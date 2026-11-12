// Qwen3-VL vision tower, model-private to MiniMax H3 (issue #177 FL2VA
// keyframe conditioning). Semantics mirror transformers'
// Qwen3VLVisionModel exactly as pinned by the sibling reference port in
// src/models/qwen_image_21/hip/pipeline.cpp (EncodeVision/VisionMerge); no
// qwen_image_21 symbol is linked or shared (repo policy: kernels stay
// model-private). Config: depth 27, hidden 1152, heads 16 (dim 72),
// intermediate 4304 gelu_pytorch_tanh, mergers with erf-GELU, LayerNorm
// eps 1e-6 everywhere, deepstack taps after blocks 8/16/24 with
// postshuffle norm, final merger to out_hidden_size 5120.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/vision_tower.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::minimax_h3 {
namespace {

namespace vt {

constexpr std::uint32_t kHidden = 1152U;
constexpr std::uint32_t kHeads = 16U;
constexpr std::uint32_t kHeadDim = 72U;
constexpr std::uint32_t kHalf = 36U;
constexpr std::uint32_t kQkv = 3456U;
constexpr std::uint32_t kIntermediate = 4304U;
constexpr std::uint32_t kMergeIn = 4608U;
constexpr std::uint32_t kOut = 5120U;
constexpr std::uint32_t kPatchDim = 1536U;
constexpr std::uint32_t kDepth = 27U;
constexpr float kNormEpsilon = 1.0e-6F;

void Fail(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool Checked(std::string_view phase, std::string* error) {
  const hipError_t status = hipGetLastError();
  if (status == hipSuccess) {
    return true;
  }
  Fail(error, std::string("MiniMax H3 vision tower ") + std::string(phase) +
                  " launch: " + hipGetErrorString(status));
  return false;
}

class Buffer {
public:
  Buffer() = default;
  ~Buffer() { Reset(); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = other.data_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }
  [[nodiscard]] bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      Fail(error, std::string("allocate MiniMax H3 vision tower buffer: ") +
                      hipGetErrorString(status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }
  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
    bytes_ = 0;
  }
  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

// --- bf16 helpers -----------------------------------------------------------

__device__ __forceinline__ float B2F(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}
__device__ __forceinline__ std::uint16_t F2B(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    return static_cast<std::uint16_t>((bits | 0x0040U) >> 16U);
  }
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}
__device__ __forceinline__ float RoundB(float value) {
  return B2F(F2B(value));
}

__global__ void Bf16ToFloatKernel(const std::uint16_t* input, float* output,
                                  std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = B2F(input[i]);
  }
}

// Linear epilogue: f32 GEMM result plus f32 bias, one bf16 conversion
// (PyTorch addmm semantics; matches the sibling reference Bias kernel).
__global__ void BiasKernel(const float* input, const std::uint16_t* bias,
                           std::uint16_t* output, std::uint32_t cols,
                           std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = F2B(input[i] + B2F(bias[i % cols]));
  }
}

// Row LayerNorm(eps 1e-6, affine) over bf16 rows of `cols` elements.
__global__ void LayerNormKernel(const std::uint16_t* input,
                                std::uint16_t* output,
                                const std::uint16_t* weight,
                                const std::uint16_t* bias, std::uint32_t cols) {
  __shared__ float scratch[256];
  const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * cols;
  float mean = 0.0F;
  for (std::uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
    mean += B2F(input[offset + c]);
  }
  scratch[threadIdx.x] = mean;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  mean = scratch[0] / static_cast<float>(cols);
  __syncthreads();
  float square = 0.0F;
  for (std::uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = B2F(input[offset + c]) - mean;
    square += v * v;
  }
  scratch[threadIdx.x] = square;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(scratch[0] / static_cast<float>(cols) + kNormEpsilon);
  __syncthreads();
  for (std::uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (B2F(input[offset + c]) - mean) * inverse;
    v *= B2F(weight[c]);
    v += B2F(bias[c]);
    output[offset + c] = F2B(v);
  }
}

__global__ void AddKernel(const std::uint16_t* left, const std::uint16_t* right,
                          std::uint16_t* output, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = F2B(B2F(left[i]) + B2F(right[i]));
  }
}

__global__ void GeluKernel(std::uint16_t* data, std::size_t n, bool tanh) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float a = B2F(data[i]);
  const float value = tanh ? 0.5F * a *
                                 (1.0F + tanhf(0.7978845608028654F *
                                               (a + 0.044715F * a * a * a)))
                           : 0.5F * a * (1.0F + erff(a * 0.7071067811865475F));
  data[i] = F2B(value);
}

// Vision RoPE (transformers apply_rotary_pos_emb_vision, half-split layout,
// frequencies interleaved cos/sin per (token, pair) with pair < 36; the
// table already encodes the recomposed h|w|h|w duplication).
__global__ void RotaryKernel(std::uint16_t* data, std::uint32_t rows,
                             std::uint32_t row_stride, std::uint32_t base_col,
                             const float* frequencies) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t pair = threadIdx.x;
  if (row >= rows || head >= kHeads || pair >= kHalf) {
    return;
  }
  const std::size_t line =
      static_cast<std::size_t>(row) * row_stride + base_col;
  const std::size_t left = line + head * kHeadDim + pair;
  const std::size_t right = left + kHalf;
  const float cosine =
      frequencies[(static_cast<std::size_t>(row) * kHalf + pair) * 2U];
  const float sine =
      frequencies[(static_cast<std::size_t>(row) * kHalf + pair) * 2U + 1U];
  const float a = B2F(data[left]);
  const float b = B2F(data[right]);
  data[left] = F2B(a * cosine - b * sine);
  data[right] = F2B(b * cosine + a * sine);
}

// Learned absolute position embedding: bilinear interpolation of the 48x48
// bf16 table onto the (height, width) patch grid, accumulated in fp32 with
// one final bf16 conversion (transformers 5.17 parity as mirrored by the
// sibling port). Token order is merge-block-major.
__global__ void PositionKernel(const std::uint16_t* input,
                               const std::uint16_t* table,
                               std::uint16_t* output, std::uint32_t height,
                               std::uint32_t width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(height) * width * kHidden) {
    return;
  }
  const std::uint32_t token = static_cast<std::uint32_t>(i / kHidden);
  const std::uint32_t c = static_cast<std::uint32_t>(i % kHidden);
  const std::uint32_t py = (token / 4U / (width / 2U)) * 2U + (token / 2U) % 2U;
  const std::uint32_t px = ((token / 4U) % (width / 2U)) * 2U + token % 2U;
  const float fy =
      static_cast<float>(py) * 47.0F / static_cast<float>(height - 1U);
  const float fx =
      static_cast<float>(px) * 47.0F / static_cast<float>(width - 1U);
  const int y0 = static_cast<int>(fy);
  const int x0 = static_cast<int>(fx);
  const int y1 = min(y0 + 1, 47);
  const int x1 = min(x0 + 1, 47);
  const float dy = fy - static_cast<float>(y0);
  const float dx = fx - static_cast<float>(x0);
  const float p0 =
      B2F(table[(static_cast<std::size_t>(y0) * 48U + x0) * kHidden + c]) *
      ((1.0F - dy) * (1.0F - dx));
  const float p1 =
      B2F(table[(static_cast<std::size_t>(y0) * 48U + x1) * kHidden + c]) *
      ((1.0F - dy) * dx);
  const float p2 =
      B2F(table[(static_cast<std::size_t>(y1) * 48U + x0) * kHidden + c]) *
      (dy * (1.0F - dx));
  const float p3 =
      B2F(table[(static_cast<std::size_t>(y1) * 48U + x1) * kHidden + c]) *
      (dy * dx);
  output[i] = F2B(B2F(input[i]) + RoundB(p0 + p1 + p2 + p3));
}

// token-major [rows][heads*dim] <-> head-major [heads][rows][dim]
__global__ void HeadLayoutKernel(const std::uint16_t* input,
                                 std::uint16_t* output, std::uint32_t rows,
                                 std::uint32_t row_stride) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * kHeads * kHeadDim) {
    return;
  }
  const std::uint32_t row = static_cast<std::uint32_t>(i / (kHeads * kHeadDim));
  const std::uint32_t rest =
      static_cast<std::uint32_t>(i % (kHeads * kHeadDim));
  const std::uint32_t head = rest / kHeadDim;
  const std::uint32_t d = rest % kHeadDim;
  output[static_cast<std::size_t>(head) * rows * kHeadDim +
         static_cast<std::size_t>(row) * kHeadDim + d] =
      input[static_cast<std::size_t>(row) * row_stride + rest];
}

// Attention tile output: head-major f32 tile [heads][count][dim] for query
// rows [start, start+count) -> bf16 token-major rows of [rows][heads*dim].
__global__ void TileOutKernel(const float* tile, std::uint16_t* destination,
                              std::uint32_t rows_total, std::uint32_t start,
                              std::uint32_t count) {
  (void)rows_total;
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(count) * kHeads * kHeadDim) {
    return;
  }
  const std::uint32_t head = static_cast<std::uint32_t>(i / (count * kHeadDim));
  const std::uint32_t rest = static_cast<std::uint32_t>(i % (count * kHeadDim));
  const std::uint32_t row = rest / kHeadDim;
  const std::uint32_t d = rest % kHeadDim;
  destination[(static_cast<std::size_t>(start + row) * kHeads + head) *
                  kHeadDim +
              d] = F2B(tile[i]);
}

// Row-parallel softmax over f32 score rows with the 1/sqrt(dim) scale baked
// in. Rows live in a [heads][tile][rows_total] score buffer; blockIdx.x
// enumerates head-major (head * count + local_query_row).
__global__ void SoftmaxKernel(const float* scores, float* probabilities,
                              std::uint32_t columns, std::uint32_t count,
                              std::uint32_t tile, float scale) {
  __shared__ float scratch[256];
  const std::uint32_t head = blockIdx.x / count;
  const std::uint32_t local = blockIdx.x % count;
  const std::size_t offset =
      (static_cast<std::size_t>(head) * tile + local) * columns;
  float maximum = -INFINITY;
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    maximum = fmaxf(maximum, scores[offset + i] * scale);
  }
  scratch[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = scratch[0];
  __syncthreads();
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    sum += expf(scores[offset + i] * scale - maximum);
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  sum = scratch[0];
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    probabilities[offset + i] =
        expf(scores[offset + i] * scale - maximum) / sum;
  }
}

constexpr unsigned kThreads = 256U;

inline dim3 Grid(std::size_t elements) {
  return dim3(
      static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads));
}

// --- host-side bf16 conversion (upload payloads) ----------------------------

std::uint16_t HostFloatToBf16(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if (std::isnan(value)) {
    return static_cast<std::uint16_t>((bits | 0x0040U) >> 16U);
  }
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

// --- inventory weight loading ----------------------------------------------

bool LoadRaw(const ModelInventory& inventory, std::string_view name,
             std::vector<std::uint16_t>& out, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    Fail(error,
         "MiniMax H3 vision tower tensor is missing: " + std::string(name));
    return false;
  }
  if (descriptor->dtype != DType::kBFloat16) {
    Fail(error,
         "MiniMax H3 vision tower tensor dtype differs: " + std::string(name));
    return false;
  }
  if (descriptor->payload_bytes % 2U != 0U ||
      descriptor->payload_bytes > std::numeric_limits<std::size_t>::max()) {
    Fail(error, "MiniMax H3 vision tower tensor payload invalid: " +
                    std::string(name));
    return false;
  }
  out.resize(static_cast<std::size_t>(descriptor->payload_bytes) / 2U);
  const std::filesystem::path path = inventory.model_root() / descriptor->shard;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    Fail(error, "cannot open MiniMax H3 vision tower shard " + path.string() +
                    ": " + open_error);
    return false;
  }
  std::size_t read_bytes = 0;
  while (read_bytes < descriptor->payload_bytes) {
    const std::size_t result = platform::ReadFileAt(
        file, reinterpret_cast<std::byte*>(out.data()) + read_bytes,
        descriptor->payload_bytes - read_bytes,
        descriptor->file_offset + read_bytes);
    if (result == 0) {
      platform::CloseFile(file);
      Fail(error, "short read loading MiniMax H3 vision tower tensor " +
                      std::string(name));
      return false;
    }
    read_bytes += result;
  }
  platform::CloseFile(file);
  return true;
}

bool Upload(hipStream_t stream, Buffer& buffer, const void* source,
            std::size_t bytes, std::string* error) {
  if (bytes == 0) {
    return true;
  }
  if (hipMemcpyAsync(buffer.data(), source, bytes, hipMemcpyHostToDevice,
                     stream) != hipSuccess) {
    Fail(error, "MiniMax H3 vision tower weight upload failed");
    return false;
  }
  return true;
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

}  // namespace vt

}  // namespace

// Per-call device session; weights load fresh (single-shot keyframe encode;
// caching is a later optimization, not a correctness concern).
struct VisionTowerWeights {
  vt::Buffer patch_weight;
  vt::Buffer patch_bias;
  vt::Buffer pos_table;
  struct Block {
    vt::Buffer norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b;
    vt::Buffer norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b;
  };
  std::array<Block, vt::kDepth> blocks;
  struct Merger {
    vt::Buffer norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b;
  };
  Merger merger;
  std::array<Merger, 3> deepstack{};
  std::uint64_t bytes{0};
};

namespace {

bool LoadMerger(const ModelInventory& inventory, const std::string& base,
                VisionTowerWeights::Merger* merger, hipStream_t stream,
                std::uint64_t* bytes, std::string* error) {
  const std::string prefix = "model.visual." + base;
  struct Item {
    const char* suffix;
    vt::Buffer VisionTowerWeights::Merger::* slot;
  };
  const Item items[] = {
      {".norm.weight", &VisionTowerWeights::Merger::norm_w},
      {".norm.bias", &VisionTowerWeights::Merger::norm_b},
      {".linear_fc1.weight", &VisionTowerWeights::Merger::fc1_w},
      {".linear_fc1.bias", &VisionTowerWeights::Merger::fc1_b},
      {".linear_fc2.weight", &VisionTowerWeights::Merger::fc2_w},
      {".linear_fc2.bias", &VisionTowerWeights::Merger::fc2_b},
  };
  for (const Item& item : items) {
    std::vector<std::uint16_t> raw;
    if (!vt::LoadRaw(inventory, prefix + item.suffix, raw, error)) {
      return false;
    }
    vt::Buffer& slot = (merger->*(item.slot));
    if (!slot.Allocate(raw.size() * 2U, error) ||
        !vt::Upload(stream, slot, raw.data(), raw.size() * 2U, error)) {
      return false;
    }
    *bytes += slot.bytes();
  }
  return true;
}

class Roc {
public:
  Roc() {
    if (rocblas_create_handle(&handle_) == rocblas_status_success) {
      (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    }
  }
  ~Roc() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  Roc(const Roc&) = delete;
  Roc& operator=(const Roc&) = delete;

  bool ok() const noexcept { return handle_ != nullptr; }

  // Linear: out_f32[batch][rows] = W[rows][cols] x in[batch][cols].
  bool Linear(const std::uint16_t* weight, const std::uint16_t* input,
              float* output, std::size_t batch, std::uint32_t rows,
              std::uint32_t cols, hipStream_t stream, std::string* error) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower set stream failed");
      return false;
    }
    const rocblas_status status = rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(cols), &alpha, weight, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(cols), input, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(cols), &beta, output, rocblas_datatype_f32_r,
        static_cast<rocblas_int>(rows), output, rocblas_datatype_f32_r,
        static_cast<rocblas_int>(rows), rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower linear failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

  // S[h] = Q[h] K[h]^T with bf16 in, f32 out (row-major per head).
  bool Scores(const std::uint16_t* key_major, std::size_t key_stride,
              const std::uint16_t* query_major, std::size_t query_stride,
              float* scores, std::size_t scores_stride,
              std::uint32_t query_rows, std::uint32_t key_rows,
              std::uint32_t dim, std::uint32_t heads, hipStream_t stream,
              std::string* error) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower set stream failed");
      return false;
    }
    const rocblas_status status = rocblas_gemm_strided_batched_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(key_rows),
        static_cast<rocblas_int>(query_rows), static_cast<rocblas_int>(dim),
        &alpha, key_major, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(dim), static_cast<rocblas_stride>(key_stride),
        query_major, rocblas_datatype_bf16_r, static_cast<rocblas_int>(dim),
        static_cast<rocblas_stride>(query_stride), &beta, scores,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(key_rows),
        static_cast<rocblas_stride>(scores_stride), scores,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(key_rows),
        static_cast<rocblas_stride>(scores_stride),
        static_cast<rocblas_int>(heads), rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower score GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

  // O[h] = P[h] V[h], f32 in and out. Probabilities live in a
  // [heads][tile][key_rows] buffer (p_batch_stride elements per head); the
  // output is a contiguous [heads][query_rows][dim] tile.
  bool Values(const float* v_head_major, const float* probabilities,
              float* output_head_major, std::uint32_t query_rows,
              std::uint32_t key_rows, std::uint32_t dim, std::uint32_t heads,
              std::size_t p_batch_stride, hipStream_t stream,
              std::string* error) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower set stream failed");
      return false;
    }
    // Row-major O[m=q][n=d] = P[m][k] V[k][n] as column-major:
    // O_cm(d,q) = V_cm(d,k) P_cm(k,q) with V head-major contiguous.
    const rocblas_status status = rocblas_gemm_strided_batched_ex(
        handle_, rocblas_operation_none, rocblas_operation_none,
        static_cast<rocblas_int>(dim), static_cast<rocblas_int>(query_rows),
        static_cast<rocblas_int>(key_rows), &alpha, v_head_major,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(dim),
        static_cast<rocblas_stride>(key_rows) * dim, probabilities,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(key_rows),
        static_cast<rocblas_stride>(p_batch_stride), &beta, output_head_major,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(dim),
        static_cast<rocblas_stride>(query_rows) * dim, output_head_major,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(dim),
        static_cast<rocblas_stride>(query_rows) * dim,
        static_cast<rocblas_int>(heads), rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      vt::Fail(error, "MiniMax H3 vision tower value GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  rocblas_handle handle_{nullptr};
};

bool EncodeOneVision(const ModelInventory& inventory,
                     const KeyframeVision& vision,
                     const CancellationToken* cancellation,
                     std::vector<std::uint16_t>* output,
                     std::vector<std::uint16_t>* deepstack,
                     std::string* error) {
  using namespace vt;  // NOLINT
  if (output == nullptr) {
    Fail(error, "MiniMax H3 vision tower output is missing");
    return false;
  }
  if (vision.grid_h < 2 || vision.grid_w < 2 || vision.grid_h % 2 != 0 ||
      vision.grid_w % 2 != 0) {
    Fail(error, "MiniMax H3 vision grid must be even and at least 2");
    return false;
  }
  const std::uint32_t rows =
      static_cast<std::uint32_t>(vision.grid_h) * vision.grid_w;
  const std::size_t merged = static_cast<std::size_t>(rows) / 4U;
  if (vision.pixel_values.size() !=
      static_cast<std::size_t>(rows) * kPatchDim) {
    Fail(error, "MiniMax H3 vision pixel payload differs from the grid");
    return false;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    Fail(error, "MiniMax H3 vision tower device query failed");
    return false;
  }
  hipStream_t stream = nullptr;
  if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
    Fail(error, "MiniMax H3 vision tower stream creation failed");
    return false;
  }
  const auto cleanup = [&]() { (void)hipStreamDestroy(stream); };

  Roc roc;
  if (!roc.ok()) {
    Fail(error, "MiniMax H3 vision tower rocBLAS init failed");
    cleanup();
    return false;
  }

  // --- weights --------------------------------------------------------------
  VisionTowerWeights weights;
  std::uint64_t loaded_bytes = 0;
  const auto load = [&](const std::string& name, std::size_t elements,
                        vt::Buffer* slot) -> bool {
    std::vector<std::uint16_t> raw;
    if (!LoadRaw(inventory, name, raw, error)) {
      return false;
    }
    if (raw.size() != elements) {
      Fail(error, "MiniMax H3 vision tower tensor shape differs: " + name);
      return false;
    }
    if (!slot->Allocate(raw.size() * 2U, error) ||
        !Upload(stream, *slot, raw.data(), raw.size() * 2U, error)) {
      return false;
    }
    loaded_bytes += slot->bytes();
    return true;
  };
  if (!load("model.visual.patch_embed.proj.weight", kHidden * kPatchDim,
            &weights.patch_weight) ||
      !load("model.visual.patch_embed.proj.bias", kHidden,
            &weights.patch_bias) ||
      !load("model.visual.pos_embed.weight", 2304U * kHidden,
            &weights.pos_table)) {
    cleanup();
    return false;
  }
  for (std::uint32_t block = 0; block < kDepth; ++block) {
    const std::string base = "model.visual.blocks." + std::to_string(block);
    auto& b = weights.blocks[block];
    if (!load(base + ".norm1.weight", kHidden, &b.norm1_w) ||
        !load(base + ".norm1.bias", kHidden, &b.norm1_b) ||
        !load(base + ".attn.qkv.weight", kQkv * kHidden, &b.qkv_w) ||
        !load(base + ".attn.qkv.bias", kQkv, &b.qkv_b) ||
        !load(base + ".attn.proj.weight", kHidden * kHidden, &b.proj_w) ||
        !load(base + ".attn.proj.bias", kHidden, &b.proj_b) ||
        !load(base + ".norm2.weight", kHidden, &b.norm2_w) ||
        !load(base + ".norm2.bias", kHidden, &b.norm2_b) ||
        !load(base + ".mlp.linear_fc1.weight", kIntermediate * kHidden,
              &b.fc1_w) ||
        !load(base + ".mlp.linear_fc1.bias", kIntermediate, &b.fc1_b) ||
        !load(base + ".mlp.linear_fc2.weight", kHidden * kIntermediate,
              &b.fc2_w) ||
        !load(base + ".mlp.linear_fc2.bias", kHidden, &b.fc2_b)) {
      cleanup();
      return false;
    }
  }
  if (!LoadMerger(inventory, "merger", &weights.merger, stream, &loaded_bytes,
                  error) ||
      !LoadMerger(inventory, "deepstack_merger_list.0", &weights.deepstack[0],
                  stream, &loaded_bytes, error) ||
      !LoadMerger(inventory, "deepstack_merger_list.1", &weights.deepstack[1],
                  stream, &loaded_bytes, error) ||
      !LoadMerger(inventory, "deepstack_merger_list.2", &weights.deepstack[2],
                  stream, &loaded_bytes, error)) {
    cleanup();
    return false;
  }

  // --- host-prepared inputs
  // ---------------------------------------------------
  std::vector<std::uint16_t> patches(static_cast<std::size_t>(rows) *
                                     kPatchDim);
  for (std::size_t i = 0; i < patches.size(); ++i) {
    patches[i] = HostFloatToBf16(vision.pixel_values[i]);
  }
  std::vector<float> frequencies(static_cast<std::size_t>(rows) * kHalf * 2U);
  for (std::uint32_t token = 0; token < rows; ++token) {
    const std::uint32_t py =
        (token / 4U / (vision.grid_w / 2U)) * 2U + (token / 2U) % 2U;
    const std::uint32_t px =
        ((token / 4U) % (vision.grid_w / 2U)) * 2U + token % 2U;
    for (std::uint32_t pair = 0; pair < kHalf; ++pair) {
      const std::uint32_t position = pair < 18U ? py : px;
      const float angle =
          static_cast<float>(position) /
          std::pow(10000.0F, static_cast<float>(pair % 18U) / 18.0F);
      frequencies[(static_cast<std::size_t>(token) * kHalf + pair) * 2U] =
          std::cos(angle);
      frequencies[(static_cast<std::size_t>(token) * kHalf + pair) * 2U + 1U] =
          std::sin(angle);
    }
  }

  // --- device buffers
  // --------------------------------------------------------- Role-sized
  // buffers, no cross-role swaps: x is the running residual, y is LayerNorm
  // output, qkv the attention projection, mlp the 4304-wide intermediate,
  // merge_out the 5120-wide merged tokens.
  Buffer pixel_device, x, y, qkv, mlp, merge_out, attn_out, head_q, head_k,
      head_v, v_float;
  Buffer probabilities, scores, out_head, rope;
  const std::size_t tile = rows < 1024U ? rows : 1024U;
  const std::size_t hidden_bytes =
      static_cast<std::size_t>(rows) * kHidden * 2U;
  const std::size_t merged_out_bytes = merged * kOut * 2U;
  if (!pixel_device.Allocate(patches.size() * 2U, error) ||
      !x.Allocate(hidden_bytes, error) || !y.Allocate(hidden_bytes, error) ||
      !qkv.Allocate(static_cast<std::size_t>(rows) * kQkv * 2U, error) ||
      !mlp.Allocate(static_cast<std::size_t>(rows) * kIntermediate * 2U,
                    error) ||
      !merge_out.Allocate(merged_out_bytes, error) ||
      !attn_out.Allocate(hidden_bytes, error) ||
      !head_q.Allocate(qkv.bytes(), error) ||
      !head_k.Allocate(qkv.bytes(), error) ||
      !head_v.Allocate(qkv.bytes(), error) ||
      !v_float.Allocate(static_cast<std::size_t>(rows) * kQkv * 4U, error) ||
      !scores.Allocate(static_cast<std::size_t>(kHeads) * tile * rows * 4U,
                       error) ||
      !probabilities.Allocate(scores.bytes(), error) ||
      !out_head.Allocate(static_cast<std::size_t>(rows) * kQkv * 4U, error) ||
      !rope.Allocate(frequencies.size() * sizeof(float), error)) {
    cleanup();
    return false;
  }
  if (hipMemcpyAsync(pixel_device.data(), patches.data(), patches.size() * 2U,
                     hipMemcpyHostToDevice, stream) != hipSuccess ||
      hipMemcpyAsync(rope.data(), frequencies.data(),
                     frequencies.size() * sizeof(float), hipMemcpyHostToDevice,
                     stream) != hipSuccess) {
    Fail(error, "MiniMax H3 vision tower input upload failed");
    cleanup();
    return false;
  }

  // Scratch for GEMM f32 results (largest row count: kIntermediate).
  Buffer gemm_out;
  if (!gemm_out.Allocate(static_cast<std::size_t>(rows) * kIntermediate * 4U,
                         error)) {
    cleanup();
    return false;
  }

  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

  // patch_embed Linear + pos_embed.
  auto linear = [&](const std::uint16_t* weight, const std::uint16_t* bias,
                    const std::uint16_t* input, std::uint32_t in_cols,
                    std::uint32_t out_cols, std::uint32_t count,
                    std::uint16_t* destination) -> bool {
    if (!roc.Linear(weight, input, gemm_out.As<float>(), count, out_cols,
                    in_cols, stream, error)) {
      return false;
    }
    BiasKernel<<<Grid(static_cast<std::size_t>(count) * out_cols), kThreads, 0,
                 stream>>>(gemm_out.As<float>(), bias, destination, out_cols,
                           static_cast<std::size_t>(count) * out_cols);
    return Checked("linear", error);
  };

  if (!linear(weights.patch_weight.As<std::uint16_t>(),
              weights.patch_bias.As<std::uint16_t>(),
              pixel_device.As<std::uint16_t>(), kPatchDim, kHidden, rows,
              x.As<std::uint16_t>())) {
    cleanup();
    return false;
  }
  PositionKernel<<<Grid(static_cast<std::size_t>(rows) * kHidden), kThreads, 0,
                   stream>>>(
      x.As<std::uint16_t>(), weights.pos_table.As<std::uint16_t>(),
      y.As<std::uint16_t>(), static_cast<std::uint32_t>(vision.grid_h),
      static_cast<std::uint32_t>(vision.grid_w));
  if (!Checked("position", error)) {
    cleanup();
    return false;
  }
  std::swap(x, y);

  for (std::uint32_t block = 0; block < kDepth; ++block) {
    if (IsCancelled(cancellation)) {
      Fail(error, "MiniMax H3 vision tower cancelled");
      cleanup();
      return false;
    }
    const auto& b = weights.blocks[block];
    const auto stage = [&]() -> bool {
      // norm1 -> qkv -> rope -> attention -> proj -> residual
      LayerNormKernel<<<rows, kThreads, 0, stream>>>(
          x.As<std::uint16_t>(), y.As<std::uint16_t>(),
          b.norm1_w.As<std::uint16_t>(), b.norm1_b.As<std::uint16_t>(),
          kHidden);
      if (!Checked("norm1", error)) {
        return false;
      }
      if (!linear(b.qkv_w.As<std::uint16_t>(), b.qkv_b.As<std::uint16_t>(),
                  y.As<std::uint16_t>(), kHidden, kQkv, rows,
                  qkv.As<std::uint16_t>())) {
        return false;
      }
      RotaryKernel<<<dim3(rows, kHeads), kHalf, 0, stream>>>(
          qkv.As<std::uint16_t>(), rows, kQkv, 0, rope.As<float>());
      RotaryKernel<<<dim3(rows, kHeads), kHalf, 0, stream>>>(
          qkv.As<std::uint16_t>(), rows, kQkv, kHidden, rope.As<float>());
      if (!Checked("rotary", error)) {
        return false;
      }
      // Full head-major transposes of the three qkv regions.
      HeadLayoutKernel<<<Grid(static_cast<std::size_t>(rows) * kQkv), kThreads,
                         0, stream>>>(qkv.As<std::uint16_t>(),
                                      head_q.As<std::uint16_t>(), rows, kQkv);
      HeadLayoutKernel<<<Grid(static_cast<std::size_t>(rows) * kQkv), kThreads,
                         0, stream>>>(qkv.As<std::uint16_t>() + kHidden,
                                      head_k.As<std::uint16_t>(), rows, kQkv);
      HeadLayoutKernel<<<Grid(static_cast<std::size_t>(rows) * kQkv), kThreads,
                         0, stream>>>(qkv.As<std::uint16_t>() + kHidden * 2U,
                                      head_v.As<std::uint16_t>(), rows, kQkv);
      if (!Checked("head layout", error)) {
        return false;
      }
      Bf16ToFloatKernel<<<Grid(static_cast<std::size_t>(rows) * kQkv), kThreads,
                          0, stream>>>(head_v.As<std::uint16_t>(),
                                       v_float.As<float>(),
                                       static_cast<std::size_t>(rows) * kQkv);
      if (!Checked("v convert", error)) {
        return false;
      }
      for (std::uint32_t start = 0; start < rows; start += tile) {
        const std::uint32_t count = rows - start < tile ? rows - start : tile;
        if (!roc.Scores(head_k.As<std::uint16_t>(), rows * kHeadDim,
                        head_q.As<std::uint16_t>() +
                            static_cast<std::size_t>(start) * kHeadDim,
                        rows * kHeadDim, scores.As<float>(),
                        static_cast<std::size_t>(tile) * rows, count, rows,
                        kHeadDim, kHeads, stream, error)) {
          return false;
        }
        SoftmaxKernel<<<kHeads * count, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<float>(), rows, count, tile,
            attention_scale);
        if (!Checked("softmax", error)) {
          return false;
        }
        if (!roc.Values(v_float.As<float>(), probabilities.As<float>(),
                        out_head.As<float>(), count, rows, kHeadDim, kHeads,
                        static_cast<std::size_t>(tile) * rows, stream, error)) {
          return false;
        }
        TileOutKernel<<<Grid(static_cast<std::size_t>(count) * kQkv), kThreads,
                        0, stream>>>(out_head.As<float>(),
                                     attn_out.As<std::uint16_t>(), rows, start,
                                     count);
        if (!Checked("attention tile", error)) {
          return false;
        }
      }
      if (!linear(b.proj_w.As<std::uint16_t>(), b.proj_b.As<std::uint16_t>(),
                  attn_out.As<std::uint16_t>(), kHidden, kHidden, rows,
                  qkv.As<std::uint16_t>())) {
        return false;
      }
      AddKernel<<<Grid(static_cast<std::size_t>(rows) * kHidden), kThreads, 0,
                  stream>>>(x.As<std::uint16_t>(), qkv.As<std::uint16_t>(),
                            x.As<std::uint16_t>(),
                            static_cast<std::size_t>(rows) * kHidden);
      if (!Checked("residual 1", error)) {
        return false;
      }
      // norm2 -> fc1 -> gelu tanh -> fc2 -> residual
      LayerNormKernel<<<rows, kThreads, 0, stream>>>(
          x.As<std::uint16_t>(), y.As<std::uint16_t>(),
          b.norm2_w.As<std::uint16_t>(), b.norm2_b.As<std::uint16_t>(),
          kHidden);
      if (!Checked("norm2", error)) {
        return false;
      }
      if (!linear(b.fc1_w.As<std::uint16_t>(), b.fc1_b.As<std::uint16_t>(),
                  y.As<std::uint16_t>(), kHidden, kIntermediate, rows,
                  mlp.As<std::uint16_t>())) {
        return false;
      }
      GeluKernel<<<Grid(static_cast<std::size_t>(rows) * kIntermediate),
                   kThreads, 0, stream>>>(
          mlp.As<std::uint16_t>(),
          static_cast<std::size_t>(rows) * kIntermediate, true);
      if (!Checked("gelu", error)) {
        return false;
      }

      if (!linear(b.fc2_w.As<std::uint16_t>(), b.fc2_b.As<std::uint16_t>(),
                  mlp.As<std::uint16_t>(), kIntermediate, kHidden, rows,
                  y.As<std::uint16_t>())) {
        return false;
      }
      AddKernel<<<Grid(static_cast<std::size_t>(rows) * kHidden), kThreads, 0,
                  stream>>>(x.As<std::uint16_t>(), y.As<std::uint16_t>(),
                            x.As<std::uint16_t>(),
                            static_cast<std::size_t>(rows) * kHidden);
      if (!Checked("residual 2", error)) {
        return false;
      }
      return true;
    }();
    if (!stage) {
      cleanup();
      return false;
    }
    if (block == 8U || block == 16U || block == 24U) {
      // Deepstack merger (postshuffle): LayerNorm over 4x1152 rows.
      const auto& m = weights.deepstack[block / 8U - 1U];
      const std::uint32_t merged_rows = static_cast<std::uint32_t>(merged);
      LayerNormKernel<<<merged_rows, kThreads, 0, stream>>>(
          x.As<std::uint16_t>(), y.As<std::uint16_t>(),
          m.norm_w.As<std::uint16_t>(), m.norm_b.As<std::uint16_t>(), kMergeIn);
      if (!Checked("deepstack norm", error)) {
        cleanup();
        return false;
      }
      if (!linear(m.fc1_w.As<std::uint16_t>(), m.fc1_b.As<std::uint16_t>(),
                  y.As<std::uint16_t>(), kMergeIn, kMergeIn, merged_rows,
                  qkv.As<std::uint16_t>())) {
        cleanup();
        return false;
      }
      GeluKernel<<<Grid(merged * kMergeIn), kThreads, 0, stream>>>(
          qkv.As<std::uint16_t>(), merged * kMergeIn, false);
      if (!Checked("deepstack gelu", error)) {
        cleanup();
        return false;
      }
      if (!linear(m.fc2_w.As<std::uint16_t>(), m.fc2_b.As<std::uint16_t>(),
                  qkv.As<std::uint16_t>(), kMergeIn, kOut, merged_rows,
                  merge_out.As<std::uint16_t>())) {
        cleanup();
        return false;
      }
      if (deepstack != nullptr) {
        std::vector<std::uint16_t> host(merged * kOut);
        if (hipMemcpyAsync(host.data(), merge_out.data(), host.size() * 2U,
                           hipMemcpyDeviceToHost, stream) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess) {
          Fail(error, "MiniMax H3 deepstack readback failed");
          cleanup();
          return false;
        }
        deepstack->insert(deepstack->end(), host.begin(), host.end());
      }
    }
  }

  // Final merger: LayerNorm over 1152, then reshape to merged rows.
  {
    const auto& m = weights.merger;
    LayerNormKernel<<<rows, kThreads, 0, stream>>>(
        x.As<std::uint16_t>(), y.As<std::uint16_t>(),
        m.norm_w.As<std::uint16_t>(), m.norm_b.As<std::uint16_t>(), kHidden);
    if (!Checked("merger norm", error)) {
      cleanup();
      return false;
    }
    if (!linear(m.fc1_w.As<std::uint16_t>(), m.fc1_b.As<std::uint16_t>(),
                y.As<std::uint16_t>(), kMergeIn, kMergeIn,
                static_cast<std::uint32_t>(merged), qkv.As<std::uint16_t>())) {
      cleanup();
      return false;
    }
    GeluKernel<<<Grid(merged * kMergeIn), kThreads, 0, stream>>>(
        qkv.As<std::uint16_t>(), merged * kMergeIn, false);
    if (!Checked("merger gelu", error)) {
      cleanup();
      return false;
    }
    output->assign(merged * kOut, 0U);
    if (!linear(m.fc2_w.As<std::uint16_t>(), m.fc2_b.As<std::uint16_t>(),
                qkv.As<std::uint16_t>(), kMergeIn, kOut,
                static_cast<std::uint32_t>(merged),
                merge_out.As<std::uint16_t>())) {
      cleanup();
      return false;
    }
    if (hipMemcpyAsync(output->data(), merge_out.data(), output->size() * 2U,
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      Fail(error, "MiniMax H3 vision tower readback failed");
      cleanup();
      return false;
    }
  }
  cleanup();
  return true;
}

}  // namespace

bool EncodeVision(const ModelInventory& inventory, const KeyframeVision& vision,
                  const CancellationToken* cancellation,
                  std::vector<std::uint16_t>* output,
                  std::vector<std::uint16_t>* deepstack, std::string* error) {
  if (deepstack != nullptr) {
    deepstack->clear();
  }
  return EncodeOneVision(inventory, vision, cancellation, output, deepstack,
                         error);
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
