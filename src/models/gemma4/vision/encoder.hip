#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/vision/encoder.hpp"
#include "src/models/gemma4/vision/prompt.hpp"
#include "src/models/gemma4/vision/weights.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::gemma4::vision {
namespace {
/// GEMM operand format: binary16 activations (and the BF16 weights converted
/// on upload, exact within its range), FP32 accumulation.
using Act = _Float16;

/// Binary16 of v, saturated to the finite range.
__device__ inline Act ToAct(float v) {
  return static_cast<Act>(fminf(fmaxf(v, -65504.0F), 65504.0F));
}

/// In place, BF16 weights to binary16 (values below 2^-24 flush to zero).
__global__ void Bf16ToHalf(std::uint16_t* data, std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float v = __uint_as_float(std::uint32_t{data[i]} << 16U);
  data[i] = __builtin_bit_cast(std::uint16_t, ToAct(v));
}
constexpr unsigned kThreads = 256;
constexpr unsigned kQkv = 3 * kHidden;
constexpr unsigned kGateUp = 2 * kFfn;
/// The down projection's K padded to whole 32-value blocks for the binary16
/// WMMA GEMM (zero weights and activations past kFfn).
constexpr unsigned kFfnPadded = (kFfn + 31) / 32 * 32;
/// Rotary pairs per axis: each half of a head rotates as NEOX pairs.
constexpr unsigned kRopePairs = kHeadDim / 4;
/// Attention operands pad each head to whole 16-wide WMMA steps and each
/// key sequence to whole kKeyTile tiles, with zeros.
constexpr unsigned kPaddedDim = 80;
constexpr unsigned kDimSteps = kPaddedDim / 16;
constexpr unsigned kAttentionWaves = 4;
/// Keys per softmax update; key sequences are padded to whole tiles.
constexpr unsigned kKeyTile = 64;

using Half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Gemma 4 vision hipBLAS failure: " +
                             std::to_string(status));
  }
}

struct Allocation {
  void* data{nullptr};
  Allocation() = default;
  explicit Allocation(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Allocation() {
    if (data != nullptr)
      (void)hipFree(data);
  }
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&& other) noexcept
      : data(std::exchange(other.data, nullptr)) {}
  Allocation& operator=(Allocation&& other) noexcept {
    if (this != &other) {
      if (data != nullptr)
        (void)hipFree(data);
      data = std::exchange(other.data, nullptr);
    }
    return *this;
  }
  template<class T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

/// Sum over a kThreads block; every thread receives the total.
__device__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

/// Patches in 2*(v/255)-1, each ordered channel, row, column like the
/// patch-embedding kernel.
__global__ void Patchify(const std::uint8_t* pixels, float* patches,
                         unsigned width, unsigned columns, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchValues)
    return;
  const unsigned t = i / kPatchValues;
  const unsigned e = i % kPatchValues;
  const unsigned c = e / (kPatchSize * kPatchSize);
  const unsigned y = (e / kPatchSize) % kPatchSize;
  const unsigned x = e % kPatchSize;
  const unsigned px = (t % columns) * kPatchSize + x;
  const unsigned py = (t / columns) * kPatchSize + y;
  patches[i] =
      2.0F * (static_cast<float>(pixels[(py * width + px) * 3 + c]) / 255.0F) -
      1.0F;
}

__global__ void AddPositions(float* hidden, const float* table,
                             unsigned columns, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned t = i / kHidden;
  const unsigned d = i % kHidden;
  hidden[i] += table[(t % columns) * kHidden + d] +
               table[(kPositions + t / columns) * kHidden + d];
}

/// y[row] = rms(x[row]) * weight (weight may be null) as binary16.
__global__ void RmsNormHalf(const float* x, const float* weight, Act* y,
                            unsigned width) {
  __shared__ float shared[kThreads];
  const float* row = x + std::size_t{blockIdx.x} * width;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads)
    sum += row[j] * row[j];
  const float scale = rsqrtf(BlockSum(sum, shared) / width + kEps);
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    y[std::size_t{blockIdx.x} * width + j] =
        ToAct(row[j] * scale * (weight != nullptr ? weight[j] : 1.0F));
  }
}

/// hidden[row] += rms(o[row]) * weight.
__global__ void PostNormAdd(float* hidden, const float* o,
                            const float* weight) {
  __shared__ float shared[kThreads];
  const std::size_t base = std::size_t{blockIdx.x} * kHidden;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    sum += o[base + j] * o[base + j];
  const float scale = rsqrtf(BlockSum(sum, shared) / kHidden + kEps);
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    hidden[base + j] += o[base + j] * scale * weight[j];
}

/// One block of three waves per token and head; wave 0, 1, 2 handles Q, K,
/// V. Each is RMS-normalized over the head (Q and K with their weights, V
/// without); Q and K are rotated. Q and K are written as binary16
/// [head][padded count][80], V transposed as [head][80][padded count], the
/// layouts FusedAttention reads. `cos`/`sin` hold [position][kRopePairs] for
/// the patch column (first half of the head) and row (second half).
__global__ void __launch_bounds__(96)
    QkvPost(const float* qkv, const float* q_norm, const float* k_norm,
            const float* cos, const float* sin, _Float16* q, _Float16* k,
            _Float16* vt, unsigned columns, unsigned padded) {
  __shared__ float values[3][kHeadDim];
  const unsigned token = blockIdx.x / kHeads;
  const unsigned head = blockIdx.x % kHeads;
  const unsigned part = threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  const float* in = qkv + std::size_t{token} * kQkv + part * kHidden +
                    head * kHeadDim;
  float x[3];
  float sum = 0;
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    x[j] = d < kHeadDim ? in[d] : 0.0F;
    sum += x[j] * x[j];
  }
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor(sum, offset, 32);
  const float scale = rsqrtf(sum / kHeadDim + kEps);
  const float* weight = part == 0 ? q_norm : part == 1 ? k_norm : nullptr;
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    if (d < kHeadDim)
      values[part][d] =
          x[j] * scale * (weight != nullptr ? weight[d] : 1.0F);
  }
  __syncthreads();
#pragma unroll
  for (unsigned j = 0; j < 3; ++j) {
    const unsigned d = lane + 32 * j;
    if (d >= kHeadDim)
      continue;
    if (part == 2) {
      vt[(std::size_t{head} * kPaddedDim + d) * padded + token] =
          static_cast<_Float16>(values[2][d]);
      continue;
    }
    const unsigned half = d / (kHeadDim / 2);
    const unsigned i = d % (kHeadDim / 2);
    const unsigned pair = i % kRopePairs;
    const unsigned position = half == 0 ? token % columns : token / columns;
    const float c = cos[position * kRopePairs + pair];
    const float s = sin[position * kRopePairs + pair];
    const unsigned first = half * (kHeadDim / 2) + pair;
    const float a = values[part][first];
    const float b = values[part][first + kRopePairs];
    const float rotated = i < kRopePairs ? a * c - b * s : b * c + a * s;
    (part == 0 ? q : k)[(std::size_t{head} * padded + token) * kPaddedDim +
                        d] = static_cast<_Float16>(rotated);
  }
}

__device__ __forceinline__ Float8 Wmma(Half16 a, Half16 b, Float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ Half16 LoadHalf16(const _Float16* p) {
  union {
    Half16 h;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(p);
  value.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return value.h;
}

/// Bidirectional scale-1 attention with an online softmax. Each wave owns 16
/// queries of one head and walks 64-key tiles (four 16-key WMMA blocks per
/// softmax update): binary16 Q K^T on the matrix cores, FP32 softmax state,
/// binary16 probabilities times V accumulated in FP32. Fragment layout
/// (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16 contiguous k, B
/// column L%16 and 16 contiguous k, and accumulator element i is row
/// 2i + L/16, column L%16. Writes token-major binary16 rows.
__global__ void __launch_bounds__(32 * kAttentionWaves)
    FusedAttention(const _Float16* q, const _Float16* k, const _Float16* vt,
                   Act* out, unsigned count, unsigned padded) {
  // Per-wave probability tiles: waves never exchange data.
  __shared__ __align__(16) _Float16
      p_lds[kAttentionWaves][16][kKeyTile + 8];
  const unsigned lane = threadIdx.x % 32;
  const unsigned wave = threadIdx.x / 32;
  const unsigned head = blockIdx.y;
  const unsigned q0 = (blockIdx.x * kAttentionWaves + wave) * 16;
  if (q0 >= count)
    return;
  const unsigned row = lane % 16;
  const unsigned half = lane / 16;
  const _Float16* qh = q + std::size_t{head} * padded * kPaddedDim;
  const _Float16* kh = k + std::size_t{head} * padded * kPaddedDim;
  const _Float16* vh = vt + std::size_t{head} * kPaddedDim * padded;
  Half16 query[kDimSteps];
#pragma unroll
  for (unsigned s = 0; s < kDimSteps; ++s)
    query[s] = LoadHalf16(qh + std::size_t{q0 + row} * kPaddedDim + s * 16);
  Float8 acc[kDimSteps] = {};
  float m[8], l[8];
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0F;
  }
  constexpr unsigned kBlocks = kKeyTile / 16;
  for (unsigned key0 = 0; key0 < count; key0 += kKeyTile) {
    Float8 score[kBlocks] = {};
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (unsigned s = 0; s < kDimSteps; ++s) {
        score[b] = Wmma(
            query[s],
            LoadHalf16(kh + std::size_t{key0 + b * 16 + row} * kPaddedDim +
                       s * 16),
            score[b]);
      }
    }
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      float x[kBlocks];
      float maximum = -INFINITY;
#pragma unroll
      for (unsigned b = 0; b < kBlocks; ++b) {
        x[b] = key0 + b * 16 + row < count ? score[b][i] : -INFINITY;
        maximum = fmaxf(maximum, x[b]);
      }
#pragma unroll
      for (unsigned offset = 8; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
      const float next = fmaxf(m[i], maximum);
      const float alpha = __expf(m[i] - next);
      float sum = 0.0F;
#pragma unroll
      for (unsigned b = 0; b < kBlocks; ++b) {
        const float p = __expf(x[b] - next);
        sum += p;
        p_lds[wave][2 * i + half][b * 16 + row] = static_cast<_Float16>(p);
      }
#pragma unroll
      for (unsigned offset = 8; offset > 0; offset /= 2)
        sum += __shfl_xor(sum, offset, 32);
      l[i] = l[i] * alpha + sum;
      m[i] = next;
#pragma unroll
      for (unsigned c = 0; c < kDimSteps; ++c)
        acc[c][i] *= alpha;
    }
    // A wave's LDS accesses complete in order; the barriers only keep the
    // compiler from moving them across the exchange.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (unsigned b = 0; b < kBlocks; ++b) {
      const Half16 probabilities = LoadHalf16(&p_lds[wave][row][b * 16]);
#pragma unroll
      for (unsigned c = 0; c < kDimSteps; ++c) {
        acc[c] = Wmma(probabilities,
                      LoadHalf16(vh + std::size_t{c * 16 + row} * padded +
                                 key0 + b * 16),
                      acc[c]);
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  }
#pragma unroll
  for (unsigned c = 0; c < kDimSteps; ++c) {
    const unsigned d = c * 16 + row;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
      const unsigned token = q0 + 2 * i + half;
      if (token < count && d < kHeadDim) {
        out[std::size_t{token} * kHidden + head * kHeadDim + d] =
            ToAct(acc[c][i] / l[i]);
      }
    }
  }
}

/// gelu_tanh(gate) * up, from fused [count][gate | up] rows, as
/// [count][kFfnPadded] rows.
__global__ void GeGlu(const float* gate_up, Act* output, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kFfnPadded)
    return;
  const std::size_t row = i / kFfnPadded;
  const unsigned j = i % kFfnPadded;
  if (j >= kFfn) {
    output[i] = Act(0);
    return;
  }
  const float x = gate_up[row * kGateUp + j];
  const float up = gate_up[row * kGateUp + kFfn + j];
  const float gelu =
      0.5F * x *
      (1.0F + tanhf(0.7978845608028654F * (x + 0.044715F * x * x * x)));
  output[i] = ToAct(gelu * up);
}

/// 3x3 mean over the patch grid times sqrt(1152), row-major pooled cells.
__global__ void Pool(const float* hidden, float* pooled, unsigned columns,
                     unsigned pooled_columns, unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned r = i / kHidden;
  const unsigned d = i % kHidden;
  const unsigned px = (r % pooled_columns) * kPoolSize;
  const unsigned py = (r / pooled_columns) * kPoolSize;
  float sum = 0;
  for (unsigned y = 0; y < kPoolSize; ++y) {
    for (unsigned x = 0; x < kPoolSize; ++x)
      sum += hidden[(std::size_t{py + y} * columns + px + x) * kHidden + d];
  }
  pooled[i] = sum / (kPoolSize * kPoolSize) * sqrtf(float{kHidden});
}

/// rms((pooled - bias) * scale) without weight, as binary16.
__global__ void StandardizeNorm(const float* pooled, const float* bias,
                                const float* scale, Act* output) {
  __shared__ float shared[kThreads];
  __shared__ float values[kHidden];
  const std::size_t base = std::size_t{blockIdx.x} * kHidden;
  float sum = 0;
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads) {
    values[j] = (pooled[base + j] - bias[j]) * scale[j];
    sum += values[j] * values[j];
  }
  const float factor = rsqrtf(BlockSum(sum, shared) / kHidden + kEps);
  for (unsigned j = threadIdx.x; j < kHidden; j += kThreads)
    output[base + j] = ToAct(values[j] * factor);
}

}  // namespace

struct Encoder::Impl {
  struct Layer {
    Allocation qkv, out, gate_up, down;
    Allocation ln1, ln2, q_norm, k_norm, attn_post_norm, ffn_post_norm;
  };

  std::unique_ptr<core::GgufReader> reader;
  Weights weights;
  std::string identity;
  std::once_flag identity_once;
  std::atomic<std::size_t> resident_bytes{0};
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<gufo::hip::HipblasLtGemm> lt;
  bool uploaded{false};
  Allocation patch, position, std_bias, std_scale, projection;
  std::vector<Layer> layers;

  static std::unique_ptr<core::GgufReader> OpenReader(
      const std::filesystem::path& path) {
    std::string error;
    auto reader = core::GgufReader::OpenFile(path, &error);
    if (!reader)
      throw std::invalid_argument(error);
    return reader;
  }

  Impl(const std::filesystem::path& path, std::uint32_t width)
      : reader(OpenReader(path)), weights(Weights::Resolve(*reader, width)) {}

  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamSynchronize(stream);
    lt.reset();
    layers.clear();
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  std::size_t uploaded_bytes{0};

  Allocation Upload(std::initializer_list<const core::GgufTensorInfo*> parts) {
    std::size_t bytes = 0;
    for (const auto* part : parts)
      bytes += part->size_bytes;
    Allocation buffer(bytes);
    std::size_t offset = 0;
    for (const auto* part : parts) {
      Check(hipMemcpyAsync(static_cast<char*>(buffer.data) + offset, part->data,
                           part->size_bytes, hipMemcpyHostToDevice, stream));
      offset += part->size_bytes;
    }
    uploaded_bytes += bytes;
    return buffer;
  }

  /// A BF16 GEMM weight, uploaded and converted to binary16.
  Allocation UploadHalf(
      std::initializer_list<const core::GgufTensorInfo*> parts) {
    std::size_t bytes = 0;
    for (const auto* part : parts)
      bytes += part->size_bytes;
    Allocation buffer = Upload(parts);
    const std::size_t count = bytes / 2;
    Bf16ToHalf<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<std::uint16_t*>(buffer.data), count);
    return buffer;
  }

  /// A [rows][cols] BF16 GEMM weight as binary16 rows of `padded` values,
  /// zero past `cols`.
  Allocation UploadHalfPadded(const core::GgufTensorInfo& part, unsigned rows,
                              unsigned cols, unsigned padded) {
    const std::size_t count = std::size_t{rows} * padded;
    Allocation buffer(count * 2);
    Check(hipMemsetAsync(buffer.data, 0, count * 2, stream));
    Check(hipMemcpy2DAsync(buffer.data, std::size_t{padded} * 2, part.data,
                           std::size_t{cols} * 2, std::size_t{cols} * 2, rows,
                           hipMemcpyHostToDevice, stream));
    Bf16ToHalf<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<std::uint16_t*>(buffer.data), count);
    uploaded_bytes += count * 2;
    return buffer;
  }

  void UploadWeights() {
    if (uploaded)
      return;
    if (stream == nullptr)
      Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (blas == nullptr)
      CheckBlas(hipblasCreate(&blas));
    CheckBlas(hipblasSetStream(blas, stream));
    lt = std::make_unique<gufo::hip::HipblasLtGemm>(
        gufo::hip::HipblasLtGemmOptions{.ignore_environment = true});
    uploaded_bytes = 0;
    patch = Upload({weights.patch});
    position = Upload({weights.position});
    std_bias = Upload({weights.std_bias});
    std_scale = Upload({weights.std_scale});
    projection = UploadHalf({weights.projection});
    layers.clear();
    layers.resize(kLayers);
    for (unsigned l = 0; l < kLayers; ++l) {
      const LayerTensors& T = weights.layers[l];
      Layer& L = layers[l];
      L.qkv = UploadHalf({T.q, T.k, T.v});
      L.out = UploadHalf({T.out});
      L.gate_up = UploadHalf({T.gate, T.up});
      L.down = UploadHalfPadded(*T.down, kHidden, kFfn, kFfnPadded);
      L.ln1 = Upload({T.ln1});
      L.ln2 = Upload({T.ln2});
      L.q_norm = Upload({T.q_norm});
      L.k_norm = Upload({T.k_norm});
      L.attn_post_norm = Upload({T.attn_post_norm});
      L.ffn_post_norm = Upload({T.ffn_post_norm});
    }
    Check(hipStreamSynchronize(stream));
    resident_bytes.store(uploaded_bytes);
    uploaded = true;
  }

  /// y[rows][out] = x[rows][in] * W[out][in]^T with binary16 inputs.
  void Gemm(const Allocation& w, const Act* x, float* y, unsigned rows,
            unsigned out, unsigned in) {
    if (qwen38_flash_next::rocm::UnquantizedF16Gemm(
            w.data, reinterpret_cast<const __half*>(x), y, rows, out, in,
            stream))
      return;
    constexpr float alpha = 1, beta = 0;
    CheckBlas(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, out, rows, in,
                            &alpha, w.data, HIP_R_16F, in, x, HIP_R_16F, in,
                            &beta, y, HIP_R_32F, out, HIPBLAS_COMPUTE_32F,
                            HIPBLAS_GEMM_DEFAULT));
  }
};

Encoder::Encoder(const std::filesystem::path& path, std::uint32_t width)
    : impl_(std::make_unique<Impl>(path, width)) {}
Encoder::~Encoder() = default;

std::shared_ptr<Encoder> Encoder::Open(const std::filesystem::path& target,
                                       const std::filesystem::path& sidecar,
                                       std::uint32_t width) {
  if (!sidecar.empty())
    return std::make_shared<Encoder>(sidecar, width);
  auto directory = target.parent_path();
  for (unsigned level = 0; level < 2; ++level) {
    const auto path = directory / "mmproj-BF16.gguf";
    if (std::filesystem::is_regular_file(path))
      return std::make_shared<Encoder>(path, width);
    directory = directory.parent_path();
  }
  return {};
}

const std::string& Encoder::identity() const {
  std::call_once(impl_->identity_once, [&] {
    const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> hash(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!hash || EVP_DigestInit_ex(hash.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("cannot initialize vision fingerprint");
    for (const auto& region : impl_->reader->GetMappedRegions()) {
      if (EVP_DigestUpdate(hash.get(), region.data, region.size) != 1)
        throw std::runtime_error("cannot hash vision projector");
    }
    std::array<unsigned char, 32> digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(hash.get(), digest.data(), &length) != 1 ||
        length != digest.size())
      throw std::runtime_error("cannot finish vision fingerprint");
    constexpr std::string_view hex = "0123456789abcdef";
    impl_->identity.reserve(digest.size() * 2);
    for (const auto byte : digest) {
      impl_->identity.push_back(hex[byte >> 4]);
      impl_->identity.push_back(hex[byte & 15]);
    }
  });
  return impl_->identity;
}

std::size_t Encoder::ResidentBytes() const noexcept {
  return impl_->resident_bytes.load();
}

Encoder::Embedding::~Embedding() {
  if (data_ != nullptr)
    (void)hipFree(data_);
}

std::shared_ptr<const Encoder::Embedding> Encoder::Encode(
    const core::Image& image, const Observer& observer,
    const CancellationCheck& is_cancelled) {
  const auto check_cancelled = [&] {
    if (is_cancelled && is_cancelled())
      throw std::runtime_error("vision encoding cancelled");
  };
  check_cancelled();
  if (image.width == 0 || image.height == 0 ||
      image.width % kSideMultiple != 0 || image.height % kSideMultiple != 0 ||
      image.width / kPatchSize > kPositions ||
      image.height / kPatchSize > kPositions ||
      std::uint64_t{image.width} * image.height > 16777216 ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument("vision encoder requires a resized RGB image");
  }
  auto& I = *impl_;
  I.UploadWeights();
  const auto stream = I.stream;
  const unsigned columns = image.width / kPatchSize;
  const unsigned patch_rows = image.height / kPatchSize;
  const unsigned count = columns * patch_rows;
  const unsigned pooled_columns = columns / kPoolSize;
  const unsigned rows = pooled_columns * (patch_rows / kPoolSize);
  const unsigned width = I.weights.output_width;

  // Rotary tables for every patch column and row, in double on the host.
  const unsigned positions = std::max(columns, patch_rows);
  std::vector<float> table(std::size_t{2} * positions * kRopePairs);
  for (unsigned p = 0; p < positions; ++p) {
    for (unsigned i = 0; i < kRopePairs; ++i) {
      const double angle =
          p * std::pow(double{kRopeTheta}, -2.0 * i / (kHeadDim / 2));
      table[p * kRopePairs + i] = static_cast<float>(std::cos(angle));
      table[(positions + p) * kRopePairs + i] =
          static_cast<float>(std::sin(angle));
    }
  }

  const unsigned padded = (count + kKeyTile - 1) / kKeyTile * kKeyTile;
  Allocation pixels(image.pixels.size());
  Allocation rope(table.size() * sizeof(float));
  Allocation hidden(std::size_t{count} * kHidden * sizeof(float));
  Allocation work(std::size_t{count} *
                  std::max({kGateUp, kQkv, kPatchValues}) * sizeof(float));
  Allocation normed(std::size_t{count} * kFfnPadded * sizeof(Act));
  // Q, K and transposed V; padding stays zero.
  const std::size_t head_values = std::size_t{kHeads} * padded * kPaddedDim;
  Allocation qkv(3 * head_values * sizeof(_Float16));
  Check(hipMemsetAsync(qkv.data, 0, 3 * head_values * sizeof(_Float16),
                       stream));
  Allocation attention(std::size_t{count} * kHidden * sizeof(Act));
  Allocation pooled(std::size_t{rows} * kHidden * sizeof(float));
  Check(hipMemcpyAsync(pixels.data, image.pixels.data(), image.pixels.size(),
                       hipMemcpyHostToDevice, stream));
  Check(hipMemcpyAsync(rope.data, table.data(), table.size() * sizeof(float),
                       hipMemcpyHostToDevice, stream));
  const float* cos = rope.As<float>();
  const float* sin = cos + std::size_t{positions} * kRopePairs;

  const auto observe = [&](const std::string& stage, const float* data,
                           std::size_t size) {
    if (!observer)
      return;
    std::vector<float> copy(size);
    Check(hipMemcpyAsync(copy.data(), data, size * sizeof(float),
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    observer(stage, copy);
  };

  // Patch embedding in FP32, as the F32 kernel is stored.
  Patchify<<<Blocks(std::size_t{count} * kPatchValues), kThreads, 0, stream>>>(
      pixels.As<std::uint8_t>(), work.As<float>(), image.width, columns, count);
  constexpr float one = 1, zero = 0;
  CheckBlas(hipblasSgemm(I.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, kHidden, count,
                         kPatchValues, &one, I.patch.As<float>(), kPatchValues,
                         work.As<float>(), kPatchValues, &zero,
                         hidden.As<float>(), kHidden));
  AddPositions<<<Blocks(std::size_t{count} * kHidden), kThreads, 0, stream>>>(
      hidden.As<float>(), I.position.As<float>(), columns, count);
  observe("patch", hidden.As<float>(), std::size_t{count} * kHidden);

  auto* q = qkv.As<_Float16>();
  auto* k = q + head_values;
  auto* vt = k + head_values;
  const dim3 attention_grid(
      (padded / 16 + kAttentionWaves - 1) / kAttentionWaves, kHeads);
  for (unsigned l = 0; l < kLayers; ++l) {
    check_cancelled();
    const Impl::Layer& L = I.layers[l];
    RmsNormHalf<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), L.ln1.As<float>(), normed.As<Act>(), kHidden);
    I.Gemm(L.qkv, normed.As<Act>(), work.As<float>(), count, kQkv, kHidden);
    QkvPost<<<count * kHeads, 96, 0, stream>>>(
        work.As<float>(), L.q_norm.As<float>(), L.k_norm.As<float>(), cos, sin,
        q, k, vt, columns, padded);
    FusedAttention<<<attention_grid, 32 * kAttentionWaves, 0, stream>>>(
        q, k, vt, attention.As<Act>(), count, padded);
    I.Gemm(L.out, attention.As<Act>(), work.As<float>(), count, kHidden,
           kHidden);
    PostNormAdd<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), work.As<float>(), L.attn_post_norm.As<float>());
    RmsNormHalf<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), L.ln2.As<float>(), normed.As<Act>(), kHidden);
    I.Gemm(L.gate_up, normed.As<Act>(), work.As<float>(), count, kGateUp,
           kHidden);
    GeGlu<<<Blocks(std::size_t{count} * kFfnPadded), kThreads, 0, stream>>>(
        work.As<float>(), normed.As<Act>(), count);
    I.Gemm(L.down, normed.As<Act>(), work.As<float>(), count, kHidden,
           kFfnPadded);
    PostNormAdd<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), work.As<float>(), L.ffn_post_norm.As<float>());
    observe("layer" + std::to_string(l), hidden.As<float>(),
            std::size_t{count} * kHidden);
  }

  Pool<<<Blocks(std::size_t{rows} * kHidden), kThreads, 0, stream>>>(
      hidden.As<float>(), pooled.As<float>(), columns, pooled_columns, rows);
  observe("pooled", pooled.As<float>(), std::size_t{rows} * kHidden);
  StandardizeNorm<<<rows, kThreads, 0, stream>>>(
      pooled.As<float>(), I.std_bias.As<float>(), I.std_scale.As<float>(),
      normed.As<Act>());
  auto embedding = std::shared_ptr<Embedding>(new Embedding);
  embedding->rows_ = rows;
  embedding->width_ = width;
  Check(hipMalloc(&embedding->data_,
                  std::size_t{rows} * width * sizeof(float)));
  I.Gemm(I.projection, normed.As<Act>(), embedding->data_, rows, width,
         kHidden);
  Check(hipGetLastError());
  Check(hipStreamSynchronize(stream));
  check_cancelled();
  observe("embedding", embedding->data_, std::size_t{rows} * width);
  return embedding;
}

}  // namespace gufo::models::gemma4::vision
