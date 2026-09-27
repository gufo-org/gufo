#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/vision/encoder.hpp"

namespace gufo::models::qwen::vision {
namespace {
using Bf16 = hip_bfloat16;
constexpr unsigned kHidden = 1152;
constexpr unsigned kHeads = 16;
constexpr unsigned kHeadDim = 72;
constexpr unsigned kFfn = 4304;
constexpr unsigned kMergeHidden = kHidden * 4;
constexpr unsigned kPatchDim = 3 * 2 * 16 * 16;
constexpr unsigned kThreads = 256;

void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
void CheckBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("vision hipBLAS failure: " +
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

__device__ __forceinline__ float Round(float x) {
  return static_cast<float>(Bf16(x));
}

__global__ void Patchify(const std::uint8_t* pixels, Bf16* patches,
                         unsigned width, unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kPatchDim)
    return;
  const unsigned patch = i / kPatchDim;
  const unsigned element = i % kPatchDim;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const unsigned channel = element / (2 * 16 * 16);
  const unsigned spatial = element % (16 * 16);
  const unsigned pixel =
      ((y * 16 + spatial / 16) * width + x * 16 + spatial % 16) * 3;
  // Match rescale_and_normalize: multiply by 1/255 then subtract/divide.
  patches[i] = Bf16(
      (static_cast<float>(pixels[pixel + channel]) * (1.0F / 255.0F) - 0.5F) /
      0.5F);
}

__global__ void PatchPosition(float* hidden, const float* bias,
                              const float* positions, unsigned grid_height,
                              unsigned grid_width, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned patch = i / kHidden;
  const unsigned dim = i % kHidden;
  const unsigned merge = patch / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (patch % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + patch % 2;
  const float fy = static_cast<float>(y) * (47.0F / (grid_height - 1));
  const float fx = static_cast<float>(x) * (47.0F / (grid_width - 1));
  const unsigned y0 = min(static_cast<unsigned>(fy), 47U);
  const unsigned x0 = min(static_cast<unsigned>(fx), 47U);
  const unsigned y1 = min(y0 + 1, 47U);
  const unsigned x1 = min(x0 + 1, 47U);
  const float wy = fy - y0;
  const float wx = fx - x0;
  // The reference sums four interpolation products in this order.
  float position =
      positions[(y0 * 48 + x0) * kHidden + dim] * ((1 - wy) * (1 - wx));
  position += positions[(y0 * 48 + x1) * kHidden + dim] * ((1 - wy) * wx);
  position += positions[(y1 * 48 + x0) * kHidden + dim] * (wy * (1 - wx));
  position += positions[(y1 * 48 + x1) * kHidden + dim] * (wy * wx);
  hidden[i] = Round(Round(Round(hidden[i]) + bias[dim]) + Round(position));
}

__global__ void LayerNorm(const float* input, Bf16* output, const float* weight,
                          const float* bias, unsigned width) {
  const unsigned row = blockIdx.x;
  __shared__ float sum[kThreads];
  float value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads)
    value += input[row * width + j];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = sum[0] / width;
  __syncthreads();
  value = 0;
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    const float centered = input[row * width + j] - mean;
    value += centered * centered;
  }
  sum[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sum[0] / width + 1e-6F);
  for (unsigned j = threadIdx.x; j < width; j += kThreads) {
    output[row * width + j] =
        Bf16((input[row * width + j] - mean) * scale * weight[j] + bias[j]);
  }
}

__global__ void BiasResidual(float* hidden, const float* projection,
                             const float* bias, unsigned size, unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    hidden[i] = Round(hidden[i] + Round(projection[i] + bias[i % width]));
}

__global__ void Activate(const float* input, const float* bias, Bf16* output,
                         unsigned size, unsigned width, bool approximate) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  const float x = Round(input[i] + bias[i % width]);
  const float gelu = approximate
                         ? 0.5F * x *
                               (1.0F + tanhf(0.7978845608028654F *
                                             (x + 0.044715F * x * x * x)))
                         : 0.5F * x * (1.0F + erff(x * 0.7071067811865475F));
  output[i] = Bf16(gelu);
}

__global__ void QkvRope(const float* input, const float* bias, Bf16* q, Bf16* k,
                        Bf16* v, unsigned count, unsigned grid_width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  const unsigned pair = d % (kHeadDim / 2);
  const unsigned merge = token / 4;
  const unsigned y = (merge / (grid_width / 2)) * 2 + (token % 4) / 2;
  const unsigned x = (merge % (grid_width / 2)) * 2 + token % 2;
  const unsigned position = pair < kHeadDim / 4 ? y : x;
  const float angle =
      position / powf(10000.0F, static_cast<float>(pair % (kHeadDim / 4)) /
                                    (kHeadDim / 4));
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const unsigned first = head * kHeadDim + pair;
  const unsigned base = token * 3 * kHidden;
  const float q0 = Round(input[base + first] + bias[first]);
  const float q1 =
      Round(input[base + first + kHeadDim / 2] + bias[first + kHeadDim / 2]);
  const float k0 = Round(input[base + kHidden + first] + bias[kHidden + first]);
  const float k1 = Round(input[base + kHidden + first + kHeadDim / 2] +
                         bias[kHidden + first + kHeadDim / 2]);
  const unsigned target = (head * count + token) * kHeadDim + d;
  q[target] = Bf16(d < kHeadDim / 2 ? q0 * cosine - q1 * sine
                                    : q1 * cosine + q0 * sine);
  k[target] = Bf16(d < kHeadDim / 2 ? k0 * cosine - k1 * sine
                                    : k1 * cosine + k0 * sine);
  v[target] = Bf16(input[base + 2 * kHidden + head * kHeadDim + d] +
                   bias[2 * kHidden + head * kHeadDim + d]);
}

// Keep bounded rows in registers through normalization. Larger rows retain
// the streaming path; caching them would spill and reread private memory.
template<unsigned kValues>
__global__ void Softmax(const float* scores, Bf16* probabilities,
                        unsigned length) {
  const unsigned row = blockIdx.x;
  __shared__ float reduce[kThreads];
  float values[kValues != 0 ? kValues : 1];
  float maximum = -INFINITY;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      values[j] =
          i < length
              ? Round(Round(scores[row * length + i]) * 0.1178511301977579F)
              : -INFINITY;
      maximum = fmaxf(maximum, values[j]);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      maximum = fmaxf(maximum, Round(Round(scores[row * length + i]) *
                                     0.1178511301977579F));
    }
  }
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] =
          fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float sum = 0;
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      if (threadIdx.x + j * kThreads < length) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
        // Limit overlapping exponential temporaries without changing their
        // order. Wider shapes otherwise spill registers on gfx1151.
        if constexpr (kValues > 16)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads)
      sum += expf(Round(Round(scores[row * length + i]) * 0.1178511301977579F) -
                  maximum);
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / reduce[0];
  if constexpr (kValues != 0) {
#pragma unroll
    for (unsigned j = 0; j < kValues; ++j) {
      const unsigned i = threadIdx.x + j * kThreads;
      if (i < length)
        probabilities[row * length + i] = Bf16(values[j] * inverse);
    }
  } else {
    for (unsigned i = threadIdx.x; i < length; i += kThreads) {
      probabilities[row * length + i] = Bf16(
          expf(Round(Round(scores[row * length + i]) * 0.1178511301977579F) -
               maximum) *
          inverse);
    }
  }
}

__global__ void AttentionRows(const float* input, Bf16* output, unsigned begin,
                              unsigned rows) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kHidden)
    return;
  const unsigned token = i / kHidden;
  const unsigned head = (i % kHidden) / kHeadDim;
  const unsigned d = i % kHeadDim;
  output[(begin + token) * kHidden + head * kHeadDim + d] =
      Bf16(input[(head * rows + token) * kHeadDim + d]);
}

__global__ void Finish(float* output, const float* bias, unsigned size,
                       unsigned width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    output[i] = Round(output[i] + bias[i % width]);
}

std::uint16_t ToBf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
}  // namespace

struct Encoder::Impl {
  std::unique_ptr<core::GgufReader> reader;
  std::uint32_t output_width;
  std::string identity;
  std::once_flag identity_once;
  std::atomic<std::size_t> resident_bytes{0};
  std::unordered_map<std::string, Allocation> weights;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<gufo::hip::HipblasLtGemm> lt;
  bool uploaded{false};

  Impl(const std::filesystem::path& path, std::uint32_t width)
      : output_width(width) {
    std::string error;
    reader = core::GgufReader::OpenFile(path, &error);
    if (!reader)
      throw std::invalid_argument(error);
    const auto expect = [&](std::string_view key, std::uint64_t value) {
      if (reader->GetMetadataUint64(key) != value) {
        throw std::invalid_argument("incompatible Qwen vision metadata: " +
                                    std::string(key));
      }
    };
    expect("clip.vision.projection_dim", width);
    expect("clip.vision.patch_size", 16);
    expect("clip.vision.embedding_length", kHidden);
    expect("clip.vision.feed_forward_length", kFfn);
    expect("clip.vision.block_count", 27);
    expect("clip.vision.attention.head_count", kHeads);
    expect("clip.vision.spatial_merge_size", 2);
    expect("clip.vision.image_size", 768);
    if ((width != 5120 && width != 2560) ||
        reader->GetMetadataString("general.architecture") != "clip" ||
        reader->GetMetadataString("general.type") != "mmproj" ||
        reader->GetMetadataBool("clip.has_vision_encoder") != true ||
        reader->GetMetadataBool("clip.use_gelu") != true ||
        reader->GetMetadataFloat32(
            "clip.vision.attention.layer_norm_epsilon") != 1e-6F) {
      throw std::invalid_argument(
          "vision sidecar does not match the Qwen3.8 model");
    }
    for (const auto key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
      const auto* value = reader->FindMetadata(key);
      const auto* array =
          value ? std::get_if<std::vector<double>>(&value->value) : nullptr;
      if (!array || *array != std::vector<double>{0.5, 0.5, 0.5}) {
        throw std::invalid_argument("unsupported Qwen image normalization");
      }
    }
    const auto* deepstack_value =
        reader->FindMetadata("clip.vision.is_deepstack_layers");
    const auto* deepstack =
        deepstack_value
            ? std::get_if<std::vector<std::uint64_t>>(&deepstack_value->value)
            : nullptr;
    if (!deepstack || deepstack->size() != 27 ||
        std::any_of(deepstack->begin(), deepstack->end(),
                    [](auto enabled) { return enabled != 0; })) {
      throw std::invalid_argument(
          "Qwen3.8 vision requires a final-layer-only merger");
    }
    if (reader->GetMetadataString("clip.projector_type") != "qwen3vl_merger") {
      throw std::invalid_argument("unsupported Qwen vision projector");
    }
    const auto require = [&](const std::string& name,
                             std::initializer_list<std::uint64_t> shape,
                             core::GgmlType type) {
      const auto* tensor = reader->FindTensor(name);
      if (tensor == nullptr ||
          tensor->dimensions != std::vector<std::uint64_t>(shape) ||
          tensor->type != type) {
        throw std::invalid_argument("invalid BF16 Qwen vision tensor: " + name);
      }
      const auto bytes =
          tensor->ElementCount() * (type == core::GgmlType::kF32 ? 4 : 2);
      const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
      const bool mapped = std::any_of(
          reader->GetMappedRegions().begin(), reader->GetMappedRegions().end(),
          [&](const auto& region) {
            const auto begin = reinterpret_cast<std::uintptr_t>(region.data);
            return address >= begin && address - begin <= region.size &&
                   bytes <= region.size - (address - begin);
          });
      if (!mapped)
        throw std::invalid_argument("truncated Qwen vision tensor: " + name);
    };
    for (unsigned layer = 0; layer < 27; ++layer) {
      const auto prefix = "v.blk." + std::to_string(layer) + ".";
      for (const auto* norm : {"ln1", "ln2"}) {
        require(prefix + norm + ".weight", {kHidden}, core::GgmlType::kF32);
        require(prefix + norm + ".bias", {kHidden}, core::GgmlType::kF32);
      }
      for (const auto& [name, in, out] :
           std::array{std::tuple{"attn_qkv", kHidden, 3 * kHidden},
                      std::tuple{"attn_out", kHidden, kHidden},
                      std::tuple{"ffn_up", kHidden, kFfn},
                      std::tuple{"ffn_down", kFfn, kHidden}}) {
        require(prefix + name + ".weight", {in, out}, core::GgmlType::kBF16);
        require(prefix + name + ".bias", {out}, core::GgmlType::kF32);
      }
    }
    require("v.patch_embd.weight", {16, 16, 3, kHidden}, core::GgmlType::kF32);
    require("v.patch_embd.weight.1", {16, 16, 3, kHidden},
            core::GgmlType::kF32);
    require("v.patch_embd.bias", {kHidden}, core::GgmlType::kF32);
    require("v.position_embd.weight", {kHidden, 2304}, core::GgmlType::kF32);
    require("v.post_ln.weight", {kHidden}, core::GgmlType::kF32);
    require("v.post_ln.bias", {kHidden}, core::GgmlType::kF32);
    require("mm.0.weight", {kMergeHidden, kMergeHidden}, core::GgmlType::kBF16);
    require("mm.0.bias", {kMergeHidden}, core::GgmlType::kF32);
    require("mm.2.weight", {kMergeHidden, width}, core::GgmlType::kBF16);
    require("mm.2.bias", {width}, core::GgmlType::kF32);
    if (reader->GetTensorCount() != 334) {
      throw std::invalid_argument(
          "unexpected tensors in Qwen vision projector");
    }
  }

  ~Impl() {
    if (stream != nullptr)
      (void)hipStreamSynchronize(stream);
    lt.reset();
    weights.clear();
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  void Upload() {
    if (uploaded)
      return;
    if (stream == nullptr)
      Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    Check(hipStreamSynchronize(stream));
    weights.clear();
    resident_bytes.store(0);
    std::size_t bytes_uploaded = 0;
    if (blas == nullptr)
      CheckBlas(hipblasCreate(&blas));
    CheckBlas(hipblasSetStream(blas, stream));
    lt = std::make_unique<gufo::hip::HipblasLtGemm>(
        gufo::hip::HipblasLtGemmOptions{.ignore_environment = true});
    for (const auto& tensor : reader->GetTensors()) {
      if (tensor.name == "v.patch_embd.weight" ||
          tensor.name == "v.patch_embd.weight.1")
        continue;
      const std::size_t bytes =
          tensor.ElementCount() * (tensor.type == core::GgmlType::kF32 ? 4 : 2);
      Allocation buffer(bytes);
      bytes_uploaded += bytes;
      Check(hipMemcpyAsync(buffer.data, tensor.data, bytes,
                           hipMemcpyHostToDevice, stream));
      weights.emplace(tensor.name, std::move(buffer));
    }
    const auto* first = static_cast<const float*>(
        reader->FindTensor("v.patch_embd.weight")->data);
    const auto* second = static_cast<const float*>(
        reader->FindTensor("v.patch_embd.weight.1")->data);
    std::vector<std::uint16_t> patch(kHidden * kPatchDim);
    for (unsigned h = 0; h < kHidden; ++h) {
      for (unsigned c = 0; c < 3; ++c) {
        for (unsigned t = 0; t < 2; ++t) {
          for (unsigned p = 0; p < 256; ++p) {
            patch[h * kPatchDim + (c * 2 + t) * 256 + p] =
                ToBf16((t == 0 ? first : second)[(h * 3 + c) * 256 + p]);
          }
        }
      }
    }
    Allocation buffer(patch.size() * sizeof(std::uint16_t));
    Check(hipMemcpyAsync(buffer.data, patch.data(),
                         patch.size() * sizeof(std::uint16_t),
                         hipMemcpyHostToDevice, stream));
    weights.emplace("patch", std::move(buffer));
    Check(hipStreamSynchronize(stream));
    resident_bytes.store(bytes_uploaded + patch.size() * sizeof(std::uint16_t));
    uploaded = true;
  }

  const float* FloatWeight(const std::string& name) const {
    return weights.at(name).As<float>();
  }
  void Gemm(const std::string& weight, const Bf16* input, float* output,
            unsigned rows, unsigned out, unsigned in) {
    const auto* w = weights.at(weight).data;
    if (lt->RunBf16(w, input, output, rows, out, in, stream))
      return;
    constexpr float alpha = 1, beta = 0;
    CheckBlas(hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, out, rows, in,
                            &alpha, w, HIP_R_16BF, in, input, HIP_R_16BF, in,
                            &beta, output, HIP_R_32F, out, HIPBLAS_COMPUTE_32F,
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
    // Hash the complete projector once. OpenSSL uses the CPU's SHA
    // instructions, avoiding seconds of scalar hashing on the first image.
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
  if (image.width == 0 || image.height == 0 || image.width % 32 != 0 ||
      image.height % 32 != 0 ||
      std::uint64_t{image.width} * image.height > 16777216 ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument("vision encoder requires a resized RGB32 grid");
  }
  impl_->Upload();
  const auto stream = impl_->stream;
  const unsigned count = (image.width / 16) * (image.height / 16);
  const unsigned merged = count / 4;
  Allocation pixels(image.pixels.size());
  Allocation hidden(std::size_t{count} * kHidden * sizeof(float));
  Allocation work(std::size_t{count} * std::max(kFfn, 3 * kHidden) *
                  sizeof(float));
  Allocation normalized(std::size_t{count} * std::max(kFfn, kPatchDim) *
                        sizeof(Bf16));
  Allocation qkv(std::size_t{3} * count * kHidden * sizeof(Bf16));
  Allocation attention(std::size_t{count} * kHidden * sizeof(Bf16));
  // Smaller attention tiles reduce scratch traffic for the 4096-patch
  // shape while preserving its BLAS reductions. Ragged tails require
  // their existing GEMM shapes to preserve rounding.
  const unsigned tile = std::min(count, count == 4096 ? 64U : 128U);
  Allocation scores(std::size_t{kHeads} * tile * count * sizeof(float));
  Allocation probabilities(std::size_t{kHeads} * tile * count * sizeof(Bf16));
  Allocation output_tile(std::size_t{tile} * kHidden * sizeof(float));
  Check(hipMemcpyAsync(pixels.data, image.pixels.data(), image.pixels.size(),
                       hipMemcpyHostToDevice, stream));
  Patchify<<<(count * kPatchDim + kThreads - 1) / kThreads, kThreads, 0,
             stream>>>(pixels.As<std::uint8_t>(), normalized.As<Bf16>(),
                       image.width, image.width / 16, count);
  impl_->Gemm("patch", normalized.As<Bf16>(), hidden.As<float>(), count,
              kHidden, kPatchDim);
  PatchPosition<<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                  stream>>>(hidden.As<float>(),
                            impl_->FloatWeight("v.patch_embd.bias"),
                            impl_->FloatWeight("v.position_embd.weight"),
                            image.height / 16, image.width / 16, count);
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
  observe("patch", hidden.As<float>(), std::size_t{count} * kHidden);
  for (unsigned layer = 0; layer < 27; ++layer) {
    check_cancelled();
    const auto prefix = "v.blk." + std::to_string(layer) + ".";
    LayerNorm<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), normalized.As<Bf16>(),
        impl_->FloatWeight(prefix + "ln1.weight"),
        impl_->FloatWeight(prefix + "ln1.bias"), kHidden);
    impl_->Gemm(prefix + "attn_qkv.weight", normalized.As<Bf16>(),
                work.As<float>(), count, 3 * kHidden, kHidden);
    auto* q = qkv.As<Bf16>();
    auto* k = q + count * kHidden;
    auto* v = k + count * kHidden;
    QkvRope<<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
              stream>>>(work.As<float>(),
                        impl_->FloatWeight(prefix + "attn_qkv.bias"), q, k, v,
                        count, image.width / 16);
    for (unsigned begin = 0; begin < count; begin += tile) {
      check_cancelled();
      const unsigned rows = std::min(tile, count - begin);
      constexpr float zero = 0, one = 1;
      CheckBlas(hipblasGemmStridedBatchedEx(
          impl_->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, count, rows, kHeadDim, &one,
          k, HIP_R_16BF, kHeadDim, count * kHeadDim, q + begin * kHeadDim,
          HIP_R_16BF, kHeadDim, count * kHeadDim, &zero, scores.data, HIP_R_32F,
          count, rows * count, kHeads, HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));
      if (count <= 1024)
        Softmax<4><<<kHeads * rows, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<Bf16>(), count);
      else if (count <= 2048)
        Softmax<8><<<kHeads * rows, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<Bf16>(), count);
      else if (count <= 4096)
        Softmax<16><<<kHeads * rows, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<Bf16>(), count);
      else if (count <= 8192)
        Softmax<32><<<kHeads * rows, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<Bf16>(), count);
      else
        Softmax<0><<<kHeads * rows, kThreads, 0, stream>>>(
            scores.As<float>(), probabilities.As<Bf16>(), count);
      CheckBlas(hipblasGemmStridedBatchedEx(
          impl_->blas, HIPBLAS_OP_N, HIPBLAS_OP_N, kHeadDim, rows, count, &one,
          v, HIP_R_16BF, kHeadDim, count * kHeadDim, probabilities.data,
          HIP_R_16BF, count, rows * count, &zero, output_tile.data, HIP_R_32F,
          kHeadDim, rows * kHeadDim, kHeads, HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));
      AttentionRows<<<(rows * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                      stream>>>(output_tile.As<float>(), attention.As<Bf16>(),
                                begin, rows);
    }
    impl_->Gemm(prefix + "attn_out.weight", attention.As<Bf16>(),
                work.As<float>(), count, kHidden, kHidden);
    BiasResidual<<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                   stream>>>(hidden.As<float>(), work.As<float>(),
                             impl_->FloatWeight(prefix + "attn_out.bias"),
                             count * kHidden, kHidden);
    LayerNorm<<<count, kThreads, 0, stream>>>(
        hidden.As<float>(), normalized.As<Bf16>(),
        impl_->FloatWeight(prefix + "ln2.weight"),
        impl_->FloatWeight(prefix + "ln2.bias"), kHidden);
    impl_->Gemm(prefix + "ffn_up.weight", normalized.As<Bf16>(),
                work.As<float>(), count, kFfn, kHidden);
    Activate<<<(count * kFfn + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        work.As<float>(), impl_->FloatWeight(prefix + "ffn_up.bias"),
        normalized.As<Bf16>(), count * kFfn, kFfn, true);
    impl_->Gemm(prefix + "ffn_down.weight", normalized.As<Bf16>(),
                work.As<float>(), count, kHidden, kFfn);
    BiasResidual<<<(count * kHidden + kThreads - 1) / kThreads, kThreads, 0,
                   stream>>>(hidden.As<float>(), work.As<float>(),
                             impl_->FloatWeight(prefix + "ffn_down.bias"),
                             count * kHidden, kHidden);
    observe("layer" + std::to_string(layer), hidden.As<float>(),
            std::size_t{count} * kHidden);
  }
  LayerNorm<<<count, kThreads, 0, stream>>>(
      hidden.As<float>(), normalized.As<Bf16>(),
      impl_->FloatWeight("v.post_ln.weight"),
      impl_->FloatWeight("v.post_ln.bias"), kHidden);
  impl_->Gemm("mm.0.weight", normalized.As<Bf16>(), work.As<float>(), merged,
              kMergeHidden, kMergeHidden);
  Activate<<<(merged * kMergeHidden + kThreads - 1) / kThreads, kThreads, 0,
             stream>>>(work.As<float>(), impl_->FloatWeight("mm.0.bias"),
                       normalized.As<Bf16>(), merged * kMergeHidden,
                       kMergeHidden, false);
  auto embedding = std::shared_ptr<Embedding>(new Embedding);
  embedding->rows_ = merged;
  embedding->width_ = impl_->output_width;
  Check(hipMalloc(&embedding->data_,
                  std::size_t{merged} * embedding->width_ * sizeof(float)));
  impl_->Gemm("mm.2.weight", normalized.As<Bf16>(), embedding->data_, merged,
              embedding->width_, kMergeHidden);
  Finish<<<(merged * embedding->width_ + kThreads - 1) / kThreads, kThreads, 0,
           stream>>>(embedding->data_, impl_->FloatWeight("mm.2.bias"),
                     merged * embedding->width_, embedding->width_);
  Check(hipGetLastError());
  Check(hipStreamSynchronize(stream));
  check_cancelled();
  observe("embedding", embedding->data_,
          std::size_t{merged} * embedding->width_);
  return embedding;
}
}  // namespace gufo::models::qwen::vision
