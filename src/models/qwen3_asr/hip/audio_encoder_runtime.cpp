#include "src/models/qwen3_asr/hip/audio_encoder_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <miopen/miopen.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen3_asr/hip/audio_ops.hpp"
#include "src/models/qwen3_asr/hip/blas.hpp"
#include "src/models/qwen3_asr/loader.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {

constexpr std::size_t kMelBins = 128;
constexpr std::size_t kChunkFrames = 100;
constexpr std::size_t kConvChannels = 480;
constexpr std::size_t kHidden = 1024;
constexpr std::size_t kFfn = 4096;
constexpr std::size_t kOutput = 2048;
constexpr std::size_t kHeads = 16;
constexpr std::size_t kHeadDim = 64;
constexpr float kLayerNormEpsilon = 1.0e-5F;
constexpr std::size_t kFlattened = 7680;
constexpr std::size_t kConvTime = 13;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void RequireHipblas(hipblasStatus_t status, std::string_view operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(status));
  }
}

void RequireMiopen(miopenStatus_t status, std::string_view operation) {
  if (status != miopenStatusSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             miopenGetErrorString(status));
  }
}

template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  void Reset(std::size_t count) {
    if (data_ != nullptr) {
      RequireHip(hipFree(data_), "hipFree Qwen3-ASR buffer");
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0U) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Qwen3-ASR HIP buffer size overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Qwen3-ASR buffer");
    count_ = count;
  }
  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

const Tensor& RequireTensor(const TensorStore& store, std::string_view name,
                            std::span<const std::uint64_t> shape) {
  const Tensor* tensor = store.Find(name);
  if (tensor == nullptr || tensor->dtype != DType::kBF16 ||
      !std::ranges::equal(tensor->shape, shape)) {
    throw std::runtime_error("unexpected Qwen3-ASR tensor: " +
                             std::string(name));
  }
  return *tensor;
}

DeviceBuffer<hip_bfloat16> UploadBfloat16Async(
    const TensorStore& store, std::string_view name,
    std::span<const std::uint64_t> shape, hipStream_t stream) {
  const Tensor& tensor = RequireTensor(store, name, shape);
  const auto elements = tensor.NumElements();
  if (!elements.has_value()) {
    throw std::runtime_error("invalid Qwen3-ASR tensor size: " +
                             std::string(name));
  }
  DeviceBuffer<hip_bfloat16> result(*elements);
  RequireHip(hipMemcpyAsync(result.get(), tensor.data, tensor.byte_count,
                            hipMemcpyHostToDevice, stream),
             "hipMemcpyAsync Qwen3-ASR weight");
  return result;
}

class TensorDescriptor {
public:
  TensorDescriptor() {
    RequireMiopen(miopenCreateTensorDescriptor(&value_),
                  "miopenCreateTensorDescriptor");
  }
  ~TensorDescriptor() {
    if (value_ != nullptr) {
      (void)miopenDestroyTensorDescriptor(value_);
    }
  }
  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor& operator=(const TensorDescriptor&) = delete;
  void Set(int n, int c, int h, int w) {
    RequireMiopen(
        miopenSet4dTensorDescriptor(value_, miopenBFloat16, n, c, h, w),
        "miopenSet4dTensorDescriptor");
  }
  [[nodiscard]] miopenTensorDescriptor_t get() const noexcept { return value_; }

private:
  miopenTensorDescriptor_t value_{nullptr};
};

class Convolution {
public:
  Convolution(miopenHandle_t handle, int batch, int input_channels,
              int input_height, int input_width, int output_channels)
      : handle_(handle) {
    input_.Set(batch, input_channels, input_height, input_width);
    weights_.Set(output_channels, input_channels, 3, 3);
    output_.Set(batch, output_channels, (input_height + 1) / 2,
                (input_width + 1) / 2);
    RequireMiopen(miopenCreateConvolutionDescriptor(&convolution_),
                  "miopenCreateConvolutionDescriptor");
    RequireMiopen(miopenInitConvolutionDescriptor(
                      convolution_, miopenConvolution, 1, 1, 2, 2, 1, 1),
                  "miopenInitConvolutionDescriptor");
    std::size_t solution_count = 0;
    RequireMiopen(miopenConvolutionForwardGetSolutionCount(
                      handle_, weights_.get(), input_.get(), convolution_,
                      output_.get(), &solution_count),
                  "miopenConvolutionForwardGetSolutionCount");
    if (solution_count == 0U) {
      throw std::runtime_error("MIOpen found no Qwen3-ASR convolution");
    }
    std::vector<miopenConvSolution_t> solutions(solution_count);
    std::size_t returned = 0;
    RequireMiopen(
        miopenConvolutionForwardGetSolution(
            handle_, weights_.get(), input_.get(), convolution_, output_.get(),
            solutions.size(), &returned, solutions.data()),
        "miopenConvolutionForwardGetSolution");
    if (returned == 0U) {
      throw std::runtime_error("MIOpen returned no Qwen3-ASR convolution");
    }
    solution_ = solutions.front().solution_id;
    workspace_.Reset(
        (solutions.front().workspace_size + sizeof(std::byte) - 1U) /
        sizeof(std::byte));
  }

  ~Convolution() {
    if (convolution_ != nullptr) {
      (void)miopenDestroyConvolutionDescriptor(convolution_);
    }
  }
  Convolution(const Convolution&) = delete;
  Convolution& operator=(const Convolution&) = delete;

  void Run(const void* input, const void* weights, void* output) {
    RequireMiopen(miopenConvolutionForwardImmediate(
                      handle_, weights_.get(), weights, input_.get(), input,
                      convolution_, output_.get(), output, workspace_.get(),
                      workspace_.size(), solution_),
                  "miopenConvolutionForwardImmediate");
  }

private:
  miopenHandle_t handle_{nullptr};
  TensorDescriptor input_;
  TensorDescriptor weights_;
  TensorDescriptor output_;
  miopenConvolutionDescriptor_t convolution_{nullptr};
  std::uint64_t solution_{0};
  DeviceBuffer<std::byte> workspace_;
};

std::string LayerName(std::size_t layer, std::string_view suffix) {
  return "thinker.audio_tower.layers." + std::to_string(layer) + "." +
         std::string(suffix);
}

struct EncoderLayerWeights {
  DeviceBuffer<hip_bfloat16> q_weight;
  DeviceBuffer<hip_bfloat16> q_bias;
  DeviceBuffer<hip_bfloat16> k_weight;
  DeviceBuffer<hip_bfloat16> k_bias;
  DeviceBuffer<hip_bfloat16> v_weight;
  DeviceBuffer<hip_bfloat16> v_bias;
  DeviceBuffer<hip_bfloat16> out_weight;
  DeviceBuffer<hip_bfloat16> out_bias;
  DeviceBuffer<hip_bfloat16> attention_norm_weight;
  DeviceBuffer<hip_bfloat16> attention_norm_bias;
  DeviceBuffer<hip_bfloat16> fc1_weight;
  DeviceBuffer<hip_bfloat16> fc1_bias;
  DeviceBuffer<hip_bfloat16> fc2_weight;
  DeviceBuffer<hip_bfloat16> fc2_bias;
  DeviceBuffer<hip_bfloat16> final_norm_weight;
  DeviceBuffer<hip_bfloat16> final_norm_bias;
};

}  // namespace

struct AudioEncoderHipRuntime::Impl {
  explicit Impl(LoadResult loaded) : model(std::move(loaded)) {
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreate Qwen3-ASR audio encoder");
    RequireHipblas(hipblasCreate(&blas), "hipblasCreate Qwen3-ASR");
    lt = std::make_unique<GemmLt>();
    RequireHipblas(hipblasSetStream(blas, stream),
                   "hipblasSetStream Qwen3-ASR");
    RequireHipblas(hipblasSetAtomicsMode(blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
                   "hipblasSetAtomicsMode Qwen3-ASR");
    RequireMiopen(miopenCreate(&miopen), "miopenCreate Qwen3-ASR");
    RequireMiopen(miopenSetStream(miopen, stream), "miopenSetStream Qwen3-ASR");
    LoadWeights();
  }

  ~Impl() {
    if (miopen != nullptr) {
      (void)miopenDestroy(miopen);
    }
    if (blas != nullptr) {
      (void)hipblasDestroy(blas);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  DeviceBuffer<hip_bfloat16> UploadBfloat16(
      const TensorStore& store, std::string_view name,
      std::span<const std::uint64_t> shape) {
    return UploadBfloat16Async(store, name, shape, stream);
  }

  void LoadWeights() {
    const TensorStore& store = *model.store;
    conv1_weight = UploadBfloat16(store, "thinker.audio_tower.conv2d1.weight",
                                  std::array<std::uint64_t, 4>{480, 1, 3, 3});
    conv1_bias = UploadBfloat16(store, "thinker.audio_tower.conv2d1.bias",
                                std::array<std::uint64_t, 1>{480});
    conv2_weight = UploadBfloat16(store, "thinker.audio_tower.conv2d2.weight",
                                  std::array<std::uint64_t, 4>{480, 480, 3, 3});
    conv2_bias = UploadBfloat16(store, "thinker.audio_tower.conv2d2.bias",
                                std::array<std::uint64_t, 1>{480});
    conv3_weight = UploadBfloat16(store, "thinker.audio_tower.conv2d3.weight",
                                  std::array<std::uint64_t, 4>{480, 480, 3, 3});
    conv3_bias = UploadBfloat16(store, "thinker.audio_tower.conv2d3.bias",
                                std::array<std::uint64_t, 1>{480});
    conv_out = UploadBfloat16(store, "thinker.audio_tower.conv_out.weight",
                              std::array<std::uint64_t, 2>{1024, 7680});
  }

  void EnsureCapacity(std::size_t requested_chunks) {
    if (requested_chunks <= chunks) {
      return;
    }
    chunks = requested_chunks;
    feature_input.Reset(chunks * kMelBins * kChunkFrames);
    input.Reset(chunks * kMelBins * kChunkFrames);
    conv1.Reset(chunks * kConvChannels * 64U * 50U);
    conv2.Reset(chunks * kConvChannels * 32U * 25U);
    conv3.Reset(chunks * kConvChannels * 16U * kConvTime);
    layout.Reset(chunks * kConvTime * kFlattened);
    projected.Reset(chunks * kConvTime * kHidden);
    compact.Reset(chunks * kConvTime * kHidden);
    first = std::make_unique<Convolution>(miopen, static_cast<int>(chunks), 1,
                                          128, 100, 480);
    second = std::make_unique<Convolution>(miopen, static_cast<int>(chunks),
                                           480, 64, 50, 480);
    third = std::make_unique<Convolution>(miopen, static_cast<int>(chunks), 480,
                                          32, 25, 480);
  }

  void EnsureEncoderWeights() {
    if (!layers.empty()) {
      return;
    }
    const TensorStore& store = *model.store;
    layers.reserve(24);
    for (std::size_t layer = 0; layer < 24; ++layer) {
      EncoderLayerWeights weights;
      weights.q_weight =
          UploadBfloat16(store, LayerName(layer, "self_attn.q_proj.weight"),
                         std::array<std::uint64_t, 2>{kHidden, kHidden});
      weights.q_bias =
          UploadBfloat16(store, LayerName(layer, "self_attn.q_proj.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.k_weight =
          UploadBfloat16(store, LayerName(layer, "self_attn.k_proj.weight"),
                         std::array<std::uint64_t, 2>{kHidden, kHidden});
      weights.k_bias =
          UploadBfloat16(store, LayerName(layer, "self_attn.k_proj.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.v_weight =
          UploadBfloat16(store, LayerName(layer, "self_attn.v_proj.weight"),
                         std::array<std::uint64_t, 2>{kHidden, kHidden});
      weights.v_bias =
          UploadBfloat16(store, LayerName(layer, "self_attn.v_proj.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.out_weight =
          UploadBfloat16(store, LayerName(layer, "self_attn.out_proj.weight"),
                         std::array<std::uint64_t, 2>{kHidden, kHidden});
      weights.out_bias =
          UploadBfloat16(store, LayerName(layer, "self_attn.out_proj.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.attention_norm_weight =
          UploadBfloat16(store, LayerName(layer, "self_attn_layer_norm.weight"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.attention_norm_bias =
          UploadBfloat16(store, LayerName(layer, "self_attn_layer_norm.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.fc1_weight =
          UploadBfloat16(store, LayerName(layer, "fc1.weight"),
                         std::array<std::uint64_t, 2>{kFfn, kHidden});
      weights.fc1_bias = UploadBfloat16(store, LayerName(layer, "fc1.bias"),
                                        std::array<std::uint64_t, 1>{kFfn});
      weights.fc2_weight =
          UploadBfloat16(store, LayerName(layer, "fc2.weight"),
                         std::array<std::uint64_t, 2>{kHidden, kFfn});
      weights.fc2_bias = UploadBfloat16(store, LayerName(layer, "fc2.bias"),
                                        std::array<std::uint64_t, 1>{kHidden});
      weights.final_norm_weight =
          UploadBfloat16(store, LayerName(layer, "final_layer_norm.weight"),
                         std::array<std::uint64_t, 1>{kHidden});
      weights.final_norm_bias =
          UploadBfloat16(store, LayerName(layer, "final_layer_norm.bias"),
                         std::array<std::uint64_t, 1>{kHidden});
      layers.push_back(std::move(weights));
    }
    ln_post_weight = UploadBfloat16(store, "thinker.audio_tower.ln_post.weight",
                                    std::array<std::uint64_t, 1>{kHidden});
    ln_post_bias = UploadBfloat16(store, "thinker.audio_tower.ln_post.bias",
                                  std::array<std::uint64_t, 1>{kHidden});
    proj1_weight =
        UploadBfloat16(store, "thinker.audio_tower.proj1.weight",
                       std::array<std::uint64_t, 2>{kHidden, kHidden});
    proj1_bias = UploadBfloat16(store, "thinker.audio_tower.proj1.bias",
                                std::array<std::uint64_t, 1>{kHidden});
    proj2_weight =
        UploadBfloat16(store, "thinker.audio_tower.proj2.weight",
                       std::array<std::uint64_t, 2>{kOutput, kHidden});
    proj2_bias = UploadBfloat16(store, "thinker.audio_tower.proj2.bias",
                                std::array<std::uint64_t, 1>{kOutput});
  }

  void EnsureTransformerCapacity(std::size_t requested_tokens) {
    if (requested_tokens <= token_capacity) {
      return;
    }
    token_capacity = requested_tokens;
    hidden.Reset(token_capacity * kHidden);
    normalized.Reset(token_capacity * kHidden);
    q.Reset(token_capacity * kHidden);
    k.Reset(token_capacity * kHidden);
    v.Reset(token_capacity * kHidden);
    attention.Reset(token_capacity * kHidden);
    attention_bfloat16.Reset(token_capacity * kHidden);
    projection.Reset(token_capacity * kOutput);
    ffn.Reset(token_capacity * kFfn);
    ffn_bfloat16.Reset(token_capacity * kFfn);
  }

  /// The audio tower always runs multi-token, and rocBLAS has no WMMA kernel
  /// for a transposed BF16 operand with a float32 output, so hipBLASLt is tried
  /// first with hipBLAS retained as the fallback.
  void Gemm(const void* weight, const void* input_bfloat16, float* output,
            std::size_t rows, std::size_t output_columns,
            std::size_t reduction) {
    if (lt != nullptr && lt->Run(weight, input_bfloat16, output, rows,
                                 output_columns, reduction, stream)) {
      return;
    }
    LaunchGemmBf16(blas, weight, input_bfloat16, output, rows, output_columns,
                   reduction, stream);
  }

  std::size_t RunFrontend(std::span<const float> log_mel, std::size_t frames) {
    if (frames == 0U || log_mel.size() != kMelBins * frames) {
      throw std::invalid_argument(
          "Qwen3-ASR log-mel tensor must have shape [128, frames]");
    }
    const std::size_t requested_chunks =
        (frames + kChunkFrames - 1U) / kChunkFrames;
    EnsureCapacity(requested_chunks);
    RequireHip(hipMemcpyAsync(feature_input.get(), log_mel.data(),
                              log_mel.size() * sizeof(float),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-ASR log-mel");
    LaunchChunkLogMel(feature_input.get(), input.get(), frames,
                      requested_chunks, stream);
    first->Run(input.get(), conv1_weight.get(), conv1.get());
    LaunchBiasGelu(conv1.get(), conv1_bias.get(), conv1.size(), kConvChannels,
                   64U * 50U, stream);
    second->Run(conv1.get(), conv2_weight.get(), conv2.get());
    LaunchBiasGelu(conv2.get(), conv2_bias.get(), conv2.size(), kConvChannels,
                   32U * 25U, stream);
    third->Run(conv2.get(), conv3_weight.get(), conv3.get());
    LaunchBiasGelu(conv3.get(), conv3_bias.get(), conv3.size(), kConvChannels,
                   16U * kConvTime, stream);
    LaunchConvOutputLayout(conv3.get(), layout.get(), requested_chunks, stream);
    Gemm(conv_out.get(), layout.get(), projected.get(),
         requested_chunks * kConvTime, kHidden, kFlattened);
    LaunchAddPositionAndCompact(projected.get(), compact.get(), frames,
                                requested_chunks, stream);

    const std::size_t tail_frames =
        frames - (requested_chunks - 1U) * kChunkFrames;
    return (requested_chunks - 1U) * kConvTime + (tail_frames + 7U) / 8U;
  }

  AudioEncoderOutput EncodeFrontend(std::span<const float> log_mel,
                                    std::size_t frames) {
    AudioEncoderOutput result;
    result.tokens = RunFrontend(log_mel, frames);
    result.values.resize(result.tokens * kHidden);
    RequireHip(hipMemcpyAsync(result.values.data(), compact.get(),
                              result.values.size() * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync Qwen3-ASR audio frontend");
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Qwen3-ASR audio frontend");
    RequireHip(hipGetLastError(), "Qwen3-ASR audio frontend");
    return result;
  }

  AudioEncoderDeviceOutput Encode(
      std::span<const float> log_mel, std::size_t frames,
      AudioEncoderTrace* trace,
      const std::function<bool()>& is_cancelled = {}) {
    const auto checkpoint = [&] {
      if (is_cancelled && is_cancelled())
        throw std::runtime_error("Qwen3-ASR audio encoder cancelled");
    };
    checkpoint();
    const std::size_t tokens = RunFrontend(log_mel, frames);
    if (trace != nullptr) {
      trace->frontend.tokens = tokens;
      trace->frontend.values.resize(tokens * kHidden);
      RequireHip(hipMemcpyAsync(trace->frontend.values.data(), compact.get(),
                                trace->frontend.values.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream),
                 "hipMemcpyAsync Qwen3-ASR audio frontend");
    }
    EnsureEncoderWeights();
    EnsureTransformerCapacity(tokens);
    const std::size_t hidden_elements = tokens * kHidden;
    RequireHip(hipMemcpyAsync(hidden.get(), compact.get(),
                              hidden_elements * sizeof(float),
                              hipMemcpyDeviceToDevice, stream),
               "hipMemcpyAsync Qwen3-ASR frontend activations");

    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
      checkpoint();
      const EncoderLayerWeights& weights = layers[layer];
      LaunchLayerNormToBfloat16(
          hidden.get(), weights.attention_norm_weight.get(),
          weights.attention_norm_bias.get(), normalized.get(), tokens, kHidden,
          kLayerNormEpsilon, stream);
      Gemm(weights.q_weight.get(), normalized.get(), q.get(), tokens, kHidden,
           kHidden);
      Gemm(weights.k_weight.get(), normalized.get(), k.get(), tokens, kHidden,
           kHidden);
      Gemm(weights.v_weight.get(), normalized.get(), v.get(), tokens, kHidden,
           kHidden);
      LaunchBiasRoundBfloat16(q.get(), weights.q_bias.get(), tokens, kHidden,
                              stream);
      LaunchBiasRoundBfloat16(k.get(), weights.k_bias.get(), tokens, kHidden,
                              stream);
      LaunchBiasRoundBfloat16(v.get(), weights.v_bias.get(), tokens, kHidden,
                              stream);
      LaunchSegmentedSelfAttention(q.get(), k.get(), v.get(), attention.get(),
                                   attention_bfloat16.get(), tokens, tokens,
                                   kHeads, kHeadDim, stream);
      Gemm(weights.out_weight.get(), attention_bfloat16.get(), projection.get(),
           tokens, kHidden, kHidden);
      LaunchBiasRoundBfloat16(projection.get(), weights.out_bias.get(), tokens,
                              kHidden, stream);
      LaunchResidualAddBfloat16(hidden.get(), projection.get(), hidden_elements,
                                stream);

      LaunchLayerNormToBfloat16(hidden.get(), weights.final_norm_weight.get(),
                                weights.final_norm_bias.get(), normalized.get(),
                                tokens, kHidden, kLayerNormEpsilon, stream);
      Gemm(weights.fc1_weight.get(), normalized.get(), ffn.get(), tokens, kFfn,
           kHidden);
      LaunchBiasGeluToBfloat16(ffn.get(), weights.fc1_bias.get(),
                               ffn_bfloat16.get(), tokens, kFfn, stream);
      Gemm(weights.fc2_weight.get(), ffn_bfloat16.get(), projection.get(),
           tokens, kHidden, kFfn);
      LaunchBiasRoundBfloat16(projection.get(), weights.fc2_bias.get(), tokens,
                              kHidden, stream);
      LaunchResidualAddBfloat16(hidden.get(), projection.get(), hidden_elements,
                                stream);

      if (layer == 0U && trace != nullptr) {
        trace->layer0.tokens = tokens;
        trace->layer0.values.resize(hidden_elements);
        RequireHip(hipMemcpyAsync(trace->layer0.values.data(), hidden.get(),
                                  hidden_elements * sizeof(float),
                                  hipMemcpyDeviceToHost, stream),
                   "hipMemcpyAsync Qwen3-ASR layer0");
      }
    }

    LaunchLayerNormToBfloat16(hidden.get(), ln_post_weight.get(),
                              ln_post_bias.get(), normalized.get(), tokens,
                              kHidden, kLayerNormEpsilon, stream);
    Gemm(proj1_weight.get(), normalized.get(), projection.get(), tokens,
         kHidden, kHidden);
    LaunchBiasGeluToBfloat16(projection.get(), proj1_bias.get(),
                             normalized.get(), tokens, kHidden, stream);
    Gemm(proj2_weight.get(), normalized.get(), projection.get(), tokens,
         kOutput, kHidden);
    LaunchBiasRoundBfloat16(projection.get(), proj2_bias.get(), tokens, kOutput,
                            stream);

    if (trace != nullptr) {
      trace->final.tokens = tokens;
      trace->final.values.resize(tokens * kOutput);
      RequireHip(hipMemcpyAsync(trace->final.values.data(), projection.get(),
                                trace->final.values.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream),
                 "hipMemcpyAsync Qwen3-ASR audio final");
    }
    std::uint32_t finite = 1U;
    RequireHip(hipMemcpyAsync(finite_flag.get(), &finite, sizeof(finite),
                              hipMemcpyHostToDevice, stream),
               "hipMemcpyAsync Qwen3-ASR finite flag");
    LaunchAllFinite(projection.get(), tokens * kOutput, finite_flag.get(),
                    stream);
    RequireHip(hipMemcpyAsync(&finite, finite_flag.get(), sizeof(finite),
                              hipMemcpyDeviceToHost, stream),
               "hipMemcpyAsync Qwen3-ASR finite result");
    RequireHip(hipStreamSynchronize(stream),
               "hipStreamSynchronize Qwen3-ASR audio encoder");
    RequireHip(hipGetLastError(), "Qwen3-ASR audio encoder");
    if (finite == 0U) {
      throw std::runtime_error(
          "Qwen3-ASR audio encoder produced non-finite embeddings");
    }
    return {
        .values = projection.get(),
        .tokens = tokens,
    };
  }

  LoadResult model;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  std::unique_ptr<GemmLt> lt;
  miopenHandle_t miopen{nullptr};
  std::size_t chunks{0};
  std::size_t token_capacity{0};
  std::unique_ptr<Convolution> first;
  std::unique_ptr<Convolution> second;
  std::unique_ptr<Convolution> third;
  DeviceBuffer<hip_bfloat16> conv1_weight;
  DeviceBuffer<hip_bfloat16> conv1_bias;
  DeviceBuffer<hip_bfloat16> conv2_weight;
  DeviceBuffer<hip_bfloat16> conv2_bias;
  DeviceBuffer<hip_bfloat16> conv3_weight;
  DeviceBuffer<hip_bfloat16> conv3_bias;
  DeviceBuffer<hip_bfloat16> conv_out;
  DeviceBuffer<float> feature_input;
  DeviceBuffer<hip_bfloat16> input;
  DeviceBuffer<hip_bfloat16> conv1;
  DeviceBuffer<hip_bfloat16> conv2;
  DeviceBuffer<hip_bfloat16> conv3;
  DeviceBuffer<hip_bfloat16> layout;
  DeviceBuffer<float> projected;
  DeviceBuffer<float> compact;
  std::vector<EncoderLayerWeights> layers;
  DeviceBuffer<hip_bfloat16> ln_post_weight;
  DeviceBuffer<hip_bfloat16> ln_post_bias;
  DeviceBuffer<hip_bfloat16> proj1_weight;
  DeviceBuffer<hip_bfloat16> proj1_bias;
  DeviceBuffer<hip_bfloat16> proj2_weight;
  DeviceBuffer<hip_bfloat16> proj2_bias;
  DeviceBuffer<float> hidden;
  DeviceBuffer<hip_bfloat16> normalized;
  DeviceBuffer<float> q;
  DeviceBuffer<float> k;
  DeviceBuffer<float> v;
  DeviceBuffer<float> attention;
  DeviceBuffer<hip_bfloat16> attention_bfloat16;
  DeviceBuffer<float> projection;
  DeviceBuffer<float> ffn;
  DeviceBuffer<hip_bfloat16> ffn_bfloat16;
  DeviceBuffer<std::uint32_t> finite_flag{1U};
};

AudioEncoderHipRuntime::AudioEncoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AudioEncoderHipRuntime::~AudioEncoderHipRuntime() = default;

std::unique_ptr<AudioEncoderHipRuntime> AudioEncoderHipRuntime::Create(
    const std::string& model_root, std::string* error) {
  try {
    LoadResult loaded = LoadModelDirectory(model_root);
    if (!loaded.ok) {
      SetError(error, loaded.error);
      return nullptr;
    }
    return std::unique_ptr<AudioEncoderHipRuntime>(
        new AudioEncoderHipRuntime(std::make_unique<Impl>(std::move(loaded))));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool AudioEncoderHipRuntime::EncodeFrontend(std::span<const float> log_mel,
                                            std::size_t frames,
                                            AudioEncoderOutput* output,
                                            std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR audio encoder output must not be null");
    return false;
  }
  *output = {};
  try {
    *output = impl_->EncodeFrontend(log_mel, frames);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool AudioEncoderHipRuntime::Encode(std::span<const float> log_mel,
                                    std::size_t frames,
                                    AudioEncoderTrace* output,
                                    std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR audio encoder trace must not be null");
    return false;
  }
  *output = {};
  try {
    (void)impl_->Encode(log_mel, frames, output);
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool AudioEncoderHipRuntime::EncodeDevice(
    std::span<const float> log_mel, std::size_t frames,
    AudioEncoderDeviceOutput* output, std::string* error,
    const std::function<bool()>& is_cancelled) {
  if (output == nullptr) {
    SetError(error, "Qwen3-ASR device encoder output must not be null");
    return false;
  }
  *output = {};
  try {
    *output = impl_->Encode(log_mel, frames, nullptr, is_cancelled);
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    SetError(error, exception.what());
    return false;
  }
}

}  // namespace gufo::models::qwen3_asr::hip

#else

namespace gufo::models::qwen3_asr::hip {

AudioEncoderHipRuntime::~AudioEncoderHipRuntime() = default;

std::unique_ptr<AudioEncoderHipRuntime> AudioEncoderHipRuntime::Create(
    const std::string&, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR audio encoder requires ENGINE_ENABLE_HIP";
  }
  return nullptr;
}

bool AudioEncoderHipRuntime::EncodeFrontend(std::span<const float>, std::size_t,
                                            AudioEncoderOutput*,
                                            std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR audio encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

bool AudioEncoderHipRuntime::Encode(std::span<const float>, std::size_t,
                                    AudioEncoderTrace*, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR audio encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

bool AudioEncoderHipRuntime::EncodeDevice(std::span<const float>, std::size_t,
                                          AudioEncoderDeviceOutput*,
                                          std::string* error,
                                          const std::function<bool()>&) {
  if (error != nullptr) {
    *error = "Qwen3-ASR audio encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
