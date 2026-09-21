#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"

#include <unordered_map>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/models/qwen3_tts/hip/speech_decoder_ops.hpp"
#include "src/models/qwen3_tts/loader.hpp"

namespace gufo::models::qwen3_tts::hip {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kCodeGroups = 16;
constexpr std::size_t kCodebookDimension = 256;
constexpr std::size_t kSamplesPerCode = 1920;
constexpr std::size_t kChunkFrames = 300;
constexpr std::size_t kLeftContextFrames = 25;

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

void RequireRocblas(rocblas_status status, std::string_view operation) {
  if (status != rocblas_status_success) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template<typename Element>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Reset(count); }
  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  void Reset(std::size_t count = 0) {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
      count_ = 0;
    }
    if (count == 0) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
      throw std::overflow_error("Qwen3-TTS speech decoder buffer overflow");
    }
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Element)),
        "hipMalloc Qwen3-TTS speech decoder");
    count_ = count;
  }

  [[nodiscard]] Element* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
  Element* data_{nullptr};
  std::size_t count_{0};
};

class DeviceRegion {
public:
  DeviceRegion() = default;
  ~DeviceRegion() {
    if (owns_device_ && allocation_ != nullptr) {
      (void)hipFree(allocation_);
    }
    if (registered_ && host_ != nullptr) {
      (void)hipHostUnregister(const_cast<std::byte*>(host_));
    }
  }

  DeviceRegion(const DeviceRegion&) = delete;
  DeviceRegion& operator=(const DeviceRegion&) = delete;

  bool Initialize(MappedRegion region, std::string* error) {
    host_ = region.data;
    size_ = region.size;
    payload_offset_ = region.payload_offset;
    core::PrefaultMappedRange(host_, size_);
    // The supported gfx1151 APU shares physical memory with the CPU, but
    // hipMalloc uses coarse-grained pages that the GPU caches and streams
    // faster than host-registered safetensors pages.
    if (TryCopy() || TryMap())
      return true;
    SetError(error,
             "cannot make Qwen3-TTS speech-tokenizer weights GPU-visible");
    return false;
  }

  [[nodiscard]] const void* Resolve(const std::byte* pointer) const {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base = reinterpret_cast<std::uintptr_t>(host_);
    if (address < base || address - base >= size_) {
      return nullptr;
    }
    return static_cast<const std::byte*>(device_) + (address - base);
  }

private:
  bool TryMap() {
    if (hipHostRegister(const_cast<std::byte*>(host_), size_,
                        hipHostRegisterMapped | hipHostRegisterReadOnly) !=
        hipSuccess) {
      return false;
    }
    registered_ = true;
    if (hipHostGetDevicePointer(&device_, const_cast<std::byte*>(host_), 0) ==
        hipSuccess) {
      return true;
    }
    (void)hipHostUnregister(const_cast<std::byte*>(host_));
    registered_ = false;
    device_ = nullptr;
    return false;
  }

  bool TryCopy() {
    constexpr std::size_t kLoadAlignment = 16;
    const std::size_t shift =
        (kLoadAlignment - (payload_offset_ % kLoadAlignment)) % kLoadAlignment;
    void* allocation = nullptr;
    if (hipMalloc(&allocation, size_ + shift) != hipSuccess) {
      return false;
    }
    allocation_ = allocation;
    owns_device_ = true;
    device_ = static_cast<std::byte*>(allocation) + shift;
    if (hipMemcpy(device_, host_, size_, hipMemcpyHostToDevice) == hipSuccess) {
      return true;
    }
    (void)hipFree(allocation_);
    allocation_ = nullptr;
    device_ = nullptr;
    owns_device_ = false;
    return false;
  }

  const std::byte* host_{nullptr};
  void* device_{nullptr};
  void* allocation_{nullptr};
  std::size_t size_{0};
  std::size_t payload_offset_{0};
  bool owns_device_{false};
  bool registered_{false};
};

const Tensor& RequireF32Tensor(const TensorStore& store, std::string_view name,
                               std::span<const std::uint64_t> shape) {
  const Tensor* tensor = store.Find(name);
  if (tensor == nullptr) {
    throw std::runtime_error("missing Qwen3-TTS speech tensor: " +
                             std::string(name));
  }
  if (tensor->dtype != DType::kF32 ||
      !std::ranges::equal(tensor->shape, shape)) {
    throw std::runtime_error("unexpected Qwen3-TTS speech tensor layout: " +
                             std::string(name));
  }
  return *tensor;
}

const float* ResolveF32(const TensorStore& store, const DeviceRegion& region,
                        std::string_view name,
                        std::span<const std::uint64_t> shape) {
  const Tensor& tensor = RequireF32Tensor(store, name, shape);
  const void* pointer = region.Resolve(tensor.data);
  if (pointer == nullptr) {
    throw std::runtime_error(
        "Qwen3-TTS speech tensor is outside mapped file: " + std::string(name));
  }
  return static_cast<const float*>(pointer);
}

class F32Gemm {
public:
  F32Gemm() {
    RequireRocblas(rocblas_create_handle(&handle_), "rocblas_create_handle");
    RequireRocblas(
        rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed),
        "rocblas_set_atomics_mode");
  }

  ~F32Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }

  F32Gemm(const F32Gemm&) = delete;
  F32Gemm& operator=(const F32Gemm&) = delete;

  void Run(const float* weight, const float* input, float* output,
           std::size_t rows, std::size_t output_columns, std::size_t reduction,
           hipStream_t stream) {
    constexpr std::size_t max_int =
        static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max());
    if (rows == 0 || output_columns == 0 || reduction == 0 || rows > max_int ||
        output_columns > max_int || reduction > max_int) {
      throw std::length_error("invalid Qwen3-TTS speech decoder GEMM shape");
    }
    RequireRocblas(rocblas_set_stream(handle_, stream), "rocblas_set_stream");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    RequireRocblas(
        rocblas_gemm_ex(
            handle_, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<rocblas_int>(output_columns),
            static_cast<rocblas_int>(rows), static_cast<rocblas_int>(reduction),
            &alpha, weight, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(reduction), input, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(reduction), &beta, output,
            rocblas_datatype_f32_r, static_cast<rocblas_int>(output_columns),
            output, rocblas_datatype_f32_r,
            static_cast<rocblas_int>(output_columns), rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "Qwen3-TTS speech decoder GEMM");
  }

private:
  rocblas_handle handle_{nullptr};
};

struct LinearWeights {
  const float* weight{nullptr};
  const float* bias{nullptr};
  std::size_t input_columns{0};
  std::size_t output_columns{0};
};

struct ConvWeights {
  const float* weight{nullptr};
  const float* bias{nullptr};
  std::size_t input_channels{0};
  std::size_t output_channels{0};
  std::size_t kernel{0};
  std::size_t dilation{1};
  std::size_t stride{1};
  bool depthwise{false};
  bool transpose{false};
  DeviceBuffer<float> transformed_weight;
};

struct SnakeBetaWeights {
  const float* alpha{nullptr};
  const float* beta{nullptr};
};

struct CodebookWeights {
  const float* embedding_sum{nullptr};
  const float* cluster_usage{nullptr};
};

struct TransformerLayerWeights {
  const float* input_norm{nullptr};
  const float* post_norm{nullptr};
  const float* attention_scale{nullptr};
  const float* mlp_scale{nullptr};
  LinearWeights query;
  LinearWeights key;
  LinearWeights value;
  LinearWeights attention_output;
  LinearWeights gate;
  LinearWeights up;
  LinearWeights down;
};

struct ConvNeXtWeights {
  ConvWeights depthwise;
  const float* norm_weight{nullptr};
  const float* norm_bias{nullptr};
  LinearWeights pointwise1;
  LinearWeights pointwise2;
  const float* gamma{nullptr};
};

struct UpsampleStageWeights {
  ConvWeights upsample;
  ConvNeXtWeights convnext;
};

struct ResidualUnitWeights {
  SnakeBetaWeights activation1;
  ConvWeights conv1;
  SnakeBetaWeights activation2;
  ConvWeights conv2;
};

struct DecoderBlockWeights {
  SnakeBetaWeights activation;
  ConvWeights upsample;
  std::array<ResidualUnitWeights, 3> residuals;
};

struct Workspace {
  std::size_t capacity_frames{0};
  DeviceBuffer<std::uint32_t> codes;
  std::array<DeviceBuffer<float>, 6> scratch;
  DeviceBuffer<float> columns;
  DeviceBuffer<float> history_input;
  DeviceBuffer<float> history_output;

  void Ensure(std::size_t frames) {
    if (frames <= capacity_frames) {
      return;
    }
    if (frames >
        std::numeric_limits<std::size_t>::max() / (kSamplesPerCode * 96U)) {
      throw std::overflow_error("Qwen3-TTS speech decoder frame overflow");
    }
    const std::size_t scratch_elements = frames * kSamplesPerCode * 96U;
    const std::size_t column_elements = frames * kSamplesPerCode * 96U * 7U;
    codes.Reset(frames * kCodeGroups);
    for (auto& buffer : scratch) {
      buffer.Reset(scratch_elements);
    }
    columns.Reset(column_elements);
    capacity_frames = frames;
  }
};

struct ConvHistory {
  DeviceBuffer<float> input;
  std::size_t rows{0};
};

struct AttentionHistory {
  DeviceBuffer<float> key;
  DeviceBuffer<float> value;
};

struct DecoderHistory {
  std::unordered_map<const ConvWeights*, ConvHistory> convolutions;
  std::vector<AttentionHistory> attention;
  std::vector<std::uint32_t> codes;
  std::size_t position{0};

  void ResetBlock() {
    position = 0;
    for (auto& [weight, history] : convolutions)
      history.rows = 0;
  }

  void CopyFrom(const DecoderHistory& source, std::size_t attention_width,
                std::size_t attention_capacity, hipStream_t stream) {
    const auto copy = [stream](DeviceBuffer<float>& destination,
                               const DeviceBuffer<float>& input,
                               std::size_t count, std::size_t capacity) {
      if (destination.size() < capacity)
        destination.Reset(capacity);
      if (count != 0)
        RequireHip(hipMemcpyAsync(destination.get(), input.get(),
                                  count * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream),
                   "copy decoder reference state");
    };
    ResetBlock();
    for (const auto& [weights, previous] : source.convolutions) {
      auto& next = convolutions[weights];
      const auto count = previous.rows * weights->input_channels;
      copy(next.input, previous.input, count, count);
      next.rows = previous.rows;
    }
    attention.resize(source.attention.size());
    for (std::size_t layer = 0; layer < attention.size(); ++layer) {
      copy(attention[layer].key, source.attention[layer].key,
           source.position * attention_width,
           attention_capacity * attention_width);
      copy(attention[layer].value, source.attention[layer].value,
           source.position * attention_width,
           attention_capacity * attention_width);
    }
    codes = source.codes;
    position = source.position;
  }
};

std::string TransformerName(std::size_t layer, std::string_view suffix) {
  return "decoder.pre_transformer.layers." + std::to_string(layer) + "." +
         std::string(suffix);
}

float* FindScratch(Workspace& workspace,
                   std::initializer_list<const float*> excluded) {
  for (auto& buffer : workspace.scratch) {
    if (std::ranges::find(excluded, buffer.get()) == excluded.end()) {
      return buffer.get();
    }
  }
  throw std::runtime_error("Qwen3-TTS speech decoder scratch exhaustion");
}

}  // namespace

struct SpeechDecoderHipRuntime::Impl {
  explicit Impl(LoadResult loaded) : model(std::move(loaded)) {
    int device = 0;
    RequireHip(hipGetDevice(&device), "hipGetDevice");
    hipDeviceProp_t properties{};
    RequireHip(hipGetDeviceProperties(&properties, device),
               "hipGetDeviceProperties");
    if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
        properties.integrated == 0) {
      throw std::runtime_error(
          "Qwen3-TTS speech decoder supports integrated gfx1151 Strix Halo");
    }
    const auto decoder_regions = model.RegionsFor("decoder.");
    if (decoder_regions.size() != 1) {
      throw std::runtime_error(
          "Qwen3-TTS speech-tokenizer safetensors mapping is missing");
    }
    RequireHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags");
    if (!speech_weights.Initialize(decoder_regions.front(), nullptr)) {
      throw std::runtime_error(
          "cannot initialize Qwen3-TTS speech weight mapping");
    }
    LoadWeights();
    RequireHip(hipStreamSynchronize(stream),
               "synchronize Qwen3-TTS speech weights");
  }

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  LinearWeights LoadLinear(std::string_view prefix, std::size_t input_columns,
                           std::size_t output_columns, bool bias = true) {
    const TensorStore& store = *model.store;
    const std::array<std::uint64_t, 2> weight_shape{
        static_cast<std::uint64_t>(output_columns),
        static_cast<std::uint64_t>(input_columns)};
    LinearWeights result{
        .weight = ResolveF32(store, speech_weights,
                             std::string(prefix) + ".weight", weight_shape),
        .bias = nullptr,
        .input_columns = input_columns,
        .output_columns = output_columns,
    };
    if (bias) {
      const std::array<std::uint64_t, 1> bias_shape{
          static_cast<std::uint64_t>(output_columns)};
      result.bias = ResolveF32(store, speech_weights,
                               std::string(prefix) + ".bias", bias_shape);
    }
    return result;
  }

  ConvWeights LoadConv(std::string_view prefix, std::size_t input_channels,
                       std::size_t output_channels, std::size_t kernel,
                       std::size_t dilation = 1, bool depthwise = false) {
    const TensorStore& store = *model.store;
    const std::array<std::uint64_t, 3> weight_shape{
        static_cast<std::uint64_t>(output_channels),
        static_cast<std::uint64_t>(depthwise ? 1 : input_channels),
        static_cast<std::uint64_t>(kernel)};
    const std::array<std::uint64_t, 1> bias_shape{
        static_cast<std::uint64_t>(output_channels)};
    return {
        .weight = ResolveF32(store, speech_weights,
                             std::string(prefix) + ".weight", weight_shape),
        .bias = ResolveF32(store, speech_weights, std::string(prefix) + ".bias",
                           bias_shape),
        .input_channels = input_channels,
        .output_channels = output_channels,
        .kernel = kernel,
        .dilation = dilation,
        .stride = 1,
        .depthwise = depthwise,
        .transpose = false,
    };
  }

  ConvWeights LoadTransposeConv(std::string_view prefix,
                                std::size_t input_channels,
                                std::size_t output_channels, std::size_t kernel,
                                std::size_t stride) {
    const TensorStore& store = *model.store;
    const std::array<std::uint64_t, 3> weight_shape{
        static_cast<std::uint64_t>(input_channels),
        static_cast<std::uint64_t>(output_channels),
        static_cast<std::uint64_t>(kernel)};
    const std::array<std::uint64_t, 1> bias_shape{
        static_cast<std::uint64_t>(output_channels)};
    ConvWeights result{
        .weight = nullptr,
        .bias = ResolveF32(store, speech_weights, std::string(prefix) + ".bias",
                           bias_shape),
        .input_channels = input_channels,
        .output_channels = output_channels,
        .kernel = kernel,
        .dilation = 1,
        .stride = stride,
        .depthwise = false,
        .transpose = true,
    };
    const float* source = ResolveF32(
        store, speech_weights, std::string(prefix) + ".weight", weight_shape);
    result.transformed_weight.Reset(input_channels * output_channels * kernel);
    LaunchTransposeConvWeight(source, result.transformed_weight.get(),
                              input_channels, output_channels, kernel, stream);
    result.weight = result.transformed_weight.get();
    return result;
  }

  const float* LoadVector(std::string_view name, std::size_t size) {
    const std::array<std::uint64_t, 1> shape{static_cast<std::uint64_t>(size)};
    return ResolveF32(*model.store, speech_weights, name, shape);
  }

  SnakeBetaWeights LoadSnakeBeta(std::string_view alpha_name,
                                 std::string_view beta_name,
                                 std::size_t channels) {
    SnakeBetaWeights result{
        .alpha = LoadVector(alpha_name, channels),
        .beta = LoadVector(beta_name, channels),
    };
    return result;
  }

  void LoadWeights() {
    const auto& config = model.config.speech_tokenizer;
    if (config.num_quantizers != kCodeGroups ||
        config.codebook_dim != 2 * kCodebookDimension ||
        config.upsample_rates != std::vector<std::uint32_t>({8, 5, 4, 3}) ||
        config.upsampling_ratios != std::vector<std::uint32_t>({2, 2})) {
      throw std::runtime_error(
          "unsupported Qwen3-TTS speech decoder configuration");
    }
    const TensorStore& store = *model.store;
    const auto load_codebook = [&](const std::string& prefix) {
      const std::array<std::uint64_t, 2> embedding_shape{config.codebook_size,
                                                         kCodebookDimension};
      const std::array<std::uint64_t, 1> usage_shape{config.codebook_size};
      return CodebookWeights{
          .embedding_sum = ResolveF32(
              store, speech_weights, prefix + "embedding_sum", embedding_shape),
          .cluster_usage = ResolveF32(store, speech_weights,
                                      prefix + "cluster_usage", usage_shape),
      };
    };
    semantic_codebook =
        load_codebook("decoder.quantizer.rvq_first.vq.layers.0._codebook.");
    acoustic_codebooks.reserve(kCodeGroups - 1);
    for (std::size_t group = 0; group + 1 < kCodeGroups; ++group) {
      acoustic_codebooks.push_back(
          load_codebook("decoder.quantizer.rvq_rest.vq.layers." +
                        std::to_string(group) + "._codebook."));
    }
    semantic_projection =
        LoadConvProjection("decoder.quantizer.rvq_first.output_proj");
    acoustic_projection =
        LoadConvProjection("decoder.quantizer.rvq_rest.output_proj");
    pre_conv = LoadConv("decoder.pre_conv.conv", 512, 1024, 3);

    transformer_input =
        LoadLinear("decoder.pre_transformer.input_proj", 1024, 512);
    transformer_layers.reserve(config.num_hidden_layers);
    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
      const std::string prefix =
          "decoder.pre_transformer.layers." + std::to_string(layer);
      transformer_layers.push_back({
          .input_norm = LoadVector(prefix + ".input_layernorm.weight", 512),
          .post_norm =
              LoadVector(prefix + ".post_attention_layernorm.weight", 512),
          .attention_scale =
              LoadVector(prefix + ".self_attn_layer_scale.scale", 512),
          .mlp_scale = LoadVector(prefix + ".mlp_layer_scale.scale", 512),
          .query = LoadLinear(prefix + ".self_attn.q_proj", 512, 1024, false),
          .key = LoadLinear(prefix + ".self_attn.k_proj", 512, 1024, false),
          .value = LoadLinear(prefix + ".self_attn.v_proj", 512, 1024, false),
          .attention_output =
              LoadLinear(prefix + ".self_attn.o_proj", 1024, 512, false),
          .gate = LoadLinear(prefix + ".mlp.gate_proj", 512, 1024, false),
          .up = LoadLinear(prefix + ".mlp.up_proj", 512, 1024, false),
          .down = LoadLinear(prefix + ".mlp.down_proj", 1024, 512, false),
      });
    }
    transformer_norm = LoadVector("decoder.pre_transformer.norm.weight", 512);
    transformer_output =
        LoadLinear("decoder.pre_transformer.output_proj", 512, 1024);

    upsample_stages.reserve(2);
    for (std::size_t stage = 0; stage < 2; ++stage) {
      const std::string prefix = "decoder.upsample." + std::to_string(stage);
      UpsampleStageWeights weights;
      weights.upsample =
          LoadTransposeConv(prefix + ".0.conv", 1024, 1024, 2, 2);
      weights.convnext.depthwise =
          LoadConv(prefix + ".1.dwconv.conv", 1024, 1024, 7, 1, true);
      weights.convnext.norm_weight =
          LoadVector(prefix + ".1.norm.weight", 1024);
      weights.convnext.norm_bias = LoadVector(prefix + ".1.norm.bias", 1024);
      weights.convnext.pointwise1 =
          LoadLinear(prefix + ".1.pwconv1", 1024, 4096);
      weights.convnext.pointwise2 =
          LoadLinear(prefix + ".1.pwconv2", 4096, 1024);
      weights.convnext.gamma = LoadVector(prefix + ".1.gamma", 1024);
      upsample_stages.push_back(std::move(weights));
    }

    decoder_input = LoadConv("decoder.decoder.0.conv", 1024, 1536, 7);
    const std::array<std::size_t, 4> rates{8, 5, 4, 3};
    std::size_t input_channels = 1536;
    for (std::size_t block = 0; block < rates.size(); ++block) {
      const std::size_t output_channels = input_channels / 2;
      const std::string prefix =
          "decoder.decoder." + std::to_string(block + 1) + ".block";
      DecoderBlockWeights weights;
      weights.activation = LoadSnakeBeta(prefix + ".0.alpha",
                                         prefix + ".0.beta", input_channels);
      weights.upsample =
          LoadTransposeConv(prefix + ".1.conv", input_channels, output_channels,
                            rates[block] * 2, rates[block]);
      const std::array<std::size_t, 3> dilations{1, 3, 9};
      for (std::size_t unit = 0; unit < weights.residuals.size(); ++unit) {
        const std::string unit_prefix = prefix + "." + std::to_string(unit + 2);
        auto& residual = weights.residuals[unit];
        residual.activation1 =
            LoadSnakeBeta(unit_prefix + ".act1.alpha",
                          unit_prefix + ".act1.beta", output_channels);
        residual.conv1 = LoadConv(unit_prefix + ".conv1.conv", output_channels,
                                  output_channels, 7, dilations[unit]);
        residual.activation2 =
            LoadSnakeBeta(unit_prefix + ".act2.alpha",
                          unit_prefix + ".act2.beta", output_channels);
        residual.conv2 = LoadConv(unit_prefix + ".conv2.conv", output_channels,
                                  output_channels, 1);
      }
      decoder_blocks.push_back(std::move(weights));
      input_channels = output_channels;
    }
    output_activation =
        LoadSnakeBeta("decoder.decoder.5.alpha", "decoder.decoder.5.beta", 96);
    output_conv = LoadConv("decoder.decoder.6.conv", 96, 1, 7);
  }

  LinearWeights LoadConvProjection(std::string_view prefix) {
    const std::array<std::uint64_t, 3> shape{512, 256, 1};
    return {
        .weight = ResolveF32(*model.store, speech_weights,
                             std::string(prefix) + ".weight", shape),
        .bias = nullptr,
        .input_columns = 256,
        .output_columns = 512,
    };
  }

  void RunLinear(const LinearWeights& weights, const float* input,
                 float* output, std::size_t rows) {
    gemm.Run(weights.weight, input, output, rows, weights.output_columns,
             weights.input_columns, stream);
    if (weights.bias != nullptr) {
      LaunchAddBias(output, weights.bias, rows, weights.output_columns, stream);
    }
  }

  void RunSnakeBeta(const SnakeBetaWeights& weights, const float* input,
                    float* output, std::size_t rows,
                    std::size_t columns) const {
    LaunchSnakeBeta(input, weights.alpha, weights.beta, output, rows, columns,
                    stream);
  }

  void RunConv(const ConvWeights& weights, const float* input, float* output,
               std::size_t input_length, std::size_t output_length) {
    const std::size_t keep = weights.transpose
                                 ? (weights.kernel - 1) / weights.stride
                                 : (weights.kernel - 1) * weights.dilation;
    if (active_history == nullptr || keep == 0) {
      RunConvRaw(weights, input, output, input_length, output_length);
      return;
    }
    auto& history = active_history->convolutions[&weights];
    if (history.input.size() < keep * weights.input_channels) {
      history.input.Reset(keep * weights.input_channels);
    }
    const std::size_t previous = history.rows;
    const std::size_t rows = previous + input_length;
    const std::size_t stride = weights.transpose ? weights.stride : 1;
    const std::size_t input_elements = rows * weights.input_channels;
    const std::size_t output_elements = rows * stride * weights.output_channels;
    if (workspace.history_input.size() < input_elements)
      workspace.history_input.Reset(input_elements);
    if (workspace.history_output.size() < output_elements)
      workspace.history_output.Reset(output_elements);
    float* joined = workspace.history_input.get();
    if (previous != 0) {
      RequireHip(
          hipMemcpyAsync(joined, history.input.get(),
                         previous * weights.input_channels * sizeof(float),
                         hipMemcpyDeviceToDevice, stream),
          "copy convolution history");
    }
    RequireHip(
        hipMemcpyAsync(joined + previous * weights.input_channels, input,
                       input_length * weights.input_channels * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "append convolution input");
    history.rows = std::min(keep, rows);
    RequireHip(
        hipMemcpyAsync(history.input.get(),
                       joined + (rows - history.rows) * weights.input_channels,
                       history.rows * weights.input_channels * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "retain convolution history");
    RunConvRaw(weights, joined, workspace.history_output.get(), rows,
               rows * stride);
    RequireHip(
        hipMemcpyAsync(output,
                       workspace.history_output.get() +
                           previous * stride * weights.output_channels,
                       output_length * weights.output_channels * sizeof(float),
                       hipMemcpyDeviceToDevice, stream),
        "copy incremental convolution output");
  }

  void RunConvRaw(const ConvWeights& weights, const float* input, float* output,
                  std::size_t input_length, std::size_t output_length) {
    if (weights.depthwise) {
      if (input_length != output_length ||
          weights.input_channels != weights.output_channels) {
        throw std::length_error(
            "invalid Qwen3-TTS depthwise convolution shape");
      }
      LaunchCausalDepthwiseConv1d(input, weights.weight, weights.bias, output,
                                  input_length, weights.input_channels,
                                  weights.kernel, weights.dilation, stream);
      return;
    }
    if (weights.transpose) {
      if (output_length != input_length * weights.stride) {
        throw std::length_error(
            "invalid Qwen3-TTS transposed convolution length");
      }
      const std::size_t expanded_columns =
          weights.output_channels * weights.kernel;
      gemm.Run(weights.weight, input, workspace.columns.get(), input_length,
               expanded_columns, weights.input_channels, stream);
      LaunchAssembleCausalConvTranspose1d(
          workspace.columns.get(), weights.bias, output, input_length,
          output_length, weights.output_channels, weights.kernel,
          weights.stride, stream);
      return;
    }
    if (input_length != output_length) {
      throw std::length_error("invalid Qwen3-TTS causal convolution length");
    }
    const std::size_t reduction = weights.input_channels * weights.kernel;
    LaunchCausalConv1dIm2Col(input, workspace.columns.get(), input_length,
                             weights.input_channels, output_length,
                             weights.kernel, weights.dilation, stream);
    gemm.Run(weights.weight, workspace.columns.get(), output, output_length,
             weights.output_channels, reduction, stream);
    LaunchAddBias(output, weights.bias, output_length, weights.output_channels,
                  stream);
  }

  void Capture(const float* device, std::size_t elements,
               std::vector<float>* output) {
    output->resize(elements);
    RequireHip(hipMemcpyAsync(output->data(), device, elements * sizeof(float),
                              hipMemcpyDeviceToHost, stream),
               "copy Qwen3-TTS speech decoder trace");
    RequireHip(hipStreamSynchronize(stream),
               "synchronize Qwen3-TTS speech decoder trace");
  }

  std::vector<float> DecodeChunk(std::span<const std::uint32_t> codes,
                                 std::size_t frames,
                                 SpeechDecoderTrace* trace) {
    workspace.Ensure(frames +
                     (active_history == nullptr ? 0 : kLeftContextFrames));
    RequireHip(
        hipMemcpyAsync(workspace.codes.get(), codes.data(), codes.size_bytes(),
                       hipMemcpyHostToDevice, stream),
        "copy Qwen3-TTS codec frames");

    float* semantic = workspace.scratch[0].get();
    float* acoustic = workspace.scratch[1].get();
    LaunchDecodeCodebook(workspace.codes.get(), kCodeGroups, 0,
                         semantic_codebook.embedding_sum,
                         semantic_codebook.cluster_usage, semantic, frames,
                         kCodebookDimension, true, stream);
    for (std::size_t group = 1; group < kCodeGroups; ++group) {
      const CodebookWeights& codebook = acoustic_codebooks[group - 1];
      LaunchDecodeCodebook(workspace.codes.get(), kCodeGroups, group,
                           codebook.embedding_sum, codebook.cluster_usage,
                           acoustic, frames, kCodebookDimension, group == 1,
                           stream);
    }
    float* semantic_projected = workspace.scratch[2].get();
    float* acoustic_projected = workspace.scratch[3].get();
    RunLinear(semantic_projection, semantic, semantic_projected, frames);
    RunLinear(acoustic_projection, acoustic, acoustic_projected, frames);
    float* hidden = workspace.scratch[0].get();
    LaunchAdd(semantic_projected, acoustic_projected, hidden, frames * 512,
              stream);
    if (trace != nullptr) {
      Capture(hidden, frames * 512, &trace->quantizer_output);
    }

    float* pre_conv_output = workspace.scratch[1].get();
    RunConv(pre_conv, hidden, pre_conv_output, frames, frames);
    if (trace != nullptr) {
      Capture(pre_conv_output, frames * 1024, &trace->pre_conv_output);
    }

    hidden = workspace.scratch[0].get();
    RunLinear(transformer_input, pre_conv_output, hidden, frames);
    std::size_t layer_index = 0;
    for (const TransformerLayerWeights& layer : transformer_layers) {
      float* normalized = FindScratch(workspace, {hidden});
      LaunchRmsNorm(hidden, layer.input_norm, normalized, frames, 512,
                    model.config.speech_tokenizer.rms_norm_eps, stream);
      float* query = FindScratch(workspace, {hidden, normalized});
      float* key = FindScratch(workspace, {hidden, normalized, query});
      float* value = FindScratch(workspace, {hidden, normalized, query, key});
      RunLinear(layer.query, normalized, query, frames);
      RunLinear(layer.key, normalized, key, frames);
      RunLinear(layer.value, normalized, value, frames);
      LaunchRope(query, key, frames,
                 model.config.speech_tokenizer.num_attention_heads,
                 model.config.speech_tokenizer.head_dim,
                 model.config.speech_tokenizer.rope_theta, stream,
                 active_history == nullptr ? 0 : active_history->position);
      float* attention =
          FindScratch(workspace, {hidden, normalized, query, key, value});
      const float* keys = key;
      const float* values = value;
      const std::size_t position =
          active_history == nullptr ? 0 : active_history->position;
      if (active_history != nullptr) {
        active_history->attention.resize(transformer_layers.size());
        auto& cache = active_history->attention[layer_index];
        const std::size_t attention_width =
            model.config.speech_tokenizer.num_attention_heads *
            model.config.speech_tokenizer.head_dim;
        const std::size_t cache_elements =
            (kChunkFrames + kLeftContextFrames) * attention_width;
        if (cache.key.size() == 0) {
          cache.key.Reset(cache_elements);
          cache.value.Reset(cache_elements);
        }
        if (position + frames > kChunkFrames + kLeftContextFrames)
          throw std::length_error(
              "incremental decoder attention exceeds its window");
        RequireHip(hipMemcpyAsync(cache.key.get() + position * attention_width,
                                  key, frames * attention_width * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream),
                   "append decoder attention keys");
        RequireHip(
            hipMemcpyAsync(cache.value.get() + position * attention_width,
                           value, frames * attention_width * sizeof(float),
                           hipMemcpyDeviceToDevice, stream),
            "append decoder attention values");
        keys = cache.key.get();
        values = cache.value.get();
      }
      LaunchSlidingCausalAttention(
          query, keys, values, attention, frames,
          model.config.speech_tokenizer.num_attention_heads,
          model.config.speech_tokenizer.head_dim,
          model.config.speech_tokenizer.sliding_window, stream, position,
          position + frames);
      RunLinear(layer.attention_output, attention, normalized, frames);
      LaunchAddScaledChannels(hidden, normalized, layer.attention_scale, frames,
                              512, stream);

      LaunchRmsNorm(hidden, layer.post_norm, normalized, frames, 512,
                    model.config.speech_tokenizer.rms_norm_eps, stream);
      float* gate = query;
      float* up = key;
      float* activation = value;
      RunLinear(layer.gate, normalized, gate, frames);
      RunLinear(layer.up, normalized, up, frames);
      LaunchSwiGlu(gate, up, activation, frames * 1024, stream);
      RunLinear(layer.down, activation, normalized, frames);
      LaunchAddScaledChannels(hidden, normalized, layer.mlp_scale, frames, 512,
                              stream);
      ++layer_index;
    }
    float* normalized = FindScratch(workspace, {hidden});
    LaunchRmsNorm(hidden, transformer_norm, normalized, frames, 512,
                  model.config.speech_tokenizer.rms_norm_eps, stream);
    float* transformer = FindScratch(workspace, {hidden, normalized});
    RunLinear(transformer_output, normalized, transformer, frames);
    if (trace != nullptr) {
      Capture(transformer, frames * 1024, &trace->transformer_output);
      trace->upsample_outputs.clear();
      trace->decoder_outputs.clear();
    }

    std::size_t length = frames;
    hidden = transformer;
    for (const UpsampleStageWeights& stage : upsample_stages) {
      const std::size_t output_length = length * stage.upsample.stride;
      float* upsampled = FindScratch(workspace, {hidden});
      RunConv(stage.upsample, hidden, upsampled, length, output_length);
      if (trace != nullptr) {
        trace->upsample_outputs.emplace_back();
        Capture(upsampled, output_length * 1024,
                &trace->upsample_outputs.back());
      }
      float* depthwise = FindScratch(workspace, {hidden, upsampled});
      RunConv(stage.convnext.depthwise, upsampled, depthwise, output_length,
              output_length);
      float* normed = FindScratch(workspace, {hidden, upsampled, depthwise});
      LaunchLayerNorm(depthwise, stage.convnext.norm_weight,
                      stage.convnext.norm_bias, normed, output_length, 1024,
                      1.0e-6F, stream);
      float* wide =
          FindScratch(workspace, {hidden, upsampled, depthwise, normed});
      RunLinear(stage.convnext.pointwise1, normed, wide, output_length);
      LaunchGelu(wide, output_length * 4096, stream);
      RunLinear(stage.convnext.pointwise2, wide, depthwise, output_length);
      LaunchAddScaledChannels(upsampled, depthwise, stage.convnext.gamma,
                              output_length, 1024, stream);
      if (trace != nullptr) {
        trace->upsample_outputs.emplace_back();
        Capture(upsampled, output_length * 1024,
                &trace->upsample_outputs.back());
      }
      hidden = upsampled;
      length = output_length;
    }

    float* decoder_hidden = FindScratch(workspace, {hidden});
    RunConv(decoder_input, hidden, decoder_hidden, length, length);
    if (trace != nullptr) {
      trace->decoder_outputs.emplace_back();
      Capture(decoder_hidden, length * 1536, &trace->decoder_outputs.back());
    }
    hidden = decoder_hidden;
    std::size_t channels = 1536;
    for (const DecoderBlockWeights& block : decoder_blocks) {
      float* activated = FindScratch(workspace, {hidden});
      RunSnakeBeta(block.activation, hidden, activated, length, channels);
      const std::size_t output_length = length * block.upsample.stride;
      float* upsampled = FindScratch(workspace, {hidden, activated});
      RunConv(block.upsample, activated, upsampled, length, output_length);
      channels /= 2;
      for (const ResidualUnitWeights& residual : block.residuals) {
        float* branch = FindScratch(workspace, {hidden, activated, upsampled});
        RunSnakeBeta(residual.activation1, upsampled, branch, output_length,
                     channels);
        float* work =
            FindScratch(workspace, {hidden, activated, upsampled, branch});
        RunConv(residual.conv1, branch, work, output_length, output_length);
        RunSnakeBeta(residual.activation2, work, branch, output_length,
                     channels);
        RunConv(residual.conv2, branch, work, output_length, output_length);
        LaunchAdd(upsampled, work, upsampled, output_length * channels, stream);
      }
      hidden = upsampled;
      length = output_length;
      if (trace != nullptr) {
        trace->decoder_outputs.emplace_back();
        Capture(hidden, length * channels, &trace->decoder_outputs.back());
      }
    }

    float* activated = FindScratch(workspace, {hidden});
    RunSnakeBeta(output_activation, hidden, activated, length, 96);
    if (trace != nullptr) {
      trace->decoder_outputs.emplace_back();
      Capture(activated, length * 96, &trace->decoder_outputs.back());
    }
    float* waveform = FindScratch(workspace, {hidden, activated});
    RunConv(output_conv, activated, waveform, length, length);
    if (trace != nullptr) {
      trace->decoder_outputs.emplace_back();
      Capture(waveform, length, &trace->decoder_outputs.back());
    }
    LaunchClamp(waveform, length, -1.0F, 1.0F, stream);
    std::vector<float> samples(length);
    RequireHip(
        hipMemcpyAsync(samples.data(), waveform, samples.size() * sizeof(float),
                       hipMemcpyDeviceToHost, stream),
        "copy Qwen3-TTS waveform");
    RequireHip(hipStreamSynchronize(stream),
               "synchronize Qwen3-TTS speech decoder");
    RequireHip(hipGetLastError(), "Qwen3-TTS speech decoder");
    if (active_history != nullptr)
      active_history->position += frames;
    return samples;
  }

  LoadResult model;
  DeviceRegion speech_weights;
  hipStream_t stream{nullptr};
  F32Gemm gemm;
  Workspace workspace;
  DecoderHistory history;
  std::unique_ptr<DecoderHistory> reference_history;
  DecoderHistory* active_history{nullptr};

  CodebookWeights semantic_codebook;
  std::vector<CodebookWeights> acoustic_codebooks;
  LinearWeights semantic_projection;
  LinearWeights acoustic_projection;
  ConvWeights pre_conv;
  LinearWeights transformer_input;
  std::vector<TransformerLayerWeights> transformer_layers;
  const float* transformer_norm{nullptr};
  LinearWeights transformer_output;
  std::vector<UpsampleStageWeights> upsample_stages;
  ConvWeights decoder_input;
  std::vector<DecoderBlockWeights> decoder_blocks;
  SnakeBetaWeights output_activation;
  ConvWeights output_conv;
};

SpeechDecoderHipRuntime::SpeechDecoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SpeechDecoderHipRuntime::~SpeechDecoderHipRuntime() = default;

std::unique_ptr<SpeechDecoderHipRuntime> SpeechDecoderHipRuntime::Create(
    const std::string& model_root, std::string* error) {
  try {
    LoadResult loaded = LoadModelDirectory(model_root);
    if (!loaded.ok) {
      SetError(error, loaded.error);
      return nullptr;
    }
    return std::unique_ptr<SpeechDecoderHipRuntime>(
        new SpeechDecoderHipRuntime(std::make_unique<Impl>(std::move(loaded))));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool SpeechDecoderHipRuntime::DecodeIncremental(
    std::span<const std::uint32_t> codes, std::size_t frames,
    std::size_t begin_frame, SpeechDecoderOutput* output, std::string* error) {
  if (output == nullptr || frames == 0 || begin_frame > frames ||
      frames > std::numeric_limits<std::size_t>::max() / kSamplesPerCode ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "invalid incremental speech decoder shape");
    return false;
  }
  *output = {};
  auto& history = impl_->history;
  if (begin_frame == 0) {
    history.codes.clear();
    history.ResetBlock();
  } else if (history.codes.size() != begin_frame * kCodeGroups ||
             !std::equal(history.codes.begin(), history.codes.end(),
                         codes.begin())) {
    SetError(error, "incremental speech decoder prefix does not match");
    return false;
  }
  for (const auto code : codes) {
    if (code >= impl_->model.config.speech_tokenizer.codebook_size) {
      SetError(error, "incremental speech decoder code is out of range");
      return false;
    }
  }
  const auto started = Clock::now();
  impl_->active_history = &history;
  struct ResetActive {
    Impl* impl;
    ~ResetActive() { impl->active_history = nullptr; }
  } reset{impl_.get()};
  try {
    for (std::size_t cursor = begin_frame; cursor < frames;) {
      if (cursor % kChunkFrames == 0) {
        history.ResetBlock();
        // The offline decoder restarts every 300 frames with 25 prior frames.
        // Reconstruct exactly that context, without emitting its audio twice.
        for (std::size_t warm = cursor - std::min(cursor, kLeftContextFrames);
             warm < cursor;) {
          const auto count = std::min<std::size_t>(16, cursor - warm);
          (void)impl_->DecodeChunk(
              codes.subspan(warm * kCodeGroups, count * kCodeGroups), count,
              nullptr);
          warm += count;
        }
      }
      const auto count = std::min({std::size_t{16}, frames - cursor,
                                   kChunkFrames - cursor % kChunkFrames});
      const auto chunk =
          codes.subspan(cursor * kCodeGroups, count * kCodeGroups);
      auto samples = impl_->DecodeChunk(chunk, count, nullptr);
      output->samples.insert(output->samples.end(), samples.begin(),
                             samples.end());
      history.codes.insert(history.codes.end(), chunk.begin(), chunk.end());
      cursor += count;
    }
    output->sample_rate =
        impl_->model.config.speech_tokenizer.output_sample_rate;
    output->code_frames = frames - begin_frame;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    history.codes.clear();
    history.ResetBlock();
    *output = {};
    SetError(error, exception.what());
    return false;
  }
}

bool SpeechDecoderHipRuntime::Decode(std::span<const std::uint32_t> codes,
                                     std::size_t frames,
                                     SpeechDecoderOutput* output,
                                     SpeechDecoderTrace* trace,
                                     std::string* error) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speech decoder output is null");
    return false;
  }
  if (frames == 0 ||
      frames > std::numeric_limits<std::size_t>::max() / kCodeGroups ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "Qwen3-TTS speech decoder expects [frames,16] codes");
    return false;
  }
  for (const std::uint32_t code : codes) {
    if (code >= impl_->model.config.speech_tokenizer.codebook_size) {
      SetError(error, "Qwen3-TTS speech decoder code is out of range");
      return false;
    }
  }
  if (trace != nullptr && frames > kChunkFrames) {
    SetError(error,
             "Qwen3-TTS speech decoder trace supports at most 300 frames");
    return false;
  }
  try {
    const auto begin = Clock::now();
    output->samples.clear();
    output->samples.reserve(frames * kSamplesPerCode);
    for (std::size_t start = 0; start < frames; start += kChunkFrames) {
      const std::size_t end = std::min(start + kChunkFrames, frames);
      const std::size_t context = std::min(start, kLeftContextFrames);
      const std::size_t chunk_start = start - context;
      const std::size_t chunk_frames = end - chunk_start;
      const auto chunk_codes =
          codes.subspan(chunk_start * kCodeGroups, chunk_frames * kCodeGroups);
      std::vector<float> decoded =
          impl_->DecodeChunk(chunk_codes, chunk_frames, trace);
      const std::size_t drop = context * kSamplesPerCode;
      const std::size_t keep = (end - start) * kSamplesPerCode;
      if (drop > decoded.size() || keep > decoded.size() - drop) {
        throw std::runtime_error(
            "Qwen3-TTS speech decoder chunk output is truncated");
      }
      output->samples.insert(
          output->samples.end(),
          decoded.begin() + static_cast<std::ptrdiff_t>(drop),
          decoded.begin() + static_cast<std::ptrdiff_t>(drop + keep));
    }
    output->sample_rate =
        impl_->model.config.speech_tokenizer.output_sample_rate;
    output->code_frames = frames;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool SpeechDecoderHipRuntime::DecodeAfterReference(
    std::span<const std::uint32_t> codes, std::size_t frames,
    std::size_t reference_frames, SpeechDecoderOutput* output,
    std::string* error) {
  if (output == nullptr || reference_frames >= frames ||
      frames > std::numeric_limits<std::size_t>::max() / kSamplesPerCode ||
      codes.size() != frames * kCodeGroups) {
    SetError(error, "invalid speech decoder reference shape");
    return false;
  }
  *output = {};
  const auto started = Clock::now();
  // End at an existing incremental work boundary. A partial reference tail
  // continues with generated codes in the same chunks as a cold request.
  const std::size_t prefix_frames =
      reference_frames - (reference_frames % kChunkFrames) % 16;
  try {
    if (prefix_frames != 0) {
      const auto prefix = codes.first(prefix_frames * kCodeGroups);
      auto& saved = impl_->reference_history;
      const std::size_t width =
          impl_->model.config.speech_tokenizer.num_attention_heads *
          impl_->model.config.speech_tokenizer.head_dim;
      if (saved == nullptr || saved->codes.size() != prefix.size() ||
          !std::equal(prefix.begin(), prefix.end(), saved->codes.begin())) {
        SpeechDecoderOutput discarded;
        if (!DecodeIncremental(prefix, prefix_frames, 0, &discarded, error))
          return false;
        auto replacement = std::make_unique<DecoderHistory>();
        replacement->CopyFrom(impl_->history, width, impl_->history.position,
                              impl_->stream);
        RequireHip(hipStreamSynchronize(impl_->stream),
                   "capture decoder reference state");
        saved = std::move(replacement);
      } else {
        impl_->history.CopyFrom(
            *saved, width, kChunkFrames + kLeftContextFrames, impl_->stream);
      }
    }
    if (!DecodeIncremental(codes, frames, prefix_frames, output, error))
      return false;
    const std::size_t drop =
        (reference_frames - prefix_frames) * kSamplesPerCode;
    output->samples.erase(
        output->samples.begin(),
        output->samples.begin() + static_cast<std::ptrdiff_t>(drop));
    output->code_frames = frames - reference_frames;
    output->decode_milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return true;
  } catch (const std::exception& exception) {
    (void)hipStreamSynchronize(impl_->stream);
    impl_->history.codes.clear();
    impl_->history.ResetBlock();
    *output = {};
    SetError(error, exception.what());
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::hip

#else

namespace gufo::models::qwen3_tts::hip {

struct SpeechDecoderHipRuntime::Impl {};

SpeechDecoderHipRuntime::SpeechDecoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SpeechDecoderHipRuntime::~SpeechDecoderHipRuntime() = default;

std::unique_ptr<SpeechDecoderHipRuntime> SpeechDecoderHipRuntime::Create(
    const std::string&, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS speech decoder requires a HIP-enabled build";
  }
  return nullptr;
}

bool SpeechDecoderHipRuntime::Decode(std::span<const std::uint32_t>,
                                     std::size_t, SpeechDecoderOutput*,
                                     SpeechDecoderTrace*, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS speech decoder requires a HIP-enabled build";
  }
  return false;
}

bool SpeechDecoderHipRuntime::DecodeIncremental(std::span<const std::uint32_t>,
                                                std::size_t, std::size_t,
                                                SpeechDecoderOutput*,
                                                std::string* error) {
  if (error != nullptr)
    *error = "Qwen3-TTS speech decoder requires a HIP-enabled build";
  return false;
}

bool SpeechDecoderHipRuntime::DecodeAfterReference(
    std::span<const std::uint32_t>, std::size_t, std::size_t,
    SpeechDecoderOutput*, std::string* error) {
  if (error != nullptr)
    *error = "Qwen3-TTS speech decoder requires a HIP-enabled build";
  return false;
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
