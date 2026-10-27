#include "src/models/qwen3_tts/hip/speech_encoder_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/hip/encoder_ops.hpp"
#include "src/models/qwen3_tts/hip/encoder_support.hpp"
#include "src/models/qwen3_tts/hip/speech_decoder_ops.hpp"
#include "src/models/qwen3_tts/loader.hpp"

namespace gufo::models::qwen3_tts::hip {
namespace {

constexpr std::size_t kSampleRate = 24000;
constexpr std::size_t kHidden = 512;
constexpr std::size_t kQuantizerDimension = 256;
constexpr std::size_t kCodebookSize = 2048;
constexpr std::size_t kCodeGroups = 16;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::size_t CeilDivide(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct ConvWeights {
  const float* weight{nullptr};
  const float* bias{nullptr};
  std::size_t input_channels{0};
  std::size_t output_channels{0};
  std::size_t kernel{0};
  std::size_t stride{1};
  std::size_t dilation{1};
  EncoderPaddingMode padding{EncoderPaddingMode::kZero};
};

struct ResidualWeights {
  ConvWeights first;
  ConvWeights second;
};

struct LinearWeights {
  const float* weight{nullptr};
  std::size_t input_columns{0};
  std::size_t output_columns{0};
};

struct TransformerWeights {
  const float* norm1_weight{nullptr};
  const float* norm1_bias{nullptr};
  const float* norm2_weight{nullptr};
  const float* norm2_bias{nullptr};
  const float* attention_scale{nullptr};
  const float* mlp_scale{nullptr};
  LinearWeights query;
  LinearWeights key;
  LinearWeights value;
  LinearWeights output;
  LinearWeights fc1;
  LinearWeights fc2;
};

struct CodebookWeights {
  const float* embedding{nullptr};
};

struct Workspace {
  explicit Workspace(std::size_t main_elements, std::size_t columns_elements,
                     std::size_t transformer_frames, std::size_t output_frames)
      : first(main_elements),
        second(main_elements),
        third(main_elements),
        fourth(main_elements),
        columns(columns_elements),
        query(transformer_frames * kHidden),
        key(transformer_frames * kHidden),
        value(transformer_frames * kHidden),
        attention(transformer_frames * kHidden),
        normalized(transformer_frames * kHidden),
        ffn(transformer_frames * 2048),
        projected(output_frames * kQuantizerDimension),
        acoustic(output_frames * kQuantizerDimension),
        codes(output_frames * kCodeGroups) {}

  EncoderDeviceBuffer<float> first;
  EncoderDeviceBuffer<float> second;
  EncoderDeviceBuffer<float> third;
  EncoderDeviceBuffer<float> fourth;
  EncoderDeviceBuffer<float> columns;
  EncoderDeviceBuffer<float> query;
  EncoderDeviceBuffer<float> key;
  EncoderDeviceBuffer<float> value;
  EncoderDeviceBuffer<float> attention;
  EncoderDeviceBuffer<float> normalized;
  EncoderDeviceBuffer<float> ffn;
  EncoderDeviceBuffer<float> projected;
  EncoderDeviceBuffer<float> acoustic;
  EncoderDeviceBuffer<std::uint32_t> codes;
};

}  // namespace

struct SpeechEncoderHipRuntime::Impl {
  explicit Impl(LoadResult loaded)
      : model(std::move(loaded)), weights(*model.store) {
    if (model.config.variant != ModelVariant::kBase) {
      throw std::invalid_argument(
          "Qwen3-TTS speech encoding requires a Base model");
    }
    RequireEncoderHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                      "hipStreamCreate Qwen3-TTS speech encoder");
    LoadWeights();
    // Weights were converted into owned device buffers by hipMemcpy.
    model.ReleaseHostPages();
  }

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  ConvWeights LoadConv(std::string_view prefix, std::size_t input_channels,
                       std::size_t output_channels, std::size_t kernel,
                       std::size_t stride = 1, std::size_t dilation = 1,
                       bool bias = true,
                       EncoderPaddingMode padding = EncoderPaddingMode::kZero) {
    const std::array<std::uint64_t, 3> weight_shape{output_channels,
                                                    input_channels, kernel};
    ConvWeights result{
        .weight = weights.Load(std::string(prefix) + ".weight", weight_shape),
        .input_channels = input_channels,
        .output_channels = output_channels,
        .kernel = kernel,
        .stride = stride,
        .dilation = dilation,
        .padding = padding,
    };
    if (bias) {
      const std::array<std::uint64_t, 1> bias_shape{output_channels};
      result.bias = weights.Load(std::string(prefix) + ".bias", bias_shape);
    }
    return result;
  }

  LinearWeights LoadLinear(std::string_view prefix, std::size_t input_columns,
                           std::size_t output_columns) {
    const std::array<std::uint64_t, 2> shape{output_columns, input_columns};
    return {
        .weight = weights.Load(std::string(prefix) + ".weight", shape),
        .input_columns = input_columns,
        .output_columns = output_columns,
    };
  }

  const float* LoadVector(std::string_view name, std::size_t elements) {
    const std::array<std::uint64_t, 1> shape{elements};
    return weights.Load(name, shape);
  }

  void LoadWeights() {
    const std::array<const char*, 6> convolution_names{
        "encoder.encoder.layers.0.conv",  "encoder.encoder.layers.3.conv",
        "encoder.encoder.layers.6.conv",  "encoder.encoder.layers.9.conv",
        "encoder.encoder.layers.12.conv", "encoder.encoder.layers.14.conv",
    };
    const std::array<std::array<std::size_t, 4>, 6> convolution_shapes{{
        {1, 64, 7, 1},
        {64, 128, 8, 4},
        {128, 256, 10, 5},
        {256, 512, 12, 6},
        {512, 1024, 16, 8},
        {1024, 512, 3, 1},
    }};
    for (std::size_t index = 0; index < convolution_names.size(); ++index) {
      const auto& shape = convolution_shapes[index];
      convolutions.push_back(LoadConv(convolution_names[index], shape[0],
                                      shape[1], shape[2], shape[3]));
    }

    const std::array<std::size_t, 4> residual_indices{1, 4, 7, 10};
    const std::array<std::size_t, 4> residual_channels{64, 128, 256, 512};
    for (std::size_t index = 0; index < residual_indices.size(); ++index) {
      const std::string prefix = "encoder.encoder.layers." +
                                 std::to_string(residual_indices[index]) +
                                 ".block.";
      const std::size_t channels = residual_channels[index];
      residuals.push_back({
          .first = LoadConv(prefix + "1.conv", channels, channels / 2, 3),
          .second = LoadConv(prefix + "3.conv", channels / 2, channels, 1),
      });
    }

    for (std::size_t layer = 0; layer < 8; ++layer) {
      const std::string prefix =
          "encoder.encoder_transformer.layers." + std::to_string(layer);
      transformers.push_back({
          .norm1_weight = LoadVector(prefix + ".input_layernorm.weight", 512),
          .norm1_bias = LoadVector(prefix + ".input_layernorm.bias", 512),
          .norm2_weight =
              LoadVector(prefix + ".post_attention_layernorm.weight", 512),
          .norm2_bias =
              LoadVector(prefix + ".post_attention_layernorm.bias", 512),
          .attention_scale =
              LoadVector(prefix + ".self_attn_layer_scale.scale", 512),
          .mlp_scale = LoadVector(prefix + ".mlp_layer_scale.scale", 512),
          .query = LoadLinear(prefix + ".self_attn.q_proj", 512, 512),
          .key = LoadLinear(prefix + ".self_attn.k_proj", 512, 512),
          .value = LoadLinear(prefix + ".self_attn.v_proj", 512, 512),
          .output = LoadLinear(prefix + ".self_attn.o_proj", 512, 512),
          .fc1 = LoadLinear(prefix + ".mlp.fc1", 512, 2048),
          .fc2 = LoadLinear(prefix + ".mlp.fc2", 2048, 512),
      });
    }
    downsample = LoadConv("encoder.downsample.conv", 512, 512, 4, 2, 1, false,
                          EncoderPaddingMode::kReplicate);
    semantic_projection = LoadConv(
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj", 512,
        256, 1, 1, 1, false);
    acoustic_projection = LoadConv(
        "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj", 512,
        256, 1, 1, 1, false);

    semantic_codebook = {
        .embedding = weights.LoadBfloat16CodebookEmbedding(
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
            "codebook.embed_sum",
            "encoder.quantizer.semantic_residual_vector_quantizer.layers.0."
            "codebook.cluster_usage",
            kCodebookSize, kQuantizerDimension),
    };
    for (std::size_t layer = 0; layer < kCodeGroups - 1; ++layer) {
      const std::string prefix =
          "encoder.quantizer.acoustic_residual_vector_quantizer.layers." +
          std::to_string(layer) + ".codebook.";
      acoustic_codebooks.push_back({
          .embedding = weights.LoadBfloat16CodebookEmbedding(
              prefix + "embed_sum", prefix + "cluster_usage", kCodebookSize,
              kQuantizerDimension),
      });
    }
  }

  std::size_t RunConv(const ConvWeights& conv, const float* input,
                      std::size_t input_length, float* output, float* columns) {
    const std::size_t output_length = CeilDivide(input_length, conv.stride);
    const std::size_t effective_kernel = (conv.kernel - 1) * conv.dilation + 1;
    const std::size_t left_padding = effective_kernel - conv.stride;
    LaunchEncoderConv1dIm2Col(input, columns, input_length, conv.input_channels,
                              output_length, conv.kernel, conv.stride,
                              conv.dilation, left_padding, conv.padding,
                              stream);
    gemm.Run(conv.weight, columns, output, output_length, conv.output_channels,
             conv.input_channels * conv.kernel, stream);
    if (conv.bias != nullptr) {
      LaunchAddBias(output, conv.bias, output_length, conv.output_channels,
                    stream);
    }
    LaunchRoundBfloat16(output, output_length * conv.output_channels, stream);
    return output_length;
  }

  void RunLinear(const LinearWeights& linear, const float* input, float* output,
                 std::size_t rows) {
    gemm.Run(linear.weight, input, output, rows, linear.output_columns,
             linear.input_columns, stream);
    LaunchRoundBfloat16(output, rows * linear.output_columns, stream);
  }

  Workspace MakeWorkspace(std::size_t samples) {
    std::size_t maximum_main = samples;
    std::size_t maximum_columns = 0;
    std::size_t length = samples;
    std::size_t channels = 1;
    auto include_conv = [&](const ConvWeights& conv) {
      const std::size_t output_length = CeilDivide(length, conv.stride);
      maximum_columns = std::max(
          maximum_columns, output_length * conv.input_channels * conv.kernel);
      maximum_main =
          std::max(maximum_main, output_length * conv.output_channels);
      length = output_length;
      channels = conv.output_channels;
      (void)channels;
    };
    include_conv(convolutions[0]);
    for (std::size_t index = 0; index < residuals.size(); ++index) {
      const auto& residual = residuals[index];
      maximum_columns =
          std::max(maximum_columns, length * residual.first.input_channels *
                                        residual.first.kernel);
      maximum_main =
          std::max(maximum_main, length * residual.first.output_channels);
      maximum_columns =
          std::max(maximum_columns, length * residual.second.input_channels *
                                        residual.second.kernel);
      include_conv(convolutions[index + 1]);
    }
    include_conv(convolutions.back());
    const std::size_t transformer_frames = length;
    include_conv(downsample);
    const std::size_t output_frames = length;
    maximum_columns = std::max(maximum_columns, output_frames * 512);
    return Workspace(maximum_main, maximum_columns, transformer_frames,
                     output_frames);
  }

  void CaptureLayer(const float* values, std::size_t channels,
                    std::size_t frames, SpeechEncoderTrace* trace) const {
    if (trace == nullptr) {
      return;
    }
    SpeechEncoderLayerTrace layer;
    layer.channels = channels;
    layer.frames = frames;
    layer.values.resize(channels * frames);
    RequireEncoderHip(hipMemcpyAsync(layer.values.data(), values,
                                     layer.values.size() * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      "copy Qwen3-TTS speech encoder layer trace");
    RequireEncoderHip(hipStreamSynchronize(stream),
                      "synchronize Qwen3-TTS speech encoder layer trace");
    trace->layers.push_back(std::move(layer));
  }

  LoadResult model;
  EncoderWeights weights;
  EncoderF32Gemm gemm;
  hipStream_t stream{nullptr};
  std::vector<ConvWeights> convolutions;
  std::vector<ResidualWeights> residuals;
  std::vector<TransformerWeights> transformers;
  ConvWeights downsample;
  ConvWeights semantic_projection;
  ConvWeights acoustic_projection;
  CodebookWeights semantic_codebook;
  std::vector<CodebookWeights> acoustic_codebooks;
};

SpeechEncoderHipRuntime::SpeechEncoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SpeechEncoderHipRuntime::~SpeechEncoderHipRuntime() = default;
SpeechEncoderHipRuntime::SpeechEncoderHipRuntime(
    SpeechEncoderHipRuntime&&) noexcept = default;
SpeechEncoderHipRuntime& SpeechEncoderHipRuntime::operator=(
    SpeechEncoderHipRuntime&&) noexcept = default;

std::unique_ptr<SpeechEncoderHipRuntime> SpeechEncoderHipRuntime::Create(
    const std::string& model_root, std::string* error) {
  LoadResult model = LoadModelDirectory(model_root);
  if (!model.ok) {
    SetError(error, model.error);
    return nullptr;
  }
  try {
    return std::unique_ptr<SpeechEncoderHipRuntime>(
        new SpeechEncoderHipRuntime(std::make_unique<Impl>(std::move(model))));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool SpeechEncoderHipRuntime::Encode(const AudioBuffer& audio,
                                     SpeechEncoderOutput* output,
                                     std::string* error,
                                     SpeechEncoderTrace* trace) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speech encoder output must not be null");
    return false;
  }
  *output = {};
  if (trace != nullptr) {
    *trace = {};
  }
  try {
    const std::vector<float> waveform = ResampleMono(audio, kSampleRate);
    if (waveform.empty()) {
      throw std::invalid_argument(
          "Qwen3-TTS speech encoder requires reference audio");
    }
    Workspace workspace = impl_->MakeWorkspace(waveform.size());
    RequireEncoderHip(hipMemcpyAsync(workspace.first.get(), waveform.data(),
                                     waveform.size() * sizeof(float),
                                     hipMemcpyHostToDevice, impl_->stream),
                      "copy Qwen3-TTS reference waveform");
    LaunchRoundBfloat16(workspace.first.get(), waveform.size(), impl_->stream);

    float* current = workspace.second.get();
    std::size_t length =
        impl_->RunConv(impl_->convolutions[0], workspace.first.get(),
                       waveform.size(), current, workspace.columns.get());
    impl_->CaptureLayer(current, impl_->convolutions[0].output_channels, length,
                        trace);
    for (std::size_t index = 0; index < impl_->residuals.size(); ++index) {
      const auto& residual = impl_->residuals[index];
      RequireEncoderHip(
          hipMemcpyAsync(workspace.third.get(), current,
                         length * residual.first.input_channels * sizeof(float),
                         hipMemcpyDeviceToDevice, impl_->stream),
          "copy Qwen3-TTS speech residual");
      LaunchElu(workspace.third.get(), length * residual.first.input_channels,
                impl_->stream);
      LaunchRoundBfloat16(workspace.third.get(),
                          length * residual.first.input_channels,
                          impl_->stream);
      (void)impl_->RunConv(residual.first, workspace.third.get(), length,
                           workspace.fourth.get(), workspace.columns.get());
      LaunchElu(workspace.fourth.get(), length * residual.first.output_channels,
                impl_->stream);
      LaunchRoundBfloat16(workspace.fourth.get(),
                          length * residual.first.output_channels,
                          impl_->stream);
      (void)impl_->RunConv(residual.second, workspace.fourth.get(), length,
                           workspace.third.get(), workspace.columns.get());
      LaunchAdd(current, workspace.third.get(), workspace.first.get(),
                length * residual.second.output_channels, impl_->stream);
      LaunchRoundBfloat16(workspace.first.get(),
                          length * residual.second.output_channels,
                          impl_->stream);
      impl_->CaptureLayer(workspace.first.get(),
                          residual.second.output_channels, length, trace);
      LaunchElu(workspace.first.get(), length * residual.second.output_channels,
                impl_->stream);
      LaunchRoundBfloat16(workspace.first.get(),
                          length * residual.second.output_channels,
                          impl_->stream);
      impl_->CaptureLayer(workspace.first.get(),
                          residual.second.output_channels, length, trace);
      length = impl_->RunConv(impl_->convolutions[index + 1],
                              workspace.first.get(), length,
                              workspace.second.get(), workspace.columns.get());
      current = workspace.second.get();
      impl_->CaptureLayer(current,
                          impl_->convolutions[index + 1].output_channels,
                          length, trace);
    }
    LaunchElu(current, length * 1024, impl_->stream);
    LaunchRoundBfloat16(current, length * 1024, impl_->stream);
    impl_->CaptureLayer(current, 1024, length, trace);
    length = impl_->RunConv(impl_->convolutions.back(), current, length,
                            workspace.first.get(), workspace.columns.get());
    current = workspace.first.get();
    impl_->CaptureLayer(current, impl_->convolutions.back().output_channels,
                        length, trace);
    if (trace != nullptr) {
      trace->convolutional_frames = length;
      trace->convolutional.resize(length * kHidden);
      RequireEncoderHip(
          hipMemcpyAsync(trace->convolutional.data(), current,
                         trace->convolutional.size() * sizeof(float),
                         hipMemcpyDeviceToHost, impl_->stream),
          "copy Qwen3-TTS convolutional trace");
      RequireEncoderHip(hipStreamSynchronize(impl_->stream),
                        "synchronize Qwen3-TTS convolutional trace");
    }

    const auto& tokenizer = impl_->model.config.speech_tokenizer;
    for (const TransformerWeights& layer : impl_->transformers) {
      LaunchLayerNorm(current, layer.norm1_weight, layer.norm1_bias,
                      workspace.normalized.get(), length,
                      tokenizer.encoder_hidden_size, tokenizer.encoder_norm_eps,
                      impl_->stream);
      LaunchRoundBfloat16(workspace.normalized.get(), length * kHidden,
                          impl_->stream);
      impl_->RunLinear(layer.query, workspace.normalized.get(),
                       workspace.query.get(), length);
      impl_->RunLinear(layer.key, workspace.normalized.get(),
                       workspace.key.get(), length);
      impl_->RunLinear(layer.value, workspace.normalized.get(),
                       workspace.value.get(), length);
      LaunchRope(workspace.query.get(), workspace.key.get(), length,
                 tokenizer.encoder_num_attention_heads,
                 tokenizer.encoder_head_dim, tokenizer.encoder_rope_theta,
                 impl_->stream);
      LaunchRoundBfloat16(workspace.query.get(), length * kHidden,
                          impl_->stream);
      LaunchRoundBfloat16(workspace.key.get(), length * kHidden, impl_->stream);
      // The encoder transformer is windowed, not fully causal. Passing the
      // sequence length here made every reference clip longer than
      // `encoder_sliding_window` frames attend outside the official window.
      LaunchSlidingCausalAttention(
          workspace.query.get(), workspace.key.get(), workspace.value.get(),
          workspace.attention.get(), length,
          tokenizer.encoder_num_attention_heads, tokenizer.encoder_head_dim,
          tokenizer.encoder_sliding_window, impl_->stream);
      LaunchRoundBfloat16(workspace.attention.get(), length * kHidden,
                          impl_->stream);
      impl_->RunLinear(layer.output, workspace.attention.get(),
                       workspace.second.get(), length);
      LaunchAddScaledChannels(current, workspace.second.get(),
                              layer.attention_scale, length, kHidden,
                              impl_->stream);
      LaunchRoundBfloat16(current, length * kHidden, impl_->stream);

      LaunchLayerNorm(current, layer.norm2_weight, layer.norm2_bias,
                      workspace.normalized.get(), length,
                      tokenizer.encoder_hidden_size, tokenizer.encoder_norm_eps,
                      impl_->stream);
      LaunchRoundBfloat16(workspace.normalized.get(), length * kHidden,
                          impl_->stream);
      impl_->RunLinear(layer.fc1, workspace.normalized.get(),
                       workspace.ffn.get(), length);
      LaunchGelu(workspace.ffn.get(), length * 2048, impl_->stream);
      LaunchRoundBfloat16(workspace.ffn.get(), length * 2048, impl_->stream);
      impl_->RunLinear(layer.fc2, workspace.ffn.get(), workspace.second.get(),
                       length);
      LaunchAddScaledChannels(current, workspace.second.get(), layer.mlp_scale,
                              length, kHidden, impl_->stream);
      LaunchRoundBfloat16(current, length * kHidden, impl_->stream);
    }
    if (trace != nullptr) {
      trace->transformer.resize(length * kHidden);
      RequireEncoderHip(
          hipMemcpyAsync(trace->transformer.data(), current,
                         trace->transformer.size() * sizeof(float),
                         hipMemcpyDeviceToHost, impl_->stream),
          "copy Qwen3-TTS transformer trace");
      RequireEncoderHip(hipStreamSynchronize(impl_->stream),
                        "synchronize Qwen3-TTS transformer trace");
    }

    length = impl_->RunConv(impl_->downsample, current, length,
                            workspace.second.get(), workspace.columns.get());
    current = workspace.second.get();
    if (trace != nullptr) {
      trace->output_frames = length;
      trace->downsample.resize(length * kHidden);
      RequireEncoderHip(hipMemcpyAsync(trace->downsample.data(), current,
                                       trace->downsample.size() * sizeof(float),
                                       hipMemcpyDeviceToHost, impl_->stream),
                        "copy Qwen3-TTS downsample trace");
    }
    (void)impl_->RunConv(impl_->semantic_projection, current, length,
                         workspace.projected.get(), workspace.columns.get());
    (void)impl_->RunConv(impl_->acoustic_projection, current, length,
                         workspace.acoustic.get(), workspace.columns.get());
    LaunchRoundBfloat16(workspace.projected.get(), length * kQuantizerDimension,
                        impl_->stream);
    LaunchRoundBfloat16(workspace.acoustic.get(), length * kQuantizerDimension,
                        impl_->stream);
    if (trace != nullptr) {
      trace->semantic_projection.resize(length * kQuantizerDimension);
      trace->acoustic_projection.resize(length * kQuantizerDimension);
      RequireEncoderHip(
          hipMemcpyAsync(trace->semantic_projection.data(),
                         workspace.projected.get(),
                         trace->semantic_projection.size() * sizeof(float),
                         hipMemcpyDeviceToHost, impl_->stream),
          "copy Qwen3-TTS semantic projection trace");
      RequireEncoderHip(
          hipMemcpyAsync(trace->acoustic_projection.data(),
                         workspace.acoustic.get(),
                         trace->acoustic_projection.size() * sizeof(float),
                         hipMemcpyDeviceToHost, impl_->stream),
          "copy Qwen3-TTS acoustic projection trace");
      RequireEncoderHip(hipStreamSynchronize(impl_->stream),
                        "synchronize Qwen3-TTS projection trace");
    }

    output->frames = length;
    output->code_groups = kCodeGroups;
    output->codes.resize(length * kCodeGroups);
    LaunchQuantizeCodebook(workspace.projected.get(),
                           impl_->semantic_codebook.embedding,
                           workspace.codes.get(), length, kQuantizerDimension,
                           kCodebookSize, kCodeGroups, 0, false, impl_->stream);
    RequireEncoderHip(
        hipMemcpyAsync(workspace.projected.get(), workspace.acoustic.get(),
                       length * kQuantizerDimension * sizeof(float),
                       hipMemcpyDeviceToDevice, impl_->stream),
        "copy Qwen3-TTS acoustic residual");
    for (std::size_t group = 1; group < kCodeGroups; ++group) {
      const CodebookWeights& codebook = impl_->acoustic_codebooks[group - 1];
      LaunchQuantizeCodebook(workspace.projected.get(), codebook.embedding,
                             workspace.codes.get(), length, kQuantizerDimension,
                             kCodebookSize, kCodeGroups, group, true,
                             impl_->stream);
    }
    RequireEncoderHip(
        hipMemcpyAsync(output->codes.data(), workspace.codes.get(),
                       output->codes.size() * sizeof(std::uint32_t),
                       hipMemcpyDeviceToHost, impl_->stream),
        "copy Qwen3-TTS speech codes");
    RequireEncoderHip(hipStreamSynchronize(impl_->stream),
                      "synchronize Qwen3-TTS speech codes");
    RequireEncoderHip(hipGetLastError(), "Qwen3-TTS speech encoder");
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    *output = {};
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts::hip

#else

namespace gufo::models::qwen3_tts::hip {

struct SpeechEncoderHipRuntime::Impl {};

SpeechEncoderHipRuntime::SpeechEncoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeechEncoderHipRuntime::~SpeechEncoderHipRuntime() = default;

std::unique_ptr<SpeechEncoderHipRuntime> SpeechEncoderHipRuntime::Create(
    const std::string&, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return nullptr;
}

bool SpeechEncoderHipRuntime::Encode(const AudioBuffer&, SpeechEncoderOutput*,
                                     std::string* error, SpeechEncoderTrace*) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
