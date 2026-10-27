#include "src/models/qwen3_tts/hip/speaker_encoder_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
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
constexpr std::size_t kMelBins = 128;
constexpr std::size_t kFft = 1024;
constexpr std::size_t kFrequencyBins = kFft / 2 + 1;
constexpr std::size_t kHop = 256;
constexpr std::size_t kPrepad = (kFft - kHop) / 2;
constexpr std::size_t kChannels = 512;
constexpr std::size_t kScale = 8;
constexpr std::size_t kWidth = kChannels / kScale;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::int64_t ReflectIndex(std::int64_t index, std::int64_t length) {
  if (length <= 1) {
    return 0;
  }
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : 2 * length - index - 2;
  }
  return index;
}

void Fft(std::span<std::complex<float>> values) {
  const std::size_t size = values.size();
  for (std::size_t index = 1, reversed = 0; index < size; ++index) {
    std::size_t bit = size >> 1U;
    for (; (reversed & bit) != 0; bit >>= 1U) {
      reversed ^= bit;
    }
    reversed ^= bit;
    if (index < reversed) {
      std::swap(values[index], values[reversed]);
    }
  }
  for (std::size_t length = 2; length <= size; length <<= 1U) {
    const float angle =
        -2.0F * std::numbers::pi_v<float> / static_cast<float>(length);
    const std::complex<float> root(std::cos(angle), std::sin(angle));
    for (std::size_t offset = 0; offset < size; offset += length) {
      std::complex<float> factor(1.0F, 0.0F);
      for (std::size_t index = 0; index < length / 2; ++index) {
        const std::complex<float> even = values[offset + index];
        const std::complex<float> odd =
            factor * values[offset + index + length / 2];
        values[offset + index] = even + odd;
        values[offset + index + length / 2] = even - odd;
        factor *= root;
      }
    }
  }
}

float HertzToMel(float hertz) {
  constexpr float scale = 200.0F / 3.0F;
  constexpr float boundary_hertz = 1000.0F;
  constexpr float boundary_mel = boundary_hertz / scale;
  constexpr float log_step = 0.06875177742094912F;
  return hertz >= boundary_hertz
             ? boundary_mel + std::log(hertz / boundary_hertz) / log_step
             : hertz / scale;
}

float MelToHertz(float mel) {
  constexpr float scale = 200.0F / 3.0F;
  constexpr float boundary_hertz = 1000.0F;
  constexpr float boundary_mel = boundary_hertz / scale;
  constexpr float log_step = 0.06875177742094912F;
  return mel >= boundary_mel
             ? boundary_hertz * std::exp(log_step * (mel - boundary_mel))
             : mel * scale;
}

std::vector<float> MakeMelFilterbank() {
  const float mel_min = HertzToMel(0.0F);
  const float mel_max = HertzToMel(12000.0F);
  std::vector<float> hertz(kMelBins + 2);
  for (std::size_t index = 0; index < hertz.size(); ++index) {
    const float mel = mel_min + (mel_max - mel_min) *
                                    static_cast<float>(index) /
                                    static_cast<float>(kMelBins + 1);
    hertz[index] = MelToHertz(mel);
  }
  std::vector<float> filter(kMelBins * kFrequencyBins, 0.0F);
  for (std::size_t mel = 0; mel < kMelBins; ++mel) {
    const float left = hertz[mel];
    const float center = hertz[mel + 1];
    const float right = hertz[mel + 2];
    const float normalization = 2.0F / std::max(right - left, 1.0e-12F);
    for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
      const float frequency = static_cast<float>(kSampleRate) * 0.5F *
                              static_cast<float>(bin) /
                              static_cast<float>(kFrequencyBins - 1);
      const float lower =
          (frequency - left) / std::max(center - left, 1.0e-12F);
      const float upper =
          (right - frequency) / std::max(right - center, 1.0e-12F);
      filter[mel * kFrequencyBins + bin] =
          std::max(0.0F, std::min(lower, upper)) * normalization;
    }
  }
  return filter;
}

struct SpeakerFeatures {
  std::vector<float> values;
  std::size_t frames{0};
};

SpeakerFeatures ComputeFeatures(const AudioBuffer& audio) {
  std::vector<float> waveform = ResampleMono(audio, kSampleRate);
  if (waveform.empty()) {
    throw std::invalid_argument(
        "Qwen3-TTS speaker encoder requires reference audio");
  }
  std::vector<float> padded(waveform.size() + 2 * kPrepad);
  for (std::size_t index = 0; index < padded.size(); ++index) {
    const std::int64_t source = ReflectIndex(
        static_cast<std::int64_t>(index) - static_cast<std::int64_t>(kPrepad),
        static_cast<std::int64_t>(waveform.size()));
    padded[index] = waveform[static_cast<std::size_t>(source)];
  }
  if (padded.size() < kFft) {
    throw std::invalid_argument(
        "Qwen3-TTS reference audio is too short for speaker encoding");
  }
  const std::size_t frames = 1 + (padded.size() - kFft) / kHop;
  std::vector<float> magnitude(kFrequencyBins * frames, 0.0F);
  std::vector<std::complex<float>> spectrum(kFft);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t index = 0; index < kFft; ++index) {
      const float window =
          0.5F -
          0.5F * std::cos(2.0F * std::numbers::pi_v<float> *
                          static_cast<float>(index) / static_cast<float>(kFft));
      spectrum[index] =
          std::complex<float>(padded[frame * kHop + index] * window, 0.0F);
    }
    Fft(spectrum);
    for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
      magnitude[bin * frames + frame] =
          std::sqrt(std::norm(spectrum[bin]) + 1.0e-9F);
    }
  }

  static const std::vector<float> filter = MakeMelFilterbank();
  SpeakerFeatures result;
  result.frames = frames;
  result.values.resize(frames * kMelBins);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t mel = 0; mel < kMelBins; ++mel) {
      long double sum = 0.0;
      for (std::size_t bin = 0; bin < kFrequencyBins; ++bin) {
        sum += static_cast<long double>(filter[mel * kFrequencyBins + bin]) *
               magnitude[bin * frames + frame];
      }
      result.values[frame * kMelBins + mel] =
          std::log(std::max(static_cast<float>(sum), 1.0e-5F));
    }
  }
  return result;
}

struct ConvWeights {
  const float* weight{nullptr};
  const float* bias{nullptr};
  std::size_t input_channels{0};
  std::size_t output_channels{0};
  std::size_t kernel{0};
  std::size_t dilation{1};
};

struct Res2NetWeights {
  ConvWeights tdnn1;
  std::array<ConvWeights, 7> blocks;
  ConvWeights tdnn2;
  ConvWeights se1;
  ConvWeights se2;
};

struct Workspace {
  explicit Workspace(std::size_t frames)
      : first(frames * 4608),
        second(frames * 4608),
        third(frames * 4608),
        fourth(frames * 4608),
        columns(frames * 4608 * 5),
        hidden1(frames * 512),
        hidden2(frames * 512),
        hidden3(frames * 512),
        chunk(frames * 64),
        previous(frames * 64),
        combined(frames * 64),
        channel_vector(3072) {}

  EncoderDeviceBuffer<float> first;
  EncoderDeviceBuffer<float> second;
  EncoderDeviceBuffer<float> third;
  EncoderDeviceBuffer<float> fourth;
  EncoderDeviceBuffer<float> columns;
  EncoderDeviceBuffer<float> hidden1;
  EncoderDeviceBuffer<float> hidden2;
  EncoderDeviceBuffer<float> hidden3;
  EncoderDeviceBuffer<float> chunk;
  EncoderDeviceBuffer<float> previous;
  EncoderDeviceBuffer<float> combined;
  EncoderDeviceBuffer<float> channel_vector;
};

}  // namespace

struct SpeakerEncoderHipRuntime::Impl {
  explicit Impl(LoadResult loaded)
      : model(std::move(loaded)), weights(*model.store) {
    if (model.config.variant != ModelVariant::kBase ||
        !model.config.speaker_encoder.has_value()) {
      throw std::invalid_argument(
          "Qwen3-TTS speaker encoding requires a Base model");
    }
    RequireEncoderHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                      "hipStreamCreate Qwen3-TTS speaker encoder");
    LoadWeights();
    // Weights were converted into owned device buffers by hipMemcpy.
    model.ReleaseHostPages();
  }

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  ConvWeights LoadConv(std::string_view suffix, std::size_t input_channels,
                       std::size_t output_channels, std::size_t kernel,
                       std::size_t dilation = 1) {
    const std::string prefix = "speaker_encoder." + std::string(suffix);
    return {
        .weight = weights.Load(prefix + ".weight",
                               std::array<std::uint64_t, 3>{
                                   output_channels, input_channels, kernel}),
        .bias = weights.Load(prefix + ".bias",
                             std::array<std::uint64_t, 1>{output_channels}),
        .input_channels = input_channels,
        .output_channels = output_channels,
        .kernel = kernel,
        .dilation = dilation,
    };
  }

  void LoadWeights() {
    block0 = LoadConv("blocks.0.conv", 128, 512, 5);
    for (std::size_t block = 1; block <= 3; ++block) {
      const std::string prefix = "blocks." + std::to_string(block);
      Res2NetWeights layer{
          .tdnn1 = LoadConv(prefix + ".tdnn1.conv", 512, 512, 1),
          .tdnn2 = LoadConv(prefix + ".tdnn2.conv", 512, 512, 1),
          .se1 = LoadConv(prefix + ".se_block.conv1", 512, 128, 1),
          .se2 = LoadConv(prefix + ".se_block.conv2", 128, 512, 1),
      };
      for (std::size_t index = 0; index < layer.blocks.size(); ++index) {
        layer.blocks[index] = LoadConv(
            prefix + ".res2net_block.blocks." + std::to_string(index) + ".conv",
            64, 64, 3, block + 1);
      }
      blocks.push_back(std::move(layer));
    }
    mfa = LoadConv("mfa.conv", 1536, 1536, 1);
    asp_tdnn = LoadConv("asp.tdnn.conv", 4608, 128, 1);
    asp_conv = LoadConv("asp.conv", 128, 1536, 1);
    fc = LoadConv("fc", 3072, 2048, 1);
  }

  void RunConv(const ConvWeights& conv, const float* input, std::size_t rows,
               float* output, float* columns) {
    const std::size_t padding = conv.dilation * (conv.kernel - 1) / 2;
    LaunchEncoderConv1dIm2Col(input, columns, rows, conv.input_channels, rows,
                              conv.kernel, 1, conv.dilation, padding,
                              EncoderPaddingMode::kReflect, stream);
    gemm.Run(conv.weight, columns, output, rows, conv.output_channels,
             conv.input_channels * conv.kernel, stream);
    LaunchAddBias(output, conv.bias, rows, conv.output_channels, stream);
    LaunchRoundBfloat16(output, rows * conv.output_channels, stream);
  }

  void Tdnn(const ConvWeights& conv, const float* input, std::size_t rows,
            float* output, float* columns) {
    RunConv(conv, input, rows, output, columns);
    LaunchRelu(output, rows * conv.output_channels, stream);
  }

  /// `input` is read again for the residual add after the Res2Net branches have
  /// run, so it must not alias any scratch buffer those branches write. Only
  /// `fourth` and the three `hidden` buffers are safe to pass here.
  void RunBlock(const Res2NetWeights& block, const float* input,
                std::size_t rows, float* output, Workspace* workspace) {
    if (input == workspace->first.get() || input == workspace->second.get() ||
        input == workspace->third.get() || input == workspace->chunk.get() ||
        input == workspace->previous.get() ||
        input == workspace->combined.get() ||
        input == workspace->channel_vector.get()) {
      throw std::invalid_argument(
          "Qwen3-TTS speaker Res2Net input aliases block scratch");
    }
    Tdnn(block.tdnn1, input, rows, workspace->first.get(),
         workspace->columns.get());
    for (std::size_t split = 0; split < kScale; ++split) {
      LaunchExtractChannels(workspace->first.get(), workspace->chunk.get(),
                            rows, kChannels, split * kWidth, kWidth, stream);
      const float* branch = workspace->chunk.get();
      if (split > 0) {
        if (split > 1) {
          LaunchAdd(workspace->chunk.get(), workspace->previous.get(),
                    workspace->combined.get(), rows * kWidth, stream);
          branch = workspace->combined.get();
        }
        Tdnn(block.blocks[split - 1], branch, rows, workspace->second.get(),
             workspace->columns.get());
        RequireEncoderHip(
            hipMemcpyAsync(workspace->previous.get(), workspace->second.get(),
                           rows * kWidth * sizeof(float),
                           hipMemcpyDeviceToDevice, stream),
            "copy Qwen3-TTS Res2Net state");
        branch = workspace->second.get();
      } else {
        RequireEncoderHip(hipMemcpyAsync(workspace->previous.get(), branch,
                                         rows * kWidth * sizeof(float),
                                         hipMemcpyDeviceToDevice, stream),
                          "copy Qwen3-TTS Res2Net first state");
      }
      LaunchInsertChannels(branch, workspace->third.get(), rows, kChannels,
                           split * kWidth, kWidth, stream);
    }
    Tdnn(block.tdnn2, workspace->third.get(), rows, workspace->first.get(),
         workspace->columns.get());
    LaunchChannelMean(workspace->first.get(), workspace->channel_vector.get(),
                      rows, kChannels, stream);
    Tdnn(block.se1, workspace->channel_vector.get(), 1, workspace->second.get(),
         workspace->columns.get());
    RunConv(block.se2, workspace->second.get(), 1,
            workspace->channel_vector.get(), workspace->columns.get());
    LaunchSigmoidMultiplyChannels(workspace->first.get(),
                                  workspace->channel_vector.get(), rows,
                                  kChannels, stream);
    LaunchAdd(workspace->first.get(), input, output, rows * kChannels, stream);
    LaunchRoundBfloat16(output, rows * kChannels, stream);
  }

  void Capture(const float* values, std::size_t count,
               std::vector<float>* out) const {
    out->resize(count);
    RequireEncoderHip(hipMemcpyAsync(out->data(), values, count * sizeof(float),
                                     hipMemcpyDeviceToHost, stream),
                      "copy Qwen3-TTS speaker encoder trace");
    RequireEncoderHip(hipStreamSynchronize(stream),
                      "synchronize Qwen3-TTS speaker encoder trace");
  }

  LoadResult model;
  EncoderWeights weights;
  EncoderF32Gemm gemm;
  hipStream_t stream{nullptr};
  ConvWeights block0;
  std::vector<Res2NetWeights> blocks;
  ConvWeights mfa;
  ConvWeights asp_tdnn;
  ConvWeights asp_conv;
  ConvWeights fc;
};

SpeakerEncoderHipRuntime::SpeakerEncoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeakerEncoderHipRuntime::~SpeakerEncoderHipRuntime() = default;
SpeakerEncoderHipRuntime::SpeakerEncoderHipRuntime(
    SpeakerEncoderHipRuntime&&) noexcept = default;
SpeakerEncoderHipRuntime& SpeakerEncoderHipRuntime::operator=(
    SpeakerEncoderHipRuntime&&) noexcept = default;

std::unique_ptr<SpeakerEncoderHipRuntime> SpeakerEncoderHipRuntime::Create(
    const std::string& model_root, std::string* error) {
  LoadResult model = LoadModelDirectory(model_root);
  if (!model.ok) {
    SetError(error, model.error);
    return nullptr;
  }
  try {
    return std::unique_ptr<SpeakerEncoderHipRuntime>(
        new SpeakerEncoderHipRuntime(std::make_unique<Impl>(std::move(model))));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool SpeakerEncoderHipRuntime::Encode(const AudioBuffer& audio,
                                      SpeakerEncoderOutput* output,
                                      std::string* error,
                                      SpeakerEncoderTrace* trace) {
  if (output == nullptr) {
    SetError(error, "Qwen3-TTS speaker encoder output must not be null");
    return false;
  }
  *output = {};
  if (trace != nullptr) {
    *trace = {};
  }
  try {
    const SpeakerFeatures features = ComputeFeatures(audio);
    Workspace workspace(features.frames);
    RequireEncoderHip(
        hipMemcpyAsync(workspace.first.get(), features.values.data(),
                       features.values.size() * sizeof(float),
                       hipMemcpyHostToDevice, impl_->stream),
        "copy Qwen3-TTS speaker features");
    LaunchRoundBfloat16(workspace.first.get(), features.values.size(),
                        impl_->stream);
    impl_->Tdnn(impl_->block0, workspace.first.get(), features.frames,
                workspace.fourth.get(), workspace.columns.get());
    if (trace != nullptr) {
      trace->frames = features.frames;
      impl_->Capture(workspace.first.get(), features.frames * kMelBins,
                     &trace->features);
      impl_->Capture(workspace.fourth.get(), features.frames * kChannels,
                     &trace->initial);
    }
    impl_->RunBlock(impl_->blocks[0], workspace.fourth.get(), features.frames,
                    workspace.hidden1.get(), &workspace);
    impl_->RunBlock(impl_->blocks[1], workspace.hidden1.get(), features.frames,
                    workspace.hidden2.get(), &workspace);
    impl_->RunBlock(impl_->blocks[2], workspace.hidden2.get(), features.frames,
                    workspace.hidden3.get(), &workspace);
    if (trace != nullptr) {
      impl_->Capture(workspace.hidden1.get(), features.frames * kChannels,
                     &trace->block1);
      impl_->Capture(workspace.hidden2.get(), features.frames * kChannels,
                     &trace->block2);
      impl_->Capture(workspace.hidden3.get(), features.frames * kChannels,
                     &trace->block3);
    }

    LaunchConcatenateThree(workspace.hidden1.get(), workspace.hidden2.get(),
                           workspace.hidden3.get(), workspace.first.get(),
                           features.frames, kChannels, impl_->stream);
    impl_->Tdnn(impl_->mfa, workspace.first.get(), features.frames,
                workspace.second.get(), workspace.columns.get());
    if (trace != nullptr) {
      impl_->Capture(workspace.second.get(), features.frames * 1536,
                     &trace->aggregate);
    }
    LaunchAttentiveStatsInput(workspace.second.get(), workspace.first.get(),
                              features.frames, 1536, 1.0e-12F, impl_->stream);
    LaunchRoundBfloat16(workspace.first.get(), features.frames * 4608,
                        impl_->stream);
    impl_->Tdnn(impl_->asp_tdnn, workspace.first.get(), features.frames,
                workspace.third.get(), workspace.columns.get());
    LaunchTanh(workspace.third.get(), features.frames * 128, impl_->stream);
    impl_->RunConv(impl_->asp_conv, workspace.third.get(), features.frames,
                   workspace.first.get(), workspace.columns.get());
    LaunchSoftmaxOverRows(workspace.first.get(), features.frames, 1536,
                          impl_->stream);
    LaunchRoundBfloat16(workspace.first.get(), features.frames * 1536,
                        impl_->stream);
    LaunchWeightedStats(workspace.second.get(), workspace.first.get(),
                        workspace.channel_vector.get(), features.frames, 1536,
                        1.0e-12F, impl_->stream);
    LaunchRoundBfloat16(workspace.channel_vector.get(), 3072, impl_->stream);
    if (trace != nullptr) {
      impl_->Capture(workspace.channel_vector.get(), 3072, &trace->pooled);
    }
    impl_->RunConv(impl_->fc, workspace.channel_vector.get(), 1,
                   workspace.first.get(), workspace.columns.get());

    output->embedding.resize(2048);
    RequireEncoderHip(
        hipMemcpyAsync(output->embedding.data(), workspace.first.get(),
                       output->embedding.size() * sizeof(float),
                       hipMemcpyDeviceToHost, impl_->stream),
        "copy Qwen3-TTS speaker embedding");
    RequireEncoderHip(hipStreamSynchronize(impl_->stream),
                      "synchronize Qwen3-TTS speaker encoder");
    RequireEncoderHip(hipGetLastError(), "Qwen3-TTS speaker encoder");
    if (trace != nullptr) {
      trace->embedding = output->embedding;
    }
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

struct SpeakerEncoderHipRuntime::Impl {};
SpeakerEncoderHipRuntime::SpeakerEncoderHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpeakerEncoderHipRuntime::~SpeakerEncoderHipRuntime() = default;

std::unique_ptr<SpeakerEncoderHipRuntime> SpeakerEncoderHipRuntime::Create(
    const std::string&, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return nullptr;
}

bool SpeakerEncoderHipRuntime::Encode(const AudioBuffer&, SpeakerEncoderOutput*,
                                      std::string* error,
                                      SpeakerEncoderTrace*) {
  if (error != nullptr) {
    *error = "Qwen3-TTS HIP support is not enabled";
  }
  return false;
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
