#ifndef GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen3_tts::hip {

struct SpeechDecoderTrace {
  // Convolutional tensors use row-major [time, channels] layout. Transformer
  // output uses the same [time, channels] layout as the official
  // implementation.
  std::vector<float> quantizer_output;
  std::vector<float> pre_conv_output;
  std::vector<float> transformer_output;
  std::vector<std::vector<float>> upsample_outputs;
  std::vector<std::vector<float>> decoder_outputs;
};

struct SpeechDecoderOutput {
  std::vector<float> samples;
  std::uint32_t sample_rate{24000};
  std::size_t code_frames{0};
  double decode_milliseconds{0.0};
};

/// Float32 HIP implementation of the Qwen3-TTS 12 Hz speech-tokenizer decoder.
///
/// Codec codes are frame-major `[frames, 16]`. Each code frame produces 1,920
/// mono samples at 24 kHz.
class SpeechDecoderHipRuntime {
public:
  ~SpeechDecoderHipRuntime();

  SpeechDecoderHipRuntime(const SpeechDecoderHipRuntime&) = delete;
  SpeechDecoderHipRuntime& operator=(const SpeechDecoderHipRuntime&) = delete;
  SpeechDecoderHipRuntime(SpeechDecoderHipRuntime&&) = delete;
  SpeechDecoderHipRuntime& operator=(SpeechDecoderHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<SpeechDecoderHipRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);

  [[nodiscard]] bool Decode(std::span<const std::uint32_t> codes,
                            std::size_t frames, SpeechDecoderOutput* output,
                            SpeechDecoderTrace* trace = nullptr,
                            std::string* error = nullptr);

  /// Decode only the new suffix of a growing codec sequence, retaining causal
  /// convolution histories and attention KV. begin_frame=0 starts a request;
  /// later calls must extend the identical prefix. Returns suffix audio only.
  /// Preserves the offline decoder's 300-frame/25-frame context boundaries.
  [[nodiscard]] bool DecodeIncremental(std::span<const std::uint32_t> codes,
                                       std::size_t frames,
                                       std::size_t begin_frame,
                                       SpeechDecoderOutput* output,
                                       std::string* error = nullptr);

private:
  struct Impl;
  explicit SpeechDecoderHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_DECODER_RUNTIME_HPP_
