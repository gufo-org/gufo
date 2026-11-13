#ifndef GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_ENCODER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_ENCODER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::models::qwen3_tts::hip {

struct SpeechEncoderOutput {
  // Frame-major [frames, code_groups].
  std::vector<std::uint32_t> codes;
  std::size_t frames{0};
  std::size_t code_groups{0};
};

/// One captured SEANet layer: the lumped `convolutional` stage covers six
/// strided convolutions and four residual units, so a mismatch there needs the
/// per-layer breakdown to localise.
struct SpeechEncoderLayerTrace {
  std::vector<float> values;
  std::size_t channels{0};
  std::size_t frames{0};
};

struct SpeechEncoderTrace {
  // All tensors use frame-major layout.
  std::vector<SpeechEncoderLayerTrace> layers;
  std::vector<float> convolutional;
  std::vector<float> transformer;
  std::vector<float> downsample;
  std::vector<float> semantic_projection;
  std::vector<float> acoustic_projection;
  std::size_t convolutional_frames{0};
  std::size_t output_frames{0};
};

/// Reference-audio codec encoding for Base prompts, on the GPU or the NPU.
class SpeechEncoderBackend {
public:
  virtual ~SpeechEncoderBackend() = default;
  [[nodiscard]] virtual bool Encode(const AudioBuffer& audio,
                                    SpeechEncoderOutput* output,
                                    std::string* error = nullptr,
                                    SpeechEncoderTrace* trace = nullptr) = 0;
};

class SpeechEncoderHipRuntime : public SpeechEncoderBackend {
public:
  ~SpeechEncoderHipRuntime();

  SpeechEncoderHipRuntime(const SpeechEncoderHipRuntime&) = delete;
  SpeechEncoderHipRuntime& operator=(const SpeechEncoderHipRuntime&) = delete;
  SpeechEncoderHipRuntime(SpeechEncoderHipRuntime&&) noexcept;
  SpeechEncoderHipRuntime& operator=(SpeechEncoderHipRuntime&&) noexcept;

  [[nodiscard]] static std::unique_ptr<SpeechEncoderHipRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);

  [[nodiscard]] bool Encode(const AudioBuffer& audio,
                            SpeechEncoderOutput* output,
                            std::string* error = nullptr,
                            SpeechEncoderTrace* trace = nullptr) override;

private:
  struct Impl;
  explicit SpeechEncoderHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_SPEECH_ENCODER_RUNTIME_HPP_
