#ifndef GUFO_MODELS_QWEN3_TTS_HIP_SPEAKER_ENCODER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_SPEAKER_ENCODER_RUNTIME_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::models::qwen3_tts::hip {

struct SpeakerEncoderOutput {
  std::vector<float> embedding;
};

/// Optional stage captures for comparing against the official ECAPA-TDNN.
/// Every tensor is frame-major, so a channel-major oracle has to be transposed
/// before comparison. `pooled` and `embedding` are single rows.
struct SpeakerEncoderTrace {
  std::size_t frames{0};
  std::vector<float> features;   // [frames, 128] rounded log-mel input
  std::vector<float> initial;    // [frames, 512] initial TDNN block
  std::vector<float> block1;     // [frames, 512] first SE-Res2Net block
  std::vector<float> block2;     // [frames, 512]
  std::vector<float> block3;     // [frames, 512]
  std::vector<float> aggregate;  // [frames, 1536] multi-layer aggregation
  std::vector<float> pooled;     // [3072] attentive statistics
  std::vector<float> embedding;  // [2048]
};

/// Reference-audio speaker embedding for Base prompts, on the GPU or the NPU.
class SpeakerEncoderBackend {
public:
  virtual ~SpeakerEncoderBackend() = default;
  [[nodiscard]] virtual bool Encode(const AudioBuffer& audio,
                                    SpeakerEncoderOutput* output,
                                    std::string* error = nullptr,
                                    SpeakerEncoderTrace* trace = nullptr) = 0;
};

class SpeakerEncoderHipRuntime : public SpeakerEncoderBackend {
public:
  ~SpeakerEncoderHipRuntime();

  SpeakerEncoderHipRuntime(const SpeakerEncoderHipRuntime&) = delete;
  SpeakerEncoderHipRuntime& operator=(const SpeakerEncoderHipRuntime&) = delete;
  SpeakerEncoderHipRuntime(SpeakerEncoderHipRuntime&&) noexcept;
  SpeakerEncoderHipRuntime& operator=(SpeakerEncoderHipRuntime&&) noexcept;

  [[nodiscard]] static std::unique_ptr<SpeakerEncoderHipRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);

  [[nodiscard]] bool Encode(const AudioBuffer& audio,
                            SpeakerEncoderOutput* output,
                            std::string* error = nullptr,
                            SpeakerEncoderTrace* trace = nullptr) override;

private:
  struct Impl;
  explicit SpeakerEncoderHipRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_SPEAKER_ENCODER_RUNTIME_HPP_
