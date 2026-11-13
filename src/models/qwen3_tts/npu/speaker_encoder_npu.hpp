#ifndef GUFO_MODELS_QWEN3_TTS_NPU_SPEAKER_ENCODER_NPU_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_SPEAKER_ENCODER_NPU_HPP_

// The Base model's ECAPA speaker encoder for `serve tts --npu`, on the CPU
// with the GPU path's BF16 rounding points. It runs once per reference
// recording, which synthesis caches.

#include <memory>
#include <string>

#include "src/models/qwen3_tts/hip/speaker_encoder_runtime.hpp"

namespace gufo::models::qwen3_tts::npu {

class SpeakerEncoderNpuRuntime : public hip::SpeakerEncoderBackend {
public:
  ~SpeakerEncoderNpuRuntime() override;
  [[nodiscard]] static std::unique_ptr<SpeakerEncoderNpuRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);
  [[nodiscard]] bool Encode(const AudioBuffer& audio,
                            hip::SpeakerEncoderOutput* output,
                            std::string* error = nullptr,
                            hip::SpeakerEncoderTrace* trace = nullptr) override;

  struct Impl;

private:
  explicit SpeakerEncoderNpuRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_SPEAKER_ENCODER_NPU_HPP_
