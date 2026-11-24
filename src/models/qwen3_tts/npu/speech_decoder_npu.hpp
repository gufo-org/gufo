#ifndef GUFO_MODELS_QWEN3_TTS_NPU_SPEECH_DECODER_NPU_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_SPEECH_DECODER_NPU_HPP_

// The Qwen3-TTS 12 Hz codec decoder on the XDNA NPU (`serve tts --npu`):
// linear layers and convolutions (as im2col products) run on
// gufo::npu::BfpGemm, normalization, attention, activations and the
// convolution bookkeeping on the CPU. Chunking, streaming history and the
// Base reference prefix follow hip::SpeechDecoderHipRuntime.

#include <memory>
#include <string>

#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"

namespace gufo::models::qwen3_tts::npu {

class SpeechDecoderNpuRuntime : public hip::SpeechDecoderBackend {
public:
  ~SpeechDecoderNpuRuntime() override;
  SpeechDecoderNpuRuntime(const SpeechDecoderNpuRuntime&) = delete;
  SpeechDecoderNpuRuntime& operator=(const SpeechDecoderNpuRuntime&) = delete;

  [[nodiscard]] static std::unique_ptr<SpeechDecoderNpuRuntime> Create(
      const std::string& model_root, std::string* error = nullptr);

  [[nodiscard]] bool Decode(std::span<const std::uint32_t> codes,
                            std::size_t frames,
                            hip::SpeechDecoderOutput* output,
                            hip::SpeechDecoderTrace* trace = nullptr,
                            std::string* error = nullptr) override;
  [[nodiscard]] bool DecodeIncremental(std::span<const std::uint32_t> codes,
                                       std::size_t frames,
                                       std::size_t begin_frame,
                                       hip::SpeechDecoderOutput* output,
                                       std::string* error = nullptr) override;
  [[nodiscard]] bool DecodeAfterReference(
      std::span<const std::uint32_t> codes, std::size_t frames,
      std::size_t reference_frames, hip::SpeechDecoderOutput* output,
      std::string* error = nullptr) override;

  struct Impl;

private:
  explicit SpeechDecoderNpuRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_SPEECH_DECODER_NPU_HPP_
