#ifndef GUFO_MODELS_QWEN3_TTS_NPU_TALKER_NPU_HPP_
#define GUFO_MODELS_QWEN3_TTS_NPU_TALKER_NPU_HPP_

// The Qwen3-TTS talker and code predictor for `serve tts --npu`, with the
// GPU off: prompt prefill and text projection on the XDNA NPU
// (gufo::npu::BfpGemm), decode steps and the operators between products on
// the CPU (see talker_npu.cpp).

#include <cstddef>
#include <memory>
#include <string>

#include "src/models/qwen3_tts/hip/talker_runtime.hpp"

namespace gufo::models::qwen3_tts::npu {

class TalkerNpuRuntime : public hip::TalkerBackend {
public:
  ~TalkerNpuRuntime() override;
  TalkerNpuRuntime(const TalkerNpuRuntime&) = delete;
  TalkerNpuRuntime& operator=(const TalkerNpuRuntime&) = delete;

  /// Loads and quantizes the talker of `model_root` onto the NPU, with KV
  /// capacity for `maximum_tokens` positions. Null with `error` set when the
  /// NPU, its designs or the weights are unavailable.
  [[nodiscard]] static std::unique_ptr<TalkerNpuRuntime> Create(
      const std::string& model_root, std::size_t maximum_tokens,
      std::string* error = nullptr);

  [[nodiscard]] bool BuildCustomVoicePrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
      std::string_view language, hip::CustomVoicePromptOutput* output,
      std::string* error = nullptr) override;
  [[nodiscard]] bool BuildVoiceDesignPrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view language,
      hip::TalkerPromptOutput* output, std::string* error = nullptr) override;
  [[nodiscard]] bool BuildVoiceClonePrompt(
      const hip::VoiceClonePromptInput& input, hip::TalkerPromptOutput* output,
      std::string* error = nullptr) override;
  [[nodiscard]] bool Generate(
      const hip::TalkerPromptOutput& prompt, std::size_t maximum_new_tokens,
      const SamplingOptions& sampling, const CancellationCheck& is_cancelled,
      hip::TalkerGenerationOutput* output, std::string* error = nullptr,
      const std::function<bool(const hip::TalkerGenerationOutput&)>& on_codes =
          {}) override;

  struct Impl;

private:
  explicit TalkerNpuRuntime(std::unique_ptr<Impl> impl);
  bool BuildConditionedPrompt(std::span<const std::uint32_t> input_ids,
                              std::span<const std::uint32_t> instruction_ids,
                              std::string_view speaker,
                              std::string_view language, bool include_speaker,
                              hip::TalkerPromptOutput* output,
                              std::string* error);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::npu

#endif  // GUFO_MODELS_QWEN3_TTS_NPU_TALKER_NPU_HPP_
