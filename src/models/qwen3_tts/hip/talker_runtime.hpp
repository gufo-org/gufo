#ifndef GUFO_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_
#define GUFO_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_tts/prompt.hpp"
#include "src/models/qwen3_tts/synthesis.hpp"

namespace gufo::models::qwen3_tts::hip {

struct TalkerPrefillOutput {
  std::vector<float> logits;
  std::vector<float> last_hidden;
  std::vector<float> layer0_output;
};

using TalkerPromptOutput = ::gufo::models::qwen3_tts::PromptOutput;
using CustomVoicePromptOutput = TalkerPromptOutput;

struct VoiceClonePromptInput {
  std::span<const std::uint32_t> input_ids;
  std::span<const std::uint32_t> reference_ids;
  // Frame-major [reference_frames, 16].
  std::span<const std::uint32_t> reference_codes;
  std::span<const float> speaker_embedding;
  std::size_t reference_frames{0};
  std::string_view language{"auto"};
  bool icl_mode{true};
};

struct TalkerGenerationOutput {
  std::vector<std::uint32_t> codes;
  std::size_t frames{0};
  std::size_t code_groups{0};
};

struct CodePredictorOutput {
  std::vector<std::uint32_t> codes;
  std::vector<float> logits;
};

/// Native gfx1151 talker prefill used as the first exact HIP parity boundary.
/// Input embeddings must use row-major [tokens, hidden_size] layout.
class TalkerHipRuntime {
public:
  ~TalkerHipRuntime();

  TalkerHipRuntime(const TalkerHipRuntime&) = delete;
  TalkerHipRuntime& operator=(const TalkerHipRuntime&) = delete;
  TalkerHipRuntime(TalkerHipRuntime&&) = delete;
  TalkerHipRuntime& operator=(TalkerHipRuntime&&) = delete;

  [[nodiscard]] static std::unique_ptr<TalkerHipRuntime> Create(
      const std::string& model_root, std::size_t maximum_tokens,
      std::string* error = nullptr);

  [[nodiscard]] bool Prefill(std::span<const float> input_embeddings,
                             std::size_t tokens, TalkerPrefillOutput* output,
                             std::string* error = nullptr);

  /// Builds the official non-streaming CustomVoice prompt embeddings.
  [[nodiscard]] bool BuildCustomVoicePrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
      std::string_view language, CustomVoicePromptOutput* output,
      std::string* error = nullptr);

  /// Builds the official VoiceDesign prompt. A non-empty instruction is
  /// required by the service even though the low-level builder accepts an
  /// empty row set for exact upstream compatibility.
  [[nodiscard]] bool BuildVoiceDesignPrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view language,
      TalkerPromptOutput* output, std::string* error = nullptr);

  /// Builds the Base model voice-clone prompt in full ICL or
  /// speaker-embedding-only mode.
  [[nodiscard]] bool BuildVoiceClonePrompt(const VoiceClonePromptInput& input,
                                           TalkerPromptOutput* output,
                                           std::string* error = nullptr);

  /// Greedily expands one first-codebook token into all codec groups.
  [[nodiscard]] bool PredictCodeFrame(std::span<const float> talker_hidden,
                                      std::uint32_t first_code,
                                      std::vector<std::uint32_t>* codes,
                                      std::string* error = nullptr);

  /// Expands a frame and retains one [vocab] logit row for each sub-codebook.
  [[nodiscard]] bool PredictCodeFrameTrace(std::span<const float> talker_hidden,
                                           std::uint32_t first_code,
                                           CodePredictorOutput* output,
                                           std::string* error = nullptr);

  /// Feeds one complete codec frame back into the cached talker.
  [[nodiscard]] bool DecodeCodeFrame(std::span<const std::uint32_t> codes,
                                     std::span<const float> text_embedding,
                                     TalkerPrefillOutput* output,
                                     std::string* error = nullptr);

  [[nodiscard]] bool BuildCodeFrameEmbedding(
      std::span<const std::uint32_t> codes,
      std::span<const float> text_embedding, std::vector<float>* embedding,
      std::string* error = nullptr);

  [[nodiscard]] bool GenerateGreedy(const TalkerPromptOutput& prompt,
                                    std::size_t maximum_new_tokens,
                                    TalkerGenerationOutput* output,
                                    std::string* error = nullptr);

  /// Generates with the checkpoint's native top-k sampling policy by default.
  [[nodiscard]] bool Generate(const TalkerPromptOutput& prompt,
                              std::size_t maximum_new_tokens,
                              const SamplingOptions& sampling,
                              TalkerGenerationOutput* output,
                              std::string* error = nullptr);

  /// Generates while checking for cancellation before prefill and every codec
  /// frame. A cancelled generation fails without returning partial codes.
  [[nodiscard]] bool Generate(
      const TalkerPromptOutput& prompt, std::size_t maximum_new_tokens,
      const SamplingOptions& sampling, const CancellationCheck& is_cancelled,
      TalkerGenerationOutput* output, std::string* error = nullptr,
      const std::function<bool(const TalkerGenerationOutput&)>& on_codes = {});

private:
  struct Impl;
  explicit TalkerHipRuntime(std::unique_ptr<Impl> impl);

  /// Shared body of the two frame predictors. `collect_logits` retains the
  /// per-sub-codebook logit rows, which only the trace entry point needs.
  [[nodiscard]] bool PredictFrame(std::span<const float> talker_hidden,
                                  std::uint32_t first_code,
                                  CodePredictorOutput* output,
                                  bool collect_logits, std::string* error);

  [[nodiscard]] bool BuildConditionedPrompt(
      std::span<const std::uint32_t> input_ids,
      std::span<const std::uint32_t> instruction_ids, std::string_view speaker,
      std::string_view language, bool include_speaker,
      TalkerPromptOutput* output, std::string* error);

  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen3_tts::hip

#endif  // GUFO_MODELS_QWEN3_TTS_HIP_TALKER_RUNTIME_HPP_
