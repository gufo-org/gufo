#ifndef GUFO_MODELS_QWEN3_ASR_PROMPT_HPP_
#define GUFO_MODELS_QWEN3_ASR_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen3_asr {

class Tokenizer;

struct PromptOptions {
  std::string context;
  std::optional<std::string> language;
  std::size_t audio_embedding_tokens{0};
};

struct Transcription {
  std::string language;
  std::string text;
};

[[nodiscard]] std::vector<std::uint32_t> BuildPrompt(
    const Tokenizer& tokenizer, const PromptOptions& options);

[[nodiscard]] Transcription ParseTranscription(
    std::string_view decoded,
    const std::optional<std::string>& forced_language);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_PROMPT_HPP_
