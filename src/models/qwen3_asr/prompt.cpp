#include "src/models/qwen3_asr/prompt.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen3_asr/tokenizer.hpp"

namespace gufo::models::qwen3_asr {
namespace {

constexpr std::uint32_t kImStart = 151644;
constexpr std::uint32_t kImEnd = 151645;
constexpr std::uint32_t kAudioStart = 151669;
constexpr std::uint32_t kAudioEnd = 151670;
constexpr std::uint32_t kAudioPad = 151676;
constexpr std::uint32_t kNewline = 198;
constexpr std::uint32_t kSystem = 8948;
constexpr std::uint32_t kUser = 872;
constexpr std::uint32_t kAssistant = 77091;
constexpr std::string_view kAsrTextTag = "<asr_text>";
constexpr std::string_view kLanguagePrefix = "language ";

void Append(std::vector<std::uint32_t>* destination,
            std::span<const std::uint32_t> values) {
  destination->insert(destination->end(), values.begin(), values.end());
}

std::string Trim(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return std::string(text);
}

}  // namespace

std::vector<std::uint32_t> BuildPrompt(const Tokenizer& tokenizer,
                                       const PromptOptions& options) {
  if (options.audio_embedding_tokens == 0U) {
    throw std::invalid_argument(
        "Qwen3-ASR prompt requires at least one audio embedding token");
  }
  std::vector<std::uint32_t> result;
  result.reserve(15U + options.audio_embedding_tokens +
                 options.context.size() / 2U);
  const std::array system_prefix{kImStart, kSystem, kNewline};
  Append(&result, system_prefix);
  if (!options.context.empty()) {
    const std::vector<std::uint32_t> context =
        tokenizer.Encode(options.context);
    Append(&result, context);
  }
  const std::array system_suffix{kImEnd, kNewline, kImStart,
                                 kUser,  kNewline, kAudioStart};
  Append(&result, system_suffix);
  result.insert(result.end(), options.audio_embedding_tokens, kAudioPad);
  const std::array audio_suffix{kAudioEnd, kImEnd,     kNewline,
                                kImStart,  kAssistant, kNewline};
  Append(&result, audio_suffix);
  if (options.language.has_value()) {
    const std::vector<std::uint32_t> language =
        tokenizer.Encode(std::string(kLanguagePrefix) + *options.language +
                         std::string(kAsrTextTag));
    Append(&result, language);
  }
  return result;
}

Transcription ParseTranscription(
    std::string_view decoded,
    const std::optional<std::string>& forced_language) {
  if (forced_language.has_value()) {
    return {
        .language = *forced_language,
        .text = Trim(decoded),
    };
  }

  const std::size_t tag = decoded.find(kAsrTextTag);
  if (tag == std::string_view::npos) {
    return {.text = Trim(decoded)};
  }
  const std::string_view metadata = decoded.substr(0, tag);
  std::string language;
  const std::size_t prefix = metadata.find(kLanguagePrefix);
  if (prefix != std::string_view::npos) {
    language = Trim(metadata.substr(prefix + kLanguagePrefix.size()));
    std::string lower = language;
    std::ranges::transform(lower, lower.begin(), [](unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
    if (lower == "none") {
      return {};
    }
  }
  return {
      .language = std::move(language),
      .text = Trim(decoded.substr(tag + kAsrTextTag.size())),
  };
}

}  // namespace gufo::models::qwen3_asr
