#include "src/models/yue2/protocol.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::models::yue2 {
namespace {

std::string_view Instruction(Planning planning) {
  switch (planning) {
    case Planning::kOff:
      return "Generate music with codec tokens from the given conditions.";
    case Planning::kMelody:
      return "Generate a melody-only ABC transcription without chord symbols, "
             "then generate music with codec tokens from the given "
             "conditions.";
    case Planning::kFull:
      break;
  }
  return "Generate a chord-annotated ABC transcription, then generate music "
         "with codec tokens from the given conditions.";
}

void Append(std::vector<std::uint32_t>* ids,
            std::span<const std::uint32_t> values) {
  ids->insert(ids->end(), values.begin(), values.end());
}

}  // namespace

std::optional<Planning> ParsePlanning(std::string_view name) {
  if (name == "off")
    return Planning::kOff;
  if (name == "melody")
    return Planning::kMelody;
  if (name == "full")
    return Planning::kFull;
  return std::nullopt;
}

std::string_view PlanningName(Planning planning) {
  switch (planning) {
    case Planning::kOff:
      return "off";
    case Planning::kMelody:
      return "melody";
    case Planning::kFull:
      break;
  }
  return "full";
}

Sampling DefaultAbcSampling() {
  return {.temperature = 0.7F,
          .top_p = 0.9F,
          .top_k = 30,
          .repetition_penalty = 1.005F,
          .penalty_window = 100,
          .min_tokens = 32,
          .max_tokens = 4096};
}

Sampling DefaultSemanticSampling() {
  return {};
}

bool ValidateSampling(const Sampling& sampling, std::string* error) {
  const auto fail = [&](std::string_view message) {
    if (error != nullptr)
      *error = message;
    return false;
  };
  if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
      !std::isfinite(sampling.repetition_penalty))
    return fail("sampling numbers must be finite");
  if (sampling.temperature < 0.0F || sampling.temperature > 5.0F)
    return fail("temperature must be in [0, 5]");
  if (!(sampling.top_p > 0.0F) || sampling.top_p > 1.0F)
    return fail("top_p must be in (0, 1]");
  if (sampling.top_k < 1)
    return fail("top_k must be at least 1");
  if (!(sampling.repetition_penalty > 0.0F))
    return fail("repetition_penalty must be positive");
  if (sampling.penalty_window < 1 || sampling.penalty_window > 100)
    return fail("penalty_window must be in [1, 100]");
  if (sampling.max_tokens < 1 || sampling.min_tokens > sampling.max_tokens)
    return fail("require 0 <= min_tokens <= max_tokens and max_tokens >= 1");
  return true;
}

std::string PromptText(const SongPrompt& prompt) {
  std::string text(Instruction(prompt.planning));
  text += "\n[Tags]\n";
  text += prompt.style;
  text += "\n[Lyrics]\n";
  text += prompt.lyrics;
  text += "\n";
  return text;
}

std::vector<std::uint32_t> PlanPrefix(const TextTokenizer& tokenizer,
                                      const SongPrompt& prompt) {
  std::vector<std::uint32_t> ids{kEndOfDocument};
  Append(&ids, tokenizer.Encode(PromptText(prompt)));
  ids.push_back(kAbcStart);
  if (prompt.planning == Planning::kOff) {
    ids.push_back(kAbcEnd);
    ids.push_back(kMusicStart);
  }
  return ids;
}

std::vector<std::uint32_t> SemanticPrefix(
    const TextTokenizer& tokenizer, const SongPrompt& prompt,
    std::span<const std::uint32_t> abc_ids) {
  std::vector<std::uint32_t> ids = PlanPrefix(tokenizer, prompt);
  if (prompt.planning == Planning::kOff)
    return ids;
  Append(&ids, abc_ids);
  ids.push_back(kAbcEnd);
  ids.push_back(kMusicStart);
  return ids;
}

std::vector<std::uint32_t> NegativePrefix(
    const TextTokenizer& tokenizer, const SongPrompt& prompt,
    std::span<const std::uint32_t> abc_ids) {
  std::vector<std::uint32_t> ids{kEndOfDocument};
  Append(&ids, tokenizer.Encode(Instruction(prompt.planning)));
  if (prompt.planning != Planning::kOff) {
    ids.push_back(kAbcStart);
    Append(&ids, abc_ids);
    ids.push_back(kAbcEnd);
  }
  ids.push_back(kMusicStart);
  return ids;
}

float DefaultGuidance(Planning planning) {
  return planning == Planning::kOff ? 1.01F : 1.0F;
}

std::vector<std::pair<std::size_t, std::size_t>> ChunkRanges(
    std::size_t frames, std::size_t prefix_tokens) {
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  if (frames == 0 || prefix_tokens + 3 >= kContext)
    return ranges;
  const std::size_t size =
      std::min((kContext - prefix_tokens - 3) / 2, kContext);
  if (size == 0)
    return ranges;
  for (std::size_t begin = 0; begin < frames; begin += size)
    ranges.emplace_back(begin, std::min(begin + size, frames));
  return ranges;
}

}  // namespace gufo::models::yue2
