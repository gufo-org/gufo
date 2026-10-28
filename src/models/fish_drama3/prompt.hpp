#ifndef GUFO_MODELS_FISH_DRAMA3_PROMPT_HPP_
#define GUFO_MODELS_FISH_DRAMA3_PROMPT_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "src/models/fish_drama3/config.hpp"
#include "src/models/fish_drama3/tokenizer.hpp"

namespace gufo::models::fish_drama3 {

// Slot 0 is the backbone input token (either text token ID or semantic_begin +
// vq[0]). Slots 1..10 are the 10 codebook tokens (0 when text-only frame).
using Drama3Frame = std::array<std::int32_t, 11>;

struct FishDacCodes {
  std::int64_t codebooks = 10;
  std::int64_t frames = 0;
  // Layout: [codebooks, frames] row-major, i.e. codes[book * frames + frame]
  std::vector<std::int32_t> codes;
};

struct Drama3Take {
  std::string text;
  FishDacCodes codes;
};

void ValidateDrama3Text(const std::string& text, const char* location);
void ValidateDrama3Codes(const FishDacCodes& codes, std::int64_t codebook_size);
[[nodiscard]] std::vector<Drama3Frame> BuildDrama3PromptPrefix(
    const ModelConfig& config, const Drama3Tokenizer& tokenizer,
    const std::vector<Drama3Take>& references = {},
    const std::vector<Drama3Take>& history = {});
[[nodiscard]] std::vector<Drama3Frame> BuildDrama3PromptSuffix(
    const Drama3Tokenizer& tokenizer, const std::string& text);
[[nodiscard]] std::vector<Drama3Frame> BuildDrama3Prompt(
    const ModelConfig& config, const Drama3Tokenizer& tokenizer,
    const std::string& text, const std::vector<Drama3Take>& references = {},
    const std::vector<Drama3Take>& history = {});

}  // namespace gufo::models::fish_drama3

#endif  // GUFO_MODELS_FISH_DRAMA3_PROMPT_HPP_
