// Hand-written scanner equivalent to the Inkling 2 split regex (tokenizer.json
// pre_tokenizer, behavior Isolated). std::regex is not used.
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// |
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | \p{N}{1,3}
// | ?[^\s\p{L}\p{N}]+[\r\n/]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace inkling2::text {

/// Appends the [begin, end) byte ranges of the pieces of `text`, in order. The
/// ranges tile `text` exactly.
void SplitPieces(std::string_view text,
                 std::vector<std::pair<std::uint32_t, std::uint32_t>>* out);

}  // namespace inkling2::text
