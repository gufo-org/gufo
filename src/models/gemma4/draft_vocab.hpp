#ifndef GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_

#include <cstdint>
#include <vector>

#include "src/models/gemma4/tokenizer.hpp"

namespace gufo::models::gemma4 {

/// Drafter head rows an eligible draft step scores. The `gemma4-assistant`
/// head projects onto all 262,144 tokens: as Q4_K a 151 MB read, 0.65 ms on
/// gfx1151 and about a third of a drafter step. Gemma's token ids are not
/// ordered by frequency, but the merge that creates a token is a usable
/// proxy, so the device keeps the head rows in DraftHeadOrder. On 26B
/// outputs the first 98,304 rows (plus the request's own tokens, see
/// qwen38_flash_next::MtpDraftVocab) serve 92-100% of code draft cycles and
/// 55-72% of English prose cycles; other scripts fall back to the full head.
///
/// Only proposals change. The target verifies every drafted token over the
/// full vocabulary, so greedy output is unchanged, and sampled proposals are
/// corrected against the top candidates they were drawn from.
///
/// Only the 26B-A4B uses it: its ~30 ms cycle gains 4-6% on code and up to
/// 3.5% on English prose. The dense 31B's ~130 ms cycle gains nothing
/// measurable and loses up to 5% to missed proposals in other languages.
inline constexpr std::uint32_t kDraftHeadRows = 98304;

/// Per-session head rows holding the request's tokens beyond kDraftHeadRows,
/// in first-use order.
inline constexpr std::uint32_t kDraftTableRows = 1024;

/// The token of each drafter head row in device order: control and
/// user-defined tokens, then tokens by the rank of the earliest merge that
/// creates them, with single Latin, punctuation and space symbols ranked
/// with the first merges; every other token follows by id.
[[nodiscard]] std::vector<TokenId> DraftHeadOrder(const Tokenizer& tokenizer);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_
