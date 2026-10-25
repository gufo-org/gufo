#include "src/models/qwen/mtp_draft_vocab.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sp = gufo::speculative;
using gufo::tokenization::TokenId;

namespace {

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr TokenId kPrefix = sp::kMtpDraftVocab;
// Two special tokens above the prefix, as in the Qwen vocabulary.
constexpr TokenId kImStart = 248045;
constexpr TokenId kImEnd = 248046;
constexpr std::size_t kWindow =
    gufo::models::qwen38_flash_next::kMtpDraftVocabWindow;
// Ordinary tokens above the prefix.
constexpr TokenId kRareA = kPrefix + 10;
constexpr TokenId kRareB = kPrefix + 20;

sp::QwenMtpDraftTable MakeTable() {
  return sp::QwenMtpDraftTable(
      {static_cast<std::int32_t>(kImStart), static_cast<std::int32_t>(kImEnd)});
}

using Specials = std::vector<std::pair<std::string, TokenId>>;
constexpr std::string_view kArch = sp::kQwenMtpDraftArchitecture;

// Only Qwen3.6 35B-A3B (qwen35moe, Qwen vocabulary) qualifies; the table opens
// with the special tokens above the prefix, ascending and once each.
void CheckSpecials() {
  const Specials specials{{"<|im_end|>", kImEnd},
                          {"<|im_start|>", kImStart},
                          {"<|low|>", 7},
                          {"<|im_end|>", kImEnd}};
  const auto ids =
      sp::QwenMtpDraftSpecials(kArch, sp::kQwenMtpDraftVocabRows, specials);
  Require(ids.has_value(), "the Qwen vocabulary lost the small head");
  Require(*ids == std::vector<std::int32_t>{static_cast<std::int32_t>(kImStart),
                                            static_cast<std::int32_t>(kImEnd)},
          "special rows");
  Require(!sp::QwenMtpDraftSpecials(kArch, 151936, specials),
          "another vocabulary drafted small");
  // The dense Qwen3.8 27B shares the vocabulary but is not qualified.
  Require(
      !sp::QwenMtpDraftSpecials("qwen35", sp::kQwenMtpDraftVocabRows, specials),
      "another architecture drafted small");
  const Specials outside{{"<|out|>", sp::kQwenMtpDraftVocabRows}};
  Require(!sp::QwenMtpDraftSpecials(kArch, sp::kQwenMtpDraftVocabRows, outside),
          "a special token outside the vocabulary");
  Specials crowded;
  for (TokenId id = 0; id < sp::kMtpDraftExtraRows; ++id)
    crowded.emplace_back("<|s" + std::to_string(id) + "|>", kPrefix + id);
  Require(!sp::QwenMtpDraftSpecials(kArch, sp::kQwenMtpDraftVocabRows, crowded),
          "special tokens filled the table");
  crowded.pop_back();
  Require(sp::QwenMtpDraftSpecials(kArch, sp::kQwenMtpDraftVocabRows, crowded)
                  ->size() == sp::kMtpDraftExtraRows - 1,
          "one row left for the request's own tokens");
}

// The anchor (the sequence's last token) is decided against the history
// before it and enters the table with the next proposal, as in Flash-Next.
void CheckAnchor() {
  auto table = MakeTable();
  std::vector<TokenId> tokens{1, 2, kRareA};
  auto update = table.Sync(tokens);
  Require(update.restart && update.added.empty(), "the anchor was tabled");
  Require(!table.Fits(kRareA), "an untabled anchor drafted small");
  Require(table.Fits(kImEnd) && table.Fits(3), "common anchors");
  tokens.push_back(3);
  update = table.Sync(tokens);
  Require(!update.restart && update.added.size() == 1 &&
              update.added[0] == static_cast<std::int32_t>(kRareA),
          "the previous anchor was not tabled");
  Require(table.rows() == 3, "rows are the specials and the tabled token");
  Require(table.Fits(3) && table.Fits(kRareA), "a lone miss left the head");
  // A second miss leaves the small head until the first leaves the window.
  tokens.insert(tokens.end(), {kRareB, 4});
  (void)table.Sync(tokens);
  Require(!table.Fits(4), "small head right after a second miss");
  tokens.resize(tokens.size() + kWindow, 5);
  (void)table.Sync(tokens);
  Require(table.Fits(5) && table.Fits(kRareB), "known text kept the full head");
}

// The table is a function of the sequence alone: catching up a step at a
// time, all at once, or after Forget gives the same rows and choices.
void CheckHistoryOnly() {
  std::vector<TokenId> tokens;
  for (TokenId step = 0; step < 80; ++step)
    tokens.push_back(step % 9 == 0 ? kPrefix + step : step);
  auto stepwise = MakeTable();
  std::vector<std::int32_t> rows;
  for (std::size_t size = 1; size <= tokens.size(); ++size) {
    const auto update =
        stepwise.Sync(std::span<const TokenId>(tokens.data(), size));
    if (update.restart)
      rows.clear();
    rows.insert(rows.end(), update.added.begin(), update.added.end());
  }
  auto whole = MakeTable();
  const auto update = whole.Sync(tokens);
  Require(std::vector<std::int32_t>(update.added.begin(), update.added.end()) ==
              rows,
          "stepwise and whole histories tabled different rows");
  Require(whole.rows() == stepwise.rows(), "table sizes differ");
  for (const TokenId anchor : {TokenId{1}, kPrefix + 9, kPrefix + 1, kImEnd})
    Require(whole.Fits(anchor) == stepwise.Fits(anchor), "choices differ");

  stepwise.Forget();
  const auto rebuilt = stepwise.Sync(tokens);
  Require(
      rebuilt.restart && std::vector<std::int32_t>(rebuilt.added.begin(),
                                                   rebuilt.added.end()) == rows,
      "Forget did not rebuild the same table");
  // A shorter sequence (a rewound session) rebuilds as well.
  const auto rewound =
      stepwise.Sync(std::span<const TokenId>(tokens.data(), 1));
  Require(rewound.restart && stepwise.rows() == 2, "rewind kept rows");
}

// Tokens outside the vocabulary are misses that never take a table row, and
// a full table turns further rare tokens into misses.
void CheckLimits() {
  auto table = MakeTable();
  std::vector<TokenId> tokens{gufo::tokenization::kInvalidTokenId,
                              sp::kQwenMtpDraftVocabRows, 1, 2};
  auto update = table.Sync(tokens);
  Require(update.added.empty() && table.rows() == 2,
          "an out-of-vocabulary ID was tabled");
  Require(!table.Fits(gufo::tokenization::kInvalidTokenId) &&
              !table.Fits(sp::kQwenMtpDraftVocabRows),
          "an out-of-vocabulary anchor drafted small");

  sp::QwenMtpDraftTable full(std::vector<std::int32_t>(
      sp::kMtpDraftExtraRows - 1, static_cast<std::int32_t>(kImEnd)));
  tokens = {kRareA, kRareB};
  tokens.resize(tokens.size() + kWindow + 2, 1);
  update = full.Sync(tokens);
  Require(update.added.size() == 1 && full.rows() == sp::kMtpDraftExtraRows,
          "the table overfilled");
  Require(full.Fits(kRareA) && !full.Fits(kRareB), "an untabled token fits");
}

}  // namespace

int main() {
  try {
    CheckSpecials();
    CheckAnchor();
    CheckHistoryOnly();
    CheckLimits();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
