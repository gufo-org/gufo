#ifndef GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"

namespace gufo::speculative {

using models::qwen38_flash_next::kMtpDraftExtraRows;
using models::qwen38_flash_next::kMtpDraftVocab;

/// Rows of the Qwen3.5/3.6/3.8 vocabulary. Flash-Next's small draft head
/// (draft_vocab.hpp) assumes this byte-level BPE vocabulary, whose early
/// merges are its frequent tokens. The Qwen3.6 35B-A3B GGUF carries the same
/// tokens, token types and merges as the Flash-Next GGUF.
inline constexpr std::uint32_t kQwenMtpDraftVocabRows = 248320;

/// The architecture whose MTP drafts through the small head: Qwen3.6
/// 35B-A3B, where it is measured. Other targets keep the full head.
inline constexpr std::string_view kQwenMtpDraftArchitecture = "qwen35moe";

/// The special tokens that open every draft table, ascending: those above
/// the prefix. Empty optional when the draft keeps the full head: another
/// architecture or vocabulary, or more special tokens than table rows.
[[nodiscard]] inline std::optional<std::vector<std::int32_t>>
QwenMtpDraftSpecials(
    std::string_view architecture, std::uint32_t vocab_size,
    std::span<const std::pair<std::string, tokenization::TokenId>> specials) {
  if (architecture != kQwenMtpDraftArchitecture ||
      vocab_size != kQwenMtpDraftVocabRows)
    return std::nullopt;
  std::vector<std::int32_t> ids;
  for (const auto& [name, id] : specials) {
    if (id >= vocab_size)
      return std::nullopt;
    if (id >= kMtpDraftVocab)
      ids.push_back(static_cast<std::int32_t>(id));
  }
  std::ranges::sort(ids);
  ids.erase(std::ranges::unique(ids).begin(), ids.end());
  if (ids.size() >= kMtpDraftExtraRows)
    return std::nullopt;
  return ids;
}

/// One session's draft table and head choice (MtpDraftVocab), kept in step
/// with the token sequence each proposal receives: committed history, then
/// the anchor. As in Flash-Next, the anchor is decided against the history
/// before it and enters the table with the next proposal. The table is a
/// function of that sequence alone, so serial, batched and restored sessions
/// draft the same way.
class QwenMtpDraftTable {
public:
  explicit QwenMtpDraftTable(std::vector<std::int32_t> specials)
      : specials_(std::move(specials)),
        vocab_(kMtpDraftExtraRows -
               static_cast<std::uint32_t>(specials_.size())) {}

  /// What Sync changed: when `restart` is set, the rows above the special
  /// tokens are dropped before `added` is appended in order.
  struct Update {
    bool restart{false};
    std::span<const std::int32_t> added;
  };

  /// The session's history was replaced (reset, restore, new request): the
  /// next Sync rescans it from the first token.
  void Forget() noexcept { stale_ = true; }

  /// Brings the table up to `tokens` and returns the change. The span stays
  /// valid until the next call.
  [[nodiscard]] Update Sync(std::span<const tokenization::TokenId> tokens) {
    const std::size_t size = tokens.empty() ? 0 : tokens.size() - 1;
    Update update;
    if (stale_ || history_.size() > size) {
      vocab_.Reset();
      history_.clear();
      stale_ = false;
      update.restart = true;
    }
    for (std::size_t index = history_.size(); index < size; ++index)
      history_.push_back(Id(tokens[index]));
    added_.clear();
    vocab_.CatchUp(
        history_, [this](std::int32_t id) { return Common(id); }, &added_);
    update.added = added_;
    return update;
  }

  /// Whether a proposal anchored on `anchor`, after the synced history,
  /// scores the small head.
  [[nodiscard]] bool Fits(tokenization::TokenId anchor) const {
    return vocab_.Fits(Id(anchor),
                       [this](std::int32_t id) { return Common(id); });
  }

  /// Table rows in use: the special tokens, then the tabled tokens.
  [[nodiscard]] std::uint32_t rows() const noexcept {
    return static_cast<std::uint32_t>(specials_.size() + vocab_.rare().size());
  }
  [[nodiscard]] std::span<const std::int32_t> specials() const noexcept {
    return specials_;
  }

private:
  /// IDs outside the vocabulary read as -1: misses that never take a row.
  [[nodiscard]] static std::int32_t Id(tokenization::TokenId token) noexcept {
    return token < kQwenMtpDraftVocabRows ? static_cast<std::int32_t>(token)
                                          : -1;
  }
  [[nodiscard]] bool Common(std::int32_t id) const {
    return id >= 0 && (static_cast<std::uint32_t>(id) < kMtpDraftVocab ||
                       std::ranges::binary_search(specials_, id));
  }

  std::vector<std::int32_t> specials_;
  models::qwen38_flash_next::MtpDraftVocab vocab_;
  std::vector<std::int32_t> history_;  ///< scanned tokens, as table IDs
  std::vector<std::int32_t> added_;
  bool stale_{true};
};

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_MTP_DRAFT_VOCAB_HPP_
