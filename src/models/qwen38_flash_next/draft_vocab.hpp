#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DRAFT_VOCAB_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Output rows the MTP draft head scores. Each draft token projects through
/// the output head; over all 248,320 rows that is a 675 MB read, about 3 ms
/// on gfx1151 and 12-18% of decode time. Byte-level BPE numbers tokens in
/// merge order, and in this vocabulary the early merges are the frequent
/// tokens: the first 98,304 cover ~99.8% of the tokens in English code and
/// prose (measured, not guaranteed; other text falls back to the full head).
/// The draft scores those rows in place, plus a small per-request table: the
/// special tokens (chat, thinking and tool-call markers at the end of the
/// vocabulary) and every other token the request has already used.
///
/// Only proposals change. The target verifies every drafted token over the
/// full vocabulary, so greedy output is unchanged. Sampled proposals are
/// corrected against the distribution they were drawn from.
inline constexpr std::uint32_t kMtpDraftVocab = 98304;

/// Rows of the per-request table: the special tokens, then the request's own
/// tokens above kMtpDraftVocab in first-use order.
inline constexpr std::uint32_t kMtpDraftExtraRows = 1024;

/// Committed tokens the head choice looks back over.
inline constexpr std::size_t kMtpDraftVocabWindow = 32;

/// Whether a request's cycle may draft through the small head; the table then
/// decides per cycle (MtpDraftVocab::Fits). Sampled requests may at any
/// session count, with penalties and constraints: verification corrects
/// their proposals against the full target distribution, and the choice
/// reads only the request's own tokens, so seeded output does not depend on
/// overlapping requests. Greedy penalties/constraints keep the full head.
/// Other greedy requests use the small head on one session or in a jointly
/// planned batch; unplanned multi-session greedy requests keep the full head.
[[nodiscard]] constexpr bool MtpDraftSubsetAllowed(
    bool sampled, bool plain_greedy, std::uint32_t sessions,
    bool planned_batch) noexcept {
  return sampled || (plain_greedy && (sessions == 1 || planned_batch));
}

/// The request's draft vocabulary and when to use it. A token is a miss when
/// it is outside the prefix, not special and not yet in the request's table
/// (or the table is full). A draft cycle uses the small head when its anchor
/// is not a miss and at most one of the last kMtpDraftVocabWindow committed
/// tokens was, so text that keeps bringing new tokens (a short reply in
/// another script) drafts over the full vocabulary as before, a lone new
/// word does not leave the small head, and a request that repeats its own
/// vocabulary (code and comments in the user's language) gets the small
/// head. Everything is a function of the request's own token history, so
/// batched, isolated and restored sessions draft the same way.
class MtpDraftVocab {
public:
  /// `rare_rows`: table rows left after the special tokens.
  explicit MtpDraftVocab(std::uint32_t rare_rows = 0) : capacity_(rare_rows) {}

  void Reset() {
    rare_.clear();
    known_.clear();
    scanned_ = 0;
    last_miss_.reset();
    prior_miss_.reset();
  }

  /// Takes in `committed[scanned()..]`. Appends the newly tabled tokens to
  /// `added` in table order. `common(id)` is true for prefix and special IDs.
  template<typename Common>
  void CatchUp(std::span<const std::int32_t> committed, Common common,
               std::vector<std::int32_t>* added) {
    for (; scanned_ < committed.size(); ++scanned_) {
      const std::int32_t id = committed[scanned_];
      if (common(id) || Known(id))
        continue;
      prior_miss_ = last_miss_;
      last_miss_ = scanned_;
      if (id < 0 || rare_.size() >= capacity_)
        continue;
      if (known_.size() <= static_cast<std::size_t>(id))
        known_.resize(static_cast<std::size_t>(id) + 1, false);
      known_[static_cast<std::size_t>(id)] = true;
      rare_.push_back(id);
      if (added != nullptr)
        added->push_back(id);
    }
  }

  /// Whether a cycle anchored on `anchor`, after the scanned tokens, drafts
  /// through the small head.
  template<typename Common>
  [[nodiscard]] bool Fits(std::int32_t anchor, Common common) const {
    return (common(anchor) || Known(anchor)) && Quiet();
  }

  /// Whether the window allows the small head; Fits also needs the anchor.
  [[nodiscard]] bool Quiet() const noexcept {
    return !prior_miss_ || *prior_miss_ + kMtpDraftVocabWindow < scanned_;
  }

  [[nodiscard]] std::size_t scanned() const noexcept { return scanned_; }
  /// Tabled rare tokens, in table order.
  [[nodiscard]] std::span<const std::int32_t> rare() const noexcept {
    return rare_;
  }

private:
  [[nodiscard]] bool Known(std::int32_t id) const {
    return id >= 0 && static_cast<std::size_t>(id) < known_.size() &&
           known_[static_cast<std::size_t>(id)];
  }

  std::uint32_t capacity_;
  std::vector<std::int32_t> rare_;
  std::vector<bool> known_;
  std::size_t scanned_{0};
  std::optional<std::size_t> last_miss_;
  std::optional<std::size_t> prior_miss_;  ///< the miss before last_miss_
};

}  // namespace gufo::models::qwen38_flash_next

#endif
