#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Prompt lookup: proposals copied from the session's own committed tokens.
/// The drafter's context (committed tokens plus the cycle's chain so far)
/// ends in a key of kKeyTokens; among the most recent earlier occurrences
/// of that key, the one whose preceding tokens match the context furthest
/// back wins, and the tokens that followed it are the proposals. A match
/// shorter than kMinMatch tokens proposes nothing: short matches ("of the")
/// are poor predictions at sampling temperature and would displace MTP
/// drafts (2026-09-25 study: an ungated 3-token rule cost 6-8% on every
/// task; >= 12 matched tokens were accepted 82-94% of the time).
///
/// Proposals depend only on the prefix, so rejection sampling (with the
/// proposal as a point mass) keeps the target distribution exact.
class PromptLookup {
public:
  static constexpr std::size_t kKeyTokens = 3;
  static constexpr std::size_t kMinMatch = 12;
  static constexpr std::size_t kMaxMatch = 64;
  static constexpr std::size_t kMaxCandidates = 16;

  struct Match {
    std::size_t start{0};   ///< committed index of the first proposal
    std::size_t length{0};  ///< matched context tokens (>= kKeyTokens)
  };

  void Clear() {
    head_.Clear();
    prev_.clear();
  }

  /// Indexes committed tokens appended since the last call. `committed`
  /// must extend what was indexed before (Clear() after anything else).
  /// A shorter history, or one whose last indexed token changed, restarts
  /// the index; proposals are verified anyway, so this only keeps a missed
  /// Clear() from proposing stale copies.
  void Extend(std::span<const std::int32_t> committed) {
    if (committed.size() < prev_.size() ||
        (!prev_.empty() && committed[prev_.size() - 1] != last_)) {
      Clear();
    }
    for (std::size_t end = prev_.size(); end < committed.size(); ++end) {
      // prev_[end] chains occurrences of the key ending just before `end`.
      std::uint32_t previous = kNone;
      if (end >= kKeyTokens) {
        previous = head_.Exchange(
            Key(committed.subspan(end - kKeyTokens, kKeyTokens)),
            static_cast<std::uint32_t>(end));
      }
      prev_.push_back(previous);
    }
    if (!prev_.empty()) {
      last_ = committed[prev_.size() - 1];
    }
    // The newest key (ending at committed.size()) has no continuation yet;
    // it is indexed once the next token commits.
  }

  /// Best match for the context `committed` + `tail` among the indexed
  /// committed tokens, or a zero-length match. `tail` holds the uncommitted
  /// chain (anchor and drafts); `committed` must be what Extend() saw.
  [[nodiscard]] Match Find(std::span<const std::int32_t> committed,
                           std::span<const std::int32_t> tail) const {
    const std::size_t context = committed.size() + tail.size();
    if (context < kKeyTokens) {
      return {};
    }
    const auto at = [&](std::size_t i) {
      return i < committed.size() ? committed[i] : tail[i - committed.size()];
    };
    std::int32_t key_tokens[kKeyTokens];
    for (std::size_t i = 0; i < kKeyTokens; ++i) {
      key_tokens[i] = at(context - kKeyTokens + i);
    }
    Match best{};
    std::uint32_t end = head_.Find(Key(key_tokens));
    for (std::size_t n = 0; n < kMaxCandidates && end != kNone; ++n) {
      // An occurrence ending at `end` proposes committed[end...]. The key
      // itself matched; extend backwards against the context.
      std::size_t length = kKeyTokens;
      while (length < kMaxMatch && end > length && context > length &&
             committed[end - length - 1] == at(context - length - 1)) {
        ++length;
      }
      if (end < committed.size() && length > best.length) {
        best = {end, length};
      }
      end = prev_[end];
    }
    return best.length >= kMinMatch ? best : Match{};
  }

private:
  static constexpr std::uint32_t kNone =
      std::numeric_limits<std::uint32_t>::max();

  /// Key -> end of its most recent occurrence: open addressing with linear
  /// probing, at most half full. Clear() keeps the capacity, so a session
  /// re-indexing after a restore allocates nothing.
  class Heads {
  public:
    /// Sets `key` to `value` and returns its previous value, or kNone.
    std::uint32_t Exchange(std::uint64_t key, std::uint32_t value) {
      if (2 * (size_ + 1) > keys_.size()) {
        Grow();
      }
      std::size_t i = Slot(key);
      while (keys_[i] != kEmpty && keys_[i] != key) {
        i = (i + 1) & (keys_.size() - 1);
      }
      if (keys_[i] == kEmpty) {
        keys_[i] = key;
        ++size_;
        values_[i] = value;
        return kNone;
      }
      const std::uint32_t previous = values_[i];
      values_[i] = value;
      return previous;
    }

    [[nodiscard]] std::uint32_t Find(std::uint64_t key) const {
      if (size_ == 0) {
        return kNone;
      }
      for (std::size_t i = Slot(key);; i = (i + 1) & (keys_.size() - 1)) {
        if (keys_[i] == key) {
          return values_[i];
        }
        if (keys_[i] == kEmpty) {
          return kNone;
        }
      }
    }

    void Clear() {
      if (size_ != 0) {
        std::fill(keys_.begin(), keys_.end(), kEmpty);
        size_ = 0;
      }
    }

  private:
    /// Keys use 63 bits (three 21-bit ids), so all ones is never a key.
    static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};

    [[nodiscard]] std::size_t Slot(std::uint64_t key) const {
      // Fibonacci hashing: the multiply's top bits index the table.
      return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >>
                                      shift_);
    }

    void Grow() {
      const std::size_t capacity = keys_.empty() ? 1024 : 2 * keys_.size();
      std::vector<std::uint64_t> keys(capacity, kEmpty);
      std::vector<std::uint32_t> values(capacity);
      keys.swap(keys_);
      values.swap(values_);
      shift_ = 64;
      for (std::size_t c = capacity; c > 1; c >>= 1) {
        --shift_;
      }
      size_ = 0;
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] != kEmpty) {
          std::size_t j = Slot(keys[i]);
          while (keys_[j] != kEmpty) {
            j = (j + 1) & (keys_.size() - 1);
          }
          keys_[j] = keys[i];
          values_[j] = values[i];
          ++size_;
        }
      }
    }

    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> values_;
    std::size_t size_{0};
    unsigned shift_{64};
  };

  /// Token ids fit in 21 bits (vocabularies up to 2M), so a key packs
  /// exactly: equal keys are equal token triples.
  static std::uint64_t Key(std::span<const std::int32_t> tokens) {
    std::uint64_t key = 0;
    for (const auto t : tokens) {
      key = (key << 21) |
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(t)) &
             0x1FFFFF);
    }
    return key;
  }

  Heads head_;
  std::vector<std::uint32_t> prev_;
  std::int32_t last_{0};  ///< committed[prev_.size() - 1] when indexed
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
