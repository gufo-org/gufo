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
/// back wins, and the tokens that followed it are the proposals. Greedy
/// requests require kMinMatch tokens; sampled requests require the longer
/// kSampledMinMatch. Common code syntax and field names alone poorly predict
/// their continuation at temperature and can displace useful neural drafts.
/// Sampled requests index only their starting context: earlier parts of a
/// freshly generated answer are especially poor sources for new prose/code.
/// Greedy requests can also copy repetition within their current answer.
///
/// Where the last verified copy left off, kResumeMinMatch tokens are
/// enough: output that edits a line of the text it is copying rejoins it a
/// few tokens later, and waiting for 12 fresh matching tokens would draft
/// that stretch with MTP alone. A copy that keeps fewer than half of its
/// tokens cannot enable this shorter gate: even a long matching prefix can
/// precede a different field value or a new function body. A new request
/// starts with nothing to resume.
///
/// A request's first speculative step copies nothing. Its position also
/// bounds later query matches: a shared chat wrapper or tool header must not
/// count as a long matching answer before that answer has been generated.
///
/// Proposals depend only on the prefix, so rejection sampling (with the
/// proposal as a point mass) keeps the target distribution exact.
class PromptLookup {
public:
  static constexpr std::size_t kKeyTokens = 3;
  static constexpr std::size_t kMinMatch = 12;
  /// Sampled code and field extraction often share short syntax without
  /// sharing the continuation. Establish a longer match before resuming.
  static constexpr std::size_t kSampledMinMatch = 24;
  static constexpr std::size_t kMaxMatch = 64;
  static constexpr std::size_t kMaxCandidates = 16;
  static constexpr std::size_t kResumeMinMatch = 5;
  /// A resumed copy continues at most this far past the last copy's end,
  /// or kResumeBack tokens before it (the output repeated a few tokens).
  static constexpr std::size_t kResumeWindow = 512;
  static constexpr std::size_t kResumeBack = 8;

  struct Match {
    std::size_t start{0};   ///< committed index of the first proposal
    std::size_t length{0};  ///< matched context tokens (>= kKeyTokens)
  };

  /// Copy state the index cannot rebuild from the tokens. Session snapshots
  /// carry it, so a restored session copies as the original would have and
  /// seeded sampled output replays.
  struct State {
    std::uint32_t resume{kNoState};         ///< Observe()'s resume point
    std::uint32_t request_start{kNoState};  ///< first speculative position
  };
  static constexpr std::uint32_t kNoState =
      std::numeric_limits<std::uint32_t>::max();

  void Clear() {
    ClearIndex();
    request_start_ = kNoResume;
  }

  /// Called when a request starts: nothing to resume yet.
  void BeginRequest() {
    resume_ = kNoResume;
    request_start_ = kNoResume;
  }

  /// Called once per decode step; false on a request's first step.
  [[nodiscard]] bool Step(std::size_t position) {
    if (request_start_ != kNoResume)
      return true;
    request_start_ = position;
    return false;
  }

  [[nodiscard]] State Save() const {
    return {.resume = resume_ == kNoResume
                          ? kNoState
                          : static_cast<std::uint32_t>(resume_),
            .request_start = request_start_ == kNoResume
                                 ? kNoState
                                 : static_cast<std::uint32_t>(request_start_)};
  }
  /// Whether `state` can belong to a history of `tokens` committed tokens.
  [[nodiscard]] static bool Valid(const State& state, std::size_t tokens) {
    return (state.request_start == kNoState || state.request_start <= tokens) &&
           (state.resume == kNoState || state.resume <= tokens);
  }
  /// Clears the index and adopts `state` (checked with Valid()).
  void Restore(const State& state) {
    Clear();
    resume_ = state.resume == kNoState ? kNoResume : state.resume;
    request_start_ =
        state.request_start == kNoState ? kNoResume : state.request_start;
  }

  /// Records a verified copy of `proposed` tokens from committed index
  /// `start` of which `accepted` were kept; the copied text continues at
  /// start + accepted. Only a mostly accepted copy enables short matches.
  void Observe(std::size_t start, std::size_t proposed, std::size_t accepted) {
    resume_ = 2 * accepted < proposed ? kNoResume : start + accepted;
  }

  /// Indexes committed tokens appended since the last call. `committed`
  /// must extend what was indexed before (Clear() after anything else).
  /// A shorter history, or one whose last indexed token changed, restarts
  /// the index; proposals are verified anyway, so this only keeps a missed
  /// Clear() from proposing stale copies.
  void Extend(std::span<const std::int32_t> committed,
              bool prompt_only = false) {
    if (prompt_only && request_start_ != kNoResume) {
      committed = committed.first(std::min(committed.size(), request_start_));
    }
    if (committed.size() < prev_.size() ||
        (!prev_.empty() && committed[prev_.size() - 1] != last_)) {
      // BeginRequest/Step may already have established the new request's
      // boundary before its shorter restored history is indexed.
      ClearIndex();
    }
    for (std::size_t end = prev_.size(); end < committed.size(); ++end) {
      // prev_[end] chains occurrences of the key ending just before `end`.
      std::uint32_t previous = kNone;
      if (end >= kKeyTokens) {
        previous =
            head_.Exchange(Key(committed.subspan(end - kKeyTokens, kKeyTokens)),
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

  /// Copies must end inside the indexed source, even when the query's
  /// committed history has grown beyond a sampled request's source boundary.
  [[nodiscard]] std::size_t IndexedTokens() const { return prev_.size(); }

  /// Best match for the context `committed` + `tail` among the indexed
  /// committed tokens, or a zero-length match. `tail` holds the uncommitted
  /// chain (anchor and drafts); `committed` must extend the indexed prefix.
  [[nodiscard]] Match Find(std::span<const std::int32_t> committed,
                           std::span<const std::int32_t> tail,
                           std::size_t minimum = kMinMatch) const {
    const std::size_t context = committed.size() + tail.size();
    const std::size_t available =
        request_start_ == kNoResume
            ? context
            : context - std::min(context, request_start_);
    const std::size_t max_match = std::min(kMaxMatch, available);
    if (max_match < kKeyTokens) {
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
      while (length < max_match && end > length &&
             committed[end - length - 1] == at(context - length - 1)) {
        ++length;
      }
      const bool resumes = resume_ != kNoResume &&
                           end + kResumeBack >= resume_ &&
                           end <= resume_ + kResumeWindow;
      if (end < prev_.size() && length > best.length &&
          (length >= minimum || (resumes && length >= kResumeMinMatch))) {
        best = {end, length};
      }
      end = prev_[end];
    }
    return best;
  }

private:
  void ClearIndex() {
    head_.Clear();
    prev_.clear();
    resume_ = kNoResume;
  }

  static constexpr std::uint32_t kNone =
      std::numeric_limits<std::uint32_t>::max();
  static constexpr std::size_t kNoResume =
      std::numeric_limits<std::size_t>::max();

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
      return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift_);
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
  std::int32_t last_{0};           ///< committed[prev_.size() - 1] when indexed
  std::size_t resume_{kNoResume};  ///< where the last copy's source continues
  std::size_t request_start_{kNoResume};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PROMPT_LOOKUP_HPP_
