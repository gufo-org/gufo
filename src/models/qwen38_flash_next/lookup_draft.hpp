#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Retrieval draft source over the token history. When the text so far ends in
/// a phrase that occurred earlier, the tokens that followed that occurrence are
/// a free guess for what comes next (copied code, quoted tool output, repeated
/// identifiers). The target model still verifies every token, so a wrong guess
/// only costs the verify width.
///
/// Positions are indexed by a hash of the last kGram tokens and chained newest
/// first, so a query inspects the most recent occurrences and keeps the one
/// with the longest verified match. The index is host-side and a pure function
/// of the token history, which keeps a request's drafts reproducible.
class LookupIndex {
public:
  static constexpr std::uint32_t kGram = 4;
  static constexpr std::uint32_t kCandidates = 8;
  static constexpr std::uint32_t kMaxMatch = 128;

  void Append(std::int32_t token) {
    text_.push_back(token);
    prev_.push_back(-1);
    const auto count = static_cast<std::int32_t>(text_.size());
    if (count < static_cast<std::int32_t>(kGram)) {
      return;
    }
    if (2 * static_cast<std::size_t>(count) > head_.size()) {
      Rebuild(std::max<std::size_t>(1024, head_.size() * 2));
      return;
    }
    const auto slot = Slot(count - 1);
    prev_[count - 1] = head_[slot];
    head_[slot] = count - 1;
  }

  [[nodiscard]] std::size_t size() const noexcept { return text_.size(); }

  /// Tokens that followed the best earlier occurrence of the current tail, at
  /// most `limit` of them; empty when no earlier match reaches `minimum_match`.
  /// `matched` receives the verified match length.
  [[nodiscard]] std::vector<std::int32_t> Draft(
      std::uint32_t minimum_match, std::size_t limit,
      std::uint32_t* matched = nullptr) const {
    std::vector<std::int32_t> draft;
    const auto count = static_cast<std::int32_t>(text_.size());
    if (matched != nullptr) {
      *matched = 0;
    }
    if (limit == 0 || count < static_cast<std::int32_t>(kGram) ||
        head_.empty()) {
      return draft;
    }
    minimum_match = std::max(minimum_match, kGram);
    std::int32_t best_end = -1;
    std::uint32_t best_length = 0;
    std::uint32_t tried = 0;
    for (std::int32_t end = head_[Slot(count - 1)];
         end >= 0 && tried < kCandidates; end = prev_[end]) {
      if (end == count - 1) {
        continue;  // the tail itself
      }
      ++tried;
      std::uint32_t length = 0;
      while (length < kMaxMatch &&
             end - static_cast<std::int32_t>(length) >= 0 &&
             text_[end - length] == text_[count - 1 - length]) {
        ++length;
      }
      if (length > best_length) {
        best_length = length;
        best_end = end;
      }
    }
    if (best_end < 0 || best_length < minimum_match) {
      return draft;
    }
    if (matched != nullptr) {
      *matched = best_length;
    }
    for (std::size_t i = 0; i < limit; ++i) {
      const auto at = static_cast<std::size_t>(best_end) + 1 + i;
      if (at >= text_.size()) {
        break;
      }
      draft.push_back(text_[at]);
    }
    return draft;
  }

private:
  [[nodiscard]] std::size_t Slot(std::int32_t end) const noexcept {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::uint32_t i = 0; i < kGram; ++i) {
      h = (h ^ static_cast<std::uint32_t>(text_[end - kGram + 1 + i])) *
          0x100000001b3ULL;
    }
    h ^= h >> 29;
    return static_cast<std::size_t>(h) & (head_.size() - 1);
  }

  void Rebuild(std::size_t capacity) {
    head_.assign(capacity, -1);
    for (std::int32_t end = static_cast<std::int32_t>(kGram) - 1;
         end < static_cast<std::int32_t>(text_.size()); ++end) {
      const auto slot = Slot(end);
      prev_[end] = head_[slot];
      head_[slot] = end;
    }
  }

  std::vector<std::int32_t> text_;
  std::vector<std::int32_t> prev_;
  std::vector<std::int32_t> head_;
};

}  // namespace gufo::models::qwen38_flash_next
