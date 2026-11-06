#ifndef GUFO_CACHE_PREFETCH_CURSOR_HPP_
#define GUFO_CACHE_PREFETCH_CURSOR_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::cache::detail {

// Scheduling state protected by the prefetcher's mutex. Offsets contains one
// cumulative byte position per file and the total size after the last file.
class PrefetchCursor {
public:
  void Reading(std::size_t index) noexcept {
    reader_ = index;
    // A reader can overtake helpers while they hint an earlier file. Skip
    // consumed files so the unsigned byte distance cannot underflow.
    next_ = std::max(next_, reader_);
  }

  [[nodiscard]] bool Ready(std::span<const std::uint64_t> offsets,
                           std::uint64_t window) const noexcept {
    return !offsets.empty() && next_ < offsets.size() - 1 &&
           offsets[next_] - offsets[reader_] < window;
  }

  [[nodiscard]] std::size_t Take() noexcept { return next_++; }

private:
  std::size_t next_{}, reader_{};
};

}  // namespace gufo::cache::detail

#endif  // GUFO_CACHE_PREFETCH_CURSOR_HPP_
