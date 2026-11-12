#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen38_flash_next/memory.hpp"

namespace gufo::models::qwen38_flash_next {

struct CopyWidthMemory {
  std::size_t per_session_bytes;
  std::size_t deferred_bytes;
};

struct CopyWidthChoice {
  std::uint32_t rows;
  CopyWidthMemory memory;
};

// Choose optional copy acceleration before session construction. A wide path
// must fit both the ordinary device gate and the currently observed runtime
// headroom after the cache preference and existing margin. This is a priority
// rule, not a cache grant or a second startup guarantee: ordinary verification
// is always the fallback, even if that preference cannot fit. Its actual text
// admission is still checked separately. Neural draft limits do not change.
// estimate must install/evaluate each candidate's real executor configuration;
// the returned candidate is the final one evaluated, including the fallback.
template<class Estimate>
[[nodiscard]] CopyWidthChoice SelectOnDemandCopyWidth(
    std::uint32_t maximum_rows, std::size_t sessions,
    std::size_t device_capacity, std::size_t runtime_capacity,
    std::size_t cache_preference, std::size_t runtime_margin,
    Estimate&& estimate) {
  if (maximum_rows < kCopySegmentRows || maximum_rows % kCopySegmentRows ||
      !sessions)
    throw std::invalid_argument("invalid copy-width admission geometry");
  auto remaining = std::min(device_capacity, runtime_capacity);
  const auto subtract = [&remaining](std::size_t bytes) {
    if (bytes > remaining) {
      remaining = 0;
      return false;
    }
    remaining -= bytes;
    return true;
  };
  const bool preference_fits =
      subtract(cache_preference) && subtract(runtime_margin);
  const auto fits = [sessions](std::size_t capacity, CopyWidthMemory memory) {
    return memory.deferred_bytes <= capacity &&
           memory.per_session_bytes <=
               (capacity - memory.deferred_bytes) / sessions;
  };
  for (auto rows = maximum_rows;; rows -= kCopySegmentRows) {
    const auto memory = estimate(rows);
    if (rows == kCopySegmentRows ||
        (preference_fits && fits(device_capacity, memory) &&
         fits(remaining, memory)))
      return {rows, memory};
  }
}

}  // namespace gufo::models::qwen38_flash_next

#endif
