#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/families/qwen/vision/admission.hpp"
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

struct CopyWidthHeadroom {
  std::size_t capacity_bytes;
  std::size_t cache_growth_bytes;
};

// The observation already excludes committed storage. Only initialized cache
// storage included in the target can reduce its remaining growth; reservations
// and estimated allocator headroom are not commitment. Host-only cache storage
// does not compete with device-local capacity. Its own host checks still apply.
[[nodiscard]] inline CopyWidthHeadroom PrepareCopyWidthHeadroom(
    const qwen::vision::StartupCapacity& capacity, std::size_t runtime_margin,
    std::size_t cache_target, std::size_t cache_committed) {
  return {capacity.available_bytes -
              std::min(capacity.available_bytes, runtime_margin),
          capacity.device_budget_only
              ? 0
              : cache_target - std::min(cache_target, cache_committed)};
}

// Prefer wider verification only if sessions, deferred work, one maximum image
// and applicable future cache storage fit together. This does not reserve cache
// or image capacity and is not a text startup condition. Buffers keep their
// selected sizes, so fall back to ordinary verification before sessions exist.
// estimate installs/evaluates the real executor configuration; its last call
// installs the returned width. Neural MTP draft limits are unchanged.
template<class Estimate>
[[nodiscard]] CopyWidthChoice SelectOnDemandCopyWidth(
    std::uint32_t widest, std::size_t sessions, std::size_t capacity,
    std::size_t image_bytes, std::size_t cache_growth_bytes,
    Estimate&& estimate) {
  constexpr auto kMax = std::numeric_limits<std::size_t>::max();
  for (auto rows = widest; rows > kCopySegmentRows; rows -= kCopySegmentRows) {
    const auto memory = estimate(rows);
    if (memory.per_session_bytes && sessions > kMax / memory.per_session_bytes)
      continue;
    const auto text = sessions * memory.per_session_bytes;
    if (memory.deferred_bytes > kMax - text ||
        image_bytes > kMax - text - memory.deferred_bytes)
      continue;
    const auto execution_and_image = text + memory.deferred_bytes + image_bytes;
    if (cache_growth_bytes > kMax - execution_and_image)
      continue;
    if (execution_and_image + cache_growth_bytes <= capacity)
      return {rows, memory};
  }
  return {kCopySegmentRows, estimate(kCopySegmentRows)};
}

}  // namespace gufo::models::qwen38_flash_next

#endif
