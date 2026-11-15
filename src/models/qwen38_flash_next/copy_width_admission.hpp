#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>

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

// Wider copy verification stays as long as the configured sessions, their
// deferred work and one maximum-resolution image still fit the startup device
// capacity. Extra segments are execution obligations even before first use
// and session buffers keep their sizes, so a configuration without that room
// falls back to one ordinary eight-row segment before any session exists.
// estimate must install/evaluate the real executor configuration; the last
// call installs the returned width. Neural MTP draft limits are unchanged.
template<class Estimate>
[[nodiscard]] CopyWidthChoice SelectOnDemandCopyWidth(std::uint32_t widest,
                                                      std::size_t sessions,
                                                      std::size_t capacity,
                                                      std::size_t image_bytes,
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
    if (text + memory.deferred_bytes + image_bytes <= capacity)
      return {rows, memory};
  }
  return {kCopySegmentRows, estimate(kCopySegmentRows)};
}

}  // namespace gufo::models::qwen38_flash_next

#endif
