#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_COPY_WIDTH_ADMISSION_HPP_

#include <cstddef>
#include <cstdint>

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

// On-demand vision keeps copy verification within one ordinary eight-row
// segment. A startup headroom reading cannot promise spare runtime capacity:
// extra segments remain execution obligations even before their first use.
// Session hidden rows and captured work also retain stable allocation sizes,
// so this cap is installed before creating any session and never widened.
// estimate must install/evaluate the real executor configuration. Neural MTP
// draft limits and the separate text/device admission gate remain unchanged.
template<class Estimate>
[[nodiscard]] CopyWidthChoice SelectOnDemandCopyWidth(Estimate&& estimate) {
  return {kCopySegmentRows, estimate(kCopySegmentRows)};
}

}  // namespace gufo::models::qwen38_flash_next

#endif
