#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "gufo/ext/component_cache_resources.hpp"
#include "gufo/ext/text.hpp"
#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

// Same requested allowance used by serving: explicit retention, the bounded
// automatic host budget, or disk-only copy-through storage. This is an upper
// preference, not a grant or a promise of payload pages: retained metadata is
// inside it; bounded staging/driver work remains in the runtime margin.
[[nodiscard]] inline std::size_t RequestedVisionCacheBytes(
    const ext::TextRunnerRamCacheOptions& ram, bool disk,
    std::size_t automatic_host_budget) {
  if (!ram.enabled)
    return disk ? std::size_t{512} << 20 : 0;
  return ram.capacity_bytes
             ? ram.capacity_bytes
             : std::min(ext::TextRunnerRamCacheOptions::kAutomaticMaxBytes,
                        automatic_host_budget);
}

// Preserve ordinary cache grants and lazy growth. Only an arriving segment
// competes with images and optional snapshots; its charge follows the pages.
inline hip::ComponentCacheHooks VisionCacheHooks(
    hip::ComponentCacheHooks upstream,
    const std::shared_ptr<qwen::vision::AdmissionBudget>& budget) {
  if (budget)
    upstream.reserve_segment = [budget](std::size_t bytes) {
      return budget->ReserveShared(bytes);
    };
  return upstream;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
