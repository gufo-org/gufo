#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_ADMISSION_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "gufo/ext/component_cache_resources.hpp"
#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen38_flash_next {

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
