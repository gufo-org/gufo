#pragma once

#include <algorithm>

#include "src/families/qwen/vision/admission.hpp"

namespace gufo::models::qwen::vision::testing {
// Upper bound for small synthetic fixtures. Production uses retained buffer
// sizes and already-prepared owners; the tests inspect actual live allocations
// independently rather than deriving coverage from this aggregate estimate.
inline std::size_t RequestMemoryBound(std::size_t prefix_rows,
                                      std::span<const ImageDimensions> images,
                                      std::uint32_t output_width) {
  auto bytes = CheckedMultiply(prefix_rows, 6 * sizeof(std::int32_t));
  std::size_t scratch = 0;
  for (const auto image : images) {
    bytes = CheckedAdd(bytes, ImageEmbeddingBytes(image, output_width));
    bytes = CheckedAdd(
        bytes, CheckedMultiply(CheckedMultiply(image.width, image.height), 3));
    scratch = std::max(scratch, ImageScratchBytes(image));
  }
  return CheckedAdd(bytes, scratch);
}

}  // namespace gufo::models::qwen::vision::testing
