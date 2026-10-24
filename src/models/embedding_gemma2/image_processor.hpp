#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_IMAGE_PROCESSOR_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_IMAGE_PROCESSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::models::embedding_gemma2 {

inline constexpr std::uint32_t kMaxImageSide = 32768;

/// Unpadded, row-major 16x16 RGB patches. The reference pads to 2,520
/// patches and masks the padding; omitting those keys gives the same result.
struct ImagePatches {
  std::uint32_t rows{0};
  std::uint32_t columns{0};
  /// [patch][pixel y][pixel x][channel], rescaled to [0,1].
  std::vector<float> pixels;
  [[nodiscard]] std::size_t soft_tokens() const {
    return std::size_t{rows} * columns / 9;
  }
};

/// Refuses settings for which the fixed processor below would be incorrect.
[[nodiscard]] bool ValidateImageProcessor(const std::string& model_dir,
                                          std::string* error = nullptr);

/// Gemma4ImageProcessor: aspect-preserving 280-soft-token budget, uint8
/// antialiased bicubic resize, rescale, then patchify. Does not crop.
[[nodiscard]] ImagePatches ProcessImage(const core::Image& image);

}  // namespace gufo::models::embedding_gemma2
#endif
