#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_IMAGE_PROCESSOR_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_IMAGE_PROCESSOR_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/image.hpp>
#include <string>
#include <vector>

namespace gufo::models::embedding_gemma2 {

inline constexpr std::uint32_t kMaxImageSide = 32768;
/// Gemma4ImageProcessor's supported `max_soft_tokens` values; the checkpoint
/// defaults to 280.
inline constexpr std::array<std::uint32_t, 5> kImageSoftTokenBudgets{
    70, 140, 280, 560, 1120};
inline constexpr std::uint32_t kDefaultImageSoftTokens = 280;
inline constexpr std::uint32_t kMaxImageSoftTokens = 1120;

/// Unpadded, row-major 16x16 RGB patches. The reference pads to nine patches
/// per soft token of the budget and masks the padding; omitting those keys
/// gives the same result.
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

/// True for a `max_soft_tokens` value in `kImageSoftTokenBudgets`.
[[nodiscard]] bool SupportedImageSoftTokens(std::uint32_t max_soft_tokens);

/// Gemma4ImageProcessor: aspect-preserving resize to at most
/// `max_soft_tokens` soft tokens (a value of `kImageSoftTokenBudgets`), uint8
/// antialiased bicubic resize, rescale, then patchify. Does not crop.
[[nodiscard]] ImagePatches ProcessImage(
    const core::Image& image,
    std::uint32_t max_soft_tokens = kDefaultImageSoftTokens);

}  // namespace gufo::models::embedding_gemma2
#endif
