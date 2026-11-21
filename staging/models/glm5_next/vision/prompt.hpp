#ifndef GUFO_MODELS_GLM5_NEXT_VISION_PROMPT_HPP_
#define GUFO_MODELS_GLM5_NEXT_VISION_PROMPT_HPP_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
#include "staging/models/glm5_next/tokenizer.hpp"
#include "staging/models/glm5_next/vision/config.hpp"

namespace gufo::models::glm5_next::vision {

/// Aspect-preserving target for the `glm5v` tower's black canvas. The canvas
/// is ceil-aligned to `patch_size * n_merge` (28) and fitted into
/// `[min_pixels, max_pixels]`: grown (ceil) when below the budget, and shrunk
/// to the largest content height whose aligned canvas still fits when above
/// it. Mirrors the reference `mtmd_image_preprocessor_glm5v::preprocess`.
struct CanvasSize {
  std::uint32_t width{0};
  std::uint32_t height{0};
};

[[nodiscard]] CanvasSize CanvasFor(std::uint32_t width, std::uint32_t height,
                                   const Config& config);

/// One image after bicubic content resize, top-left placement on the black
/// canvas and mean/std normalization, ready for the tower. `pixels` is CHW,
/// `3 * ny * nx` floats normalized as `(v / 255 - mean) / std`; black padding
/// therefore holds `-mean / std`, not zero. `nx` and `ny` are the canvas
/// sides (multiples of `AlignSize()`) and `tokens == (nx / align) * (ny / align)`.
struct PreparedImage {
  std::vector<float> pixels;
  std::uint32_t nx{0};
  std::uint32_t ny{0};
  std::uint32_t tokens{0};
};

[[nodiscard]] PreparedImage PrepareImage(const core::Image& image,
                                         const Config& config);

/// One image's placeholder span inside a built prompt plus the pixels the
/// encoder must run over. `offset` is the absolute index of the first filler
/// token and `count` is the number of filler tokens (the encoder's
/// `n_tokens`); the executor overwrites exactly those `count` embedding rows.
struct VisionSlot {
  std::uint32_t offset{0};
  std::uint32_t count{0};
  std::vector<float> pixels;
  std::uint32_t nx{0};
  std::uint32_t ny{0};
};

/// A tokenized prompt with every `kImageTag` marker expanded into its
/// placeholder span, plus the per-image slots needed to fill them.
struct PreparedPrompt {
  std::vector<TokenId> tokens;
  std::vector<VisionSlot> images;
};

/// Expands a rendered chat prompt into token ids, replacing each `kImageTag`
/// marker with `n_tokens` copies of the marker (the chat template already
/// emits the surrounding begin/end-of-image tags). `images`
/// are consumed in marker order. Returns an empty prompt and fills
/// `error_msg` if the marker is missing from the vocabulary or the marker
/// count differs from `images`.
[[nodiscard]] PreparedPrompt BuildPrompt(const GlmTokenizer& tokenizer,
                                         std::string_view rendered,
                                         const std::vector<core::Image>& images,
                                         const Config& config,
                                         std::string* error_msg);

}  // namespace gufo::models::glm5_next::vision

#endif  // GUFO_MODELS_GLM5_NEXT_VISION_PROMPT_HPP_
