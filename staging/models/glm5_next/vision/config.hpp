#ifndef GUFO_MODELS_GLM5_NEXT_VISION_CONFIG_HPP_
#define GUFO_MODELS_GLM5_NEXT_VISION_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next::vision {

/// The projector post-LayerNorm epsilon (`NORM_TYPE_NORMAL, 1e-5` in the
/// reference graph; not a GGUF value).
inline constexpr float kLayerNormEps = 1e-5F;

/// Architecture parameters of the GLM-5 vision tower (`clip` GGUF with the
/// `glm5v` projector). A 24-block CLIP-style ViT over 14x14 patches, hidden
/// 1024, 16 heads of width 64, with per-head Q/K RMSNorm, vision RoPE over
/// the first half of each head (x/y split, theta 10000), swiglu-clamp FFNs,
/// a 2x2 stride-2 conv merger and an fc + LayerNorm + gelu_erf + FFN
/// projection to the text width (4096). Values are read from the mmproj
/// file; `FromGguf` locks the geometry this runtime was written and tested
/// for.
struct Config {
  std::uint32_t projection_dim{0};       ///< Output width to the trunk, 4096.
  std::uint32_t image_size{0};           ///< Reference square side, 448.
  std::uint32_t patch_size{0};           ///< 14.
  std::uint32_t embedding_length{0};     ///< Vision hidden, 1024.
  std::uint32_t feed_forward_length{0};  ///< 4096.
  std::uint32_t block_count{0};          ///< 24.
  std::uint32_t head_count{0};           ///< 16.
  std::uint32_t merge_size{2};           ///< Merger window, fixed 2.
  float layer_norm_epsilon{1e-5F};
  float swiglu_clamp{10.0F};     ///< Both FFN stages clamp at +/- this.
  std::uint64_t min_pixels{0};   ///< Smallest model canvas (12544).
  std::uint64_t max_pixels{0};   ///< Largest model canvas (6272000).
  std::vector<float> image_mean;  ///< 3 entries.
  std::vector<float> image_std;   ///< 3 entries.

  /// Default RoPE base when the file stores no explicit theta.
  static constexpr float kRopeTheta = 10000.0F;

  [[nodiscard]] std::uint32_t HeadDim() const noexcept {
    return head_count == 0 ? 0 : embedding_length / head_count;
  }
  /// Resize canvas is a multiple of `patch_size * merge_size` so the merger
  /// conv (kernel 2, stride 2) divides the patch grid exactly.
  [[nodiscard]] std::uint32_t AlignSize() const noexcept {
    return patch_size * merge_size;
  }
  /// Head dims [0, half) rotate with the y position, [half, 2*half) with x.
  [[nodiscard]] std::uint32_t RopeHalfDim() const noexcept {
    return HeadDim() / 4;
  }
  /// Rotated dims per head: pairs (d, d + 2*half).
  [[nodiscard]] std::uint32_t RopeDim() const noexcept {
    return 2 * RopeHalfDim();
  }

  /// Reads and validates the `clip.*` metadata for the `glm5v` projector.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::glm5_next::vision

#endif  // GUFO_MODELS_GLM5_NEXT_VISION_CONFIG_HPP_
