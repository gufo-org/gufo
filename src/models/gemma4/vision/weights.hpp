#ifndef GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_

#include <array>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::vision {

inline constexpr std::uint32_t kHidden = 1152;
inline constexpr std::uint32_t kHeads = 16;
inline constexpr std::uint32_t kHeadDim = 72;
inline constexpr std::uint32_t kFfn = 4304;
inline constexpr std::uint32_t kLayers = 27;
/// Rows of each learned position table (x and y).
inline constexpr std::uint32_t kPositions = 10240;
inline constexpr float kEps = 1e-6F;
inline constexpr float kRopeTheta = 100.0F;
/// Pixel values per patch, ordered channel, row, column.
inline constexpr std::uint32_t kPatchValues = 3 * 16 * 16;

struct LayerTensors {
  const core::GgufTensorInfo* ln1{nullptr};  ///< pre-attention RMS weight
  const core::GgufTensorInfo* q{nullptr};
  const core::GgufTensorInfo* k{nullptr};
  const core::GgufTensorInfo* v{nullptr};
  const core::GgufTensorInfo* q_norm{nullptr};
  const core::GgufTensorInfo* k_norm{nullptr};
  const core::GgufTensorInfo* out{nullptr};
  const core::GgufTensorInfo* attn_post_norm{nullptr};
  const core::GgufTensorInfo* ln2{nullptr};  ///< pre-FFN RMS weight
  const core::GgufTensorInfo* gate{nullptr};
  const core::GgufTensorInfo* up{nullptr};
  const core::GgufTensorInfo* down{nullptr};
  const core::GgufTensorInfo* ffn_post_norm{nullptr};
};

/// Validated view of a Gemma 4 `gemma4v` BF16 sidecar. Matrices are BF16
/// [out][in]; norms, the patch embedding ([1152][3][16][16]), the position
/// tables ([2][10240][1152]) and the standardization vectors are F32.
struct Weights {
  std::uint32_t output_width{0};
  const core::GgufTensorInfo* patch{nullptr};
  const core::GgufTensorInfo* position{nullptr};
  const core::GgufTensorInfo* std_bias{nullptr};
  const core::GgufTensorInfo* std_scale{nullptr};
  const core::GgufTensorInfo* projection{nullptr};  ///< [output_width][1152]
  std::array<LayerTensors, kLayers> layers{};

  /// Throws std::invalid_argument when the sidecar is not a Gemma 4 vision
  /// projector for a text model of `output_width`.
  [[nodiscard]] static Weights Resolve(const core::GgufReader& reader,
                                       std::uint32_t output_width);
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_WEIGHTS_HPP_
