#ifndef GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_
#define GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_

#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/gemma4/vision/weights.hpp"

namespace gufo::models::gemma4::vision {

/// Scalar reference of the Gemma 4 vision tower and embedder
/// (transformers `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`; llama.cpp
/// `clip_graph_gemma4v` with `clip.use_gelu`). Weights are widened to FP32
/// and every reduction accumulates in double; attention is quadratic, so it
/// only suits small images. Output rows are row-major over the pooled grid.
class Reference {
public:
  using Observer =
      std::function<void(std::string_view stage, std::span<const float>)>;

  explicit Reference(const Weights& weights) : weights_(weights) {}

  /// `image` must already be resized (sides multiples of 48). Stages:
  /// "patch" (patch embedding plus positions), "layer<i>" (residual after
  /// block i), "pooled" (3x3 mean times sqrt(1152)) and "embedding".
  [[nodiscard]] std::vector<float> Encode(const core::Image& image,
                                          const Observer& observer = {}) const;

private:
  const Weights& weights_;
};

}  // namespace gufo::models::gemma4::vision

#endif  // GUFO_MODELS_GEMMA4_VISION_REFERENCE_HPP_
