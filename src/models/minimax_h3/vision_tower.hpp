#ifndef GUFO_MODELS_MINIMAX_H3_VISION_TOWER_HPP_
#define GUFO_MODELS_MINIMAX_H3_VISION_TOWER_HPP_

// Qwen3-VL vision tower for MiniMax H3 FL2VA keyframe conditioning
// (upstream issue #177). The tower is the `model.visual.*` stack inside the
// pinned FL2VA/text_encoder shards (vision_config: depth 27, hidden 1152,
// heads 16, head dimension 72, intermediate 4304, spatial merge 2, temporal
// patch 2, num_position_embeddings 2304, deepstack indexes 8/16/24,
// out_hidden_size 5120). Semantics follow transformers' Qwen3VLVisionModel as
// mirrored by the sanctioned reference port in
// src/models/qwen_image_21/hip/pipeline.cpp (EncodeVision, lines 57-136);
// this implementation is model-private to minimax_h3 and never links
// qwen_image_21. local://h3-fl2va-reference.md Part 4 and
// local://gufo-h3-internals.md §3 are the authoritative feature contract.

#include <cstdint>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

// One keyframe prepared for the vision tower. `pixel_values` holds
// grid_h * grid_w rows of 3 * temporal_patch(2) * 16 * 16 = 1536 float32
// elements in the Qwen2VLImageProcessorFast patch layout: element index
// ((c * 2 + t) * 16 + y) * 16 + x with the single frame duplicated across
// both temporal taps, normalized as (uint8 / 255 - 0.5) / 0.5. The row
// order is merge-block-major: consecutive groups of four rows form the
// 2x2 spatial-merge block consumed by the patch mergers.
struct KeyframeVision {
  std::vector<float> pixel_values;
  int grid_h{0};  // Patches per column axis (image height / 16).
  int grid_w{0};  // Patches per row axis (image width / 16).
};

// Encodes one keyframe. On success:
//   *output     holds (grid_h / 2 * grid_w / 2) rows of 5120 bfloat16 bits
//               (the final `model.visual.merger` output), and
//   *deepstack  holds the three `model.visual.deepstack_merger_list` taps
//               (extracted after blocks 8, 16, and 24) in tap-major order:
//               3 * (grid_h / 2 * grid_w / 2) * 5120 bfloat16 bits.
// `deepstack` may be null when the caller only needs merged embeddings.
[[nodiscard]] bool EncodeVision(const ModelInventory& inventory,
                                const KeyframeVision& vision,
                                const CancellationToken* cancellation,
                                std::vector<std::uint16_t>* output,
                                std::vector<std::uint16_t>* deepstack,
                                std::string* error = nullptr);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_VISION_TOWER_HPP_
