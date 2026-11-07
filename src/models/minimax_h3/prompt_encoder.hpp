#ifndef GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
#define GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"
#include "src/models/minimax_h3/vision_tower.hpp"

namespace gufo::minimax_h3 {

inline constexpr std::size_t kPromptEncoderLayers = 50;
inline constexpr std::size_t kPromptEncoderHiddenSize = 5120;
inline constexpr std::size_t kPromptEncoderIntermediateSize = 25600;
inline constexpr std::size_t kPromptEncoderQueryHeads = 64;
inline constexpr std::size_t kPromptEncoderKeyValueHeads = 8;
inline constexpr std::size_t kPromptEncoderHeadDimension = 128;
inline constexpr std::size_t kPromptEncoderVocabularySize = 151936;

struct PromptEncoderOptions {
  std::size_t layer_count{kPromptEncoderLayers};
  bool prefetch_layers{true};
  bool vision_inputs{false};
  bool deepstack_inputs{false};
};

struct PromptEncoderLayerTelemetry {
  std::size_t layer{0};
  std::uint64_t weight_bytes{0};
  double load_ms{0.0};
  double prefetch_wait_ms{0.0};
  double submit_ms{0.0};
  double gpu_ms{0.0};
  std::uint64_t dispatches{0};
};

struct PromptEncoderTelemetry {
  std::vector<PromptEncoderLayerTelemetry> layers;
  std::uint64_t embedding_weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t peak_device_bytes{0};
  std::uint64_t peak_registered_host_bytes{0};
  std::uint64_t live_registered_host_bytes_after{0};
  std::uint64_t total_dispatches{0};
};

struct PromptEmbedding {
  std::size_t tokens{0};
  std::size_t width{0};
  std::vector<std::uint16_t> values;
  // Per-row modality tags for the DiT AdaLN table (1 = text, 0 = video).
  // Empty means "all text" (the text-only path is unchanged); FL2VA
  // presentation fills it: label and prompt rows are 1, vision blocks
  // including both marker rows are 0.
  std::vector<std::uint8_t> tags;
};

// FL2VA keyframe presentation (issue #177), host-side and testable: the
// token sequence, per-row modality tags, image_pad row indexes, and the
// interleaved-mRoPE (t,h,w) positions for every row. Contract per diffusers
// encoders.py:285-289 with the transformers' Qwen3VL position advance
// (image blocks advance the cursor by max(h, w) merged rows).
struct KeyframePresentation {
  std::vector<std::uint32_t> ids;
  std::vector<std::uint8_t> tags;         // 1 = text (label/prompt), 0 = vision
  std::vector<std::uint32_t> image_rows;  // rows receiving tower embeddings
  std::vector<std::int32_t> rope_positions;  // [rows][3] (t, h, w)
};

// `grids` are the per-keyframe vision patch grids in packed order (the same
// order as KeyframeVision grid_h/grid_w): rows = grid_h * grid_w, merged
// half-grids define the marker expansion count (grid_h * grid_w / 4).
[[nodiscard]] bool BuildKeyframePresentation(
    const Tokenizer& tokenizer, std::string_view prompt,
    const std::vector<std::pair<int, int>>& grids, KeyframePresentation* output,
    std::string* error = nullptr);

[[nodiscard]] bool EncodePromptLayer50(const ModelInventory& inventory,
                                       std::span<const std::uint32_t> token_ids,
                                       const PromptEncoderOptions& options,
                                       const CancellationToken* cancellation,
                                       PromptEmbedding* output,
                                       PromptEncoderTelemetry* telemetry,
                                       std::string* error = nullptr);

// FL2VA keyframe presentation (issue #177): per keyframe "<Picture {i+1}>: "
// plus vision_start / image_pad x (H*W/1024) / vision_end, then the raw
// prompt (no chat template, add_special_tokens=false). Vision rows receive
// the tower embeddings; deepstack taps add into language layers 0-2; rope
// uses interleaved mRoPE (identical to the text-only table wherever the
// three axes agree).
[[nodiscard]] bool EncodePromptMultimodal(
    const ModelInventory& inventory, const Tokenizer& tokenizer,
    std::string_view prompt, const std::vector<KeyframeVision>& keyframes,
    const CancellationToken* cancellation, PromptEmbedding* output,
    PromptEncoderTelemetry* telemetry, std::string* error = nullptr);

[[nodiscard]] std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept;

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_HPP_
