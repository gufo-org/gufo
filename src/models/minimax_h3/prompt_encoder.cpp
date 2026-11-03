#include "src/models/minimax_h3/prompt_encoder.hpp"

#include <string>

#include "src/families/qwen/control_tokens/control_tokens.hpp"

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

bool BuildKeyframePresentation(const Tokenizer& tokenizer,
                               std::string_view prompt,
                               const std::vector<std::pair<int, int>>& grids,
                               KeyframePresentation* output,
                               std::string* error) {
  if (output == nullptr || grids.empty()) {
    SetError(error, "MiniMax H3 keyframe presentation requires keyframes");
    return false;
  }
  std::uint32_t vision_start = 0;
  std::uint32_t vision_end = 0;
  std::uint32_t image_pad = 0;
  if (!tokenizer.AddedTokenId(tokenization::kVisionStart, &vision_start) ||
      !tokenizer.AddedTokenId(tokenization::kVisionEnd, &vision_end) ||
      !tokenizer.AddedTokenId(tokenization::kImagePad, &image_pad)) {
    SetError(error, "MiniMax H3 tokenizer lacks the Qwen vision markers");
    return false;
  }
  *output = {};
  std::int32_t position = 0;
  const auto push_text = [&](std::string_view text, std::uint8_t tag) -> bool {
    std::vector<std::uint32_t> encoded;
    if (!tokenizer.Encode(text, false, &encoded, error) || encoded.empty()) {
      if (error != nullptr && error->empty()) {
        SetError(error, "MiniMax H3 keyframe presentation text is empty");
      }
      return false;
    }
    for (const std::uint32_t id : encoded) {
      output->ids.push_back(id);
      output->tags.push_back(tag);
      output->rope_positions.push_back(position);
      output->rope_positions.push_back(position);
      output->rope_positions.push_back(position);
      ++position;
    }
    return true;
  };
  const auto push_marker = [&](std::uint32_t id) {
    output->ids.push_back(id);
    output->tags.push_back(0);  // vision blocks (markers included) are rows 0
    output->rope_positions.push_back(position);
    output->rope_positions.push_back(position);
    output->rope_positions.push_back(position);
    ++position;
  };
  for (std::size_t index = 0; index < grids.size(); ++index) {
    const int grid_h = grids[index].first;
    const int grid_w = grids[index].second;
    if (grid_h < 2 || grid_w < 2 || grid_h % 2 != 0 || grid_w % 2 != 0) {
      SetError(error, "MiniMax H3 keyframe vision grid must be even");
      return false;
    }
    const auto half_h = static_cast<std::int32_t>(grid_h / 2);
    const auto half_w = static_cast<std::int32_t>(grid_w / 2);
    if (!push_text("<Picture " + std::to_string(index + 1) + ">: ", 1)) {
      return false;
    }
    push_marker(vision_start);
    const std::int32_t block_start = position;
    for (std::int32_t y = 0; y < half_h; ++y) {
      for (std::int32_t x = 0; x < half_w; ++x) {
        output->image_rows.push_back(
            static_cast<std::uint32_t>(output->ids.size()));
        output->ids.push_back(image_pad);
        output->tags.push_back(0);
        output->rope_positions.push_back(block_start);
        output->rope_positions.push_back(block_start + y);
        output->rope_positions.push_back(block_start + x);
      }
    }
    position += half_h > half_w ? half_h : half_w;
    push_marker(vision_end);
  }
  if (!push_text(prompt.empty() ? " " : prompt, 1)) {
    return false;
  }
  return true;
}

#if !defined(ENGINE_ENABLE_HIP)
bool EncodePromptLayer50(const ModelInventory&, std::span<const std::uint32_t>,
                         const PromptEncoderOptions&, const CancellationToken*,
                         PromptEmbedding*, PromptEncoderTelemetry*,
                         std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

bool EncodePromptMultimodal(const ModelInventory&, const Tokenizer&,
                            std::string_view,
                            const std::vector<KeyframeVision>&,
                            const CancellationToken*, PromptEmbedding*,
                            PromptEncoderTelemetry*, std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 prompt encoder requires ENGINE_ENABLE_HIP";
  }
  return false;
}

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return 0;
}
#endif

}  // namespace gufo::minimax_h3
