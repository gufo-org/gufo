// FL2VA keyframe presentation invariants (issue #177). Structural checks on
// BuildKeyframePresentation: ordering, modality tags, image_pad expansion,
// and the interleaved-mRoPE cursor advance. The tokenizer file comes from
// the external checkpoint; the test skips with 77 when it is absent.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/tokenizer.hpp"

namespace {

namespace h3 = gufo::minimax_h3;

int failures = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL presentation: " << message << '\n';
    ++failures;
  }
}

constexpr std::uint32_t kVisionStart = 151652U;
constexpr std::uint32_t kVisionEnd = 151653U;
constexpr std::uint32_t kImagePad = 151655U;

void TestPresentation(const h3::Tokenizer& tokenizer) {
  const std::vector<std::pair<int, int>> grids{{32, 32}, {32, 32}};
  h3::KeyframePresentation presentation;
  std::string error;
  Check(h3::BuildKeyframePresentation(tokenizer, "a red fox", grids,
                                      &presentation, &error),
        "builder succeeds for two 512-canvas keyframes");
  if (failures != 0) {
    return;
  }
  Check(presentation.ids.size() == presentation.tags.size() &&
            presentation.rope_positions.size() == presentation.ids.size() * 3U,
        "tags and rope positions align with rows");
  // Vision rows: 2 markers + (grid_h/2 * grid_w/2) = 256 pads per keyframe.
  const std::size_t pads_per_block = 16U * 16U;
  std::size_t pads = 0;
  std::size_t starts = 0;
  std::size_t ends = 0;
  for (std::size_t row = 0; row < presentation.ids.size(); ++row) {
    if (presentation.ids[row] == kImagePad) {
      ++pads;
      Check(presentation.tags[row] == 0, "image_pad rows are video-tagged");
    } else if (presentation.ids[row] == kVisionStart) {
      ++starts;
      Check(presentation.tags[row] == 0, "vision_start is video-tagged");
    } else if (presentation.ids[row] == kVisionEnd) {
      ++ends;
      Check(presentation.tags[row] == 0, "vision_end is video-tagged");
    }
  }
  Check(starts == 2 && ends == 2 && pads == 2 * pads_per_block,
        "two blocks with 256 pads each");
  Check(presentation.image_rows.size() == pads,
        "one embedding slot per image_pad row");
  for (std::size_t index = 0; index < presentation.image_rows.size(); ++index) {
    Check(presentation.ids[presentation.image_rows[index]] == kImagePad,
          "image row indexes point at pads");
  }
  // Cursor semantics inside each block: constant t, (y, x) ramp over the
  // 16x16 merged grid, and a >= max(h, w) = 16 advance before the next block.
  const auto rope = [&](std::size_t row, int axis) {
    return presentation
        .rope_positions[row * 3U + static_cast<std::size_t>(axis)];
  };
  const std::size_t first_block = presentation.image_rows.front();
  Check(rope(first_block, 1) == rope(first_block, 2),
        "block origin has h == w");
  Check(rope(first_block + 1U, 2) == rope(first_block, 2) + 1 &&
            rope(first_block + 1U, 1) == rope(first_block, 1),
        "x advances before y inside the row");
  Check(rope(first_block + 16U, 1) == rope(first_block, 1) + 1,
        "y advances per merged row");
  for (std::size_t row = first_block; row < first_block + pads_per_block;
       ++row) {
    Check(rope(row, 0) == rope(first_block, 0),
          "temporal axis is constant across the block");
  }
  // Text rows keep t == h == w.
  for (std::size_t row = 0; row < presentation.ids.size(); ++row) {
    const bool vision_row = presentation.tags[row] == 0;
    if (!vision_row) {
      Check(rope(row, 0) == rope(row, 1) && rope(row, 1) == rope(row, 2),
            "text rows use equal axes");
    }
  }
  const std::size_t second_block = presentation.image_rows[pads_per_block];
  // Block 2 starts after the 16-row cursor advance, the closing marker, the
  // second label, and the opening marker: at least 17 rows later.
  Check(rope(second_block, 0) - rope(first_block, 0) >= 17,
        "second block temporal cursor advanced");

  // Single-keyframe path and odd grids.
  h3::KeyframePresentation single;
  Check(h3::BuildKeyframePresentation(tokenizer, "one", {{16, 18}}, &single,
                                      &error),
        "single keyframe accepted");
  Check(single.image_rows.size() == 8U * 9U, "grid 16x18 -> 72 pads");
  std::string rejection;
  Check(!h3::BuildKeyframePresentation(tokenizer, "x", {{15, 16}}, &single,
                                       &rejection),
        "odd grid rejected");
  Check(!h3::BuildKeyframePresentation(tokenizer, "x", {}, &single, &rejection),
        "empty keyframe list rejected");
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root == nullptr) {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset\n";
    return 77;
  }
  const std::filesystem::path path =
      std::filesystem::path(root) / "FL2VA/tokenizer/tokenizer.json";
  h3::Tokenizer tokenizer;
  std::string error;
  if (!h3::Tokenizer::Load(path, &tokenizer, &error)) {
    std::cerr << "FAIL tokenizer load: " << error << '\n';
    return 1;
  }
  TestPresentation(tokenizer);
  if (failures != 0) {
    return 1;
  }
  std::cout << "MiniMax H3 keyframe presentation invariants passed\n";
  return 0;
}
