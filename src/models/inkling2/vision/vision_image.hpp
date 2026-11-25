// CPU-side image handling for the Inkling 2 vision encoder: decode (stb_image),
// grid planning and the (rare) over-limit downscale. Pad + normalise run on the
// GPU inside VisionEncoder. MIT. Grid/limit rules follow llama.cpp
// tools/mtmd/mtmd-image.cpp (e412c6f), see THIRD_PARTY.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gufo::models::inkling2::vision {

inline constexpr int kPatch = 20;  // patch size
inline constexpr int kMerge = 2;   // 2x2 merge
inline constexpr int kAlign =
    kPatch * kMerge;  // 40: image sides are padded to a multiple of this
inline constexpr int kMaxTokens = 4096;  // image tokens per image
inline constexpr int64_t kMaxPixels =
    int64_t(kMaxTokens) * kAlign * kAlign;  // 4096 * 1600 = 6,553,600

struct ImageU8 {
  int w = 0, h = 0;
  std::vector<uint8_t> rgb;  // w*h*3, interleaved RGB
};

struct ImagePlan {
  int src_w = 0,
      src_h = 0;  // after the optional downscale (== input when not too large)
  int pad_w = 0, pad_h = 0;    // multiples of 40
  int grid_w = 0, grid_h = 0;  // patches (pad / 20)
  int tok_w = 0, tok_h = 0;    // image tokens (pad / 40)
  int n_tokens = 0;
  bool downscaled = false;
};

/// Plans the padded grid for an image of w x h. Images whose padded area
/// exceeds 4096 * 1600 pixels are first scaled (aspect preserved, sides
/// multiples of 40).
ImagePlan PlanImage(int w, int h);

/// Scales `img` down (aspect kept, triangle filter) so its longer side is at
/// most `max_side`; never enlarges. Returns whether it changed the image. Used
/// for `detail: low` (512 pixels).
bool ShrinkToFit(ImageU8* img, int max_side);

/// Returns img itself when no downscale is needed; otherwise resizes to the
/// plan size (separable triangle filter; not bit-identical to mtmd's bicubic,
/// which only matters for images larger than ~2560 x 2560).
const ImageU8& ApplyPlan(const ImageU8& img, const ImagePlan& plan,
                         ImageU8* scratch);

}  // namespace gufo::models::inkling2::vision
