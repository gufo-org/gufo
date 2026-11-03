#ifndef GUFO_MODELS_MINIMAX_H3_CONDITIONING_HPP_
#define GUFO_MODELS_MINIMAX_H3_CONDITIONING_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/image.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace gufo::minimax_h3 {

// FL2VA keyframe conditioning (upstream issue #177). Keyframes are resized
// onto the target canvas by the released index rule (the geometry anchor is
// stretched; a follower is aspect-cover cropped), VisualVAE-encoded with the
// reference posterior-sampling lineage, rounded through fp16, channel
// normalized, lightly noised at 0.999, and patchified into extra leading
// video rows for the packed sequence. References:
// local://h3-fl2va-reference.md and diffusers modular_pipelines/minimax_h3.

// Fixed by the reference implementation (diffusers keyframe_encode_seed,
// encoders.py:131): a fresh posterior generator per keyframe, independent of
// the request seed, so the same keyframe always encodes to the same anchor.
inline constexpr std::uint64_t kH3KeyframePosteriorSeed = 42;

struct PreparedKeyframe {
  // Canvas-sized RGB8 pixels in display orientation.
  std::vector<std::uint8_t> rgb;
  int width{0};
  int height{0};
  // Normalized anchor latent, 24 * (height/16) * (width/16), F32 channel-
  // major. Filled by EncodeKeyframeLatent.
  std::vector<float> latent;
};

// Reads and decodes a keyframe file to RGB8 in display orientation, within
// the shared encoded-image limit. The geometry anchor's size then selects the
// canvas (ResolveKeyframeParameters) before any keyframe is resampled.
[[nodiscard]] bool DecodeKeyframe(const std::filesystem::path& path,
                                  core::Image* out, std::string* error);

// Resample only; no device or RNG work. `follower` selects the aspect-cover
// crop path; the geometry anchor (first in packed order) uses the stretch
// path. Exact canvas matches pass through without resampling.
[[nodiscard]] bool PrepareKeyframe(const core::Image& decoded, int canvas_width,
                                   int canvas_height, bool follower,
                                   PreparedKeyframe* out, std::string* error);

// VisualVAE encode, posterior sample with the dedicated fixed-seed stream
// (fresh per keyframe), fp16 rounding, and per-channel normalization.
[[nodiscard]] bool EncodeKeyframeLatent(const ModelInventory& inventory,
                                        PreparedKeyframe& keyframe,
                                        const CancellationToken* cancellation,
                                        VideoVaeTelemetry* telemetry,
                                        std::string* error);

// mixed = t * latent + (1 - t) * augmentation evaluated in float32 with the
// reference's separate multiply/add rounding (no fused contraction) and
// t = kH3KeyframeConditionTimestep. `augmentation` must cover the latent.
[[nodiscard]] bool MixKeyframeNoise(const PreparedKeyframe& keyframe,
                                    std::span<const float> augmentation,
                                    std::vector<float>* mixed,
                                    std::string* error);

using KeyframeLatentNormalization =
    std::array<float, kH3VideoVaeLatentChannels>;

// latents_mean / latents_std from FL2VA/video_vae/config.json, the same
// pinned constants the decoder inverts before pixel reconstruction.
[[nodiscard]] bool ReadKeyframeNormalization(
    const ModelInventory& inventory, KeyframeLatentNormalization* mean,
    KeyframeLatentNormalization* deviation, std::string* error);

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_CONDITIONING_HPP_
