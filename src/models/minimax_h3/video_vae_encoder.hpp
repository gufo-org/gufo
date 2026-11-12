#ifndef GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_
#define GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae.hpp"

namespace gufo::minimax_h3 {

// Geometry of the released VisualVAE encoder, read literally from
// FL2VA/video_vae/source/config.json (ch 128, ch_mult [1,2,2,4,4,8],
// space_down [2,2,2,2,1,1], time_down [1,2,2,1,1,1], double_z encoder).
inline constexpr int kH3VaeEncoderInputChannels = 3;
inline constexpr int kH3VaeEncoderLevels = 6;
inline constexpr int kH3VaeEncoderResidualBlocks = 2;
inline constexpr std::array<int, kH3VaeEncoderLevels>
    kH3VaeEncoderLevelChannels{128, 256, 256, 512, 512, 1024};
inline constexpr std::array<int, kH3VaeEncoderLevels>
    kH3VaeEncoderSpatialDownsample{2, 2, 2, 2, 1, 1};
// conv_out emits 2*z_channels (double_z): channels 0..23 are the posterior
// mean, 24..47 the log variance. quant_conv is 48 -> 48 with a 1x1 kernel.
inline constexpr int kH3VaeEncoderMomentChannels = 48;
inline constexpr int kH3VaeEncoderSpatialRatio = 16;
// norm.py:get_group_norm_3d pins GroupNorm(32 groups, eps 1e-6, affine) and
// use_t_isolated_gn=true keeps statistics per plane.
inline constexpr int kH3VaeEncoderGroupNormGroups = 32;
inline constexpr float kH3VaeEncoderGroupNormEpsilon = 1.0e-6F;
// klvae.py:setup_forward / config.json vae_tile_size + vae_tile_overlap_min.
// The released FL2VA path runs the encoder tiled; tile geometry is therefore
// part of the numerical contract, not an allocator heuristic.
inline constexpr int kH3VaeEncoderTilePixels = 256;
inline constexpr int kH3VaeEncoderTileOverlapMinimum = 64;
// modular_pipeline.py canvas_multiple: 16 (spatial ratio) * 2 (latent patch).
inline constexpr int kH3VaeEncoderCanvasMultiple = 32;

struct VideoVaeEncoderTileAxis {
  int extent{0};            // canvas extent along this axis, in pixels
  int length{0};            // tile length in pixels
  std::vector<int> starts;  // pixel offsets, ascending, first is 0
  std::vector<int>
      overlaps;  // pixels with the previous tile; size + 1 == tiles
};

struct VideoVaeEncoderTilePlan {
  int height{0};
  int width{0};
  VideoVaeEncoderTileAxis vertical;    // rows
  VideoVaeEncoderTileAxis horizontal;  // columns
  int tile_pixels{0};
  int tile_latent_height{0};
  int tile_latent_width{0};
  int latent_height{0};
  int latent_width{0};
  int tiles{0};
};

// Reproduces klvae.py:AutoencoderKL.split_tiles (and the identical diffusers
// _split_tiles): the smallest tile count that still covers the extent with the
// minimum overlap, then the slack widened round-robin in whole latent steps so
// every tile boundary stays latent-aligned.
[[nodiscard]] std::optional<VideoVaeEncoderTileAxis>
ResolveVideoVaeEncoderTileAxis(int extent, std::string* error = nullptr);

[[nodiscard]] std::optional<VideoVaeEncoderTilePlan>
ResolveVideoVaeEncoderTilePlan(int height, int width,
                               std::string* error = nullptr);

// Linear cross-fade of per-tile 48-channel moments, in float32, matching
// klvae.py:AutoencoderKL.blend / diffusers _stitch_tiles: rows blend before
// columns (both reading the *untrimmed* neighbours), then each non-last tile is
// trimmed by its own overlap. `tile_moments` is row-major over the plan and
// every entry holds (48, tile_latent_height, tile_latent_width) values.
[[nodiscard]] bool StitchVideoVaeEncoderTiles(
    const VideoVaeEncoderTilePlan& plan,
    std::span<const std::vector<float>> tile_moments,
    std::vector<float>* moments, std::string* error = nullptr);

// One encoder activation, planar CHW, captured for the first tile only.
struct VideoVaeEncoderStage {
  std::string name;  // conv_in, down.0..down.5, norm_out, pre_quant, moments
  int channels{0};
  int height{0};
  int width{0};
  std::vector<float> values;
};

struct VideoVaeEncoderTrace {
  int tile_height{0};
  int tile_width{0};
  std::vector<VideoVaeEncoderStage> stages;

  [[nodiscard]] const VideoVaeEncoderStage* Find(
      std::string_view name) const noexcept;
};

// Single-frame VisualVAE encoder (issue #177 keyframe conditioning). The
// posterior split, sampling, fp16 rounding and latent normalization live in
// conditioning.cpp; this class stops at the 48-channel quant_conv moments.
class VideoVaeEncoder {
public:
  ~VideoVaeEncoder();

  VideoVaeEncoder(const VideoVaeEncoder&) = delete;
  VideoVaeEncoder& operator=(const VideoVaeEncoder&) = delete;
  VideoVaeEncoder(VideoVaeEncoder&&) noexcept;
  VideoVaeEncoder& operator=(VideoVaeEncoder&&) noexcept;

  [[nodiscard]] static std::unique_ptr<VideoVaeEncoder> Create(
      const ModelInventory& inventory, const CancellationToken* cancellation,
      VideoVaeTelemetry* telemetry, std::string* error = nullptr);

  // `rgb_imagenet_f32` is planar (3, height, width) float32 already mapped
  // uint8/255 -> ImageNet. `moments` receives (48, height/16, width/16).
  [[nodiscard]] bool EncodeKeyframeMoments(
      const float* rgb_imagenet_f32, int height, int width,
      const CancellationToken* cancellation, std::vector<float>* moments,
      std::string* error = nullptr);

  // Same encode with the per-tile progress callback (mirroring
  // VideoVaeDecoder) and an optional stage trace of the first tile.
  [[nodiscard]] bool EncodeKeyframeMoments(
      const float* rgb_imagenet_f32, int height, int width,
      const CancellationToken* cancellation, std::vector<float>* moments,
      VideoVaeEncoderTrace* trace, VideoVaeProgress progress,
      void* progress_opaque, VideoVaeTelemetry* telemetry,
      std::string* error = nullptr);

  [[nodiscard]] std::uint64_t weight_bytes() const noexcept;

private:
  struct Impl;
  explicit VideoVaeEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_ENCODER_HPP_
