#include "src/models/minimax_h3/video_vae_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::optional<VideoVaeEncoderTileAxis> ResolveVideoVaeEncoderTileAxis(
    int extent, std::string* error) {
  VideoVaeEncoderTileAxis axis;
  axis.extent = extent;
  if (extent < kH3VaeEncoderSpatialRatio ||
      extent % kH3VaeEncoderSpatialRatio != 0) {
    SetError(
        error,
        "MiniMax H3 VisualVAE encoder tile extent must be a multiple of 16");
    return std::nullopt;
  }
  if (extent <= kH3VaeEncoderTilePixels) {
    // klvae.py:split_tiles keeps the short axis intact: one tile of `extent`.
    axis.length = extent;
    axis.starts = {0};
    return axis;
  }
  int count = (extent + kH3VaeEncoderTilePixels - 1) / kH3VaeEncoderTilePixels;
  while (kH3VaeEncoderTilePixels * count -
             kH3VaeEncoderTileOverlapMinimum * (count - 1) <
         extent) {
    ++count;
  }
  axis.length = kH3VaeEncoderTilePixels;
  axis.starts.assign(static_cast<std::size_t>(count), 0);
  axis.overlaps.assign(static_cast<std::size_t>(count - 1),
                       kH3VaeEncoderTileOverlapMinimum);
  const int remaining = kH3VaeEncoderTilePixels * count -
                        kH3VaeEncoderTileOverlapMinimum * (count - 1) - extent;
  for (int unit = 0; unit < remaining / kH3VaeEncoderSpatialRatio; ++unit) {
    axis.overlaps[static_cast<std::size_t>(unit % (count - 1))] +=
        kH3VaeEncoderSpatialRatio;
  }
  for (int index = 1; index < count; ++index) {
    axis.starts[static_cast<std::size_t>(index)] =
        axis.starts[static_cast<std::size_t>(index - 1)] +
        kH3VaeEncoderTilePixels -
        axis.overlaps[static_cast<std::size_t>(index - 1)];
  }
  return axis;
}

std::optional<VideoVaeEncoderTilePlan> ResolveVideoVaeEncoderTilePlan(
    int height, int width, std::string* error) {
  if (height < kH3VaeEncoderCanvasMultiple ||
      width < kH3VaeEncoderCanvasMultiple ||
      height % kH3VaeEncoderCanvasMultiple != 0 ||
      width % kH3VaeEncoderCanvasMultiple != 0) {
    SetError(error,
             "MiniMax H3 VisualVAE encoder keyframe must be a multiple "
             "of 32 pixels on both axes");
    return std::nullopt;
  }
  auto vertical = ResolveVideoVaeEncoderTileAxis(height, error);
  if (!vertical) {
    return std::nullopt;
  }
  auto horizontal = ResolveVideoVaeEncoderTileAxis(width, error);
  if (!horizontal) {
    return std::nullopt;
  }
  VideoVaeEncoderTilePlan plan;
  plan.height = height;
  plan.width = width;
  plan.tile_pixels = vertical->length;
  plan.vertical = std::move(*vertical);
  plan.horizontal = std::move(*horizontal);
  plan.tile_latent_height = plan.vertical.length / kH3VaeEncoderSpatialRatio;
  plan.tile_latent_width = plan.horizontal.length / kH3VaeEncoderSpatialRatio;
  plan.latent_height = height / kH3VaeEncoderSpatialRatio;
  plan.latent_width = width / kH3VaeEncoderSpatialRatio;
  const std::size_t tiles =
      plan.vertical.starts.size() * plan.horizontal.starts.size();
  if (tiles > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    SetError(error, "MiniMax H3 VisualVAE encoder tile grid is too large");
    return std::nullopt;
  }
  plan.tiles = static_cast<int>(tiles);
  // Every boundary is latent aligned, so the tiles cover the canvas exactly.
  const int covered_vertical =
      plan.vertical.starts.back() + plan.vertical.length;
  const int covered_horizontal =
      plan.horizontal.starts.back() + plan.horizontal.length;
  if (covered_vertical != height || covered_horizontal != width) {
    SetError(error,
             "MiniMax H3 VisualVAE encoder tiling does not cover the keyframe");
    return std::nullopt;
  }
  return plan;
}

bool StitchVideoVaeEncoderTiles(const VideoVaeEncoderTilePlan& plan,
                                std::span<const std::vector<float>> tiles,
                                std::vector<float>* moments,
                                std::string* error) {
  if (moments != nullptr) {
    moments->clear();
  }
  const int rows = static_cast<int>(plan.vertical.starts.size());
  const int columns = static_cast<int>(plan.horizontal.starts.size());
  const std::size_t tile_elements =
      static_cast<std::size_t>(kH3VaeEncoderMomentChannels) *
      plan.tile_latent_height * plan.tile_latent_width;
  if (moments == nullptr || rows < 1 || columns < 1 ||
      plan.tiles != rows * columns ||
      tiles.size() != static_cast<std::size_t>(plan.tiles) ||
      plan.tile_latent_height < 1 || plan.tile_latent_width < 1) {
    SetError(error, "invalid MiniMax H3 VisualVAE encoder stitch request");
    return false;
  }
  for (const std::vector<float>& tile : tiles) {
    if (tile.size() != tile_elements) {
      SetError(error, "MiniMax H3 VisualVAE encoder tile payload differs");
      return false;
    }
  }
  const int tile_height = plan.tile_latent_height;
  const int tile_width = plan.tile_latent_width;
  moments->assign(static_cast<std::size_t>(kH3VaeEncoderMomentChannels) *
                      plan.latent_height * plan.latent_width,
                  0.0F);
  // klvae.py:tiled_encode reads the untrimmed neighbours: rows blend first,
  // then columns, then each tile keeps everything except its own trailing
  // overlap. Weights are the linear ramp 1 - pos/n and pos/n in float32.
  for (int row = 0; row < rows; ++row) {
    const int overlap_up =
        row == 0 ? 0
                 : plan.vertical.overlaps[static_cast<std::size_t>(row - 1)] /
                       kH3VaeEncoderSpatialRatio;
    const int keep_height =
        tile_height -
        (row + 1 < rows
             ? plan.vertical.overlaps[static_cast<std::size_t>(row)] /
                   kH3VaeEncoderSpatialRatio
             : 0);
    for (int column = 0; column < columns; ++column) {
      const int overlap_left =
          column == 0
              ? 0
              : plan.horizontal.overlaps[static_cast<std::size_t>(column - 1)] /
                    kH3VaeEncoderSpatialRatio;
      const int keep_width =
          tile_width -
          (column + 1 < columns
               ? plan.horizontal.overlaps[static_cast<std::size_t>(column)] /
                     kH3VaeEncoderSpatialRatio
               : 0);
      const std::vector<float>& current =
          tiles[static_cast<std::size_t>(row * columns + column)];
      const std::vector<float>* above =
          row == 0
              ? nullptr
              : &tiles[static_cast<std::size_t>((row - 1) * columns + column)];
      const std::vector<float>* left =
          column == 0
              ? nullptr
              : &tiles[static_cast<std::size_t>(row * columns + column - 1)];
      const int origin_y = plan.vertical.starts[static_cast<std::size_t>(row)] /
                           kH3VaeEncoderSpatialRatio;
      const int origin_x =
          plan.horizontal.starts[static_cast<std::size_t>(column)] /
          kH3VaeEncoderSpatialRatio;
      for (int channel = 0; channel < kH3VaeEncoderMomentChannels; ++channel) {
        const std::size_t plane =
            static_cast<std::size_t>(channel) * tile_height * tile_width;
        for (int y = 0; y < keep_height; ++y) {
          for (int x = 0; x < keep_width; ++x) {
            const std::size_t local =
                plane + static_cast<std::size_t>(y) * tile_width + x;
            float value = current[local];
            if (above != nullptr && y < overlap_up) {
              const std::size_t source =
                  plane +
                  static_cast<std::size_t>(tile_height - overlap_up + y) *
                      tile_width +
                  x;
              const float weight =
                  static_cast<float>(y) / static_cast<float>(overlap_up);
              value = (*above)[source] * (1.0F - weight) + value * weight;
            }
            if (left != nullptr && x < overlap_left) {
              const std::size_t source =
                  plane + static_cast<std::size_t>(y) * tile_width +
                  (tile_width - overlap_left + x);
              const float weight =
                  static_cast<float>(x) / static_cast<float>(overlap_left);
              value = (*left)[source] * (1.0F - weight) + value * weight;
            }
            const std::size_t destination =
                static_cast<std::size_t>(channel) * plan.latent_height *
                    plan.latent_width +
                static_cast<std::size_t>(origin_y + y) * plan.latent_width +
                (origin_x + x);
            (*moments)[destination] = value;
          }
        }
      }
    }
  }
  return true;
}

const VideoVaeEncoderStage* VideoVaeEncoderTrace::Find(
    std::string_view name) const noexcept {
  const auto found =
      std::ranges::find(stages, name, &VideoVaeEncoderStage::name);
  return found == stages.end() ? nullptr : &*found;
}

#if !defined(ENGINE_ENABLE_HIP)

struct VideoVaeEncoder::Impl {};

VideoVaeEncoder::VideoVaeEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VideoVaeEncoder::~VideoVaeEncoder() = default;
VideoVaeEncoder::VideoVaeEncoder(VideoVaeEncoder&&) noexcept = default;
VideoVaeEncoder& VideoVaeEncoder::operator=(VideoVaeEncoder&&) noexcept =
    default;

std::unique_ptr<VideoVaeEncoder> VideoVaeEncoder::Create(
    const ModelInventory&, const CancellationToken*, VideoVaeTelemetry*,
    std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE encoder requires a HIP-enabled build";
  }
  return nullptr;
}

bool VideoVaeEncoder::EncodeKeyframeMoments(const float*, int, int,
                                            const CancellationToken*,
                                            std::vector<float>*,
                                            std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE encoder requires a HIP-enabled build";
  }
  return false;
}

bool VideoVaeEncoder::EncodeKeyframeMoments(
    const float*, int, int, const CancellationToken*, std::vector<float>*,
    VideoVaeEncoderTrace*, VideoVaeProgress, void*, VideoVaeTelemetry*,
    std::string* error) {
  if (error != nullptr) {
    *error = "MiniMax H3 VisualVAE encoder requires a HIP-enabled build";
  }
  return false;
}

std::uint64_t VideoVaeEncoder::weight_bytes() const noexcept {
  return 0;
}

#endif

}  // namespace gufo::minimax_h3
