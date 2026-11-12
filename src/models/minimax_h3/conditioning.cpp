#include "src/models/minimax_h3/conditioning.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <gufo/ext/image.hpp>
#include <memory>
#include <utility>

#include "src/models/minimax_h3/image_processor.hpp"
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/sampling.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

// Reference pixel contract: uint8 -> f32 / 255 -> ImageNet normalize
// (FL2VA/video_vae/normalize.py; encoders.py:129 keeps the two eager steps).
constexpr float kPixelMean[3] = {0.485F, 0.456F, 0.406F};
constexpr float kPixelStd[3] = {0.229F, 0.224F, 0.225F};

}  // namespace

bool DecodeKeyframe(const std::filesystem::path& path, core::Image* out,
                    std::string* error) {
  if (out == nullptr) {
    SetError(error, "MiniMax H3 keyframe output is missing");
    return false;
  }
  std::error_code filesystem_error;
  const std::uintmax_t encoded_bytes =
      std::filesystem::file_size(path, filesystem_error);
  if (filesystem_error || encoded_bytes == 0 ||
      encoded_bytes > core::kMaxEncodedImageBytes) {
    SetError(error,
             "MiniMax H3 keyframe file is empty, unreadable, or over "
             "the encoded image limit");
    return false;
  }
  core::Image decoded;
  try {
    decoded = core::DecodeImage(core::ReadImageFile(path));
  } catch (const std::exception& exception) {
    SetError(error, std::string("MiniMax H3 keyframe decode failed: ") +
                        exception.what());
    return false;
  }
  if (decoded.width == 0 || decoded.height == 0 ||
      decoded.pixels.size() !=
          static_cast<std::size_t>(decoded.width) * decoded.height * 3U) {
    SetError(error, "MiniMax H3 keyframe decoded to an invalid RGB image");
    return false;
  }
  *out = std::move(decoded);
  return true;
}

bool PrepareKeyframe(const core::Image& decoded, int canvas_width,
                     int canvas_height, bool follower, PreparedKeyframe* out,
                     std::string* error) {
  if (out == nullptr || canvas_width < kH3VideoVaeSpatialRatio ||
      canvas_height < kH3VideoVaeSpatialRatio ||
      canvas_width % kH3VideoVaeSpatialRatio != 0 ||
      canvas_height % kH3VideoVaeSpatialRatio != 0) {
    SetError(error, "invalid MiniMax H3 keyframe preparation request");
    return false;
  }
  out->rgb.clear();
  const bool resized = image_processor::PrepareKeyframe(
      decoded, canvas_width, canvas_height,
      follower ? image_processor::KeyframeFit::kCoverCrop
               : image_processor::KeyframeFit::kStretch,
      &out->rgb, error);
  if (!resized) {
    return false;
  }
  out->width = canvas_width;
  out->height = canvas_height;
  out->latent.clear();
  return true;
}

bool ReadKeyframeNormalization(const ModelInventory& inventory,
                               KeyframeLatentNormalization* mean,
                               KeyframeLatentNormalization* deviation,
                               std::string* error) {
  if (mean == nullptr || deviation == nullptr) {
    SetError(error, "invalid MiniMax H3 keyframe normalization output");
    return false;
  }
  try {
    const json::Value config = json::ParseFile(
        inventory.model_root() / "FL2VA/video_vae/config.json", 1U << 20U);
    const json::Value* mean_value = config.Find("latents_mean");
    const json::Value* deviation_value = config.Find("latents_std");
    if (mean_value == nullptr || deviation_value == nullptr ||
        !mean_value->IsArray() || !deviation_value->IsArray() ||
        mean_value->AsArray().size() != mean->size() ||
        deviation_value->AsArray().size() != deviation->size()) {
      throw json::Error("latent normalization shape differs");
    }
    for (std::size_t index = 0; index < mean->size(); ++index) {
      (*mean)[index] =
          static_cast<float>(mean_value->AsArray()[index].AsDouble());
      (*deviation)[index] =
          static_cast<float>(deviation_value->AsArray()[index].AsDouble());
      if (!std::isfinite((*mean)[index]) ||
          !std::isfinite((*deviation)[index]) || (*deviation)[index] <= 0.0F) {
        throw json::Error("latent normalization contains invalid values");
      }
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax H3 keyframe config failed: " +
                        std::string(exception.what()));
    return false;
  }
}

bool EncodeKeyframeLatent(const ModelInventory& inventory,
                          PreparedKeyframe& keyframe,
                          const CancellationToken* cancellation,
                          VideoVaeTelemetry* telemetry, std::string* error) {
  const int latent_height = keyframe.height / kH3VideoVaeSpatialRatio;
  const int latent_width = keyframe.width / kH3VideoVaeSpatialRatio;
  if (keyframe.width % kH3VideoVaeSpatialRatio != 0 ||
      keyframe.height % kH3VideoVaeSpatialRatio != 0 || latent_height < 2 ||
      latent_width < 2 ||
      keyframe.rgb.size() !=
          static_cast<std::size_t>(keyframe.width) * keyframe.height * 3U) {
    SetError(error, "MiniMax H3 keyframe canvas is not VAE-aligned");
    return false;
  }
  KeyframeLatentNormalization mean{};
  KeyframeLatentNormalization deviation{};
  if (!ReadKeyframeNormalization(inventory, &mean, &deviation, error)) {
    return false;
  }
  std::unique_ptr<VideoVaeEncoder> encoder =
      VideoVaeEncoder::Create(inventory, cancellation, telemetry, error);
  if (encoder == nullptr) {
    return false;
  }
  const std::size_t plane =
      static_cast<std::size_t>(keyframe.width) * keyframe.height;
  std::vector<float> pixels(3U * plane);
  for (std::size_t index = 0; index < plane; ++index) {
    for (std::size_t channel = 0; channel < 3U; ++channel) {
      const float scaled =
          static_cast<float>(keyframe.rgb[index * 3U + channel]) / 255.0F;
      pixels[channel * plane + index] =
          (scaled - kPixelMean[channel]) / kPixelStd[channel];
    }
  }
  std::vector<float> moments;
  if (!encoder->EncodeKeyframeMoments(pixels.data(), keyframe.height,
                                      keyframe.width, cancellation, &moments,
                                      error)) {
    return false;
  }
  const std::size_t latent_elements =
      static_cast<std::size_t>(kH3VideoVaeLatentChannels) * latent_height *
      latent_width;
  if (moments.size() != latent_elements * 2U) {
    SetError(error,
             "MiniMax H3 keyframe encoder returned an unexpected "
             "moment count");
    return false;
  }
  // Posterior sample (DiagonalGaussianDistribution.sample in fp32):
  // z = mean + exp(0.5 * clamp(logvar, -30, 20)) * epsilon. A fresh fixed-
  // seed stream per keyframe reproduces the reference contract that anchors
  // depend only on pixels (diffusers: Generator().manual_seed(42),
  // encoders.py:131; identical shapes draw identical epsilon, as upstream).
  std::vector<float>& latent = keyframe.latent;
  latent.resize(latent_elements);
  NormalRng posterior(kH3KeyframePosteriorSeed);
  const std::size_t spatial =
      static_cast<std::size_t>(latent_height) * latent_width;
  for (std::size_t element = 0; element < latent_elements; ++element) {
    const float logvar =
        std::clamp(moments[latent_elements + element], -30.0F, 20.0F);
    const float sigma = std::exp(0.5F * logvar);
    // Elements run channel-major over (1, lh, lw); torch.chunk(dim=1) puts
    // mean in channels 0..23 and logvar in 24..47 of the same layout, so
    // the logvar of `element` sits exactly latent_elements positions later.
    latent[element] = moments[element] + sigma * posterior.NextNormal();
  }
  // Explicit fp16 round-trip before normalization (encoders.py:132).
  for (std::size_t element = 0; element < latent_elements; ++element) {
    latent[element] =
        static_cast<float>(static_cast<_Float16>(latent[element]));
    const std::size_t channel = element / spatial;
    latent[element] = (latent[element] - mean[channel]) / deviation[channel];
  }
  return true;
}

bool MixKeyframeNoise(const PreparedKeyframe& keyframe,
                      std::span<const float> augmentation,
                      std::vector<float>* mixed, std::string* error) {
  if (mixed == nullptr || keyframe.latent.empty() ||
      augmentation.size() != keyframe.latent.size()) {
    SetError(error, "invalid MiniMax H3 keyframe noise mix");
    return false;
  }
  mixed->resize(keyframe.latent.size());
  // Reference scale_noise with the t=0.999 fp32 tensor: (1 - t) is evaluated
  // in float32 and the products keep eager float32 rounding boundaries
  // (scheduling_minimax_h3.py; before_denoise.py:958-959). Contraction into
  // FMA would shift ~1e-8 of the anchor and is amplified downstream.
  {
#pragma clang fp contract(off)
    const float clean = kH3KeyframeConditionTimestep;
    const float noisy = 1.0F - kH3KeyframeConditionTimestep;
    for (std::size_t element = 0; element < mixed->size(); ++element) {
      (*mixed)[element] =
          clean * keyframe.latent[element] + noisy * augmentation[element];
    }
  }
  return true;
}

}  // namespace gufo::minimax_h3
