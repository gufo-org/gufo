#ifndef GUFO_MODELS_YUE2_HIP_VAE_HPP_
#define GUFO_MODELS_YUE2_HIP_VAE_HPP_

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/models/yue2/config.hpp"

namespace gufo::models::yue2::hip {

/// Called after each decoded tile; returning false cancels.
using TileProgress = std::function<bool(int completed, int total)>;

/// The FP32 Oobleck decoder of YuE2-Vae: latents at 25 Hz to 48 kHz stereo.
/// Decodes in tiles with exact halo crops, as upstream `decode_tiled`.
class VaeDecoder {
public:
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  [[nodiscard]] static std::unique_ptr<VaeDecoder> Create(
      const std::filesystem::path& vae_root, std::string* error);

  /// `latents` is `[frames][64]`; `audio` receives frame-interleaved stereo,
  /// `1920 * frames - 64` frames, not clamped.
  [[nodiscard]] bool Decode(std::span<const float> latents,
                            const TileProgress& progress,
                            std::vector<float>* audio, std::string* error);

  [[nodiscard]] const VaeConfig& config() const;

private:
  struct Impl;
  explicit VaeDecoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::yue2::hip

#endif  // GUFO_MODELS_YUE2_HIP_VAE_HPP_
