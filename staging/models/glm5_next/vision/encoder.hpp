#ifndef GUFO_MODELS_GLM5_NEXT_VISION_ENCODER_HPP_
#define GUFO_MODELS_GLM5_NEXT_VISION_ENCODER_HPP_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "staging/models/glm5_next/vision/config.hpp"

namespace gufo::models::glm5_next::vision {

/// ROCm vision tower for the `glm5v` projector. Uploads the mmproj tower
/// once as FP32 and runs patch embed (folded temporal-2 Conv3d), 24 ViT
/// blocks (RMS norms, per-head Q/K norm, vision RoPE, swiglu-clamp FFNs),
/// post_ln, the 2x2 merger conv, fc + LayerNorm + gelu_erf and the
/// swiglu-clamp projector FFN. Mirrors `ReferenceEncoder`'s contract: CHW
/// pixels already normalized with the config mean/std, nx/ny multiples of
/// patch*merge, output row-major [n_tokens x projection_dim].
class Encoder {
public:
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  /// Opens the mmproj sidecar, binds and uploads every tower tensor.
  [[nodiscard]] static std::unique_ptr<Encoder> Open(
      const std::string& mmproj_path, std::string* error_msg);

  /// Runs the tower over one image. `pixels` is CHW, `3 * ny * nx` floats
  /// normalized with the config mean/std. `out` receives
  /// `(nx/patch/merge) * (ny/patch/merge) * projection_dim` floats.
  [[nodiscard]] bool Encode(const float* pixels, std::uint32_t nx,
                            std::uint32_t ny, std::vector<float>& out,
                            std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept;

private:
  struct Impl;
  explicit Encoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::glm5_next::vision

#endif  // GUFO_MODELS_GLM5_NEXT_VISION_ENCODER_HPP_
