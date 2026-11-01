#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::ext {

struct ImageBuffer {
  int width{0};
  int height{0};
  std::vector<std::uint8_t> rgba;
};

struct ImageMetrics {
  double prompt_ms{0};
  double denoise_ms{0};
  double vae_ms{0};
  double total_ms{0};
  std::size_t resident_weight_bytes{0};
};

struct ImageRequest {
  std::string prompt;
  std::vector<ImageBuffer> images;
  int width{1024};
  int height{1024};
  int steps{40};
  std::uint64_t seed{0};
};

struct ImageResult {
  ImageBuffer image;
  ImageMetrics metrics;
};

using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for image generation models.
class ImageRunner {
public:
  virtual ~ImageRunner() = default;

  [[nodiscard]] virtual ImageResult Generate(
      const ImageRequest& request,
      const CancellationCheck& is_cancelled = {}) = 0;
};

using ImageProbe = bool (*)(std::string_view model_root);
using ImageArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string* error);
using ImageRunnerFactory = std::unique_ptr<ImageRunner> (*)(
    const std::string& model_root, std::string* error);

struct ImageProvider {
  std::string id;
  ImageProbe probe = nullptr;
  ImageArtifactResolver resolve = nullptr;
  ImageRunnerFactory create = nullptr;
};

void RegisterImageProvider(ImageProvider provider);
const ImageProvider* FindImageProvider(std::string_view id);
const ImageProvider* ProbeImageProvider(std::string_view model_root);

}  // namespace gufo::ext
