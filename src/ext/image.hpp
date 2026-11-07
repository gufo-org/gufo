#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
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
  double total_ms{0};  // Includes lora_ms.
  std::size_t resident_weight_bytes{0};
  int steps{0};       // Denoising steps executed after resolving the default.
  double lora_ms{0};  // Adapter switch, before prompt encoding.
};

/// A LoRA adapter registered with ImageRunner::AddLora, selected by name; the
/// scale multiplies the adapter's own alpha/rank.
struct ImageLora {
  std::string name;
  float scale{1};
  bool operator==(const ImageLora&) const = default;
};

struct ImageLoraInfo {
  std::string format;
  std::size_t targets{0};  // Base weights the adapter patches.
  int min_rank{0};
  int max_rank{0};
};

struct ImageRequest {
  std::string prompt;
  std::vector<ImageBuffer> images;
  int width{1024};
  int height{1024};
  // Zero selects the checkpoint's default; runners reject unsupported counts.
  int steps{0};
  std::uint64_t seed{0};
  // Applied in order; scale 0 entries are ignored.
  std::vector<ImageLora> loras{};
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

  /// Validate and register an adapter before serving. Runners without
  /// adapter support reject every adapter.
  virtual ImageLoraInfo AddLora(const std::string& name,
                                const std::string& path) {
    (void)path;
    throw std::invalid_argument(
        "this image model does not support LoRA "
        "adapters: " +
        name);
  }
  /// Adapters applied to the weights now; empty while a switch is unfinished.
  [[nodiscard]] virtual std::vector<ImageLora> ActiveLoras() const {
    return {};
  }
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
