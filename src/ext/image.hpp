#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/options.hpp"
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
  int steps{0};  // Denoising steps executed after resolving the default.
};

/// Caller's background intent, forwarded unchanged from the HTTP request.
/// kAuto leaves the background to the model and returns its pixels as
/// produced. kOpaque asks for an opaque image; the host composites any
/// remaining alpha over white after generation, so every provider supports it.
/// kTransparent asks for a transparent background. An explicit capability
/// declaration must support it and owns model-specific conditioning. Legacy
/// runners retain the host's historical transparent prompt framing.
enum class ImageBackground : std::uint8_t { kAuto, kOpaque, kTransparent };

struct ImageRequest {
  std::string prompt;
  std::vector<ImageBuffer> images;
  int width{1024};
  int height{1024};
  // Zero selects the checkpoint's default; runners reject unsupported counts.
  int steps{0};
  std::uint64_t seed{0};
  ImageBackground background{ImageBackground::kAuto};
};

/// Request features a runner accepts beyond text-to-image generation. The
/// host rejects unsupported requests with HTTP 400 before inference. An
/// explicit empty declaration accepts neither feature.
struct ImageCapabilities {
  /// Accepts reference images (`/v1/images/edits`).
  bool edits{false};
  /// Accepts `ImageBackground::kTransparent`.
  bool transparent_background{false};
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
  /// Explicit feature declaration, or nullopt for legacy runners. Legacy
  /// runners keep edits enabled and receive the historical RGBA prompt framing
  /// for transparent requests. Declaring capabilities opts into unchanged
  /// prompts and provider-owned conditioning.
  [[nodiscard]] virtual std::optional<ImageCapabilities> Capabilities() const {
    return std::nullopt;
  }
  /// Optional provider-specific request validation, called after the host's
  /// general checks and capability checks but before `Generate`. Throw
  /// `std::invalid_argument` to reject the request with HTTP 400.
  virtual void ValidateRequest(const ImageRequest& request) const {
    (void)request;
  }
};

/// Command whose manifest options reach an image provider. A declaration for
/// the bare `serve` command also applies (see `OptionApplies`).
inline constexpr std::string_view kImageServeCommand = "serve image";

/// Startup inputs for an image runner. `options` holds only explicitly
/// supplied manifest options of the selected provider, already parsed to their
/// declared types and checked by `ValidateSelectedOptions`; existing `path`
/// values name existing files or directories. Defaults for absent options
/// (for example a sidecar beside the model) remain the provider's choice.
struct ImageRunnerOptions {
  std::string model_root;
  ParsedOptions options;
};

using ImageProbe = bool (*)(std::string_view model_root);
using ImageArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string* error);
using ImageRunnerFactory = std::unique_ptr<ImageRunner> (*)(
    const std::string& model_root, std::string* error);
using ImageRunnerOptionsFactory = std::unique_ptr<ImageRunner> (*)(
    const ImageRunnerOptions& options, std::string* error);

struct ImageProvider {
  std::string id;
  ImageProbe probe = nullptr;
  ImageArtifactResolver resolve = nullptr;
  ImageRunnerFactory create = nullptr;
  /// Preferred over create when set; receives validated startup options. A
  /// provider that declares manifest options for `serve image` must set it.
  ImageRunnerOptionsFactory create_with_options = nullptr;
};

void RegisterImageProvider(ImageProvider provider);
const ImageProvider* FindImageProvider(std::string_view id);
const ImageProvider* ProbeImageProvider(std::string_view model_root);

}  // namespace gufo::ext
