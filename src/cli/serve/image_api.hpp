#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/ext/image.hpp"

namespace gufo::ext {
class Registry;
}

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

/// Adapts a callback to `ext::ImageRunner` with explicit capabilities; used to
/// inject runners in tests and embeddings without a registered provider.
class CallbackImageRunner final : public ext::ImageRunner {
public:
  using Callback = std::function<ext::ImageResult(
      const ext::ImageRequest&, const ext::CancellationCheck&)>;
  CallbackImageRunner(Callback callback, ext::ImageCapabilities capabilities)
      : callback_(std::move(callback)), capabilities_(capabilities) {}
  ext::ImageResult Generate(const ext::ImageRequest& request,
                            const ext::CancellationCheck& cancelled) override {
    return callback_(request, cancelled);
  }
  std::optional<ext::ImageCapabilities> Capabilities() const override {
    return capabilities_;
  }

private:
  Callback callback_;
  ext::ImageCapabilities capabilities_;
};

/// Selects the image provider for a model directory: the provider whose probe
/// claims it, else the built-in Qwen-Image provider (historical default).
/// Explicit startup options are then validated against the selected provider:
/// options of another provider, a missing provider registration and `path`
/// values that do not exist throw `std::invalid_argument`. No weights load.
const ext::ImageProvider& SelectImageProvider(
    const ext::Registry& registry, const std::filesystem::path& root,
    const ext::ParsedOptions& options);

/// Creates the runner through the provider's factory, forwarding only the
/// validated options. Throws `std::runtime_error` when the factory fails or
/// the provider has no execution factory in this build.
std::shared_ptr<ext::ImageRunner> CreateImageRunner(
    const ext::ImageProvider& provider, const std::filesystem::path& root,
    const ext::ParsedOptions& options);

class ImageService {
public:
  using Request = ext::ImageRequest;
  using Result = ext::ImageResult;
  using CancellationCheck = ext::CancellationCheck;

  explicit ImageService(std::shared_ptr<ext::ImageRunner> runner,
                        std::string model_id = "Qwen-Image-2.1",
                        std::vector<std::string> model_aliases = {});
  ~ImageService();
  const std::string& model_id() const { return model_id_; }
  /// True for the primary ID or a configured alias (#957).
  bool serves_model(std::string_view name) const {
    return name == model_id_ ||
           std::find(model_aliases_.begin(), model_aliases_.end(),
                     std::string(name)) != model_aliases_.end();
  }
  const std::vector<std::string>& model_aliases() const {
    return model_aliases_;
  }
  ext::ImageCapabilities capabilities() const {
    return capabilities_.value_or(ext::ImageCapabilities{true, true});
  }
  /// Capability and provider validation; throws `std::invalid_argument`.
  void Validate(const Request& request) const;
  Result Generate(const Request& request, const CancellationCheck& cancelled);

private:
  std::string model_id_;
  std::vector<std::string> model_aliases_;
  std::shared_ptr<ext::ImageRunner> model_;
  std::optional<ext::ImageCapabilities> capabilities_;
  std::timed_mutex gate_;
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);

}  // namespace gufo::server
#endif
