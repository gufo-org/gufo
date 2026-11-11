#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
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
  /// Accepts every adapter without reading it.
  ext::ImageLoraInfo AddLora(const std::string&, const std::string&) override {
    return {};
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

/// Adapters registered at startup and the set applied when a request omits
/// `loras`.
struct ImageLoraOptions {
  std::vector<std::pair<std::string, std::filesystem::path>> files;
  std::vector<ext::ImageLora> defaults;
};

/// Checks adapter names and defaults without opening any file; throws
/// `std::invalid_argument`.
void ValidateLoraOptions(const ImageLoraOptions& loras);

/// How `GET /sdapi/v1/loras` lists a registered adapter, as in
/// stable-diffusion.cpp: `path` is a unique selection ID (the file name,
/// or `NAME/file name` when several adapters share it) and `display_path`
/// the file name (the full path when shared).
struct ImageLoraListing {
  std::string path;
  std::string display_path;
};

class ImageService {
public:
  using Request = ext::ImageRequest;
  using Result = ext::ImageResult;
  using CancellationCheck = ext::CancellationCheck;

  explicit ImageService(std::shared_ptr<ext::ImageRunner> runner,
                        std::string model_id = "Qwen-Image-2.1",
                        std::vector<std::string> model_aliases = {},
                        ImageLoraOptions loras = {});
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
  const ImageLoraOptions& loras() const { return loras_; }
  /// Parallel to `loras().files`.
  std::vector<ImageLoraListing> LoraListings() const;
  bool HasLora(std::string_view name) const;
  /// Empty for a CallbackImageRunner (no files are read).
  const ext::ImageLoraInfo& lora_info(std::string_view name) const;
  std::vector<ext::ImageLora> ActiveLoras() const;
  Result Generate(const Request& request, const CancellationCheck& cancelled);

private:
  std::string model_id_;
  std::vector<std::string> model_aliases_;
  ImageLoraOptions loras_;
  std::map<std::string, ext::ImageLoraInfo, std::less<>> lora_info_;
  std::shared_ptr<ext::ImageRunner> model_;
  std::optional<ext::ImageCapabilities> capabilities_;
  std::timed_mutex gate_;
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);
/// `GET /sdapi/v1/loras`: registered adapters in registration order.
HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service);

}  // namespace gufo::server
#endif
