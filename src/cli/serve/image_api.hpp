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

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

/// Adapters registered at startup and the set applied when a request omits
/// `loras`.
struct ImageLoraOptions {
  std::vector<std::pair<std::string, std::filesystem::path>> files;
  std::vector<ext::ImageLora> defaults;
};

class ImageService {
public:
  using Request = ext::ImageRequest;
  using Result = ext::ImageResult;
  using CancellationCheck = ext::CancellationCheck;
  using Runner =
      std::function<Result(const Request&, const CancellationCheck&)>;

  explicit ImageService(const std::filesystem::path& model_root,
                        std::string model_id = "Qwen-Image-2.1",
                        Runner runner = {},
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
  const ImageLoraOptions& loras() const { return loras_; }
  bool HasLora(std::string_view name) const;
  /// Empty with an injected runner (no files are read).
  const ext::ImageLoraInfo& lora_info(std::string_view name) const;
  std::vector<ext::ImageLora> ActiveLoras() const;
  Result Generate(const Request& request, const CancellationCheck& cancelled);

private:
  std::string model_id_;
  std::vector<std::string> model_aliases_;
  Runner runner_;
  ImageLoraOptions loras_;
  std::map<std::string, ext::ImageLoraInfo, std::less<>> lora_info_;
  std::shared_ptr<ext::ImageRunner> model_;
  std::timed_mutex gate_;
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);
/// `GET /sdapi/v1/loras`: registered adapters in registration order.
HttpResponse HandleImageLorasRequest(const HttpRequest& request,
                                     const ImageService& service);

}  // namespace gufo::server
#endif
