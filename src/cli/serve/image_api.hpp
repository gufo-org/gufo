#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "src/ext/image.hpp"

namespace gufo::server {

struct HttpRequest;
struct HttpResponse;

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
  Result Generate(const Request& request, const CancellationCheck& cancelled);

private:
  std::string model_id_;
  std::vector<std::string> model_aliases_;
  Runner runner_;
  std::shared_ptr<ext::ImageRunner> model_;
  std::timed_mutex gate_;
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);

}  // namespace gufo::server
#endif
