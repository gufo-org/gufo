#ifndef GUFO_SERVER_IMAGE_API_HPP_
#define GUFO_SERVER_IMAGE_API_HPP_

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

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
                        Runner runner = {});
  ~ImageService();
  const std::string& model_id() const { return model_id_; }
  Result Generate(const Request& request, const CancellationCheck& cancelled);

private:
  std::string model_id_;
  Runner runner_;
  std::shared_ptr<ext::ImageRunner> model_;
  std::timed_mutex gate_;
};

HttpResponse HandleImageApiRequest(const HttpRequest& request,
                                   ImageService& service);

}  // namespace gufo::server
#endif
