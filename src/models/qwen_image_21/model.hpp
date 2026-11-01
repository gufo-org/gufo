#ifndef GUFO_MODELS_QWEN_IMAGE_21_MODEL_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_MODEL_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <gufo/ext/image.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::qwen_image_21 {

using Image = ext::ImageBuffer;
using Request = ext::ImageRequest;
using Metrics = ext::ImageMetrics;
using Result = ext::ImageResult;
using CancellationCheck = ext::CancellationCheck;

Image DecodeImage(std::span<const std::uint8_t> bytes);
std::vector<std::uint8_t> EncodePng(const Image& image);
Image ResizeImage(const Image& image, int width, int height);
/// Keep edit noise independent of the noise that generated its references.
/// Supply the canonical resized RGBA references; generation retains its seed.
std::uint64_t InitialNoiseSeed(std::uint64_t seed,
                               std::span<const Image> images);
/// Development observer; tensors are FP32 views of the actual BF16 boundary.
struct Observer {
  std::function<void(std::string_view, std::span<const float>, int, int)> write;
  // Filter before copying tensors to the host, so trajectory checks do not
  // download every intermediate block.
  std::function<bool(std::string_view)> include;
  bool Wants(std::string_view name) const {
    return write && (!include || include(name));
  }
};

void ValidateRequest(const Request& request);
std::vector<float> FlowSigmas(int steps, int image_tokens);

class Model {
public:
  explicit Model(const std::filesystem::path& root);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// One execution at a time. Requests have independent RNG and prefix caches.
  Result Generate(const Request& request,
                  const CancellationCheck& cancelled = {},
                  const Observer& observer = {});
  std::vector<std::uint32_t> Tokenize(std::string_view text) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen_image_21
#endif
