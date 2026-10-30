#ifndef GUFO_MODELS_D1_VISION_HPP_
#define GUFO_MODELS_D1_VISION_HPP_
#include <memory>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
namespace gufo::models::d1 {
struct Picture {
  int height{}, width{};       // patch grid, in merge-block order
  std::vector<float> patches;  // [patch][RGB][16][16], normalized to [-1,1]
};
Picture ProcessImage(const core::Image& image);
class Vision {
public:
  ~Vision();
  static std::unique_ptr<Vision> Load(const std::string& path);
  std::vector<float> Encode(const Picture& picture);

private:
  Vision();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::d1
#endif
