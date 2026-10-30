#ifndef GUFO_MODELS_LFM3_VL_VISION_HPP_
#define GUFO_MODELS_LFM3_VL_VISION_HPP_
#include <memory>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/image.hpp"
namespace gufo::models::lfm3_vl {
struct Picture {
  int height{}, width{};       // patch grid, in merge-block order
  std::vector<float> patches;  // [patch][RGB][16][16], normalized to [-1,1]
};
Picture ProcessImage(const core::Image& image, unsigned max_pixels = 16777216);
class Vision {
public:
  ~Vision();
  static std::unique_ptr<Vision> Load(const std::string& path);
  std::vector<float> Encode(const Picture& picture);
  const std::string& Identity() const;

private:
  Vision();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::models::lfm3_vl
#endif
