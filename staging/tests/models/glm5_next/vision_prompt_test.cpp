#include "staging/models/glm5_next/vision/prompt.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/image.hpp"

namespace glm = gufo::models::glm5_next;
namespace vision = gufo::models::glm5_next::vision;
using gufo::core::Image;

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// The reference glm5v canvas budgets and 0.5/0.5 normalization, spelled the
// way FromGguf produces for the shipped sidecar.
vision::Config ReferenceConfig() {
  vision::Config config;
  config.projection_dim = 4096;
  config.image_size = 448;
  config.patch_size = 14;
  config.embedding_length = 1024;
  config.feed_forward_length = 4096;
  config.block_count = 24;
  config.head_count = 16;
  config.min_pixels = 12544;
  config.max_pixels = 6272000;
  config.image_mean = {0.5F, 0.5F, 0.5F};
  config.image_std = {0.5F, 0.5F, 0.5F};
  return config;
}

Image SolidImage(std::uint32_t width, std::uint32_t height,
                 std::uint8_t level) {
  Image image;
  image.width = width;
  image.height = height;
  image.pixels.assign(static_cast<std::size_t>(width) * height * 3, level);
  return image;
}

void TestCanvas() {
  const vision::Config config = ReferenceConfig();
  Require(config.AlignSize() == 28, "align size must be 28");

  // Zero sides resolve to no canvas.
  Require(vision::CanvasFor(0, 10, config).width == 0, "zero width");
  Require(vision::CanvasFor(10, 0, config).height == 0, "zero height");

  // Below the minimum budget: uniformly grown so the ceil-aligned canvas
  // reaches at least min_pixels (112*112 = 12544 exactly).
  const auto small = vision::CanvasFor(10, 10, config);
  Require(small.width == 112 && small.height == 112, "min budget growth");

  // Aspect preserved under growth: 32x128 -> 56x224.
  const auto tall = vision::CanvasFor(32, 128, config);
  Require(tall.width == 56 && tall.height == 224, "min budget aspect");

  // Within the budgets: only ceil alignment to 28.
  const auto mid = vision::CanvasFor(200, 120, config);
  Require(mid.width == 224 && mid.height == 140, "ceil align only");

  // Already 28-aligned stays as-is.
  const auto exact = vision::CanvasFor(448, 448, config);
  Require(exact.width == 448 && exact.height == 448, "aligned passthrough");

  // Over the maximum: the largest aligned pair inside 6272000 for a square
  // is 2492 (2520^2 exceeds the budget).
  const auto huge = vision::CanvasFor(4096, 4096, config);
  Require(huge.width == 2492 && huge.height == 2492, "max budget shrink");
  Require(static_cast<std::int64_t>(huge.width) * huge.height <=
              static_cast<std::int64_t>(config.max_pixels),
          "max budget respected");
}

void TestPrepareImage() {
  const vision::Config config = ReferenceConfig();

  // Solid content survives the bicubic pass exactly, and an image under the
  // minimum budget is upscaled to fill the whole aligned canvas.
  const Image white_small = SolidImage(56, 56, 255);
  const vision::PreparedImage up = vision::PrepareImage(white_small, config);
  Require(up.nx == 112 && up.ny == 112, "upscaled canvas");
  Require(up.tokens == 16, "token count after upscale");
  Require(up.pixels.size() == 3UL * 112 * 112, "CHW size");
  const std::size_t plane = 112UL * 112;
  for (std::size_t c = 0; c < 3; ++c) {
    Require(up.pixels[c * plane] == 1.0F, "white normalizes to +1");
  }

  // A 200x120 image inside the budgets fills a 224x140 canvas top-left at
  // native scale; the right and bottom padding is black, which normalizes
  // to -mean/std, not zero.
  const Image white_wide = SolidImage(200, 120, 255);
  const vision::PreparedImage pad = vision::PrepareImage(white_wide, config);
  Require(pad.nx == 224 && pad.ny == 140, "padded canvas");
  Require(pad.tokens == 40, "token count with padding");
  Require(pad.pixels.size() == 3UL * 224 * 140, "padded CHW size");
  const std::size_t cp = 224UL * 140;
  Require(pad.pixels[0 * cp + 0] == 1.0F, "content white");
  Require(pad.pixels[0 * cp + 119 * 224 + 199] == 1.0F, "content corner");
  Require(pad.pixels[0 * cp + 139 * 224 + 223] == -1.0F, "padding black");
  Require(pad.pixels[2 * cp + 0 * 224 + 223] == -1.0F, "right padding");
  Require(pad.pixels[2 * cp + 139 * 224 + 0] == -1.0F, "bottom padding");
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t p = 0; p < cp; ++p) {
      Require(std::fabs(pad.pixels[c * cp + p]) == 1.0F, "binary canvas");
    }
  }

  // Truncated pixel buffers produce no prepared image.
  Image broken = SolidImage(8, 8, 10);
  broken.pixels.pop_back();
  const vision::PreparedImage empty = vision::PrepareImage(broken, config);
  Require(empty.nx == 0 && empty.ny == 0 && empty.pixels.empty(),
          "bad input rejected");
}

void Run(const char* name, void (*test)()) {
  try {
    test();
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string(name) + ": " + error.what());
  }
}

}  // namespace

int main() {
  try {
    Run("TestCanvas", TestCanvas);
    Run("TestPrepareImage", TestPrepareImage);
    std::cout << "glm5_next vision_prompt: OK" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "glm5_next vision_prompt FAILED: " << error.what()
              << std::endl;
    return 1;
  }
  return 0;
}
