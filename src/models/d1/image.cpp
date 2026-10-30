#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/models/d1/vision.hpp"
namespace gufo::models::d1 {
namespace {
// Torchvision's uint8 antialiased bicubic path uses the Pillow filter
// (a = -0.5), separable passes, signed fixed-point coefficients, and uint8
// saturation after each pass. A floating-point resize followed by one cast
// changes pixels and, consequently, the embedding.
double Cubic(double x) {
  x = std::abs(x);
  if (x < 1.0) {
    return ((1.5 * x - 2.5) * x) * x + 1.0;
  }
  if (x < 2.0) {
    return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
  }
  return 0.0;
}

struct Filter {
  int begin;
  unsigned precision{0};
  std::vector<std::int32_t> weights;
};

std::vector<Filter> Filters(int input, int output) {
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;
  std::vector<Filter> filters;
  filters.reserve(output);
  std::vector<std::vector<double>> normalized;
  double maximum = 0.0;
  for (int i = 0; i < output; ++i) {
    const double center = (i + 0.5) * scale;
    const int begin = std::max(0, static_cast<int>(center - support + 0.5));
    const int end = std::min(input, static_cast<int>(center + support + 0.5));
    std::vector<double> weights;
    double sum = 0.0;
    for (int j = begin; j < end; ++j) {
      const double value = Cubic((j - center + 0.5) / filter_scale);
      weights.push_back(value);
      sum += value;
    }
    for (double& value : weights) {
      value /= sum;
      maximum = std::max(maximum, value);
    }
    filters.push_back({begin, 0, {}});
    normalized.push_back(std::move(weights));
  }
  unsigned precision = 0;
  while (precision < 22 &&
         static_cast<int>(0.5 + maximum * (1 << (precision + 1))) < (1 << 15)) {
    ++precision;
  }
  for (std::size_t i = 0; i < filters.size(); ++i) {
    filters[i].precision = precision;
    for (double value : normalized[i]) {
      value *= (1 << precision);
      filters[i].weights.push_back(
          static_cast<std::int32_t>(value < 0.0 ? value - 0.5 : value + 0.5));
    }
  }
  return filters;
}

std::vector<std::uint8_t> Resize(const core::Image& image, int width,
                                 int height) {
  const int source_width = static_cast<int>(image.width);
  const int source_height = static_cast<int>(image.height);
  if (source_width == width && source_height == height) {
    return image.pixels;
  }
  const auto horizontal = Filters(source_width, width);
  const auto vertical = Filters(source_height, height);
  std::vector<std::uint8_t> intermediate(std::size_t{image.height} * width * 3);
  std::vector<std::uint8_t> output(static_cast<std::size_t>(height) * width *
                                   3);
  for (int y = 0; y < source_height; ++y) {
    for (int x = 0; x < width; ++x) {
      const auto& filter = horizontal[x];
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = 1 << (filter.precision - 1);
        for (std::size_t k = 0; k < filter.weights.size(); ++k) {
          sum += static_cast<std::int64_t>(filter.weights[k]) *
                 image.pixels[(static_cast<std::size_t>(y) * source_width +
                               filter.begin + k) *
                                  3 +
                              c];
        }
        intermediate[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> filter.precision, 0, 255));
      }
    }
  }
  for (int y = 0; y < height; ++y) {
    const auto& filter = vertical[y];
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        std::int64_t sum = 1 << (filter.precision - 1);
        for (std::size_t k = 0; k < filter.weights.size(); ++k) {
          sum += static_cast<std::int64_t>(filter.weights[k]) *
                 intermediate[((filter.begin + k) * width + x) * 3 + c];
        }
        output[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> filter.precision, 0, 255));
      }
    }
  }
  return output;
}

}  // namespace
Picture ProcessImage(const core::Image& image) {
  if (!image.width || !image.height ||
      std::size_t(image.width) * image.height > core::kMaxImagePixels ||
      image.pixels.size() != std::size_t(image.width) * image.height * 3 ||
      double(std::max(image.width, image.height)) /
              std::min(image.width, image.height) >
          200)
    throw std::invalid_argument("invalid d1 image");
  const double h = image.height, w = image.width;
  int height = int(std::nearbyint(h / 32)) * 32;
  int width = int(std::nearbyint(w / 32)) * 32;
  constexpr double minimum = 65536, maximum = 1638400;
  if (double(height) * width > maximum) {
    double scale = std::sqrt(h * w / maximum);
    height = std::max(32, int(std::floor(h / scale / 32)) * 32);
    width = std::max(32, int(std::floor(w / scale / 32)) * 32);
  } else if (double(height) * width < minimum) {
    double scale = std::sqrt(minimum / (h * w));
    height = int(std::ceil(h * scale / 32)) * 32;
    width = int(std::ceil(w * scale / 32)) * 32;
  }
  auto pixels = Resize(image, width, height);
  Picture p{height / 16, width / 16, {}};
  p.patches.reserve(std::size_t(height) * width * 3);
  for (int by = 0; by < p.height; by += 2)
    for (int bx = 0; bx < p.width; bx += 2)
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 16; ++y)
              for (int x = 0; x < 16; ++x)
                p.patches.push_back((pixels[((by + dy) * 16 + y) * width * 3 +
                                            ((bx + dx) * 16 + x) * 3 + c] /
                                         255.0F -
                                     0.5F) /
                                    0.5F);
  return p;
}
}  // namespace gufo::models::d1
