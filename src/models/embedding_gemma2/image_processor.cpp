#include "src/models/embedding_gemma2/image_processor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gufo::models::embedding_gemma2 {
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

bool ValidateImageProcessor(const std::string& model_dir, std::string* error) {
  std::ifstream file(std::filesystem::path(model_dir) /
                     "processor_config.json");
  std::ostringstream contents;
  contents << file.rdbuf();
  try {
    const auto root = json::parse(contents.str());
    const auto* config = root.find("image_processor");
    const auto enabled = [&](const char* key, bool expected) {
      const auto* value = config ? config->find(key) : nullptr;
      return value && value->is_bool() && value->as_bool() == expected;
    };
    if (config && config->is_object() &&
        config->member_str("image_processor_type") == "Gemma4ImageProcessor" &&
        enabled("do_convert_rgb", true) && enabled("do_resize", true) &&
        enabled("do_rescale", true) && enabled("do_normalize", false) &&
        config->member_size("patch_size") == 16 &&
        config->member_size("pooling_kernel_size") == 3 &&
        config->member_size("max_soft_tokens") == kDefaultImageSoftTokens &&
        config->member_size("resample") == 3 &&
        std::abs(config->member_double("rescale_factor") - 1.0 / 255.0) <
            1e-15) {
      return true;
    }
  } catch (const std::exception&) {
  }
  if (error)
    *error =
        "embedding_gemma2: unsupported or missing image processor "
        "configuration";
  return false;
}

bool SupportedImageSoftTokens(std::uint32_t max_soft_tokens) {
  return std::ranges::find(kImageSoftTokenBudgets, max_soft_tokens) !=
         kImageSoftTokenBudgets.end();
}

ImagePatches ProcessImage(const core::Image& image,
                          std::uint32_t max_soft_tokens) {
  if (!SupportedImageSoftTokens(max_soft_tokens)) {
    throw std::invalid_argument(
        "image max_soft_tokens must be one of 70, 140, 280, 560, or 1120");
  }
  if (image.width == 0 || image.height == 0 || image.width > kMaxImageSide ||
      image.height > kMaxImageSide ||
      std::size_t{image.width} * image.height > core::kMaxImagePixels ||
      image.pixels.size() != std::size_t{image.width} * image.height * 3) {
    throw std::invalid_argument(
        "invalid embedding image dimensions or RGB data");
  }
  const int budget = static_cast<int>(max_soft_tokens);
  constexpr int kStride = 48;
  const double factor = std::sqrt(
      (budget * 9.0 * 256) / (static_cast<double>(image.width) * image.height));
  int height =
      static_cast<int>(std::floor(factor * image.height / kStride)) * kStride;
  int width =
      static_cast<int>(std::floor(factor * image.width / kStride)) * kStride;
  if (height == 0) {
    height = kStride;
    width = static_cast<int>(std::min<std::uint32_t>(image.width / image.height,
                                                     max_soft_tokens)) *
            kStride;
  } else if (width == 0) {
    width = kStride;
    height = static_cast<int>(std::min<std::uint32_t>(
                 image.height / image.width, max_soft_tokens)) *
             kStride;
  }
  const auto resized = Resize(image, width, height);
  ImagePatches result;
  result.rows = static_cast<std::uint32_t>(height / 16);
  result.columns = static_cast<std::uint32_t>(width / 16);
  result.pixels.reserve(resized.size());
  for (int py = 0; py < height; py += 16) {
    for (int px = 0; px < width; px += 16) {
      for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
          for (int c = 0; c < 3; ++c) {
            result.pixels.push_back(
                resized[((py + y) * width + px + x) * 3 + c] * (1.0F / 255.0F));
          }
        }
      }
    }
  }
  return result;
}

}  // namespace gufo::models::embedding_gemma2
