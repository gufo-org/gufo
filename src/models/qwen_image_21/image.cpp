#include "src/core/image.hpp"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "src/models/qwen_image_21/model.hpp"

namespace gufo::models::qwen_image_21 {
namespace {

void CheckImage(const Image& image) {
  if (image.width <= 0 || image.height <= 0 ||
      static_cast<std::uint64_t>(image.width) * image.height >
          core::kMaxImagePixels ||
      image.rgba.size() !=
          static_cast<std::size_t>(image.width) * image.height * 4)
    throw std::invalid_argument("invalid RGBA image dimensions");
}

struct Filter {
  int begin{0};
  std::vector<std::int32_t> weights;
};

struct PngOutput {
  std::vector<std::uint8_t> bytes;
  std::size_t used{0};
};

void WritePng(png_structp png, png_bytep data, png_size_t size) {
  auto& output = *static_cast<PngOutput*>(png_get_io_ptr(png));
  if (size > output.bytes.size() - output.used)
    png_error(png, "output PNG exceeds its allocation");
  std::memcpy(output.bytes.data() + output.used, data, size);
  output.used += size;
}

std::vector<Filter> LanczosFilters(int input, int output) {
  std::vector<Filter> result(output);
  const double scale = static_cast<double>(input) / output;
  const double filter_scale = std::max(scale, 1.0);
  const double support = filter_scale * 3;
  for (int dst = 0; dst < output; ++dst) {
    const double center = (dst + 0.5) * scale;
    const int start = std::max(0, static_cast<int>(center - support + 0.5));
    const int end = std::min(input, static_cast<int>(center + support + 0.5));
    auto& f = result[dst];
    f.begin = start;
    std::vector<double> values;
    double sum = 0;
    for (int i = start; i < end; ++i) {
      const double x = (i - center + 0.5) / filter_scale;
      const auto sinc = [](double z) {
        return z == 0 ? 1.0
                      : std::sin(z * std::numbers::pi) / (z * std::numbers::pi);
      };
      const double w = (x >= -3 && x < 3) ? sinc(x) * sinc(x / 3) : 0;
      values.push_back(w);
      sum += w;
    }
    for (double value : values)
      f.weights.push_back(static_cast<std::int32_t>(value / sum * (1 << 22) +
                                                    (value < 0 ? -0.5 : 0.5)));
  }
  return result;
}

}  // namespace

Image DecodeImage(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > core::kMaxEncodedImageBytes || bytes.size() < 8)
    throw std::invalid_argument("invalid or oversized input image");
  if (png_sig_cmp(bytes.data(), 0, 8) != 0) {
    const auto rgb = core::DecodeImage(bytes);
    Image image{static_cast<int>(rgb.width), static_cast<int>(rgb.height), {}};
    image.rgba.resize(static_cast<std::size_t>(rgb.width) * rgb.height * 4);
    for (std::size_t i = 0; i < image.rgba.size() / 4; ++i) {
      std::memcpy(image.rgba.data() + 4 * i, rgb.pixels.data() + 3 * i, 3);
      image.rgba[4 * i + 3] = 255;
    }
    return image;
  }
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_memory(&png, bytes.data(), bytes.size()))
    throw std::invalid_argument("cannot decode PNG image");
  const auto cleanup = std::unique_ptr<png_image, decltype(&png_image_free)>(
      &png, png_image_free);
  if (!png.width || !png.height ||
      static_cast<std::uint64_t>(png.width) * png.height >
          core::kMaxImagePixels)
    throw std::invalid_argument("PNG image dimensions exceed limits");
  png.format = PNG_FORMAT_RGBA;
  Image image{static_cast<int>(png.width), static_cast<int>(png.height), {}};
  image.rgba.resize(PNG_IMAGE_SIZE(png));
  if (!png_image_finish_read(&png, nullptr, image.rgba.data(), 0, nullptr))
    throw std::invalid_argument("invalid PNG image data");
  return image;
}

std::vector<std::uint8_t> EncodePng(const Image& image) {
  CheckImage(image);
  png_image geometry{};
  geometry.width = image.width;
  geometry.height = image.height;
  geometry.format = PNG_FORMAT_RGBA;
  // Allocate before setjmp; callbacks only mutate heap state and cannot throw.
  const auto output = std::make_unique<PngOutput>();
  output->bytes.resize(PNG_IMAGE_PNG_SIZE_MAX(geometry));
  auto png = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, nullptr,
      [](png_structp p, png_const_charp) { png_longjmp(p, 1); }, nullptr);
  if (!png)
    throw std::runtime_error("cannot allocate PNG encoder");
  auto info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    throw std::runtime_error("cannot allocate PNG metadata");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    throw std::runtime_error("cannot encode output PNG");
  }
  png_set_write_fn(png, output.get(), WritePng, nullptr);
  png_set_IHDR(png, info, image.width, image.height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_set_sRGB(png, info, PNG_sRGB_INTENT_PERCEPTUAL);
  // Low compression avoids blocking the response on an expensive zlib search.
  // Adaptive filtering and every RGBA byte are preserved.
  png_set_compression_level(png, 1);
  png_set_filter(png, PNG_FILTER_TYPE_BASE, PNG_ALL_FILTERS);
  png_write_info(png, info);
  for (int row = 0; row < image.height; ++row)
    png_write_row(png, const_cast<png_bytep>(image.rgba.data() +
                                             static_cast<std::size_t>(row) *
                                                 image.width * 4));
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  output->bytes.resize(output->used);
  return std::move(output->bytes);
}

Image ResizeImage(const Image& image, int width, int height) {
  CheckImage(image);
  if (width <= 0 || height <= 0 ||
      static_cast<std::uint64_t>(width) * height > core::kMaxImagePixels)
    throw std::invalid_argument("invalid resized image dimensions");
  if (width == image.width && height == image.height)
    return image;
  // Pillow resizes RGBA in premultiplied space using 22-bit integer Lanczos
  // coefficients. Retain each pass's uint8 rounding, including alpha.
  Image source = image;
  for (std::size_t i = 0; i < source.rgba.size(); i += 4)
    for (int c = 0; c < 3; ++c) {
      const int v = source.rgba[i + c] * source.rgba[i + 3] + 128;
      source.rgba[i + c] = static_cast<std::uint8_t>((v + (v >> 8)) >> 8);
    }
  const auto horizontal = LanczosFilters(image.width, width);
  const auto vertical = LanczosFilters(image.height, height);
  Image temp{width, image.height,
             std::vector<std::uint8_t>(static_cast<std::size_t>(width) *
                                       image.height * 4)};
  Image output{
      width, height,
      std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height * 4)};
  for (int y = 0; y < image.height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 4; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = horizontal[x];
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 source.rgba[(static_cast<std::size_t>(y) * image.width +
                              f.begin + k) *
                                 4 +
                             c];
        temp.rgba[(static_cast<std::size_t>(y) * width + x) * 4 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 4; ++c) {
        std::int64_t sum = 1 << 21;
        const auto& f = vertical[y];
        for (std::size_t k = 0; k < f.weights.size(); ++k)
          sum += static_cast<std::int64_t>(f.weights[k]) *
                 temp.rgba[((f.begin + k) * width + x) * 4 + c];
        output.rgba[(static_cast<std::size_t>(y) * width + x) * 4 + c] =
            static_cast<std::uint8_t>(
                std::clamp<std::int64_t>(sum >> 22, 0, 255));
      }
  for (std::size_t i = 0; i < output.rgba.size(); i += 4) {
    const int alpha = output.rgba[i + 3];
    if (alpha != 0 && alpha != 255)
      for (int c = 0; c < 3; ++c)
        output.rgba[i + c] = static_cast<std::uint8_t>(
            std::min(255, output.rgba[i + c] * 255 / alpha));
  }
  return output;
}

void ValidateRequest(const Request& request) {
  if (request.prompt.size() > (64U << 10U))
    throw std::invalid_argument("image prompt exceeds 64 KiB");
  if (request.width < 32 || request.height < 32 || request.width % 32 ||
      request.height % 32 || request.width > 4096 || request.height > 4096 ||
      static_cast<std::uint64_t>(request.width) * request.height > 4194304)
    throw std::invalid_argument(
        "image size must be multiples of 32, at most 4096 per side and 4 "
        "megapixels");
  if (request.steps < 2 || request.steps > 100)
    throw std::invalid_argument("image steps must be between 2 and 100");
  if (request.images.size() > 10)
    throw std::invalid_argument(
        "Qwen-Image supports at most 10 reference images");
  std::size_t pixels = 0;
  for (const auto& image : request.images) {
    CheckImage(image);
    pixels += static_cast<std::size_t>(image.width) * image.height;
  }
  if (pixels > core::kMaxImagePixels)
    throw std::invalid_argument(
        "reference images exceed 32 megapixels combined");
}

std::vector<float> FlowSigmas(int steps, int image_tokens) {
  if (steps < 2 || steps > 100 || image_tokens < 1)
    throw std::invalid_argument("invalid image flow schedule");
  const double mu = 0.5 + (image_tokens - 256) * (0.9 - 0.5) / (8192 - 256);
  const float shift = static_cast<float>(std::exp(mu));
  std::vector<float> sigmas(steps + 1);
  for (int i = 0; i < steps; ++i) {
    const float t = static_cast<float>(1.0 - static_cast<double>(i) / steps);
    sigmas[i] = shift / (shift + (1.0F / t - 1.0F));
  }
  const float scale = (1.0F - sigmas[steps - 1]) / (1.0F - 0.02F);
  for (int i = 0; i < steps; ++i)
    sigmas[i] = 1.0F - (1.0F - sigmas[i]) / scale;
  sigmas[steps] = 0;
  return sigmas;
}

}  // namespace gufo::models::qwen_image_21
