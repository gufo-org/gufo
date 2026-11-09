// DecodePng and its orientation helpers from src/core/image.cpp in gufo 0.28.1.
#include <png.h>

#include <algorithm>
#include <csetjmp>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::core::libpng_reference {
namespace {

void ValidateDimensions(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0 ||
      std::uint64_t{width} * height > kMaxImagePixels) {
    throw std::invalid_argument(
        "image dimensions exceed the decoded pixel limit");
  }
}

struct PngDecoder {
  png_structp png{nullptr};
  png_infop info{nullptr};
  std::span<const std::uint8_t> encoded;
  std::size_t offset{0};
  Image image;
  std::vector<std::uint8_t> grayscale16;
  std::vector<png_bytep> rows;
  ~PngDecoder() {
    if (png)
      png_destroy_read_struct(&png, &info, nullptr);
  }
};

Image DecodePng(std::span<const std::uint8_t> bytes) {
  auto state = std::make_unique<PngDecoder>();
  state->encoded = bytes;
  state->png = png_create_read_struct(
      PNG_LIBPNG_VER_STRING, nullptr,
      [](png_structp png, png_const_charp) { png_longjmp(png, 1); },
      [](png_structp, png_const_charp) {});
  if (!state->png)
    throw std::runtime_error("cannot initialize PNG decoder");
  state->info = png_create_info_struct(state->png);
  if (!state->info)
    throw std::runtime_error("cannot initialize PNG metadata");
  if (setjmp(png_jmpbuf(state->png)) != 0)
    throw std::invalid_argument("invalid PNG image");
  png_set_read_fn(
      state->png, state.get(),
      [](png_structp png, png_bytep output, png_size_t count) {
        auto& input = *static_cast<PngDecoder*>(png_get_io_ptr(png));
        if (count > input.encoded.size() - input.offset)
          png_error(png, "truncated PNG");
        std::memcpy(output, input.encoded.data() + input.offset, count);
        input.offset += count;
      });
  png_read_info(state->png, state->info);
  const auto width = png_get_image_width(state->png, state->info);
  const auto height = png_get_image_height(state->png, state->info);
  ValidateDimensions(width, height);
  const int depth = png_get_bit_depth(state->png, state->info);
  const int color = png_get_color_type(state->png, state->info);
  const bool gray16 = color == PNG_COLOR_TYPE_GRAY && depth == 16;
  // PIL.convert("RGB") ignores gamma/ICC metadata and drops alpha without
  // compositing. The simplified libpng API applies gamma, so use explicit
  // transforms.
  if (color == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(state->png);
  if (color == PNG_COLOR_TYPE_GRAY && depth < 8)
    png_set_expand_gray_1_2_4_to_8(state->png);
  if (depth == 16 && !gray16)
    png_set_strip_16(state->png);
  if ((color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) &&
      !gray16)
    png_set_gray_to_rgb(state->png);
  png_set_strip_alpha(state->png);
  (void)png_set_interlace_handling(state->png);
  png_read_update_info(state->png, state->info);
  const std::size_t row_bytes = std::size_t{width} * (gray16 ? 2 : 3);
  if (png_get_rowbytes(state->png, state->info) != row_bytes) {
    throw std::invalid_argument("unsupported PNG pixel layout");
  }
  state->image = {width, height,
                  std::vector<std::uint8_t>(std::size_t{width} * height * 3)};
  if (gray16)
    state->grayscale16.resize(row_bytes * height);
  auto* pixels =
      gray16 ? state->grayscale16.data() : state->image.pixels.data();
  state->rows.resize(height);
  for (std::size_t y = 0; y < height; ++y)
    state->rows[y] = pixels + y * row_bytes;
  png_read_image(state->png, state->rows.data());
  png_read_end(state->png, state->info);
  if (gray16) {
    // Pillow's 16-bit grayscale mode converts to RGB by saturating the integer
    // sample, unlike 16-bit RGB's high-byte conversion.
    for (std::size_t i = 0; i < std::size_t{width} * height; ++i) {
      const auto value =
          std::min(255U, (unsigned{pixels[i * 2]} << 8) | pixels[i * 2 + 1]);
      std::fill_n(state->image.pixels.data() + i * 3, 3,
                  static_cast<std::uint8_t>(value));
    }
  }
  return std::move(state->image);
}

unsigned TiffOrientation(std::span<const std::uint8_t> data) {
  if (data.size() < 8)
    return 1;
  const bool little = data[0] == 'I' && data[1] == 'I';
  if (!little && !(data[0] == 'M' && data[1] == 'M'))
    return 1;
  const auto read = [&](std::size_t pos, std::size_t size) -> std::uint32_t {
    if (pos > data.size() || size > data.size() - pos)
      return 0;
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < size; ++i) {
      value |= std::uint32_t{data[pos + i]}
               << (8 * (little ? i : size - 1 - i));
    }
    return value;
  };
  if (read(2, 2) != 42)
    return 1;
  const std::size_t ifd = read(4, 4);
  if (ifd > data.size() || data.size() - ifd < 2)
    return 1;
  const std::size_t count = read(ifd, 2);
  if (count > (data.size() - ifd - 2) / 12)
    return 1;
  for (std::size_t i = 0; i < count; ++i) {
    const auto p = ifd + 2 + i * 12;
    if (read(p, 2) == 0x112 && read(p + 2, 2) == 3 && read(p + 4, 4) == 1) {
      const unsigned value = read(p + 8, 2);
      return value >= 1 && value <= 8 ? value : 1;
    }
  }
  return 1;
}

unsigned PngOrientation(std::span<const std::uint8_t> bytes) {
  for (std::size_t offset = 8; offset + 12 <= bytes.size();) {
    const std::size_t length = (std::uint32_t{bytes[offset]} << 24) |
                               (std::uint32_t{bytes[offset + 1]} << 16) |
                               (std::uint32_t{bytes[offset + 2]} << 8) |
                               bytes[offset + 3];
    if (length > bytes.size() - offset - 12)
      break;
    if (std::memcmp(bytes.data() + offset + 4, "eXIf", 4) == 0) {
      return TiffOrientation(bytes.subspan(offset + 8, length));
    }
    offset += length + 12;
  }
  return 1;
}

Image Orient(Image image, unsigned orientation) {
  if (orientation == 1)
    return image;
  const bool transpose = orientation >= 5;
  Image result{transpose ? image.height : image.width,
               transpose ? image.width : image.height,
               std::vector<std::uint8_t>(image.pixels.size())};
  for (std::uint32_t y = 0; y < result.height; ++y) {
    for (std::uint32_t x = 0; x < result.width; ++x) {
      std::uint32_t sx = x;
      std::uint32_t sy = y;
      switch (orientation) {
        case 2:
          sx = image.width - 1 - x;
          break;
        case 3:
          sx = image.width - 1 - x;
          sy = image.height - 1 - y;
          break;
        case 4:
          sy = image.height - 1 - y;
          break;
        case 5:
          sx = y;
          sy = x;
          break;
        case 6:
          sx = y;
          sy = image.height - 1 - x;
          break;
        case 7:
          sx = image.width - 1 - y;
          sy = image.height - 1 - x;
          break;
        case 8:
          sx = image.width - 1 - y;
          sy = x;
          break;
        default:
          break;
      }
      std::copy_n(
          image.pixels.data() + (std::size_t{sy} * image.width + sx) * 3, 3,
          result.pixels.data() + (std::size_t{y} * result.width + x) * 3);
    }
  }
  return result;
}

}  // namespace

Image DecodeImage(std::span<const std::uint8_t> bytes) {
  return Orient(DecodePng(bytes), PngOrientation(bytes));
}

}  // namespace gufo::core::libpng_reference
