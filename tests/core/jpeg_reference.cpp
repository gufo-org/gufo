// libjpeg-turbo JPEG decoding as DecodeImage did before the Rust port, frozen
// as the reference for jpeg_decode_test. It includes EXIF orientation.

#include <cstdio>
// jpeglib.h uses FILE but doesn't include <stdio.h>.
#include <jpeglib.h>

#include <algorithm>
#include <csetjmp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::core::libjpeg_reference {
namespace {

void ValidateDimensions(std::uint32_t width, std::uint32_t height) {
  if (width == 0 || height == 0 ||
      std::uint64_t{width} * height > kMaxImagePixels) {
    throw std::invalid_argument(
        "image dimensions exceed the decoded pixel limit");
  }
}

struct JpegError {
  jpeg_error_mgr manager{};
  std::jmp_buf jump;
};

struct JpegDecoder {
  jpeg_decompress_struct decoder{};
  JpegError error;
  Image image;
  std::vector<std::uint8_t> cmyk_row;
  bool created{false};
  ~JpegDecoder() {
    if (created)
      jpeg_destroy_decompress(&decoder);
  }
};

Image DecodeJpeg(std::span<const std::uint8_t> bytes) {
  // Keep every modified C++ object on the heap, outside libjpeg's longjmp.
  auto state = std::make_unique<JpegDecoder>();
  state->decoder.err = jpeg_std_error(&state->error.manager);
  state->error.manager.error_exit = [](j_common_ptr common) {
    std::longjmp(reinterpret_cast<JpegError*>(common->err)->jump, 1);
  };
  state->error.manager.output_message = [](j_common_ptr) {};
  if (setjmp(state->error.jump) != 0) {
    throw std::invalid_argument("invalid JPEG image");
  }
  jpeg_create_decompress(&state->decoder);
  state->created = true;
  jpeg_mem_src(&state->decoder, bytes.data(), bytes.size());
  if (jpeg_read_header(&state->decoder, TRUE) != JPEG_HEADER_OK) {
    throw std::invalid_argument("invalid JPEG header");
  }
  ValidateDimensions(state->decoder.image_width, state->decoder.image_height);
  const bool cmyk = state->decoder.jpeg_color_space == JCS_CMYK ||
                    state->decoder.jpeg_color_space == JCS_YCCK;
  state->decoder.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
  jpeg_start_decompress(&state->decoder);
  state->image.width = state->decoder.output_width;
  state->image.height = state->decoder.output_height;
  state->image.pixels.resize(std::size_t{state->image.width} *
                             state->image.height * 3);
  if (cmyk)
    state->cmyk_row.resize(std::size_t{state->image.width} * 4);
  while (state->decoder.output_scanline < state->decoder.output_height) {
    auto* rgb =
        state->image.pixels.data() +
        std::size_t{state->decoder.output_scanline} * state->image.width * 3;
    JSAMPROW row = cmyk ? state->cmyk_row.data() : rgb;
    if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1) {
      throw std::invalid_argument("truncated JPEG image");
    }
    if (cmyk) {
      // Pillow decodes JPEG CMYK with inverted channels, then converts to RGB.
      for (std::size_t x = 0; x < state->image.width; ++x) {
        for (std::size_t c = 0; c < 3; ++c) {
          const unsigned product =
              unsigned{row[x * 4 + c]} * row[x * 4 + 3] + 128;
          rgb[x * 3 + c] =
              static_cast<std::uint8_t>((product + (product >> 8)) >> 8);
        }
      }
    }
  }
  jpeg_finish_decompress(&state->decoder);
  if (state->error.manager.num_warnings != 0)
    throw std::invalid_argument("corrupt or truncated JPEG image");
  return std::move(state->image);
}
// Bounds-checked TIFF orientation; shared by JPEG APP1 and PNG eXIf.
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

unsigned JpegOrientation(std::span<const std::uint8_t> bytes) {
  for (std::size_t offset = 2; offset < bytes.size();) {
    if (bytes[offset] != 0xff)
      break;
    // JPEG permits repeated FF fill bytes before a marker.
    while (offset < bytes.size() && bytes[offset] == 0xff)
      ++offset;
    if (offset == bytes.size())
      break;
    const auto marker = bytes[offset++];
    if (marker == 0xda || marker == 0xd9)
      break;
    if (marker == 0x01 || marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7))
      continue;
    if (bytes.size() - offset < 2)
      break;
    const std::size_t length =
        (unsigned{bytes[offset]} << 8) | bytes[offset + 1];
    if (length < 2 || length > bytes.size() - offset)
      break;
    const auto data = bytes.subspan(offset + 2, length - 2);
    if (marker == 0xe1 && data.size() >= 14 &&
        std::memcmp(data.data(), "Exif\0\0", 6) == 0) {
      return TiffOrientation(data.subspan(6));
    }
    offset += length;
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
  if (bytes.empty() || bytes.size() > kMaxEncodedImageBytes) {
    throw std::invalid_argument("image exceeds encoded byte limit or is empty");
  }
  return Orient(DecodeJpeg(bytes), JpegOrientation(bytes));
}

}  // namespace gufo::core::libjpeg_reference
