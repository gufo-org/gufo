// jpeglib.h requires size_t and FILE to be declared first.
// clang-format off
#include <cstddef>
#include <cstdio>
#include <jpeglib.h>
// clang-format on
#include <webp/encode.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/image.hpp"

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
using Bytes = std::vector<std::uint8_t>;
using gufo::core::ImageFormat;
using gufo::core::InspectImage;

void Be32(Bytes& bytes, std::size_t at, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes[at++] = static_cast<std::uint8_t>(value >> shift);
}

Bytes PngHeader(std::uint32_t width, std::uint32_t height, unsigned depth = 8,
                unsigned color = 2) {
  Bytes bytes{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n',
              0,    0,   0,   13,  'I',  'H',  'D',  'R'};
  bytes.resize(33);
  Be32(bytes, 16, width);
  Be32(bytes, 20, height);
  bytes[24] = static_cast<std::uint8_t>(depth);
  bytes[25] = static_cast<std::uint8_t>(color);
  return bytes;
}

Bytes JpegHeader(unsigned width, unsigned height, unsigned components = 3) {
  Bytes bytes{0xff,
              0xd8,
              0xff,
              0xc2,
              0,
              static_cast<std::uint8_t>(8 + components * 3),
              8,
              static_cast<std::uint8_t>(height >> 8),
              static_cast<std::uint8_t>(height),
              static_cast<std::uint8_t>(width >> 8),
              static_cast<std::uint8_t>(width),
              static_cast<std::uint8_t>(components)};
  for (unsigned i = 0; i < components; ++i) {
    bytes.push_back(static_cast<std::uint8_t>(i + 1));
    bytes.push_back(0x11);
    bytes.push_back(0);
  }
  bytes.insert(bytes.end(), {0xff, 0xda});
  return bytes;
}

Bytes LosslessWebpHeader(unsigned width, unsigned height) {
  Bytes bytes{'R', 'I', 'F', 'F', 18, 0, 0, 0,    'W', 'E', 'B', 'P', 'V',
              'P', '8', 'L', 6,   0,  0, 0, 0x2f, 0,   0,   0,   0,   0};
  const auto dimensions = (width - 1) | ((height - 1) << 14);
  for (unsigned i = 0; i < 4; ++i)
    bytes[21 + i] = static_cast<std::uint8_t>(dimensions >> (8 * i));
  return bytes;
}

Bytes Exif(unsigned orientation) {
  return {'E', 'x',  'i', 'f', 0,
          0,   'I',  'I', 42,  0,
          8,   0,    0,   0,   1,
          0,   0x12, 1,   3,   0,
          1,   0,    0,   0,   static_cast<std::uint8_t>(orientation),
          0,   0,    0,   0,   0,
          0,   0};
}

Bytes EncodeJpeg(unsigned components, bool progressive, unsigned orientation) {
  jpeg_compress_struct encoder{};
  jpeg_error_mgr errors{};
  encoder.err = jpeg_std_error(&errors);
  jpeg_create_compress(&encoder);
  unsigned char* encoded = nullptr;
  unsigned long length = 0;
  jpeg_mem_dest(&encoder, &encoded, &length);
  encoder.image_width = 17;
  encoder.image_height = 9;
  encoder.input_components = static_cast<int>(components);
  encoder.in_color_space = components == 4 ? JCS_CMYK : JCS_RGB;
  jpeg_set_defaults(&encoder);
  if (progressive)
    jpeg_simple_progression(&encoder);
  jpeg_start_compress(&encoder, TRUE);
  const auto exif = Exif(orientation);
  jpeg_write_marker(&encoder, JPEG_APP0 + 1, exif.data(), exif.size());
  Bytes row(17 * components, 100);
  while (encoder.next_scanline < encoder.image_height) {
    JSAMPROW pixels = row.data();
    jpeg_write_scanlines(&encoder, &pixels, 1);
  }
  jpeg_finish_compress(&encoder);
  Bytes bytes(encoded, encoded + length);
  std::free(encoded);
  jpeg_destroy_compress(&encoder);
  return bytes;
}

Bytes EncodeWebp(bool lossless, bool alpha) {
  Bytes pixels(17 * 9 * (alpha ? 4 : 3), 100);
  std::uint8_t* encoded = nullptr;
  const auto length =
      alpha      ? WebPEncodeRGBA(pixels.data(), 17, 9, 17 * 4, 75, &encoded)
      : lossless ? WebPEncodeLosslessRGB(pixels.data(), 17, 9, 17 * 3, &encoded)
                 : WebPEncodeRGB(pixels.data(), 17, 9, 17 * 3, 75, &encoded);
  assert(length && encoded);
  Bytes bytes(encoded, encoded + length);
  WebPFree(encoded);
  return bytes;
}

void Rejected(std::span<const std::uint8_t> bytes) {
  bool rejected = false;
  try {
    (void)InspectImage(bytes);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void DimensionsAndWorkspace() {
  // Header-only fixtures exercise the real planner without allocating pixels
  // for large geometry. Successful inspection does not claim valid contents.
  const auto small = InspectImage(PngHeader(32, 16));
  const auto large = InspectImage(PngHeader(8192, 4096));
  const auto rgba16 = InspectImage(PngHeader(8192, 4096, 16, 6));
  assert(small.format == ImageFormat::kPng && small.width == 32);
  assert(small.height == 16 && small.original_width == 32);
  assert(small.peak_bytes < large.peak_bytes);
  assert(large.peak_bytes < rgba16.peak_bytes);
  assert(small.peak_bytes < (1ULL << 30));
  assert(rgba16.peak_bytes > 8192ULL * 4096 * 8);

  auto subbyte = PngHeader(8, 8, 1, 0);
  const auto packed = InspectImage(subbyte);
  subbyte[28] = 1;
  assert(InspectImage(subbyte).peak_bytes > packed.peak_bytes);

  const auto rgb = InspectImage(JpegHeader(32, 16));
  const auto cmyk = InspectImage(JpegHeader(32, 16, 4));
  const auto big_jpeg = InspectImage(JpegHeader(8192, 4096));
  assert(rgb.format == ImageFormat::kJpeg);
  assert(rgb.peak_bytes < cmyk.peak_bytes);
  assert(cmyk.peak_bytes < big_jpeg.peak_bytes);
  assert(big_jpeg.peak_bytes > 8192ULL * 4096 * (3 + 3 * 2));
  auto lossless_jpeg = JpegHeader(32, 16);
  lossless_jpeg[3] = 0xc3;
  lossless_jpeg[6] = 7;
  assert(InspectImage(lossless_jpeg).peak_bytes == rgb.peak_bytes);

  const auto small_webp = InspectImage(LosslessWebpHeader(16, 8));
  const auto large_webp = InspectImage(LosslessWebpHeader(8192, 4096));
  assert(small_webp.format == ImageFormat::kWebp);
  assert(small_webp.peak_bytes < large_webp.peak_bytes);
  // Valid dimensions need not imply an affordable decoder. Huffman table
  // storage can dominate a malicious or complex lossless input.
  assert(large_webp.peak_bytes > (1ULL << 30));

  // The valid header walk performs no C++ heap allocation, even when its
  // resulting bound cannot fit in a small runtime budget.
  const auto png = PngHeader(8192, 4096, 16, 6);
  const auto jpeg = JpegHeader(8192, 4096);
  const auto webp = EncodeWebp(true, false);
  allocations = 0;
  count_allocations = true;
  (void)InspectImage(png);
  (void)InspectImage(jpeg);
  (void)InspectImage(webp);
  count_allocations = false;
  assert(allocations == 0);
}

void MalformedAndLimits() {
  Rejected({});
  Rejected(Bytes{1, 2, 3});
  Rejected(PngHeader(0, 1));
  Rejected(PngHeader(8192, 4097));
  Rejected(PngHeader(UINT32_MAX, UINT32_MAX));
  Rejected(PngHeader(10, 10, 16, 3));
  Rejected(PngHeader(10, 10, 3, 2));
  Rejected(JpegHeader(0, 10));
  Rejected(JpegHeader(65535, 65535));
  Rejected(JpegHeader(10, 10, 2));
  Rejected(LosslessWebpHeader(16384, 16384));
  auto jpeg = JpegHeader(10, 10);
  jpeg[4] = 0xff;
  Rejected(jpeg);
  jpeg = JpegHeader(10, 10);
  jpeg[13] = 0xf1;
  Rejected(jpeg);
  auto png = PngHeader(10, 10);
  for (std::size_t length = 0; length < png.size(); ++length)
    Rejected(std::span(png).first(length));
  auto webp = EncodeWebp(false, false);
  webp.resize(20);
  Rejected(webp);
  // Encoded-size rejection precedes any parsing/decoder allocation.
  const Bytes excessive(gufo::core::kMaxEncodedImageBytes + 1, 0);
  Rejected(excessive);
}

void DecoderAgreement() {
  for (const auto components : {3U, 4U}) {
    for (const bool progressive : {false, true}) {
      for (unsigned orientation = 1; orientation <= 8; ++orientation) {
        const auto bytes = EncodeJpeg(components, progressive, orientation);
        const auto plan = InspectImage(bytes);
        const auto image = gufo::core::DecodeImage(bytes);
        assert(plan.original_width == 17 && plan.original_height == 9);
        assert(plan.orientation == orientation);
        assert(plan.width == image.width && plan.height == image.height);
        assert(plan.peak_bytes >=
               image.pixels.size() * (orientation == 1 ? 1 : 2));
      }
    }
  }
  const auto lossy = EncodeWebp(false, false);
  const auto lossless = EncodeWebp(true, false);
  const auto alpha = EncodeWebp(false, true);
  assert(InspectImage(lossless).peak_bytes > InspectImage(lossy).peak_bytes);
  assert(InspectImage(alpha).peak_bytes >= InspectImage(lossless).peak_bytes);
  for (const auto& bytes : {lossy, lossless, alpha}) {
    const auto plan = InspectImage(bytes);
    const auto image = gufo::core::DecodeImage(bytes);
    assert(plan.format == ImageFormat::kWebp);
    assert(plan.original_width == 17 && plan.original_height == 9);
    assert(plan.width == image.width && plan.height == image.height);
    assert(plan.peak_bytes >= image.pixels.size());
  }
}
}  // namespace

// Test-local guard for the inspection path; codec internals are additionally
// audited for their allocation-free header APIs. Decoding is not instrumented.
void* operator new(std::size_t size) {
  if (count_allocations)
    ++allocations;
  if (void* memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

int main() {
  DimensionsAndWorkspace();
  MalformedAndLimits();
  DecoderAgreement();
}
