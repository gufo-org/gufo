#include "src/core/png.hpp"

#include <gufo/png.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "src/core/rust/callback.hpp"

namespace gufo::core {
namespace {

void CheckPng(GufoPngStatus status) {
  if (status == GUFO_PNG_INVALID)
    throw std::invalid_argument("invalid PNG image");
  if (status != GUFO_PNG_OK)
    throw std::runtime_error("PNG decoder failed");
}

struct PngDecoderClose {
  void operator()(GufoPngDecoder* decoder) const { gufo_png_close(decoder); }
};

// Returns the decoded pixels of width * height * channels bytes.
std::vector<std::uint8_t> DecodePng(std::span<const std::uint8_t> bytes,
                                    std::uint32_t format, std::size_t channels,
                                    std::uint32_t& width,
                                    std::uint32_t& height) {
  GufoPngDecoder* opened = nullptr;
  GufoPngDimensions size{};
  const GufoPngStatus status =
      gufo_png_open(bytes.data(), bytes.size(), format, &opened, &size);
  const std::unique_ptr<GufoPngDecoder, PngDecoderClose> decoder(opened);
  CheckPng(status);
  if (size.width == 0 || size.height == 0 ||
      std::uint64_t{size.width} * size.height > kMaxImagePixels) {
    throw std::invalid_argument(
        "image dimensions exceed the decoded pixel limit");
  }
  width = size.width;
  height = size.height;
  std::vector<std::uint8_t> pixels(std::size_t{width} * height * channels);
  CheckPng(gufo_png_decode(decoder.get(), pixels.data(), pixels.size()));
  return pixels;
}

}  // namespace

bool IsPng(std::span<const std::uint8_t> bytes) noexcept {
  constexpr std::array<std::uint8_t, 8> kSignature{0x89, 'P',  'N',  'G',
                                                   '\r', '\n', 0x1a, '\n'};
  return bytes.size() >= kSignature.size() &&
         std::memcmp(bytes.data(), kSignature.data(), kSignature.size()) == 0;
}

Image DecodePngRgb(std::span<const std::uint8_t> bytes) {
  Image image;
  image.pixels = DecodePng(bytes, GUFO_PNG_RGB8, 3, image.width, image.height);
  return image;
}

RgbaImage DecodeImageRgba(std::span<const std::uint8_t> bytes) {
  RgbaImage image;
  if (IsPng(bytes)) {
    if (bytes.size() > kMaxEncodedImageBytes)
      throw std::invalid_argument("image exceeds encoded byte limit");
    image.rgba = DecodePng(bytes, GUFO_PNG_RGBA8, 4, image.width, image.height);
    return image;
  }
  const auto rgb = DecodeImage(bytes);
  image.width = rgb.width;
  image.height = rgb.height;
  image.rgba.resize(std::size_t{rgb.width} * rgb.height * 4);
  for (std::size_t i = 0; i < image.rgba.size() / 4; ++i) {
    std::memcpy(image.rgba.data() + 4 * i, rgb.pixels.data() + 3 * i, 3);
    image.rgba[4 * i + 3] = 255;
  }
  return image;
}

std::vector<std::uint8_t> EncodePngRgba(std::uint32_t width,
                                        std::uint32_t height,
                                        std::span<const std::uint8_t> rgba) {
  struct Output {
    std::vector<std::uint8_t> bytes;
    ffi::CallbackFailure failure;
  } out;
  const GufoPngStatus status = gufo_png_encode_rgba8(
      rgba.data(), rgba.size(), width, height,
      [](void* context, const std::uint8_t* data, std::size_t len) noexcept {
        auto& out = *static_cast<Output*>(context);
        out.failure.Guard([&] { out.bytes.assign(data, data + len); });
      },
      &out);
  out.failure.Rethrow();
  if (status != GUFO_PNG_OK)
    throw std::runtime_error("cannot encode output PNG");
  return std::move(out.bytes);
}

}  // namespace gufo::core
