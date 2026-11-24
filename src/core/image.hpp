#ifndef GUFO_CORE_IMAGE_HPP_
#define GUFO_CORE_IMAGE_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

inline constexpr std::size_t kMaxEncodedImageBytes = 20 * 1024 * 1024;
inline constexpr std::size_t kMaxImagePixels = 32 * 1024 * 1024;

struct Image {
  std::uint32_t width{0};
  std::uint32_t height{0};
  /// Packed, unpremultiplied RGB8, in display orientation.
  std::vector<std::uint8_t> pixels;
};

/// PNG, JPEG and WebP. Decoding checks dimensions before allocating pixels.
[[nodiscard]] Image DecodeImage(std::span<const std::uint8_t> bytes);

struct RgbaImage {
  std::uint32_t width{0};
  std::uint32_t height{0};
  /// Packed, unpremultiplied RGBA8.
  std::vector<std::uint8_t> rgba;
};

/// PNG as libpng's simplified API decodes it to 8-bit RGBA: tRNS becomes
/// alpha, 16-bit samples round to 8 bits, color samples convert from the file
/// gamma to sRGB, and EXIF orientation is ignored. Interlaced 16-bit PNGs
/// decode correctly, unlike libpng 1.6.59. JPEG and WebP decode as
/// DecodeImage does, with opaque alpha. Checks kMaxEncodedImageBytes and
/// kMaxImagePixels; invalid input throws std::invalid_argument.
[[nodiscard]] RgbaImage DecodeImageRgba(std::span<const std::uint8_t> bytes);
/// An 8-bit RGBA PNG with an sRGB chunk, for width * height pixels in rgba,
/// each dimension at most 1,000,000. Throws std::runtime_error otherwise.
[[nodiscard]] std::vector<std::uint8_t> EncodePngRgba(
    std::uint32_t width, std::uint32_t height,
    std::span<const std::uint8_t> rgba);
[[nodiscard]] std::vector<std::uint8_t> ReadImageFile(
    const std::filesystem::path& path);
/// Shared across all messages of one HTTP request.
struct ImageReadBudget {
  /// False when the loaded model takes no image input; reading is refused
  /// before any data URL is decoded or any URL is fetched.
  bool accepts_images{true};
  std::size_t remaining_bytes{kMaxEncodedImageBytes};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::now() + std::chrono::seconds(15)};
};

/// Network-order IPv4/IPv6 address; excludes local and special-use ranges.
[[nodiscard]] bool IsPublicImageAddress(
    std::span<const std::uint8_t> address) noexcept;
/// Base64 data URLs or public HTTPS destinations. DNS results and every
/// redirect's actual socket address are checked before connecting.
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url,
                                                     ImageReadBudget& budget);
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url);
/// Audio uses the same bounded reader and destination checks. Supports raw
/// base64, audio data URLs and public HTTPS. The decoder checks the bytes.
[[nodiscard]] std::vector<std::uint8_t> ReadAudioUrl(std::string_view url,
                                                     ImageReadBudget& budget);

/// Video uses the same shared byte/deadline and public HTTPS protections.
[[nodiscard]] std::vector<std::uint8_t> ReadVideoUrl(std::string_view url,
                                                     ImageReadBudget& budget);

}  // namespace gufo::core
#endif
