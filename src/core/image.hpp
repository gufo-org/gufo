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
[[nodiscard]] std::vector<std::uint8_t> ReadImageFile(
    const std::filesystem::path& path);
/// Shared across all messages of one HTTP request.
struct ImageReadBudget {
  std::size_t remaining_bytes{kMaxEncodedImageBytes};
  std::size_t remaining_images{16};
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

}  // namespace gufo::core
#endif
