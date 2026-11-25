#ifndef GUFO_CORE_PNG_HPP_
#define GUFO_CORE_PNG_HPP_

#include <cstdint>
#include <span>

#include "src/core/image.hpp"

namespace gufo::core {

[[nodiscard]] bool IsPng(std::span<const std::uint8_t> bytes) noexcept;

/// Packed RGB8 as Pillow's convert("RGB"), before EXIF orientation. Checks
/// dimensions against kMaxImagePixels before allocating pixels.
[[nodiscard]] Image DecodePngRgb(std::span<const std::uint8_t> bytes);

}  // namespace gufo::core

#endif
