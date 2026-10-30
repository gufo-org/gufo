#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_FORMAT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_FORMAT_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

/// The 8-bit K/V cache. A row (one position) of `width` = kv_heads * d values
/// holds `width` int8 codes followed by width / kKvBlock F16 scales, one per
/// 16 consecutive values. Writers round each value x to F16 first; a block's
/// scale is f16(max |x| * (1 / 127.0f)), one F32 product rounded to F16, and
/// its codes round(x / scale), ties to even, clamped to +-127 (all zero when
/// the scale is zero). Readers see f16(scale * code): every attention kernel
/// consumes exactly those F16 values.
inline constexpr std::uint32_t kKvBlock = 16;
[[nodiscard]] constexpr std::size_t KvRowBytes(std::uint32_t width) noexcept {
  return std::size_t{width} + std::size_t{width} / kKvBlock * 2;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_FORMAT_HPP_
