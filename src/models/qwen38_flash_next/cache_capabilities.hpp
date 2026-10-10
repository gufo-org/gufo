#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen38_flash_next {

// Served through the continuation cache. The adapter (cache redesign card 14)
// returns this record and decides persistent encoding when it lands.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = true};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_CACHE_CAPABILITIES_HPP_
