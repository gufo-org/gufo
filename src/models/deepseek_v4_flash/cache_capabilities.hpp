#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::deepseek_v4_flash {

// Served through the continuation cache. The adapter (cache redesign card 17)
// returns this record and decides persistent encoding when it lands.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = true};

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_
