#ifndef GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen {

// Served through the continuation cache. The adapter (cache redesign cards
// 15-16) returns this record and decides persistent encoding when it lands.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = true};

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_
