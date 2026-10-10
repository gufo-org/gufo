#ifndef GUFO_MODELS_QWEN_IMAGE_21_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen_image_21 {

// Diffusion denoising trajectories are not token-prefix continuations.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::qwen_image_21

#endif  // GUFO_MODELS_QWEN_IMAGE_21_CACHE_CAPABILITIES_HPP_
