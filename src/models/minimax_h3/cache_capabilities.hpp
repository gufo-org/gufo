#ifndef GUFO_MODELS_MINIMAX_H3_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_MINIMAX_H3_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::minimax_h3 {

// Diffusion denoising trajectories are not token-prefix continuations.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::minimax_h3

#endif  // GUFO_MODELS_MINIMAX_H3_CACHE_CAPABILITIES_HPP_
