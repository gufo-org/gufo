#ifndef GUFO_MODELS_QWEN3_TTS_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN3_TTS_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen3_tts {

// Every request prefills its own talker prompt. Streaming WebSocket input is
// split into speech segments that are synthesized as independent requests.
// Reusing a voice-clone speaker embedding or reference prompt would be a
// separate capability, not a token-prefix continuation.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_CACHE_CAPABILITIES_HPP_
