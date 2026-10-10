#ifndef GUFO_MODELS_QWEN3_ASR_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN3_ASR_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen3_asr {

// Every request prefills its own prompt and audio embeddings. Live WebSocket
// sessions transcribe each committed buffer as an independent request, so no
// request resumes another's decoder state. Reusing session context would be a
// separate capability, not a token-prefix continuation.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_CACHE_CAPABILITIES_HPP_
