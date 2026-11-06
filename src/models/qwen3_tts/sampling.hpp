#ifndef GUFO_MODELS_QWEN3_TTS_SAMPLING_HPP_
#define GUFO_MODELS_QWEN3_TTS_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "src/ext/tts.hpp"

namespace gufo::models::qwen3_tts {

using SamplingOptions = ext::TtsSamplingOptions;

struct TokenScore {
  std::uint32_t token;
  float score;
};

[[nodiscard]] bool ValidSamplingOptions(const SamplingOptions& options);

/// Applies temperature, top-k (including boundary ties), then nucleus
/// filtering. top_k=0 disables top-k. A null RNG selects the greedy token.
/// Rejects nonfinite eligible logits instead of returning arbitrary audio.
std::uint32_t SampleCodec(std::vector<TokenScore>& candidates,
                          std::size_t top_k, float top_p, float temperature,
                          std::mt19937* random);

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_SAMPLING_HPP_
