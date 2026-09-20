#ifndef GUFO_MODELS_QWEN3_TTS_SAMPLING_HPP_
#define GUFO_MODELS_QWEN3_TTS_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace gufo::models::qwen3_tts {

struct SamplingOptions {
  bool sample{true};
  std::uint32_t seed{42};
  std::size_t top_k{50};
  float temperature{0.9F};
  float top_p{1.0F};
  bool predictor_sample{true};
  std::size_t predictor_top_k{50};
  float predictor_temperature{0.9F};
  float predictor_top_p{1.0F};
  float repetition_penalty{1.05F};
};

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
