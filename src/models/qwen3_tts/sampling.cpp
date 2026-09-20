#include "src/models/qwen3_tts/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gufo::models::qwen3_tts {

bool ValidSamplingOptions(const SamplingOptions& options) {
  const auto temperature = [](float value) {
    return std::isfinite(value) && value > 0.0F;
  };
  const auto top_p = [](float value) {
    return std::isfinite(value) && value > 0.0F && value <= 1.0F;
  };
  return temperature(options.temperature) &&
         temperature(options.predictor_temperature) &&
         temperature(options.repetition_penalty) && top_p(options.top_p) &&
         top_p(options.predictor_top_p);
}

std::uint32_t SampleCodec(std::vector<TokenScore>& candidates,
                          std::size_t top_k, float top_p, float temperature,
                          std::mt19937* random) {
  if (candidates.empty() || !std::isfinite(temperature) ||
      temperature <= 0.0F || !std::isfinite(top_p) || top_p <= 0.0F ||
      top_p > 1.0F) {
    throw std::invalid_argument("Qwen3-TTS sampler input is invalid");
  }
  for (const auto& candidate : candidates) {
    if (!std::isfinite(candidate.score)) {
      throw std::runtime_error("Qwen3-TTS sampler received nonfinite logits");
    }
  }
  const auto higher = [](const TokenScore& left, const TokenScore& right) {
    return left.score == right.score ? left.token < right.token
                                     : left.score > right.score;
  };
  if (random == nullptr) {
    return std::min_element(candidates.begin(), candidates.end(), higher)
        ->token;
  }
  if (top_k != 0 && top_k < candidates.size()) {
    std::nth_element(candidates.begin(), candidates.begin() + top_k - 1,
                     candidates.end(), higher);
    const float cutoff = candidates[top_k - 1].score;
    std::erase_if(candidates, [cutoff](const TokenScore& token) {
      return token.score < cutoff;
    });
  }
  std::sort(candidates.begin(), candidates.end(), higher);
  const double maximum = candidates.front().score;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    weights.push_back(
        std::exp((static_cast<double>(candidate.score) - maximum) /
                 static_cast<double>(temperature)));
  }
  if (top_p < 1.0F) {
    const double threshold =
        std::accumulate(weights.begin(), weights.end(), 0.0) * top_p;
    double cumulative = 0.0;
    std::size_t count = 0;
    do {
      cumulative += weights[count++];
    } while (count < weights.size() && cumulative < threshold);
    weights.resize(count);
    candidates.resize(count);
  }
  std::discrete_distribution<std::size_t> distribution(weights.begin(),
                                                       weights.end());
  return candidates[distribution(*random)].token;
}

}  // namespace gufo::models::qwen3_tts
