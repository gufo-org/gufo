#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace gufo::sampling {

inline std::uint64_t NextRandom(std::uint64_t* state) {
  if (state == nullptr) {
    throw std::invalid_argument("sampling requires RNG state");
  }
  std::uint64_t value = *state;
  if (value == 0) {
    value = UINT64_C(0x9e3779b97f4a7c15);
  }
  value ^= value >> 12U;
  value ^= value << 25U;
  value ^= value >> 27U;
  *state = value;
  return value * UINT64_C(0x2545f4914f6cdd1d);
}

inline double Uniform(std::uint64_t* state) {
  const std::uint64_t value = NextRandom(state);
  return static_cast<double>((value >> 40U) & UINT64_C(0xffffff)) / 16777216.0;
}

namespace detail {

struct SoftmaxNormalizer {
  double maximum{-std::numeric_limits<double>::infinity()};
  double weight_sum{0.0};
  std::uint32_t best_token{0};
};

[[nodiscard]] inline SoftmaxNormalizer Normalize(std::span<const float> logits,
                                                 float temperature) {
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (logits.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument("logit distribution exceeds token ID range");
  }
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "softmax temperature must be finite and positive");
  }

  SoftmaxNormalizer normalizer;
  bool has_finite_logit = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    const double value = static_cast<double>(logits[index]);
    if (!std::isfinite(value)) {
      continue;
    }
    if (!has_finite_logit || value > normalizer.maximum) {
      normalizer.maximum = value;
      normalizer.best_token = static_cast<std::uint32_t>(index);
      has_finite_logit = true;
    }
  }
  if (!has_finite_logit) {
    throw std::runtime_error("logit distribution contains no finite values");
  }

  const double inverse_temperature = 1.0 / static_cast<double>(temperature);
  for (const float value : logits) {
    if (std::isfinite(value)) {
      normalizer.weight_sum +=
          std::exp((static_cast<double>(value) - normalizer.maximum) *
                   inverse_temperature);
    }
  }
  if (!(normalizer.weight_sum > 0.0) || !std::isfinite(normalizer.weight_sum)) {
    throw std::runtime_error("logit softmax normalization failed");
  }
  return normalizer;
}

[[nodiscard]] inline double Weight(float logit, float temperature,
                                   double maximum) noexcept {
  if (!std::isfinite(logit)) {
    return 0.0;
  }
  return std::exp((static_cast<double>(logit) - maximum) /
                  static_cast<double>(temperature));
}

[[nodiscard]] inline double SparseProbability(
    std::uint32_t token, std::span<const std::uint32_t> candidate_ids,
    std::span<const float> candidate_probabilities) {
  if (candidate_ids.size() != candidate_probabilities.size()) {
    throw std::invalid_argument("sparse probability row is malformed");
  }
  double probability = 0.0;
  for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
    const float value = candidate_probabilities[index];
    if (!std::isfinite(value) || value < 0.0F) {
      throw std::invalid_argument(
          "sparse probability row contains an invalid value");
    }
    if (candidate_ids[index] == token) {
      probability += static_cast<double>(value);
    }
  }
  return probability;
}

}  // namespace detail

/// Selects one token from a full-vocabulary logit distribution.
///
/// Temperature zero is exact argmax and does not consume RNG state. Positive
/// temperature uses a numerically stable full-vocabulary softmax.
[[nodiscard]] inline std::uint32_t SampleLogits(std::span<const float> logits,
                                                float temperature,
                                                std::uint64_t* rng_state) {
  if (logits.empty()) {
    throw std::invalid_argument("cannot sample an empty logit distribution");
  }
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "sampling temperature must be finite and nonnegative");
  }

  std::uint32_t best_token = 0;
  float best_logit = -std::numeric_limits<float>::infinity();
  bool has_finite_logit = false;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    const float value = logits[index];
    if (!std::isfinite(value)) {
      continue;
    }
    if (!has_finite_logit || value > best_logit) {
      best_logit = value;
      best_token = static_cast<std::uint32_t>(index);
      has_finite_logit = true;
    }
  }
  if (!has_finite_logit) {
    throw std::runtime_error("logit distribution contains no finite values");
  }
  if (temperature == 0.0F) {
    return best_token;
  }

  const auto normalizer = detail::Normalize(logits, temperature);
  double sample = Uniform(rng_state) * normalizer.weight_sum;
  std::uint32_t last_finite_token = normalizer.best_token;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      continue;
    }
    last_finite_token = static_cast<std::uint32_t>(index);
    sample -= detail::Weight(logits[index], temperature, normalizer.maximum);
    if (sample <= 0.0) {
      return last_finite_token;
    }
  }
  return last_finite_token;
}

[[nodiscard]] inline double TokenProbability(std::span<const float> logits,
                                             float temperature,
                                             std::uint32_t token) {
  if (token >= logits.size()) {
    throw std::out_of_range("sampled token exceeds target vocabulary");
  }
  const auto normalizer = detail::Normalize(logits, temperature);
  return detail::Weight(logits[token], temperature, normalizer.maximum) /
         normalizer.weight_sum;
}

/// Samples the normalized positive part of p_target - q_draft.
[[nodiscard]] inline std::uint32_t SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const std::uint32_t> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state) {
  const auto normalizer = detail::Normalize(target_logits, temperature);
  double residual_sum = 0.0;
  for (std::size_t token = 0; token < target_logits.size(); ++token) {
    const double target_probability =
        detail::Weight(target_logits[token], temperature, normalizer.maximum) /
        normalizer.weight_sum;
    const double draft_probability =
        detail::SparseProbability(static_cast<std::uint32_t>(token),
                                  candidate_ids, candidate_probabilities);
    residual_sum += std::max(target_probability - draft_probability, 0.0);
  }
  if (!(residual_sum > 0.0) || !std::isfinite(residual_sum)) {
    return SampleLogits(target_logits, temperature, rng_state);
  }

  double sample = Uniform(rng_state) * residual_sum;
  std::uint32_t last_positive_token = normalizer.best_token;
  for (std::size_t token = 0; token < target_logits.size(); ++token) {
    const double target_probability =
        detail::Weight(target_logits[token], temperature, normalizer.maximum) /
        normalizer.weight_sum;
    const double draft_probability =
        detail::SparseProbability(static_cast<std::uint32_t>(token),
                                  candidate_ids, candidate_probabilities);
    const double residual =
        std::max(target_probability - draft_probability, 0.0);
    if (residual <= 0.0) {
      continue;
    }
    last_positive_token = static_cast<std::uint32_t>(token);
    sample -= residual;
    if (sample <= 0.0) {
      return last_positive_token;
    }
  }
  return last_positive_token;
}

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
