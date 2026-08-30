#ifndef GUFO_CORE_SAMPLING_HPP_
#define GUFO_CORE_SAMPLING_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace gufo::sampling {

using TokenId = std::uint32_t;

/// Model-independent controls for selecting a token from a logit row.
///
/// Filters run in this order: penalties, top-k, top-p, min-p, temperature.
/// The defaults preserve Gufo's existing greedy decoding behavior.
struct SamplingConfig {
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  float min_p{0.0F};
  std::size_t min_keep{0};
  std::int64_t seed{-1};
  float repeat_penalty{1.0F};
  std::size_t repeat_last_n{64};
  float frequency_penalty{0.0F};
  float presence_penalty{0.0F};

  void Validate() const;

  [[nodiscard]] bool penalties_enabled() const noexcept;
  [[nodiscard]] bool uses_random_sampling() const noexcept;
  [[nodiscard]] bool can_use_unmodified_argmax() const noexcept;
};

struct Probability {
  TokenId token{0};
  double value{0.0};
};

/// A normalized, post-filter sampling distribution.
class SamplingDistribution {
public:
  SamplingDistribution() = default;
  explicit SamplingDistribution(std::vector<Probability> entries);

  [[nodiscard]] std::span<const Probability> entries() const noexcept;
  [[nodiscard]] TokenId best_token() const;
  [[nodiscard]] double probability(TokenId token) const noexcept;
  [[nodiscard]] TokenId Sample(std::uint64_t* rng_state) const;
  [[nodiscard]] TokenId SampleResidual(
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities,
      std::uint64_t* rng_state) const;

private:
  std::vector<Probability> entries_;
};

/// Builds the exact distribution used for both normal and speculative decode.
[[nodiscard]] SamplingDistribution BuildDistribution(
    std::span<const float> logits, const SamplingConfig& config,
    std::span<const TokenId> recent_tokens = {});

/// Request-owned sampling state: controls, bounded token history, and RNG.
///
/// Call Accept only for committed tokens. The type is copyable so speculative
/// decode can work on a tentative state and discard it on rollback.
class SamplerState {
public:
  explicit SamplerState(SamplingConfig config = {},
                        std::span<const TokenId> initial_history = {});
  ~SamplerState() = default;
  SamplerState(const SamplerState& other);
  SamplerState& operator=(const SamplerState& other);
  SamplerState(SamplerState&&) noexcept = default;
  SamplerState& operator=(SamplerState&&) noexcept = default;

  [[nodiscard]] const SamplingConfig& config() const noexcept;
  [[nodiscard]] std::span<const TokenId> history() const noexcept;
  [[nodiscard]] std::uint64_t rng_state() const noexcept;
  [[nodiscard]] std::uint64_t* mutable_rng_state() noexcept;
  void SetRngState(std::uint64_t state) noexcept;

  void ResetHistory(std::span<const TokenId> tokens);
  void Accept(TokenId token);
  void Accept(std::span<const TokenId> tokens);

  [[nodiscard]] SamplingDistribution Distribution(
      std::span<const float> logits) const;
  [[nodiscard]] TokenId Sample(std::span<const float> logits);
  [[nodiscard]] TokenId SampleResidual(
      std::span<const float> target_logits,
      std::span<const TokenId> candidate_ids,
      std::span<const float> candidate_probabilities);
  [[nodiscard]] double Uniform();

private:
  void TrimHistory();
  void RebuildPenaltyCounts();
  [[nodiscard]] double AdjustedLogit(TokenId token, float logit) const noexcept;
  [[nodiscard]] TokenId SampleGreedy(std::span<const float> logits) const;
  [[nodiscard]] TokenId SampleLinear(std::span<const float> logits);
  [[nodiscard]] TokenId SampleSelected(std::span<const float> logits);

  SamplingConfig config_;
  std::vector<TokenId> history_;
  std::vector<std::pair<TokenId, std::uint32_t>> penalty_counts_;
  std::vector<Probability> candidate_scratch_;
  std::uint64_t rng_state_{0};
};

[[nodiscard]] std::uint64_t NextRandom(std::uint64_t* state);
[[nodiscard]] double Uniform(std::uint64_t* state);

// Compatibility helpers for callers that have not yet adopted SamplerState.
[[nodiscard]] TokenId SampleLogits(std::span<const float> logits,
                                   float temperature, std::uint64_t* rng_state);
[[nodiscard]] double TokenProbability(std::span<const float> logits,
                                      float temperature, TokenId token);
[[nodiscard]] TokenId SampleResidual(
    std::span<const float> target_logits, float temperature,
    std::span<const TokenId> candidate_ids,
    std::span<const float> candidate_probabilities, std::uint64_t* rng_state);

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_HPP_
