#ifndef GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_
#define GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_

// Internal to the sampler: the full-vocabulary row queries behind top-k and
// top-p. Include only from sampling.cpp and its harnesses.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "src/core/sampling.hpp"

namespace gufo::sampling {

constexpr std::size_t kSkipBlock = 64;

/// Largest value in a block, ignoring NaN; +inf is returned as is.
[[nodiscard]] inline float BlockMaximum(std::span<const float> block) noexcept {
  if (block.size() == kSkipBlock) {
    // A fixed trip count with independent lanes lets the compiler vectorize.
    std::array<float, 8> lanes;
    lanes.fill(-std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < kSkipBlock; i += lanes.size())
      for (std::size_t lane = 0; lane < lanes.size(); ++lane)
        lanes[lane] =
            block[i + lane] > lanes[lane] ? block[i + lane] : lanes[lane];
    float maximum = lanes[0];
    for (const float lane : lanes)
      maximum = lane > maximum ? lane : maximum;
    return maximum;
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (const float value : block)
    maximum = value > maximum ? value : maximum;
  return maximum;
}

[[nodiscard]] inline bool IsBetterProbability(
    const Probability& left, const Probability& right) noexcept {
  if (left.value == right.value) {
    return left.token < right.token;
  }
  return left.value > right.value;
}

/// The adjusted logit of a token with sparse penalty state. Defined once, in
/// sampling.cpp, so every sampling path rounds identically.
[[nodiscard]] double Penalize(double logit, const SamplingConfig& config,
                              const TokenPenalty& penalty);

// Adds penalties to logits lazily and answers the three questions the
// top-p/top-k samplers ask of a full vocabulary row: its maximum, its best
// candidates, and its softmax mass. Per-block maxima let each query skip the
// blocks that cannot affect the answer.
class LogitRow {
public:
  struct Interval {
    double low;
    double high;
  };

  /// `scratch` outlives the row and is reused across rows.
  LogitRow(std::span<const float> logits, const SamplingConfig& config,
           std::span<const TokenPenalty> penalties, LogitRowScratch* scratch);

  [[nodiscard]] std::size_t size() const noexcept { return logits_.size(); }

  /// The best min(limit, finite count) candidates, in IsBetterProbability
  /// order when `sorted` and in no particular order otherwise.
  void Collect(std::size_t limit, std::vector<Probability>* out,
               bool sorted = true) const;

  /// Sum of exp((logit - maximum) / T) over finite logits in index order.
  [[nodiscard]] double ExactSoftmaxSum() const;

  /// A range containing ExactSoftmaxSum(), from vectorized float
  /// approximations. Empty when the row or temperature is outside the range
  /// the error bounds cover.
  [[nodiscard]] std::optional<Interval> ApproxSoftmaxSum() const;

private:
  using PenaltyIt = std::span<const TokenPenalty>::iterator;

  [[nodiscard]] double Adjusted(std::size_t index, PenaltyIt& penalty) const;
  [[nodiscard]] double HistogramThreshold(std::size_t limit) const;
  void Scan();

  std::span<const float> logits_;
  const SamplingConfig& config_;
  std::span<const TokenPenalty> penalties_;
  LogitRowScratch* scratch_;
  std::vector<double> block_max_;
  std::vector<std::uint8_t> scalar_block_;
  std::size_t finite_blocks_{0};
  double maximum_{-std::numeric_limits<double>::infinity()};
};

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_
