#ifndef GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_
#define GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_

// Internal to the sampler: the full-vocabulary row queries behind top-k and
// top-p. Include only from sampling.cpp and its harnesses.

#if defined(__GNUC__) && defined(__x86_64__)
#include <immintrin.h>
#endif

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

#if defined(__GNUC__) && defined(__x86_64__)
// AVX-512 path for the top-k skip scan. The scalar loop drops NaN values
// (`value > maximum ? value : maximum` keeps `maximum` when `value` is NaN),
// so the vector form keeps the value in the first operand and the
// accumulator second: _mm512_max_ps returns the accumulator whenever the
// value is NaN, and accumulators here never hold NaN.
__attribute__((target("avx512f"))) inline float BlockMaximumAvx512(
    const float* p, std::size_t n) noexcept {
  const __m512 neg_inf =
      _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __m512 a0 = neg_inf;
  __m512 a1 = neg_inf;
  __m512 a2 = neg_inf;
  __m512 a3 = neg_inf;
  std::size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    a0 = _mm512_max_ps(_mm512_loadu_ps(p + i), a0);
    a1 = _mm512_max_ps(_mm512_loadu_ps(p + i + 16), a1);
    a2 = _mm512_max_ps(_mm512_loadu_ps(p + i + 32), a2);
    a3 = _mm512_max_ps(_mm512_loadu_ps(p + i + 48), a3);
  }
  a0 = _mm512_max_ps(a1, a0);
  a0 = _mm512_max_ps(a2, a0);
  a0 = _mm512_max_ps(a3, a0);
  for (; i + 16 <= n; i += 16)
    a0 = _mm512_max_ps(_mm512_loadu_ps(p + i), a0);
  if (i < n) {
    const __mmask16 tail = static_cast<__mmask16>((1U << (n - i)) - 1U);
    a0 = _mm512_mask_max_ps(a0, tail, _mm512_maskz_loadu_ps(tail, p + i), a0);
  }
  return _mm512_reduce_max_ps(a0);
}

#endif

/// Largest value in a block, ignoring NaN; +inf is returned as is.
[[nodiscard]] inline float BlockMaximum(std::span<const float> block) noexcept {
#if defined(__GNUC__) && defined(__x86_64__)
  static const bool has_avx512f = __builtin_cpu_supports("avx512f");
  if (has_avx512f)
    return BlockMaximumAvx512(block.data(), block.size());
#endif
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
