#ifndef GUFO_SCHED_PREFILL_RATE_HPP_
#define GUFO_SCHED_PREFILL_RATE_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>

namespace gufo::sched {

// One estimator per scheduler/model pool. Only the scheduler worker uses it.
// Scheduling needs the cost of prefill work now, not a lifetime average
// dominated by an old warmup or by tiny cached continuations.
class PrefillRate {
public:
  using Milliseconds = std::chrono::duration<double, std::milli>;

  void Observe(std::size_t tokens, double milliseconds) {
    if (tokens < 256 || !std::isfinite(milliseconds) || milliseconds < 1)
      return;
    samples_[next_] = {static_cast<double>(tokens), milliseconds};
    next_ = (next_ + 1) % samples_.size();
  }

  /// Time to prefill `tokens` at the recent rate; empty until calibrated.
  [[nodiscard]] std::optional<Milliseconds> Estimate(std::size_t tokens) const {
    double observed_tokens = 0, milliseconds = 0;
    for (const auto& sample : samples_) {
      observed_tokens += sample.tokens;
      milliseconds += sample.milliseconds;
    }
    if (observed_tokens == 0)
      return std::nullopt;
    return Milliseconds(static_cast<double>(tokens) *
                        (milliseconds / observed_tokens));
  }

  /// Records a pass of a few tokens that started `depth` tokens into its
  /// context. Its tokens cost little; it reads the context's attention rows
  /// for them, so its time follows the depth, which the per-token rate above
  /// does not see: seven tokens at 76K took a dense model 351 ms. Longer
  /// passes are not samples: their tokens' cost grows with the depth too.
  void ObserveContext(std::size_t tokens, std::size_t depth,
                      double milliseconds) {
    if (tokens == 0 || tokens > kContextTokens ||
        !std::isfinite(milliseconds) || milliseconds <= 0)
      return;
    context_[context_next_] = {static_cast<double>(depth), milliseconds};
    context_next_ = (context_next_ + 1) % context_.size();
    context_count_ = std::min(context_count_ + 1, context_.size());
  }

  /// Time any pass spends on a context `depth` tokens deep before its own
  /// tokens count: a fixed part plus a part per token of depth. Empty until
  /// a few-token pass was measured. It is the median line through the recent
  /// ones (Theil-Sen), which a stalled pass cannot tilt. Passes at one depth
  /// give no slope: their median then holds at that depth and deeper and
  /// shrinks with the depth below it, so no estimate exceeds what was
  /// measured without a slope to support it.
  [[nodiscard]] std::optional<Milliseconds> EstimateContext(
      std::size_t depth) const {
    if (context_count_ == 0)
      return std::nullopt;
    std::array<double, kContextSamples> rest{};
    std::array<double, kContextSamples*(kContextSamples - 1) / 2> slopes{};
    std::size_t pairs = 0;
    double deepest = 0;
    for (std::size_t i = 0; i < context_count_; ++i) {
      rest[i] = context_[i].milliseconds;
      deepest = std::max(deepest, context_[i].depth);
      for (std::size_t j = i + 1; j < context_count_; ++j)
        if (const double apart = context_[i].depth - context_[j].depth;
            std::abs(apart) >= kSlopeDepth)
          slopes[pairs++] =
              (context_[i].milliseconds - context_[j].milliseconds) / apart;
    }
    const auto median = [](auto first, auto last) {
      const auto middle = first + (last - first) / 2;
      std::nth_element(first, middle, last);
      return *middle;
    };
    const double at = static_cast<double>(depth);
    if (pairs == 0) {
      const double measured =
          median(rest.begin(), rest.begin() + context_count_);
      return Milliseconds(at < deepest ? measured * at / deepest : measured);
    }
    const double slope =
        std::max(0.0, median(slopes.begin(), slopes.begin() + pairs));
    for (std::size_t i = 0; i < context_count_; ++i)
      rest[i] = std::max(0.0, rest[i] - slope * context_[i].depth);
    return Milliseconds(median(rest.begin(), rest.begin() + context_count_) +
                        slope * at);
  }

private:
  struct Sample {
    double tokens{}, milliseconds{};
  };
  struct ContextSample {
    double depth{}, milliseconds{};
  };
  static constexpr std::size_t kContextSamples = 16;
  /// The most tokens of a pass that samples its context.
  static constexpr std::size_t kContextTokens = 16;
  /// Two passes this far apart in depth give a slope.
  static constexpr double kSlopeDepth = 4096;
  std::array<Sample, 8> samples_{};
  std::size_t next_{};
  std::array<ContextSample, kContextSamples> context_{};
  std::size_t context_next_{}, context_count_{};
};

}  // namespace gufo::sched

#endif  // GUFO_SCHED_PREFILL_RATE_HPP_
