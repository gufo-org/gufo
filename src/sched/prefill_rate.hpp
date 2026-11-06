#ifndef GUFO_SCHED_PREFILL_RATE_HPP_
#define GUFO_SCHED_PREFILL_RATE_HPP_

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

private:
  struct Sample {
    double tokens{}, milliseconds{};
  };
  std::array<Sample, 8> samples_{};
  std::size_t next_{};
};

}  // namespace gufo::sched

#endif  // GUFO_SCHED_PREFILL_RATE_HPP_
