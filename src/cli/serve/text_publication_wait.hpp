#ifndef GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_
#define GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace gufo::server {

struct TextPendingPublication {
  static constexpr std::size_t kMinimumGainTokens = 4096;
  std::uint64_t checkpoint_id{};
  std::size_t available_tokens{};
  std::size_t pending_tokens{};

  [[nodiscard]] std::size_t Gain() const {
    return pending_tokens - std::min(pending_tokens, available_tokens);
  }
};

namespace detail {

// One estimator per scheduler/model pool. Only the scheduler worker uses it.
// Admission needs an estimate of useful saved work, not a lifetime average
// dominated by an old warmup or by tiny cached continuations.
class PublicationWaitEstimate {
public:
  using Milliseconds = std::chrono::duration<double, std::milli>;
  static constexpr auto kMaximumWait = std::chrono::seconds(120);

  void Observe(std::size_t tokens, double milliseconds) {
    if (tokens < 256 || !std::isfinite(milliseconds) || milliseconds < 1)
      return;
    samples_[next_] = {static_cast<double>(tokens), milliseconds};
    next_ = (next_ + 1) % samples_.size();
  }

  [[nodiscard]] std::optional<Milliseconds> Limit(std::size_t gain) const {
    double tokens = 0, milliseconds = 0;
    for (const auto& sample : samples_) {
      tokens += sample.tokens;
      milliseconds += sample.milliseconds;
    }
    if (tokens == 0)
      return std::nullopt;
    const auto cost = static_cast<double>(gain) * (milliseconds / tokens);
    return Milliseconds(std::min(Milliseconds(kMaximumWait).count(), cost));
  }

private:
  struct Sample {
    double tokens{}, milliseconds{};
  };
  std::array<Sample, 8> samples_{};
  std::size_t next_{};
};

}  // namespace detail
}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_
