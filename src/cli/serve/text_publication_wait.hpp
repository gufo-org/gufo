#ifndef GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_
#define GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "src/sched/prefill_rate.hpp"

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

// Admission waits for a pending publication at most as long as recomputing
// its gain would take.
class PublicationWaitEstimate : public sched::PrefillRate {
public:
  static constexpr auto kMaximumWait = std::chrono::seconds(120);

  [[nodiscard]] std::optional<Milliseconds> Limit(std::size_t gain) const {
    const auto cost = Estimate(gain);
    if (!cost)
      return std::nullopt;
    return std::min(Milliseconds(kMaximumWait), *cost);
  }
};

}  // namespace detail
}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_PUBLICATION_WAIT_HPP_
