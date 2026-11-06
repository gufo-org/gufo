#ifndef GUFO_SCHED_POLICY_HPP_
#define GUFO_SCHED_POLICY_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <variant>

namespace gufo::sched {

/// Prompt tokens one prefill work unit may consume while other sessions
/// decode, for runners that have no scheduling profile.
inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;

struct PolicyOptions {
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
  /// The runner can stop a prompt between passes and resume it.
  bool incremental_prefill{false};
};

struct PrefillingRequest {
  std::size_t remaining_tokens{0};
};

struct DecodingRequest {
  /// Owed a step: it waited through a prefill unit since its last one.
  bool due{false};
  /// Has produced at least one token.
  bool started{false};
};

/// The scheduler's resident requests at one work-unit decision, each queue
/// in its service order.
struct WorkQueues {
  std::span<const PrefillingRequest> prefilling{};
  std::span<const DecodingRequest> decoding{};
  /// A request holds a session while its prompt snapshot completes.
  bool capturing{false};
  /// Every resident request is being prepared for one batched multi-token
  /// decode plan.
  bool forming_batch{false};
  /// No prefill unit has run beside decoders since the last decode.
  bool after_decode{true};
};

/// Prefill the front of `prefilling`.
struct PrefillWork {
  /// Decoders are runnable and wait for this unit.
  bool beside_decoders{false};
  /// A short prompt or a pending capture waits for this unit.
  bool peer_waiting{false};
  /// Prompt tokens this unit may consume; empty leaves the whole remaining
  /// prompt to the runner.
  std::optional<std::size_t> max_tokens;
};

/// Decode one round from `decoding`.
struct DecodeWork {
  /// Requests the decode plan is chosen for.
  std::size_t candidates{0};
  /// Take only due requests, in queue order.
  bool due_only{false};
};

using Work = std::variant<PrefillWork, DecodeWork>;

/// Chooses the next work unit. It holds no request state: the serving
/// scheduler owns the queues and executes each decision.
class Policy {
public:
  explicit Policy(PolicyOptions options = {});

  [[nodiscard]] const PolicyOptions& options() const noexcept {
    return options_;
  }

  /// Requires at least one prefilling or decoding request.
  [[nodiscard]] Work Next(const WorkQueues& queues) const;

private:
  [[nodiscard]] bool Short(const PrefillingRequest& request) const noexcept {
    return request.remaining_tokens <= options_.decode_active_tokens;
  }

  PolicyOptions options_;
};

}  // namespace gufo::sched

#endif  // GUFO_SCHED_POLICY_HPP_
