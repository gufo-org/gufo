#ifndef GUFO_SCHED_POLICY_HPP_
#define GUFO_SCHED_POLICY_HPP_

#include <chrono>
#include <cstddef>
#include <optional>
#include <span>
#include <variant>

#include "src/sched/prefill_rate.hpp"
#include "src/sched/profile.hpp"

namespace gufo::sched {

/// The policy's prompt granularity. A prompt within it runs unsplit beside
/// decoders, a layered pass gets at most one group per unit of its tokens
/// (arXiv 2510.08055, section 4.4), and a long prompt alone stops after each
/// unit while a short peer or a capture waits.
inline constexpr std::size_t kPrefillUnitTokens = 512;

/// Longest a decoder should wait for one layer group of a prompt pass. The
/// group count follows from the measured pass time, up to the profile's
/// maximum and one group per unit of the pass's tokens.
inline constexpr std::chrono::milliseconds kTargetDecoderWait{250};

struct PolicyOptions {
  /// The runner's profile; every field must be at least one.
  TextSchedulingProfile profile{};
  /// The runner can stop a prompt between passes and resume it. Without it,
  /// every prompt runs whole, in one unsplit pass.
  bool incremental_prefill{false};
  /// Tests shrink it to keep prompts small.
  std::size_t unit_tokens{kPrefillUnitTokens};
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

/// Prefill one request of `prefilling`.
struct PrefillWork {
  /// Index of that request in `WorkQueues::prefilling`; the front unless a
  /// profiled pass lets a peer that one pass finishes go first.
  std::size_t request{0};
  /// Decoders are runnable and wait for this unit.
  bool beside_decoders{false};
  /// A short prompt or a pending capture waits for this unit.
  bool peer_waiting{false};
  /// Prompt tokens this unit may consume; empty leaves the whole remaining
  /// prompt to the runner.
  std::optional<std::size_t> max_tokens;
  /// Layer groups the pass is split into, with a decode round between
  /// groups (BetweenGroups). One runs it unsplit.
  std::size_t layer_groups{1};
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
  /// Throws std::invalid_argument for an incomplete profile.
  explicit Policy(PolicyOptions options);

  [[nodiscard]] const PolicyOptions& options() const noexcept {
    return options_;
  }

  /// Requires at least one prefilling or decoding request. `rate` is the
  /// recent prefill rate of passes that ran alone.
  [[nodiscard]] Work Next(const WorkQueues& queues,
                          const PrefillRate& rate = {}) const;

  /// One decode step of the round between two layer groups of a pass, from
  /// the decoders that have not stepped in it yet: as many as the profile
  /// batches. The host repeats it until each has stepped once. Empty
  /// without decoders.
  [[nodiscard]] std::optional<DecodeWork> BetweenGroups(
      std::span<const DecodingRequest> decoding) const;

private:
  [[nodiscard]] std::size_t LayerGroups(const PrefillRate& rate,
                                        std::size_t remaining_tokens) const;
  [[nodiscard]] bool Short(const PrefillingRequest& request) const noexcept {
    return request.remaining_tokens <= options_.unit_tokens;
  }

  PolicyOptions options_;
};

}  // namespace gufo::sched

#endif  // GUFO_SCHED_POLICY_HPP_
