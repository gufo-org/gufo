#ifndef GUFO_SCHED_PROFILE_HPP_
#define GUFO_SCHED_PROFILE_HPP_

#include <cstddef>
#include <functional>

namespace gufo::sched {

/// How a text runner wants its prompts scheduled while other sessions
/// decode. A runner without one gets passes bounded by the host's
/// decode-active budget instead.
struct TextSchedulingProfile {
  /// Prompt tokens one pass covers beside decoders: the model's efficient
  /// width, normally its full prefill pass.
  std::size_t pass_tokens{0};
  /// Layer groups one such pass may be split into, with a decode round for
  /// the other sessions between groups. One runs the pass unsplit.
  std::size_t max_layer_groups{1};
  /// Sessions one decode step between layer groups may batch. Each round
  /// steps every decoder once, in as many steps as this width needs.
  std::size_t max_yield_decode_width{1};

  bool operator==(const TextSchedulingProfile&) const = default;
};

/// Runs the host's decode round between two layer groups of a prefill pass.
/// It decodes only other sessions, never touches the prefilling state and
/// does not throw. The prefilling request's cancellation is reported through
/// the state's cancellation check, as for any pass.
using PrefillYield = std::function<void()>;

}  // namespace gufo::sched

#endif  // GUFO_SCHED_PROFILE_HPP_
