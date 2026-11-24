#include "src/sched/policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gufo::sched {

Policy::Policy(PolicyOptions options) : options_(options) {
  if (options_.unit_tokens == 0)
    throw std::invalid_argument("prefill unit must be at least one token");
  const auto& profile = options_.profile;
  if (profile.pass_tokens == 0)
    throw std::invalid_argument(
        "text runner declares no scheduling profile "
        "(TextRunnerDescriptor::scheduling)");
  if (profile.max_layer_groups == 0 || profile.max_yield_decode_width == 0)
    throw std::invalid_argument("invalid text scheduling profile");
}

std::size_t Policy::LayerGroups(const PrefillRate& rate,
                                const PrefillingRequest& request) const {
  const auto& profile = options_.profile;
  const std::size_t remaining_tokens = request.remaining_tokens;
  // A short remainder, such as a cached continuation, is a short pass: each
  // group adds a decode round to its time to first token, and a unit of
  // tokens is what one group may carry, so its tokens never split it. Its
  // context can: deep in a long one the pass reads every attention row for
  // those few tokens. A group carries at most the decoders' wait of that
  // time, once few-token passes measured it; until then it runs unsplit.
  if (remaining_tokens <= options_.unit_tokens) {
    const auto context = rate.EstimateContext(request.position);
    if (!context)
      return 1;
    return std::clamp<std::size_t>(
        static_cast<std::size_t>(std::ceil(
            *context / PrefillRate::Milliseconds(kTargetDecoderWait))),
        1, profile.max_layer_groups);
  }
  const std::size_t tokens = std::min(remaining_tokens, profile.pass_tokens);
  const auto pass = rate.Estimate(tokens);
  // Until a lone pass has been measured, assume a full pass needs the most
  // groups.
  const double groups =
      pass ? std::ceil(*pass / PrefillRate::Milliseconds(kTargetDecoderWait))
           : std::ceil(static_cast<double>(profile.max_layer_groups) *
                       static_cast<double>(tokens) /
                       static_cast<double>(profile.pass_tokens));
  // At most one group per unit of these tokens, which is how arXiv
  // 2510.08055 (section 4.4) sizes its groups: every group adds a decode
  // round to the prompt's time to first token. A dense model's full pass
  // saves too little over unit-sized passes to pay for more rounds; a deep
  // pass still gets groups no longer than a unit-sized pass at that depth.
  const std::size_t units =
      (tokens + options_.unit_tokens - 1) / options_.unit_tokens;
  return std::clamp<std::size_t>(static_cast<std::size_t>(groups), 1,
                                 std::min(profile.max_layer_groups, units));
}

std::optional<DecodeWork> Policy::BetweenGroups(
    std::span<const DecodingRequest> decoding) const {
  if (decoding.empty())
    return std::nullopt;
  return DecodeWork{
      .candidates =
          std::min(decoding.size(), options_.profile.max_yield_decode_width)};
}

Work Policy::Next(const WorkQueues& queues, const PrefillRate& rate) const {
  if (queues.prefilling.empty() && queues.decoding.empty())
    throw std::invalid_argument("work decision requires a resident request");

  const auto due = std::ranges::count_if(
      queues.decoding,
      [](const DecodingRequest& request) { return request.due; });
  // Give simultaneous new requests one bounded chunk to form their first
  // batch. Once decoding starts, every due decoder runs before more prefill.
  const bool assemble_initial_batch =
      queues.forming_batch && queues.after_decode &&
      !queues.prefilling.empty() && Short(queues.prefilling.front()) &&
      std::ranges::none_of(queues.decoding, &DecodingRequest::started);
  if (!queues.prefilling.empty() && !queues.decoding.empty() &&
      (due == 0 || assemble_initial_batch)) {
    // A runner that cannot stop a prompt mid-way runs all of it.
    if (!options_.incremental_prefill)
      return PrefillWork{.beside_decoders = true};
    // The runner keeps its efficient pass width and splits the pass by
    // layers, so decoders wait for one group, not one pass. Full passes would
    // make a prompt that one pass finishes wait a whole pass of each longer
    // peer per turn; it goes first, in queue order, and a longer prompt waits
    // at most one pass per such peer.
    const auto& profile = options_.profile;
    const auto fits = std::ranges::find_if(
        queues.prefilling, [&](const PrefillingRequest& request) {
          return request.remaining_tokens <= profile.pass_tokens;
        });
    const std::size_t index =
        fits == queues.prefilling.end()
            ? 0
            : static_cast<std::size_t>(fits - queues.prefilling.begin());
    return PrefillWork{
        .request = index,
        .beside_decoders = true,
        .max_tokens = profile.pass_tokens,
        .layer_groups = LayerGroups(rate, queues.prefilling[index])};
  }

  if (!queues.decoding.empty()) {
    return DecodeWork{.candidates = due != 0 ? static_cast<std::size_t>(due)
                                             : queues.decoding.size(),
                      .due_only = due != 0};
  }

  // Bound work for short waiting prefills and captures as well as decoders.
  // A long prefill alone keeps the profile's pass width while a free session
  // could admit an arrival, which then waits for one such pass, not a wider
  // one the runner takes when nobody can join (#470). Otherwise it retains
  // the runner's own widest pass.
  const bool peer_waiting =
      queues.capturing ||
      std::any_of(queues.prefilling.begin() + 1, queues.prefilling.end(),
                  [&](const auto& peer) { return Short(peer); });
  std::optional<std::size_t> max_tokens;
  if (options_.incremental_prefill && peer_waiting)
    max_tokens = options_.unit_tokens;
  else if (options_.incremental_prefill && queues.admission_open)
    max_tokens = options_.profile.pass_tokens;
  return PrefillWork{.peer_waiting = peer_waiting, .max_tokens = max_tokens};
}

}  // namespace gufo::sched
