#include "src/sched/policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gufo::sched {

Policy::Policy(PolicyOptions options) : options_(options) {
  if (options_.decode_active_tokens == 0)
    throw std::invalid_argument(
        "active-decode prefill budget must be at least one token");
  if (const auto& profile = options_.profile) {
    if (!options_.incremental_prefill)
      throw std::invalid_argument(
          "a scheduling profile requires incremental prefill");
    if (profile->pass_tokens == 0 || profile->max_layer_groups == 0 ||
        profile->max_yield_decode_width == 0)
      throw std::invalid_argument("invalid text scheduling profile");
  }
}

std::size_t Policy::LayerGroups(const PrefillRate& rate,
                                std::size_t remaining_tokens) const {
  const auto& profile = *options_.profile;
  // A short remainder, such as a cached continuation, is a short pass: each
  // group adds a decode round to its time to first token. One within the
  // legacy budget runs unsplit, as it would without a profile.
  if (remaining_tokens <= options_.decode_active_tokens)
    return 1;
  const std::size_t tokens = std::min(remaining_tokens, profile.pass_tokens);
  const auto pass = rate.Estimate(tokens);
  // Until a lone pass has been measured, assume a full pass needs the most
  // groups.
  const double groups =
      pass ? std::ceil(*pass / PrefillRate::Milliseconds(kTargetDecoderWait))
           : std::ceil(static_cast<double>(profile.max_layer_groups) *
                       static_cast<double>(tokens) /
                       static_cast<double>(profile.pass_tokens));
  // At most one group per legacy chunk of these tokens, which is how
  // arXiv 2510.08055 (section 4.4) sizes its groups: every group adds a
  // decode round to the prompt's time to first token. A dense model's full
  // pass saves too little over legacy chunks to pay for more rounds; a deep
  // pass still gets groups no longer than a legacy chunk at that depth.
  const std::size_t legacy_chunks =
      (tokens + options_.decode_active_tokens - 1) /
      options_.decode_active_tokens;
  return std::clamp<std::size_t>(
      static_cast<std::size_t>(groups), 1,
      std::min(profile.max_layer_groups, legacy_chunks));
}

std::optional<DecodeWork> Policy::BetweenGroups(
    std::span<const DecodingRequest> decoding) const {
  if (decoding.empty())
    return std::nullopt;
  const std::size_t width =
      options_.profile ? options_.profile->max_yield_decode_width : 1;
  return DecodeWork{.candidates = std::min(decoding.size(), width)};
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
    // A profiled runner keeps its efficient pass width and splits the pass
    // by layers instead, so decoders wait for one group, not one pass.
    if (options_.profile) {
      // Full passes would make a prompt that one pass finishes wait a whole
      // pass of each longer peer per turn (chunks of the legacy budget
      // interleaved them). It goes first, in queue order; a longer prompt
      // waits at most one pass per such peer.
      const auto fits = std::ranges::find_if(
          queues.prefilling, [&](const PrefillingRequest& request) {
            return request.remaining_tokens <= options_.profile->pass_tokens;
          });
      const std::size_t index =
          fits == queues.prefilling.end()
              ? 0
              : static_cast<std::size_t>(fits - queues.prefilling.begin());
      return PrefillWork{.request = index,
                         .beside_decoders = true,
                         .max_tokens = options_.profile->pass_tokens,
                         .layer_groups = LayerGroups(
                             rate, queues.prefilling[index].remaining_tokens)};
    }
    return PrefillWork{
        .beside_decoders = true,
        .max_tokens = options_.incremental_prefill
                          ? std::optional(options_.decode_active_tokens)
                          : std::nullopt};
  }

  if (!queues.decoding.empty()) {
    return DecodeWork{.candidates = due != 0 ? static_cast<std::size_t>(due)
                                             : queues.decoding.size(),
                      .due_only = due != 0};
  }

  // Bound work for short waiting prefills and captures as well as decoders.
  // Long prefills alone retain the model's efficient chunk size.
  const bool peer_waiting =
      queues.capturing ||
      std::any_of(queues.prefilling.begin() + 1, queues.prefilling.end(),
                  [&](const auto& peer) { return Short(peer); });
  return PrefillWork{
      .peer_waiting = peer_waiting,
      .max_tokens = options_.incremental_prefill && peer_waiting
                        ? std::optional(options_.decode_active_tokens)
                        : std::nullopt};
}

}  // namespace gufo::sched
