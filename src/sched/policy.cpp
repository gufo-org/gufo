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

std::size_t Policy::LayerGroups(const PrefillRate& rate) const {
  const auto& profile = *options_.profile;
  // Until a lone pass has been measured, keep decoder waits short.
  const auto pass = rate.Estimate(profile.pass_tokens);
  if (!pass)
    return profile.max_layer_groups;
  const auto groups =
      std::ceil(*pass / PrefillRate::Milliseconds(kTargetDecoderWait));
  return std::clamp<std::size_t>(static_cast<std::size_t>(groups), 1,
                                 profile.max_layer_groups);
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
    if (options_.profile)
      return PrefillWork{.beside_decoders = true,
                         .max_tokens = options_.profile->pass_tokens,
                         .layer_groups = LayerGroups(rate)};
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
