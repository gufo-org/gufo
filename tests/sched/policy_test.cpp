#include "src/sched/policy.hpp"

#include <cassert>
#include <stdexcept>
#include <variant>
#include <vector>

#include "src/sched/prefill_rate.hpp"

using namespace gufo::sched;

namespace {

const Policy kIncremental({.incremental_prefill = true});

template<class Exception, class F>
bool Throws(F&& function) {
  try {
    function();
  } catch (const Exception&) {
    return true;
  }
  return false;
}

PrefillWork Prefill(const Work& work) {
  assert(std::holds_alternative<PrefillWork>(work));
  return std::get<PrefillWork>(work);
}

DecodeWork Decode(const Work& work) {
  assert(std::holds_alternative<DecodeWork>(work));
  return std::get<DecodeWork>(work);
}

void Preconditions() {
  assert(Throws<std::invalid_argument>(
      [] { Policy({.decode_active_tokens = 0}); }));
  assert(Throws<std::invalid_argument>([] { (void)kIncremental.Next({}); }));
}

void LonePromptKeepsFullPasses() {
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 9000}};
  const auto work = Prefill(kIncremental.Next({.prefilling = prefilling}));
  assert(!work.beside_decoders && !work.peer_waiting && !work.max_tokens);
}

void WaitingPeersBoundPrefill() {
  // A short prompt behind a long one bounds the long one's unit.
  const std::vector<PrefillingRequest> short_peer{{.remaining_tokens = 9000},
                                                  {.remaining_tokens = 512}};
  auto work = Prefill(kIncremental.Next({.prefilling = short_peer}));
  assert(work.peer_waiting && work.max_tokens == 512u);

  // A long peer does not; the front request is not its own peer.
  const std::vector<PrefillingRequest> long_peer{{.remaining_tokens = 100},
                                                 {.remaining_tokens = 513}};
  work = Prefill(kIncremental.Next({.prefilling = long_peer}));
  assert(!work.peer_waiting && !work.max_tokens);

  // A pending capture is a waiting peer.
  work =
      Prefill(kIncremental.Next({.prefilling = long_peer, .capturing = true}));
  assert(work.peer_waiting && work.max_tokens == 512u);
}

void DecodersBoundPrefillUntilDue() {
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 9000}};
  std::vector<DecodingRequest> decoding{{.started = true}, {.started = true}};
  auto work = Prefill(
      kIncremental.Next({.prefilling = prefilling, .decoding = decoding}));
  assert(work.beside_decoders && !work.peer_waiting && work.max_tokens == 512u);

  // A due decoder runs before more prefill; only due ones are candidates.
  decoding[1].due = true;
  const auto decode = Decode(
      kIncremental.Next({.prefilling = prefilling, .decoding = decoding}));
  assert(decode.due_only && decode.candidates == 1);
}

void DecodersWithoutPrefill() {
  const std::vector<DecodingRequest> idle{{.started = true}, {.started = true}};
  auto decode = Decode(kIncremental.Next({.decoding = idle}));
  assert(!decode.due_only && decode.candidates == 2);

  const std::vector<DecodingRequest> due{
      {.due = true, .started = true}, {.started = true}, {.due = true}};
  decode = Decode(kIncremental.Next({.decoding = due}));
  assert(decode.due_only && decode.candidates == 2);
}

void FirstBatchAssembles() {
  // New requests forming a batch get one short chunk before due decoders,
  // but only before any of them has produced a token, right after a decode.
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 300}};
  std::vector<DecodingRequest> decoding{{.due = true}};
  const WorkQueues forming{
      .prefilling = prefilling, .decoding = decoding, .forming_batch = true};
  assert(Prefill(kIncremental.Next(forming)).beside_decoders);

  auto queues = forming;
  queues.after_decode = false;
  assert(std::holds_alternative<DecodeWork>(kIncremental.Next(queues)));

  queues = forming;
  queues.forming_batch = false;
  assert(std::holds_alternative<DecodeWork>(kIncremental.Next(queues)));

  const std::vector<PrefillingRequest> long_prompt{{.remaining_tokens = 513}};
  queues = forming;
  queues.prefilling = long_prompt;
  assert(std::holds_alternative<DecodeWork>(kIncremental.Next(queues)));

  decoding[0].started = true;
  assert(std::holds_alternative<DecodeWork>(kIncremental.Next(forming)));
}

void WholePromptRunners() {
  // A runner that cannot resume a prompt mid-way always gets all of it.
  const Policy whole({.decode_active_tokens = 64});
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 9000},
                                                  {.remaining_tokens = 1}};
  const std::vector<DecodingRequest> decoding{{.started = true}};
  assert(!Prefill(whole.Next({.prefilling = prefilling, .decoding = decoding}))
              .max_tokens);
  const auto work = Prefill(whole.Next({.prefilling = prefilling}));
  assert(work.peer_waiting && !work.max_tokens);
}

void CustomBudget() {
  const Policy policy(
      {.decode_active_tokens = 2048, .incremental_prefill = true});
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 9000},
                                                  {.remaining_tokens = 2048}};
  assert(Prefill(policy.Next({.prefilling = prefilling})).max_tokens == 2048u);
}

void PrefillRateTracksRecentWork() {
  PrefillRate rate;
  assert(!rate.Estimate(4096));
  rate.Observe(255, 100);
  rate.Observe(4096, 0.5);
  assert(!rate.Estimate(4096));
  rate.Observe(4096, 2000);
  assert(rate.Estimate(2048)->count() == 1000);
  for (int i = 0; i < 8; ++i)
    rate.Observe(1000, 100);
  assert(rate.Estimate(2048)->count() == 204.8);
}

void ProfiledPassesSplitByLayers() {
  const Policy policy(
      {.incremental_prefill = true,
       .profile = TextSchedulingProfile{.pass_tokens = 4096,
                                        .max_layer_groups = 8,
                                        .max_yield_decode_width = 2}});
  const std::vector<PrefillingRequest> prefilling{{.remaining_tokens = 9000}};
  const std::vector<DecodingRequest> decoding{{.started = true}};
  const WorkQueues queues{.prefilling = prefilling, .decoding = decoding};

  // Uncalibrated: the most groups.
  auto work = Prefill(policy.Next(queues));
  assert(work.beside_decoders && work.max_tokens == 4096u &&
         work.layer_groups == 8);

  // 2.1 s per pass: ceil(2100 / 250) = 9 groups, capped at 8.
  PrefillRate rate;
  rate.Observe(4096, 2100);
  assert(Prefill(policy.Next(queues, rate)).layer_groups == 8);
  // 0.6 s: three groups of at most 250 ms.
  PrefillRate fast;
  fast.Observe(4096, 600);
  assert(Prefill(policy.Next(queues, fast)).layer_groups == 3);
  // Shorter than one target wait: unsplit.
  PrefillRate tiny;
  tiny.Observe(4096, 100);
  assert(Prefill(policy.Next(queues, tiny)).layer_groups == 1);

  // A shorter remainder is a shorter pass, with fewer groups: 1,000 tokens
  // at 2.1 s per 4,096 take about 513 ms, three groups of 250 ms, but get
  // no more than legacy 512-token chunks would: two.
  const std::vector<PrefillingRequest> tail{{.remaining_tokens = 1000}};
  const WorkQueues tail_queues{.prefilling = tail, .decoding = decoding};
  work = Prefill(policy.Next(tail_queues, rate));
  assert(work.max_tokens == 4096u && work.layer_groups == 2);
  // 1,600 tokens take about 820 ms: four groups, the legacy chunk count.
  const std::vector<PrefillingRequest> mid{{.remaining_tokens = 1600}};
  assert(Prefill(policy.Next({.prefilling = mid, .decoding = decoding}, rate))
             .layer_groups == 4);
  // A cached continuation's few tokens run unsplit.
  const std::vector<PrefillingRequest> continuation{{.remaining_tokens = 300}};
  assert(Prefill(policy.Next({.prefilling = continuation, .decoding = decoding},
                             rate))
             .layer_groups == 1);
  // Up to the legacy 512-token budget, even a slow pass runs unsplit, as it
  // would without a profile: 512 tokens at 2 ms each would be five groups.
  PrefillRate slow;
  slow.Observe(4096, 8192);
  const std::vector<PrefillingRequest> budget{{.remaining_tokens = 512}};
  assert(
      Prefill(policy.Next({.prefilling = budget, .decoding = decoding}, slow))
          .layer_groups == 1);
  // Past it, a slow pass gets at most one group per legacy chunk: 513 tokens
  // take about 1 s, four target waits, but two chunks.
  const std::vector<PrefillingRequest> past_budget{{.remaining_tokens = 513}};
  assert(Prefill(policy.Next({.prefilling = past_budget, .decoding = decoding},
                             slow))
             .layer_groups == 2);
  // A slow full pass (8 s) reaches the profile's maximum.
  assert(Prefill(policy.Next(queues, slow)).layer_groups == 8);
  // Uncalibrated, the most groups scale with the remainder: ceil(8 * 1000 /
  // 4096) = 2, and a continuation still runs unsplit.
  assert(Prefill(policy.Next(tail_queues)).layer_groups == 2);
  assert(
      Prefill(policy.Next({.prefilling = continuation, .decoding = decoding}))
          .layer_groups == 1);

  // Alone, or beside a waiting short peer, the profile does not apply.
  work = Prefill(policy.Next({.prefilling = prefilling}, fast));
  assert(!work.max_tokens && work.layer_groups == 1);
  const std::vector<PrefillingRequest> short_peer{{.remaining_tokens = 9000},
                                                  {.remaining_tokens = 100}};
  work = Prefill(policy.Next({.prefilling = short_peer}, fast));
  assert(work.max_tokens == 512u && work.layer_groups == 1);
}

void OnePassPeersGoFirst() {
  const Policy policy({.incremental_prefill = true,
                       .profile = TextSchedulingProfile{
                           .pass_tokens = 4096, .max_layer_groups = 8}});
  const std::vector<DecodingRequest> decoding{{.started = true}};
  PrefillRate rate;
  rate.Observe(4096, 2100);
  // A 12K prompt ahead of two that one pass finishes: the first of those
  // goes, with groups sized for its own 800 tokens (about 410 ms, two).
  const std::vector<PrefillingRequest> burst{{.remaining_tokens = 12000},
                                             {.remaining_tokens = 800},
                                             {.remaining_tokens = 3000}};
  auto work =
      Prefill(policy.Next({.prefilling = burst, .decoding = decoding}, rate));
  assert(work.request == 1 && work.beside_decoders &&
         work.max_tokens == 4096u && work.layer_groups == 2);
  // Exactly one pass still fits; only longer peers keep queue order.
  const std::vector<PrefillingRequest> full{{.remaining_tokens = 9000},
                                            {.remaining_tokens = 4097},
                                            {.remaining_tokens = 4096}};
  assert(Prefill(policy.Next({.prefilling = full, .decoding = decoding}, rate))
             .request == 2);
  const std::vector<PrefillingRequest> longer{{.remaining_tokens = 9000},
                                              {.remaining_tokens = 5000}};
  assert(
      Prefill(policy.Next({.prefilling = longer, .decoding = decoding}, rate))
          .request == 0);
  // Without decoders, or without a profile, the front runs.
  assert(Prefill(policy.Next({.prefilling = burst}, rate)).request == 0);
  const Policy unprofiled({.incremental_prefill = true});
  assert(Prefill(unprofiled.Next({.prefilling = burst, .decoding = decoding}))
             .request == 0);
}

void RoundsBetweenGroups() {
  const Policy unprofiled({.incremental_prefill = true});
  const Policy batched(
      {.incremental_prefill = true,
       .profile = TextSchedulingProfile{.pass_tokens = 4096,
                                        .max_layer_groups = 4,
                                        .max_yield_decode_width = 2}});
  assert(!batched.BetweenGroups({}));
  const std::vector<DecodingRequest> three(3);
  const auto round = batched.BetweenGroups(three);
  assert(round && round->candidates == 2 && !round->due_only);
  assert(unprofiled.BetweenGroups(three)->candidates == 1);
  const std::vector<DecodingRequest> one(1);
  assert(batched.BetweenGroups(one)->candidates == 1);
}

void InvalidProfiles() {
  const TextSchedulingProfile valid{.pass_tokens = 4096, .max_layer_groups = 4};
  assert(Throws<std::invalid_argument>([&] { Policy({.profile = valid}); }));
  auto zero_pass = valid;
  zero_pass.pass_tokens = 0;
  auto zero_groups = valid;
  zero_groups.max_layer_groups = 0;
  auto zero_width = valid;
  zero_width.max_yield_decode_width = 0;
  for (const auto& profile : {zero_pass, zero_groups, zero_width})
    assert(Throws<std::invalid_argument>(
        [&] { Policy({.incremental_prefill = true, .profile = profile}); }));
}

}  // namespace

int main() {
  Preconditions();
  LonePromptKeepsFullPasses();
  WaitingPeersBoundPrefill();
  DecodersBoundPrefillUntilDue();
  DecodersWithoutPrefill();
  FirstBatchAssembles();
  WholePromptRunners();
  CustomBudget();
  PrefillRateTracksRecentWork();
  ProfiledPassesSplitByLayers();
  OnePassPeersGoFirst();
  RoundsBetweenGroups();
  InvalidProfiles();
}
