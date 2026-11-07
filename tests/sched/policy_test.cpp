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
}
