#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "src/core/json_constraint.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/runtime/dspark_policy.h"
#include "src/models/deepseek_v4_flash/runtime/native_internal.h"

using gufo::models::deepseek_v4_flash::DsparkSamplerBridge;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;

namespace {
void Expect(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

ds4_dspark_candidates Candidates() {
  ds4_dspark_candidates result{};
  for (int i = 0; i < 8; ++i) {
    result.ids[i] = i;
    result.logits[i] = (3 - i) * 0.3F;
  }
  result.confidence = 0.8F;
  return result;
}

template<typename F>
void ExpectThrows(F action) {
  bool threw = false;
  try {
    action();
  } catch (const std::exception&) {
    threw = true;
  }
  Expect(threw, "malformed proposal must fail");
}

void CheckDistribution(SamplingConfig config, bool sharp = false) {
  constexpr std::array<float, 10> target{0.5F,  -0.3F, 0.7F, 0.1F, 0.2F,
                                         -0.5F, -0.2F, 0.4F, 0.9F, -0.1F};
  constexpr std::array<gufo::sampling::TokenId, 5> history{8, 2, 2, 1, 0};
  config.seed = 731;
  SamplerState state(config, history);
  const auto expected = state.Distribution(target);
  auto candidates = Candidates();
  if (sharp)
    candidates.logits[0] = 12;
  constexpr unsigned trials = 120000;
  std::array<unsigned, 10> counts{};
  unsigned accepted = 0;
  for (unsigned i = 0; i < trials; ++i) {
    DsparkSamplerBridge bridge(state);
    const auto* hook = bridge.hook();
    const int draft = hook->propose(hook->ctx, 0, &candidates);
    const int result =
        hook->verify(hook->ctx, 0, target.data(), target.size(), draft);
    Expect(result >= 0 && result < 10, "target vocabulary");
    accepted += result == draft;
    ++counts[result];
    state.SetRngState(bridge.rng_state());
  }
  double target_mass_on_q = 0;
  for (unsigned token = 0; token < 10; ++token) {
    const double p = expected.probability(token);
    const double observed = (double)counts[token] / trials;
    const double tolerance = 6 * std::sqrt(p * (1 - p) / trials) + 2.0 / trials;
    Expect(std::abs(observed - p) <= tolerance,
           "p/q and residual preserve the complete filtered target");
    if (token < 8)
      target_mass_on_q += p;
  }
  Expect(accepted <= trials, "acceptance count");
  if (target_mass_on_q < 1)
    Expect(counts[8] + counts[9] > 0, "residual samples outside sparse q");
}

void CheckConditionalProposals() {
  const SamplingConfig config{.temperature = 0.7F,
                              .top_k = 10,
                              .seed = 73,
                              .repeat_last_n = 3,
                              .frequency_penalty = 0.3F};
  const std::array<gufo::sampling::TokenId, 3> history{1, 1, 4};
  SamplerState initial(config, history);
  std::array<float, 10> first{};
  std::array<std::array<float, 10>, 10> second{};
  for (unsigned i = 0; i < 10; ++i) {
    first[i] = 0.1F * i;
    for (unsigned j = 0; j < 10; ++j)
      second[i][j] = 0.2F * ((i * 3 + j * 7) % 10);
  }
  const auto p = initial.Distribution(first);
  std::array<std::array<double, 10>, 10> expected{};
  for (unsigned i = 0; i < 10; ++i) {
    auto conditional = initial;
    conditional.Accept(i);
    const auto next = conditional.Distribution(second[i]);
    for (unsigned j = 0; j < 10; ++j)
      expected[i][j] = p.probability(i) * next.probability(j);
  }
  constexpr unsigned trials = 120000;
  std::array<std::array<unsigned, 10>, 10> counts{};
  for (unsigned trial = 0; trial < trials; ++trial) {
    DsparkSamplerBridge bridge(initial);
    const auto* hook = bridge.hook();
    auto candidates = Candidates();
    if (trial % 2 == 0)
      candidates.logits[3] = 12;  // Mix exact delta and probabilistic q.
    const int a = hook->propose(hook->ctx, 0, &candidates);
    candidates = Candidates();
    if (a % 2 == 0)
      candidates.logits[1] = 12;
    // A synthetic Markov dependence on the preceding proposed token.
    for (auto& id : candidates.ids)
      id = (id + a + 1) % 10;
    const int b = hook->propose(hook->ctx, 1, &candidates);
    const int x = hook->verify(hook->ctx, 0, first.data(), 10, a);
    // Rejection discards the unused draft and continues from the residual.
    const int y = x == a ? hook->verify(hook->ctx, 1, second[x].data(), 10, b)
                         : hook->sample(hook->ctx, second[x].data(), 10);
    ++counts[x][y];
    initial.SetRngState(bridge.rng_state());
  }
  for (unsigned i = 0; i < 10; ++i)
    for (unsigned j = 0; j < 10; ++j) {
      const double probability = expected[i][j];
      const double tolerance =
          6 * std::sqrt(probability * (1 - probability) / trials) +
          2.0 / trials;
      Expect(std::abs((double)counts[i][j] / trials - probability) <= tolerance,
             "conditional q, rejected suffix and committed penalties preserve "
             "joint p");
    }
}
}  // namespace

int main() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float extreme[]{-3e35F, -2e35F, -1e35F};
  const float mixed[]{nan, inf, -inf, -3e35F, -3e35F};
  Expect(ds4_sample_argmax(extreme, 3) == 2,
         "argmax covers finite float range");
  Expect(ds4_sample_argmax(mixed, 5) == 3,
         "argmax skips nonfinite values and keeps first tie");
  Expect(ds4_sample_argmax(mixed, 3) == -1 &&
             ds4_sample_argmax(nullptr, 3) == -1 &&
             ds4_sample_argmax(extreme, 0) == -1,
         "invalid argmax reports failure");
  std::uint64_t rng = 5;
  for (int top_k : {0, 2})
    Expect(ds4_sample_top_p_min_p(mixed, 3, 1, top_k, 0.9F, 0.1F, &rng) == -1,
           "nonfinite sampled target fails closed");
  CheckConditionalProposals();
  CheckDistribution({.temperature = 1.F, .top_k = 10}, true);
  for (size_t c : {1u, 2u, 4u, 8u}) {
    ds4_dspark_confidence_policy cold{.depth = 4096, .concurrency = c};
    Expect(!cold.Include(0.01F) && cold.included == 0,
           "low predicted survival selects AR before drawing");
    ds4_dspark_confidence_policy hot{.depth = 4096, .concurrency = c};
    bool profitable = false;
    for (unsigned width = 1; width <= 3; ++width)
      profitable |= hot.Cost(width) < 100 * (width + 1);
    Expect(hot.Include(1.F) == profitable,
           "high confidence still respects measured hardware cost");
    const auto included = hot.included;
    Expect(!hot.Include(0.F) && hot.included == included,
           "stopping never retracts an earlier proposal");
    Expect(!hot.Include(NAN) && !hot.Include(INFINITY),
           "non-finite confidence fails closed");
  }
  for (const auto config : {
           SamplingConfig{.temperature = 0.6F, .top_k = 10},
           SamplingConfig{.temperature = 1.0F, .top_k = 10},
           SamplingConfig{.temperature = 0.6F, .top_k = 4, .top_p = 0.95F},
           SamplingConfig{.temperature = 1.0F,
                          .top_p = 0.95F,
                          .min_p = 0.2F,
                          .min_keep = 3,
                          .repeat_penalty = 1.2F,
                          .frequency_penalty = 0.3F,
                          .presence_penalty = 0.1F},
           SamplingConfig{.temperature = 1.0F, .top_k = 1, .min_keep = 2},
       })
    CheckDistribution(config);

  SamplerState state({.temperature = 0.7F, .top_k = 10, .seed = 12});
  auto candidates = Candidates();
  const std::array<float, 10> logits{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  DsparkSamplerBridge a(state), b(state);
  const auto *ha = a.hook(), *hb = b.hook();
  const int da = ha->propose(ha->ctx, 0, &candidates);
  const int db = hb->propose(hb->ctx, 0, &candidates);
  Expect(da == db, "proposal seed replay");
  Expect(ha->verify(ha->ctx, 0, logits.data(), logits.size(), da) ==
                 hb->verify(hb->ctx, 0, logits.data(), logits.size(), db) &&
             a.rng_state() == b.rng_state(),
         "verification seed replay");
  DsparkSamplerBridge missing(state);
  ExpectThrows([&] {
    missing.hook()->verify(missing.hook()->ctx, 0, logits.data(), logits.size(),
                           0);
  });
  for (const auto invalid : {NAN, INFINITY}) {
    auto broken = candidates;
    broken.logits[2] = invalid;
    ExpectThrows([&] { ha->propose(ha->ctx, 0, &broken); });
  }
  auto duplicate = candidates;
  duplicate.ids[2] = duplicate.ids[1];
  ExpectThrows([&] { ha->propose(ha->ctx, 0, &duplicate); });
  auto sharp = candidates;
  sharp.logits[3] = 30;
  const auto before_sharp = a.rng_state();
  Expect(ha->propose(ha->ctx, 0, &sharp) == 3 && a.rng_state() == before_sharp,
         "concentrated q becomes an exact delta without a proposal draw");
  sharp.ids[2] = sharp.ids[1];
  ExpectThrows([&] { ha->propose(ha->ctx, 0, &sharp); });
  auto invalid_id = candidates;
  invalid_id.ids[7] = -1;
  ExpectThrows([&] { ha->propose(ha->ctx, 0, &invalid_id); });
  auto empty = candidates;
  std::fill(std::begin(empty.logits), std::end(empty.logits), -INFINITY);
  ExpectThrows([&] { ha->propose(ha->ctx, 0, &empty); });
  auto outside = candidates;
  outside.ids[0] = 100;
  const int draft = ha->propose(ha->ctx, 0, &outside);
  ExpectThrows(
      [&] { ha->verify(ha->ctx, 0, logits.data(), logits.size(), draft); });
  SamplerState greedy({.temperature = 0, .repeat_penalty = 1.1F});
  DsparkSamplerBridge greedy_bridge(greedy);
  Expect(greedy_bridge.hook()->propose == nullptr &&
             greedy_bridge.hook()->verify == nullptr &&
             greedy_bridge.hook()->try_accept_argmax == nullptr,
         "greedy retains point-mass behavior");
  SamplerState unfiltered({.temperature = 0.7F, .seed = 12});
  DsparkSamplerBridge delta(unfiltered);
  Expect(delta.hook()->propose == nullptr && delta.hook()->verify == nullptr &&
             delta.hook()->try_accept_argmax == nullptr,
         "expensive full-vocabulary p retains the point-mass fallback");
  const auto expected = unfiltered.Sample(logits);
  Expect(delta.hook()->sample(delta.hook()->ctx, logits.data(),
                              logits.size()) == static_cast<int>(expected) &&
             delta.rng_state() == unfiltered.rng_state(),
         "point-mass fallback preserves the sampled AR seed trace");
  const std::array<std::string, 3> pieces{"{", "\"x\":1", "}"};
  auto constraint = std::make_shared<gufo::sampling::TokenConstraint>();
  constraint->grammar = gufo::sampling::JsonConstraint::Object();
  constraint->vocabulary =
      std::make_shared<gufo::sampling::ConstraintVocabulary>(
          pieces.size(), [&](std::uint32_t i) {
            return gufo::sampling::ConstraintVocabulary::Piece{pieces[i],
                                                               false};
          });
  SamplerState constrained({.seed = 19, .constraint = constraint});
  DsparkSamplerBridge fast(constrained);
  const auto* hook = fast.hook();
  Expect(hook->try_accept_argmax != nullptr,
         "greedy constraints can use exact target argmaxes");
  ExpectThrows([&] { hook->try_accept_argmax(hook->ctx, 3); });
  Expect(!hook->try_accept_argmax(hook->ctx, -1) &&
             !hook->try_accept_argmax(hook->ctx, 2),
         "invalid and forbidden argmaxes leave the grammar unchanged");
  Expect(hook->try_accept_argmax(hook->ctx, 0) &&
             !hook->try_accept_argmax(hook->ctx, 0) &&
             hook->try_accept_argmax(hook->ctx, 1) &&
             hook->try_accept_argmax(hook->ctx, 2) &&
             !hook->try_accept_argmax(hook->ctx, 1),
         "accepted argmaxes advance the grammar exactly once");
  Expect(fast.rng_state() == constrained.rng_state(),
         "greedy constraint acceptance consumes no random draws");
  std::puts(
      "DSpark sampling: distribution, residual, replay and malformed rows "
      "pass");
}
