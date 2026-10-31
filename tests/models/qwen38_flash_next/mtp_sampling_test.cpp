#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckLengthController() {
  qfn::MtpLengthController controller(7);
  const auto initial = controller.Choose(7);
  Require(controller.Choose(0) == 0, "empty draft budget was ignored");
  Require(controller.Choose(1) == 1, "one-draft budget was ignored");
  controller.Observe(0, 0);
  Require(controller.Choose(7) == initial, "empty cycle changed policy");

  for (unsigned i = 0; i < 12; ++i) {
    const auto length = controller.Choose(7);
    controller.Observe(length, length);
  }
  Require(controller.Choose(7) == 7,
          "fully accepted prefixes did not grow to the maximum length");
  Require(controller.Choose(2) == 2, "remaining output budget was ignored");
  for (unsigned i = 0; i < 12; ++i) {
    controller.Observe(0, controller.Choose(7));
  }
  Require(controller.Choose(7) == 0,
          "unprofitable speculation did not back off to AR");
  qfn::MtpLengthController backed_off(7);
  Require(backed_off.Restore(controller.State()), "backoff restore failed");
  for (unsigned i = 0; i < 16; ++i) {
    Require(controller.Choose(7) == 0 && backed_off.Choose(7) == 0,
            "AR interval ended early");
    controller.ObserveArToken();
    backed_off.ObserveArToken();
  }
  Require(controller.Choose(7) == 1 && backed_off.Choose(7) == 1,
          "AR backoff did not retry a proposal");
  controller.Reset();
  Require(controller.Choose(7) == initial, "reset retained old acceptance");
  controller.Observe(5, 6);
  qfn::MtpLengthController restored(7);
  Require(restored.Restore(controller.State()), "controller restore failed");
  for (unsigned budget = 0; budget <= 8; ++budget) {
    Require(restored.Choose(budget) == controller.Choose(budget),
            "restored policy changes draft widths");
  }
  auto invalid = controller.State();
  invalid.successes[0] = -1.0F;
  Require(!restored.Restore(invalid) && !restored.Restore({}),
          "invalid controller state accepted");

  // The unverified suffix is not additional evidence of failure.
  qfn::MtpLengthController short_tail(7);
  qfn::MtpLengthController long_tail(7);
  for (unsigned accepted : {2U, 0U, 1U, 2U, 1U, 0U}) {
    short_tail.Observe(accepted, accepted + 1);
    long_tail.Observe(accepted, 7);
    for (unsigned budget = 0; budget <= 9; ++budget) {
      Require(short_tail.Choose(budget) == long_tail.Choose(budget),
              "unverified suffix biased the acceptance estimate");
    }
  }
  for (unsigned limit = 1; limit <= 7; ++limit) {
    qfn::MtpLengthController a(limit);
    qfn::MtpLengthController b(limit);
    for (unsigned cycle = 0; cycle < 32; ++cycle) {
      const auto budget = cycle % 10;
      const auto length = a.Choose(budget);
      Require(length == b.Choose(budget) && length <= budget && length <= limit,
              "controller violated a limit or changed on replay");
      const unsigned accepted = length == 0 ? 0 : cycle % (length + 1);
      a.Observe(accepted, length);
      b.Observe(accepted, length);
      if (length == 0) {
        a.ObserveArToken();
        b.ObserveArToken();
      }
    }
  }
  qfn::MtpLengthController early_failure(7), late_failure(7);
  for (unsigned i = 0; i < 12; ++i) {
    early_failure.Observe(0, 7);
    late_failure.Observe(4, 7);
  }
  Require(early_failure.Choose(7) == 0 && late_failure.Choose(7) > 1,
          "conditional acceptance by depth was collapsed to one probability");
}

void CheckOccupancyOverride() {
  // About half of the first drafts are accepted: one draft pays for itself at
  // the C1 cost row but not at C8, where a lone request would fall back to AR.
  qfn::MtpLengthController configured_8(7, 8), configured_1(7, 1), lone(7, 8);
  for (unsigned i = 0; i < 12; ++i) {
    const unsigned accepted = i % 2;
    configured_8.Observe(accepted, 1);
    configured_1.Observe(accepted, 1);
    lone.Observe(accepted, 1, 0, 1);
  }
  Require(configured_8.Choose(7) == 0 && configured_1.Choose(7) > 0,
          "the C8 cost row did not price out a half-accepted draft");
  for (unsigned budget = 0; budget <= 8; ++budget) {
    Require(lone.Choose(budget, 0, 1) == configured_1.Choose(budget),
            "an occupancy override did not price like a configured lone "
            "decoder");
  }
}

void CheckPolicyConcurrency() {
  // Greedy and unseeded sampled cycles price at the live occupancy; seeded
  // sampled cycles and planned batches keep the configured row.
  Require(qfn::MtpPolicyConcurrency(false, false, false, 1) == 1 &&
              qfn::MtpPolicyConcurrency(false, true, false, 3) == 3,
          "an unplanned greedy cycle did not price at its occupancy");
  Require(qfn::MtpPolicyConcurrency(true, false, false, 1) == 1,
          "an unseeded sampled cycle did not price at its occupancy");
  Require(qfn::MtpPolicyConcurrency(true, true, false, 1) == 0,
          "a seeded sampled cycle did not keep the configured row");
  Require(qfn::MtpPolicyConcurrency(false, false, true, 2) == 0 &&
              qfn::MtpPolicyConcurrency(true, false, true, 2) == 0,
          "a planned batch did not keep its plan's row");
}

void CheckCompactProposals() {
  const std::array<float, 5> target_logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    sampling::SamplerState sampler(test.config, history);
    const auto p = sampler.Distribution(target_logits);
    for (std::uint64_t seed = 0; seed < 16; ++seed) {
      auto rng = seed;
      auto replay_rng = seed;
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &rng);
      const auto replay =
          qfn::SampleMtpProposal(candidates, sampler, &replay_rng);
      Require(q.token == replay.token && q.ids == replay.ids &&
                  q.probabilities == replay.probabilities && rng == replay_rng,
              "compact proposal is not reproducible");
      double sum = 0.0;
      std::array<double, 5> draft{};
      for (std::size_t i = 0; i < q.size; ++i) {
        const double mass = q.probabilities[i] * double(1U << 24);
        Require(mass >= 0.0 && mass == std::floor(mass),
                "proposal probability is not an exact discrete mass");
        draft[q.ids[i]] += q.probabilities[i];
        sum += q.probabilities[i];
      }
      Require(
          sum == 1.0 && q.probability > 0.0F && draft[q.token] == q.probability,
          "sampled q differs from the exported distribution");
      double accepted = 0.0, residual = 0.0;
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        accepted += std::min(probability, draft[i]);
        residual += std::max(probability - draft[i], 0.0);
      }
      Require(std::abs(accepted + residual - 1.0) < 1e-12,
              "acceptance and residual mass do not conserve the target");
      for (std::size_t i = 0; i < draft.size(); ++i) {
        const double probability = p.probability(i);
        const double correction =
            residual > 0.0
                ? (1.0 - accepted) * std::max(probability - draft[i], 0.0) /
                      residual
                : 0.0;
        Require(std::abs(std::min(probability, draft[i]) + correction -
                         probability) < 1e-12,
                "bounded draft support changes target output probability");
      }
    }
  }
  sampling::SamplerState counted({.temperature = 1,
                                  .repeat_penalty = 1.5F,
                                  .repeat_last_n = 1,
                                  .frequency_penalty = 0.25F,
                                  .presence_penalty = 0.5F},
                                 std::array<sampling::TokenId, 2>{1, 1});
  counted.Accept(std::array<sampling::TokenId, 2>{3, 3});
  counted.Accept(std::vector<sampling::TokenId>(80, 0));
  counted.Accept(4);
  std::uint64_t rng = 17;
  const auto proposal = qfn::SampleMtpProposal(candidates, counted, &rng);
  // Token 3 is outside the repetition window but keeps both generated counts.
  // Token 1 occurs only in the prompt. Token 4 gets all three penalties.
  const std::array<double, 3> adjusted{1.2 - 2 * 0.25 - 0.5, 0.1,
                                       -0.7 * 1.5 - 0.25 - 0.5};
  double sum = 0;
  for (auto logit : adjusted)
    sum += std::exp(logit);
  for (std::size_t i = 0; i < proposal.size; ++i) {
    const auto index = std::find(candidates.ids.begin(),
                                 candidates.ids.begin() + 3, proposal.ids[i]) -
                       candidates.ids.begin();
    Require(index < 3 &&
                std::abs(proposal.probabilities[i] -
                         std::exp(adjusted[index]) / sum) < 2.0 / (1U << 24),
            "compact MTP proposal lost full generated penalty counts");
  }
}

void CheckCalibratedCosts() {
  qfn::MtpLengthController single(7, 1), concurrent(7, 8);
  auto state = single.State();
  state.successes.fill(6.0F);
  state.failures.fill(4.0F);
  state.explored_depth = 7;
  state.failed_depths = 127;
  Require(single.Restore(state) && concurrent.Restore(state),
          "calibrated controller state rejected");
  Require(single.Choose(7, 0) > concurrent.Choose(7, 0),
          "concurrent verification cost did not shorten the draft chain");
  single.Observe(0, 1);
  concurrent.Observe(0, 1);
  Require(single.Choose(7) > 0 && concurrent.Choose(7) == 0,
          "measured verification cost did not change the AR break-even point");

  for (unsigned c = 1; c <= 8; ++c) {
    qfn::MtpLengthController perfect(7, c);
    for (unsigned cycle = 0; cycle < 8; ++cycle) {
      const auto length = perfect.Choose(7);
      perfect.Observe(length, length);
    }
    Require(perfect.Choose(7) == 7,
            "perfect acceptance did not retain the fastest measured width");
    for (unsigned depth : {0, 4096, 32768, 131072, 262144}) {
      const auto costs = qfn::MtpCycleCosts(depth, c);
      Require(costs[0] > 0, "AR baseline cost is not positive");
      for (std::size_t i = 1; i < costs.size(); ++i)
        Require(std::isfinite(costs[i]) && costs[i] > costs[i - 1],
                "calibrated costs do not increase with verification work");
    }
  }
}

void CheckSampledSubsetCosts() {
  qfn::MtpLengthController single(7), configured_8(7, 8);
  Require(single.Choose(7, 4096, 0, true) == single.Choose(7, 4096),
          "sampled prior widened the first cycle before any acceptance");
  auto state = single.State();
  state.successes.fill(4.0F);
  state.failures.fill(1.0F);
  state.explored_depth = 7;
  state.failed_depths = 127;
  Require(single.Restore(state) && configured_8.Restore(state),
          "sampled-head controller state rejected");
  Require(single.Choose(7, 0, 0, true) > single.Choose(7),
          "cheaper sampled proposals did not widen a profitable chain");
  for (unsigned context : {0, 2048, 4096, 32768, 131072, 262144}) {
    const auto full = qfn::MtpCycleCosts(context, 1);
    const auto subset = qfn::MtpCycleCosts(context, 1, true);
    for (unsigned width = 1; width < subset.size(); ++width) {
      Require(std::isfinite(subset[width]) && subset[width] > subset[width - 1],
              "sampled-head extrapolation crosses verification widths");
      if (context <= 32768 && width >= 2)
        Require(subset[width] / subset[0] < full[width] / full[0],
                "sampled-head cost lost its measured relative saving");
    }
    for (unsigned c = 2; c <= 8; ++c)
      Require(qfn::MtpCycleCosts(context, c, true) ==
                  qfn::MtpCycleCosts(context, c),
              "C1 sampled-head calibration changed a concurrent cost");
    for (unsigned budget = 0; budget <= 8; ++budget) {
      Require(configured_8.Choose(budget, context, 0, true) ==
                  configured_8.Choose(budget, context),
              "seeded concurrency used lone-decoder costs");
      Require(configured_8.Choose(budget, context, 1, true) ==
                  single.Choose(budget, context, 0, true),
              "lone unseeded decoder did not use the sampled-head curve");
    }
  }

  // Restoring after head fallback must replay the same policy decisions.
  // Neither switching heads nor changing the output budget adds RNG state.
  qfn::MtpLengthController replay(7);
  for (unsigned cycle = 0; cycle < 64; ++cycle) {
    Require(replay.Restore(single.State()), "sampled-head restore failed");
    const bool subset = cycle % 3 != 0;
    const unsigned budget = cycle % 9;
    const unsigned context = cycle * 2048;
    const auto length = single.Choose(budget, context, 0, subset);
    Require(length == replay.Choose(budget, context, 0, subset) &&
                length <= budget && length <= 7,
            "head fallback or snapshot restore changed bounded replay");
    const auto accepted = length == 0 ? 0 : cycle % (length + 1);
    single.Observe(accepted, length, context, 0, subset);
    replay.Observe(accepted, length, context, 0, subset);
    if (length == 0) {
      single.ObserveArToken();
      replay.ObserveArToken();
    }
    Require(single.Choose(7, context, 0, subset) ==
                replay.Choose(7, context, 0, subset),
            "restored acceptance history changed the next sampled width");
  }
}

void CheckBatchProfitability() {
  qfn::MtpLengthController a(7), b(7);
  for (unsigned i = 0; i < 12; ++i) {
    a.Observe(7, 7);
    b.Observe(7, 7);
  }
  qfn::MtpBatchController policy;
  std::array<qfn::MtpBatchController::Row, 2> rows{{{&a, 7}, {&b, 7}}};
  Require(policy.Choose(rows, 0) > 0,
          "transient occupancy paid an unnecessary plain control");
  for (unsigned i = 0; i < 2; ++i)
    policy.Observe(2, 0, policy.Choose(rows, 0), 100.0F);
  Require(policy.Choose(rows, 0) == 0,
          "stable occupancy omitted its plain control");
  // Perfect proposals are unprofitable when measured cycle cost is too high.
  for (unsigned w = 0; w <= 7; ++w)
    for (unsigned repeat = 0; repeat < 3; ++repeat)
      policy.Observe(2, 0, w, w == 0 ? 50.0F : 1000.0F);
  Require(policy.Choose(rows, 0) == 0,
          "perfect acceptance overrode measured unprofitable execution");
  policy.Observe(2, 0, 0, 1000.0F);  // Previous speculative catch-up debt.
  policy.Observe(2, 0, 0, 50.0F);
  Require(policy.Choose(rows, 0) == 0,
          "catch-up debt inflated the measured plain baseline");
  // Context and occupancy samples must not contaminate each other.
  Require(policy.Choose(rows, 32768) > 0,
          "deep batch inherited unprofitable shallow timings");
  for (unsigned i = 0; i < 32; ++i)
    policy.Observe(2, 0, 3, 60.0F);
  Require(policy.Choose(rows, 0) == 3,
          "batch did not select the profitable measured width");
  policy.Observe(2, 0, 7, 500.0F);
  policy.Observe(2, 0, 3, 2000.0F);
  Require(policy.Choose(rows, 0) == 3,
          "a width transition charged the preceding chain to the new width");
  policy.SkipCycle();  // A copy replaced the neural draft work.
  policy.Observe(2, 0, 3, 100000.0F);
  Require(policy.Choose(rows, 0) == 3,
          "lookup catch-up contaminated the neural width estimate");
  // Copies can leave the physical batch with an AR cost but no measured
  // neural widths. Repeatedly widening probes never measures a steady cost:
  // Observe correctly discards every transition's preceding catch-up work.
  qfn::MtpBatchController after_copy;
  for (unsigned i = 0; i < 3; ++i)
    after_copy.Observe(2, 0, 0, 50.0F);
  for (unsigned i = 0; i < 3; ++i) {
    Require(after_copy.Choose(rows, 0) == 1,
            "unmeasured batch widened before qualifying its cheapest draft");
    after_copy.Observe(2, 0, 1, 1000.0F);
  }
  Require(after_copy.Choose(rows, 0) > 1,
          "qualifying one draft permanently capped speculative width");
  qfn::MtpLengthController rising_a(7), rising_b(7);
  rising_a.Observe(4, 4);
  rising_b.Observe(4, 4);
  const std::array<qfn::MtpBatchController::Row, 2> rising{
      {{&rising_a, 7}, {&rising_b, 7}}};
  qfn::MtpBatchController measured;
  for (unsigned i = 0; i < 3; ++i)
    measured.Observe(2, 0, 0, 50.0F);
  for (unsigned i = 0; i < 3; ++i)
    measured.Observe(2, 0, 1, 10.0F);
  Require(measured.Choose(rising, 0) == 1,
          "acceptance probe ignored a cheaper measured batch width");
  rows[1].budget = 1;
  Require(policy.Choose(rows, 0) <= 1, "batch exceeded one member's budget");
  policy.Observe(2, 0, 0, std::numeric_limits<float>::quiet_NaN());
  Require(policy.Choose(rows, 0) <= 1, "invalid timing corrupted policy");
}

void CheckSampledOutputFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 3;
  candidates.ids = {3, 1, 4};
  candidates.logits = {1.2F, 0.1F, -0.7F};
  constexpr unsigned trials = 100000;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 73});
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = qfn::SampleMtpProposal(candidates, sampler, &draft_rng);
      const auto token = qfn::VerifyMtpProposal(logits, q, sampler).token;
      ++counts[token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      Require(std::abs(double(counts[token]) / trials - p.probability(token)) <
                  0.008,
              "proposal and correction draws distort target frequencies");
    }
  }
}

// Gemma 4 draws proposals from the request sampler without top-p, so q may
// put mass on tokens the target's top-p removes; verification must still
// reproduce the target's frequencies, and the copy keeps every other
// setting.
void CheckUnfilteredProposalFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  qfn::MtpCandidateLogits candidates;
  candidates.size = 4;
  candidates.ids = {3, 1, 4, 2};
  candidates.logits = {1.2F, 0.1F, -0.7F, 0.9F};
  constexpr unsigned trials = 100000;
  for (const float top_p : {0.6F, 0.9F}) {
    sampling::SamplingConfig config;
    config.temperature = 1.0F;
    config.top_k = 4;
    config.top_p = top_p;
    config.seed = 41;
    sampling::SamplerState sampler(config);
    const auto proposals = sampler.WithoutTopP();
    Require(proposals.config().top_p == 1.0F && proposals.config().top_k == 4 &&
                proposals.config().temperature == 1.0F,
            "WithoutTopP changed another setting");
    const auto p = sampler.Distribution(logits);
    std::array<unsigned, 5> counts{};
    for (unsigned trial = 0; trial < trials; ++trial) {
      auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
      const auto q = qfn::SampleMtpProposal(candidates, proposals, &draft_rng);
      ++counts[qfn::VerifyMtpProposal(logits, q, sampler).token];
    }
    for (unsigned token = 0; token < counts.size(); ++token) {
      Require(std::abs(double(counts[token]) / trials - p.probability(token)) <
                  0.008,
              "unfiltered proposals distort target frequencies");
    }
  }
}

// Prompt lookup proposes copied tokens as point masses. Whatever token is
// copied, verification must reproduce the target frequencies: accept with
// p(token), else resample from p without it.
void CheckPointMassFrequencies() {
  const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
  constexpr unsigned trials = 100000;
  for (const float temperature : {0.7F, 1.0F}) {
    sampling::SamplerState sampler({.temperature = temperature, .seed = 29});
    const auto p = sampler.Distribution(logits);
    for (const sampling::TokenId copied : {2U, 4U}) {
      qfn::MtpProposal proposal;
      proposal.size = 1;
      proposal.ids[0] = proposal.token = copied;
      proposal.probabilities[0] = proposal.probability = 1;
      std::array<unsigned, 5> counts{};
      for (unsigned trial = 0; trial < trials; ++trial) {
        ++counts[qfn::VerifyMtpProposal(logits, proposal, sampler).token];
      }
      for (unsigned token = 0; token < counts.size(); ++token) {
        Require(std::abs(double(counts[token]) / trials -
                         p.probability(token)) < 0.008,
                "point-mass proposals distort target frequencies");
      }
    }
  }
}

void CheckVerificationBoundaries() {
  const std::array<float, 3> logits{0, -1, -2};
  const std::array<sampling::TokenId, 3> history{1, 1, 2};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    auto config = test.config;
    config.temperature = 0.5F;
    config.seed = 6;
    // top-p and min-p cutoffs immediately around the finite FP64 boundary.
    for (const float top_p : {0.8F, 0.8668133F, 0.8668134F, 1.0F}) {
      config.top_p = top_p;
      for (const float min_p : {0.0F, 0.13533527F, 0.13533530F}) {
        config.min_p = min_p;
        sampling::SamplerState sampler(config, history);
        sampler.Accept(1);
        const auto target = sampler.Distribution(logits);
        for (unsigned draft = 0; draft < logits.size(); ++draft) {
          qfn::MtpProposal proposal;
          proposal.size = 1;
          proposal.ids[0] = proposal.token = draft;
          proposal.probabilities[0] = proposal.probability = 1;
          for (unsigned draw = 0; draw < 16; ++draw) {
            auto reference = sampler;
            auto replay = sampler;
            const bool accepted =
                reference.Uniform() < target.probability(draft);
            const auto expected =
                accepted ? draft
                         : target.SampleResidual(
                               std::span(proposal.ids).first(1),
                               std::span(proposal.probabilities).first(1),
                               reference.mutable_rng_state());
            const auto actual =
                qfn::VerifyMtpProposal(logits, proposal, sampler);
            const auto repeated =
                qfn::VerifyMtpProposal(logits, proposal, replay);
            Require(actual.accepted == accepted && actual.token == expected &&
                        sampler.rng_state() == reference.rng_state() &&
                        actual.token == repeated.token &&
                        actual.accepted == repeated.accepted &&
                        sampler.rng_state() == replay.rng_state(),
                    "MTP acceptance/residual must use the canonical AR "
                    "distribution exactly");
          }
        }
      }
    }
  }
}

int main() {
  try {
    CheckVerificationBoundaries();
    CheckLengthController();
    CheckOccupancyOverride();
    CheckPolicyConcurrency();
    CheckCalibratedCosts();
    CheckSampledSubsetCosts();
    CheckBatchProfitability();
    CheckCompactProposals();
    CheckSampledOutputFrequencies();
    CheckUnfilteredProposalFrequencies();
    CheckPointMassFrequencies();
    const std::array<float, 5> logits{0.3F, -0.2F, 1.1F, 0.7F, -1.0F};
    const std::array<sampling::TokenId, 4> history{2, 3, 2, 0};
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      if (test.config.uses_random_sampling())
        continue;
      for (unsigned accepted = 0; accepted < 4; ++accepted) {
        sampling::SamplerState ar(test.config, history);
        auto speculative = ar;
        for (unsigned i = 0; i < accepted; ++i) {
          const auto token = ar.Sample(logits);
          ar.Accept(token);
          Require(qfn::VerifyDraft(logits, static_cast<std::int32_t>(token),
                                   speculative, [](auto) { return false; }) ==
                      qfn::DraftDecision::kAccept,
                  "matching target sample was rejected");
        }
        const auto before = speculative.rng_state();
        const auto saved_history = std::vector(speculative.history().begin(),
                                               speculative.history().end());
        const auto token = ar.Sample(logits);
        Require(qfn::VerifyDraft(logits, (token + 1) % logits.size(),
                                 speculative, [](auto) { return false; }) ==
                    qfn::DraftDecision::kReject,
                "mismatching target sample was accepted");
        Require(speculative.rng_state() == before &&
                    std::vector(speculative.history().begin(),
                                speculative.history().end()) == saved_history,
                "rejected suffix consumed RNG or history");
        Require(speculative.Sample(logits) == token,
                "deferred rejection did not reproduce the AR token");
        ar.Accept(token);
        speculative.Accept(token);
        Require(ar.rng_state() == speculative.rng_state() &&
                    std::vector(ar.history().begin(), ar.history().end()) ==
                        std::vector(speculative.history().begin(),
                                    speculative.history().end()),
                "accepted prefix differs from AR state");
        auto stop_reference = speculative;
        const auto stop_token = stop_reference.Sample(logits);
        Require(qfn::VerifyDraft(
                    logits, (stop_token + 1) % logits.size(), speculative,
                    [&](auto t) {
                      return t == static_cast<std::int32_t>(stop_token);
                    }) == qfn::DraftDecision::kStop,
                "target stop must end verification even when draft differs");
        Require(speculative.rng_state() == stop_reference.rng_state(),
                "stop draw was discarded");
      }
    }
    std::cout
        << "MTP policy replay, greedy prefix/RNG/stops and compact proposal "
           "probabilities pass all sampling configurations\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
