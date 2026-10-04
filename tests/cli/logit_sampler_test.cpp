#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void TestGreedySelectsFiniteArgmaxWithoutAdvancingRng() {
  const std::array<float, 4> logits = {
      -1.0F, 5.0F, std::numeric_limits<float>::quiet_NaN(), 2.0F};
  std::uint64_t rng_state = 1234;

  const auto token = gufo::sampling::SampleLogits(logits, 0.0F, &rng_state);

  Expect(token == 1, "greedy sampling selects the finite argmax");
  Expect(rng_state == 1234, "greedy sampling does not consume RNG state");
}

void TestDefaultConfigPreservesGreedyDecoding() {
  const std::array<float, 4> logits = {-1.0F, 5.0F, 3.0F, 2.0F};
  gufo::sampling::SamplerState sampler;

  const auto before = sampler.rng_state();
  const auto token = sampler.Sample(logits);

  Expect(token == 1, "default sampler is greedy");
  Expect(sampler.rng_state() == before,
         "default sampler does not consume RNG state");
}

void TestExternalArgmaxRequiresAppliedPenalties() {
  using gufo::sampling::SamplerState;
  const std::array<float, 2> logits{1.0F, 0.75F};
  SamplerState sampler({.presence_penalty = 1.5F});
  sampler.Accept(0);
  Expect(sampler.Sample(logits) == 1, "penalty changes the greedy winner");
  Expect(!sampler.CanSelectArgmax(0), "raw argmax cannot bypass penalties");
  Expect(sampler.CanSelectArgmax(1, /*penalties_applied=*/true),
         "an exactly penalized argmax avoids downloading verification logits");
  SamplerState sampled({.temperature = .7F, .presence_penalty = 1.5F});
  Expect(!sampled.CanSelectArgmax(1, /*penalties_applied=*/true),
         "applied penalties never bypass random sampling");
}

void TestTemperatureSamplingIsDeterministicAndNonGreedy() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  bool saw_first = false;
  bool saw_non_first = false;

  for (std::uint64_t seed = 1; seed <= 128; ++seed) {
    auto left_state = seed;
    auto right_state = seed;
    const auto left = gufo::sampling::SampleLogits(logits, 1.0F, &left_state);
    const auto right = gufo::sampling::SampleLogits(logits, 1.0F, &right_state);
    Expect(left == right && left_state == right_state,
           "temperature sampling is reproducible from the same RNG state");
    saw_first = saw_first || left == 0;
    saw_non_first = saw_non_first || left != 0;
  }

  Expect(saw_first && saw_non_first,
         "temperature sampling does not collapse to greedy argmax");
}

void TestTopKFiltersTheCandidateSet() {
  const std::array<float, 5> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-k retains exactly k candidates");
  Expect(distribution.probability(0) > 0.0 && distribution.probability(1) > 0.0,
         "top-k retains the highest-logit candidates");
  Expect(distribution.probability(2) == 0.0,
         "top-k removes lower-logit candidates");
}

void TestTopPFiltersByCumulativeProbability() {
  const std::array<float, 3> logits = {3.0F, 2.0F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_p = 0.7F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "top-p retains the smallest cumulative-probability prefix");
  Expect(distribution.probability(2) == 0.0,
         "top-p removes the low-probability tail");
  config.top_p = 0.0F;
  for (const auto floor : {0U, 1U, 2U}) {
    config.min_keep = floor;
    const auto zero = gufo::sampling::BuildDistribution(logits, config);
    Expect(zero.entries().size() == std::max(floor, 1U),
           "top-p zero preserves the minimum token count");
    Expect(zero.best_token() == 0, "top-p zero keeps the highest probability");
  }
}

void TestMinPFiltersRelativeToTheBestToken() {
  const std::array<float, 3> logits = {3.0F, 2.4F, 1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 2,
         "min-p retains candidates above the relative threshold");
  Expect(distribution.probability(2) == 0.0,
         "min-p removes candidates far below the best token");
}

void TestMinKeepProvidesAFilterFloor() {
  const std::array<float, 4> logits = {5.0F, 4.0F, 3.0F, 2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 1;
  config.top_p = 0.1F;
  config.min_p = 0.99F;
  config.min_keep = 3;

  const auto distribution = gufo::sampling::BuildDistribution(logits, config);

  Expect(distribution.entries().size() == 3,
         "min-keep prevents enabled filters from shrinking below its floor");
}

void TestTemperatureBeforeProbabilityFilters() {
  // Independent closed-form expectations for softmax([0,-1,-2] / T).
  // Also exercise the bounded (>1024 vocabulary) candidate selector.
  for (const std::size_t size : {3U, 2048U}) {
    std::vector<float> logits(size, -1000.0F);
    logits[0] = 0.0F;
    logits[1] = -1.0F;
    logits[2] = -2.0F;
    for (const bool nucleus : {false, true}) {
      gufo::sampling::SamplingConfig config{.temperature = 0.5F, .seed = 42};
      if (nucleus)
        config.top_p = 0.8F;
      else
        config.min_p = 0.2F;
      gufo::sampling::SamplerState sampler(config);
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            sampler.Distribution(logits)}) {
        Expect(distribution.entries().size() == 1 &&
                   distribution.probability(0) == 1.0,
               "temperature sharpens probabilities before filtering");
      }
      for (unsigned draw = 0; draw < 32; ++draw)
        Expect(sampler.Sample(logits) == 0,
               "ordinary decode uses the tempered candidate set");

      config.temperature = 2.0F;
      config.top_p = nucleus ? 0.8F : 1.0F;
      config.min_p = nucleus ? 0.0F : 0.5F;
      const double first = 1.0 / (1.0 + std::exp(-0.5));
      for (const auto distribution :
           {gufo::sampling::BuildDistribution(logits, config),
            gufo::sampling::SamplerState(config).Distribution(logits)}) {
        Expect(
            distribution.entries().size() == 2 &&
                std::abs(distribution.probability(0) - first) < 1e-12 &&
                std::abs(distribution.probability(1) - (1.0 - first)) < 1e-12,
            "temperature flattens probabilities before filtering");
      }
    }
  }
}

void TestRepeatPenaltyUsesCommittedHistory() {
  const std::array<float, 3> logits = {2.0F, 1.0F, 0.0F};
  const std::array<gufo::sampling::TokenId, 1> history = {0};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 3.0F;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 1,
         "repeat penalty can demote a recently used token");
}

void TestFrequencyAndPresencePenaltiesUseCounts() {
  const std::array<float, 2> logits = {2.0F, 1.4F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 0};

  gufo::sampling::SamplingConfig frequency_config;
  frequency_config.frequency_penalty = 0.4F;
  const auto frequency_distribution =
      gufo::sampling::BuildDistribution(logits, frequency_config, {}, history);
  Expect(frequency_distribution.best_token() == 1,
         "frequency penalty scales with occurrence count");

  gufo::sampling::SamplingConfig presence_config;
  presence_config.presence_penalty = 0.7F;
  const auto presence_distribution =
      gufo::sampling::BuildDistribution(logits, presence_config, {}, history);
  Expect(presence_distribution.best_token() == 1,
         "presence penalty applies once when a token is present");
}

void TestRepeatWindowIsBounded() {
  const std::array<float, 2> logits = {2.0F, 2.1F};
  const std::array<gufo::sampling::TokenId, 2> history = {0, 1};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 2.0F;
  config.repeat_last_n = 1;

  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);

  Expect(distribution.best_token() == 0,
         "repeat-last-n excludes tokens outside the configured window");
}

void TestFastGreedyMatchesNormalizedPenaltyRoute() {
  const std::array<float, 5> logits = {3.0F, 2.9F, 2.8F, 2.7F, 2.6F};
  const std::array<gufo::sampling::TokenId, 4> history = {0, 0, 1, 3};
  gufo::sampling::SamplingConfig config;
  config.repeat_penalty = 1.5F;
  config.frequency_penalty = 0.2F;
  config.presence_penalty = 0.1F;
  gufo::sampling::SamplerState sampler(config, history);

  const auto expected =
      gufo::sampling::BuildDistribution(logits, config, history).best_token();
  const auto actual = sampler.Sample(logits);

  Expect(actual == expected,
         "linear greedy route matches the normalized penalty route");
}

void TestFastTopKSamplingNeverEscapesSelectedSet() {
  const std::array<float, 6> logits = {8.0F, 7.0F, 1.0F, 0.0F, -1.0F, -2.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.top_k = 2;

  bool saw_first = false;
  bool saw_second = false;
  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    const auto token = sampler.Sample(logits);
    Expect(token <= 1, "top-k fast route retains only selected candidates");
    saw_first = saw_first || token == 0;
    saw_second = saw_second || token == 1;
  }
  Expect(saw_first && saw_second,
         "top-k fast route samples the retained distribution");
}

void TestFastMinPSamplingNeverEscapesRelativeThreshold() {
  const std::array<float, 4> logits = {3.0F, 2.5F, 1.0F, -4.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.min_p = 0.5F;

  for (std::int64_t seed = 1; seed <= 256; ++seed) {
    config.seed = seed;
    gufo::sampling::SamplerState sampler(config);
    Expect(sampler.Sample(logits) <= 1,
           "min-p linear route rejects tokens below the relative threshold");
  }
}

void TestSeedAndStateAreRequestLocal() {
  const std::array<float, 3> logits = {0.0F, 0.0F, 0.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 1.0F;
  config.seed = 42;
  config.repeat_last_n = 2;
  gufo::sampling::SamplerState left(config);
  gufo::sampling::SamplerState right(config);

  for (int sample = 0; sample < 16; ++sample) {
    Expect(left.Sample(logits) == right.Sample(logits),
           "equal seeds produce equal token sequences");
  }
  left.Accept(1);
  left.Accept(2);
  left.Accept(0);
  Expect(left.history().size() == 2 && left.history()[0] == 2 &&
             left.history()[1] == 0,
         "sampler retains only its bounded committed history");
  Expect(right.history().empty(), "sampler histories are request-local");
}

void TestDistributionIsNormalized() {
  const std::array<float, 4> logits = {2.0F, 1.0F, 0.0F, -1.0F};
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.8F;
  const auto distribution = gufo::sampling::BuildDistribution(logits, config);
  double sum = 0.0;
  for (const auto& entry : distribution.entries()) {
    sum += entry.value;
  }
  Expect(std::abs(sum - 1.0) < 1e-12,
         "post-filter probabilities are normalized");
}

void TestGreedySparsePenaltyWalk() {
  using namespace gufo::sampling;
  std::vector<float> logits(16385);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = float(int((i * 7919) % 65521) - 32768) / 32768;
  logits[0] = logits[128] = std::numeric_limits<float>::quiet_NaN();
  logits[777] = std::numeric_limits<float>::infinity();
  for (const float repeat : {.7F, 1.F, 1.3F}) {
    for (const float penalty : {-1.5F, 1.5F}) {
      SamplingConfig config{.seed = 73,
                            .repeat_penalty = repeat,
                            .repeat_last_n = 3,
                            .frequency_penalty = penalty / 3,
                            .presence_penalty = penalty};
      const std::vector<TokenId> history{0, 128, 777, 16384, UINT32_MAX};
      SamplerState sampler(config, history);
      std::vector<TokenId> generated;
      for (unsigned step = 0; step < 16; ++step) {
        const auto expected =
            BuildDistribution(logits, config, history, generated);
        Expect(
            sampler.Sample(logits) == expected.best_token(),
            "sparse greedy walk matches independent distribution construction");
        const TokenId token =
            step % 3 == 0 ? 128 : (step * 977) % logits.size();
        generated.push_back(token);
        sampler.Accept(token);
      }
    }
  }
}

void TestSamplingFailsClosedOnInvalidInputs() {
  bool empty_rejected = false;
  try {
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits({}, 1.0F, &rng_state);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  Expect(empty_rejected, "empty logits are rejected");

  bool missing_rng_rejected = false;
  try {
    const std::array<float, 2> logits = {0.0F, 1.0F};
    (void)gufo::sampling::SampleLogits(logits, 1.0F, nullptr);
  } catch (const std::invalid_argument&) {
    missing_rng_rejected = true;
  }
  Expect(missing_rng_rejected, "sampled decoding requires RNG state");

  bool non_finite_rejected = false;
  try {
    const std::array<float, 2> logits = {
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::infinity()};
    std::uint64_t rng_state = 1;
    (void)gufo::sampling::SampleLogits(logits, 1.0F, &rng_state);
  } catch (const std::runtime_error&) {
    non_finite_rejected = true;
  }
  Expect(non_finite_rejected, "an all-non-finite distribution is rejected");

  bool invalid_config_rejected = false;
  try {
    gufo::sampling::SamplingConfig config;
    config.top_p = -0.01F;
    config.Validate();
  } catch (const std::invalid_argument&) {
    invalid_config_rejected = true;
  }
  Expect(invalid_config_rejected, "invalid sampling controls are rejected");
}

void TestZeroDrawAndNonFiniteCandidates() {
  // Inverse xorshift state whose next 53-bit uniform is exactly zero.
  constexpr std::uint64_t zero_draw_state = UINT64_C(0x98d76a164d99a710);
  const std::array<float, 3> cold_logits{-1000.0F, 0.0F, -1.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 0});
  sampler.SetRngState(zero_draw_state);
  Expect(sampler.Sample(cold_logits) == 1,
         "zero draw must not select underflowed zero probability");
  const std::array<float, 5> masked_logits{
      std::numeric_limits<float>::quiet_NaN(), 4.0F,
      std::numeric_limits<float>::infinity(), 0.0F,
      -std::numeric_limits<float>::infinity()};
  for (const auto seed : {0, 73, 808}) {
    gufo::sampling::SamplerState masked({.temperature = 1.0F, .seed = seed});
    auto rng = masked.rng_state();
    const auto expected =
        gufo::sampling::BuildDistribution(masked_logits, masked.config())
            .Sample(&rng);
    Expect(masked.Sample(masked_logits) == expected,
           "linear CPU sampling must ignore nonfinite candidates");
  }
}

void TestStrategyReplayAgainstReference() {
  const std::array<float, 5> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (const auto seed : {0, 1, 73, 808}) {
      auto config = test.config;
      config.seed = seed;
      std::vector<gufo::sampling::TokenId> history{1, 1, 2, 3};
      gufo::sampling::SamplerState sampler(config, history);
      std::vector<gufo::sampling::TokenId> generated;
      auto replay = sampler;
      for (int step = 0; step < 16; ++step) {
        const auto distribution = gufo::sampling::BuildDistribution(
            logits, config, history, generated);
        std::vector<gufo::sampling::Probability> ordered(
            distribution.entries().begin(), distribution.entries().end());
        if (config.top_k == 0 && config.top_p == 1.0F &&
            !(config.min_p > 0.0F && config.min_keep > 1)) {
          std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
        }
        auto rng = sampler.rng_state();
        double threshold = gufo::sampling::Uniform(&rng);
        auto expected = distribution.best_token();
        if (config.temperature > 0.0F) {
          for (const auto& entry : ordered) {
            if (entry.value > 0.0 && threshold < entry.value) {
              expected = entry.token;
              break;
            }
            threshold -= entry.value;
          }
        }
        const auto token = sampler.Sample(logits);
        Expect(token == expected && token == replay.Sample(logits), test.name);
        generated.push_back(token);
        sampler.Accept(token);
        replay.Accept(token);
      }
    }
  }
}

void TestFilteredDistributionAgainstFullSort() {
  std::vector<float> logits(4099);
  const std::array<gufo::sampling::TokenId, 6> history{8, 8, 8, 11, 17, 29};
  for (bool concentrated : {false, true}) {
    for (std::size_t i = 0; i < logits.size(); ++i)
      logits[i] = concentrated ? -(float)((i * 73) % logits.size()) * 0.2F
                               : std::sin((float)i * 0.7F);
    for (const auto& test : gufo::test::QwenSamplingCases()) {
      gufo::sampling::SamplerState sampler(test.config, history);
      const auto rng = sampler.rng_state();
      const auto expected =
          gufo::sampling::BuildDistribution(logits, test.config, history);
      const auto actual = sampler.Distribution(logits);
      const auto positive = std::ranges::count_if(
          expected.entries(), [](const auto& e) { return e.value > 0; });
      Expect(actual.entries().size() == static_cast<std::size_t>(positive),
             test.name);
      for (const auto& entry : expected.entries())
        Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
               test.name);
      Expect(sampler.rng_state() == rng,
             "materializing probabilities consumes no random draws");
    }
  }
}

void TestBlockSkippedTopKAgainstFullSort() {
  // Large rows exercise block skipping: ties straddling the top-k boundary,
  // non-finite entries and penalties that lift a low token into the set.
  std::vector<float> logits(70001);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = std::floor(std::sin(static_cast<float>(i) * 0.37F) * 8.0F);
  logits[5] = std::numeric_limits<float>::quiet_NaN();
  logits[640] = std::numeric_limits<float>::infinity();
  logits[641] = -std::numeric_limits<float>::infinity();
  logits[69999] = 9.0F;
  const std::array<gufo::sampling::TokenId, 7> history{
      3, 700, 701, 33333, 50000, 69999, 70000};
  for (const gufo::sampling::SamplingConfig config :
       {gufo::sampling::SamplingConfig{
            .temperature = 1.0F, .top_k = 64, .top_p = 0.95F},
        gufo::sampling::SamplingConfig{.temperature = 1.0F, .top_k = 1},
        gufo::sampling::SamplingConfig{.temperature = 0.7F, .top_k = 300},
        gufo::sampling::SamplingConfig{.temperature = 1.0F,
                                       .top_k = 64,
                                       .repeat_penalty = 1.3F,
                                       .presence_penalty = -9.0F},
        gufo::sampling::SamplingConfig{.temperature = 0.8F,
                                       .top_k = 20,
                                       .repeat_penalty = 0.5F,
                                       .frequency_penalty = 2.0F}}) {
    gufo::sampling::SamplerState sampler(config, history);
    sampler.Accept(std::span(history).first(3));
    const auto expected = gufo::sampling::BuildDistribution(
        logits, config, history, std::span(history).first(3));
    const auto actual = sampler.Distribution(logits);
    Expect(actual.entries().size() == expected.entries().size(),
           "block-skipped top-k keeps the reference candidate count");
    for (const auto& entry : expected.entries())
      Expect(std::abs(actual.probability(entry.token) - entry.value) < 1e-12,
             "block-skipped top-k keeps the reference distribution");
  }
}

void TestDeferredResidualReplay() {
  const std::array<float, 3> logits{100.0F, -100.0F, -100.0F};
  gufo::sampling::SamplerState sampler({.temperature = 1.0F, .seed = 7});
  // A retained residual must win over a fresh draw, even when that fresh
  // draw would almost certainly select a different token.
  sampler.DeferSample(2);
  const auto rng = sampler.rng_state();
  auto copied = sampler;
  gufo::sampling::SamplerState assigned;
  assigned = sampler;
  bool duplicate_rejected = false;
  try {
    sampler.DeferSample(1);
  } catch (const std::logic_error&) {
    duplicate_rejected = true;
  }
  Expect(duplicate_rejected, "pending residual cannot be overwritten");
  gufo::sampling::SamplerState published({.temperature = 1.0F});
  published.Accept(1);
  published.CopyDrawStateFrom(sampler);
  Expect(published.history().size() == 1 && published.history()[0] == 1,
         "publishing draw state leaves committed history unchanged");
  for (auto* state : {&sampler, &copied, &assigned, &published}) {
    Expect(state->Sample(logits) == 2 && state->rng_state() == rng,
           "copied residual is consumed without another RNG draw");
    Expect(state->Sample(logits) == 0 && state->rng_state() != rng,
           "residual is consumed only once");
  }
  sampler.DeferSample(2);
  sampler.ResetHistory({});
  Expect(sampler.Sample(logits) == 0, "new history discards pending residual");
  sampler.DeferSample(2);
  sampler.SetRngState(rng);
  Expect(sampler.Sample(logits) == 0, "RNG reset discards pending residual");
}

void TestCanonicalTargetDistribution() {
  std::vector<float> logits(4099, -1000.0F);
  logits[0] = 0;
  logits[17] = -1;
  logits[4098] = -2;
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    for (float temperature : {0.5F, 0.99999994F, 2.0F}) {
      auto config = test.config;
      config.temperature = temperature;
      config.seed = 19;
      gufo::sampling::SamplerState sampler(config);
      for (unsigned step = 0; step < 16; ++step) {
        const auto distribution = sampler.Distribution(logits);
        auto rng = sampler.rng_state();
        const auto expected = distribution.Sample(&rng);
        Expect(sampler.Sample(logits) == expected && sampler.rng_state() == rng,
               "AR and verification share exact support, CDF and RNG draws");
        sampler.Accept(expected);
      }
    }
  }
}

void TestResponsePenaltyScope() {
  using gufo::sampling::SamplerState;
  using gufo::sampling::TokenId;
  const std::array<float, 3> logits{3, 2, 1};
  SamplerState sampler({.temperature = 1,
                        .seed = 7,
                        .repeat_last_n = 0,
                        .frequency_penalty = .5F,
                        .presence_penalty = .25F},
                       std::vector<TokenId>{0, 0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "frequency/presence must not penalize prompt tokens");
  sampler.Accept(std::vector<TokenId>{0, 0});
  sampler.Accept(std::vector<TokenId>(80, 2));
  const std::array<float, 3> expected_logits{1.75F, 2, -39.25F};
  const auto expected =
      gufo::sampling::BuildDistribution(expected_logits, {.temperature = 1});
  for (const auto token : {0U, 1U, 2U})
    Expect(std::abs(sampler.Distribution(logits).probability(token) -
                    expected.probability(token)) < 1e-12,
           "full response counts must outlive the repetition window");
  auto rejected = sampler;
  rejected.Accept(1);
  Expect(sampler.Distribution(logits).best_token() == 1 &&
             rejected.Distribution(logits).best_token() == 0,
         "tentative penalty counts must not leak through speculative rollback");
  auto replay = sampler;
  Expect(replay.Sample(logits) == sampler.Sample(logits),
         "penalty copies preserve RNG replay");
  sampler.ResetHistory(std::vector<TokenId>{0, 0});
  Expect(sampler.Distribution(logits).best_token() == 0,
         "new request resets generated counts");
}

}  // namespace

int main() {
  bool sub_float_resolution = false;
  for (std::uint64_t seed = 1; seed < 100; ++seed) {
    auto state = seed;
    auto reference = seed;
    const auto bits = gufo::sampling::NextRandom(&reference);
    const double draw = gufo::sampling::Uniform(&state);
    Expect(draw >= 0 && draw < 1 && state == reference,
           "uniform stays half-open and consumes one RNG word");
    Expect(draw == static_cast<double>(bits >> 11) * 0x1.0p-53,
           "uniform retains all 53 random mantissa bits");
    sub_float_resolution |=
        draw != static_cast<double>(static_cast<float>(draw));
  }
  Expect(sub_float_resolution, "uniform must exceed float resolution");
  TestCanonicalTargetDistribution();
  TestResponsePenaltyScope();
  TestGreedySelectsFiniteArgmaxWithoutAdvancingRng();
  TestDefaultConfigPreservesGreedyDecoding();
  TestExternalArgmaxRequiresAppliedPenalties();
  TestTemperatureSamplingIsDeterministicAndNonGreedy();
  TestTopKFiltersTheCandidateSet();
  TestTopPFiltersByCumulativeProbability();
  TestMinPFiltersRelativeToTheBestToken();
  TestMinKeepProvidesAFilterFloor();
  TestTemperatureBeforeProbabilityFilters();
  TestRepeatPenaltyUsesCommittedHistory();
  TestFrequencyAndPresencePenaltiesUseCounts();
  TestRepeatWindowIsBounded();
  TestFastGreedyMatchesNormalizedPenaltyRoute();
  TestFastTopKSamplingNeverEscapesSelectedSet();
  TestFastMinPSamplingNeverEscapesRelativeThreshold();
  TestSeedAndStateAreRequestLocal();
  TestDistributionIsNormalized();
  TestGreedySparsePenaltyWalk();
  TestSamplingFailsClosedOnInvalidInputs();
  TestZeroDrawAndNonFiniteCandidates();
  TestStrategyReplayAgainstReference();
  TestFilteredDistributionAgainstFullSort();
  TestBlockSkippedTopKAgainstFullSort();
  TestDeferredResidualReplay();
  std::cout << "All logit sampler tests passed.\n";
  return 0;
}
