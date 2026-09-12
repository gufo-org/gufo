#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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
      gufo::sampling::BuildDistribution(logits, frequency_config, history);
  Expect(frequency_distribution.best_token() == 1,
         "frequency penalty scales with occurrence count");

  gufo::sampling::SamplingConfig presence_config;
  presence_config.presence_penalty = 0.7F;
  const auto presence_distribution =
      gufo::sampling::BuildDistribution(logits, presence_config, history);
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
    config.top_p = 0.0F;
    config.Validate();
  } catch (const std::invalid_argument&) {
    invalid_config_rejected = true;
  }
  Expect(invalid_config_rejected, "invalid sampling controls are rejected");
}

void TestZeroDrawAndNonFiniteCandidates() {
  // Inverse xorshift state whose next 24-bit uniform is exactly zero.
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
      auto replay = sampler;
      for (int step = 0; step < 16; ++step) {
        const auto distribution =
            gufo::sampling::BuildDistribution(logits, config, history);
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
        history.push_back(token);
        sampler.Accept(token);
        replay.Accept(token);
      }
    }
  }
}

}  // namespace

int main() {
  TestGreedySelectsFiniteArgmaxWithoutAdvancingRng();
  TestDefaultConfigPreservesGreedyDecoding();
  TestTemperatureSamplingIsDeterministicAndNonGreedy();
  TestTopKFiltersTheCandidateSet();
  TestTopPFiltersByCumulativeProbability();
  TestMinPFiltersRelativeToTheBestToken();
  TestMinKeepProvidesAFilterFloor();
  TestRepeatPenaltyUsesCommittedHistory();
  TestFrequencyAndPresencePenaltiesUseCounts();
  TestRepeatWindowIsBounded();
  TestFastGreedyMatchesNormalizedPenaltyRoute();
  TestFastTopKSamplingNeverEscapesSelectedSet();
  TestFastMinPSamplingNeverEscapesRelativeThreshold();
  TestSeedAndStateAreRequestLocal();
  TestDistributionIsNormalized();
  TestSamplingFailsClosedOnInvalidInputs();
  TestZeroDrawAndNonFiniteCandidates();
  TestStrategyReplayAgainstReference();
  std::cout << "All logit sampler tests passed.\n";
  return 0;
}
