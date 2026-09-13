#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/models/qwen27b/sampling_cases.hpp"

namespace {

void CheckHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

std::uint32_t Sample(std::span<const float> logits,
                     const gufo::hip::GpuSamplingParameters& parameters,
                     std::span<const std::uint32_t> penalty_tokens = {},
                     std::span<const std::uint32_t> penalty_counts = {}) {
  if (penalty_tokens.size() != penalty_counts.size()) {
    throw std::invalid_argument("penalty vectors must have equal lengths");
  }

  float* device_logits = nullptr;
  std::uint32_t* device_token = nullptr;
  CheckHip(hipMalloc(&device_logits, logits.size_bytes()), "allocate logits");
  CheckHip(hipMalloc(&device_token, sizeof(std::uint32_t)), "allocate token");
  CheckHip(hipMemcpy(device_logits, logits.data(), logits.size_bytes(),
                     hipMemcpyHostToDevice),
           "upload logits");

  gufo::hip::GpuSamplingWorkspace workspace;
  gufo::hip::AllocateGpuSamplingWorkspace(&workspace, logits.size(),
                                          penalty_tokens.size());
  gufo::hip::LaunchGPUSampling(device_logits, device_token, logits.size(),
                               parameters, penalty_tokens.data(),
                               penalty_counts.data(), penalty_tokens.size(),
                               &workspace);

  std::uint32_t token = 0;
  CheckHip(
      hipMemcpy(&token, device_token, sizeof(token), hipMemcpyDeviceToHost),
      "download token");
  gufo::hip::FreeGpuSamplingWorkspace(&workspace);
  CheckHip(hipFree(device_token), "free token");
  CheckHip(hipFree(device_logits), "free logits");
  return token;
}

struct SpeculativeSample {
  std::uint32_t token{0};
  bool accepted{false};
};

SpeculativeSample SampleSpeculatively(
    std::span<const float> logits,
    const gufo::hip::GpuSamplingParameters& parameters,
    std::uint32_t draft_token, float draft_token_probability,
    std::span<const std::uint32_t> draft_candidate_ids,
    std::span<const float> draft_candidate_probabilities,
    float acceptance_uniform, float residual_uniform,
    std::span<const std::uint32_t> penalty_tokens = {},
    std::span<const std::uint32_t> penalty_counts = {}) {
  if (draft_candidate_ids.size() != draft_candidate_probabilities.size() ||
      penalty_tokens.size() != penalty_counts.size()) {
    throw std::invalid_argument("sparse vectors must have equal lengths");
  }

  float* device_logits = nullptr;
  std::uint32_t* device_token = nullptr;
  std::uint32_t* device_accepted = nullptr;
  CheckHip(hipMalloc(&device_logits, logits.size_bytes()), "allocate logits");
  CheckHip(hipMalloc(&device_token, sizeof(std::uint32_t)), "allocate token");
  CheckHip(hipMalloc(&device_accepted, sizeof(std::uint32_t)),
           "allocate acceptance");
  CheckHip(hipMemcpy(device_logits, logits.data(), logits.size_bytes(),
                     hipMemcpyHostToDevice),
           "upload logits");

  gufo::hip::GpuSamplingWorkspace workspace;
  gufo::hip::AllocateGpuSamplingWorkspace(
      &workspace, logits.size(),
      std::max(draft_candidate_ids.size(), penalty_tokens.size()));
  gufo::hip::LaunchGPUSpeculativeSampling(
      device_logits, device_token, device_accepted, logits.size(), parameters,
      draft_token, draft_token_probability, draft_candidate_ids.data(),
      draft_candidate_probabilities.data(), draft_candidate_ids.size(),
      acceptance_uniform, residual_uniform, penalty_tokens.data(),
      penalty_counts.data(), penalty_tokens.size(), &workspace);

  SpeculativeSample result;
  std::uint32_t accepted = 0;
  CheckHip(hipMemcpy(&result.token, device_token, sizeof(result.token),
                     hipMemcpyDeviceToHost),
           "download speculative token");
  CheckHip(hipMemcpy(&accepted, device_accepted, sizeof(accepted),
                     hipMemcpyDeviceToHost),
           "download speculative acceptance");
  result.accepted = accepted != 0U;
  gufo::hip::FreeGpuSamplingWorkspace(&workspace);
  CheckHip(hipFree(device_accepted), "free acceptance");
  CheckHip(hipFree(device_token), "free token");
  CheckHip(hipFree(device_logits), "free logits");
  return result;
}

void TestGreedySparsePenalty() {
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F};
  const std::vector<std::uint32_t> penalty_tokens = {0U};
  const std::vector<std::uint32_t> penalty_counts = {1U};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.repeat_penalty = 2.0F;

  Expect(Sample(logits, parameters, penalty_tokens, penalty_counts) == 1U,
         "sparse repeat penalty changes the greedy winner");
}

void TestLinearTemperatureAndMinP() {
  const std::vector<float> logits = {3.0F, 2.5F, 0.0F, -1.0F};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;
  parameters.min_p = 0.5F;
  parameters.uniform = 0.9F;

  Expect(Sample(logits, parameters) == 1U,
         "linear temperature route samples within the min-p set");
}

void TestOrderedFilterChain() {
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;
  parameters.top_k = 3;
  parameters.top_p = 0.8F;
  parameters.min_p = 0.1F;
  parameters.uniform = 0.99F;

  Expect(Sample(logits, parameters) == 1U,
         "ordered route applies top-k, top-p, min-p, then temperature");
}

void TestTopKOneIsDeterministic() {
  const std::vector<float> logits = {1.0F, 5.0F, 4.0F};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;
  parameters.top_k = 1;
  parameters.uniform = 0.99F;

  Expect(Sample(logits, parameters) == 1U,
         "top-k one remains deterministic argmax");
}

void TestTopKOneRespectsMinimumKeep() {
  const std::vector<float> logits = {std::log(0.20F), std::log(0.80F)};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;
  parameters.top_k = 1;
  parameters.min_keep = 2;
  parameters.uniform = 0.99F;
  Expect(Sample(logits, parameters) == 0U,
         "min-keep overrides top-k one instead of silently selecting argmax");
  const std::vector<std::uint32_t> ids{0, 1};
  const std::vector<float> probabilities{0.8F, 0.2F};
  const auto result = SampleSpeculatively(logits, parameters, 0, 0.8F, ids,
                                          probabilities, 0.9F, 0.5F);
  Expect(!result.accepted && result.token == 1,
         "top-k one with min-keep uses lossless sampled verification");
}

void CheckSamplingDistribution(const gufo::sampling::SamplingConfig& config) {
  const std::vector<float> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  const std::vector<std::uint32_t> history{1, 1, 2, 3};
  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history);
  const gufo::sampling::SamplerState sampler(config, history);
  std::vector<std::uint32_t> ids, counts;
  for (std::uint32_t token = 0; token < logits.size(); ++token) {
    const auto count = std::ranges::count(sampler.history(), token);
    if (count > 0 && config.penalties_enabled()) {
      ids.push_back(token);
      counts.push_back(static_cast<std::uint32_t>(count));
    }
  }
  std::vector<gufo::sampling::Probability> ordered(
      distribution.entries().begin(), distribution.entries().end());
  const bool linear = config.top_k == 0 && config.top_p == 1.0F &&
                      !(config.min_p > 0.0F && config.min_keep > 1);
  if (linear)
    std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
  const auto quantile = [](const auto& entries, double uniform) {
    auto selected = entries.front().token;
    for (const auto& entry : entries) {
      if (entry.value <= 0.0)
        continue;
      selected = entry.token;
      if (uniform < entry.value)
        break;
      uniform -= entry.value;
    }
    return selected;
  };
  gufo::hip::GpuSamplingParameters parameters{
      .temperature = config.temperature,
      .top_k = config.top_k,
      .top_p = config.top_p,
      .min_p = config.min_p,
      .min_keep = config.min_keep,
      .repeat_penalty = config.repeat_penalty,
      .frequency_penalty = config.frequency_penalty,
      .presence_penalty = config.presence_penalty,
  };
  for (const float uniform : {0.0F, 0.23F, 0.91F, 0.99999994F}) {
    parameters.uniform = uniform;
    Expect(
        Sample(logits, parameters, ids, counts) == quantile(ordered, uniform),
        "AR quantile differs from the double-precision reference");
  }
  if (!config.uses_random_sampling())
    return;  // Greedy proposal verification is covered by the verifier/model.

  // Deliberately different sparse q: includes filtered-out proposals and omits
  // target tokens. Test p/q on both sides of each acceptance boundary, then
  // every residual CDF boundary. This checks probabilities without Monte Carlo
  // noise or an expensive model-generation sweep.
  const std::vector<std::uint32_t> draft_ids{1, 3, 4};
  const std::vector<float> draft_probabilities{0.5F, 0.25F, 0.25F};
  constexpr double tolerance = 2e-5;
  std::size_t rejected_candidate = draft_ids.size();
  float rejection_draw = 0.0F;
  for (std::size_t i = 0; i < draft_ids.size(); ++i) {
    const double ratio = std::min(
        1.0, distribution.probability(draft_ids[i]) / draft_probabilities[i]);
    if (ratio > tolerance) {
      const auto result = SampleSpeculatively(
          logits, parameters, draft_ids[i], draft_probabilities[i], draft_ids,
          draft_probabilities, static_cast<float>(ratio - tolerance), 0.5F, ids,
          counts);
      Expect(result.accepted && result.token == draft_ids[i],
             "DFlash2 acceptance below p/q differs from reference");
    }
    if (ratio < 1.0 - tolerance) {
      const auto result = SampleSpeculatively(
          logits, parameters, draft_ids[i], draft_probabilities[i], draft_ids,
          draft_probabilities, static_cast<float>(ratio + tolerance), 0.5F, ids,
          counts);
      Expect(!result.accepted,
             "DFlash2 rejection above p/q differs from reference");
      rejected_candidate = i;
      rejection_draw = static_cast<float>((1.0 + ratio) * 0.5);
    }
  }
  Expect(rejected_candidate < draft_ids.size(),
         "test q must exercise rejection");
  auto residual = ordered;
  double total = 0.0;
  for (auto& entry : residual) {
    for (std::size_t i = 0; i < draft_ids.size(); ++i)
      if (entry.token == draft_ids[i])
        entry.value = std::max(0.0, entry.value - draft_probabilities[i]);
    total += entry.value;
  }
  Expect(total > 0.0, "test q must leave positive residual mass");
  for (auto& entry : residual)
    entry.value /= total;
  std::vector<float> draws{0.0F, 0.23F, 0.91F, 0.99999994F};
  double cumulative = 0.0;
  for (const auto& entry : residual) {
    cumulative += entry.value;
    if (entry.value > 2 * tolerance) {
      draws.push_back(static_cast<float>(cumulative - entry.value * 0.5));
      if (cumulative > tolerance && cumulative < 1.0 - tolerance) {
        draws.push_back(static_cast<float>(cumulative - tolerance));
        draws.push_back(static_cast<float>(cumulative + tolerance));
      }
    }
  }
  for (const float uniform : draws) {
    const auto result = SampleSpeculatively(
        logits, parameters, draft_ids[rejected_candidate],
        draft_probabilities[rejected_candidate], draft_ids, draft_probabilities,
        rejection_draw, uniform, ids, counts);
    Expect(!result.accepted && result.token == quantile(residual, uniform),
           "DFlash2 residual CDF differs from normalized positive p-q");
  }
}

void TestFilterCombinations() {
  for (const auto& test : gufo::test::QwenSamplingCases()) {
    std::cout << "Sampling strategy: " << test.name << '\n';
    CheckSamplingDistribution(test.config);
  }
  for (const float temperature : {0.0F, 0.05F, 0.8F, 2.0F}) {
    for (const int top_k : {0, 1, 3}) {
      for (const float top_p : {1.0F, 0.65F}) {
        for (const float min_p : {0.0F, 0.4F}) {
          for (const std::size_t minimum : {0U, 2U}) {
            for (const bool penalized : {false, true}) {
              gufo::sampling::SamplingConfig config{
                  .temperature = temperature,
                  .top_k = top_k,
                  .top_p = top_p,
                  .min_p = min_p,
                  .min_keep = minimum,
                  .repeat_penalty = 1.2F,
                  .repeat_last_n = penalized ? 8U : 0U,
                  .frequency_penalty = 0.2F,
                  .presence_penalty = -0.1F,
              };
              CheckSamplingDistribution(config);
            }
          }
        }
      }
    }
  }
  auto oversized = gufo::sampling::SamplingConfig{
      .temperature = 0.8F, .top_k = 100, .top_p = 0.01F, .min_keep = 100};
  CheckSamplingDistribution(oversized);
  std::cout
      << "216 AR policies; random policies also check p/q and residual CDF\n";
}

void TestSpeculativeResidualSampling() {
  const std::vector<float> logits = {std::log(0.20F), std::log(0.80F)};
  const std::vector<std::uint32_t> draft_ids = {0U, 1U};
  const std::vector<float> draft_probabilities = {0.80F, 0.20F};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;

  const auto result =
      SampleSpeculatively(logits, parameters, 0U, 0.80F, draft_ids,
                          draft_probabilities, 0.90F, 0.50F);
  Expect(!result.accepted, "low target/draft ratio rejects the proposal");
  Expect(result.token == 1U,
         "GPU rejection samples the target-minus-draft residual");
}

void TestFilteredSpeculativeResidualSampling() {
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F};
  const std::vector<std::uint32_t> draft_ids = {0U, 1U, 2U};
  const std::vector<float> draft_probabilities = {0.10F, 0.10F, 0.80F};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 0.8F;
  parameters.top_k = 2;
  parameters.top_p = 0.95F;
  parameters.min_p = 0.05F;

  const auto result =
      SampleSpeculatively(logits, parameters, 2U, 0.80F, draft_ids,
                          draft_probabilities, 0.50F, 0.0F);
  Expect(!result.accepted, "filtered-out draft token is rejected");
  Expect(result.token == 0U,
         "filtered speculative correction stays in the target support");
}

void TestSamplingBoundaryValues() {
  // Includes empty GPU segments and a finite logit whose probability
  // underflows to zero. A zero uniform must still choose nonzero mass.
  const std::vector<float> logits = {-1000.0F, -2.0F, 3.0F};
  for (const int top_k : {0, 2}) {
    gufo::hip::GpuSamplingParameters parameters;
    parameters.temperature = 1e-38F;
    parameters.top_k = top_k;
    for (const float uniform : {0.0F, 0.5F, 0.99999994F}) {
      parameters.uniform = uniform;
      Expect(Sample(logits, parameters) == 2,
             "cold sampling must remain finite and skip zero probability");
    }
    parameters.temperature = 1.0F;
    parameters.uniform = 0.0F;
    Expect(Sample(logits, parameters) == (top_k == 0 ? 1U : 2U),
           "zero uniform must skip empty segments and zero weights");
    const std::vector<std::uint32_t> ids{0, 1};
    const std::vector<float> probabilities{0.8F, 0.2F};
    parameters.temperature = 1e-38F;
    const auto result = SampleSpeculatively(logits, parameters, 0, 0.8F, ids,
                                            probabilities, 0.5F, 0.0F);
    Expect(!result.accepted && result.token == 2,
           "cold residual sampling must recover target mass outside draft "
           "support");
  }

  // Inverse xorshift state whose next output is UINT64_MAX. The RNG keeps
  // 24 bits, so its largest draw must remain below one after the GPU cast.
  std::uint64_t maximum_draw_state = UINT64_C(0xa8d395be4b19cce8);
  const double maximum_draw = gufo::sampling::Uniform(&maximum_draw_state);
  const float gpu_draw = static_cast<float>(maximum_draw);
  Expect(maximum_draw == gpu_draw && gpu_draw == std::nextafter(1.0F, 0.0F),
         "the largest RNG draw remains exactly representable below one");
  const std::vector<float> equal_logits(4, 0.0F);
  const std::vector<std::uint32_t> ids{0, 1, 2, 3};
  const std::vector<float> probabilities{0.25F, 0.5F, 0.125F, 0.125F};
  for (const int top_k : {0, 4}) {
    gufo::hip::GpuSamplingParameters parameters;
    parameters.temperature = 1.0F;
    parameters.top_k = top_k;
    const auto result = SampleSpeculatively(
        equal_logits, parameters, 0, 0.25F, ids, probabilities, gpu_draw, 0.5F);
    Expect(result.accepted && result.token == 0,
           "p(y) = q(y) accepts even the largest draw on both GPU routes");
  }
}

void TestExactNucleusBoundaryAndGreedyTies() {
  // Four candidates per GPU segment: the nucleus ends inside a segment,
  // exactly at 257 / 1024 of the unfiltered probability mass.
  const std::vector<float> equal_logits(1024, 0.0F);
  gufo::hip::GpuSamplingParameters parameters;
  parameters.temperature = 1.0F;
  parameters.top_p = 257.0F / 1024.0F;
  parameters.uniform = 0.99999994F;
  const auto nucleus = Sample(equal_logits, parameters);
  const std::vector<std::uint32_t> ids{257};
  const std::vector<float> probabilities{1.0F};
  const auto speculative = SampleSpeculatively(
      equal_logits, parameters, 257, 1.0F, ids, probabilities, 0.0F, 0.5F);
  std::vector<float> tied_logits(512, -2.0F);
  tied_logits[1] = tied_logits[256] = 1.0F;
  const auto greedy = Sample(tied_logits, {});
  std::cerr << "Boundary regression: nucleus=" << nucleus
            << " (expected 256), draft accepted=" << speculative.accepted
            << " (expected false), greedy tie=" << greedy << " (expected 1)\n";
  Expect(nucleus == 256 && !speculative.accepted && greedy == 1,
         "exact nucleus cutoff and lowest-token greedy ties");
  parameters = {};
  parameters.temperature = 1.0F;
  parameters.top_k = 1;
  Expect(Sample(tied_logits, parameters) == 1,
         "top-k one uses the same lowest-token tie as greedy");
  const std::vector<float> extreme_logits{-3e35F, -2e35F, -1e35F};
  Expect(Sample(extreme_logits, {}) == 2,
         "greedy reduction covers the entire finite float range");

  float* device_logits = nullptr;
  std::uint32_t* device_token = nullptr;
  CheckHip(hipMalloc(&device_logits, tied_logits.size() * sizeof(float)),
           "allocate batched tie");
  CheckHip(hipMalloc(&device_token, sizeof(std::uint32_t)),
           "allocate tie token");
  CheckHip(hipMemcpy(device_logits, tied_logits.data(),
                     tied_logits.size() * sizeof(float), hipMemcpyHostToDevice),
           "upload batched tie");
  gufo::hip::LaunchBatchedGPUArgmax(device_logits, device_token, 1,
                                    tied_logits.size());
  std::uint32_t batched_token = 0;
  CheckHip(hipMemcpy(&batched_token, device_token, sizeof(batched_token),
                     hipMemcpyDeviceToHost),
           "download batched tie");
  CheckHip(hipFree(device_token), "free tie token");
  CheckHip(hipFree(device_logits), "free batched tie");
  Expect(batched_token == 1, "batched verification uses the same greedy tie");
}

void TestFullVocabularyQuantiles() {
  // Exercise real Qwen vocabulary geometry and substantial mass outside the
  // sparse draft support. For a dense tail, compare CDF mass rather than token
  // identity: FP32 reductions can move a draw across very small intervals.
  std::vector<float> logits(248320);
  for (std::size_t i = 0; i < logits.size(); ++i)
    logits[i] = -10.0F + 0.01F * static_cast<float>(i % 29);
  logits[1] = 2.0F;
  logits[3] = 1.1F;
  logits[4] = -0.3F;
  logits[258] = 0.4F;
  const std::vector<std::uint32_t> draft_ids{1, 3, 4};
  const std::vector<float> draft_probabilities{0.5F, 0.25F, 0.25F};
  for (const auto& config :
       {gufo::sampling::SamplingConfig{.temperature = 0.8F},
        gufo::sampling::SamplingConfig{.temperature = 2.0F, .top_p = 0.9F},
        gufo::sampling::SamplingConfig{.temperature = 0.8F, .top_k = 40},
        gufo::sampling::SamplingConfig{
            .temperature = 0.8F, .min_p = 0.8F, .min_keep = 3}}) {
    const auto distribution = gufo::sampling::BuildDistribution(logits, config);
    std::vector<gufo::sampling::Probability> ordered(
        distribution.entries().begin(), distribution.entries().end());
    if (config.top_k == 0 && config.top_p == 1.0F &&
        !(config.min_p > 0.0F && config.min_keep > 1))
      std::ranges::sort(ordered, {}, &gufo::sampling::Probability::token);
    auto residual = ordered;
    double total = 0.0;
    for (auto& entry : residual) {
      for (std::size_t i = 0; i < draft_ids.size(); ++i)
        if (entry.token == draft_ids[i])
          entry.value = std::max(0.0, entry.value - draft_probabilities[i]);
      total += entry.value;
    }
    for (auto& entry : residual)
      entry.value /= total;
    const auto contains_quantile = [](const auto& entries, std::uint32_t token,
                                      double uniform) {
      constexpr double tolerance = 2e-5;
      double cumulative = 0.0;
      for (const auto& entry : entries) {
        if (entry.token == token)
          return entry.value > 0.0 && cumulative <= uniform + tolerance &&
                 cumulative + entry.value >= uniform - tolerance;
        cumulative += entry.value;
      }
      return false;
    };
    gufo::hip::GpuSamplingParameters parameters{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .min_keep = config.min_keep,
    };
    for (const float uniform : {0.0F, 0.23F, 0.91F, 0.99999994F}) {
      parameters.uniform = uniform;
      Expect(contains_quantile(ordered, Sample(logits, parameters), uniform),
             "full-vocabulary AR CDF differs from reference");
      // Token 4 has excess draft mass in every configuration.
      const auto result =
          SampleSpeculatively(logits, parameters, 4, 0.25F, draft_ids,
                              draft_probabilities, 0.99F, uniform);
      Expect(!result.accepted &&
                 contains_quantile(residual, result.token, uniform),
             "full-vocabulary DFlash2 residual CDF differs from reference");
    }
  }
}

}  // namespace

int main() {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return 77;
  }
  TestGreedySparsePenalty();
  TestLinearTemperatureAndMinP();
  TestOrderedFilterChain();
  TestTopKOneIsDeterministic();
  TestTopKOneRespectsMinimumKeep();
  TestFilterCombinations();
  TestSpeculativeResidualSampling();
  TestFilteredSpeculativeResidualSampling();
  TestSamplingBoundaryValues();
  TestExactNucleusBoundaryAndGreedyTies();
  TestFullVocabularyQuantiles();
  std::cout << "All Qwen GPU sampling tests passed.\n";
  return 0;
}
