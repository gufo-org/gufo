#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/families/qwen/kernels/dflash_kernels.hpp"
#include "src/families/qwen/kernels/ops/token.hpp"
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

std::uint32_t Sample(
    std::span<const float> logits,
    const gufo::hip::GpuSamplingParameters& parameters,
    std::span<const gufo::sampling::TokenPenalty> penalties = {}) {
  float* device_logits = nullptr;
  std::uint32_t* device_token = nullptr;
  CheckHip(hipMalloc(&device_logits, logits.size_bytes()), "allocate logits");
  CheckHip(hipMalloc(&device_token, sizeof(std::uint32_t)), "allocate token");
  CheckHip(hipMemcpy(device_logits, logits.data(), logits.size_bytes(),
                     hipMemcpyHostToDevice),
           "upload logits");

  gufo::hip::GpuSamplingWorkspace workspace;
  gufo::hip::AllocateGpuSamplingWorkspace(&workspace, logits.size(),
                                          penalties.size());
  gufo::hip::LaunchGPUSampling(device_logits, device_token, logits.size(),
                               parameters, penalties.data(), penalties.size(),
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
    double acceptance_uniform, double residual_uniform,
    std::span<const gufo::sampling::TokenPenalty> penalties = {}) {
  if (draft_candidate_ids.size() != draft_candidate_probabilities.size()) {
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
      std::max(draft_candidate_ids.size(), penalties.size()));
  gufo::hip::LaunchGPUSpeculativeSampling(
      device_logits, device_token, device_accepted, logits.size(), parameters,
      draft_token, draft_token_probability, draft_candidate_ids.data(),
      draft_candidate_probabilities.data(), draft_candidate_ids.size(),
      acceptance_uniform, residual_uniform, penalties.data(), penalties.size(),
      &workspace);

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

void TestInvalidTargetRows() {
  bool failed = false;
  try {
    (void)gufo::hip::CheckedSampleToken(gufo::hip::kInvalidSampleToken, 248320);
  } catch (const std::runtime_error&) {
    failed = true;
  }
  Expect(failed,
         "host must propagate invalid device samples as request failures");
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (std::size_t size : {3U, 16383U, 16384U, 248320U}) {
    std::vector<float> logits(size);
    for (std::size_t i = 0; i < size; ++i)
      logits[i] = i % 3 == 0 ? nan : i % 3 == 1 ? inf : -inf;
    for (int route : {0, 1, 2, 3}) {
      gufo::hip::GpuSamplingParameters config;
      config.temperature = route == 0 ? 0 : 1;
      config.top_k = route == 2 ? 40 : 0;
      config.top_p = route == 3 ? 0.9F : 1;
      config.uniform = 0.5;
      Expect(Sample(logits, config) == gufo::hip::kInvalidSampleToken,
             "ordinary sampler must report all-nonfinite target failure");
      if (route) {
        const std::vector<std::uint32_t> ids{0};
        const std::vector<float> mass{1};
        const auto sampled =
            SampleSpeculatively(logits, config, 0, 1, ids, mass, 0.0, 0.5);
        Expect(sampled.token == gufo::hip::kInvalidSampleToken,
               "speculative sampler must report all-nonfinite target failure");
      }
      logits.back() = -1e35F;
      Expect(Sample(logits, config) == size - 1,
             "nonfinite logits cannot outrank a finite candidate");
      logits.back() = -inf;
    }
  }
  gufo::hip::GpuSamplingParameters config;
  config.temperature = 1;
  config.uniform = 0.5 - 0x1.0p-40;
  Expect(Sample(std::vector<float>{0, 0}, config) == 0,
         "GPU draw must retain bits below float resolution");
  config.uniform = 0.5;
  Expect(Sample(std::vector<float>{0, 0}, config) == 1,
         "GPU CDF is half-open at an exact boundary");
}

void TestGreedySparsePenalty() {
  const std::vector<float> logits = {5.0F, 4.0F, 3.0F};
  const std::vector<gufo::sampling::TokenPenalty> penalties = {{0, 0, 1}};
  gufo::hip::GpuSamplingParameters parameters;
  parameters.repeat_penalty = 2.0F;

  Expect(Sample(logits, parameters, penalties) == 1U,
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
         "ordered route applies temperature, top-k, top-p, then min-p");
}

void TestTemperedFilterExpectations() {
  // Expected sets come from softmax([0,-1,-2]/T), not the CPU sampler.
  for (const std::size_t size : {3U, 131072U}) {
    std::vector<float> logits(size, -1000.0F);
    logits[0] = 0;
    logits[1] = -1;
    logits[2] = -2;
    for (const bool nucleus : {false, true}) {
      gufo::hip::GpuSamplingParameters parameters{.temperature = 0.5F};
      parameters.top_p = nucleus ? 0.8F : 1.0F;
      parameters.min_p = nucleus ? 0.0F : 0.2F;
      parameters.uniform = 0.99F;
      Expect(Sample(logits, parameters) == 0,
             "tempered filters must exclude the second token");
      const std::vector<std::uint32_t> ids{1};
      const std::vector<float> probabilities{1};
      const auto rejected = SampleSpeculatively(logits, parameters, 1, 1, ids,
                                                probabilities, 0, 0.99F);
      Expect(!rejected.accepted && rejected.token == 0,
             "verification and residual use the tempered target support");
      parameters.temperature = 2.0F;
      parameters.min_p = nucleus ? 0.0F : 0.5F;
      Expect(Sample(logits, parameters) == 1,
             "higher temperature retains the second token");
      const auto accepted = SampleSpeculatively(logits, parameters, 1, 1, ids,
                                                probabilities, 0.3F, 0.99F);
      Expect(accepted.accepted && accepted.token == 1,
             "acceptance uses the flattened target probability");
    }
  }
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

void TestFullGenerationPenalties() {
  const std::vector<float> logits{3, 2, 1};
  gufo::sampling::SamplerState sampler({.repeat_last_n = 0,
                                        .frequency_penalty = 0.5F,
                                        .presence_penalty = 0.25F},
                                       std::vector<std::uint32_t>{0, 0, 0});
  gufo::hip::GpuSamplingParameters p{.frequency_penalty = 0.5F,
                                     .presence_penalty = 0.25F};
  Expect(Sample(logits, p, sampler.penalties()) == 0,
         "prompt occurrences must not count as generated text");
  sampler.Accept(std::vector<std::uint32_t>{0, 0});
  sampler.Accept(std::vector<std::uint32_t>(80, 2));
  // [3 - 2*.5 - .25, 2, 1 - 80*.5 - .25] -> token 1.
  Expect(
      Sample(logits, p, sampler.penalties()) == 1,
      "GPU penalties retain response counts beyond 64 with no repeat window");
  p.temperature = 1;
  const std::vector<std::uint32_t> ids{0};
  const std::vector<float> q{1};
  const auto result = SampleSpeculatively(logits, p, 0, 1, ids, q, 0.99F, 0.5F,
                                          sampler.penalties());
  Expect(!result.accepted && result.token == 1,
         "GPU residual verification uses the same full generated counts");
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
  const std::vector<std::uint32_t> generated{3, 1, 4};
  const auto distribution =
      gufo::sampling::BuildDistribution(logits, config, history, generated);
  gufo::sampling::SamplerState sampler(config, history);
  sampler.Accept(generated);
  const auto penalties = sampler.penalties();
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
    Expect(Sample(logits, parameters, penalties) == quantile(ordered, uniform),
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
          draft_probabilities, static_cast<float>(ratio - tolerance), 0.5F,
          penalties);
      Expect(result.accepted && result.token == draft_ids[i],
             "DFlash2 acceptance below p/q differs from reference");
    }
    if (ratio < 1.0 - tolerance) {
      const auto result = SampleSpeculatively(
          logits, parameters, draft_ids[i], draft_probabilities[i], draft_ids,
          draft_probabilities, static_cast<float>(ratio + tolerance), 0.5F,
          penalties);
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
        rejection_draw, uniform, penalties);
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

void TestDuplicateDraftCandidates() {
  const std::vector<float> logits{std::log(0.20F), std::log(0.30F),
                                  std::log(0.50F)};
  const std::vector<std::uint32_t> unique_ids{0, 1};
  const std::vector<float> unique_probabilities{0.8F, 0.2F};
  const std::vector<std::uint32_t> duplicate_ids{0, 1, 0, 99};
  const std::vector<float> duplicate_probabilities{0.4F, 0.2F, 0.4F, 0.0F};
  for (const int top_k : {0, 3}) {
    gufo::hip::GpuSamplingParameters parameters{.temperature = 1.0F,
                                                .top_k = top_k};
    for (const float uniform : {0.0F, 0.1F, 0.5F, 0.99999994F}) {
      const auto expected =
          SampleSpeculatively(logits, parameters, 0, 0.8F, unique_ids,
                              unique_probabilities, 0.9F, uniform);
      const auto actual =
          SampleSpeculatively(logits, parameters, 0, 0.8F, duplicate_ids,
                              duplicate_probabilities, 0.9F, uniform);
      Expect(!actual.accepted && actual.token == expected.token,
             "duplicate or out-of-vocabulary candidates alter residual q");
    }
  }
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
  // 53 bits; passing a float here would round the largest draw to one.
  std::uint64_t maximum_draw_state = UINT64_C(0xa8d395be4b19cce8);
  const double maximum_draw = gufo::sampling::Uniform(&maximum_draw_state);
  const double gpu_draw = maximum_draw;
  Expect(maximum_draw == gpu_draw && gpu_draw == std::nextafter(1.0, 0.0),
         "the largest RNG draw remains exactly representable below one");
  const std::vector<float> equal_logits(4, 0.0F);
  const std::vector<std::uint32_t> ids{0, 1, 2, 3};
  const std::vector<float> probabilities{0.25F, 0.5F, 0.125F, 0.125F};
  for (const int top_k : {0, 4}) {
    gufo::hip::GpuSamplingParameters parameters;
    parameters.temperature = 1.0F;
    parameters.top_k = top_k;
    const auto result = SampleSpeculatively(equal_logits, parameters, 0, 0.25F,
                                            ids, probabilities, gpu_draw, 0.5F);
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
                                    tied_logits.size(), {});
  std::uint32_t batched_token = 0;
  CheckHip(hipMemcpy(&batched_token, device_token, sizeof(batched_token),
                     hipMemcpyDeviceToHost),
           "download batched tie");
  CheckHip(hipFree(device_token), "free tie token");
  CheckHip(hipFree(device_logits), "free batched tie");
  Expect(batched_token == 1, "batched verification uses the same greedy tie");
}

void TestFullVocabularyQuantiles() {
  // Exercise the parallel verifier's dispatch boundary, real Qwen vocabulary
  // and ragged segments, with substantial mass outside the draft support.
  // Compare CDF mass: FP32 reductions can cross very small token intervals.
  for (const std::size_t vocab : {16383U, 16384U, 248320U, 248323U}) {
    std::vector<float> logits(vocab);
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
          gufo::sampling::SamplingConfig{.temperature = 1e-38F},
          gufo::sampling::SamplingConfig{.temperature = 0.8F, .min_p = 0.5F},
          gufo::sampling::SamplingConfig{.temperature = 2.0F, .top_p = 0.9F},
          gufo::sampling::SamplingConfig{.temperature = 0.8F, .top_k = 40},
          gufo::sampling::SamplingConfig{
              .temperature = 0.8F, .min_p = 0.8F, .min_keep = 3}}) {
      const auto distribution =
          gufo::sampling::BuildDistribution(logits, config);
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
      const auto contains_quantile = [](const auto& entries,
                                        std::uint32_t token, double uniform) {
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
}

void TestBatchedArgmaxTiles() {
  for (const std::size_t vocab : {4095U, 4096U, 4097U, 248323U}) {
    constexpr std::size_t rows = 8;
    std::vector<float> logits(rows * vocab, -3e35F);
    for (std::size_t row = 0; row < rows; ++row) {
      auto values = std::span(logits).subspan(row * vocab, vocab);
      if (row == 0) {
        std::fill(values.begin(), values.end(), NAN);
        values[17] = INFINITY;
        values[19] = -INFINITY;
      } else if (row == 1) {
        values.back() = -1e35F;
      } else if (row == 2) {
        values[4094] = values.back() = 37.0F;
      } else if (row == 3) {
        values[1] = values[256] = 44.0F;
      } else if (row == 4) {
        std::fill(values.begin(), values.end(), 0.0F);
      } else if (row == 5) {
        values[0] = NAN;
        values[vocab / 2] = INFINITY;
        values.back() = 3.0F;
      } else if (row == 6) {
        values[vocab > 4096 ? 4096 : 1024] = values.back() = 9.0F;
      } else {
        values[17] = -2e35F;
      }
    }
    std::vector<std::uint32_t> expected(rows, gufo::hip::kInvalidSampleToken);
    for (std::size_t row = 0; row < rows; ++row) {
      float maximum = -INFINITY;
      for (std::size_t id = 0; id < vocab; ++id) {
        const auto value = logits[row * vocab + id];
        if (std::isfinite(value) && value > maximum) {
          maximum = value;
          expected[row] = static_cast<std::uint32_t>(id);
        }
      }
    }
    const auto scratch_count = 2 * rows * ((vocab + 4095) / 4096);
    float* device_logits = nullptr;
    float* scratch = nullptr;
    std::uint32_t* tokens = nullptr;
    CheckHip(hipMalloc(&device_logits, logits.size() * sizeof(float)),
             "allocate tiled argmax logits");
    CheckHip(hipMalloc(&scratch, (scratch_count + 2) * sizeof(float)),
             "allocate guarded argmax scratch");
    CheckHip(hipMalloc(&tokens, rows * sizeof(std::uint32_t)),
             "allocate batched argmax tokens");
    CheckHip(hipMemcpy(device_logits, logits.data(),
                       logits.size() * sizeof(float), hipMemcpyHostToDevice),
             "upload tiled argmax fixtures");
    CheckHip(hipMemset(scratch, 0xA5, (scratch_count + 2) * sizeof(float)),
             "poison argmax scratch");
    for (const std::size_t width : {1U, 2U, 4U, 6U, 8U}) {
      gufo::hip::LaunchBatchedGPUArgmax(device_logits, tokens, width, vocab,
                                        {scratch + 1, scratch_count});
      std::vector<std::uint32_t> actual(width);
      CheckHip(hipMemcpy(actual.data(), tokens, width * sizeof(std::uint32_t),
                         hipMemcpyDeviceToHost),
               "read tiled argmax");
      Expect(std::equal(actual.begin(), actual.end(), expected.begin()),
             "argmax tiles must preserve finite extrema and lowest-ID ties");
    }
    std::uint32_t first = 0, last = 0;
    CheckHip(hipMemcpy(&first, scratch, sizeof(first), hipMemcpyDeviceToHost),
             "read leading argmax guard");
    CheckHip(hipMemcpy(&last, scratch + scratch_count + 1, sizeof(last),
                       hipMemcpyDeviceToHost),
             "read trailing argmax guard");
    Expect(first == 0xA5A5A5A5U && last == first,
           "argmax scratch must remain within the supplied span");
    CheckHip(hipFree(tokens), "free argmax tokens");
    CheckHip(hipFree(scratch), "free argmax scratch");
    CheckHip(hipFree(device_logits), "free argmax logits");
  }
}

void TestExactCandidatePrefix() {
  for (const std::size_t size : {1023U, 1024U, 1025U, 32769U, 248320U}) {
    std::vector<float> logits(size);
    std::vector<std::uint32_t> expected(size);
    float* device = nullptr;
    CheckHip(hipMalloc(&device, size * sizeof(float)), "candidate input");
    gufo::hip::GpuSamplingWorkspace workspace;
    gufo::hip::AllocateGpuSamplingWorkspace(&workspace, size, 0);
    for (unsigned fixture = 0; fixture < 4; ++fixture) {
      for (std::size_t i = 0; i < size; ++i) {
        if (fixture == 0)
          logits[i] =
              static_cast<float>(static_cast<int>(i * 131U % 997) - 498) /
              97.0F;
        else if (fixture == 1)
          logits[i] = i % 2 ? -0.0F : 0.0F;
        else if (fixture == 2)
          logits[i] = i % 3 ? -INFINITY : NAN;
        else
          logits[i] = i % 1024 == 1023 ? 10.0F : -1.0F;
      }
      if (fixture != 1) {
        logits[size / 2] = INFINITY;
        logits[size - 1] = -INFINITY;
      }
      std::iota(expected.begin(), expected.end(), 0U);
      const auto finite = [&](auto id) {
        return std::isfinite(logits[id]) ? logits[id] : -INFINITY;
      };
      std::stable_sort(expected.begin(), expected.end(),
                       [&](auto a, auto b) { return finite(a) > finite(b); });
      CheckHip(hipMemcpy(device, logits.data(), size * sizeof(float),
                         hipMemcpyHostToDevice),
               "candidate upload");
      gufo::hip::LaunchGPUSortLogits(device, &workspace);
      std::vector<float> reference(64), actual(64);
      std::vector<std::uint32_t> reference_ids(64), ids(64);
      CheckHip(hipMemcpy(reference.data(), workspace.sorted_logits,
                         64 * sizeof(float), hipMemcpyDeviceToHost),
               "full sort logits");
      CheckHip(hipMemcpy(reference_ids.data(), workspace.sorted_token_ids,
                         64 * sizeof(std::uint32_t), hipMemcpyDeviceToHost),
               "full sort ids");
      Expect(std::equal(reference_ids.begin(), reference_ids.end(),
                        expected.begin()),
             "full sort independent ordering");
      for (const std::size_t keep : {1U, 20U, 32U, 33U, 64U}) {
        gufo::hip::LaunchGPUSortLogits(device, &workspace, nullptr, keep);
        CheckHip(hipMemcpy(actual.data(), workspace.sorted_logits,
                           keep * sizeof(float), hipMemcpyDeviceToHost),
                 "bounded logits");
        CheckHip(hipMemcpy(ids.data(), workspace.sorted_token_ids,
                           keep * sizeof(std::uint32_t), hipMemcpyDeviceToHost),
                 "bounded ids");
        Expect(std::memcmp(actual.data(), reference.data(),
                           keep * sizeof(float)) == 0 &&
                   std::equal(ids.begin(), ids.begin() + keep,
                              reference_ids.begin()),
               "bounded candidates must retain exact scores and tie order");
      }
    }
    gufo::hip::FreeGpuSamplingWorkspace(&workspace);
    CheckHip(hipFree(device), "free candidate input");
  }
  std::cout << "Bounded candidate prefixes match full sort and CPU ordering.\n";
}

void TestDraftSamplingControls() {
  using namespace gufo;
  constexpr std::size_t vocab = 2051, rows = 4, count = 8, rank = 16;
  const std::array<sampling::TokenId, count> ids{3,    4,    5,    6,
                                                 1025, 1026, 1027, 2048};
  std::vector<float> logits(rows * vocab, -20.0F);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t i = 0; i < count; ++i)
      logits[row * vocab + ids[i]] = 4.0F - float(i / 2) * 0.2F;
  for (int policy = 0; policy < 8; ++policy) {
    sampling::SamplingConfig config{
        .temperature = 0.7F,
        .repeat_penalty = policy == 1 || policy == 7 ? 1.2F
                          : policy == 2              ? 0.8F
                                                     : 1.0F,
        .repeat_last_n = 2,
        .frequency_penalty = policy == 3   ? 0.2F
                             : policy == 4 ? -0.2F
                                           : 0.0F,
        .presence_penalty = policy == 5 || policy == 7 ? 1.5F
                            : policy == 6              ? -1.5F
                                                       : 0.0F,
    };
    if (policy == 7)
      config.temperature = 0.0001F;
    sampling::SamplerState expected(config, std::span(ids).first(4));
    expected.Accept(3);
    expected.Accept(5);
    std::vector<void*> allocations;
    const auto allocate = [&](std::size_t bytes) {
      void* pointer = nullptr;
      CheckHip(hipMalloc(&pointer, std::max<std::size_t>(bytes, 4)),
               "draft control allocation");
      allocations.push_back(pointer);
      return pointer;
    };
    const auto upload = [&](const void* source, std::size_t bytes) {
      auto* pointer = allocate(bytes);
      if (bytes)
        CheckHip(hipMemcpy(pointer, source, bytes, hipMemcpyHostToDevice),
                 "draft control upload");
      return pointer;
    };
    auto* d_logits =
        static_cast<float*>(upload(logits.data(), logits.size() * 4));
    auto* codebook = allocate(vocab * rank * 2);
    CheckHip(hipMemset(codebook, 0, vocab * rank * 2),
             "zero transition oracle");
    auto* hidden = static_cast<float*>(allocate(rows * rank * 4));
    CheckHip(hipMemset(hidden, 0, rows * rank * 4), "zero selector oracle");
    auto* tokens = static_cast<std::uint32_t*>(allocate((rows + 1) * 4));
    CheckHip(hipMemset(tokens, 0, (rows + 1) * 4), "set draft anchor");
    auto* partial = static_cast<float*>(allocate(
        rows * hip::kernels::DFlashSelectorScratchElements(vocab) * 4));
    auto* partial_ids = static_cast<std::uint32_t*>(allocate(
        rows * hip::kernels::DFlashSelectorScratchElements(vocab) * 4));
    const std::array<float, rows> uniforms{0.0F, 0.23F, 0.91F, 0.99999994F};
    auto* draws =
        static_cast<float*>(upload(uniforms.data(), sizeof(uniforms)));
    auto* candidates = static_cast<std::uint32_t*>(allocate(rows * count * 4));
    auto* probabilities = static_cast<float*>(allocate(rows * count * 4));
    const auto penalties = expected.penalties();
    const auto history = expected.history();
    hip::kernels::DFlashSampling controls{
        .repeat_penalty = config.repeat_penalty,
        .frequency_penalty = config.frequency_penalty,
        .presence_penalty = config.presence_penalty,
        .repeat_last_n = std::uint32_t(config.repeat_last_n),
        .penalties = static_cast<const sampling::TokenPenalty*>(
            upload(penalties.data(), penalties.size_bytes())),
        .penalty_count = std::uint32_t(penalties.size()),
        .history = static_cast<const sampling::TokenId*>(
            upload(history.data(), history.size_bytes())),
        .history_count = std::uint32_t(history.size()),
    };
    const std::array sequences{hip::kernels::DFlashSelectorSequence{
        rows, config.temperature, controls}};
    hip::kernels::LaunchDFlashSelectorBatch(
        d_logits, hidden, codebook, codebook, tokens, nullptr, partial,
        partial_ids, draws, candidates, probabilities, sequences, vocab, rank,
        count, nullptr);
    std::array<std::uint32_t, rows + 1> actual{};
    std::array<std::uint32_t, rows * count> actual_ids{};
    std::array<float, rows * count> actual_probabilities{};
    CheckHip(
        hipMemcpy(actual.data(), tokens, sizeof(actual), hipMemcpyDeviceToHost),
        "draft controlled samples");
    CheckHip(hipMemcpy(actual_ids.data(), candidates, sizeof(actual_ids),
                       hipMemcpyDeviceToHost),
             "draft controlled IDs");
    CheckHip(hipMemcpy(actual_probabilities.data(), probabilities,
                       sizeof(actual_probabilities), hipMemcpyDeviceToHost),
             "draft controlled probabilities");
    for (std::size_t row = 0; row < rows; ++row) {
      // The base lattice keeps the selector's existing lane-order tie breaks.
      // Its set is known independently; filter a full-vocabulary CPU row so
      // numeric sampler ties use actual token IDs, not compact row indices.
      auto selected = std::span(actual_ids).subspan(row * count, count);
      std::array<std::uint32_t, count> sorted{};
      std::ranges::copy(selected, sorted.begin());
      std::ranges::sort(sorted);
      Expect(std::ranges::equal(sorted, ids), "selector candidate set changed");
      std::vector<float> scores(vocab, -INFINITY);
      for (auto id : ids)
        scores[id] = logits[row * vocab + id];
      const auto distribution = expected.Distribution(scores);
      double cumulative = 0.0;
      bool checked = false;
      for (std::size_t i = 0; i < count; ++i) {
        const double probability = distribution.probability(selected[i]);
        if (std::abs(actual_probabilities[row * count + i] - probability) >=
            2e-5)
          std::cerr << "draft policy=" << policy << " row=" << row
                    << " token=" << selected[i]
                    << " gpu=" << actual_probabilities[row * count + i]
                    << " cpu=" << probability << '\n';
        Expect(std::abs(actual_probabilities[row * count + i] - probability) <
                   2e-5,
               "draft q differs from independent CPU distribution");
        if (actual[row + 1] == selected[i]) {
          Expect(probability > 0 && cumulative <= uniforms[row] + 2e-5 &&
                     cumulative + probability >= uniforms[row] - 2e-5,
                 "draft sample is outside its reported q CDF");
          checked = true;
        }
        cumulative += probability;
      }
      Expect(checked, "draft sample must belong to its reported candidates");
      expected.Accept(actual[row + 1]);
    }
    for (auto* allocation : allocations)
      CheckHip(hipFree(allocation), "free draft control allocation");
  }
  std::cout << "DFlash sampling controls match independent conditional q.\n";
}

}  // namespace

int main() {
  TestInvalidTargetRows();
  TestFullGenerationPenalties();
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return 77;
  }
  TestGreedySparsePenalty();
  TestLinearTemperatureAndMinP();
  TestOrderedFilterChain();
  TestTemperedFilterExpectations();
  TestTopKOneIsDeterministic();
  TestTopKOneRespectsMinimumKeep();
  TestFilterCombinations();
  TestSpeculativeResidualSampling();
  TestDuplicateDraftCandidates();
  TestFilteredSpeculativeResidualSampling();
  TestSamplingBoundaryValues();
  TestExactNucleusBoundaryAndGreedyTies();
  TestFullVocabularyQuantiles();
  TestBatchedArgmaxTiles();
  TestExactCandidatePrefix();
  TestDraftSamplingControls();
  std::cout << "All Qwen GPU sampling tests passed.\n";
  return 0;
}
