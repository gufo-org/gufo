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

void TestFilterCombinations() {
  // Compare quantiles against the independent double-precision CPU
  // distribution. Exercise every filter together, including minimum floors,
  // disabled history, greedy and cold/hot random sampling.
  const std::vector<float> logits{0.4F, 2.0F, -1.0F, 1.1F, -0.3F};
  const std::vector<std::uint32_t> history{1, 1, 3};
  const std::vector<std::uint32_t> ids{1, 3};
  const std::vector<std::uint32_t> counts{2, 1};
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
              const auto distribution =
                  gufo::sampling::BuildDistribution(logits, config, history);
              std::vector<gufo::sampling::Probability> ordered(
                  distribution.entries().begin(), distribution.entries().end());
              const bool linear =
                  top_k == 0 && top_p == 1.0F && !(min_p > 0.0F && minimum > 1);
              if (linear)
                std::ranges::sort(ordered, {},
                                  &gufo::sampling::Probability::token);
              for (const float uniform : {0.0F, 0.23F, 0.91F}) {
                double threshold = uniform;
                auto expected = ordered.back().token;
                for (const auto& entry : ordered) {
                  if (entry.value > 0.0 && threshold < entry.value) {
                    expected = entry.token;
                    break;
                  }
                  threshold -= entry.value;
                }
                gufo::hip::GpuSamplingParameters parameters{
                    .temperature = temperature,
                    .top_k = top_k,
                    .top_p = top_p,
                    .min_p = min_p,
                    .min_keep = minimum,
                    .repeat_penalty = config.repeat_penalty,
                    .frequency_penalty = config.frequency_penalty,
                    .presence_penalty = config.presence_penalty,
                    .uniform = uniform,
                };
                Expect(Sample(logits, parameters,
                              penalized ? std::span(ids)
                                        : std::span<const std::uint32_t>{},
                              penalized ? std::span(counts)
                                        : std::span<const std::uint32_t>{}) ==
                           expected,
                       "GPU filter combination differs from CPU distribution");
              }
            }
          }
        }
      }
    }
  }
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
  std::cout << "All Qwen GPU sampling tests passed.\n";
  return 0;
}
