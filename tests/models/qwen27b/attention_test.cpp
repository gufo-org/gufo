#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

// Official DFlash2 config: 32 query heads, 8 KV heads, head dimension 128.
constexpr std::uint32_t kNumQueryHeads = 32;
constexpr std::uint32_t kNumKvHeads = 8;
constexpr std::uint32_t kHeadDim = 128;
constexpr std::uint32_t kBlockCount = 8;
constexpr std::uint32_t kSlidingWindow = 2048;

float Sample(std::size_t index, std::size_t salt) {
  return std::sin(static_cast<float>((index * 7U) + (salt * 13U) + 1U)) * 0.5F;
}

enum class Route { kScalar, kWave };

// Independent equations from z-lab/dflash model_mlx.py, pinned in eval/README.
// Keep these small operator checks with the ring/attention checks.
void TestDynamicConvolution() {
  using gufo::test::DeviceBuffer;
  constexpr std::uint32_t tokens = 8, hidden = 5120, taps = 2, group_size = 16;
  constexpr std::uint32_t groups = hidden / group_size;
  std::vector<float> input(tokens * hidden),
      dynamic(tokens * 2 * taps * groups), base(2 * taps * hidden);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = Sample(i, 11);
  for (std::size_t i = 0; i < dynamic.size(); ++i)
    dynamic[i] = Sample(i, 12);
  for (std::size_t i = 0; i < base.size(); ++i)
    base[i] = Sample(i, 13);
  DeviceBuffer<float> d_input(input), d_dynamic(dynamic), d_base(base),
      d_output(input.size());
  for (std::uint32_t side = 0; side < 2; ++side) {
    gufo::hip::kernels::LaunchDFlashGroupedDynamicConv(
        d_input.data(), d_dynamic.data(), d_base.data(), d_output.data(),
        tokens, hidden, taps, group_size, side, nullptr);
    const auto output = d_output.CopyToHost();
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t channel = 0; channel < hidden; ++channel) {
        double reference = 0;
        for (std::uint32_t tap = 0; tap < taps && tap <= token; ++tap) {
          const double value = input[(token - tap) * hidden + channel];
          reference += base[(side * taps + tap) * hidden + channel] * value;
          reference += dynamic[((token * 2 + side) * taps + tap) * groups +
                               channel / group_size] *
                       value;
        }
        gufo::test::Expect(
            std::isfinite(output[token * hidden + channel]) &&
                std::abs(output[token * hidden + channel] - reference) < 2e-7,
            "DFlash grouped convolution differs from the causal equation");
      }
    }
  }
}

void TestProposalVerification(std::span<const std::uint32_t> ids,
                              std::span<const float> q, std::uint32_t vocab) {
  using gufo::test::DeviceBuffer;
  using gufo::test::Expect;
  Expect(std::ranges::find(ids, 0U) == ids.end() &&
             std::ranges::find(ids, 1U) == ids.end(),
         "residual fixture must include tokens outside draft top-k");
  std::vector<float> logits(vocab, -INFINITY);
  logits[0] = std::log(0.55F);
  logits[1] = std::log(0.10F);
  logits[ids[0]] = std::log(0.13F);
  logits[ids[1]] = std::log(0.07F);
  logits[ids[2]] = std::log(0.15F);
  DeviceBuffer<float> d_logits(logits);
  DeviceBuffer<std::uint32_t> d_token(1), d_accepted(1);
  gufo::hip::GpuSamplingWorkspace workspace;
  gufo::hip::AllocateGpuSamplingWorkspace(&workspace, vocab, ids.size());
  for (const bool filtered : {false, true}) {
    gufo::sampling::SamplingConfig config;
    config.temperature = filtered ? 0.8F : 1.0F;
    config.top_k = filtered ? 4 : 0;
    config.top_p = filtered ? 0.95F : 1.0F;
    config.min_p = filtered ? 0.05F : 0.0F;
    config.repeat_penalty = filtered ? 1.2F : 1.0F;
    const std::vector<std::uint32_t> history = {ids[0], ids[0]};
    const auto p = gufo::sampling::BuildDistribution(logits, config, history);
    std::vector<gufo::sampling::Probability> residual;
    double rejection_mass = 0;
    for (const auto& entry : p.entries()) {
      const auto found = std::ranges::find(ids, entry.token);
      const double draft_p = found == ids.end() ? 0 : q[found - ids.begin()];
      const double mass = std::max(entry.value - draft_p, 0.0);
      if (mass > 0) {
        residual.push_back({entry.token, mass});
        rejection_mass += mass;
      }
    }
    // Linear GPU sampling walks token IDs; the filtered route walks sorted
    // target logits. Select a midpoint in every nonempty residual interval.
    if (!filtered)
      std::ranges::sort(residual, {}, &gufo::sampling::Probability::token);
    gufo::hip::GpuSamplingParameters parameters{
        .temperature = config.temperature,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .repeat_penalty = config.repeat_penalty,
    };
    const std::array<gufo::sampling::TokenPenalty, 1> penalties{
        {{ids[0], 0, 1}}};
    const auto verify = [&](std::size_t candidate, double accept_u,
                            double residual_u, bool accepted,
                            std::uint32_t expected) {
      gufo::hip::LaunchGPUSpeculativeSampling(
          d_logits.data(), d_token.data(), d_accepted.data(), vocab, parameters,
          ids[candidate], q[candidate], ids.data(), q.data(), ids.size(),
          accept_u, residual_u, penalties.data(), filtered ? 1 : 0, &workspace);
      Expect(
          d_accepted.CopyToHost()[0] == static_cast<std::uint32_t>(accepted) &&
              d_token.CopyToHost()[0] == expected,
          "GPU verifier differs from independent p/q and residual equations");
    };
    std::vector<double> emitted(vocab, 0);
    for (std::size_t candidate = 0; candidate < ids.size(); ++candidate) {
      if (q[candidate] == 0)
        continue;
      const double acceptance =
          std::min(1.0, p.probability(ids[candidate]) / q[candidate]);
      if (acceptance > 0)
        verify(candidate, acceptance * 0.5, 0.5, true, ids[candidate]);
      emitted[ids[candidate]] += q[candidate] * acceptance;
      if (acceptance < 1) {
        double cumulative = 0;
        for (const auto& entry : residual) {
          const double probability = entry.value / rejection_mass;
          verify(candidate, (1 + acceptance) * 0.5,
                 cumulative + probability * 0.5, false, entry.token);
          emitted[entry.token] += q[candidate] * (1 - acceptance) * probability;
          cumulative += probability;
        }
      }
    }
    for (std::uint32_t token = 0; token < vocab; ++token)
      Expect(std::abs(emitted[token] - p.probability(token)) < 2e-6,
             "selector plus verifier must recover the complete target "
             "distribution");
  }
  gufo::hip::FreeGpuSamplingWorkspace(&workspace);
}

void TestSelector(std::uint32_t vocab) {
  using gufo::test::DeviceBuffer;
  // Cross a partial-top-k partition boundary and leave a short final partition.
  constexpr std::uint32_t rank = 256, anchor = 3;
  std::vector<float> logits(vocab), projected(rank);
  std::vector<std::uint16_t> predecessors(vocab * rank),
      successors(vocab * rank);
  const auto bf16 = [](float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >>
                                      16);
  };
  const auto fp32 = [](std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
  };
  for (std::size_t i = 0; i < logits.size(); ++i) {
    // Distinct unary scores, deliberately scattered across both partitions.
    logits[i] = static_cast<float>((i * 193) % vocab) / vocab * 4.0F;
  }
  for (std::size_t i = 0; i < rank; ++i)
    projected[i] = Sample(i, 17);
  for (std::size_t i = 0; i < predecessors.size(); ++i) {
    predecessors[i] = bf16(Sample(i, 18));
    successors[i] = bf16(Sample(i, 19));
  }
  DeviceBuffer<float> d_logits(logits), d_projected(projected), d_confidence(1),
      d_uniform(std::vector<float>{0.37F});
  DeviceBuffer<std::uint16_t> d_predecessors(predecessors),
      d_successors(successors);
  DeviceBuffer<std::uint32_t> d_anchor(std::vector<std::uint32_t>{anchor}),
      d_selected(1);
  const auto scratch_size =
      gufo::hip::kernels::DFlashSelectorScratchElements(vocab);
  DeviceBuffer<float> d_scores(scratch_size);
  DeviceBuffer<std::uint32_t> d_ids(scratch_size);
  std::vector<std::uint32_t> sorted(vocab);
  std::iota(sorted.begin(), sorted.end(), 0U);
  std::ranges::sort(sorted,
                    [&](auto a, auto b) { return logits[a] > logits[b]; });
  for (const std::uint32_t top_k : {1U, 7U, 16U}) {
    DeviceBuffer<std::uint32_t> d_candidates(top_k);
    DeviceBuffer<float> d_probabilities(top_k);
    std::vector<double> scores(top_k);
    for (std::size_t c = 0; c < top_k; ++c) {
      scores[c] = logits[sorted[c]];
      for (std::size_t r = 0; r < rank; ++r) {
        scores[c] +=
            static_cast<double>(fp32(predecessors[anchor * rank + r])) *
            projected[r] * fp32(successors[sorted[c] * rank + r]);
      }
    }
    for (const float temperature : {0.0F, 0.8F, 1e-38F}) {
      // Unwritten partial slots must never enter the candidate set.
      d_scores.CopyFrom(std::vector<float>(scratch_size, 1e20F));
      d_ids.CopyFrom(std::vector<std::uint32_t>(scratch_size, vocab - 1));
      gufo::hip::kernels::LaunchDFlashSelectorStep(
          d_logits.data(), d_projected.data(), d_predecessors.data(),
          d_successors.data(), d_anchor.data(), d_selected.data(),
          d_confidence.data(), d_scores.data(), d_ids.data(), temperature,
          d_uniform.data(), d_candidates.data(), d_probabilities.data(), vocab,
          rank, top_k, nullptr);
      const auto candidates = d_candidates.CopyToHost();
      gufo::test::Expect(
          std::equal(candidates.begin(), candidates.end(), sorted.begin()),
          "DFlash selector top-k differs from full vocabulary sort");
      const auto probabilities = d_probabilities.CopyToHost();
      const double maximum = *std::ranges::max_element(scores);
      const double divisor = temperature > 0 ? temperature : 1.0;
      double denominator = 0;
      for (const double value : scores)
        denominator += std::exp((value - maximum) / divisor);
      std::size_t selected = std::ranges::max_element(scores) - scores.begin();
      if (temperature > 0) {
        double cumulative = 0;
        selected = top_k - 1;
        for (std::size_t c = 0; c < top_k; ++c) {
          const double p =
              std::exp((scores[c] - maximum) / divisor) / denominator;
          if (!std::isfinite(probabilities[c]) ||
              std::abs(probabilities[c] - p) >= 2e-6) {
            std::cerr << "selector vocab=" << vocab << " k=" << top_k
                      << " temperature=" << temperature << " candidate=" << c
                      << " actual=" << probabilities[c] << " expected=" << p
                      << '\n';
          }
          gufo::test::Expect(std::isfinite(probabilities[c]) &&
                                 std::abs(probabilities[c] - p) < 2e-6,
                             "DFlash sparse proposal probability is incorrect");
          const double previous = cumulative;
          cumulative += p;
          if (previous < 0.37 && cumulative >= 0.37)
            selected = c;
        }
      }
      gufo::test::Expect(d_selected.CopyToHost()[0] == sorted[selected],
                         "DFlash selector chose the wrong candidate");
      const double confidence =
          temperature > 0
              ? std::exp((scores[selected] - maximum) / divisor) / denominator
              : 1.0;
      if (temperature == 0) {
        for (std::size_t c = 0; c < top_k; ++c)
          gufo::test::Expect(probabilities[c] == (c == selected ? 1.0F : 0.0F),
                             "greedy proposal probabilities must be one-hot");
      }
      gufo::test::Expect(
          std::abs(d_confidence.CopyToHost()[0] - confidence) < 2e-6,
          "DFlash selected-token probability is incorrect");
      if (vocab == 2065 && top_k == 16 && temperature == 0.8F)
        TestProposalVerification(candidates, probabilities, vocab);
    }
  }
}

void TestSelectorZeroUniform() {
  using gufo::test::DeviceBuffer;
  const auto bf16 = [](float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >>
                                      16);
  };
  // Reranking makes the first unary candidate have zero sampling mass.
  DeviceBuffer<float> logits(std::vector<float>{1, 0, -2000}),
      projected(std::vector<float>{1}), uniform(std::vector<float>{0}),
      confidence(1), probabilities(2);
  DeviceBuffer<std::uint16_t> predecessors(
      std::vector<std::uint16_t>{bf16(0), bf16(0), bf16(1)}),
      successors(std::vector<std::uint16_t>{bf16(-1000), bf16(0), bf16(0)});
  DeviceBuffer<std::uint32_t> anchor(std::vector<std::uint32_t>{2}),
      selected(1), candidates(2);
  const auto size = gufo::hip::kernels::DFlashSelectorScratchElements(3);
  DeviceBuffer<float> scores(size);
  DeviceBuffer<std::uint32_t> ids(size);
  gufo::hip::kernels::LaunchDFlashSelectorStep(
      logits.data(), projected.data(), predecessors.data(), successors.data(),
      anchor.data(), selected.data(), confidence.data(), scores.data(),
      ids.data(), 1.0F, uniform.data(), candidates.data(), probabilities.data(),
      3, 1, 2, nullptr);
  gufo::test::Expect(
      selected.CopyToHost()[0] == 1 && confidence.CopyToHost()[0] == 1.0F,
      "a zero random draw must not select zero probability");
}

void TestBatchedSelector() {
  using gufo::hip::kernels::DFlashSelectorSequence;
  using gufo::test::DeviceBuffer;
  using gufo::test::Expect;
  constexpr std::uint32_t vocab = 2065, rank = 32;
  std::vector<std::uint16_t> predecessor(vocab * rank), successor(vocab * rank);
  for (std::size_t index = 0; index < predecessor.size(); ++index) {
    predecessor[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(Sample(index, 31)) >> 16);
    successor[index] = static_cast<std::uint16_t>(
        std::bit_cast<std::uint32_t>(Sample(index, 47)) >> 16);
  }
  DeviceBuffer<std::uint16_t> d_predecessor(predecessor),
      d_successor(successor);
  const auto exact = [](const auto& left, const auto& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(),
                      [](float a, float b) {
                        return std::bit_cast<std::uint32_t>(a) ==
                               std::bit_cast<std::uint32_t>(b);
                      });
  };
  for (const auto users : {2U, 4U, 6U, 8U}) {
    std::vector<DFlashSelectorSequence> sequences(users);
    std::uint32_t rows = 0;
    for (std::uint32_t index = 0; index < users; ++index) {
      sequences[index] = {1U + (index * 3U) % 7U,
                          std::array{0.0F, 0.8F, 1e-38F, 1.2F}[index % 4]};
      rows += sequences[index].count;
    }
    std::vector<std::uint32_t> tokens(rows + users, 0);
    for (std::uint32_t index = 0, offset = 0; index < users; ++index) {
      tokens[offset] = 7U + index;
      offset += sequences[index].count + 1U;
    }
    std::vector<float> logits(rows * vocab), hidden(rows * rank),
        uniforms(rows);
    for (std::size_t index = 0; index < logits.size(); ++index)
      logits[index] = Sample(index, 73);
    for (std::size_t index = 0; index < hidden.size(); ++index)
      hidden[index] = Sample(index, 89);
    for (std::uint32_t row = 0; row < rows; ++row)
      uniforms[row] = static_cast<float>((row * 37U) % 101U) / 101.0F;
    DeviceBuffer<float> d_logits(logits), d_hidden(hidden),
        d_uniforms(uniforms), d_confidences(rows);
    DeviceBuffer<std::uint32_t> d_tokens(tokens);
    const auto partials =
        gufo::hip::kernels::DFlashSelectorScratchElements(vocab);
    DeviceBuffer<float> d_scores(rows * partials);
    DeviceBuffer<std::uint32_t> d_ids(rows * partials);
    for (const auto top_k : {1U, 7U, 16U}) {
      DeviceBuffer<float> d_probabilities(
          std::vector<float>(rows * top_k, -1.0F));
      DeviceBuffer<std::uint32_t> d_candidates(
          std::vector<std::uint32_t>(rows * top_k, vocab));
      d_tokens.CopyFrom(tokens);
      for (std::uint32_t index = 0, first_row = 0; index < users; ++index) {
        const auto sequence = sequences[index];
        for (std::uint32_t step = 0; step < sequence.count; ++step) {
          const auto row = first_row + step;
          gufo::hip::kernels::LaunchDFlashSelectorStep(
              d_logits.data() + row * vocab, d_hidden.data() + row * rank,
              d_predecessor.data(), d_successor.data(),
              d_tokens.data() + row + index, d_tokens.data() + row + index + 1U,
              d_confidences.data() + row, d_scores.data(), d_ids.data(),
              sequence.temperature, d_uniforms.data() + row,
              sequence.temperature > 0 ? d_candidates.data() + row * top_k
                                       : nullptr,
              sequence.temperature > 0 ? d_probabilities.data() + row * top_k
                                       : nullptr,
              vocab, rank, top_k, nullptr);
        }
        first_row += sequence.count;
      }
      const auto expected_tokens = d_tokens.CopyToHost();
      const auto expected_candidates = d_candidates.CopyToHost();
      const auto expected_probabilities = d_probabilities.CopyToHost();
      const auto expected_confidences = d_confidences.CopyToHost();
      d_tokens.CopyFrom(tokens);
      d_candidates.CopyFrom(std::vector<std::uint32_t>(rows * top_k, vocab));
      d_probabilities.CopyFrom(std::vector<float>(rows * top_k, -1.0F));
      d_confidences.CopyFrom(std::vector<float>(rows, -1.0F));
      d_scores.CopyFrom(std::vector<float>(rows * partials, 1e20F));
      d_ids.CopyFrom(std::vector<std::uint32_t>(rows * partials, vocab - 1));
      gufo::hip::kernels::LaunchDFlashSelectorBatch(
          d_logits.data(), d_hidden.data(), d_predecessor.data(),
          d_successor.data(), d_tokens.data(), d_confidences.data(),
          d_scores.data(), d_ids.data(), d_uniforms.data(), d_candidates.data(),
          d_probabilities.data(), sequences, vocab, rank, top_k, nullptr);
      const auto actual_tokens = d_tokens.CopyToHost();
      const auto actual_candidates = d_candidates.CopyToHost();
      const auto actual_probabilities = d_probabilities.CopyToHost();
      const auto actual_confidences = d_confidences.CopyToHost();
      const bool matches =
          actual_tokens == expected_tokens &&
          actual_candidates == expected_candidates &&
          exact(actual_probabilities, expected_probabilities) &&
          exact(actual_confidences, expected_confidences);
      if (!matches) {
        std::cerr << "selector C=" << users << " top_k=" << top_k
                  << " tokens=" << (actual_tokens == expected_tokens)
                  << " candidates="
                  << (actual_candidates == expected_candidates)
                  << " probabilities="
                  << exact(actual_probabilities, expected_probabilities)
                  << " confidences="
                  << exact(actual_confidences, expected_confidences) << '\n';
      }
      Expect(matches,
             "batched selectors must preserve ragged private token chains, "
             "sampled probabilities and uniforms");
    }
  }
}

/// Runs the reference and one candidate route over the same operands and
/// reports the largest absolute difference. The kernels accumulate the QK dot
/// product in a different order -- the reference walks head_dim in one thread,
/// the candidates reduce across a wave -- so the comparison is a float
/// tolerance, not bit equality.
float MaxRouteDifference(Route route, std::uint32_t current_pos,
                         std::uint32_t history_length,
                         std::uint32_t block_count) {
  const std::size_t q_dim = static_cast<std::size_t>(kNumQueryHeads) * kHeadDim;
  const std::size_t kv_dim = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t q_elements = block_count * q_dim;
  const std::size_t block_elements = block_count * kv_dim;
  const std::size_t history_elements =
      static_cast<std::size_t>(current_pos + block_count) * kv_dim;

  std::vector<float> h_q(q_elements);
  std::vector<float> h_block_k(block_elements);
  std::vector<float> h_block_v(block_elements);
  std::vector<float> h_injected_k(history_elements);
  std::vector<float> h_injected_v(history_elements);
  for (std::size_t index = 0; index < q_elements; ++index) {
    h_q[index] = Sample(index, 1);
  }
  for (std::size_t index = 0; index < block_elements; ++index) {
    h_block_k[index] = Sample(index, 2);
    h_block_v[index] = Sample(index, 3);
  }
  for (std::size_t index = 0; index < history_elements; ++index) {
    h_injected_k[index] = Sample(index, 4);
    h_injected_v[index] = Sample(index, 5);
  }

  const auto upload = [](const std::vector<float>& host) {
    float* device = nullptr;
    HIP_CHECK(hipMalloc(&device, host.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(device, host.data(), host.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    return device;
  };

  // Store only live history in a wrapped ring. Poison all other slots so an
  // off-by-one mask or addressing error cannot accidentally compare equal.
  std::vector<float> ring_k(kSlidingWindow * kv_dim, NAN);
  std::vector<float> ring_v(kSlidingWindow * kv_dim, NAN);
  const auto first =
      history_length > kSlidingWindow ? history_length - kSlidingWindow : 0U;
  for (auto pos = first; pos < history_length; ++pos) {
    std::copy_n(h_injected_k.data() + pos * kv_dim, kv_dim,
                ring_k.data() + (pos % kSlidingWindow) * kv_dim);
    std::copy_n(h_injected_v.data() + pos * kv_dim, kv_dim,
                ring_v.data() + (pos % kSlidingWindow) * kv_dim);
  }
  float* d_ring_k = upload(ring_k);
  float* d_ring_v = upload(ring_v);
  float* d_q = upload(h_q);
  float* d_block_k = upload(h_block_k);
  float* d_block_v = upload(h_block_v);
  float* d_injected_k = upload(h_injected_k);
  float* d_injected_v = upload(h_injected_v);
  float* d_scalar_out = nullptr;
  float* d_wave_out = nullptr;
  HIP_CHECK(hipMalloc(&d_scalar_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wave_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_scalar_out, 0, q_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_wave_out, 0, q_elements * sizeof(float)));

  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  gufo::hip::kernels::LaunchDFlashNonCausalAttentionScalar(
      d_q, d_injected_k, d_injected_v, d_block_k, d_block_v, d_scalar_out,
      current_pos, history_length, block_count, kSlidingWindow, kNumQueryHeads,
      kNumKvHeads, kHeadDim, scale, nullptr, current_pos + block_count);
  if (route == Route::kWave) {
    gufo::hip::kernels::LaunchDFlashNonCausalAttentionWave(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, kSlidingWindow, kNumQueryHeads,
        kNumKvHeads, kHeadDim, scale, nullptr, kSlidingWindow);
  }
  if (route == Route::kScalar) {
    gufo::hip::kernels::LaunchDFlashNonCausalAttentionScalar(
        d_q, d_ring_k, d_ring_v, d_block_k, d_block_v, d_wave_out, current_pos,
        history_length, block_count, kSlidingWindow, kNumQueryHeads,
        kNumKvHeads, kHeadDim, scale, nullptr, kSlidingWindow);
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> scalar_out(q_elements);
  std::vector<float> wave_out(q_elements);
  HIP_CHECK(hipMemcpy(scalar_out.data(), d_scalar_out,
                      q_elements * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wave_out.data(), d_wave_out, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_ring_k));
  HIP_CHECK(hipFree(d_ring_v));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_block_k));
  HIP_CHECK(hipFree(d_block_v));
  HIP_CHECK(hipFree(d_injected_k));
  HIP_CHECK(hipFree(d_injected_v));
  HIP_CHECK(hipFree(d_scalar_out));
  HIP_CHECK(hipFree(d_wave_out));

  float max_difference = 0.0F;
  for (std::size_t index = 0; index < q_elements; ++index) {
    gufo::test::Expect(std::isfinite(wave_out[index]),
                       "DFlash candidate attention produced a non-finite "
                       "output");
    max_difference =
        std::max(max_difference, std::abs(scalar_out[index] - wave_out[index]));
  }
  // Independent double-precision softmax for the first/last query and GQA
  // group. All proposal keys are visible; only historical keys are windowed.
  for (const auto query : {0U, block_count - 1}) {
    for (const auto head : {0U, kNumQueryHeads - 1}) {
      const auto kv_head = head / (kNumQueryHeads / kNumKvHeads);
      const auto q_offset = query * q_dim + head * kHeadDim;
      std::vector<double> scores(history_length + block_count, -INFINITY);
      double maximum = -INFINITY;
      for (std::size_t key = 0; key < scores.size(); ++key) {
        if (key < history_length && current_pos + query - key >= kSlidingWindow)
          continue;
        const auto* keys =
            key < history_length ? h_injected_k.data() : h_block_k.data();
        const auto row = key < history_length ? key : key - history_length;
        double dot = 0;
        for (std::size_t d = 0; d < kHeadDim; ++d) {
          dot += static_cast<double>(h_q[q_offset + d]) *
                 keys[row * kv_dim + kv_head * kHeadDim + d];
        }
        scores[key] = dot * scale;
        maximum = std::max(maximum, scores[key]);
      }
      double denominator = 0;
      for (auto& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (std::size_t d = 0; d < kHeadDim; ++d) {
        double reference = 0;
        for (std::size_t key = 0; key < scores.size(); ++key) {
          const auto* values =
              key < history_length ? h_injected_v.data() : h_block_v.data();
          const auto row = key < history_length ? key : key - history_length;
          reference +=
              scores[key] * values[row * kv_dim + kv_head * kHeadDim + d];
        }
        gufo::test::Expect(
            std::abs(wave_out[q_offset + d] - reference / denominator) < 1e-5,
            "DFlash attention differs from the windowed softmax equation");
      }
    }
  }
  return max_difference;
}

void TestRouteEquivalence(std::uint32_t current_pos,
                          std::uint32_t history_length, const char* label,
                          std::uint32_t block_count = kBlockCount) {
  gufo::test::Expect(MaxRouteDifference(Route::kScalar, current_pos,
                                        history_length, block_count) == 0.0F,
                     "ring addressing must preserve scalar attention exactly");
  const float wave = MaxRouteDifference(Route::kWave, current_pos,
                                        history_length, block_count);
  std::cout << "DFlash non-causal attention " << label
            << ": max |scalar - wave| = " << wave << "\n";
  gufo::test::Expect(wave < 1e-5F,
                     "DFlash wave attention diverges from the reference");
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen DFlash non-causal attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestDynamicConvolution();
  TestSelectorZeroUniform();
  TestSelector(2065);
  TestSelector(248320);
  TestBatchedSelector();
  // No injected history at all: only the eight in-block keys are attended.
  TestRouteEquivalence(0, 0, "empty history");
  // Shallow: the whole history is inside the sliding window.
  TestRouteEquivalence(128, 128, "128-token history");
  // Prefetch boundaries and unequal draft widths cover both head-group routes.
  for (const auto rows : {1U, 3U, 4U, 7U}) {
    TestRouteEquivalence(31, 31, "short prefetch tail", rows);
    TestRouteEquivalence(32, 32, "complete prefetch tile", rows);
    TestRouteEquivalence(33, 33, "prefetch tile plus tail", rows);
    TestRouteEquivalence(2051, 2051, "ragged wrapped window", rows);
  }
  // Deep enough that the window clips the history, which is the case the
  // coalesced route exists for.
  TestRouteEquivalence(2047, 2047, "window minus one");
  TestRouteEquivalence(2048, 2048, "window boundary");
  TestRouteEquivalence(2051, 2051, "wrapped window");
  TestRouteEquivalence(4096, 4096, "4096-token history, window-clipped");
  std::cout << "Qwen DFlash non-causal attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen DFlash non-causal attention ops "
               "test.\n";
  return 77;
#endif
}
