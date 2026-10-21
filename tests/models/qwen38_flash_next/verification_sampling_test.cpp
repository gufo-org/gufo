#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
namespace {

void Check(hipError_t code) {
  if (code != hipSuccess)
    throw std::runtime_error(hipGetErrorString(code));
}

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Delete {
  void operator()(void* p) const { (void)hipFree(p); }
};

template<typename T>
auto Allocate(std::size_t n) {
  T* p = nullptr;
  Check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)));
  return std::unique_ptr<T, Delete>(p);
}

void Compare(std::span<const float> full, std::span<const float> compact,
             std::span<const sampling::LogitCandidate> candidates,
             const sampling::SamplerState& state) {
  const auto a = state.Distribution(full);
  const auto b = state.Distribution(compact);
  const auto c = state.DistributionFromCandidates(candidates);
  Require(a.entries().size() == b.entries().size(), "target support differs");
  Require(a.entries().size() == c.entries().size(), "compact support differs");
  for (std::size_t i = 0; i < a.entries().size(); ++i)
    Require(a.entries()[i].token == b.entries()[i].token &&
                a.entries()[i].token == c.entries()[i].token &&
                std::bit_cast<std::uint64_t>(a.entries()[i].value) ==
                    std::bit_cast<std::uint64_t>(b.entries()[i].value) &&
                std::bit_cast<std::uint64_t>(a.entries()[i].value) ==
                    std::bit_cast<std::uint64_t>(c.entries()[i].value),
            "target probabilities are not bit exact");
  // Include both accepted and rejected point-mass and neural proposals.
  q::MtpCandidateLogits draft;
  draft.size = std::min<std::size_t>(64, full.size());
  for (std::size_t i = 0; i < draft.size; ++i) {
    draft.ids[i] = (i * 37) % full.size();
    draft.logits[i] = float(int(i % 13) - 6) / 10;
  }
  auto proposal_rng = std::uint64_t{987};
  for (unsigned seed = 0; seed < 16; ++seed) {
    auto left = state;
    auto right = state;
    left.SetRngState(seed);
    right.SetRngState(seed);
    auto proposal = q::SampleMtpProposal(draft, state, &proposal_rng);
    if (seed % 2 == 0) {
      proposal.size = 1;
      proposal.ids[0] = seed % 4 ? a.best_token() : full.size() - 1;
      proposal.probabilities[0] = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1;
    }
    const auto x = q::VerifyMtpProposal(full, proposal, left);
    const auto y =
        q::VerifyMtpProposal(candidates, full.size(), proposal, right);
    Require(x.token == y.token && x.accepted == y.accepted &&
                left.rng_state() == right.rng_state(),
            "verification or correction RNG differs");
  }
}

void Run(unsigned vocab, unsigned rows, unsigned mode, bool bench) {
  constexpr auto keep = q::kMtpCandidates;
  const auto stride = q::rocm::MtpCandidateWorkspaceSize(vocab);
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> logits(std::size_t(rows) * vocab);
  std::vector<sampling::TokenPenalty> penalties;
  std::vector<sampling::SamplerState> states;
  q::rocm::GreedyPenaltyRows batch{};
  sampling::SamplingConfig config{.temperature = 0.8F,
                                  .top_k = mode % 3 == 0   ? 1
                                           : mode % 3 == 1 ? 20
                                                           : 64,
                                  .top_p = 0.95F,
                                  .min_p = mode % 2 ? 0.1F : 0.0F,
                                  .min_keep = mode == 3 ? 64U : 1U,
                                  .seed = 83,
                                  .repeat_penalty = mode % 2 ? 0.7F : 1.3F,
                                  .repeat_last_n = 512,
                                  .frequency_penalty = mode % 2 ? -0.4F : 0.3F,
                                  .presence_penalty = mode % 2 ? -1.5F : 1.0F};
  if (mode == 6) {
    config.repeat_penalty = 1;
    config.frequency_penalty = 0;
    config.presence_penalty = 0;
  }
  std::vector<sampling::TokenId> history(std::min(vocab, 300U));
  for (unsigned i = 0; i < history.size(); ++i)
    history[i] = (i * 17) % vocab;
  sampling::SamplerState state(config, history);
  state.Accept(history);
  for (unsigned row = 0; row < rows; ++row) {
    states.push_back(state);
    const auto counts = state.penalties();
    penalties.insert(penalties.end(), counts.begin(), counts.end());
    batch.offsets[row + 1] = penalties.size();
    auto values = std::span(logits).subspan(std::size_t(row) * vocab, vocab);
    for (unsigned i = 0; i < vocab; ++i)
      values[i] = float(int((i * 7919U + row * 71U) % 65521U) - 32768) / 4096;
    if (mode == 3)
      for (unsigned i = 0; i < vocab; ++i)
        values[i] = i % 2 ? 0.0F : -0.0F;
    if (mode == 4) {
      std::fill(values.begin(), values.end(), -inf);
      for (unsigned i = 0; i < std::min(vocab, 17U); ++i)
        values[i] = float(i);
    }
    if (mode == 5) {
      values[0] = inf;
      values[vocab - 1] = std::numeric_limits<float>::quiet_NaN();
    }
    state.Accept((row * 13 + 3) % vocab);
  }
  const auto count = rows * keep + penalties.size();
  auto gpu_logits = Allocate<float>(logits.size());
  auto gpu_penalties = Allocate<sampling::TokenPenalty>(penalties.size());
  auto ids = Allocate<std::uint32_t>(rows * stride);
  auto scratch = Allocate<std::uint32_t>(rows * stride);
  auto gpu_output = Allocate<sampling::LogitCandidate>(count);
  Check(hipMemcpy(gpu_logits.get(), logits.data(),
                  logits.size() * sizeof(float), hipMemcpyHostToDevice));
  Check(hipMemcpy(gpu_penalties.get(), penalties.data(),
                  penalties.size() * sizeof(penalties[0]),
                  hipMemcpyHostToDevice));
  batch.penalties = gpu_penalties.get();
  std::vector<sampling::LogitCandidate> output(count);
  const auto select = [&] {
    q::rocm::VerificationCandidates(gpu_logits.get(), batch, ids.get(),
                                    scratch.get(), gpu_output.get(), rows,
                                    vocab, nullptr);
    Check(hipMemcpy(output.data(), gpu_output.get(), count * sizeof(output[0]),
                    hipMemcpyDeviceToHost));
  };
  select();
  std::vector<float> compact(logits.size(), -inf);
  for (unsigned row = 0; row < rows; ++row) {
    const auto full =
        std::span(logits).subspan(std::size_t(row) * vocab, vocab);
    auto reconstructed =
        std::span(compact).subspan(std::size_t(row) * vocab, vocab);
    const auto result = std::span(output).subspan(
        row * keep + batch.offsets[row],
        keep + batch.offsets[row + 1] - batch.offsets[row]);
    // Independent full CPU sort; no production selector or penalties.
    std::vector<unsigned> expected;
    std::vector<bool> penalized(vocab);
    for (const auto& penalty : states[row].penalties())
      penalized[penalty.token] = true;
    for (unsigned id = 0; id < vocab; ++id)
      if (std::isfinite(full[id]) && !penalized[id])
        expected.push_back(id);
    std::ranges::sort(expected, [&](auto a, auto b) {
      return full[a] > full[b] || (full[a] == full[b] && a < b);
    });
    expected.resize(std::min<std::size_t>(keep, expected.size()));
    for (unsigned i = 0; i < keep; ++i)
      Require(
          result[i].token == (i < expected.size() ? expected[i] : UINT32_MAX),
          "unpenalized top-k differs from independent sort");
    for (const auto& candidate : result) {
      if (candidate.token >= vocab)
        continue;
      Require(std::bit_cast<std::uint32_t>(candidate.value) ==
                  std::bit_cast<std::uint32_t>(full[candidate.token]),
              "candidate logit bits changed");
      reconstructed[candidate.token] = candidate.value;
    }
    Compare(full, reconstructed, result, states[row]);
  }
  if (bench) {
    // Diagnostic only: complete selection, transfer and CPU target
    // distributions; serving measurements decide whether to retain it.
    std::vector<float> downloaded(logits.size());
    double sink = 0;
    const auto measure = [&](bool reduced) {
      std::vector<double> samples;
      for (unsigned repeat = 0; repeat < 22; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        if (reduced) {
          select();
        } else {
          Check(hipMemcpy(downloaded.data(), gpu_logits.get(),
                          downloaded.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
        }
        for (unsigned row = 0; row < rows; ++row) {
          const auto distribution =
              reduced
                  ? states[row].DistributionFromCandidates(
                        std::span(output).subspan(
                            row * keep + batch.offsets[row],
                            keep + batch.offsets[row + 1] - batch.offsets[row]))
                  : states[row].Distribution(
                        std::span(downloaded)
                            .subspan(std::size_t(row) * vocab, vocab));
          sink += distribution.entries()[0].value;
        }
        if (repeat >= 2)
          samples.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count());
      }
      std::ranges::sort(samples);
      return samples[samples.size() / 2];
    };
    const double full_us = measure(false);
    const double compact_us = measure(true);
    std::cout << "rows=" << rows << " full_us=" << full_us
              << " compact_us=" << compact_us << " sink=" << sink << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--bench") {
      for (unsigned rows : {1, 3, 7})
        Run(248320, rows, 1, true);
    } else {
      for (unsigned vocab : {67, 1023, 1025, 248320})
        for (unsigned rows = 1; rows <= 7; ++rows)
          for (unsigned mode = 0; mode < 7; ++mode)
            Run(vocab, rows, mode, false);
      std::cout
          << "Exact candidate union, target probabilities and rejection "
             "RNG: ragged vocabularies, widths 1..7, penalties and ties\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
