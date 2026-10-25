// Qwen3.5/3.6-35B-A3B (qwen35moe) target checks: GPU executor logits against
// the CPU reference forward pass (teacher forcing on identical prefixes), plus
// GPU prefill/decode self-consistency. Model path comes from
// GUFO_QWEN35BA3B_MODEL; the test skips when it is unset.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::array<const char*, 2> kTexts{
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs when the requested page is unavailable.",
    "A train leaves the station at noon and travels at sixty kilometers per "
    "hour. Another train leaves one hour later at ninety kilometers per hour.",
};

struct CpuReference {
  gufo::models::QwenModelWeights weights;
  gufo::models::QwenKvCache kv;
  gufo::models::QwenSsmCache ssm;
  gufo::models::QwenScratchArena arena;

  explicit CpuReference(gufo::models::QwenModelWeights w)
      : weights(std::move(w)),
        kv(weights.config.FullAttentionLayerCount(),
           weights.config.num_key_value_heads, 8192, weights.config.head_dim),
        ssm(weights.config.num_layers, weights.config.SsmQkvSize(),
            weights.config.ssm_conv_kernel, weights.config.ssm_time_step_rank,
            weights.config.ssm_state_size, weights.config.SsmValueSize()),
        arena(weights.config) {}

  std::span<const float> Step(Token token, std::uint32_t pos) {
    gufo::models::ForwardModel(token, pos, weights, kv, ssm, arena,
                               arena.logits);
    return arena.logits;
  }
};

void CheckCpuGpuLogitParity(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model,
    const gufo::core::GgufReader& reader) {
  std::string error;
  auto executor = Executor::Create(model, &error, 4096);
  Expect(executor != nullptr, error);

  for (const char* text : kTexts) {
    auto cpu_weights =
        gufo::models::QwenModelWeights::LoadFromGguf(reader, &error);
    Expect(cpu_weights.has_value(), error);
    const auto tokens = executor->GetTokenizer().Encode(text);
    Expect(tokens.size() >= 12, "parity fixture must tokenize");
    constexpr std::size_t kSteps = 12;

    CpuReference cpu(std::move(*cpu_weights));
    cpu.kv.Reset();
    cpu.ssm.Reset();

    // Teacher forcing on identical prefixes. The GPU exposes logits for the
    // final prefilled position only, so each position re-prefills its prefix.
    for (std::size_t p = 0; p < kSteps; ++p) {
      const auto cpu_logits =
          cpu.Step(tokens[p], static_cast<std::uint32_t>(p));
      executor->Reset();
      (void)executor->ForwardPromptBatch(std::span(tokens).first(p + 1), 0,
                                         true);
      const auto view = executor->CopyLastLogits();
      Expect(!view.empty(), "GPU logits must be nonempty");
      const std::vector<float> gpu_logits(view.begin(), view.end());
      const auto result =
          gufo::testing::CompareLogits(cpu_logits, gpu_logits, 5e-2F, 5e-2F);
      Expect(result.finite, "GPU logits must be finite");
      Expect(result.top1_match, "GPU/CPU greedy mismatch at position " +
                                    std::to_string(p) + ": " + result.details);
      std::cout << "pos " << p << " max_abs=" << result.max_abs_diff
                << " rmse=" << result.root_mean_square_error << "\n";
    }
  }
}

void CheckPrefillDecodeParity(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model) {
  std::string error;
  auto executor = Executor::Create(model, &error, 4096);
  Expect(executor != nullptr, error);
  const auto tokens = executor->GetTokenizer().Encode(kTexts[0]);
  Expect(tokens.size() >= 16, "parity fixture must tokenize");

  executor->Reset();
  const Token prefill_next = executor->ForwardPromptBatch(tokens);
  executor->Reset();
  Token decode_next = 0;
  for (std::size_t p = 0; p < tokens.size(); ++p) {
    decode_next =
        executor->ForwardToken(tokens[p], static_cast<std::uint32_t>(p));
  }
  Expect(prefill_next == decode_next,
         "prefill and single-step decode disagree on the next token");
}

// Concurrent speculative verification stacks every session's draft rows into
// one MoE batch (up to 8 sessions x 8 rows). Three sessions with 8 + 8 + 7
// rows exceed the Q8_0 gated vector kernel's 8-row launch limit and leave a
// partial slice; each row must agree with that session verifying alone.
void CheckConcurrentVerification(
    const std::shared_ptr<const gufo::hip::QwenGpuModel>& model) {
  constexpr std::array<std::size_t, 3> kRows{8, 8, 7};
  std::vector<std::unique_ptr<Executor>> sessions;
  std::vector<std::vector<Token>> tokens;
  std::vector<std::uint32_t> positions;
  std::vector<std::unique_ptr<gufo::hip::QwenGpuSnapshot>> snapshots;
  std::vector<std::vector<std::vector<float>>> expected;
  for (std::size_t s = 0; s < kRows.size(); ++s) {
    std::string error;
    sessions.push_back(Executor::Create(model, &error, 4096));
    Expect(sessions.back() != nullptr, error);
    auto& session = *sessions.back();
    tokens.push_back(session.GetTokenizer().Encode(kTexts[s % kTexts.size()]));
    Expect(tokens.back().size() > kRows[s] + 4, "fixture text is too short");
    const auto position =
        static_cast<std::uint32_t>(tokens.back().size() - kRows[s]);
    positions.push_back(position);
    session.Reset();
    (void)session.ForwardPromptBatch(std::span(tokens.back()).first(position));
    snapshots.push_back(session.SaveSnapshot(position));
    (void)session.ForwardVerificationChunk(
        std::span(tokens.back()).subspan(position), position, true);
    auto& rows = expected.emplace_back();
    for (std::size_t r = 0; r < kRows[s]; ++r) {
      const auto view = session.CopyVerificationLogits(r);
      rows.emplace_back(view.begin(), view.end());
    }
    session.RestoreSnapshot(*snapshots.back());
  }

  std::vector<gufo::hip::QwenGpuVerificationItem> items;
  for (std::size_t s = 0; s < sessions.size(); ++s) {
    items.push_back({sessions[s].get(),
                     std::span(tokens[s]).subspan(positions[s]), positions[s],
                     true});
  }
  const auto predictions = Executor::ForwardVerificationBatch(items);
  Expect(predictions.size() == sessions.size(),
         "concurrent verification must answer every session");
  for (std::size_t s = 0; s < sessions.size(); ++s) {
    Expect(predictions[s].size() == kRows[s],
           "concurrent verification row count mismatch");
    for (std::size_t r = 0; r < kRows[s]; ++r) {
      const auto view = sessions[s]->CopyVerificationLogits(r);
      const std::vector<float> actual(view.begin(), view.end());
      const auto result =
          gufo::testing::CompareLogits(expected[s][r], actual, 5e-2F, 5e-2F);
      Expect(result.finite, "concurrent verification logits must be finite");
      Expect(result.top1_match,
             "concurrent verification changed the greedy token: session " +
                 std::to_string(s) + " row " + std::to_string(r) + ": " +
                 result.details);
    }
  }
  std::cout << "concurrent verification: sessions=" << sessions.size()
            << " rows=" << kRows[0] + kRows[1] + kRows[2] << " top1 match\n";
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_QWEN35BA3B_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "GUFO_QWEN35BA3B_MODEL not set; skipping\n";
    return 77;
  }
  try {
    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(model_path, &error);
    Expect(reader != nullptr,
           error.empty() ? "failed to open model GGUF" : error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    Expect(model->GetConfig().IsMoE(), "fixture must be a MoE model");

    CheckCpuGpuLogitParity(model, *reader);
    CheckPrefillDecodeParity(model);
    CheckConcurrentVerification(model);
  } catch (const std::exception& e) {
    std::cerr << "qwen35ba3b_target_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen35ba3b_target_test: PASS\n";
  return 0;
}
