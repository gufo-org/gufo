// Layer-wise hidden-state probe for the qwen35moe bring-up: compares the CPU
// reference hidden state against the GPU executor's tapped layers for a short
// fixed prompt. Diagnostics only; exits 77 without GUFO_QWEN27B_MODEL.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {
using Token = gufo::tokenization::TokenId;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_QWEN27B_MODEL");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "GUFO_QWEN27B_MODEL not set; skipping\n";
    return 77;
  }
  try {
    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(model_path, &error);
    Expect(reader != nullptr, error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    const auto& config = model->GetConfig();

    const std::vector<Token> tokens =
        model->GetTokenizer().Encode("The capital of France is");
    Expect(!tokens.empty(), "probe prompt must tokenize");

    // CPU reference hidden states after each layer.
    auto cpu_weights =
        gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
    Expect(cpu_weights.has_value(), error);
    gufo::models::QwenKvCache kv(config.FullAttentionLayerCount(),
                                 config.num_key_value_heads, 8192,
                                 config.head_dim);
    gufo::models::QwenSsmCache ssm(
        config.num_layers, config.SsmQkvSize(), config.ssm_conv_kernel,
        config.ssm_time_step_rank, config.ssm_state_size,
        config.SsmValueSize());
    gufo::models::QwenScratchArena arena(config);

    constexpr std::array<std::uint32_t, 4> kTaps{0, 1, 2, 3};
    const std::size_t hidden = config.hidden_size;
    std::vector<std::vector<float>> cpu_hidden(kTaps.size());

    gufo::models::ForwardEmbedding(tokens[0], cpu_weights->token_embd, hidden,
                                   arena.hidden);
    for (std::uint32_t l = 0; l <= kTaps.back(); ++l) {
      gufo::models::ForwardLayer(arena.hidden, cpu_weights->layers[l], config,
                                 kv, ssm, l, 0, arena);
      const auto it = std::ranges::find(kTaps, l);
      if (it != kTaps.end()) {
        cpu_hidden[static_cast<std::size_t>(it - kTaps.begin())].assign(
            arena.hidden.begin(), arena.hidden.end());
      }
    }
    // CPU logits after the last tap layer are not needed; the probe stops at
    // layer 3.

    // GPU: capture the same layers for the same first token.
    auto executor = gufo::hip::QwenGpuExecutor::Create(model, &error, 4096);
    Expect(executor != nullptr, error);
    executor->Reset();
    executor->SetPromptHiddenCapture(true, kTaps);
    (void)executor->ForwardPromptBatch(std::span(tokens).first(1), 0, false);
    const auto gpu_hidden = executor->GetPromptHiddenStates();
    Expect(gpu_hidden.size() >= kTaps.size() * hidden,
           "GPU hidden capture too small");

    bool any_bad = false;
    for (std::size_t tap = 0; tap < kTaps.size(); ++tap) {
      const float* gpu_row = gpu_hidden.data() + tap * hidden;
      double max_abs = 0.0;
      double sum_sq = 0.0;
      double ref_sq = 0.0;
      std::size_t worst = 0;
      for (std::size_t i = 0; i < hidden; ++i) {
        const double d =
            std::fabs(static_cast<double>(cpu_hidden[tap][i]) - gpu_row[i]);
        if (d > max_abs) {
          max_abs = d;
          worst = i;
        }
        sum_sq += d * d;
        ref_sq += static_cast<double>(cpu_hidden[tap][i]) * cpu_hidden[tap][i];
      }
      const double rmse = std::sqrt(sum_sq / hidden);
      const double ref_norm = std::sqrt(ref_sq / hidden);
      std::cout << "layer " << kTaps[tap] << " max_abs=" << max_abs
                << " rmse=" << rmse << " ref_rms=" << ref_norm
                << " worst_idx=" << worst << " cpu=" << cpu_hidden[tap][worst]
                << " gpu=" << gpu_row[worst] << "\n";
      any_bad |= ref_norm > 0.0 && rmse > 0.02 * ref_norm + 1e-3;
    }
    Expect(!any_bad, "GPU/CPU hidden states diverge");
  } catch (const std::exception& e) {
    std::cerr << "qwen27bprobe_layer_probe: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen27bprobe_layer_probe: PASS\n";
  return 0;
}
