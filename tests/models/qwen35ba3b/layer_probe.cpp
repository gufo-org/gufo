// Layer-wise hidden-state probe for the qwen35moe bring-up: compares the CPU
// reference hidden state against the GPU executor's tapped layers for a short
// fixed prompt, then runs the real layer-0 MoE weights through every GPU
// kernel family against the CPU module on identical input. Diagnostics only;
// exits 77 without GUFO_QWEN35BA3B_MODEL.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/modules/moe.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {
using Token = gufo::tokenization::TokenId;

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct HostBuffer {
  void* data = nullptr;
  std::size_t bytes = 0;
  HostBuffer(const void* src, std::size_t size) : bytes(size) {
    HIP_CHECK(hipMalloc(&data, size));
    HIP_CHECK(hipMemcpy(data, src, size, hipMemcpyHostToDevice));
  }
  ~HostBuffer() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
};

/// Runs the real weights of one MoE layer through the GPU decode sequence and
/// the prefill-specific router/shexp routes, comparing against the CPU module.
void CheckRealWeightsMoe(const gufo::models::QwenLayerWeights& layer,
                         const gufo::core::ModelConfig& config,
                         const std::vector<float>& x, std::uint32_t layer_idx,
                         hipStream_t stream) {
  const auto view = gufo::models::qwen::MakeMoeView(layer, config);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t n_experts = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::size_t expert_ff = config.expert_ff_length;
  const std::size_t shared_ff = config.expert_shared_ff_length;

  HostBuffer d_x(x.data(), x.size() * sizeof(float));
  HostBuffer d_router(view.router.data, view.router.EncodedSizeBytes());
  HostBuffer d_shexp_gi(view.shexp_gate_inp.data,
                        view.shexp_gate_inp.EncodedSizeBytes());
  HostBuffer d_gate(view.gate_exps.data, view.gate_exps.EncodedSizeBytes());
  HostBuffer d_up(view.up_exps.data, view.up_exps.EncodedSizeBytes());
  HostBuffer d_down(view.down_exps.data, view.down_exps.EncodedSizeBytes());
  HostBuffer d_shexp_gate(view.shexp_gate.data,
                          view.shexp_gate.EncodedSizeBytes());
  HostBuffer d_shexp_up(view.shexp_up.data, view.shexp_up.EncodedSizeBytes());
  HostBuffer d_shexp_down(view.shexp_down.data,
                          view.shexp_down.EncodedSizeBytes());

  const std::size_t slots = n_used;
  std::vector<float> zero_f(std::max<std::size_t>(n_experts, hidden * slots),
                            0.0F);
  HostBuffer d_logits(zero_f.data(), n_experts * sizeof(float));
  HostBuffer d_shexp_gate_scalar(zero_f.data(), sizeof(float));
  std::vector<std::int32_t> zero_i(slots, 0);
  HostBuffer d_ids(zero_i.data(), slots * sizeof(std::int32_t));
  HostBuffer d_weights(zero_f.data(), slots * sizeof(float));
  HostBuffer d_gate_e(zero_f.data(), slots * expert_ff * sizeof(float));
  HostBuffer d_up_e(zero_f.data(), slots * expert_ff * sizeof(float));
  HostBuffer d_down_e(zero_f.data(), slots * hidden * sizeof(float));
  HostBuffer d_shexp_act(zero_f.data(), shared_ff * sizeof(float));
  HostBuffer d_shexp_out(zero_f.data(), hidden * sizeof(float));
  HostBuffer d_out(zero_f.data(), hidden * sizeof(float));

  // --- decode-style router: LaunchGEMV ---
  gufo::hip::LaunchGEMV(
      d_router.data, view.router.type, static_cast<const float*>(d_x.data),
      static_cast<float*>(d_logits.data), n_experts, hidden, stream);
  gufo::hip::LaunchGEMV(d_shexp_gi.data, view.shexp_gate_inp.type,
                        static_cast<const float*>(d_x.data),
                        static_cast<float*>(d_shexp_gate_scalar.data), 1,
                        hidden, stream);
  gufo::hip::LaunchMoeRouterTopK(
      static_cast<const float*>(d_logits.data), n_experts,
      static_cast<std::int32_t*>(d_ids.data),
      static_cast<float*>(d_weights.data), 1, n_experts, n_used, stream);

  // --- prefill-style router: LaunchHipblasGEMM (batch 1) into a second buffer
  // ---
  HostBuffer d_logits2(zero_f.data(), n_experts * sizeof(float));
  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));
  HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  std::vector<std::uint8_t> scratch(hidden * sizeof(hip_bfloat16));
  HostBuffer d_scratch(scratch.data(), scratch.size());
  gufo::hip::LaunchHipblasGEMM(handle, d_router.data, false,
                               static_cast<const float*>(d_x.data),
                               static_cast<float*>(d_logits2.data), 1,
                               n_experts, hidden, d_scratch.data, stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  hipblasDestroy(handle);

  std::vector<float> logits(n_experts);
  HIP_CHECK(hipMemcpy(logits.data(), d_logits.data, n_experts * sizeof(float),
                      hipMemcpyDeviceToHost));
  std::vector<float> logits2(n_experts);
  HIP_CHECK(hipMemcpy(logits2.data(), d_logits2.data, n_experts * sizeof(float),
                      hipMemcpyDeviceToHost));
  float router_max_diff = 0.0F;
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    router_max_diff =
        std::max(router_max_diff, std::fabs(logits[e] - logits2[e]));
  }
  std::cout << "layer " << layer_idx
            << " router GEMV-vs-blas max_diff=" << router_max_diff << "\n";
  Expect(router_max_diff < 0.05F, "router GEMM routes disagree");

  // --- shared expert: decode-style kernels ---
  gufo::hip::LaunchFusedSwiGLUGEMV(
      d_shexp_gate.data, view.shexp_gate.type, d_shexp_up.data,
      view.shexp_up.type, static_cast<const float*>(d_x.data),
      static_cast<float*>(d_shexp_act.data), shared_ff, hidden, stream);
  gufo::hip::LaunchGEMV(d_shexp_down.data, view.shexp_down.type,
                        static_cast<const float*>(d_shexp_act.data),
                        static_cast<float*>(d_shexp_out.data), hidden,
                        shared_ff, stream);

  // --- routed experts: MMQ + fallback ---
  const bool q8 = view.gate_exps.type == gufo::core::GgmlType::kQ8_0 &&
                  view.up_exps.type == gufo::core::GgmlType::kQ8_0 &&
                  view.down_exps.type == gufo::core::GgmlType::kQ8_0;
  if (q8) {
    Expect(qfn_mmq_moe_gated_vec(
               static_cast<int>(gufo::core::GgmlType::kQ8_0), d_gate.data,
               d_up.data, static_cast<const float*>(d_x.data),
               static_cast<const std::int32_t*>(d_ids.data),
               static_cast<float*>(d_gate_e.data), static_cast<int>(expert_ff),
               static_cast<int>(hidden), 1, static_cast<int>(n_experts),
               static_cast<int>(n_used), stream) == 0,
           "gated vec failed");
    Expect(qfn_mmq_moe_vec(
               static_cast<int>(gufo::core::GgmlType::kQ8_0), d_down.data,
               static_cast<const float*>(d_gate_e.data),
               static_cast<const std::int32_t*>(d_ids.data),
               static_cast<float*>(d_down_e.data), static_cast<int>(hidden),
               static_cast<int>(expert_ff), static_cast<int>(n_used),
               static_cast<int>(n_experts), 1, stream, nullptr, nullptr) == 0,
           "moe vec failed");
  } else {
    gufo::hip::LaunchMoeSlotSwigluGemv(
        d_gate.data, view.gate_exps.type, d_up.data, view.up_exps.type,
        static_cast<const float*>(d_x.data),
        static_cast<const std::int32_t*>(d_ids.data),
        static_cast<float*>(d_gate_e.data), expert_ff, hidden, n_used, n_used,
        stream);
    gufo::hip::LaunchMoeSlotGemv(d_down.data, view.down_exps.type,
                                 static_cast<const float*>(d_gate_e.data),
                                 static_cast<const std::int32_t*>(d_ids.data),
                                 static_cast<float*>(d_down_e.data), hidden,
                                 expert_ff, n_used, 1, stream);
  }

  gufo::hip::LaunchMoeEpilogue(
      static_cast<const float*>(d_down_e.data),
      static_cast<const float*>(d_weights.data),
      static_cast<const float*>(d_shexp_out.data),
      static_cast<const float*>(d_shexp_gate_scalar.data),
      static_cast<float*>(d_out.data), 1, n_used, hidden, stream);

  std::vector<float> gpu_out(hidden);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(gpu_out.data(), d_out.data, hidden * sizeof(float),
                      hipMemcpyDeviceToHost));

  gufo::models::QwenScratchArena arena(config);
  std::vector<float> cpu_out(hidden, 0.0F);
  const gufo::models::qwen::CpuModuleContext ctx;
  gufo::models::qwen::MoeForward(ctx, view, x, arena, cpu_out);

  double max_abs = 0.0;
  double sum_sq = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < hidden; ++i) {
    const double d = std::fabs(static_cast<double>(gpu_out[i]) - cpu_out[i]);
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
    sum_sq += d * d;
  }
  const double cpu_scale = std::sqrt(
      std::inner_product(cpu_out.begin(), cpu_out.end(), cpu_out.begin(), 0.0) /
      static_cast<double>(hidden));
  std::cout << "layer " << layer_idx
            << " real-weights MoE vs CPU: max_abs=" << max_abs
            << " rmse=" << std::sqrt(sum_sq / hidden)
            << " cpu_rms=" << cpu_scale << " worst_idx=" << worst
            << " gpu=" << gpu_out[worst] << " cpu=" << cpu_out[worst] << "\n";
  Expect(max_abs < 0.02 + 0.02 * cpu_scale + 0.05 * std::fabs(cpu_out[worst]),
         "real-weights MoE GPU/CPU mismatch");
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
    Expect(reader != nullptr, error);
    auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(model != nullptr, error);
    const auto& config = model->GetConfig();

    const std::vector<Token> tokens =
        model->GetTokenizer().Encode("The capital of France is");
    Expect(!tokens.empty(), "probe prompt must tokenize");

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
    // Snapshot the FFN-normed input of layer 0 for the real-weights check.
    gufo::models::ForwardRMSNorm(arena.hidden, cpu_weights->layers[0].attn_norm,
                                 1e-6F, arena.normed);
    std::vector<float> layer0_normed(arena.normed.begin(), arena.normed.end());

    for (std::uint32_t l = 0; l <= kTaps.back(); ++l) {
      gufo::models::ForwardLayer(arena.hidden, cpu_weights->layers[l], config,
                                 kv, ssm, l, 0, arena);
      const auto it = std::ranges::find(kTaps, l);
      if (it != kTaps.end()) {
        cpu_hidden[static_cast<std::size_t>(it - kTaps.begin())].assign(
            arena.hidden.begin(), arena.hidden.end());
      }
    }

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
    if (any_bad) {
      std::cout << "NOTE: GPU/CPU hidden states diverge; continuing into the "
                   "real-weights MoE diagnostics\n";
    }

    // Real-weights MoE kernel checks: SSM/q8_0 layer 0, BF16 layer 1, and a
    // full-attention MoE layer (3).
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    Expect(qfn_mmq_init(0) == 0, "qfn_mmq_init failed");
    CheckRealWeightsMoe(cpu_weights->layers[0], config, layer0_normed, 0,
                        stream);
    gufo::models::ForwardRMSNorm(arena.hidden, cpu_weights->layers[1].attn_norm,
                                 1e-6F, arena.normed);
    CheckRealWeightsMoe(
        cpu_weights->layers[1], config,
        std::vector<float>(arena.normed.begin(), arena.normed.end()), 1,
        stream);
    HIP_CHECK(hipStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::cerr << "qwen35ba3b_layer_probe: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen35ba3b_layer_probe: PASS\n";
  return 0;
}
