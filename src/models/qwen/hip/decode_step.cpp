#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/decode_step.hpp"

#include <stdexcept>

#include "qfn_mmq.h"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/modules/moe.hpp"

namespace gufo::hip {

/// Single-token routed MoE FFN (qwen35moe): router + top-k, routed experts
/// (MMQ Q8_0 kernels where possible, per-slot GEMV fallback otherwise),
/// sigmoid-gated shared expert, weighted combine into `out`.
void ExecuteMoeDecodeStep(hipStream_t stream, const QwenMoeScratch& moe,
                          const models::QwenLayerWeights& layer,
                          const core::ModelConfig& config, const float* x,
                          float* out) {
  const auto view = models::qwen::MakeMoeView(layer, config);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t n_experts = config.expert_count;
  const std::uint32_t n_used = config.expert_used_count;
  const std::size_t expert_ff = config.expert_ff_length;
  const std::size_t shared_ff = config.expert_shared_ff_length;

  LaunchGEMV(view.router.data, view.router.type, x, moe.router_logits.data(),
             n_experts, hidden, stream);
  LaunchGEMV(view.shexp_gate_inp.data, view.shexp_gate_inp.type, x,
             moe.shexp_gate.data(), 1, hidden, stream);
  LaunchMoeRouterTopK(moe.router_logits.data(), n_experts, moe.ids.data(),
                      moe.weights.data(), 1, n_experts, n_used, stream);

  // Shared expert: plain dense SwiGLU FFN of width shared_ff.
  LaunchFusedSwiGLUGEMV(view.shexp_gate.data, view.shexp_gate.type,
                        view.shexp_up.data, view.shexp_up.type, x,
                        moe.shexp_act.data(), shared_ff, hidden, stream);
  LaunchGEMV(view.shexp_down.data, view.shexp_down.type, moe.shexp_act.data(),
             moe.shexp_out.data(), hidden, shared_ff, stream);

  // Routed experts. The MMQ gated kernel fuses gate+up+SwiGLU for the formats
  // it covers; anything else takes the per-slot warp GEMV.
  constexpr int kGgmlQ8_0 = static_cast<int>(core::GgmlType::kQ8_0);
  const bool mmq_gated = view.gate_exps.type == core::GgmlType::kQ8_0 &&
                         view.up_exps.type == core::GgmlType::kQ8_0;
  if (mmq_gated) {
    if (qfn_mmq_moe_gated_vec(
            kGgmlQ8_0, view.gate_exps.data, view.up_exps.data, x,
            moe.ids.data(), moe.gate_e.data(), static_cast<int>(expert_ff),
            static_cast<int>(hidden), 1, static_cast<int>(n_experts),
            static_cast<int>(n_used), stream) != 0) {
      throw std::runtime_error("MoE gated expert projection failed");
    }
  } else {
    LaunchMoeSlotSwigluGemv(view.gate_exps.data, view.gate_exps.type,
                            view.up_exps.data, view.up_exps.type, x,
                            moe.ids.data(), moe.gate_e.data(), expert_ff,
                            hidden, n_used, n_used, stream);
  }
  if (view.down_exps.type == core::GgmlType::kQ8_0) {
    // Each (token, slot) pair carries its own activation row, so the down
    // projection runs as U independent rows with one expert id each.
    if (qfn_mmq_moe_vec(kGgmlQ8_0, view.down_exps.data, moe.gate_e.data(),
                        moe.ids.data(), moe.down_e.data(),
                        static_cast<int>(hidden), static_cast<int>(expert_ff),
                        static_cast<int>(n_used), static_cast<int>(n_experts),
                        1, stream, nullptr, nullptr) != 0) {
      throw std::runtime_error("MoE down expert projection failed");
    }
  } else {
    LaunchMoeSlotGemv(view.down_exps.data, view.down_exps.type,
                      moe.gate_e.data(), moe.ids.data(), moe.down_e.data(),
                      hidden, expert_ff, n_used, 1, stream);
  }
  LaunchMoeEpilogue(moe.down_e.data(), moe.weights.data(), moe.shexp_out.data(),
                    moe.shexp_gate.data(), out, 1, n_used, hidden, stream);
}

void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy) {
  if (!detail::DispatchTelemetryEnabled()) {
    return;
  }
  for (std::uint32_t layer_index = 0; layer_index < weights.config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    detail::EmitQwenRouteResolution(
        "decode", layer_index, layer.is_full_attention ? "attention" : "ssm",
        resolution.plan.Fingerprint(),
        static_cast<std::uint32_t>(resolution.rejected));
  }
}

void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits,
                       models::qwen::vision::DeviceInput* image_input) {
  const auto* rope = image_input != nullptr ? image_input->rope() : nullptr;
  (void)token_id;
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  auto scratch = arena.GetScratchView();
  auto& decode_scratch = scratch.decode;
  auto& attention_scratch = scratch.attention;
  auto& ssm_scratch = scratch.ssm;
  auto& ffn_scratch = scratch.ffn;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = decode_scratch.prompt_tokens.data();
  const std::uint32_t* d_in_pos = decode_scratch.prompt_tokens.data() + 1;
  auto* d_out_token = decode_scratch.sampled_token.data();

  // 1. Embedding lookup
  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        d_in_token, decode_scratch.hidden.data(), hidden_size,
                        arena.stream);

  if (image_input != nullptr)
    image_input->Inject(decode_scratch.hidden.data(), pos, 1, hidden_size, 1,
                        arena.stream);

  // 2. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const auto route_plan = ResolveQwenLayerRoute(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    const gufo::models::qwen::HipModuleContext module_ctx(
        static_cast<void*>(arena.stream), l, pos);

    {
      // Pre-RMSNorm, routed through the norm module (HIP backend). Same
      // kernel, same args, same arena slices (d_hidden in, d_normed out);
      // behavior identical to the former inline `LaunchRMSNorm` call.
      const auto norm_view =
          gufo::models::qwen::MakeAttnNormView(layer, config);
      gufo::models::qwen::NormForward(
          module_ctx, norm_view, decode_scratch.hidden, decode_scratch.normed);
    }

    if (layer.is_full_attention) {
      // Full attention path
      const std::size_t total_k = arena.GetAttentionKvPlaneElements();
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      LaunchFusedQKVProjections(
          layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
          layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          attention_scratch.k.data(), attention_scratch.v.data(),
          q_projection_size, kv_size, hidden_size, arena.stream);
      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(ssm_scratch.qkv.data(), attention_scratch.q.data(),
                     ssm_scratch.gate.data(), config.num_attention_heads,
                     config.head_dim, arena.stream);

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the
      // policy toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchFusedQKNormRoPEKvWrite(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data),
            attention_scratch.q.data(), attention_scratch.k.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            attn_layer_idx, d_in_pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, 1e-6F,
            arena.stream, rope);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              attention_scratch.q.data(), config.num_attention_heads,
              config.head_dim, 1e-6F, arena.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              attention_scratch.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              attention_scratch.k.data(), config.num_key_value_heads,
              config.head_dim, 1e-6F, arena.stream);
        }

        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(attention_scratch.q.data(), attention_scratch.k.data(),
                   config.num_attention_heads, config.num_key_value_heads,
                   config.head_dim, config.rotary_dim, d_in_pos,
                   config.rope_theta, arena.stream, rope);
      }

      // Softmax Attention + Gating
      if (use_split_k_decode) {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream, attention_scratch.split_k.data(),
            fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, OffsetIfPresent(arena.d_kv_cache, total_k),
            arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena.d_attention_kv_f16), total_k),
            ssm_scratch.out.data(), attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      gufo::models::qwen::QuantGemm(
          module_ctx, layer.attn_output, ssm_scratch.out.first(attention_size),
          hidden_size, attention_size, attention_scratch.output);
    } else {
      // SSM path

      LaunchFusedSSMInputProjections(
          layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
          layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
          layer.ssm_beta.data, layer.ssm_beta.type,
          decode_scratch.normed.data(), ssm_scratch.qkv.data(),
          ssm_scratch.gate.data(), ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), hidden_size, ssm_qkv_size, ssm_inner_size,
          time_step_rank, arena.stream);
      LaunchSSMConvRecurrence(
          ssm_scratch.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          arena.d_ssm_conv_state, ssm_scratch.conv_out.data(),
          arena.d_ssm_deltanet_state, ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          ssm_scratch.gate.data(), ssm_scratch.out.data(),
          config.SsmLayerIndex(l), ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena.stream, arena.GetSsmReplayCapture(),
          arena.GetRecurrentStateStorage());

      LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type, ssm_scratch.out.data(),
                 attention_scratch.output.data(), hidden_size, ssm_inner_size,
                 arena.stream);
    }

    {
      gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                      attention_scratch.output);

      {
        // FFN Pre-RMSNorm
        LaunchRMSNorm(decode_scratch.hidden.data(),
                      static_cast<const float*>(layer.ffn_norm.data),
                      decode_scratch.normed.data(), hidden_size, 1e-6F,
                      arena.stream);
      }
    }

    // Fused SwiGLU FFN (dense) or routed MoE FFN
    if (config.IsMoE()) {
      ExecuteMoeDecodeStep(arena.stream, scratch.moe, layer, config,
                           decode_scratch.normed.data(),
                           ffn_scratch.out.data());
    } else {
      // Non-fused FFN, routed through the module. Same fused SwiGLU kernel +
      // same down GEMV as the former inline calls; behavior identical. The
      // module reads the arena device spans (x = d_normed, act_scratch =
      // d_ffn_act, out = d_ffn_out) directly.
      const auto ffn_view = gufo::models::qwen::MakeFfnView(layer, config);
      gufo::models::qwen::FfnForward(
          module_ctx, ffn_view, decode_scratch.normed, ffn_scratch.activation,
          ffn_scratch.activation, ffn_scratch.activation, ffn_scratch.out);
    }

    // Residual Add
    gufo::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                    ffn_scratch.out);

    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      HIP_CHECK(hipMemcpyAsync(
          arena.d_target_layer_features + (*tap_index * hidden_size),
          decode_scratch.hidden.data(), hidden_size * sizeof(float),
          hipMemcpyDeviceToDevice, arena.stream));
    }
  }

  if (compute_logits) {
    // 3. Final Output Norm
    LaunchRMSNorm(decode_scratch.hidden.data(),
                  static_cast<const float*>(weights.output_norm.data),
                  decode_scratch.normed.data(), hidden_size, 1e-6F,
                  arena.stream);

    // 4. LM Head Logits GEMV on final token
    LaunchGEMV(weights.output.data, weights.output.type,
               decode_scratch.normed.data(), decode_scratch.logits.data(),
               vocab_size, hidden_size, arena.stream);

    // 5. Parallel GPU Argmax
    LaunchBatchedGPUArgmax(decode_scratch.logits.data(), d_out_token, 1,
                           vocab_size, ffn_scratch.out, arena.stream);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
