#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/decode_step.hpp"

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/modules.hpp"

namespace strix::hip {
namespace {

[[nodiscard]] constexpr bool SupportsDenseFusedProjection(
    core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

}  // namespace

void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits) {
  (void)token_id;
  const auto& config = weights.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  auto scratch = arena.GetScratchView();
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = scratch.prompt_tokens.data();
  const std::uint32_t* d_in_pos = scratch.prompt_tokens.data() + 1;
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena.d_alpha_buf);

  // 1. Embedding lookup
  LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                        d_in_token, arena.d_hidden, hidden_size, arena.stream);

  // opt-c014-layer-prefetch: touch the next layer's weight pages on a side
  // stream while the current layer computes, so the next layer's projection
  // kernels do not stall on first-touch page walks. The join below keeps the
  // side stream bounded to exactly one layer ahead. The unfused route (no
  // prefetch) stays wired behind the policy toggle as the reference.
  auto PrefetchLayerWeights = [](const models::QwenLayerWeights& layer,
                                 hipStream_t stream) {
    auto PrefetchTensor = [stream](const models::QwenTensorRef& tensor) {
      if (tensor.empty()) {
        return;
      }
      const std::size_t encoded_bytes = tensor.EncodedSizeBytes();
      if (encoded_bytes != 0) {
        LaunchLayerWeightPrefetch(tensor.data, encoded_bytes, stream);
      }
    };
    PrefetchTensor(layer.attn_norm);
    PrefetchTensor(layer.ffn_norm);
    PrefetchTensor(layer.attn_q);
    PrefetchTensor(layer.attn_k);
    PrefetchTensor(layer.attn_v);
    PrefetchTensor(layer.attn_output);
    PrefetchTensor(layer.attn_q_norm);
    PrefetchTensor(layer.attn_k_norm);
    PrefetchTensor(layer.attn_qkv);
    PrefetchTensor(layer.attn_gate);
    PrefetchTensor(layer.ssm_a);
    PrefetchTensor(layer.ssm_conv1d);
    PrefetchTensor(layer.ssm_dt);
    PrefetchTensor(layer.ssm_alpha);
    PrefetchTensor(layer.ssm_beta);
    PrefetchTensor(layer.ssm_norm);
    PrefetchTensor(layer.ssm_out);
    PrefetchTensor(layer.ffn_gate);
    PrefetchTensor(layer.ffn_up);
    PrefetchTensor(layer.ffn_down);
  };

  // 2. Layer stack
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights.layers[l];
    const auto route_plan = ResolveQwenLayerRoute(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    const strix::models::qwen::HipModuleContext module_ctx(
        static_cast<void*>(arena.stream), l, pos);
    const bool fuse_ffn_norm_swiglu =
        route_plan.fuse_ffn_swiglu &&
        layer.ffn_gate.type == core::GgmlType::kBF16 &&
        layer.ffn_up.type == core::GgmlType::kBF16;

    // opt-c014-layer-prefetch: start the next layer's page touch on the side
    // stream while this layer runs, then join before layer l+1 computes.
    // The side stream records the completion marker; the captured main
    // stream waits on it (the capture-compatible dependency direction), so
    // HIP-graph capture keeps working.
    if (route_plan.prefetch_next_layer && l > 0) {
      HIP_CHECK(hipStreamWaitEvent(arena.stream, arena.prefetch_event, 0));
    }
    if (route_plan.prefetch_next_layer && l + 1 < config.num_layers) {
      PrefetchLayerWeights(weights.layers[l + 1], arena.prefetch_stream);
      HIP_CHECK(hipEventRecord(arena.prefetch_event, arena.prefetch_stream));
    }

    // opt-c010-rmsnorm-projection: fuse the layer pre-RMSNorm into the
    // projection GEMVs below (QKV / SSM input / FFN SwiGLU), so the
    // projection kernel prepares its own normed input. The unfused chain
    // (RMSNormKernel + projection kernel) stays wired as the reference.
    const bool dense_input_projections =
        layer.is_full_attention
            ? SupportsDenseFusedProjection(layer.attn_q.type) &&
                  SupportsDenseFusedProjection(layer.attn_k.type) &&
                  SupportsDenseFusedProjection(layer.attn_v.type)
            : SupportsDenseFusedProjection(layer.attn_qkv.type) &&
                  SupportsDenseFusedProjection(layer.attn_gate.type) &&
                  SupportsDenseFusedProjection(layer.ssm_alpha.type) &&
                  SupportsDenseFusedProjection(layer.ssm_beta.type);
    const bool fuse_input_rmsnorm_projection =
        route_plan.fuse_rmsnorm_projection && dense_input_projections;
    if (!fuse_input_rmsnorm_projection) {
      // Pre-RMSNorm, routed through the norm module (HIP backend). Same
      // kernel, same args, same arena slices (d_hidden in, d_normed out);
      // behavior identical to the former inline `LaunchRMSNorm` call.
      const auto norm_view =
          strix::models::qwen::MakeAttnNormView(layer, config);
      strix::models::qwen::NormForward(module_ctx, norm_view, scratch.hidden,
                                       scratch.normed);
    }

    // opt-c010-ssm-gate-residual: the SSM branch folds the post-SSM residual
    // add into the ssm_out GEMV; the common residual-add step below then
    // runs only the FFN pre-norm for SSM layers.
    bool ssm_residual_folded = false;

    if (layer.is_full_attention) {
      // Full attention path
      const bool q_bf16 = layer.attn_q.type == core::GgmlType::kBF16;
      const bool k_bf16 = layer.attn_k.type == core::GgmlType::kBF16;
      const bool v_bf16 = layer.attn_v.type == core::GgmlType::kBF16;
      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena.GetMaxContext() * config.head_dim;
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      if (fuse_input_rmsnorm_projection) {
        LaunchFusedRMSNormQKVProjections(
            arena.d_hidden, static_cast<const float*>(layer.attn_norm.data),
            1e-6F, layer.attn_q.data, q_bf16, layer.attn_k.data, k_bf16,
            layer.attn_v.data, v_bf16, arena.d_ssm_qkv, arena.d_k,
            arena.d_v, q_projection_size, kv_size, hidden_size,
            arena.stream);
      } else {
        LaunchFusedQKVProjections(
            layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
            layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
            arena.d_normed, arena.d_ssm_qkv, arena.d_k, arena.d_v,
            q_projection_size, kv_size, hidden_size, arena.stream);
      }

      // De-interleave Q and Gate from attn_q projection
      LaunchUnpackQG(arena.d_ssm_qkv, arena.d_q, arena.d_ssm_gate,
                     config.num_attention_heads, config.head_dim,
                     arena.stream);

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the
      // policy toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchFusedQKNormRoPEKvWrite(
            arena.d_q, arena.d_k, arena.d_v,
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), arena.d_q,
            arena.d_k, arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            attn_layer_idx, d_in_pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, 1e-6F,
            arena.stream);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchPerHeadRMSNorm(
              arena.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena.d_q, config.num_attention_heads, config.head_dim, 1e-6F,
              arena.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchPerHeadRMSNorm(
              arena.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena.d_k, config.num_key_value_heads, config.head_dim, 1e-6F,
              arena.stream);
        }

        // RoPE (using device pos pointer for graph capture invariance)
        LaunchRoPE(arena.d_q, arena.d_k, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   config.rotary_dim, d_in_pos, config.rope_theta,
                   arena.stream);
      }

      // Softmax Attention + Gating
      if (use_split_k_decode) {
        LaunchAttention(
            arena.d_q, arena.d_k, arena.d_v, arena.d_ssm_gate,
            arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            arena.d_ssm_out, attn_layer_idx, pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream, scratch.split_k_attention.data(),
            fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            arena.d_q, arena.d_k, arena.d_v, arena.d_ssm_gate,
            arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            arena.d_ssm_out, attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      strix::models::qwen::QuantGemm(
          module_ctx, layer.attn_output,
          scratch.ssm_out.first(attention_size), hidden_size, attention_size,
          scratch.attention_out);
    } else {
      // SSM path
      ssm_residual_folded =
          route_plan.fuse_ssm_epilogue &&
          SupportsDenseFusedProjection(layer.ssm_out.type);
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;

      if (fuse_input_rmsnorm_projection) {
        LaunchFusedRMSNormSSMInputProjections(
            arena.d_hidden, static_cast<const float*>(layer.attn_norm.data),
            1e-6F, layer.attn_qkv.data, qkv_bf16, layer.attn_gate.data,
            gate_bf16, layer.ssm_alpha.data, alpha_bf16, layer.ssm_beta.data,
            beta_bf16, arena.d_ssm_qkv, arena.d_ssm_gate,
            arena.d_alpha_buf, arena.d_beta_buf, hidden_size, ssm_qkv_size,
            ssm_inner_size, time_step_rank, arena.stream);
      } else {
        LaunchFusedSSMInputProjections(
            layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
            layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
            layer.ssm_beta.data, layer.ssm_beta.type, arena.d_normed,
            arena.d_ssm_qkv, arena.d_ssm_gate, arena.d_alpha_buf,
            arena.d_beta_buf, hidden_size, ssm_qkv_size, ssm_inner_size,
            time_step_rank, arena.stream);
      }

      LaunchSSMConvRecurrence(
          arena.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
          arena.d_ssm_conv_state, arena.d_conv_out,
          arena.d_ssm_deltanet_state, arena.d_alpha_buf, arena.d_beta_buf,
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data), arena.d_ssm_gate,
          arena.d_ssm_out, l, ssm_qkv_size, config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), arena.stream, arena.GetSsmReplayCapture());

      // opt-c010-ssm-gate-residual: fold the post-SSM residual add into the
      // ssm_out GEMV epilogue (y = A*x + hidden). The unfused chain (GEMV
      // into d_attn_out + residual add) stays wired as the reference.
      if (ssm_residual_folded) {
        LaunchGEMVResidual(layer.ssm_out.data, layer.ssm_out.type,
                           arena.d_ssm_out, arena.d_hidden,
                           arena.d_hidden, hidden_size, ssm_inner_size,
                           arena.stream);
      } else {
        LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type, arena.d_ssm_out,
                   arena.d_attn_out, hidden_size, ssm_inner_size,
                   arena.stream);
      }
    }

    // Residual Add + FFN Pre-RMSNorm fused into one kernel
    // (opt-c010-residual-rmsnorm). The unfused chain stays wired behind the
    // policy toggle as the independent reference.
    // When the SSM residual is folded into the ssm_out GEMV above, the
    // residual-add step is already applied, so only the FFN pre-norm runs.
    if (ssm_residual_folded) {
      if (!fuse_ffn_norm_swiglu) {
        LaunchRMSNorm(arena.d_hidden,
                      static_cast<const float*>(layer.ffn_norm.data),
                      arena.d_normed, hidden_size, 1e-6F, arena.stream);
      }
    } else if (route_plan.fuse_residual_rmsnorm) {
      LaunchFusedResidualAddRMSNorm(
          arena.d_hidden, arena.d_attn_out, arena.d_hidden,
          static_cast<const float*>(layer.ffn_norm.data), arena.d_normed,
          hidden_size, 1e-6F, arena.stream);
    } else {
      strix::models::qwen::ResidualAdd(module_ctx, scratch.hidden,
                                       scratch.attention_out);

      if (!fuse_ffn_norm_swiglu) {
        // FFN Pre-RMSNorm
        LaunchRMSNorm(arena.d_hidden,
                      static_cast<const float*>(layer.ffn_norm.data),
                      arena.d_normed, hidden_size, 1e-6F, arena.stream);
      }
    }

    // Fused SwiGLU FFN
    if (fuse_ffn_norm_swiglu) {
      // Cross-module fusion: RMSNorm folded into the SwiGLU GEMV. Owned by
      // the composition layer (Option B); the standalone FFN below is the
      // module. Keep the fused launch + shared down GEMV inline.
      LaunchFusedRMSNormSwiGLUGEMV(
          arena.d_hidden, static_cast<const float*>(layer.ffn_norm.data),
          1e-6F, layer.ffn_gate.data, layer.ffn_up.data, arena.d_ffn_act,
          intermediate_size, hidden_size, arena.stream);
      LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type, arena.d_ffn_act,
                 arena.d_ffn_out, hidden_size, intermediate_size,
                 arena.stream);
    } else {
      // Non-fused FFN, routed through the module. Same fused SwiGLU kernel +
      // same down GEMV as the former inline calls; behavior identical. The
      // module reads the arena device spans (x = d_normed, act_scratch =
      // d_ffn_act, out = d_ffn_out) directly.
      const auto ffn_view = strix::models::qwen::MakeFfnView(layer, config);
      strix::models::qwen::FfnForward(
          module_ctx, ffn_view, scratch.normed, scratch.ffn_activation,
          scratch.ffn_activation, scratch.ffn_activation, scratch.ffn_out);
    }

    // Residual Add
    strix::models::qwen::ResidualAdd(module_ctx, scratch.hidden,
                                     scratch.ffn_out);
  }

  if (compute_logits) {
    // 3. Final Output Norm
    LaunchRMSNorm(arena.d_hidden,
                  static_cast<const float*>(weights.output_norm.data),
                  arena.d_normed, hidden_size, 1e-6F, arena.stream);

    // 4. LM Head Logits GEMV on final token
    LaunchGEMV(weights.output.data, weights.output.type, arena.d_normed,
               scratch.logits.data(), vocab_size, hidden_size, arena.stream);

    // 5. Parallel GPU Argmax
    LaunchGPUArgmax(scratch.logits.data(), d_out_token, vocab_size,
                    arena.stream);
  }
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
