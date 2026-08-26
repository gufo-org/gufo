#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/detail/decode_step.hpp"

#include "src/core/hip/detail/dispatch_telemetry.hpp"
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
      strix::models::qwen::NormForward(
          module_ctx, norm_view, decode_scratch.hidden, decode_scratch.normed);
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
            decode_scratch.hidden.data(),
            static_cast<const float*>(layer.attn_norm.data), 1e-6F,
            layer.attn_q.data, q_bf16, layer.attn_k.data, k_bf16,
            layer.attn_v.data, v_bf16, ssm_scratch.qkv.data(),
            attention_scratch.k.data(), attention_scratch.v.data(),
            q_projection_size, kv_size, hidden_size, arena.stream);
      } else {
        LaunchFusedQKVProjections(
            layer.attn_q.data, layer.attn_q.type, layer.attn_k.data,
            layer.attn_k.type, layer.attn_v.data, layer.attn_v.type,
            decode_scratch.normed.data(), ssm_scratch.qkv.data(),
            attention_scratch.k.data(), attention_scratch.v.data(),
            q_projection_size, kv_size, hidden_size, arena.stream);
      }

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
            arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            attn_layer_idx, d_in_pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, 1e-6F,
            arena.stream);
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
                   config.rope_theta, arena.stream);
      }

      // Softmax Attention + Gating
      if (use_split_k_decode) {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            ssm_scratch.out.data(), attn_layer_idx, pos, arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream, attention_scratch.split_k.data(),
            fused_qknorm_rope_kv);
      } else {
        LaunchAttention(
            attention_scratch.q.data(), attention_scratch.k.data(),
            attention_scratch.v.data(), ssm_scratch.gate.data(),
            arena.d_kv_cache, arena.d_kv_cache + total_k,
            arena.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena.d_attention_kv_f16) + total_k,
            ssm_scratch.out.data(), attn_layer_idx, d_in_pos,
            arena.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena.stream,
            fused_qknorm_rope_kv);
      }

      // Output projection, routed through the quant_gemm module (HIP
      // backend). Same kernel, same args, same arena slices (d_ssm_out in,
      // d_attn_out out); behavior-identical to the former inline `LaunchGEMV`.
      strix::models::qwen::QuantGemm(
          module_ctx, layer.attn_output, ssm_scratch.out.first(attention_size),
          hidden_size, attention_size, attention_scratch.output);
    } else {
      // SSM path
      ssm_residual_folded = route_plan.fuse_ssm_epilogue &&
                            SupportsDenseFusedProjection(layer.ssm_out.type);
      const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
      const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
      const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
      const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;

      if (fuse_input_rmsnorm_projection) {
        LaunchFusedRMSNormSSMInputProjections(
            decode_scratch.hidden.data(),
            static_cast<const float*>(layer.attn_norm.data), 1e-6F,
            layer.attn_qkv.data, qkv_bf16, layer.attn_gate.data, gate_bf16,
            layer.ssm_alpha.data, alpha_bf16, layer.ssm_beta.data, beta_bf16,
            ssm_scratch.qkv.data(), ssm_scratch.gate.data(),
            ssm_scratch.alpha.data(), ssm_scratch.beta.data(), hidden_size,
            ssm_qkv_size, ssm_inner_size, time_step_rank, arena.stream);
      } else {
        LaunchFusedSSMInputProjections(
            layer.attn_qkv.data, layer.attn_qkv.type, layer.attn_gate.data,
            layer.attn_gate.type, layer.ssm_alpha.data, layer.ssm_alpha.type,
            layer.ssm_beta.data, layer.ssm_beta.type,
            decode_scratch.normed.data(), ssm_scratch.qkv.data(),
            ssm_scratch.gate.data(), ssm_scratch.alpha.data(),
            ssm_scratch.beta.data(), hidden_size, ssm_qkv_size, ssm_inner_size,
            time_step_rank, arena.stream);
      }

      LaunchSSMConvRecurrence(
          ssm_scratch.qkv.data(),
          static_cast<const float*>(layer.ssm_conv1d.data),
          arena.d_ssm_conv_state, ssm_scratch.conv_out.data(),
          arena.d_ssm_deltanet_state, ssm_scratch.alpha.data(),
          ssm_scratch.beta.data(), static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data),
          static_cast<const float*>(layer.ssm_norm.data),
          ssm_scratch.gate.data(), ssm_scratch.out.data(), l, ssm_qkv_size,
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), arena.stream,
          arena.GetSsmReplayCapture());

      // opt-c010-ssm-gate-residual: fold the post-SSM residual add into the
      // ssm_out GEMV epilogue (y = A*x + hidden). The unfused chain (GEMV
      // into d_attn_out + residual add) stays wired as the reference.
      if (ssm_residual_folded) {
        LaunchGEMVResidual(layer.ssm_out.data, layer.ssm_out.type,
                           ssm_scratch.out.data(), decode_scratch.hidden.data(),
                           decode_scratch.hidden.data(), hidden_size,
                           ssm_inner_size, arena.stream);
      } else {
        LaunchGEMV(layer.ssm_out.data, layer.ssm_out.type,
                   ssm_scratch.out.data(), attention_scratch.output.data(),
                   hidden_size, ssm_inner_size, arena.stream);
      }
    }

    // Residual Add + FFN Pre-RMSNorm fused into one kernel
    // (opt-c010-residual-rmsnorm). The unfused chain stays wired behind the
    // policy toggle as the independent reference.
    // When the SSM residual is folded into the ssm_out GEMV above, the
    // residual-add step is already applied, so only the FFN pre-norm runs.
    if (ssm_residual_folded) {
      if (!fuse_ffn_norm_swiglu) {
        LaunchRMSNorm(decode_scratch.hidden.data(),
                      static_cast<const float*>(layer.ffn_norm.data),
                      decode_scratch.normed.data(), hidden_size, 1e-6F,
                      arena.stream);
      }
    } else if (route_plan.fuse_residual_rmsnorm) {
      LaunchFusedResidualAddRMSNorm(
          decode_scratch.hidden.data(), attention_scratch.output.data(),
          decode_scratch.hidden.data(),
          static_cast<const float*>(layer.ffn_norm.data),
          decode_scratch.normed.data(), hidden_size, 1e-6F, arena.stream);
    } else {
      strix::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                       attention_scratch.output);

      if (!fuse_ffn_norm_swiglu) {
        // FFN Pre-RMSNorm
        LaunchRMSNorm(decode_scratch.hidden.data(),
                      static_cast<const float*>(layer.ffn_norm.data),
                      decode_scratch.normed.data(), hidden_size, 1e-6F,
                      arena.stream);
      }
    }

    // Fused SwiGLU FFN
    if (fuse_ffn_norm_swiglu) {
      // Cross-module fusion: RMSNorm folded into the SwiGLU GEMV. Owned by
      // the composition layer (Option B); the standalone FFN below is the
      // module. Keep the fused launch + shared down GEMV inline.
      LaunchFusedRMSNormSwiGLUGEMV(
          decode_scratch.hidden.data(),
          static_cast<const float*>(layer.ffn_norm.data), 1e-6F,
          layer.ffn_gate.data, layer.ffn_up.data, ffn_scratch.activation.data(),
          intermediate_size, hidden_size, arena.stream);
      LaunchGEMV(layer.ffn_down.data, layer.ffn_down.type,
                 ffn_scratch.activation.data(), ffn_scratch.out.data(),
                 hidden_size, intermediate_size, arena.stream);
    } else {
      // Non-fused FFN, routed through the module. Same fused SwiGLU kernel +
      // same down GEMV as the former inline calls; behavior identical. The
      // module reads the arena device spans (x = d_normed, act_scratch =
      // d_ffn_act, out = d_ffn_out) directly.
      const auto ffn_view = strix::models::qwen::MakeFfnView(layer, config);
      strix::models::qwen::FfnForward(
          module_ctx, ffn_view, decode_scratch.normed, ffn_scratch.activation,
          ffn_scratch.activation, ffn_scratch.activation, ffn_scratch.out);
    }

    // Residual Add
    strix::models::qwen::ResidualAdd(module_ctx, decode_scratch.hidden,
                                     ffn_scratch.out);

    if (const auto tap_index = arena.GetTargetLayerCaptureIndex(l);
        tap_index.has_value()) {
      (void)hipMemcpyAsync(
          arena.d_target_layer_features + (*tap_index * hidden_size),
          decode_scratch.hidden.data(), hidden_size * sizeof(float),
          hipMemcpyDeviceToDevice, arena.stream);
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
    LaunchGPUArgmax(decode_scratch.logits.data(), d_out_token, vocab_size,
                    arena.stream);
  }
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
