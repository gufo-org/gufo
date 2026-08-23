#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/models/qwen/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_executor.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen/modules/modules.hpp"

namespace strix::hip {
// Per-token decode composition step. Extracted from ForwardToken's ExecuteStep
// lambda (#42 PART 1): owns the layer loop + cross-module fusions behind the
// detail:: policy toggles, keeping the module calls pure. Graph-capture-safe:
// no heap allocation or throwing in the body; all device buffers come from
// `arena` and every kernel launch uses `arena.stream`.
static void ExecuteDecodeStep(QwenGpuArena& arena,
                              const models::QwenModelWeights& weights,
                              tokenization::TokenId token_id,
                              std::uint32_t pos, bool compute_logits) {
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
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  // GPU parameter buffers
  const std::uint32_t* d_in_token = arena.d_prompt_tokens + 0;
  const std::uint32_t* d_in_pos = arena.d_prompt_tokens + 1;
  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena.d_alpha_buf);

    // 1. Embedding lookup
    LaunchEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                          d_in_token, arena.d_hidden, hidden_size,
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
        const std::size_t element_bytes =
            tensor.type == core::GgmlType::kBF16 ? 2U : 4U;
        LaunchLayerWeightPrefetch(tensor.data,
                                  tensor.num_elements * element_bytes, stream);
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

      // opt-c014-layer-prefetch: start the next layer's page touch on the side
      // stream while this layer runs, then join before layer l+1 computes.
      // The side stream records the completion marker; the captured main
      // stream waits on it (the capture-compatible dependency direction), so
      // HIP-graph capture keeps working.
      if (detail::ShouldPrefetchNextLayer() && l + 1 < config.num_layers) {
        if (l > 0) {
          HIP_CHECK(
              hipStreamWaitEvent(arena.stream, arena.prefetch_event, 0));
        }
        PrefetchLayerWeights(weights.layers[l + 1], arena.prefetch_stream);
        HIP_CHECK(
            hipEventRecord(arena.prefetch_event, arena.prefetch_stream));
      }

      // opt-c010-rmsnorm-projection: fuse the layer pre-RMSNorm into the
      // projection GEMVs below (QKV / SSM input / FFN SwiGLU), so the
      // projection kernel prepares its own normed input. The unfused chain
      // (RMSNormKernel + projection kernel) stays wired as the reference.
      const bool fused_rmsnorm_proj = detail::ShouldFuseRMSNormProjection();
      if (!fused_rmsnorm_proj) {
        // Pre-RMSNorm, routed through the norm module (HIP backend). Same
        // kernel, same args, same arena slices (d_hidden in, d_normed out);
        // behavior identical to the former inline `LaunchRMSNorm` call.
        strix::models::qwen::ModuleCtx norm_ctx;
        norm_ctx.backend = strix::models::qwen::Backend::Hip;
        norm_ctx.config = &config;
        norm_ctx.stream = static_cast<void*>(arena.stream);
        norm_ctx.layer_idx = l;
        // arena_ is a QwenGpuArena (HIP device buffers), while ModuleCtx::arena
        // is the CPU QwenScratchArena. The norm module reads the device spans
        // passed in directly, so ctx.arena stays null here.
        norm_ctx.arena = nullptr;
        const auto norm_view =
            strix::models::qwen::MakeAttnNormView(layer, config);
        strix::models::qwen::NormForward(
            norm_ctx, norm_view,
            std::span<const float>(arena.d_hidden, hidden_size),
            std::span<float>(arena.d_normed, hidden_size));
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

        if (fused_rmsnorm_proj) {
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
        const bool fused_qknorm_rope_kv = detail::ShouldFuseQKNormRoPEKvWrite();
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
              config.head_dim, arena.stream,
              static_cast<float*>(arena.d_scratch_bf16), fused_qknorm_rope_kv);
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
        strix::models::qwen::ModuleCtx qg_ctx;
        qg_ctx.backend = strix::models::qwen::Backend::Hip;
        qg_ctx.config = &config;
        qg_ctx.stream = static_cast<void*>(arena.stream);
        qg_ctx.layer_idx = l;
        // ModuleCtx::arena is the CPU QwenScratchArena; the module reads the
        // device spans passed in directly, so ctx.arena stays null here.
        qg_ctx.arena = nullptr;
        strix::models::qwen::QuantGemm(
            qg_ctx, layer.attn_output,
            std::span<const float>(arena.d_ssm_out, attention_size),
            hidden_size, attention_size,
            std::span<float>(arena.d_attn_out, hidden_size));
      } else {
        // SSM path
        ssm_residual_folded = detail::ShouldFuseSSMGateResidual();
        const bool qkv_bf16 = layer.attn_qkv.type == core::GgmlType::kBF16;
        const bool gate_bf16 = layer.attn_gate.type == core::GgmlType::kBF16;
        const bool alpha_bf16 = layer.ssm_alpha.type == core::GgmlType::kBF16;
        const bool beta_bf16 = layer.ssm_beta.type == core::GgmlType::kBF16;

        if (fused_rmsnorm_proj) {
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
        LaunchRMSNorm(arena.d_hidden,
                      static_cast<const float*>(layer.ffn_norm.data),
                      arena.d_normed, hidden_size, 1e-6F, arena.stream);
      } else if (detail::ShouldFuseResidualAddRMSNorm()) {
        LaunchFusedResidualAddRMSNorm(
            arena.d_hidden, arena.d_attn_out, arena.d_hidden,
            static_cast<const float*>(layer.ffn_norm.data), arena.d_normed,
            hidden_size, 1e-6F, arena.stream);
      } else {
        strix::models::qwen::ModuleCtx res_ctx;
        res_ctx.backend = strix::models::qwen::Backend::Hip;
        res_ctx.config = &config;
        res_ctx.stream = static_cast<void*>(arena.stream);
        res_ctx.layer_idx = l;
        // arena_ is a QwenGpuArena (HIP device buffers), while ModuleCtx::arena
        // is the CPU QwenScratchArena. The residual module reads the device
        // spans passed in directly, so ctx.arena stays null here.
        res_ctx.arena = nullptr;
        strix::models::qwen::ResidualAdd(
            res_ctx, std::span<float>(arena.d_hidden, hidden_size),
            std::span<const float>(arena.d_attn_out, hidden_size));

        if (fused_rmsnorm_proj) {
          // FFN Pre-RMSNorm is folded into the SwiGLU GEMV below.
        } else {
          // FFN Pre-RMSNorm
          LaunchRMSNorm(arena.d_hidden,
                        static_cast<const float*>(layer.ffn_norm.data),
                        arena.d_normed, hidden_size, 1e-6F, arena.stream);
        }
      }

      // Fused SwiGLU FFN
      const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
      const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;

      if (fused_rmsnorm_proj && ffn_g_bf16 && ffn_u_bf16) {
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
        strix::models::qwen::ModuleCtx ffn_ctx;
        ffn_ctx.backend = strix::models::qwen::Backend::Hip;
        ffn_ctx.config = &config;
        ffn_ctx.stream = static_cast<void*>(arena.stream);
        ffn_ctx.layer_idx = l;
        // arena_ is a QwenGpuArena (HIP device buffers), while ModuleCtx::arena
        // is the CPU QwenScratchArena. The ffn module reads the device spans
        // passed in directly, so ctx.arena stays null here.
        ffn_ctx.arena = nullptr;
        const auto ffn_view = strix::models::qwen::MakeFfnView(layer, config);
        strix::models::qwen::FfnForward(
            ffn_ctx, ffn_view,
            std::span<const float>(arena.d_normed, hidden_size),
            std::span<float>(arena.d_ffn_act, intermediate_size),
            std::span<float>(arena.d_ffn_act, intermediate_size),
            std::span<float>(arena.d_ffn_act, intermediate_size),
            std::span<float>(arena.d_ffn_out, hidden_size));
      }

      // Residual Add
      strix::models::qwen::ModuleCtx res_ctx;
      res_ctx.backend = strix::models::qwen::Backend::Hip;
      res_ctx.config = &config;
      res_ctx.stream = static_cast<void*>(arena.stream);
      res_ctx.layer_idx = l;
      // arena_ is a QwenGpuArena (HIP device buffers), while ModuleCtx::arena
      // is the CPU QwenScratchArena. The residual module reads the device spans
      // passed in directly, so ctx.arena stays null here.
      res_ctx.arena = nullptr;
      strix::models::qwen::ResidualAdd(
          res_ctx, std::span<float>(arena.d_hidden, hidden_size),
          std::span<const float>(arena.d_ffn_out, hidden_size));
    }

    if (compute_logits) {
      // 3. Final Output Norm
      LaunchRMSNorm(arena.d_hidden,
                    static_cast<const float*>(weights.output_norm.data),
                    arena.d_normed, hidden_size, 1e-6F, arena.stream);

      // 4. LM Head Logits GEMV on final token
      LaunchGEMV(weights.output.data, weights.output.type, arena.d_normed,
                 arena.d_logits, vocab_size, hidden_size, arena.stream);

      // 5. Parallel GPU Argmax
      LaunchGPUArgmax(arena.d_logits, d_out_token, vocab_size, arena.stream);
    }
}
tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  if (pos >= arena_.GetMaxContext()) {
    throw std::length_error("token position exceeds the GPU context length");
  }

  if (replaying_ssm_state_ && !compute_logits) {
    if (arena_.CanReplaySsmPosition(pos)) {
      ReplaySsmState(pos);
      return 0;
    }
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  } else if (replaying_ssm_state_) {
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  }

  arena_.MarkSsmReplayPosition(pos);
  last_hidden_offset_ = 0;
  const auto& config = weights_.config;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  auto* d_out_token = reinterpret_cast<std::uint32_t*>(arena_.d_alpha_buf);


  // Copy token_id and pos to GPU device memory
  const std::uint32_t in_params[2] = {token_id, pos};
  HIP_CHECK(hipMemcpyAsync(arena_.d_prompt_tokens, in_params, sizeof(in_params),
                           hipMemcpyHostToDevice, arena_.stream));


  if (compute_logits && !use_split_k_decode && graph_executor_.IsEnabled()) {
    if (graph_executor_.IsCaptured()) {
      graph_executor_.Launch(arena_.stream);
    } else {
      const bool ok = graph_executor_.TryCapture(
          arena_.stream,
          [&]() { ExecuteDecodeStep(arena_, weights_, token_id, pos, compute_logits); });
      if (ok) {
        graph_executor_.Launch(arena_.stream);
      } else {
        ExecuteDecodeStep(arena_, weights_, token_id, pos, compute_logits);
      }
    }
  } else {
    ExecuteDecodeStep(arena_, weights_, token_id, pos, compute_logits);
  }


  if (!compute_logits) {
    return 0;
  }

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}



}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
