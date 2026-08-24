#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {
tokenization::TokenId QwenGpuExecutor::ForwardPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t batch_size = prompt_tokens.size();
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const float eps = 1e-6F;

  if (batch_size == 0) {
    return 0;
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error("prompt chunk exceeds the GPU batch length");
  }
  auto scratch = arena_.GetScratchView(batch_size);

  // 1. Copy prompt token IDs to GPU
  std::vector<std::uint32_t> host_tokens(prompt_tokens.begin(),
                                         prompt_tokens.end());
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  // 2. Batched Embedding lookup: d_hidden [B, hidden_size]
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);

  const bool do_profile = (std::getenv("STRIX_PROFILE") != nullptr);
  auto t_start = std::chrono::high_resolution_clock::now();
  double time_attn_proj = 0, time_ssm_recur = 0, time_ssm_out = 0;
  double time_ffn = 0, time_norm = 0;
  double time_dequant = 0, time_gemm = 0;

  constexpr std::size_t hipblaslt_min_dimension = 1024;
  const auto launch_bf16_gemm = [&](const void* weights, const void* input,
                                    float* output, std::size_t m,
                                    std::size_t k) {
    const bool use_hipblaslt =
        m >= hipblaslt_min_dimension && k >= hipblaslt_min_dimension &&
        arena_.hipblaslt_gemm->RunBf16(weights, input, output, batch_size, m, k,
                                       arena_.stream);
    if (!use_hipblaslt) {
      LaunchHipblasGEMMBF16(arena_.hipblas_handle, weights, input, output,
                            batch_size, m, k, arena_.stream);
    }
  };

  // Centralized weight dispatch: BF16 -> direct BF16 GEMM on the BF16
  // activations; F32 -> hipblas fp32 GEMM on the fp32 input; any quant
  // (Q8_0/Q5_K/Q6_K/Q8_K) -> batched quant GEMM directly on the quantized
  // weights (no dequantize-to-BF16, no BF16 GEMM).
  const auto gemm_weight = [&](const models::QwenTensorRef& w,
                               const void* bf16_input,
                               const float* fp32_input, float* output,
                               std::size_t m, std::size_t k) {
    if (w.type == core::GgmlType::kBF16) {
      launch_bf16_gemm(w.data, bf16_input, output, m, k);
    } else if (w.type == core::GgmlType::kF32) {
      LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                        output, batch_size, m, k, arena_.d_scratch_bf16,
                        arena_.stream);
    } else {
      auto tg0 = std::chrono::high_resolution_clock::now();
      LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch_size, m,
                             k, arena_.stream);
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        time_gemm += std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - tg0)
                         .count();
      }
    }
  };

  // 3. Layer stack across all 32 layers
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const auto& layer = weights_.layers[l];
    const auto route_resolution = ResolveQwenLayerRouteWithReasons(
        policy_, QwenExecutionMode::kPrefill, layer.is_full_attention);
    const auto& route_plan = route_resolution.plan;
    detail::EmitQwenRouteResolution(
        "prefill", l, layer.is_full_attention ? "attention" : "ssm",
        route_plan.Fingerprint(),
        static_cast<std::uint32_t>(route_resolution.rejected));

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    LaunchBatchedRMSNorm(arena_.d_hidden,
                         static_cast<const float*>(layer.attn_norm.data),
                         arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                         hidden_size, eps, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_norm += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }

    if (layer.is_full_attention) {
      gemm_weight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, q_projection_size, hidden_size);
      gemm_weight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_k, kv_size, hidden_size);
      gemm_weight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_v, kv_size, hidden_size);

      LaunchBatchedUnpackQG(arena_.d_ssm_qkv, arena_.d_q, arena_.d_ssm_gate,
                            batch_size, config.num_attention_heads,
                            config.head_dim, arena_.stream);

      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena_.GetMaxContext() * config.head_dim;
      const std::uint32_t attn_layer_idx = l / config.full_attention_interval;

      // QK-Norm + RoPE + KV-cache write fused into one kernel
      // (opt-c010-qk-rope-kv). The unfused chain stays wired behind the policy
      // toggle as the independent reference.
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (fused_qknorm_rope_kv) {
        LaunchBatchedFusedQKNormRoPEKvWrite(
            arena_.d_q, arena_.d_k, arena_.d_v,
            static_cast<const float*>(layer.attn_q_norm.data),
            static_cast<const float*>(layer.attn_k_norm.data), arena_.d_q,
            arena_.d_k, arena_.d_kv_cache, arena_.d_kv_cache + total_k,
            arena_.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
            attn_layer_idx, start_pos, batch_size, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.rotary_dim, config.rope_theta, eps,
            arena_.stream);
      } else {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_q, static_cast<const float*>(layer.attn_q_norm.data),
              arena_.d_q, batch_size, config.num_attention_heads,
              config.head_dim, eps, arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              arena_.d_k, static_cast<const float*>(layer.attn_k_norm.data),
              arena_.d_k, batch_size, config.num_key_value_heads,
              config.head_dim, eps, arena_.stream);
        }

        LaunchBatchedRoPE(
            arena_.d_q, arena_.d_k, batch_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rotary_dim,
            start_pos, config.rope_theta, arena_.stream);
      }

      const std::size_t visible_context =
          static_cast<std::size_t>(start_pos) + batch_size;
      enum class SelectedAttention {
        kBaseline,
        kTiled,
        kComposableKernel,
      };
      SelectedAttention selected_attention = SelectedAttention::kBaseline;
      bool tiled_rejected = false;
      bool ck_rejected = false;
      detail::DispatchPrefillAttention(
          visible_context,
          [&] {
            const bool launched = LaunchBatchedAttentionTile(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_attention_kv_f16,
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
                    total_k,
                arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream);
            selected_attention = launched ? SelectedAttention::kTiled
                                          : SelectedAttention::kBaseline;
            tiled_rejected = !launched;
            return launched;
          },
          [&] {
            const bool launched = LaunchBatchedAttentionCk(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_attention_kv_f16,
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
                    total_k,
                arena_.d_scratch_bf16, arena_.d_ssm_out, attn_layer_idx,
                start_pos, batch_size, arena_.GetMaxContext(),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream);
            selected_attention = launched ? SelectedAttention::kComposableKernel
                                          : SelectedAttention::kBaseline;
            ck_rejected = !launched;
            return launched;
          },
          [&] {
            selected_attention = SelectedAttention::kBaseline;
            LaunchBatchedAttention(
                arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
                arena_.d_kv_cache, arena_.d_kv_cache + total_k,
                arena_.d_attention_kv_f16,
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
                    total_k,
                arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
                arena_.GetMaxContext(), config.num_attention_heads,
                config.num_key_value_heads, config.head_dim, arena_.stream,
                fused_qknorm_rope_kv);
          });
      if (selected_attention == SelectedAttention::kTiled) {
        detail::EmitAttentionDispatch("prefill_tiled", "");
      } else if (selected_attention == SelectedAttention::kComposableKernel) {
        detail::EmitAttentionDispatch(
            "prefill_composable_kernel",
            tiled_rejected ? "prefill_tiled: rejected" : "");
      } else if (!detail::ShouldAttemptOptimizedAttention(visible_context)) {
        detail::EmitAttentionDispatch(
            "prefill_baseline",
            "prefill_tiled: below_threshold; prefill_composable_kernel: "
            "below_threshold");
      } else {
        detail::EmitAttentionDispatch(
            "prefill_baseline",
            tiled_rejected && ck_rejected
                ? "prefill_tiled: rejected; prefill_composable_kernel: "
                  "rejected"
                : "optimized_attention: rejected");
      }

      LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                            batch_size * attention_size, arena_.stream);
      gemm_weight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                  arena_.d_attn_out, hidden_size, attention_size);
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    } else {
      gemm_weight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, ssm_qkv_size, hidden_size);
      gemm_weight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_gate, ssm_inner_size, hidden_size);
      gemm_weight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_alpha_buf, time_step_rank, hidden_size);
      gemm_weight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_beta_buf, time_step_rank, hidden_size);
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      // opt-c010-ssm-gate-residual: fuse the per-head post-RMSNorm + SiLU
      // gate into the DeltaNet recurrence epilogue (one launch, no raw_out
      // global round trip). The unfused chain (recurrence +
      // BatchedSSMPostNormGateKernel) stays wired as the reference.
      if (route_plan.fuse_ssm_epilogue) {
        LaunchBatchedSSMConvRecurrenceNormGate(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, l, batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream);
      } else {
        LaunchBatchedSSMConvRecurrence(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out, l, batch_size, ssm_qkv_size,
            config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream);
      }

      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_recur +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }

      LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                            batch_size * ssm_inner_size, arena_.stream);
      gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                  arena_.d_attn_out, hidden_size, ssm_inner_size);
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_ssm_out +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    }

    // Residual Add + FFN RMSNorm fused into one kernel
    // (opt-c010-residual-rmsnorm). The unfused chain stays wired behind the
    // policy toggle as the independent reference.
    if (route_plan.fuse_residual_rmsnorm) {
      LaunchBatchedFusedResidualAddRMSNorm(
          arena_.d_hidden, arena_.d_attn_out, arena_.d_hidden,
          static_cast<const float*>(layer.ffn_norm.data), arena_.d_normed,
          arena_.d_scratch_bf16, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_attn_out,
                               arena_.d_hidden, batch_size, hidden_size,
                               arena_.stream);

      // FFN RMSNorm (generates BF16 into d_scratch_bf16 directly)
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.ffn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    const bool ffn_g_bf16 = layer.ffn_gate.type == core::GgmlType::kBF16;
    const bool ffn_u_bf16 = layer.ffn_up.type == core::GgmlType::kBF16;

    // Fused FFN gate/up projection with SwiGLU activation into one kernel
    // (opt-c010-ffn-swiglu). The fused kernel supports the BF16 weight route;
    // the unfused chain stays wired behind the policy toggle as the
    // independent reference.
    if (route_plan.fuse_ffn_swiglu && ffn_g_bf16 && ffn_u_bf16) {
      LaunchBatchedFusedSwiGLUGEMM(
          layer.ffn_gate.data, true, layer.ffn_up.data, true, arena_.d_normed,
          arena_.d_ffn_act, arena_.d_scratch_bf16, batch_size,
          intermediate_size, hidden_size, arena_.stream);
    } else {
      gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ffn_gate, intermediate_size, hidden_size);

      gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ffn_up, intermediate_size, hidden_size);

      LaunchBatchedSwiGLUActivation(
          arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_ffn_act,
          arena_.d_scratch_bf16, batch_size * intermediate_size, arena_.stream);
    }

    gemm_weight(layer.ffn_down, arena_.d_scratch_bf16, arena_.d_ffn_act,
                arena_.d_ffn_out, hidden_size, intermediate_size);

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                             batch_size, hidden_size, arena_.stream);

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_ffn += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }
  }

  if (do_profile) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "\n[STRIX_PROFILE B=" << batch_size << "] Total: " << total_ms
              << " ms (" << (batch_size / (total_ms / 1000.0)) << " tok/s)\n"
              << "  - Norms:      " << time_norm << " ms\n"
              << "  - Input Proj: " << time_attn_proj << " ms\n"
              << "  - SSM Recur:  " << time_ssm_recur << " ms\n"
              << "  - SSM Out:    " << time_ssm_out << " ms\n"
              << "  - FFN (3 GEMM): " << time_ffn << " ms\n"
              << "  - Dequant:    " << time_dequant << " ms\n"
              << "  - GEMM(deq):  " << time_gemm << " ms\n";
  }

  if (capture_prompt_hidden_) {
    const std::size_t old_size = h_prompt_hidden_.size();
    const std::size_t chunk_elements = batch_size * hidden_size;
    h_prompt_hidden_.resize(old_size + chunk_elements);
    HIP_CHECK(hipMemcpyAsync(h_prompt_hidden_.data() + old_size,
                             scratch.decode.hidden.data(),
                             chunk_elements * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
  }
  last_hidden_offset_ = (batch_size - 1) * hidden_size;

  if (!compute_logits) {
    return 0;
  }

  // 4. Output Norm for final token
  const float* final_hidden =
      scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size);
  LaunchRMSNorm(final_hidden,
                static_cast<const float*>(weights_.output_norm.data),
                scratch.decode.normed.data(), hidden_size, eps, arena_.stream);

  // 5. LM Head Logits GEMV on final token
  LaunchGEMV(weights_.output.data, weights_.output.type,
             scratch.decode.normed.data(), scratch.decode.logits.data(),
             vocab_size, hidden_size, arena_.stream);

  // 6. GPU Argmax. The sampled-token view enters its documented alias epoch
  // only after all SSM layer uses of alpha have completed.
  auto* d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUArgmax(scratch.decode.logits.data(), d_out_token, vocab_size,
                  arena_.stream);

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
