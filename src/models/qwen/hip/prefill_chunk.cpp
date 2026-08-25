#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
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

  // Execute the pure Qwen route decision while keeping hipBLASLt failure as a
  // runtime fallback to hipBLAS, not as resolver state.
  const auto gemm_weight = [&](const models::QwenTensorRef& w,
                               const void* bf16_input, const float* fp32_input,
                               float* output, std::size_t m, std::size_t k,
                               const void* q8_act = nullptr) {
    const auto resolution = models::qwen::ResolveQwenGemmRoute(
        {.type = w.type,
         .batch_size = batch_size,
         .m = m,
         .k = k,
         .mode = models::qwen::QwenGemmMode::kHipPrefill,
         .capabilities = {.can_try_hipblaslt =
                              arena_.hipblaslt_gemm != nullptr}});
    if (!resolution.accepted()) {
      std::abort();
    }
    switch (resolution.route) {
      case models::qwen::QwenGemmRoute::kHipPrefillBf16LtTryThenBlas:
        if (arena_.hipblaslt_gemm->RunBf16(w.data, bf16_input, output,
                                           batch_size, m, k, arena_.stream)) {
          return;
        }
        [[fallthrough]];
      case models::qwen::QwenGemmRoute::kHipPrefillBf16Blas:
        LaunchHipblasGEMMBF16(arena_.hipblas_handle, w.data, bf16_input, output,
                              batch_size, m, k, arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillF32Blas:
        LaunchHipblasGEMM(arena_.hipblas_handle, w.data, false, fp32_input,
                          output, batch_size, m, k, arena_.d_scratch_bf16,
                          arena_.stream);
        return;
      case models::qwen::QwenGemmRoute::kHipPrefillQuantDirect: {
        const auto tg0 = std::chrono::high_resolution_clock::now();
        // Route Q8_0 weights through the native W8A8 WMMA matrix-core kernel
        // (zero scratch dequantization). Other quant types (Q6_K / Q5_K) and
        // batch==1 decode path keep their proven routes.
        if (w.type == core::GgmlType::kQ8_0) {
          if (q8_act != nullptr) {
            LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, q8_act, output,
                                               batch_size, m, k, arena_.stream);
          } else {
            LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output,
                                   batch_size, m, k, arena_.stream);
          }
        } else if (batch_size > 1) {
          const auto td0 = std::chrono::high_resolution_clock::now();
          LaunchDequantizeToBf16(w.type, w.data, arena_.d_weights_bf16, m * k,
                                 arena_.stream);
          if (do_profile) {
            HIP_CHECK(hipStreamSynchronize(arena_.stream));
            time_dequant += std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - td0)
                                .count();
          }
          if (arena_.hipblaslt_gemm != nullptr && m >= 1024 && k >= 1024 &&
              arena_.hipblaslt_gemm->RunBf16(arena_.d_weights_bf16, bf16_input,
                                             output, batch_size, m, k,
                                             arena_.stream)) {
            // hipBLASLt executed successfully
          } else {
            LaunchHipblasGEMMBF16(arena_.hipblas_handle, arena_.d_weights_bf16,
                                  bf16_input, output, batch_size, m, k,
                                  arena_.stream);
          }
        } else {
          LaunchBatchedQuantGEMM(w.type, w.data, bf16_input, output, batch_size,
                                 m, k, arena_.stream);
        }
        if (do_profile) {
          HIP_CHECK(hipStreamSynchronize(arena_.stream));
          time_gemm += std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - tg0)
                           .count();
        }
        return;
      }
      default:
        std::abort();
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

    const auto is_q8 = [](const models::QwenTensorRef& w) {
      return w.type == core::GgmlType::kQ8_0;
    };
    // opt-c173-norm-quant: when every projection reading a norm is Q8_0 the
    // FP32 normed row and the BF16 staging copy are both dead, so the norm can
    // write the tiled Q8_1 activation directly and skip two round trips.
    const bool norm_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        (layer.is_full_attention
             ? (is_q8(layer.attn_q) && is_q8(layer.attn_k) &&
                is_q8(layer.attn_v))
             : (is_q8(layer.attn_qkv) && is_q8(layer.attn_gate) &&
                is_q8(layer.ssm_alpha) && is_q8(layer.ssm_beta)));
    // Where these hold, the BF16 and FP32 buffers passed to gemm_weight below
    // are stale: the route for a Q8_0 tensor always reads the quantized
    // activation, and the flags are exactly the condition that every consumer
    // is Q8_0, so no route can reach them.
    const bool ffn_feeds_q8_only =
        IsFusedRMSNormQuantizeQ8_1Supported(hidden_size) &&
        !route_plan.fuse_ffn_swiglu && is_q8(layer.ffn_gate) &&
        is_q8(layer.ffn_up);

    // Pre-layer RMSNorm (generates BF16 into d_scratch_bf16 directly)
    if (norm_feeds_q8_only) {
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, /*residual=*/nullptr,
          static_cast<const float*>(layer.attn_norm.data), /*sum_out=*/nullptr,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
    } else {
      LaunchBatchedRMSNorm(arena_.d_hidden,
                           static_cast<const float*>(layer.attn_norm.data),
                           arena_.d_normed, arena_.d_scratch_bf16, batch_size,
                           hidden_size, eps, arena_.stream);
    }

    if (do_profile) {
      HIP_CHECK(hipStreamSynchronize(arena_.stream));
      auto t1 = std::chrono::high_resolution_clock::now();
      time_norm += std::chrono::duration<double, std::milli>(t1 - t0).count();
      t0 = t1;
    }

    if (layer.is_full_attention) {
      // The quantized activation is read only by Q8_0 projections; when q/k/v
      // are all BF16 (as in the Q8_K_XL artifact) nothing consumes it, and when
      // they are all Q8_0 the fused norm already wrote it.
      if (!norm_feeds_q8_only && (layer.attn_q.type == core::GgmlType::kQ8_0 ||
                                  layer.attn_k.type == core::GgmlType::kQ8_0 ||
                                  layer.attn_v.type == core::GgmlType::kQ8_0)) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_q, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, q_projection_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_k, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_k, kv_size, hidden_size, arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_v, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_v, kv_size, hidden_size, arena_.d_scratch_q8_act);

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
      // opt-c165-attn-split: at depth, the fully visible prefix is most of the
      // attention work and needs no causal mask, so it goes through AOTriton's
      // pretuned flash attention while the tiled kernel keeps only the N x N
      // diagonal. The two partial softmaxes merge exactly by log-sum-exp.
      // opt-c177-attn-wmma: one masked WMMA pass over the whole visible range,
      // which replaces both the tiled kernel and the AOTriton prefix plus merge
      // that the split path used. Falls through to the previous routes when the
      // shape is unsupported or an alternative is pinned.
      // opt-c180-kv-resync: the fused QK-norm/RoPE kernel above already wrote
      // this chunk's K and V into *both* the FP32 cache and its FP16 mirror,
      // and earlier chunks did the same for the prefix, as do all three decode
      // paths. So the attention launcher's own pack pass rewrites identical
      // bytes, and its prefix sync re-converts a prefix that already matches --
      // work that grows linearly with depth. The unfused fallback below only
      // applies RoPE in place and never touches the cache, so it still needs
      // the pack.
      const bool kv_already_written = fused_qknorm_rope_kv;
      bool wmma_attention = false;
      if (detail::ShouldUseWmmaPrefillAttention()) {
        wmma_attention = LaunchQwenWmmaAttention(
            arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
            arena_.d_kv_cache, arena_.d_kv_cache + total_k,
            arena_.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
            arena_.d_ssm_out, attn_layer_idx, start_pos, batch_size,
            arena_.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena_.stream,
            /*lse_out=*/nullptr, /*key_begin=*/0,
            /*skip_kv_write=*/kv_already_written);
        if (wmma_attention) {
          detail::EmitAttentionDispatch("prefill_wmma", "");
        }
      }

      bool split_attention = wmma_attention;
      if (!wmma_attention &&
          detail::ShouldUsePrefillAttentionSplit(start_pos, batch_size)) {
        LaunchConvertQueriesToHalf(arena_.d_q, arena_.d_attn_q_f16,
                                   batch_size * attention_size, arena_.stream);
        auto* layer_k_f16 =
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) +
            (static_cast<std::size_t>(attn_layer_idx) * arena_.GetMaxContext() *
             kv_size);
        auto* layer_v_f16 = layer_k_f16 + total_k;

        // The tiled launcher owns the KV pack/sync, so run the diagonal first.
        const bool diagonal = LaunchBatchedAttentionTile(
            arena_.d_q, arena_.d_k, arena_.d_v, arena_.d_ssm_gate,
            arena_.d_kv_cache, arena_.d_kv_cache + total_k,
            arena_.d_attention_kv_f16,
            static_cast<std::uint16_t*>(arena_.d_attention_kv_f16) + total_k,
            arena_.d_attn_prefix_out, attn_layer_idx, start_pos, batch_size,
            arena_.GetMaxContext(), config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, arena_.stream,
            arena_.d_attn_lse_diag, start_pos, false);
        if (diagonal &&
            LaunchQwenAotritonPrefixAttention(
                static_cast<const __half*>(arena_.d_attn_q_f16), layer_k_f16,
                layer_v_f16, static_cast<__half*>(arena_.d_attn_prefix_f16),
                arena_.d_attn_lse_prefix, batch_size, start_pos,
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream)) {
          LaunchMergeSplitAttention(
              arena_.d_attn_prefix_f16, arena_.d_attn_lse_prefix,
              arena_.d_attn_prefix_out, arena_.d_attn_lse_diag,
              arena_.d_ssm_gate, arena_.d_ssm_out, batch_size,
              config.num_attention_heads, config.head_dim, arena_.stream);
          split_attention = true;
          detail::EmitAttentionDispatch("prefill_split_aotriton", "");
        }
      }

      if (!split_attention) {
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
              selected_attention = launched
                                       ? SelectedAttention::kComposableKernel
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
      }

      // opt-c172-fp32-quant: the Q8_0 route reads only the quantized
      // activation, so the BF16 staging buffer is dead. Quantizing straight
      // from FP32 drops one launch and one round trip (about 75 MB per layer
      // at batch 2048) and keeps the activation's full precision going into the
      // Q8_1 codes instead of rounding to BF16 first.
      if (layer.attn_output.type == core::GgmlType::kQ8_0) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            attention_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * attention_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     attention_size, arena_.stream);
      }
      gemm_weight(layer.attn_output, arena_.d_scratch_bf16, arena_.d_ssm_out,
                  arena_.d_attn_out, hidden_size, attention_size,
                  arena_.d_scratch_q8_act);
      if (do_profile) {
        HIP_CHECK(hipStreamSynchronize(arena_.stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_attn_proj +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
      }
    } else {
      if (!norm_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      gemm_weight(layer.attn_qkv, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_qkv, ssm_qkv_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.attn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_ssm_gate, ssm_inner_size, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_alpha, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_alpha_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
      gemm_weight(layer.ssm_beta, arena_.d_scratch_bf16, arena_.d_normed,
                  arena_.d_beta_buf, time_step_rank, hidden_size,
                  arena_.d_scratch_q8_act);
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
      // opt-c170-deltanet-rowsplit: the recurrence is serial in the token index
      // but independent across state rows, so hoisting the row-uniform k/q
      // norms and gates into two tiny prologue kernels lets the 128 rows of a
      // head spread over 16 waves instead of being pinned to one block behind
      // four barriers per token. The previous single-block kernel stays wired
      // as the reference and is selectable with STRIX_SSM_RECURRENCE=baseline.
      // opt-c174-ssm-epilogue-quant: the gated SSM row feeds only the Q8_0
      // ssm_out projection, so the epilogue can emit the quantized activation
      // and skip the FP32 round trip.
      const bool ssm_epilogue_q8 =
          layer.ssm_out.type == core::GgmlType::kQ8_0 &&
          IsFusedSSMEpilogueQuantizeQ8_1Supported(config.SsmValueSize(),
                                                  ssm_inner_size);
      const bool ssm_row_split = detail::ShouldUseSsmRowSplitRecurrence(
          IsDeltaNetRowSplitSupported(config.ssm_state_size,
                                      config.SsmValueSize()),
          arena_.d_ssm_kq_scales != nullptr &&
              arena_.d_ssm_alpha_beta != nullptr);
      if (ssm_row_split) {
        LaunchBatchedSSMConvRecurrenceRowSplit(
            arena_.d_ssm_qkv, static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, arena_.d_conv_out,
            arena_.d_ssm_deltanet_state, arena_.d_alpha_buf, arena_.d_beta_buf,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data), arena_.d_ssm_gate,
            arena_.d_ssm_out,
            ssm_epilogue_q8 ? arena_.d_scratch_q8_act : nullptr,
            arena_.d_ssm_kq_scales, arena_.d_ssm_alpha_beta, l, batch_size,
            ssm_qkv_size, config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena_.stream);
      } else if (route_plan.fuse_ssm_epilogue) {
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

      if (ssm_row_split && ssm_epilogue_q8) {
        // The recurrence epilogue already wrote the quantized activation.
      } else if (layer.ssm_out.type == core::GgmlType::kQ8_0) {
        LaunchQuantizeActivationQ8_1FromFp32(
            arena_.d_ssm_out, arena_.d_scratch_q8_act, batch_size,
            ssm_inner_size, arena_.stream);
      } else {
        LaunchFloatToBfloat16(arena_.d_ssm_out, arena_.d_scratch_bf16,
                              batch_size * ssm_inner_size, arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     ssm_inner_size, arena_.stream);
      }
      gemm_weight(layer.ssm_out, arena_.d_scratch_bf16, arena_.d_ssm_out,
                  arena_.d_attn_out, hidden_size, ssm_inner_size,
                  arena_.d_scratch_q8_act);
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
    } else if (ffn_feeds_q8_only) {
      // The post-attention residual add folds into the norm: one pass reads the
      // hidden state and the attention output, writes the updated hidden state
      // for the next residual link, and emits the Q8_1 activation.
      LaunchBatchedFusedRMSNormQuantizeQ8_1(
          arena_.d_hidden, arena_.d_attn_out,
          static_cast<const float*>(layer.ffn_norm.data), arena_.d_hidden,
          arena_.d_scratch_q8_act, batch_size, hidden_size, eps, arena_.stream);
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
      if (!ffn_feeds_q8_only) {
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     hidden_size, arena_.stream);
      }
      if (layer.ffn_gate.type == core::GgmlType::kQ8_0 &&
          layer.ffn_up.type == core::GgmlType::kQ8_0) {
        LaunchBatchedDualQuantGEMMPreQuantized(
            layer.ffn_gate.type, layer.ffn_gate.data, layer.ffn_up.data,
            arena_.d_scratch_q8_act, arena_.d_ffn_gate, arena_.d_ffn_up,
            batch_size, intermediate_size, hidden_size, arena_.stream);
      } else {
        gemm_weight(layer.ffn_gate, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_gate, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);

        gemm_weight(layer.ffn_up, arena_.d_scratch_bf16, arena_.d_normed,
                    arena_.d_ffn_up, intermediate_size, hidden_size,
                    arena_.d_scratch_q8_act);
      }

      // opt-c164-swiglu-quant: when ffn_down reads Q8_0 the only consumer of
      // the activation is the quantized buffer, so SwiGLU can write it
      // directly. That drops the FP32 activation and BF16 scratch round trips
      // (about 500 MB per layer at batch 2048) and one kernel launch. Other
      // ffn_down formats still need the FP32/BF16 forms, so they keep the
      // unfused chain.
      if (layer.ffn_down.type == core::GgmlType::kQ8_0) {
        LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            arena_.d_ffn_gate, arena_.d_ffn_up, arena_.d_scratch_q8_act,
            batch_size, intermediate_size, arena_.stream);
      } else {
        LaunchBatchedSwiGLUActivation(arena_.d_ffn_gate, arena_.d_ffn_up,
                                      arena_.d_ffn_act, arena_.d_scratch_bf16,
                                      batch_size * intermediate_size,
                                      arena_.stream);
        LaunchQuantizeActivationQ8_1(arena_.d_scratch_bf16,
                                     arena_.d_scratch_q8_act, batch_size,
                                     intermediate_size, arena_.stream);
      }
    }

    gemm_weight(layer.ffn_down, arena_.d_scratch_bf16, arena_.d_ffn_act,
                arena_.d_ffn_out, hidden_size, intermediate_size,
                arena_.d_scratch_q8_act);

    LaunchBatchedResidualAdd(arena_.d_hidden, arena_.d_ffn_out, arena_.d_hidden,
                             batch_size, hidden_size, arena_.stream);

    // DFlash draft feature tap for target layers [1, 7, 13, 19, 25]
    if (arena_.d_target_layer_features != nullptr) {
      const std::size_t last_tok_offset = (batch_size - 1) * hidden_size;
      std::size_t tap_idx = 999;
      if (l == 1) tap_idx = 0;
      else if (l == 7) tap_idx = 1;
      else if (l == 13) tap_idx = 2;
      else if (l == 19) tap_idx = 3;
      else if (l == 25) tap_idx = 4;

      if (tap_idx < 5) {
        (void)hipMemcpyAsync(
            arena_.d_target_layer_features + (tap_idx * hidden_size),
            arena_.d_hidden + last_tok_offset,
            hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream);
      }
    }

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
    HIP_CHECK(hipMemcpyAsync(
        h_prompt_hidden_.data() + old_size, scratch.decode.hidden.data(),
        chunk_elements * sizeof(float), hipMemcpyDeviceToHost, arena_.stream));
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

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos) {
  const std::size_t batch_size = candidate_tokens.size();
  if (batch_size == 0) {
    return {};
  }
  const auto& config = weights_.config;
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t vocab_size = config.vocab_size;
  const float eps = 1e-6F;

  // 1. Run all 64 layers across all candidate tokens in a single parallel prefill chunk pass
  (void)ForwardPromptChunk(candidate_tokens, start_pos, false);

  auto scratch = arena_.GetScratchView(batch_size);

  // 2. Batched Output RMSNorm across all candidate tokens
  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights_.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, eps, arena_.stream);

  // 3. Batched LM head GEMV & Argmax
  auto* d_out_tokens = scratch.decode.sampled_token.data();
  for (std::size_t b = 0; b < batch_size; ++b) {
    LaunchGEMV(weights_.output.data, weights_.output.type,
               scratch.decode.normed.data() + (b * hidden_size),
               scratch.decode.logits.data(),
               vocab_size, hidden_size, arena_.stream,
               models::qwen::QwenGemmMode::kHipMtp);
    LaunchGPUArgmax(scratch.decode.logits.data(),
                    d_out_tokens + b, vocab_size, arena_.stream);
  }

  std::vector<tokenization::TokenId> predictions(batch_size, 0);
  HIP_CHECK(hipMemcpyAsync(predictions.data(), d_out_tokens,
                           batch_size * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return predictions;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
