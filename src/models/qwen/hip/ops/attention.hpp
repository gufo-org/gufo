#ifndef GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::hip {

/// Computes Fused QKV Projections for Full Attention layers in a single kernel
void LaunchFusedQKVProjections(const void* q_w, core::GgmlType q_type,
                               const void* k_w, core::GgmlType k_type,
                               const void* v_w, core::GgmlType v_type,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream = nullptr);

/// Fused layer pre-RMSNorm + QKV projections (opt-c010-rmsnorm-projection).
/// Computes the norm over x and feeds the projections, matching the unfused
/// RMSNormKernel + LaunchFusedQKVProjections chain bit-for-bit.
void LaunchFusedRMSNormQKVProjections(
    const float* x, const float* norm_w, float eps, const void* q_w,
    bool q_is_bf16, const void* k_w, bool k_is_bf16, const void* v_w,
    bool v_is_bf16, float* q_out, float* k_out, float* v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size, hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention with KV-cache and optional gating
/// on GPU. Production supplies canonical FP16 K/V; the independent validation
/// fallback supplies FP32 K/V. Tests may supply both representations. When
/// skip_kv_write is true the selected KV cache is assumed already written
/// (e.g. by the fused QK norm+RoPE kernel, opt-c010-qk-rope-kv).
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr,
                     float* split_k_scratch = nullptr,
                     bool skip_kv_write = false);

/// Computes Grouped-Query Softmax Attention reading position from device memory
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr, bool skip_kv_write = false);

/// Fuses per-head Q/K RMSNorm, RoPE, and the KV-cache write for a single decode
/// token into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, and K/V into every non-null cache
/// representation supplied by the caller.
void LaunchFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr);

/// Batched Causal Attention for B tokens with KV Cache. When skip_kv_write is
/// true the KV cache is assumed already written by the fused prefill kernel
/// (opt-c010-qk-rope-kv). When out_context_bf16 is non-null the context is also
/// written as BF16, which saves the caller a conversion launch when the next
/// projection consumes BF16.
void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            void* k_cache_f16, void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream = nullptr,
                            bool skip_kv_write = false,
                            void* out_context_bf16 = nullptr);

/// Batched fuse of per-head Q/K RMSNorm, RoPE, and the KV-cache write across B
/// tokens into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, and K/V into every non-null cache
/// representation supplied by the caller.
void LaunchBatchedFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr);

/// Qwen3.8-specific causal GQA tile for gfx1151. The kernel processes 16 query
/// positions and two query heads per block while reusing one FP16 K/V tile.
/// Returns false for unsupported model shapes.
/// `lse_out`, when non-null, makes this one half of a split attention: the
/// kernel starts at `key_begin`, writes the partial log-sum-exp per
/// [head, token], and leaves the SiLU gate to the merge step. `skip_kv_write`
/// suppresses the KV-cache pack/sync when another half already did it.
[[nodiscard]] bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, float* lse_out = nullptr,
    std::uint32_t key_begin = 0, bool skip_kv_write = false);

/// Non-causal attention of every query in the chunk against the first
/// `prefix_length` cached keys, through AOTriton's pretuned gfx11xx flash
/// attention. Returns false if AOTriton rejects the shape.
/// Masked prefill attention on the WMMA matrix cores
/// (opt-c177-attn-wmma). Same contract as LaunchBatchedAttentionTile -- it
/// packs the FP16 K/V cache unless `skip_kv_write`, masks keys in
/// [`key_begin`, min(context_end, query position]), and emits the partial
/// log-sum-exp into `lse_out` (suppressing the gate) when that is non-null.
/// Returns false when the shape is unsupported, so the caller can fall back.
[[nodiscard]] bool LaunchQwenWmmaAttention(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, float* lse_out = nullptr,
    std::uint32_t key_begin = 0, bool skip_kv_write = false);

[[nodiscard]] bool LaunchQwenAotritonPrefixAttention(
    const __half* q_half, const void* k_cache_f16, const void* v_cache_f16,
    __half* out_prefix, float* lse_prefix, std::size_t batch_size,
    std::uint32_t prefix_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Converts FP32 queries to FP16 in place-compatible [token][head][dim] order.
void LaunchConvertQueriesToHalf(const float* q, void* q_half,
                                std::size_t num_elements,
                                hipStream_t stream = nullptr);

/// Combines the prefix and diagonal partial attentions by log-sum-exp and
/// applies the SiLU gate to the merged result.
void LaunchMergeSplitAttention(const void* out_prefix, const float* lse_prefix,
                               const float* out_diag, const float* lse_diag,
                               const float* gate, float* out,
                               std::size_t batch_size, std::uint32_t num_heads,
                               std::uint32_t head_dim,
                               hipStream_t stream = nullptr);

/// Causal GQA through ROCm Composable Kernel. Inputs and outputs remain FP32
/// at the executor boundary; the fused attention operator uses FP16 tiles with
/// FP32 accumulation and online softmax. Returns false for unsupported shapes.
[[nodiscard]] bool LaunchBatchedAttentionCk(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    void* scratch_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

/// Large-batch causal attention using float32 QK/PV GEMMs and one reusable
/// [batch, context] score buffer.
void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* scores, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_
