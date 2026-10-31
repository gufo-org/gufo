#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_OPS_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::fish_drama3::hip {

// Convert float32 buffer to BF16 (uint16_t) on GPU.
void LaunchTowerFloatToBf16(const float* input, std::uint16_t* output,
                            std::size_t elements, hipStream_t stream);

// RMSNorm: output = rms_norm(input, eps) * weight
void LaunchTowerRmsNorm(const float* input, const float* weight, float* output,
                        std::size_t rows, std::size_t columns, float eps,
                        hipStream_t stream);

// Residual add + RMSNorm:
// x_inout += update; normed_out = rms_norm(x_inout, eps) * weight
void LaunchTowerResidualAddRmsNorm(float* x_inout, const float* update,
                                   const float* weight, float* normed_out,
                                   std::size_t rows, std::size_t columns,
                                   float eps, hipStream_t stream);

// Fused step 1 of Tower layer:
// emb = bf16_to_f32(embedding_table[codes[r]]) * label_mask[r] + (extra ?
// extra[r] : 0) concat_out[r, 0..dim-1]     = rms_norm(emb, eps) * enorm
// concat_out[r, dim..2*dim-1] = rms_norm(hidden[r], eps) * hnorm
void LaunchTowerEmbedAndConcatNorm(const std::uint16_t* embedding_table_bf16,
                                   const std::int32_t* codes,
                                   const float* label_mask, const float* extra,
                                   const float* hidden, const float* enorm,
                                   const float* hnorm, float* concat_out,
                                   std::size_t steps, std::size_t dim,
                                   float eps, hipStream_t stream);

// Fused QKV split + per-head Q/K RMSNorm + partial Interleaved RoPE + KV cache
// write. qkv layout per row: [Q: heads*head_dim, Gate: heads*head_dim, K:
// kv_heads*head_dim, V: kv_heads*head_dim]
void LaunchTowerSplitQkvNormRopeCache(const float* qkv, const float* q_norm,
                                      const float* k_norm, float* q_out,
                                      float* key_cache, float* value_cache,
                                      std::size_t steps, std::size_t start_pos,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim,
                                      std::size_t rotary_dim, float rope_theta,
                                      float eps, hipStream_t stream);

// Causal GQA Attention + output sigmoid gate:
// Reads q [steps, heads, head_dim], key_cache [start_pos + steps, kv_heads,
// head_dim], value_cache [start_pos + steps, kv_heads, head_dim], and gate from
// qkv [steps, 9216] (offset heads*head_dim). Writes gated attention output
// [steps, heads * head_dim].
void LaunchTowerGqaAttentionAndGate(const float* q, const float* key_cache,
                                    const float* value_cache, const float* qkv,
                                    float* attended_out, std::size_t steps,
                                    std::size_t start_pos, std::size_t heads,
                                    std::size_t kv_heads, std::size_t head_dim,
                                    std::size_t qkv_stride, hipStream_t stream);

// BF16 Weight Matrix x FP32 Vector (or small batch) -> FP32 Output:
// weight_bf16 is [rows, cols], input_f32 is [batch, cols], output_f32 is
// [batch, rows].
void LaunchTowerBf16Gemv(const std::uint16_t* weight_bf16,
                         const float* input_f32, float* output_f32,
                         std::size_t batch, std::size_t rows, std::size_t cols,
                         hipStream_t stream);

// Fused BF16 Gate+Up GEMV + SwiGLU for decode (batch == 1):
// Computes output[r] = silu(dot(w1[r], input)) * dot(w3[r], input)
void LaunchTowerBf16SwiGluGemv(const std::uint16_t* w1_bf16,
                               const std::uint16_t* w3_bf16,
                               const float* input_f32, float* output_f32,
                               std::size_t rows, std::size_t cols,
                               hipStream_t stream);

void LaunchTowerSwiGlu(const float* gate, const float* up, float* output,
                       std::size_t elements, hipStream_t stream);

// GPU-side Top-K / Top-P / Greedy sampling on a single row of logits
// [vocab_size]. Writes the sampled token index to *out_code_dev and optionally
// out_codes_array[book_idx].
void LaunchTowerSampleTopKTopP(const float* logits, std::size_t vocab_size,
                               float temperature, int top_k, float top_p,
                               float uniform_sample, std::int32_t* out_code_dev,
                               std::int32_t* out_codes_array,
                               std::size_t book_idx, hipStream_t stream);

}  // namespace gufo::models::fish_drama3::hip
#endif

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_TOWER_OPS_HPP_
