#ifndef GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_OPS_HPP_
#define GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_OPS_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::fish_drama3::hip {

void LaunchBackboneRmsNorm(const float* input, const float* weight,
                           float* output, std::size_t rows, std::size_t columns,
                           float eps, hipStream_t stream);

void LaunchBackboneResidualAdd(float* x_inout, const float* update,
                               std::size_t elements, hipStream_t stream);

void LaunchBackboneResidualAddRmsNorm(float* x_inout, const float* update,
                                      const float* weight, float* normed_out,
                                      std::size_t rows, std::size_t columns,
                                      float eps, hipStream_t stream);

void LaunchBackboneFullQkvProjFusedGemv(const std::uint16_t* q_w,
                                        const std::uint16_t* k_w,
                                        const std::uint16_t* v_w,
                                        const float* input_f32, float* q_out,
                                        float* k_out, float* v_out,
                                        std::size_t q_rows, std::size_t kv_rows,
                                        std::size_t cols, hipStream_t stream);

void LaunchBackboneLinearProjFusedGemv(
    const std::uint16_t* qkv_w, const std::uint16_t* z_w,
    const std::uint16_t* b_w, const std::uint16_t* a_w, const float* input_f32,
    float* qkv_out, float* z_out, float* b_out, float* a_out,
    std::size_t qkv_rows, std::size_t z_rows, std::size_t ba_rows,
    std::size_t cols, hipStream_t stream);

void LaunchBackboneBf16Gemv(const std::uint16_t* weight_bf16,
                            const float* input_f32, float* output_f32,
                            std::size_t batch, std::size_t rows,
                            std::size_t cols, hipStream_t stream);

void LaunchBackboneFloatToBf16(const float* input, std::uint16_t* output,
                               std::size_t elements, hipStream_t stream);

// Full Attention ops (layers where (l + 1) % 4 == 0):
// q_full is [steps, heads * 2 * head_dim] where each head has [head_dim Q,
// head_dim Gate]. k_in and v_in are [steps, kv_heads * head_dim]. Applies
// per-head Gemma RMSNorm (q_norm, k_norm) and partial NeoX (half-split) RoPE on
// rotary_dim, writes K and V into key_cache and value_cache at start_pos +
// step, and writes rotated Q to q_out.
void LaunchBackboneFullAttnPrep(
    const float* q_full, const float* k_in, const float* v_in,
    const float* q_norm, const float* k_norm, float* q_out, float* key_cache,
    float* value_cache, std::size_t steps, std::size_t start_pos,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t rotary_dim, float rope_theta, float eps, hipStream_t stream);

// Causal GQA Attention + sigmoid output gate from q_full.
void LaunchBackboneFullAttnGqaAndGate(const float* q, const float* key_cache,
                                      const float* value_cache,
                                      const float* q_full, float* attended_out,
                                      std::size_t steps, std::size_t start_pos,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim, hipStream_t stream);

// Linear Attention (Gated DeltaNet) ops (layers where (l + 1) % 4 != 0):
// 1. Causal Conv1d (kernel = 4) + SiLU on qkv [steps, 8192], updating
// conv_state [3, 8192].
void LaunchBackboneLinearConv1dSilu(const float* qkv_in,
                                    const float* conv_weight, float* conv_state,
                                    float* qkv_conv_out, std::size_t steps,
                                    std::size_t channels, hipStream_t stream);

// 2. Gated DeltaNet recurrent step across `steps` frames:
// Register-resident 32x128x128 state update + Gated RMSNorm with silu(z).
void LaunchBackboneGatedDeltaNetRecurrent(
    const float* qkv_conv, const float* z, const float* b, const float* a,
    const float* dt_bias, const float* neg_exp_a, const float* norm_weight,
    float* ssm_state, float* attn_out, std::size_t steps, std::size_t k_heads,
    std::size_t v_heads, std::size_t head_dim, float eps, hipStream_t stream);

// Native FP8 (F8_E4M3 + 128x128 block scale_inv) MoE + BF16 Shared Expert ops:
// 1. Router: computes 256 expert logits + 1 shared_expert_gate logit for
// `steps` tokens,
//    applies softmax, selects top-8 experts, normalizes top-8 weights, and
//    computes sigmoid(shared_gate).
void LaunchBackboneMoERouter(const std::uint16_t* router_weight_bf16,
                             const std::uint16_t* shared_gate_weight_bf16,
                             const float* ffn_in, std::int32_t* top_indices,
                             float* top_weights, float* shared_gate_scale,
                             std::size_t steps, std::size_t num_experts,
                             std::size_t top_k, std::size_t hidden_size,
                             hipStream_t stream);

// 2. Fused Gate + Up + SwiGLU for top-8 FP8 routed experts AND 1 BF16 shared
// expert:
//    For each token s in [0, steps):
//    - Slots k = 0..top_k-1 compute routed expert top_indices[s*top_k + k]
//    using F8_E4M3 weights + 128x128 scale_inv.
//    - Slot k = top_k computes shared expert using BF16 weights.
//    Writes output to expert_inter [steps, top_k + 1, inter_size].
void LaunchBackboneMoEGateUpSwiGlu(
    const std::uint8_t* experts_gate_fp8, const float* experts_gate_scale,
    const std::uint8_t* experts_up_fp8, const float* experts_up_scale,
    const std::uint16_t* shared_gate_bf16, const std::uint16_t* shared_up_bf16,
    const float* ffn_in, const std::int32_t* top_indices, float* expert_inter,
    std::size_t steps, std::size_t top_k, std::size_t inter_size,
    std::size_t hidden_size, hipStream_t stream);

// 3. Fused Down projection + weighted combination + residual add into
// hidden_inout:
//    Combines the top_k FP8 routed experts (weighted by top_weights) and the
//    BF16 shared expert (weighted by shared_gate_scale) and adds the result
//    into hidden_inout [steps, hidden_size].
void LaunchBackboneMoEDownCombine(
    const std::uint8_t* experts_down_fp8, const float* experts_down_scale,
    const std::uint16_t* shared_down_bf16, const float* expert_inter,
    const std::int32_t* top_indices, const float* top_weights,
    const float* shared_gate_scale, float* hidden_inout, std::size_t steps,
    std::size_t top_k, std::size_t inter_size, std::size_t hidden_size,
    hipStream_t stream);

}  // namespace gufo::models::fish_drama3::hip
#endif

#endif  // GUFO_MODELS_FISH_DRAMA3_HIP_BACKBONE_OPS_HPP_
