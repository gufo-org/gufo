// SPDX-License-Identifier: MIT
// Raw-pointer C ABI for DeepSeek's HIP quantized matrix kernels.
// Functions return zero on success and nonzero on failure. Device pointers
// remain caller-owned; every operation uses the supplied HIP stream.

#pragma once

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current HIP device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: HIP device ordinal (0 for the primary GPU).
// Returns 0 on success.
int ds4_mmq_init(int device);
void ds4_mmq_cleanup(void);
void ds4_mmq_set_aligned_q81_scratch(void *ptr, size_t bytes);

// ds4: routed MoE column-tile bound.
//
// The routed path sizes its column-tile grid from ncols_max, and without a
// better bound that has to be the whole chunk width: any one expert could in
// principle receive every token. Every block outside an expert's real bucket
// then exits immediately at the top of mul_mat_q, but it still costs a
// dispatch and a full 31.5 KiB shared-memory reservation, which caps residency
// at two workgroups per CU. At a 4,096-token chunk over 256 experts the mean
// bucket is n_tokens * n_expert_used / n_experts = 96 rows, so about 96% of the
// grid is empty.
//
// The caller already reads the per-expert assignment counts back to the host to
// build the Q2 down hot list, so it can pass the real maximum here. rows <= 0
// restores the conservative bound. Purely a launch-geometry bound: the blocks
// it removes did no work, so results are unchanged.
void ds4_mmq_set_routed_max_expert_rows(int rows);

// Column-tile width for the next routed gate/up pass, when the caller knows the
// whole bucket distribution and not just its maximum.
//
// The default rule takes the widest tile that minimizes the tile count for the
// chunk width, which is the right answer for a dense column range and the wrong
// one for a routed one. A tile executes all its columns whether or not the
// bucket fills them, so the cost is `sum_e ceil(c_e / w) * w` column-slots plus
// one weight-panel reload per tile: at a 4,096-token chunk the mean bucket is 96
// rows and the 80-column tile is nearly full, while at 512 tokens it is twelve
// rows and about 85% of the matrix-core issue is padding.
//
// cols <= 0, or a width the device cannot host, restores the default rule.
// Purely a tiling choice: each output element still walks K once in the same
// order, so results are unchanged.
void ds4_mmq_set_routed_tile_cols(int cols);

// The width the model above picks for one per-expert assignment-count array.
// Returns 0 when the counts are unusable, which the setter treats as "default".
int ds4_mmq_routed_tile_cols_for_counts(const unsigned int *counts,
                                        int n_experts);

// Dense Q8_0 matmul through the mul_mat_q template:
//
//   out[col, row] = sum_k W[row, k] * X[k, col]      0 <= row < M, 0 <= col < N
//
// Layouts (matching ggml + llama.cpp mmq conventions, all on device):
//   W:       [M rows, K cols], row-major, packed in Q8_0 blocks. K must be a
//            multiple of 256.
//   X_f32:   [N rows, K cols] F32 row-major (logical [K, N] with K
//            innermost - i.e. for each "column" col of the logical [K, N]
//            matrix, K contiguous floats live at X[col*K .. col*K + K]).
//   out_f32: caller-allocated, M*N floats. mmq writes in column-major:
//            out[col*M + row]. Callers expecting row-major must transpose.
//
// Returns 0 on success, non-zero on validation or launch failure.
int ds4_mmq_q8_0_dense(
    const void  * W_q8_0,
    const float * X_f32,
    float       * out_f32,
    int           M,
    int           N,
    int           K,
    hipStream_t  stream);

// Paired IQ2_XXS MoE matmul. For each (token, slot-within-token's-top-k) pair
// it computes gate AND up over the same activation:
//
//   out[col, row] = sum_k W[ids[token, slot], row, k] * X[token, k]
//
// where col = token * n_expert_used + slot, row in [0, M), so the Q8_1
// quantize of X and the mm_ids_helper bookkeeping happen once. Both weights
// have the same shape (M, K, n_experts).
//
// Layouts:
//   W_a/W_b: device pointers, [n_experts, M rows, K cols] in IQ2_XXS blocks.
//            Per-expert slab is M*K/256 blocks stored contiguously; experts
//            are stacked.
//   X_f32:   device pointer, [n_tokens, K] F32 row-major (K innermost).
//   ids:     device pointer, [n_tokens, n_expert_used] int32_t row-major.
//            ids[t*n_expert_used + s] is the expert id for token t's
//            s-th routing slot.  Values must be in [0, n_experts).
//   out_a/out_b: caller-allocated, M * n_tokens * n_expert_used floats.
//            Column-major: out[col*M + row].
//
// K must be a multiple of 256. Each token's selected experts must be unique,
// which bounds every expert bucket by n_tokens instead of
// n_tokens * n_expert_used.
//
// Returns 0 on success; on error neither output is guaranteed valid.
int ds4_mmq_iq2_xxs_moe_pair_token_bound(
    const void    * W_a,
    const void    * W_b,
    const float   * X_f32,
    const int32_t * ids,
    float         * out_a,
    float         * out_b,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    hipStream_t    stream);

// Non-stream-K variant for ROCm prefill. The up result stays in the MMQ
// write-back register and is combined with the materialized gate, router
// weight, clamp, and SwiGLU directly into pair-major FP32/FP16 mid outputs.
// discard must be a valid, disjoint device pointer but is not written.
int ds4_mmq_iq2_xxs_moe_pair_token_bound_swiglu(
    const void    * W_gate,
    const void    * W_up,
    const float   * X_f32,
    const int32_t * ids,
    const float   * router_weights,
    float         * gate_f32,
    float         * discard,
    float         * mid_f32,
    void          * mid_f16,
    int             M,
    int             K,
    int             n_tokens,
    int             n_experts,
    int             n_expert_used,
    float           clamp,
    hipStream_t    stream);

// Set the thread-local stream that the internal cuda pool uses for
// hipMallocAsync / hipFreeAsync.  Defaults to hipStreamPerThread.
// Use the graph capture stream so pool
// allocations land on the captured stream and don't invalidate capture.
// Pass NULL to reset to hipStreamPerThread.
void ds4_pool_set_stream(hipStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif
