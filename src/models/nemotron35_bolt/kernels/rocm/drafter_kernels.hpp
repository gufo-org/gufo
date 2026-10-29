#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DRAFTER_KERNELS_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DRAFTER_KERNELS_HPP_

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private HIP kernels of the LiLiCorr DFlash drafter, following
/// llama.cpp's graph (src/models/dflash.cpp). Row-major float32 unless stated;
/// every launch is asynchronous on `stream`.
namespace gufo::models::nemotron35_bolt::rocm {

/// out = bf16(x), round to nearest even.
hipError_t ToBf16(const float* x, __hip_bfloat16* out, std::size_t n,
                  hipStream_t stream);

/// out[t][r] = sum_c w[r][c] * x[t][c] for BF16 weights of `rows` x `cols`,
/// streaming the weights once; `cols` must be a multiple of 1024 and
/// n_tokens at most kGemvMaxTokens.
inline constexpr int kGemvMaxTokens = 8;
hipError_t GemvBf16(const __hip_bfloat16* w, const float* x, float* out,
                    int rows, int cols, int n_tokens, hipStream_t stream);

enum class Activation : std::uint8_t { kNone, kSilu };
/// x[r][c] = act(x[r][c] * scale + bias[c]); bias may be null.
hipError_t ScaleBias(float* x, float scale, const float* bias, int rows,
                     int cols, Activation act, hipStream_t stream);

/// out = silu(gate * gate_scale) * up * up_scale.
hipError_t SiluGate(const float* gate, float gate_scale, const float* up,
                    float up_scale, float* out, std::size_t n,
                    hipStream_t stream);

/// DFlash2 dynamic depthwise convolution, causal within each block of
/// `block` tokens: out[t][c] = sum over taps i <= t % block of
/// (dynamic[t][side][i][c / group] + base[side][i][c]) * x[t - i][c].
/// `dynamic` holds 2 * kernel * (hidden / group) floats per token.
hipError_t DflashConv(const float* x, const float* dynamic, const float* base,
                      float* out, int n_tokens, int block, int hidden,
                      int group, int kernel, int side, hipStream_t stream);

/// NeoX rotary embedding in place on [n_tokens][heads][head_dim] rows at
/// positions *first + t.
hipError_t RopeNeox(float* x, const int* first, int n_tokens, int heads,
                    int head_dim, float theta, hipStream_t stream);

/// Stores K and V rows (kv_dim floats) as half in ring slots
/// (*first + t) % ring.
hipError_t RingAppend(const float* k, const float* v, __half* k_ring,
                      __half* v_ring, const int* first, int n_tokens,
                      int kv_dim, int ring, hipStream_t stream);

/// ring[(*first + t) % ring_rows] = x[t] for rows of `width` floats.
hipError_t RowsToRing(const float* x, float* ring, const int* first,
                      int n_tokens, int width, int ring_rows,
                      hipStream_t stream);

/// Drafter attention for queries at positions *first + t, t < n_tokens: every
/// query sees the ring positions from max(0, p - window + 1) up to the last
/// query's position, the whole block included (non-causal). Per-head sink
/// logits join the softmax normalizer. head_dim is fixed at 128 and
/// heads / kv_heads at 16.
hipError_t DrafterAttention(const float* q, const __half* k_ring,
                            const __half* v_ring, const float* sinks,
                            float* out, const int* first, int n_tokens,
                            int heads, int kv_heads, int ring, int window,
                            float scale, hipStream_t stream);

/// Per row: the 8 largest logits in descending order (ties: lower id), and
/// the row's log-sum-exp.
inline constexpr int kLiliCorrTopK = 8;
hipError_t TopK8(const float* logits, int rows, int vocab, std::int32_t* ids,
                 float* values, float* lse, hipStream_t stream);

/// LiLiCorr candidate features of `slots` x 8 nodes from TopK8: log p, p,
/// log p - log p of rank 0, rank / 7, rank == 0; then LayerNorm (eps 1e-5)
/// with the affine `w`, `b`. out holds 5 floats per node.
hipError_t LiliCorrFeatures(const float* values, const float* lse,
                            const float* w, const float* b, float* out,
                            int slots, hipStream_t stream);

/// x[node] = token[node] + pass[slot] + feature[node] + slot_embd[slot] +
/// rank_embd[rank], node = slot * 8 + rank.
hipError_t LiliCorrCombine(const float* token, const float* pass,
                           const float* feature, const float* slot_embd,
                           const float* rank_embd, float* x, int slots,
                           int width, hipStream_t stream);

/// Self-attention over `nodes` rows of q|k|v (3 * heads * 128 floats) with an
/// additive bias[head][query][key]; out holds heads * 128 floats per node.
hipError_t NodeAttention(const float* qkv, const float* bias, float* out,
                         int nodes, int heads, float scale, hipStream_t stream);

/// out[node] = x[node] | anchor | x[node] * anchor (3 * width floats).
hipError_t FactorInput(const float* x, const float* anchor, float* out,
                       int nodes, int width, hipStream_t stream);

/// x[r] = (x[r] + bias) / max(|x[r] + bias|, eps).
hipError_t BiasL2Norm(float* x, const float* bias, int rows, int width,
                      float eps, hipStream_t stream);

/// LiLiCorr lattice scores, scaled by `scale`: scores[k] =
/// in[k] . anchor for the first slot's candidates, then for each later slot s
/// scores[8 + (s - 1) * 64 + prev * 8 + next] = in[s * 8 + next] .
/// out[(s - 1) * 8 + prev].
hipError_t LatticeScores(const float* in, const float* out, const float* anchor,
                         float* scores, int slots, int width, float scale,
                         hipStream_t stream);

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_DRAFTER_KERNELS_HPP_
