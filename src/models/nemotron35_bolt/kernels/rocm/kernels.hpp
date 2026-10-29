#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private HIP kernels of the Nemotron 3.5 Bolt trunk other than the
/// quantized matrix products (kernels/rocm/mmq). Row-major float32 unless
/// stated; every launch is asynchronous on `stream`. Operator order and
/// formulas follow the CPU reference (reference.cpp).
namespace gufo::models::nemotron35_bolt::rocm {

/// x[t] = dequantized Q8_0 row tokens[t] of the embedding table.
hipError_t EmbedQ8_0(const void* table, const std::int32_t* tokens, float* x,
                     int n_tokens, int hidden, hipStream_t stream);

/// out[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w for `rows` rows of `width`.
hipError_t RmsNorm(const float* x, const float* w, float* out, int rows,
                   int width, float eps, hipStream_t stream);
/// RmsNorm with the F32 result rounded to FP16 (nearest even) as it stores.
hipError_t RmsNorm(const float* x, const float* w, __half* out, int rows,
                   int width, float eps, hipStream_t stream);

/// Bytes of one Q8_1 block as the vector products read it: half scale
/// d = max|x| / 127, half sum of x, then 32 int8 values round(x / d).
inline constexpr std::size_t kQ8_1BlockBytes = 36;

/// RMSNorm of short chunks (width a multiple of 128, at most 4096), writing
/// F32 rows to `out` and/or Q8_1 rows to `q8` (either may be null).
hipError_t RmsNormQ8(const float* x, const float* w, float* out, void* q8,
                     int rows, int width, float eps, hipStream_t stream);

/// x rows (width a multiple of 128, at most 4096) to Q8_1 rows.
hipError_t QuantizeQ8_1(const float* x, void* q8, int rows, int width,
                        hipStream_t stream);

/// x[i] += scale * a[i] (+ b[i] when b is not null).
hipError_t AddScaled(float* x, const float* a, float scale, const float* b,
                     std::size_t n, hipStream_t stream);

/// max(x, 0)^2 in place.
hipError_t Relu2(float* x, std::size_t n, hipStream_t stream);

/// Mamba-2 causal depthwise convolution with SiLU. `proj` holds T rows of
/// `stride` floats (the unscaled ssm_in output); the x|B|C channels start at
/// `xbc_offset` and are multiplied by `in_scale`. `state` holds kernel-1 rows
/// of raw scaled inputs before the first token, oldest first.
hipError_t MambaConv(const float* proj, int stride, int xbc_offset,
                     float in_scale, const float* weight, const float* bias,
                     const float* state, float* out, int n_tokens, int channels,
                     int kernel, hipStream_t stream);

/// Shifts the last kernel-1 raw inputs of this chunk into `state`.
hipError_t MambaConvState(const float* proj, int stride, int xbc_offset,
                          float in_scale, float* state, int n_tokens,
                          int channels, int kernel, hipStream_t stream);

/// MambaConv and MambaConvState of a one-token chunk for each of `rows`
/// independent sequences: proj row r is sequence r's token and states[r]
/// (device memory) its history. The same arithmetic per row.
hipError_t MambaConvRows(const float* proj, int stride, int xbc_offset,
                         float in_scale, const float* weight, const float* bias,
                         float* const* states, float* out, int rows,
                         int channels, int kernel, hipStream_t stream);

struct MambaScanArgs {
  const float* xbc;  ///< [T][channels] convolved x|B|C
  int channels;
  const float* proj;  ///< [T][stride] ssm_in output; dt at dt_offset
  int stride;
  int dt_offset;
  float in_scale;
  const float* dt_bias;  ///< [heads]
  const float* a;        ///< [heads], -exp(A_log)
  const float* d;        ///< [heads]
  float* state;          ///< [heads][d_state][head_dim]
  float* y;              ///< [T][heads * head_dim]
  int n_tokens;
  int heads;
  int head_dim;
  int groups;
  /// False leaves `state` as it was, for a verification pass that keeps
  /// its inputs and replays only the accepted prefix.
  bool write_state{true};
  /// When set, the n_tokens rows are one-token sequences: row r scans with
  /// the state at row_states[r] (device memory) instead of `state`.
  float* const* row_states{nullptr};
};

/// Selective scan with a scalar decay per head, token by token; d_state is
/// fixed at 128 and head_dim at 64.
hipError_t MambaScan(const MambaScanArgs& args, hipStream_t stream);

/// MambaGateNorm for short chunks: the result goes to Q8_1 rows of `inner`
/// values instead of back into y. Groups hold 1024 values.
hipError_t MambaGateNormQ8(const float* y, const float* proj, int stride,
                           float in_scale, const float* weight, void* q8,
                           int n_tokens, int inner, int groups, float eps,
                           hipStream_t stream);

/// y = RMSNorm per group(y * silu(z * in_scale)) * w, in place; z is the first
/// `inner` floats of each proj row.
hipError_t MambaGateNorm(float* y, const float* proj, int stride,
                         float in_scale, const float* weight, int n_tokens,
                         int inner, int groups, float eps, hipStream_t stream);
/// MambaGateNorm storing the result to `out` rounded to FP16 (nearest even);
/// y keeps the gated values.
hipError_t MambaGateNorm(float* y, const float* proj, int stride,
                         float in_scale, const float* weight, __half* out,
                         int n_tokens, int inner, int groups, float eps,
                         hipStream_t stream);

/// Stores T rows of K and V (kv_dim floats) as half at positions
/// *pos..*pos+T-1; `pos` is device memory so a captured graph can replay.
hipError_t KvAppend(const float* k, const float* v, __half* k_cache,
                    __half* v_cache, const int* pos, int n_tokens, int kv_dim,
                    hipStream_t stream);

/// KvAppend of one token for each of `rows` sequences: row r at pos[r] of
/// k_caches[r] and v_caches[r] (device memory).
hipError_t KvAppendRows(const float* k, const float* v, __half* const* k_caches,
                        __half* const* v_caches, const int* pos, int rows,
                        int kv_dim, hipStream_t stream);

/// Causal GQA without positional encoding for queries at positions
/// *pos..*pos+T-1 over cache positions 0..*pos+t; head_dim is fixed at 128
/// and every KV head serves 16 query heads. Runs on FP16 WMMA: the scaled
/// query and the softmax probabilities round to FP16, scores, normalizer and
/// output accumulate in F32.
hipError_t Attention(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, const int* pos,
                     int n_tokens, int heads, int kv_heads, float scale,
                     hipStream_t stream);

/// Router logits out[t][r] = sum_c w[r][c] * x[t][c] for F32 weights of
/// `rows` x `cols`, streaming the weights once; `cols` must be a multiple of
/// 1024 and n_tokens at most kRouterMaxTokens.
inline constexpr int kRouterMaxTokens = 8;
hipError_t RouterLogits(const float* w, const float* x, float* out, int rows,
                        int cols, int n_tokens, hipStream_t stream);

/// Attention for chunks of up to kAttentionShortTokens queries (decode and
/// verification): the same arithmetic as Attention with the keys split over
/// kAttentionSplits blocks per KV head, each reading its keys once for all
/// 16 query heads, then merged. `partial` holds n_tokens * heads *
/// kAttentionSplits * (head_dim + 2) floats.
inline constexpr int kAttentionSplits = 64;
inline constexpr int kAttentionShortTokens = 8;
hipError_t AttentionShort(const float* q, const __half* k_cache,
                          const __half* v_cache, float* out, const int* pos,
                          int n_tokens, int heads, int kv_heads, float scale,
                          float* partial, hipStream_t stream);

/// AttentionShort of one query for each of `rows` sequences (at most
/// kAttentionShortTokens): row r at pos[r] over k_caches[r] / v_caches[r],
/// with the arithmetic of a one-token AttentionShort.
hipError_t AttentionShortRows(const float* q, const __half* const* k_caches,
                              const __half* const* v_caches, float* out,
                              const int* pos, int rows, int heads, int kv_heads,
                              float scale, float* partial, hipStream_t stream);

/// Sigmoid scores, selection by score + bias, top-k ids and weights of the
/// raw scores, optionally normalized (sum clamped at 6.103515625e-5), times
/// `scale`. Ties choose the lower expert index. n_experts <= 512.
hipError_t RouterTopK(const float* logits, const float* bias, int n_experts,
                      int n_tokens, int k, bool normalize, float scale,
                      std::int32_t* ids, float* weights, hipStream_t stream);

/// out[t] = sum_s weights[t * k + s] * slots[t * k + s] (rows of `width`).
hipError_t WeightedSlotSum(const float* slots, const float* weights, float* out,
                           int n_tokens, int k, int width, hipStream_t stream);

/// WeightedSlotSum into Q8_1 rows (width a multiple of 128, at most 4096).
hipError_t WeightedSlotSumQ8(const float* slots, const float* weights, void* q8,
                             int n_tokens, int k, int width,
                             hipStream_t stream);

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_KERNELS_HPP_
