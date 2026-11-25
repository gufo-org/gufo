// Simple, clearly-correct HIP kernels for the Inkling 2 Mini forward pass.
// One kernel per operation, F32 activations,
// dequantize-on-the-fly weights in their native formats. No performance work:
// see ext/lab for the optimised kernels. MIT. Quantization block layouts follow
// llama.cpp ggml-common.h (see THIRD_PARTY.md).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::inkling2::hip {

// ggml tensor type ids (the values stored in a GGUF).
enum GgmlTypeId : int {
  kTF32 = 0,
  kTF16 = 1,
  kTQ5_0 = 6,
  kTQ8_0 = 8,
  kTQ4_K = 12,
  kTQ5_K = 13,
  kTQ6_K = 14,
  kTIQ4_NL = 20,
  kTIQ4_XS = 23,
  kTBF16 = 30,
  kTNVFP4 = 40,
};

/// Bytes of one row of `k` elements of `type`; 0 when the type is not supported
/// or k does not fill whole blocks.
std::size_t RowBytes(int type, std::size_t k);

/// out[t*ldo + r] = sum_k W[r][k] * x[t*ldx + k] for r < n, t < t_count.
/// `w` is n rows of RowBytes(type, k) bytes.
void Gemm(hipStream_t s, int type, const void* w, int n, int k, const float* x,
          int ldx, float* out, int ldo, int t_count);

/// Expert-grouped GEMM. Pairs are sorted by expert: expert e owns
/// sorted_pairs[offsets[e] .. offsets[e+1]). For pair p the input row is
/// x[(gather_div ? p / gather_div : p) * ldx] and the result row is out[p * ldo
/// + r] = scale[e] * sum_k W[e][r][k] * x_row[k]. `w` holds n_expert x n rows.
/// `scale` may be null (1.0).
void MoeGemm(hipStream_t s, int type, const void* w, int n, int k, int n_expert,
             const float* scale, const float* x, int ldx, int gather_div,
             const int* sorted_pairs, const int* offsets, float* out, int ldo,
             int n_pairs);

/// Row `ids[t]` of a `type` matrix with k columns, dequantised to F32 (token
/// embedding).
void EmbedRows(hipStream_t s, int type, const void* w, int k, const int* ids,
               int t_count, float* out, int ldo);

/// out = x / sqrt(mean(x^2) + eps) * w (w may be null). Rows of `dim` elements.
void RmsNorm(hipStream_t s, const float* x, int ldx, const float* w, float* out,
             int ldo, int dim, int rows, float eps, void* out16 = nullptr);

/// Attention-residual mix. `states` (up to 8 device row-major [t][dim] buffers)
/// are RMS-normalised per token; for every query row q: p = softmax_j(query[q]
/// . S_j / sqrt(dim)), out[t][q] = sum_j p_j S_j. out has row stride nq*dim.
struct MixStates {
  const float* ptr[8];
  int n;
};
void Mix(hipStream_t s, const MixStates& states, const float* query, int nq,
         int dim, int t_count, float* out, float eps, float* inv_out = nullptr);

/// tau[t] = 1 + alpha * ln(max(1, (pos0 + t + 1) / floor)).
void LogScale(hipStream_t s, float* tau, int pos0, int t_count, float floor_,
              float alpha);

/// Short causal depthwise convolution (kernel 4). state is [C][3] (oldest
/// first). out = (x_t + sum_i w[c][i] x_{t-3+i} + dynamic) * scale; dyn is
/// [t][C/hd*4] or null.
void Conv(hipStream_t s, const float* x, const float* state, const float* w,
          const float* dyn, float* out, int t_count, int channels, int hd,
          float scale);
void ConvStateUpdate(hipStream_t s, const float* x, float* state, int t_count,
                     int channels);
/// Same state advance as ConvStateUpdate, reading the pre-ubatch `old_state`
/// and writing the state after `t_count` tokens into a separate `dst_state`.
void ConvStateCapture(hipStream_t s, const float* x, const float* old_state,
                      float* dst_state, int t_count, int channels);

void Add(hipStream_t s, const float* a, const float* b, float* out, int n);
void Copy(hipStream_t s, const float* a, float* out, int n);

/// Router: sigmoid, biased top-k selection, renormalised x route_scale weights.
void RouterTopK(hipStream_t s, const float* logits, const float* bias, int* ids,
                float* wts, int t_count, int n_expert, int n_used,
                float route_scale);
void Situ(hipStream_t s, const float* gu, float* out, int n_ff, int rows,
          float beta, float lb);
void MoeCombine(hipStream_t s, const float* y, const float* wts, float* out,
                int dim, int t_count, int n_used);
void Softcap(hipStream_t s, float* x, std::int64_t n, float cap);

}  // namespace gufo::models::inkling2::hip
