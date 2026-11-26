// Kernels of the Inkling 2 Mini DSpark drafter (see ../dspark.hpp).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::inkling2::dsp {

inline constexpr int kEmbd = 2304, kQ = 3072, kKV = 1024, kHead = 128,
                     kHeads = 24, kKvHeads = 8, kFf = 9216, kRank = 256,
                     kJoint = 2 * kRank + 2 * kEmbd,  // 5120
    kBlock = 8, kFeat = 4, kLayers = 3;

/// mix[n][j] (j < 4: the three layers' context inputs, then the Markov head's)
/// = rms_norm(sum_i alpha[j][i] * feat[n][i]) * w[j]; out16 gets F16 copies of
/// j < 3 for the projections.
struct MixArgs {
  const float* feat;  // [n][4][2304]
  float alpha[4][4];
  const float* w[4];
  float* out;       // [n][4][2304]
  _Float16* out16;  // [3][ld16][2304]
  int n;
  float eps;
  int ld16 = 0;  // rows between out16's layers (0: n)
};
void Mix(hipStream_t s, const MixArgs& a);

/// Context K/V of rows [0, n) at positions pos0 + r: per-head RMS norm of K
/// times `knorm`, NeoX RoPE (theta), F16 into the cache rows
/// (pos % cap).
struct KvStoreArgs {
  const float* k;  // [n][1024]
  const float* v;  // [n][1024]
  int ldk, ldv;
  const float* knorm;  // [128]
  _Float16* kc;        // [cap][1024]
  _Float16* vc;
  int cap, pos0, n;
  float theta, eps;
};
void KvStore(hipStream_t s, const KvStoreArgs& a);

/// x[0] = rms_norm(token_embd[anchor]) * tok_norm (Q8_0 rows), x[1..7] =
/// mask; h = rms_norm(x) * w (F32 and F16).
void Embed(hipStream_t s, const void* token_embd, const float* tok_norm,
           const int* anchor, const float* mask, const float* w, float* x,
           float* h, _Float16* h16, float eps);

/// Per-head RMS norm (weights [128]) and NeoX RoPE of the 8 block rows of Q
/// [8][3072] and K [8][1024] at positions pos[0] + r, in place.
void QkNormRope(hipStream_t s, float* q, float* k, const float* qnorm,
                const float* knorm, const int* pos, float theta, float eps);

/// Block attention: every block row attends to the context rows [lo, pos)
/// (lo = pos - window, or 0) of the cache and to all 8 block rows (k, v).
/// out16 [8][3072] (F16, the o projection's input).
struct AttnArgs {
  const float* q;  // [8][3072] (normed, rotated)
  const float* k;  // [8][1024]
  const float* v;  // [8][1024]
  const _Float16* kc;
  const _Float16* vc;
  int cap;
  const int* pos;  // device: anchor position
  int window;      // 0: global
  int first;       // first injected context position
  float* ws;       // split workspace
  _Float16* out16;
  int nsplits = 32;
};
inline constexpr int kAttnSplits = 32;
void Attn(hipStream_t s, const AttnArgs& a);

/// x += y; h = rms_norm(x) * w (F32 and F16 copies), rows [0, n).
void AddNorm(hipStream_t s, float* x, const float* y, const float* w, float* h,
             _Float16* h16, int n, float eps);
/// out16 = silu(g) * u, [n][9216] from gu [n][2][9216]? (gate rows, then up
/// rows, each [n][9216]).
void Swiglu(hipStream_t s, const float* g, const float* u, _Float16* out16,
            int n);

/// One Markov RNN step of the chain (row `i` of the block):
/// joint = W_j [state, w1[prev], h_i, f] + b; state = cand + sig(gate) (state -
/// cand) with cand = tanh; lat = tanh, at joint + 3 * 256 (LrLogits8).
struct MarkovArgs {
  const uint16_t* w1;  // BF16 [201024][256]
  const uint16_t* wj;  // BF16 [768][5120]
  const float* bj;     // [768]
  const float* h;      // [8][2304] block rows (output-normed)
  const float* f;      // [2304] Markov context of the token before the block
  float* state;        // [256]
  float* joint;        // [768] scratch
  const int* prev;     // device: previous token
  int row;
};
void MarkovStep(hipStream_t s, const MarkovArgs& a);

/// The proposal logits of one Markov step without a head pass: W_U (h_i + W_o
/// lat + b_o) = base + (W_U W_o) lat + W_U b_o, with base = W_U h_i (Gemv8 of
/// the block, once per draft), pt = (W_U W_o)^T [256][n] as an int8 table with
/// one scale per row (n % 4 == 0, as QuantPt makes from the F32 table) and
/// cb = W_U b_o precomputed.
void LrLogits8(hipStream_t s, const int8_t* pt, const float* ps,
               const float* cb, const float* base, const float* lat,
               float* logits, int n);
/// Batched LrLogits8 for `count` (1..8) sessions sharing one read of `pt`,
/// `ps` and `cb`.
struct LrBatch {
  const float* base[8];
  const float* lat[8];
  float* out[8];
  int count;
};
void LrLogits8Batch(hipStream_t s, const int8_t* pt, const float* ps,
                    const float* cb, const LrBatch& batch, int n);
void QuantPt(hipStream_t s, const float* src, int n, int8_t* q, float* scale);
/// Row-wise squared L2 norms of BF16 [rows][256] into `out` [rows].
void RowNormSqBf16(hipStream_t s, const uint16_t* src, int rows, float* out);
/// Quantizes and transposes row-major BF16 [n][256] into int8 [256][n] + [n]
/// scales.
void QuantPtBf16(hipStream_t s, const uint16_t* src, int n, int8_t* q,
                 float* scale);

/// tokens[row] = map[argmax(logits[0, n))] (first maximum; map null: the
/// index itself), also *prev; conf[row] = the softmax probability of that
/// maximum.
void Argmax(hipStream_t s, const float* logits, int n, const int* map,
            float* ws, int* tokens, int row, int* prev, float* conf);

/// y[r][m] = W[m] . x[r] for `nrows` (1..64) rows: W Q8_0 [M][K], x F16
/// [nrows][K] (K a multiple of 32, M of 16). `qa` ([ntiles][K/32][2][16] v4i)
/// and `qs` ([ntiles][K/32][16] float) are device quantization scratch.
void Gemv8(hipStream_t s, const void* w, int m, int k, const _Float16* x,
           float* y, int nrows, void* qa, float* qs);
/// Gemv8 of n (<= 3) matrices of the same input in one launch.
void Gemv8N(hipStream_t s, int n, const void* const* w, const int* m, int k,
            const _Float16* x, float* const* y, int nrows, void* qa, float* qs);

/// Coupled sampling of one proposal: p over the n proposal-head rows (in
/// token order) = softmax(softcap(logits) / temperature), rows below the
/// min-p floor dropped; the token where the running sum first passes u
/// (the target sampler's linear draw with the uniform it will use), mapped
/// through `map`, goes to tokens[row] and *prev, its p to conf[row].
/// ws: 3 * 128 + 1 floats.
void CoupledSample(hipStream_t s, const float* logits, int n, const int* map,
                   float softcap, float temperature, float min_p, float u,
                   float* ws, int* tokens, int row, int* prev, float* conf);

/// dst[i] = F16(src[i]), i < n.
void ToHalf(hipStream_t s, const float* src, std::size_t n, _Float16* dst);

/// Copies the Q8_0 rows ids[0, n) of `src` (2304 columns) to `dst`.
void GatherQ8Rows(hipStream_t s, const void* src, const int* ids, int n,
                  void* dst);

}  // namespace gufo::models::inkling2::dsp
