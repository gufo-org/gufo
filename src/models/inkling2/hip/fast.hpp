// Fast kernels for the Inkling 2 Mini forward pass: the
// validated lab kernels (ext/lab: GEMV, fused projections, router, SITU MoE,
// split-K attention, WMMA grouped GEMMs) integrated into the engine, plus the
// small fused per-visit ops. MIT. Weight formats with a fast path: Q8_0 (dense,
// experts) and NVFP4 (experts). The engine falls back to the simple kernels of
// kernels.hpp for every other type.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::inkling2::fast {

/// Device table entry of one visit slot of a session (KV cache and conv
/// history). The engine keeps one table on the device that is overwritten from
/// the running session before every decode step, so a captured graph does not
/// depend on the session.
struct SlotEntry {
  void* k;
  void* v;
  float* conv;
  int cap;  // KV rows (position p lives in row p % cap)
  int pad;
};

enum Fmt : int { kFmtQ8 = 0, kFmtNvfp4 = 1 };

// ------------------------------------------------------------------ decode
// GEMV family (T = 1)
struct ProjSpec {  // fused Q8_0 GEMV over up to 5 matrices of K = 2304
  const void* w[5];
  int m[5];
  int in[5];  // input vector index (0..ni-1)
  int n;
  // Optional extra matrices of 24..64 rows (F32 / F16 / BF16, ggml types)
  // computed by the first blocks of the same launch from the un-quantised
  // inputs: y[b * tys + r] = W[r] . x[b][tin] (sink and dynamic-tap
  // projections).
  int tn = 0;
  const void* tw[3];
  int ttype[3];
  int trows[3];
  int tin[3];
  float* ty[3];
  int tys[3];
};
/// y[yoff_i + r] for matrix i, x = [ni][2304]. Batched: `nb` tokens share
/// one read of the weights; token b reads x + b * ldx and writes y + b * ldy
/// (nb = 1: ldx, ldy unused).
void ProjMulti(hipStream_t s, const ProjSpec& p, const float* x, int ni,
               float* y, int nb = 1, int ldx = 0, int ldy = 0);
/// y[r] = (softcap > 0 ? softcap * tanh(. / softcap) : .) of W[r] . x, Q8_0
/// with K 2304 or 3072. Batched: x = [nb][k] contiguous, y = [nb][m].
void GemvQ8(hipStream_t s, int k, const void* w, const float* x, float* y,
            int m, float softcap, int nb = 1);
/// F32 (or F16 when `half`) [128][2304] router: logits workspace [nb][128],
/// counter [nb] (zero, self-resetting), ids / wts [nb][8], x [nb][2304].
void Router(hipStream_t s, const void* w, const float* bias, const float* x,
            float* logit_ws, unsigned* counter, int* ids, float* wts,
            float rscale, int nb = 1, bool half = false);
/// Batched: x [nb][2304], ids [nb][8], a [nb * 8][ff]. One block row per
/// (token, slot) pair.
void MoeGu(hipStream_t s, int fmt, const void* w, const float* escale,
           const float* x, const int* ids, float* a, int ff, float beta,
           float lb, int nb = 1);
/// Same maths with one block row per (token, expert) pair: yp [nb][8][d]
/// weighted partial outputs that the next ConvResid adds in slot order
/// (ConvResidArgs::nsum = 8). Smaller blocks (one expert's activations to
/// quantise).
/// Wide steps (9 to 64 tokens) through int8 WMMA, with the outputs of
/// ProjMulti / GemvQ8 (k = 2304 or 3072): the inputs are first quantized by
/// QuantRows (`per` inputs a token, token stride ldt floats) into qa
/// (kQaBytesPerVector each) and qs (kQsFloatsPerVector each).
inline constexpr std::size_t kQaBytesPerVector = 2 * 3072 / 32 * 2 * 16;
inline constexpr std::size_t kQsFloatsPerVector = 2 * 3072 / 16;
void QuantRows(hipStream_t s, int k, const float* x, int per, int ldt, int nb,
               void* qa, float* qs);
bool ProjMma(hipStream_t s, const ProjSpec& p, const void* qa, const float* qs,
             int per, float* y, int nb, int ldx, int ldy, const float* x);
void GemvMma(hipStream_t s, int k, const void* w, const void* qa,
             const float* qs, float* y, int m, float softcap, int nb);
/// Pairs of a step grouped by expert (MoeRouteGroup), for MoeGuGrouped /
/// MoeDownGrouped: the same outputs as MoeGu / MoeDownE, each expert's
/// weights read once for all its pairs.
struct MoeGroups {
  int* sorted;  // [pairs]
  int* ue;      // [128] used experts
  int* ub;      // [128] start of each one's pairs in `sorted`
  int* un;      // [128] its pair count
  int* nd;      // [1] used experts
  void* qx;     // the next grouped kernel's quantized inputs (MoeQuantGrouped)
};
void MoeRouteGroup(hipStream_t s, const int* ids, int pairs, MoeGroups g);
/// Quantizes the nv inputs x [nv][k] (k = 2304: the tokens' MoE inputs; 1152:
/// the pairs' activations) of the next grouped kernel into g.qx, once for all
/// the blocks that read them. g.qx holds kMoeQuantBytes per pair.
inline constexpr std::size_t kMoeQuantBytes = 2 * 1152 + 2 * 72 * 4;
void MoeQuantGrouped(hipStream_t s, int fmt, const float* x, int nv, int k,
                     MoeGroups g);
/// a [pair][ff]: SITU(gate, up) of every pair, inputs from MoeQuantGrouped(k
/// = 2304); `slots` >= used experts.
void MoeGuGrouped(hipStream_t s, int fmt, const void* w, const float* escale,
                  MoeGroups g, int slots, float* a, int ff, float beta,
                  float lb);
/// yp [pair][d]: weighted outputs, inputs from MoeQuantGrouped(k = 1152).
void MoeDownGrouped(hipStream_t s, int fmt, const void* w, const float* escale,
                    MoeGroups g, int slots, const float* wts, float* yp, int d);
/// MoeGu then MoeDownE (same outputs) in one launch for NVFP4 experts; cnt
/// [2][nb * 8] (zero, self-resetting). False (nothing launched) for other
/// formats.
bool MoeGuDown(hipStream_t s, int fmt, const void* wgu, const float* gscale,
               const void* wd, const float* dscale, const float* x,
               const int* ids, const float* wts, float* a, float* yp,
               unsigned* cnt, int ff, int d, float beta, float lb, int nb);
void MoeDownE(hipStream_t s, int fmt, const void* w, const float* escale,
              const float* a, const int* ids, const float* wts, float* yp,
              int d, int nb = 1);

// ------------------------------------------------------------------ small
// fused ops
/// out = (x conv-history) conv; block != null: blk = cur, cur = conv; else cur
/// = cur + conv.
struct ConvResidArgs {
  const float* x;  // [2304] projection output
  const float* w;  // [2304][4]
  const SlotEntry* slot;
  int conv_off;
  float scale;
  float* cur;
  float* blk;    // may be null
  int nsum = 0;  // > 1: x holds [nb][nsum][2304] partial rows that are added in
                 // slot order first
  // Verification (all nb rows are consecutive positions of one session): row
  // b's history is rows b-3..b-1 of this step (the stored history before
  // row 0), the history is not written, and each row's input goes to
  // cap[b * 6656 + conv_off + c] (CommitVerifyConv applies it).
  float* cap = nullptr;
  // With `cap`: each row's index within its session's rows (a step may verify
  // several sessions, each one's rows consecutive); null: all rows one session.
  const int* ri = nullptr;
};
/// Rows of one verification step (Engine::kMaxVerifyRows).
inline constexpr int kVerifyRows = 64;
/// Batched (nb tokens): x, cur, blk are [nb][2304]; token b uses slot[b * 48].
void ConvResid(hipStream_t s, const ConvResidArgs& a, int nb = 1);
/// After a verification step: every history of the session's 48 slots
/// becomes the last 3 inputs of (history, the first `keep` of its captured
/// rows, which start at step row `row0`).
/// `cap` is [48][kVerifyRows][6656].
void CommitVerifyConv(hipStream_t s, const SlotEntry* tbl, const float* cap,
                      int keep, int row0 = 0);
/// out[r] = the first maximum of the finite values of row r of x [rows][n]
/// (ws: rows * 64 * 2 floats).
void RowsArgmax(hipStream_t s, const float* x, int rows, int n, float* ws,
                int* out);
/// GPU linear sampling (temperature + optional min_p, FP64 softmax weights)
/// for `rows` rows of `vocab` logits. Only rows with `temp[r] > 0` are
/// sampled; `ws` holds rows * (64 * 2 + 1) doubles.
struct LinearSampleArgs {
  const float* logits;  // [rows][vocab]
  int rows;
  int vocab;
  const float* temp;   // [rows] (device; <= 0 skips row)
  const float* min_p;  // [rows] (device)
  const double* u;     // [rows] (device)
  double* ws;          // [rows][129] (device)
  int* out;            // [rows] (device)
};
void RowsLinearSample(hipStream_t s, const LinearSampleArgs& a);

/// Attention-residual mix: states (<= 8 vectors of 2304) RMS-normalised,
/// softmax of query . state / sqrt(2304), out[q] = sum_j p_j S_j / rms_j. One
/// block per query row.
struct MixJob {
  const float* st[8];
  int n;
  const float* query;  // [nq][2304]
  float* out;          // [nq][2304]
  int nq;
  int ntok = 1;  // prefill: token count; every state pointer is then a
                 // [ntok][2304] matrix
  float*
      outq[4];  // optional per-query output matrices [ntok][2304] (else `out`)
  const float* nw[4];  // when non-null: out rows are additionally
                       // RMS-normalised with this weight
  void* outh[4] = {};  // optional F16 copies of the outq rows [ntok][2304]
  int has_cr;  // nq == 1 only: apply `cr` (post-conv + residual of the previous
               // op) first
  ConvResidArgs cr;
  int ostride = 0;  // token stride of `out` (0: nq * 2304); decode batches
};
/// Reads every range of the list, in order (a cache prefetch: nothing is
/// written).
struct TouchList {
  static constexpr int kMax = 16;
  const void* p[kMax];
  size_t bytes[kMax];
  int n = 0;
  void Add(const void* q, size_t b) {
    if (b > 0 && q != nullptr && n < kMax) {
      p[n] = q;
      bytes[n] = b;
      ++n;
    }
  }
};
void TouchN(hipStream_t s, const TouchList& l, int blocks);
void Mix(hipStream_t s, const MixJob& j, float eps);

/// Several RMS norms of 2304 rows in one launch.
struct NormJobs {
  const float* x[3];
  const float* w[3];  // may be null
  float* out[3];
  int n;
  int xs[3];  // token strides (decode batches)
  int os[3];
};
void Rms(hipStream_t s, const NormJobs& j, float eps, int nb = 1);

struct CtlJobs {  // loop controllers of up to 4 streams
  const float* x[4];
  const float* msg[4];
  const void* down[4];    // [64][4608]  F32, or F16 when `half`
  const void* gain[4];    // [2304][64]
  const void* forget[4];  // [2304][64]
  int half = 0;
  float* g64[4];
  float* out[4];
  int n;
  int xs[4], ms[4], gs[4],
      os[4];  // token strides of x, msg, g64, out (decode batches)
};
void CtlGate(hipStream_t s, const CtlJobs& j, float eps, int nb = 1);
/// The Ctl apply step, then Rms of norm job i on stream i's output
/// (nj.n == j.n) in the same launch. cnt: [nb][4] zero, self-resetting.
void CtlApplyRms(hipStream_t s, const CtlJobs& j, const NormJobs& nj,
                 unsigned* cnt, float eps, int nb = 1);

struct TinyJobs {  // y = W x for matrices with 32..64 rows of K = 2304, F32 /
                   // F16 / BF16
  const void* w[3];
  int type[3];  // ggml type
  const float* x[3];
  float* y[3];
  int rows[3];
  int n;
  int xs[3], ys[3];  // token strides (decode batches)
};
void Tiny(hipStream_t s, const TinyJobs& j, int nb = 1);

struct KvPrepArgs {
  const float* k;   // [1024] raw K projection
  const float* v;   // [1024]
  const float* dk;  // [8][4] dynamic taps or null
  const float* dv;
  const float* wk;  // [1024][4] static taps
  const float* wv;
  const SlotEntry* slot;
  int conv_off_k,
      conv_off_v;  // float offsets of the conv history inside the slot
  const int* pos;
  float scale, qk_thr, eps;
  int ld = 0, ldd = 0;      // token strides of k / v and of dk / dv (decode
                            // batches); slot advances 48 entries per token
  float* cap = nullptr;     // verification capture (see ConvResidArgs::cap)
  const int* ri = nullptr;  // see ConvResidArgs::ri
};
void KvPrep(hipStream_t s, const KvPrepArgs& a, int nb = 1);

struct AttnArgs {
  const SlotEntry* slot;
  const int* pos;
  const float* q;       // [24][128] raw
  const float* r;       // [24][16] raw
  const float* qscale;  // [24][128]
  const float* rclip;   // [24][16]
  const void* bt;       // f16 [1024][16]
  const float* sink;    // [24]
  const float* gate;    // [24*128] or null
  float* out;           // [24][128]
  float* ws_o;
  float* ws_m;
  float* ws_l;
  int swa;
  int chunk, nsplit;
  float qk_thr, r_thr, tau_floor, tau_alpha;
  int ldp = 0;     // token stride of q / r / gate (decode batches); out is
                   // [nb][3072], sink [nb][24]
  int ws_tok = 0;  // token stride of ws_m / ws_l (floats); ws_o is ws_tok * 128
  int nrows = 0;   // attn_decode_rows: rows of the step (groups of NB per z)
};
void AttnDecode(hipStream_t s, const AttnArgs& a, int nb = 1);
/// KvPrep and the attention in one launch, then the split reduction, for rows
/// of distinct sessions (no verification capture): the same arithmetic.
void KvAttnDecode(hipStream_t s, const KvPrepArgs& kv, const AttnArgs& a,
                  int nb = 1);
/// AttnDecode of nb (2 to 4) rows of one session at consecutive positions
/// (a verification step) on a global visit: one block reads each K/V tile for
/// all rows; every row's result is AttnDecode's.
void AttnDecodeRows(hipStream_t s, const AttnArgs& a, int nb);

/// Counting sort of pair -> expert (P <= 4096): offsets [E+1], sorted pair ids.
void RouteBuild(hipStream_t s, const int* ids, int p, int e, int* off,
                int* sorted);

// ------------------------------------------------------------------ prefill (T
// > 1)
struct GemmItems {  // work-item list for a dense GEMM over T tokens (built on
                    // the host per ubatch)
  const int* nitems;
  const int* item_e;
  const int* item_start;
  const int* item_n;
  int max_items;
  int tile = 64;  // tokens per item: 64 or 128
};
struct MoeWs {  // device scratch of the expert grouping
  int *cnt, *goff, *fill, *nitems, *item_e, *item_start, *item_n, *postok,
      *pair;
  float* posw;
};
void CvtF16(hipStream_t s, const float* x, void* h, size_t n);
/// Sets `*non_exact = 1` if any of the `n` F32 values has non-zero low 16 bits.
void CheckBf16Exact(hipStream_t s, const float* x, size_t n,
                    unsigned* non_exact);
/// Exact high-16-bit truncation of `n` F32 values to BF16 (`uint16_t`).
void F32ToBf16(hipStream_t s, const float* x, void* dst, size_t n);
/// BF16 copy of a BF16-exact F32 [rows][64] controller gain/forget matrix in
/// the k-interleaved layout k_ctl_apply_norm reads: [16][rows][4].
void CtlWeightsK4(hipStream_t s, const float* x, void* dst, int rows);
/// Copies `rows x cols` weights of ggml type `type` (0=F32, 1=F16, 30=BF16)
/// into `padded_rows x cols` F32, zero-padding rows `[rows, padded_rows)`.
void PadRowsToF32(hipStream_t s, const void* src, int type, int rows,
                  int padded_rows, int cols, float* dst);
/// out[t][r] = W[r] . x[t] (Q8_0, K 2304 or 3072, m % 64 == 0); activations f16
/// [T][k].
void GemmDense(hipStream_t s, const void* w, int m, int k, const void* ah,
               float* out, const GemmItems& it);
/// Groups the p = T*8 (token, slot) pairs by expert into work items of `tile`
/// tokens.
void MoeRoute(hipStream_t s, const int* ids, const float* wts, int p, int tile,
              const MoeWs& ws);
void MoeGuPrefill(hipStream_t s, int fmt, int tile, const void* w,
                  const float* escale, const void* ah, void* act16,
                  const MoeWs& ws, int max_items, float beta, float lb, int ff);
/// Weighted output rows of every (token, slot) pair into `pairs` [ntok * 8][d];
/// MoeTail sums them per token in slot order (deterministic).
void MoeDownPrefill(hipStream_t s, int fmt, int tile, const void* w,
                    const float* escale, const void* act16, float* pairs,
                    const MoeWs& ws, int max_items, int d);

struct PrefillAttn {
  int n, pos0, cap, swa;
  const float *k, *v;  // [n][1024] post-conv (when kraw is null)
  // When set: raw K/V projections, convolved here (histories, static taps,
  // optional dynamic taps, scale) instead of k / v.
  const float *kraw = nullptr, *vraw = nullptr, *kst = nullptr, *vst = nullptr;
  const float *kw = nullptr, *vw = nullptr, *kdyn = nullptr, *vdyn = nullptr;
  float conv_scale = 1.f;
  const float *q, *r;  // [n][24][128], [n][24][16] raw
  const float *qscale, *rclip;
  void *kc, *vc;  // f16 caches
  void* qn;       // f16 [n][3072] scratch
  float* rn;      // f32 [n][384] scratch
  const void* bt;
  const float *sink, *gate;
  float* out;   // [n][24][128]
  void* out16;  // when set: F16 [n][24][128] instead of `out`
  float qk_thr, r_thr, eps, tau_floor, tau_alpha;
};
/// kv_write + Q/R prologue + WMMA flash attention (gate fused when `gate` is
/// set).
void AttnPrefill(hipStream_t s, const PrefillAttn& a);
/// After MoeDownPrefill: cur[t] += post conv (w, scale) of the slot sums of
/// `pairs`, with the convolution history `state` [c][3], which is then advanced
/// by the t inputs.
void MoeTail(hipStream_t s, const float* pairs, float* state, const float* w,
             float* cur, int t, int c, float scale);
/// Captures the MoE post-convolution history after the first `t` tokens of
/// `pairs` ([T][8][c]) without modifying `old_state`.
void MoeTailStateCapture(hipStream_t s, const float* pairs,
                         const float* old_state, float* dst_state, int t,
                         int c);
/// cur[t] += post conv of x with the history `state`, which is then advanced
/// (hip::Conv + ConvStateUpdate + Add in one pass).
void ConvAdd(hipStream_t s, const float* x, float* state, const float* w,
             float* cur, int t, int c, float scale);

/// Prefill loop controller and router over T tokens (F32 or exact-BF16 weights,
/// tiled SIMT).
struct CtlPJob {
  const float* x;  // [T][ldx]
  int ldx;
  const float* msg;  // [T][ldm]
  int ldm;
  const void* down;    // [64][4608]  F32, or BF16 when `half`
  const void* gain;    // [2304][64]
  const void* forget;  // [2304][64]
  float* g64;          // [T][64] scratch
  float* out;          // [T][2304]
  // optional [T] inverse RMS of each msg row (from hip::Mix), else computed
  // here
  const float* minv = nullptr;
  // optional [T] inverse RMS of each x row, summed as the gate prologue sums it
  const float* xinv = nullptr;
  int half = 0;
  // Optional RMS norm of the controlled rows (weight `nw`), fused into the
  // apply: writes `nout` (F32, may be null) and `nout16` (F16) instead of
  // `out`. Same values as hip::RmsNorm of `out`. Reads `gain4`/`forget4` (the
  // CtlWeightsK4 copies) instead of `gain`/`forget`.
  const float* nw = nullptr;
  const void* gain4 = nullptr;
  const void* forget4 = nullptr;
  float* nout = nullptr;
  void* nout16 = nullptr;
};
struct CtlPJobs {
  CtlPJob j[3];
  int n;
};
void CtlPrefill(hipStream_t s, const CtlPJobs& jobs, int t, float eps);
/// logits[t][r] = W[r] . x[t] for the [128][2304] router (F32, or BF16 when
/// `half`).
void RouterPrefill(hipStream_t s, const void* w, const float* x, float* logits,
                   int t, bool half = false);
/// Sigmoid, biased top-8 and renormalisation of router logits [t][128] (wave
/// per token).
void TopkPrefill(hipStream_t s, const float* logits, const float* bias,
                 int* ids, float* wts, int t, float rscale);
/// out[t][r] = W[r] . x[t] (r < rows <= 64, K = 2304, W padded to 64 rows,
/// F32).
void SkinnyPrefill(hipStream_t s, const float* x, int ldx, const float* w,
                   int rows, float* out, int ldo, int t);

}  // namespace gufo::models::inkling2::fast
