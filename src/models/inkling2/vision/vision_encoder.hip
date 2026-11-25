// Inkling 2 Mini vision encoder: kernels + host driver. MIT.
// Reference: llama.cpp tools/mtmd/models/inkling2v.cpp, mtmd-graph.cpp
// (build_vit, build_inp, resize_position_embeddings, build_rope_2d,
// build_patch_merge_permute) at e412c6f.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "mmproj_gguf.hpp"
#include "vision_encoder.hpp"

namespace gufo::models::inkling2::vision {
namespace {

constexpr int kT_F32 = 0, kT_F16 = 1, kT_Q8 = 8, kT_BF16 = 30;
constexpr int kEmbd = 1024, kHeads = 16, kHeadDim = 64, kFfn = 4096,
              kLayers = 23;
constexpr int kPosGrid = 32;  // learned position table is 32 x 32
constexpr int kPatchElems = kPatch * kPatch * 3;  // 1200

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

enum Epi { kEpiStore = 0, kEpiGelu = 1, kEpiResScale = 2 };

__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// f32 -> f16 of a value first rounded to f32: the empty asm keeps the
// compiler from fusing the producing FMA into a single-rounding v_fma_mix.
__device__ __forceinline__ _Float16 to_h(float x) {
  asm("" : "+v"(x));
  return (_Float16)x;
}

// f16(a * b) rounded once, as the scaled f32 -> f16 conversion compiles
// (v_fma_mixlo_f16 a, b, -0).
__device__ __forceinline__ _Float16 mul_to_h(float a, float b) {
  uint32_t r;
  asm("v_fma_mixlo_f16 %0, %1, %2, neg(0)" : "=v"(r) : "v"(a), "v"(b));
  return __builtin_bit_cast(_Float16, uint16_t(r));
}

__device__ __forceinline__ float bf16_to_f(uint16_t u) {
  return __uint_as_float(uint32_t(u) << 16);
}

// One weight element (row = pointer to the start of the output row).
template<int WT>
__device__ __forceinline__ float load_w1(const uint8_t* row, int k) {
  if (WT == kT_F32)
    return reinterpret_cast<const float*>(row)[k];
  if (WT == kT_F16)
    return __half2float(reinterpret_cast<const __half*>(row)[k]);
  if (WT == kT_BF16)
    return bf16_to_f(reinterpret_cast<const uint16_t*>(row)[k]);
  const uint8_t* b = row + (k >> 5) * 34;
  return __half2float(*reinterpret_cast<const __half*>(b)) *
         float(reinterpret_cast<const int8_t*>(b + 2)[k & 31]);
}

__device__ __forceinline__ void epilogue(float v, int m, int n, int ldc,
                                         const float* bias, const float* ls,
                                         float* C, int epi) {
  if (bias)
    v += bias[n];
  if (epi == kEpiGelu)
    v = gelu_erf(v);
  float* c = C + size_t(m) * ldc + n;
  if (epi == kEpiResScale)
    *c += ls[n] * v;
  else
    *c = v;
}

// The value of lane ln of the other 16-lane half (ln ^ 16).
__device__ __forceinline__ float other_half(float v) {
  return __int_as_float(__builtin_amdgcn_permlanex16(
      0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
}
__device__ __forceinline__ uint32_t other_half(uint32_t v) {
  return uint32_t(__builtin_amdgcn_permlanex16(0, int(v), 0x76543210,
                                               0xfedcba98, false, false));
}

__device__ __forceinline__ v16h
ld16h(const _Float16* p) {  // p must be 16-byte aligned
  typedef _Float16 v8h __attribute__((ext_vector_type(8)));
  const v8h a = *reinterpret_cast<const v8h*>(p),
            b = *reinterpret_cast<const v8h*>(p + 8);
  v16h r;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    r[i] = a[i];
    r[8 + i] = b[i];
  }
  return r;
}

// ---- WMMA GEMM: C[m][n] = epi( sum_k A[m][k] * W[n][k] + bias[n] ), A and W
// f16 rows (row strides multiples of 8 halves), FP32 accumulation by
// v_wmma_f32_16x16x16_f16 in ascending K16 order from zero. Block tile BM x BN
// staged through LDS in K32 steps (next step prefetched into registers);
// waves form a WM x WN grid, each owning (BM / WM) x (BN / WN) outputs. N must
// be a multiple of BN, K of 32; A rows past M read row M - 1 and are not
// stored. Fragments: A lane -> row ln, 16 consecutive k; B lane -> weight row
// ln; C: acc[i] = (row 2i + hi, col ln).
enum FastEpi { kFStore = 0, kFResScale = 1, kFGeluH = 2, kFQkv = 3 };

struct FastEpiArgs {
  const float* bias;  // [N] or null
  const float* ls;    // kFResScale
  float* C;           // kFStore / kFResScale (f32, row stride ldc)
  int ldc;
  _Float16* H;  // kFGeluH: f16 rows (stride ldc); kFQkv: Q rows (stride 1024)
  // kFQkv: Q (scaled by qscale), K and V^T (per column, keys contiguous,
  // stride ldvt), 2-D RoPE on Q and K from rope[pos][16] = (cos, sin).
  _Float16 *K16, *VT;
  int ldvt, gw;
  float qscale;
  const float2* rope;
};

// kFQkv epilogue for one 16-row tile of the wave: columns 0..1023 -> Q,
// 1024..2047 -> K (bias, 2-D RoPE, f16; Q also scaled by qscale), 2048.. ->
// V^T (bias, f16). The section is uniform per wave.
template<int TN>
__device__ __forceinline__ void qkv_epilogue(const v8f (&acc)[TN], int mrow,
                                             int ncol, int M, int lane,
                                             const FastEpiArgs& e) {
  const int ln = lane & 15, hi = lane >> 4;
  const int sec = ncol >> 10;  // 0 Q, 1 K, 2 V
#pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    const int n = ncol + 16 * tn + ln, c = n & 1023;
    const float bn = e.bias[n];
    if (sec == 2) {
      // V^T: gather 8 consecutive rows per lane, one 16-byte store
      float v[8];
#pragma unroll
      for (int i = 0; i < 8; i++)
        v[i] = acc[tn][i] + bn;
      float r[4];
#pragma unroll
      for (int j = 0; j < 4; j++)
        r[j] = other_half(hi ? v[j] : v[4 + j]);
      typedef _Float16 v8h __attribute__((ext_vector_type(8)));
      v8h o;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        o[2 * j] = to_h(hi ? r[j] : v[j]);
        o[2 * j + 1] = to_h(hi ? v[4 + j] : r[j]);
      }
      if (mrow < M)  // rows up to ldvt (a multiple of 128) are masked
        *reinterpret_cast<v8h*>(e.VT + size_t(c) * e.ldvt + mrow + 8 * hi) = o;
      continue;
    }
    const int d = c & 63, half = d >> 5, i2 = (d & 31) >> 1;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int m = mrow + 2 * i + hi;
      const float x = acc[tn][i] + bn;
      const float px = __shfl_xor(x, 1);
      const int pw = m == 0 ? 0 : (m - 1) % e.gw + 1,
                ph = m == 0 ? 0 : (m - 1) / e.gw + 1;
      const float2 cs = e.rope[(half == 0 ? pw : ph) * 16 + i2];
      // k_rope's contraction: x0 cs - x1 sn = fma(x0, cs, -(x1 sn)),
      // x0 sn + x1 cs = fma(x1, cs, x0 sn)
      const float y = (d & 1) == 0 ? __builtin_fmaf(x, cs.x, -(px * cs.y))
                                   : __builtin_fmaf(x, cs.x, px * cs.y);
      if (m < M) {
        if (sec == 0)
          e.H[size_t(m) * kEmbd + c] = mul_to_h(y, e.qscale);
        else
          e.K16[size_t(m) * kEmbd + c] = to_h(y);
      }
    }
  }
}

template<int BM, int BN, int WM, int WN, int EPI>
__global__ void __launch_bounds__(WM * WN * 32)
    k_gemm_t(const _Float16* __restrict__ A, int lda,
             const _Float16* __restrict__ W, int ldw, int M, int K,
             FastEpiArgs ep) {
  constexpr int NT = WM * WN * 32, BK = 32, LDK = BK + 8, SEG = BK / 8;
  constexpr int TM = BM / WM / 16, TN = BN / WN / 16;
  constexpr int CA = BM * SEG / NT, CB = BN * SEG / NT;
  static_assert(CA * NT == BM * SEG && CB * NT == BN * SEG);
  __shared__ _Float16 sA[BM * LDK], sB[BN * LDK];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, ln = lane & 15;
  const int wm = wave / WN, wn = wave % WN;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  // this thread's 16-byte chunks: rows tid / SEG + c * (NT / SEG), segment
  // tid % SEG
  const int lr = tid / SEG, lseg = (tid % SEG) * 8;
  const _Float16* ga = A + size_t(min(m0 + lr, M - 1)) * lda + lseg;
  const _Float16* gb = W + size_t(n0 + lr) * ldw + lseg;
  constexpr int RS = NT / SEG;  // rows per chunk step
  size_t astep[CA];
#pragma unroll
  for (int c = 0; c < CA; c++)
    astep[c] = size_t(min(m0 + lr + c * RS, M - 1) - min(m0 + lr, M - 1)) * lda;
  _Float16* sa = sA + lr * LDK + lseg;
  _Float16* sb = sB + lr * LDK + lseg;
  v4u ra[CA], rb[CB];
#pragma unroll
  for (int c = 0; c < CA; c++)
    ra[c] = *reinterpret_cast<const v4u*>(ga + astep[c]);
#pragma unroll
  for (int c = 0; c < CB; c++)
    rb[c] = *reinterpret_cast<const v4u*>(gb + size_t(c * RS) * ldw);
  v8f acc[TM][TN];
#pragma unroll
  for (int t = 0; t < TM; t++)
#pragma unroll
    for (int r = 0; r < TN; r++)
      acc[t][r] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  const _Float16* fa = sA + ((wm * TM) * 16 + ln) * LDK;
  const _Float16* fb = sB + ((wn * TN) * 16 + ln) * LDK;
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int c = 0; c < CA; c++)
      *reinterpret_cast<v4u*>(sa + c * RS * LDK) = ra[c];
#pragma unroll
    for (int c = 0; c < CB; c++)
      *reinterpret_cast<v4u*>(sb + c * RS * LDK) = rb[c];
    __syncthreads();
    if (k0 + BK < K) {
#pragma unroll
      for (int c = 0; c < CA; c++)
        ra[c] = *reinterpret_cast<const v4u*>(ga + astep[c] + k0 + BK);
#pragma unroll
      for (int c = 0; c < CB; c++)
        rb[c] =
            *reinterpret_cast<const v4u*>(gb + size_t(c * RS) * ldw + k0 + BK);
    }
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      // A struct per fragment: clang 24 (ROCm 10.1) promotes a plain
      // two-fragment array to one <32 x half> vector and extracts each
      // fragment through i256, 16 extra VALU instructions per fragment.
      struct {
        v16h v;
      } b[TN];
#pragma unroll
      for (int r = 0; r < TN; r++)
        b[r].v = ld16h(fb + r * 16 * LDK + kk);
#pragma unroll
      for (int t = 0; t < TM; t++) {
        const v16h a = ld16h(fa + t * 16 * LDK + kk);
#pragma unroll
        for (int r = 0; r < TN; r++)
          acc[t][r] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[r].v, acc[t][r]);
      }
    }
    __syncthreads();
  }
  const int ncol = n0 + wn * TN * 16;
  if constexpr (EPI == kFQkv) {
#pragma unroll
    for (int t = 0; t < TM; t++)
      qkv_epilogue<TN>(acc[t], m0 + (wm * TM + t) * 16, ncol, M, lane, ep);
  } else {
    // per-column bias / layer scale, loaded once
    float bn[TN], sn[TN];
#pragma unroll
    for (int r = 0; r < TN; r++) {
      bn[r] = ep.bias ? ep.bias[ncol + 16 * r + ln] : 0.f;
      sn[r] = EPI == kFResScale ? ep.ls[ncol + 16 * r + ln] : 0.f;
    }
    const bool has_bias = ep.bias != nullptr;
    const int hi = lane >> 4;
#pragma unroll
    for (int t = 0; t < TM; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int m = m0 + (wm * TM + t) * 16 + 2 * i + hi;
        if (m >= M)
          continue;
#pragma unroll
        for (int r = 0; r < TN; r++) {
          const int n = ncol + 16 * r + ln;
          float v = acc[t][r][i];
          if (has_bias)
            v += bn[r];
          if constexpr (EPI == kFGeluH) {
            v = gelu_erf(v);
            ep.H[size_t(m) * ep.ldc + n] = to_h(v);
          } else if constexpr (EPI == kFResScale) {
            float* c = ep.C + size_t(m) * ep.ldc + n;
            *c += sn[r] * v;
          } else {
            ep.C[size_t(m) * ep.ldc + n] = v;
          }
        }
      }
  }
}

// ---- exact FP32 GEMM (64x64 tile, 16x16 threads, 4x4 each); any K, N, M
// ---------------------
template<int WT, int EPI>
__global__ void __launch_bounds__(256)
    k_gemm_ref(const float* __restrict__ A, int lda,
               const uint8_t* __restrict__ W, size_t rowbytes, int M, int N,
               int K, const float* __restrict__ bias,
               const float* __restrict__ ls, float* C, int ldc) {
  __shared__ float As[16][65], Ws[16][65];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int q = 0; q < 4; q++) {
      const int idx = threadIdx.x + 256 * q, r = idx >> 4, kk = idx & 15,
                k = k0 + kk;
      As[kk][r] = (m0 + r < M && k < K) ? A[size_t(m0 + r) * lda + k] : 0.f;
      Ws[kk][r] = (n0 + r < N && k < K)
                      ? load_w1<WT>(W + size_t(n0 + r) * rowbytes, k)
                      : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        a[i] = As[kk][ty + 16 * i];
        b[i] = Ws[kk][tx + 16 * i];
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int m = m0 + ty + 16 * i, n = n0 + tx + 16 * j;
      if (m < M && n < N)
        epilogue(acc[i][j], m, n, ldc, bias, ls, C, EPI);
    }
}

// ---- elementwise / small kernels
// ------------------------------------------------------------

// u8 image (w x h, RGB) -> patches [n_patch][1200] (c, y, x order = the
// patch_embd weight layout), with -1 padding (in 0..255 units, as mtmd does)
// and (v - mean*255) / (std*255) normalisation.
__global__ void k_patchify(const uint8_t* __restrict__ img, int w, int h,
                           int gw, int gh, const float* mean255,
                           const float* std255, float* __restrict__ P) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(gw) * gh * kPatchElems)
    return;
  const int p = int(idx / kPatchElems), e = int(idx % kPatchElems);
  const int c = e / (kPatch * kPatch), y = (e / kPatch) % kPatch,
            x = e % kPatch;
  const int X = (p % gw) * kPatch + x, Y = (p / gw) * kPatch + y;
  const float raw =
      (X < w && Y < h) ? float(img[(size_t(Y) * w + X) * 3 + c]) : -1.0f;
  P[idx] = (raw - mean255[c]) / std255[c];
}

// X[0] = class_embd + class_pos_embd; X[1 + p] = E[p] + bilinear(pos_table,
// 32x32 -> gh x gw, align_corners = false)[p]   (ggml_interpolate semantics,
// see ops.cpp upscale_f32).
__global__ void k_embed(const float* __restrict__ E,
                        const float* __restrict__ cls,
                        const float* __restrict__ cls_pos,
                        const float* __restrict__ pos, int gw, int gh,
                        float* __restrict__ X) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n_pos = gw * gh + 1;
  if (idx >= size_t(n_pos) * kEmbd)
    return;
  const int r = int(idx / kEmbd), e = int(idx % kEmbd);
  if (r == 0) {
    X[idx] = cls[e] + cls_pos[e];
    return;
  }
  const int p = r - 1, px = p % gw, py = p / gw;
  const float sfx = float(gw) / kPosGrid, sfy = float(gh) / kPosGrid;
  const float fx = (float(px) + 0.5f) / sfx - 0.5f,
              fy = (float(py) + 0.5f) / sfy - 0.5f;
  int x0 = int(floorf(fx)), x1 = x0 + 1, y0 = int(floorf(fy)), y1 = y0 + 1;
  x0 = max(0, min(x0, kPosGrid - 1));
  x1 = max(0, min(x1, kPosGrid - 1));
  y0 = max(0, min(y0, kPosGrid - 1));
  y1 = max(0, min(y1, kPosGrid - 1));
  const float dx = fmaxf(0.f, fminf(fx - float(x0), 1.f)),
              dy = fmaxf(0.f, fminf(fy - float(y0), 1.f));
  const float a = pos[size_t(y0 * kPosGrid + x0) * kEmbd + e],
              b = pos[size_t(y0 * kPosGrid + x1) * kEmbd + e];
  const float c = pos[size_t(y1 * kPosGrid + x0) * kEmbd + e],
              d = pos[size_t(y1 * kPosGrid + x1) * kEmbd + e];
  X[idx] =
      E[size_t(p) * kEmbd + e] + (a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) +
                                  c * (1 - dx) * dy + d * dx * dy);
}

// LayerNorm over rows of 1024 (in-place safe: every element is read before any
// is written).
template<typename T>
__global__ void __launch_bounds__(256)
    k_layernorm(const float* x, const float* __restrict__ w,
                const float* __restrict__ b, T* y, int ldy, float eps) {
  __shared__ float red[8];
  const size_t row = size_t(blockIdx.x) * kEmbd;
  float v[4], s = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = x[row + threadIdx.x + 256 * i];
    s += v[i];
  }
  auto bsum = [&](float t) {
    for (int o = 16; o; o >>= 1)
      t += __shfl_xor(t, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0)
      red[threadIdx.x >> 5] = t;
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < 8; i++)
      r += red[i];
    return r;
  };
  const float mean = bsum(s) / kEmbd;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float d = v[i] - mean;
    q += d * d;
  }
  const float rstd = rsqrtf(bsum(q) / kEmbd + eps);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int e = threadIdx.x + 256 * i;
    float r = (v[i] - mean) * rstd * w[e] + b[e];
    asm("" : "+v"(r));  // round to f32 first (no v_fma_mix for f16 rows)
    y[size_t(blockIdx.x) * ldy + e] = T(r);
  }
}

// 2-D axial RoPE on Q and K [n_pos][1024]: per head, dims 0..31 rotate with
// pos_w (adjacent pairs (2i, 2i+1), theta^(-2i/32)), dims 32..63 with pos_h.
// CLS is (0, 0); patch p is (x + 1, y + 1).
__global__ void k_rope(float* __restrict__ Q, float* __restrict__ K, int gw,
                       int n_pos, float theta) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size_t(n_pos) * kHeads * 32)
    return;
  const int pr = int(idx & 31), head = int((idx >> 5) % kHeads),
            t = int(idx / (32 * kHeads));
  const int half = pr >> 4, i = pr & 15;
  const int pw = t == 0 ? 0 : (t - 1) % gw + 1,
            ph = t == 0 ? 0 : (t - 1) / gw + 1;
  const float pos = float(half == 0 ? pw : ph);
  const float ang = pos * powf(theta, -float(2 * i) / 32.0f);
  float sn, cs;
  sincosf(ang, &sn, &cs);
  const size_t o = size_t(t) * kEmbd + head * kHeadDim + half * 32 + 2 * i;
  float x0 = Q[o], x1 = Q[o + 1];
  Q[o] = x0 * cs - x1 * sn;
  Q[o + 1] = x0 * sn + x1 * cs;
  x0 = K[o];
  x1 = K[o + 1];
  K[o] = x0 * cs - x1 * sn;
  K[o + 1] = x0 * sn + x1 * cs;
}

// Full (non-causal) attention, online softmax. Block = one head x 64 queries
// (thread = query); K/V tiles of 32 keys staged in LDS (broadcast reads).
__global__ void __launch_bounds__(64)
    k_attn(const float* __restrict__ Q, const float* __restrict__ K,
           const float* __restrict__ V, float* __restrict__ O, int n,
           float scale) {
  __shared__ float4 Ks[32][16], Vs[32][16];
  const int head = blockIdx.y, q = blockIdx.x * 64 + threadIdx.x;
  const bool live = q < n;
  float qv[kHeadDim], acc[kHeadDim];
  const float qs = scale * 1.4426950408889634f;
#pragma unroll
  for (int d = 0; d < kHeadDim; d++) {
    qv[d] = live ? Q[size_t(q) * kEmbd + head * kHeadDim + d] * qs : 0.f;
    acc[d] = 0.f;
  }
  float mrun = -INFINITY, lrun = 0.f;
  for (int k0 = 0; k0 < n; k0 += 32) {
    const int nk = min(32, n - k0);
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 16; i += 64) {
      const int j = i >> 4, c = i & 15;
      float4 kk = make_float4(0, 0, 0, 0), vv = kk;
      if (j < nk) {
        const size_t o = size_t(k0 + j) * kEmbd + head * kHeadDim + c * 4;
        kk = *reinterpret_cast<const float4*>(K + o);
        vv = *reinterpret_cast<const float4*>(V + o);
      }
      Ks[j][c] = kk;
      Vs[j][c] = vv;
    }
    __syncthreads();
    float s[32], tmax = -INFINITY;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      float a = 0.f;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        const float4 kk = Ks[j][c];
        a += qv[4 * c] * kk.x + qv[4 * c + 1] * kk.y + qv[4 * c + 2] * kk.z +
             qv[4 * c + 3] * kk.w;
      }
      a = j < nk ? a : -INFINITY;
      s[j] = a;
      tmax = fmaxf(tmax, a);
    }
    const float mnew = fmaxf(mrun, tmax), corr = exp2f(mrun - mnew);
    lrun *= corr;
#pragma unroll
    for (int d = 0; d < kHeadDim; d++)
      acc[d] *= corr;
#pragma unroll
    for (int j = 0; j < 32; j++) {
      const float p = exp2f(s[j] - mnew);
      lrun += p;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        const float4 vv = Vs[j][c];
        acc[4 * c] += p * vv.x;
        acc[4 * c + 1] += p * vv.y;
        acc[4 * c + 2] += p * vv.z;
        acc[4 * c + 3] += p * vv.w;
      }
    }
    mrun = mnew;
  }
  if (live) {
    const float inv = 1.f / lrun;
#pragma unroll
    for (int d = 0; d < kHeadDim; d++)
      O[size_t(q) * kEmbd + head * kHeadDim + d] = acc[d] * inv;
  }
}

// cos/sin of the 2-D RoPE angles: rope[pos][i] = (cos, sin)(pos *
// theta^(-2i/32)), the same expressions as k_rope.
__global__ void k_rope_table(float2* __restrict__ rope, int n_pos_max,
                             float theta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pos_max * 16)
    return;
  const int i = idx & 15;
  const float pos = float(idx >> 4);
  const float ang = pos * powf(theta, -float(2 * i) / 32.0f);
  float sn, cs;
  sincosf(ang, &sn, &cs);
  rope[idx] = make_float2(cs, sn);
}

// One 64-key step of the transposed flash attention below for the wave's 16
// queries: Ks = 64 keys x 64 dims (row stride 72), Vt = 64 dims x 64 keys (row
// stride LDV), lim = keys left from this step (masked when < 64).
template<int LDV>
__device__ __forceinline__ void attn_step(const v16h (&qf)[4],
                                          const _Float16* Ks,
                                          const _Float16* Vt, float& m,
                                          float (&l)[8], v8f (&oacc)[4],
                                          int lim, int ln, int hi) {
  constexpr int LD = 72;
  // S^T = K Q^T: s[j][i] = S[query ln][key 16 j + 2 i + hi], K16 order
  v8f s[4];
#pragma unroll
  for (int j = 0; j < 4; j++)
    s[j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int kk = 0; kk < 4; kk++)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const v16h kb = ld16h(&Ks[(j * 16 + ln) * LD + kk * 16]);
      s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kb, qf[kk], s[j]);
    }
  if (lim < 64) {
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int i = 0; i < 8; i++)
        if (j * 16 + 2 * i + hi >= lim)
          s[j][i] = -INFINITY;
  }
  float mx = -INFINITY, mn = INFINITY;
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      mx = fmaxf(mx, s[j][i]);
      mn = fminf(mn, s[j][i]);
    }
  const float mnew = fmaxf(m, fmaxf(mx, other_half(mx)));
  const float carg = m - mnew;
  const bool slow =
      __builtin_amdgcn_ballot_w32((mn - mnew < -126.f) |
                                  ((carg < -126.f) & (carg != -INFINITY))) != 0;
  float corr;
  auto step = [&](auto exp_fn) {
    corr = exp_fn(carg);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float ps = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float p = exp_fn(s[j][i] - mnew);
        ps += p;
        s[j][i] = p;
      }
      l[i] = l[i] * corr + ps;
    }
  };
  if (slow)
    step([](float x) { return exp2f(x); });
  else
    step([](float x) { return __builtin_amdgcn_exp2f(x); });
  m = mnew;
  // P^T as f16 pairs: ph[j][u] = keys 16 j + 2 (2u) + hi, 16 j + 2 (2u+1) + hi
  uint32_t ph[4][4];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const _Float16 lo = (_Float16)s[j][2 * u], h2 = (_Float16)s[j][2 * u + 1];
      ph[j][u] = uint32_t(__builtin_bit_cast(uint16_t, lo)) |
                 (uint32_t(__builtin_bit_cast(uint16_t, h2)) << 16);
    }
  // O *= corr; multiplying by exactly 1 changes nothing
  if (__builtin_amdgcn_ballot_w32(corr != 1.f) != 0) {
#pragma unroll
    for (int t = 0; t < 4; t++)
#pragma unroll
      for (int i = 0; i < 8; i++)
        oacc[t][i] *= corr;
  }
  // O^T += V^T P^T: the B fragment of query ln is keys kk*16 + 0..15, i.e. this
  // lane's keys 2i + hi interleaved with the other half's. Keys 4u..4u+3 =
  // (even.lo, odd.lo), (even.hi, odd.hi) from one v_perm_b32 each (own bytes
  // 4..7, other half's 0..3; the selector depends on hi).
  const uint32_t sel0 = hi ? 0x05040100u : 0x01000504u,
                 sel1 = hi ? 0x07060302u : 0x03020706u;
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    uint32_t w[8];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const uint32_t own = ph[kk][u], oth = other_half(own);
      w[2 * u] = __builtin_amdgcn_perm(own, oth, sel0);
      w[2 * u + 1] = __builtin_amdgcn_perm(own, oth, sel1);
    }
    const v16h pb = __builtin_bit_cast(v16h, w);
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const v16h va = ld16h(&Vt[(t * 16 + ln) * LDV + kk * 16]);
      oacc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, pb, oacc[t]);
    }
  }
}

// WMMA flash attention, transposed. Q is pre-scaled by (1/8) * log2(e); K is
// [key][1024] and V^T [1024][keys] (row stride ldvt, a multiple of 128), all
// f16. Block = 8 waves of 16 queries; 128-key tiles are staged in LDS (K as
// [key][d], V^T as [d][key]; the next tile is prefetched into registers) and
// consumed in two 64-key softmax steps. Each step computes S^T = K Q^T and
// O^T += V^T P^T: v_wmma_f32_16x16x16_f16 is exactly symmetric in its
// operands (wmma(B, A, C^T) == wmma(A, B, C)^T, checked bitwise on gfx1151),
// so every value equals the row-major kernel's (S = Q K^T in K16 order, P =
// f16(exp2(S - m)), O += P V per 64 keys), while a lane owns one query: lane
// (ln, hi) holds query ln and keys / dims 2i + hi of each 16-block. The
// softmax state keeps that kernel's arithmetic: one running max per query,
// the sum as 16 per-(key mod 16) partials (l = l * corr + the lane's 4 keys of
// the step) reduced by the same pairwise tree at the end, O *= corr before
// P V. exp2f lowers to v_exp_f32 with an x + 64 / ldexp(-64) fix-up for
// arguments below -126; when no argument in the wave is a finite value below
// -126 it is skipped (same value; masked keys give exp2(-inf) = 0 either
// way). Keys past n are zero in LDS and masked to -inf. O rows are f16
// (stride ldo).
__global__ void __launch_bounds__(256)
    k_attn_h(const _Float16* __restrict__ Q, const _Float16* __restrict__ K,
             const _Float16* __restrict__ VT, int ldvt,
             _Float16* __restrict__ O, int ldo, int n) {
  constexpr int LD = 72, NT = 256, KT = 128, LDV = KT + 8;
  __shared__ _Float16 Ks[KT * LD], Vt[64 * LDV];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, ln = lane & 15,
            hi = lane >> 4;
  const int head = blockIdx.y, q0 = blockIdx.x * 128 + wave * 16;
  v16h qf[4];
  {
    const _Float16* qp =
        Q + size_t(min(q0 + ln, n - 1)) * kEmbd + head * kHeadDim;
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
      qf[kk] = ld16h(qp + kk * 16);
  }
  v8f oacc[4];
#pragma unroll
  for (int t = 0; t < 4; t++)
    oacc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY, l[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    l[i] = 0.f;
  const _Float16* kbase = K + head * kHeadDim;
  const _Float16* vbase = VT + size_t(head) * kHeadDim * ldvt;
  // thread chunk c = tid + i NT: K row c / 8, halves 8 (c % 8); V^T row
  // c / 16, keys 8 (c % 16)
  constexpr int CH = KT * 8 / NT, VS = KT / 8;
  v4u kr[CH], vr[CH];
  auto gload = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CH; i++) {
      const int c = tid + i * NT;
      kr[i] = *reinterpret_cast<const v4u*>(
          kbase + size_t(min(k0 + (c >> 3), n - 1)) * kEmbd + (c & 7) * 8);
      vr[i] = *reinterpret_cast<const v4u*>(vbase + size_t(c / VS) * ldvt + k0 +
                                            (c % VS) * 8);
    }
  };
  gload(0);
  for (int k0 = 0; k0 < n; k0 += KT) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < CH; i++) {
      const int c = tid + i * NT;
      const int krow = c >> 3, kseg = (c & 7) * 8, vrow = c / VS,
                key = (c % VS) * 8;
      v4u kv = kr[i], vv = vr[i];
      if (k0 + KT > n) {  // keys past n are zero
        if (k0 + krow >= n)
          kv = (v4u){0, 0, 0, 0};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          if (k0 + key + 2 * e >= n)
            vv[e] &= 0xffff0000u;
          if (k0 + key + 2 * e + 1 >= n)
            vv[e] &= 0x0000ffffu;
        }
      }
      *reinterpret_cast<v4u*>(&Ks[krow * LD + kseg]) = kv;
      *reinterpret_cast<v4u*>(&Vt[vrow * LDV + key]) = vv;
    }
    __syncthreads();
    if (k0 + KT < n)
      gload(k0 + KT);
    attn_step<LDV>(qf, Ks, Vt, m, l, oacc, n - k0, ln, hi);
    if (k0 + 64 < n)
      attn_step<LDV>(qf, Ks + 64 * LD, Vt + 64, m, l, oacc, n - k0 - 64, ln,
                     hi);
  }
  // the row kernel's xor tree over the 16 partials (lanes 2i + hi):
  // ((l0 + l1) + (l2 + l3)) + ... in pairs
  float s1[8];
#pragma unroll
  for (int i = 0; i < 8; i++)
    s1[i] = l[i] + other_half(l[i]);
  const float s2[4] = {s1[0] + s1[1], s1[2] + s1[3], s1[4] + s1[5],
                       s1[6] + s1[7]};
  const float s3[2] = {s2[0] + s2[1], s2[2] + s2[3]};
  const float inv = 1.f / (s3[0] + s3[1]);
  const int q = q0 + ln;
  // O^T rows 2i + hi -> 8 consecutive dims per lane: the even half writes
  // dims 0..7 of each 16-block, the odd half 8..15
#pragma unroll
  for (int t = 0; t < 4; t++) {
    float x[8], r[4];
#pragma unroll
    for (int i = 0; i < 8; i++)
      x[i] = oacc[t][i] * inv;
#pragma unroll
    for (int u = 0; u < 4; u++)
      r[u] = other_half(hi ? x[u] : x[4 + u]);
    typedef _Float16 v8h __attribute__((ext_vector_type(8)));
    v8h o;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      o[2 * u] = to_h(hi ? r[u] : x[u]);
      o[2 * u + 1] = to_h(hi ? x[4 + u] : r[u]);
    }
    if (q < n)
      *reinterpret_cast<v8h*>(O + size_t(q) * ldo + head * kHeadDim + t * 16 +
                              8 * hi) = o;
  }
}

// X rows 1.. (patches, raster) -> M[token (i, j)][(dh, dw, e)] = patch (2i +
// dh, 2j + dw)
// (== ggml build_patch_merge_permute; tokens in raster order of the merged
// grid).
__global__ void k_merge(const float* __restrict__ X, int gw, int tw,
                        float* __restrict__ M, size_t total) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  const int tok = int(idx / (4 * kEmbd)), f = int(idx % (4 * kEmbd));
  const int dh = f / (2 * kEmbd), dw = (f / kEmbd) & 1, e = f % kEmbd;
  const int ti = tok / tw, tj = tok % tw;
  const int p = (2 * ti + dh) * gw + (2 * tj + dw);
  M[idx] = X[size_t(1 + p) * kEmbd + e];
}

// k_merge writing f16 rows (stride ldm) for the projector GEMM.
__global__ void k_merge_h(const float* __restrict__ X, int gw, int tw,
                          _Float16* __restrict__ M, int ldm, size_t total) {
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  const int tok = int(idx / (4 * kEmbd)), f = int(idx % (4 * kEmbd));
  const int dh = f / (2 * kEmbd), dw = (f / kEmbd) & 1, e = f % kEmbd;
  const int ti = tok / tw, tj = tok % tw;
  const int p = (2 * ti + dh) * gw + (2 * tj + dw);
  M[size_t(tok) * ldm + f] = (_Float16)X[size_t(1 + p) * kEmbd + e];
}

// Test aid: x -> Q8_0 -> x per 32-block (d = fp16(amax/127), q = round(x/d)),
// as ggml's quantize_row_q8_0.
__global__ void k_q8_roundtrip(const float* __restrict__ x,
                               float* __restrict__ y, size_t n_blocks) {
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= n_blocks)
    return;
  const float* p = x + b * 32;
  float amax = 0.f;
  for (int i = 0; i < 32; i++)
    amax = fmaxf(amax, fabsf(p[i]));
  const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
  const float dh = __half2float(__float2half(d));
  for (int i = 0; i < 32; i++)
    y[b * 32 + i] = roundf(p[i] * id) * dh;
}

// ---- host
// ---------------------------------------------------------------------------------------

#define HIPOK(call)                                               \
  do {                                                            \
    hipError_t e_ = (call);                                       \
    if (e_ != hipSuccess) {                                       \
      if (err)                                                    \
        *err = std::string(#call) + ": " + hipGetErrorString(e_); \
      return false;                                               \
    }                                                             \
  } while (0)

struct DevLinear {
  const uint8_t* w = nullptr;
  int type = 0, n = 0, k = 0;
  size_t rowbytes = 0;
};

struct DevLayer {
  const float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ls1, *ls2;
  const float *bq, *bk, *bv, *bo, *b_up, *b_down;
  const float* bqkv = nullptr;  // fast mode: bq | bk | bv
  DevLinear q, k, v, o, up, down;
  DevLinear qkv;  // fast mode: Wq | Wk | Wv (q, k, v stay empty)
};

size_t RowBytesOf(int type, size_t k) {
  switch (type) {
    case kT_F32:
      return k * 4;
    case kT_F16:
    case kT_BF16:
      return k * 2;
    case kT_Q8:
      return k / 32 * 34;
  }
  return 0;
}

float HalfToF(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0)
    v = std::ldexp(float(m), -24);
  else if (e == 31)
    v = m ? NAN : INFINITY;
  else
    v = std::ldexp(float(m | 1024), int(e) - 25);
  return s ? -v : v;
}

}  // namespace

struct VisionEncoder::Impl {
  Options opt;
  bool loaded = false;
  float eps = 1e-5f;
  float theta = 10000.f;
  int proj_dim = 0;
  std::vector<void*> allocs;
  const float *d_cls = nullptr, *d_cls_pos = nullptr, *d_pos = nullptr,
              *d_pre_w = nullptr, *d_pre_b = nullptr;
  const float *d_mean255 = nullptr, *d_std255 = nullptr;
  DevLinear patch, proj1, proj2;
  const float *b_proj1 = nullptr, *b_proj2 = nullptr;
  DevLayer layers[kLayers];

  // workspace (grown on demand)
  struct Buf {
    float* p = nullptr;
    size_t cap = 0;
  };
  // exact mode: f32 activations
  Buf X, XN, Q, K, V, O, H, OUT, QA;
  // fast mode: X f32 residual, f16 rows XH (LN output, attention output,
  // merged tokens), Q16/K16, VT (V^T), HH (FFN hidden, projector hidden), PF
  // (patches), EF (patch embeddings), RP (RoPE table)
  Buf XH, Q16, K16, VT, HH, PF, EF, RP;
  bool exact = false;  // exact reference mode: original weight types, FP32 FMA
                       // GEMMs, FP32 attention
  const char* tap_dir =
      nullptr;  // set by tests: dump X after each stage to this directory
  void Tap(hipStream_t s, const char* name, const float* dev, int rows,
           int cols) {
    if (!tap_dir)
      return;
    hipStreamSynchronize(s);
    std::vector<float> h(size_t(rows) * cols);
    hipMemcpy(h.data(), dev, h.size() * 4, hipMemcpyDeviceToHost);
    std::string path = std::string(tap_dir) + "/" + name + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
      return;
    int hdr[4] = {rows, cols, 0, 0};
    std::fwrite(hdr, 4, 4, f);
    std::fwrite(h.data(), 4, h.size(), f);
    std::fclose(f);
  }
  uint8_t* d_img = nullptr;
  size_t img_cap = 0;
  size_t ws_bytes = 0;

  ~Impl() {
    for (void* p : allocs)
      hipFree(p);
    for (Buf* b : {&X, &XN, &Q, &K, &V, &O, &H, &OUT, &QA, &XH, &Q16, &K16, &VT,
                   &HH, &PF, &EF, &RP})
      if (b->p)
        hipFree(b->p);
    if (d_img)
      hipFree(d_img);
  }

  bool Upload(const void* host, size_t bytes, const void** dev,
              std::string* err) {
    void* d = nullptr;
    HIPOK(hipMalloc(&d, std::max<size_t>(bytes, 16)));
    allocs.push_back(d);
    HIPOK(hipMemcpy(d, host, bytes, hipMemcpyHostToDevice));
    *dev = d;
    return true;
  }
  // small tensor -> F32 device array (F32/F16/BF16 sources)
  bool UploadF32(const MmprojTensor* t, uint64_t expect, const float** dev,
                 std::string* err) {
    if (!t) {
      if (err)
        *err = "missing tensor";
      return false;
    }
    if (t->nelem() != expect) {
      if (err)
        *err = "bad shape: " + t->name;
      return false;
    }
    std::vector<float> f(expect);
    for (uint64_t i = 0; i < expect; i++) {
      if (t->type == kT_F32)
        std::memcpy(&f[i], t->data + i * 4, 4);
      else if (t->type == kT_F16) {
        uint16_t h;
        std::memcpy(&h, t->data + i * 2, 2);
        f[i] = HalfToF(h);
      } else if (t->type == kT_BF16) {
        uint16_t h;
        std::memcpy(&h, t->data + i * 2, 2);
        uint32_t u = uint32_t(h) << 16;
        std::memcpy(&f[i], &u, 4);
      } else {
        if (err)
          *err = "unsupported small tensor type: " + t->name;
        return false;
      }
    }
    return Upload(f.data(), expect * 4, reinterpret_cast<const void**>(dev),
                  err);
  }
  bool UploadLinear(const MmprojTensor* t, int n, int k, DevLinear* out,
                    std::string* err) {
    if (!t) {
      if (err)
        *err = "missing linear tensor";
      return false;
    }
    if (t->ne.size() != 2 || t->ne[0] != uint64_t(k) ||
        t->ne[1] != uint64_t(n)) {
      if (err)
        *err = "bad shape: " + t->name;
      return false;
    }
    if (!RowBytesOf(t->type, k) || (t->type == kT_Q8 && k % 32) || k % 16) {
      if (err)
        *err = "unsupported type/shape: " + t->name;
      return false;
    }
    const void* d;
    if (exact) {
      if (!Upload(t->data, t->nbytes, &d, err))
        return false;
      out->w = static_cast<const uint8_t*>(d);
      out->type = t->type;
      out->n = n;
      out->k = k;
      out->rowbytes = RowBytesOf(t->type, k);
      return true;
    }
    // fast mode: dequantise once to f16 (Q8_0 -> f16 rounds each d*q to 11
    // bits)
    const int ldh = k + 16;
    const size_t count = size_t(n) * ldh;
    const auto h = std::make_unique_for_overwrite<_Float16[]>(count);
    DequantF16(t, n, k, h.get(), ldh);
    if (!Upload(h.get(), count * 2, &d, err))
      return false;
    out->w = static_cast<const uint8_t*>(d);
    out->type = kT_F16;
    out->n = n;
    out->k = k;
    out->rowbytes = size_t(ldh) * 2;
    return true;
  }
  // Rows of t -> f16 rows of h (stride ldh; the ldh - k pad halves are
  // zeroed), split across host threads.
  static void DequantF16(const MmprojTensor* t, int n, int k, _Float16* h,
                         int ldh) {
    auto rows = [&](int r0, int r1) {
      for (int r = r0; r < r1; r++) {
        DequantRow(t->type, t->data + size_t(r) * RowBytesOf(t->type, k), k,
                   h + size_t(r) * ldh);
        std::fill(h + size_t(r) * ldh + k, h + size_t(r + 1) * ldh,
                  (_Float16)0);
      }
    };
    const int nt =
        int(std::clamp(std::thread::hardware_concurrency(), 1u, 16u));
    const int per = (n + nt - 1) / nt;
    std::vector<std::thread> pool;
    for (int i = 1; i < nt && i * per < n; i++)
      pool.emplace_back(rows, i * per, std::min(n, (i + 1) * per));
    rows(0, std::min(n, per));
    for (std::thread& th : pool)
      th.join();
  }
  static void DequantRow(int type, const uint8_t* row, int k, _Float16* out) {
    for (int c = 0; c < k; c++) {
      float v;
      if (type == kT_F32)
        std::memcpy(&v, row + size_t(c) * 4, 4);
      else if (type == kT_F16) {
        uint16_t u;
        std::memcpy(&u, row + size_t(c) * 2, 2);
        v = HalfToF(u);
      } else if (type == kT_BF16) {
        uint16_t u;
        std::memcpy(&u, row + size_t(c) * 2, 2);
        uint32_t x = uint32_t(u) << 16;
        std::memcpy(&v, &x, 4);
      } else {
        const uint8_t* b = row + size_t(c / 32) * 34;
        uint16_t u;
        std::memcpy(&u, b, 2);
        v = HalfToF(u) * float(int8_t(b[2 + c % 32]));
      }
      out[c] = (_Float16)v;
    }
  }
  // fast mode: Wq | Wk | Wv as one f16 [3 * 1024][1024 + 16] linear
  bool UploadQkv(const MmprojTensor* const (&t)[3], DevLinear* out,
                 std::string* err) {
    constexpr int ldh = kEmbd + 16;
    const size_t count = size_t(3) * kEmbd * ldh;
    const auto h = std::make_unique_for_overwrite<_Float16[]>(count);
    for (int i = 0; i < 3; i++) {
      if (!t[i] || t[i]->ne.size() != 2 || t[i]->ne[0] != uint64_t(kEmbd) ||
          t[i]->ne[1] != uint64_t(kEmbd) || !RowBytesOf(t[i]->type, kEmbd)) {
        if (err)
          *err = "bad or missing attention q/k/v weight";
        return false;
      }
      DequantF16(t[i], kEmbd, kEmbd, h.get() + size_t(i) * kEmbd * ldh, ldh);
    }
    const void* d;
    if (!Upload(h.get(), count * 2, &d, err))
      return false;
    *out = {static_cast<const uint8_t*>(d), kT_F16, 3 * kEmbd, kEmbd,
            size_t(ldh) * 2};
    return true;
  }
  bool Grow(Buf& b, size_t n, std::string* err) {
    if (b.cap >= n)
      return true;
    if (b.p) {
      hipFree(b.p);
      ws_bytes -= b.cap * 4;
      b.p = nullptr;
      b.cap = 0;
    }
    HIPOK(hipMalloc(reinterpret_cast<void**>(&b.p), n * 4));
    b.cap = n;
    ws_bytes += n * 4;
    return true;
  }

  template<int WT, int EPI>
  void RefT(hipStream_t s, const DevLinear& L, const float* A, int lda, int M,
            float* C, int ldc, const float* bias, const float* ls) {
    dim3 g((L.n + 63) / 64, (M + 63) / 64);
    k_gemm_ref<WT, EPI><<<g, 256, 0, s>>>(A, lda, L.w, L.rowbytes, M, L.n, L.k,
                                          bias, ls, C, ldc);
  }
  template<int WT>
  void RefW(hipStream_t s, const DevLinear& L, const float* A, int lda, int M,
            float* C, int ldc, const float* bias, const float* ls, int epi) {
    if (epi == kEpiStore)
      RefT<WT, kEpiStore>(s, L, A, lda, M, C, ldc, bias, ls);
    else if (epi == kEpiGelu)
      RefT<WT, kEpiGelu>(s, L, A, lda, M, C, ldc, bias, ls);
    else
      RefT<WT, kEpiResScale>(s, L, A, lda, M, C, ldc, bias, ls);
  }

  // Exact mode: C = epi(A W^T + bias) with the file's weight types, FP32.
  void Gemm(hipStream_t s, const DevLinear& L, const float* A, int lda, int M,
            float* C, int ldc, const float* bias, const float* ls, int epi) {
    if (opt.emulate_q8_acts && L.type == kT_Q8 && lda == L.k) {
      const size_t nb = size_t(M) * L.k / 32;
      k_q8_roundtrip<<<unsigned((nb + 255) / 256), 256, 0, s>>>(A, QA.p, nb);
      A = QA.p;
    }
    switch (L.type) {
      case kT_F32:
        RefW<kT_F32>(s, L, A, lda, M, C, ldc, bias, ls, epi);
        break;
      case kT_F16:
        RefW<kT_F16>(s, L, A, lda, M, C, ldc, bias, ls, epi);
        break;
      case kT_BF16:
        RefW<kT_BF16>(s, L, A, lda, M, C, ldc, bias, ls, epi);
        break;
      default:
        RefW<kT_Q8>(s, L, A, lda, M, C, ldc, bias, ls, epi);
        break;
    }
  }

  // Fast mode: f16 A rows (stride lda) times the f16 linear L.
  template<int EPI>
  void FastGemm(hipStream_t s, const DevLinear& L, const _Float16* A, int lda,
                int M, const FastEpiArgs& ep) {
    const _Float16* W = reinterpret_cast<const _Float16*>(L.w);
    const int ldw = int(L.rowbytes / 2);
    k_gemm_t<128, 128, 2, 4, EPI>
        <<<dim3(L.n / 128, (M + 127) / 128), 256, 0, s>>>(A, lda, W, ldw, M,
                                                          L.k, ep);
  }
  bool EncodeFast(hipStream_t s, int iw, int ih, const ImagePlan& plan,
                  std::string* err);
};

VisionEncoder::VisionEncoder() : impl_(new Impl) {}
VisionEncoder::~VisionEncoder() = default;
int VisionEncoder::embedding_dim() const {
  return impl_->proj_dim;
}
size_t VisionEncoder::device_bytes() const {
  size_t b = impl_->ws_bytes;
  return b;  // weights tracked separately by the caller via hipMemGetInfo if
             // needed
}

bool VisionEncoder::Load(const std::string& path, std::string* err,
                         bool exact_reference) {
  Impl& m = *impl_;
  m.exact = exact_reference;
  MmprojGguf g;
  if (!g.Open(path, err))
    return false;
  auto fail = [&](const std::string& s) {
    if (err)
      *err = s;
    return false;
  };
  std::string proj;
  if (!g.GetStr("clip.vision.projector_type", &proj) || proj != "inkling2v")
    return fail("projector_type is not inkling2v");
  uint32_t u;
  auto need = [&](const char* key, uint32_t want) {
    return g.GetU32(key, &u) && u == want;
  };
  if (!need("clip.vision.block_count", kLayers) ||
      !need("clip.vision.embedding_length", kEmbd) ||
      !need("clip.vision.attention.head_count", kHeads) ||
      !need("clip.vision.feed_forward_length", kFfn) ||
      !need("clip.vision.patch_size", kPatch) ||
      !need("clip.vision.spatial_merge_size", kMerge))
    return fail(
        "unexpected clip.vision hyper-parameters (want 23 layers, 1024, 16 "
        "heads, 4096, patch 20, merge 2)");
  if (!g.GetU32("clip.vision.projection_dim", &u))
    return fail("missing projection_dim");
  m.proj_dim = int(u);
  if (!exact_reference && (m.proj_dim <= 0 || m.proj_dim % 128))
    return fail("projection_dim must be a positive multiple of 128");
  if (!g.GetF32("clip.vision.attention.layer_norm_epsilon", &m.eps))
    m.eps = 1e-5f;
  std::vector<float> mean, stdv;
  if (!g.GetF32Array("clip.vision.image_mean", &mean) ||
      !g.GetF32Array("clip.vision.image_std", &stdv) || mean.size() != 3 ||
      stdv.size() != 3)
    return fail("missing image_mean/image_std");
  for (int c = 0; c < 3; c++) {
    mean[c] *= 255.f;
    stdv[c] *= 255.f;
  }
  if (!m.Upload(mean.data(), 12, reinterpret_cast<const void**>(&m.d_mean255),
                err))
    return false;
  if (!m.Upload(stdv.data(), 12, reinterpret_cast<const void**>(&m.d_std255),
                err))
    return false;

  if (!m.UploadF32(g.Find("v.class_embd"), kEmbd, &m.d_cls, err))
    return false;
  if (!m.UploadF32(g.Find("v.class_position_embd.weight"), kEmbd, &m.d_cls_pos,
                   err))
    return false;
  if (!m.UploadF32(g.Find("v.position_embd.weight"),
                   uint64_t(kPosGrid) * kPosGrid * kEmbd, &m.d_pos, err))
    return false;
  if (!m.UploadF32(g.Find("v.pre_ln.weight"), kEmbd, &m.d_pre_w, err))
    return false;
  if (!m.UploadF32(g.Find("v.pre_ln.bias"), kEmbd, &m.d_pre_b, err))
    return false;
  {  // patch embedding: kept F32 (K = 1200 is not a multiple of 32)
    const MmprojTensor* t = g.Find("v.patch_embd.weight");
    if (!t || t->nelem() != uint64_t(kEmbd) * kPatchElems)
      return fail("bad v.patch_embd.weight");
    std::vector<float> f(t->nelem());
    for (uint64_t i = 0; i < f.size(); i++) {
      if (t->type == kT_F32)
        std::memcpy(&f[i], t->data + i * 4, 4);
      else if (t->type == kT_F16) {
        uint16_t h;
        std::memcpy(&h, t->data + i * 2, 2);
        f[i] = HalfToF(h);
      } else if (t->type == kT_BF16) {
        uint16_t h;
        std::memcpy(&h, t->data + i * 2, 2);
        uint32_t x = uint32_t(h) << 16;
        std::memcpy(&f[i], &x, 4);
      } else
        return fail("unsupported patch_embd type");
    }
    const void* d;
    if (!m.Upload(f.data(), f.size() * 4, &d, err))
      return false;
    m.patch = {static_cast<const uint8_t*>(d), kT_F32, kEmbd, kPatchElems,
               size_t(kPatchElems) * 4};
  }
  for (int i = 0; i < kLayers; i++) {
    DevLayer& L = m.layers[i];
    const std::string p = "v.blk." + std::to_string(i) + ".";
    auto f1 = [&](const char* n, const float** d, uint64_t cnt) {
      return m.UploadF32(g.Find(p + n), cnt, d, err);
    };
    if (!f1("ln1.weight", &L.ln1_w, kEmbd) ||
        !f1("ln1.bias", &L.ln1_b, kEmbd) ||
        !f1("ln2.weight", &L.ln2_w, kEmbd) ||
        !f1("ln2.bias", &L.ln2_b, kEmbd) || !f1("ls1.weight", &L.ls1, kEmbd) ||
        !f1("ls2.weight", &L.ls2, kEmbd) || !f1("attn_q.bias", &L.bq, kEmbd) ||
        !f1("attn_k.bias", &L.bk, kEmbd) || !f1("attn_v.bias", &L.bv, kEmbd) ||
        !f1("attn_out.bias", &L.bo, kEmbd) ||
        !f1("ffn_up.bias", &L.b_up, kFfn) ||
        !f1("ffn_down.bias", &L.b_down, kEmbd))
      return false;
    if (m.exact) {
      if (!m.UploadLinear(g.Find(p + "attn_q.weight"), kEmbd, kEmbd, &L.q,
                          err) ||
          !m.UploadLinear(g.Find(p + "attn_k.weight"), kEmbd, kEmbd, &L.k,
                          err) ||
          !m.UploadLinear(g.Find(p + "attn_v.weight"), kEmbd, kEmbd, &L.v, err))
        return false;
    } else {
      const MmprojTensor* const qkv[3] = {g.Find(p + "attn_q.weight"),
                                          g.Find(p + "attn_k.weight"),
                                          g.Find(p + "attn_v.weight")};
      if (!m.UploadQkv(qkv, &L.qkv, err))
        return false;
      void* d = nullptr;
      if (hipMalloc(&d, 3 * kEmbd * sizeof(float)) != hipSuccess)
        return fail("hipMalloc failed for the q/k/v bias");
      m.allocs.push_back(d);
      float* b = static_cast<float*>(d);
      if (hipMemcpy(b, L.bq, kEmbd * 4, hipMemcpyDeviceToDevice) !=
              hipSuccess ||
          hipMemcpy(b + kEmbd, L.bk, kEmbd * 4, hipMemcpyDeviceToDevice) !=
              hipSuccess ||
          hipMemcpy(b + 2 * kEmbd, L.bv, kEmbd * 4, hipMemcpyDeviceToDevice) !=
              hipSuccess)
        return fail("q/k/v bias copy failed");
      L.bqkv = b;
    }
    if (!m.UploadLinear(g.Find(p + "attn_out.weight"), kEmbd, kEmbd, &L.o,
                        err) ||
        !m.UploadLinear(g.Find(p + "ffn_up.weight"), kFfn, kEmbd, &L.up, err) ||
        !m.UploadLinear(g.Find(p + "ffn_down.weight"), kEmbd, kFfn, &L.down,
                        err))
      return false;
  }
  const int merged = 4 * kEmbd;
  if (!m.UploadLinear(g.Find("mm.1.weight"), merged, merged, &m.proj1, err) ||
      !m.UploadLinear(g.Find("mm.2.weight"), m.proj_dim, merged, &m.proj2, err))
    return false;
  // optional projector biases (absent in the shipped file)
  if (g.Find("mm.1.bias") &&
      !m.UploadF32(g.Find("mm.1.bias"), merged, &m.b_proj1, err))
    return false;
  if (g.Find("mm.2.bias") &&
      !m.UploadF32(g.Find("mm.2.bias"), uint64_t(m.proj_dim), &m.b_proj2, err))
    return false;
  m.loaded = true;
  return true;
}

void VisionEncoder::ReleaseWorkspace() {
  Impl& m = *impl_;
  for (Impl::Buf* b :
       {&m.X, &m.XN, &m.Q, &m.K, &m.V, &m.O, &m.H, &m.OUT, &m.QA, &m.XH, &m.Q16,
        &m.K16, &m.VT, &m.HH, &m.PF, &m.EF, &m.RP}) {
    if (b->p)
      hipFree(b->p);
    b->p = nullptr;
    b->cap = 0;
  }
  if (m.d_img)
    hipFree(m.d_img);
  m.d_img = nullptr;
  m.img_cap = 0;
  m.ws_bytes = 0;
}

bool VisionEncoder::Encode(const uint8_t* rgb, int w, int h, hipStream_t s,
                           Result* out, std::string* err) {
  Impl& m = *impl_;
  if (!m.loaded) {
    if (err)
      *err = "encoder not loaded";
    return false;
  }
  if (w <= 0 || h <= 0) {
    if (err)
      *err = "empty image";
    return false;
  }
  const ImagePlan plan = PlanImage(w, h);
  ImageU8 scratch;
  const uint8_t* px = rgb;
  int iw = w, ih = h;
  if (plan.downscaled) {
    ImageU8 src;
    src.w = w;
    src.h = h;
    src.rgb.assign(rgb, rgb + size_t(w) * h * 3);
    const ImageU8& r = ApplyPlan(src, plan, &scratch);
    iw = r.w;
    ih = r.h;
    px = r.rgb.data();
  }
  const int gw = plan.grid_w, gh = plan.grid_h, np = gw * gh, n_pos = np + 1,
            tw = plan.tok_w, nt = plan.n_tokens;
  const size_t img_bytes = size_t(iw) * ih * 3;
  if (m.img_cap < img_bytes) {
    if (m.d_img)
      hipFree(m.d_img);
    m.d_img = nullptr;
    HIPOK(hipMalloc(reinterpret_cast<void**>(&m.d_img), img_bytes));
    m.img_cap = img_bytes;
  }
  const size_t sz_pos = size_t(n_pos) * kEmbd;
  if (m.exact) {
    if (!m.Grow(m.X, sz_pos, err) || !m.Grow(m.XN, sz_pos, err) ||
        !m.Grow(m.Q, sz_pos, err) || !m.Grow(m.K, sz_pos, err) ||
        !m.Grow(m.V, sz_pos, err) || !m.Grow(m.O, sz_pos, err) ||
        !m.Grow(m.H, std::max(size_t(n_pos) * kFfn, size_t(np) * kPatchElems),
                err) ||
        !m.Grow(m.OUT, size_t(nt) * m.proj_dim, err) ||
        (m.opt.emulate_q8_acts && !m.Grow(m.QA, size_t(n_pos) * kFfn, err)))
      return false;
  }
  // The upload is synchronous w.r.t. the host buffer so the caller may reuse
  // `rgb` right away.
  HIPOK(hipMemcpyAsync(m.d_img, px, img_bytes, hipMemcpyHostToDevice, s));
  HIPOK(hipStreamSynchronize(s));
  if (!m.exact) {
    if (!m.EncodeFast(s, iw, ih, plan, err))
      return false;
    out->embd = m.OUT.p;
    out->n_tokens = nt;
    out->tok_w = plan.tok_w;
    out->tok_h = plan.tok_h;
    out->dim = m.proj_dim;
    out->plan = plan;
    return true;
  }

  auto blocks = [](size_t n, int t) { return unsigned((n + t - 1) / t); };
  k_patchify<<<blocks(size_t(np) * kPatchElems, 256), 256, 0, s>>>(
      m.d_img, iw, ih, gw, gh, m.d_mean255, m.d_std255, m.H.p);
  m.Gemm(s, m.patch, m.H.p, kPatchElems, np, m.Q.p, kEmbd, nullptr, nullptr,
         kEpiStore);  // E -> Q
  k_embed<<<blocks(sz_pos, 256), 256, 0, s>>>(m.Q.p, m.d_cls, m.d_cls_pos,
                                              m.d_pos, gw, gh, m.X.p);
  m.Tap(s, "embeddings", m.X.p, n_pos, kEmbd);
  k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, m.d_pre_w, m.d_pre_b, m.X.p, kEmbd,
                                    m.eps);
  m.Tap(s, "pre_ln", m.X.p, n_pos, kEmbd);

  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  for (int il = 0; il < kLayers; il++) {
    const DevLayer& L = m.layers[il];
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln1_w, L.ln1_b, m.XN.p, kEmbd,
                                      m.eps);
    m.Gemm(s, L.q, m.XN.p, kEmbd, n_pos, m.Q.p, kEmbd, L.bq, nullptr,
           kEpiStore);
    m.Gemm(s, L.k, m.XN.p, kEmbd, n_pos, m.K.p, kEmbd, L.bk, nullptr,
           kEpiStore);
    m.Gemm(s, L.v, m.XN.p, kEmbd, n_pos, m.V.p, kEmbd, L.bv, nullptr,
           kEpiStore);
    k_rope<<<blocks(size_t(n_pos) * kHeads * 32, 256), 256, 0, s>>>(
        m.Q.p, m.K.p, gw, n_pos, m.theta);
    k_attn<<<dim3((n_pos + 63) / 64, kHeads), 64, 0, s>>>(m.Q.p, m.K.p, m.V.p,
                                                          m.O.p, n_pos, scale);
    if (m.tap_dir && il == 0)
      m.Tap(s, "attnO-0", m.O.p, n_pos, kEmbd);
    m.Gemm(s, L.o, m.O.p, kEmbd, n_pos, m.X.p, kEmbd, L.bo, L.ls1,
           kEpiResScale);
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln2_w, L.ln2_b, m.XN.p, kEmbd,
                                      m.eps);
    m.Gemm(s, L.up, m.XN.p, kEmbd, n_pos, m.H.p, kFfn, L.b_up, nullptr,
           kEpiGelu);
    m.Gemm(s, L.down, m.H.p, kFfn, n_pos, m.X.p, kEmbd, L.b_down, L.ls2,
           kEpiResScale);
    if (m.tap_dir)
      m.Tap(s, ("layer_out-" + std::to_string(il)).c_str(), m.X.p, n_pos,
            kEmbd);
  }
  const size_t total = size_t(nt) * 4 * kEmbd;
  k_merge<<<blocks(total, 256), 256, 0, s>>>(m.X.p, gw, tw, m.Q.p, total);
  m.Gemm(s, m.proj1, m.Q.p, 4 * kEmbd, nt, m.K.p, 4 * kEmbd, m.b_proj1, nullptr,
         kEpiGelu);
  m.Gemm(s, m.proj2, m.K.p, 4 * kEmbd, nt, m.OUT.p, m.proj_dim, m.b_proj2,
         nullptr, kEpiStore);
  HIPOK(hipGetLastError());
  m.Tap(s, "projected", m.OUT.p, nt, m.proj_dim);

  out->embd = m.OUT.p;
  out->n_tokens = nt;
  out->tok_w = plan.tok_w;
  out->tok_h = plan.tok_h;
  out->dim = m.proj_dim;
  out->plan = plan;
  return true;
}

// Fast path. Per layer: LN1 -> f16, one Q|K|V GEMM (RoPE, Q scale, f16 and
// V^T in the epilogue), attention (f16 out), O GEMM (+= ls1 * ...), LN2 ->
// f16, up GEMM (GELU -> f16), down GEMM (+= ls2 * ...). Every output is the
// same WMMA sequence on the same f16 operands as the unfused kernels.
bool VisionEncoder::Impl::EncodeFast(hipStream_t s, int iw, int ih,
                                     const ImagePlan& plan, std::string* err) {
  Impl& m = *this;
  const int gw = plan.grid_w, gh = plan.grid_h, np = gw * gh, n_pos = np + 1,
            tw = plan.tok_w, nt = plan.n_tokens;
  constexpr int ldx = kEmbd + 16, ldf = kFfn + 16, ldm = 4 * kEmbd + 16;
  const int ldvt = (n_pos + 127) / 128 * 128;
  const int pos_max = std::max(gw, gh) + 1;
  auto halves = [](size_t n) { return (n + 1) / 2 + 8; };
  if (!m.Grow(m.X, size_t(n_pos) * kEmbd, err) ||
      !m.Grow(m.XH, halves(std::max(size_t(n_pos) * ldx, size_t(nt) * ldm)),
              err) ||
      !m.Grow(m.Q16, halves(size_t(n_pos) * kEmbd), err) ||
      !m.Grow(m.K16, halves(size_t(n_pos) * kEmbd), err) ||
      !m.Grow(m.VT, halves(size_t(kEmbd) * ldvt), err) ||
      !m.Grow(m.HH, halves(std::max(size_t(n_pos) * ldf, size_t(nt) * ldm)),
              err) ||
      !m.Grow(m.PF, size_t(np) * kPatchElems, err) ||
      !m.Grow(m.EF, size_t(np) * kEmbd, err) ||
      !m.Grow(m.RP, size_t(pos_max) * 16 * 2, err) ||
      !m.Grow(m.OUT, size_t(nt) * m.proj_dim, err))
    return false;
  _Float16* XH = reinterpret_cast<_Float16*>(m.XH.p);
  _Float16* HH = reinterpret_cast<_Float16*>(m.HH.p);
  _Float16* Q16 = reinterpret_cast<_Float16*>(m.Q16.p);
  _Float16* K16 = reinterpret_cast<_Float16*>(m.K16.p);
  _Float16* VT = reinterpret_cast<_Float16*>(m.VT.p);
  float2* rope = reinterpret_cast<float2*>(m.RP.p);
  auto blocks = [](size_t n, int t) { return unsigned((n + t - 1) / t); };

  k_patchify<<<blocks(size_t(np) * kPatchElems, 256), 256, 0, s>>>(
      m.d_img, iw, ih, gw, gh, m.d_mean255, m.d_std255, m.PF.p);
  RefW<kT_F32>(s, m.patch, m.PF.p, kPatchElems, np, m.EF.p, kEmbd, nullptr,
               nullptr, kEpiStore);
  k_embed<<<blocks(size_t(n_pos) * kEmbd, 256), 256, 0, s>>>(
      m.EF.p, m.d_cls, m.d_cls_pos, m.d_pos, gw, gh, m.X.p);
  k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, m.d_pre_w, m.d_pre_b, m.X.p, kEmbd,
                                    m.eps);
  k_rope_table<<<blocks(size_t(pos_max) * 16, 256), 256, 0, s>>>(rope, pos_max,
                                                                 m.theta);

  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  FastEpiArgs qkv{};
  qkv.H = Q16;
  qkv.K16 = K16;
  qkv.VT = VT;
  qkv.ldvt = ldvt;
  qkv.gw = gw;
  qkv.qscale = scale * 1.4426950408889634f;
  qkv.rope = rope;
  for (int il = 0; il < kLayers; il++) {
    const DevLayer& L = m.layers[il];
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln1_w, L.ln1_b, XH, ldx, m.eps);
    qkv.bias = L.bqkv;
    m.FastGemm<kFQkv>(s, L.qkv, XH, ldx, n_pos, qkv);
    k_attn_h<<<dim3((n_pos + 127) / 128, kHeads), 256, 0, s>>>(
        Q16, K16, VT, ldvt, XH, ldx, n_pos);
    FastEpiArgs o{};
    o.bias = L.bo;
    o.ls = L.ls1;
    o.C = m.X.p;
    o.ldc = kEmbd;
    m.FastGemm<kFResScale>(s, L.o, XH, ldx, n_pos, o);
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln2_w, L.ln2_b, XH, ldx, m.eps);
    FastEpiArgs up{};
    up.bias = L.b_up;
    up.H = HH;
    up.ldc = ldf;
    m.FastGemm<kFGeluH>(s, L.up, XH, ldx, n_pos, up);
    FastEpiArgs down{};
    down.bias = L.b_down;
    down.ls = L.ls2;
    down.C = m.X.p;
    down.ldc = kEmbd;
    m.FastGemm<kFResScale>(s, L.down, HH, ldf, n_pos, down);
  }
  const size_t total = size_t(nt) * 4 * kEmbd;
  k_merge_h<<<blocks(total, 256), 256, 0, s>>>(m.X.p, gw, tw, XH, ldm, total);
  FastEpiArgs p1{};
  p1.bias = m.b_proj1;
  p1.H = HH;
  p1.ldc = ldm;
  m.FastGemm<kFGeluH>(s, m.proj1, XH, ldm, nt, p1);
  FastEpiArgs p2{};
  p2.bias = m.b_proj2;
  p2.C = m.OUT.p;
  p2.ldc = m.proj_dim;
  m.FastGemm<kFStore>(s, m.proj2, HH, ldm, nt, p2);
  HIPOK(hipGetLastError());
  return true;
}

}  // namespace gufo::models::inkling2::vision
