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

enum Epi { kEpiStore = 0, kEpiGelu = 1, kEpiResScale = 2 };

__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
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

// ---- WMMA GEMM: C[m][n] = epi( sum_k A[m][k] * W[n][k] + bias[n] ), A and W
// f16 -------------------- Block = 4 waves; a wave owns 64 tokens x 64 outputs
// (16 accumulators) and reads its operand fragments straight from global/L2 (no
// LDS). K % 16 == 0, N % 256 not required (waves past N exit). A fragment: lane
// -> row ln, 16 consecutive k (lanes hi duplicate); B fragment: lane -> weight
// row ln; C: acc[i] = (row 2i + hi, col ln). Rows of K f32 (stride lda) -> f16
// rows with stride ldy (padded: pow-2 row strides camp on L2 channels).
__global__ void k_rows_to_f16(const float* __restrict__ x, int lda,
                              _Float16* __restrict__ y, int ldy, int M, int K) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(M) * K)
    return;
  const int r = int(i / K), c = int(i % K);
  y[size_t(r) * ldy + c] = (_Float16)x[size_t(r) * lda + c];
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

template<int EPI, int NR>
__global__ void __launch_bounds__(128)
    k_gemm_h(const _Float16* __restrict__ A, int lda,
             const _Float16* __restrict__ W, int ldw, int M, int N, int K,
             const float* __restrict__ bias, const float* __restrict__ ls,
             float* C, int ldc) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ln = lane & 15,
            hi = lane >> 4;
  const int n0 = (blockIdx.x * 4 + wave) * (16 * NR);
  if (n0 >= N)
    return;
  const int m0 = blockIdx.y * 64;
  const _Float16* wr[NR];
  const _Float16* ar[4];
#pragma unroll
  for (int r = 0; r < NR; r++)
    wr[r] = W + size_t(n0 + 16 * r + ln) * ldw;
#pragma unroll
  for (int t = 0; t < 4; t++)
    ar[t] = A + size_t(min(m0 + 16 * t + ln, M - 1)) * lda;
  v8f acc[4][NR];
#pragma unroll
  for (int t = 0; t < 4; t++)
#pragma unroll
    for (int r = 0; r < NR; r++)
      acc[t][r] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    v16h b[NR];
#pragma unroll
    for (int r = 0; r < NR; r++)
      b[r] = ld16h(wr[r] + k0);
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const v16h a = ld16h(ar[t] + k0);
#pragma unroll
      for (int r = 0; r < NR; r++)
        acc[t][r] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[r], acc[t][r]);
    }
  }
#pragma unroll
  for (int t = 0; t < 4; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int m = m0 + 16 * t + 2 * i + hi;
      if (m < M) {
#pragma unroll
        for (int r = 0; r < NR; r++)
          epilogue(acc[t][r][i], m, n0 + 16 * r + ln, ldc, bias, ls, C, EPI);
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
__global__ void __launch_bounds__(256)
    k_layernorm(const float* x, const float* __restrict__ w,
                const float* __restrict__ b, float* y, float eps) {
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
    y[row + e] = (v[i] - mean) * rstd * w[e] + b[e];
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

// f32 -> f16 (optionally scaled) for the WMMA attention operands.
__global__ void k_to_f16(const float* __restrict__ x, _Float16* __restrict__ y,
                         size_t n, float scale) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (_Float16)(x[i] * scale);
}

// WMMA flash attention. Q16 is pre-scaled by (1/8) * log2(e). Block = 4 waves =
// 64 queries of one head; each wave owns 16 queries and streams 64-key tiles (K
// as [key][d], V transposed to [d][key] in LDS). S = Q K^T and O += P V use
// v_wmma_f32_16x16x16_f16; softmax state is f32 in registers (C layout: acc[i]
// = row 2i + hi, column ln); P goes through LDS to become the A operand.
__global__ void __launch_bounds__(128)
    k_attn_wmma(const _Float16* __restrict__ Q, const _Float16* __restrict__ K,
                const _Float16* __restrict__ V, float* __restrict__ O, int n) {
  constexpr int LD = 72;
  __shared__ _Float16 Ks[64 * LD], Vt[64 * LD], Ps[4][16 * LD];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, ln = lane & 15,
            hi = lane >> 4;
  const int head = blockIdx.y, q0 = blockIdx.x * 64 + wave * 16;
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
  float m[8], l[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    m[i] = -INFINITY;
    l[i] = 0.f;
  }
  for (int k0 = 0; k0 < n; k0 += 64) {
    __syncthreads();
    for (int c = tid; c < 512; c += 128) {
      const int key = c >> 3, seg = c & 7, gk = k0 + key;
      uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
      if (gk < n) {
        const size_t o = size_t(gk) * kEmbd + head * kHeadDim + seg * 8;
        kv = *reinterpret_cast<const uint4*>(K + o);
        vv = *reinterpret_cast<const uint4*>(V + o);
      }
      *reinterpret_cast<uint4*>(&Ks[key * LD + seg * 8]) = kv;
      const _Float16* vh = reinterpret_cast<const _Float16*>(&vv);
#pragma unroll
      for (int e = 0; e < 8; e++)
        Vt[(seg * 8 + e) * LD + key] = vh[e];
    }
    __syncthreads();
    v8f sc[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
      sc[j] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int kk = 0; kk < 4; kk++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const v16h kb = ld16h(&Ks[(j * 16 + ln) * LD + kk * 16]);
        sc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[kk], kb, sc[j]);
      }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float v = (k0 + j * 16 + ln < n) ? sc[j][i] : -INFINITY;
        sc[j][i] = v;
        mx = fmaxf(mx, v);
      }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1)
        mx = fmaxf(mx, __shfl_xor(mx, o, 16));
      const float mnew = fmaxf(m[i], mx), corr = exp2f(m[i] - mnew);
      float ps = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float p = exp2f(sc[j][i] - mnew);
        ps += p;
        Ps[wave][(2 * i + hi) * LD + j * 16 + ln] = (_Float16)p;
      }
      l[i] = l[i] * corr + ps;
      m[i] = mnew;
#pragma unroll
      for (int t = 0; t < 4; t++)
        oacc[t][i] *= corr;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      const v16h pa = ld16h(&Ps[wave][ln * LD + kk * 16]);
#pragma unroll
      for (int t = 0; t < 4; t++) {
        const v16h vb = ld16h(&Vt[(t * 16 + ln) * LD + kk * 16]);
        oacc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, oacc[t]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float ls = l[i];
#pragma unroll
    for (int o = 1; o < 16; o <<= 1)
      ls += __shfl_xor(ls, o, 16);
    const float inv = 1.f / ls;
    const int q = q0 + 2 * i + hi;
    if (q < n) {
#pragma unroll
      for (int t = 0; t < 4; t++)
        O[size_t(q) * kEmbd + head * kHeadDim + t * 16 + ln] = oacc[t][i] * inv;
    }
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
  DevLinear q, k, v, o, up, down;
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
  Buf X, XN, Q, K, V, O, H, OUT, QA, Q16, K16, V16, AH;
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
    for (Buf* b :
         {&X, &XN, &Q, &K, &V, &O, &H, &OUT, &QA, &Q16, &K16, &V16, &AH})
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
    std::vector<_Float16> h(size_t(n) * ldh, (_Float16)0);
    for (int r = 0; r < n; r++) {
      const uint8_t* row = t->data + size_t(r) * RowBytesOf(t->type, k);
      for (int c = 0; c < k; c++) {
        float v;
        if (t->type == kT_F32)
          std::memcpy(&v, row + size_t(c) * 4, 4);
        else if (t->type == kT_F16) {
          uint16_t u;
          std::memcpy(&u, row + size_t(c) * 2, 2);
          v = HalfToF(u);
        } else if (t->type == kT_BF16) {
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
        h[size_t(r) * ldh + c] = (_Float16)v;
      }
    }
    if (!Upload(h.data(), h.size() * 2, &d, err))
      return false;
    out->w = static_cast<const uint8_t*>(d);
    out->type = kT_F16;
    out->n = n;
    out->k = k;
    out->rowbytes = size_t(ldh) * 2;
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

  // C = epi(A W^T + bias). Fast mode: weights are f16 on the device, A is
  // rounded to f16 first.
  void Gemm(hipStream_t s, const DevLinear& L, const float* A, int lda, int M,
            float* C, int ldc, const float* bias, const float* ls, int epi) {
    if (!exact && L.type == kT_F16) {
      const int ldh =
          L.k + 16;  // padded f16 row stride (halves) for both operands
      {
        const size_t n = size_t(M) * L.k;
        k_rows_to_f16<<<unsigned((n + 255) / 256), 256, 0, s>>>(
            A, lda, reinterpret_cast<_Float16*>(AH.p), ldh, M, L.k);
      }
      const _Float16* Ah = reinterpret_cast<const _Float16*>(AH.p);
      const _Float16* W = reinterpret_cast<const _Float16*>(L.w);
      const int mb = (M + 63) / 64;
      const bool wide =
          (L.n / 256) * mb >=
          120;  // enough blocks to fill the GPU with 64-wide waves
      const int nr = wide ? 4 : 2;
      dim3 g((L.n + 64 * nr - 1) / (64 * nr), mb);
#define GH(EPIV)                                                            \
  if (wide)                                                                 \
    k_gemm_h<EPIV, 4>                                                       \
        <<<g, 128, 0, s>>>(Ah, ldh, W, ldh, M, L.n, L.k, bias, ls, C, ldc); \
  else                                                                      \
    k_gemm_h<EPIV, 2>                                                       \
        <<<g, 128, 0, s>>>(Ah, ldh, W, ldh, M, L.n, L.k, bias, ls, C, ldc);
      if (epi == kEpiStore) {
        GH(kEpiStore)
      } else if (epi == kEpiGelu) {
        GH(kEpiGelu)
      } else {
        GH(kEpiResScale)
      }
#undef GH
      return;
    }
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
};

VisionEncoder::VisionEncoder() : impl_(new Impl) {}
VisionEncoder::~VisionEncoder() = default;
void VisionEncoder::set_options(const Options& o) {
  impl_->opt = o;
}
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
    if (!m.UploadLinear(g.Find(p + "attn_q.weight"), kEmbd, kEmbd, &L.q, err) ||
        !m.UploadLinear(g.Find(p + "attn_k.weight"), kEmbd, kEmbd, &L.k, err) ||
        !m.UploadLinear(g.Find(p + "attn_v.weight"), kEmbd, kEmbd, &L.v, err) ||
        !m.UploadLinear(g.Find(p + "attn_out.weight"), kEmbd, kEmbd, &L.o,
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
  if (!m.Grow(m.X, sz_pos, err) || !m.Grow(m.XN, sz_pos, err) ||
      !m.Grow(m.Q, sz_pos, err) || !m.Grow(m.K, sz_pos, err) ||
      !m.Grow(m.V, sz_pos, err) || !m.Grow(m.O, sz_pos, err) ||
      !m.Grow(m.H, std::max(size_t(n_pos) * kFfn, size_t(np) * kPatchElems),
              err) ||
      !m.Grow(m.OUT, size_t(nt) * m.proj_dim, err) ||
      (m.opt.emulate_q8_acts && !m.Grow(m.QA, size_t(n_pos) * kFfn, err)) ||
      (!m.exact && (!m.Grow(m.AH, size_t(n_pos) * (kFfn + 16) / 2 + 8, err) ||
                    !m.Grow(m.Q16, sz_pos / 2 + 8, err) ||
                    !m.Grow(m.K16, sz_pos / 2 + 8, err) ||
                    !m.Grow(m.V16, sz_pos / 2 + 8, err))))
    return false;
  // The upload is synchronous w.r.t. the host buffer so the caller may reuse
  // `rgb` right away.
  HIPOK(hipMemcpyAsync(m.d_img, px, img_bytes, hipMemcpyHostToDevice, s));
  HIPOK(hipStreamSynchronize(s));

  auto blocks = [](size_t n, int t) { return unsigned((n + t - 1) / t); };
  k_patchify<<<blocks(size_t(np) * kPatchElems, 256), 256, 0, s>>>(
      m.d_img, iw, ih, gw, gh, m.d_mean255, m.d_std255, m.H.p);
  m.Gemm(s, m.patch, m.H.p, kPatchElems, np, m.Q.p, kEmbd, nullptr, nullptr,
         kEpiStore);  // E -> Q
  k_embed<<<blocks(sz_pos, 256), 256, 0, s>>>(m.Q.p, m.d_cls, m.d_cls_pos,
                                              m.d_pos, gw, gh, m.X.p);
  m.Tap(s, "embeddings", m.X.p, n_pos, kEmbd);
  k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, m.d_pre_w, m.d_pre_b, m.X.p, m.eps);
  m.Tap(s, "pre_ln", m.X.p, n_pos, kEmbd);

  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  for (int il = 0; il < kLayers; il++) {
    const DevLayer& L = m.layers[il];
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln1_w, L.ln1_b, m.XN.p, m.eps);
    m.Gemm(s, L.q, m.XN.p, kEmbd, n_pos, m.Q.p, kEmbd, L.bq, nullptr,
           kEpiStore);
    m.Gemm(s, L.k, m.XN.p, kEmbd, n_pos, m.K.p, kEmbd, L.bk, nullptr,
           kEpiStore);
    m.Gemm(s, L.v, m.XN.p, kEmbd, n_pos, m.V.p, kEmbd, L.bv, nullptr,
           kEpiStore);
    k_rope<<<blocks(size_t(n_pos) * kHeads * 32, 256), 256, 0, s>>>(
        m.Q.p, m.K.p, gw, n_pos, m.theta);
    if (!m.exact) {
      k_to_f16<<<blocks(sz_pos, 256), 256, 0, s>>>(
          m.Q.p, reinterpret_cast<_Float16*>(m.Q16.p), sz_pos,
          scale * 1.4426950408889634f);
      k_to_f16<<<blocks(sz_pos, 256), 256, 0, s>>>(
          m.K.p, reinterpret_cast<_Float16*>(m.K16.p), sz_pos, 1.0f);
      k_to_f16<<<blocks(sz_pos, 256), 256, 0, s>>>(
          m.V.p, reinterpret_cast<_Float16*>(m.V16.p), sz_pos, 1.0f);
      k_attn_wmma<<<dim3((n_pos + 63) / 64, kHeads), 128, 0, s>>>(
          reinterpret_cast<const _Float16*>(m.Q16.p),
          reinterpret_cast<const _Float16*>(m.K16.p),
          reinterpret_cast<const _Float16*>(m.V16.p), m.O.p, n_pos);
    } else {
      k_attn<<<dim3((n_pos + 63) / 64, kHeads), 64, 0, s>>>(
          m.Q.p, m.K.p, m.V.p, m.O.p, n_pos, scale);
    }
    if (m.tap_dir && il == 0)
      m.Tap(s, "attnO-0", m.O.p, n_pos, kEmbd);
    m.Gemm(s, L.o, m.O.p, kEmbd, n_pos, m.X.p, kEmbd, L.bo, L.ls1,
           kEpiResScale);
    k_layernorm<<<n_pos, 256, 0, s>>>(m.X.p, L.ln2_w, L.ln2_b, m.XN.p, m.eps);
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

}  // namespace gufo::models::inkling2::vision
