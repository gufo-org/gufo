// Fast kernels for Inkling 2 Mini. See fast.hpp.
// MIT. Ported from ext/lab (kern.h, kern2.h, attn/attn_kernels.h of this
// repository).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "src/models/inkling2/hip/fast.hpp"

namespace gufo::models::inkling2::fast {
namespace {

constexpr int WS = 32;
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef uint32_t v4u_a2 __attribute__((ext_vector_type(4), aligned(2)));
typedef uint32_t v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));

constexpr int FMT_Q8 = 0, FMT_NVFP4 = 1;
template<int F>
struct Fm;
template<>
struct Fm<FMT_Q8> {
  static constexpr int BYTES = 34, UEL = 32;
};
template<>
struct Fm<FMT_NVFP4> {
  static constexpr int BYTES = 36, UEL = 64;
};

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
  return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false);
}
__device__ __forceinline__ uint32_t dec4(uint32_t n) {
  uint32_t sel = n & 0x07070707u;
  uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);
  uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);
  uint32_t m = ((n >> 3) & 0x01010101u) * 255u;
  return (neg & m) | (pos & ~m);
}
__device__ __forceinline__ float ue4m3(uint32_t x) {
  uint32_t e = (x >> 3) & 15u, m = x & 7u;
  float nrm = __uint_as_float(((e + 119u) << 23) | (m << 20));
  float sub = (float)m * (1.0f / 1024.0f);
  float r = e ? nrm : sub;
  return (x == 0x7Fu) ? 0.f : r;
}
__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
  return ((uint32_t)(uint8_t)(int8_t)(int)rintf(a)) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(b) << 8) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(c) << 16) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(d) << 24);
}

// Quantize activation vectors (contiguous, K floats each) to int8 with a per-16
// f32 scale in LDS, plus a second int8 plane holding the quantisation residual
// (about 14 bit of activation precision in total). Plane 1 lives at aq +
// NV*CPU*NU and asc + NV*K/16.
template<int K, int NV, int UEL, int NPER = NV>
__device__ __forceinline__ void quant_act(const float* __restrict__ x, v4u* aq,
                                          float* asc, int res, int ldt = 0) {
  constexpr int NG = K / 16, NU = K / UEL, CPU = UEL / 16, PLQ = NV * CPU * NU,
                PLS = NV * NG;
  for (int it = threadIdx.x; it < NV * NG; it += blockDim.x) {
    int v = it / NG, g = it % NG;
    const float4* xp = (const float4*)(x + (size_t)(v / NPER) * ldt +
                                       (size_t)(v % NPER) * K + g * 16);
    float f[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float4 t = xp[i];
      f[4 * i] = t.x;
      f[4 * i + 1] = t.y;
      f[4 * i + 2] = t.z;
      f[4 * i + 3] = t.w;
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)),
                               fmaxf(fabsf(t.z), fabsf(t.w))));
    }
    const float sc = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
    float r[16];
    uint32_t q[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float q0 = rintf(f[4 * i] * id), q1 = rintf(f[4 * i + 1] * id),
            q2 = rintf(f[4 * i + 2] * id), q3 = rintf(f[4 * i + 3] * id);
      q[i] = pack4(q0, q1, q2, q3);
      r[4 * i] = f[4 * i] - q0 * sc;
      r[4 * i + 1] = f[4 * i + 1] - q1 * sc;
      r[4 * i + 2] = f[4 * i + 2] - q2 * sc;
      r[4 * i + 3] = f[4 * i + 3] - q3 * sc;
    }
    asc[v * NG + g] = sc;
    const int idx = (v * CPU + g % CPU) * NU + g / CPU;
    aq[idx] = v4u{q[0], q[1], q[2], q[3]};
    if (res) {
      // residual: |r| <= sc/2; scale sc/254 maps it onto +-127
      const float sc2 = sc * (1.0f / 254.0f),
                  id2 = sc > 0.f ? 254.0f / sc : 0.f;
      uint32_t q2p[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
        q2p[i] = pack4(r[4 * i] * id2, r[4 * i + 1] * id2, r[4 * i + 2] * id2,
                       r[4 * i + 3] * id2);
      asc[PLS + v * NG + g] = sc2;
      aq[PLQ + idx] = v4u{q2p[0], q2p[1], q2p[2], q2p[3]};
    }
  }
  __syncthreads();
}

template<int B, int K, int VS = 1>
__device__ __forceinline__ void q8_unit_c(v4u w0, v4u w1, float d,
                                          const v4u* aq, const float* asc,
                                          int u, int vb, float (&acc)[B]) {
  constexpr int NU = K / 32;
#pragma unroll
  for (int b = 0; b < B; b++) {
    v4u a0 = aq[((vb + b * VS) * 2 + 0) * NU + u],
        a1 = aq[((vb + b * VS) * 2 + 1) * NU + u];
    int s0 = dot4(w0.x, a0.x, 0);
    s0 = dot4(w0.y, a0.y, s0);
    s0 = dot4(w0.z, a0.z, s0);
    s0 = dot4(w0.w, a0.w, s0);
    int s1 = dot4(w1.x, a1.x, 0);
    s1 = dot4(w1.y, a1.y, s1);
    s1 = dot4(w1.z, a1.z, s1);
    s1 = dot4(w1.w, a1.w, s1);
    const float* sc = asc + (vb + b * VS) * (K / 16) + 2 * u;
    acc[b] += d * (sc[0] * (float)s0 + sc[1] * (float)s1);
  }
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void q8_unit(const uint8_t* p, const v4u* aq,
                                        const float* asc, int u, int vb,
                                        float (&acc)[B]) {
  v4u w0 = *(const v4u_a2*)(p + 2);
  v4u w1 = *(const v4u_a2*)(p + 18);
  float d = __half2float(*(const __half*)p);
  q8_unit_c<B, K, VS>(w0, w1, d, aq, asc, u, vb, acc);
}

// A unit's 64 weights decoded to int8 (l0, l1, h0, h1 per 16-element block)
// and its four block scales.
struct NvDec {
  uint32_t w[16];
  float ds[4];
};
__device__ __forceinline__ void nv_decode(uint32_t sw, v4u q0, v4u q1,
                                          NvDec& d) {
  uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int s = 0; s < 4; s++) {
    uint32_t w0 = w[2 * s], w1 = w[2 * s + 1];
    d.w[4 * s] = dec4(w0 & 0x0F0F0F0Fu);
    d.w[4 * s + 1] = dec4(w1 & 0x0F0F0F0Fu);
    d.w[4 * s + 2] = dec4((w0 >> 4) & 0x0F0F0F0Fu);
    d.w[4 * s + 3] = dec4((w1 >> 4) & 0x0F0F0F0Fu);
    d.ds[s] = ue4m3((sw >> (8 * s)) & 0xFFu);
  }
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void nv_unit_d(const NvDec& d, const v4u* aq,
                                          const float* asc, int u, int vb,
                                          float (&acc)[B]) {
  constexpr int NU = K / 64, NB16 = K / 16;
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const uint32_t l0 = d.w[4 * s], l1 = d.w[4 * s + 1], h0 = d.w[4 * s + 2],
                   h1 = d.w[4 * s + 3];
    const float ds = d.ds[s];
#pragma unroll
    for (int b = 0; b < B; b++) {
      v4u a = aq[((vb + b * VS) * 4 + s) * NU + u];
      int t = dot4(l0, a.x, 0);
      t = dot4(l1, a.y, t);
      t = dot4(h0, a.z, t);
      t = dot4(h1, a.w, t);
      acc[b] += ds * asc[(vb + b * VS) * NB16 + u * 4 + s] * (float)t;
    }
  }
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void nv_unit_c(uint32_t sw, v4u q0, v4u q1,
                                          const v4u* aq, const float* asc,
                                          int u, int vb, float (&acc)[B]) {
  NvDec d;
  nv_decode(sw, q0, q1, d);
  nv_unit_d<B, K, VS>(d, aq, asc, u, vb, acc);
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void nv_unit(const uint8_t* p, const v4u* aq,
                                        const float* asc, int u, int vb,
                                        float (&acc)[B]) {
  uint32_t sw = *(const uint32_t*)p;
  v4u q0 = *(const v4u_a4*)(p + 4);
  v4u q1 = *(const v4u_a4*)(p + 20);
  nv_unit_c<B, K, VS>(sw, q0, q1, aq, asc, u, vb, acc);
}

// Streamed weight reads that bypass last-level-cache (MALL) allocation: buffer
// loads with DLC. They run at the same rate as plain loads, but do not evict
// what the decode prefetch placed in the cache. The resource is wave-uniform
// (an expert's or a matrix's base); offsets stay below 2^31.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t wrsrc(const void* base) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), (short)0,
                                           0x7FFFFFFF, 0x31004000);
}
constexpr int kDlc = 4;  // cache policy: DLC (no MALL allocation)
__device__ __forceinline__ v4u wld16(__amdgpu_buffer_rsrc_t r, int off) {
  return __builtin_bit_cast(
      v4u, __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, kDlc));
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void q8_unit_dl(__amdgpu_buffer_rsrc_t r, int off,
                                           const v4u* aq, const float* asc,
                                           int u, int vb, float (&acc)[B]) {
  v4u w0 = wld16(r, off + 2);
  v4u w1 = wld16(r, off + 18);
  float d = __half2float(
      __ushort_as_half(__builtin_amdgcn_raw_buffer_load_b16(r, off, 0, kDlc)));
  q8_unit_c<B, K, VS>(w0, w1, d, aq, asc, u, vb, acc);
}
template<int B, int K, int VS = 1>
__device__ __forceinline__ void nv_unit_dl(__amdgpu_buffer_rsrc_t r, int off,
                                           const v4u* aq, const float* asc,
                                           int u, int vb, float (&acc)[B]) {
  uint32_t sw = __builtin_amdgcn_raw_buffer_load_b32(r, off, 0, kDlc);
  v4u q0 = wld16(r, off + 4);
  v4u q1 = wld16(r, off + 20);
  nv_unit_c<B, K, VS>(sw, q0, q1, aq, asc, u, vb, acc);
}

// A lane's units of one weight row, loaded ahead of the products (the same
// units, products and order as row_dot).
template<int FMT, int K, int LPR>
struct RowRegs {
  static constexpr int NU = K / Fm<FMT>::UEL, NI = (NU + LPR - 1) / LPR;
  uint32_t h[NI];  // NVFP4: scale word; Q8: the f16 scale's bits
  v4u q0[NI], q1[NI];
};
// TS (tail split, row_dot_g): every lane loads a last unit, lane lir the
// unit NFULL * LPR + lir % T of the T left over.
template<int FMT, int K, int LPR, bool TS = false>
__device__ __forceinline__ void row_load(const uint8_t* row, int lir,
                                         RowRegs<FMT, K, LPR>& r) {
  using R = RowRegs<FMT, K, LPR>;
  constexpr int BY = Fm<FMT>::BYTES, T = R::NU % LPR;
#pragma unroll
  for (int i = 0; i < R::NI; ++i) {
    const int u = (TS && T > 0 && i == R::NI - 1)
                      ? (R::NU / LPR) * LPR + lir % (T > 0 ? T : 1)
                      : i * LPR + lir;
    if (TS || (R::NU % LPR == 0) || u < R::NU) {
      const uint8_t* p = row + (size_t)u * BY;
      if (FMT == FMT_Q8) {
        r.h[i] = *(const uint16_t*)p;
        r.q0[i] = *(const v4u_a2*)(p + 2);
        r.q1[i] = *(const v4u_a2*)(p + 18);
      } else {
        r.h[i] = *(const uint32_t*)p;
        r.q0[i] = *(const v4u_a4*)(p + 4);
        r.q1[i] = *(const v4u_a4*)(p + 20);
      }
    }
  }
}
// row_load with the row at byte `roff` of the resource r (DLC reads).
template<int FMT, int K, int LPR, bool TS = false>
__device__ __forceinline__ void row_load_dl(__amdgpu_buffer_rsrc_t rs, int roff,
                                            int lir, RowRegs<FMT, K, LPR>& r) {
  using R = RowRegs<FMT, K, LPR>;
  constexpr int BY = Fm<FMT>::BYTES, T = R::NU % LPR;
#pragma unroll
  for (int i = 0; i < R::NI; ++i) {
    const int u = (TS && T > 0 && i == R::NI - 1)
                      ? (R::NU / LPR) * LPR + lir % (T > 0 ? T : 1)
                      : i * LPR + lir;
    if (TS || (R::NU % LPR == 0) || u < R::NU) {
      const int off = roff + u * BY;
      if (FMT == FMT_Q8) {
        r.h[i] = __builtin_amdgcn_raw_buffer_load_b16(rs, off, 0, kDlc);
        r.q0[i] = wld16(rs, off + 2);
        r.q1[i] = wld16(rs, off + 18);
      } else {
        r.h[i] = __builtin_amdgcn_raw_buffer_load_b32(rs, off, 0, kDlc);
        r.q0[i] = wld16(rs, off + 4);
        r.q1[i] = wld16(rs, off + 20);
      }
    }
  }
}
// NVFP4 rows decoded once, for all the pairs that read them.
template<int K, int LPR>
struct RowDec {
  static constexpr int NU = K / 64, NI = (NU + LPR - 1) / LPR;
  NvDec d[NI];
};
template<int K, int LPR, bool TS = false>
__device__ __forceinline__ void row_decode(const RowRegs<FMT_NVFP4, K, LPR>& r,
                                           int lir, RowDec<K, LPR>& d) {
  using R = RowDec<K, LPR>;
#pragma unroll
  for (int i = 0; i < R::NI; ++i)
    if (TS || (R::NU % LPR == 0) || i * LPR + lir < R::NU)
      nv_decode(r.h[i], r.q0[i], r.q1[i], d.d[i]);
}

template<int FMT, int K, int LPR, int B, int VS = 1>
__device__ __forceinline__ void row_dot(const uint8_t* row, const v4u* aq,
                                        const float* asc, int plq, int pls,
                                        int res, int vb, int lir,
                                        float (&acc)[B]) {
  constexpr int NU = K / Fm<FMT>::UEL, BY = Fm<FMT>::BYTES;
#pragma unroll
  for (int b = 0; b < B; b++)
    acc[b] = 0.f;
#pragma unroll
  for (int u0 = 0; u0 < NU; u0 += LPR) {
    int u = u0 + lir;
    if ((NU % LPR == 0) || u < NU) {
      if (FMT == FMT_Q8) {
        q8_unit<B, K, VS>(row + (size_t)u * BY, aq, asc, u, vb, acc);
        if (res)
          q8_unit<B, K, VS>(row + (size_t)u * BY, aq + plq, asc + pls, u, vb,
                            acc);
      } else {
        nv_unit<B, K, VS>(row + (size_t)u * BY, aq, asc, u, vb, acc);
        if (res)
          nv_unit<B, K, VS>(row + (size_t)u * BY, aq + plq, asc + pls, u, vb,
                            acc);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      acc[b] += __shfl_xor(acc[b], off, WS);
}

// row_dot with the row at byte `roff` of the resource r (DLC reads).
template<int FMT, int K, int LPR, int B, int VS = 1>
__device__ __forceinline__ void row_dot_dl(__amdgpu_buffer_rsrc_t r, int roff,
                                           const v4u* aq, const float* asc,
                                           int plq, int pls, int res, int vb,
                                           int lir, float (&acc)[B]) {
  constexpr int NU = K / Fm<FMT>::UEL, BY = Fm<FMT>::BYTES;
#pragma unroll
  for (int b = 0; b < B; b++)
    acc[b] = 0.f;
#pragma unroll
  for (int u0 = 0; u0 < NU; u0 += LPR) {
    int u = u0 + lir;
    if ((NU % LPR == 0) || u < NU) {
      const int off = roff + u * BY;
      if (FMT == FMT_Q8) {
        q8_unit_dl<B, K, VS>(r, off, aq, asc, u, vb, acc);
        if (res)
          q8_unit_dl<B, K, VS>(r, off, aq + plq, asc + pls, u, vb, acc);
      } else {
        nv_unit_dl<B, K, VS>(r, off, aq, asc, u, vb, acc);
        if (res)
          nv_unit_dl<B, K, VS>(r, off, aq + plq, asc + pls, u, vb, acc);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      acc[b] += __shfl_xor(acc[b], off, WS);
}

// ---------------- dense GEMV (B = 1) with optional softcap ----------------
template<int FMT, int K, int LPR, int NIT, int BW, int NB = 1>
__global__ void __launch_bounds__(BW* WS)
    k_gemv(const uint8_t* __restrict__ W, const float* __restrict__ x,
           float* __restrict__ y, int M, float softcap, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[2 * NB * CPU * NU];
  __shared__ float asc[2 * NB * (K / 16)];
  quant_act<K, NB, Fm<FMT>::UEL>(x, aq, asc, res);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
      lir = lane % LPR;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < M ? row : M - 1;
    float acc[NB];
    row_dot<FMT, K, LPR, NB>(W + rb * rr, aq, asc, NB * CPU * NU, NB * (K / 16),
                             res, 0, lir, acc);
    if (row < M && lir == 0)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        y[(size_t)b * M + row] =
            softcap > 0.f ? softcap * tanhf(acc[b] / softcap) : acc[b];
  }
}

// ---------------- fused multi-matrix Q8_0 GEMV (K = 2304) ----------------
struct MultiP {
  const uint8_t* W[5];
  int M[5];
  int in[5];
  int yoff[5];
  int cum[6];
  int n;
  int res;
  int ldx;  // token stride of x (floats)
  int ldy;  // token stride of y
  // tiny F32 / F16 / BF16 matrices handled by the first `ntb` blocks
  int tn, ntb;
  const void* tw[3];
  int ttype[3], trows[3], tin[3], tys[3];
  float* ty[3];
};

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o, 32);
  return v;
}
// One wave = one row of a tiny matrix (K = 2304, F32 / F16 / BF16) against NB
// un-quantised input vectors.
template<int NB>
__device__ __forceinline__ void tiny_row(const MultiP& P,
                                         const float* __restrict__ x, int row,
                                         int lane, int b0 = 0,
                                         int nvalid = NB) {
  int m = 0;
  while (m < P.tn - 1 && row >= P.trows[m]) {
    row -= P.trows[m];
    ++m;
  }
  if (row >= P.trows[m])
    return;
  float acc[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    acc[b] = 0.f;
  const int ty = P.ttype[m];
  const float* xv = x + (size_t)P.tin[m] * 2304;
  if (ty == 0) {
    const float4* wp =
        (const float4*)((const float*)P.tw[m] + (size_t)row * 2304);
    float4 wv[18];
#pragma unroll
    for (int i = 0; i < 18; ++i)
      wv[i] = wp[lane + 32 * i];
    // the token loop stays rolled (unrolled, all tokens' input loads are
    // hoisted and the kernel spills); each token's sum is the same
#pragma unroll 1
    for (int b = 0; b < NB; ++b) {
      const float4* xp = (const float4*)(xv + (size_t)b * P.ldx);
#pragma unroll
      for (int i = 0; i < 18; ++i) {
        const float4 a = xp[lane + 32 * i];
        acc[b] += wv[i].x * a.x + wv[i].y * a.y + wv[i].z * a.z + wv[i].w * a.w;
      }
    }
  } else {
    // F16 / BF16: 2304 halves = 288 x 16 B; 9 chunks of 8 per lane
    const uint4* wp =
        (const uint4*)((const uint16_t*)P.tw[m] + (size_t)row * 2304);
    uint4 wv[9];
#pragma unroll
    for (int i = 0; i < 9; ++i)
      wv[i] = wp[lane + 32 * i];
#pragma unroll 1
    for (int b = 0; b < NB; ++b) {
      const float4* xp = (const float4*)(xv + (size_t)b * P.ldx);
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        const uint32_t u[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
        const float4 a0 = xp[2 * (lane + 32 * i)],
                     a1 = xp[2 * (lane + 32 * i) + 1];
        const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          float lo, hi;
          if (ty == 1) {
            lo = __half2float(
                __ushort_as_half((unsigned short)(u[h] & 0xFFFFu)));
            hi = __half2float(__ushort_as_half((unsigned short)(u[h] >> 16)));
          } else {
            lo = __uint_as_float(u[h] << 16);
            hi = __uint_as_float(u[h] & 0xFFFF0000u);
          }
          acc[b] += lo * a[2 * h] + hi * a[2 * h + 1];
        }
      }
    }
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float v = wsum(acc[b]);
    if (lane == 0 && b < nvalid)
      P.ty[m][(size_t)(b0 + b) * P.tys[m] + row] = v;
  }
}

template<int K, int LPR, int NIT, int BW, int NI, int NB = 1>
__global__ void __launch_bounds__(BW* WS)
    k_gemv_multi(MultiP P, const float* __restrict__ x, float* __restrict__ y,
                 int Mtot) {
  if (blockIdx.x < P.ntb) {  // the tiny matrices first, so they never queue
                             // behind the long GEMV
    tiny_row<NB>(P, x, blockIdx.x * BW + threadIdx.x / WS, threadIdx.x % WS);
    return;
  }
  const int bx = blockIdx.x - P.ntb;
  constexpr int NU = K / 32, RPW = WS / LPR;
  __shared__ v4u aq[2 * NB * NI * 2 * NU];
  __shared__ float asc[2 * NB * NI * (K / 16)];
  quant_act<K, NB * NI, 32, NI>(x, aq, asc, P.res, P.ldx);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
      lir = lane % LPR;
  size_t rb = (size_t)NU * 34;
  int base = (bx * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < Mtot ? row : Mtot - 1;
    int m = 0;
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i < P.n && rr >= P.cum[i])
        m = i;
    const uint8_t* Wm = P.W[0];
    int inm = P.in[0], yo = P.yoff[0], c0 = P.cum[0];
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i == m) {
        Wm = P.W[i];
        inm = P.in[i];
        yo = P.yoff[i];
        c0 = P.cum[i];
      }
    int lr = rr - c0;
    float acc[NB];
    row_dot<FMT_Q8, K, LPR, NB, NI>(Wm + rb * lr, aq, asc, NB * NI * 2 * NU,
                                    NB * NI * (K / 16), P.res, inm, lir, acc);
    if (row < Mtot && lir == 0)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        y[(size_t)b * P.ldy + yo + lr] = acc[b];
  }
}

// k_gemv_multi for several inputs per token where every block's rows belong
// to matrices of one input (the matrix boundaries are multiples of a block's
// rows): a block quantizes only that input of its NB tokens, so up to 8 tokens
// fit in LDS and one launch reads the weights once. Each row's quantized input
// and dot product are k_gemv_multi's.
template<int K, int LPR, int NIT, int BW, int NB>
__global__ void __launch_bounds__(BW* WS)
    k_gemv_multi1(MultiP P, const float* __restrict__ x, float* __restrict__ y,
                  int Mtot) {
  if (blockIdx.x < P.ntb) {
    tiny_row<NB>(P, x, blockIdx.x * BW + threadIdx.x / WS, threadIdx.x % WS);
    return;
  }
  const int bx = blockIdx.x - P.ntb;
  constexpr int NU = K / 32, RPW = WS / LPR;
  __shared__ v4u aq[2 * NB * 2 * NU];
  __shared__ float asc[2 * NB * (K / 16)];
  const int brow = min(bx * BW * RPW * NIT, Mtot - 1);
  int bin = P.in[0];
#pragma unroll
  for (int i = 1; i < 5; i++)
    if (i < P.n && brow >= P.cum[i])
      bin = P.in[i];
  quant_act<K, NB, 32, 1>(x + (size_t)bin * K, aq, asc, P.res, P.ldx);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
      lir = lane % LPR;
  size_t rb = (size_t)NU * 34;
  int base = (bx * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < Mtot ? row : Mtot - 1;
    int m = 0;
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i < P.n && rr >= P.cum[i])
        m = i;
    const uint8_t* Wm = P.W[0];
    int yo = P.yoff[0], c0 = P.cum[0];
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i == m) {
        Wm = P.W[i];
        yo = P.yoff[i];
        c0 = P.cum[i];
      }
    int lr = rr - c0;
    float acc[NB];
    row_dot<FMT_Q8, K, LPR, NB>(Wm + rb * lr, aq, asc, NB * 2 * NU,
                                NB * (K / 16), P.res, 0, lir, acc);
    if (row < Mtot && lir == 0)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        y[(size_t)b * P.ldy + yo + lr] = acc[b];
  }
}

// The tiny matrices of a wide step (k_gemv_multi's first blocks), CB tokens
// at a time: their own launch, so they do not raise k_gemv_wide's registers.
template<int BW, int CB>
__global__ void __launch_bounds__(BW* WS)
    k_tiny_wide(MultiP P, const float* __restrict__ x, int nb) {
  for (int c0 = 0; c0 < nb; c0 += CB)
    tiny_row<CB>(P, x + (size_t)c0 * P.ldx, blockIdx.x * BW + threadIdx.x / WS,
                 threadIdx.x % WS, c0, min(CB, nb - c0));
}
// ---------------- exact int8 WMMA GEMV (wide steps) ----------------
// Each 16-element integer dot product of q8_unit / nv_unit is one entry of a
// 16 x 16 x 16 int8 WMMA (integer sums are exact in any order). The float
// work around them is replayed per output in the GEMV's own order: lane
// `lir` of a row group accumulates units lir, lir + LPR, ... (plane 0, then
// plane 1, per unit) and the lanes are combined by the xor butterfly whose
// lane-0 value the GEMV writes. So every output is bit for bit the GEMV's.
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
struct F8 {
  float v[8];
};
// Quantized activations in global memory for the WMMA kernels: vector v
// (input v % per of token v / per, at x + (v / per) * ldt + (v % per) * K) as
// quant_act writes it for one vector: qa[v][2 planes][CPU * NU], qs[v][2][K /
// 16].
template<int K, int UEL>
__global__ void __launch_bounds__(256)
    k_quant_rows(const float* __restrict__ x, int per, int ldt, v4u* qa,
                 float* qs, int res) {
  constexpr int NG = K / 16, NU = K / UEL, CPU = UEL / 16;
  const int v = blockIdx.x;
  quant_act<K, 1, UEL>(x + (size_t)(v / per) * ldt + (size_t)(v % per) * K,
                       qa + (size_t)v * 2 * CPU * NU, qs + (size_t)v * 2 * NG,
                       res);
}

// The leaf of one output tile: Q8_0 units l, l + lpr, ... of the lane's
// weight row `wrow` against the A rows `av` (quantized vectors), scales in
// `ssc` ([2][16][K / 16]); q8_unit's arithmetic per entry.
template<int K, int LPR>
__device__ __forceinline__ F8 mma_leaf(int l, const uint8_t* wrow,
                                       const v4u* av, const float* ssc,
                                       int half, int res) {
  constexpr int NU = K / 32, NG = K / 16, NI = (NU + LPR - 1) / LPR;
  F8 acc;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    acc.v[i] = 0.f;
  // every unit's weights and inputs are loaded before the first product
  v4u w0[NI], w1[NI], a[NI][2][2];
  float d[NI];
#pragma unroll
  for (int k = 0; k < NI; ++k) {
    const int u = l + k * LPR;
    if (NU % LPR == 0 || u < NU) {
      const uint8_t* pu = wrow + (size_t)u * 34;
      w0[k] = *(const v4u_a2*)(pu + 2);
      w1[k] = *(const v4u_a2*)(pu + 18);
      d[k] = __half2float(*(const __half*)pu);
#pragma unroll
      for (int pl = 0; pl < 2; ++pl)
        if (pl <= res) {
          a[k][pl][0] = av[pl * 2 * NU + u];
          a[k][pl][1] = av[pl * 2 * NU + NU + u];
        }
    }
  }
#pragma unroll
  for (int k = 0; k < NI; ++k) {
    const int u = l + k * LPR;
    if (NU % LPR == 0 || u < NU) {
      const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int pl = 0; pl < 2; ++pl) {
        if (pl > res)
          break;
        const v8i c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, (v4i)a[k][pl][0], true, (v4i)w0[k], z, false);
        const v8i c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
            true, (v4i)a[k][pl][1], true, (v4i)w1[k], z, false);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float* sc = ssc + (pl * 16 + 2 * i + half) * NG + 2 * u;
          acc.v[i] += d[k] * (sc[0] * (float)c0[i] + sc[1] * (float)c1[i]);
        }
      }
    }
  }
  return acc;
}

// Q8_0 rows (P's matrices, one input per block) x 16-token tiles. Block
// (tile, rbk) takes weight rows rbk * 16 + 0..15 (lane c16's B column) for the
// tile's tokens (A rows; C entry i of a lane is token 2 i + half). Its NW
// waves compute the LPR leaves (wave w: leaves w, w + NW, ...) into LDS, and
// the block combines them in the butterfly's order.
template<int K, int LPR, int NW>
__global__ void __launch_bounds__(NW * 32)
    k_gemv_mma(MultiP P, const v4u* __restrict__ qa,
               const float* __restrict__ qs, int per, float* __restrict__ y,
               int Mtot, int nb, float softcap) {
  constexpr int NU = K / 32, NG = K / 16, QV = 2 * 2 * NU;
  __shared__ float ssc[2][16][NG];
  static_assert(LPR % NW == 0, "leaves per wave");
  constexpr int NLW = LPR / NW;  // leaves of a wave: w, w + NW, ...
  __shared__ float lv[NW][8][32];
  const int t0 = blockIdx.x * 16, brow = blockIdx.y * 16;
  int bin = P.in[0];
#pragma unroll
  for (int i = 1; i < 5; i++)
    if (i < P.n && brow >= P.cum[i])
      bin = P.in[i];
  for (int i = threadIdx.x; i < 2 * 16 * NG; i += NW * 32) {
    const int pl = i / (16 * NG), r = (i / NG) % 16, g = i % NG;
    const int tok = min(t0 + r, nb - 1);
    ssc[pl][r][g] = qs[(size_t)(tok * per + bin) * 2 * NG + pl * NG + g];
  }
  __syncthreads();
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32, c16 = lane & 15,
            half = lane >> 4;
  const int row = brow + c16;
  const int rr = row < Mtot ? row : Mtot - 1;
  int m = 0;
#pragma unroll
  for (int i = 1; i < 5; i++)
    if (i < P.n && rr >= P.cum[i])
      m = i;
  const uint8_t* Wm = P.W[0];
  int yo = P.yoff[0], cm = P.cum[0];
#pragma unroll
  for (int i = 1; i < 5; i++)
    if (i == m) {
      Wm = P.W[i];
      yo = P.yoff[i];
      cm = P.cum[i];
    }
  const int lr = rr - cm;
  const uint8_t* wrow = Wm + (size_t)lr * NU * 34;
  const int tokA = min(t0 + c16, nb - 1);
  const v4u* av = qa + (size_t)(tokA * per + bin) * QV;
  F8 lf[NLW];
#pragma unroll
  for (int j = 0; j < NLW; ++j)
    lf[j] = mma_leaf<K, LPR>(w + j * NW, wrow, av, &ssc[0][0][0], half, P.res);
  // the butterfly's steps for lane 0 (offsets LPR / 2 .. 1): leaf l += leaf
  // l + off for l < off. Offsets of NW and more pair leaves of one wave (in
  // registers), the others go through LDS, every entry by the block's
  // threads in turn.
#pragma unroll
  for (int off = LPR / 2; off >= NW; off >>= 1)
#pragma unroll
    for (int j = 0; j * NW < off; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        lf[j].v[i] += lf[j + off / NW].v[i];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    lv[w][i][lane] = lf[0].v[i];
  __syncthreads();
  for (int off = NW / 2; off >= 1; off >>= 1) {
    for (int idx = threadIdx.x; idx < off * 8 * 32; idx += NW * 32) {
      const int l = idx / 256, i = (idx / 32) % 8, ln = idx % 32;
      lv[l][i][ln] += lv[l + off][i][ln];
    }
    __syncthreads();
  }
  if (w != 0)
    return;
  F8 r;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    r.v[i] = lv[0][i][lane];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int tok = t0 + 2 * i + half;
    if (tok < nb && row < Mtot)
      y[(size_t)tok * P.ldy + yo + lr] =
          softcap > 0.f ? softcap * tanhf(r.v[i] / softcap) : r.v[i];
  }
}

// ---------------- router: F32 [128][2304] GEMV + sigmoid + biased top-8
// ----------------
template<int K, bool H>
__global__ void __launch_bounds__(256)
    k_router(const void* __restrict__ Wv, const float* __restrict__ bias,
             const float* __restrict__ x0, float* __restrict__ logit0,
             unsigned* __restrict__ counter0, int* __restrict__ ids0,
             float* __restrict__ wts0, float rscale) {
  const int tk = blockIdx.y;  // token of the batch
  const float* x = x0 + (size_t)tk * K;
  float* logit = logit0 + tk * 128;
  unsigned* counter = counter0 + tk;
  int* ids = ids0 + tk * 8;
  float* wts = wts0 + tk * 8;
  int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * 8 + wave;
  const float4* xp = (const float4*)x;
  float s = 0.f;
  if (H) {
    const uint2* wp = (const uint2*)((const uint16_t*)Wv + (size_t)row * K);
    uint2 wv[K / 128];
#pragma unroll
    for (int i = 0; i < K / 128; i++)
      wv[i] = wp[lane + 32 * i];
#pragma unroll
    for (int i = 0; i < K / 128; i++) {
      const float4 b = xp[lane + 32 * i];
      s += __uint_as_float(wv[i].x << 16) * b.x +
           __uint_as_float(wv[i].x & 0xFFFF0000u) * b.y +
           __uint_as_float(wv[i].y << 16) * b.z +
           __uint_as_float(wv[i].y & 0xFFFF0000u) * b.w;
    }
  } else {
    const float4* wp = (const float4*)((const float*)Wv + (size_t)row * K);
#pragma unroll
    for (int i = 0; i < K / 128; i++) {
      float4 a = wp[lane + 32 * i], b = xp[lane + 32 * i];
      s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    s += __shfl_xor(s, o, 32);
  if (lane == 0)
    logit[row] = s;
  __threadfence();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) {
    unsigned t = atomicAdd(&counter[0], 1u);
    last = (t == gridDim.x - 1);
  }
  __syncthreads();
  if (!last)
    return;
  __threadfence();
  if (wave != 0)
    return;
  float sc[4], sel[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    int i = lane + 32 * k;
    float l = __builtin_nontemporal_load(&logit[i]);
    sc[k] = 1.f / (1.f + expf(-l));
    sel[k] = sc[k] + bias[i];
  }
  float ssum = 0.f;
  float myw[8];
  int myi[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float bv = -1e30f;
    int bi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (sel[k] > bv) {
        bv = sel[k];
        bi = lane + 32 * k;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor(bv, o, 32);
      int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    float scv = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (lane + 32 * k == bi) {
        scv = sc[k];
        sel[k] = -1e30f;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      scv += __shfl_xor(scv, o, 32);
    myw[r] = scv;
    myi[r] = bi;
    ssum += scv;
  }
  if (lane < 8) {
    float w = 0.f;
    int id = 0;
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r == lane) {
        w = myw[r];
        id = myi[r];
      }
    ids[lane] = id;
    wts[lane] = w / (ssum + 1e-20f) * rscale;
  }
  if (lane == 0)
    counter[0] = 0;
}

// ---------------- MoE gate/up + SITU ----------------
// The SITU constants of Inkling 2 Mini (ParseConfig refuses other values; the
// host launchers check them). As constants, g / 4 compiles to the multiply by
// 0.25 that equals it, which makes the MoE gate/up epilogues cheaper.
constexpr float kSituBeta = 4.f, kSituLinearBeta = 25.f;
__device__ __forceinline__ float situ(float g, float u) {
  constexpr float beta = kSituBeta, lb = kSituLinearBeta;
  return beta * tanhf(g / beta) * (1.f / (1.f + expf(-g))) *
         (lb * tanhf(u / lb));
}
template<int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW* WS)
    k_moe_gu(const uint8_t* __restrict__ W, const float* __restrict__ escale,
             const float* __restrict__ x, const int* __restrict__ ids,
             float* __restrict__ a, int FF, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16,
                RPP = WS / LPR / 2;
  __shared__ v4u aq[2 * CPU * NU];
  __shared__ float asc[2 * (K / 16)];
  // Grid x is the (token, expert) assignment and y the row block, so the blocks
  // of different tokens that chose the same expert read its rows at about the
  // same time (cache hits instead of a second DRAM read).
  // One session: row blocks of one expert run next to each other (grid x is the
  // row block).
  const bool rows_first = gridDim.y == 8;
  const int p = rows_first ? blockIdx.y : blockIdx.x,
            rbk = rows_first ? blockIdx.x : blockIdx.y;
  quant_act<K, 1, Fm<FMT>::UEL>(x + (size_t)(p >> 3) * K, aq, asc, res);
  int e = ids[p];
  float sc = escale ? escale[e] : 1.f;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  const uint8_t* We = W + (size_t)e * 2 * FF * rb;
  const __amdgpu_buffer_rsrc_t wr = wrsrc(We);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
      lir = lane % LPR, isup = g & 1, pg = g >> 1;
  int base = (rbk * BW + wave) * RPP * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int j = base + it * RPP + pg;
    int jj = j < FF ? j : FF - 1;
    float acc[1];
    row_dot_dl<FMT, K, LPR, 1>(wr, (int)(rb * (isup ? FF + jj : jj)), aq, asc,
                               CPU * NU, K / 16, res, 0, lir, acc);
    float other = __shfl_xor(acc[0], LPR, WS);
    if (!isup && lir == 0 && j < FF)
      a[(size_t)p * FF + j] = situ(acc[0] * sc, other * sc);
  }
}

// ---------------- MoE down, one expert per block row (weighted partial
// outputs) ----------------
template<int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW* WS)
    k_moe_down_e(const uint8_t* __restrict__ W,
                 const float* __restrict__ escale, const float* __restrict__ a,
                 const int* __restrict__ ids, const float* __restrict__ wts,
                 float* __restrict__ yp, int D, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[2 * CPU * NU];
  __shared__ float asc[2 * (K / 16)];
  // assignment (token * 8 + slot) first, as in k_moe_gu; one session: row
  // blocks first
  const bool rows_first = gridDim.y == 8;
  const int p = rows_first ? blockIdx.y : blockIdx.x,
            rbk = rows_first ? blockIdx.x : blockIdx.y;
  quant_act<K, 1, Fm<FMT>::UEL>(a + (size_t)p * K, aq, asc, res);
  const int e = ids[p];
  const float ew = wts[p] * (escale ? escale[e] : 1.f);
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  const __amdgpu_buffer_rsrc_t wr = wrsrc(W + (size_t)e * D * rb);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
      lir = lane % LPR;
  int base = (rbk * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < D ? row : D - 1;
    float acc[1];
    row_dot_dl<FMT, K, LPR, 1>(wr, (int)(rr * rb), aq, asc, CPU * NU, K / 16,
                               res, 0, lir, acc);
    if (row < D && lir == 0)
      yp[(size_t)p * D + row] = ew * acc[0];
  }
}

// ---------------- MoE gate/up + SITU and down in one launch ----------------
// NVFP4 experts, per pair: the first `ngu` blocks are k_moe_gu's (same pair and
// row block order, same products), the rest k_moe_down_e's. A down block loads
// its weight units before it waits for the gate/up blocks of its pair (their
// count in cnt[p]), so its weight stream starts while the gate/up tail runs
// instead of after a kernel boundary. Blocks dispatch in index order, so every
// gate/up block is resident or done once a down block waits. The last down
// block of a pair resets both counters for the next launch.
template<int K1, int K2, int LPR1, int LPR2, int BW>
__global__ void __launch_bounds__(BW* WS)
    k_moe_gu_down(const uint8_t* __restrict__ Wgu,
                  const float* __restrict__ gscale,
                  const uint8_t* __restrict__ Wd,
                  const float* __restrict__ dscale, const float* __restrict__ x,
                  const int* __restrict__ ids, const float* __restrict__ wts,
                  float* __restrict__ a, float* __restrict__ yp,
                  unsigned* __restrict__ cnt, int FF, int D, int npair,
                  int gu_rb, int dn_rb, int rows_first, int res) {
  using F = Fm<FMT_NVFP4>;
  constexpr int UEL = F::UEL, BY = F::BYTES, CPU = UEL / 16;
  constexpr int NU1 = K1 / UEL, NU2 = K2 / UEL;
  constexpr int RPP = WS / LPR1 / 2, RPW = WS / LPR2;
  __shared__ v4u aq[2 * CPU * NU1];
  __shared__ float asc[2 * (K1 / 16)];
  const int ngu = npair * gu_rb;
  const int wave = threadIdx.x / WS, lane = threadIdx.x % WS;
  if ((int)blockIdx.x < ngu) {
    const int b = blockIdx.x;
    const int p = rows_first ? b / gu_rb : b % npair,
              rbk = rows_first ? b % gu_rb : b / npair;
    quant_act<K1, 1, UEL>(x + (size_t)(p >> 3) * K1, aq, asc, res);
    const int e = ids[p];
    const float sc = gscale ? gscale[e] : 1.f;
    const size_t rb = (size_t)NU1 * BY;
    const uint8_t* We = Wgu + (size_t)e * 2 * FF * rb;
    const __amdgpu_buffer_rsrc_t wr = wrsrc(We);
    const int g = lane / LPR1, lir = lane % LPR1, isup = g & 1, pg = g >> 1;
    const int j = (rbk * BW + wave) * RPP + pg;
    const int jj = j < FF ? j : FF - 1;
    float acc[1];
    row_dot_dl<FMT_NVFP4, K1, LPR1, 1>(wr, (int)(rb * (isup ? FF + jj : jj)),
                                       aq, asc, CPU * NU1, K1 / 16, res, 0, lir,
                                       acc);
    const float other = __shfl_xor(acc[0], LPR1, WS);
    if (!isup && lir == 0 && j < FF)
      a[(size_t)p * FF + j] = situ(acc[0] * sc, other * sc);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0)
      __hip_atomic_fetch_add(&cnt[p], 1u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    return;
  }
  const int b = blockIdx.x - ngu;
  const int p = rows_first ? b / dn_rb : b % npair,
            rbk = rows_first ? b % dn_rb : b / npair;
  const int e = ids[p];
  const size_t rb = (size_t)NU2 * BY;
  const __amdgpu_buffer_rsrc_t wr = wrsrc(Wd + (size_t)e * D * rb);
  const int g = lane / LPR2, lir = lane % LPR2;
  const int row = (rbk * BW + wave) * RPW + g;
  const int rr = row < D ? row : D - 1;
  // this lane's weight units (row_dot's units and order), loaded first
  constexpr int NI = (NU2 + LPR2 - 1) / LPR2;
  uint32_t sw[NI];
  v4u q0[NI], q1[NI];
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int u = i * LPR2 + lir;
    if ((NU2 % LPR2 == 0) || u < NU2) {
      const int off = (int)(rr * rb) + u * BY;
      sw[i] = __builtin_amdgcn_raw_buffer_load_b32(wr, off, 0, kDlc);
      q0[i] = wld16(wr, off + 4);
      q1[i] = wld16(wr, off + 20);
    }
  }
  if (threadIdx.x == 0)
    while (__hip_atomic_load(&cnt[p], __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) < (unsigned)gu_rb)
      __builtin_amdgcn_s_sleep(1);
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  quant_act<K2, 1, UEL>(a + (size_t)p * K2, aq, asc, res);
  const float ew = wts[p] * (dscale ? dscale[e] : 1.f);
  float acc[1] = {0.f};
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int u = i * LPR2 + lir;
    if ((NU2 % LPR2 == 0) || u < NU2) {
      nv_unit_c<1, K2>(sw[i], q0[i], q1[i], aq, asc, u, 0, acc);
      if (res)
        nv_unit_c<1, K2>(sw[i], q0[i], q1[i], aq + CPU * NU2, asc + K2 / 16, u,
                         0, acc);
    }
  }
#pragma unroll
  for (int off = LPR2 / 2; off > 0; off >>= 1)
    acc[0] += __shfl_xor(acc[0], off, WS);
  if (row < D && lir == 0)
    yp[(size_t)p * D + row] = ew * acc[0];
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned done = __hip_atomic_fetch_add(
        &cnt[npair + p], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (done == (unsigned)dn_rb - 1) {
      cnt[p] = 0;
      cnt[npair + p] = 0;
    }
  }
}

// ---------------- MoE of many rows, grouped by expert ----------------
// Verification steps of 8 (one session) to 64 rows: the (row, slot) pairs
// that chose the same expert are processed by one block row, whose lanes keep
// their weight units in registers for all those pairs, kGrp at a time. Every
// pair's dot products are k_moe_gu's / k_moe_down_e's (row_dot over the same
// units, lanes and shuffles), so each output is the one the per-pair kernels
// write.
constexpr int kGrp = 4;
constexpr int kGgBw = 16;  // waves per grouped-kernel block
// Blocks per row block of the grouped kernels: they loop over the distinct
// experts (20 to 64 per step), so a grid sized for every possible slot (8 per
// row) does not launch hundreds of empty blocks (8-row MoE 5% faster, 16-row
// 10%).
constexpr int kGgSlotBlocks = 24;
// Pairs sorted by expert: ue[k] / ub[k] / un[k] = the k-th used expert, the
// start of its pairs in `sorted` and their count; *nd = used experts.
__global__ void __launch_bounds__(256)
    k_route_group(const int* __restrict__ ids, int P, int* __restrict__ sorted,
                  int* __restrict__ ue, int* __restrict__ ub,
                  int* __restrict__ un, int* __restrict__ nd) {
  __shared__ int cnt[128], off[129], cur[128];
  const int t = threadIdx.x;
  if (t < 128)
    cnt[t] = 0;
  __syncthreads();
  for (int p = t; p < P; p += blockDim.x)
    atomicAdd(&cnt[ids[p]], 1);
  __syncthreads();
  if (t == 0) {
    int run = 0, k = 0;
    for (int e = 0; e < 128; ++e) {
      off[e] = run;
      cur[e] = run;
      if (cnt[e] > 0) {
        ue[k] = e;
        ub[k] = run;
        un[k] = cnt[e];
        ++k;
      }
      run += cnt[e];
    }
    *nd = k;
  }
  __syncthreads();
  for (int p = t; p < P; p += blockDim.x)
    sorted[atomicAdd(&cur[ids[p]], 1)] = p;
}

// f(B) for the first n (1..MAXB) pairs, B fixed at compile time.
template<int FMT, int K, int LPR, int MAXB, class F>
__device__ __forceinline__ void with_rows(int n, F&& f) {
  if constexpr (MAXB >= 8) {
    if (n >= 8) {
      f(std::integral_constant<int, 8>{});
      return;
    }
  }
  if constexpr (MAXB >= 7) {
    if (n == 7) {
      f(std::integral_constant<int, 7>{});
      return;
    }
  }
  if constexpr (MAXB >= 6) {
    if (n == 6) {
      f(std::integral_constant<int, 6>{});
      return;
    }
  }
  if constexpr (MAXB >= 5) {
    if (n == 5) {
      f(std::integral_constant<int, 5>{});
      return;
    }
  }
  if constexpr (MAXB >= 4) {
    if (n == 4) {
      f(std::integral_constant<int, 4>{});
      return;
    }
  }
  if constexpr (MAXB >= 3) {
    if (n == 3) {
      f(std::integral_constant<int, 3>{});
      return;
    }
  }
  if constexpr (MAXB >= 2) {
    if (n == 2) {
      f(std::integral_constant<int, 2>{});
      return;
    }
  }
  f(std::integral_constant<int, 1>{});
}

// Quantized inputs of the grouped kernels, once per step: vector v's int8
// planes (quant_act's values, in its LDS order) and then its two planes'
// scales at qx + v * kQB<K>.
template<int K>
constexpr int kQB = 2 * K + 2 * (K / 16) * 4;
template<int K, int UEL>
__global__ void __launch_bounds__(256)
    k_quant_vec(const float* __restrict__ x, int nv, uint8_t* __restrict__ qx) {
  constexpr int NG = K / 16, NU = K / UEL, CPU = UEL / 16;
  const int it = blockIdx.x * blockDim.x + threadIdx.x;
  if (it >= nv * NG)
    return;
  const int v = it / NG, g = it % NG;
  const float4* xp = (const float4*)(x + (size_t)v * K + g * 16);
  float f[16];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float4 t = xp[i];
    f[4 * i] = t.x;
    f[4 * i + 1] = t.y;
    f[4 * i + 2] = t.z;
    f[4 * i + 3] = t.w;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)),
                             fmaxf(fabsf(t.z), fabsf(t.w))));
  }
  const float sc = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
  float r[16];
  uint32_t q[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float q0 = rintf(f[4 * i] * id), q1 = rintf(f[4 * i + 1] * id),
          q2 = rintf(f[4 * i + 2] * id), q3 = rintf(f[4 * i + 3] * id);
    q[i] = pack4(q0, q1, q2, q3);
    r[4 * i] = f[4 * i] - q0 * sc;
    r[4 * i + 1] = f[4 * i + 1] - q1 * sc;
    r[4 * i + 2] = f[4 * i + 2] - q2 * sc;
    r[4 * i + 3] = f[4 * i + 3] - q3 * sc;
  }
  uint8_t* o = qx + (size_t)v * kQB<K>;
  v4u* q0p = (v4u*)o;
  v4u* q1p = (v4u*)(o + K);
  float* s0 = (float*)(o + 2 * K);
  const int idx = (g % CPU) * NU + g / CPU;
  s0[g] = sc;
  q0p[idx] = v4u{q[0], q[1], q[2], q[3]};
  const float sc2 = sc * (1.0f / 254.0f), id2 = sc > 0.f ? 254.0f / sc : 0.f;
  uint32_t q2p[4];
#pragma unroll
  for (int i = 0; i < 4; i++)
    q2p[i] = pack4(r[4 * i] * id2, r[4 * i + 1] * id2, r[4 * i + 2] * id2,
                   r[4 * i + 3] * id2);
  s0[NG + g] = sc2;
  q1p[idx] = v4u{q2p[0], q2p[1], q2p[2], q2p[3]};
}

// Products of the grouped kernels. Every pair's quantized input is read from
// qx (through the vector caches): no LDS staging or barriers, and a wave runs
// all its expert's pairs, B at a time, with its row's weights decoded once.
// Each product and sum is nv_unit's / q8_unit's in row_dot's order.
template<int B, int K>
__device__ __forceinline__ void nv_unit_g(const NvDec& d,
                                          const uint8_t* const (&qv)[B],
                                          int plane, int u, float (&acc)[B]) {
  constexpr int NU = K / 64, NG = K / 16;
#pragma unroll
  for (int s = 0; s < 4; s++) {
    const uint32_t l0 = d.w[4 * s], l1 = d.w[4 * s + 1], h0 = d.w[4 * s + 2],
                   h1 = d.w[4 * s + 3];
    const float ds = d.ds[s];
#pragma unroll
    for (int b = 0; b < B; b++) {
      const v4u a = ((const v4u*)(qv[b] + plane * K))[s * NU + u];
      const float sc = ((const float*)(qv[b] + 2 * K))[plane * NG + u * 4 + s];
      int t = dot4(l0, a.x, 0);
      t = dot4(l1, a.y, t);
      t = dot4(h0, a.z, t);
      t = dot4(h1, a.w, t);
      acc[b] += ds * sc * (float)t;
    }
  }
}
// row_dot for B pairs. The T units left after the full passes (only T of
// the LPR lanes have one) are split over the lanes: lane lir continues pairs
// lir / T + q * (LPR / T) of lane lir % T, its partial sums moved there and
// back by shuffles, so every lane's sum takes the same terms in the same
// order and the butterfly is row_dot's.
template<int K, int LPR, int B>
__device__ __forceinline__ void row_dot_g(const RowDec<K, LPR>& r,
                                          const uint8_t* const (&qv)[B],
                                          int res, int lir, float (&acc)[B]) {
  using R = RowDec<K, LPR>;
  constexpr int NF = R::NU / LPR, T = R::NU % LPR, PL = LPR / T,
                TPL = (B + PL - 1) / PL;  // tail tasks per lane
  static_assert(T > 0 && LPR % T == 0, "tail split");
#pragma unroll
  for (int b = 0; b < B; b++)
    acc[b] = 0.f;
#pragma unroll
  for (int i = 0; i < NF; ++i) {
    const int u = i * LPR + lir;
    nv_unit_g<B, K>(r.d[i], qv, 0, u, acc);
    if (res)
      nv_unit_g<B, K>(r.d[i], qv, 1, u, acc);
  }
  // lane lir continues pairs lir / T + q * PL of lane lir % T
  const int base = (threadIdx.x % WS) - lir, tl = lir % T, tb = lir / T;
  float t1[TPL][1];
  const uint8_t* q1[TPL][1];
#pragma unroll
  for (int q = 0; q < TPL; ++q) {
    t1[q][0] = 0.f;
    q1[q][0] = qv[0];
  }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float x = __shfl(acc[b], base + tl, WS);
    if (tb == b % PL) {
      t1[b / PL][0] = x;
      q1[b / PL][0] = qv[b];
    }
  }
  const int u = NF * LPR + tl;
#pragma unroll
  for (int q = 0; q < TPL; ++q)
    if (tb + q * PL < B) {
      nv_unit_g<1, K>(r.d[R::NI - 1], q1[q], 0, u, t1[q]);
      if (res)
        nv_unit_g<1, K>(r.d[R::NI - 1], q1[q], 1, u, t1[q]);
    }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float x = __shfl(t1[b / PL][0], base + (b % PL) * T + tl, WS);
    if (lir < T)
      acc[b] = x;
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      acc[b] += __shfl_xor(acc[b], off, WS);
}

// Q8_0 counterpart of nv_unit_g / row_dot_g (q8_unit_c's products).
template<int B, int K>
__device__ __forceinline__ void q8_unit_g(v4u w0, v4u w1, float d,
                                          const uint8_t* const (&qv)[B],
                                          int plane, int u, float (&acc)[B]) {
  constexpr int NU = K / 32, NG = K / 16;
#pragma unroll
  for (int b = 0; b < B; b++) {
    const v4u* aq = (const v4u*)(qv[b] + plane * K);
    const v4u a0 = aq[u], a1 = aq[NU + u];
    const float* sc = (const float*)(qv[b] + 2 * K) + plane * NG + 2 * u;
    int s0 = dot4(w0.x, a0.x, 0);
    s0 = dot4(w0.y, a0.y, s0);
    s0 = dot4(w0.z, a0.z, s0);
    s0 = dot4(w0.w, a0.w, s0);
    int s1 = dot4(w1.x, a1.x, 0);
    s1 = dot4(w1.y, a1.y, s1);
    s1 = dot4(w1.z, a1.z, s1);
    s1 = dot4(w1.w, a1.w, s1);
    acc[b] += d * (sc[0] * (float)s0 + sc[1] * (float)s1);
  }
}
template<int K, int LPR, int B>
__device__ __forceinline__ void row_dot_q8g(const RowRegs<FMT_Q8, K, LPR>& r,
                                            const uint8_t* const (&qv)[B],
                                            int res, int lir, float (&acc)[B]) {
  using R = RowRegs<FMT_Q8, K, LPR>;
  constexpr int NF = R::NU / LPR, T = R::NU % LPR, PL = LPR / T,
                TPL = (B + PL - 1) / PL;
  static_assert(T > 0 && LPR % T == 0, "tail split");
  auto dq = [&](int i) {
    return __half2float(__ushort_as_half((unsigned short)(r.h[i] & 0xFFFFu)));
  };
#pragma unroll
  for (int b = 0; b < B; b++)
    acc[b] = 0.f;
#pragma unroll
  for (int i = 0; i < NF; ++i) {
    const int u = i * LPR + lir;
    const float d = dq(i);
    q8_unit_g<B, K>(r.q0[i], r.q1[i], d, qv, 0, u, acc);
    if (res)
      q8_unit_g<B, K>(r.q0[i], r.q1[i], d, qv, 1, u, acc);
  }
  const int base = (threadIdx.x % WS) - lir, tl = lir % T, tb = lir / T;
  float t1[TPL][1];
  const uint8_t* q1[TPL][1];
#pragma unroll
  for (int q = 0; q < TPL; ++q) {
    t1[q][0] = 0.f;
    q1[q][0] = qv[0];
  }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float x = __shfl(acc[b], base + tl, WS);
    if (tb == b % PL) {
      t1[b / PL][0] = x;
      q1[b / PL][0] = qv[b];
    }
  }
  const int u = NF * LPR + tl;
  const float d = dq(R::NI - 1);
#pragma unroll
  for (int q = 0; q < TPL; ++q)
    if (tb + q * PL < B) {
      q8_unit_g<1, K>(r.q0[R::NI - 1], r.q1[R::NI - 1], d, q1[q], 0, u, t1[q]);
      if (res)
        q8_unit_g<1, K>(r.q0[R::NI - 1], r.q1[R::NI - 1], d, q1[q], 1, u,
                        t1[q]);
    }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const float x = __shfl(t1[b / PL][0], base + (b % PL) * T + tl, WS);
    if (lir < T)
      acc[b] = x;
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      acc[b] += __shfl_xor(acc[b], off, WS);
}

template<int FMT, int K, int LPR, int BW>
__global__ void __launch_bounds__(BW* WS)
    k_moe_gu_gg(const uint8_t* __restrict__ W, const float* __restrict__ escale,
                const uint8_t* __restrict__ qx, const int* __restrict__ sorted,
                const int* __restrict__ ue, const int* __restrict__ ub,
                const int* __restrict__ un, const int* __restrict__ nd,
                float* __restrict__ a, int FF, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, RPP = WS / LPR / 2;
  // A launch has kGgSlotBlocks blocks per row block, each looping over the
  // step's distinct experts: most slots of a wide grid would be empty.
  const int rbk = blockIdx.y, ndv = *nd;
  for (int k = blockIdx.x; k < ndv; k += gridDim.x) {
    const int e = ue[k] & 127, b0 = ub[k], n = un[k];
    const size_t rb = (size_t)NU * Fm<FMT>::BYTES;
    const uint8_t* We = W + (size_t)e * 2 * FF * rb;
    const int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
              lir = lane % LPR, isup = g & 1, pg = g >> 1;
    const int j = (rbk * BW + wave) * RPP + pg;
    const int jj = j < FF ? j : FF - 1;
    RowRegs<FMT, K, LPR> wr;
    row_load_dl<FMT, K, LPR, true>(wrsrc(We), (int)(rb * (isup ? FF + jj : jj)),
                                   lir, wr);
    [[maybe_unused]] RowDec<K, LPR> wd;
    if constexpr (FMT == FMT_NVFP4)
      row_decode<K, LPR, true>(wr, lir, wd);
    const float sc = escale ? escale[e] : 1.f;
    for (int c0 = 0; c0 < n; c0 += kGrp) {
      const int cn = min(kGrp, n - c0);
      int ps[kGrp];
      const uint8_t* qv[kGrp];
#pragma unroll
      for (int b = 0; b < kGrp; ++b) {
        ps[b] = sorted[b0 + c0 + min(b, cn - 1)];
        qv[b] = qx + (size_t)(ps[b] >> 3) * kQB<K>;
      }
      with_rows<FMT, K, LPR, kGrp>(cn, [&](auto nb_) {
        constexpr int B = decltype(nb_)::value;
        const uint8_t* qb[B];
#pragma unroll
        for (int b = 0; b < B; ++b)
          qb[b] = qv[b];
        float acc[B];
        if constexpr (FMT == FMT_NVFP4)
          row_dot_g<K, LPR, B>(wd, qb, res, lir, acc);
        else
          row_dot_q8g<K, LPR, B>(wr, qb, res, lir, acc);
        // after the butterfly every lane of a row holds its sums: gate lane b
        // takes pair b (one SITU per pair instead of one per pair and lane)
        float gv = acc[0], uv = __shfl_xor(acc[0], LPR, WS);
        int pb = ps[0];
#pragma unroll
        for (int b = 1; b < B; ++b) {
          const float o = __shfl_xor(acc[b], LPR, WS);
          if (lir == b) {
            gv = acc[b];
            uv = o;
            pb = ps[b];
          }
        }
        if (!isup && lir < B && j < FF)
          a[(size_t)pb * FF + j] = situ(gv * sc, uv * sc);
      });
    }
  }
}

template<int FMT, int K, int LPR, int BW>
__global__ void __launch_bounds__(BW* WS)
    k_moe_down_gg(const uint8_t* __restrict__ W,
                  const float* __restrict__ escale,
                  const uint8_t* __restrict__ qx,
                  const int* __restrict__ sorted, const int* __restrict__ ue,
                  const int* __restrict__ ub, const int* __restrict__ un,
                  const int* __restrict__ nd, const float* __restrict__ wts,
                  float* __restrict__ yp, int D, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, RPW = WS / LPR;
  // A launch has kGgSlotBlocks blocks per row block, each looping over the
  // step's distinct experts: most slots of a wide grid would be empty.
  const int rbk = blockIdx.y, ndv = *nd;
  for (int k = blockIdx.x; k < ndv; k += gridDim.x) {
    const int e = ue[k] & 127, b0 = ub[k], n = un[k];
    const size_t rb = (size_t)NU * Fm<FMT>::BYTES;
    const int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR,
              lir = lane % LPR;
    const int row = (rbk * BW + wave) * RPW + g;
    const int rr = row < D ? row : D - 1;
    RowRegs<FMT, K, LPR> wr;
    row_load_dl<FMT, K, LPR, true>(wrsrc(W + (size_t)e * D * rb),
                                   (int)(rr * rb), lir, wr);
    [[maybe_unused]] RowDec<K, LPR> wd;
    if constexpr (FMT == FMT_NVFP4)
      row_decode<K, LPR, true>(wr, lir, wd);
    const float es = escale ? escale[e] : 1.f;
    for (int c0 = 0; c0 < n; c0 += kGrp) {
      const int cn = min(kGrp, n - c0);
      int ps[kGrp];
      const uint8_t* qv[kGrp];
#pragma unroll
      for (int b = 0; b < kGrp; ++b) {
        ps[b] = sorted[b0 + c0 + min(b, cn - 1)];
        qv[b] = qx + (size_t)ps[b] * kQB<K>;
      }
      with_rows<FMT, K, LPR, kGrp>(cn, [&](auto nb_) {
        constexpr int B = decltype(nb_)::value;
        const uint8_t* qb[B];
#pragma unroll
        for (int b = 0; b < B; ++b)
          qb[b] = qv[b];
        float acc[B];
        if constexpr (FMT == FMT_NVFP4)
          row_dot_g<K, LPR, B>(wd, qb, res, lir, acc);
        else
          row_dot_q8g<K, LPR, B>(wr, qb, res, lir, acc);
        if (row < D && lir == 0)
#pragma unroll
          for (int b = 0; b < B; ++b) {
            const int p = ps[b];
            yp[(size_t)p * D + row] = (wts[p] * es) * acc[b];
          }
      });
    }
  }
}

// ---------------- small ops ----------------
__device__ __forceinline__ float block_sum(float v, float* red /*[32]*/) {
  const int wid = threadIdx.x / 32, lane = threadIdx.x % 32,
            nw = blockDim.x / 32;
  v = wsum(v);
  __syncthreads();
  if (lane == 0)
    red[wid] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i)
    t += red[i];
  return t;
}

// post-conv + residual: out = (x + w . history) * scale; block: blk = cur, cur
// = out; else cur += out.
__device__ __forceinline__ void conv_resid_elem(const ConvResidArgs& a, int c,
                                                int tok = 0) {
  const size_t to = (size_t)tok * 2304;
  float* st = a.slot[tok * 48].conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
  auto input = [&](int r) {
    float xc;
    if (a.nsum == 8) {
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; ++e)
        v[e] = a.x[((size_t)r * 8 + e) * 2304 + c];
      xc = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        xc += v[e];
    } else if (a.nsum > 1) {
      xc = 0.f;
      for (int e = 0; e < a.nsum; ++e)
        xc += a.x[((size_t)r * a.nsum + e) * 2304 + c];
    } else {
      xc = a.x[(size_t)r * 2304 + c];
    }
    return xc;
  };
  const float xc = input(tok);
  float xin[4] = {st[0], st[1], st[2], xc};
  if (a.cap) {
    const int j = a.ri ? a.ri[tok] : tok;  // rows of this session before tok
    for (int l = 1; l <= 3 && l <= j; ++l)
      xin[3 - l] = input(tok - l);
    if (j < 3)
      for (int l = j + 1; l <= 3; ++l)
        xin[3 - l] = st[3 - l + j];
    a.cap[(size_t)tok * 6656 + a.conv_off + c] = xc;
  }
  float y = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    y += a.w[(size_t)c * 4 + i] * xin[i];
  if (!a.cap) {
    st[0] = xin[1];
    st[1] = xin[2];
    st[2] = xin[3];
  }
  const float o = (xin[3] + y) * a.scale;
  const float cv = a.cur[to + c];
  if (a.blk) {
    a.blk[to + c] = cv;
    a.cur[to + c] = o;
  } else {
    a.cur[to + c] = cv + o;
  }
}
// The same for a whole 512-thread block (k_mix prologue): every load of the 5
// elements per thread is issued before the first one is consumed, instead of
// one dependent chain per element.
__device__ __forceinline__ void conv_resid_block(const ConvResidArgs& a,
                                                 int tok) {
  constexpr int R = 5;  // ceil(2304 / 512)
  const size_t to = (size_t)tok * 2304;
  const SlotEntry* sl = a.slot + tok * 48;
  float xin[R][4], wv[R][4], cv[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      const float* st = sl->conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
      xin[r][0] = st[0];
      xin[r][1] = st[1];
      xin[r][2] = st[2];
      auto input = [&](int row) {
        float xc;
        if (a.nsum == 8) {
          float v[8];
#pragma unroll
          for (int e = 0; e < 8; ++e)
            v[e] = a.x[((size_t)row * 8 + e) * 2304 + c];
          xc = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e)
            xc += v[e];
        } else {
          xc = a.x[(size_t)row * 2304 + c];
        }
        return xc;
      };
      const float xc = input(tok);
      xin[r][3] = xc;
      if (a.cap) {
        const int j = a.ri ? a.ri[tok] : tok;
        for (int l = 1; l <= 3 && l <= j; ++l)
          xin[r][3 - l] = input(tok - l);
        if (j < 3)
          for (int l = j + 1; l <= 3; ++l)
            xin[r][3 - l] = st[3 - l + j];
        a.cap[(size_t)tok * 6656 + a.conv_off + c] = xc;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        wv[r][i] = a.w[(size_t)c * 4 + i];
      cv[r] = a.cur[to + c];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      float* st = sl->conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
      float y = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        y += wv[r][i] * xin[r][i];
      if (!a.cap) {
        st[0] = xin[r][1];
        st[1] = xin[r][2];
        st[2] = xin[r][3];
      }
      const float o = (xin[r][3] + y) * a.scale;
      if (a.blk) {
        a.blk[to + c] = cv[r];
        a.cur[to + c] = o;
      } else {
        a.cur[to + c] = cv[r] + o;
      }
    }
  }
}
__global__ void __launch_bounds__(256) k_conv_resid(ConvResidArgs a) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c < 2304)
    conv_resid_elem(a, c, blockIdx.z);
}

// Attention-residual mix, one block (16 waves) per query row. Optional
// prologue: the previous op's post-conv + residual (single-row jobs); optional
// epilogue: RMS norm with weights.
__global__ void __launch_bounds__(512) k_mix(MixJob j, float eps) {
  __shared__ float sc[8], coef[8], red[16];
  const int q = blockIdx.x, tok = blockIdx.y, wave = threadIdx.x / 32,
            lane = threadIdx.x % 32;
  const size_t toff = (size_t)tok * 2304;
  if (j.has_cr) {
    conv_resid_block(j.cr, tok);
    __syncthreads();
  }
  const float* qv = j.query + (size_t)q * 2304;
  if (wave < j.n) {
    const float4* xp = (const float4*)(j.st[wave] + toff);
    const float4* qp = (const float4*)qv;
    float ss = 0.f, d = 0.f;
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      const float4 x = xp[lane + 32 * i], y = qp[lane + 32 * i];
      ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
      d += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    }
    ss = wsum(ss);
    d = wsum(d);
    if (lane == 0) {
      const float inv = rsqrtf(ss * (1.0f / 2304.0f) + eps);
      sc[wave] = d * inv * 0.020833333f;  // 1/sqrt(2304) = 1/48
      coef[wave] = inv;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int i = 0; i < j.n; ++i)
      m = fmaxf(m, sc[i]);
    float z = 0.f;
    for (int i = 0; i < j.n; ++i) {
      sc[i] = expf(sc[i] - m);
      z += sc[i];
    }
    for (int i = 0; i < j.n; ++i)
      sc[i] = sc[i] / z * coef[i];
  }
  __syncthreads();
  float4 a[2];
  float ssq = 0.f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = threadIdx.x + r * 512;
    a[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < 576) {
      for (int k = 0; k < j.n; ++k) {
        const float4 x = ((const float4*)(j.st[k] + toff))[i];
        a[r].x += sc[k] * x.x;
        a[r].y += sc[k] * x.y;
        a[r].z += sc[k] * x.z;
        a[r].w += sc[k] * x.w;
      }
      ssq +=
          a[r].x * a[r].x + a[r].y * a[r].y + a[r].z * a[r].z + a[r].w * a[r].w;
    }
  }
  float scale = 1.0f;
  const float* nw = q < 4 ? j.nw[q] : nullptr;  // up to 4 normed queries
  if (nw) {
    ssq = wsum(ssq);
    if (lane == 0)
      red[wave] = ssq;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 16; ++k)
      t += red[k];
    scale = rsqrtf(t * (1.0f / 2304.0f) + eps);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = threadIdx.x + r * 512;
    if (i < 576) {
      float4 o = a[r];
      if (nw) {
        const float4 w = ((const float4*)nw)[i];
        o.x *= scale * w.x;
        o.y *= scale * w.y;
        o.z *= scale * w.z;
        o.w *= scale * w.w;
      }
      float* ob = q < 4 && j.outq[q]
                      ? j.outq[q] + toff
                      : j.out +
                            (j.ostride ? (size_t)tok * j.ostride
                                       : (size_t)tok * j.nq * 2304) +
                            (size_t)q * 2304;
      ((float4*)ob)[i] = o;
      if (q < 4 && j.outh[q]) {
        _Float16* hb = static_cast<_Float16*>(j.outh[q]) + toff + 4 * i;
        hb[0] = (_Float16)o.x;
        hb[1] = (_Float16)o.y;
        hb[2] = (_Float16)o.z;
        hb[3] = (_Float16)o.w;
      }
    }
  }
}

struct NormP {
  NormJobs j;
};
__device__ __forceinline__ void rms_job(const NormJobs& j, int r, int tk,
                                        float eps, float* red /*[32]*/) {
  const float* x = j.x[r] + (size_t)tk * j.xs[r];
  const float* w = j.w[r];
  float* o = j.out[r] + (size_t)tk * j.os[r];
  // 256 threads (both launches): element i = threadIdx.x + 256 t, the
  // loads issued together, each thread's sum in t order
  float v[9], ss = 0.f;
#pragma unroll
  for (int t = 0; t < 9; ++t)
    v[t] = x[threadIdx.x + 256 * t];
#pragma unroll
  for (int t = 0; t < 9; ++t)
    ss = __fmaf_rn(v[t], v[t], ss);
  const float tot = block_sum(ss, red);
  const float inv = rsqrtf(tot * (1.0f / 2304.0f) + eps);
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int i = threadIdx.x + 256 * t;
    o[i] = v[t] * inv * (w ? w[i] : 1.0f);
  }
}
__global__ void __launch_bounds__(256) k_rms(NormJobs j, float eps) {
  __shared__ float red[32];
  rms_job(j, blockIdx.x, blockIdx.y, eps, red);
}

// Loop controller gate: g64[row] = down[row] . [x/rms(x); msg/rms(msg)]. One
// block per (row, stream).
__device__ __forceinline__ float4
ld4w(const void* base, size_t i4,
     bool h) {  // 4 weights starting at element 4*i4
  if (!h)
    return ((const float4*)base)[i4];
  // BF16 copies of BF16-exact F32 weights: widening is a shift, so the values
  // are the file's.
  const uint2 u = ((const uint2*)base)[i4];
  return make_float4(
      __uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
      __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
}
template<bool H>
__global__ void __launch_bounds__(256) k_ctl_gate(CtlJobs j, float eps) {
  __shared__ float red[4][8];
  const int row = blockIdx.x, s = blockIdx.y, tk = blockIdx.z;
  const float4* x = (const float4*)(j.x[s] + (size_t)tk * j.xs[s]);
  const float4* m = (const float4*)(j.msg[s] + (size_t)tk * j.ms[s]);
  const size_t drow =
      (size_t)row * 1152;  // 4608 / 4 groups of four weights per row
  float sx = 0.f, sm = 0.f, dx = 0.f, dm = 0.f;
  for (int i = threadIdx.x; i < 576; i += blockDim.x) {
    const float4 xv = x[i], mv = m[i], d0 = ld4w(j.down[s], drow + i, H),
                 d1 = ld4w(j.down[s], drow + 576 + i, H);
    sx += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    sm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
    dx += d0.x * xv.x + d0.y * xv.y + d0.z * xv.z + d0.w * xv.w;
    dm += d1.x * mv.x + d1.y * mv.y + d1.z * mv.z + d1.w * mv.w;
  }
  sx = wsum(sx);
  sm = wsum(sm);
  dx = wsum(dx);
  dm = wsum(dm);
  const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    red[0][wid] = sx;
    red[1][wid] = sm;
    red[2][wid] = dx;
    red[3][wid] = dm;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < 8; ++w)
      for (int k = 0; k < 4; ++k)
        t[k] += red[k][w];
    j.g64[s][(size_t)tk * j.gs[s] + row] =
        t[2] * rsqrtf(t[0] * (1.0f / 2304.0f) + eps) +
        t[3] * rsqrtf(t[1] * (1.0f / 2304.0f) + eps);
  }
}
template<bool H>
__device__ __forceinline__ void ctl_apply_elem(const CtlJobs& j, int i, int s,
                                               int tk) {
  const float4* g = (const float4*)(j.g64[s] + (size_t)tk * j.gs[s]);
  float ga = 0.f, fo = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float4 gv = g[k], a = ld4w(j.gain[s], (size_t)i * 16 + k, H),
                 b = ld4w(j.forget[s], (size_t)i * 16 + k, H);
    ga += a.x * gv.x + a.y * gv.y + a.z * gv.z + a.w * gv.w;
    fo += b.x * gv.x + b.y * gv.y + b.z * gv.z + b.w * gv.w;
  }
  const float cv = j.x[s][(size_t)tk * j.xs[s] + i];
  j.out[s][(size_t)tk * j.os[s] + i] =
      cv - tanhf(fo) * cv + tanhf(ga) * j.msg[s][(size_t)tk * j.ms[s] + i];
}
// ctl_apply_elem, then (last block of a stream and token) k_rms of norm job s.
template<bool H>
__global__ void __launch_bounds__(256)
    k_ctl_apply_rms(CtlJobs j, NormJobs nj, unsigned* cnt, float eps) {
  __shared__ float red[32];
  __shared__ unsigned last;
  const int i = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y, tk = blockIdx.z;
  if (i < 2304)
    ctl_apply_elem<H>(j, i, s, tk);
  __threadfence();
  __syncthreads();
  unsigned* c = cnt + (size_t)tk * 4 + s;
  if (threadIdx.x == 0)
    last = atomicAdd(c, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last)
    return;
  __threadfence();
  if (threadIdx.x == 0)
    *c = 0;
  rms_job(nj, s, tk, eps, red);
}

// y[r] = W[r] . x for a few small matrices (K = 2304). One wave per row.
__device__ __forceinline__ float ld_w(const void* w, int type, size_t i) {
  if (type == 0)
    return ((const float*)w)[i];
  if (type == 1)
    return __half2float(((const __half*)w)[i]);
  return __uint_as_float((uint32_t)((const uint16_t*)w)[i] << 16);
}
__global__ void __launch_bounds__(256) k_tiny(TinyJobs j) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * 8 + wave;
  int m = 0;
  while (m < j.n && row >= j.rows[m]) {
    row -= j.rows[m];
    ++m;
  }
  if (m >= j.n)
    return;
  const void* W = j.w[m];
  const int ty = j.type[m];
  const float* x = j.x[m] + (size_t)blockIdx.y * j.xs[m];
  float s = 0.f;
  for (int i = lane; i < 2304; i += 32)
    s += ld_w(W, ty, (size_t)row * 2304 + i) * x[i];
  s = wsum(s);
  if (lane == 0)
    j.y[m][(size_t)blockIdx.y * j.ys[m] + row] = s;
}

// K/V conv (+ conv-state update), K clip / V rms, f16 KV write of kv head
// `kvh`, token `tk`, by a block of 128 threads.
__device__ __forceinline__ void kvprep_block(const KvPrepArgs& a, int kvh,
                                             int tk, float* red /*[32]*/) {
  const int d = threadIdx.x, c = kvh * 128 + d;
  const int pos = a.pos[tk];
  const SlotEntry* slot = a.slot + tk * 48;
  const float* ak = a.k + (size_t)tk * a.ld;
  const float* av = a.v + (size_t)tk * a.ld;
  float* sk = slot->conv + (size_t)a.conv_off_k * 3 + (size_t)c * 3;
  float* sv = slot->conv + (size_t)a.conv_off_v * 3 + (size_t)c * 3;
  float kx[4] = {sk[0], sk[1], sk[2], ak[c]};
  float vx[4] = {sv[0], sv[1], sv[2], av[c]};
  if (a.cap) {
    const int j = a.ri ? a.ri[tk] : tk;
    for (int l = 1; l <= 3 && l <= j; ++l) {
      kx[3 - l] = a.k[(size_t)(tk - l) * a.ld + c];
      vx[3 - l] = a.v[(size_t)(tk - l) * a.ld + c];
    }
    if (j < 3)
      for (int l = j + 1; l <= 3; ++l) {
        kx[3 - l] = sk[3 - l + j];
        vx[3 - l] = sv[3 - l + j];
      }
    a.cap[(size_t)tk * 6656 + a.conv_off_k + c] = kx[3];
    a.cap[(size_t)tk * 6656 + a.conv_off_v + c] = vx[3];
  }
  float ky = 0.f, vy = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ky += a.wk[(size_t)c * 4 + i] * kx[i];
    vy += a.wv[(size_t)c * 4 + i] * vx[i];
  }
  if (a.dk) {
    const float* dk = a.dk + (size_t)tk * a.ldd;
    const float* dv = a.dv + (size_t)tk * a.ldd;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
      ky += dk[kvh * 4 + lag] * kx[3 - lag];
      vy += dv[kvh * 4 + lag] * vx[3 - lag];
    }
  }
  if (!a.cap) {
    sk[0] = kx[1];
    sk[1] = kx[2];
    sk[2] = kx[3];
    sv[0] = vx[1];
    sv[1] = vx[2];
    sv[2] = vx[3];
  }
  const float kc = (kx[3] + ky) * a.scale, vc = (vx[3] + vy) * a.scale;
  const float kss = block_sum(kc * kc, red);
  const float vss = block_sum(vc * vc, red);
  const float kf = fminf(1.0f, a.qk_thr / (sqrtf(kss) + 1e-6f));
  const float vf = rsqrtf(vss * (1.0f / 128.0f) + a.eps);
  const int cell = pos % slot->cap;
  ((__half*)slot->k)[(size_t)cell * 1024 + c] = __float2half(kc * kf);
  ((__half*)slot->v)[(size_t)cell * 1024 + c] = __float2half(vc * vf);
}
// One block per kv head.
__global__ void __launch_bounds__(128) k_kvprep(KvPrepArgs a) {
  __shared__ float red[32];
  kvprep_block(a, blockIdx.x, blockIdx.z, red);
}

// single-block counting sort of (token, slot) pairs by expert
__global__ void __launch_bounds__(1024)
    k_route_build(const int* ids, int P, int E, int* off, int* sorted) {
  __shared__ int cnt[256], cur[256];
  for (int e = threadIdx.x; e < E; e += blockDim.x)
    cnt[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x)
    atomicAdd(&cnt[ids[p]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int e = 0; e < E; ++e) {
      off[e] = run;
      cur[e] = run;
      run += cnt[e];
    }
    off[E] = run;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x)
    sorted[atomicAdd(&cur[ids[p]], 1)] = p;
}

// ---------------- prefill: tiled SIMT kernels for the skinny F32/BF16 matrices
// ---------------- out[t][r] = sum_k W[r][k] * in[t][k] with the input read
// from up to two segments (x, msg), each RMS-normalised per token when `rms`.
struct GateP {
  const float* x;
  int ldx;
  const float* msg;
  int ldm;
  const void* W;
  int ldw;
  int k1, k2, rms;
  float* out;
  int ldo;
  int rows;  // valid output rows of the tile range (<= 64 per tile; 0 = all)
  const float* inv2 = nullptr;  // precomputed per-token msg inverse RMS
  const float* inv1 = nullptr;  // precomputed per-token x inverse RMS
  int half = 0;                 // 1: BF16 weights (exact F32 high 16 bits)
};
struct GateJobs {
  GateP j[3];
};

__device__ __forceinline__ float4 load_w4(const void* W, size_t off, int half) {
  if (!half)
    return *(const float4*)(static_cast<const float*>(W) + off);
  const uint2 u = *(const uint2*)(static_cast<const uint16_t*>(W) + off);
  return make_float4(
      __uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
      __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
}

// Register-tiled: a block computes 64 tokens x 64 rows, each thread 4 x 4 of
// them (rows tr + 16i, tokens tt + 16j), accumulating k in ascending order.
__global__ void __launch_bounds__(256)
    k_gate_p64(GateJobs jobs, int T, float eps) {
  const GateP& p = jobs.j[blockIdx.z];
  __shared__ float Wt[64][33];
  __shared__ float xs[64][33];
  __shared__ float inv[2][64];
  const int tid = threadIdx.x, t0 = blockIdx.x * 64, r0 = blockIdx.y * 64;
  const int wave = tid / 32, lane = tid % 32;
  for (int q = 0; q < 8; ++q) {
    const int tk = wave * 8 + q;
    const int t = min(t0 + tk, T - 1);
    float s1 = 1.f, s2 = 1.f;
    if (p.rms) {
      float a = 0.f, b = 0.f;
      if (!p.inv1)
        for (int i = lane; i < p.k1; i += 32) {
          float v = p.x[(size_t)t * p.ldx + i];
          a += v * v;
        }
      if (!p.inv2)
        for (int i = lane; i < p.k2; i += 32) {
          float v = p.msg[(size_t)t * p.ldm + i];
          b += v * v;
        }
      a = wsum(a);
      b = wsum(b);
      s1 = p.inv1 ? p.inv1[t] : rsqrtf(a / p.k1 + eps);
      s2 = p.inv2 ? p.inv2[t] : p.k2 ? rsqrtf(b / p.k2 + eps) : 1.f;
    }
    if (lane == 0) {
      inv[0][tk] = s1;
      inv[1][tk] = s2;
    }
  }
  const int tr = tid & 15, tt = tid >> 4;
  float acc[4][4] = {};
  const int ktot = p.k1 + p.k2;
  for (int kc = 0; kc < ktot; kc += 32) {
    __syncthreads();
    for (int i = tid; i < 64 * 8; i += 256) {
      const int row = i >> 3, c4 = i & 7;
      const float4 v =
          load_w4(p.W, (size_t)(r0 + row) * p.ldw + kc + c4 * 4, p.half);
      Wt[row][c4 * 4] = v.x;
      Wt[row][c4 * 4 + 1] = v.y;
      Wt[row][c4 * 4 + 2] = v.z;
      Wt[row][c4 * 4 + 3] = v.w;
    }
    const bool seg2 = kc >= p.k1;
    const float* src = seg2 ? p.msg : p.x;
    const int ld = seg2 ? p.ldm : p.ldx, kk = seg2 ? kc - p.k1 : kc;
    for (int i = tid; i < 64 * 8; i += 256) {
      const int tk = i >> 3, c4 = i & 7;
      const int t = min(t0 + tk, T - 1);
      const float4 v = *(const float4*)(src + (size_t)t * ld + kk + c4 * 4);
      const float sc = inv[seg2 ? 1 : 0][tk];
      xs[tk][c4 * 4] = v.x * sc;
      xs[tk][c4 * 4 + 1] = v.y * sc;
      xs[tk][c4 * 4 + 2] = v.z * sc;
      xs[tk][c4 * 4 + 3] = v.w * sc;
    }
    __syncthreads();
#pragma unroll 4
    for (int k = 0; k < 32; ++k) {
      float w[4], x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        w[i] = Wt[tr + 16 * i][k];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        x[j] = xs[tt + 16 * j][k];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j] += w[i] * x[j];
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = r0 + tr + 16 * i;
    if (p.rows && r >= p.rows)
      continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t0 + tt + 16 * j;
      if (t < T)
        p.out[(size_t)t * p.ldo + r] = acc[i][j];
    }
  }
}

// Same products as k_gate_p64 (every output accumulates k in ascending order
// with the same fused multiply-adds, inputs scaled by the same inverse RMS), so
// the results are identical. Faster: a thread keeps TT/16 tokens x RB/16 rows
// and reads each operand as a float4 along k from LDS (one read per four
// k-steps instead of four), the next 32-k chunk is fetched into registers
// while the current one is consumed from the other LDS buffer (one barrier per
// chunk).
template<int TT, int RB>
__global__ void __launch_bounds__(256)
    k_gate_v2(GateJobs jobs, int T, float eps) {
  constexpr int KC = 32, LS = KC + 4, NT = TT / 16, NR = RB / 16;
  // float4 loads per thread and chunk (XL = 1 with TT = 16: half the threads)
  constexpr int WL = RB * KC / 4 / 256,
                XL = TT * KC / 4 >= 256 ? TT * KC / 4 / 256 : 1;
  static_assert(WL >= 1, "tile too small");
  const GateP& p = jobs.j[blockIdx.z];
  __shared__ float Ws[2][RB * LS];
  __shared__ float Xs[2][TT * LS];
  __shared__ float inv[2][TT];
  const int tid = threadIdx.x, t0 = blockIdx.x * TT, r0 = blockIdx.y * RB;
  const int wave = tid / 32, lane = tid % 32;
  {
    constexpr int TW = TT / 8;
    for (int q = 0; q < TW; ++q) {
      const int tk = wave * TW + q;
      const int t = min(t0 + tk, T - 1);
      float s1 = 1.f, s2 = 1.f;
      if (p.rms) {
        float a = 0.f, b = 0.f;
        if (!p.inv1) {
          const float* xr = p.x + (size_t)t * p.ldx;
          for (int i = lane; i < p.k1; i += 32) {
            const float v = xr[i];
            a += v * v;
          }
        }
        if (!p.inv2) {
          const float* mr = p.msg + (size_t)t * p.ldm;
          for (int i = lane; i < p.k2; i += 32) {
            const float v = mr[i];
            b += v * v;
          }
        }
        const float sa = wsum(a), sb = wsum(b);
        s1 = p.inv1 ? p.inv1[t] : rsqrtf(sa / p.k1 + eps);
        s2 = p.inv2 ? p.inv2[t] : p.k2 ? rsqrtf(sb / p.k2 + eps) : 1.f;
      }
      if (lane == 0) {
        inv[0][tk] = s1;
        inv[1][tk] = s2;
      }
    }
  }
  __syncthreads();
  const int ktot = p.k1 + p.k2, nc = ktot / KC;
  float4 wr[WL], xv[XL];
  auto gload = [&](int c) {
    const int kc = c * KC;
#pragma unroll
    for (int m = 0; m < WL; ++m) {
      const int i = tid + 256 * m, row = i >> 3, c4 = i & 7;
      wr[m] = load_w4(p.W, (size_t)(r0 + row) * p.ldw + kc + c4 * 4, p.half);
    }
    const bool seg2 = kc >= p.k1;
    const float* src = seg2 ? p.msg : p.x;
    const int ld = seg2 ? p.ldm : p.ldx, kk = seg2 ? kc - p.k1 : kc;
#pragma unroll
    for (int m = 0; m < XL; ++m) {
      const int i = tid + 256 * m, tk = i >> 3, c4 = i & 7;
      const int t = min(t0 + min(tk, TT - 1), T - 1);
      xv[m] = *(const float4*)(src + (size_t)t * ld + kk + c4 * 4);
    }
  };
  auto sstore = [&](int c, int buf) {
    const bool seg2 = c * KC >= p.k1;
#pragma unroll
    for (int m = 0; m < WL; ++m) {
      const int i = tid + 256 * m, row = i >> 3, c4 = i & 7;
      *(float4*)&Ws[buf][row * LS + c4 * 4] = wr[m];
    }
#pragma unroll
    for (int m = 0; m < XL; ++m) {
      const int i = tid + 256 * m, tk = i >> 3, c4 = i & 7;
      if (tk >= TT)
        break;
      const float sc = inv[seg2 ? 1 : 0][tk];
      float4 v = xv[m];
      v.x = v.x * sc;
      v.y = v.y * sc;
      v.z = v.z * sc;
      v.w = v.w * sc;
      *(float4*)&Xs[buf][tk * LS + c4 * 4] = v;
    }
  };
  const int tr = tid & 15, tt = tid >> 4;
  float acc[NR][NT];
#pragma unroll
  for (int i = 0; i < NR; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
      acc[i][j] = 0.f;
  gload(0);
  sstore(0, 0);
  __syncthreads();
  for (int c = 0; c < nc; ++c) {
    const int buf = c & 1;
    if (c + 1 < nc)
      gload(c + 1);
    const float* wb = Ws[buf];
    const float* xb = Xs[buf];
#pragma unroll 2
    for (int k4 = 0; k4 < KC; k4 += 4) {
      float4 w4[NR], x4[NT];
#pragma unroll
      for (int i = 0; i < NR; ++i)
        w4[i] = *(const float4*)&wb[(tr + 16 * i) * LS + k4];
#pragma unroll
      for (int j = 0; j < NT; ++j)
        x4[j] = *(const float4*)&xb[(tt + 16 * j) * LS + k4];
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] += w4[i].x * x4[j].x;
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] += w4[i].y * x4[j].y;
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] += w4[i].z * x4[j].z;
#pragma unroll
      for (int i = 0; i < NR; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
          acc[i][j] += w4[i].w * x4[j].w;
    }
    if (c + 1 < nc)
      sstore(c + 1, buf ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const int r = r0 + tr + 16 * i;
    if (p.rows && r >= p.rows)
      continue;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int t = t0 + tt + 16 * j;
      if (t < T)
        p.out[(size_t)t * p.ldo + r] = acc[i][j];
    }
  }
}

// Register-tiled over 64 tokens x 64 channels (each thread 4 x 4) in ascending
// k order (2 chunks of 32 k so LDS stays <= 28 KiB -> >= 2 blocks/CU; BF16
// weights stored as uint16 in LDS -> 18 KiB).
template<bool H>
struct CtlWType {
  using T = float;
};
template<>
struct CtlWType<true> {
  using T = uint16_t;
};

template<bool H>
__global__ void __launch_bounds__(256) k_ctl_apply_v4(CtlPJobs jobs, int T) {
  constexpr int LS = 36;
  const CtlPJob& p = jobs.j[blockIdx.z];
  using WT = typename CtlWType<H>::T;
  __shared__ __attribute__((aligned(16))) WT wg[64 * LS], wf[64 * LS];
  __shared__ __attribute__((aligned(16))) float gs[64 * LS];
  const int tid = threadIdx.x, t0 = blockIdx.y * 64, c0 = blockIdx.x * 64;
  const int tr = tid & 15, tt = tid >> 4;
  float ga[4][4] = {}, fo[4][4] = {};
  for (int k0 = 0; k0 < 64; k0 += 32) {
    if (k0 > 0)
      __syncthreads();
    for (int i = tid; i < 64 * 8; i += 256) {
      const int row = i >> 3, c4 = i & 7;
      const size_t woff = (size_t)(c0 + row) * 64 + k0 + c4 * 4;
      if constexpr (H) {
        *(uint2*)&wg[row * LS + c4 * 4] =
            *(const uint2*)(static_cast<const uint16_t*>(p.gain) + woff);
        *(uint2*)&wf[row * LS + c4 * 4] =
            *(const uint2*)(static_cast<const uint16_t*>(p.forget) + woff);
      } else {
        *(float4*)&wg[row * LS + c4 * 4] =
            *(const float4*)(static_cast<const float*>(p.gain) + woff);
        *(float4*)&wf[row * LS + c4 * 4] =
            *(const float4*)(static_cast<const float*>(p.forget) + woff);
      }
      const int t = min(t0 + row, T - 1);
      *(float4*)&gs[row * LS + c4 * 4] =
          *(const float4*)(p.g64 + (size_t)t * 64 + k0 + c4 * 4);
    }
    __syncthreads();
#pragma unroll 2
    for (int k4 = 0; k4 < 32; k4 += 4) {
      float4 a[4], b[4], g[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (H) {
          const uint2 ug = *(const uint2*)&wg[(tr + 16 * i) * LS + k4];
          const uint2 uf = *(const uint2*)&wf[(tr + 16 * i) * LS + k4];
          a[i] = make_float4(
              __uint_as_float(ug.x << 16), __uint_as_float(ug.x & 0xFFFF0000u),
              __uint_as_float(ug.y << 16), __uint_as_float(ug.y & 0xFFFF0000u));
          b[i] = make_float4(
              __uint_as_float(uf.x << 16), __uint_as_float(uf.x & 0xFFFF0000u),
              __uint_as_float(uf.y << 16), __uint_as_float(uf.y & 0xFFFF0000u));
        } else {
          a[i] = *(const float4*)&wg[(tr + 16 * i) * LS + k4];
          b[i] = *(const float4*)&wf[(tr + 16 * i) * LS + k4];
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        g[j] = *(const float4*)&gs[(tt + 16 * j) * LS + k4];
#define CTL_STEP(F)                                   \
  _Pragma("unroll") for (int i = 0; i < 4; ++i)       \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) { \
    ga[i][j] += a[i].F * g[j].F;                      \
    fo[i][j] += b[i].F * g[j].F;                      \
  }
      CTL_STEP(x)
      CTL_STEP(y)
      CTL_STEP(z)
      CTL_STEP(w)
#undef CTL_STEP
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = t0 + tt + 16 * j;
    if (t >= T)
      continue;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = c0 + tr + 16 * i;
      const float cv = p.x[(size_t)t * p.ldx + c];
      const float mv = p.msg[(size_t)t * p.ldm + c];
      p.out[(size_t)t * 2304 + c] =
          cv - tanhf(fo[i][j]) * cv + tanhf(ga[i][j]) * mv;
    }
  }
}

// k_ctl_apply_v4<true> followed by hip::RmsNorm of each controlled row, in
// one pass: a block takes R tokens of one job and all 2304 channels, thread
// tid the channels tid + 256 m (m = 0..8) as rmsnorm_k does. Each output
// accumulates k = 0..63 in ascending order with the same products, the row's
// squares are summed per thread in m order, then per wave and over the eight
// waves as rmsnorm_k sums them, so the values are identical. The un-normalised
// rows are not written.
template<int R>
__global__ void __launch_bounds__(256)
    k_ctl_apply_norm(CtlPJobs jobs, int T, float eps) {
  // The jobs of a token block are consecutive blocks (they share x rows).
  const CtlPJob& p = jobs.j[blockIdx.x % jobs.n];
  const int tid = threadIdx.x, t0 = (blockIdx.x / jobs.n) * R;
  const uint16_t* gw = static_cast<const uint16_t*>(p.gain4);
  const uint16_t* fw = static_cast<const uint16_t*>(p.forget4);
  __shared__ float red[R][8];
  float ga[R][9], fo[R][9];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < 9; ++m)
      ga[r][m] = fo[r][m] = 0.f;
  int tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r)
    tr[r] = min(t0 + r, T - 1);
  for (int k4 = 0; k4 < 64; k4 += 4) {
    float4 g[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
      g[r] = *(const float4*)(p.g64 + (size_t)tr[r] * 64 + k4);
#pragma unroll
    for (int m = 0; m < 9; ++m) {
      const size_t woff = ((size_t)(k4 >> 2) * 2304 + tid + 256 * m) * 4;
      const uint2 ug = *(const uint2*)(gw + woff);
      const uint2 uf = *(const uint2*)(fw + woff);
      const float4 a = make_float4(
          __uint_as_float(ug.x << 16), __uint_as_float(ug.x & 0xFFFF0000u),
          __uint_as_float(ug.y << 16), __uint_as_float(ug.y & 0xFFFF0000u));
      const float4 b = make_float4(
          __uint_as_float(uf.x << 16), __uint_as_float(uf.x & 0xFFFF0000u),
          __uint_as_float(uf.y << 16), __uint_as_float(uf.y & 0xFFFF0000u));
#pragma unroll
      for (int r = 0; r < R; ++r) {
        ga[r][m] = __fmaf_rn(a.x, g[r].x, ga[r][m]);
        fo[r][m] = __fmaf_rn(b.x, g[r].x, fo[r][m]);
        ga[r][m] = __fmaf_rn(a.y, g[r].y, ga[r][m]);
        fo[r][m] = __fmaf_rn(b.y, g[r].y, fo[r][m]);
        ga[r][m] = __fmaf_rn(a.z, g[r].z, ga[r][m]);
        fo[r][m] = __fmaf_rn(b.z, g[r].z, fo[r][m]);
        ga[r][m] = __fmaf_rn(a.w, g[r].w, ga[r][m]);
        fo[r][m] = __fmaf_rn(b.w, g[r].w, fo[r][m]);
      }
    }
  }
  const int wave = tid >> 5, lane = tid & 31;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float ss = 0.f;
#pragma unroll
    for (int m = 0; m < 9; ++m) {
      const int c = tid + 256 * m;
      const float cv = p.x[(size_t)tr[r] * p.ldx + c];
      const float mv = p.msg[(size_t)tr[r] * p.ldm + c];
      const float o = cv - tanhf(fo[r][m]) * cv + tanhf(ga[r][m]) * mv;
      ga[r][m] = o;
      ss = __fmaf_rn(o, o, ss);
    }
    ss = wsum(ss);
    if (lane == 0)
      red[r][wave] = ss;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (t0 + r >= T)
      break;
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      tot += red[r][i];
    const float inv = rsqrtf(tot / 2304.f + eps);
    float* orow = p.nout ? p.nout + (size_t)(t0 + r) * 2304 : nullptr;
    _Float16* hrow = static_cast<_Float16*>(p.nout16) + (size_t)(t0 + r) * 2304;
#pragma unroll
    for (int m = 0; m < 9; ++m) {
      const int c = tid + 256 * m;
      const float o = ga[r][m] * inv * p.nw[c];
      if (orow)
        orow[c] = o;
      hrow[c] = (_Float16)o;
    }
  }
}

// sigmoid + biased top-8 + renormalise, one wave per token (router logits
// [T][128])
__global__ void __launch_bounds__(256)
    k_topk_p(const float* __restrict__ logit, const float* __restrict__ bias,
             int* __restrict__ ids, float* __restrict__ wts, int T,
             float rscale) {
  const int lane = threadIdx.x % 32, tok = blockIdx.x * 8 + threadIdx.x / 32;
  if (tok >= T)
    return;
  float sc[4], sel[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    int i = lane + 32 * k;
    float l = logit[(size_t)tok * 128 + i];
    sc[k] = 1.f / (1.f + expf(-l));
    sel[k] = sc[k] + bias[i];
  }
  float ssum = 0.f;
  float myw[8];
  int myi[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float bv = -1e30f;
    int bi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (sel[k] > bv) {
        bv = sel[k];
        bi = lane + 32 * k;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor(bv, o, 32);
      int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    float scv = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (lane + 32 * k == bi) {
        scv = sc[k];
        sel[k] = -1e30f;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      scv += __shfl_xor(scv, o, 32);
    myw[r] = scv;
    myi[r] = bi;
    ssum += scv;
  }
  if (lane < 8) {
    float w = 0.f;
    int id = 0;
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r == lane) {
        w = myw[r];
        id = myi[r];
      }
    ids[(size_t)tok * 8 + lane] = id;
    wts[(size_t)tok * 8 + lane] = w / (ssum + 1e-20f) * rscale;
  }
}

// ---------------- attention decode (lab/attn/attn_kernels.h, device-side
// position) ----------------
namespace ia {
constexpr int HD = 128, NH = 24, NKV = 8, RD = 16, WIN = 1024, CELL = NKV * HD;
typedef _Float16 h16;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float tau_of(int pos, float floor_, float alpha) {
  float x = (float)(pos + 1) / floor_;
  return 1.0f + alpha * logf(fmaxf(1.0f, x));
}
__device__ __forceinline__ v16h cat16(uint4 a, uint4 b) {
  union {
    uint4 u[2];
    v16h v;
  } x;
  x.u[0] = a;
  x.u[1] = b;
  return x.v;
}
__device__ __forceinline__ v2h as_v2h(uint32_t u) {
  return __builtin_bit_cast(v2h, u);
}
__device__ __forceinline__ void cbar() {
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ float2 h2_to_f2(uint32_t u) {
  v2h v = as_v2h(u);
  return make_float2((float)v[0], (float)v[1]);
}
__device__ __forceinline__ float bias_dot(const v2h* rp, const __half* bt,
                                          int d) {
  const uint4* bp = (const uint4*)(bt + (size_t)d * RD);
  uint4 b0 = bp[0], b1 = bp[1];
  float s = 0.f;
  s = __builtin_amdgcn_fdot2(rp[0], as_v2h(b0.x), s, false);
  s = __builtin_amdgcn_fdot2(rp[1], as_v2h(b0.y), s, false);
  s = __builtin_amdgcn_fdot2(rp[2], as_v2h(b0.z), s, false);
  s = __builtin_amdgcn_fdot2(rp[3], as_v2h(b0.w), s, false);
  s = __builtin_amdgcn_fdot2(rp[4], as_v2h(b1.x), s, false);
  s = __builtin_amdgcn_fdot2(rp[5], as_v2h(b1.y), s, false);
  s = __builtin_amdgcn_fdot2(rp[6], as_v2h(b1.z), s, false);
  s = __builtin_amdgcn_fdot2(rp[7], as_v2h(b1.w), s, false);
  return s;
}

// Split length of a global-attention decode row at position p: the engine's
// bucket for p alone (Impl::BucketCtx; chunk as DLayer sets it), so a row's
// arithmetic never depends on the other rows of its step. Window layers use
// the fixed chunk of the arguments.
__device__ __forceinline__ int row_chunk(int chunk, int swa, int p) {
  if (swa)
    return chunk;
  int b = 2048;
  while (b < p + 1)
    b *= 2;
  int c = min(max(b / 32, 128), 2048);
  return max(c, (b / 256 + 63) / 64 * 64);
}

// NQ = 1 (3 rows: the 3 q heads of one kv head)
template<int NW>
__device__ __forceinline__ void attn_decode_body(const AttnArgs& a) {
  constexpr int NR = 3;
  __shared__ __attribute__((aligned(16))) __half sQ[16][HD + 8];
  __shared__ float sR[16][16];
  __shared__ __attribute__((aligned(16))) float sP[NW][16][16];
  __shared__ float sA[NW][16];
  __shared__ __attribute__((aligned(16))) float sO[NW][NR][HD];
  __shared__ float sM[NW][16], sL[NW][16];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.x, split = blockIdx.y, tk = blockIdx.z;
  const int p0 = a.pos[tk];
  const SlotEntry* slot = a.slot + tk * 48;
  const float* aq_ = a.q + (size_t)tk * a.ldp;
  const float* ar_ = a.r + (size_t)tk * a.ldp;
  float* ws_o = a.ws_o + (size_t)tk * a.ws_tok * 128;
  float* ws_m = a.ws_m + (size_t)tk * a.ws_tok;
  float* ws_l = a.ws_l + (size_t)tk * a.ws_tok;
  const __half* kc = (const __half*)slot->k;
  const __half* vc = (const __half*)slot->v;
  const int cap = slot->cap;
  const int kb = a.swa ? max(0, p0 - (WIN - 1)) : 0;
  const int chunk = row_chunk(a.chunk, a.swa, p0);
  const int lo = kb + split * chunk;
  const int hi = min(lo + chunk, p0 + 1);
  const __half* bt = (const __half*)a.bt;

  for (int r = w; r < 16; r += NW) {
    if (r < NR) {
      const int h = kvh * 3 + r;
      const float tau = a.swa ? 1.0f : tau_of(p0, a.tau_floor, a.tau_alpha);
      float4 x = ((const float4*)(aq_ + (size_t)h * HD))[lane];
      float4 s = ((const float4*)(a.qscale + h * HD))[lane];
      x.x *= s.x;
      x.y *= s.y;
      x.z *= s.z;
      x.w *= s.w;
      float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
      float sc = fminf(1.0f, a.qk_thr / (sqrtf(ss) + 1e-6f)) * tau *
                 0.08838834764831845f;
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(x.x * sc, x.y * sc);
      qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
      float rv = 0.f;
      if (lane < RD)
        rv = ar_[(size_t)h * RD + lane] * a.rclip[h * RD + lane];
      float rs = wsum(rv * rv);
      float rf = fminf(1.0f, a.r_thr / (sqrtf(rs) + 1e-6f)) * tau;
      if (lane < RD)
        sR[r][lane] = rv * rf;
    } else {
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(0.f, 0.f);
      qp[1] = __floats2half2_rn(0.f, 0.f);
      if (lane < RD)
        sR[r][lane] = 0.f;
    }
  }
  __syncthreads();
  const int q16 = lane & 15, half = lane >> 4;
  v16h bq[8];
#pragma unroll
  for (int s = 0; s < 8; s++)
    bq[s] = cat16(*(const uint4*)&sQ[q16][s * 16],
                  *(const uint4*)&sQ[q16][s * 16 + 8]);
  v2h rp[8];
#pragma unroll
  for (int c = 0; c < 8; c++) {
    rp[c][0] = (h16)sR[q16][2 * c];
    rp[c][1] = (h16)sR[q16][2 * c + 1];
  }
  const int posq = (q16 < NR) ? p0 : -1;

  float m = -INFINITY, l = 0.f;
  float acc[NR][4];
#pragma unroll
  for (int r = 0; r < NR; r++)
    acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;

  for (int jt = lo + w * 16; jt < hi; jt += NW * 16) {
    const int pk = min(jt + q16, hi - 1);
    const uint4* kp = (const uint4*)(kc + (size_t)(pk % cap) * CELL + kvh * HD);
    uint4 kr[16];
#pragma unroll
    for (int u = 0; u < 16; u++)
      kr[u] = kp[u];
    uint2 vr[16];
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int pv = min(jt + k, hi - 1);
      vr[k] =
          *(const uint2*)(vc + (size_t)(pv % cap) * CELL + kvh * HD + lane * 4);
    }
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 8; s++)
      c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          cat16(kr[2 * s], kr[2 * s + 1]), bq[s], c);

    float sv[8], mx = -INFINITY;
    const bool needb = (posq >= 0) && (posq - (jt + 15) < WIN);
    const bool anyb = __any(needb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int pos = jt + 2 * i + half, d = posq - pos;
      const bool ok = (d >= 0) && (pos < hi) && (!a.swa || d < WIN);
      float s = c[i];
      if (anyb && ok && d < WIN)
        s += bias_dot(rp, bt, d);
      sv[i] = ok ? s : -INFINITY;
      mx = fmaxf(mx, sv[i]);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float mn = fmaxf(m, mx);
    float alpha = 1.f, ps = 0.f;
    if (mn != -INFINITY) {
      alpha = __expf(m - mn);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float p = __expf(sv[i] - mn);
        ps += p;
        sP[w][2 * i + half][q16] = p;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++)
        sP[w][2 * i + half][q16] = 0.f;
    }
    l = l * alpha + ps;
    m = mn;
    sA[w][q16] = alpha;
    cbar();
    float al[NR];
#pragma unroll
    for (int r = 0; r < NR; r++)
      al[r] = sA[w][r];
#pragma unroll
    for (int r = 0; r < NR; r++) {
      acc[r][0] *= al[r];
      acc[r][1] *= al[r];
      acc[r][2] *= al[r];
      acc[r][3] *= al[r];
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
      float2 v01 = h2_to_f2(vr[k].x), v23 = h2_to_f2(vr[k].y);
      float p[NR];
      {
        float4 t4 = *(const float4*)&sP[w][k][0];
        p[0] = t4.x;
        p[1] = t4.y;
        p[2] = t4.z;
      }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        acc[r][0] = fmaf(p[r], v01.x, acc[r][0]);
        acc[r][1] = fmaf(p[r], v01.y, acc[r][1]);
        acc[r][2] = fmaf(p[r], v23.x, acc[r][2]);
        acc[r][3] = fmaf(p[r], v23.y, acc[r][3]);
      }
    }
    cbar();
  }
  const float ltot = l + __shfl_xor(l, 16, 32);
  if (lane < 16) {
    sM[w][lane] = m;
    sL[w][lane] = ltot;
  }
#pragma unroll
  for (int r = 0; r < NR; r++)
    *(float4*)&sO[w][r][lane * 4] =
        make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
  __syncthreads();
  const size_t base = ((size_t)split * NKV + kvh) * 12;
  for (int idx = tid; idx < NR * HD; idx += NW * 32) {
    const int r = idx / HD, dim = idx % HD;
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++)
      M = fmaxf(M, sM[w2][r]);
    float o = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++)
        o += __expf(sM[w2][r] - M) * sO[w2][r][dim];
    }
    ws_o[(base + r) * HD + dim] = o;
  }
  if (tid < NR) {
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++)
      M = fmaxf(M, sM[w2][tid]);
    float L = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++)
        L += __expf(sM[w2][tid] - M) * sL[w2][tid];
    }
    ws_m[base + tid] = M;
    ws_l[base + tid] = L;
  }
}

template<int NW>
__global__ __launch_bounds__(NW * 32) void attn_decode(AttnArgs a) {
  attn_decode_body<NW>(a);
}

// The rows of one verification step (consecutive positions of one session) in
// one block per (kv head, split): their 3 heads each are query slots
// 3 * row + head of the WMMA attn_decode fills with 3, so every K/V tile is
// read once for all rows. Global visits only (the split partition starts at key
// 0 for every row); keys past a row's position are masked, which leaves its
// sums exactly as attn_decode computes them, so each row's result is
// attn_decode's.
template<int NW, int NB>
__global__ __launch_bounds__(NW * 32) void attn_decode_rows(AttnArgs a) {
  constexpr int NR = 3 * NB;
  // blockIdx.x = kv head * groups + group: a step's groups of up to NB rows
  // are adjacent blocks, which run at the same time and share the K/V tiles
  // through L2 (read once from memory)
  const int groups = (a.nrows + NB - 1) / NB;
  const int g0 = (blockIdx.x % groups) * NB, nrow = min(NB, a.nrows - g0);
  a.q += (size_t)g0 * a.ldp;
  a.r += (size_t)g0 * a.ldp;
  a.pos += g0;
  a.ws_o += (size_t)g0 * a.ws_tok * 128;
  a.ws_m += (size_t)g0 * a.ws_tok;
  a.ws_l += (size_t)g0 * a.ws_tok;
  const int nr = 3 * nrow;
  __shared__ __attribute__((aligned(16))) __half sQ[16][HD + 8];
  __shared__ float sR[16][16];
  __shared__ __attribute__((aligned(16))) float sP[NW][16][16];
  __shared__ float sA[NW][16];
  __shared__ __attribute__((aligned(16))) float sO[NW][NR][HD];
  __shared__ float sM[NW][16], sL[NW][16];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.x / groups, split = blockIdx.y;
  const int plast = a.pos[nrow - 1];
  const SlotEntry* slot = a.slot;  // every row is the same session
  const __half* kc = (const __half*)slot->k;
  const __half* vc = (const __half*)slot->v;
  const int cap = slot->cap;
  const int chunk = row_chunk(a.chunk, 0, a.pos[0]);  // one bucket per step
  const int lo = split * chunk;
  const int hi = min(lo + chunk, plast + 1);
  const __half* bt = (const __half*)a.bt;

  for (int r = w; r < 16; r += NW) {
    if (r < nr) {
      const int tk = r / 3, h = kvh * 3 + r % 3;
      const float* aq_ = a.q + (size_t)tk * a.ldp;
      const float* ar_ = a.r + (size_t)tk * a.ldp;
      const float tau = tau_of(a.pos[tk], a.tau_floor, a.tau_alpha);
      float4 x = ((const float4*)(aq_ + (size_t)h * HD))[lane];
      float4 s = ((const float4*)(a.qscale + h * HD))[lane];
      x.x *= s.x;
      x.y *= s.y;
      x.z *= s.z;
      x.w *= s.w;
      float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
      float sc = fminf(1.0f, a.qk_thr / (sqrtf(ss) + 1e-6f)) * tau *
                 0.08838834764831845f;
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(x.x * sc, x.y * sc);
      qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
      float rv = 0.f;
      if (lane < RD)
        rv = ar_[(size_t)h * RD + lane] * a.rclip[h * RD + lane];
      float rs = wsum(rv * rv);
      float rf = fminf(1.0f, a.r_thr / (sqrtf(rs) + 1e-6f)) * tau;
      if (lane < RD)
        sR[r][lane] = rv * rf;
    } else {
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(0.f, 0.f);
      qp[1] = __floats2half2_rn(0.f, 0.f);
      if (lane < RD)
        sR[r][lane] = 0.f;
    }
  }
  __syncthreads();
  const int q16 = lane & 15, half = lane >> 4;
  v16h bq[8];
#pragma unroll
  for (int s = 0; s < 8; s++)
    bq[s] = cat16(*(const uint4*)&sQ[q16][s * 16],
                  *(const uint4*)&sQ[q16][s * 16 + 8]);
  v2h rp[8];
#pragma unroll
  for (int c = 0; c < 8; c++) {
    rp[c][0] = (h16)sR[q16][2 * c];
    rp[c][1] = (h16)sR[q16][2 * c + 1];
  }
  const int posq = (q16 < nr) ? a.pos[q16 / 3] : -1;

  float m = -INFINITY, l = 0.f;
  float acc[NR][4];
#pragma unroll
  for (int r = 0; r < NR; r++)
    acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;

  for (int jt = lo + w * 16; jt < hi; jt += NW * 16) {
    const int pk = min(jt + q16, hi - 1);
    const uint4* kp = (const uint4*)(kc + (size_t)(pk % cap) * CELL + kvh * HD);
    uint4 kr[16];
#pragma unroll
    for (int u = 0; u < 16; u++)
      kr[u] = kp[u];
    uint2 vr[16];
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int pv = min(jt + k, hi - 1);
      vr[k] =
          *(const uint2*)(vc + (size_t)(pv % cap) * CELL + kvh * HD + lane * 4);
    }
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 8; s++)
      c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          cat16(kr[2 * s], kr[2 * s + 1]), bq[s], c);

    float sv[8], mx = -INFINITY;
    const bool needb = (posq >= 0) && (posq - (jt + 15) < WIN);
    const bool anyb = __any(needb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int pos = jt + 2 * i + half, d = posq - pos;
      const bool ok = (d >= 0) && (pos < hi);
      float s = c[i];
      if (anyb && ok && d < WIN)
        s += bias_dot(rp, bt, d);
      sv[i] = ok ? s : -INFINITY;
      mx = fmaxf(mx, sv[i]);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float mn = fmaxf(m, mx);
    float alpha = 1.f, ps = 0.f;
    if (mn != -INFINITY) {
      alpha = __expf(m - mn);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float p = __expf(sv[i] - mn);
        ps += p;
        sP[w][2 * i + half][q16] = p;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++)
        sP[w][2 * i + half][q16] = 0.f;
    }
    l = l * alpha + ps;
    m = mn;
    sA[w][q16] = alpha;
    cbar();
    float al[NR];
#pragma unroll
    for (int r = 0; r < NR; r++)
      al[r] = sA[w][r];
#pragma unroll
    for (int r = 0; r < NR; r++) {
      acc[r][0] *= al[r];
      acc[r][1] *= al[r];
      acc[r][2] *= al[r];
      acc[r][3] *= al[r];
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
      float2 v01 = h2_to_f2(vr[k].x), v23 = h2_to_f2(vr[k].y);
      float p[(NR + 3) / 4 * 4];
#pragma unroll
      for (int r4 = 0; r4 < (NR + 3) / 4; r4++) {
        const float4 t4 = *(const float4*)&sP[w][k][4 * r4];
        p[4 * r4] = t4.x;
        p[4 * r4 + 1] = t4.y;
        p[4 * r4 + 2] = t4.z;
        p[4 * r4 + 3] = t4.w;
      }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        acc[r][0] = fmaf(p[r], v01.x, acc[r][0]);
        acc[r][1] = fmaf(p[r], v01.y, acc[r][1]);
        acc[r][2] = fmaf(p[r], v23.x, acc[r][2]);
        acc[r][3] = fmaf(p[r], v23.y, acc[r][3]);
      }
    }
    cbar();
  }
  const float ltot = l + __shfl_xor(l, 16, 32);
  if (lane < 16) {
    sM[w][lane] = m;
    sL[w][lane] = ltot;
  }
#pragma unroll
  for (int r = 0; r < NR; r++)
    *(float4*)&sO[w][r][lane * 4] =
        make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
  __syncthreads();
  const size_t base = ((size_t)split * NKV + kvh) * 12;
  for (int idx = tid; idx < nr * HD; idx += NW * 32) {
    const int r = idx / HD, dim = idx % HD;
    float* ws_o = a.ws_o + (size_t)(r / 3) * a.ws_tok * 128;
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++)
      M = fmaxf(M, sM[w2][r]);
    float o = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++)
        o += __expf(sM[w2][r] - M) * sO[w2][r][dim];
    }
    ws_o[(base + r % 3) * HD + dim] = o;
  }
  if (tid < nr) {
    float* ws_m = a.ws_m + (size_t)(tid / 3) * a.ws_tok;
    float* ws_l = a.ws_l + (size_t)(tid / 3) * a.ws_tok;
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++)
      M = fmaxf(M, sM[w2][tid]);
    float L = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++)
        L += __expf(sM[w2][tid] - M) * sL[w2][tid];
    }
    ws_m[base + tid % 3] = M;
    ws_l[base + tid % 3] = L;
  }
}

// Combines the splits of head h, token tk (128 threads): the sink, the gate.
__device__ __forceinline__ void reduce_head(const AttnArgs& a, int h, int tk,
                                            float* sc /*[1024]*/,
                                            float* sred /*[4]*/) {
  const int kvh = h / 3, r = h % 3, dim = threadIdx.x;
  const float* ws_o = a.ws_o + (size_t)tk * a.ws_tok * 128;
  const float* ws_m = a.ws_m + (size_t)tk * a.ws_tok;
  const float* ws_l = a.ws_l + (size_t)tk * a.ws_tok;
  const float sk = a.sink[tk * 24 + h];
  float mmax = sk;
  for (int s = dim; s < a.nsplit; s += 128) {
    sc[s] = ws_m[((size_t)s * NKV + kvh) * 12 + r];
    mmax = fmaxf(mmax, sc[s]);
  }
  mmax = fmaxf(mmax, __shfl_xor(mmax, 16, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 8, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 4, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 2, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 1, 32));
  if ((dim & 31) == 0)
    sred[dim >> 5] = mmax;
  __syncthreads();
  const float M = fmaxf(fmaxf(sred[0], sred[1]), fmaxf(sred[2], sred[3]));
  float zp = 0.f;
  for (int s = dim; s < a.nsplit; s += 128) {
    const float ms = sc[s];
    const float e = (ms == -INFINITY) ? 0.f : __expf(ms - M);
    sc[s] = e;
    zp +=
        e * ((ms == -INFINITY) ? 0.f : ws_l[((size_t)s * NKV + kvh) * 12 + r]);
  }
  zp = wsum(zp);
  __syncthreads();
  if ((dim & 31) == 0)
    sred[dim >> 5] = zp;
  __syncthreads();
  const float Z = __expf(sk - M) + sred[0] + sred[1] + sred[2] + sred[3];
  float o = 0.f;
#pragma unroll 8
  for (int s = 0; s < a.nsplit; s++)
    o += sc[s] * ws_o[(((size_t)s * NKV + kvh) * 12 + r) * HD + dim];
  o /= Z;
  if (a.gate)
    o *= 1.0f /
         (1.0f + __expf(-a.gate[(size_t)tk * a.ldp + (size_t)h * HD + dim]));
  a.out[(size_t)tk * 3072 + (size_t)h * HD + dim] = o;
}
__global__ __launch_bounds__(128) void decode_reduce(AttnArgs a) {
  __shared__ float sc[1024];
  __shared__ float sred[4];
  reduce_head(a, blockIdx.x, blockIdx.z, sc, sred);
}

// attn_decode of distinct sessions' rows with k_kvprep folded in: the block of
// the split holding the row's position first writes the kv head's K/V row.
// decode_reduce combines the splits in its own launch (24 heads x rows blocks
// in parallel; folded into the last split block it serialises the three heads
// of a kv head, 0.5% slower per token). Same arithmetic as the three launches.
template<int NW>
__global__ __launch_bounds__(NW * 32) void attn_decode_fused(AttnArgs a,
                                                             KvPrepArgs kv) {
  __shared__ float red[32];
  const int kvh = blockIdx.x, split = blockIdx.y, tk = blockIdx.z;
  {
    const int p0 = a.pos[tk];
    const int kb = a.swa ? max(0, p0 - (WIN - 1)) : 0;
    if ((p0 - kb) / row_chunk(a.chunk, a.swa, p0) == split) {
      kvprep_block(kv, kvh, tk, red);
      __syncthreads();
    }
  }
  attn_decode_body<NW>(a);
}
}  // namespace ia

// ================================================================================================
// Prefill (T > 1): LDS-tiled f16-WMMA grouped GEMM (lab kern3.h / kern4.h
// `k_gemm4`), GPU-side expert grouping, and the WMMA flash attention
// (lab/attn/attn_kernels.h `attn_prefill`).
// ================================================================================================
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h_a16 __attribute__((ext_vector_type(16), aligned(16)));
typedef uint32_t v4u_a16 __attribute__((ext_vector_type(4), aligned(16)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

struct GP3 {
  const uint8_t* W;
  size_t wexp;
  const float* escale;
  const _Float16* ah;
  const int* actrow;
  const int* nitems;
  const int* item_e;
  const int* item_start;
  const int* item_n;
  float* out;
  _Float16* outh;
  int ldo;
  const int* postok;
  const int* pair;  // EPI 2: (token * 8 + slot) of each sorted position:
                    // deterministic, non-atomic write
  const float* posw;
};

__device__ __forceinline__ void nib4_to_half(uint32_t n, uint32_t& a,
                                             uint32_t& b) {
  uint32_t m = n & 0x07070707u, t = (n << 4) & 0x80808080u;
  uint32_t sA = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x05000400u),
           sB = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x07000600u);
  a = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sA) |
      __builtin_amdgcn_perm(t, 0u, 0x050C040Cu);
  b = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sB) |
      __builtin_amdgcn_perm(t, 0u, 0x070C060Cu);
}
__device__ __forceinline__ v16h dq_nv16(uint32_t w0, uint32_t w1, float ds) {
  v8u r;
  uint32_t a, b;
  nib4_to_half(w0 & 0x0F0F0F0Fu, a, b);
  r[0] = a;
  r[1] = b;
  nib4_to_half(w1 & 0x0F0F0F0Fu, a, b);
  r[2] = a;
  r[3] = b;
  nib4_to_half((w0 >> 4) & 0x0F0F0F0Fu, a, b);
  r[4] = a;
  r[5] = b;
  nib4_to_half((w1 >> 4) & 0x0F0F0F0Fu, a, b);
  r[6] = a;
  r[7] = b;
  return __builtin_bit_cast(v16h, r) * (_Float16)ds;
}
__device__ __forceinline__ v16h dq_q8_16(v4u w, float d) {
  v8u r;
  uint32_t x[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t u = x[i] ^ 0x80808080u;
    r[2 * i] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    r[2 * i + 1] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
  }
  return (__builtin_bit_cast(v16h, r) - (_Float16)1152.0f) * (_Float16)d;
}

// EPI 0: f32 out[pos*ldo+row]; EPI 1: gate/up + SITU -> outh[pos*FF+j]; EPI 2:
// down, atomicAdd into out[postok[pos]*ldo+row] weighted by posw*expert scale.
template<int FMT, int EPI, int BN, int K, int RG>
__global__ void __launch_bounds__(64 * RG) k_gemm4(GP3 p, int rgn) {
  constexpr int FF = 1152, STR = 72, NT = 64 * RG, BM = 32 * RG, TTW = BN / 32,
                NK = K / 64, NCH = (BN * 8 + NT - 1) / NT;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  __shared__ _Float16 Ws[2][BM * STR];
  // GA: activations go straight from global memory into the WMMA A operand (one
  // token row of 16 consecutive k values per lane) when at most two waves share
  // a token row; otherwise they are staged in LDS like the weights. Same values
  // and WMMAs either way.
  constexpr bool GA = RG <= 2;
  __shared__ _Float16 As[GA ? 1 : 2][GA ? 8 : BN * STR];
  const int total = *p.nitems * rgn;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15,
            hi = lane >> 4, rg = wave % RG, th = wave / RG;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int item = unit / rgn, rgi = unit % rgn;
    const int e = p.item_e[item], start = p.item_start[item],
              n = p.item_n[item];
    const uint8_t* We = p.W + (size_t)e * p.wexp;
    int wr = tid >> 1, wh = tid & 1, srow;
    if (EPI == 1) {
      int h2 = wr >> 5, sub = (wr >> 4) & 1, r = wr & 15;
      srow = (sub ? FF : 0) + rgi * (16 * RG) + h2 * 16 + r;
    } else {
      srow = rgi * BM + wr;
    }
    const uint8_t* wp = We + (size_t)srow * RB;
    uint32_t raw[9];
    v4u areg[NCH];
    const _Float16* ap[NCH];
    int arow_l[NCH];
    bool aok[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      int idx = tid + NT * i, row = idx >> 3, ch = idx & 7;
      aok[i] = !GA && idx < BN * 8;
      arow_l[i] = row * STR + ch * 8;
      int pos = row < n ? start + row : start;
      int ar = p.actrow ? p.actrow[pos] : pos;
      ap[i] = p.ah + (size_t)ar * K + ch * 8;
    }
    const _Float16* garow[GA ? TTW : 1];
    if constexpr (GA) {
#pragma unroll
      for (int t = 0; t < TTW; t++) {
        const int row = th * (BN / 2) + 16 * t + ln;
        const int pos = row < n ? start + row : start;
        garow[t] = p.ah + (size_t)(p.actrow ? p.actrow[pos] : pos) * K;
      }
    }
    auto gload = [&](int kb) {
      if (FMT == FMT_NVFP4) {
        const uint8_t* b = wp + (size_t)kb * 36;
        raw[0] = *(const uint32_t*)b;
        v4u q = *(const v4u_a4*)(b + 4 + 16 * wh);
        raw[1] = q.x;
        raw[2] = q.y;
        raw[3] = q.z;
        raw[4] = q.w;
      } else {
        const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34;
        raw[0] = *(const uint16_t*)b;
        v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
        raw[1] = a.x;
        raw[2] = a.y;
        raw[3] = a.z;
        raw[4] = a.w;
        raw[5] = c.x;
        raw[6] = c.y;
        raw[7] = c.z;
        raw[8] = c.w;
      }
#pragma unroll
      for (int i = 0; i < NCH; i++)
        if (aok[i])
          areg[i] = *(const v4u*)(ap[i] + kb * 64);
    };
    auto sstore = [&](int buf) {
      _Float16* dst = &Ws[buf][wr * STR + 32 * wh];
      if (FMT == FMT_NVFP4) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
          float ds = ue4m3((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu);
          *(v16h_a16*)(dst + 16 * s2) =
              dq_nv16(raw[1 + 2 * s2], raw[2 + 2 * s2], ds);
        }
      } else {
        float d = __half2float(__ushort_as_half((unsigned short)raw[0]));
        *(v16h_a16*)dst = dq_q8_16((v4u){raw[1], raw[2], raw[3], raw[4]}, d);
        *(v16h_a16*)(dst + 16) =
            dq_q8_16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
      }
#pragma unroll
      for (int i = 0; i < NCH; i++)
        if (aok[i])
          *(v4u_a16*)&As[buf][arow_l[i]] = areg[i];
    };
    v8f facc[TTW][2];
#pragma unroll
    for (int t = 0; t < TTW; t++) {
      facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      facc[t][1] = facc[t][0];
    }
    gload(0);
    sstore(0);
    __syncthreads();
    for (int kb = 0; kb < NK; kb++) {
      int cur = kb & 1;
      // A struct per fragment: clang 24 (ROCm 10.1) promotes small plain
      // fragment arrays to one wide vector and extracts each fragment through
      // an integer, 16 extra VALU instructions per fragment.
      struct Fragment {
        v16h v;
      } av[GA ? TTW : 1][4];
      if constexpr (GA) {
#pragma unroll
        for (int t = 0; t < TTW; t++)
#pragma unroll
          for (int ks = 0; ks < 4; ks++)
            av[t][ks].v = *(const v16h_a16*)(garow[t] + kb * 64 + ks * 16);
      }
      if (kb + 1 < NK)
        gload(kb + 1);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        Fragment bf[2];
#pragma unroll
        for (int r = 0; r < 2; r++)
          bf[r].v = *(
              const v16h_a16*)&Ws[cur][(rg * 32 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
        for (int t = 0; t < TTW; t++) {
          v16h a;
          if constexpr (GA)
            a = av[t][ks].v;
          else
            a = *(const v16h_a16*)&As[cur][(th * (BN / 2) + 16 * t + ln) * STR +
                                           ks * 16];
#pragma unroll
          for (int r = 0; r < 2; r++)
            facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bf[r].v,
                                                                    facc[t][r]);
        }
      }
      if (kb + 1 < NK)
        sstore(cur ^ 1);
      __syncthreads();
    }
    float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
    for (int t = 0; t < TTW; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int pl = th * (BN / 2) + 16 * t + 2 * i + hi;
        if (pl >= n)
          continue;
        int pd = start + pl;
        if (EPI == 0) {
          int row = rgi * BM + rg * 32 + ln;
          p.out[(size_t)pd * p.ldo + row] = facc[t][0][i];
          p.out[(size_t)pd * p.ldo + row + 16] = facc[t][1][i];
        } else if (EPI == 1) {
          int j = rgi * (16 * RG) + rg * 16 + ln;
          p.outh[(size_t)pd * FF + j] =
              (_Float16)situ(facc[t][0][i] * esc, facc[t][1][i] * esc);
        } else {
          float w = p.posw[pd] * esc;
          // One row per (token, slot) pair; MoeTail adds the 8 slots in a fixed
          // order. (An f32 atomicAdd here made two identical prefills differ in
          // the last bits.)
          size_t o = (size_t)p.pair[pd] * p.ldo;
          int row = rgi * BM + rg * 32 + ln;
          p.out[o + row] = w * facc[t][0][i];
          p.out[o + row + 16] = w * facc[t][1][i];
        }
      }
    __syncthreads();
  }
}

// MoE gate/up + SITU for 128-token items: 8 waves, each 32 tokens x (32 gate +
// 32 up) channels, so six LDS loads feed eight WMMAs (k_gemm4: four feed four).
// Same dequantisation, WMMA operand order and ascending-k accumulation per
// output as k_gemm4<FMT, 1, ...>, so the results are identical.
template<int FMT>
__global__ void __launch_bounds__(256) k_gu128(GP3 p, int rgn) {
  constexpr int FF = 1152, STR = 72, BM = 128, K = 2304, NK = K / 64;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  // Only the dequantised weights go through LDS. A WMMA A operand is one token
  // row of 16 consecutive k values per lane, so each wave loads its activations
  // from global memory.
  __shared__ _Float16 Ws[BM * STR];
  const int total = *p.nitems * rgn;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15,
            hi = lane >> 4, cg = wave & 1, tg = wave >> 1;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int item = unit / rgn, rgi = unit % rgn;
    const int e = p.item_e[item], start = p.item_start[item],
              n = p.item_n[item];
    const uint8_t* We = p.W + (size_t)e * p.wexp;
    // LDS row wr: 16-channel group h2, gate (sub 0) or up (sub 1), channel r.
    const int wr = tid >> 1, wh = tid & 1;
    const int srow =
        (((wr >> 4) & 1) ? FF : 0) + rgi * 64 + (wr >> 5) * 16 + (wr & 15);
    const uint8_t* wp = We + (size_t)srow * RB;
    const _Float16* arow[2];
#pragma unroll
    for (int t = 0; t < 2; t++) {
      const int row = tg * 32 + 16 * t + ln;
      const int pos = row < n ? start + row : start;
      arow[t] = p.ah + (size_t)p.actrow[pos] * K;
    }
    uint32_t raw[9];
    auto gload = [&](int kb) {
      if (FMT == FMT_NVFP4) {
        const uint8_t* b = wp + (size_t)kb * 36;
        raw[0] = *(const uint32_t*)b;
        v4u q = *(const v4u_a4*)(b + 4 + 16 * wh);
        raw[1] = q.x;
        raw[2] = q.y;
        raw[3] = q.z;
        raw[4] = q.w;
      } else {
        const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34;
        raw[0] = *(const uint16_t*)b;
        v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
        raw[1] = a.x;
        raw[2] = a.y;
        raw[3] = a.z;
        raw[4] = a.w;
        raw[5] = c.x;
        raw[6] = c.y;
        raw[7] = c.z;
        raw[8] = c.w;
      }
    };
    auto sstore = [&]() {
      _Float16* dst = &Ws[wr * STR + 32 * wh];
      if (FMT == FMT_NVFP4) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
          float ds = ue4m3((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu);
          *(v16h_a16*)(dst + 16 * s2) =
              dq_nv16(raw[1 + 2 * s2], raw[2 + 2 * s2], ds);
        }
      } else {
        float d = __half2float(__ushort_as_half((unsigned short)raw[0]));
        *(v16h_a16*)dst = dq_q8_16((v4u){raw[1], raw[2], raw[3], raw[4]}, d);
        *(v16h_a16*)(dst + 16) =
            dq_q8_16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
      }
    };
    v8f facc[2][2][2];  // [token tile][channel tile][gate, up]
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int c = 0; c < 2; c++)
        facc[t][c][0] = facc[t][c][1] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    gload(0);
    sstore();
    __syncthreads();
    // 16-token tiles wholly past the item's end are padding: their WMMAs are
    // skipped.
    const bool live[2] = {tg * 32 < n, tg * 32 + 16 < n};
    for (int kb = 0; kb < NK; kb++) {
      v16h a[2][4];
#pragma unroll
      for (int t = 0; t < 2; t++)
        if (live[t])
#pragma unroll
          for (int ks = 0; ks < 4; ks++)
            a[t][ks] = *(const v16h_a16*)(arow[t] + kb * 64 + ks * 16);
      if (kb + 1 < NK)
        gload(kb + 1);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        v16h bf[2][2];
#pragma unroll
        for (int c = 0; c < 2; c++)
#pragma unroll
          for (int g = 0; g < 2; g++)
            bf[c][g] = *(
                const v16h_a16*)&Ws[(((2 * cg + c) * 32) + 16 * g + ln) * STR +
                                    ks * 16];
#pragma unroll
        for (int t = 0; t < 2; t++)
          if (live[t])
#pragma unroll
            for (int c = 0; c < 2; c++)
#pragma unroll
              for (int g = 0; g < 2; g++)
                facc[t][c][g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a[t][ks], bf[c][g], facc[t][c][g]);
      }
      __syncthreads();
      if (kb + 1 < NK) {
        sstore();
        __syncthreads();
      }
    }
    const float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int c = 0; c < 2; c++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int pl = tg * 32 + 16 * t + 2 * i + hi;
          if (pl >= n)
            continue;
          const int j = rgi * 64 + (2 * cg + c) * 16 + ln;
          p.outh[(size_t)(start + pl) * FF + j] =
              (_Float16)situ(facc[t][c][0][i] * esc, facc[t][c][1][i] * esc);
        }
  }
}

// Dense (EPI 0) and MoE down (EPI 2) products for 128-token items: 8 waves over
// 128 tokens x 128 rows, each wave 32 tokens x 64 rows (six LDS/global loads
// feed eight WMMAs). Weights go through LDS, activations straight from global
// memory. Same dequantisation, WMMA operand order, ascending-k accumulation and
// epilogue arithmetic per output as k_gemm4<FMT, EPI, ...>.
template<int FMT, int EPI, int K>
__device__ __forceinline__ void g128_unit(const GP3& p, int unit, int rgn,
                                          _Float16* Ws, int* Ps, float* Pw) {
  constexpr int STR = 72, BM = 128, NK = K / 64;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15,
            hi = lane >> 4, cg = wave & 1, tg = wave >> 1;
  {
    const int item = unit / rgn, rgi = unit % rgn;
    const int e = p.item_e[item], start = p.item_start[item],
              n = p.item_n[item];
    const uint8_t* We = p.W + (size_t)e * p.wexp;
    const int wr = tid >> 1, wh = tid & 1;
    const uint8_t* wp = We + (size_t)(rgi * BM + wr) * RB;
    const _Float16* arow[2];
#pragma unroll
    for (int t = 0; t < 2; t++) {
      const int row = tg * 32 + 16 * t + ln;
      const int pos = row < n ? start + row : start;
      arow[t] = p.ah + (size_t)(p.actrow ? p.actrow[pos] : pos) * K;
    }
    uint32_t raw[9];
    auto gload = [&](int kb) {
      if (FMT == FMT_NVFP4) {
        const uint8_t* b = wp + (size_t)kb * 36;
        raw[0] = *(const uint32_t*)b;
        v4u q = *(const v4u_a4*)(b + 4 + 16 * wh);
        raw[1] = q.x;
        raw[2] = q.y;
        raw[3] = q.z;
        raw[4] = q.w;
      } else {
        const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34;
        raw[0] = *(const uint16_t*)b;
        v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
        raw[1] = a.x;
        raw[2] = a.y;
        raw[3] = a.z;
        raw[4] = a.w;
        raw[5] = c.x;
        raw[6] = c.y;
        raw[7] = c.z;
        raw[8] = c.w;
      }
    };
    auto sstore = [&]() {
      _Float16* dst = &Ws[wr * STR + 32 * wh];
      if (FMT == FMT_NVFP4) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
          float ds = ue4m3((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu);
          *(v16h_a16*)(dst + 16 * s2) =
              dq_nv16(raw[1 + 2 * s2], raw[2 + 2 * s2], ds);
        }
      } else {
        float d = __half2float(__ushort_as_half((unsigned short)raw[0]));
        *(v16h_a16*)dst = dq_q8_16((v4u){raw[1], raw[2], raw[3], raw[4]}, d);
        *(v16h_a16*)(dst + 16) =
            dq_q8_16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
      }
    };
    v8f facc[2][4];  // [token tile][16-row tile]
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        facc[t][r] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    // MoE down: the item's (token, slot) rows and weights for the epilogue,
    // loaded now instead of beside the stores.
    if (EPI == 2 && tid < 128) {
      const int pd = start + (tid < n ? tid : 0);
      Ps[tid] = p.pair[pd];
      Pw[tid] = p.posw[pd];
    }
    gload(0);
    sstore();
    __syncthreads();
    // 16-token tiles wholly past the item's end are padding: their WMMAs are
    // skipped.
    const bool live[2] = {tg * 32 < n, tg * 32 + 16 < n};
    // The live tiles are wave-uniform: one straight-line body per count, so the
    // LDS loads of the next k-step overlap the WMMAs of this one.
    // NTC 1 or 2: that many live tiles, no checks; 0: check each tile.
    auto step = [&](int kb, auto ntc, const bool* lv) {
      constexpr int NTC = decltype(ntc)::value, NTL = NTC ? NTC : 2;
      v16h a[NTL][4];
#pragma unroll
      for (int t = 0; t < NTL; t++)
        if (NTC || lv[t])
#pragma unroll
          for (int ks = 0; ks < 4; ks++)
            a[t][ks] = *(const v16h_a16*)(arow[t] + kb * 64 + ks * 16);
      if (kb + 1 < NK)
        gload(kb + 1);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        v16h bf[4];
#pragma unroll
        for (int r = 0; r < 4; r++)
          bf[r] =
              *(const v16h_a16*)&Ws[(cg * 64 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
        for (int t = 0; t < NTL; t++)
          if (NTC || lv[t])
#pragma unroll
            for (int r = 0; r < 4; r++)
              facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                  a[t][ks], bf[r], facc[t][r]);
      }
    };
    // K = 3072 measured faster with the per-tile checks inside one body.
    constexpr bool kSplit = K != 3072;
    for (int kb = 0; kb < NK; kb++) {
      if (!kSplit)
        step(kb, std::integral_constant<int, 0>{}, live);
      else if (live[1])
        step(kb, std::integral_constant<int, 2>{}, live);
      else if (live[0])
        step(kb, std::integral_constant<int, 1>{}, live);
      else if (kb + 1 < NK)
        gload(kb + 1);
      __syncthreads();
      if (kb + 1 < NK) {
        sstore();
        __syncthreads();
      }
    }
    const float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
    for (int t = 0; t < 2; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int pl = tg * 32 + 16 * t + 2 * i + hi;
        if (pl >= n)
          continue;
        const int pd = start + pl;
        const size_t prow = EPI == 2 ? (size_t)Ps[pl] * p.ldo : 0;
        const float pw = EPI == 2 ? Pw[pl] : 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
          const int row = rgi * BM + cg * 64 + 16 * r + ln;
          if (EPI == 0) {
            p.out[(size_t)pd * p.ldo + row] = facc[t][r][i];
          } else {
            const float w = pw * esc;
            p.out[prow + row] = w * facc[t][r][i];
          }
        }
      }
  }
}

template<int FMT, int EPI, int K>
__global__ void __launch_bounds__(256) k_g128(GP3 p, int rgn) {
  __shared__ _Float16 Ws[128 * 72];
  __shared__ int Ps[EPI == 2 ? 128 : 1];
  __shared__ float Pw[EPI == 2 ? 128 : 1];
  const int total = *p.nitems * rgn;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    g128_unit<FMT, EPI, K>(p, unit, rgn, Ws, Ps, Pw);
    __syncthreads();
  }
}

// MoE tail of a prefill ubatch in one pass: x[t] = sum of the 8 slot rows of
// token t (slot order, as the old per-token sum), y = post conv of x with the
// history `state` (as hip::conv_k), cur[t] += y (as hip::add_k). A block takes
// 16 tokens of 256 channels and keeps the 4-token window in registers.
// SL = 1: the same with the input rows themselves (hip::conv_k + hip::add_k of
// the attention output).
template<int SL>
__global__ void __launch_bounds__(256)
    k_moe_tail(const float* __restrict__ pairs, const float* __restrict__ state,
               const float* __restrict__ w, float* __restrict__ cur, int T,
               int C, float scale) {
  const int c = blockIdx.x * 256 + threadIdx.x, t0 = blockIdx.y * 16;
  if (c >= C)
    return;
  auto slot_sum = [&](int t) {
    float a = pairs[((size_t)t * SL) * C + c];
#pragma unroll
    for (int k = 1; k < SL; k++)
      a += pairs[((size_t)t * SL + k) * C + c];
    return a;
  };
  float xin[4];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int idx = t0 - 3 + i;
    xin[i + 1] = idx >= 0 ? slot_sum(idx) : state[(size_t)c * 3 + idx + 3];
  }
  const float w0 = w[(size_t)c * 4], w1 = w[(size_t)c * 4 + 1],
              w2 = w[(size_t)c * 4 + 2], w3 = w[(size_t)c * 4 + 3];
  const int t1 = min(t0 + 16, T);
  for (int t = t0; t < t1; ++t) {
    xin[0] = xin[1];
    xin[1] = xin[2];
    xin[2] = xin[3];
    xin[3] = slot_sum(t);
    float y = 0.f;
    y += w0 * xin[0];
    y += w1 * xin[1];
    y += w2 * xin[2];
    y += w3 * xin[3];
    const size_t o = (size_t)t * C + c;
    {
      // Rounded product, then rounded sum: the two kernels this replaces could
      // not contract them.
#pragma clang fp contract(off)
      const float out = (xin[3] + y) * scale;
      cur[o] = cur[o] + out;
    }
  }
}

// The convolution history after the ubatch: its last 3 inputs (as
// hip::conv_state_k), the inputs being the slot sums of k_moe_tail.
template<int SL>
__global__ void k_moe_tail_state(const float* __restrict__ pairs,
                                 float* __restrict__ state, int T, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  float old[3] = {state[(size_t)c * 3], state[(size_t)c * 3 + 1],
                  state[(size_t)c * 3 + 2]};
  float nw[3];
  for (int j = 0; j < 3; ++j) {
    const int idx = T + j;
    if (idx < 3) {
      nw[j] = old[idx];
    } else {
      const size_t t = idx - 3;
      float a = pairs[(t * SL) * C + c];
#pragma unroll
      for (int k = 1; k < SL; k++)
        a += pairs[(t * SL + k) * C + c];
      nw[j] = a;
    }
  }
  for (int j = 0; j < 3; ++j)
    state[(size_t)c * 3 + j] = nw[j];
}

template<int SL>
__global__ void k_moe_tail_state_capture(const float* __restrict__ pairs,
                                         const float* __restrict__ old_state,
                                         float* __restrict__ dst_state, int T,
                                         int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  for (int j = 0; j < 3; ++j) {
    const int idx = T + j;
    if (idx < 3) {
      dst_state[(size_t)c * 3 + j] = old_state[(size_t)c * 3 + idx];
    } else {
      const size_t t = idx - 3;
      float a = pairs[(t * SL) * C + c];
#pragma unroll
      for (int k = 1; k < SL; k++)
        a += pairs[(t * SL + k) * C + c];
      dst_state[(size_t)c * 3 + j] = a;
    }
  }
}

__global__ void k_cvt_f16(const float* __restrict__ x, _Float16* __restrict__ h,
                          size_t n) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    float4 v = *(const float4*)(x + i);
    h[i] = (_Float16)v.x;
    h[i + 1] = (_Float16)v.y;
    h[i + 2] = (_Float16)v.z;
    h[i + 3] = (_Float16)v.w;
  } else {
    for (; i < n; ++i)
      h[i] = (_Float16)x[i];
  }
}

// GPU-side expert grouping (E = 128)
__global__ void k_hist(const int* __restrict__ ids, int P, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P)
    atomicAdd(&cnt[ids[i]], 1);
}
__global__ void k_scan_items(const int* cnt, int tile, int* goff, int* fill,
                             int* nitems, int* ie, int* is, int* in) {
  __shared__ int sc[128], st[128];
  int e = threadIdx.x;
  int c = cnt[e], tiles = (c + tile - 1) / tile;
  sc[e] = c;
  st[e] = tiles;
  __syncthreads();
  for (int o = 1; o < 128; o <<= 1) {
    int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0;
    __syncthreads();
    sc[e] += a;
    st[e] += b;
    __syncthreads();
  }
  int ec = sc[e] - c, et = st[e] - tiles;
  goff[e] = ec;
  fill[e] = 0;
  for (int t = 0; t < tiles; t++) {
    int idx = et + t;
    ie[idx] = e;
    is[idx] = ec + t * tile;
    in[idx] = min(tile, c - t * tile);
  }
  if (e == 127) {
    *nitems = st[127];
    goff[128] = sc[127];
  }
}
__global__ void k_scatter(const int* __restrict__ ids,
                          const float* __restrict__ w, int P, const int* goff,
                          int* fill, int* postok, int* pair, float* posw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P)
    return;
  int e = ids[i];
  int pos = goff[e] + atomicAdd(&fill[e], 1);
  postok[pos] = i / 8;
  pair[pos] = i;
  posw[pos] = w[i];
}

// ---------------- prefill attention prologue / KV write / flash attention
// ----------------
__global__ __launch_bounds__(128) void k_kv_write(const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  int n, int pos0, int cap,
                                                  __half* kc, __half* vc,
                                                  float qk_thr, float eps) {
  using namespace ia;
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= n * NKV)
    return;
  const int t = gw / NKV, kvh = gw % NKV;
  const size_t off = (size_t)t * CELL + kvh * HD;
  float4 kx = ((const float4*)(k + off))[lane];
  float4 vx = ((const float4*)(v + off))[lane];
  float ks = wsum(kx.x * kx.x + kx.y * kx.y + kx.z * kx.z + kx.w * kx.w);
  float vs = wsum(vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w);
  float kf = fminf(1.0f, qk_thr / (sqrtf(ks) + 1e-6f));
  float vf = rsqrtf(vs * (1.0f / HD) + eps);
  int cell = (pos0 + t) % cap;
  __half2* kp = (__half2*)(kc + (size_t)cell * CELL + kvh * HD + lane * 4);
  __half2* vp = (__half2*)(vc + (size_t)cell * CELL + kvh * HD + lane * 4);
  kp[0] = __floats2half2_rn(kx.x * kf, kx.y * kf);
  kp[1] = __floats2half2_rn(kx.z * kf, kx.w * kf);
  vp[0] = __floats2half2_rn(vx.x * vf, vx.y * vf);
  vp[1] = __floats2half2_rn(vx.z * vf, vx.w * vf);
}

// k_kv_write of the short-convolved K and V: the convolution (hip::conv_k with
// the same products and order) is evaluated here from the raw projections, so
// the convolved rows never go through memory.
struct KvConv {
  const float *k, *v;        // raw [n][1024]
  const float *kst, *vst;    // histories [1024][3]
  const float *kw, *vw;      // static taps [1024][4]
  const float *kdyn, *vdyn;  // dynamic taps [n][8][4] or null
  float scale;
};
__device__ __forceinline__ float conv_elem(const float* x, const float* st,
                                           const float* w, const float* dyn,
                                           int t, int c, float scale) {
  float xin[4];
  for (int i = 0; i < 4; ++i) {
    const int idx = t - 3 + i;
    xin[i] = idx >= 0 ? x[(size_t)idx * 1024 + c] : st[(size_t)c * 3 + idx + 3];
  }
  float y = 0.f;
  for (int i = 0; i < 4; ++i)
    y += w[(size_t)c * 4 + i] * xin[i];
  if (dyn) {
    const float* d = dyn + (size_t)t * 8 * 4 + (c / 128) * 4;
    for (int lag = 0; lag < 4; ++lag)
      y += d[lag] * xin[3 - lag];
  }
  return (xin[3] + y) * scale;
}
__global__ __launch_bounds__(128) void k_kv_conv_write(KvConv cv, int n,
                                                       int pos0, int cap,
                                                       __half* kc, __half* vc,
                                                       float qk_thr,
                                                       float eps) {
  using namespace ia;
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= n * NKV)
    return;
  const int t = gw / NKV, kvh = gw % NKV, c0 = kvh * HD + lane * 4;
  float4 kx, vx;
  kx.x = conv_elem(cv.k, cv.kst, cv.kw, cv.kdyn, t, c0, cv.scale);
  kx.y = conv_elem(cv.k, cv.kst, cv.kw, cv.kdyn, t, c0 + 1, cv.scale);
  kx.z = conv_elem(cv.k, cv.kst, cv.kw, cv.kdyn, t, c0 + 2, cv.scale);
  kx.w = conv_elem(cv.k, cv.kst, cv.kw, cv.kdyn, t, c0 + 3, cv.scale);
  vx.x = conv_elem(cv.v, cv.vst, cv.vw, cv.vdyn, t, c0, cv.scale);
  vx.y = conv_elem(cv.v, cv.vst, cv.vw, cv.vdyn, t, c0 + 1, cv.scale);
  vx.z = conv_elem(cv.v, cv.vst, cv.vw, cv.vdyn, t, c0 + 2, cv.scale);
  vx.w = conv_elem(cv.v, cv.vst, cv.vw, cv.vdyn, t, c0 + 3, cv.scale);
  float ks = wsum(kx.x * kx.x + kx.y * kx.y + kx.z * kx.z + kx.w * kx.w);
  float vs = wsum(vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w);
  float kf = fminf(1.0f, qk_thr / (sqrtf(ks) + 1e-6f));
  float vf = rsqrtf(vs * (1.0f / HD) + eps);
  int cell = (pos0 + t) % cap;
  __half2* kp = (__half2*)(kc + (size_t)cell * CELL + kvh * HD + lane * 4);
  __half2* vp = (__half2*)(vc + (size_t)cell * CELL + kvh * HD + lane * 4);
  kp[0] = __floats2half2_rn(kx.x * kf, kx.y * kf);
  kp[1] = __floats2half2_rn(kx.z * kf, kx.w * kf);
  vp[0] = __floats2half2_rn(vx.x * vf, vx.y * vf);
  vp[1] = __floats2half2_rn(vx.z * vf, vx.w * vf);
}

__global__ __launch_bounds__(128) void k_qr_prep(
    const float* __restrict__ q, const float* __restrict__ r,
    const float* __restrict__ qscale, const float* __restrict__ rclip, int n,
    int pos0, int swa, float qk_thr, float r_thr, float tau_floor,
    float tau_alpha, __half* qn, float* rn) {
  using namespace ia;
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= n * NH)
    return;
  const int t = gw / NH, h = gw % NH;
  const float tau = swa ? 1.0f : tau_of(pos0 + t, tau_floor, tau_alpha);
  float4 x = ((const float4*)(q + (size_t)gw * HD))[lane];
  float4 s = ((const float4*)(qscale + h * HD))[lane];
  x.x *= s.x;
  x.y *= s.y;
  x.z *= s.z;
  x.w *= s.w;
  float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
  float sc =
      fminf(1.0f, qk_thr / (sqrtf(ss) + 1e-6f)) * tau * 0.08838834764831845f;
  __half2* qp = (__half2*)(qn + (size_t)gw * HD + lane * 4);
  qp[0] = __floats2half2_rn(x.x * sc, x.y * sc);
  qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
  float rv = 0.f;
  if (lane < RD)
    rv = r[(size_t)gw * RD + lane] * rclip[h * RD + lane];
  float rs = wsum(rv * rv);
  float rf = fminf(1.0f, r_thr / (sqrtf(rs) + 1e-6f)) * tau;
  if (lane < RD)
    rn[(size_t)gw * RD + lane] = rv * rf;
}

struct PfArgs {
  const __half* kc;
  const __half* vc;
  int cap, swa, U, p0;
  const __half* qn;
  const float* rn;
  const float* sink;
  const float* gate;
  const __half* bt;
  float* out;
  _Float16* out16;
};
namespace ia {
constexpr int PF_KS = HD + 8;  // K tile row stride (halves)
__device__ __forceinline__ uint32_t as_u32(v2h v) {
  return __builtin_bit_cast(uint32_t, v);
}
}  // namespace ia

namespace ia {
// shfl_xor(v, 16) across the half-waves without LDS.
__device__ __forceinline__ uint32_t xor16u(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false,
                                      false);
}
__device__ __forceinline__ float xor16f(float v) {
  return __builtin_bit_cast(float, xor16u(__builtin_bit_cast(uint32_t, v)));
}
}  // namespace ia

// Prefill flash attention: a block of NTT query tiles x 3 heads (one KV head)
// shares each K/V tile, the next tile's rows are loaded while this one
// computes (double-buffered LDS, one barrier per KT keys), and the
// probability exchange between half-waves uses permlanex16 / v_perm. The
// online softmax steps over 16 keys from the block's first key (0 on global
// layers, the window's start on window layers).
template<int NTT, int KT>
__global__ __launch_bounds__(NTT * 96) void attn_prefill_db(PfArgs a) {
  using namespace ia;
  constexpr int PF_KT = KT, PF_VS = KT + 8;
  constexpr int NT = NTT * 96;
  __shared__ __attribute__((aligned(16))) __half sK2[2][PF_KT][PF_KS];
  __shared__ __attribute__((aligned(16))) __half sVt2[2][HD][PF_VS];
  // Rows d0 .. d0 + BTR - 1 of the relative-bias table: every distance a key
  // tile meets (the same values bias_dot reads from global memory).
  constexpr int BTR = 16 * NTT + PF_KT;
  __shared__ __attribute__((aligned(16))) __half sB2[2][BTR][RD];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.y;
  const int hh = w % 3, tt = w / 3, h = kvh * 3 + hh;
  const int q16 = lane & 15, half = lane >> 4;
  const int tbase = blockIdx.x * 16 * NTT;
  const int tb = tbase + tt * 16;
  const int t = tb + q16;
  const bool rowvalid = t < a.U;
  const int tcl = min(t, a.U - 1);
  const int posq = rowvalid ? a.p0 + t : -1;

  v16h bq[8];
  {
    const uint4* qp = (const uint4*)(a.qn + ((size_t)tcl * NH + h) * HD);
#pragma unroll
    for (int s = 0; s < 8; s++)
      bq[s] = cat16(qp[2 * s], qp[2 * s + 1]);
  }
  v2h rp[8];
  {
    const float2* rq = (const float2*)(a.rn + ((size_t)tcl * NH + h) * RD);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      float2 f = rq[c];
      rp[c][0] = (h16)f.x;
      rp[c][1] = (h16)f.y;
    }
  }
  float m = rowvalid ? a.sink[(size_t)t * NH + h] : 0.f;
  float l = (half == 0) ? 1.f : 0.f;
  v8f acc[8];
#pragma unroll
  for (int d = 0; d < 8; d++)
    acc[d] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

  const uint32_t sel_lo = half ? 0x01000504u : 0x05040100u;
  const uint32_t sel_hi = half ? 0x03020706u : 0x07060302u;
  const int pfirst = a.p0 + tbase;
  const int plast = min(a.p0 + a.U - 1, a.p0 + tbase + 16 * NTT - 1);
  // The 16-key steps start at multiples of 16 (absolute positions), so a
  // row's softmax steps do not depend on where its prompt chunk starts.
  const int kfirst = a.swa ? max(0, pfirst - (WIN - 1)) & ~15 : 0;
  const int ntiles = (plast - kfirst + PF_KT) / PF_KT;
  const int wmin = a.p0 + tb, wmax = a.p0 + tb + 15;

  // Tile t + 1's K / V / bias rows are loaded into registers while tile t
  // computes, and written to the other LDS buffer after it: one barrier per
  // tile.
  constexpr int NK = (PF_KT * 16 + NT - 1) / NT, NV = (PF_KT * 8 + NT - 1) / NT;
  uint4 rk[NK], rv[NV][2], rb;
  auto gload = [&](int tile) {
    const int kbase = kfirst + tile * PF_KT;
    const int d0 = pfirst - (kbase + PF_KT - 1);
#pragma unroll
    for (int j = 0; j < NK; j++) {
      const int idx = tid + j * NT;
      if (idx < PF_KT * 16) {
        const int key = idx >> 4, c = idx & 15;
        const int pos = min(kbase + key, plast);
        rk[j] = *(const uint4*)(a.kc + (size_t)(pos % a.cap) * CELL + kvh * HD +
                                c * 8);
      }
    }
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = tid + j * NT;
      if (idx < (PF_KT / 2) * 16) {
        const int kp = idx >> 4, c = idx & 15;
        const int p0_ = min(kbase + 2 * kp, plast),
                  p1_ = min(kbase + 2 * kp + 1, plast);
        rv[j][0] = *(const uint4*)(a.vc + (size_t)(p0_ % a.cap) * CELL +
                                   kvh * HD + c * 8);
        rv[j][1] = *(const uint4*)(a.vc + (size_t)(p1_ % a.cap) * CELL +
                                   kvh * HD + c * 8);
      }
    }
    if (tid < BTR * 2) {
      const int row = tid >> 1, hf = tid & 1, d = d0 + row;
      if (d >= 0 && d < WIN)
        rb = *(const uint4*)(a.bt + (size_t)d * RD + hf * 8);
    }
  };
  auto store = [&](int tile, int b) {
    const int kbase = kfirst + tile * PF_KT;
    const int d0 = pfirst - (kbase + PF_KT - 1);
    if (tid < BTR * 2) {
      const int row = tid >> 1, hf = tid & 1, d = d0 + row;
      if (d >= 0 && d < WIN)
        *(uint4*)&sB2[b][row][hf * 8] = rb;
    }
#pragma unroll
    for (int j = 0; j < NK; j++) {
      const int idx = tid + j * NT;
      if (idx < PF_KT * 16) {
        const int key = idx >> 4, c = idx & 15;
        *(uint4*)&sK2[b][key][c * 8] = rk[j];
      }
    }
#pragma unroll
    for (int j = 0; j < NV; j++) {
      const int idx = tid + j * NT;
      if (idx < (PF_KT / 2) * 16) {
        const int kp = idx >> 4, c = idx & 15;
        uint4 v0 = rv[j][0], v1 = rv[j][1];
        uint32_t* dst = (uint32_t*)&sVt2[b][c * 8][2 * kp];
        constexpr int RS = PF_VS / 2;
        uint32_t x0[4] = {v0.x, v0.y, v0.z, v0.w},
                 x1[4] = {v1.x, v1.y, v1.z, v1.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          dst[(2 * e) * RS] = (x0[e] & 0xffffu) | (x1[e] << 16);
          dst[(2 * e + 1) * RS] = (x0[e] >> 16) | (x1[e] & 0xffff0000u);
        }
      }
    }
  };
  gload(0);
  store(0, 0);
  __syncthreads();
  if (ntiles > 1)
    gload(1);
  int tile = 0;
  // Far tiles (global layers): every key is at least WIN before the block's
  // first query, so no row masks a key or adds a bias, and the same steps run
  // without that work. The accumulators are rescaled only in the steps where
  // some row of the wave meets a new maximum: every other step's factor is
  // exp(0) = 1 (l keeps the general step's expression, which the compiler
  // would otherwise sum in another order).
  const int nfar =
      a.swa ? 0 : min(ntiles, max(0, (pfirst - WIN + 1 - kfirst) / PF_KT));
  for (; tile < nfar; tile++) {
    const int b = tile & 1;
    __half(*sK)[PF_KS] = sK2[b];
    __half(*sVt)[PF_VS] = sVt2[b];
#pragma unroll
    for (int kt = 0; kt < PF_KT / 16; kt++) {
      v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < 8; s++)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            cat16(*(const uint4*)&sK[kt * 16 + q16][s * 16],
                  *(const uint4*)&sK[kt * 16 + q16][s * 16 + 8]),
            bq[s], c);
      float mx = c[0];
#pragma unroll
      for (int i = 1; i < 8; i++)
        mx = fmaxf(mx, c[i]);
      mx = fmaxf(mx, xor16f(mx));
      const float mn = fmaxf(m, mx);
      const bool grew = __any(rowvalid && mn != m);
      float p[8], ps = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        p[i] = __expf(c[i] - mn);
        ps += p[i];
      }
      const float alpha = __expf(m - mn);
      l = l * alpha + ps;
      m = mn;
      if (grew) {
#pragma unroll
        for (int dt = 0; dt < 8; dt++)
          acc[dt] *= alpha;
      }
      uint32_t mine[4], theirs[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h pr;
        pr[0] = (h16)p[2 * i];
        pr[1] = (h16)p[2 * i + 1];
        mine[i] = as_u32(pr);
        theirs[i] = xor16u(mine[i]);
      }
      v16h pf;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const v2h lo =
            as_v2h(__builtin_amdgcn_perm(theirs[i], mine[i], sel_lo));
        const v2h hi =
            as_v2h(__builtin_amdgcn_perm(theirs[i], mine[i], sel_hi));
        pf[4 * i + 0] = lo[0];
        pf[4 * i + 1] = lo[1];
        pf[4 * i + 2] = hi[0];
        pf[4 * i + 3] = hi[1];
      }
#pragma unroll
      for (int dt = 0; dt < 8; dt++) {
        const __half* vp = &sVt[dt * 16 + q16][kt * 16];
        acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            cat16(*(const uint4*)vp, *(const uint4*)(vp + 8)), pf, acc[dt]);
      }
    }
    if (tile + 1 < ntiles)
      store(tile + 1, b ^ 1);
    __syncthreads();
    if (tile + 2 < ntiles)
      gload(tile + 2);
  }
  for (; tile < ntiles; tile++) {
    const int b = tile & 1;
    const int kbase = kfirst + tile * PF_KT;
    const int d0 = pfirst - (kbase + PF_KT - 1);
    __half(*sK)[PF_KS] = sK2[b];
    __half(*sVt)[PF_VS] = sVt2[b];
    __half(*sB)[RD] = sB2[b];
#pragma unroll
    for (int kt = 0; kt < PF_KT / 16; kt++) {
      const int pos0 = kbase + kt * 16;
      if (pos0 > wmax)
        continue;
      if (a.swa && (wmin - (pos0 + 15)) >= WIN)
        continue;
      v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < 8; s++)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            cat16(*(const uint4*)&sK[kt * 16 + q16][s * 16],
                  *(const uint4*)&sK[kt * 16 + q16][s * 16 + 8]),
            bq[s], c);
      float sv[8], mx = -INFINITY;
      const bool needb = rowvalid && (posq - (pos0 + 15) < WIN);
      const bool anyb = __any(needb);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int pos = pos0 + 2 * i + half, d = posq - pos;
        const bool ok = (d >= 0) && (pos <= plast) && (!a.swa || d < WIN);
        float s = c[i];
        if (anyb && ok && d < WIN)
          s += bias_dot(rp, &sB[0][0], d - d0);
        sv[i] = ok ? s : -INFINITY;
        mx = fmaxf(mx, sv[i]);
      }
      mx = fmaxf(mx, xor16f(mx));
      const float mn = fmaxf(m, mx);
      const float alpha = __expf(m - mn);
      float p[8], ps = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        p[i] = __expf(sv[i] - mn);
        ps += p[i];
      }
      l = l * alpha + ps;
      m = mn;
      // A step holding keys this row masks (its own step's later keys, the
      // keys before a window) is summed on the VALU with those keys' V
      // zeroed: the WMMA's sum depends on the A entries a zero probability
      // meets, and those rows differ with the prompt's chunking (later
      // tokens, ring rows). Its WMMA column is zero, which leaves acc exact.
      const bool edge =
          rowvalid &&
          ((posq >= pos0 && posq < pos0 + 15) ||
           (a.swa && posq - (WIN - 1) > pos0 && posq - (WIN - 1) <= pos0 + 15));
      uint32_t mine[4], theirs[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h pr;
        pr[0] = (h16)p[2 * i];
        pr[1] = (h16)p[2 * i + 1];
        mine[i] = as_u32(pr);
        theirs[i] = xor16u(mine[i]);
      }
      uint32_t pe[8] = {};  // edge: keys 2j, 2j + 1 of the step
      if (edge) {
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
          // key 2jj + e sits in the half e: its p is p[jj] of that half, in
          // pair jj >> 1 (element jj & 1) of mine / theirs
          const uint32_t a0 = half == 0 ? mine[jj >> 1] : theirs[jj >> 1];
          const uint32_t a1 = half == 1 ? mine[jj >> 1] : theirs[jj >> 1];
          const uint32_t lo = (jj & 1) ? (a0 >> 16) : (a0 & 0xffffu);
          const uint32_t hi = (jj & 1) ? (a1 >> 16) : (a1 & 0xffffu);
          pe[jj] = lo | (hi << 16);
        }
        // the other half holds the same row: its column is zero too
#pragma unroll
        for (int i = 0; i < 4; i++)
          mine[i] = theirs[i] = 0;
      }
      v16h pf;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const v2h lo =
            as_v2h(__builtin_amdgcn_perm(theirs[i], mine[i], sel_lo));
        const v2h hi =
            as_v2h(__builtin_amdgcn_perm(theirs[i], mine[i], sel_hi));
        pf[4 * i + 0] = lo[0];
        pf[4 * i + 1] = lo[1];
        pf[4 * i + 2] = hi[0];
        pf[4 * i + 3] = hi[1];
      }
#pragma unroll
      for (int dt = 0; dt < 8; dt++) {
        acc[dt] *= alpha;
        const __half* vp = &sVt[dt * 16 + q16][kt * 16];
        acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            cat16(*(const uint4*)vp, *(const uint4*)(vp + 8)), pf, acc[dt]);
      }
      if (edge) {
        uint32_t vm[8];  // halves of the keys this row sees
#pragma unroll
        for (int jj = 0; jj < 8; jj++) {
          uint32_t mk = 0;
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int d = posq - (pos0 + 2 * jj + e);
            if (d >= 0 && (!a.swa || d < WIN))
              mk |= 0xffffu << (16 * e);
          }
          vm[jj] = mk;
        }
#pragma unroll
        for (int dt = 0; dt < 8; dt++)
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const uint4* vr =
                (const uint4*)&sVt[dt * 16 + 2 * i + half][kt * 16];
            const uint4 v0 = vr[0], v1 = vr[1];
            const uint32_t vv[8] = {v0.x, v0.y, v0.z, v0.w,
                                    v1.x, v1.y, v1.z, v1.w};
            float e = 0.f;
#pragma unroll
            for (int jj = 0; jj < 8; jj++)
              e = __builtin_amdgcn_fdot2(as_v2h(pe[jj]),
                                         as_v2h(vv[jj] & vm[jj]), e, false);
            acc[dt][i] += e;
          }
      }
    }
    if (tile + 1 < ntiles)
      store(tile + 1, b ^ 1);
    __syncthreads();
    if (tile + 2 < ntiles)
      gload(tile + 2);
  }
  if (!rowvalid)
    return;
  const float inv = 1.0f / (l + xor16f(l));
  const size_t ob = ((size_t)t * NH + h) * HD;
  const float* gp = a.gate ? a.gate + (size_t)t * (NH * HD) + h * HD : nullptr;
#pragma unroll
  for (int dt = 0; dt < 8; dt++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int dim = dt * 16 + 2 * i + half;
      float o = acc[dt][i] * inv;
      if (gp)
        o *= 1.0f / (1.0f + __expf(-gp[dim]));
      if (a.out16)
        a.out16[ob + dim] = (_Float16)o;
      else
        a.out[ob + dim] = o;
    }
}

template<int N>
using I = std::integral_constant<int, N>;
void Chk(const char* what) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2 fast ") + what + ": " +
                             hipGetErrorString(e));
}

}  // namespace

// The fast GEMVs quantise their activations to int8 with a second plane holding
// the residual.
constexpr int g_res = 1;

// ---------------------------------------------------------------------------------------------
void ProjMulti(hipStream_t s, const ProjSpec& p, const float* x, int ni,
               float* y, int nb, int ldx, int ldy) {
  MultiP mp{};
  mp.ldx = ldx;
  mp.ldy = ldy;
  int cum = 0;
  for (int i = 0; i < p.n; ++i) {
    mp.W[i] = static_cast<const uint8_t*>(p.w[i]);
    mp.M[i] = p.m[i];
    mp.in[i] = p.in[i];
    mp.yoff[i] = cum;
    mp.cum[i] = cum;
    cum += p.m[i];
  }
  mp.cum[p.n] = cum;
  mp.n = p.n;
  mp.res = g_res;
  int trows_total = 0;
  mp.tn = p.tn;
  for (int i = 0; i < p.tn; ++i) {
    mp.tw[i] = p.tw[i];
    mp.ttype[i] = p.ttype[i] == 0 ? 0 : (p.ttype[i] == 1 ? 1 : 2);
    mp.trows[i] = p.trows[i];
    mp.tin[i] = p.tin[i];
    mp.ty[i] = p.ty[i];
    mp.tys[i] = p.tys[i];
    trows_total += p.trows[i];
  }
  auto run = [&](auto lpr, auto nit, auto bw) {
    constexpr int LPR = decltype(lpr)::value, NIT = decltype(nit)::value,
                  BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR) * NIT;
    mp.ntb = (trows_total + BW - 1) / BW;
    const dim3 grid((cum + RPB - 1) / RPB + mp.ntb);
    // several inputs: blocks of one input each when every change of input
    // falls on a block boundary
    bool one_input_blocks = ni > 1;
    for (int i = 1; i < mp.n; ++i)
      if (mp.in[i] != mp.in[i - 1] && mp.cum[i] % RPB != 0)
        one_input_blocks = false;
    if (nb == 1) {
      if (ni == 1)
        k_gemv_multi<2304, LPR, NIT, BW, 1>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
      else if (one_input_blocks)
        k_gemv_multi1<2304, LPR, NIT, BW, 1>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
      else
        k_gemv_multi<2304, LPR, NIT, BW, 3>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
    } else if (ni == 1) {
      switch (nb) {
        case 2:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 2>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        case 3:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 3>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        case 4:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 4>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        case 5:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 5>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        case 6:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 6>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        case 7:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 7>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
        default:
          k_gemv_multi<2304, LPR, NIT, BW, 1, 8>
              <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
          break;
      }
    } else if (one_input_blocks) {
      // up to 5 tokens in one launch; 6 to 8 as two halves (more tokens fill
      // the LDS and leave one block per CU)
      auto launch = [&](int n, const float* xx, float* yy, const MultiP& m) {
        switch (n) {
          case 2:
            k_gemv_multi1<2304, LPR, NIT, BW, 2>
                <<<grid, BW * WS, 0, s>>>(m, xx, yy, cum);
            break;
          case 3:
            k_gemv_multi1<2304, LPR, NIT, BW, 3>
                <<<grid, BW * WS, 0, s>>>(m, xx, yy, cum);
            break;
          case 4:
            k_gemv_multi1<2304, LPR, NIT, BW, 4>
                <<<grid, BW * WS, 0, s>>>(m, xx, yy, cum);
            break;
          default:
            k_gemv_multi1<2304, LPR, NIT, BW, 5>
                <<<grid, BW * WS, 0, s>>>(m, xx, yy, cum);
            break;
        }
      };
      if (nb <= 5) {
        launch(nb, x, y, mp);
      } else {
        const int h = (nb + 1) / 2;
        MultiP mp2 = mp;
        for (int i = 0; i < mp2.tn; ++i)
          mp2.ty[i] += h * (size_t)mp2.tys[i];
        launch(h, x, y, mp);
        launch(nb - h, x + h * (size_t)ldx, y + h * (size_t)ldy, mp2);
      }
    } else if (nb > 4) {
      // three inputs per token: groups of three rows (and a last group of one
      // or two), every row with its own arithmetic
      for (int r0 = 0; r0 < nb;) {
        const int g = nb - r0 >= 3 ? (nb - r0 == 4 ? 2 : 3) : nb - r0;
        MultiP mg = mp;
        for (int i = 0; i < mg.tn; ++i)
          mg.ty[i] += r0 * (size_t)mg.tys[i];
        const float* xg = x + r0 * (size_t)ldx;
        float* yg = y + r0 * (size_t)ldy;
        if (g == 3)
          k_gemv_multi<2304, LPR, NIT, BW, 3, 3>
              <<<grid, BW * WS, 0, s>>>(mg, xg, yg, cum);
        else if (g == 2)
          k_gemv_multi<2304, LPR, NIT, BW, 3, 2>
              <<<grid, BW * WS, 0, s>>>(mg, xg, yg, cum);
        else
          k_gemv_multi<2304, LPR, NIT, BW, 3>
              <<<grid, BW * WS, 0, s>>>(mg, xg, yg, cum);
        r0 += g;
      }
    } else {
      // three inputs per token: 4 tokens would need more than 64 KiB of LDS, so
      // they run as two pairs
      if (nb == 2)
        k_gemv_multi<2304, LPR, NIT, BW, 3, 2>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
      else if (nb == 3)
        k_gemv_multi<2304, LPR, NIT, BW, 3, 3>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
      else {
        MultiP mp2 = mp;
        for (int i = 0; i < mp2.tn; ++i)
          mp2.ty[i] += 2 * (size_t)mp2.tys[i];
        k_gemv_multi<2304, LPR, NIT, BW, 3, 2>
            <<<grid, BW * WS, 0, s>>>(mp, x, y, cum);
        k_gemv_multi<2304, LPR, NIT, BW, 3, 2><<<grid, BW * WS, 0, s>>>(
            mp2, x + 2 * (size_t)ldx, y + 2 * (size_t)ldy, cum);
      }
    }
  };
  // 4+ tokens: each block's quantized inputs take most of a CU's LDS, so
  // blocks of 16 waves keep twice the loads in flight (rows' sums unchanged);
  // one input and 6+ tokens: two rows per wave (more blocks, 0.4 to 0.7% per
  // verification step)
  if (nb >= 6 && ni == 1)
    run(I<16>{}, I<2>{}, I<16>{});
  else if (nb >= 4)
    run(I<16>{}, I<4>{}, I<16>{});
  else
    run(I<16>{}, I<4>{}, I<8>{});
  Chk("proj_multi");
}

void GemvQ8(hipStream_t s, int k, const void* w, const float* x, float* y,
            int m, float softcap, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  if (k == 2304) {
    // several tokens: two rows per wave (more loads in flight at the lower
    // occupancy); each row's dot product is the same
    auto run = [&](auto nit, auto bw) {
      constexpr int LPR = 32, NIT = decltype(nit)::value,
                    BW = decltype(bw)::value, RPB = BW * (WS / LPR) * NIT;
      const dim3 grid((m + RPB - 1) / RPB);
      switch (nb) {
        case 1:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 1>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 2:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 2>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 3:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 3>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 4:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 4>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 5:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 5>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 6:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 6>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        case 7:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 7>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
        default:
          k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 8>
              <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
          break;
      }
    };
    if (nb == 1)
      run(I<1>{}, I<8>{});
    else if (nb < 4)
      run(I<2>{}, I<8>{});
    else
      run(I<2>{}, I<16>{});
  } else if (k == 3072) {
    auto run = [&](auto bw) {
      constexpr int LPR = 16, NIT = 1, BW = decltype(bw)::value,
                    RPB = BW * (WS / LPR) * NIT;
      const dim3 grid((m + RPB - 1) / RPB);
      auto go = [&](auto nbc) {
        constexpr int NB = decltype(nbc)::value;
        k_gemv<FMT_Q8, 3072, LPR, NIT, BW, NB>
            <<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res);
      };
      switch (nb) {
        case 1:
          go(I<1>{});
          break;
        case 2:
          go(I<2>{});
          break;
        case 3:
          go(I<3>{});
          break;
        case 4:
          go(I<4>{});
          break;
        case 5:
          go(I<5>{});
          break;
        case 6:
          go(I<6>{});
          break;
        case 7:
          go(I<7>{});
          break;
        default:
          go(I<8>{});
          break;
      }
    };
    if (nb < 4)
      run(I<8>{});
    else
      run(I<16>{});
  } else {
    throw std::runtime_error("inkling2 fast: unsupported gemv K");
  }
  Chk("gemv");
}

void Router(hipStream_t s, const void* w, const float* bias, const float* x,
            float* logit_ws, unsigned* counter, int* ids, float* wts,
            float rscale, int nb, bool half) {
  if (half)
    k_router<2304, true><<<dim3(16, nb), 256, 0, s>>>(
        w, bias, x, logit_ws, counter, ids, wts, rscale);
  else
    k_router<2304, false><<<dim3(16, nb), 256, 0, s>>>(
        w, bias, x, logit_ws, counter, ids, wts, rscale);
  Chk("router");
}

// The kernels evaluate SITU with Inkling 2 Mini's constants.
static void CheckSitu(float beta, float lb) {
  if (beta != kSituBeta || lb != kSituLinearBeta)
    throw std::runtime_error("inkling2 fast: unsupported SITU constants");
}

void MoeGu(hipStream_t s, int fmt, const void* w, const float* escale,
           const float* x, const int* ids, float* a, int ff, float beta,
           float lb, int nb) {
  CheckSitu(beta, lb);
  const uint8_t* W = static_cast<const uint8_t*>(w);
  auto run = [&](auto fm, auto lpr, auto nit, auto bw) {
    constexpr int FM = decltype(fm)::value, LPR = decltype(lpr)::value,
                  NIT = decltype(nit)::value, BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR / 2) * NIT;
    k_moe_gu<FM, 2304, LPR, NIT, BW>
        <<<nb == 1 ? dim3((ff + RPB - 1) / RPB, 8)
                   : dim3(8 * nb, (ff + RPB - 1) / RPB),
           BW * WS, 0, s>>>(W, escale, x, ids, a, ff, g_res);
  };
  if (fmt == kFmtNvfp4) {
    run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<8>{});
  } else {
    run(I<FMT_Q8>{}, I<16>{}, I<1>{}, I<4>{});
  }
  Chk("moe_gu");
}

void QuantRows(hipStream_t s, int k, const float* x, int per, int ldt, int nb,
               void* qa, float* qs) {
  if (k == 2304)
    k_quant_rows<2304, 32><<<nb * per, 256, 0, s>>>(
        x, per, ldt, static_cast<v4u*>(qa), qs, g_res);
  else
    k_quant_rows<3072, 32><<<nb * per, 256, 0, s>>>(
        x, per, ldt, static_cast<v4u*>(qa), qs, g_res);
  Chk("quant_rows");
}
namespace {
constexpr int kMmaWaves = 8;
MultiP OneMatrix(const void* w, int m, int ldy) {
  MultiP mp{};
  mp.W[0] = static_cast<const uint8_t*>(w);
  mp.M[0] = m;
  mp.n = 1;
  mp.cum[1] = m;
  mp.res = g_res;
  mp.ldy = ldy;
  return mp;
}
}  // namespace
bool ProjMma(hipStream_t s, const ProjSpec& p, const void* qa, const float* qs,
             int per, float* y, int nb, int ldx, int ldy, const float* x) {
  constexpr int RPB = 16;
  MultiP mp{};
  mp.ldx = ldx;
  mp.ldy = ldy;
  int cum = 0;
  for (int i = 0; i < p.n; ++i) {
    mp.W[i] = static_cast<const uint8_t*>(p.w[i]);
    mp.M[i] = p.m[i];
    mp.in[i] = p.in[i];
    mp.yoff[i] = cum;
    mp.cum[i] = cum;
    cum += p.m[i];
  }
  mp.cum[p.n] = cum;
  mp.n = p.n;
  mp.res = g_res;
  for (int i = 1; i < mp.n; ++i)
    if (mp.in[i] != mp.in[i - 1] && mp.cum[i] % RPB != 0)
      return false;
  int trows_total = 0;
  mp.tn = p.tn;
  for (int i = 0; i < p.tn; ++i) {
    mp.tw[i] = p.tw[i];
    mp.ttype[i] = p.ttype[i] == 0 ? 0 : (p.ttype[i] == 1 ? 1 : 2);
    mp.trows[i] = p.trows[i];
    mp.tin[i] = p.tin[i];
    mp.ty[i] = p.ty[i];
    mp.tys[i] = p.tys[i];
    trows_total += p.trows[i];
  }
  if (mp.tn > 0) {
    constexpr int BW = 16;
    mp.ntb = (trows_total + BW - 1) / BW;
    k_tiny_wide<BW, 8><<<dim3(mp.ntb), BW * WS, 0, s>>>(mp, x, nb);
  }
  k_gemv_mma<2304, 16, kMmaWaves>
      <<<dim3((nb + 15) / 16, (cum + RPB - 1) / RPB), kMmaWaves * 32, 0, s>>>(
          mp, static_cast<const v4u*>(qa), qs, per, y, cum, nb, 0.f);
  Chk("proj_mma");
  return true;
}
void GemvMma(hipStream_t s, int k, const void* w, const void* qa,
             const float* qs, float* y, int m, float softcap, int nb) {
  constexpr int RPB = 16;
  const MultiP mp = OneMatrix(w, m, m);
  const dim3 grid((nb + 15) / 16, (m + RPB - 1) / RPB);
  if (k == 2304)
    k_gemv_mma<2304, 32, kMmaWaves><<<grid, kMmaWaves * 32, 0, s>>>(
        mp, static_cast<const v4u*>(qa), qs, 1, y, m, nb, softcap);
  else
    k_gemv_mma<3072, 16, kMmaWaves><<<grid, kMmaWaves * 32, 0, s>>>(
        mp, static_cast<const v4u*>(qa), qs, 1, y, m, nb, softcap);
  Chk("gemv_mma");
}
void MoeRouteGroup(hipStream_t s, const int* ids, int pairs, MoeGroups g) {
  k_route_group<<<1, 256, 0, s>>>(ids, pairs, g.sorted, g.ue, g.ub, g.un, g.nd);
  Chk("route_group");
}
void MoeGuGrouped(hipStream_t s, int fmt, const void* w, const float* escale,
                  MoeGroups g, int slots, float* a, int ff, float beta,
                  float lb) {
  CheckSitu(beta, lb);
  const uint8_t* W = static_cast<const uint8_t*>(w);
  const uint8_t* qx = static_cast<const uint8_t*>(g.qx);
  auto run = [&](auto fm) {
    constexpr int FM = decltype(fm)::value, RPB = kGgBw * (WS / 16 / 2);
    k_moe_gu_gg<FM, 2304, 16, kGgBw>
        <<<dim3(std::min(slots, kGgSlotBlocks), (ff + RPB - 1) / RPB),
           kGgBw * WS, 0, s>>>(W, escale, qx, g.sorted, g.ue, g.ub, g.un, g.nd,
                               a, ff, g_res);
  };
  if (fmt == kFmtNvfp4)
    run(I<FMT_NVFP4>{});
  else
    run(I<FMT_Q8>{});
  Chk("moe_gu_grouped");
}
void MoeQuantGrouped(hipStream_t s, int fmt, const float* x, int nv, int k,
                     MoeGroups g) {
  uint8_t* qx = static_cast<uint8_t*>(g.qx);
  const dim3 grid((nv * (k / 16) + 255) / 256);
  if (k == 2304) {
    if (fmt == kFmtNvfp4)
      k_quant_vec<2304, 64><<<grid, 256, 0, s>>>(x, nv, qx);
    else
      k_quant_vec<2304, 32><<<grid, 256, 0, s>>>(x, nv, qx);
  } else {
    if (fmt == kFmtNvfp4)
      k_quant_vec<1152, 64><<<grid, 256, 0, s>>>(x, nv, qx);
    else
      k_quant_vec<1152, 32><<<grid, 256, 0, s>>>(x, nv, qx);
  }
  Chk("moe_quant_grouped");
}
void MoeDownGrouped(hipStream_t s, int fmt, const void* w, const float* escale,
                    MoeGroups g, int slots, const float* wts, float* yp,
                    int d) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  const uint8_t* qx = static_cast<const uint8_t*>(g.qx);
  auto run = [&](auto fm) {
    constexpr int FM = decltype(fm)::value, RPB = kGgBw * (WS / 8);
    k_moe_down_gg<FM, 1152, 8, kGgBw>
        <<<dim3(std::min(slots, kGgSlotBlocks), (d + RPB - 1) / RPB),
           kGgBw * WS, 0, s>>>(W, escale, qx, g.sorted, g.ue, g.ub, g.un, g.nd,
                               wts, yp, d, g_res);
  };
  if (fmt == kFmtNvfp4)
    run(I<FMT_NVFP4>{});
  else
    run(I<FMT_Q8>{});
  Chk("moe_down_grouped");
}

void MoeDownE(hipStream_t s, int fmt, const void* w, const float* escale,
              const float* a, const int* ids, const float* wts, float* yp,
              int d, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  auto run = [&](auto fm, auto lpr, auto nit, auto bw) {
    constexpr int FM = decltype(fm)::value, LPR = decltype(lpr)::value,
                  NIT = decltype(nit)::value, BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR) * NIT;
    k_moe_down_e<FM, 1152, LPR, NIT, BW>
        <<<nb == 1 ? dim3((d + RPB - 1) / RPB, 8)
                   : dim3(8 * nb, (d + RPB - 1) / RPB),
           BW * WS, 0, s>>>(W, escale, a, ids, wts, yp, d, g_res);
  };
  if (fmt == kFmtNvfp4) {
    run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<8>{});
  } else {
    run(I<FMT_Q8>{}, I<8>{}, I<4>{}, I<4>{});
  }
  Chk("moe_down_e");
}

bool MoeGuDown(hipStream_t s, int fmt, const void* wgu, const float* gscale,
               const void* wd, const float* dscale, const float* x,
               const int* ids, const float* wts, float* a, float* yp,
               unsigned* cnt, int ff, int d, float beta, float lb, int nb) {
  if (fmt != kFmtNvfp4 || ff != 1152 || d != 2304)
    return false;
  CheckSitu(beta, lb);
  constexpr int BW = 8, LPR1 = 16, LPR2 = 8;
  const int gu_rb = ff / (BW * (WS / LPR1 / 2)), dn_rb = d / (BW * (WS / LPR2));
  const int npair = 8 * nb;
  k_moe_gu_down<2304, 1152, LPR1, LPR2, BW>
      <<<npair*(gu_rb + dn_rb), BW * WS, 0, s>>>(
          static_cast<const uint8_t*>(wgu), gscale,
          static_cast<const uint8_t*>(wd), dscale, x, ids, wts, a, yp, cnt, ff,
          d, npair, gu_rb, dn_rb, nb == 1 ? 1 : 0, g_res);
  Chk("moe_gu_down");
  return true;
}

// Reads every range of the list so that a later kernel finds it in the
// last-level cache; the loaded values only feed a store that never happens.
__global__ void __launch_bounds__(256) k_touchn(TouchList l, unsigned* sink) {
  // four loads in flight per thread: a few blocks reach the DRAM rate
  constexpr int U = 4;
  uint32_t acc = 0;
  for (int r = 0; r < l.n; ++r) {
    const uint4* p = static_cast<const uint4*>(l.p[r]);
    const size_t n16 = l.bytes[r] / 16;
    size_t i = (size_t)blockIdx.x * 256 * U + threadIdx.x;
    for (; i + 256 * (U - 1) < n16; i += (size_t)gridDim.x * 256 * U) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u)
        v[u] = p[i + 256 * u];
#pragma unroll
      for (int u = 0; u < U; ++u)
        acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n16; i += 256)
      acc ^= p[i].x;
  }
  if (acc == 0x9e3779b9u && sink)
    *sink = acc;
}
void TouchN(hipStream_t s, const TouchList& l, int blocks) {
  if (l.n == 0)
    return;
  k_touchn<<<blocks, 256, 0, s>>>(l, nullptr);
  Chk("touchn");
}

void Mix(hipStream_t s, const MixJob& j, float eps) {
  k_mix<<<dim3(j.nq, j.ntok), 512, 0, s>>>(j, eps);
  Chk("mix");
}
void Rms(hipStream_t s, const NormJobs& j, float eps, int nb) {
  k_rms<<<dim3(j.n, nb), 256, 0, s>>>(j, eps);
  Chk("rms");
}
void CtlGate(hipStream_t s, const CtlJobs& j, float eps, int nb) {
  if (j.half)
    k_ctl_gate<true><<<dim3(64, j.n, nb), 256, 0, s>>>(j, eps);
  else
    k_ctl_gate<false><<<dim3(64, j.n, nb), 256, 0, s>>>(j, eps);
  Chk("ctl_gate");
}
void CtlApplyRms(hipStream_t s, const CtlJobs& j, const NormJobs& nj,
                 unsigned* cnt, float eps, int nb) {
  if (j.half)
    k_ctl_apply_rms<true><<<dim3(9, j.n, nb), 256, 0, s>>>(j, nj, cnt, eps);
  else
    k_ctl_apply_rms<false><<<dim3(9, j.n, nb), 256, 0, s>>>(j, nj, cnt, eps);
  Chk("ctl apply rms");
}
void Tiny(hipStream_t s, const TinyJobs& j, int nb) {
  int rows = 0;
  for (int i = 0; i < j.n; ++i)
    rows += j.rows[i];
  k_tiny<<<dim3((rows + 7) / 8, nb), 256, 0, s>>>(j);
  Chk("tiny");
}
void KvPrep(hipStream_t s, const KvPrepArgs& a, int nb) {
  k_kvprep<<<dim3(8, 1, nb), 128, 0, s>>>(a);
  Chk("kvprep");
}
void AttnDecode(hipStream_t s, const AttnArgs& a, int nb) {
  ia::attn_decode<4><<<dim3(8, a.nsplit, nb), 128, 0, s>>>(a);
  ia::decode_reduce<<<dim3(24, 1, nb), 128, 0, s>>>(a);
  Chk("attn_decode");
}
void KvAttnDecode(hipStream_t s, const KvPrepArgs& kv, const AttnArgs& a,
                  int nb) {
  ia::attn_decode_fused<4><<<dim3(8, a.nsplit, nb), 128, 0, s>>>(a, kv);
  ia::decode_reduce<<<dim3(24, 1, nb), 128, 0, s>>>(a);
  Chk("attn_decode_fused");
}
void AttnDecodeRows(hipStream_t s, const AttnArgs& a0, int nb) {
  AttnArgs a = a0;
  a.nrows = nb;
  // up to 5 rows (15 of the 16 query slots) per block; 6 to 8 rows as two
  // groups in the same launch
  const int groups = nb <= 5 ? 1 : 2, per = (nb + groups - 1) / groups;
  const dim3 grid(8 * groups, a.nsplit, 1);
  switch (per) {
    case 2:
      ia::attn_decode_rows<4, 2><<<grid, 128, 0, s>>>(a);
      break;
    case 3:
      ia::attn_decode_rows<4, 3><<<grid, 128, 0, s>>>(a);
      break;
    case 4:
      ia::attn_decode_rows<4, 4><<<grid, 128, 0, s>>>(a);
      break;
    default:
      ia::attn_decode_rows<4, 5><<<grid, 128, 0, s>>>(a);
      break;
  }
  ia::decode_reduce<<<dim3(24, 1, nb), 128, 0, s>>>(a);
  Chk("attn_decode_rows");
}
void ConvResid(hipStream_t s, const ConvResidArgs& a, int nb) {
  k_conv_resid<<<dim3(9, 1, nb), 256, 0, s>>>(a);
  Chk("conv_resid");
}
__global__ void __launch_bounds__(256)
    k_conv_commit(const SlotEntry* tbl, const float* cap, int keep, int row0) {
  const int ch = blockIdx.x * 256 + threadIdx.x, slot = blockIdx.y;
  if (ch >= 6656)
    return;
  float* st = tbl[slot].conv + (size_t)ch * 3;
  // the slot's rows [kVerifyRows][6656], from step row row0
  const float* cp = cap + ((size_t)slot * kVerifyRows + row0) * 6656 + ch;
  // the last three of [st[0..3), cp[0..keep)]
  float v[3];
  for (int i = 0; i < 3; ++i) {
    const int j = keep + i;
    v[i] = j < 3 ? st[j] : cp[(size_t)(j - 3) * 6656];
  }
  for (int i = 0; i < 3; ++i)
    st[i] = v[i];
}
__global__ void __launch_bounds__(256)
    k_rows_argmax1(const float* x, int n, float* ws) {
  const int r = blockIdx.y;
  const float* xr = x + (size_t)r * n;
  float v = -INFINITY;
  int ix = 0x7fffffff;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
    const float y = xr[i];
    if (y > v) {  // NaN and -inf never win, as in the greedy sampler
      v = y;
      ix = i;
    }
  }
  __shared__ float bv[8];
  __shared__ int bi[8];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o, 32);
    const int oi = __shfl_xor(ix, o, 32);
    if (ov > v || (ov == v && oi < ix)) {
      v = ov;
      ix = oi;
    }
  }
  if (threadIdx.x % 32 == 0) {
    bv[threadIdx.x / 32] = v;
    bi[threadIdx.x / 32] = ix;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < 8; ++w)
      if (bv[w] > v || (bv[w] == v && bi[w] < ix)) {
        v = bv[w];
        ix = bi[w];
      }
    float* o = ws + ((size_t)r * gridDim.x + blockIdx.x) * 2;
    o[0] = v;
    o[1] = __int_as_float(ix);
  }
}
__global__ void k_rows_argmax2(const float* ws, int nb, int* out) {
  const int r = threadIdx.x;
  float v = -INFINITY;
  int ix = 0x7fffffff;
  for (int b = 0; b < nb; ++b) {
    const float y = ws[((size_t)r * nb + b) * 2];
    const int i = __float_as_int(ws[((size_t)r * nb + b) * 2 + 1]);
    if (y > v || (y == v && i < ix)) {
      v = y;
      ix = i;
    }
  }
  out[r] = ix == 0x7fffffff ? 0 : ix;
}
void RowsArgmax(hipStream_t s, const float* x, int rows, int n, float* ws,
                int* out) {
  constexpr int nb = 64;
  k_rows_argmax1<<<dim3(nb, rows), 256, 0, s>>>(x, n, ws);
  k_rows_argmax2<<<1, rows, 0, s>>>(ws, nb, out);
  Chk("rows_argmax");
}

namespace {
constexpr int kLsBlocks = 64;
constexpr int kLsStride = 2 * kLsBlocks + 1;

__device__ __forceinline__ double wsum_d(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o, 32);
  return v;
}

__global__ void __launch_bounds__(256) k_ls_max(LinearSampleArgs a) {
  const int r = blockIdx.y;
  if (!(a.temp[r] > 0.f))
    return;
  const int per = (a.vocab + kLsBlocks - 1) / kLsBlocks;
  const int lo = blockIdx.x * per, hi = min(a.vocab, lo + per);
  const float* x = a.logits + (size_t)r * a.vocab;
  float v = -INFINITY;
  for (int i = lo + threadIdx.x; i < hi; i += 256) {
    const float y = x[i];
    if (isfinite(y) && y > v)
      v = y;
  }
  __shared__ float red[8];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor(v, o, 32));
  if (threadIdx.x % 32 == 0)
    red[threadIdx.x / 32] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < 8; ++w)
      v = fmaxf(v, red[w]);
    a.ws[(size_t)r * kLsStride + blockIdx.x] = static_cast<double>(v);
  }
}

__global__ void __launch_bounds__(256) k_ls_sum(LinearSampleArgs a) {
  const int r = blockIdx.y;
  const float t_f = a.temp[r];
  if (!(t_f > 0.f))
    return;
  __shared__ double red[8];
  __shared__ double mx_s;
  double* ws = a.ws + (size_t)r * kLsStride;
  if (threadIdx.x == 0) {
    double m = -INFINITY;
    for (int b = 0; b < kLsBlocks; ++b)
      if (ws[b] > m)
        m = ws[b];
    mx_s = m;
  }
  __syncthreads();
  const double m = mx_s;
  const double inv_t = 1.0 / static_cast<double>(t_f);
  const float mp = a.min_p[r];
  const double thr =
      mp > 0.f ? m + static_cast<double>(t_f) * log(static_cast<double>(mp))
               : -INFINITY;
  const int per = (a.vocab + kLsBlocks - 1) / kLsBlocks;
  const int lo = blockIdx.x * per, hi = min(a.vocab, lo + per);
  const float* x = a.logits + (size_t)r * a.vocab;
  double z = 0.0;
  for (int i = lo + threadIdx.x; i < hi; i += 256) {
    const float y = x[i];
    if (isfinite(y) && static_cast<double>(y) >= thr)
      z += exp((static_cast<double>(y) - m) * inv_t);
  }
  z = wsum_d(z);
  if (threadIdx.x % 32 == 0)
    red[threadIdx.x / 32] = z;
  __syncthreads();
  if (threadIdx.x == 0) {
    double s = 0.0;
    for (int w = 0; w < 8; ++w)
      s += red[w];
    ws[kLsBlocks + blockIdx.x] = s;
    if (blockIdx.x == 0)
      ws[2 * kLsBlocks] = m;
  }
}

__global__ void __launch_bounds__(256) k_ls_pick(LinearSampleArgs a) {
  const int r = blockIdx.x;
  const float t_f = a.temp[r];
  if (!(t_f > 0.f))
    return;
  __shared__ int chunk;
  __shared__ double before, total;
  __shared__ double part[256];
  const double* ws = a.ws + (size_t)r * kLsStride;
  const double m = ws[2 * kLsBlocks];
  const double inv_t = 1.0 / static_cast<double>(t_f);
  const float mp = a.min_p[r];
  const double thr =
      mp > 0.f ? m + static_cast<double>(t_f) * log(static_cast<double>(mp))
               : -INFINITY;
  if (threadIdx.x == 0) {
    double z = 0.0;
    for (int b = 0; b < kLsBlocks; ++b)
      z += ws[kLsBlocks + b];
    const double target = a.u[r] * z;
    // Rounding can leave the target at or past the last running sum: pick
    // the last chunk holding any allowed token then, never an empty one.
    double run = 0.0, run_last = 0.0;
    int c = -1, last = kLsBlocks - 1;
    for (int b = 0; b < kLsBlocks; ++b) {
      const double w = ws[kLsBlocks + b];
      const double nxt = run + w;
      if (w > 0.0) {
        last = b;
        run_last = run;
      }
      if (target < nxt) {
        c = b;
        break;
      }
      run = nxt;
    }
    chunk = c >= 0 ? c : last;
    before = c >= 0 ? run : run_last;
    total = z;
  }
  __syncthreads();
  const int per = (a.vocab + kLsBlocks - 1) / kLsBlocks;
  const int lo = chunk * per, hi = min(a.vocab, lo + per);
  const int len = hi - lo, sl = (len + 255) / 256;
  const int a0 = lo + threadIdx.x * sl, a1 = min(hi, a0 + sl);
  const float* x = a.logits + (size_t)r * a.vocab;
  double s = 0.0;
  for (int i = a0; i < a1; ++i) {
    const float y = x[i];
    if (isfinite(y) && static_cast<double>(y) >= thr)
      s += exp((static_cast<double>(y) - m) * inv_t);
  }
  part[threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    const double target = a.u[r] * total;
    double run = before;
    // The selected slice is the one whose sum crosses the target, else (when
    // rounding leaves the target past every sum) the last non-empty slice;
    // within it the crossing token, else its last allowed token. Every
    // fallback is an allowed token.
    int sel = -1, last = -1;
    for (int th = 0; th < 256; ++th) {
      if (part[th] > 0.0)
        last = th;
      if (target < run + part[th]) {
        sel = th;
        break;
      }
      run += part[th];
    }
    if (sel < 0)
      sel = last;
    int tok = hi - 1;  // only when the row allows no token at all
    if (sel >= 0) {
      const int b0 = lo + sel * sl, b1 = min(hi, b0 + sl);
      for (int i = b0; i < b1; ++i) {
        const float y = x[i];
        if (isfinite(y) && static_cast<double>(y) >= thr) {
          tok = i;
          run += exp((static_cast<double>(y) - m) * inv_t);
          if (target < run)
            break;
        }
      }
    }
    a.out[r] = tok;
  }
}
}  // namespace

void RowsLinearSample(hipStream_t s, const LinearSampleArgs& a) {
  if (a.rows <= 0)
    return;
  k_ls_max<<<dim3(kLsBlocks, a.rows), 256, 0, s>>>(a);
  k_ls_sum<<<dim3(kLsBlocks, a.rows), 256, 0, s>>>(a);
  k_ls_pick<<<a.rows, 256, 0, s>>>(a);
  Chk("rows_linear_sample");
}

void CommitVerifyConv(hipStream_t s, const SlotEntry* tbl, const float* cap,
                      int keep, int row0) {
  k_conv_commit<<<dim3(26, 48), 256, 0, s>>>(tbl, cap, keep, row0);
  Chk("conv_commit");
}
void RouteBuild(hipStream_t s, const int* ids, int p, int e, int* off,
                int* sorted) {
  k_route_build<<<1, 1024, 0, s>>>(ids, p, e, off, sorted);
  Chk("route_build");
}

// ---------------------------------------------------------------------------------------------
// prefill host API
void CvtF16(hipStream_t s, const float* x, void* h, size_t n) {
  k_cvt_f16<<<(unsigned)((n / 4 + 255) / 256), 256, 0, s>>>(
      x, static_cast<_Float16*>(h), n);
  Chk("cvt_f16");
}

namespace {
__global__ void __launch_bounds__(256)
    k_check_bf16_exact(const uint32_t* __restrict__ x, size_t n,
                       unsigned* non_exact) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t n4 = n / 4;
  if (i < n4) {
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    if (((v.x | v.y | v.z | v.w) & 0xFFFFu) != 0u)
      *non_exact = 1u;
  } else if (i == n4) {
    for (size_t k = n4 * 4; k < n; ++k)
      if ((x[k] & 0xFFFFu) != 0u)
        *non_exact = 1u;
  }
}

__global__ void __launch_bounds__(256)
    k_f32_to_bf16(const uint32_t* __restrict__ x, uint16_t* __restrict__ dst,
                  size_t n) {
  const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  const size_t n4 = n / 4;
  if (i < n4) {
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    const uint2 out = make_uint2((v.x >> 16) | (v.y & 0xFFFF0000u),
                                 (v.z >> 16) | (v.w & 0xFFFF0000u));
    reinterpret_cast<uint2*>(dst)[i] = out;
  } else if (i == n4) {
    for (size_t k = n4 * 4; k < n; ++k)
      dst[k] = static_cast<uint16_t>(x[k] >> 16);
  }
}

__global__ void __launch_bounds__(256)
    k_pad_rows_to_f32(const void* __restrict__ src, int type, int rows,
                      int padded_rows, int cols, float* __restrict__ dst) {
  const size_t total = (size_t)padded_rows * cols;
  const size_t idx = (size_t)blockIdx.x * 256 + threadIdx.x;
  if (idx >= total)
    return;
  const size_t valid = (size_t)rows * cols;
  if (idx >= valid) {
    dst[idx] = 0.f;
    return;
  }
  if (type == 0)
    dst[idx] = static_cast<const float*>(src)[idx];
  else if (type == 1)
    dst[idx] = __half2float(static_cast<const __half*>(src)[idx]);
  else
    dst[idx] = __uint_as_float(
        static_cast<uint32_t>(static_cast<const uint16_t*>(src)[idx]) << 16);
}
}  // namespace

void CheckBf16Exact(hipStream_t s, const float* x, size_t n,
                    unsigned* non_exact) {
  const size_t threads = n / 4 + 1;
  k_check_bf16_exact<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>(
      reinterpret_cast<const uint32_t*>(x), n, non_exact);
  Chk("check_bf16_exact");
}

void F32ToBf16(hipStream_t s, const float* x, void* dst, size_t n) {
  const size_t threads = n / 4 + 1;
  k_f32_to_bf16<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>(
      reinterpret_cast<const uint32_t*>(x), static_cast<uint16_t*>(dst), n);
  Chk("f32_to_bf16");
}

__global__ void k_ctl_weights_k4(const uint32_t* __restrict__ x,
                                 uint16_t* __restrict__ d, int rows) {
  const int i = blockIdx.x * 256 + threadIdx.x;  // source index c * 64 + k
  if (i >= rows * 64)
    return;
  const int c = i >> 6, k = i & 63;
  d[((size_t)(k >> 2) * rows + c) * 4 + (k & 3)] = (uint16_t)(x[i] >> 16);
}
void CtlWeightsK4(hipStream_t s, const float* x, void* dst, int rows) {
  k_ctl_weights_k4<<<(rows * 64 + 255) / 256, 256, 0, s>>>(
      reinterpret_cast<const uint32_t*>(x), static_cast<uint16_t*>(dst), rows);
  Chk("ctl_weights_k4");
}

void PadRowsToF32(hipStream_t s, const void* src, int type, int rows,
                  int padded_rows, int cols, float* dst) {
  const size_t total = (size_t)padded_rows * cols;
  k_pad_rows_to_f32<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
      src, type, rows, padded_rows, cols, dst);
  Chk("pad_rows_to_f32");
}

void GemmDense(hipStream_t s, const void* w, int m, int k, const void* ah,
               float* out, const GemmItems& it) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp = 0;
  p.ah = static_cast<const _Float16*>(ah);
  p.nitems = it.nitems;
  p.item_e = it.item_e;
  p.item_start = it.item_start;
  p.item_n = it.item_n;
  p.out = out;
  p.ldo = m;
  constexpr int RG = 2, BM = 32 * RG;
  if (m % BM)
    throw std::runtime_error(
        "inkling2 fast: dense gemm rows not a multiple of 64");
  const int rgn = m / BM;
  const dim3 grid(it.max_items * rgn);
  if (it.tile == 128) {
    if (m % 128)
      throw std::runtime_error(
          "inkling2 fast: dense gemm rows not a multiple of 128");
    const int rgn128 = m / 128;
    const dim3 grid128(it.max_items * rgn128);
    if (k == 2304)
      k_g128<FMT_Q8, 0, 2304><<<grid128, 256, 0, s>>>(p, rgn128);
    else if (k == 3072)
      k_g128<FMT_Q8, 0, 3072><<<grid128, 256, 0, s>>>(p, rgn128);
    else
      throw std::runtime_error("inkling2 fast: unsupported dense gemm K");
  } else if (k == 2304)
    k_gemm4<FMT_Q8, 0, 64, 2304, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
  else if (k == 3072)
    k_gemm4<FMT_Q8, 0, 64, 3072, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
  else if (k == 9216)  // the DSpark drafter's FFN down projection
    k_gemm4<FMT_Q8, 0, 64, 9216, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
  else
    throw std::runtime_error("inkling2 fast: unsupported dense gemm K");
  Chk("gemm_dense");
}

void MoeRoute(hipStream_t s, const int* ids, const float* wts, int p, int tile,
              const MoeWs& ws) {
  hipMemsetAsync(ws.cnt, 0, 128 * 4, s);
  k_hist<<<(p + 255) / 256, 256, 0, s>>>(ids, p, ws.cnt);
  k_scan_items<<<1, 128, 0, s>>>(ws.cnt, tile, ws.goff, ws.fill, ws.nitems,
                                 ws.item_e, ws.item_start, ws.item_n);
  k_scatter<<<(p + 255) / 256, 256, 0, s>>>(ids, wts, p, ws.goff, ws.fill,
                                            ws.postok, ws.pair, ws.posw);
  Chk("moe_route");
}

template<int FMT, int BN, int RG>
static void gu_launch(hipStream_t s, GP3& p, int max_items) {
  constexpr int rgn = 1152 / (16 * RG);
  k_gemm4<FMT, 1, BN, 2304, RG>
      <<<dim3(max_items * rgn), 64 * RG, 0, s>>>(p, rgn);
}
template<int FMT, int BN, int RG>
static void dn_launch(hipStream_t s, GP3& p, int max_items) {
  constexpr int rgn = 2304 / (32 * RG);
  k_gemm4<FMT, 2, BN, 1152, RG>
      <<<dim3(max_items * rgn), 64 * RG, 0, s>>>(p, rgn);
}

void MoeGuPrefill(hipStream_t s, int fmt, int tile, const void* w,
                  const float* escale, const void* ah, void* act16,
                  const MoeWs& ws, int max_items, float beta, float lb,
                  int ff) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp =
      (size_t)(fmt == kFmtNvfp4 ? 2304 / 64 * 36 : 2304 / 32 * 34) * 2 * ff;
  p.escale = escale;
  p.ah = static_cast<const _Float16*>(ah);
  p.actrow = ws.postok;
  p.nitems = ws.nitems;
  p.item_e = ws.item_e;
  p.item_start = ws.item_start;
  p.item_n = ws.item_n;
  p.outh = static_cast<_Float16*>(act16);
  CheckSitu(beta, lb);
  if (fmt == kFmtNvfp4) {
    if (tile == 32)
      gu_launch<FMT_NVFP4, 32, 2>(s, p, max_items);
    else if (tile == 128)
      k_gu128<FMT_NVFP4><<<dim3(max_items * 18), 256, 0, s>>>(p, 18);
    else
      gu_launch<FMT_NVFP4, 64, 4>(s, p, max_items);
  } else {
    if (tile == 32)
      gu_launch<FMT_Q8, 32, 2>(s, p, max_items);
    else if (tile == 128)
      k_gu128<FMT_Q8><<<dim3(max_items * 18), 256, 0, s>>>(p, 18);
    else
      gu_launch<FMT_Q8, 64, 4>(s, p, max_items);
  }
  Chk("moe_gu_prefill");
}

void MoeDownPrefill(hipStream_t s, int fmt, int tile, const void* w,
                    const float* escale, const void* act16, float* pairs,
                    const MoeWs& ws, int max_items, int d) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp = (size_t)(fmt == kFmtNvfp4 ? 1152 / 64 * 36 : 1152 / 32 * 34) * d;
  p.escale = escale;
  p.ah = static_cast<const _Float16*>(act16);
  p.actrow = nullptr;
  p.nitems = ws.nitems;
  p.item_e = ws.item_e;
  p.item_start = ws.item_start;
  p.item_n = ws.item_n;
  p.out = pairs;
  p.ldo = d;
  p.postok = ws.postok;
  p.pair = ws.pair;
  p.posw = ws.posw;
  if (tile == 128) {
    if (fmt == kFmtNvfp4)
      k_g128<FMT_NVFP4, 2, 1152><<<dim3(max_items * 18), 256, 0, s>>>(p, 18);
    else
      k_g128<FMT_Q8, 2, 1152><<<dim3(max_items * 18), 256, 0, s>>>(p, 18);
  } else if (fmt == kFmtNvfp4) {
    if (tile == 32)
      dn_launch<FMT_NVFP4, 32, 2>(s, p, max_items);
    else
      dn_launch<FMT_NVFP4, 64, 4>(s, p, max_items);
  } else {
    if (tile == 32)
      dn_launch<FMT_Q8, 32, 2>(s, p, max_items);
    else
      dn_launch<FMT_Q8, 64, 4>(s, p, max_items);
  }
  Chk("moe_down_prefill");
}

void MoeTail(hipStream_t s, const float* pairs, float* state, const float* w,
             float* cur, int t, int c, float scale) {
  k_moe_tail<8><<<dim3((c + 255) / 256, (t + 15) / 16), 256, 0, s>>>(
      pairs, state, w, cur, t, c, scale);
  k_moe_tail_state<8><<<(c + 127) / 128, 128, 0, s>>>(pairs, state, t, c);
  Chk("moe_tail");
}
void MoeTailStateCapture(hipStream_t s, const float* pairs,
                         const float* old_state, float* dst_state, int t,
                         int c) {
  k_moe_tail_state_capture<8>
      <<<(c + 127) / 128, 128, 0, s>>>(pairs, old_state, dst_state, t, c);
  Chk("moe_tail_state_capture");
}

void ConvAdd(hipStream_t s, const float* x, float* state, const float* w,
             float* cur, int t, int c, float scale) {
  k_moe_tail<1><<<dim3((c + 255) / 256, (t + 15) / 16), 256, 0, s>>>(
      x, state, w, cur, t, c, scale);
  k_moe_tail_state<1><<<(c + 127) / 128, 128, 0, s>>>(x, state, t, c);
  Chk("conv_add");
}

void AttnPrefill(hipStream_t s, const PrefillAttn& a) {
  if (a.kraw) {
    const KvConv cv{a.kraw, a.vraw, a.kst,  a.vst,       a.kw,
                    a.vw,   a.kdyn, a.vdyn, a.conv_scale};
    k_kv_conv_write<<<(a.n * 8 + 3) / 4, 128, 0, s>>>(
        cv, a.n, a.pos0, a.cap, static_cast<__half*>(a.kc),
        static_cast<__half*>(a.vc), a.qk_thr, a.eps);
  } else {
    k_kv_write<<<(a.n * 8 + 3) / 4, 128, 0, s>>>(
        a.k, a.v, a.n, a.pos0, a.cap, static_cast<__half*>(a.kc),
        static_cast<__half*>(a.vc), a.qk_thr, a.eps);
  }
  k_qr_prep<<<(a.n * 24 + 3) / 4, 128, 0, s>>>(
      a.q, a.r, a.qscale, a.rclip, a.n, a.pos0, a.swa, a.qk_thr, a.r_thr,
      a.tau_floor, a.tau_alpha, static_cast<__half*>(a.qn), a.rn);
  PfArgs pa{};
  pa.kc = static_cast<const __half*>(a.kc);
  pa.vc = static_cast<const __half*>(a.vc);
  pa.cap = a.cap;
  pa.swa = a.swa;
  pa.U = a.n;
  pa.p0 = a.pos0;
  pa.qn = static_cast<const __half*>(a.qn);
  pa.rn = a.rn;
  pa.sink = a.sink;
  pa.gate = a.gate;
  pa.bt = static_cast<const __half*>(a.bt);
  pa.out = a.out;
  pa.out16 = static_cast<_Float16*>(a.out16);
  // Global visits past 1K keys with enough queries to fill the GPU take
  // 128-query blocks (pp32K attention 33.6 -> 27.4 ms per visit). Window
  // visits keep 64-query blocks: their first key follows the block's first
  // query, so a larger block would regroup their softmax steps.
  if (!a.swa && a.pos0 >= 1024 && a.n >= 512)
    attn_prefill_db<8, 32><<<dim3((a.n + 127) / 128, 8), 768, 0, s>>>(pa);
  else
    attn_prefill_db<4, 32><<<dim3((a.n + 63) / 64, 8), 384, 0, s>>>(pa);
  Chk("attn_prefill");
}

void CtlPrefill(hipStream_t s, const CtlPJobs& jobs, int t, float eps) {
  GateJobs g{};
  for (int i = 0; i < jobs.n; ++i) {
    const CtlPJob& c = jobs.j[i];
    g.j[i] = {c.x, c.ldx, c.msg, c.ldm, c.down, 4608,   2304,  2304,
              1,   c.g64, 64,    0,     c.minv, c.xinv, c.half};
  }
  // One job has too few token blocks for k_gate_p64's serial chunks; three
  // jobs keep the GPU busy and k_gate_p64 moves fewer bytes.
  if (jobs.n == 1)
    k_gate_v2<64, 64><<<dim3((t + 63) / 64, 1, 1), 256, 0, s>>>(g, t, eps);
  else
    k_gate_p64<<<dim3((t + 63) / 64, 1, jobs.n), 256, 0, s>>>(g, t, eps);
  if (jobs.n > 0 && jobs.j[0].nw) {
    // 4 tokens per block (2 and 8 measured slower)
    constexpr int R = 4;
    k_ctl_apply_norm<R>
        <<<dim3((t + R - 1) / R * jobs.n), 256, 0, s>>>(jobs, t, eps);
  } else if (jobs.n > 0 && jobs.j[0].half)
    k_ctl_apply_v4<true>
        <<<dim3(36, (t + 63) / 64, jobs.n), 256, 0, s>>>(jobs, t);
  else
    k_ctl_apply_v4<false>
        <<<dim3(36, (t + 63) / 64, jobs.n), 256, 0, s>>>(jobs, t);
  Chk("ctl_prefill");
}

void RouterPrefill(hipStream_t s, const void* w, const float* x, float* logits,
                   int t, bool half) {
  GateJobs g{};
  g.j[0] = {x, 2304,   nullptr, 0, w,       2304,    2304,        0,
            0, logits, 128,     0, nullptr, nullptr, half ? 1 : 0};
  k_gate_p64<<<dim3((t + 63) / 64, 2, 1), 256, 0, s>>>(g, t, 0.f);
  Chk("router_prefill");
}

void TopkPrefill(hipStream_t s, const float* logits, const float* bias,
                 int* ids, float* wts, int t, float rscale) {
  k_topk_p<<<(t + 7) / 8, 256, 0, s>>>(logits, bias, ids, wts, t, rscale);
  Chk("topk_prefill");
}

void SkinnyPrefill(hipStream_t s, const float* x, int ldx, const float* w,
                   int rows, float* out, int ldo, int t) {
  GateJobs g{};
  g.j[0] = {x, ldx, nullptr, 0, w, 2304, 2304, 0, 0, out, ldo, rows};
  k_gate_v2<64, 32><<<dim3((t + 63) / 64, 1, 1), 256, 0, s>>>(g, t, 0.f);
  Chk("skinny_prefill");
}

}  // namespace gufo::models::inkling2::fast
