// Simple HIP kernels for Inkling 2 Mini. See kernels.hpp.
// MIT. Dequantisation follows llama.cpp ggml-quants.c / ggml-common.h (MIT).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "src/models/inkling2/hip/kernels.hpp"

namespace gufo::models::inkling2::hip {
namespace {

constexpr int kWave = 32;      // gfx11 runs wave32
constexpr int kGemmWaves = 8;  // one weight row per wave

__device__ __forceinline__ float f16_to_f32(uint16_t u) {
  __half_raw r;
  r.x = u;
  return __half2float(__half(r));
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
__device__ __forceinline__ float bf16_to_f32(uint16_t u) {
  return __uint_as_float(static_cast<uint32_t>(u) << 16);
}

__constant__ int8_t kFp4[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                0, -1, -2, -3, -4, -6, -8, -12};
__constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1,    13,   25,  38,  53,  69,  89,  113};

// ggml_ue4m3_to_fp32 (includes the 0.5 factor of the doubled e2m1 table).
__device__ __forceinline__ float ue4m3(uint32_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0f;
  const int e = (x >> 3) & 0xF;
  const int m = x & 0x7;
  const float raw = e == 0 ? ldexpf(static_cast<float>(m), -9)
                           : ldexpf(1.0f + static_cast<float>(m) / 8.0f, e - 7);
  return raw * 0.5f;
}

// get_scale_min_k4 of ggml-quants.c.
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q,
                                             uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// A type is read in "items": IE consecutive elements dequantised together. A
// block holds IPB items and BB bytes. deq(block, sub, w) writes the IE elements
// of item `sub`.
template<int T>
struct Q;

template<>
struct Q<kTF32> {
  static constexpr int IE = 4, IPB = 1, BB = 16;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    for (int i = 0; i < 4; ++i)
      w[i] = __uint_as_float(ld32(b + 4 * i));
  }
};
template<>
struct Q<kTF16> {
  static constexpr int IE = 8, IPB = 1, BB = 16;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    for (int i = 0; i < 8; ++i)
      w[i] = f16_to_f32(ld16(b + 2 * i));
  }
};
template<>
struct Q<kTBF16> {
  static constexpr int IE = 8, IPB = 1, BB = 16;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    for (int i = 0; i < 8; ++i)
      w[i] = bf16_to_f32(ld16(b + 2 * i));
  }
};
template<>
struct Q<kTQ8_0> {
  static constexpr int IE = 32, IPB = 1, BB = 34;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    const float d = f16_to_f32(ld16(b));
    for (int i = 0; i < 32; ++i)
      w[i] = d * static_cast<float>(static_cast<int8_t>(b[2 + i]));
  }
};
template<>
struct Q<kTNVFP4> {
  static constexpr int IE = 16, IPB = 4, BB = 36;
  __device__ static void deq(const uint8_t* b, int sub, float* w) {
    const float d = ue4m3(b[sub]);
    const uint8_t* qs = b + 4 + sub * 8;
    for (int j = 0; j < 8; ++j) {
      w[j] = static_cast<float>(kFp4[qs[j] & 0xF]) * d;
      w[j + 8] = static_cast<float>(kFp4[qs[j] >> 4]) * d;
    }
  }
};
template<>
struct Q<kTQ5_0> {
  static constexpr int IE = 32, IPB = 1, BB = 22;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    const float d = f16_to_f32(ld16(b));
    const uint32_t qh = ld32(b + 2);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
      const int x0 = ((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
      const int x1 = ((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
      w[j] = static_cast<float>(x0) * d;
      w[j + 16] = static_cast<float>(x1) * d;
    }
  }
};
template<>
struct Q<kTIQ4_NL> {
  static constexpr int IE = 32, IPB = 1, BB = 18;
  __device__ static void deq(const uint8_t* b, int, float* w) {
    const float d = f16_to_f32(ld16(b));
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
      w[j] = d * static_cast<float>(kIq4nl[qs[j] & 0xF]);
      w[j + 16] = d * static_cast<float>(kIq4nl[qs[j] >> 4]);
    }
  }
};
template<>
struct Q<kTIQ4_XS> {
  static constexpr int IE = 32, IPB = 8, BB = 136;
  __device__ static void deq(const uint8_t* b, int ib, float* w) {
    const float d = f16_to_f32(ld16(b));
    const uint32_t sh = ld16(b + 2);
    const uint8_t* sl = b + 4;
    const uint8_t* qs = b + 8 + 16 * ib;
    const int ls =
        ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    for (int j = 0; j < 16; ++j) {
      w[j] = dl * static_cast<float>(kIq4nl[qs[j] & 0xF]);
      w[j + 16] = dl * static_cast<float>(kIq4nl[qs[j] >> 4]);
    }
  }
};
template<>
struct Q<kTQ4_K> {
  static constexpr int IE = 32, IPB = 8, BB = 144;
  __device__ static void deq(const uint8_t* b, int s, float* w) {
    const float d = f16_to_f32(ld16(b));
    const float dmin = f16_to_f32(ld16(b + 2));
    uint8_t sc, m;
    scale_min_k4(s, b + 4, &sc, &m);
    const uint8_t* qs = b + 16 + 32 * (s / 2);
    const float d1 = d * sc, m1 = dmin * m;
    for (int l = 0; l < 32; ++l)
      w[l] =
          d1 * static_cast<float>((s & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
  }
};
template<>
struct Q<kTQ5_K> {
  static constexpr int IE = 32, IPB = 8, BB = 176;
  __device__ static void deq(const uint8_t* b, int s, float* w) {
    const float d = f16_to_f32(ld16(b));
    const float dmin = f16_to_f32(ld16(b + 2));
    uint8_t sc, m;
    scale_min_k4(s, b + 4, &sc, &m);
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48 + 32 * (s / 2);
    const float d1 = d * sc, m1 = dmin * m;
    for (int l = 0; l < 32; ++l) {
      const int lo = (s & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
      const int hi = (qh[l] >> s) & 1 ? 16 : 0;
      w[l] = d1 * static_cast<float>(lo + hi) - m1;
    }
  }
};
template<>
struct Q<kTQ6_K> {
  static constexpr int IE = 32, IPB = 8, BB = 210;
  __device__ static void deq(const uint8_t* b, int s, float* w) {
    const int half = s / 4, qd = s % 4;
    const uint8_t* ql = b + 64 * half;
    const uint8_t* qh = b + 128 + 32 * half;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * half;
    const float d = f16_to_f32(ld16(b + 208));
    for (int l = 0; l < 32; ++l) {
      const int lo =
          qd < 2 ? (ql[(qd & 1) * 32 + l] & 0xF) : (ql[(qd & 1) * 32 + l] >> 4);
      const int hi = (qh[l] >> (2 * qd)) & 3;
      const int q = (lo | (hi << 4)) - 32;
      w[l] =
          d * static_cast<float>(sc[l / 16 + 2 * qd]) * static_cast<float>(q);
    }
  }
};

// ---------------------------------------------------------------------------------------------
// GEMM: one weight row per wave, lanes stride over items, TT tokens per block
// row-tile.
template<int T, int TT>
__global__ void gemm_k(const uint8_t* __restrict__ W, size_t row_bytes, int N,
                       int K, const float* __restrict__ x, int ldx,
                       float* __restrict__ out, int ldo, int t_count) {
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int row = blockIdx.x * kGemmWaves + wave;
  if (row >= N)
    return;
  const int t0 = blockIdx.y * TT;
  const uint8_t* wr = W + static_cast<size_t>(row) * row_bytes;
  constexpr int IE = Q<T>::IE, IPB = Q<T>::IPB, BB = Q<T>::BB;
  const int nitems = K / IE;
  float acc[TT];
  for (int t = 0; t < TT; ++t)
    acc[t] = 0.f;
  for (int it = lane; it < nitems; it += kWave) {
    float w[IE];
    Q<T>::deq(wr + static_cast<size_t>(it / IPB) * BB, it % IPB, w);
    for (int t = 0; t < TT; ++t) {
      if (t0 + t >= t_count)
        break;
      const float* xp =
          x + static_cast<size_t>(t0 + t) * ldx + static_cast<size_t>(it) * IE;
      float s = 0.f;
      for (int j = 0; j < IE; ++j)
        s += w[j] * xp[j];
      acc[t] += s;
    }
  }
  for (int t = 0; t < TT; ++t) {
    float v = acc[t];
    for (int o = kWave / 2; o > 0; o >>= 1)
      v += __shfl_xor(v, o, kWave);
    if (lane == 0 && t0 + t < t_count)
      out[static_cast<size_t>(t0 + t) * ldo + row] = v;
  }
}

template<int T, int TT>
__global__ void moe_gemm_k(const uint8_t* __restrict__ W, size_t row_bytes,
                           int N, int K, const float* __restrict__ scale,
                           const float* __restrict__ x, int ldx, int gdiv,
                           const int* __restrict__ sorted,
                           const int* __restrict__ off, float* __restrict__ out,
                           int ldo) {
  const int e = blockIdx.y;
  const int begin = off[e], cnt = off[e + 1] - begin;
  if (cnt == 0)
    return;
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int row = blockIdx.x * kGemmWaves + wave;
  if (row >= N)
    return;
  const uint8_t* wr = W + (static_cast<size_t>(e) * N + row) * row_bytes;
  const float sc = scale ? scale[e] : 1.0f;
  constexpr int IE = Q<T>::IE, IPB = Q<T>::IPB, BB = Q<T>::BB;
  const int nitems = K / IE;
  for (int base = 0; base < cnt; base += TT) {
    float acc[TT];
    const float* xr[TT];
    int pr[TT];
    for (int t = 0; t < TT; ++t) {
      acc[t] = 0.f;
      pr[t] = base + t < cnt ? sorted[begin + base + t] : -1;
      xr[t] = pr[t] >= 0
                  ? x + static_cast<size_t>(gdiv ? pr[t] / gdiv : pr[t]) * ldx
                  : nullptr;
    }
    for (int it = lane; it < nitems; it += kWave) {
      float w[IE];
      Q<T>::deq(wr + static_cast<size_t>(it / IPB) * BB, it % IPB, w);
      for (int t = 0; t < TT; ++t) {
        if (pr[t] < 0)
          break;
        const float* xp = xr[t] + static_cast<size_t>(it) * IE;
        float s = 0.f;
        for (int j = 0; j < IE; ++j)
          s += w[j] * xp[j];
        acc[t] += s;
      }
    }
    for (int t = 0; t < TT; ++t) {
      float v = acc[t];
      for (int o = kWave / 2; o > 0; o >>= 1)
        v += __shfl_xor(v, o, kWave);
      if (lane == 0 && pr[t] >= 0)
        out[static_cast<size_t>(pr[t]) * ldo + row] = v * sc;
    }
  }
}

template<int T>
__global__ void embed_k(const uint8_t* __restrict__ W, size_t row_bytes, int K,
                        const int* __restrict__ ids, float* __restrict__ out,
                        int ldo) {
  const int t = blockIdx.x, lane = threadIdx.x;
  constexpr int IE = Q<T>::IE, IPB = Q<T>::IPB, BB = Q<T>::BB;
  const uint8_t* wr = W + static_cast<size_t>(ids[t]) * row_bytes;
  for (int it = lane; it < K / IE; it += blockDim.x) {
    float w[IE];
    Q<T>::deq(wr + static_cast<size_t>(it / IPB) * BB, it % IPB, w);
    for (int j = 0; j < IE; ++j)
      out[static_cast<size_t>(t) * ldo + it * IE + j] = w[j];
  }
}

template<int T>
struct TypeTag {
  static constexpr int value = T;
};

template<class F>
void DispatchType(int type, F&& f) {
  switch (type) {
    case kTF32:
      f(TypeTag<kTF32>{});
      break;
    case kTF16:
      f(TypeTag<kTF16>{});
      break;
    case kTBF16:
      f(TypeTag<kTBF16>{});
      break;
    case kTQ8_0:
      f(TypeTag<kTQ8_0>{});
      break;
    case kTNVFP4:
      f(TypeTag<kTNVFP4>{});
      break;
    case kTQ5_0:
      f(TypeTag<kTQ5_0>{});
      break;
    case kTIQ4_NL:
      f(TypeTag<kTIQ4_NL>{});
      break;
    case kTIQ4_XS:
      f(TypeTag<kTIQ4_XS>{});
      break;
    case kTQ4_K:
      f(TypeTag<kTQ4_K>{});
      break;
    case kTQ5_K:
      f(TypeTag<kTQ5_K>{});
      break;
    case kTQ6_K:
      f(TypeTag<kTQ6_K>{});
      break;
    default:
      throw std::runtime_error("inkling2: unsupported weight type " +
                               std::to_string(type));
  }
}

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2 HIP launch ") + what + ": " +
                             hipGetErrorString(e));
}

// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float wave_sum(float v) {
  for (int o = kWave / 2; o > 0; o >>= 1)
    v += __shfl_xor(v, o, kWave);
  return v;
}

__global__ void rmsnorm_k(const float* __restrict__ x, int ldx,
                          const float* __restrict__ w, float* __restrict__ out,
                          int ldo, int dim, float eps,
                          _Float16* __restrict__ out16) {
  const int row = blockIdx.x;
  const float* xr = x + static_cast<size_t>(row) * ldx;
  __shared__ float red[32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    ss += xr[i] * xr[i];
  ss = wave_sum(ss);
  const int wid = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  if (lane == 0)
    red[wid] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < blockDim.x / kWave; ++i)
    tot += red[i];
  const float inv = rsqrtf(tot / static_cast<float>(dim) + eps);
  float* orow = out ? out + static_cast<size_t>(row) * ldo : nullptr;
  _Float16* hrow = out16 ? out16 + static_cast<size_t>(row) * dim : nullptr;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float o = xr[i] * inv * (w ? w[i] : 1.0f);
    if (out)
      orow[i] = o;
    if (hrow)
      hrow[i] = (_Float16)o;
  }
}

__global__ void mix_k(MixStates st, const float* __restrict__ query, int nq,
                      int dim, float* __restrict__ out, float eps,
                      float* __restrict__ inv_out, int t_count) {
  const int t = blockIdx.x;
  const int wid = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  constexpr int kWaves = 8;
  __shared__ float inv[8];
  __shared__ float score[4][8];
  // per-state inverse RMS
  for (int j = wid; j < st.n; j += kWaves) {
    const float* xr = st.ptr[j] + static_cast<size_t>(t) * dim;
    float ss = 0.f;
    // Unrolled so the loads are issued ahead; the additions keep their order.
#pragma unroll 8
    for (int i = lane; i < dim; i += kWave)
      ss += xr[i] * xr[i];
    ss = wave_sum(ss);
    if (lane == 0)
      inv[j] = rsqrtf(ss / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  for (int pr = wid; pr < nq * st.n; pr += kWaves) {
    const int q = pr / st.n, j = pr % st.n;
    const float* xr = st.ptr[j] + static_cast<size_t>(t) * dim;
    const float* qr = query + static_cast<size_t>(q) * dim;
    float d = 0.f;
#pragma unroll 8
    for (int i = lane; i < dim; i += kWave)
      d += qr[i] * xr[i];
    d = wave_sum(d);
    if (lane == 0)
      score[q][j] = d * inv[j] * rsqrtf(static_cast<float>(dim));
  }
  __syncthreads();
  if (threadIdx.x < nq) {
    const int q = threadIdx.x;
    float m = -INFINITY;
    for (int j = 0; j < st.n; ++j)
      m = fmaxf(m, score[q][j]);
    float z = 0.f;
    for (int j = 0; j < st.n; ++j) {
      score[q][j] = expf(score[q][j] - m);
      z += score[q][j];
    }
    for (int j = 0; j < st.n; ++j)
      score[q][j] /= z;
  }
  __syncthreads();
  // Each state value is read once for every query; every output sums the states
  // in order j.
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (j < st.n)
        x[j] = st.ptr[j][static_cast<size_t>(t) * dim + i];
    for (int q = 0; q < nq; ++q) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j < st.n)
          a += score[q][j] * inv[j] * x[j];
      out[(static_cast<size_t>(t) * nq + q) * dim + i] = a;
    }
  }
  if (inv_out) {
    // inv_out[q][t]: the inverse RMS of output row q, summed as the loop
    // controllers' gate prologue sums it (lane-strided, then a wave sum).
    __threadfence_block();
    __syncthreads();
    if (wid < nq) {
      const float* o = out + (static_cast<size_t>(t) * nq + wid) * dim;
      float b = 0.f;
      for (int i = lane; i < dim; i += kWave) {
        float v = o[i];
        b += v * v;
      }
      b = wave_sum(b);
      if (lane == 0)
        inv_out[static_cast<size_t>(wid) * t_count + t] = rsqrtf(b / dim + eps);
    }
  }
}

// mix_k for dim 2304 with the same arithmetic in fewer dependent load
// phases: a wave keeps its state row in registers for the sum of squares and
// every query's dot product (each sums lane-strided, in i order, then a wave
// sum, as mix_k does), and the output loop issues all of its state loads at
// once. Identical results.
__global__ void __launch_bounds__(256)
    mix2304_k(MixStates st, const float* __restrict__ query, int nq, int dim,
              float* __restrict__ out, float eps, float* __restrict__ inv_out,
              int t_count) {
  constexpr int D = 2304, M = D / kWave, R = D / 256;
  const int t = blockIdx.x;
  const int wid = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  __shared__ float inv[8];
  __shared__ float score[4][8];
  // dim is the runtime 2304: rsqrtf and the divisions see the same operands
  // as in mix_k.
  const float fdim = static_cast<float>(dim);
  if (wid < st.n) {
    const float* xr = st.ptr[wid] + static_cast<size_t>(t) * D;
    float x[M];
#pragma unroll
    for (int m = 0; m < M; ++m)
      x[m] = xr[lane + kWave * m];
    float ss = 0.f;
#pragma unroll
    for (int m = 0; m < M; ++m)
      ss = __fmaf_rn(x[m], x[m], ss);
    ss = wave_sum(ss);
    const float iv = rsqrtf(ss / fdim + eps);
    if (lane == 0)
      inv[wid] = iv;
    for (int q = 0; q < nq; ++q) {
      const float* qr = query + static_cast<size_t>(q) * D;
      float d = 0.f;
#pragma unroll
      for (int m = 0; m < M; ++m)
        d = __fmaf_rn(qr[lane + kWave * m], x[m], d);
      d = wave_sum(d);
      if (lane == 0)
        score[q][wid] = d * iv * rsqrtf(fdim);
    }
  }
  __syncthreads();
  if (threadIdx.x < nq) {
    const int q = threadIdx.x;
    float m = -INFINITY;
    for (int j = 0; j < st.n; ++j)
      m = fmaxf(m, score[q][j]);
    float z = 0.f;
    for (int j = 0; j < st.n; ++j) {
      score[q][j] = expf(score[q][j] - m);
      z += score[q][j];
    }
    for (int j = 0; j < st.n; ++j)
      score[q][j] /= z;
  }
  __syncthreads();
  float x[R][8];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (j < st.n)
        x[r][j] = st.ptr[j][static_cast<size_t>(t) * D + threadIdx.x + 256 * r];
  for (int q = 0; q < nq; ++q) {
    float c[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      c[j] = j < st.n ? score[q][j] * inv[j] : 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j < st.n)
          a = __fmaf_rn(c[j], x[r][j], a);
      out[(static_cast<size_t>(t) * nq + q) * D + threadIdx.x + 256 * r] = a;
    }
  }
  if (inv_out) {
    __threadfence_block();
    __syncthreads();
    if (wid < nq) {
      const float* o = out + (static_cast<size_t>(t) * nq + wid) * D;
      float v[M];
#pragma unroll
      for (int m = 0; m < M; ++m)
        v[m] = o[lane + kWave * m];
      float b = 0.f;
#pragma unroll
      for (int m = 0; m < M; ++m)
        b = __fmaf_rn(v[m], v[m], b);
      b = wave_sum(b);
      if (lane == 0)
        inv_out[static_cast<size_t>(wid) * t_count + t] = rsqrtf(b / dim + eps);
    }
  }
}

__global__ void logscale_k(float* tau, int pos0, int t_count, float floor_,
                           float alpha) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= t_count)
    return;
  const float r = fmaxf(1.0f, static_cast<float>(pos0 + t + 1) / floor_);
  tau[t] = 1.0f + alpha * logf(r);
}

__global__ void conv_k(const float* __restrict__ x,
                       const float* __restrict__ state,
                       const float* __restrict__ w,
                       const float* __restrict__ dyn, float* __restrict__ out,
                       int t_count, int C, int hd, float scale) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (c >= C)
    return;
  float xin[4];
  for (int i = 0; i < 4; ++i) {
    const int idx = t - 3 + i;
    xin[i] = idx >= 0 ? x[static_cast<size_t>(idx) * C + c]
                      : state[static_cast<size_t>(c) * 3 + idx + 3];
  }
  float y = 0.f;
  for (int i = 0; i < 4; ++i)
    y += w[static_cast<size_t>(c) * 4 + i] * xin[i];
  if (dyn) {
    const int nh = C / hd;
    const float* d = dyn + static_cast<size_t>(t) * nh * 4 + (c / hd) * 4;
    for (int lag = 0; lag < 4; ++lag)
      y += d[lag] * xin[3 - lag];
  }
  out[static_cast<size_t>(t) * C + c] = (xin[3] + y) * scale;
}

__global__ void conv_state_k(const float* __restrict__ x,
                             float* __restrict__ state, int t_count, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  float old[3] = {state[static_cast<size_t>(c) * 3],
                  state[static_cast<size_t>(c) * 3 + 1],
                  state[static_cast<size_t>(c) * 3 + 2]};
  float nw[3];
  for (int j = 0; j < 3; ++j) {
    const int idx = t_count + j;  // index in concat(state(3), x(T))
    nw[j] = idx < 3 ? old[idx] : x[static_cast<size_t>(idx - 3) * C + c];
  }
  for (int j = 0; j < 3; ++j)
    state[static_cast<size_t>(c) * 3 + j] = nw[j];
}

__global__ void conv_state_capture_k(const float* __restrict__ x,
                                     const float* __restrict__ old_state,
                                     float* __restrict__ dst_state, int t_count,
                                     int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C)
    return;
  for (int j = 0; j < 3; ++j) {
    const int idx = t_count + j;
    dst_state[static_cast<size_t>(c) * 3 + j] =
        idx < 3 ? old_state[static_cast<size_t>(c) * 3 + idx]
                : x[static_cast<size_t>(idx - 3) * C + c];
  }
}

__global__ void add_k(const float* a, const float* b, float* o, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    o[i] = a[i] + b[i];
}

__global__ void router_k(const float* __restrict__ logits,
                         const float* __restrict__ bias, int* __restrict__ ids,
                         float* __restrict__ wts, int t_count, int E, int K,
                         float rscale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= t_count)
    return;
  const float* lr = logits + static_cast<size_t>(t) * E;
  int sel[16];
  float sc[16];
  float sum = 0.f;
  for (int k = 0; k < K; ++k) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < E; ++e) {
      bool used = false;
      for (int j = 0; j < k; ++j)
        used |= sel[j] == e;
      if (used)
        continue;
      const float s = 1.0f / (1.0f + expf(-lr[e]));
      const float v = s + bias[e];
      if (v > bv) {
        bv = v;
        best = e;
      }
    }
    sel[k] = best;
    sc[k] = 1.0f / (1.0f + expf(-lr[best]));
    sum += sc[k];
  }
  for (int k = 0; k < K; ++k) {
    ids[static_cast<size_t>(t) * K + k] = sel[k];
    wts[static_cast<size_t>(t) * K + k] = sc[k] / (sum + 1e-20f) * rscale;
  }
}

__global__ void situ_k(const float* gu, float* out, int n_ff, int rows,
                       float beta, float lb) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(rows) * n_ff)
    return;
  const size_t p = i / n_ff;
  const int c = static_cast<int>(i % n_ff);
  const float g = gu[p * 2 * n_ff + c], u = gu[p * 2 * n_ff + n_ff + c];
  const float act = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g)));
  out[i] = act * (lb * tanhf(u / lb));
}

__global__ void moe_comb_k(const float* y, const float* wts, float* out,
                           int dim, int t_count, int K) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(t_count) * dim)
    return;
  const int t = static_cast<int>(i / dim), c = static_cast<int>(i % dim);
  float a = 0.f;
  for (int k = 0; k < K; ++k)
    a += wts[static_cast<size_t>(t) * K + k] *
         y[(static_cast<size_t>(t) * K + k) * dim + c];
  out[i] = a;
}

__global__ void softcap_k(float* x, std::int64_t n, float cap) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = cap * tanhf(x[i] / cap);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
std::size_t RowBytes(int type, std::size_t k) {
  std::size_t bytes = 0;
  switch (type) {
    case kTF32:
      return k % 4 ? 0 : k * 4;
    case kTF16:
    case kTBF16:
      return k % 8 ? 0 : k * 2;
    case kTQ8_0:
      return k % 32 ? 0 : k / 32 * 34;
    case kTNVFP4:
      return k % 64 ? 0 : k / 64 * 36;
    case kTQ5_0:
      return k % 32 ? 0 : k / 32 * 22;
    case kTIQ4_NL:
      return k % 32 ? 0 : k / 32 * 18;
    case kTIQ4_XS:
      return k % 256 ? 0 : k / 256 * 136;
    case kTQ4_K:
      return k % 256 ? 0 : k / 256 * 144;
    case kTQ5_K:
      return k % 256 ? 0 : k / 256 * 176;
    case kTQ6_K:
      return k % 256 ? 0 : k / 256 * 210;
    default:
      return bytes;
  }
}

void Gemm(hipStream_t s, int type, const void* w, int n, int k, const float* x,
          int ldx, float* out, int ldo, int t_count) {
  const size_t rb = RowBytes(type, k);
  if (rb == 0)
    throw std::runtime_error("inkling2: bad gemm shape");
  DispatchType(type, [&](auto tag) {
    constexpr int T = decltype(tag)::value;
    const dim3 block(kGemmWaves * kWave);
    if (t_count == 1) {
      const dim3 grid((n + kGemmWaves - 1) / kGemmWaves, 1);
      gemm_k<T, 1><<<grid, block, 0, s>>>(static_cast<const uint8_t*>(w), rb, n,
                                          k, x, ldx, out, ldo, t_count);
    } else {
      const dim3 grid((n + kGemmWaves - 1) / kGemmWaves, (t_count + 7) / 8);
      gemm_k<T, 8><<<grid, block, 0, s>>>(static_cast<const uint8_t*>(w), rb, n,
                                          k, x, ldx, out, ldo, t_count);
    }
  });
  Check(hipGetLastError(), "gemm");
}

void MoeGemm(hipStream_t s, int type, const void* w, int n, int k, int n_expert,
             const float* scale, const float* x, int ldx, int gather_div,
             const int* sorted_pairs, const int* offsets, float* out, int ldo,
             int n_pairs) {
  const size_t rb = RowBytes(type, k);
  if (rb == 0)
    throw std::runtime_error("inkling2: bad moe gemm shape");
  DispatchType(type, [&](auto tag) {
    constexpr int T = decltype(tag)::value;
    const dim3 block(kGemmWaves * kWave);
    const dim3 grid((n + kGemmWaves - 1) / kGemmWaves, n_expert);
    if (n_pairs <= 8) {
      moe_gemm_k<T, 1><<<grid, block, 0, s>>>(
          static_cast<const uint8_t*>(w), rb, n, k, scale, x, ldx, gather_div,
          sorted_pairs, offsets, out, ldo);
    } else {
      moe_gemm_k<T, 8><<<grid, block, 0, s>>>(
          static_cast<const uint8_t*>(w), rb, n, k, scale, x, ldx, gather_div,
          sorted_pairs, offsets, out, ldo);
    }
  });
  Check(hipGetLastError(), "moe_gemm");
}

void EmbedRows(hipStream_t s, int type, const void* w, int k, const int* ids,
               int t_count, float* out, int ldo) {
  const size_t rb = RowBytes(type, k);
  if (rb == 0)
    throw std::runtime_error("inkling2: bad embedding shape");
  DispatchType(type, [&](auto tag) {
    constexpr int T = decltype(tag)::value;
    embed_k<T><<<t_count, 128, 0, s>>>(static_cast<const uint8_t*>(w), rb, k,
                                       ids, out, ldo);
  });
  Check(hipGetLastError(), "embed");
}

void RmsNorm(hipStream_t s, const float* x, int ldx, const float* w, float* out,
             int ldo, int dim, int rows, float eps, void* out16) {
  if (rows <= 0)
    return;
  rmsnorm_k<<<rows, 256, 0, s>>>(x, ldx, w, out, ldo, dim, eps,
                                 static_cast<_Float16*>(out16));
  Check(hipGetLastError(), "rmsnorm");
}

void LogScale(hipStream_t s, float* tau, int pos0, int t_count, float floor_,
              float alpha) {
  logscale_k<<<(t_count + 127) / 128, 128, 0, s>>>(tau, pos0, t_count, floor_,
                                                   alpha);
  Check(hipGetLastError(), "logscale");
}

void Conv(hipStream_t s, const float* x, const float* state, const float* w,
          const float* dyn, float* out, int t_count, int channels, int hd,
          float scale) {
  const dim3 grid((channels + 127) / 128, t_count);
  conv_k<<<grid, 128, 0, s>>>(x, state, w, dyn, out, t_count, channels, hd,
                              scale);
  Check(hipGetLastError(), "conv");
}

void Mix(hipStream_t s, const MixStates& states, const float* query, int nq,
         int dim, int t_count, float* out, float eps, float* inv_out) {
  if (nq > 4 || states.n > 8)
    throw std::runtime_error("inkling2: mix too large");
  if (dim == 2304)
    mix2304_k<<<t_count, 256, 0, s>>>(states, query, nq, dim, out, eps, inv_out,
                                      t_count);
  else
    mix_k<<<t_count, 256, 0, s>>>(states, query, nq, dim, out, eps, inv_out,
                                  t_count);
  Check(hipGetLastError(), "mix");
}
void ConvStateUpdate(hipStream_t s, const float* x, float* state, int t_count,
                     int channels) {
  conv_state_k<<<(channels + 127) / 128, 128, 0, s>>>(x, state, t_count,
                                                      channels);
  Check(hipGetLastError(), "conv_state");
}
void ConvStateCapture(hipStream_t s, const float* x, const float* old_state,
                      float* dst_state, int t_count, int channels) {
  conv_state_capture_k<<<(channels + 127) / 128, 128, 0, s>>>(
      x, old_state, dst_state, t_count, channels);
  Check(hipGetLastError(), "conv_state_capture");
}
void Add(hipStream_t s, const float* a, const float* b, float* out, int n) {
  add_k<<<(n + 255) / 256, 256, 0, s>>>(a, b, out, n);
  Check(hipGetLastError(), "add");
}
void Copy(hipStream_t s, const float* a, float* out, int n) {
  Check(hipMemcpyAsync(out, a, static_cast<size_t>(n) * 4,
                       hipMemcpyDeviceToDevice, s),
        "copy");
}

void RouterTopK(hipStream_t s, const float* logits, const float* bias, int* ids,
                float* wts, int t_count, int n_expert, int n_used,
                float route_scale) {
  if (n_used > 16)
    throw std::runtime_error("inkling2: too many experts used");
  router_k<<<(t_count + 63) / 64, 64, 0, s>>>(logits, bias, ids, wts, t_count,
                                              n_expert, n_used, route_scale);
  Check(hipGetLastError(), "router");
}
void Situ(hipStream_t s, const float* gu, float* out, int n_ff, int rows,
          float beta, float lb) {
  const size_t n = static_cast<size_t>(rows) * n_ff;
  situ_k<<<(n + 255) / 256, 256, 0, s>>>(gu, out, n_ff, rows, beta, lb);
  Check(hipGetLastError(), "situ");
}
void MoeCombine(hipStream_t s, const float* y, const float* wts, float* out,
                int dim, int t_count, int n_used) {
  const size_t n = static_cast<size_t>(t_count) * dim;
  moe_comb_k<<<(n + 255) / 256, 256, 0, s>>>(y, wts, out, dim, t_count, n_used);
  Check(hipGetLastError(), "moe_combine");
}
void Softcap(hipStream_t s, float* x, std::int64_t n, float cap) {
  softcap_k<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap);
  Check(hipGetLastError(), "softcap");
}

}  // namespace gufo::models::inkling2::hip
