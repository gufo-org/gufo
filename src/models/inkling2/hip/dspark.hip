// Kernels of the Inkling 2 Mini DSpark drafter. The drafter only proposes
// tokens: the target verifies every one of them exactly, so these kernels are
// written for speed and plain F32 arithmetic, not for a fixed rounding order.
#include <hip/hip_fp16.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "src/models/inkling2/hip/dspark.hpp"

namespace gufo::models::inkling2::dsp {
namespace {

void Chk(const char* what) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2 dspark ") + what + ": " +
                             hipGetErrorString(e));
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor(v, o, 32);
  return v;
}
// Sum over a 256-thread block (red: 8 floats of LDS).
__device__ __forceinline__ float bsum256(float v, float* red) {
  v = wsum(v);
  const int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0)
    red[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    t += red[i];
  return t;
}
__device__ __forceinline__ float bf16f(uint16_t b) {
  return __uint_as_float(static_cast<uint32_t>(b) << 16);
}
__device__ __forceinline__ float sigm(float x) {
  return 1.f / (1.f + __expf(-x));
}

// ---- feature mixes: block per (row, j) -------------------------------------
__global__ void __launch_bounds__(256) k_mix(MixArgs a) {
  __shared__ float red[8];
  const int r = blockIdx.x, j = blockIdx.y;
  const float* f = a.feat + (size_t)r * 4 * kEmbd;
  float v[9], ss = 0.f;
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int c = threadIdx.x + 256 * t;
    float x = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      x += a.alpha[j][i] * f[(size_t)i * kEmbd + c];
    v[t] = x;
    ss += x * x;
  }
  const float inv = rsqrtf(bsum256(ss, red) / kEmbd + a.eps);
  float* o = a.out + ((size_t)r * 4 + j) * kEmbd;
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int c = threadIdx.x + 256 * t;
    const float y = v[t] * inv * a.w[j][c];
    o[c] = y;
    if (j < 3)
      a.out16[((size_t)j * (a.ld16 ? a.ld16 : a.n) + r) * kEmbd + c] =
          (_Float16)y;
  }
}

// NeoX rotation of one 128-dim head held as lane values d and d + 64
// (lanes 0..63 of a 64-thread group), position p.
__device__ __forceinline__ void rope_pair(float& x0, float& x1, int d, int p,
                                          float theta) {
  const float inv = powf(theta, -(2.f * d) / kHead);
  float sn, cs;
  sincosf(p * inv, &sn, &cs);
  const float a = x0, b = x1;
  x0 = a * cs - b * sn;
  x1 = b * cs + a * sn;
}

// ---- context K/V store: block per (row, kv head), 64 threads ---------------
__global__ void __launch_bounds__(64) k_kv_store(KvStoreArgs a) {
  __shared__ float red[2];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const float* kr = a.k + (size_t)r * a.ldk + h * kHead;
  float x0 = kr[d], x1 = kr[d + 64];
  float ss = wsum(x0 * x0 + x1 * x1);
  if ((d & 31) == 0)
    red[d / 32] = ss;
  __syncthreads();
  const float inv = rsqrtf((red[0] + red[1]) / kHead + a.eps);
  x0 = x0 * inv * a.knorm[d];
  x1 = x1 * inv * a.knorm[d + 64];
  const int p = a.pos0 + r;
  rope_pair(x0, x1, d, p, a.theta);
  const size_t row = (size_t)(p % a.cap) * kKV + h * kHead;
  a.kc[row + d] = (_Float16)x0;
  a.kc[row + d + 64] = (_Float16)x1;
  const float* vr = a.v + (size_t)r * a.ldv + h * kHead;
  a.vc[row + d] = (_Float16)vr[d];
  a.vc[row + d + 64] = (_Float16)vr[d + 64];
}

// ---- block embedding + first norm: block per row ---------------------------
__global__ void __launch_bounds__(256)
    k_embed(const uint8_t* __restrict__ emb, const float* __restrict__ tnorm,
            const int* __restrict__ anchor, const float* __restrict__ mask,
            const float* __restrict__ w, float* x, float* h, _Float16* h16,
            float eps) {
  __shared__ float red[8];
  const int r = blockIdx.x;
  float v[9], ss = 0.f;
  if (r == 0) {
    const uint8_t* row = emb + (size_t)(*anchor) * (kEmbd / 32) * 34;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
      const int c = threadIdx.x + 256 * t, b = c / 32, i = c % 32;
      const uint8_t* blk = row + (size_t)b * 34;
      const float d = __half2float(*(const __half*)blk);
      v[t] = d * (float)(int8_t)blk[2 + i];
      ss += v[t] * v[t];
    }
    const float inv = rsqrtf(bsum256(ss, red) / kEmbd + eps);
    ss = 0.f;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
      const int c = threadIdx.x + 256 * t;
      v[t] = v[t] * inv * tnorm[c];
      ss += v[t] * v[t];
    }
  } else {
#pragma unroll
    for (int t = 0; t < 9; ++t) {
      v[t] = mask[threadIdx.x + 256 * t];
      ss += v[t] * v[t];
    }
  }
  const float inv = rsqrtf(bsum256(ss, red) / kEmbd + eps);
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int c = threadIdx.x + 256 * t;
    const size_t o = (size_t)r * kEmbd + c;
    x[o] = v[t];
    const float y = v[t] * inv * w[c];
    h[o] = y;
    h16[o] = (_Float16)y;
  }
}

// ---- per-head q/k norm + rope: block per (row, head of q or k) -------------
__global__ void __launch_bounds__(64)
    k_qk(float* q, float* k, const float* qn, const float* kn, const int* pos,
         float theta, float eps) {
  __shared__ float red[2];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const bool isq = h < kHeads;
  float* x = isq ? q + (size_t)r * kQ + h * kHead
                 : k + (size_t)r * kKV + (h - kHeads) * kHead;
  const float* w = isq ? qn : kn;
  float x0 = x[d], x1 = x[d + 64];
  const float ss = wsum(x0 * x0 + x1 * x1);
  if ((d & 31) == 0)
    red[d / 32] = ss;
  __syncthreads();
  const float inv = rsqrtf((red[0] + red[1]) / kHead + eps);
  x0 = x0 * inv * w[d];
  x1 = x1 * inv * w[d + 64];
  rope_pair(x0, x1, d, *pos + r, theta);
  x[d] = x0;
  x[d + 64] = x1;
}

// ---- block attention
// --------------------------------------------------------- grid (query head,
// split): a 256-thread block holds the 8 block rows of one query head; wave w
// (8 waves) takes keys w, w + 8, ... of its split, lane l holds dims 4l..4l+3.
// Online softmax per row in registers, merged across the waves in LDS, then
// across the splits.
__global__ void __launch_bounds__(256) k_attn(AttnArgs a) {
  constexpr int NQ = kBlock;
  const int h = blockIdx.x, kvh = h / 3, split = blockIdx.y,
            w = threadIdx.x / 32, l = threadIdx.x % 32;
  const int pos = *a.pos;
  const int lo = max(a.first, a.window ? max(0, pos - a.window) : 0);
  const int nctx = pos - lo, nkeys = nctx + kBlock;
  const int nsplits = a.nsplits;
  const int per = (nkeys + nsplits - 1) / nsplits;
  const int k0 = split * per, k1 = min(nkeys, k0 + per);
  float4 qv[NQ];
#pragma unroll
  for (int r = 0; r < NQ; ++r)
    qv[r] = ((const float4*)(a.q + (size_t)r * kQ + h * kHead))[l];
  constexpr float scale = 0.08838834764831845f;
  float m[NQ], s[NQ];
  float4 o[NQ];
#pragma unroll
  for (int r = 0; r < NQ; ++r) {
    m[r] = -INFINITY;
    s[r] = 0.f;
    o[r] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  for (int j = k0 + w; j < k1; j += 8) {
    float4 kk, vv;
    if (j < nctx) {
      const size_t row = (size_t)((lo + j) % a.cap) * kKV + kvh * kHead + 4 * l;
      const __half2* kp = (const __half2*)(a.kc + row);
      const __half2* vp = (const __half2*)(a.vc + row);
      const float2 k01 = __half22float2(kp[0]), k23 = __half22float2(kp[1]);
      const float2 v01 = __half22float2(vp[0]), v23 = __half22float2(vp[1]);
      kk = make_float4(k01.x, k01.y, k23.x, k23.y);
      vv = make_float4(v01.x, v01.y, v23.x, v23.y);
    } else {
      const int b = j - nctx;
      kk = ((const float4*)(a.k + (size_t)b * kKV + kvh * kHead))[l];
      vv = ((const float4*)(a.v + (size_t)b * kKV + kvh * kHead))[l];
    }
#pragma unroll
    for (int r = 0; r < NQ; ++r) {
      const float sc = wsum(qv[r].x * kk.x + qv[r].y * kk.y + qv[r].z * kk.z +
                            qv[r].w * kk.w) *
                       scale;
      const float mn = fmaxf(m[r], sc), al = __expf(m[r] - mn),
                  p = __expf(sc - mn);
      s[r] = s[r] * al + p;
      o[r].x = o[r].x * al + p * vv.x;
      o[r].y = o[r].y * al + p * vv.y;
      o[r].z = o[r].z * al + p * vv.z;
      o[r].w = o[r].w * al + p * vv.w;
      m[r] = mn;
    }
  }
  __shared__ float sm[8][NQ][2];
  __shared__ float so[8][4][kHead];
#pragma unroll
  for (int r = 0; r < NQ; ++r) {
    if (l == 0) {
      sm[w][r][0] = m[r];
      sm[w][r][1] = s[r];
    }
  }
  float* ws = a.ws + ((size_t)h * nsplits + split) * NQ * (kHead + 2);
  for (int half = 0; half < 2; ++half) {
    if (half > 0)
      __syncthreads();
#pragma unroll
    for (int ri = 0; ri < 4; ++ri) {
      const int r = half * 4 + ri;
      so[w][ri][4 * l] = o[r].x;
      so[w][ri][4 * l + 1] = o[r].y;
      so[w][ri][4 * l + 2] = o[r].z;
      so[w][ri][4 * l + 3] = o[r].w;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 4 * kHead; e += 256) {
      const int ri = e / kHead, r = half * 4 + ri, d = e % kHead;
      float mm = -INFINITY;
      for (int x = 0; x < 8; ++x)
        mm = fmaxf(mm, sm[x][r][0]);
      float ss = 0.f, oo = 0.f;
      if (mm > -INFINITY)
        for (int x = 0; x < 8; ++x) {
          const float f = __expf(sm[x][r][0] - mm);
          ss += sm[x][r][1] * f;
          oo += so[x][ri][d] * f;
        }
      ws[(size_t)r * (kHead + 2) + 2 + d] = oo;
      if (d == 0) {
        ws[(size_t)r * (kHead + 2)] = mm;
        ws[(size_t)r * (kHead + 2) + 1] = ss;
      }
    }
  }
}
__global__ void __launch_bounds__(128) k_attn_combine(AttnArgs a) {
  constexpr int NQ = kBlock;
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int nsplits = a.nsplits;
  const float* ws = a.ws + (size_t)h * nsplits * NQ * (kHead + 2);
  float mm = -INFINITY;
  for (int x = 0; x < nsplits; ++x)
    mm = fmaxf(mm, ws[((size_t)x * NQ + r) * (kHead + 2)]);
  float ss = 0.f, oo = 0.f;
  for (int x = 0; x < nsplits; ++x) {
    const float* e = ws + ((size_t)x * NQ + r) * (kHead + 2);
    if (e[0] == -INFINITY)
      continue;
    const float f = __expf(e[0] - mm);
    ss += e[1] * f;
    oo += e[2 + d] * f;
  }
  a.out16[(size_t)r * kQ + h * kHead + d] = (_Float16)(oo / ss);
}

// ---- residual + norm
// ----------------------------------------------------------
__global__ void __launch_bounds__(256)
    k_add_norm(float* x, const float* y, const float* w, float* h,
               _Float16* h16, float eps) {
  __shared__ float red[8];
  const int r = blockIdx.x;
  float v[9], ss = 0.f;
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const size_t o = (size_t)r * kEmbd + threadIdx.x + 256 * t;
    v[t] = x[o] + y[o];
    x[o] = v[t];
    ss += v[t] * v[t];
  }
  const float inv = rsqrtf(bsum256(ss, red) / kEmbd + eps);
#pragma unroll
  for (int t = 0; t < 9; ++t) {
    const int c = threadIdx.x + 256 * t;
    const size_t o = (size_t)r * kEmbd + c;
    const float hv = v[t] * inv * w[c];
    h[o] = hv;
    if (h16)
      h16[o] = (_Float16)hv;
  }
}

__global__ void __launch_bounds__(256)
    k_swiglu(const float* g, const float* u, _Float16* o, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n)
    return;
  const float gv = g[i];
  o[i] = (_Float16)(gv * sigm(gv) * u[i]);
}

// ---- Markov RNN
// ---------------------------------------------------------------- joint (768
// rows of BF16 [5120]) = W_j [state, w1[prev], h_i, f] + b: the 5120 inputs
// gathered into LDS once per block, then one wave per row reading two BF16 per
// lane and step.
__global__ void __launch_bounds__(256) k_joint(MarkovArgs a) {
  __shared__ float in[kJoint];
  {
    const uint16_t* w1 = a.w1 + (size_t)(*a.prev) * kRank;
    const float* hi = a.h + (size_t)a.row * kEmbd;
    for (int c = threadIdx.x; c < kJoint; c += 256)
      in[c] = c < kRank               ? a.state[c]
              : c < 2 * kRank         ? bf16f(w1[c - kRank])
              : c < 2 * kRank + kEmbd ? hi[c - 2 * kRank]
                                      : a.f[c - 2 * kRank - kEmbd];
  }
  __syncthreads();
  const int row = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (row >= 3 * kRank)
    return;
  const uint32_t* wr =
      reinterpret_cast<const uint32_t*>(a.wj + (size_t)row * kJoint);
  float acc = 0.f;
  for (int c = l; c < kJoint / 2; c += 32) {
    const uint32_t w = wr[c];
    acc += __uint_as_float(w << 16) * in[2 * c] +
           __uint_as_float(w & 0xFFFF0000u) * in[2 * c + 1];
  }
  acc = wsum(acc);
  if (l == 0)
    a.joint[row] = acc + a.bj[row];
}
// state update (one block of 256): gate | cand | latent chunks
__global__ void __launch_bounds__(256) k_state(MarkovArgs a, float* lat) {
  const int i = threadIdx.x;
  const float g = sigm(a.joint[i]), c = tanhf(a.joint[kRank + i]);
  a.state[i] = c + g * (a.state[i] - c);
  lat[i] = tanhf(a.joint[2 * kRank + i]);
}
// ---- 8-row Q8_0 GEMV
// Block: 8 waves x 4 rows; the inputs [8][2304] of one K chunk in LDS as F16.
// A lane takes Q8 blocks lane, lane + 32, lane + 64 of the chunk, holds the 4
// rows' 32 codes as F16 pairs and accumulates every input with fdot2.
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
struct Gemv8Set {  // up to three matrices of the same input, one launch
  const uint8_t* w[3];
  float* y[3];
  int m[3];
  int blocks[3];  // row blocks of 32 of each
  int n;
};
// Row-wise squared L2 norms of BF16 [rows][256], 8 rows per 256-thread block.
__global__ void __launch_bounds__(256)
    k_row_norm_sq_bf16(const uint16_t* __restrict__ src, int rows,
                       float* __restrict__ out) {
  const int row = blockIdx.x * 8 + threadIdx.x / 32;
  const int l = threadIdx.x % 32;
  if (row >= rows)
    return;
  const uint16_t* p = src + (size_t)row * kRank;
  float s = 0.f;
#pragma unroll
  for (int c = l; c < kRank; c += 32) {
    const float v = bf16f(p[c]);
    s += v * v;
  }
  s = wsum(s);
  if (l == 0)
    out[row] = s;
}
// Quantize + transpose row-major BF16 [n][256] into int8 [256][n] + [n] scales.
__global__ void __launch_bounds__(256)
    k_quant_pt_bf16(const uint16_t* __restrict__ src, int n,
                    int8_t* __restrict__ q, float* __restrict__ scale) {
  const int r = blockIdx.x * 256 + threadIdx.x;
  if (r >= n)
    return;
  const uint16_t* row = src + (size_t)r * kRank;
  float mx = 0.f;
  for (int j = 0; j < kRank; ++j)
    mx = fmaxf(mx, fabsf(bf16f(row[j])));
  const float sc = mx > 0.f ? mx / 127.f : 1.f;
  const float inv = 1.f / sc;
  for (int j = 0; j < kRank; ++j)
    q[(size_t)j * n + r] = (int8_t)__float2int_rn(bf16f(row[j]) * inv);
  scale[r] = sc;
}
// int8 table with one scale per row r (over the 256 ranks)
__global__ void __launch_bounds__(256)
    k_quant_pt(const float* __restrict__ src, int n, int8_t* __restrict__ q,
               float* __restrict__ scale) {
  const int r = blockIdx.x * 256 + threadIdx.x;
  if (r >= n)
    return;
  float mx = 0.f;
  for (int j = 0; j < kRank; ++j)
    mx = fmaxf(mx, fabsf(src[(size_t)j * n + r]));
  const float sc = mx > 0.f ? mx / 127.f : 1.f;
  for (int j = 0; j < kRank; ++j)
    q[(size_t)j * n + r] = (int8_t)__float2int_rn(src[(size_t)j * n + r] / sc);
  scale[r] = sc;
}
// logits[r] = base[r] + sum_j pt[j][r] lat[j] + cb[r] from the int8 table:
// four rows per thread, shared across NS sessions so `pt` (16 MB) is read once per Markov step.
template<int NS>
__global__ void __launch_bounds__(256)
    k_lr_logits8_batch(const int8_t* __restrict__ pt,
                       const float* __restrict__ ps,
                       const float* __restrict__ cb, LrBatch batch, int n) {
  __shared__ float sl[NS][kRank];
#pragma unroll
  for (int s = 0; s < NS; ++s)
    if (s < batch.count)
      sl[s][threadIdx.x] = batch.lat[s][threadIdx.x];
  __syncthreads();
  const int r = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (r >= n)
    return;
  if constexpr (NS == 1) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 16
    for (int j = 0; j < kRank; ++j) {
      const char4 p = *(const char4*)(pt + (size_t)j * n + r);
      const float l = sl[0][j];
      a[0] += (float)p.x * l;
      a[1] += (float)p.y * l;
      a[2] += (float)p.z * l;
      a[3] += (float)p.w * l;
    }
    const float* b = batch.base[0];
    float* o = batch.out[0];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      o[r + i] = b[r + i] + ps[r + i] * a[i] + cb[r + i];
    return;
  }
  float a[NS][4];
#pragma unroll
  for (int s = 0; s < NS; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      a[s][i] = 0.f;
#pragma unroll 8
  for (int j = 0; j < kRank; ++j) {
    const char4 p = *(const char4*)(pt + (size_t)j * n + r);
    const float p0 = (float)p.x, p1 = (float)p.y, p2 = (float)p.z,
                p3 = (float)p.w;
#pragma unroll
    for (int s = 0; s < NS; ++s)
      if (s < batch.count) {
        const float l = sl[s][j];
        a[s][0] += p0 * l;
        a[s][1] += p1 * l;
        a[s][2] += p2 * l;
        a[s][3] += p3 * l;
      }
  }
  const float4 psv = *(const float4*)(ps + r);
  const float4 cbv = *(const float4*)(cb + r);
#pragma unroll
  for (int s = 0; s < NS; ++s)
    if (s < batch.count) {
      const float* b = batch.base[s] + r;
      float* o = batch.out[s] + r;
      o[0] = b[0] + psv.x * a[s][0] + cbv.x;
      o[1] = b[1] + psv.y * a[s][1] + cbv.y;
      o[2] = b[2] + psv.z * a[s][2] + cbv.z;
      o[3] = b[3] + psv.w * a[s][3] + cbv.w;
    }
}
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i_a2 __attribute__((ext_vector_type(4), aligned(2)));

// Per-32-element Q8_0 quantization of F16 activations [nrows][K] into
// WMMA-coalesced tiles [ntiles][K/32][2][16] (v4i) and [ntiles][K/32][16] (f32).
__global__ void __launch_bounds__(256)
    k_quant_q8(const _Float16* __restrict__ x, int nrows, int K,
               v4i* __restrict__ qa, float* __restrict__ qs) {
  __shared__ __attribute__((aligned(16))) int8_t sb[8][32];
  const int nb = K / 32;
  const int ntiles = (nrows + 15) / 16;
  const int total_warps = ntiles * 16 * nb;
  const int wid = blockIdx.x * 8 + threadIdx.x / 32;
  const int w = threadIdx.x / 32;
  const int l = threadIdx.x % 32;
  if (wid >= total_warps)
    return;
  const int b = wid % nb;
  const int row = wid / nb;
  const int tile = row / 16;
  const int r = row % 16;
  const size_t qbase = ((size_t)tile * nb + b) * 32 + r;
  const size_t sidx = ((size_t)tile * nb + b) * 16 + r;
  if (row >= nrows) {
    if (l == 0) {
      const v4i z = {0, 0, 0, 0};
      qa[qbase] = z;
      qa[qbase + 16] = z;
      qs[sidx] = 0.f;
    }
    return;
  }
  const float v = (float)x[(size_t)row * K + b * 32 + l];
  float mx = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    mx = fmaxf(mx, __shfl_xor(mx, o, 32));
  const float sc = mx > 0.f ? mx / 127.f : 0.f;
  const float inv = mx > 0.f ? 127.f / mx : 0.f;
  sb[w][l] = (int8_t)__float2int_rn(v * inv);
  __builtin_amdgcn_wave_barrier();
  if (l == 0) {
    qa[qbase] = *(const v4i*)&sb[w][0];
    qa[qbase + 16] = *(const v4i*)&sb[w][16];
    qs[sidx] = sc;
  }
}

// Int8 WMMA Q8_0 GEMM: 16 output rows per block, NW waves split K/32 in
// lockstep, reading each weight block once across all NTILES x 16 token rows.
template<int NW, int NTILES, int MAX_I>
__global__ void __launch_bounds__(NW * 32)
    k_gemv_wmma(Gemv8Set set, int K, const v4i* __restrict__ qa,
                const float* __restrict__ qs, int nrows) {
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int c16 = lane & 15;
  const int half = lane >> 4;
  int by = blockIdx.x, mi = 0;
  while (mi + 1 < set.n && by >= set.blocks[mi])
    by -= set.blocks[mi++];
  const uint8_t* __restrict__ W = set.w[mi];
  const int M = set.m[mi];
  float* __restrict__ y = set.y[mi];
  const int r0 = by * 16;
  const int nb = K / 32;
  const size_t rb = (size_t)nb * 34;
  const uint8_t* Wrow = W + (size_t)(r0 + c16) * rb;

  float acc[NTILES][MAX_I];
#pragma unroll
  for (int t = 0; t < NTILES; ++t)
#pragma unroll
    for (int i = 0; i < MAX_I; ++i)
      acc[t][i] = 0.f;

  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int b = wave; b < nb; b += NW) {
    const uint8_t* pu = Wrow + (size_t)b * 34;
    const float dw = __half2float(*(const __half*)pu);
    const v4i w0 = *(const v4i_a2*)(pu + 2);
    const v4i w1 = *(const v4i_a2*)(pu + 18);
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      const size_t qbase = ((size_t)t * nb + b) * 32 + c16;
      const v4i a0 = qa[qbase];
      const v4i a1 = qa[qbase + 16];
      const float sa = qs[((size_t)t * nb + b) * 16 + c16];
      v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, w0, z,
                                                         false);
      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, w1, c,
                                                     false);
#pragma unroll
      for (int i = 0; i < MAX_I; ++i) {
        const float s_row = __shfl(sa, 2 * i + half, 16);
        acc[t][i] += (dw * s_row) * (float)c[i];
      }
    }
  }

  constexpr int RPT = MAX_I * 2;
  __shared__ float red[NW][RPT][16];
  for (int t = 0; t < NTILES; ++t) {
    if (t > 0)
      __syncthreads();
#pragma unroll
    for (int i = 0; i < MAX_I; ++i)
      red[wave][2 * i + half][c16] = acc[t][i];
    __syncthreads();
    if (wave == 0) {
#pragma unroll
      for (int i = 0; i < MAX_I; ++i) {
        const int tr = 2 * i + half;
        const int gr = t * 16 + tr;
        if (gr < nrows && r0 + c16 < M) {
          float sum = 0.f;
#pragma unroll
          for (int w = 0; w < NW; ++w)
            sum += red[w][tr][c16];
          y[(size_t)gr * M + r0 + c16] = sum;
        }
      }
    }
  }
}

// ---- coupled sampling over contiguous chunks
// --------------------------------------
constexpr int kCsBlocks = 128;
__device__ __forceinline__ float softcapf(float x, float c) {
  return c > 0.f ? c * tanhf(x / c) : x;
}
// pass 1: block maxima of the capped logits (contiguous chunks)
__global__ void __launch_bounds__(256)
    k_cs_max(const float* x, int n, float cap, float* ws) {
  const int per = (n + kCsBlocks - 1) / kCsBlocks;
  const int lo = blockIdx.x * per, hi = min(n, lo + per);
  float v = -INFINITY;
  for (int i = lo + threadIdx.x; i < hi; i += 256)
    v = fmaxf(v, softcapf(x[i], cap));
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
    ws[blockIdx.x] = v;
  }
}
// pass 2: block sums of the weights
__device__ __forceinline__ float cs_weight(float l, float cap, float m, float t,
                                           float floor_) {
  const float c = softcapf(l, cap);
  return c >= floor_ ? __expf((c - m) / t) : 0.f;
}
__global__ void __launch_bounds__(256)
    k_cs_sum(const float* x, int n, float cap, float t, float min_p,
             float* ws) {
  __shared__ float red[8];
  __shared__ float mx;
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int b = 0; b < kCsBlocks; ++b)
      m = fmaxf(m, ws[b]);
    mx = m;
  }
  __syncthreads();
  const float m = mx;
  const float floor_ = min_p > 0.f ? m + t * __logf(min_p) : -INFINITY;
  const int per = (n + kCsBlocks - 1) / kCsBlocks;
  const int lo = blockIdx.x * per, hi = min(n, lo + per);
  float z = 0.f;
  for (int i = lo + threadIdx.x; i < hi; i += 256)
    z += cs_weight(x[i], cap, m, t, floor_);
  z = wsum(z);
  if (threadIdx.x % 32 == 0)
    red[threadIdx.x / 32] = z;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < 8; ++w)
      s += red[w];
    ws[kCsBlocks + blockIdx.x] = s;
    if (blockIdx.x == 0)
      ws[3 * kCsBlocks] = m;
  }
}
// pass 3 (one block): the chunk where the running sum passes u * Z, then the
// token inside it
__global__ void __launch_bounds__(256)
    k_cs_pick(const float* x, int n, const int* map, float cap, float t,
              float min_p, float u, const float* ws, int* tokens, int row,
              int* prev, float* conf) {
  __shared__ int chunk;
  __shared__ float before, total;
  __shared__ float part[256];
  const float m = ws[3 * kCsBlocks];
  const float floor_ = min_p > 0.f ? m + t * __logf(min_p) : -INFINITY;
  if (threadIdx.x == 0) {
    float z = 0.f;
    for (int b = 0; b < kCsBlocks; ++b)
      z += ws[kCsBlocks + b];
    const float target = u * z;
    float run = 0.f;
    int c = kCsBlocks - 1;
    for (int b = 0; b < kCsBlocks; ++b) {
      const float nxt = run + ws[kCsBlocks + b];
      if (target < nxt) {
        c = b;
        break;
      }
      run = nxt;
    }
    chunk = c;
    before = run;
    total = z;
  }
  __syncthreads();
  const int per = (n + kCsBlocks - 1) / kCsBlocks;
  const int lo = chunk * per, hi = min(n, lo + per);
  // each thread sums its slice of the chunk, then thread 0 walks the slices
  const int len = hi - lo, sl = (len + 255) / 256;
  const int a0 = lo + threadIdx.x * sl, a1 = min(hi, a0 + sl);
  float s = 0.f;
  for (int i = a0; i < a1; ++i)
    s += cs_weight(x[i], cap, m, t, floor_);
  part[threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    const float target = u * total;
    float run = before;
    int tok = hi - 1;
    for (int th = 0; th < 256; ++th) {
      if (target < run + part[th]) {
        const int b0 = lo + th * sl, b1 = min(hi, b0 + sl);
        for (int i = b0; i < b1; ++i) {
          const float w = cs_weight(x[i], cap, m, t, floor_);
          run += w;
          if (target < run) {
            tok = i;
            break;
          }
        }
        break;
      }
      run += part[th];
    }
    const int id = map ? map[tok] : tok;
    tokens[row] = id;
    *prev = id;
    conf[row] = cs_weight(x[tok], cap, m, t, floor_) / total;
  }
}

// ---- argmax (first maximum) and its softmax probability
// ----------------------
__global__ void __launch_bounds__(256)
    k_argmax1(const float* x, int n, float* ws) {
  __shared__ float bv[8], bz[8];
  __shared__ int bi[8];
  float v = -INFINITY;
  int ix = 0x7fffffff;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
    const float y = x[i];
    if (y > v) {
      v = y;
      ix = i;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o, 32);
    const int oi = __shfl_xor(ix, o, 32);
    if (ov > v || (ov == v && oi < ix)) {
      v = ov;
      ix = oi;
    }
  }
  const int w = threadIdx.x / 32;
  if (threadIdx.x % 32 == 0) {
    bv[w] = v;
    bi[w] = ix;
  }
  __syncthreads();
  float bm = bv[0];
  int bx = bi[0];
  for (int i = 1; i < 8; ++i)
    if (bv[i] > bm || (bv[i] == bm && bi[i] < bx)) {
      bm = bv[i];
      bx = bi[i];
    }
  // sum of exp(x - bm) over this block's elements
  float z = 0.f;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
    z += __expf(x[i] - bm);
  z = wsum(z);
  if (threadIdx.x % 32 == 0)
    bz[w] = z;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < 8; ++i)
      t += bz[i];
    ws[3 * blockIdx.x] = bm;
    ws[3 * blockIdx.x + 1] = __int_as_float(bx);
    ws[3 * blockIdx.x + 2] = t;
  }
}
__global__ void k_argmax2(const float* ws, int nb, const int* map, int* tokens,
                          int row, int* prev, float* conf) {
  if (threadIdx.x != 0)
    return;
  float v = -INFINITY;
  int ix = 0;
  for (int b = 0; b < nb; ++b) {
    const float y = ws[3 * b];
    const int i = __float_as_int(ws[3 * b + 1]);
    if (y > v || (y == v && i < ix)) {
      v = y;
      ix = i;
    }
  }
  float z = 0.f;
  for (int b = 0; b < nb; ++b)
    z += ws[3 * b + 2] * __expf(ws[3 * b] - v);
  const int tok = map ? map[ix] : ix;
  tokens[row] = tok;
  *prev = tok;
  conf[row] = 1.f / z;
}

__global__ void __launch_bounds__(256)
    k_gather_q8(const uint8_t* src, const int* ids, int n, uint8_t* dst) {
  constexpr int RB = (kEmbd / 32) * 34;  // 2448 bytes per row
  const int r = blockIdx.x;
  if (r >= n)
    return;
  const uint8_t* s = src + (size_t)ids[r] * RB;
  uint8_t* d = dst + (size_t)r * RB;
  for (int i = threadIdx.x; i < RB; i += 256)
    d[i] = s[i];
}

}  // namespace

void Mix(hipStream_t s, const MixArgs& a) {
  k_mix<<<dim3(a.n, 4), 256, 0, s>>>(a);
  Chk("mix");
}
void KvStore(hipStream_t s, const KvStoreArgs& a) {
  k_kv_store<<<dim3(a.n, kKvHeads), 64, 0, s>>>(a);
  Chk("kv_store");
}
void Embed(hipStream_t s, const void* token_embd, const float* tok_norm,
           const int* anchor, const float* mask, const float* w, float* x,
           float* h, _Float16* h16, float eps) {
  k_embed<<<kBlock, 256, 0, s>>>(static_cast<const uint8_t*>(token_embd),
                                 tok_norm, anchor, mask, w, x, h, h16, eps);
  Chk("embed");
}
void QkNormRope(hipStream_t s, float* q, float* k, const float* qnorm,
                const float* knorm, const int* pos, float theta, float eps) {
  k_qk<<<dim3(kBlock, kHeads + kKvHeads), 64, 0, s>>>(q, k, qnorm, knorm, pos,
                                                      theta, eps);
  Chk("qk");
}
void Attn(hipStream_t s, const AttnArgs& a) {
  k_attn<<<dim3(kHeads, a.nsplits), 256, 0, s>>>(a);
  k_attn_combine<<<dim3(kHeads, kBlock), 128, 0, s>>>(a);
  Chk("attn");
}
void AddNorm(hipStream_t s, float* x, const float* y, const float* w, float* h,
             _Float16* h16, int n, float eps) {
  k_add_norm<<<n, 256, 0, s>>>(x, y, w, h, h16, eps);
  Chk("add_norm");
}
void Swiglu(hipStream_t s, const float* g, const float* u, _Float16* out16,
            int n) {
  k_swiglu<<<(n + 255) / 256, 256, 0, s>>>(g, u, out16, n);
  Chk("swiglu");
}
void MarkovStep(hipStream_t s, const MarkovArgs& a) {
  // lat lives right after the joint rows
  float* lat = a.joint + 3 * kRank;
  k_joint<<<3 * kRank / 8, 256, 0, s>>>(a);
  k_state<<<1, 256, 0, s>>>(a, lat);
  Chk("markov");
}
void QuantPt(hipStream_t s, const float* src, int n, int8_t* q, float* scale) {
  k_quant_pt<<<(n + 255) / 256, 256, 0, s>>>(src, n, q, scale);
  Chk("quant pt");
}
void RowNormSqBf16(hipStream_t s, const uint16_t* src, int rows, float* out) {
  k_row_norm_sq_bf16<<<(rows + 7) / 8, 256, 0, s>>>(src, rows, out);
  Chk("row norm");
}
void QuantPtBf16(hipStream_t s, const uint16_t* src, int n, int8_t* q,
                 float* scale) {
  k_quant_pt_bf16<<<(n + 255) / 256, 256, 0, s>>>(src, n, q, scale);
  Chk("quant pt bf16");
}
void LrLogits8Batch(hipStream_t s, const int8_t* pt, const float* ps,
                    const float* cb, const LrBatch& batch, int n) {
  const int grid = (n / 4 + 255) / 256;
  if (batch.count <= 1)
    k_lr_logits8_batch<1><<<grid, 256, 0, s>>>(pt, ps, cb, batch, n);
  else if (batch.count <= 2)
    k_lr_logits8_batch<2><<<grid, 256, 0, s>>>(pt, ps, cb, batch, n);
  else if (batch.count <= 4)
    k_lr_logits8_batch<4><<<grid, 256, 0, s>>>(pt, ps, cb, batch, n);
  else
    k_lr_logits8_batch<8><<<grid, 256, 0, s>>>(pt, ps, cb, batch, n);
  Chk("lr logits");
}
void LrLogits8(hipStream_t s, const int8_t* pt, const float* ps,
               const float* cb, const float* base, const float* lat,
               float* logits, int n) {
  LrBatch b{};
  b.base[0] = base;
  b.lat[0] = lat;
  b.out[0] = logits;
  b.count = 1;
  LrLogits8Batch(s, pt, ps, cb, b, n);
}
void Argmax(hipStream_t s, const float* logits, int n, const int* map,
            float* ws, int* tokens, int row, int* prev, float* conf) {
  constexpr int nb = 128;
  k_argmax1<<<nb, 256, 0, s>>>(logits, n, ws);
  k_argmax2<<<1, 32, 0, s>>>(ws, nb, map, tokens, row, prev, conf);
  Chk("argmax");
}
void Gemv8N(hipStream_t s, int n, const void* const* w, const int* m, int k,
            const _Float16* x, float* const* y, int nrows, void* qa,
            float* qs) {
  if (nrows < 1 || nrows > 64 || k % 32 != 0)
    throw std::runtime_error("inkling2 dspark: gemv8 shape");
  Gemv8Set set{};
  set.n = n;
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    if (m[i] % 16)
      throw std::runtime_error("inkling2 dspark: gemv8 shape");
    set.w[i] = static_cast<const uint8_t*>(w[i]);
    set.y[i] = y[i];
    set.m[i] = m[i];
    set.blocks[i] = m[i] / 16;
    blocks += m[i] / 16;
  }
  const int ntiles = (nrows + 15) / 16;
  const int qwarps = ntiles * 16 * (k / 32);
  v4i* qap = static_cast<v4i*>(qa);
  k_quant_q8<<<(qwarps + 7) / 8, 256, 0, s>>>(x, nrows, k, qap, qs);
  constexpr int NW = 4;
  if (nrows <= 8)
    k_gemv_wmma<NW, 1, 4><<<blocks, NW * 32, 0, s>>>(set, k, qap, qs, nrows);
  else if (nrows <= 16)
    k_gemv_wmma<NW, 1, 8><<<blocks, NW * 32, 0, s>>>(set, k, qap, qs, nrows);
  else if (nrows <= 32)
    k_gemv_wmma<NW, 2, 8><<<blocks, NW * 32, 0, s>>>(set, k, qap, qs, nrows);
  else if (nrows <= 48)
    k_gemv_wmma<NW, 3, 8><<<blocks, NW * 32, 0, s>>>(set, k, qap, qs, nrows);
  else
    k_gemv_wmma<NW, 4, 8><<<blocks, NW * 32, 0, s>>>(set, k, qap, qs, nrows);
  Chk("gemv8");
}
void Gemv8(hipStream_t s, const void* w, int m, int k, const _Float16* x,
           float* y, int nrows, void* qa, float* qs) {
  Gemv8N(s, 1, &w, &m, k, x, &y, nrows, qa, qs);
}
void CoupledSample(hipStream_t s, const float* logits, int n, const int* map,
                   float softcap, float temperature, float min_p, float u,
                   float* ws, int* tokens, int row, int* prev, float* conf) {
  k_cs_max<<<kCsBlocks, 256, 0, s>>>(logits, n, softcap, ws);
  k_cs_sum<<<kCsBlocks, 256, 0, s>>>(logits, n, softcap, temperature, min_p,
                                     ws);
  k_cs_pick<<<1, 256, 0, s>>>(logits, n, map, softcap, temperature, min_p, u,
                              ws, tokens, row, prev, conf);
  Chk("coupled sample");
}
__global__ void __launch_bounds__(256)
    k_to_half(const float* __restrict__ x, std::size_t n, _Float16* y) {
  const std::size_t i = (std::size_t)blockIdx.x * 256 + threadIdx.x;
  if (i < n)
    y[i] = (_Float16)x[i];
}
void ToHalf(hipStream_t s, const float* src, std::size_t n, _Float16* dst) {
  k_to_half<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src, n, dst);
  Chk("to half");
}
void GatherQ8Rows(hipStream_t s, const void* src, const int* ids, int n,
                  void* dst) {
  k_gather_q8<<<n, 256, 0, s>>>(static_cast<const uint8_t*>(src), ids, n,
                                static_cast<uint8_t*>(dst));
  Chk("gather");
}

}  // namespace gufo::models::inkling2::dsp
