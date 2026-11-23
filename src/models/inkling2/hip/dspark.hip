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
      a.out16[((size_t)j * a.n + r) * kEmbd + c] = (_Float16)y;
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
  const int lo = a.window ? max(0, pos - a.window) : 0;
  const int nctx = pos - lo, nkeys = nctx + kBlock;
  const int per = (nkeys + kAttnSplits - 1) / kAttnSplits;
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
  __shared__ float so[8][NQ][kHead];
#pragma unroll
  for (int r = 0; r < NQ; ++r) {
    if (l == 0) {
      sm[w][r][0] = m[r];
      sm[w][r][1] = s[r];
    }
    so[w][r][4 * l] = o[r].x;
    so[w][r][4 * l + 1] = o[r].y;
    so[w][r][4 * l + 2] = o[r].z;
    so[w][r][4 * l + 3] = o[r].w;
  }
  __syncthreads();
  float* ws = a.ws + ((size_t)h * kAttnSplits + split) * NQ * (kHead + 2);
  for (int e = threadIdx.x; e < NQ * kHead; e += 256) {
    const int r = e / kHead, d = e % kHead;
    float mm = -INFINITY;
    for (int x = 0; x < 8; ++x)
      mm = fmaxf(mm, sm[x][r][0]);
    float ss = 0.f, oo = 0.f;
    if (mm > -INFINITY)
      for (int x = 0; x < 8; ++x) {
        const float f = __expf(sm[x][r][0] - mm);
        ss += sm[x][r][1] * f;
        oo += so[x][r][d] * f;
      }
    ws[(size_t)r * (kHead + 2) + 2 + d] = oo;
    if (d == 0) {
      ws[(size_t)r * (kHead + 2)] = mm;
      ws[(size_t)r * (kHead + 2) + 1] = ss;
    }
  }
}
__global__ void __launch_bounds__(128) k_attn_combine(AttnArgs a) {
  constexpr int NQ = kBlock;
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const float* ws = a.ws + (size_t)h * kAttnSplits * NQ * (kHead + 2);
  float mm = -INFINITY;
  for (int x = 0; x < kAttnSplits; ++x)
    mm = fmaxf(mm, ws[((size_t)x * NQ + r) * (kHead + 2)]);
  float ss = 0.f, oo = 0.f;
  for (int x = 0; x < kAttnSplits; ++x) {
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
// rows of BF16 [5120]) = W_j [state, w1[prev], h_i, f] + b: one wave per row,
// the 5120 inputs gathered on the fly.
__global__ void __launch_bounds__(256) k_joint(MarkovArgs a) {
  const int row = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (row >= 3 * kRank)
    return;
  const uint16_t* wr = a.wj + (size_t)row * kJoint;
  const uint16_t* w1 = a.w1 + (size_t)(*a.prev) * kRank;
  const float* hi = a.h + (size_t)a.row * kEmbd;
  float acc = 0.f;
  for (int c = l; c < kJoint; c += 32) {
    float x;
    if (c < kRank)
      x = a.state[c];
    else if (c < 2 * kRank)
      x = bf16f(w1[c - kRank]);
    else if (c < 2 * kRank + kEmbd)
      x = hi[c - 2 * kRank];
    else
      x = a.f[c - 2 * kRank - kEmbd];
    acc += bf16f(wr[c]) * x;
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
// heff = h_i + W_o lat + b_o (2304 rows of BF16 [256]), wave per row
__global__ void __launch_bounds__(256) k_heff(MarkovArgs a, const float* lat) {
  const int row = blockIdx.x * 8 + threadIdx.x / 32, l = threadIdx.x % 32;
  if (row >= kEmbd)
    return;
  const uint16_t* wr = a.wo + (size_t)row * kRank;
  float acc = 0.f;
#pragma unroll
  for (int c = l; c < kRank; c += 32)
    acc += bf16f(wr[c]) * lat[c];
  acc = wsum(acc);
  if (l == 0)
    a.heff[row] = a.h[(size_t)a.row * kEmbd + row] + acc + a.bo[row];
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
  k_attn<<<dim3(kHeads, kAttnSplits), 256, 0, s>>>(a);
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
  k_heff<<<kEmbd / 8, 256, 0, s>>>(a, lat);
  Chk("markov");
}
void Argmax(hipStream_t s, const float* logits, int n, const int* map,
            float* ws, int* tokens, int row, int* prev, float* conf) {
  constexpr int nb = 128;
  k_argmax1<<<nb, 256, 0, s>>>(logits, n, ws);
  k_argmax2<<<1, 32, 0, s>>>(ws, nb, map, tokens, row, prev, conf);
  Chk("argmax");
}
void GatherQ8Rows(hipStream_t s, const void* src, const int* ids, int n,
                  void* dst) {
  k_gather_q8<<<n, 256, 0, s>>>(static_cast<const uint8_t*>(src), ids, n,
                                static_cast<uint8_t*>(dst));
  Chk("gather");
}

}  // namespace gufo::models::inkling2::dsp
