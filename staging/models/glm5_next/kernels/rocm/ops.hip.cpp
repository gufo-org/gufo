#include "staging/models/glm5_next/kernels/rocm/ops.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

// Fused per-token operators for the glm5-next trunk. Every formula follows
// docs/models/glm-5.3-flash/IMPLEMENTATION.md; silu uses ggml's x/(1+e^-x)
// and the norms use 1/sqrtf (not rsqrtf) so the values track the scalar
// oracle. The parity test (staging/tests/models/glm5_next/rocm_ops_test.cpp)
// checks each operator against an independent host double reference.
namespace gufo::models::glm5_next::rocm {
namespace {

constexpr unsigned kBlock = 256;
constexpr unsigned kMaxGrid = 65535U;

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0F + expf(-x));
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// Block reduce for kBlock threads; thread 0 returns the sum. Callers publish
// the result through shared memory before reuse of `red`.
__device__ __forceinline__ float BlockReduceSum(float v, float* red) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = WarpReduceSum(v);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = (lane < kBlock / 32) ? red[lane] : 0.0F;
    v = WarpReduceSum(v);
  }
  return v;
}

unsigned Grid1D(std::uint32_t count) {
  const std::uint64_t blocks =
      (static_cast<std::uint64_t>(count) + kBlock - 1) / kBlock;
  return blocks < kMaxGrid ? static_cast<unsigned>(blocks) : kMaxGrid;
}

std::uint32_t CeilPow2(std::uint32_t v) {
  std::uint32_t p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

__global__ void AddBiasKernel(float* data, const float* bias,
                              std::uint32_t total, std::uint32_t bias_len) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    data[i] += bias[i % bias_len];
  }
}

__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ dst, std::uint32_t n,
                              std::uint32_t x_stride, std::uint32_t dst_stride,
                              float eps) {
  __shared__ float red[kBlock / 32];
  __shared__ float scale;
  const float* xr = x + static_cast<std::size_t>(blockIdx.x) * x_stride;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < n; i += kBlock) {
    const float v = xr[i];
    sum += v * v;
  }
  sum = BlockReduceSum(sum, red);
  if (threadIdx.x == 0) {
    scale = 1.0F / sqrtf(sum / static_cast<float>(n) + eps);
  }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < n; i += kBlock) {
    dst[static_cast<std::size_t>(blockIdx.x) * dst_stride + i] =
        w[i] * (xr[i] * scale);
  }
}

__global__ void LayerNormKernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                float* __restrict__ dst, std::uint32_t n,
                                float eps) {
  __shared__ float red[kBlock / 32];
  __shared__ float stats[2];
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < n; i += kBlock) {
    sum += x[i];
  }
  sum = BlockReduceSum(sum, red);
  if (threadIdx.x == 0) {
    stats[0] = sum / static_cast<float>(n);
  }
  __syncthreads();
  const float mean = stats[0];
  float var = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < n; i += kBlock) {
    const float d = x[i] - mean;
    var += d * d;
  }
  sum = BlockReduceSum(var, red);
  if (threadIdx.x == 0) {
    stats[1] = 1.0F / sqrtf(sum / static_cast<float>(n) + eps);
  }
  __syncthreads();
  const float inv = stats[1];
  for (std::uint32_t i = threadIdx.x; i < n; i += kBlock) {
    dst[i] = w[i] * ((x[i] - mean) * inv) + b[i];
  }
}

// One thread computes the 24 mHC coefficients: the contract is the exact
// F32 operation order of the scalar sinkhorn, so a deterministic single
// walk is the reference, not a parallel reduction.
__global__ void HcCoeffsKernel(const float* __restrict__ mixes,
                               const float* __restrict__ scale,
                               const float* __restrict__ base, unsigned iters,
                               float eps, float* __restrict__ pre,
                                float* __restrict__ post,
                                float* __restrict__ comb) {
  const std::size_t row = blockIdx.y;
  mixes += row * 24;
  pre += row * 4;
  post += row * 4;
  comb += row * 16;
  float a[24];
  for (int i = 0; i < 24; ++i) {
    const int role = i < 4 ? 0 : (i < 8 ? 1 : 2);
    a[i] = mixes[i] * scale[role] + base[i];
  }
  for (int i = 0; i < 4; ++i) {
    pre[i] = Sigmoid(a[i]) + eps;
    post[i] = Sigmoid(a[4 + i]) * 2.0F;
  }
  float c[4][4];  // [src][dst] at offset 8 + src*4 + dst (comb idst + hc*isrc)
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      c[s][d] = a[8 + s * 4 + d];
    }
  }
  // Exact F32 order of the ggml reference: row softmax over dst with
  // comb = comb * (1 / sum) + eps, one column normalization, then
  // (rows, columns) pairs for the remaining iterations, each with a
  // sum seeded by eps.
  for (int s = 0; s < 4; ++s) {
    float mx = c[s][0];
    for (int d = 1; d < 4; ++d) {
      mx = fmaxf(mx, c[s][d]);
    }
    float sum = 0.0F;
    for (int d = 0; d < 4; ++d) {
      c[s][d] = expf(c[s][d] - mx);
      sum += c[s][d];
    }
    const float inv = 1.0F / sum;
    for (int d = 0; d < 4; ++d) {
      c[s][d] = c[s][d] * inv + eps;
    }
  }
  for (int d = 0; d < 4; ++d) {
    float sum = eps;
    for (int s = 0; s < 4; ++s) {
      sum += c[s][d];
    }
    const float inv = 1.0F / sum;
    for (int s = 0; s < 4; ++s) {
      c[s][d] *= inv;
    }
  }
  for (unsigned it = 1; it < iters; ++it) {
    for (int s = 0; s < 4; ++s) {
      float sum = eps;
      for (int d = 0; d < 4; ++d) {
        sum += c[s][d];
      }
      const float inv = 1.0F / sum;
      for (int d = 0; d < 4; ++d) {
        c[s][d] *= inv;
      }
    }
    for (int d = 0; d < 4; ++d) {
      float sum = eps;
      for (int s = 0; s < 4; ++s) {
        sum += c[s][d];
      }
      const float inv = 1.0F / sum;
      for (int s = 0; s < 4; ++s) {
        c[s][d] *= inv;
      }
    }
  }
  for (int i = 0; i < 16; ++i) {
    comb[i] = c[i / 4][i % 4];
  }
}

__global__ void HcInputsKernel(const float* __restrict__ streams,
                               const float* __restrict__ pre,
                               std::uint32_t pre_stride, std::uint32_t n,
                               float* __restrict__ dst) {
  const std::size_t row = blockIdx.y;
  streams += row * 4 * n;
  pre += row * pre_stride;
  dst += row * n;
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (i >= static_cast<std::size_t>(n)) {
    return;
  }
  const float p0 = pre[0];
  const float p1 = pre[1];
  const float p2 = pre[2];
  const float p3 = pre[3];
  float acc = streams[i] * p0;
  acc = fmaf(streams[static_cast<std::size_t>(n) + i], p1, acc);
  acc = fmaf(streams[2 * static_cast<std::size_t>(n) + i], p2, acc);
  acc = fmaf(streams[3 * static_cast<std::size_t>(n) + i], p3, acc);
  dst[i] = acc;
}

__global__ void HcWritebackKernel(const float* __restrict__ y,
                                  const float* __restrict__ res,
                                  const float* __restrict__ post,
                                  const float* __restrict__ comb,
                                  std::uint32_t n, float* __restrict__ out) {
  const std::size_t row = blockIdx.y;
  y += row * n;
  res += row * 4 * n;
  post += row * 4;
  comb += row * 16;
  out += row * 4 * n;
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * kBlock + threadIdx.x;
  if (i >= static_cast<std::size_t>(n)) {
    return;
  }
  // comb is stored [src][dst]; the stream D sees sum_src res_src * comb_srcD.
  float cd[4][4];
  for (unsigned d = 0; d < 4; ++d) {
    for (unsigned s = 0; s < 4; ++s) {
      cd[d][s] = comb[s * 4 + d];
    }
  }
  const float yv = y[i];
  const float r0 = res[i];
  const float r1 = res[static_cast<std::size_t>(n) + i];
  const float r2 = res[2 * static_cast<std::size_t>(n) + i];
  const float r3 = res[3 * static_cast<std::size_t>(n) + i];
  for (unsigned d = 0; d < 4; ++d) {
    float acc = post[d] * yv;
    acc = fmaf(r0, cd[d][0], acc);
    acc = fmaf(r1, cd[d][1], acc);
    acc = fmaf(r2, cd[d][2], acc);
    acc = fmaf(r3, cd[d][3], acc);
    out[static_cast<std::size_t>(d) * n + i] = acc;
  }
}

__global__ void KdaConvStepKernel(std::uint64_t pos,
                                  const float* __restrict__ raw,
                                  const float* __restrict__ w,
                                  float* __restrict__ hist, std::uint32_t n,
                                  float* __restrict__ dst) {
  const std::uint64_t s0 = pos % 3;  // holds position p-3
  for (std::uint32_t ch = blockIdx.x * kBlock + threadIdx.x; ch < n;
       ch += gridDim.x * kBlock) {
    const float x3 = hist[s0 * n + ch];
    const float x2 = hist[((s0 + 1) % 3) * n + ch];
    const float x1 = hist[((s0 + 2) % 3) * n + ch];
    const float x0 = raw[ch];
    const float acc = w[ch * 4 + 0] * x3 + w[ch * 4 + 1] * x2 +
                      w[ch * 4 + 2] * x1 + w[ch * 4 + 3] * x0;
    dst[ch] = Silu(acc);
    hist[s0 * n + ch] = x0;
  }
}

// blockDim.x == d. State is [heads][d][d] with rows i strided by d.
__global__ void KdaFoldSkKernel(float* __restrict__ state,
                                const float* __restrict__ eg,
                                const float* __restrict__ k, unsigned d,
                                float* __restrict__ sk) {
  const unsigned h = blockIdx.y;
  const unsigned j = threadIdx.x;
  float* st = state + (static_cast<std::size_t>(h) * d) * d;
  float acc = 0.0F;
  for (unsigned i = 0; i < d; ++i) {
    const float s = st[static_cast<std::size_t>(i) * d + j] * eg[h * d + i];
    st[static_cast<std::size_t>(i) * d + j] = s;
    acc = fmaf(k[h * d + i], s, acc);
  }
  sk[h * d + j] = acc;
}

__global__ void KdaUpdateOutKernel(float* __restrict__ state,
                                   const float* __restrict__ k,
                                   const float* __restrict__ q,
                                   const float* __restrict__ v,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ sk, unsigned d,
                                   float* __restrict__ o) {
  const unsigned h = blockIdx.y;
  const unsigned j = threadIdx.x;
  float* st = state + (static_cast<std::size_t>(h) * d) * d;
  const float dj = beta[h] * (v[h * d + j] - sk[h * d + j]);
  float acc = 0.0F;
  for (unsigned i = 0; i < d; ++i) {
    const float s = st[static_cast<std::size_t>(i) * d + j] + k[h * d + i] * dj;
    st[static_cast<std::size_t>(i) * d + j] = s;
    acc = fmaf(q[h * d + i], s, acc);
  }
  o[h * d + j] = acc;
}

// Chunked conv: identical arithmetic and ring updates to running
// KdaConvStepKernel for each position of the chunk in order.
__global__ void KdaConvStepChunkKernel(std::uint64_t pos0,
                                       const float* __restrict__ raw,
                                       const float* __restrict__ w,
                                       float* __restrict__ hist,
                                       std::uint32_t n, std::uint32_t ntok,
                                       float* __restrict__ dst) {
  for (std::uint32_t ch = blockIdx.x * kBlock + threadIdx.x; ch < n;
       ch += gridDim.x * kBlock) {
    const float w0 = w[ch * 4 + 0];
    const float w1 = w[ch * 4 + 1];
    const float w2 = w[ch * 4 + 2];
    const float w3 = w[ch * 4 + 3];
    for (std::uint32_t t = 0; t < ntok; ++t) {
      const std::uint64_t pos = pos0 + t;
      const std::uint32_t s0 = static_cast<std::uint32_t>(pos % 3);
      const std::size_t rb = static_cast<std::size_t>(t) * n + ch;
      const float x3 = hist[s0 * n + ch];
      const float x2 = hist[((s0 + 1) % 3) * n + ch];
      const float x1 = hist[((s0 + 2) % 3) * n + ch];
      const float x0 = raw[rb];
      const float acc = w0 * x3 + w1 * x2 + w2 * x1 + w3 * x0;
      dst[rb] = Silu(acc);
      hist[s0 * n + ch] = x0;
    }
  }
}

// Fused fold+update over a chunk, state resident in dynamic shared memory
// (a workgroup per head, a thread per output column j). The per-token
// arithmetic and order match KdaFoldSkKernel + KdaUpdateOutKernel exactly;
// `sk` never leaves registers.
__global__ void KdaFoldUpdateChunkKernel(
    float* __restrict__ state, const float* __restrict__ eg,
    const float* __restrict__ k, const float* __restrict__ q,
    const float* __restrict__ v, const float* __restrict__ beta,
    std::uint32_t heads, std::uint32_t d, std::uint32_t ntok,
    float* __restrict__ o) {
  extern __shared__ float st[];  // [d][d]
  const std::size_t dd = static_cast<std::size_t>(d) * d;
  float* gst = state + static_cast<std::size_t>(blockIdx.x) * dd;
  const unsigned j = threadIdx.x;
  const std::size_t inner = static_cast<std::size_t>(heads) * d;
  for (std::size_t t = j; t < dd; t += blockDim.x) {
    st[t] = gst[t];
  }
  const std::size_t hcol = static_cast<std::size_t>(blockIdx.x) * d;
  for (std::uint32_t r = 0; r < ntok; ++r) {
    const std::size_t base = r * inner + hcol;
    float sk = 0.0F;
    for (unsigned i = 0; i < d; ++i) {
      const std::size_t at = static_cast<std::size_t>(i) * d + j;
      const float s = st[at] * eg[base + i];
      st[at] = s;
      sk = fmaf(k[base + i], s, sk);
    }
    const float dj =
        beta[static_cast<std::size_t>(r) * heads + blockIdx.x] *
        (v[base + j] - sk);
    float acc = 0.0F;
    for (unsigned i = 0; i < d; ++i) {
      const std::size_t at = static_cast<std::size_t>(i) * d + j;
      const float s = st[at] + k[base + i] * dj;
      st[at] = s;
      acc = fmaf(q[base + i], s, acc);
    }
    o[base + j] = acc;
  }
  for (std::size_t t = j; t < dd; t += blockDim.x) {
    gst[t] = st[t];
  }
}

__global__ void KdaOutGateKernel(const float* __restrict__ o,
                                 const float* __restrict__ w,
                                 const float* __restrict__ g2,
                                 float* __restrict__ dst, unsigned d,
                                 float eps) {
  __shared__ float red[kBlock / 32];
  __shared__ float scale;
  const float* orow = o + static_cast<std::size_t>(blockIdx.x) * d;
  float sum = 0.0F;
  for (unsigned i = threadIdx.x; i < d; i += kBlock) {
    sum += orow[i] * orow[i];
  }
  sum = BlockReduceSum(sum, red);
  if (threadIdx.x == 0) {
    scale = 1.0F / sqrtf(sum / static_cast<float>(d) + eps);
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < d; i += kBlock) {
    dst[static_cast<std::size_t>(blockIdx.x) * d + i] =
        w[i] * (orow[i] * scale) *
        Sigmoid(g2[static_cast<std::size_t>(blockIdx.x) * d + i]);
  }
}

__global__ void IdxScoreKernel(const float* __restrict__ pool,
                               std::uint32_t pool_stride, std::uint32_t seen0,
                               unsigned kpool, unsigned key_len, unsigned heads,
                               const float* __restrict__ iq,
                               std::size_t iq_stride,
                               const float* __restrict__ ape,
                               const float* __restrict__ w,
                               std::size_t w_stride, float q_scale,
                               float* __restrict__ scores,
                               std::size_t scores_stride) {
  __shared__ float kp[128];
  __shared__ float total;
  const unsigned P = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  iq += static_cast<std::size_t>(t) * iq_stride;
  w += static_cast<std::size_t>(t) * w_stride;
  scores += static_cast<std::size_t>(t) * scores_stride;
  if (threadIdx.x == 0) {
    total = 0.0F;
  }
  if (P * kpool >= seen0 + t) {
    return;  // block belongs to a shorter token's pool list
  }
  const unsigned seen = seen0 + t - P * kpool;  // >= 1 for launched pools
  const unsigned taps = seen < kpool ? seen : kpool;
  const float* rows = pool + static_cast<std::size_t>(P) * kpool * pool_stride;
  for (unsigned j = threadIdx.x; j < key_len; j += kBlock) {
    float g[4];
    for (unsigned t = 0; t < taps; ++t) {
      g[t] = ape[static_cast<std::size_t>(t) * key_len + j] +
             rows[static_cast<std::size_t>(t) * pool_stride + key_len + j];
    }
    float mx = g[0];
    for (unsigned t = 1; t < taps; ++t) {
      mx = fmaxf(mx, g[t]);
    }
    float sum = 0.0F;
    for (unsigned t = 0; t < taps; ++t) {
      g[t] = expf(g[t] - mx);
      sum += g[t];
    }
    const float inv = 1.0F / sum;
    float acc = 0.0F;
    for (unsigned t = 0; t < taps; ++t) {
      acc = fmaf(rows[static_cast<std::size_t>(t) * pool_stride + j],
                 g[t] * inv, acc);
    }
    kp[j] = acc;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x < heads) {
      float acc = 0.0F;
      const float* iqh = iq + static_cast<std::size_t>(threadIdx.x) * key_len;
      for (unsigned j = 0; j < key_len; ++j) {
        acc = fmaf(iqh[j], kp[j], acc);
      }
      if (acc < 0.0F) {
        acc = 0.0F;
      }
      acc = WarpReduceSum(acc * w[threadIdx.x]);
      if (threadIdx.x == 0) {
        total = acc;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    scores[P] = total * q_scale;
  }
}

// Ordered u64 key: ascending sort yields score-descending order, ties by
// ascending index (low half is the plain index; keeping the first K2 of the
// ascending order puts the smaller index first on value ties). Out-of-range
// entries use the all-ones neutral key, which sorts last.
__device__ __forceinline__ std::uint64_t ScoreKey(float score,
                                                  std::uint32_t index) {
  std::uint32_t u;
  __builtin_memcpy(&u, &score, sizeof(u));
  const std::uint32_t asc = (u & 0x80000000U) ? ~u : (u ^ 0x80000000U);
  return (static_cast<std::uint64_t>(~asc) << 32) | index;
}

// Full bitonic sort of C (power of two) u64 values in shared memory to
// ascending order. blockDim.x <= 1024 with an inner stride loop.
template<int C>
__device__ void BitonicSortAsc(std::uint64_t* s) {
  for (int k = 2; k <= C; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      for (int a = threadIdx.x; a < C; a += blockDim.x) {
        if (a & j) {
          continue;
        }
        const int b = a ^ j;
        const bool asc = (a & k) == 0;
        const std::uint64_t va = s[a];
        const std::uint64_t vb = s[b];
        if (asc ? (va > vb) : (va < vb)) {
          s[a] = vb;
          s[b] = va;
        }
      }
    }
  }
}

// First stage: block b keys elements [b * 2K2, b * 2K2 + 2K2) of scores
// (neutral past n) and writes its K2 best, sorted ascending, as candidate
// run b at out_c[b * K2 ..]. Rounds stay packed: run j lives at j * K2.
template<int K2>
__global__ void BitonicGatherKernel(const float* __restrict__ scores,
                                    std::uint64_t* __restrict__ out_c,
                                    std::int32_t* __restrict__ out_idx,
                                    std::uint32_t n, unsigned k, bool final) {
  __shared__ std::uint64_t s[2 * K2];
  const std::uint64_t base = static_cast<std::size_t>(blockIdx.x) * (2 * K2);
  for (unsigned i = threadIdx.x; i < 2 * K2; i += blockDim.x) {
    s[i] = (base + i < n) ? ScoreKey(scores[base + i],
                                     static_cast<std::uint32_t>(base + i))
                          : 0xFFFFFFFFFFFFFFFFULL;
  }
  BitonicSortAsc<2 * K2>(s);
  __syncthreads();
  if (final) {
    // Final blocks may have fewer threads than k (thread cap 1024), so
    // stride the write; k <= K2 <= 2K2 keeps the source in range.
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
      out_idx[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(s[i]));
    }
  } else {
    for (unsigned i = threadIdx.x; i < K2; i += blockDim.x) {
      out_c[static_cast<std::size_t>(blockIdx.x) * K2 + i] = s[i];
    }
  }
}

// Merge stage: block b combines the two sorted runs 2b and 2b + 1 (region
// [b * 2K2 .. (b + 1) * 2K2)) into the best K2 at out_c[b * K2 ..]. For an
// odd list count the LAST block (gridDim - 1 = ceil(lists / 2) - 1) has no
// partner (run lists - 1 = 2 * blockIdx) and passes it through unchanged.
template<int K2>
__global__ void BitonicMergeKernel(const std::uint64_t* __restrict__ in,
                                   std::uint64_t* __restrict__ out_c,
                                   std::int32_t* __restrict__ out_idx,
                                   unsigned lists, unsigned k, bool final) {
  __shared__ std::uint64_t s[2 * K2];
  const std::uint64_t base = static_cast<std::size_t>(blockIdx.x) * (2 * K2);
  if ((lists & 1U) && static_cast<unsigned>(blockIdx.x) + 1 == gridDim.x) {
    // Odd tail: run 2b (= lists - 1) starts at in[base].
    if (final) {
      for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
        out_idx[i] =
            static_cast<std::int32_t>(static_cast<std::uint32_t>(in[base + i]));
      }
    } else {
      for (unsigned i = threadIdx.x; i < K2; i += blockDim.x) {
        out_c[static_cast<std::size_t>(blockIdx.x) * K2 + i] = in[base + i];
      }
    }
    return;
  }
  for (unsigned i = threadIdx.x; i < 2 * K2; i += blockDim.x) {
    s[i] = in[base + i];
  }
  BitonicSortAsc<2 * K2>(s);
  __syncthreads();
  if (final) {
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
      out_idx[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(s[i]));
    }
  } else {
    for (unsigned i = threadIdx.x; i < K2; i += blockDim.x) {
      out_c[static_cast<std::size_t>(blockIdx.x) * K2 + i] = s[i];
    }
  }
}

__global__ void SwigluClampKernel(const float* __restrict__ gate,
                                  const float* __restrict__ up,
                                  float* __restrict__ dst, std::uint32_t total,
                                  std::uint32_t n, std::uint32_t gate_stride,
                                  std::uint32_t up_stride,
                                  std::uint32_t dst_stride, float clamp_gate,
                                  float clamp_up) {
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < total;
       i += gridDim.x * kBlock) {
    const std::uint32_t r = i / n;
    const std::uint32_t c = i % n;
    const float g =
        fminf(gate[static_cast<std::size_t>(r) * gate_stride + c], clamp_gate);
    const float u =
        fminf(fmaxf(up[static_cast<std::size_t>(r) * up_stride + c], -clamp_up),
              clamp_up);
    dst[static_cast<std::size_t>(r) * dst_stride + c] = g * Sigmoid(g) * u;
  }
}

__global__ void WeightedSumAddKernel(float* __restrict__ dst,
                                     const float* __restrict__ src,
                                     const float* __restrict__ weights,
                                      std::uint32_t rows, std::uint32_t n,
                                      std::uint32_t stride) {
  const std::size_t row = blockIdx.y;
  dst += row * n;
  src += row * static_cast<std::size_t>(rows) * stride;
  weights += weights != nullptr ? row * rows : 0;
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < n;
       i += gridDim.x * kBlock) {
    float acc = 0.0F;
    if (weights != nullptr) {
      for (std::uint32_t r = 0; r < rows; ++r) {
        acc = fmaf(weights[r], src[static_cast<std::size_t>(r) * stride + i],
                   acc);
      }
    } else {
      for (std::uint32_t r = 0; r < rows; ++r) {
        acc += src[static_cast<std::size_t>(r) * stride + i];
      }
    }
    dst[i] += acc;
  }
}

__global__ void MoERouteKernel(const float* __restrict__ logits,
                               const float* __restrict__ bias, unsigned experts,
                               unsigned top_k, float scale,
                                float* __restrict__ weights,
                                std::int32_t* __restrict__ ids) {
  const std::size_t row = blockIdx.y;
  logits += row * experts;
  weights += row * top_k;
  ids += row * top_k;
  __shared__ float s[1024];   // selection keys (sigmoid + bias)
  __shared__ float p[1024];   // unbiased sigmoid probabilities (weights)
  __shared__ std::uint64_t red[1024 / 32];  // one warp-leader key each
  const unsigned t = threadIdx.x;
  const float prob = t < experts ? Sigmoid(logits[t]) : -FLT_MAX;
  p[t] = prob;
  s[t] = t < experts ? prob + bias[t] : -FLT_MAX;
  __syncthreads();
  for (unsigned r = 0; r < top_k; ++r) {
    // Descending sortable score with the inverted index in the low half:
    // the max wins ties with the smallest expert id.
    const float v = s[t];
    std::uint32_t u;
    __builtin_memcpy(&u, &v, sizeof(u));
    const std::uint32_t asc = (u & 0x80000000U) ? ~u : (u ^ 0x80000000U);
    std::uint64_t key = (static_cast<std::uint64_t>(asc) << 32) |
                        static_cast<std::uint32_t>(1023 - t);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const std::uint64_t other = static_cast<std::uint64_t>(
          __shfl_down(static_cast<std::uintptr_t>(key), off));
      if (other > key) {
        key = other;
      }
    }
    if ((t & 31) == 0) {
      red[t >> 5] = key;
    }
    __syncthreads();
    if (t == 0) {
      std::uint64_t best = red[0];
      for (unsigned w = 1; w < 1024 / 32; ++w) {
        if (red[w] > best) {
          best = red[w];
        }
      }
      const std::int32_t id =
          1023 - static_cast<std::int32_t>(
                     static_cast<std::uint32_t>(best & 0xFFFFFFFFULL));
      ids[r] = id;
      weights[r] = p[id];
      s[id] = -FLT_MAX;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sum = 0.0F;
    for (unsigned q = 0; q < top_k; ++q) {
      sum += weights[q];
    }
    // ffn_moe_weights_sum_clamped: clamp to the smallest F16 value (the
    // reference divides by the clamped sum, then scales).
    const float div = fmaxf(sum, 6.103515625e-5F);
    for (unsigned q = 0; q < top_k; ++q) {
      weights[q] = weights[q] / div * scale;
    }
  }
}

__global__ void ScaleKernel(float* __restrict__ x, std::uint32_t total,
                            float s) {
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < total;
       i += gridDim.x * kBlock) {
    x[i] *= s;
  }
}

__global__ void SigmoidInPlaceKernel(float* __restrict__ x,
                                     std::uint32_t total) {
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < total;
       i += gridDim.x * kBlock) {
    x[i] = Sigmoid(x[i]);
  }
}

__global__ void KdaEGateKernel(const float* __restrict__ g1,
                               const float* __restrict__ a,
                               float* __restrict__ eg, std::uint32_t total,
                               std::uint32_t heads, std::uint32_t d,
                               float lower_bound) {
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < total;
       i += gridDim.x * kBlock) {
    const float v = -(g1[i] * a[(i / d) % heads]);
    eg[i] = expf(lower_bound * Sigmoid(v));
  }
}

__global__ void L2NormHeadsKernel(float* __restrict__ x, std::uint32_t d,
                                  float eps) {
  __shared__ float red[kBlock / 32];
  __shared__ float inv;
  float acc = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < d; i += kBlock) {
    const float v = x[static_cast<std::size_t>(blockIdx.x) * d + i];
    acc = fmaf(v, v, acc);
  }
  acc = BlockReduceSum(acc, red);
  if (threadIdx.x == 0) {
    inv = 1.0F / sqrtf(acc + eps);
  }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < d; i += kBlock) {
    const std::size_t at = static_cast<std::size_t>(blockIdx.x) * d + i;
    x[at] *= inv;
  }
}

__global__ void BuildSelKernel(const std::int32_t* __restrict__ pool_sel,
                               std::size_t pool_sel_stride, std::int32_t k_fixed,
                               std::uint32_t pos_first, std::uint32_t kpool,
                               std::uint32_t top_pools, std::int32_t n_tail_fix,
                               std::uint32_t use_tail,
                               std::int32_t* __restrict__ sel,
                               std::size_t sel_stride) {
  const std::uint32_t t = blockIdx.y;
  pool_sel += static_cast<std::size_t>(t) * pool_sel_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  const std::int32_t pos = static_cast<std::int32_t>(pos_first + t);
  std::int32_t k = k_fixed;
  std::int32_t n_tail = n_tail_fix;
  if (k_fixed < 0) {
    const std::uint32_t pools = (pos_first + t + 1U) / kpool;
    k = static_cast<std::int32_t>(pools < top_pools ? pools : top_pools);
    n_tail = use_tail != 0U
                 ? static_cast<std::int32_t>((pos_first + t + 1U) % kpool)
                 : 0;
  }
  const std::uint32_t total = 4U * k + n_tail;
  for (std::uint32_t i = blockIdx.x * kBlock + threadIdx.x; i < total;
       i += gridDim.x * kBlock) {
    if (i < 4U * k) {
      sel[i] = 4 * pool_sel[i / 4] + static_cast<std::int32_t>(i % 4);
    } else {
      sel[i] = pos - static_cast<std::int32_t>(n_tail - (i - 4U * k) - 1U);
    }
  }
}

}  // namespace

void AddBias(float* data, const float* bias, std::uint32_t total,
             std::uint32_t bias_len, hipStream_t stream) {
  AddBiasKernel<<<Grid1D(total), kBlock, 0, stream>>>(data, bias, total,
                                                      bias_len);
}

void RmsNorm(const float* x, const float* w, float* dst, std::uint32_t rows,
             std::uint32_t n, std::uint32_t x_stride, std::uint32_t dst_stride,
             float eps, hipStream_t stream) {
  RmsNormKernel<<<rows, kBlock, 0, stream>>>(x, w, dst, n, x_stride, dst_stride,
                                             eps);
}

void LayerNorm(const float* x, const float* w, const float* b, float* dst,
               std::uint32_t n, float eps, hipStream_t stream) {
  LayerNormKernel<<<1, kBlock, 0, stream>>>(x, w, b, dst, n, eps);
}

void HcCoeffs(const float* mixes, const float* scale, const float* base,
              std::uint32_t sinkhorn_iters, float eps, float* pre, float* post,
              float* comb, std::uint32_t rows, hipStream_t stream) {
  HcCoeffsKernel<<<dim3(1, rows), 1, 0, stream>>>(mixes, scale, base,
                                                  sinkhorn_iters, eps, pre,
                                                  post, comb);
}

void HcInputs(const float* streams, const float* pre,
              std::uint32_t pre_stride, std::uint32_t hc_count,
              std::uint32_t n, float* dst, std::uint32_t rows,
              hipStream_t stream) {
  (void)hc_count;  // contract: four streams (config hc_count)
  HcInputsKernel<<<dim3(Grid1D(n), rows), kBlock, 0, stream>>>(streams, pre,
                                                               pre_stride, n,
                                                               dst);
}

void HcWriteback(const float* y, const float* res, const float* post,
                 const float* comb, std::uint32_t hc_count, std::uint32_t n,
                 float* out, std::uint32_t rows, hipStream_t stream) {
  (void)hc_count;
  HcWritebackKernel<<<dim3(Grid1D(n), rows), kBlock, 0, stream>>>(
      y, res, post, comb, n, out);
}

void KdaConvStep(std::uint64_t pos, const float* raw, const float* w,
                 float* hist, std::uint32_t n, float* dst, hipStream_t stream) {
  KdaConvStepKernel<<<Grid1D(n), kBlock, 0, stream>>>(pos, raw, w, hist, n,
                                                      dst);
}

void KdaFoldSk(float* state, const float* eg, const float* k,
               std::uint32_t heads, std::uint32_t d, float* sk,
               hipStream_t stream) {
  // Contract: d == 128 (kda_head_dim); one thread per output column.
  dim3 grid(1, heads);
  KdaFoldSkKernel<<<grid, d, 0, stream>>>(state, eg, k, d, sk);
}

void KdaUpdateOut(float* state, const float* k, const float* q, const float* v,
                  const float* beta, const float* sk, std::uint32_t heads,
                  std::uint32_t d, float* o, hipStream_t stream) {
  dim3 grid(1, heads);
  KdaUpdateOutKernel<<<grid, d, 0, stream>>>(state, k, q, v, beta, sk, d, o);
}

void KdaOutGate(const float* o, const float* w, const float* g2, float* dst,
                std::uint32_t heads, std::uint32_t d, float eps,
                hipStream_t stream) {
  KdaOutGateKernel<<<heads, kBlock, 0, stream>>>(o, w, g2, dst, d, eps);
}

 void IdxScore(const float* pool, std::uint32_t pool_stride,
               std::uint32_t tokens, std::uint32_t kpool, std::uint32_t key_len,
               std::uint32_t heads, const float* iq, const float* ape,
               const float* w, float q_scale, float* scores,
               hipStream_t stream) {
   // Contract: kpool == 4, key_len <= 128, heads <= 32 (DSa indexer shapes).
   const std::uint32_t pools = (tokens + kpool - 1) / kpool;
   if (pools == 0) {
     return;
   }
   IdxScoreKernel<<<dim3(pools, 1U), kBlock, 0, stream>>>(
       pool, pool_stride, tokens, kpool, key_len, heads, iq, 0U, ape, w, 0U,
       q_scale, scores, 0U);
 }

 void IdxScoreBatch(const float* pool, std::uint32_t pool_stride,
                    std::uint32_t seen0, std::uint32_t kpool,
                    std::uint32_t key_len, std::uint32_t heads,
                    const float* iq, std::size_t iq_stride, const float* ape,
                    const float* w, std::size_t w_stride, float q_scale,
                    std::uint32_t tokens, float* scores,
                    std::size_t scores_stride, hipStream_t stream) {
   const std::uint32_t pools = (seen0 + tokens - 1U + kpool - 1U) / kpool;
   if (pools == 0U || tokens == 0U) {
     return;
   }
   IdxScoreKernel<<<dim3(pools, tokens), kBlock, 0, stream>>>(
       pool, pool_stride, seen0, kpool, key_len, heads, iq, iq_stride, ape, w,
       w_stride, q_scale, scores, scores_stride);
 }
void TopKDesc(const float* scores, std::uint32_t n, std::uint32_t k,
              std::int32_t* out, std::uint64_t* scratch, hipStream_t stream) {
  const std::uint32_t k2 = CeilPow2(k < 2 ? 2 : k);
  const unsigned threads = k2 <= 512 ? 2 * k2 : 1024U;
  std::uint32_t lists = (n + 2 * k2 - 1) / (2 * k2);
  // Single-list tokens (the common prefill case below 4K context: the
  // pool count never exceeds CeilPow2's 2*k2 bound) run the gather as the
  // final stage with the same k2-sized network as the multi-list path;
  // the order, keys and neutral fills are identical, only C shrinks.
  const bool single = lists <= 1U;
  if (single) {
    lists = 1U;
  }
  switch (k2) {
    case 2:
      BitonicGatherKernel<2>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 4:
      BitonicGatherKernel<4>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 8:
      BitonicGatherKernel<8>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 16:
      BitonicGatherKernel<16>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 32:
      BitonicGatherKernel<32>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 64:
      BitonicGatherKernel<64>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 128:
      BitonicGatherKernel<128>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 256:
      BitonicGatherKernel<256>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 512:
      BitonicGatherKernel<512>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    case 1024:
      BitonicGatherKernel<1024>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
    default:
      BitonicGatherKernel<2048>
          <<<lists, threads, 0, stream>>>(scores, scratch, out, n, k, single);
      break;
  }
  // Merge rounds ping-pong inside scratch: buffer B starts after the first
  // round's candidates (contract scratch >= n + 4096 uint64, and the first
  // round holds at most ceil(n / 2) keys).
  // Two fixed regions: A = candidates of round 0 (at most n/2 + k2 keys, so
  // 2 * ceil(n / 2) + padding fits the n + 4096 contract). Candidates halve
  // per round, so alternating writes never catch up with the region still
  // being read.
  std::uint64_t* src = scratch;
  std::uint64_t* dst = scratch + ((static_cast<std::size_t>(lists) * k2 + 63) &
                                  ~static_cast<std::size_t>(63));
  while (lists > 1) {
    const std::uint32_t next = (lists + 1) / 2;
    const bool final = next == 1;
    switch (k2) {
      case 2:
        BitonicMergeKernel<2>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 4:
        BitonicMergeKernel<4>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 8:
        BitonicMergeKernel<8>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 16:
        BitonicMergeKernel<16>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 32:
        BitonicMergeKernel<32>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 64:
        BitonicMergeKernel<64>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 128:
        BitonicMergeKernel<128>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 256:
        BitonicMergeKernel<256>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 512:
        BitonicMergeKernel<512>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      case 1024:
        BitonicMergeKernel<1024>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
      default:
        BitonicMergeKernel<2048>
            <<<next, threads, 0, stream>>>(src, dst, out, lists, k, final);
        break;
    }
    lists = next;
    const std::uint64_t* prev = src;
    src = dst;
    dst = const_cast<std::uint64_t*>(prev);
  }
}

void MoERoute(const float* logits, const float* bias, std::uint32_t experts,
              std::uint32_t top_k, float scale, float* weights,
              std::int32_t* ids, std::uint32_t rows, hipStream_t stream) {
  MoERouteKernel<<<dim3(1, rows), 1024, 0, stream>>>(logits, bias, experts,
                                                     top_k, scale, weights,
                                                     ids);
}

void Scale(float* x, std::uint32_t total, float s, hipStream_t stream) {
  if (total == 0) {
    return;
  }
  ScaleKernel<<<Grid1D(total), kBlock, 0, stream>>>(x, total, s);
}

void SigmoidInPlace(float* x, std::uint32_t total, hipStream_t stream) {
  if (total == 0) {
    return;
  }
  SigmoidInPlaceKernel<<<Grid1D(total), kBlock, 0, stream>>>(x, total);
}

void KdaEGate(const float* g1, const float* a, std::uint32_t heads,
              std::uint32_t d, float lower_bound, float* eg, std::uint32_t rows,
              hipStream_t stream) {
  const std::uint32_t total = rows * heads * d;  // contract: fits uint32
  KdaEGateKernel<<<Grid1D(total), kBlock, 0, stream>>>(g1, a, eg, total, heads,
                                                       d, lower_bound);
}

void KdaConvStepChunk(std::uint64_t pos0, const float* raw, const float* w,
                      float* hist, std::uint32_t n, std::uint32_t ntok,
                      float* dst, hipStream_t stream) {
  if (ntok == 0) {
    return;
  }
  KdaConvStepChunkKernel<<<Grid1D(n), kBlock, 0, stream>>>(pos0, raw, w, hist,
                                                           n, ntok, dst);
}

bool KdaFoldUpdateChunk(float* state, const float* eg, const float* k,
                        const float* q, const float* v, const float* beta,
                        std::uint32_t heads, std::uint32_t d,
                        std::uint32_t ntok, float* o, hipStream_t stream) {
  const std::size_t lds_bytes = sizeof(float) * static_cast<std::size_t>(d) * d;
  static const bool attr_ok =
      hipFuncSetAttribute(reinterpret_cast<const void*>(
                              &KdaFoldUpdateChunkKernel),
                          hipFuncAttributeMaxDynamicSharedMemorySize,
                          64 * 1024) == hipSuccess;
  if (!attr_ok) {
    return false;
  }
  KdaFoldUpdateChunkKernel<<<heads, d, lds_bytes, stream>>>(
      state, eg, k, q, v, beta, heads, d, ntok, o);
  return hipGetLastError() == hipSuccess;
}

void L2NormHeads(float* x, std::uint32_t heads, std::uint32_t d, float eps,
                 hipStream_t stream) {
  L2NormHeadsKernel<<<heads, kBlock, 0, stream>>>(x, d, eps);
}

void BuildSel(const std::int32_t* pool_sel, std::uint32_t k, std::int32_t pos,
              std::uint32_t n_tail, std::int32_t* sel, hipStream_t stream) {
  const std::uint32_t total = 4U * k + n_tail;
  if (total == 0) {
    return;
  }
  BuildSelKernel<<<dim3(Grid1D(total), 1U), kBlock, 0, stream>>>(
      pool_sel, 0U, static_cast<std::int32_t>(k), static_cast<std::uint32_t>(pos),
      4U, 0U, static_cast<std::int32_t>(n_tail), 0U, sel, 0U);
}

void BuildSelBatch(const std::int32_t* pool_sel, std::size_t pool_sel_stride,
                   std::uint32_t pos_first, std::uint32_t top_pools,
                   std::uint32_t kpool, std::uint32_t use_tail,
                   std::uint32_t tokens, std::uint32_t max_total,
                   std::int32_t* sel, std::size_t sel_stride,
                   hipStream_t stream) {
  if (tokens == 0U || max_total == 0U) {
    return;
  }
  BuildSelKernel<<<dim3(Grid1D(max_total), tokens), kBlock, 0, stream>>>(
      pool_sel, pool_sel_stride, -1, pos_first, kpool, top_pools, -1, use_tail,
      sel, sel_stride);
}

void SwigluClamp(const float* gate, const float* up, float* dst,
                 std::uint32_t m, std::uint32_t n, std::uint32_t gate_stride,
                 std::uint32_t up_stride, std::uint32_t dst_stride,
                 float clamp_gate, float clamp_up, hipStream_t stream) {
  const std::uint64_t total = static_cast<std::uint64_t>(m) * n;
  SwigluClampKernel<<<Grid1D(static_cast<std::uint32_t>(total)), kBlock, 0,
                      stream>>>(
      gate, up, dst, static_cast<std::uint32_t>(total), n, gate_stride,
      up_stride, dst_stride, clamp_gate, clamp_up);
}

void WeightedSumAdd(float* dst, const float* src, const float* weights,
                    std::uint32_t rows, std::uint32_t n, std::uint32_t stride,
                    std::uint32_t batch, hipStream_t stream) {
  WeightedSumAddKernel<<<dim3(Grid1D(n), batch), kBlock, 0, stream>>>(
      dst, src, weights, rows, n, stride);
}

}  // namespace gufo::models::glm5_next::rocm
