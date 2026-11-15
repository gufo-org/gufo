// GLM-5.3-Flash DSA MLA gather microbench: score and combine variants.
// Baselines are copies of MlaGatherScoreKernel/MlaGatherCombineKernel from
// staging/models/glm5_next/kernels/rocm/attention.hip.cpp. A4 variants:
// S1 = WMMA F16 tiles over packed q with gathered key rows in LDS,
// C1 = 64-thread rank-slice combine with uint4 loads and 4-way j unroll.
// C2 = head-grouped combine (width 2/4) sharing each gathered row load across
// w heads; required bit-exact against C1.
// Small shapes report NMSE of each variant against the baseline output;
// the large shape is the pp2048 batched call (tokens=2048, heads=64,
// rank=512, tile=64, kpool=4, top_pools=512).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr unsigned kRank = 512;
constexpr unsigned kHeads = 64;
constexpr unsigned kTile = 64;

#define CUDA_CHECK(x)                                                    \
  do {                                                                   \
    const hipError_t err__ = (x);                                        \
    if (err__ != hipSuccess) {                                           \
      std::fprintf(stderr, "hip error %s at %s:%d\n",                    \
                   hipGetErrorString(err__), __FILE__, __LINE__);        \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

__host__ __device__ unsigned DsaSelCount(unsigned pos_first, unsigned t,
                                         unsigned kpool, unsigned top_pools,
                                         unsigned use_tail) {
  const unsigned tokens = pos_first + t + 1U;
  const unsigned pools = tokens / kpool;
  const unsigned k = pools < top_pools ? pools : top_pools;
  return k * kpool + (use_tail != 0U ? tokens % kpool : 0U);
}

__device__ __forceinline__ float WarpReduceMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down(v, offset));
  }
  return v;
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

__device__ __forceinline__ float BlockReduceMax(float v, float* red) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  v = WarpReduceMax(v);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = (lane < warps) ? red[lane] : -FLT_MAX;
    v = WarpReduceMax(v);
  }
  return v;
}

// ---------------------------------------------------------------- baseline
__global__ __launch_bounds__(256) void ScoreBaseKernel(
    const float* __restrict__ q_abs, std::size_t q_stride,
    const __half* __restrict__ cache, const std::int32_t* __restrict__ sel,
    std::size_t sel_stride, unsigned n_sel, unsigned pos_first, unsigned kpool,
    unsigned top_pools, unsigned use_tail, unsigned tile, unsigned heads,
    unsigned hspan, float* __restrict__ scores, std::size_t scores_stride) {
  const unsigned t = blockIdx.z;
  sel += static_cast<std::size_t>(t) * sel_stride;
  scores += static_cast<std::size_t>(t) * scores_stride;
  q_abs += static_cast<std::size_t>(t) * q_stride;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  const unsigned lane = threadIdx.x & 31U;
  const unsigned key = blockIdx.x * 8U + (threadIdx.x >> 5U);
  if (key >= n_sel) {
    return;
  }
  const uint4* rp = reinterpret_cast<const uint4*>(
      cache + static_cast<std::size_t>(sel[key]) * kRank);
  const uint4 r0 = rp[lane];
  const uint4 r1 = rp[lane + 32U];
  float v[16];
  const __half2* p0 = reinterpret_cast<const __half2*>(&r0);
  const __half2* p1 = reinterpret_cast<const __half2*>(&r1);
#pragma unroll
  for (unsigned k = 0; k < 4U; ++k) {
    const float2 f0 = __half22float2(p0[k]);
    const float2 f1 = __half22float2(p1[k]);
    v[2U * k] = f0.x;
    v[2U * k + 1U] = f0.y;
    v[8U + 2U * k] = f1.x;
    v[8U + 2U * k + 1U] = f1.y;
  }
  const unsigned head0 = blockIdx.y * hspan;
  for (unsigned h = 0; h < hspan; ++h) {
    const unsigned head = head0 + h;
    const float4* qa = reinterpret_cast<const float4*>(
        q_abs + static_cast<std::size_t>(head) * kRank);
    const float4 c0 = qa[2U * lane];
    const float4 c1 = qa[2U * lane + 1U];
    const float4 c2 = qa[kRank / 8U + 2U * lane];
    const float4 c3 = qa[kRank / 8U + 2U * lane + 1U];
    float acc = 0.0F;
    acc = __fmaf_rn(v[0], c0.x, acc);
    acc = __fmaf_rn(v[1], c0.y, acc);
    acc = __fmaf_rn(v[2], c0.z, acc);
    acc = __fmaf_rn(v[3], c0.w, acc);
    acc = __fmaf_rn(v[4], c1.x, acc);
    acc = __fmaf_rn(v[5], c1.y, acc);
    acc = __fmaf_rn(v[6], c1.z, acc);
    acc = __fmaf_rn(v[7], c1.w, acc);
    acc = __fmaf_rn(v[8], c2.x, acc);
    acc = __fmaf_rn(v[9], c2.y, acc);
    acc = __fmaf_rn(v[10], c2.z, acc);
    acc = __fmaf_rn(v[11], c2.w, acc);
    acc = __fmaf_rn(v[12], c3.x, acc);
    acc = __fmaf_rn(v[13], c3.y, acc);
    acc = __fmaf_rn(v[14], c3.z, acc);
    acc = __fmaf_rn(v[15], c3.w, acc);
    acc += __shfl_down_sync(0xffffffffULL, acc, 16U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 8U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 4U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 2U);
    acc += __shfl_down_sync(0xffffffffULL, acc, 1U);
    if (lane == 0U) {
      scores[(static_cast<std::size_t>(key) / tile) *
                 (static_cast<std::size_t>(heads) * tile) +
             static_cast<std::size_t>(head) * tile + (key % tile)] = acc;
    }
  }
}

__global__ void CombineBaseKernel(const float* __restrict__ scores,
                                  std::size_t scores_stride,
                                  const __half* __restrict__ cache,
                                  const std::int32_t* __restrict__ sel,
                                  std::size_t sel_stride, unsigned n_sel,
                                  unsigned pos_first, unsigned kpool,
                                  unsigned top_pools, unsigned use_tail,
                                  unsigned tile, unsigned heads,
                                  float* __restrict__ latent,
                                  std::size_t latent_stride) {
  extern __shared__ float smem[];
  const unsigned t = blockIdx.y;
  scores += static_cast<std::size_t>(t) * scores_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  latent += static_cast<std::size_t>(t) * latent_stride;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  float* ex = smem;
  float* red = smem + n_sel;
  const unsigned head = blockIdx.x;
  const std::size_t score_row = static_cast<std::size_t>(head) * tile;

  float pm = -FLT_MAX;
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    pm = fmaxf(pm, scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                          score_row + (j % tile)]);
  }
  __syncthreads();
  pm = BlockReduceMax(pm, red);
  if (threadIdx.x == 0) {
    red[0] = pm;
  }
  __syncthreads();
  const float max_s = red[0];

  float psum = 0.0F;
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    ex[j] = expf(scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                        score_row + (j % tile)] -
                 max_s);
    psum += ex[j];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  psum = WarpReduceSum(psum);
  if (lane == 0) {
    red[threadIdx.x >> 5] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0F;
    for (unsigned w = 0; w < blockDim.x / 32; ++w) {
      s += red[w];
    }
    red[0] = s;
  }
  __syncthreads();
  const float inv = 1.0F / red[0];

  float acc = 0.0F;
  for (unsigned j = 0; j < n_sel; ++j) {
    const __half* row = cache + static_cast<std::size_t>(sel[j]) * kRank;
    acc += ex[j] * __half2float(row[threadIdx.x]);
  }
  latent[static_cast<std::size_t>(head) * kRank + threadIdx.x] = acc * inv;
}

// ------------------------------------------------------------------- S1
// Packs q into an F16 hi/lo pair: q ~= hi + lo with residual below 2^-22
// relative, so the two WMMA dots recover the fp32 scoring tier. Layout is
// [tokens][2][heads][rank]: hi at +0, lo at +per for each token.
__global__ void PackQKernel(const float* __restrict__ q_abs,
                            __half* __restrict__ q_p, std::size_t per,
                            std::size_t total) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i < total) {
    const std::size_t t = i / per;
    const std::size_t rem = i % per;
    const __half hi = __float2half_rn(q_abs[i]);
    q_p[t * 2U * per + rem] = hi;
    q_p[t * 2U * per + per + rem] =
        __float2half_rn(q_abs[i] - __half2float(hi));
  }
}

// WMMA F16 scoring over 32-key tiles: two waves gather 32 gathered cache
// rows into LDS, then wave w scores its 16 keys against 16-head fragments
// loaded directly from the packed q hi and lo halves into one fp32
// accumulator. acc rows = heads, cols = keys, so the tile store matches
// the scores layout with ldm = tile.
__global__ __launch_bounds__(64) void ScoreWmmaKernel(
    const __half* __restrict__ q_p, const __half* __restrict__ cache,
    const std::int32_t* __restrict__ sel, std::size_t sel_stride,
    unsigned n_sel, unsigned pos_first, unsigned kpool, unsigned top_pools,
    unsigned use_tail, float* __restrict__ scores, std::size_t scores_stride) {
  namespace wmma = rocwmma;
  constexpr unsigned kKeysPerBlock = 32U;
  __shared__ __half k_sh[kKeysPerBlock][kRank];
  const unsigned t = blockIdx.y;
  sel += static_cast<std::size_t>(t) * sel_stride;
  scores += static_cast<std::size_t>(t) * scores_stride;
  q_p += static_cast<std::size_t>(t) * 2U * kHeads * kRank;
  const __half* q_lo = q_p + kHeads * kRank;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  const unsigned key_base = blockIdx.x * kKeysPerBlock;
  if (key_base >= n_sel) {
    return;
  }
  // Gather rows coalesced: 32 rows x 128 uint4 over 64 threads.
  for (unsigned i = threadIdx.x; i < kKeysPerBlock * (kRank / 8U);
       i += blockDim.x) {
    const unsigned r = i / (kRank / 8U);
    const unsigned c = i % (kRank / 8U);
    const unsigned key = key_base + r;
    *reinterpret_cast<uint4*>(&k_sh[r][c * 8U]) =
        key < n_sel
            ? reinterpret_cast<const uint4*>(
                  cache + static_cast<std::size_t>(sel[key]) * kRank)[c]
            : make_uint4(0U, 0U, 0U, 0U);
  }
  __syncthreads();
  const unsigned lane = threadIdx.x & 31U;
  const unsigned warp = threadIdx.x >> 5U;
  const unsigned key0 = key_base + warp * 16U;
  for (unsigned h0 = 0; h0 < kHeads; h0 += 16U) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0F);
#pragma unroll 4
    for (unsigned d0 = 0; d0 < kRank; d0 += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_lo;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, q_p + static_cast<std::size_t>(h0) * kRank +
                                          d0,
                             kRank);
      wmma::load_matrix_sync(a_lo, q_lo + static_cast<std::size_t>(h0) * kRank +
                                        d0,
                             kRank);
      wmma::load_matrix_sync(b_frag, &k_sh[warp * 16U][d0], kRank);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::mma_sync(c_frag, a_lo, b_frag, c_frag);
    }
    wmma::store_matrix_sync(
        scores + (static_cast<std::size_t>(key0 / kTile) *
                  (static_cast<std::size_t>(kHeads) * kTile)) +
            static_cast<std::size_t>(h0) * kTile + (key0 % kTile),
        c_frag, kTile, wmma::mem_row_major);
  }
}

// ------------------------------------------------------------------- C1
template <unsigned kKeyW>
__global__ void CombineVecKernel(const float* __restrict__ scores,
                                 std::size_t scores_stride,
                                 const __half* __restrict__ cache,
                                 const std::int32_t* __restrict__ sel,
                                 std::size_t sel_stride, unsigned n_sel,
                                 unsigned pos_first, unsigned kpool,
                                 unsigned top_pools, unsigned use_tail,
                                 unsigned tile, unsigned heads,
                                 float* __restrict__ latent,
                                 std::size_t latent_stride) {
  extern __shared__ float smem[];
  const unsigned t = blockIdx.y;
  scores += static_cast<std::size_t>(t) * scores_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  latent += static_cast<std::size_t>(t) * latent_stride;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  float* ex = smem;
  float* red = smem + n_sel;
  const unsigned head = blockIdx.x;
  const std::size_t score_row = static_cast<std::size_t>(head) * tile;

  float pm = -FLT_MAX;
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    pm = fmaxf(pm, scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                          score_row + (j % tile)]);
  }
  __syncthreads();
  pm = BlockReduceMax(pm, red);
  if (threadIdx.x == 0) {
    red[0] = pm;
  }
  __syncthreads();
  const float max_s = red[0];

  float psum = 0.0F;
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    ex[j] = expf(scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                        score_row + (j % tile)] -
                 max_s);
    psum += ex[j];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  psum = WarpReduceSum(psum);
  if (lane == 0) {
    red[threadIdx.x >> 5] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0F;
    for (unsigned w = 0; w < blockDim.x / 32; ++w) {
      s += red[w];
    }
    red[0] = s;
  }
  __syncthreads();
  const float inv = 1.0F / red[0];

  // Each of 64 threads owns 8 consecutive rank columns as one uint4
  // (8 halves per selected row); kKeyW keys are loaded back-to-back to
  // deepen L2 latency hiding, accumulated per column in ascending j.
  float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const std::size_t col = static_cast<std::size_t>(threadIdx.x) * 8U;
  unsigned j = 0U;
  for (; j + (kKeyW - 1U) < n_sel; j += kKeyW) {
    uint4 u[kKeyW];
    float w[kKeyW];
#pragma unroll
    for (unsigned p = 0; p < kKeyW; ++p) {
      u[p] = *reinterpret_cast<const uint4*>(
          cache + static_cast<std::size_t>(sel[j + p]) * kRank + col);
    }
#pragma unroll
    for (unsigned p = 0; p < kKeyW; ++p) {
      w[p] = ex[j + p];
    }
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e) {
#pragma unroll
      for (unsigned p = 0; p < kKeyW; ++p) {
        const float2 f = __half22float2(
            reinterpret_cast<const __half2*>(&u[p])[e]);
        // Explicit FMA pins the instruction across unroll widths so any
        // kKeyW reproduces the kKeyW=4 reference bits exactly.
        acc[2U * e] = __fmaf_rn(w[p], f.x, acc[2U * e]);
        acc[2U * e + 1U] = __fmaf_rn(w[p], f.y, acc[2U * e + 1U]);
      }
    }
  }
  for (; j < n_sel; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(
        cache + static_cast<std::size_t>(sel[j]) * kRank + col);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
    const float w = ex[j];
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e) {
      const float2 f = __half22float2(h[e]);
      acc[2U * e] = __fmaf_rn(w, f.x, acc[2U * e]);
      acc[2U * e + 1U] = __fmaf_rn(w, f.y, acc[2U * e + 1U]);
    }
  }
  float* out = latent + static_cast<std::size_t>(head) * kRank + col;
#pragma unroll
  for (unsigned e = 0; e < 8U; ++e) {
    out[e] = acc[e] * inv;
  }
}

// ------------------------------------------------------------------- C2
// Width-kHeadW head-grouped combine: the 64-thread block computes w heads,
// loads each selected row's uint4 column slice once and feeds w independent
// accumulator sets, so row traffic and LSU issue drop by w. Exp values are
// staged in LDS as w interleaved components (one float4 load serves all w
// heads per key). Per-(head, column) max/exp/psum/accumulate order is
// identical to C1, so the result must be bit-exact against the vec kernel.
template <unsigned kHeadW, unsigned kKeyW = 4U>
__global__ void CombineQuadKernel(const float* __restrict__ scores,
                                  std::size_t scores_stride,
                                  const __half* __restrict__ cache,
                                  const std::int32_t* __restrict__ sel,
                                  std::size_t sel_stride, unsigned n_sel,
                                  unsigned pos_first, unsigned kpool,
                                  unsigned top_pools, unsigned use_tail,
                                  unsigned tile, unsigned heads,
                                  float* __restrict__ latent,
                                  std::size_t latent_stride) {
  extern __shared__ float smem[];
  const unsigned t = blockIdx.y;
  scores += static_cast<std::size_t>(t) * scores_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  latent += static_cast<std::size_t>(t) * latent_stride;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  float* ex = smem;  // [n_sel][kHeadW] components
  float* red = smem + static_cast<std::size_t>(n_sel) * kHeadW;
  const unsigned head0 = blockIdx.x * kHeadW;

  float pm[kHeadW];
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    pm[h] = -FLT_MAX;
  }
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
#pragma unroll
    for (unsigned h = 0; h < kHeadW; ++h) {
      pm[h] = fmaxf(pm[h], srow[(head0 + h) * tile]);
    }
  }
  __syncthreads();
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    float m = BlockReduceMax(pm[h], red + h * 33U);
    if (threadIdx.x == 0) {
      red[h * 33U] = m;
    }
    __syncthreads();
  }
  float maxw[kHeadW];
  float ps[kHeadW];
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    maxw[h] = red[h * 33U];
    ps[h] = 0.0F;
  }
  for (unsigned j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
    float e[kHeadW];
#pragma unroll
    for (unsigned h = 0; h < kHeadW; ++h) {
      e[h] = expf(srow[(head0 + h) * tile] - maxw[h]);
      ps[h] += e[h];
      ex[static_cast<std::size_t>(j) * kHeadW + h] = e[h];
    }
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    const float s = WarpReduceSum(ps[h]);
    if (lane == 0) {
      red[h * 33U + (threadIdx.x >> 5)] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (unsigned h = 0; h < kHeadW; ++h) {
      float s = 0.0F;
      for (unsigned w = 0; w < blockDim.x / 32; ++w) {
        s += red[h * 33U + w];
      }
      red[h * 33U] = s;
    }
  }
  __syncthreads();
  float inv[kHeadW];
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    inv[h] = 1.0F / red[h * 33U];
  }

  float acc[kHeadW][8];
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
#pragma unroll
    for (unsigned e = 0; e < 8U; ++e) {
      acc[h][e] = 0.0F;
    }
  }
  const std::size_t col = static_cast<std::size_t>(threadIdx.x) * 8U;
  unsigned j = 0U;
  for (; j + (kKeyW - 1U) < n_sel; j += kKeyW) {
    uint4 u[kKeyW];
    float ew[kKeyW][kHeadW];
#pragma unroll
    for (unsigned p = 0; p < kKeyW; ++p) {
      u[p] = *reinterpret_cast<const uint4*>(
          cache + static_cast<std::size_t>(sel[j + p]) * kRank + col);
#pragma unroll
      for (unsigned h = 0; h < kHeadW; ++h) {
        ew[p][h] = ex[static_cast<std::size_t>(j + p) * kHeadW + h];
      }
    }
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e) {
#pragma unroll
      for (unsigned p = 0; p < kKeyW; ++p) {
        const float2 f = __half22float2(
            reinterpret_cast<const __half2*>(&u[p])[e]);
#pragma unroll
        for (unsigned h = 0; h < kHeadW; ++h) {
          acc[h][2U * e] = __fmaf_rn(ew[p][h], f.x, acc[h][2U * e]);
          acc[h][2U * e + 1U] = __fmaf_rn(ew[p][h], f.y, acc[h][2U * e + 1U]);
        }
      }
    }
  }
  for (; j < n_sel; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(
        cache + static_cast<std::size_t>(sel[j]) * kRank + col);
    const __half2* hd = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e) {
      const float2 f = __half22float2(hd[e]);
#pragma unroll
      for (unsigned h = 0; h < kHeadW; ++h) {
        const float w = ex[static_cast<std::size_t>(j) * kHeadW + h];
        acc[h][2U * e] = __fmaf_rn(w, f.x, acc[h][2U * e]);
        acc[h][2U * e + 1U] = __fmaf_rn(w, f.y, acc[h][2U * e + 1U]);
      }
    }
  }
#pragma unroll
  for (unsigned h = 0; h < kHeadW; ++h) {
    float* out = latent + static_cast<std::size_t>(head0 + h) * kRank + col;
#pragma unroll
    for (unsigned e = 0; e < 8U; ++e) {
      out[e] = acc[h][e] * inv[h];
    }
  }
}

// ---------------------------------------------------------------- harness
std::mt19937& Rng() {
  static std::mt19937 rng(12345U);
  return rng;
}

double Nmse(const std::vector<float>& a, const std::vector<float>& b,
            const std::vector<char>& valid) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!valid[i]) {
      continue;
    }
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(a[i]) * a[i];
  }
  return num / (den + 1e-30);
}

double WorstRel(const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<char>& valid, double floor_eps) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!valid[i]) {
      continue;
    }
    const double d =
        std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    const double denom =
        std::max(std::abs(static_cast<double>(a[i])), floor_eps);
    if (d / denom > worst) {
      worst = d / denom;
    }
  }
  return worst;
}

float TimeKernel(const std::function<void()>& launch, int warm, int reps) {
  for (int i = 0; i < warm; ++i) {
    launch();
  }
  hipEvent_t e0;
  hipEvent_t e1;
  CUDA_CHECK(hipEventCreate(&e0));
  CUDA_CHECK(hipEventCreate(&e1));
  CUDA_CHECK(hipEventRecord(e0));
  for (int i = 0; i < reps; ++i) {
    launch();
  }
  CUDA_CHECK(hipEventRecord(e1));
  CUDA_CHECK(hipEventSynchronize(e1));
  float ms = 0.0F;
  CUDA_CHECK(hipEventElapsedTime(&ms, e0, e1));
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return ms * 1000.0F / static_cast<float>(reps);
}

void RunCase(unsigned tokens, std::uint64_t pos_first_dummy, bool time_large) {
  const unsigned kpool = 4U;
  const unsigned top_pools = 512U;
  const unsigned use_tail = 1U;
  const std::size_t sel_capacity = 4ULL * top_pools + kpool - 1ULL;
  const unsigned n_sel_max =
      (std::min(tokens / kpool, top_pools) * kpool +
       (tokens % kpool));  // use_tail = 1
  const std::size_t tiles_cap = (sel_capacity + kTile - 1U) / kTile;
  const std::size_t scores_stride = tiles_cap * kHeads * kTile;
  const std::size_t q_stride = static_cast<std::size_t>(kHeads) * kRank;

  std::printf("== tokens=%u n_sel_max=%u (score tiles grid %u x %u)\n",
              tokens, n_sel_max, (n_sel_max + 7U) / 8U, tokens);

  std::vector<float> h_q(static_cast<std::size_t>(tokens) * q_stride);
  std::vector<__half> h_cache(static_cast<std::size_t>(tokens) * kRank);
  std::vector<std::int32_t> h_sel(static_cast<std::size_t>(tokens) *
                                  sel_capacity);
  auto& rng = Rng();
  std::normal_distribution<float> n(0.0F, 1.0F);
  for (auto& x : h_q) {
    x = n(rng);
  }
  for (auto& x : h_cache) {
    x = __float2half_rn(n(rng));
  }
  for (unsigned t = 0; t < tokens; ++t) {
    const unsigned ns = DsaSelCount(0U, t, kpool, top_pools, use_tail);
    (void)ns;
    for (std::size_t j = 0; j < sel_capacity; ++j) {
      h_sel[static_cast<std::size_t>(t) * sel_capacity + j] =
          static_cast<std::int32_t>(j % tokens);
    }
  }

  float* d_q;
  __half* d_cache;
  std::int32_t* d_sel;
  float* d_scores_a;
  float* d_scores_b;
  __half* d_qp;
  float* d_lat_a;
  float* d_lat_b;
  float* d_lat_c;
  CUDA_CHECK(hipMalloc(&d_q, h_q.size() * sizeof(float)));
  CUDA_CHECK(hipMalloc(&d_cache, h_cache.size() * sizeof(__half)));
  CUDA_CHECK(hipMalloc(&d_sel, h_sel.size() * sizeof(std::int32_t)));
  CUDA_CHECK(hipMalloc(&d_scores_a,
                       static_cast<std::size_t>(tokens) * scores_stride *
                           sizeof(float)));
  CUDA_CHECK(hipMalloc(&d_scores_b,
                       static_cast<std::size_t>(tokens) * scores_stride *
                           sizeof(float)));
  CUDA_CHECK(hipMalloc(&d_qp, 2ULL * static_cast<std::size_t>(tokens) *
                                   q_stride * sizeof(__half)));
  CUDA_CHECK(hipMalloc(
      &d_lat_a, static_cast<std::size_t>(tokens) * q_stride * sizeof(float)));
  CUDA_CHECK(hipMalloc(
      &d_lat_b, static_cast<std::size_t>(tokens) * q_stride * sizeof(float)));
  CUDA_CHECK(hipMalloc(
      &d_lat_c, static_cast<std::size_t>(tokens) * q_stride * sizeof(float)));
  CUDA_CHECK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                       hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d_cache, h_cache.data(),
                       h_cache.size() * sizeof(__half),
                       hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d_sel, h_sel.data(),
                       h_sel.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice));

  const std::size_t smem = (sel_capacity + 32) * sizeof(float);

  auto launch_score_base = [&]() {
    const unsigned hspan = n_sel_max >= 4096U ? 32U : 16U;
    ScoreBaseKernel<<<dim3((n_sel_max + 7U) / 8U, kHeads / hspan, tokens),
                      256>>>(d_q, q_stride, d_cache, d_sel, sel_capacity, 0U,
                             0U, kpool, top_pools, use_tail, kTile, kHeads,
                             hspan, d_scores_a, scores_stride);
  };
  auto launch_score_wmma = [&]() {
    PackQKernel<<<(static_cast<unsigned>(h_q.size()) + 255U) / 256U, 256>>>(
        d_q, d_qp, q_stride, h_q.size());
    ScoreWmmaKernel<<<dim3((n_sel_max + 31U) / 32U, tokens), 64>>>(
        d_qp, d_cache, d_sel, sel_capacity, 0U, 0U, kpool, top_pools, use_tail,
        d_scores_b, scores_stride);
  };
  auto launch_combine_base = [&]() {
    CombineBaseKernel<<<dim3(kHeads, tokens), kRank, smem>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_a, q_stride);
  };
  auto launch_combine_vec = [&]() {
    CombineVecKernel<4U><<<dim3(kHeads, tokens), 64, smem>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_b, q_stride);
  };
  auto launch_combine_vec8 = [&]() {
    CombineVecKernel<8U><<<dim3(kHeads, tokens), 64, smem>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_c, q_stride);
  };
  const std::size_t smem_q2 =
      (sel_capacity + 32) * 2 * sizeof(float) + 32U * 33U * sizeof(float);
  const std::size_t smem_q4 =
      (sel_capacity + 32) * 4 * sizeof(float) + 32U * 33U * sizeof(float);
  auto launch_combine_q2 = [&]() {
    CombineQuadKernel<2U><<<dim3(kHeads / 2U, tokens), 64, smem_q2>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_c, q_stride);
  };
  auto launch_combine_q4 = [&]() {
    CombineQuadKernel<4U><<<dim3(kHeads / 4U, tokens), 64, smem_q4>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_c, q_stride);
  };
  auto launch_combine_q48 = [&]() {
    CombineQuadKernel<4U, 8U><<<dim3(kHeads / 4U, tokens), 64, smem_q4>>>(
        d_scores_a, scores_stride, d_cache, d_sel, sel_capacity, 0U, 0U, kpool,
        top_pools, use_tail, kTile, kHeads, d_lat_c, q_stride);
  };

  // Validity mask over the score tile layout for j < n_sel(t).
  std::vector<float> a_scores(static_cast<std::size_t>(tokens) * scores_stride);
  std::vector<float> b_scores(a_scores.size());
  std::vector<char> v_scores(a_scores.size(), 0);

  if (time_large) {
    const float t_sb = TimeKernel(launch_score_base, 2, 10);
    const float t_sw = TimeKernel(launch_score_wmma, 2, 10);
    const float t_cb = TimeKernel(launch_combine_base, 2, 10);
    const float t_cv = TimeKernel(launch_combine_vec, 2, 10);
    const float t_cv8 = TimeKernel(launch_combine_vec8, 2, 10);
    const float t_c2 = TimeKernel(launch_combine_q2, 2, 10);
    const float t_c4 = TimeKernel(launch_combine_q4, 2, 10);
    const float t_c48 = TimeKernel(launch_combine_q48, 2, 10);
    double keys = 0.0;
    for (unsigned t = 0; t < tokens; ++t) {
      keys += DsaSelCount(0U, t, kpool, top_pools, use_tail);
    }
    const double gflop = keys * kHeads * kRank * 2.0 / 1e9;
    std::printf(
        "  score base %8.1f us  wmma %8.1f us (%.2fx) ~%.1f TF/s wmma\n", t_sb,
        t_sw, t_sb / t_sw, gflop / (t_sw * 1e-6) );
    std::printf("  combine base %8.1f us  vec %8.1f us (%.2fx)\n", t_cb, t_cv,
                t_cb / t_cv);
    std::printf("  combine quad2 %8.1f us (%.2fx vs vec)  quad4 %8.1f us "
                "(%.2fx vs vec)\n",
                t_c2, t_cv / t_c2, t_c4, t_cv / t_c4);
    std::printf("  combine vec8 %8.1f us (%.2fx vs vec)  quad4x8 %8.1f us "
                "(%.2fx vs vec)\n",
                t_cv8, t_cv / t_cv8, t_c48, t_cv / t_c48);
  }

  // Correctness pass on a small token count with per-token n_sel mask.
  launch_score_base();
  launch_score_wmma();
  CUDA_CHECK(hipDeviceSynchronize());
  CUDA_CHECK(hipMemcpy(a_scores.data(), d_scores_a,
                       a_scores.size() * sizeof(float), hipMemcpyDeviceToHost));
  CUDA_CHECK(hipMemcpy(b_scores.data(), d_scores_b,
                       b_scores.size() * sizeof(float), hipMemcpyDeviceToHost));
  const unsigned check_tokens = time_large ? 256U : tokens;
  std::vector<char> vcheck(a_scores.size(), 0);
  for (unsigned t = 0; t < check_tokens; ++t) {
    const unsigned ns = DsaSelCount(0U, t, kpool, top_pools, use_tail);
    for (unsigned j = 0; j < ns; ++j) {
      for (unsigned h = 0; h < kHeads; ++h) {
        vcheck[static_cast<std::size_t>(t) * scores_stride +
               (j / kTile) * (kHeads * kTile) + h * kTile + (j % kTile)] = 1;
      }
    }
  }
  std::printf("  score NMSE(wmma vs base, %u tokens) = %.3e  worstrel = %.3e\n",
              check_tokens, Nmse(a_scores, b_scores, vcheck),
              WorstRel(a_scores, b_scores, vcheck, 1e-3));

  launch_combine_base();
  launch_combine_vec();
  CUDA_CHECK(hipDeviceSynchronize());
  std::vector<float> a_lat(static_cast<std::size_t>(tokens) * q_stride, 0.0F);
  std::vector<float> b_lat(a_lat.size());
  CUDA_CHECK(hipMemcpy(a_lat.data(), d_lat_a, a_lat.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
  CUDA_CHECK(hipMemcpy(b_lat.data(), d_lat_b, b_lat.size() * sizeof(float),
                       hipMemcpyDeviceToHost));
  std::vector<char> v_lat(a_lat.size(), 1);
  std::printf("  combine NMSE(vec vs base) = %.3e\n",
              Nmse(a_lat, b_lat, v_lat));

  auto bit_diff = [](const std::vector<float>& ref,
                     const std::vector<float>& got) {
    std::size_t n = 0;
    std::size_t first = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      std::uint32_t rb = 0;
      std::uint32_t gb = 0;
      std::memcpy(&rb, &ref[i], sizeof(rb));
      std::memcpy(&gb, &got[i], sizeof(gb));
      if (rb != gb) {
        if (n == 0) {
          first = i;
        }
        ++n;
      }
    }
    return std::pair<std::size_t, std::size_t>(n, first);
  };
  std::vector<float> c_lat(a_lat.size(), 0.0F);
  auto check_quad = [&](const char* name,
                        const std::function<void()>& launch) {
    launch();
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemcpy(c_lat.data(), d_lat_c,
                         c_lat.size() * sizeof(float), hipMemcpyDeviceToHost));
    const auto d = bit_diff(b_lat, c_lat);
    std::printf("  combine %s vs vec bit-diff = %zu%s (first idx %zu, t=%zu)\n",
                name, d.first, d.first == 0 ? " (bit-exact)" : "", d.second,
                d.second / q_stride);
  };
  check_quad("vec8  ", launch_combine_vec8);
  check_quad("quad2", launch_combine_q2);
  check_quad("quad4", launch_combine_q4);
  check_quad("quad4x8", launch_combine_q48);

  (void)v_scores;
  hipFree(d_q);
  hipFree(d_cache);
  hipFree(d_sel);
  hipFree(d_scores_a);
  hipFree(d_scores_b);
  hipFree(d_qp);
  hipFree(d_lat_a);
  hipFree(d_lat_b);
  hipFree(d_lat_c);
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  RunCase(37U, 0ULL, false);
  RunCase(2048U, 0ULL, true);
  return 0;
}
