#include "staging/models/glm5_next/kernels/rocm/attention.hpp"

#include <rocwmma/rocwmma.hpp>

#include <cfloat>
#include <cmath>
#include <cstddef>

// Absorbed MLA decode attention over the gathered latent rows; see
// attention.hpp for the layouts. The score pass runs on F16 WMMA tiles
// over the packed absorbed queries (quality contract: the NMSE gate in
// staging/tests/models/glm5_next/rocm_attention_test.cpp); the combine
// pass sums the latent rows with per-thread column slices and j-strided
// partials, so the host double oracle in
// staging/tests/models/glm5_next/rocm_attention_test.cpp mirrors the
// weights and tolerances rather than a fixed summation order. Weights
// here are the dequantized F32 views the executor holds; the quantized
// read paths are a phase-3 tuning concern.
namespace gufo::models::glm5_next::rocm {
namespace {

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

// Block-wide max for blockDim threads (blockDim % 32 == 0); thread 0 holds
// it after the final __syncthreads.
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

__global__ void MlaQabsKernel(const float* __restrict__ q,
                              const float* __restrict__ kb, float scale,
                              float* __restrict__ q_abs) {
  __shared__ float sq[kMlaKvRank / 2];
  const std::size_t head = blockIdx.x;
  for (unsigned i = threadIdx.x; i < kMlaKvRank / 2; i += blockDim.x) {
    sq[i] = q[head * (kMlaKvRank / 2) + i];
  }
  __syncthreads();
  const std::size_t row = head * kMlaKvRank + threadIdx.x;
  const float* w = kb + row * (kMlaKvRank / 2);
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kMlaKvRank / 2; ++i) {
    acc += w[i] * sq[i];
  }
  q_abs[row] = scale * acc;
}

// One warp per selected slot: lanes load the row as two coalesced uint4
// chunks, unpack their 16 halves into registers, then loop the head slice
// pairing registers against the float4 q pairs covering the same dims and
// reduce with shuffles. Head-slice blocks keep small n_sel latency low.
__device__ unsigned DsaSelCount(std::uint32_t pos_first, std::uint32_t t,
                                std::uint32_t kpool, std::uint32_t top_pools,
                                std::uint32_t use_tail) {
  const std::uint32_t tokens = pos_first + t + 1U;
  const std::uint32_t pools = tokens / kpool;
  const std::uint32_t k = pools < top_pools ? pools : top_pools;
  return k * kpool + (use_tail != 0U ? tokens % kpool : 0U);
}

// Packs the absorbed queries for the WMMA scorer into an F16 hi/lo pair
// (q ~= hi + lo, residual below 2^-22 relative) laid out as
// [tokens][2][heads][rank]: token t writes q_pack[2*t*per + rem] (hi) and
// + per (lo) for q_abs[(t * q_stride == 0 ? per : q_stride) + rem] < per,
// where per = heads * rank (heads % 16 == 0).
__global__ void MlaPackQKernel(const float* __restrict__ q_abs,
                               std::size_t q_stride,
                               __half* __restrict__ q_pack,
                               std::uint32_t heads, std::size_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) {
    const std::size_t per = static_cast<std::size_t>(heads) * kMlaKvRank;
    const std::size_t rem = i % per;
    const std::size_t t = i / per;
    const float f =
        q_abs[t * (q_stride == 0U ? per : q_stride) + rem];
    const __half hi = __float2half_rn(f);
    q_pack[t * 2U * per + rem] = hi;
    q_pack[t * 2U * per + per + rem] =
        __float2half_rn(f - __half2float(hi));
  }
}

// Score pass on F16 WMMA tiles: a 64-thread block scores a 32-key tile,
// staging the gathered latent rows in LDS once; each wave keeps its 16
// key rows LDS-resident across all head tiles and loads the 16-head
// query fragments straight from the packed hi/lo F16 q, accumulating both
// halves into the same fp32 accumulator tile. Accumulator rows are heads
// and columns keys, so the store lands in the dense [tiles][heads][tile]
// layout with ldm = tile.
__global__ __launch_bounds__(64) void MlaGatherScoreKernel(
    const __half* __restrict__ q_p, const __half* __restrict__ cache,
    const std::int32_t* __restrict__ sel, std::size_t sel_stride,
    std::uint32_t n_sel, std::uint32_t pos_first, std::uint32_t kpool,
    std::uint32_t top_pools, std::uint32_t use_tail, std::uint32_t tile,
    std::uint32_t heads, float* __restrict__ scores,
    std::size_t scores_stride) {
  namespace wmma = rocwmma;
  constexpr std::uint32_t kKeys = 32U;
  __shared__ __half k_sh[kKeys][kMlaKvRank];
  const std::uint32_t t = blockIdx.z;
  sel += static_cast<std::size_t>(t) * sel_stride;
  scores += static_cast<std::size_t>(t) * scores_stride;
  q_p += static_cast<std::size_t>(t) * 2U * heads * kMlaKvRank;
  const __half* q_lo = q_p + static_cast<std::size_t>(heads) * kMlaKvRank;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  const std::uint32_t key_base = blockIdx.x * kKeys;
  if (key_base >= n_sel) {
    return;
  }
  for (std::uint32_t i = threadIdx.x; i < kKeys * (kMlaKvRank / 8U);
       i += blockDim.x) {
    const std::uint32_t r = i / (kMlaKvRank / 8U);
    const std::uint32_t c = i % (kMlaKvRank / 8U);
    const std::uint32_t key = key_base + r;
    *reinterpret_cast<uint4*>(&k_sh[r][c * 8U]) =
        key < n_sel
            ? reinterpret_cast<const uint4*>(
                  cache + static_cast<std::size_t>(sel[key]) * kMlaKvRank)[c]
            : make_uint4(0U, 0U, 0U, 0U);
  }
  __syncthreads();
  const std::uint32_t key0 = key_base + (threadIdx.x >> 5U) * 16U;
  for (std::uint32_t h0 = 0; h0 < heads; h0 += 16U) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0F);
    for (std::uint32_t d0 = 0; d0 < kMlaKvRank; d0 += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_lo;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(a_frag,
                             q_p + static_cast<std::size_t>(h0) * kMlaKvRank +
                                       d0,
                             kMlaKvRank);
      wmma::load_matrix_sync(a_lo,
                             q_lo + static_cast<std::size_t>(h0) * kMlaKvRank +
                                        d0,
                             kMlaKvRank);
      wmma::load_matrix_sync(b_frag, &k_sh[(threadIdx.x >> 5U) * 16U][d0],
                             kMlaKvRank);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::mma_sync(c_frag, a_lo, b_frag, c_frag);
    }
    wmma::store_matrix_sync(
        scores + (key0 / tile) * (static_cast<std::size_t>(heads) * tile) +
            static_cast<std::size_t>(h0) * tile + (key0 % tile),
        c_frag, tile, wmma::mem_row_major);
  }
}

// Head-grouped softmax combine: one 64-thread block computes four heads and
// loads each selected latent row once, feeding the rows' uint4 column slices
// into four accumulator sets with eight keys in flight. Exp values live in
// LDS interleaved per key. The per-(head, column) max/exp/psum/accumulate
// order matches the previous one-head kernel instruction for instruction
// (explicit FMAs pin contraction across unroll widths), so the output is
// bit-identical while gathered-row traffic drops 4x.
constexpr std::uint32_t kMlaCombineHeadW = 4U;
constexpr std::uint32_t kMlaCombineKeyW = 8U;

__global__ void MlaGatherCombineKernel(const float* __restrict__ scores,
                                       std::size_t scores_stride,
                                       const __half* __restrict__ cache,
                                       const std::int32_t* __restrict__ sel,
                                       std::size_t sel_stride,
                                       std::uint32_t n_sel,
                                       std::uint32_t pos_first,
                                       std::uint32_t kpool,
                                       std::uint32_t top_pools,
                                       std::uint32_t use_tail,
                                       std::uint32_t tile, std::uint32_t heads,
                                       float* __restrict__ latent,
                                       std::size_t latent_stride) {
  extern __shared__ float smem[];  // [n_sel][kMlaCombineHeadW] + reducers
  const std::uint32_t t = blockIdx.y;
  scores += static_cast<std::size_t>(t) * scores_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  latent += static_cast<std::size_t>(t) * latent_stride;
  if (n_sel == 0U) {
    n_sel = DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  }
  float* ex = smem;
  float* red = smem + n_sel * kMlaCombineHeadW;
  const std::uint32_t head0 = blockIdx.x * kMlaCombineHeadW;

  // Softmax max over all tiles is exact under any reduce order; fold the
  // per-thread strided partials with a block max per head.
  float pm[kMlaCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    pm[h] = -FLT_MAX;
  }
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
#pragma unroll
    for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
      pm[h] = fmaxf(pm[h], srow[(head0 + h) * tile]);
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    float m = BlockReduceMax(pm[h], red + h * 33U);
    if (threadIdx.x == 0) {
      red[h * 33U] = m;
    }
    __syncthreads();
  }
  float maxw[kMlaCombineHeadW];
  float ps[kMlaCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    maxw[h] = red[h * 33U];
    ps[h] = 0.0F;
  }
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
#pragma unroll
    for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
      const float e =
          expf(srow[(head0 + h) * tile] - maxw[h]);
      ex[static_cast<std::size_t>(j) * kMlaCombineHeadW + h] = e;
      ps[h] += e;
    }
  }
  __syncthreads();
  // Sum of exponentials: strided partials then a block tree sum per head.
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    const float s = WarpReduceSum(ps[h]);
    if (lane == 0) {
      red[h * 33U + (threadIdx.x >> 5)] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
      float s = 0.0F;
      for (unsigned w = 0; w < blockDim.x / 32; ++w) {
        s += red[h * 33U + w];
      }
      red[h * 33U] = s;
    }
  }
  __syncthreads();
  float inv[kMlaCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    inv[h] = 1.0F / red[h * 33U];
  }

  // Latent columns in selection order; each of the 64 threads owns an
  // 8-column slice loaded as one uint4 per selected row and shared by the
  // four heads, eight rows in flight, accumulated per head and column in
  // ascending selection order.
  float acc[kMlaCombineHeadW][8];
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
#pragma unroll
    for (std::uint32_t e = 0; e < 8U; ++e) {
      acc[h][e] = 0.0F;
    }
  }
  const std::size_t col = static_cast<std::size_t>(threadIdx.x) * 8U;
  std::uint32_t j = 0U;
  for (; j + (kMlaCombineKeyW - 1U) < n_sel; j += kMlaCombineKeyW) {
    uint4 u[kMlaCombineKeyW];
    float ew[kMlaCombineKeyW][kMlaCombineHeadW];
#pragma unroll
    for (std::uint32_t p = 0; p < kMlaCombineKeyW; ++p) {
      u[p] = *reinterpret_cast<const uint4*>(
          cache +
          static_cast<std::size_t>(sel[j + p]) * kMlaKvRank + col);
#pragma unroll
      for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
        ew[p][h] = ex[static_cast<std::size_t>(j + p) * kMlaCombineHeadW + h];
      }
    }
#pragma unroll
    for (std::uint32_t e = 0; e < 4U; ++e) {
#pragma unroll
      for (std::uint32_t p = 0; p < kMlaCombineKeyW; ++p) {
        const float2 f = __half22float2(
            reinterpret_cast<const __half2*>(&u[p])[e]);
#pragma unroll
        for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
          acc[h][2U * e] = __fmaf_rn(ew[p][h], f.x, acc[h][2U * e]);
          acc[h][2U * e + 1U] =
              __fmaf_rn(ew[p][h], f.y, acc[h][2U * e + 1U]);
        }
      }
    }
  }
  for (; j < n_sel; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(
        cache + static_cast<std::size_t>(sel[j]) * kMlaKvRank + col);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (std::uint32_t e = 0; e < 4U; ++e) {
      const float2 f = __half22float2(h[e]);
#pragma unroll
      for (std::uint32_t hh = 0; hh < kMlaCombineHeadW; ++hh) {
        const float w = ex[static_cast<std::size_t>(j) * kMlaCombineHeadW + hh];
        acc[hh][2U * e] = __fmaf_rn(w, f.x, acc[hh][2U * e]);
        acc[hh][2U * e + 1U] = __fmaf_rn(w, f.y, acc[hh][2U * e + 1U]);
      }
    }
  }
#pragma unroll
  for (std::uint32_t h = 0; h < kMlaCombineHeadW; ++h) {
    float* out =
        latent + static_cast<std::size_t>(head0 + h) * kMlaKvRank + col;
#pragma unroll
    for (std::uint32_t e = 0; e < 8U; ++e) {
      out[e] = acc[h][e] * inv[h];
    }
  }
}

__global__ void MlaApplyVbKernel(const float* __restrict__ latent,
                                 const float* __restrict__ vb,
                                 float* __restrict__ out) {
  __shared__ float sl[kMlaKvRank];
  const std::size_t head = blockIdx.x;
  for (unsigned i = threadIdx.x; i < kMlaKvRank; i += blockDim.x) {
    sl[i] = latent[head * kMlaKvRank + i];
  }
  __syncthreads();
  const std::size_t row = head * (kMlaKvRank / 2) + threadIdx.x;
  const float* w = vb + row * kMlaKvRank;
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kMlaKvRank; ++i) {
    acc += w[i] * sl[i];
  }
  out[row] = acc;
}

// Raw RDNA3 WMMA plumbing (fragment = 16 halves lane-contiguous, C[i] at
// row lane % 16, column 2 * i + lane / 16); the load helper mirrors the
// proven pattern in src/models/minimax_h3/attention.hip.
using Fragment = __attribute__((ext_vector_type(16))) unsigned short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

// Fused single-pass prefill attention over the selected latent rows;
// replaces the score+combine pair on one launch. One block per query
// token (256 threads = 8 waves); wave pairs (group = wave >> 1 owns 16
// heads, half = wave & 1) split the 512-dim scoring dot into latent-dim
// halves merged through LDS, then both waves replay the merged chunk
// softmax (base-2 running max/sum) and the P x V WMMA tiles over the
// group's 512 output channels, split 256 per half. K tiles and
// transposed V tiles share one staging union; probabilities are F16.
// Supports heads in {16, 32, 48, 64} (s_part groups of 16).
__global__ __launch_bounds__(256, 1) void MlaFusedAttendKernel(
    const __half* __restrict__ q_pack, const __half* __restrict__ cache,
    const std::int32_t* __restrict__ sel, std::size_t sel_stride,
    std::uint32_t pos_first, std::uint32_t kpool, std::uint32_t top_pools,
    std::uint32_t use_tail, std::uint32_t heads, float* __restrict__ latent,
    std::size_t latent_stride) {
  constexpr int kKeysTile = 32;
  constexpr int kRowStride = static_cast<int>(kMlaKvRank) + 8;
  constexpr int kVtStride = 40;
  union Stage {
    __half key[kKeysTile][kRowStride];
    __half value[kMlaKvRank][kVtStride];
  };
  __shared__ Stage st;
  __shared__ float s_part[4][2][16][34];  // [group][half][head][key]
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int group = wave >> 1;
  const int half = wave & 1;
  const std::size_t per = static_cast<std::size_t>(heads) * kMlaKvRank;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t n =
      DsaSelCount(pos_first, t, kpool, top_pools, use_tail);
  sel += static_cast<std::size_t>(t) * sel_stride;
  float* out_tok = latent + static_cast<std::size_t>(t) * latent_stride;
  const int head_row = static_cast<int>(group * 16U + lane % 16U);
  const std::uint16_t* q_hi = reinterpret_cast<const std::uint16_t*>(
      q_pack + static_cast<std::size_t>(t) * 2U * per +
      static_cast<std::size_t>(head_row) * kMlaKvRank);
  const std::uint16_t* q_lo = q_hi + per;

  Accumulator result[16];
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    result[d] = {};
  }
  float maximum = -INFINITY;
  float sum = 0.0F;

  for (std::uint32_t base = 0; base < n; base += kKeysTile) {
    // Stage the gathered K rows (zero-filled tails).
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int i = tid + z * 256;
      const int r = i / static_cast<int>(kMlaKvRank / 8U);
      const int c = i % static_cast<int>(kMlaKvRank / 8U);
      const std::uint32_t j = base + static_cast<std::uint32_t>(r);
      const uint4 v = j < n ? *reinterpret_cast<const uint4*>(
                                  cache + static_cast<std::size_t>(sel[j]) *
                                              kMlaKvRank +
                                  static_cast<std::size_t>(c) * 8U)
                            : make_uint4(0U, 0U, 0U, 0U);
      *reinterpret_cast<uint4*>(&st.key[r][c * 8]) = v;
    }
    __syncthreads();

    // Partial chunk scores over this half of the latent dim.
    Accumulator scores[2];
    scores[0] = {};
    scores[1] = {};
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      const int d0 = half * 256 + d * 16;
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) {
        const Fragment k = LoadFragment(
            reinterpret_cast<const std::uint16_t*>(&st.key[ct * 16 + lane % 16]
                                                       [0]) +
            d0);
        const Fragment q = LoadFragment(q_hi + d0);
        scores[ct] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, q, scores[ct]);
        const Fragment ql = LoadFragment(q_lo + d0);
        scores[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, ql,
                                                                scores[ct]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }

    // Publish the partial dots for the dim-half merge.
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int key = ct * 16 + 2 * i + lane / 16;
        s_part[group][half][lane % 16][key] = scores[ct][i];
      }
    }
    __syncthreads();  // K dead below; partial scores visible

    // Stage the transposed V tile (uint4 row reads, scalar LDS writes).
#pragma unroll
    for (int z = 0; z < 8; ++z) {
      const int i = tid + z * 256;
      const int r = i / static_cast<int>(kMlaKvRank / 8U);
      const int c = (i % static_cast<int>(kMlaKvRank / 8U)) * 8;
      const std::uint32_t j = base + static_cast<std::uint32_t>(r);
      union {
        uint4 packed;
        __half h[8];
      } v;
      if (j < n) {
        v.packed = *reinterpret_cast<const uint4*>(
            cache + static_cast<std::size_t>(sel[j]) * kMlaKvRank + c);
      } else {
        v.packed = make_uint4(0U, 0U, 0U, 0U);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        st.value[c + e][r] = v.h[e];
      }
    }
    __syncthreads();  // VT ready

    // Merge the dim halves, mask the tail, run the chunk softmax.
    float s[2][8];
    float next = maximum;
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int key = ct * 16 + 2 * i + lane / 16;
        const std::uint32_t jkey = base + static_cast<std::uint32_t>(key);
        float v = s_part[group][0][lane % 16][key] +
                  s_part[group][1][lane % 16][key];
        if (jkey >= n) {
          v = -INFINITY;
        }
        // log2(e); the softmax scale is already folded into q.
        s[ct][i] = v * 1.4426950408889634F;
        next = fmaxf(next, s[ct][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));
    const float correction = __builtin_amdgcn_exp2f(maximum - next);
    float subtotal = 0.0F;
    Fragment probability[2];
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(s[ct][i] - next);
        subtotal += p;
        const unsigned own = __half_as_ushort(__float2half_rn(p));
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability[ct][2 * i] = lane < 16 ? own : peer;
        probability[ct][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum = sum * correction + subtotal;
    maximum = next;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      result[d] *= correction;
    }
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      const Fragment p = probability[ct];
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        const Fragment v = LoadFragment(
            reinterpret_cast<const std::uint16_t*>(
                &st.value[half * 256 + d * 16 + lane % 16][0]) +
            ct * 16);
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v, p, result[d]);
        if ((d & 1) == 1) {
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
    __syncthreads();  // VT consumed before the next chunk re-stages K
  }

  const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
  for (int d = 0; d < 16; ++d) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out_tok[static_cast<std::size_t>(head_row) * kMlaKvRank + half * 256 +
              d * 16 + 2 * i + lane / 16] = result[d][i] * reciprocal;
    }
  }
}

}  // namespace

void MlaQabs(const float* q, const float* kb, std::uint32_t heads, float scale,
             float* q_abs, hipStream_t stream) {
  MlaQabsKernel<<<heads, kMlaKvRank, 0, stream>>>(q, kb, scale, q_abs);
}

std::uint32_t MlaGatherTiles(std::uint32_t n_sel, std::uint32_t tile) {
  return (n_sel + tile - 1) / tile;
}

void MlaGatherScore(const float* q_abs, __half* q_pack, const __half* cache,
                    const std::int32_t* sel, std::uint32_t n_sel,
                    std::uint32_t tile, std::uint32_t heads, float* scores,
                    hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(heads) * kMlaKvRank;
  MlaPackQKernel<<<static_cast<std::uint32_t>((total + 255U) / 256U), 256, 0U,
                   stream>>>(q_abs, 0U, q_pack, heads, total);
  const dim3 grid((n_sel + 31U) / 32U, 1U, 1U);
  MlaGatherScoreKernel<<<grid, 64, 0, stream>>>(q_pack, cache, sel, 0U, n_sel,
                                                0U, 0U, 0U, 0U, tile, heads,
                                                scores, 0U);
}

void MlaGatherScoreBatch(const float* q_abs, std::size_t q_stride,
                         __half* q_pack, const __half* cache,
                         const std::int32_t* sel, std::size_t sel_stride,
                         std::uint32_t n_sel_max, std::uint32_t pos_first,
                         std::uint32_t kpool, std::uint32_t top_pools,
                         std::uint32_t use_tail, std::uint32_t tokens,
                         std::uint32_t tile, std::uint32_t heads, float* scores,
                         std::size_t scores_stride, hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(tokens) * heads * kMlaKvRank;
  MlaPackQKernel<<<static_cast<std::uint32_t>((total + 255U) / 256U), 256, 0U,
                   stream>>>(q_abs, q_stride, q_pack, heads, total);
  const dim3 grid((n_sel_max + 31U) / 32U, 1U, tokens);
  MlaGatherScoreKernel<<<grid, 64, 0, stream>>>(
      q_pack, cache, sel, sel_stride, 0U, pos_first, kpool, top_pools,
      use_tail, tile, heads, scores, scores_stride);
}

void MlaGatherCombine(const float* scores, const __half* cache,
                      const std::int32_t* sel, std::uint32_t tile,
                      std::uint32_t n_sel, std::uint32_t heads, float* latent,
                      hipStream_t stream) {
  const std::size_t smem =
      (n_sel + 33U) * kMlaCombineHeadW * sizeof(float);
  MlaGatherCombineKernel<<<dim3(heads / kMlaCombineHeadW, 1U), kMlaKvRank / 8U,
                           smem, stream>>>(
      scores, 0U, cache, sel, 0U, n_sel, 0U, 1U, 0U, 0U, tile, heads, latent,
      0U);
}

void MlaGatherCombineBatch(const float* scores, std::size_t scores_stride,
                           const __half* cache, const std::int32_t* sel,
                           std::size_t sel_stride, std::uint32_t n_sel_max,
                           std::uint32_t pos_first, std::uint32_t kpool,
                           std::uint32_t top_pools, std::uint32_t use_tail,
                           std::uint32_t tokens, std::uint32_t tile,
                           std::uint32_t heads, float* latent,
                           std::size_t latent_stride, hipStream_t stream) {
  const std::size_t smem =
      (n_sel_max + 33U) * kMlaCombineHeadW * sizeof(float);
  MlaGatherCombineKernel<<<dim3(heads / kMlaCombineHeadW, tokens),
                           kMlaKvRank / 8U, smem, stream>>>(
      scores, scores_stride, cache, sel, sel_stride, 0U, pos_first, kpool,
      top_pools, use_tail, tile, heads, latent, latent_stride);
}

void MlaFusedAttend(const float* q_abs, std::size_t q_stride,
                    __half* q_pack, const __half* cache,
                    const std::int32_t* sel, std::size_t sel_stride,
                    std::uint32_t pos_first, std::uint32_t kpool,
                    std::uint32_t top_pools, std::uint32_t use_tail,
                    std::uint32_t tokens, std::uint32_t heads, float* latent,
                    std::size_t latent_stride, hipStream_t stream) {
  const std::size_t total =
      static_cast<std::size_t>(tokens) * heads * kMlaKvRank;
  MlaPackQKernel<<<static_cast<std::uint32_t>((total + 255U) / 256U), 256, 0U,
                   stream>>>(q_abs, q_stride, q_pack, heads, total);
  MlaFusedAttendKernel<<<tokens, heads * 4U, 0U, stream>>>(
      q_pack, cache, sel, sel_stride, pos_first, kpool, top_pools, use_tail,
      heads, latent, latent_stride);
}

void MlaApplyVb(const float* latent, const float* vb, std::uint32_t heads,
                float* out, hipStream_t stream) {
  MlaApplyVbKernel<<<heads, kMlaKvRank / 2, 0, stream>>>(latent, vb, out);
}

}  // namespace gufo::models::glm5_next::rocm
