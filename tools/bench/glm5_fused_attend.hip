// GLM-5.3-Flash B8 microbench: single-pass fused DSA attention.
// Baseline arms are copies of MlaPackQ/MlaGatherScore/MlaGatherCombine from
// staging/models/glm5_next/kernels/rocm/attention.hip.cpp (per-token n_sel
// comes from an array instead of DsaSelCount; arithmetic unchanged).
// The fused arm ports the WMMA fragment plumbing of
// src/models/minimax_h3/attention.hip (Short-path softmax ordering adapted
// from Composable Kernel, ROCm 7.2.3, MIT; no CK headers are required).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(expr)                                                     \
  do {                                                                  \
    hipError_t e_ = (expr);                                             \
    if (e_ != hipSuccess) {                                             \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),     \
                  __FILE__, __LINE__);                                  \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kDim = 512U;
constexpr std::size_t kPer = std::size_t(kHeads) * kDim;

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

__device__ __forceinline__ float WarpReduceMax(float v) {
  for (int o = 16; o > 0; o >>= 1) {
    v = fmaxf(v, __shfl_down(v, o));
  }
  return v;
}

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_down(v, o);
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

// ---------------------------------------------------------------------------
// Baseline copies (production kernels; n_sel==0 now reads nsel[t]).
// ---------------------------------------------------------------------------

__global__ void BasePack(const float* __restrict__ q_abs,
                         __half* __restrict__ q_pack, std::uint32_t total) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < total) {
    const std::size_t rem = i % kPer;
    const std::size_t t = i / kPer;
    const float f = q_abs[i];
    const __half hi = __float2half_rn(f);
    q_pack[t * 2U * kPer + rem] = hi;
    q_pack[(t * 2U + 1U) * kPer + rem] = __float2half_rn(f - __half2float(hi));
  }
}

__global__ __launch_bounds__(64) void BaseScore(
    const __half* __restrict__ q_p, const __half* __restrict__ cache,
    const std::int32_t* __restrict__ sel, std::size_t sel_stride,
    const float* __restrict__ nsel, std::uint32_t tile, std::uint32_t heads,
    float* __restrict__ scores, std::size_t scores_stride) {
  namespace wmma = rocwmma;
  constexpr std::uint32_t kKeysLocal = 32U;
  __shared__ __half k_sh[kKeysLocal][kDim];
  const std::uint32_t t = blockIdx.z;
  const std::uint32_t n_sel = static_cast<std::uint32_t>(nsel[t]);
  sel += static_cast<std::size_t>(t) * sel_stride;
  scores += static_cast<std::size_t>(t) * scores_stride;
  q_p += static_cast<std::size_t>(t) * 2U * heads * kDim;
  const __half* q_lo = q_p + static_cast<std::size_t>(heads) * kDim;
  const std::uint32_t key_base = blockIdx.x * kKeysLocal;
  if (key_base >= n_sel) {
    return;
  }
  for (std::uint32_t i = threadIdx.x; i < kKeysLocal * (kDim / 8U);
       i += blockDim.x) {
    const std::uint32_t r = i / (kDim / 8U);
    const std::uint32_t c = i % (kDim / 8U);
    const std::uint32_t key = key_base + r;
    *reinterpret_cast<uint4*>(&k_sh[r][c * 8U]) =
        key < n_sel
            ? reinterpret_cast<const uint4*>(
                  cache + static_cast<std::size_t>(sel[key]) * kDim)[c]
            : make_uint4(0U, 0U, 0U, 0U);
  }
  __syncthreads();
  const std::uint32_t key0 = key_base + (threadIdx.x >> 5U) * 16U;
  for (std::uint32_t h0 = 0; h0 < heads; h0 += 16U) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0F);
    for (std::uint32_t d0 = 0; d0 < kDim; d0 += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
          a_lo;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(a_frag,
                             q_p + static_cast<std::size_t>(h0) * kDim + d0,
                             kDim);
      wmma::load_matrix_sync(a_lo,
                             q_lo + static_cast<std::size_t>(h0) * kDim + d0,
                             kDim);
      wmma::load_matrix_sync(b_frag, &k_sh[(threadIdx.x >> 5U) * 16U][d0],
                             kDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::mma_sync(c_frag, a_lo, b_frag, c_frag);
    }
    wmma::store_matrix_sync(
        scores + (key0 / tile) * (static_cast<std::size_t>(heads) * tile) +
            static_cast<std::size_t>(h0) * tile + (key0 % tile),
        c_frag, tile, wmma::mem_row_major);
  }
}

constexpr std::uint32_t kCombineHeadW = 4U;
constexpr std::uint32_t kCombineKeyW = 8U;

__global__ void BaseCombine(const float* __restrict__ scores,
                            std::size_t scores_stride,
                            const __half* __restrict__ cache,
                            const std::int32_t* __restrict__ sel,
                            std::size_t sel_stride,
                            const float* __restrict__ nsel,
                            std::uint32_t tile, std::uint32_t heads,
                            float* __restrict__ latent,
                            std::size_t latent_stride) {
  extern __shared__ float smem[];
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t n_sel = static_cast<std::uint32_t>(nsel[t]);
  scores += static_cast<std::size_t>(t) * scores_stride;
  sel += static_cast<std::size_t>(t) * sel_stride;
  latent += static_cast<std::size_t>(t) * latent_stride;
  float* ex = smem;
  float* red = smem + n_sel * kCombineHeadW;
  const std::uint32_t head0 = blockIdx.x * kCombineHeadW;

  float pm[kCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    pm[h] = -FLT_MAX;
  }
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
#pragma unroll
    for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
      pm[h] = fmaxf(pm[h], srow[(head0 + h) * tile]);
    }
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    float m = BlockReduceMax(pm[h], red + h * 33U);
    if (threadIdx.x == 0) {
      red[h * 33U] = m;
    }
    __syncthreads();
  }
  float maxw[kCombineHeadW];
  float ps[kCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    maxw[h] = red[h * 33U];
    ps[h] = 0.0F;
  }
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    const float* srow =
        scores + static_cast<std::size_t>(j / tile) * (heads * tile) +
        (j % tile);
#pragma unroll
    for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
      const float e = expf(srow[(head0 + h) * tile] - maxw[h]);
      ex[static_cast<std::size_t>(j) * kCombineHeadW + h] = e;
      ps[h] += e;
    }
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    const float s = WarpReduceSum(ps[h]);
    if (lane == 0) {
      red[h * 33U + (threadIdx.x >> 5)] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
      float s = 0.0F;
      for (unsigned w = 0; w < blockDim.x / 32; ++w) {
        s += red[h * 33U + w];
      }
      red[h * 33U] = s;
    }
  }
  __syncthreads();
  float inv[kCombineHeadW];
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    inv[h] = 1.0F / red[h * 33U];
  }

  float acc[kCombineHeadW][8];
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
#pragma unroll
    for (std::uint32_t e = 0; e < 8U; ++e) {
      acc[h][e] = 0.0F;
    }
  }
  const std::size_t col = static_cast<std::size_t>(threadIdx.x) * 8U;
  std::uint32_t j = 0U;
  for (; j + (kCombineKeyW - 1U) < n_sel; j += kCombineKeyW) {
    uint4 u[kCombineKeyW];
    float ew[kCombineKeyW][kCombineHeadW];
#pragma unroll
    for (std::uint32_t p = 0; p < kCombineKeyW; ++p) {
      u[p] = *reinterpret_cast<const uint4*>(
          cache + static_cast<std::size_t>(sel[j + p]) * kDim + col);
#pragma unroll
      for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
        ew[p][h] = ex[static_cast<std::size_t>(j + p) * kCombineHeadW + h];
      }
    }
#pragma unroll
    for (std::uint32_t e = 0; e < 4U; ++e) {
#pragma unroll
      for (std::uint32_t p = 0; p < kCombineKeyW; ++p) {
        const float2 f =
            __half22float2(reinterpret_cast<const __half2*>(&u[p])[e]);
#pragma unroll
        for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
          acc[h][2U * e] = __fmaf_rn(ew[p][h], f.x, acc[h][2U * e]);
          acc[h][2U * e + 1U] =
              __fmaf_rn(ew[p][h], f.y, acc[h][2U * e + 1U]);
        }
      }
    }
  }
  for (; j < n_sel; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(
        cache + static_cast<std::size_t>(sel[j]) * kDim + col);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (std::uint32_t e = 0; e < 4U; ++e) {
      const float2 f = __half22float2(h[e]);
#pragma unroll
      for (std::uint32_t hh = 0; hh < kCombineHeadW; ++hh) {
        const float w = ex[static_cast<std::size_t>(j) * kCombineHeadW + hh];
        acc[hh][2U * e] = __fmaf_rn(w, f.x, acc[hh][2U * e]);
        acc[hh][2U * e + 1U] = __fmaf_rn(w, f.y, acc[hh][2U * e + 1U]);
      }
    }
  }
#pragma unroll
  for (std::uint32_t h = 0; h < kCombineHeadW; ++h) {
    float* out = latent + static_cast<std::size_t>(head0 + h) * kDim + col;
#pragma unroll
    for (std::uint32_t e = 0; e < 8U; ++e) {
      out[e] = acc[h][e] * inv[h];
    }
  }
}

// ---------------------------------------------------------------------------
// Fused single-pass attention.
//
// One block per token, 256 threads / 8 waves. Wave pairs (group = wave >> 1,
// half = wave & 1) own the 16-head group; scores accumulate one latent dim
// half per wave and exchange partial dots through LDS, then both waves of a
// pair replay the merged chunk softmax (base-2 running max/sum) and the
// P x V WMMA tiles over the group's 512 channels, split 256 per half.
// Scores and channel accumulation are FP32; probabilities are F16.
// K tiles and transposed V tiles share one staging union. The FASTVT path
// stages V from a transposed shadow cache with dense selected rows.
// ---------------------------------------------------------------------------

template<bool HILO, bool FASTVT>
__global__ __launch_bounds__(256, 1) void FusedAttend(
    const __half* __restrict__ q_pack, const __half* __restrict__ cache,
    const __half* __restrict__ cache_t, const std::int32_t* __restrict__ sel,
    std::size_t sel_stride, std::size_t cache_t_rows,
    const float* __restrict__ nsel, float* __restrict__ latent) {
  constexpr int kKeysTile = 32;
  constexpr int kRowStride = static_cast<int>(kDim) + 8;
  constexpr int kVtStride = 40;
  union Stage {
    __half key[kKeysTile][kRowStride];
    __half value[kDim][kVtStride];
  };
  __shared__ Stage st;
  __shared__ float s_part[4][2][16][34];  // [group][half][head][key]
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int group = wave >> 1;
  const int half = wave & 1;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t n = static_cast<std::uint32_t>(nsel[t]);
  sel += static_cast<std::size_t>(t) * sel_stride;
  float* out_tok = latent + static_cast<std::size_t>(t) * kPer;
  const int head_row = static_cast<int>(group * 16U + lane % 16U);
  const std::uint16_t* q_hi = reinterpret_cast<const std::uint16_t*>(
      q_pack + static_cast<std::size_t>(t) * 2U * kPer +
      static_cast<std::size_t>(head_row) * kDim);
  const std::uint16_t* q_lo = q_hi + kPer;

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
      const int r = i / static_cast<int>(kDim / 8U);
      const int c = i % static_cast<int>(kDim / 8U);
      const std::uint32_t j = base + static_cast<std::uint32_t>(r);
      const uint4 v =
          j < n ? *reinterpret_cast<const uint4*>(
                      cache + static_cast<std::size_t>(sel[j]) * kDim +
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
        if (HILO) {
          const Fragment ql = LoadFragment(q_lo + d0);
          scores[ct] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k, ql, scores[ct]);
        }
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

    // Stage the transposed V tile.
    if (FASTVT) {
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int i = tid + z * 256;
        const int c = i >> 2;
        const int kv = i & 3;
        *reinterpret_cast<uint4*>(&st.value[c][kv * 8]) =
            *reinterpret_cast<const uint4*>(
                cache_t + static_cast<std::size_t>(c) * cache_t_rows +
                static_cast<std::size_t>(sel[base]) + kv * 8U);
      }
    } else {
      // Transposed gather: uint4 row reads, scalar LDS writes.
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int i = tid + z * 256;
        const int r = i / static_cast<int>(kDim / 8U);
        const int c = (i % static_cast<int>(kDim / 8U)) * 8;
        const std::uint32_t j = base + static_cast<std::uint32_t>(r);
        union {
          uint4 packed;
          __half h[8];
        } v;
        if (j < n) {
          v.packed = *reinterpret_cast<const uint4*>(
              cache + static_cast<std::size_t>(sel[j]) * kDim + c);
        } else {
          v.packed = make_uint4(0U, 0U, 0U, 0U);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          st.value[c + e][r] = v.h[e];
        }
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
      out_tok[static_cast<std::size_t>(head_row) * kDim + half * 256 +
              d * 16 + 2 * i + lane / 16] = result[d][i] * reciprocal;
    }
  }
}

// ---------------------------------------------------------------------------
// Host harness.
// ---------------------------------------------------------------------------

template<typename R>
double Nmse(const std::vector<float>& a, const std::vector<R>& b) {
  double num = 0.0;
  double den = 0.0;
  double maxabs = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
    maxabs = std::max(maxabs, std::fabs(d));
  }
  std::printf(" (maxabs %.3e)", maxabs);
  return num / (den + 1e-300);
}

struct Shape {
  std::uint32_t tokens;
  std::uint32_t pos_first;   // causal offset
  std::uint32_t nsel_cap;    // selection cap
  std::uint32_t sparse_sel;  // strided random subset at the cap
};

void RunShape(const Shape& s, std::mt19937& rng) {
  const std::size_t cap_rows =
      static_cast<std::size_t>(s.pos_first + s.tokens + 8U);
  std::printf("== tokens=%u pos_first=%u cap=%u sparse=%u (cache rows %zu)\n",
              s.tokens, s.pos_first, s.nsel_cap, s.sparse_sel, cap_rows);

  std::vector<float> q_abs(std::size_t(s.tokens) * kPer);
  std::vector<float> cache_f(cap_rows * kDim);
  {
    std::normal_distribution<float> g(0.0F, 0.6F);
    for (auto& x : q_abs) x = g(rng);
    for (auto& x : cache_f) x = g(rng);
  }
  std::vector<std::int32_t> sel(std::size_t(s.tokens) * cap_rows);
  std::vector<float> nsel(s.tokens);
  {
    std::uniform_int_distribution<std::int32_t> pick(0, (1 << 30) - 1);
    for (std::uint32_t t = 0; t < s.tokens; ++t) {
      const std::uint32_t avail = s.pos_first + t + 1U;
      const std::uint32_t n = avail < s.nsel_cap ? avail : s.nsel_cap;
      nsel[t] = static_cast<float>(n);
      std::vector<std::int32_t> chosen;
      chosen.reserve(n);
      if (avail > n && s.sparse_sel) {
        // n distinct rows spread over [off, off + (avail - n)], off < n, so
        // max index < avail.
        const double step = static_cast<double>(avail - n) / (n - 1);
        const std::int32_t off = pick(rng) % static_cast<std::int32_t>(n);
        double acc = static_cast<double>(off);
        for (std::uint32_t j = 0; j < n; ++j) {
          chosen.push_back(static_cast<std::int32_t>(acc));
          acc += step;
        }
        std::sort(chosen.begin(), chosen.end());
      } else {
        for (std::uint32_t j = 0; j < n; ++j) {
          chosen.push_back(static_cast<std::int32_t>(j));
        }
      }
      for (std::uint32_t j = 0; j < n; ++j) {
        sel[std::size_t(t) * cap_rows + j] = chosen[j];
      }
    }
  }

  const std::uint32_t tile = 32U;
  const std::uint32_t tiles_max = (s.nsel_cap + tile - 1U) / tile;
  float* d_q_abs;
  __half *d_cache, *d_cache_t, *d_q_pack;
  std::int32_t* d_sel;
  float *d_nsel, *d_lat_base, *d_lat_fused, *d_scores;
  CHECK(hipMalloc(&d_q_abs, q_abs.size() * sizeof(float)));
  CHECK(hipMalloc(&d_cache, cache_f.size() * sizeof(__half)));
  CHECK(hipMalloc(&d_cache_t, cap_rows * kDim * sizeof(__half)));
  CHECK(hipMalloc(&d_q_pack, 2.0 * q_abs.size() * sizeof(__half)));
  CHECK(hipMalloc(&d_sel, sel.size() * sizeof(std::int32_t)));
  CHECK(hipMalloc(&d_nsel, nsel.size() * sizeof(float)));
  CHECK(hipMalloc(&d_lat_base, std::size_t(s.tokens) * kPer * sizeof(float)));
  CHECK(hipMalloc(&d_lat_fused, std::size_t(s.tokens) * kPer * sizeof(float)));
  CHECK(hipMalloc(
      &d_scores, std::size_t(s.tokens) * tiles_max * kHeads * tile *
                     sizeof(float)));
  CHECK(hipMemcpy(d_q_abs, q_abs.data(), q_abs.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  std::vector<__half> cache_h(cache_f.size());
  for (std::size_t i = 0; i < cache_f.size(); ++i) {
    cache_h[i] = __float2half_rn(cache_f[i]);
  }
  CHECK(hipMemcpy(d_cache, cache_h.data(), cache_h.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  std::vector<__half> vt(cap_rows * kDim);
  for (std::size_t r = 0; r < cap_rows; ++r) {
    for (std::size_t c = 0; c < kDim; ++c) {
      vt[c * cap_rows + r] = cache_h[r * kDim + c];
    }
  }
  CHECK(hipMemcpy(d_cache_t, vt.data(), vt.size() * sizeof(__half),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_sel, sel.data(), sel.size() * sizeof(std::int32_t),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_nsel, nsel.data(), nsel.size() * sizeof(float),
                  hipMemcpyHostToDevice));

  const std::size_t total = std::size_t(s.tokens) * kPer;
  const std::size_t scores_stride = std::size_t(tiles_max) * kHeads * tile;

  auto pack = [&]() {
    BasePack<<<static_cast<std::uint32_t>((total + 255U) / 256U), 256>>>(
        d_q_abs, d_q_pack, static_cast<std::uint32_t>(total));
  };
  auto baseline = [&]() {
    BaseScore<<<dim3(tiles_max, 1U, s.tokens), 64>>>(
        d_q_pack, d_cache, d_sel, cap_rows, d_nsel, tile, kHeads, d_scores,
        scores_stride);
    BaseCombine<<<dim3(kHeads / kCombineHeadW, s.tokens), 64,
                  (s.nsel_cap * kCombineHeadW + 256U) * sizeof(float)>>>(
        d_scores, scores_stride, d_cache, d_sel, cap_rows, d_nsel, tile,
        kHeads, d_lat_base, kPer);
  };
  auto fused = [&](auto hilo, auto fastvt) {
    if (hilo) {
      if (fastvt) {
        FusedAttend<true, true><<<s.tokens, 256>>>(
            d_q_pack, d_cache, d_cache_t, d_sel, cap_rows, cap_rows, d_nsel,
            d_lat_fused);
      } else {
        FusedAttend<true, false><<<s.tokens, 256>>>(
            d_q_pack, d_cache, nullptr, d_sel, cap_rows, cap_rows, d_nsel,
            d_lat_fused);
      }
    } else {
      if (fastvt) {
        FusedAttend<false, true><<<s.tokens, 256>>>(
            d_q_pack, d_cache, d_cache_t, d_sel, cap_rows, cap_rows, d_nsel,
            d_lat_fused);
      } else {
        FusedAttend<false, false><<<s.tokens, 256>>>(
            d_q_pack, d_cache, nullptr, d_sel, cap_rows, cap_rows, d_nsel,
            d_lat_fused);
      }
    }
  };

  auto timek = [&](const char* name, auto fn) -> float {
    for (int i = 0; i < 2; ++i) fn();
    CHECK(hipDeviceSynchronize());
    CHECK(hipGetLastError());
    hipEvent_t a, b;
    hipEventCreate(&a);
    hipEventCreate(&b);
    float best = 1e9F;
    for (int w = 0; w < 4; ++w) {
      hipEventRecord(a);
      for (int i = 0; i < 5; ++i) fn();
      hipEventRecord(b);
      CHECK(hipEventSynchronize(b));
      float ms;
      hipEventElapsedTime(&ms, a, b);
      best = std::min(best, ms / 5.0F);
    }
    hipEventDestroy(a);
    hipEventDestroy(b);
    std::printf("  %-26s %8.3f ms\n", name, best);
    CHECK(hipGetLastError());
    return best;
  };

  // One correctness + NMSE pass.
  pack();
  baseline();
  fused(std::true_type{}, std::false_type{});
  CHECK(hipDeviceSynchronize());
  CHECK(hipGetLastError());

  std::vector<float> lat_base(std::size_t(s.tokens) * kPer);
  std::vector<float> lat_fused(std::size_t(s.tokens) * kPer);
  CHECK(hipMemcpy(lat_base.data(), d_lat_base,
                  lat_base.size() * sizeof(float), hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(lat_fused.data(), d_lat_fused,
                  lat_fused.size() * sizeof(float), hipMemcpyDeviceToHost));

  {
    const std::uint32_t t_or = s.tokens <= 32U ? s.tokens : 4U;
    std::vector<double> oracle(std::size_t(t_or) * kPer, 0.0);
    for (std::uint32_t t = 0; t < t_or; ++t) {
      const std::uint32_t n = static_cast<std::uint32_t>(nsel[t]);
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        std::vector<double> sc(n);
        double mx = -1e300;
        for (std::uint32_t j = 0; j < n; ++j) {
          const float* q = &q_abs[std::size_t(t) * kPer + h * kDim];
          double acc = 0.0;
          const __half* row =
              &cache_h[std::size_t(sel[std::size_t(t) * cap_rows + j]) * kDim];
          for (std::uint32_t d = 0; d < kDim; ++d) {
            acc += static_cast<double>(q[d]) * static_cast<double>(row[d]);
          }
          sc[j] = acc;
          mx = std::max(mx, sc[j]);
        }
        double sumv = 0.0;
        std::vector<double> p(n);
        for (std::uint32_t j = 0; j < n; ++j) {
          p[j] = std::exp(sc[j] - mx) ;
          sumv += p[j];
        }
        for (std::uint32_t d = 0; d < kDim; ++d) {
          double acc = 0.0;
          for (std::uint32_t j = 0; j < n; ++j) {
            const __half* row = &cache_h[std::size_t(
                sel[std::size_t(t) * cap_rows + j]) * kDim];
            acc += p[j] / sumv * static_cast<double>(row[d]);
          }
          oracle[std::size_t(t) * kPer + h * kDim + d] = acc;
        }
      }
    }
    std::vector<float> base_w(std::size_t(t_or) * kPer),
        fused_w(std::size_t(t_or) * kPer);
    for (std::size_t i = 0; i < oracle.size(); ++i) {
      base_w[i] = lat_base[i];
      fused_w[i] = lat_fused[i];
    }
    std::printf("  NMSE oracle baseline: %.3e\n", Nmse(base_w, oracle));
    std::printf("  NMSE oracle fused   : %.3e\n", Nmse(fused_w, oracle));
    std::printf("  NMSE fused  baseline: %.3e\n", Nmse(fused_w, base_w));
  }

  // Timings.
  const float t_base = timek("baseline score+combine", baseline);
  const float t_pack = timek("pack only", pack);
  const float t_f_hilo = timek("fused hilo (VT generic)",
                               [&]() { fused(std::true_type{}, std::false_type{}); });
  const float t_f_hi = timek("fused hi (VT generic)",
                             [&]() { fused(std::false_type{}, std::false_type{}); });
  float t_f_fast = 0.0F;
  float t_pack_base_pair = t_base;
  if (s.sparse_sel == 0U) {
    t_f_fast = timek("fused hilo (VT fast)",
                     [&]() { fused(std::true_type{}, std::true_type{}); });
    t_pack_base_pair = t_base + t_pack;
  }
  std::printf("  ratio (base+pack)/fused: hilo=%.3f hi=%.3f fast=%.3f\n",
              t_pack_base_pair / t_f_hilo, (t_base + t_pack) / t_f_hi,
              t_f_fast > 0.0F ? t_pack_base_pair / t_f_fast : 0.0F);

  CHECK(hipFree(d_q_abs));
  CHECK(hipFree(d_cache));
  CHECK(hipFree(d_cache_t));
  CHECK(hipFree(d_q_pack));
  CHECK(hipFree(d_sel));
  CHECK(hipFree(d_nsel));
  CHECK(hipFree(d_lat_base));
  CHECK(hipFree(d_lat_fused));
  CHECK(hipFree(d_scores));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  CHECK(hipSetDevice(0));
  const int skip = argc > 1 ? std::atoi(argv[1]) : 0;
  std::mt19937 rng(12345U);
  if (skip <= 0) RunShape({24U, 0U, 2048U, 0U}, rng);      // correctness window
  if (skip <= 1) RunShape({2048U, 0U, 2048U, 0U}, rng);    // pp2048 dense causal
  RunShape({1024U, 61440U, 512U, 1U}, rng);                // sparse cap
  return 0;
}
