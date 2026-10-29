#ifndef GUFO_MODELS_LFM3_HIP_Q8_PREFILL_HPP_
#define GUFO_MODELS_LFM3_HIP_Q8_PREFILL_HPP_
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
namespace gufo::models::lfm3::gpu {
// Prompt activations are split into FP16 high parts and FP16 remainders,
// [token][K] each, after scaling every row by a power of two so its maximum
// lies in [2^14, 2^15). The pair keeps about 22 significant bits of every
// element, not only of the row maximum. Row scales follow both planes.
inline std::size_t SplitActivationBytes(std::size_t tokens, std::size_t K) {
  return tokens * K * 2 * sizeof(__half) + tokens * sizeof(float);
}
struct SplitActivations {
  __half *high, *low;
  float* scale;
  static SplitActivations In(void* base, std::size_t tokens, std::size_t K) {
    auto* half = static_cast<__half*>(base);
    return {half, half + tokens * K,
            reinterpret_cast<float*>(half + 2 * tokens * K)};
  }
};
__global__ void SplitRows(const float* __restrict__ x, SplitActivations out,
                          int K) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* src = x + std::size_t(row) * K;
  float maximum = 0;
  for (int k = tid; k < K; k += 256)
    maximum = fmaxf(maximum, fabsf(src[k]));
  for (int d = 16; d; d /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, d, 32));
  __shared__ float maxima[8];
  if (!(tid % 32))
    maxima[tid / 32] = maximum;
  __syncthreads();
  maximum = 0;
  for (float v : maxima)
    maximum = fmaxf(maximum, v);
  const int e = maximum > 0 ? 14 - ilogbf(maximum) : 0;
  if (!tid)
    out.scale[row] = ldexpf(1.0F, -e);
  for (int k = tid; k < K; k += 256) {
    const float v = ldexpf(src[k], e);
    const __half h = __float2half_rn(v);
    out.high[std::size_t(row) * K + k] = h;
    out.low[std::size_t(row) * K + k] = __float2half_rn(v - __half2float(h));
  }
}
namespace detail {
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// Exact signed Q8 codes as FP16: 0x64XX is 1024 + XX, with XX = q + 128.
__device__ __forceinline__ void Codes(std::uint32_t codes, __half2& lo,
                                      __half2& hi) {
  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  codes ^= 0x80808080U;
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, 0x64646464U, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, 0x64646464U, 0x03070206U);
  lo = __hadd2(__builtin_bit_cast(__half2, p0), magic);
  hi = __hadd2(__builtin_bit_cast(__half2, p1), magic);
}
}  // namespace detail
// Up to four projections of the same split input share one launch; each
// owns a contiguous range of row blocks.
struct Projection {
  const void* weight;
  float* out;
  int rows;
};
struct Projections {
  static constexpr int kMax = 4;
  Projection job[kMax];
  int first_block[kMax];
  int count;
};
// Q8_0 weights x split activations. For each K32 block the low and high
// products chain into one FP32 WMMA accumulator, which is then scaled by the
// block's weight scale; row scales apply after the K reduction. Block = BM
// rows x BN tokens, waves = WM row groups x WN token groups. Global loads for
// the next K block are issued before the current block's matrix work, and two
// LDS stages leave one barrier per K block. With kSwiGLU, job 0 is the FFN
// gate and job 1 its up projection: a block holds BM / 2 rows of each and
// writes silu(gate) * up to job 0's output.
template<int BM, int BN, int WM, int WN, bool kSwiGLU = false>
__launch_bounds__(256) __global__
    void SplitQ8Gemm(Projections jobs, SplitActivations x, int tokens, int K) {
  using detail::v16h;
  using detail::v8f;
  static_assert(WM * WN == 8 && (!kSwiGLU || WM % 2 == 0));
  constexpr int RM = BM / 16 / WM, RN = BN / 16 / WN;
  constexpr int kAPer = (BM + 255) / 256, kBPer = (2 * BN + 255) / 256;
  constexpr int kTileRows = kSwiGLU ? BM / 2 : BM;
  // One K block of a row or token is four 16-byte chunks; neighbouring rows
  // permute them so fragment reads cover all bank groups.
  __shared__ __attribute__((aligned(16))) uint4 s_a[2][BM][4];
  __shared__ __attribute__((aligned(16))) uint4 s_b[2][2][BN][4];
  __shared__ float s_d[2][BM];
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };
  const int blocks = K / 32, tid = threadIdx.x, lane = tid & 31,
            wave = tid >> 5, sub = lane & 15, half = lane >> 4;
  const int wr = wave / WN, wt = wave % WN;
  int job = 0;
  for (int j = 1; j < jobs.count; ++j)
    if (int(blockIdx.y) >= jobs.first_block[j])
      job = j;
  const int M = jobs.job[job].rows;
  const int r_block = (blockIdx.y - jobs.first_block[job]) * kTileRows,
            t_block = blockIdx.x * BN;
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = p * 256 + tid;
    const int source = kSwiGLU ? idx / kTileRows : job;
    const int r = r_block + idx % kTileRows;
    a_live[p] = idx < BM && r < M;
    a_ptr[p] = static_cast<const std::uint8_t*>(jobs.job[source].weight) +
               std::size_t(a_live[p] ? r : M - 1) * blocks * 34;
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = p * 256 + tid, t = t_block + idx % BN;
    b_ptr[p] = idx < 2 * BN && t < tokens
                   ? (idx / BN ? x.low : x.high) + std::size_t(t) * K
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  std::uint16_t a_d[kAPer];
  uint4 b_data[kBPer][4];
  const auto fetch = [&](int kb) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const std::uint8_t* block = a_ptr[p] + std::size_t(kb) * 34;
      a_d[p] = a_live[p] ? *reinterpret_cast<const std::uint16_t*>(block) : 0;
      __builtin_memcpy(&a_codes[p][0], block + 2, 16);
      __builtin_memcpy(&a_codes[p][1], block + 18, 16);
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      if (b_ptr[p]) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + kb * 32);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          b_data[p][c] = src[c];
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          b_data[p][c] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  const auto commit = [&](int stage) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int row = p * 256 + tid;
      if (row >= BM)
        break;
      const std::uint32_t words[8] = {
          a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
          a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
      __half2 h[16];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        detail::Codes(words[i], h[2 * i], h[2 * i + 1]);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint4 v;
        __builtin_memcpy(&v, &h[4 * c], 16);
        s_a[stage][row][swizzle(row, c)] = v;
      }
      s_d[stage][row] = __half2float(__builtin_bit_cast(__half, a_d[p]));
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = p * 256 + tid;
      if (idx >= 2 * BN)
        break;
      const int plane = idx / BN, t = idx % BN;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        s_b[stage][plane][t][swizzle(t, c)] = b_data[p][c];
    }
  };
  v8f acc[RM][RN] = {};
  const v8f zero{};
  fetch(0);
  commit(0);
  __syncthreads();
  for (int kb = 0; kb < blocks; ++kb) {
    const int stage = kb & 1;
    if (kb + 1 < blocks)
      fetch(kb + 1);
    // Keeps the next block's loads ahead of the matrix work.
    __builtin_amdgcn_sched_barrier(0);
    v16h a0[RM], a1[RM];
    float d[RM][8];
#pragma unroll
    for (int i = 0; i < RM; ++i) {
      const int row = (wr * RM + i) * 16 + sub;
      uint4 c[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        c[q] = s_a[stage][row][swizzle(row, q)];
      __builtin_memcpy(&a0[i], &c[0], 32);
      __builtin_memcpy(&a1[i], &c[2], 32);
#pragma unroll
      for (int l = 0; l < 8; ++l)
        d[i][l] = s_d[stage][(wr * RM + i) * 16 + 2 * l + half];
    }
#pragma unroll
    for (int j = 0; j < RN; ++j) {
      const int t = (wt * RN + j) * 16 + sub;
      v16h b[2][2];
#pragma unroll
      for (int plane = 0; plane < 2; ++plane) {
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          c[q] = s_b[stage][plane][t][swizzle(t, q)];
        __builtin_memcpy(&b[plane][0], &c[0], 32);
        __builtin_memcpy(&b[plane][1], &c[2], 32);
      }
#pragma unroll
      for (int i = 0; i < RM; ++i) {
        v8f c = detail::Wmma(a0[i], b[1][0], zero);
        c = detail::Wmma(a1[i], b[1][1], c);
        c = detail::Wmma(a0[i], b[0][0], c);
        c = detail::Wmma(a1[i], b[0][1], c);
#pragma unroll
        for (int l = 0; l < 8; ++l)
          acc[i][j][l] = fmaf(d[i][l], c[l], acc[i][j][l]);
      }
    }
    if (kb + 1 < blocks)
      commit(stage ^ 1);
    __syncthreads();
  }
  float* y = jobs.job[job].out;
  // Up waves hand their scaled results to the gate waves through LDS.
  [[maybe_unused]] auto* s_up = reinterpret_cast<float (*)[BN]>(&s_a[0][0][0]);
  static_assert(!kSwiGLU || sizeof(float) * kTileRows * BN <= sizeof(s_a));
  if constexpr (kSwiGLU) {
    if (wr >= WM / 2)
#pragma unroll
      for (int j = 0; j < RN; ++j) {
        const int local = (wt * RN + j) * 16 + sub, token = t_block + local;
        const float scale = token < tokens ? x.scale[token] : 0;
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int l = 0; l < 8; ++l)
            s_up[(wr * RM + i) * 16 + 2 * l + half - kTileRows][local] =
                acc[i][j][l] * scale;
      }
    __syncthreads();
    if (wr >= WM / 2)
      return;
  }
#pragma unroll
  for (int j = 0; j < RN; ++j) {
    const int local = (wt * RN + j) * 16 + sub, token = t_block + local;
    if (token >= tokens)
      continue;
    const float scale = x.scale[token];
#pragma unroll
    for (int i = 0; i < RM; ++i)
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int tile_row = (wr * RM + i) * 16 + 2 * l + half;
        const int row = r_block + tile_row;
        if (row >= M)
          continue;
        float value = acc[i][j][l] * scale;
        if constexpr (kSwiGLU) {
          // The separate SwiGLU kernel's formula.
          const float g = value, u = s_up[tile_row][local];
          const float sig_g = 1.0F / (1.0F + expf(-g));
          value = (g * sig_g) * u;
        }
        y[std::size_t(token) * M + row] = value;
      }
  }
}
}  // namespace gufo::models::lfm3::gpu
#endif
