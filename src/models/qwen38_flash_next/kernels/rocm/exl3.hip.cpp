// EXL3 trellis linears for gfx11 wave32. The decode primitives and the
// GEMV structure are adapted from Crucible (engine/exl3/exl3.h,
// engine/exl3/kernels/exl3_gemv.h, engine/gfx1151/common.h), Copyright 2026
// jaedync, Apache-2.0; the format is ExLlamaV3's (MIT). See
// THIRD_PARTY_NOTICES.md.
//
// Decode path, per 16x16 tile and lane t: lane t owns ring positions
// 8t..8t+7, i.e. rows {r0, r0+1, r0+8, r0+9} of columns {c0, c0+8} with
// r0 = 2 (t % 4) and c0 = t / 4. Its eight 16-bit states are one contiguous
// 7K+16-bit window, so three dword loads cover them. Products use
// v_dot2_f32_f16 into fp32 with one accumulator chain per (row, column) in a
// fixed tile order, and split-K partials are summed in a fixed order: a
// row's result never depends on how many rows share the launch.
#include "src/models/qwen38_flash_next/kernels/rocm/exl3.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>

#include "src/models/qwen38_flash_next/exl3/format.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

namespace fmt = ::gufo::models::qwen38_flash_next::exl3;

constexpr int kWaves = 8;
constexpr int kBlockThreads = kWaves * 32;
/// LDS words of staged transformed input per block (32 KiB).
constexpr int kStageWords = 8192;

__device__ __forceinline__ float RoundH(float v) {
  return __half2float(__float2half(v));
}

/// Value of lane ^ M: DPP row_xmask moves data within 16-lane rows and
/// permlanex16 swaps the rows, so no ds_bpermute goes through the LDS
/// crossbar. All 32 lanes must be active.
template<int M>
__device__ __forceinline__ float XorLane(float v) {
  const int x = __float_as_int(v);
  if constexpr (M == 16) {
    return __int_as_float(__builtin_amdgcn_permlanex16(
        x, x, 0x76543210U, 0xFEDCBA98U, false, false));
  } else {
    return __int_as_float(
        __builtin_amdgcn_update_dpp(0, x, 0x160 + M, 0xF, 0xF, false));
  }
}

/// One butterfly: v + o on lanes without bit M, o - v on lanes with it, as
/// one fma with the lane's sign.
template<int M>
__device__ __forceinline__ void WhtStage(float v[4], int lane) {
  const float sign = (lane & M) ? -1.0F : 1.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    v[i] = __builtin_fmaf(v[i], sign, XorLane<M>(v[i]));
}

/// Unnormalized 128-point Walsh-Hadamard transform of a wave's 4 values per
/// lane (lane l holds elements 4l..4l+3). Callers scale by kHadamardNorm.
__device__ __forceinline__ void Wht128(float v[4]) {
  const int lane = threadIdx.x & 31;
  const float a0 = v[0] + v[1];
  const float a1 = v[0] - v[1];
  const float a2 = v[2] + v[3];
  const float a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
  WhtStage<1>(v, lane);
  WhtStage<2>(v, lane);
  WhtStage<4>(v, lane);
  WhtStage<8>(v, lane);
  WhtStage<16>(v, lane);
}

/// state * 0x83DCD12D mod 2^32 for the 16-bit state in the low half of x
/// (the high half is ignored). A packed 16-bit multiply with op_sel_hi
/// picking x for the high lane gives hi = ((x * 0x83DC) mod 2^16) << 16 and
/// the low lane x * 0 = 0; one 16x16 mad then adds x * 0xD12D. Two
/// full-rate ops instead of a quarter-rate 32-bit multiply, and no mask.
__device__ __forceinline__ std::uint32_t Mul1Product(std::uint32_t x) {
  std::uint32_t hi;
  std::uint32_t p;
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]"
      : "=v"(hi)
      : "v"(x), "s"(0x83DC0000U));
  asm("v_mad_u32_u16 %0, %1.l, 0xd12d, %2" : "=v"(p) : "v"(x), "v"(hi));
  return p;
}

/// Mul1Product of the state in the high half of x: op_sel reads it in place
/// of a shift.
__device__ __forceinline__ std::uint32_t Mul1ProductHi(std::uint32_t x) {
  std::uint32_t hi;
  std::uint32_t p;
  asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel:[1,0] op_sel_hi:[1,1]"
      : "=v"(hi)
      : "v"(x), "s"(0x83DC0000U));
  asm("v_mad_u32_u16 %0, %1.h, 0xd12d, %2" : "=v"(p) : "v"(x), "v"(hi));
  return p;
}

/// fp16 codebook values of two products: fma(1024 + bytesum(p), 0x1eee,
/// 0xc931) with one rounding. Each byte sum is at most 1020, so 0x6400 plus
/// a sum never carries out of its half.
__device__ __forceinline__ __half2 Mul1Pair(std::uint32_t p0,
                                            std::uint32_t p1) {
  const std::uint32_t s0 = __builtin_amdgcn_sad_u8(p0, 0U, 0x64006400U);
  std::uint32_t packed = __builtin_amdgcn_sad_hi_u8(p1, 0U, s0);
  const __half2 h = *reinterpret_cast<__half2*>(&packed);
  const __half2 inv =
      __halves2half2(__ushort_as_half(0x1EEE), __ushort_as_half(0x1EEE));
  const __half2 bias =
      __halves2half2(__ushort_as_half(0xC931), __ushort_as_half(0xC931));
  return __hfma2(h, inv, bias);
}

/// The lane's 8 weights of one tile as 4 half2:
/// P0 = rows (r0, r0+1) col c0, P1 = rows (r0+8, r0+9) col c0,
/// P2 = rows (r0, r0+1) col c0+8, P3 = rows (r0+8, r0+9) col c0+8.
template<int K>
__device__ __forceinline__ void DecodeLane(std::uint32_t w0, std::uint32_t w1,
                                           std::uint32_t w2, int shift,
                                           __half2 P[4]) {
  std::uint32_t st[8];
  fmt::LaneStates<K>(w0, w1, w2, shift, st);
  std::uint32_t p[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    p[j] = Mul1Product(st[j]);
  P[0] = Mul1Pair(p[0], p[1]);
  P[1] = Mul1Pair(p[2], p[3]);
  P[2] = Mul1Pair(p[4], p[5]);
  P[3] = Mul1Pair(p[6], p[7]);
}

__device__ __forceinline__ float Dot2(__half2 a, __half2 b, float c) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  return __builtin_amdgcn_fdot2(*reinterpret_cast<h2*>(&a),
                                *reinterpret_cast<h2*>(&b), c, false);
}

// gfx11 raw buffer descriptor word 3 (32-bit data, raw out-of-bounds
// checking): loads past the stream return 0 without touching memory.
constexpr int kBufferWord3 = 0x31004000;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
StreamRsrc(const std::uint32_t* base, std::uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(
      const_cast<std::uint32_t*>(base), static_cast<short>(0),
      static_cast<int>(bytes), kBufferWord3);
}

struct LinearArgs {
  Exl3Weights w;
  const float* x;
  const __half* x16;  // F16 inputs instead of x (input transform only)
  std::uint32_t ldx;
  std::uint32_t x_div;
  const std::int32_t* ids;
  float* y;
  std::uint32_t ldy;
  std::uint32_t rows;
  __half* xh;
  float* partial;
  int splits;
  unsigned* done;  // per (row block, column group) split arrival counters
};

__device__ __forceinline__ std::uint32_t ExpertOf(const LinearArgs& a,
                                                  std::uint32_t row) {
  return a.ids != nullptr ? static_cast<std::uint32_t>(a.ids[row]) : 0U;
}

/// One wave per (row, 128-block): xh = half(H(half(half(x) * suh)) /
/// sqrt(128)).
__global__ void __launch_bounds__(kBlockThreads) Exl3InputKernel(LinearArgs a) {
  const std::uint32_t blocks = a.w.in / 128;
  const std::uint32_t item = blockIdx.x * kWaves + (threadIdx.x >> 5);
  if (item >= a.rows * blocks)
    return;
  const std::uint32_t row = item / blocks;
  const std::uint32_t blk = item % blocks;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t c0 = blk * 128 + lane * 4;
  const std::size_t at = static_cast<std::size_t>(row / a.x_div) * a.ldx + c0;
  const __half* suh =
      a.w.suh + static_cast<std::size_t>(ExpertOf(a, row)) * a.w.in + c0;
  float x[4];
  if (a.x16 != nullptr) {
    const uint2 h = *reinterpret_cast<const uint2*>(a.x16 + at);
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
    x[0] = lo.x;
    x[1] = lo.y;
    x[2] = hi.x;
    x[3] = hi.y;
  } else {
    const float4 f = *reinterpret_cast<const float4*>(a.x + at);
    x[0] = RoundH(f.x);
    x[1] = RoundH(f.y);
    x[2] = RoundH(f.z);
    x[3] = RoundH(f.w);
  }
  const uint2 sh = *reinterpret_cast<const uint2*>(suh);
  const float2 s01 = __half22float2(*reinterpret_cast<const __half2*>(&sh.x));
  const float2 s23 = __half22float2(*reinterpret_cast<const __half2*>(&sh.y));
  float v[4] = {RoundH(x[0] * s01.x), RoundH(x[1] * s01.y),
                RoundH(x[2] * s23.x), RoundH(x[3] * s23.y)};
  Wht128(v);
  __half* dst = a.xh + static_cast<std::size_t>(row) * a.w.in + c0;
  const __half2 out[2] = {
      __floats2half2_rn(v[0] * fmt::kHadamardNorm, v[1] * fmt::kHadamardNorm),
      __floats2half2_rn(v[2] * fmt::kHadamardNorm, v[3] * fmt::kHadamardNorm)};
  *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(out);
}

/// Block (column group, split, row group): 8 waves, one 16-column block
/// each, over the split's tile range; R rows share the decoded weights.
template<int K, int R>
__device__ __forceinline__ void GemvBody(const LinearArgs& a,
                                         const unsigned zb) {
  __shared__ std::uint32_t xs[kStageWords];
  __shared__ unsigned last_split;
  constexpr int T = fmt::TileDwords(K);
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int A = static_cast<int>(a.w.in >> 4);
  const int span = A / a.splits;
  const int s = static_cast<int>(blockIdx.y);
  const int a0 = s * span;
  const std::uint32_t row0 = zb * R;
  const int nrows = static_cast<int>(min(a.rows - row0, std::uint32_t{R}));
  const int b = static_cast<int>(blockIdx.x) * kWaves + wave;
  const fmt::LaneWindow<K> win(lane);
  const int q = lane & 3;
  const std::uint32_t expert = ExpertOf(a, row0);
  // The wave's tile stream as a buffer resource: per-lane offsets are fixed,
  // the tile offset is scalar, and the prefetch past the span reads zeros.
  const __amdgpu_buffer_rsrc_t rs =
      StreamRsrc(a.w.tiles + expert * a.w.ExpertWords() +
                     (static_cast<std::size_t>(b) * A + a0) * T,
                 static_cast<std::uint32_t>(span * T * 4));
  const int v0 = 4 * win.d0;
  const int v1 = 4 * ((win.d0 + 1) % T);
  const int v2 = 4 * ((win.d0 + 2) % T);
  // Eight tiles in flight: slot f holds tile t0 + f. Spans are multiples
  // of eight (Splits), so the slots need no bounds checks.
  constexpr int kRing = 8;
  std::uint32_t ring[kRing][3];
  const auto load = [&](int t, std::uint32_t (&w)[3]) {
    const int so = t * T * 4;
    w[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, v0, so, 0);
    w[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, v1, so, 0);
    w[2] = __builtin_amdgcn_raw_buffer_load_b32(rs, v2, so, 0);
  };
#pragma unroll
  for (int f = 0; f < kRing; ++f)
    load(f, ring[f]);
  // The weight loads above stream while the inputs are transformed.
  // Inputs are staged per chunk of tiles, [R][chunk * 16] fp16, one wave per
  // (row, 128-input block); a wide row group stages several chunks. Every
  // row keeps one accumulation chain in tile order, so its result is the
  // same at every row-group width.
  constexpr int kChunk = 1024 / R >= 8 ? 1024 / R / 8 * 8 : 8;
  float acc[R][2];
#pragma unroll
  for (int r = 0; r < R; ++r)
    acc[r][0] = acc[r][1] = 0.0F;
  for (int c0 = 0; c0 < span; c0 += kChunk) {
    const int chunk = min(kChunk, span - c0);
    if (c0 > 0)
      __syncthreads();  // every wave is done with the previous chunk
    {
      auto* xsh = reinterpret_cast<__half*>(xs);
      const int blocks = chunk / 8;
      for (int task = wave; task < R * blocks; task += kWaves) {
        const int r = task / blocks;
        const int blk = task - r * blocks;
        const int c = (a0 + c0) * 16 + blk * 128 + lane * 4;
        float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        if (r < nrows) {
          const std::uint32_t row = row0 + r;
          const float* src =
              a.x + static_cast<std::size_t>(row / a.x_div) * a.ldx + c;
          const __half* suh =
              a.w.suh + static_cast<std::size_t>(ExpertOf(a, row)) * a.w.in + c;
#pragma unroll
          for (int i = 0; i < 4; ++i)
            v[i] = RoundH(RoundH(src[i]) * __half2float(suh[i]));
        }
        Wht128(v);
        __half* dst = xsh + r * chunk * 16 + blk * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          dst[i] = __float2half(v[i] * fmt::kHadamardNorm);
      }
    }
    __syncthreads();
    for (int t0 = c0; t0 < c0 + chunk; t0 += kRing) {
#pragma unroll
      for (int f = 0; f < kRing; ++f) {
        const int t = t0 + f;
        const std::uint32_t w0 = ring[f][0];
        const std::uint32_t w1 = ring[f][1];
        const std::uint32_t w2 = ring[f][2];
        load(t + kRing, ring[f]);
        __half2 P[4];
        DecodeLane<K>(w0, w1, w2, win.shift, P);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const std::uint32_t* xr = xs + (r * chunk + (t - c0)) * 8;
          std::uint32_t xa = xr[q];
          std::uint32_t xb = xr[4 + q];
          const __half2 ha = *reinterpret_cast<__half2*>(&xa);
          const __half2 hb = *reinterpret_cast<__half2*>(&xb);
          acc[r][0] = Dot2(P[0], ha, acc[r][0]);
          acc[r][0] = Dot2(P[1], hb, acc[r][0]);
          acc[r][1] = Dot2(P[2], ha, acc[r][1]);
          acc[r][1] = Dot2(P[3], hb, acc[r][1]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float v = acc[r][c];
      v += __shfl_xor(v, 1, 32);
      v += __shfl_xor(v, 2, 32);
      acc[r][c] = v;
    }
  }
  // The block covers one 128-column group: its 8 waves' 16 columns.
  const int col = 16 * wave + lane / 4;
  const int group = static_cast<int>(blockIdx.x);
  float* sums = reinterpret_cast<float*>(xs);  // [R][128]
  if (a.splits == 1) {
    __syncthreads();  // every wave is done with the staged inputs
    if (q == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        sums[r * 128 + col] = acc[r][0];
        sums[r * 128 + col + 8] = acc[r][1];
      }
    }
    __syncthreads();
  } else {
    if (q == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (r < nrows) {
          float* out =
              a.partial +
              (static_cast<std::size_t>(s) * a.rows + row0 + r) * a.w.out +
              group * 128;
          out[col] = acc[r][0];
          out[col + 8] = acc[r][1];
        }
      }
    }
    // The last split to arrive reduces every split in a fixed order.
    __threadfence();
    __syncthreads();
    unsigned* counter = a.done + zb * gridDim.x + blockIdx.x;
    if (threadIdx.x == 0)
      last_split =
          atomicAdd(counter, 1U) == static_cast<unsigned>(a.splits - 1);
    __syncthreads();
    if (!last_split)
      return;
    __threadfence();
    if (threadIdx.x == 0)
      *counter = 0U;  // ready for the next launch (graph replays reuse it)
  }
  // Output transform: one wave per row: H, scale, svh.
  const __half* svh_base = a.w.svh + group * 128 + lane * 4;
  for (int r = wave; r < nrows; r += kWaves) {
    const std::uint32_t row = row0 + r;
    float v[4];
    if (a.splits == 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v[i] = 0.0F + sums[r * 128 + lane * 4 + i];
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v[i] = 0.0F;
      for (int sp = 0; sp < a.splits; ++sp) {
        const float* p =
            a.partial +
            (static_cast<std::size_t>(sp) * a.rows + row) * a.w.out +
            group * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v[i] += __builtin_nontemporal_load(p + i);
      }
    }
    Wht128(v);
    const __half* svh =
        svh_base + static_cast<std::size_t>(ExpertOf(a, row)) * a.w.out;
    float* dst =
        a.y + static_cast<std::size_t>(row) * a.ldy + group * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[i] = v[i] * (fmt::kHadamardNorm * __half2float(svh[i]));
  }
}

template<int K, int R>
__global__ void __launch_bounds__(kBlockThreads) Exl3GemvKernel(LinearArgs a) {
  GemvBody<K, R>(a, blockIdx.z);
}

/// Two matrices of one shape over the same rows in one launch (gate and up
/// of routed experts): row blocks [0, zsplit) take `a`, the rest `b`.
template<int K, int R>
__global__ void __launch_bounds__(kBlockThreads)
    Exl3GemvPairKernel(LinearArgs a, LinearArgs b, unsigned zsplit) {
  if (blockIdx.z < zsplit)
    GemvBody<K, R>(a, blockIdx.z);
  else
    GemvBody<K, R>(b, blockIdx.z - zsplit);
}

template<int K>
void LaunchGemv(const LinearArgs& a, int rows_per_block, hipStream_t stream) {
  const std::uint32_t groups = (a.w.out / 16 + kWaves - 1) / kWaves;
  const std::uint32_t row_blocks =
      (a.rows + rows_per_block - 1) / rows_per_block;
  const dim3 grid(groups, static_cast<unsigned>(a.splits), row_blocks);
  if (rows_per_block == 1) {
    hipLaunchKernelGGL((Exl3GemvKernel<K, 1>), grid, dim3(kBlockThreads), 0,
                       stream, a);
  } else if (rows_per_block == kExl3GemvWideRows) {
    hipLaunchKernelGGL((Exl3GemvKernel<K, kExl3GemvWideRows>), grid,
                       dim3(kBlockThreads), 0, stream, a);
  } else {
    hipLaunchKernelGGL((Exl3GemvKernel<K, kExl3GemvRows>), grid,
                       dim3(kBlockThreads), 0, stream, a);
  }
}

template<int K>
void LaunchGemvPair(const LinearArgs& a, const LinearArgs& b,
                    hipStream_t stream) {
  const std::uint32_t groups = (a.w.out / 16 + kWaves - 1) / kWaves;
  const dim3 grid(groups, static_cast<unsigned>(a.splits), 2 * a.rows);
  hipLaunchKernelGGL((Exl3GemvPairKernel<K, 1>), grid, dim3(kBlockThreads), 0,
                     stream, a, b, a.rows);
}

/// Split-K factor of a shape: enough blocks for the GPU at one row (four
/// experts' rows for expert stacks), never fewer than the staging buffer
/// needs, and a divisor of the tile rows. It depends on the shape only, so
/// a row's accumulation order is the same at every batch width.
int Splits(const Exl3Weights& w, bool experts) {
  const int A = static_cast<int>(w.in / 16);
  const int groups = static_cast<int>((w.out / 16 + kWaves - 1) / kWaves);
  const int lds_min =
      (A * static_cast<int>(kExl3GemvRows) * 8 + kStageWords - 1) / kStageWords;
  const int target = 160 / std::max(1, groups * (experts ? 10 : 1));
  // Spans are whole prefetch rings (A is a multiple of 8: inputs are
  // multiples of 128) and fit the staging buffer.
  const auto valid = [&](int s) {
    return s >= std::max(1, lds_min) && A % s == 0 && (A / s) % 8 == 0;
  };
  const int want = std::max({1, lds_min, std::min(target, A / 8)});
  for (int s = want; s >= 1; --s)
    if (valid(s))
      return s;
  for (int s = want + 1; s <= A / 8; ++s)
    if (valid(s))
      return s;
  return 1;
}

// ---- matrix-core route --------------------------------------------------

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kGemmWaves = 8;    // 16-column blocks: 128 output columns
constexpr int kFusedWaves = 20;  // fused-input GEMM: 640 output columns
constexpr int kGemmTiles = 8;    // fused: 16-row token tiles, 128 rows
// Transformed-input and scratch GEMMs: one column block per wave and 256
// rows, so most experts of a prompt pass take one tile.
constexpr int kTransformedTiles = 16;
static_assert(kExl3GemmRows == kGemmTiles * 16 &&
              kExl3TransformedGemmRows == kTransformedTiles * 16);

struct GemmArgs {
  Exl3Weights w;
  const __half* xh;  // transformed inputs, [x row][in]
  std::uint32_t ldx;
  Exl3GemmOut out;
  float* y;
  std::uint32_t ldy;
  std::uint32_t rows;
  Exl3GemmRoute route;
};

/// The 8 trellis states lane l decodes for the WMMA A operand of one tile:
/// output column m = l % 16 and inputs 8 (l / 16) .. 8 (l / 16) + 7, as the
/// pairs of consecutive ring positions p0(q), p0(q) + 1, q = 0..3. Each pair
/// is one window of K + 16 bits that two dwords hold.
template<int K>
struct AFragWindow {
  int dword[4];  ///< first dword of pair q's window (the second follows)
  int shift[4];  ///< 1..32: the window starts at bit `shift` of dword:next
  GUFO_EXL3_HD explicit AFragWindow(int lane) {
    const int m = lane & 15;
    const int jsel = (m >= 8 ? 4 : 0) + (lane >= 16 ? 2 : 0);
    constexpr int kBits = 256 * K;
    for (int q = 0; q < 4; ++q) {
      const int p0 = 8 * (4 * (m & 7) + q) + jsel;
      const int s = (((p0 + 1) * K - 16) % kBits + kBits) % kBits;
      const int dw = (s - 1 >= 0) ? (s - 1) / 32 : -1;
      dword[q] = (dw + 8 * K) % (8 * K);
      shift[q] = s - dw * 32;
    }
  }
};

/// Lane l's half of the A operand (W^T rows: output m, 8 inputs) as 4 half2
/// in input order, from the 8 words its window loaded.
template<int K>
__device__ __forceinline__ void DecodeAHalf(const std::uint32_t (&w)[8],
                                            const AFragWindow<K>& win,
                                            std::uint32_t own[4]) {
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const std::uint32_t a =
        fmt::AlignBit(w[2 * q], w[2 * q + 1], 32 - win.shift[q]);
    const __half2 h = Mul1Pair(Mul1ProductHi(a), Mul1Product(a >> (16 - K)));
    own[q] = __builtin_bit_cast(std::uint32_t, h);
  }
}

/// Block (256-column group, row tile). Per 64-input step the block stages
/// its rows' transformed inputs (F16) in LDS, the B operand. Each lane
/// decodes its half of the A operand (W^T: output columns x inputs) of each
/// of its wave's two column blocks in registers, and one permlanex16 per
/// dword exchanges the halves. D = Y^T: lane l holds column 2i + l/16 of
/// the block, row l % 16 of the tile.
template<int K, bool Fused, int W, int CB, int RT>
__global__ void __launch_bounds__(W * 32) Exl3GemmKernel(GemmArgs g) {
  constexpr int kBlocks = CB;  // column blocks per wave
  // 64 inputs per step keep the block within half the LDS, so two blocks
  // share a CU. Fused steps are one 128-input Hadamard block.
  constexpr int kStep = Fused ? 128 : 64;
  constexpr int kStepPitch = kStep + 8;
  constexpr int kParts = kStep / 8;
  // The output transform reuses the stage for 32 rows of fp32.
  // The block's columns; output rows are padded by 4 floats so the D
  // writes of a wave's 16 lanes (one per row) hit distinct banks.
  constexpr int kCols = W * kBlocks * 16;
  static_assert(kCols % 128 == 0);
  constexpr int kOutPitch = kCols + 4;
  __shared__ __half
      stage[(RT * 16 * kStepPitch > 16 * kOutPitch * 2) ? RT * 16 * kStepPitch
                                                        : 16 * kOutPitch * 2];
  constexpr int T = fmt::TileDwords(K);
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int A = static_cast<int>(g.w.in >> 4);
  const int B = static_cast<int>(g.w.out >> 4);
  // Wave's column blocks: one in each 128-column half of the group.
  int cb[kBlocks];
  bool has[kBlocks];
#pragma unroll
  for (int u = 0; u < kBlocks; ++u) {
    cb[u] = static_cast<int>(blockIdx.x) * kBlocks * W + u * W + wave;
    has[u] = cb[u] < B;
  }
  std::uint32_t expert = 0;
  int begin = 0;
  int end = 0;
  if (g.route.tiles != nullptr) {
    const std::int32_t item = g.route.tiles[blockIdx.y];
    expert = static_cast<std::uint32_t>(item & 0xFFFF);
    begin = g.route.bounds[expert] + (item >> 16) * RT * 16;
    end = min(g.route.bounds[expert + 1], begin + RT * 16);
  } else {
    begin = static_cast<int>(blockIdx.y) * RT * 16;
    end = min(static_cast<int>(g.rows), begin + RT * 16);
  }
  if (begin >= end)
    return;
  const int live = (end - begin + 15) / 16;
  const auto x_row = [&](int p) -> int {
    return g.route.x_rows != nullptr ? g.route.x_rows[p] : p;
  };
  // Each column block's tile stream as a buffer resource: fixed per-lane
  // offsets, a scalar tile offset, and zeros past the stream.
  __amdgpu_buffer_rsrc_t rs[kBlocks];
#pragma unroll
  for (int u = 0; u < kBlocks; ++u)
    rs[u] = StreamRsrc(g.w.tiles + expert * g.w.ExpertWords() +
                           static_cast<std::size_t>(has[u] ? cb[u] : 0) * A * T,
                       has[u] ? static_cast<std::uint32_t>(A * T * 4) : 0U);
  // A gfx1151 WMMA reads row r of its A operand from one lane only: lane r
  // for even rows, lane r + 16 for odd ones. The reading lane decodes
  // inputs 0..7 of its row and its partner inputs 8..15, so after one
  // permlanex16 per dword every reading lane holds [own | peer] in input
  // order; the partners' copies are ignored, and no lane selects.
  const int m = lane & 15;
  const bool reader = ((m & 1) == 0) == (lane < 16);
  const AFragWindow<K> win(reader ? m : m + 16);
  int voff[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    voff[2 * q] = 4 * win.dword[q];
    voff[2 * q + 1] = 4 * ((win.dword[q] + 1) % T);
  }
  v8f acc[kBlocks][RT];
#pragma unroll
  for (int u = 0; u < kBlocks; ++u)
#pragma unroll
    for (int j = 0; j < RT; ++j)
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  // The next tile of every column block stays in flight.
  constexpr int kAhead = 1;
  constexpr int kTilesPerChunk = kStep / 16;
  constexpr int kStageLoads = RT * 16 * kParts / (W * 32);
  std::uint32_t ring[kBlocks][kAhead][8];
  const auto load_tile = [&](int u, int a, std::uint32_t (&w)[8]) {
    const int so = a * T * 4;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      w[i] = __builtin_amdgcn_raw_buffer_load_b32(rs[u], voff[i], so, 0);
  };
#pragma unroll
  for (int u = 0; u < kBlocks; ++u)
#pragma unroll
    for (int f = 0; f < kAhead; ++f)
      load_tile(u, f, ring[u][f]);
  // Each thread's staged rows, resolved once: rows past the tile read 0.
  int stage_row[kStageLoads];
#pragma unroll
  for (int n = 0; n < kStageLoads; ++n) {
    const int r = (n * W * 32 + static_cast<int>(threadIdx.x)) / kParts;
    const int p = begin + r;
    stage_row[n] = r < live * 16 && p < end ? x_row(p) : -1;
  }
  // The next step's staged inputs are loaded while this step computes.
  uint4 staged[kStageLoads];
  const auto load_stage = [&](int k) {
#pragma unroll
    for (int n = 0; n < kStageLoads; ++n) {
      const int i = n * W * 32 + static_cast<int>(threadIdx.x);
      const int row = stage_row[n];
      staged[n] = row >= 0 ? *reinterpret_cast<const uint4*>(
                                 g.xh + static_cast<std::size_t>(row) * g.ldx +
                                 k + (i % kParts) * 8)
                           : uint4{0U, 0U, 0U, 0U};
    }
  };
  // Fused: wave w stages rows w, w + 8, ..., 4 raw F16 inputs per lane,
  // and transforms them (suh, H) on the way into LDS.
  constexpr int kFusedRows = Fused ? (RT * 16 + W - 1) / W : 1;
  int fused_row[kFusedRows];
  uint2 raw[kFusedRows];
  uint2 raw_suh{0U, 0U};
  const __half* suh = g.w.suh + static_cast<std::size_t>(expert) * g.w.in;
  if constexpr (Fused) {
#pragma unroll
    for (int n = 0; n < kFusedRows; ++n) {
      const int r = n * W + wave;
      const int p = begin + r;
      fused_row[n] = r < live * 16 && p < end ? x_row(p) : -1;
      static_assert(Fused || W == kGemmWaves);
    }
  }
  const auto load_fused = [&](int k) {
#pragma unroll
    for (int n = 0; n < kFusedRows; ++n) {
      const int row = fused_row[n];
      raw[n] =
          row >= 0
              ? *reinterpret_cast<const uint2*>(
                    g.xh + static_cast<std::size_t>(row) * g.ldx + k + lane * 4)
              : uint2{0U, 0U};
    }
    raw_suh = *reinterpret_cast<const uint2*>(suh + k + lane * 4);
  };
  if constexpr (Fused)
    load_fused(0);
  else
    load_stage(0);
  for (int k0 = 0; k0 < static_cast<int>(g.w.in); k0 += kStep) {
    __syncthreads();
    if constexpr (Fused) {
      const __half2 s01 = *reinterpret_cast<const __half2*>(&raw_suh.x);
      const __half2 s23 = *reinterpret_cast<const __half2*>(&raw_suh.y);
#pragma unroll
      for (int n = 0; n < kFusedRows; ++n) {
        // Rows past the live tiles are never read; padding rows inside them
        // stage zeros (wave-uniform).
        if (n * W + wave >= live * 16)
          break;
        if (fused_row[n] < 0) {
          *reinterpret_cast<uint2*>(stage + (n * W + wave) * kStepPitch +
                                    lane * 4) = uint2{0U, 0U};
          continue;
        }
        // An F16 product is exact in F32, so the packed F16 multiply rounds
        // as RoundH(x * suh).
        const float2 lo = __half22float2(
            __hmul2(*reinterpret_cast<const __half2*>(&raw[n].x), s01));
        const float2 hi2 = __half22float2(
            __hmul2(*reinterpret_cast<const __half2*>(&raw[n].y), s23));
        float v[4] = {lo.x, lo.y, hi2.x, hi2.y};
        Wht128(v);
        const __half2 h[2] = {__floats2half2_rn(v[0] * fmt::kHadamardNorm,
                                                v[1] * fmt::kHadamardNorm),
                              __floats2half2_rn(v[2] * fmt::kHadamardNorm,
                                                v[3] * fmt::kHadamardNorm)};
        *reinterpret_cast<uint2*>(stage + (n * W + wave) * kStepPitch +
                                  lane * 4) =
            *reinterpret_cast<const uint2*>(h);
      }
    } else {
#pragma unroll
      for (int n = 0; n < kStageLoads; ++n) {
        const int i = n * W * 32 + static_cast<int>(threadIdx.x);
        *reinterpret_cast<uint4*>(stage + (i / kParts) * kStepPitch +
                                  (i % kParts) * 8) = staged[n];
      }
    }
    __syncthreads();
    if (k0 + kStep < static_cast<int>(g.w.in)) {
      if constexpr (Fused)
        load_fused(k0 + kStep);
      else
        load_stage(k0 + kStep);
    }
#pragma unroll
    for (int kk = 0; kk < kTilesPerChunk; ++kk) {
      const int a = k0 / 16 + kk;
      v16h af[kBlocks];
#pragma unroll
      for (int u = 0; u < kBlocks; ++u) {
        std::uint32_t own[4];
        DecodeAHalf<K>(ring[u][kk % kAhead], win, own);
        load_tile(u, a + kAhead, ring[u][kk % kAhead]);
        std::uint32_t frag[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          frag[i] = own[i];
          frag[4 + i] = __builtin_amdgcn_permlanex16(
              own[i], own[i], 0x76543210U, 0xFEDCBA98U, false, false);
        }
        __builtin_memcpy(&af[u], frag, 32);
      }
#pragma unroll
      for (int j = 0; j < RT; ++j) {
        if (j < live) {
          v16h bf;
          __builtin_memcpy(
              &bf, stage + (j * 16 + (lane & 15)) * kStepPitch + kk * 16, 32);
#pragma unroll
          for (int u = 0; u < kBlocks; ++u)
            if (has[u])  // wave-uniform: a 640-wide output fills 2.5 groups
              acc[u][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[u], bf,
                                                                     acc[u][j]);
        }
      }
    }
  }
  // Output transform per 16-row tile through the staging buffer: [row]
  // [block columns] fp32, then one wave per (row, 128-column group): H,
  // scale, svh.
  static_assert(sizeof(stage) >= 16 * kOutPitch * sizeof(float));
  float* out = reinterpret_cast<float*>(stage);
  const __half* svh = g.w.svh + static_cast<std::size_t>(expert) * g.w.out;
  constexpr int kGroups = kCols / 128;
  const int col0 = static_cast<int>(blockIdx.x) * kCols;
  for (int j = 0; j < RT; ++j) {
    if (j >= live)
      break;
    __syncthreads();
#pragma unroll
    for (int t = 0; t < RT; ++t) {
      if (t != j)
        continue;
#pragma unroll
      for (int u = 0; u < kBlocks; ++u) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int col = (u * W + wave) * 16 + 2 * i + (lane >> 4);
          out[(lane & 15) * kOutPitch + col] = acc[u][t][i];
        }
      }
    }
    __syncthreads();
    for (int task = wave; task < 16 * kGroups; task += W) {
      const int r = task / kGroups;
      const int grp = task % kGroups;
      const int p = begin + j * 16 + r;
      const int c = col0 + grp * 128 + lane * 4;
      if (p >= end || col0 + grp * 128 >= static_cast<int>(g.w.out))
        continue;
      const int row = g.route.y_rows != nullptr ? g.route.y_rows[p] : p;
      if (row < 0)
        continue;
      const float4 o = *reinterpret_cast<const float4*>(out + r * kOutPitch +
                                                        grp * 128 + lane * 4);
      float v[4] = {o.x, o.y, o.z, o.w};
      Wht128(v);
      const uint2 sv = *reinterpret_cast<const uint2*>(svh + c);
      const float2 s01 =
          __half22float2(*reinterpret_cast<const __half2*>(&sv.x));
      const float2 s23 =
          __half22float2(*reinterpret_cast<const __half2*>(&sv.y));
      const float scale[4] = {s01.x, s01.y, s23.x, s23.y};
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v[i] *= fmt::kHadamardNorm * scale[i];
      const std::size_t at = static_cast<std::size_t>(row) * g.ldy + c;
      if (g.out.gate != nullptr) {
        // SwiGLU into the next projection's F16 inputs.
        const float4 gv = *reinterpret_cast<const float4*>(g.out.gate + at);
        const float gs[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v[i] *= gs[i] / (1.0F + __expf(-gs[i]));
      }
      if (g.out.next_suh != nullptr) {
        // The next linear's input transform, as Exl3InputKernel.
        const uint2 sh = *reinterpret_cast<const uint2*>(
            g.out.next_suh + static_cast<std::size_t>(expert) * g.w.out + c);
        const __half2 n01 = *reinterpret_cast<const __half2*>(&sh.x);
        const __half2 n23 = *reinterpret_cast<const __half2*>(&sh.y);
        const float2 a01 =
            __half22float2(__hmul2(__floats2half2_rn(v[0], v[1]), n01));
        const float2 a23 =
            __half22float2(__hmul2(__floats2half2_rn(v[2], v[3]), n23));
        v[0] = a01.x;
        v[1] = a01.y;
        v[2] = a23.x;
        v[3] = a23.y;
        Wht128(v);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          v[i] *= fmt::kHadamardNorm;
      }
      if (g.out.y_half != nullptr) {
        const __half2 h[2] = {__floats2half2_rn(v[0], v[1]),
                              __floats2half2_rn(v[2], v[3])};
        *reinterpret_cast<uint2*>(g.out.y_half + at) =
            *reinterpret_cast<const uint2*>(h);
      } else {
        *reinterpret_cast<float4*>(g.y + at) = float4{v[0], v[1], v[2], v[3]};
      }
    }
  }
}

/// Block (128-output group g, 128-input block ib): decode the 8x8 tiles of
/// the block into LDS (fp32 [input][output]), apply the input Hadamard to
/// every column and the output Hadamard to every row, scale by suh and svh,
/// and write the Q8_0 blocks of the 128 output rows (ggml's reference
/// quantize_row_q8_0 arithmetic).
template<int K>
__global__ void __launch_bounds__(kBlockThreads)
    Exl3DequantQ8Kernel(Exl3Weights w, std::uint32_t expert,
                        std::uint8_t* dst) {
  __shared__ float m[128 * 128];
  // Element (a, b) lives at row a, column (a + b) mod 128: column walks
  // and row walks both spread over the LDS banks.
  const auto cell = [](int a, int b) { return a * 128 + ((a + b) & 127); };
  constexpr int T = fmt::TileDwords(K);
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int A = static_cast<int>(w.in >> 4);
  const int g = static_cast<int>(blockIdx.x);
  const int ib = static_cast<int>(blockIdx.y);
  const std::uint32_t* tiles = w.tiles + expert * w.ExpertWords();
  const fmt::LaneWindow<K> win(lane);
  const int r0 = 2 * (lane & 3);
  const int c0 = lane >> 2;
  // All of the wave's tile loads are issued before the first decode.
  constexpr int kWaveTiles = 64 / kWaves;
  std::uint32_t tw[kWaveTiles][3];
#pragma unroll
  for (int n = 0; n < kWaveTiles; ++n) {
    const int t = n * kWaves + wave;
    const std::uint32_t* tile =
        tiles +
        (static_cast<std::size_t>(g * 8 + t / 8) * A + ib * 8 + t % 8) * T;
    tw[n][0] = tile[win.d0];
    tw[n][1] = tile[(win.d0 + 1) % T];
    tw[n][2] = tile[(win.d0 + 2) % T];
  }
#pragma unroll
  for (int n = 0; n < kWaveTiles; ++n) {
    const int t = n * kWaves + wave;
    __half2 P[4];
    DecodeLane<K>(tw[n][0], tw[n][1], tw[n][2], win.shift, P);
    const int a = (t % 8) * 16;
    const int b = (t / 8) * 16;
    const float2 p0 = __half22float2(P[0]);
    const float2 p1 = __half22float2(P[1]);
    const float2 p2 = __half22float2(P[2]);
    const float2 p3 = __half22float2(P[3]);
    m[cell(a + r0, b + c0)] = p0.x;
    m[cell(a + r0 + 1, b + c0)] = p0.y;
    m[cell(a + r0 + 8, b + c0)] = p1.x;
    m[cell(a + r0 + 9, b + c0)] = p1.y;
    m[cell(a + r0, b + c0 + 8)] = p2.x;
    m[cell(a + r0 + 1, b + c0 + 8)] = p2.y;
    m[cell(a + r0 + 8, b + c0 + 8)] = p3.x;
    m[cell(a + r0 + 9, b + c0 + 8)] = p3.y;
  }
  __syncthreads();
  // Input transform of every output column: H over the inputs.
  for (int b = wave; b < 128; b += kWaves) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      v[i] = m[cell(lane * 4 + i, b)];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m[cell(lane * 4 + i, b)] = v[i] * fmt::kHadamardNorm;
  }
  __syncthreads();
  // Output transform of every input row: H over the outputs.
  for (int a = wave; a < 128; a += kWaves) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
      v[i] = m[cell(a, lane * 4 + i)];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m[cell(a, lane * 4 + i)] = v[i] * fmt::kHadamardNorm;
  }
  __syncthreads();
  // Q8_0 blocks: output row b, inputs [32 q, 32 q + 32) of this block,
  // quantized in registers, staged in LDS as each row's 136 contiguous
  // bytes (4 blocks of 34), then stored as dwords.
  const __half* suh = w.suh + static_cast<std::size_t>(expert) * w.in;
  const __half* svh = w.svh + static_cast<std::size_t>(expert) * w.out;
  const std::size_t row_bytes = static_cast<std::size_t>(w.in / 32) * 34;
  constexpr int kTasks = 128 * 4 / kBlockThreads;
  std::uint16_t packed[kTasks][17];
#pragma unroll
  for (int n = 0; n < kTasks; ++n) {
    const int task = n * kBlockThreads + static_cast<int>(threadIdx.x);
    const int b = task >> 2;
    const int q = task & 3;
    const int o = g * 128 + b;
    const float so = __half2float(svh[o]);
    float x[32];
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int a = q * 32 + i;
      x[i] = m[cell(a, b)] * so * __half2float(suh[ib * 128 + a]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    packed[n][0] = __half_as_ushort(__float2half(d));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const auto lo = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(__float2int_rn(x[2 * i] * id)));
      const auto hi = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(__float2int_rn(x[2 * i + 1] * id)));
      packed[n][1 + i] = static_cast<std::uint16_t>(lo | (hi << 8));
    }
  }
  __syncthreads();  // every thread is done reading m
  auto* bytes = reinterpret_cast<std::uint16_t*>(m);  // [128][68] halves
#pragma unroll
  for (int n = 0; n < kTasks; ++n) {
    const int task = n * kBlockThreads + static_cast<int>(threadIdx.x);
    const int b = task >> 2;
    const int q = task & 3;
#pragma unroll
    for (int i = 0; i < 17; ++i)
      bytes[b * 68 + q * 17 + i] = packed[n][i];
  }
  __syncthreads();
  // Rows of 34 dwords; row_bytes and the block offset are multiples of 4
  // (Exl3DequantizeQ8 requires in % 128 == 0).
  const auto* words = reinterpret_cast<const std::uint32_t*>(m);
  for (int i = static_cast<int>(threadIdx.x); i < 128 * 34;
       i += kBlockThreads) {
    const int b = i / 34;
    const int wd = i % 34;
    auto* row = reinterpret_cast<std::uint32_t*>(
        dst + static_cast<std::size_t>(g * 128 + b) * row_bytes + ib * 136);
    row[wd] = words[b * 34 + wd];
  }
}

__global__ void Exl3RepackKernel(const std::uint8_t* base,
                                 const std::uint64_t* src_offsets,
                                 std::uint32_t* dst, std::uint32_t src_out,
                                 std::uint32_t in, std::uint32_t out, int k,
                                 std::uint32_t experts, std::size_t dst_stride,
                                 const std::uint32_t* in_blocks,
                                 const std::uint32_t* out_blocks) {
  const std::size_t T = 8 * static_cast<std::size_t>(k);
  const std::size_t A = in / 16;
  const std::size_t B = out / 16;
  const std::size_t src_B = src_out / 16;
  const std::size_t per_expert = A * B * T;
  const std::size_t total = per_expert * experts;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = i / per_expert;
    const std::size_t r = i % per_expert;
    const std::size_t w = r % T;
    const std::size_t tile = r / T;  // b * A + a
    const std::size_t b = tile / A;
    const std::size_t a = tile % A;
    const std::size_t sa =
        in_blocks != nullptr ? in_blocks[a / 8] * std::size_t{8} + a % 8 : a;
    const std::size_t sb =
        out_blocks != nullptr ? out_blocks[b / 8] * std::size_t{8} + b % 8 : b;
    // Payloads are only 2-byte aligned in the file: read uint16 pairs.
    const auto* s =
        reinterpret_cast<const std::uint16_t*>(base + src_offsets[e]) +
        2 * ((sa * src_B + sb) * T + w);
    dst[e * dst_stride + r] = static_cast<std::uint32_t>(s[0]) |
                              (static_cast<std::uint32_t>(s[1]) << 16);
  }
}

__global__ void Exl3GatherScalesKernel(const std::uint8_t* base,
                                       const std::uint64_t* offsets,
                                       std::uint32_t count,
                                       std::uint32_t src_len,
                                       const std::uint32_t* blocks,
                                       std::uint32_t dst_len, __half* dst) {
  const std::size_t total = static_cast<std::size_t>(count) * dst_len;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t part = i / dst_len;
    const std::size_t j = i % dst_len;
    const std::size_t sj =
        blocks != nullptr ? blocks[j / 128] * std::size_t{128} + j % 128 : j;
    const auto* src = reinterpret_cast<const __half*>(base + offsets[part]);
    dst[i] = sj < src_len ? src[sj] : __float2half(0.0F);
  }
}

}  // namespace

bool Exl3Linear(const Exl3Weights& w, const float* x, std::uint32_t ldx,
                std::uint32_t x_div, const std::int32_t* ids, float* y,
                std::uint32_t ldy, std::uint32_t rows,
                const Exl3Scratch& scratch, hipStream_t stream) {
  if (rows == 0)
    return true;
  if (w.tiles == nullptr || w.in % 128 != 0 || w.out % 128 != 0 ||
      w.k < fmt::kMinTileK || w.k > fmt::kMaxTileK || x_div == 0)
    return false;
  const bool experts = ids != nullptr;
  const int splits = Splits(w, experts);
  const std::uint32_t groups = w.out / 128;
  if (splits > 1 && (scratch.done == nullptr || scratch.done_elems < groups))
    return false;
  // Rows per launch: what the split partials and arrival counters hold, in
  // whole row groups.
  std::size_t chunk = rows;
  if (splits > 1) {
    chunk = std::min(
        scratch.partial_elems / (static_cast<std::size_t>(splits) * w.out),
        scratch.done_elems / groups);
    if (chunk < rows)
      chunk = chunk / kExl3GemvRows * kExl3GemvRows / x_div * x_div;
  }
  if (chunk == 0)
    return false;
  for (std::uint32_t r0 = 0; r0 < rows;
       r0 += static_cast<std::uint32_t>(chunk)) {
    const auto n =
        static_cast<std::uint32_t>(std::min<std::size_t>(chunk, rows - r0));
    LinearArgs a{w,
                 x + static_cast<std::size_t>(r0 / x_div) * ldx,
                 nullptr,
                 ldx,
                 x_div,
                 ids != nullptr ? ids + r0 : nullptr,
                 y + static_cast<std::size_t>(r0) * ldy,
                 ldy,
                 n,
                 scratch.xh,
                 scratch.partial,
                 splits,
                 scratch.done};
    // Wide verification shares each decoded tile across 64 rows.
    const int rows_per_block = experts || n == 1 ? 1
                               : n > static_cast<std::uint32_t>(kExl3GemvRows)
                                   ? static_cast<int>(kExl3GemvWideRows)
                                   : static_cast<int>(kExl3GemvRows);
    switch (w.k) {
      case 2:
        LaunchGemv<2>(a, rows_per_block, stream);
        break;
      case 3:
        LaunchGemv<3>(a, rows_per_block, stream);
        break;
      case 4:
        LaunchGemv<4>(a, rows_per_block, stream);
        break;
      case 5:
        LaunchGemv<5>(a, rows_per_block, stream);
        break;
      case 6:
        LaunchGemv<6>(a, rows_per_block, stream);
        break;
      default:
        return false;
    }
  }
  return hipGetLastError() == hipSuccess;
}

bool Exl3LinearPair(const Exl3Weights& wa, const Exl3Weights& wb,
                    const float* x, std::uint32_t ldx, std::uint32_t x_div,
                    const std::int32_t* ids, float* ya, float* yb,
                    std::uint32_t ldy, std::uint32_t rows,
                    const Exl3Scratch& scratch, hipStream_t stream) {
  const bool same = wa.in == wb.in && wa.out == wb.out && wa.k == wb.k;
  const int splits = Splits(wa, ids != nullptr);
  const std::uint32_t groups = wa.out / 128;
  const std::size_t partial = static_cast<std::size_t>(splits) * rows * wa.out;
  if (!same || ids == nullptr || rows == 0 || wa.tiles == nullptr ||
      wb.tiles == nullptr || x_div == 0 || scratch.done == nullptr ||
      2 * partial > scratch.partial_elems ||
      2 * static_cast<std::size_t>(rows) * groups > scratch.done_elems ||
      wa.k < fmt::kMinTileK || wa.k > fmt::kMaxTileK) {
    return Exl3Linear(wa, x, ldx, x_div, ids, ya, ldy, rows, scratch, stream) &&
           Exl3Linear(wb, x, ldx, x_div, ids, yb, ldy, rows, scratch, stream);
  }
  LinearArgs a{wa,          x,   nullptr, ldx,        x_div,           ids,
               ya,          ldy, rows,    scratch.xh, scratch.partial, splits,
               scratch.done};
  LinearArgs b = a;
  b.w = wb;
  b.y = yb;
  b.partial = scratch.partial + partial;
  b.done = scratch.done + static_cast<std::size_t>(rows) * groups;
  switch (wa.k) {
    case 2:
      LaunchGemvPair<2>(a, b, stream);
      break;
    case 3:
      LaunchGemvPair<3>(a, b, stream);
      break;
    case 4:
      LaunchGemvPair<4>(a, b, stream);
      break;
    case 5:
      LaunchGemvPair<5>(a, b, stream);
      break;
    case 6:
      LaunchGemvPair<6>(a, b, stream);
      break;
    default:
      return false;
  }
  return hipGetLastError() == hipSuccess;
}

bool Exl3LinearGemm(const Exl3Weights& w, const float* x, std::uint32_t ldx,
                    std::uint32_t x_div, const std::int32_t* ids,
                    std::uint32_t x_rows, float* y, std::uint32_t ldy,
                    std::uint32_t rows, const Exl3GemmRoute& route,
                    const Exl3Scratch& scratch, hipStream_t stream,
                    const Exl3GemmOut& out) {
  if (w.tiles == nullptr || w.in % 128 != 0 || w.out % 128 != 0 ||
      w.k < fmt::kMinTileK || w.k > fmt::kMaxTileK || x_div == 0 ||
      (route.tiles != nullptr && !out.fused_input && !out.x_transformed &&
       static_cast<std::size_t>(x_rows) * w.in > scratch.xh_elems))
    return false;
  const std::uint32_t row_tiles =
      route.tiles != nullptr
          ? route.n_tiles
          : (rows + kExl3TransformedGemmRows - 1) / kExl3TransformedGemmRows;
  if (row_tiles == 0 || x_rows == 0)
    return true;
  if (route.tiles == nullptr &&
      static_cast<std::size_t>(x_rows) * w.in > scratch.xh_elems) {
    // Dense rows run in chunks of whole tiles that fit the scratch.
    const auto chunk = static_cast<std::uint32_t>(scratch.xh_elems / w.in /
                                                  kExl3TransformedGemmRows *
                                                  kExl3TransformedGemmRows);
    if (chunk == 0 || x_div != 1 || ids != nullptr)
      return false;
    for (std::uint32_t r0 = 0; r0 < rows; r0 += chunk) {
      const std::uint32_t n = std::min(chunk, rows - r0);
      if (!Exl3LinearGemm(w, x + static_cast<std::size_t>(r0) * ldx, ldx, 1,
                          nullptr, n, y + static_cast<std::size_t>(r0) * ldy,
                          ldy, n, route, scratch, stream, out))
        return false;
    }
    return true;
  }
  if (out.next_suh != nullptr && (out.y_half == nullptr || ldy != w.out))
    return false;
  if (out.fused_input || out.x_transformed) {
    if (route.tiles == nullptr || out.x_half == nullptr)
      return false;
    const GemmArgs g{w, out.x_half, w.in, out, y, ldy, rows, route};
    const dim3 grid((w.out + 127) / 128, row_tiles);
    const dim3 block(kGemmWaves * 32);
    // Fused blocks cover 640 columns, so each row is transformed once per
    // 640 outputs.
    const dim3 fgrid((w.out + kFusedWaves * 32 - 1) / (kFusedWaves * 32),
                     row_tiles);
    const dim3 fblock(kFusedWaves * 32);
    if (out.x_transformed) {
      switch (w.k) {
        case 2:
          hipLaunchKernelGGL(
              (Exl3GemmKernel<2, false, kGemmWaves, 1, kTransformedTiles>),
              grid, block, 0, stream, g);
          break;
        case 3:
          hipLaunchKernelGGL(
              (Exl3GemmKernel<3, false, kGemmWaves, 1, kTransformedTiles>),
              grid, block, 0, stream, g);
          break;
        case 4:
          hipLaunchKernelGGL(
              (Exl3GemmKernel<4, false, kGemmWaves, 1, kTransformedTiles>),
              grid, block, 0, stream, g);
          break;
        case 5:
          hipLaunchKernelGGL(
              (Exl3GemmKernel<5, false, kGemmWaves, 1, kTransformedTiles>),
              grid, block, 0, stream, g);
          break;
        case 6:
          hipLaunchKernelGGL(
              (Exl3GemmKernel<6, false, kGemmWaves, 1, kTransformedTiles>),
              grid, block, 0, stream, g);
          break;
        default:
          return false;
      }
      return hipGetLastError() == hipSuccess;
    }
    switch (w.k) {
      case 2:
        hipLaunchKernelGGL(
            (Exl3GemmKernel<2, true, kFusedWaves, 2, kGemmTiles>), fgrid,
            fblock, 0, stream, g);
        break;
      case 3:
        hipLaunchKernelGGL(
            (Exl3GemmKernel<3, true, kFusedWaves, 2, kGemmTiles>), fgrid,
            fblock, 0, stream, g);
        break;
      case 4:
        hipLaunchKernelGGL(
            (Exl3GemmKernel<4, true, kFusedWaves, 2, kGemmTiles>), fgrid,
            fblock, 0, stream, g);
        break;
      case 5:
        hipLaunchKernelGGL(
            (Exl3GemmKernel<5, true, kFusedWaves, 2, kGemmTiles>), fgrid,
            fblock, 0, stream, g);
        break;
      case 6:
        hipLaunchKernelGGL(
            (Exl3GemmKernel<6, true, kFusedWaves, 2, kGemmTiles>), fgrid,
            fblock, 0, stream, g);
        break;
      default:
        return false;
    }
    return hipGetLastError() == hipSuccess;
  }
  // Transformed inputs once per row, with that row's expert suh.
  LinearArgs in{w, x,      out.x_half, ldx,     x_div, ids,    nullptr,
                0, x_rows, scratch.xh, nullptr, 1,     nullptr};
  const std::uint32_t in_items = x_rows * (w.in / 128);
  hipLaunchKernelGGL(Exl3InputKernel, dim3((in_items + kWaves - 1) / kWaves),
                     dim3(kBlockThreads), 0, stream, in);
  const GemmArgs g{w, scratch.xh, w.in, out, y, ldy, rows, route};
  const dim3 block(kGemmWaves * 32);
  const dim3 grid((w.out + 127) / 128, row_tiles);
  switch (w.k) {
    case 2:
      hipLaunchKernelGGL(
          (Exl3GemmKernel<2, false, kGemmWaves, 1, kTransformedTiles>), grid,
          block, 0, stream, g);
      break;
    case 3:
      hipLaunchKernelGGL(
          (Exl3GemmKernel<3, false, kGemmWaves, 1, kTransformedTiles>), grid,
          block, 0, stream, g);
      break;
    case 4:
      hipLaunchKernelGGL(
          (Exl3GemmKernel<4, false, kGemmWaves, 1, kTransformedTiles>), grid,
          block, 0, stream, g);
      break;
    case 5:
      hipLaunchKernelGGL(
          (Exl3GemmKernel<5, false, kGemmWaves, 1, kTransformedTiles>), grid,
          block, 0, stream, g);
      break;
    case 6:
      hipLaunchKernelGGL(
          (Exl3GemmKernel<6, false, kGemmWaves, 1, kTransformedTiles>), grid,
          block, 0, stream, g);
      break;
    default:
      return false;
  }
  return hipGetLastError() == hipSuccess;
}

bool Exl3DequantizeQ8(const Exl3Weights& w, std::uint32_t expert, void* dst,
                      hipStream_t stream) {
  if (w.tiles == nullptr || w.in % 128 != 0 || w.out % 128 != 0 ||
      expert >= w.experts)
    return false;
  const dim3 grid(w.out / 128, w.in / 128);
  auto* out = static_cast<std::uint8_t*>(dst);
  switch (w.k) {
    case 2:
      hipLaunchKernelGGL(Exl3DequantQ8Kernel<2>, grid, dim3(kBlockThreads), 0,
                         stream, w, expert, out);
      break;
    case 3:
      hipLaunchKernelGGL(Exl3DequantQ8Kernel<3>, grid, dim3(kBlockThreads), 0,
                         stream, w, expert, out);
      break;
    case 4:
      hipLaunchKernelGGL(Exl3DequantQ8Kernel<4>, grid, dim3(kBlockThreads), 0,
                         stream, w, expert, out);
      break;
    case 5:
      hipLaunchKernelGGL(Exl3DequantQ8Kernel<5>, grid, dim3(kBlockThreads), 0,
                         stream, w, expert, out);
      break;
    case 6:
      hipLaunchKernelGGL(Exl3DequantQ8Kernel<6>, grid, dim3(kBlockThreads), 0,
                         stream, w, expert, out);
      break;
    default:
      return false;
  }
  return hipGetLastError() == hipSuccess;
}

void Exl3Repack(const std::uint8_t* base, const std::uint64_t* src_offsets,
                std::uint32_t* dst, std::uint32_t src_in, std::uint32_t src_out,
                std::uint32_t in, std::uint32_t out, int k,
                std::uint32_t experts, std::size_t dst_stride,
                const std::uint32_t* in_blocks, const std::uint32_t* out_blocks,
                hipStream_t stream) {
  (void)src_in;
  hipLaunchKernelGGL(Exl3RepackKernel, dim3(2048), dim3(256), 0, stream, base,
                     src_offsets, dst, src_out, in, out, k, experts, dst_stride,
                     in_blocks, out_blocks);
}

void Exl3GatherScales(const std::uint8_t* base, const std::uint64_t* offsets,
                      std::uint32_t count, std::uint32_t src_len,
                      const std::uint32_t* blocks, std::uint32_t dst_len,
                      __half* dst, hipStream_t stream) {
  hipLaunchKernelGGL(Exl3GatherScalesKernel, dim3(256), dim3(256), 0, stream,
                     base, offsets, count, src_len, blocks, dst_len, dst);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
