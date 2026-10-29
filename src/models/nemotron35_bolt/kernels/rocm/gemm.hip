#include "src/models/nemotron35_bolt/kernels/rocm/gemm.hpp"

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

#include "src/models/nemotron35_bolt/kernels/rocm/iq2xxs.hip.hpp"

namespace gufo::models::nemotron35_bolt::rocm {
namespace {

using V16h = _Float16 __attribute__((ext_vector_type(16)));
using V8f = float __attribute__((ext_vector_type(8)));
/// Native vectors: arrays of HIP's uint4 struct can end up in scratch.
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

// Tile: kRows weight rows by kTokens tokens (a template argument), kSteps
// 16-value K steps staged per pass, waves in a kWaveRows x kWaveTokens grid.
constexpr int kRows = kNvfp4GemmRows;
constexpr int kSteps = 2;
constexpr int kThreads = 256;
constexpr int kRowTiles = kRows / 16;
/// Halves per staged 16-value row: payload plus bank padding.
constexpr int kStride = 16 + 8;
constexpr int kBlockBytes = 36;  ///< NVFP4: four scales, 32 nibble bytes.
static_assert(kRows * kSteps == kThreads, "a weight sub-row per thread");

/// Unsigned E4M3 scale times 0.5, as ggml_ue4m3_to_fp32; 0x7F reads 0.
__device__ __forceinline__ float Ue4m3Half(std::uint32_t x) {
  const std::uint32_t e = (x >> 3) & 15U;
  const std::uint32_t m = x & 7U;
  const float v = e == 0 ? static_cast<float>(m) * 0x1p-10F
                         : __uint_as_float(((e + 119U) << 23) | (m << 20));
  return x == 0x7FU ? 0.0F : v;
}

/// E2M1 code times two: {0, 1, 2, 3, 4, 6, 8, 12}, negated by bit 3.
__device__ __forceinline__ float E2m1x2(std::uint32_t code) {
  const std::uint32_t magnitude =
      __builtin_amdgcn_perm(0x0C080604U, 0x03020100U, code & 7U) & 0xFFU;
  const float v = static_cast<float>(magnitude);
  return (code & 8U) != 0 ? -v : v;
}

__device__ __forceinline__ std::uint32_t PackHalves(float lo, float hi) {
  return static_cast<std::uint32_t>(__half_as_ushort(__float2half(lo))) |
         (static_cast<std::uint32_t>(__half_as_ushort(__float2half(hi))) << 16);
}

__device__ __forceinline__ V16h LoadFragment(const _Float16* p) {
  union {
    V16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

template<int kTokens, int kWaveRows>
__global__ __launch_bounds__(kThreads) void Nvfp4GemmKernel(
    const std::uint8_t* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ out, int rows, int cols, int n_tokens) {
  constexpr int kWaveTokens = kThreads / 32 / kWaveRows;
  constexpr int kTokenTiles = kTokens / 16;
  constexpr int kWaveRowTiles = kRowTiles / kWaveRows;
  constexpr int kWaveTokenTiles = kTokenTiles / kWaveTokens;
  /// Activation sub-rows (16 values) each thread stages per pass.
  constexpr int kTokenSubs = kTokens * kSteps / kThreads;
  __shared__ _Float16 s_w[kSteps][kRowTiles][16][kStride];
  __shared__ _Float16 s_x[kSteps][kTokenTiles][16][kStride];
  static_assert(sizeof(s_w) >= kThreads / 32 * 256 * sizeof(float),
                "the weight stage doubles as the store transpose scratch");
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int sl = lane % 16;
  const int wr = wave / kWaveTokens;
  const int wt = wave % kWaveTokens;
  // Consecutive blocks share a token tile, which stays cached, and stream
  // the compact weights.
  const int r_block = static_cast<int>(blockIdx.x) * kRows;
  const int t_block = static_cast<int>(blockIdx.y) * kTokens;
  const int steps = cols / 16;

  // Staged sub-rows: weight row r_block + tid / kSteps, tokens likewise;
  // tokens past the chunk reread the last one and are dropped at the store.
  const int s_row = tid / kSteps;
  const int s_step = tid % kSteps;
  const std::uint8_t* w_row =
      w + static_cast<std::size_t>(r_block + s_row) * (cols / 64) * kBlockBytes;
  const __half* x_row[kTokenSubs];
#pragma unroll
  for (int q = 0; q < kTokenSubs; ++q) {
    const int t = min(t_block + s_row + q * (kThreads / kSteps), n_tokens - 1);
    x_row[q] = x + static_cast<std::size_t>(t) * cols;
  }

  // The next stage's weights and activations load into registers while the
  // current stage feeds the matrix units.
  std::uint32_t scale = 0;
  uint2 q8{};
  U32x4 xv[kTokenSubs][2];
  const auto load = [&](int k0) {
    const int k = k0 + s_step;  // 16-value step: NVFP4 sub-block k % 4
    const std::uint8_t* block = w_row + (k / 4) * kBlockBytes;
    scale = block[k % 4];
    q8 = *reinterpret_cast<const uint2*>(block + 4 + 8 * (k % 4));
#pragma unroll
    for (int q = 0; q < kTokenSubs; ++q) {
      const U32x4* xs = reinterpret_cast<const U32x4*>(x_row[q] + k * 16);
      xv[q][0] = xs[0];
      xv[q][1] = xs[1];
    }
  };
  load(0);
  V8f acc[kWaveRowTiles][kWaveTokenTiles] = {};
  for (int k0 = 0; k0 < steps; k0 += kSteps) {
    __syncthreads();
    {
      // Byte j holds value j (low nibble) and j + 8 (high nibble).
      const float d = Ue4m3Half(scale);
      const std::uint32_t bytes[2] = {q8.x, q8.y};
      std::uint32_t h[8];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const std::uint32_t b0 = (bytes[p / 2] >> (16 * (p % 2))) & 0xFFU;
        const std::uint32_t b1 = (bytes[p / 2] >> (16 * (p % 2) + 8)) & 0xFFU;
        h[p] = PackHalves(E2m1x2(b0 & 15U) * d, E2m1x2(b1 & 15U) * d);
        h[p + 4] = PackHalves(E2m1x2(b0 >> 4) * d, E2m1x2(b1 >> 4) * d);
      }
      auto* dst = reinterpret_cast<uint4*>(
          &s_w[s_step][s_row / 16][s_row % 16][0]);
      dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
      dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
#pragma unroll
      for (int q = 0; q < kTokenSubs; ++q) {
        const int t = s_row + q * (kThreads / kSteps);
        auto* xd = reinterpret_cast<U32x4*>(&s_x[s_step][t / 16][t % 16][0]);
        xd[0] = xv[q][0];
        xd[1] = xv[q][1];
      }
    }
    __syncthreads();
    if (k0 + kSteps < steps)
      load(k0 + kSteps);
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
      V16h a[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i)
        a[i] = LoadFragment(&s_w[s][wr * kWaveRowTiles + i][sl][0]);
#pragma unroll
      for (int j = 0; j < kWaveTokenTiles; ++j) {
        const V16h b = LoadFragment(&s_x[s][wt * kWaveTokenTiles + j][sl][0]);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b,
                                                                 acc[i][j]);
      }
    }
  }

  // C element l of a wave32 tile is row 2l + lane / 16 of the weight tile
  // and token lane % 16; out is [token][row], so transpose through the dead
  // weight stage.
  __syncthreads();
  const int hi = lane / 16;
  float* scratch = reinterpret_cast<float*>(&s_w[0][0][0][0]) + wave * 256;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokenTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l)
        scratch[sl * 16 + 2 * l + hi] = acc[i][j][l];
      __builtin_amdgcn_wave_barrier();
      const int r0 = r_block + (wr * kWaveRowTiles + i) * 16;
      const int t0 = t_block + (wt * kWaveTokenTiles + j) * 16;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = s * 32 + lane;
        const int t = t0 + flat / 16;
        if (t < n_tokens)
          out[static_cast<std::size_t>(t) * rows + r0 + flat % 16] =
              scratch[flat];
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace

hipError_t Nvfp4Gemm(const void* weights, const __half* x, float* out,
                     int rows, int cols, int n_tokens, hipStream_t stream) {
  if (rows <= 0 || rows % kRows != 0 || cols <= 0 || cols % 64 != 0 ||
      n_tokens < 1)
    return hipErrorInvalidValue;
  const auto* w = static_cast<const std::uint8_t*>(weights);
  const unsigned row_blocks = static_cast<unsigned>(rows / kRows);
  // Tall matrices take 256-token tiles: each weight tile feeds twice the
  // tokens, and enough blocks remain to fill the GPU.
  if (row_blocks >= 64) {
    const dim3 grid(row_blocks, static_cast<unsigned>((n_tokens + 255) / 256));
    Nvfp4GemmKernel<256, 4>
        <<<grid, kThreads, 0, stream>>>(w, x, out, rows, cols, n_tokens);
  } else {
    const dim3 grid(row_blocks, static_cast<unsigned>((n_tokens + 127) / 128));
    Nvfp4GemmKernel<128, 2>
        <<<grid, kThreads, 0, stream>>>(w, x, out, rows, cols, n_tokens);
  }
  return hipGetLastError();
}


namespace {

// Routed IQ2_XXS products on BF16 WMMA. A pass stages one 32-value group
// per weight row (two 16-value WMMA steps).
using Bf16Fragment = unsigned short __attribute__((ext_vector_type(16)));
using Accumulator = float __attribute__((ext_vector_type(8)));
constexpr int kIqThreads = 256;
constexpr int kIqStride = 16 + 8;  ///< Halves per staged row, with padding.

__device__ __forceinline__ Bf16Fragment LoadBf16Fragment(
    const std::uint16_t* p) {
  union {
    Bf16Fragment f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

struct Iq2xxsKernelArgs {
  const std::uint8_t* weights;
  const uint4* grid;
  const float* x;
  float* out;
  int rows;
  int cols;
  bool relu2;
  const std::int32_t* order;
  const int4* tiles;
  const std::int32_t* num_tiles;
  int used;
  bool x_per_slot;
};

/// A block multiplies kIq2xxsRoutedRows weight rows of one expert with up
/// to kIq2xxsRoutedTokens of its slots. A thread owns a weight row: it loads
/// the row's super-block (66 bytes) once for its eight 32-value groups, and
/// stages each group's exact signed grid values (BF16) and its scale. Every
/// wave takes two row tiles and both token tiles; a group's two WMMA steps
/// sum into a fresh tile that is scaled in F32 into the accumulator, as the
/// vector kernel scales each group's dot product.
__global__ __launch_bounds__(kIqThreads) void Iq2xxsRoutedKernel(
    Iq2xxsKernelArgs a) {
  constexpr int kRowTiles = kIq2xxsRoutedRows / 16;
  constexpr int kTokenTiles = kIq2xxsRoutedTokens / 16;
  constexpr int kWaveRowTiles = kRowTiles / (kIqThreads / 32);
  static_assert(kIq2xxsRoutedRows == kIqThreads, "a thread per weight row");
  __shared__ std::uint16_t s_w[2][kRowTiles][16][kIqStride];
  __shared__ std::uint16_t s_x[2][kTokenTiles][16][kIqStride];
  /// Group scales by C layout: [row tile][lane / 16][element].
  __shared__ float s_db[kRowTiles][2][8];
  __shared__ uint4 s_grid[256];
  static_assert(sizeof(s_w) >= kIqThreads / 32 * 256 * sizeof(float),
                "the weight stage doubles as the store transpose scratch");
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int sl = lane % 16;
  const int hi = lane / 16;
  if (static_cast<int>(blockIdx.y) >= *a.num_tiles)
    return;
  const int4 tile = a.tiles[blockIdx.y];
  const int expert = tile.x;
  const int first = tile.y;
  const int count = tile.z;
  s_grid[tid] = a.grid[tid];
  const int r_block = static_cast<int>(blockIdx.x) * kIq2xxsRoutedRows;
  const int blocks = a.cols / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kIq2xxsBlockBytes;
  const std::uint8_t* w_row =
      a.weights +
      (static_cast<std::size_t>(expert) * a.rows + r_block + tid) * row_bytes;

  // Token tid / 2, columns tid % 2 * 16 of a group; columns past the tile
  // reread its last slot and are dropped at the store.
  const bool stages_x = tid < 2 * kIq2xxsRoutedTokens;
  const int slot = a.order[first + min(tid / 2, count - 1)];
  const float* x_row =
      a.x + static_cast<std::size_t>(a.x_per_slot ? slot : slot / a.used) *
                a.cols +
      (tid % 2) * 16;

  // This row's super-block: its scale and the eight groups' index and
  // sign/scale words; the next one loads while these are staged.
  float d = 0.0F;
  uint4 q[4];
  const auto load_block = [&](int sb, float* d_out, uint4* q_out) {
    const std::uint8_t* block = w_row + static_cast<std::size_t>(sb) *
                                            kIq2xxsBlockBytes;
    *d_out = __half2float(*reinterpret_cast<const __half*>(block));
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q_out[i] = reinterpret_cast<const uint4*>(block + 2)[i];
  };
  float4 xv[4];
  const auto load_x = [&](int pass) {
    if (stages_x) {
      const float4* xs = reinterpret_cast<const float4*>(x_row + pass * 32);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        xv[i] = xs[i];
    }
  };
  load_block(0, &d, q);
  load_x(0);
  Accumulator acc[kWaveRowTiles][kTokenTiles] = {};
  for (int sb = 0; sb < blocks; ++sb) {
    float d_next = 0.0F;
    uint4 q_next[4];
    if (sb + 1 < blocks)
      load_block(sb + 1, &d_next, q_next);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int pass = sb * 8 + j;
      const std::uint32_t index = j % 2 == 0 ? q[j / 2].x : q[j / 2].z;
      const std::uint32_t aux = j % 2 == 0 ? q[j / 2].y : q[j / 2].w;
      __syncthreads();
      {
        const int r = tid % 16;
        s_db[tid / 16][r % 2][r / 2] =
            d * (0.5F + static_cast<float>(aux >> 28)) * 0.25F;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          std::uint32_t words[8];
#pragma unroll
          for (int g = 0; g < 2; ++g) {
            const int e = 2 * h + g;
            const SignedGrid sg(s_grid[(index >> (8 * e)) & 0xFFU],
                                (aux >> (7 * e)) & 127U);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              words[4 * g + i] = sg.w[i];
          }
          auto* dst = reinterpret_cast<uint4*>(&s_w[h][tid / 16][tid % 16][0]);
          dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
          dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
        }
      }
      if (stages_x) {
        const int token = tid / 2;
        auto* dst = reinterpret_cast<uint4*>(
            &s_x[tid % 2][token / 16][token % 16][0]);
        dst[0] =
            make_uint4(PackBf16(xv[0].x, xv[0].y), PackBf16(xv[0].z, xv[0].w),
                       PackBf16(xv[1].x, xv[1].y), PackBf16(xv[1].z, xv[1].w));
        dst[1] =
            make_uint4(PackBf16(xv[2].x, xv[2].y), PackBf16(xv[2].z, xv[2].w),
                       PackBf16(xv[3].x, xv[3].y), PackBf16(xv[3].z, xv[3].w));
      }
      __syncthreads();
      if (pass + 1 < blocks * 8)
        load_x(pass + 1);
      Bf16Fragment b[2][kTokenTiles];
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int t = 0; t < kTokenTiles; ++t)
          b[s][t] = LoadBf16Fragment(&s_x[s][t][sl][0]);
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rt = wave * kWaveRowTiles + i;
        const Bf16Fragment f0 = LoadBf16Fragment(&s_w[0][rt][sl][0]);
        const Bf16Fragment f1 = LoadBf16Fragment(&s_w[1][rt][sl][0]);
        const float4 db0 = *reinterpret_cast<const float4*>(&s_db[rt][hi][0]);
        const float4 db1 = *reinterpret_cast<const float4*>(&s_db[rt][hi][4]);
        const float db[8] = {db0.x, db0.y, db0.z, db0.w,
                             db1.x, db1.y, db1.z, db1.w};
#pragma unroll
        for (int t = 0; t < kTokenTiles; ++t) {
          Accumulator sum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
              f0, b[0][t], Accumulator{});
          sum = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(f1, b[1][t], sum);
#pragma unroll
          for (int l = 0; l < 8; ++l)
            acc[i][t][l] = fmaf(db[l], sum[l], acc[i][t][l]);
        }
      }
    }
    d = d_next;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q[i] = q_next[i];
  }

  // C element l of a wave32 tile is row 2l + lane / 16 and token lane % 16;
  // out rows are slots, so transpose through the dead weight stage.
  __syncthreads();
  float* scratch = reinterpret_cast<float*>(&s_w[0][0][0][0]) + wave * 256;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int t = 0; t < kTokenTiles; ++t) {
#pragma unroll
      for (int l = 0; l < 8; ++l)
        scratch[sl * 16 + 2 * l + hi] = acc[i][t][l];
      __builtin_amdgcn_wave_barrier();
      const int r0 = r_block + (wave * kWaveRowTiles + i) * 16;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = s * 32 + lane;
        const int c = t * 16 + flat / 16;
        if (c < count) {
          float v = scratch[flat];
          if (a.relu2) {
            v = fmaxf(v, 0.0F);
            v *= v;
          }
          a.out[static_cast<std::size_t>(a.order[first + c]) * a.rows + r0 +
                flat % 16] = v;
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

__global__ void RouteCountKernel(const std::int32_t* __restrict__ ids,
                                 int slots, std::int32_t* __restrict__ counts) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s < slots)
    atomicAdd(&counts[ids[s]], 1);
}

/// One block: offsets of every expert's slots and its tiles; the counts
/// become the scatter cursors.
__global__ __launch_bounds__(1024) void RouteTilesKernel(
    int experts, std::int32_t* __restrict__ cursor, int4* __restrict__ tiles,
    std::int32_t* __restrict__ num_tiles) {
  __shared__ int s_slots[1024];
  __shared__ int s_tiles[1024];
  const int e = static_cast<int>(threadIdx.x);
  const int count = e < experts ? cursor[e] : 0;
  const int n_tiles =
      (count + kIq2xxsRoutedTokens - 1) / kIq2xxsRoutedTokens;
  s_slots[e] = count;
  s_tiles[e] = n_tiles;
  __syncthreads();
  // Inclusive scans.
  for (int stride = 1; stride < 1024; stride *= 2) {
    const int add_slots = e >= stride ? s_slots[e - stride] : 0;
    const int add_tiles = e >= stride ? s_tiles[e - stride] : 0;
    __syncthreads();
    s_slots[e] += add_slots;
    s_tiles[e] += add_tiles;
    __syncthreads();
  }
  const int slot0 = s_slots[e] - count;
  const int tile0 = s_tiles[e] - n_tiles;
  if (e < experts) {
    cursor[e] = slot0;
    for (int k = 0; k < n_tiles; ++k)
      tiles[tile0 + k] = make_int4(e, slot0 + k * kIq2xxsRoutedTokens,
                                   min(kIq2xxsRoutedTokens,
                                       count - k * kIq2xxsRoutedTokens),
                                   0);
  }
  if (e == 1023)
    *num_tiles = s_tiles[e];
}

__global__ void RouteScatterKernel(const std::int32_t* __restrict__ ids,
                                   int slots, std::int32_t* __restrict__ cursor,
                                   std::int32_t* __restrict__ order) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s < slots)
    order[atomicAdd(&cursor[ids[s]], 1)] = s;
}

}  // namespace

int Iq2xxsRouteMaxTiles(int slots, int experts) {
  return slots / kIq2xxsRoutedTokens + experts;
}

hipError_t Iq2xxsRouteSlots(const std::int32_t* ids, int slots, int experts,
                            const Iq2xxsRoute& route, hipStream_t stream) {
  if (slots <= 0 || experts <= 0 || experts > 1024 ||
      route.max_tiles < Iq2xxsRouteMaxTiles(slots, experts))
    return hipErrorInvalidValue;
  const hipError_t cleared = hipMemsetAsync(
      route.cursor, 0, static_cast<std::size_t>(experts) * sizeof(std::int32_t),
      stream);
  if (cleared != hipSuccess)
    return cleared;
  const unsigned blocks = static_cast<unsigned>((slots + 255) / 256);
  RouteCountKernel<<<blocks, 256, 0, stream>>>(ids, slots, route.cursor);
  RouteTilesKernel<<<1, 1024, 0, stream>>>(experts, route.cursor, route.tiles,
                                           route.num_tiles);
  RouteScatterKernel<<<blocks, 256, 0, stream>>>(ids, slots, route.cursor,
                                                 route.order);
  return hipGetLastError();
}

hipError_t Iq2xxsRoutedGemm(const Iq2xxsGemmArgs& args, hipStream_t stream) {
  if (args.route == nullptr || args.rows <= 0 ||
      args.rows % kIq2xxsRoutedRows != 0 || args.cols <= 0 ||
      args.cols % 256 != 0 || args.used < 1)
    return hipErrorInvalidValue;
  const Iq2xxsKernelArgs k{
      .weights = static_cast<const std::uint8_t*>(args.weights),
      .grid = reinterpret_cast<const uint4*>(args.grid),
      .x = args.x,
      .out = args.out,
      .rows = args.rows,
      .cols = args.cols,
      .relu2 = args.relu2,
      .order = args.route->order,
      .tiles = args.route->tiles,
      .num_tiles = args.route->num_tiles,
      .used = args.used,
      .x_per_slot = args.x_per_slot,
  };
  const dim3 grid(static_cast<unsigned>(args.rows / kIq2xxsRoutedRows),
                  static_cast<unsigned>(args.route->max_tiles));
  Iq2xxsRoutedKernel<<<grid, kIqThreads, 0, stream>>>(k);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm
