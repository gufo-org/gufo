#include "src/models/nemotron35_bolt/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>

#include <cstddef>

#include "src/models/nemotron35_bolt/kernels/rocm/iq2xxs.hip.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/lanes.hip.hpp"

namespace gufo::models::nemotron35_bolt::rocm {
namespace {


constexpr int kLanes = 32;
constexpr int kWaves = 8;
constexpr int kThreads = kLanes * kWaves;
/// Rows each wave streams; they share the activations held in registers.
constexpr int kRowsPerWave = 4;
constexpr int kRowsPerBlock = kWaves * kRowsPerWave;
/// IQ2_XXS super-blocks (256 columns) of activations staged per pass.
constexpr int kChunkBlocks = 8;

/// Each wave streams kRowsPerWave rows and keeps one activation slice per
/// token in registers for all of them. With kGroups, a lane owns a whole
/// 32-value group and a wave reads four consecutive super-blocks (264
/// contiguous bytes) of a row per step, which needs a multiple of four
/// super-blocks per row; otherwise a lane owns eight values, group lane / 4
/// and grid entry lane % 4, of one super-block per step. A routed block
/// serves one (token, slot) pair, a dense block every token.
template<int kTokens, bool kGroups>
__global__ __launch_bounds__(kThreads) void Iq2xxsGemvKernel(
    Iq2xxsGemvArgs a) {
  constexpr int kStepBlocks = kGroups ? kLanes / 8 : 1;
  // Several tokens stage fewer super-blocks, so more blocks fit per CU.
  constexpr int kChunk = kTokens > 1 ? kChunkBlocks / 2 : kChunkBlocks;
  static_assert(kChunk % kStepBlocks == 0);
  static_assert(kThreads == 256, "one grid entry per thread");
  __shared__ uint4 s_grid[256];
  __shared__ uint4 s_x[kTokens][kChunk * kLanes];
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kLanes;
  const int lane = tid % kLanes;
  const bool routed = a.ids != nullptr;
  const int slot = static_cast<int>(blockIdx.y);
  const int expert = routed ? a.ids[slot] : 0;
  const int blocks = a.cols / 256;
  const std::size_t row_bytes =
      static_cast<std::size_t>(blocks) * kIq2xxsBlockBytes;
  const auto* w = static_cast<const std::uint8_t*>(a.weights) +
                  static_cast<std::size_t>(expert) * a.rows * row_bytes;
  const int row0 =
      static_cast<int>(blockIdx.x) * kRowsPerBlock + wave * kRowsPerWave;
  const int x_row0 = routed ? (a.x_per_slot ? slot : slot / a.used) : 0;
  const int x_rows = routed ? 1 : a.n_tokens;
  s_grid[tid] = reinterpret_cast<const uint4*>(a.grid)[tid];
  float acc[kRowsPerWave][kTokens] = {};
  for (int b0 = 0; b0 < blocks; b0 += kChunk) {
    const int nb = min(kChunk, blocks - b0);
    __syncthreads();
    // Absent tokens stage zeros, so the loops below have fixed trip counts.
    for (int i = tid; i < kTokens * kChunk * kLanes; i += kThreads) {
      const int t = i / (kChunk * kLanes);
      const int v = i % (kChunk * kLanes);
      uint4 packed = make_uint4(0U, 0U, 0U, 0U);
      if (t < x_rows && v < nb * kLanes) {
        const float* src = a.x +
                           static_cast<std::size_t>(x_row0 + t) * a.cols +
                           static_cast<std::size_t>(b0) * 256 + v * 8;
        const float4 lo = *reinterpret_cast<const float4*>(src);
        const float4 hi = *reinterpret_cast<const float4*>(src + 4);
        packed = make_uint4(PackBf16(lo.x, lo.y), PackBf16(lo.z, lo.w),
                            PackBf16(hi.x, hi.y), PackBf16(hi.z, hi.w));
      }
      s_x[t][v] = packed;
    }
    __syncthreads();
    for (int s0 = 0; s0 < nb; s0 += kStepBlocks) {
      if constexpr (kGroups && kTokens == 1) {
        const int bl = s0 + lane / 8;
        const int group = lane % 8;
        uint4 xv[kTokens][4];
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            xv[t][e] = s_x[t][bl * kLanes + group * 4 + e];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + bl) * kIq2xxsBlockBytes;
          const auto* q16 =
              reinterpret_cast<const std::uint16_t*>(block + 2 + 8 * group);
          const std::uint32_t index =
              q16[0] | (static_cast<std::uint32_t>(q16[1]) << 16);
          const std::uint32_t aux =
              q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          const float db = GroupScale(block, aux);
          float dot[kTokens] = {};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const SignedGrid g(s_grid[(index >> (8 * e)) & 0xFFU],
                               (aux >> (7 * e)) & 127U);
#pragma unroll
            for (int t = 0; t < kTokens; ++t)
              dot[t] = g.Dot(xv[t][e], dot[t]);
          }
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db, dot[t], acc[r][t]);
        }
      } else if constexpr (kGroups) {
        const int bl = s0 + lane / 8;
        const int group = lane % 8;
        // Several tokens: per grid entry, every row against the tokens'
        // activations, so one entry's slice is in registers instead of
        // four. The sums keep the one-token order.
        std::uint32_t index[kRowsPerWave], aux[kRowsPerWave];
        float db[kRowsPerWave];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + bl) * kIq2xxsBlockBytes;
          const auto* q16 =
              reinterpret_cast<const std::uint16_t*>(block + 2 + 8 * group);
          index[r] = q16[0] | (static_cast<std::uint32_t>(q16[1]) << 16);
          aux[r] = q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          db[r] = GroupScale(block, aux[r]);
        }
        float dot[kRowsPerWave][kTokens] = {};
#pragma unroll 1
        for (int e = 0; e < 4; ++e) {
          uint4 xv[kTokens];
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            xv[t] = s_x[t][bl * kLanes + group * 4 + e];
#pragma unroll
          for (int r = 0; r < kRowsPerWave; ++r) {
            const SignedGrid g(s_grid[(index[r] >> (8 * e)) & 0xFFU],
                               (aux[r] >> (7 * e)) & 127U);
#pragma unroll
            for (int t = 0; t < kTokens; ++t)
              dot[r][t] = g.Dot(xv[t], dot[r][t]);
          }
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r)
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db[r], dot[r][t], acc[r][t]);
      } else {
        const int group = lane / 4;
        const int part = lane % 4;
        uint4 xv[kTokens];
#pragma unroll
        for (int t = 0; t < kTokens; ++t)
          xv[t] = s_x[t][s0 * kLanes + lane];
#pragma unroll
        for (int r = 0; r < kRowsPerWave; ++r) {
          const std::uint8_t* block =
              w + static_cast<std::size_t>(row0 + r) * row_bytes +
              static_cast<std::size_t>(b0 + s0) * kIq2xxsBlockBytes;
          const std::uint8_t* q = block + 2 + 8 * group;
          const auto* q16 = reinterpret_cast<const std::uint16_t*>(q);
          const std::uint32_t aux =
              q16[2] | (static_cast<std::uint32_t>(q16[3]) << 16);
          const float db = GroupScale(block, aux);
          const SignedGrid g(s_grid[q[part]], (aux >> (7 * part)) & 127U);
#pragma unroll
          for (int t = 0; t < kTokens; ++t)
            acc[r][t] = fmaf(db, g.Dot(xv[t], 0.0F), acc[r][t]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRowsPerWave; ++r) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t)
      acc[r][t] = WaveSum(acc[r][t]);
  }
  if (lane != 0)
    return;
#pragma unroll
  for (int t = 0; t < kTokens; ++t) {
    if (t >= x_rows)
      break;
    float* out = a.out + static_cast<std::size_t>(routed ? slot : t) * a.rows;
#pragma unroll
    for (int r = 0; r < kRowsPerWave; ++r) {
      float v = acc[r][t];
      if (a.relu2) {
        v = fmaxf(v, 0.0F);
        v *= v;
      }
      out[row0 + r] = a.accumulate ? out[row0 + r] + v : v;
    }
  }
}

template<int kTokens>
void LaunchIq2xxs(const Iq2xxsGemvArgs& args, dim3 grid, hipStream_t stream) {
  if ((args.cols / 256) % (kLanes / 8) == 0)
    Iq2xxsGemvKernel<kTokens, true><<<grid, kThreads, 0, stream>>>(args);
  else
    Iq2xxsGemvKernel<kTokens, false><<<grid, kThreads, 0, stream>>>(args);
}

}  // namespace

void Iq2xxsGridBf16(const std::uint64_t* grid, std::uint32_t* words) {
  for (int e = 0; e < 256; ++e) {
    for (int p = 0; p < 4; ++p) {
      // Magnitudes are small integers, so their BF16 encoding is the upper
      // half of the F32 one.
      std::uint32_t pair = 0;
      for (int h = 0; h < 2; ++h) {
        const auto m = static_cast<float>((grid[e] >> (8 * (2 * p + h))) & 0xFF);
        pair |= (__builtin_bit_cast(std::uint32_t, m) >> 16) << (16 * h);
      }
      words[e * 4 + p] = pair;
    }
  }
}

hipError_t Iq2xxsGemv(const Iq2xxsGemvArgs& args, hipStream_t stream) {
  if (args.rows <= 0 || args.rows % kRowsPerBlock != 0 || args.cols <= 0 ||
      args.cols % 256 != 0 || args.n_tokens < 1 ||
      args.n_tokens > kGemvMaxTokens || args.used < 1)
    return hipErrorInvalidValue;
  const auto row_blocks = static_cast<unsigned>(args.rows / kRowsPerBlock);
  if (args.ids != nullptr) {
    LaunchIq2xxs<1>(args,
                    dim3(row_blocks,
                         static_cast<unsigned>(args.n_tokens * args.used)),
                    stream);
  } else if (args.n_tokens == 1) {
    LaunchIq2xxs<1>(args, dim3(row_blocks), stream);
  } else {
    LaunchIq2xxs<kGemvMaxTokens>(args, dim3(row_blocks), stream);
  }
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm
