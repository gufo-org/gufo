#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_PREFILL_QUANT_GEMM_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_PREFILL_QUANT_GEMM_HPP_
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// Tiled Q8_1 activation layout (opt-c163-blocked-w8a8).
//
// Activations are written straight into WMMA B-fragment order so both the LDS
// staging copy and the fragment reads are contiguous:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block. The 576-byte stride keeps
// every fragment and scale vector 16-byte aligned. For a token count that is a
// multiple of 16 this is exactly the same total size as the previous row-major
// block_q8_1 payload (36 bytes per 32 elements), followed by the sum sidecar.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
constexpr std::size_t kQ8ActSumTileBytes = kQ8ActTileTokens * sizeof(float);

/// Byte size of the tiled Q8_1 activation buffer for a [batch, k] tensor.
__host__ __device__ inline std::size_t Q8ActTiledBytes(std::size_t batch,
                                                       std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return tiles * (k / 32) * kQ8ActTileBytes;
}

/// Byte size of the per-token activation-sum sidecar. Keeping it
/// after the tiled payload leaves the Q8_1 layout and every Q8 kernel
/// unchanged.
__host__ __device__ inline std::size_t Q8ActSumBytes(std::size_t batch,
                                                     std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return tiles * (k / 32) * kQ8ActSumTileBytes;
}

__device__ __forceinline__ float* Q8ActSumSidecar(void* base, std::size_t batch,
                                                  std::size_t k) {
  return reinterpret_cast<float*>(static_cast<std::uint8_t*>(base) +
                                  Q8ActTiledBytes(batch, k));
}

__device__ __forceinline__ const float* Q8ActSumSidecar(const void* base,
                                                        std::size_t batch,
                                                        std::size_t k) {
  return reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(base) +
                                        Q8ActTiledBytes(batch, k));
}

__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

__device__ __forceinline__ const std::int8_t* Q8ActTile(const void* base,
                                                        std::size_t num_blocks,
                                                        std::size_t tt,
                                                        std::size_t kb) {
  return static_cast<const std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// Writes one lane's quantized byte and (lane 0) the block scale into the tiled
/// layout. `lane_id` is the element index inside the 32-element block.
template<bool StoreActivationSum>
__device__ __forceinline__ void StoreQ8ActLane(void* y, std::size_t batch,
                                               std::size_t num_blocks,
                                               std::size_t tok, std::size_t blk,
                                               std::size_t lane_id, float d,
                                               std::int8_t q) {
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
  if constexpr (StoreActivationSum) {
    int qsum = static_cast<int>(q);
    for (int off = 16; off > 0; off >>= 1) {
      qsum += __shfl_xor(qsum, off);
    }
    if (lane_id == 0) {
      float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
      sums[(((tt * num_blocks) + blk) * kQ8ActTileTokens) + tl] =
          d * static_cast<float>(qsum);
    }
  }
}

template<int WaveSize>
using WmmaQuantAccumulator =
    std::conditional_t<WaveSize == 64, int32x4_t, int32x8_t>;

template<int WaveSize>
__device__ __forceinline__ WmmaQuantAccumulator<WaveSize> WmmaQuant(
    int32x4_t a, int32x4_t b, WmmaQuantAccumulator<WaveSize> c) {
  if constexpr (WaveSize == 64) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w64(true, a, true, b, c,
                                                      true);
  } else {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                      true);
  }
}
// Both wave sizes preserve every K32 accumulator update, including the affine
// minimum correction and the separate Q6/Q3 half-block scales. Activation
// staging uses 32-thread groups; the native WMMA wave determines C ownership.
template<int BM, int BN, int BK, int WM, int WN, core::GgmlType WType,
         int WaveSize = 32>
__launch_bounds__(WM * WN * WaveSize, 1) __global__
    void WKQuantA8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                        const void* __restrict__ x_blocks,
                                        float* __restrict__ y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k) {
  constexpr bool PerHalfScale =
      WType == core::GgmlType::kQ6_K || WType == core::GgmlType::kQ3_K;
  constexpr bool HasOffset =
      WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K;

  static_assert(WaveSize == 32 || WaveSize == 64);
  static_assert((WaveSize == 32 && WM * WN == 8) ||
                (WaveSize == 64 && WM * WN == 4));
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int kWaves = WM * WN;
  constexpr int kStageWaves = kThreads / 32;
  constexpr int kAccumulatorElements = 256 / WaveSize;
  using Accumulator = WmmaQuantAccumulator<WaveSize>;
  static_assert(BN / 16 <= kStageWaves || (BN / 16) % kStageWaves == 0,
                "token subtiles must divide evenly across the waves");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  constexpr int kScaleHalves = PerHalfScale ? 2 : 1;
  constexpr int kOffRowTiles = HasOffset ? kRowTiles : 1;

  constexpr int kStageTiles = (kTokTiles + kStageWaves - 1) / kStageWaves;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kScaleHalves][kRowTiles][16];
  __shared__ float s_off[BK][kOffRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];
  __shared__ float s_sx[HasOffset ? BK : 1][kTokTiles][16];

  const std::size_t num_blocks = k / 32;
  const std::size_t row_bytes = QuantRowBytes(WType, k);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid / WaveSize;
  const int stage_wave = tid >> 5;
  const int lane_id = tid % WaveSize;
  const int stage_lane = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][kAccumulatorElements];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < kAccumulatorElements; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  constexpr int kFetchUnits = BM * BK;
  constexpr int kPrefetch = (kFetchUnits + kThreads - 1) / kThreads;
  constexpr bool kPartialFetch = (kPrefetch * kThreads) != kFetchUnits;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw0[kPrefetch];
  float r_dw1[kPrefetch];
  float r_off[kPrefetch];
  int32x4_t r_b[kStageTiles][BK];
  float r_dx[kStageTiles][BK];
  float r_sx[kStageTiles][HasOffset ? BK : 1];

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const std::uint8_t* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int idx = (p * kThreads) + tid;
    const bool unit_live = !kPartialFetch || idx < kFetchUnits;
    const int idx_clamped = unit_live ? idx : 0;
    const int r = static_cast<int>(r_block) + (idx_clamped / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = static_cast<const std::uint8_t*>(w) +
               (static_cast<std::size_t>(r_clamped) * row_bytes);
    row_live[p] = (r < m_i && unit_live) ? 1.0F : 0.0F;
  }

  const std::size_t num_act_tiles =
      (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  int b_tile[kStageTiles];
  const std::int8_t* b_base[kStageTiles];
  std::size_t act_tile[kStageTiles];
  float tile_scale[kStageTiles];
  bool b_live[kStageTiles];
#pragma unroll
  for (int t = 0; t < kStageTiles; ++t) {
    b_tile[t] = stage_wave + (t * kStageWaves);
    const std::size_t wanted = tt_block + static_cast<std::size_t>(b_tile[t]);
    const bool live = (b_tile[t] < kTokTiles) && (wanted < num_act_tiles);
    act_tile[t] = live ? wanted : 0;
    tile_scale[t] = live ? 1.0F : 0.0F;
    b_live[t] = b_tile[t] < kTokTiles;
    b_base[t] = reinterpret_cast<const std::int8_t*>(x_blocks) +
                (act_tile[t] * num_blocks * kQ8ActTileBytes);
  }
  const float* activation_sums = Q8ActSumSidecar(x_blocks, batch, k);

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      const int idx_clamped = (!kPartialFetch || idx < kFetchUnits) ? idx : 0;
      const int kb = kb0 + (idx_clamped % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const float live = (kb < num_kb) ? row_live[p] : 0.0F;
      QuantSub16 lo;
      QuantSub16 hi;
      DecodeQuantSub16<WaveSize == 64>(
          WType, w_row[p], static_cast<std::size_t>(kb_clamped) * 2, lo);
      DecodeQuantSub16<WaveSize == 64>(
          WType, w_row[p], (static_cast<std::size_t>(kb_clamped) * 2) + 1, hi);
      __builtin_memcpy(&r_q0[p], lo.q, 16);
      __builtin_memcpy(&r_q1[p], hi.q, 16);
      r_dw0[p] = lo.scale * live;
      r_dw1[p] = hi.scale * live;
      if constexpr (HasOffset) {
        r_off[p] = lo.offset * live;
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile = b_base[t] + (static_cast<std::size_t>(kb_clamped) *
                                        kQ8ActTileBytes);
        r_b[t][i] = reinterpret_cast<const int32x4_t*>(tile)[stage_lane];
        r_dx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                     reinterpret_cast<const float*>(
                         tile + kQ8ActScaleOffset)[lane_id & 15];
        if constexpr (HasOffset) {
          r_sx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                       activation_sums[(((act_tile[t] * num_blocks) +
                                         static_cast<std::size_t>(kb_clamped)) *
                                        kQ8ActTileTokens) +
                                       static_cast<std::size_t>(lane_id & 15)];
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      if constexpr (kPartialFetch) {
        if (idx >= kFetchUnits) {
          continue;
        }
      }
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      const int slot = WaveSize == 32
                           ? ((rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2)))
                           : ((rl % 4) * 4 + rl / 4);
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][0][rs][slot] = r_dw0[p];
      if constexpr (PerHalfScale) {
        s_dw[kk][1][rs][slot] = r_dw1[p];
      }
      if constexpr (HasOffset) {
        s_off[kk][rs][slot] = r_off[p];
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile[t]][stage_lane] = r_b[t][i];
        if (stage_lane < 16) {
          s_dx[i][b_tile[t]][stage_lane] = r_dx[t][i];
          if constexpr (HasOffset) {
            s_sx[i][b_tile[t]][stage_lane] = r_sx[t][i];
          }
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw0[kWaveRowTiles][kAccumulatorElements];

      float dw1[PerHalfScale ? kWaveRowTiles : 1][kAccumulatorElements];
      float off[HasOffset ? kWaveRowTiles : 1][kAccumulatorElements];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo = *reinterpret_cast<const float4*>(
            &s_dw[kb][0][rs][half_id * kAccumulatorElements]);
        dw0[i][0] = lo.x;
        dw0[i][1] = lo.y;
        dw0[i][2] = lo.z;
        dw0[i][3] = lo.w;
        if constexpr (WaveSize == 32) {
          const float4 up = *reinterpret_cast<const float4*>(
              &s_dw[kb][0][rs][(half_id * kAccumulatorElements) + 4]);
          dw0[i][4] = up.x;
          dw0[i][5] = up.y;
          dw0[i][6] = up.z;
          dw0[i][7] = up.w;
        }
        if constexpr (PerHalfScale) {
          const float4 hlo = *reinterpret_cast<const float4*>(
              &s_dw[kb][1][rs][half_id * kAccumulatorElements]);
          dw1[i][0] = hlo.x;
          dw1[i][1] = hlo.y;
          dw1[i][2] = hlo.z;
          dw1[i][3] = hlo.w;
          if constexpr (WaveSize == 32) {
            const float4 hup = *reinterpret_cast<const float4*>(
                &s_dw[kb][1][rs][(half_id * kAccumulatorElements) + 4]);
            dw1[i][4] = hup.x;
            dw1[i][5] = hup.y;
            dw1[i][6] = hup.z;
            dw1[i][7] = hup.w;
          }
        }
        if constexpr (HasOffset) {
          const float4 olo = *reinterpret_cast<const float4*>(
              &s_off[kb][rs][half_id * kAccumulatorElements]);
          off[i][0] = olo.x;
          off[i][1] = olo.y;
          off[i][2] = olo.z;
          off[i][3] = olo.w;
          if constexpr (WaveSize == 32) {
            const float4 oup = *reinterpret_cast<const float4*>(
                &s_off[kb][rs][(half_id * kAccumulatorElements) + 4]);
            off[i][4] = oup.x;
            off[i][5] = oup.y;
            off[i][6] = oup.z;
            off[i][7] = oup.w;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];

        float sx = 0.0F;
        if constexpr (HasOffset) {
          sx = s_sx[kb][ts][sub_lane];
        }
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          if constexpr (PerHalfScale) {
            Accumulator c0{};
            Accumulator c1{};
            c0 = WmmaQuant<WaveSize>(a0[i], b0, c0);
            c1 = WmmaQuant<WaveSize>(a1[i], b1, c1);
#pragma unroll
            for (int l = 0; l < kAccumulatorElements; ++l) {
              acc[i][j][l] += (dw0[i][l] * dx) * static_cast<float>(c0[l]) +
                              (dw1[i][l] * dx) * static_cast<float>(c1[l]);
            }
          } else {
            Accumulator c{};
            c = WmmaQuant<WaveSize>(a0[i], b0, c);
            c = WmmaQuant<WaveSize>(a1[i], b1, c);
#pragma unroll
            for (int l = 0; l < kAccumulatorElements; ++l) {
              acc[i][j][l] += (dw0[i][l] * dx) * static_cast<float>(c[l]);
            }
          }
          if constexpr (HasOffset) {
#pragma unroll
            for (int l = 0; l < kAccumulatorElements; ++l) {
              acc[i][j][l] -= off[i][l] * sx;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();

  constexpr int kScratchBytes =
      static_cast<int>(sizeof(s_a) > sizeof(s_b) ? sizeof(s_a) : sizeof(s_b));
  constexpr int kWavesPerPass = kScratchBytes / (256 * 4);
  static_assert(kWavesPerPass >= 1,
                "no staging array holds even one wave's output tile");
  constexpr int kEpiloguePasses = (kWaves + kWavesPerPass - 1) / kWavesPerPass;
  float* const scratch_base =
      (sizeof(s_a) > sizeof(s_b) ? reinterpret_cast<float*>(&s_a[0][0][0])
                                 : reinterpret_cast<float*>(&s_b[0][0][0]));
  float* tile_scratch = scratch_base + ((wave_id % kWavesPerPass) * 256);
#pragma unroll
  for (int pass = 0; pass < kEpiloguePasses; ++pass) {
    if constexpr (kEpiloguePasses > 1) {
      __syncthreads();
      if ((wave_id / kWavesPerPass) != pass) {
        continue;
      }
    }
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
        for (int l = 0; l < kAccumulatorElements; ++l) {
          tile_scratch[(sub_lane * 16) + ((WaveSize / 16) * l) + half_id] =
              acc[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const std::size_t r0 =
            r_block +
            static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
        const std::size_t t0 =
            t_block +
            static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
        if (t0 + 16 <= batch && r0 + 16 <= m) {
          const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
          for (int st = 0; st < kAccumulatorElements; ++st) {
            const int flat = (st * WaveSize) + lane_id;
            y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
              static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
          }
        } else {
#pragma unroll
          for (int st = 0; st < kAccumulatorElements; ++st) {
            const int flat = (st * WaveSize) + lane_id;
            const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
            const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
            if (tok < batch && r < m) {
              y[(tok * m) + r] = tile_scratch[flat];
            }
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
}

namespace detail {
[[nodiscard]] bool TryLaunchQuantPrefillWave64(core::GgmlType type,
                                               const void* w, const void* x,
                                               float* y, std::size_t batch,
                                               std::size_t m, std::size_t k,
                                               hipStream_t stream);
}  // namespace detail

}  // namespace gufo::hip
#endif
#endif
