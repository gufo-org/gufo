#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// Bound weight byte offsets and activation/output element offsets before
// selecting 32-bit device indexing. Division avoids overflow in the check.
// Batch >= 2 also leaves room for the rounded-up final row/tile indices.
constexpr bool FitsSmallBatch32BitIndices(std::size_t batch, std::size_t m,
                                          std::size_t k,
                                          std::size_t row_bytes) noexcept {
  constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
  return batch >= 2 && batch <= 8 && m != 0 && k != 0 && row_bytes != 0 &&
         m <= limit / batch && k <= limit / batch && m <= limit / row_bytes;
}

// Immediate XOR masks avoid the lane-address calculation of __shfl_xor.
// Keep the descending butterfly order identical to scalar decoding.
template<int Offset = 16>
__device__ __forceinline__ float ReduceKQuantWave(float value) {
  const int other = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, value),
                                                (Offset << 10) | 31);
  const float sum = value + __builtin_bit_cast(float, other);
  if constexpr (Offset > 1) {
    return ReduceKQuantWave<Offset / 2>(sum);
  } else {
    return sum;
  }
}

// Assign different token columns to the lanes as their K partials merge.
// Each output keeps the descending butterfly tree of scalar decoding, while
// successive stages need half as many shuffles and additions per lane.
template<std::size_t Columns, int Offset>
__device__ __forceinline__ float ReduceKQuantColumns(
    const float (&values)[Columns]) {
  static_assert(Columns > 0 && (Columns & (Columns - 1)) == 0);
  static_assert(Offset >= 0);
  static_assert(Columns == 1 ||
                Columns <= 2 * static_cast<std::size_t>(Offset));
  if constexpr (Columns > 1) {
    constexpr std::size_t kHalf = Columns / 2;
    const bool high = (threadIdx.x & static_cast<unsigned>(Offset)) != 0;
    float next[kHalf];
#pragma unroll
    for (std::size_t j = 0; j < kHalf; ++j) {
      const float own = high ? values[j + kHalf] : values[j];
      const float other = high ? values[j] : values[j + kHalf];
      const int partner = __builtin_amdgcn_ds_swizzle(
          __builtin_bit_cast(int, other), (Offset << 10) | 31);
      next[j] = own + __builtin_bit_cast(float, partner);
    }
    return ReduceKQuantColumns<kHalf, Offset / 2>(next);
  } else if constexpr (Offset > 0) {
    return ReduceKQuantWave<Offset>(values[0]);
  } else {
    return values[0];
  }
}

// These GEMMs exchange data through LDS only. Global inputs are read-only,
// and output stores follow the final tile. Complete LDS traffic and preserve
// compiler memory ordering without invalidating the global read cache.
__device__ __forceinline__ __attribute__((convergent)) void SyncKQuantTile() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

// Exact FP32-activation kernels shared by decoding, speculative verification,
// DFlash2 and the focused microbenchmark. Each lane visits its input groups in
// the same order at every batch width; tile geometry never changes arithmetic.
// Float4-aligned LDS rows amortize activation loads across output rows.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool StreamWeights = true,
         std::uint32_t HardwareWaveSize = 32, std::size_t TokenGroups = 1>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void BatchedExactBf16GEMMFp32VecKernel(const hip_bfloat16* __restrict__ A,
                                           const float* __restrict__ X,
                                           float* __restrict__ Y, std::size_t M,
                                           std::size_t K) {
  static_assert(HardwareWaveSize == 32 || HardwareWaveSize == 64);
  static_assert(TokenGroups > 0);
  X += (blockIdx.x % TokenGroups) * Batch * K;
  Y += (blockIdx.x % TokenGroups) * Batch * M;
  constexpr std::size_t kValuesPerVector = 8;
  constexpr std::size_t kVectorsPerTile = 32;
  constexpr std::size_t kStride = kValuesPerVector + 4;
  constexpr std::size_t kTileStride = kVectorsPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      (((blockIdx.x / TokenGroups) * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t vector_count = K / kValuesPerVector;
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < vector_count;
       tile_base += kVectorsPerTile) {
    constexpr std::size_t kTileHalves = Batch * kVectorsPerTile * 2;
    for (std::size_t flat = threadIdx.x; flat < kTileHalves;
         flat += blockDim.x) {
      const std::size_t token = flat / (kVectorsPerTile * 2);
      const std::size_t within = flat % (kVectorsPerTile * 2);
      const std::size_t vector = within >> 1U;
      const std::size_t half = within & 1U;
      const std::size_t source_vector = tile_base + vector;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_vector < vector_count) {
        value = *reinterpret_cast<const float4*>(
            X + (token * K) + (source_vector * kValuesPerVector) + (half * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                 (vector * kStride) + (half * 4)) = value;
    }
    if constexpr (Batch == 16)
      SyncKQuantTile();
    else
      __syncthreads();

    const std::size_t vector = tile_base + lane;
    if (vector < vector_count) {
      uint4 packed[RowsPerWave];
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
        const std::size_t row = row_base + r;
        const std::size_t safe_row = row < M ? row : M - 1;
        // Select the cache hint at dispatch: a runtime branch here makes
        // the large feature projection slower, even when it is uniform.
        if constexpr (!StreamWeights) {
          packed[r] =
              reinterpret_cast<const uint4*>(A + (safe_row * K))[vector];
        } else {
          typedef std::uint32_t PackedWeights
              __attribute__((ext_vector_type(4), may_alias));
          const auto words = __builtin_nontemporal_load(
              reinterpret_cast<const PackedWeights*>(A + (safe_row * K)) +
              vector);
          packed[r] = {words[0], words[1], words[2], words[3]};
        }
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        const float* input =
            staged_x + (token * kTileStride) + (lane * kStride);
        const float4 x0 = *reinterpret_cast<const float4*>(input);
        const float4 x1 = *reinterpret_cast<const float4*>(input + 4);
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const auto* weights =
              reinterpret_cast<const hip_bfloat16*>(&packed[r]);
          sums[r][token] += (static_cast<float>(weights[0]) * x0.x) +
                            (static_cast<float>(weights[1]) * x0.y) +
                            (static_cast<float>(weights[2]) * x0.z) +
                            (static_cast<float>(weights[3]) * x0.w) +
                            (static_cast<float>(weights[4]) * x1.x) +
                            (static_cast<float>(weights[5]) * x1.y) +
                            (static_cast<float>(weights[6]) * x1.z) +
                            (static_cast<float>(weights[7]) * x1.w);
        }
      }
    }
    if constexpr (Batch == 16)
      SyncKQuantTile();
    else
      __syncthreads();
  }

  // Tail elements outside the vectorized span, matching the GEMV epilogue.
  for (std::size_t k = (vector_count * kValuesPerVector) + lane; k < K;
       k += 32) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      const std::size_t safe_row = row < M ? row : M - 1;
      const float weight = static_cast<float>(A[(safe_row * K) + k]);
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        sums[r][token] += weight * X[(token * K) + k];
      }
    }
  }

  if constexpr (Batch == 16) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const float value = ReduceKQuantColumns<Batch, 16>(sums[r]);
      const std::size_t row = row_base + r;
      if ((lane & 1U) == 0 && row < M)
        Y[(lane / 2) * M + row] = value;
    }
  } else {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        for (int offset = 16; offset > 0; offset >>= 1) {
          sums[r][token] += __shfl_xor(sums[r][token], offset);
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
        const std::size_t row = row_base + r;
        if (row >= M) {
          continue;
        }
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          Y[(token * M) + row] = sums[r][token];
        }
      }
    }
  }
}

template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool NarrowIndex = false,
         bool XorStage = false, std::uint32_t HardwareWaveSize = 32,
         std::size_t TokenGroups = 1, std::size_t TokensPerStep = Batch,
         bool DistributedOutput = false>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0ExactFp32VecGEMMKernel(const void* __restrict__ w,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,
                                              std::size_t wide_m,
                                              std::size_t wide_k) {
  static_assert(HardwareWaveSize == 32 || HardwareWaveSize == 64);
  static_assert(TokenGroups > 0 && TokensPerStep > 0 &&
                Batch % TokensPerStep == 0);
  using Index = std::conditional_t<NarrowIndex, std::uint32_t, std::size_t>;
  const Index m = static_cast<Index>(wide_m);
  const Index k = static_cast<Index>(wide_k);
  // Adjacent workgroups reuse a weight row across independent token groups.
  // The second grid dimension still distributes narrow projections by token.
  const Index token_group =
      static_cast<Index>(blockIdx.y) * TokenGroups + blockIdx.x % TokenGroups;
  x += token_group * Batch * k;
  y += token_group * Batch * m;
  constexpr Index kBlocksPerTile = 32;
  constexpr Index kVectorsPerBlock = kQ8_0BlockSize / 4;
  // Transpose the vector groups and XOR their block indices to fit sixteen
  // activation rows in 64 KiB of LDS without changing the dot-product order.
  constexpr Index kStride = kQ8_0BlockSize + (XorStage ? 0 : 4);
  constexpr Index kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const Index lane = threadIdx.x & 31u;
  const Index warp_id = threadIdx.x >> 5u;
  const Index row_base =
      (((blockIdx.x / TokenGroups) * WavesPerBlock) + warp_id) * RowsPerWave;
  const Index num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (Index tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    constexpr Index kTileVectors = Batch * kBlocksPerTile * kVectorsPerBlock;
    for (Index flat = threadIdx.x; flat < kTileVectors; flat += blockDim.x) {
      const Index token = flat / (kBlocksPerTile * kVectorsPerBlock);
      const Index within = flat % (kBlocksPerTile * kVectorsPerBlock);
      const Index block = within / kVectorsPerBlock;
      const Index vector = within % kVectorsPerBlock;
      const Index source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (token * k) + (source_block * kQ8_0BlockSize) + (vector * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                 (XorStage ? vector * 128 + (block ^ vector) * 4
                                           : block * kStride + vector * 4)) =
          value;
    }
    if constexpr (NarrowIndex) {
      SyncKQuantTile();
    } else {
      __syncthreads();
    }

    const Index block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* rows[RowsPerWave];
      float scale[RowsPerWave];
#pragma unroll
      for (Index r = 0; r < RowsPerWave; ++r) {
        const Index row = row_base + r;
        const Index safe_row = row < m ? row : m - 1;
        rows[r] = &base[(safe_row * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      constexpr bool kPreloadWeights = TokensPerStep < Batch;
      constexpr Index kPackedGroups = kPreloadWeights ? kVectorsPerBlock : 1;
      std::uint32_t packed[RowsPerWave][kPackedGroups];
      if constexpr (kPreloadWeights) {
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (Index group = 0; group < kVectorsPerBlock; ++group) {
            __builtin_memcpy(&packed[r][group], rows[r]->qs + group * 4,
                             sizeof(std::uint32_t));
          }
        }
      }
#pragma unroll
      for (Index token_base = 0; token_base < Batch;
           token_base += TokensPerStep) {
        if constexpr (kPreloadWeights) {
          // Finish a few independent dots at a time to bound register use.
          __builtin_amdgcn_sched_barrier(0);
        }
        float block_dots[RowsPerWave][TokensPerStep] = {};
#pragma unroll
        for (Index group = 0; group < kVectorsPerBlock; ++group) {
          if constexpr (DistributedOutput && TokensPerStep == Batch) {
            // Keep one coefficient group live through its token dots.
            __builtin_amdgcn_sched_barrier(0);
          }
          if constexpr (!kPreloadWeights) {
#pragma unroll
            for (Index r = 0; r < RowsPerWave; ++r) {
              __builtin_memcpy(&packed[r][0], rows[r]->qs + group * 4,
                               sizeof(std::uint32_t));
            }
          }
#pragma unroll
          for (Index local_token = 0; local_token < TokensPerStep;
               ++local_token) {
            const Index token = token_base + local_token;
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + token * kTileStride +
                (XorStage ? group * 128 + (lane ^ group) * 4
                          : lane * kStride + group * 4));
#pragma unroll
            for (Index r = 0; r < RowsPerWave; ++r) {
              const std::uint32_t p = packed[r][kPreloadWeights ? group : 0];
              block_dots[r][local_token] +=
                  static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) *
                  xv.x;
              block_dots[r][local_token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 8U) & 0xFFU)) *
                  xv.y;
              block_dots[r][local_token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 16U) & 0xFFU)) *
                  xv.z;
              block_dots[r][local_token] +=
                  static_cast<float>(
                      static_cast<std::int8_t>((p >> 24U) & 0xFFU)) *
                  xv.w;
            }
          }
        }
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (Index local_token = 0; local_token < TokensPerStep;
               ++local_token) {
            sums[r][token_base + local_token] +=
                scale[r] * block_dots[r][local_token];
          }
        }
      }
    }
    if constexpr (NarrowIndex) {
      SyncKQuantTile();
    } else {
      __syncthreads();
    }
  }

  if constexpr (DistributedOutput) {
    static_assert(Batch == 16);
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
      const float value = ReduceKQuantColumns<Batch, 16>(sums[r]);
      const Index row = row_base + r;
      if ((lane & 1U) == 0 && row < m)
        y[(lane / 2) * m + row] = value;
    }
  } else {
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
      for (Index token = 0; token < Batch; ++token) {
        if constexpr (NarrowIndex) {
          sums[r][token] = ReduceKQuantWave(sums[r][token]);
        } else {
          for (int offset = 16; offset > 0; offset >>= 1) {
            sums[r][token] += __shfl_xor(sums[r][token], offset);
          }
        }
      }
    }
    if (lane == 0) {
#pragma unroll
      for (Index r = 0; r < RowsPerWave; ++r) {
        const Index row = row_base + r;
        if (row >= m) {
          continue;
        }
#pragma unroll
        for (Index token = 0; token < Batch; ++token) {
          y[(token * m) + row] = sums[r][token];
        }
      }
    }
  }
}

template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, core::GgmlType WType,
         std::size_t TilesPerStage = 1, std::uint32_t MinWaves = 12,
         std::size_t TokenGroups = 1, bool NarrowIndex = false,
         std::size_t TokensPerStep = Batch, std::uint32_t HardwareWaveSize = 32,
         std::uint32_t LogicalLanes = 32,
         bool DistributedOutput = HardwareWaveSize == 64 &&
                                  (Batch == 14 || Batch == 16 ||
                                   (Batch == 8 &&
                                    (WType == core::GgmlType::kQ4_K ||
                                     WType == core::GgmlType::kQ5_K ||
                                     WType == core::GgmlType::kIQ4_XS))),
         bool FusedSwiGLU = false>
__launch_bounds__(WavesPerBlock * 32, (MinWaves * 32 / HardwareWaveSize > 0
                                           ? MinWaves * 32 / HardwareWaveSize
                                           : 1)) __global__
    void SmallBatchKQuantExactFp32GEMMKernel(const void* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             std::size_t wide_m,
                                             std::size_t wide_k) {
  static_assert(TokensPerStep > 0 && Batch % TokensPerStep == 0);
  static_assert(HardwareWaveSize == 32 || HardwareWaveSize == 64);
  static_assert(LogicalLanes == 16 || LogicalLanes == 32);
  static_assert(LogicalLanes == 32 || TilesPerStage == 1);
  static_assert(!FusedSwiGLU || (DistributedOutput && RowsPerWave % 2 == 0));
  // Sixteen-lane groups keep both original lane partials independently, then
  // combine them in the same descending butterfly order as scalar decoding.
  // Only the occupancy hint counts hardware waves.
  using Index = std::conditional_t<NarrowIndex, std::uint32_t, std::size_t>;
  const Index m = static_cast<Index>(wide_m);
  const Index k = static_cast<Index>(wide_k);
  x += (blockIdx.x % TokenGroups) * Batch * k;
  const Index output_rows = FusedSwiGLU ? m / 2 : m;
  y += (blockIdx.x % TokenGroups) * Batch * output_rows;
  constexpr Index kSubElems = 16;
  constexpr Index kPhases = 32 / LogicalLanes;
  constexpr Index kSubsPerTile = LogicalLanes * TilesPerStage;
  constexpr Index kVectorsPerSub = kSubElems / 4;
  constexpr bool kCompact =
      (WType == core::GgmlType::kQ4_K && Batch >= 3 && RowsPerWave == 4 &&
       TilesPerStage == 2 && NarrowIndex) ||
      (Batch == 8 && TilesPerStage == 1 &&
       (WType == core::GgmlType::kQ5_K ||
        (WType == core::GgmlType::kIQ4_XS && RowsPerWave == 4 && NarrowIndex)));
  constexpr Index kStride = kSubElems + (kCompact ? 0 : 4);
  constexpr Index kTileStride = kSubsPerTile * kStride;
  constexpr bool kHasOffset =
      WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K;
  __shared__ float staged_x[Batch * kTileStride];
  // Affine formats use the same activation sum for every output row.
  // Compute it once during staging, in the decode GEMV's left-to-right order.
  __shared__ float staged_sums[kHasOffset ? Batch * kSubsPerTile : 1];

  const Index lane = threadIdx.x % LogicalLanes;
  const Index warp_id = threadIdx.x / LogicalLanes;
  const Index row_base =
      (((blockIdx.x / TokenGroups) * WavesPerBlock * kPhases) + warp_id) *
      RowsPerWave;
  const Index num_sub = k / kSubElems;
  const Index row_bytes = QuantRowBytes(WType, k);
  float sums[kPhases][RowsPerWave][Batch] = {};

  for (Index tile_base = 0; tile_base < num_sub;
       tile_base += 32 * TilesPerStage) {
#pragma unroll
    for (Index phase = 0; phase < kPhases; ++phase) {
      // One thread stages a whole (token, sub-block) so it can accumulate that
      // sub-block's activation sum sequentially while it has the values.
      for (Index flat = threadIdx.x; flat < Batch * kSubsPerTile;
           flat += blockDim.x) {
        const Index token = flat / kSubsPerTile;
        const Index sub = flat % kSubsPerTile;
        const Index source_sub = tile_base + phase * LogicalLanes + sub;
        float* dst = staged_x + (token * kTileStride) + (sub * kStride);
        float total = 0.0F;
#pragma unroll
        for (Index vector = 0; vector < kVectorsPerSub; ++vector) {
          float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
          if (source_sub < num_sub) {
            value = *reinterpret_cast<const float4*>(
                x + (token * k) + (source_sub * kSubElems) + (vector * 4));
          }
          // Compact staging removes per-sub-block padding. Both reads and
          // writes permute float4 groups; arithmetic is unchanged.
          const Index group = kCompact ? vector ^ ((sub >> 1U) & 3U) : vector;
          *reinterpret_cast<float4*>(dst + (group * 4)) = value;
          if constexpr (kHasOffset) {
            // Left to right, term by term, matching the GEMV's scalar loop.
            total += value.x;
            total += value.y;
            total += value.z;
            total += value.w;
          }
        }
        if constexpr (kHasOffset) {
          staged_sums[(token * kSubsPerTile) + sub] = total;
        }
      }
      SyncKQuantTile();

      for (Index tile = 0; tile < TilesPerStage; ++tile) {
        const Index slot = tile * 32 + lane;
        const Index sub = tile_base + phase * LogicalLanes + slot;
        if (sub < num_sub) {
          QuantSub16 decoded[RowsPerWave];
#pragma unroll
          for (Index r = 0; r < RowsPerWave; ++r) {
            const Index row =
                FusedSwiGLU
                    ? (row_base + r) / 2 + ((row_base + r) % 2) * output_rows
                    : row_base + r;
            const Index safe_row = row < m ? row : m - 1;
            DecodeQuantSub16<NarrowIndex>(
                WType,
                static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes),
                sub, decoded[r]);
          }
          // Group independent token dots without reordering an output's sums.
#pragma unroll
          for (Index token_base = 0; token_base < Batch;
               token_base += TokensPerStep) {
            if constexpr (LogicalLanes == 16 ||
                          (RowsPerWave == 6 && HardwareWaveSize == 64 &&
                           ((Batch == 16 && (WType == core::GgmlType::kQ4_K ||
                                             WType == core::GgmlType::kQ5_K)) ||
                            (WType == core::GgmlType::kQ6_K &&
                             (Batch == 14 || Batch == 16))))) {
              // Bound live token loads to avoid spilling the output partials.
              // This fence preserves every dot and reduction order.
              __builtin_amdgcn_sched_barrier(0);
            }
            float dots[RowsPerWave][TokensPerStep] = {};
            if constexpr (WType == core::GgmlType::kQ5_K && Batch == 8 &&
                          RowsPerWave == 4 && NarrowIndex &&
                          TokensPerStep == 1) {
              // Convert each row's four coefficients before its dependent FMAs.
              // This schedule is faster for the batch-eight Q5 FFN geometries;
              // each output retains the scalar dot product's accumulation
              // order.
#pragma unroll
              for (Index group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
                for (Index r = 0; r < RowsPerWave; ++r) {
                  const std::int8_t* q = decoded[r].q + group * 4;
                  const float q0 = static_cast<float>(q[0]);
                  const float q1 = static_cast<float>(q[1]);
                  const float q2 = static_cast<float>(q[2]);
                  const float q3 = static_cast<float>(q[3]);
#pragma unroll
                  for (Index local_token = 0; local_token < TokensPerStep;
                       ++local_token) {
                    const Index token = token_base + local_token;
                    const Index input_group =
                        kCompact ? group ^ ((slot >> 1U) & 3U) : group;
                    const float4 xv = *reinterpret_cast<const float4*>(
                        staged_x + token * kTileStride + slot * kStride +
                        input_group * 4);
                    // Alternating operand positions helps paired FMA issue on
                    // gfx1151 without changing products or accumulation order.
                    const bool swap = ((r ^ token) & 1U) != 0U;
                    dots[r][local_token] =
                        swap ? __builtin_fmaf(xv.x, q0, dots[r][local_token])
                             : __builtin_fmaf(q0, xv.x, dots[r][local_token]);
                    dots[r][local_token] =
                        swap ? __builtin_fmaf(xv.y, q1, dots[r][local_token])
                             : __builtin_fmaf(q1, xv.y, dots[r][local_token]);
                    dots[r][local_token] =
                        swap ? __builtin_fmaf(xv.z, q2, dots[r][local_token])
                             : __builtin_fmaf(q2, xv.z, dots[r][local_token]);
                    dots[r][local_token] =
                        swap ? __builtin_fmaf(xv.w, q3, dots[r][local_token])
                             : __builtin_fmaf(q3, xv.w, dots[r][local_token]);
                  }
                }
              }
            } else {
#pragma unroll
              for (Index group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
                for (Index local_token = 0; local_token < TokensPerStep;
                     ++local_token) {
                  const Index token = token_base + local_token;
                  const Index input_group =
                      kCompact ? group ^ ((slot >> 1U) & 3U) : group;
                  const float4 xv = *reinterpret_cast<const float4*>(
                      staged_x + (token * kTileStride) + (slot * kStride) +
                      (input_group * 4));
#pragma unroll
                  for (Index r = 0; r < RowsPerWave; ++r) {
                    const std::int8_t* q = decoded[r].q + (group * 4);
                    dots[r][local_token] += static_cast<float>(q[0]) * xv.x;
                    dots[r][local_token] += static_cast<float>(q[1]) * xv.y;
                    dots[r][local_token] += static_cast<float>(q[2]) * xv.z;
                    dots[r][local_token] += static_cast<float>(q[3]) * xv.w;
                  }
                }
              }
            }
#pragma unroll
            for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
              for (Index local_token = 0; local_token < TokensPerStep;
                   ++local_token) {
                const Index token = token_base + local_token;
                if constexpr (kHasOffset) {
                  sums[phase][r][token] +=
                      (decoded[r].scale * dots[r][local_token]) -
                      (decoded[r].offset *
                       staged_sums[(token * kSubsPerTile) + slot]);
                } else {
                  // Decode rounds the scaled dot before accumulating it. Keep
                  // contraction disabled only here; the dot above still uses
                  // its original FMAs. Symmetric formats need no activation
                  // sums.
#pragma clang fp contract(off)
                  const float contribution =
                      decoded[r].scale * dots[r][local_token];
                  sums[phase][r][token] += contribution;
                }
              }
            }
          }
        }
      }
      SyncKQuantTile();
    }
  }

  if constexpr (DistributedOutput) {
    static_assert(Batch == 8 || Batch == 14 || Batch == 16);
    constexpr Index kColumns = Batch == 8 ? 8 : 16;
    constexpr Index kLanesPerToken = LogicalLanes / kColumns;
    float final[RowsPerWave];
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
      float values[kColumns] = {};
#pragma unroll
      for (Index token = 0; token < Batch; ++token) {
        values[token] = sums[0][r][token];
        if constexpr (kPhases == 2) {
          values[token] += sums[1][r][token];
        }
      }
      final[r] = ReduceKQuantColumns<kColumns, LogicalLanes / 2>(values);
    }
    const Index token = lane / kLanesPerToken;
    if (lane % kLanesPerToken == 0 && token < Batch) {
      if constexpr (FusedSwiGLU) {
#pragma unroll
        for (Index r = 0; r < RowsPerWave / 2; ++r) {
          const Index row = row_base / 2 + r;
          if (row < output_rows) {
            const float gate = final[2 * r];
            const float up = final[2 * r + 1];
            const float sigmoid = 1.0F / (1.0F + expf(-gate));
            y[token * output_rows + row] = (gate * sigmoid) * up;
          }
        }
      } else {
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
          const Index row = row_base + r;
          if (row < m) {
            y[token * m + row] = final[r];
          }
        }
      }
    }
  } else {
#pragma unroll
    for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
      for (Index token = 0; token < Batch; ++token) {
        if constexpr (kPhases == 2) {
          sums[0][r][token] += sums[1][r][token];
        }
        sums[0][r][token] =
            ReduceKQuantWave<LogicalLanes / 2>(sums[0][r][token]);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (Index r = 0; r < RowsPerWave; ++r) {
        const Index row = row_base + r;
        if (row >= m) {
          continue;
        }
#pragma unroll
        for (Index token = 0; token < Batch; ++token) {
          y[(token * m) + row] = sums[0][r][token];
        }
      }
    }
  }
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_
