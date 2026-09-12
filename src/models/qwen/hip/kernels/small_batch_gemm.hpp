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
// Batch >= 3 also leaves room for the rounded-up final row/tile indices.
constexpr bool FitsSmallBatch32BitIndices(std::size_t batch, std::size_t m,
                                          std::size_t k,
                                          std::size_t row_bytes) noexcept {
  constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
  return batch >= 3 && batch <= 8 && m != 0 && k != 0 && row_bytes != 0 &&
         m <= limit / batch && k <= limit / batch && m <= limit / row_bytes;
}

// Exact FP32-activation kernels shared by decoding, speculative verification,
// DFlash2 and the focused microbenchmark. Each lane visits its input groups in
// the same order at every batch width; tile geometry never changes arithmetic.
// Float4-aligned LDS rows amortize activation loads across output rows.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, bool StreamWeights = true>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void BatchedExactBf16GEMMFp32VecKernel(const hip_bfloat16* __restrict__ A,
                                           const float* __restrict__ X,
                                           float* __restrict__ Y, std::size_t M,
                                           std::size_t K) {
  constexpr std::size_t kValuesPerVector = 8;
  constexpr std::size_t kVectorsPerTile = 32;
  constexpr std::size_t kStride = kValuesPerVector + 4;
  constexpr std::size_t kTileStride = kVectorsPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
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

template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0ExactFp32VecGEMMKernel(const void* __restrict__ w,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,
                                              std::size_t m, std::size_t k) {
  // A second grid dimension distributes narrow projections across tokens.
  x += static_cast<std::size_t>(blockIdx.y) * Batch * k;
  y += static_cast<std::size_t>(blockIdx.y) * Batch * m;
  constexpr std::size_t kBlocksPerTile = 32;
  constexpr std::size_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::size_t kStride = kQ8_0BlockSize + 4;
  constexpr std::size_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    constexpr std::size_t kTileVectors =
        Batch * kBlocksPerTile * kVectorsPerBlock;
    for (std::size_t flat = threadIdx.x; flat < kTileVectors;
         flat += blockDim.x) {
      const std::size_t token = flat / (kBlocksPerTile * kVectorsPerBlock);
      const std::size_t within = flat % (kBlocksPerTile * kVectorsPerBlock);
      const std::size_t block = within / kVectorsPerBlock;
      const std::size_t vector = within % kVectorsPerBlock;
      const std::size_t source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (token * k) + (source_block * kQ8_0BlockSize) + (vector * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                 (block * kStride) + (vector * 4)) = value;
    }
    __syncthreads();

    const std::size_t block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* rows[RowsPerWave];
      float scale[RowsPerWave];
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
        const std::size_t row = row_base + r;
        const std::size_t safe_row = row < m ? row : m - 1;
        rows[r] = &base[(safe_row * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      float block_dots[RowsPerWave][Batch] = {};
#pragma unroll
      for (std::size_t group = 0; group < kVectorsPerBlock; ++group) {
        std::uint32_t packed[RowsPerWave];
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          __builtin_memcpy(&packed[r], rows[r]->qs + (group * 4),
                           sizeof(std::uint32_t));
        }
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          const float4 xv = *reinterpret_cast<const float4*>(
              staged_x + (token * kTileStride) + (lane * kStride) +
              (group * 4));
#pragma unroll
          for (std::size_t r = 0; r < RowsPerWave; ++r) {
            const std::uint32_t p = packed[r];
            block_dots[r][token] +=
                static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 8U) & 0xFFU)) *
                                    xv.y;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 16U) & 0xFFU)) *
                                    xv.z;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 24U) & 0xFFU)) *
                                    xv.w;
          }
        }
      }
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          sums[r][token] += scale[r] * block_dots[r][token];
        }
      }
    }
    __syncthreads();
  }

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
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
}

template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, core::GgmlType WType,
         std::size_t TilesPerStage = 1, std::uint32_t MinWaves = 12,
         std::size_t TokenGroups = 1, bool NarrowIndex = false>
__launch_bounds__(WavesPerBlock * 32, MinWaves) __global__
    void SmallBatchKQuantExactFp32GEMMKernel(const void* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             std::size_t wide_m,
                                             std::size_t wide_k) {
  using Index = std::conditional_t<NarrowIndex, std::uint32_t, std::size_t>;
  const Index m = static_cast<Index>(wide_m);
  const Index k = static_cast<Index>(wide_k);
  x += (blockIdx.x % TokenGroups) * Batch * k;
  y += (blockIdx.x % TokenGroups) * Batch * m;
  constexpr Index kSubElems = 16;
  constexpr Index kSubsPerTile = 32 * TilesPerStage;
  constexpr Index kVectorsPerSub = kSubElems / 4;
  constexpr bool kCompact =
      Batch == 8 && WType == core::GgmlType::kQ5_K && TilesPerStage == 1;
  constexpr Index kStride = kSubElems + (kCompact ? 0 : 4);
  constexpr Index kTileStride = kSubsPerTile * kStride;
  constexpr bool kHasOffset =
      WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K;
  __shared__ float staged_x[Batch * kTileStride];
  // Affine formats use the same activation sum for every output row.
  // Compute it once during staging, in the decode GEMV's left-to-right order.
  __shared__ float staged_sums[kHasOffset ? Batch * kSubsPerTile : 1];

  const Index lane = threadIdx.x & 31u;
  const Index warp_id = threadIdx.x >> 5u;
  const Index row_base =
      (((blockIdx.x / TokenGroups) * WavesPerBlock) + warp_id) * RowsPerWave;
  const Index num_sub = k / kSubElems;
  const Index row_bytes = QuantRowBytes(WType, k);
  float sums[RowsPerWave][Batch] = {};

  for (Index tile_base = 0; tile_base < num_sub;
       tile_base += kSubsPerTile) {
    // One thread stages a whole (token, sub-block) so it can accumulate that
    // sub-block's activation sum sequentially while it has the values.
    for (Index flat = threadIdx.x; flat < Batch * kSubsPerTile;
         flat += blockDim.x) {
      const Index token = flat / kSubsPerTile;
      const Index sub = flat % kSubsPerTile;
      const Index source_sub = tile_base + sub;
      float* dst = staged_x + (token * kTileStride) + (sub * kStride);
      float total = 0.0F;
#pragma unroll
      for (Index vector = 0; vector < kVectorsPerSub; ++vector) {
        float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
        if (source_sub < num_sub) {
          value = *reinterpret_cast<const float4*>(
              x + (token * k) + (source_sub * kSubElems) + (vector * 4));
        }
        // Compact Q5 staging needs 17 KiB including sums, instead of 21 KiB.
        // Both reads and writes permute float4 groups; arithmetic is unchanged.
        const Index group =
            kCompact ? vector ^ ((sub >> 1U) & 3U) : vector;
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
    __syncthreads();

    for (Index tile = 0; tile < TilesPerStage; ++tile) {
      const Index slot = tile * 32 + lane;
      const Index sub = tile_base + slot;
      if (sub < num_sub) {
        QuantSub16 decoded[RowsPerWave];
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
          const Index row = row_base + r;
          const Index safe_row = row < m ? row : m - 1;
          DecodeQuantSub16<NarrowIndex>(
              WType,
              static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes), sub,
              decoded[r]);
        }
        float dots[RowsPerWave][Batch] = {};
#pragma unroll
        for (Index group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
          for (Index token = 0; token < Batch; ++token) {
            const Index input_group =
                kCompact ? group ^ ((slot >> 1U) & 3U) : group;
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + (token * kTileStride) + (slot * kStride) +
                (input_group * 4));
#pragma unroll
            for (Index r = 0; r < RowsPerWave; ++r) {
              const std::int8_t* q = decoded[r].q + (group * 4);
              dots[r][token] += static_cast<float>(q[0]) * xv.x;
              dots[r][token] += static_cast<float>(q[1]) * xv.y;
              dots[r][token] += static_cast<float>(q[2]) * xv.z;
              dots[r][token] += static_cast<float>(q[3]) * xv.w;
            }
          }
        }
#pragma unroll
        for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (Index token = 0; token < Batch; ++token) {
            if constexpr (kHasOffset) {
              sums[r][token] += (decoded[r].scale * dots[r][token]) -
                                (decoded[r].offset *
                                 staged_sums[(token * kSubsPerTile) + slot]);
            } else {
              // Decode rounds the scaled dot before accumulating it. Keep
              // contraction disabled only here; the dot above still uses its
              // original FMAs. Symmetric formats need no activation sums.
#pragma clang fp contract(off)
              const float contribution = decoded[r].scale * dots[r][token];
              sums[r][token] += contribution;
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (Index r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (Index token = 0; token < Batch; ++token) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], offset);
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

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_
