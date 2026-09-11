#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// Exact FP32-activation kernels shared by decoding, speculative verification,
// DFlash2 and the focused microbenchmark. Each lane visits its input groups in
// the same order at every batch width; tile geometry never changes arithmetic.
// Padded float4-aligned LDS rows amortize activation loads across output rows.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave>
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
        packed[r] = reinterpret_cast<const uint4*>(A + (safe_row * K))[vector];
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
         std::size_t TilesPerStage = 1, std::uint32_t MinWaves = 12>
__launch_bounds__(WavesPerBlock * 32, MinWaves) __global__
    void SmallBatchKQuantExactFp32GEMMKernel(const void* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             std::size_t m, std::size_t k) {
  constexpr std::size_t kSubElems = 16;
  constexpr std::size_t kSubsPerTile = 32 * TilesPerStage;
  constexpr std::size_t kVectorsPerSub = kSubElems / 4;
  constexpr std::size_t kStride = kSubElems + 4;
  constexpr std::size_t kTileStride = kSubsPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];
  // Every output row uses the same activation sum for each token/sub-block.
  // Compute it once during staging, in the decode GEMV's left-to-right order.
  __shared__ float staged_sums[Batch * kSubsPerTile];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_sub = k / kSubElems;
  const std::size_t row_bytes = QuantRowBytes(WType, k);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_sub;
       tile_base += kSubsPerTile) {
    // One thread stages a whole (token, sub-block) so it can accumulate that
    // sub-block's activation sum sequentially while it has the values.
    for (std::size_t flat = threadIdx.x; flat < Batch * kSubsPerTile;
         flat += blockDim.x) {
      const std::size_t token = flat / kSubsPerTile;
      const std::size_t sub = flat % kSubsPerTile;
      const std::size_t source_sub = tile_base + sub;
      float* dst = staged_x + (token * kTileStride) + (sub * kStride);
      float total = 0.0F;
#pragma unroll
      for (std::size_t vector = 0; vector < kVectorsPerSub; ++vector) {
        float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
        if (source_sub < num_sub) {
          value = *reinterpret_cast<const float4*>(
              x + (token * k) + (source_sub * kSubElems) + (vector * 4));
        }
        *reinterpret_cast<float4*>(dst + (vector * 4)) = value;
        // Left to right, term by term, matching the GEMV's scalar loop.
        total += value.x;
        total += value.y;
        total += value.z;
        total += value.w;
      }
      staged_sums[(token * kSubsPerTile) + sub] = total;
    }
    __syncthreads();

    for (std::size_t tile = 0; tile < TilesPerStage; ++tile) {
      const std::size_t slot = tile * 32 + lane;
      const std::size_t sub = tile_base + slot;
      if (sub < num_sub) {
        QuantSub16 decoded[RowsPerWave];
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const std::size_t row = row_base + r;
          const std::size_t safe_row = row < m ? row : m - 1;
          DecodeQuantSub16(
              WType,
              static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes), sub,
              decoded[r]);
        }
        float dots[RowsPerWave][Batch] = {};
#pragma unroll
        for (std::size_t group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
          for (std::size_t token = 0; token < Batch; ++token) {
            const float4 xv = *reinterpret_cast<const float4*>(
                staged_x + (token * kTileStride) + (slot * kStride) +
                (group * 4));
#pragma unroll
            for (std::size_t r = 0; r < RowsPerWave; ++r) {
              const std::int8_t* q = decoded[r].q + (group * 4);
              dots[r][token] += static_cast<float>(q[0]) * xv.x;
              dots[r][token] += static_cast<float>(q[1]) * xv.y;
              dots[r][token] += static_cast<float>(q[2]) * xv.z;
              dots[r][token] += static_cast<float>(q[3]) * xv.w;
            }
          }
        }
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
          for (std::size_t token = 0; token < Batch; ++token) {
            // The subtraction stays even for the symmetric formats, where
            // `offset` is zero. Dropping it lets the compiler contract
            // `sums += scale * dot` into a single-rounding FMA, while the
            // decode GEMV rounds the product and the sum separately -- which
            // broke bit-exactness for all five symmetric formats (33 of 64 rows
            // differed). Exactness is the contract here; the saved adds are not
            // worth it.
            //
            sums[r][token] += (decoded[r].scale * dots[r][token]) -
                              (decoded[r].offset *
                               staged_sums[(token * kSubsPerTile) + slot]);
          }
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

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_SMALL_BATCH_GEMM_HPP_
