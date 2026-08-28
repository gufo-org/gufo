#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void BatchedGEMMKernel(const void* __restrict__ A, bool is_bf16,
                           const float* __restrict__ X, float* __restrict__ Y,
                           std::size_t batch_size, std::size_t M,
                           std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t b = blockIdx.y;
  if (m >= M || b >= batch_size) {
    return;
  }

  const float* x_row = X + (b * K);
  float* y_row = Y + (b * M);

  float dot = 0.0F;
  if (is_bf16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const auto* row_vec = reinterpret_cast<const uint4*>(row);
    const std::size_t K_vec = K / 8;
    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const auto* row_vec = reinterpret_cast<const float4*>(row);
    const std::size_t K_vec = K / 4;
    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 a_v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 x_v = *reinterpret_cast<const float4*>(&x_row[k]);
      dot +=
          (a_v.x * x_v.x) + (a_v.y * x_v.y) + (a_v.z * x_v.z) + (a_v.w * x_v.w);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < K; k += 32) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    y_row[m] = dot;
  }
}

void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }
  constexpr std::uint32_t kWaves = 4;
  const dim3 grid(static_cast<unsigned int>((M + kWaves - 1) / kWaves),
                  static_cast<unsigned int>(batch_size));
  BatchedGEMMKernel<kWaves>
      <<<grid, kWaves * 32, 0, stream>>>(A, is_bf16, X, Y, batch_size, M, K);
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 3) __global__
    void BatchedExactBf16GEMMFp32Lds8Kernel(const hip_bfloat16* __restrict__ A,
                                            const float* __restrict__ X,
                                            float* __restrict__ Y,
                                            std::size_t M, std::size_t K) {
  constexpr std::size_t kBatch = 8;
  constexpr std::size_t kValuesPerVector = 8;
  constexpr std::size_t kVectorsPerTile = 32;
  constexpr std::size_t kPaddedVectorSize = kValuesPerVector + 1;
  constexpr std::size_t kTileStride = kVectorsPerTile * kPaddedVectorSize;
  __shared__ float staged_x[kBatch * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t safe_row = row < M ? row : M - 1;
  const auto* row_vec = reinterpret_cast<const uint4*>(A + (safe_row * K));
  const std::size_t vector_count = K / kValuesPerVector;
  float sums[kBatch] = {};

  for (std::size_t tile_base = 0; tile_base < vector_count;
       tile_base += kVectorsPerTile) {
    constexpr std::size_t kTileElements =
        kBatch * kVectorsPerTile * kValuesPerVector;
    for (std::size_t flat = threadIdx.x; flat < kTileElements;
         flat += blockDim.x) {
      const std::size_t token = flat / (kVectorsPerTile * kValuesPerVector);
      const std::size_t within = flat % (kVectorsPerTile * kValuesPerVector);
      const std::size_t vector = within / kValuesPerVector;
      const std::size_t element = within % kValuesPerVector;
      const std::size_t source_vector = tile_base + vector;
      staged_x[(token * kTileStride) + (vector * kPaddedVectorSize) + element] =
          source_vector < vector_count
              ? X[(token * K) + (source_vector * kValuesPerVector) + element]
              : 0.0F;
    }
    __syncthreads();

    const std::size_t vector = tile_base + lane;
    if (row < M && vector < vector_count) {
      const uint4 packed_weights = row_vec[vector];
      const auto* weights =
          reinterpret_cast<const hip_bfloat16*>(&packed_weights);
#pragma unroll
      for (std::size_t token = 0; token < kBatch; ++token) {
        const float* input =
            staged_x + (token * kTileStride) + (lane * kPaddedVectorSize);
        sums[token] += (static_cast<float>(weights[0]) * input[0]) +
                       (static_cast<float>(weights[1]) * input[1]) +
                       (static_cast<float>(weights[2]) * input[2]) +
                       (static_cast<float>(weights[3]) * input[3]) +
                       (static_cast<float>(weights[4]) * input[4]) +
                       (static_cast<float>(weights[5]) * input[5]) +
                       (static_cast<float>(weights[6]) * input[6]) +
                       (static_cast<float>(weights[7]) * input[7]);
      }
    }
    __syncthreads();
  }

  for (std::size_t k = (vector_count * kValuesPerVector) + lane; k < K;
       k += 32) {
    const float weight = static_cast<float>(A[(safe_row * K) + k]);
#pragma unroll
    for (std::size_t token = 0; token < kBatch; ++token) {
      sums[token] += weight * X[(token * K) + k];
    }
  }

#pragma unroll
  for (std::size_t token = 0; token < kBatch; ++token) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token] += __shfl_xor(sums[token], offset);
    }
  }

  if (lane == 0 && row < M) {
#pragma unroll
    for (std::size_t token = 0; token < kBatch; ++token) {
      Y[(token * M) + row] = sums[token];
    }
  }
}

void LaunchExactBf16GEMMFp32Batch8(const void* A, const float* X, float* Y,
                                   std::size_t M, std::size_t K,
                                   hipStream_t stream) {
  if (M == 0 || K == 0) {
    return;
  }
  constexpr std::uint32_t kWaves = 8;
  const unsigned int blocks =
      static_cast<unsigned int>((M + kWaves - 1) / kWaves);
  BatchedExactBf16GEMMFp32Lds8Kernel<kWaves>
      <<<blocks, kWaves * 32, 0, stream>>>(static_cast<const hip_bfloat16*>(A),
                                           X, Y, M, K);
}

__global__ void FloatToBfloat16Kernel(const float* __restrict__ in,
                                      hip_bfloat16* __restrict__ out,
                                      std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    out[idx] = hip_bfloat16(in[idx]);
  }
}

void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  FloatToBfloat16Kernel<<<num_blocks, block_size, 0, stream>>>(
      in, static_cast<hip_bfloat16*>(out), num_elements);
}

// opt-c162-q8k-prefill-dequant: device Q8_K weight dequantization to BF16 for
// the prefill hipblas GEMM path. hipBLAS cannot consume block_q8_K weights, so
// prefill first dequantizes the weight region into a BF16 scratch buffer and
// runs the GF16 GEMM on that scratch. One block per 256-wide Q8_K block, one
// thread per element; matches the CPU DequantizeQ8_K oracle
// (out[j] = d * qs[j] for j in [0,256) per block). Decode is unaffected: it
// already computes at Q8 via Q8KRowDot.
// Q8_K block layout is defined in the shared quant_ops.hpp header.

// Q5_K / Q6_K block layouts are defined in the shared quant_ops.hpp
// header; dequant to BF16 uses Q5KValueFp / Q6KValueFp below.
__global__ void DequantizeQ8KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t b = (blockIdx.x * (blockDim.x >> 5u)) + warp_id;
  if (b >= n_blocks) {
    return;
  }
  const auto* blk = static_cast<const Q8KBlock*>(w) + b;
  const float d = blk->d;
  hip_bfloat16* out_blk = out + (b * kQ8KBlockSize);
#pragma unroll
  for (std::size_t k = 0; k < 8; ++k) {
    const std::size_t j = lane_id + (32u * k);
    out_blk[j] = hip_bfloat16(d * static_cast<float>(blk->qs[j]));
  }
}

void LaunchDequantizeQ8KToBf16(const void* w, hip_bfloat16* out,
                               std::size_t n_elems, hipStream_t stream) {
  if (n_elems == 0) {
    return;
  }
  // Requires the weight region to be a whole number of 256-wide Q8_K blocks.
  if (n_elems % kQ8KBlockSize != 0) {
    return;
  }
  const std::size_t n_blocks = n_elems / kQ8KBlockSize;
  constexpr unsigned int block_threads = 256;
  constexpr unsigned int warps_per_block = block_threads / 32;
  const std::size_t grid = (n_blocks + warps_per_block - 1U) / warps_per_block;
  DequantizeQ8KToBf16Kernel<<<static_cast<unsigned int>(grid), block_threads, 0,
                              stream>>>(w, out, n_blocks);
}

// Per-value dequant helpers (fp32), matching the CPU oracles
// Q5Value/Q6Value/DequantizeQ8_0 in ggml_dequant.cpp. GetQKScaleMin (used by
// Q5KValueFp) is defined in the shared quant_ops.hpp header.
__device__ float Q5KValueFp(const Q5KBlock& block, std::size_t index) noexcept {
  const std::size_t gg = index / 64;  // 0..3
  const std::size_t wv = index % 64;  // 0..63
  const std::size_t lane = wv % 32;   // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((block.qh[lane] >> bit) & 1U) ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);        // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQKScaleMin(sis, block.scales, sc, m);
  return __half2float(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         __half2float(block.dmin) * static_cast<float>(m);
}

__device__ float Q6KValueFp(const Q6KBlock& block, std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t segment = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh_byte = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = (qh_byte >> 0U) & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh_byte >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh_byte >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh_byte >> 6U) & 0x03U;
      break;
  }
  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return __half2float(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

__device__ float Q8_0ValueFp(const Q8_0Block& block,
                             std::size_t index) noexcept {
  return __half2float(block.d) * static_cast<float>(block.qs[index]);
}

// Type-dispatched dequant-to-BF16 kernels.
__global__ void DequantizeQ8_0ToBf16Kernel(const void* __restrict__ w,
                                           hip_bfloat16* __restrict__ out,
                                           std::size_t n_blocks) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t blk_in_warp = lane_id >> 2u;     // 0..7
  const std::size_t elem_off = (lane_id & 3u) * 8u;  // 0, 8, 16, 24
  const std::size_t b =
      (blockIdx.x * ((blockDim.x >> 5u) * 8u)) + (warp_id * 8u) + blk_in_warp;
  if (b >= n_blocks) {
    return;
  }
  const auto* blk = static_cast<const Q8_0Block*>(w) + b;
  const float d = __half2float(blk->d);
  const auto q8_raw = *reinterpret_cast<const uint64_t*>(blk->qs + elem_off);
  const auto* qs = reinterpret_cast<const int8_t*>(&q8_raw);

  hip_bfloat16 bf[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bf[i] = hip_bfloat16(d * static_cast<float>(qs[i]));
  }

  *reinterpret_cast<uint4*>(&out[(b * kQ8_0BlockSize) + elem_off]) =
      *reinterpret_cast<const uint4*>(bf);
}

__global__ void DequantizeQ5KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t b = (blockIdx.x * (blockDim.x >> 5u)) + warp_id;
  if (b >= n_blocks) {
    return;
  }
  const auto* blk = static_cast<const Q5KBlock*>(w) + b;
  hip_bfloat16* out_blk = out + (b * kQ5KBlockSize);
#pragma unroll
  for (std::size_t k = 0; k < 8; ++k) {
    const std::size_t j = lane_id + (32u * k);
    out_blk[j] = hip_bfloat16(Q5KValueFp(*blk, j));
  }
}

__global__ void DequantizeQ6KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t b = (blockIdx.x * (blockDim.x >> 5u)) + warp_id;
  if (b >= n_blocks) {
    return;
  }
  const auto* blk = static_cast<const Q6KBlock*>(w) + b;
  hip_bfloat16* out_blk = out + (b * kQ6KBlockSize);
#pragma unroll
  for (std::size_t k = 0; k < 8; ++k) {
    const std::size_t j = lane_id + (32u * k);
    out_blk[j] = hip_bfloat16(Q6KValueFp(*blk, j));
  }
}

// opt-c162-q8k-prefill-dequant (extended): type-dispatched entry point that
// converts a quantized weight region (Q8_0/Q5_K/Q6_K/Q8_K) into a BF16 scratch
// for the prefill hipblas GEMM path.
void LaunchDequantizeToBf16(core::GgmlType type, const void* w,
                            hip_bfloat16* out, std::size_t n_elems,
                            hipStream_t stream) {
  if (n_elems == 0) {
    return;
  }
  constexpr unsigned int block_threads = 256;
  constexpr unsigned int warps_per_block = block_threads / 32;

  if (type == core::GgmlType::kQ8_0) {
    if (n_elems % kQ8_0BlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ8_0BlockSize;
    constexpr unsigned int blocks_per_tb = warps_per_block * 8u;
    const std::size_t grid = (n_blocks + blocks_per_tb - 1U) / blocks_per_tb;
    DequantizeQ8_0ToBf16Kernel<<<static_cast<unsigned int>(grid), block_threads,
                                 0, stream>>>(w, out, n_blocks);
    return;
  }
  if (type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
      type == core::GgmlType::kQ8_K) {
    if (n_elems % kQ5KBlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ5KBlockSize;
    const std::size_t grid =
        (n_blocks + warps_per_block - 1U) / warps_per_block;
    if (type == core::GgmlType::kQ5_K) {
      DequantizeQ5KToBf16Kernel<<<static_cast<unsigned int>(grid),
                                  block_threads, 0, stream>>>(w, out, n_blocks);
    } else if (type == core::GgmlType::kQ6_K) {
      DequantizeQ6KToBf16Kernel<<<static_cast<unsigned int>(grid),
                                  block_threads, 0, stream>>>(w, out, n_blocks);
    } else {
      LaunchDequantizeQ8KToBf16(w, out, n_elems, stream);
    }
    return;
  }
  // Unsupported quant type: no-op.
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
