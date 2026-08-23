#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen/hip/qwen_gpu_quant_ops.hpp"

namespace strix::hip {

__global__ void BatchedGEMMKernel(const void* __restrict__ A, bool is_bf16,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y, std::size_t batch_size,
                                  std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= M || b >= batch_size)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t block_threads = blockDim.x;
  const float* x_row = X + (b * K);
  float* y_row = Y + (b * M);

  float dot = 0.0F;
  if (is_bf16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
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
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      y_row[m] = block_sum;
    }
  }
}

void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(M),
                  static_cast<unsigned int>(batch_size));
  BatchedGEMMKernel<<<grid, 128, 0, stream>>>(A, is_bf16, X, Y, batch_size, M,
                                              K);
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
// Q8_K block layout is defined in the shared qwen_gpu_quant_ops.hpp header.

// Q5_K / Q6_K block layouts are defined in the shared qwen_gpu_quant_ops.hpp
// header; dequant to BF16 uses Q5KValueFp / Q6KValueFp below.
__global__ void DequantizeQ8KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  const auto* blk = static_cast<const Q8KBlock*>(w) + b;
  out[(b * kQ8KBlockSize) + j] =
      hip_bfloat16(blk->d * static_cast<float>(blk->qs[j]));
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
  DequantizeQ8KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                              static_cast<unsigned int>(kQ8KBlockSize), 0,
                              stream>>>(w, out, n_blocks);
}

// Per-value dequant helpers (fp32), matching the CPU oracles
// Q5Value/Q6Value/DequantizeQ8_0 in ggml_dequant.cpp. GetQKScaleMin (used by
// Q5KValueFp) is defined in the shared qwen_gpu_quant_ops.hpp header.
__device__ float Q5KValueFp(const Q5KBlock& block,
                            std::size_t index) noexcept {
  const std::size_t gg = index / 64;   // 0..3
  const std::size_t wv = index % 64;   // 0..63
  const std::size_t lane = wv % 32;    // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((block.qh[lane] >> bit) & 1U) ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);  // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQKScaleMin(sis, block.scales, sc, m);
  return __half2float(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         __half2float(block.dmin) * static_cast<float>(m);
}

__device__ float Q6KValueFp(const Q6KBlock& block,
                            std::size_t index) noexcept {
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

// Type-dispatched dequant-to-BF16 kernels. One block per quant block, one
// thread per element; out[b*qk + j] = (hip_bfloat16)(per_value_dequant).
__global__ void DequantizeQ8_0ToBf16Kernel(const void* __restrict__ w,
                                           hip_bfloat16* __restrict__ out,
                                           std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ8_0BlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q8_0Block*>(w) + b;
  out[(b * kQ8_0BlockSize) + j] = hip_bfloat16(Q8_0ValueFp(*blk, j));
}

__global__ void DequantizeQ5KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ5KBlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q5KBlock*>(w) + b;
  out[(b * kQ5KBlockSize) + j] = hip_bfloat16(Q5KValueFp(*blk, j));
}

__global__ void DequantizeQ6KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ6KBlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q6KBlock*>(w) + b;
  out[(b * kQ6KBlockSize) + j] = hip_bfloat16(Q6KValueFp(*blk, j));
}

// opt-c162-q8k-prefill-dequant (extended): type-dispatched entry point that
// converts a quantized weight region (Q8_0/Q5_K/Q6_K/Q8_K) into a BF16 scratch
// for the prefill hipblas GEMM path. Q8_0 uses a 32-thread block (QK=32);
// Q5_K/Q6_K/Q8_K use a 256-thread block (QK=256). Requires n_elems to be a
// whole number of blocks. Unsupported types and non-multiple sizes are no-ops.
void LaunchDequantizeToBf16(core::GgmlType type, const void* w,
                            hip_bfloat16* out, std::size_t n_elems,
                            hipStream_t stream) {
  if (n_elems == 0) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    if (n_elems % kQ8_0BlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ8_0BlockSize;
    DequantizeQ8_0ToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                 static_cast<unsigned int>(kQ8_0BlockSize), 0,
                                 stream>>>(w, out, n_blocks);
    return;
  }
  if (type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
      type == core::GgmlType::kQ8_K) {
    if (n_elems % kQ5KBlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ5KBlockSize;
    if (type == core::GgmlType::kQ5_K) {
      DequantizeQ5KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                  static_cast<unsigned int>(kQ5KBlockSize), 0,
                                  stream>>>(w, out, n_blocks);
    } else if (type == core::GgmlType::kQ6_K) {
      DequantizeQ6KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                  static_cast<unsigned int>(kQ6KBlockSize), 0,
                                  stream>>>(w, out, n_blocks);
    } else {
      // kQ8_K: reuse the existing single-type kernel (one implementation).
      LaunchDequantizeQ8KToBf16(w, out, n_elems, stream);
    }
    return;
  }
  // Unsupported quant type: no-op.
}

// opt-c162-prefill-quant-direct: batched prefill projection that computes the
// matmul DIRECTLY on quantized weights (block_q8_0 / block_q8_K / block_q5_K /
// block_q6_K) with NO dequantize-to-BF16 and NO BF16 GEMM. One warp per output
// row; each warp reads the row's quant blocks and the bf16 activation and
// reduces via QuantWarpBlockDot (num_warps=1 => full row dot in lane 0).
// Output y[b*m + row] = qdot(weights[row], x[b]).
__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const hip_bfloat16* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot = QuantWarpBlockDot<hip_bfloat16>(type, w, row, x + (b * k),
                                                    k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    return;
  }
  constexpr std::size_t kWarp = 32;
  const std::size_t warps_per_block = 8;
  const std::size_t rows_per_block = warps_per_block;  // one row per warp
  const std::size_t num_blocks_x = (m + rows_per_block - 1) / rows_per_block;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * warps_per_block));
  BatchedQuantGEMVKernel<<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
