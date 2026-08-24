#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel(const void* __restrict__ A, core::GgmlType type,
                            const float* __restrict__ x,
                            float* __restrict__ y, std::size_t M,
                            std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  float sumf = 0.0F;
  if (type == core::GgmlType::kQ8_K) {
    const std::size_t num_blocks = K / kQ8KBlockSize;
    const auto* row_blocks =
        static_cast<const Q8KBlock*>(A) + (m * num_blocks);

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlock& wblk = row_blocks[b];
      const float d_w = wblk.d;
      const float* xb = x + (b * kQ8KBlockSize);

      // Pass 1: per-lane max|x| over the 8 x values this lane owns, then a warp
      // reduction to get the block max and derive the activation scale.
      float local_max = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        local_max = fmaxf(local_max, fabsf(xb[lane_id + (32 * k)]));
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;

      // Pass 2: quantize x to int8 and integer MAC with the weight qs.
      int acc = 0;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32 * k);
        const float xv = (scale > 0.0F) ? (xb[idx] / scale) : 0.0F;
        int xq = static_cast<int>(roundf(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[idx]) * xq;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }

      sumf += d_w * scale * static_cast<float>(acc);
    }
  } else if (type == core::GgmlType::kQ8_0) {
    // Q8_0 path (QK = 32 = warp width): one element per lane. Scale derivation,
    // quantize/clamp, int-MAC and end-scale mirror the Q8_K path AND
    // Q8_0RowDot exactly (verified by the Q8_0 GEMV equivalence test).
    const std::size_t num_blocks = K / kQ8_0BlockSize;
    const auto* row_blocks =
        static_cast<const Q8_0Block*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8_0Block& wblk = row_blocks[b];
      const float d_w = __half2float(wblk.d);
      const float* xb = x + (b * kQ8_0BlockSize);

      float local_max = fabsf(xb[lane_id]);
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;

      const float xv = (scale > 0.0F) ? (xb[lane_id] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      int acc = static_cast<int>(wblk.qs[lane_id]) * xq;
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }

      sumf += d_w * scale * static_cast<float>(acc);
    }
  } else if (type == core::GgmlType::kQ5_K) {
    // Q5_K path (QK = 256): dequant each 256-wide element to fp via Q5KValue
    // and accumulate a fp MAC against x. One warp per row; each lane owns 8
    // elements (lane_id + 32*k) then a warp reduction. Mirrors CPU
    // DequantizeQ5_K + dot exactly modulo fp accumulation order.
    const std::size_t num_blocks = K / kQ5KBlockSize;
    const auto* row_blocks =
        static_cast<const Q5KBlock*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q5KBlock& wblk = row_blocks[b];
      const float* xb = x + (b * kQ5KBlockSize);
      float acc = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32u * k);
        acc += Q5KValue(wblk, idx) * xb[idx];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }
      sumf += acc;
    }
  } else if (type == core::GgmlType::kQ6_K) {
    // Q6_K path (QK = 256): dequant each 256-wide element to fp via Q6KValue
    // and accumulate a fp MAC against x. Mirrors CPU DequantizeQ6_K + dot.
    const std::size_t num_blocks = K / kQ6KBlockSize;
    const auto* row_blocks =
        static_cast<const Q6KBlock*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q6KBlock& wblk = row_blocks[b];
      const float* xb = x + (b * kQ6KBlockSize);
      float acc = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32u * k);
        acc += Q6KValue(wblk, idx) * xb[idx];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }
      sumf += acc;
    }
  }

  if (lane_id == 0) {
    y[m] = sumf;
  }
}

// opt-c1xx-q8k-gemv: launches the Q8_K on-the-fly integer-dot GEMV. One warp
// per output row (4 warps per block). Requires K % 256 == 0 (Q8_K block size).
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
  const std::size_t grid = (M + kWavesPerBlock - 1U) / kWavesPerBlock;
  Q8KBlockGEMVKernel<kWavesPerBlock>
      <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
