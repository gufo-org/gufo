#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/small_batch_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

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
void LaunchExactBf16GEMMFp32SmallBatch(const void* A, const float* X, float* Y,
                                       std::size_t batch_size, std::size_t M,
                                       std::size_t K, hipStream_t stream) {
  if (M == 0 || K == 0 || batch_size == 0) {
    return;
  }
  const bool exact_width =
      ((K == 5120 && (M == 12288 || M == 248320)) ||
       (M == 5120 && K == 6144));
  const bool wide = batch_size > 16 && exact_width;
  if (exact_width && batch_size >= 9 && batch_size <= 15 && batch_size != 12) {
    if (M == 5120 && K == 6144) {
      const auto launch = [&]<std::size_t Batch>() {
        BatchedExactBf16GEMMFp32VecKernel<16, Batch, 1>
            <<<(M + 15) / 16, 512, 0, stream>>>(
                static_cast<const hip_bfloat16*>(A), X, Y, M, K);
      };
      switch (batch_size) {
        case 9: launch.template operator()<9>(); break;
        case 10: launch.template operator()<10>(); break;
        case 11: launch.template operator()<11>(); break;
        case 13: launch.template operator()<13>(); break;
        case 14: launch.template operator()<14>(); break;
        case 15: launch.template operator()<15>(); break;
      }
      return;
    }
    if (detail::TryLaunchBf16SmallBatchWave64(
            A, X, Y, batch_size, M, K, stream))
      return;
  }
  if (wide &&
      detail::TryLaunchBf16SmallBatchWave64(A, X, Y, batch_size, M, K, stream))
    return;
  if (batch_size > 8 && batch_size != 12 && batch_size != 16) {
    for (std::size_t offset = 0; offset < batch_size;) {
      const std::size_t remaining = batch_size - offset;
      // Balanced groups avoid an expensive short remainder at common cohort
      // sizes: 36=24+12, 40=24+16, 48=24+24 and 60=24+24+12.
      const bool prefer24 =
          remaining == 36 || remaining == 40 || remaining == 48 || remaining == 60;
      const std::size_t width = wide && remaining >= 32 && !prefer24 ? 32
                                : wide && remaining >= 24 && remaining != 28
                                    ? 24
                                : remaining >= 16 ? 16
                                : exact_width && remaining > 8 ? remaining
                                : remaining >= 12
                                    ? 12
                                    : std::min<std::size_t>(8, remaining);
      LaunchExactBf16GEMMFp32SmallBatch(A, X + offset * K, Y + offset * M,
                                        width, M, K, stream);
      offset += width;
    }
    return;
  }
  constexpr std::uint32_t kVecWaves = 16;
  constexpr std::size_t kVecRowsPerWave = 2;
  constexpr std::size_t kVecRowsPerBlock = kVecWaves * kVecRowsPerWave;
  const unsigned int vec_blocks =
      static_cast<unsigned int>((M + kVecRowsPerBlock - 1) / kVecRowsPerBlock);
  const auto* a_vec = static_cast<const hip_bfloat16*>(A);
  // Injection reuses each 10 MiB K/V matrix across sixteen-row chunks.
  // Keep it cached; the much larger feature projection and decode weights
  // use streaming loads. Specialize the hint to keep it out of the GPU loop.
  if (batch_size == 16 && M == 1024 && K == 5120) {
    BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 16, kVecRowsPerWave, false>
        <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
    return;
  }
  if ((batch_size == 12 || batch_size == 16) && M == 5120 && K == 6144) {
    const auto launch = [&]<std::size_t Batch>() {
      BatchedExactBf16GEMMFp32VecKernel<16, Batch, 1>
          <<<(M + 15) / 16, 512, 0, stream>>>(a_vec, X, Y, M, K);
    };
    if (batch_size == 12)
      launch.template operator()<12>();
    else
      launch.template operator()<16>();
    return;
  }
  if (batch_size == 12 && K == 5120 &&
      (M == 12288 || M == 17408 || M == 248320)) {
    BatchedExactBf16GEMMFp32VecKernel<8, 12, 2>
        <<<(M + 15) / 16, 256, 0, stream>>>(a_vec, X, Y, M, K);
    return;
  }
  // One row per wave wins on DFlash2's attention output/FFN and the target's
  // large BF16 query projection; each preserves the same accumulation order.
  if (batch_size == 8 &&
      ((M == 5120 && K == 4096) || (K == 5120 && (M == 12288 || M == 17408)))) {
    BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 8, 1>
        <<<(M + kVecWaves - 1) / kVecWaves, kVecWaves * 32, 0, stream>>>(
            a_vec, X, Y, M, K);
    return;
  }
  switch (batch_size) {
    case 12:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 12, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 16:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 16, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 1:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 1, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 2:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 2, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 3:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 3, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 4:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 4, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 5:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 5, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 6:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 6, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    case 7:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 7, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
    default:
      BatchedExactBf16GEMMFp32VecKernel<kVecWaves, 8, kVecRowsPerWave>
          <<<vec_blocks, kVecWaves * 32, 0, stream>>>(a_vec, X, Y, M, K);
      return;
  }
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
