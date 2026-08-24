#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace strix::hip {

constexpr std::size_t kLargeGemvInput = 16384;

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_1Row(const hip_bfloat16* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               const float* __restrict__ residual,
                               std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* row = A + (m * K);
  const auto* row_vec = reinterpret_cast<const uint4*>(row);
  const std::size_t K_vec = K / 8;

  float dot = 0.0F;
  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 v = row_vec[vi];
    const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

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
    dot += static_cast<float>(row[k]) * x[k];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    const float r = (residual != nullptr) ? residual[m] : 0.0F;
    y[m] = dot + r;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_2Rows(const hip_bfloat16* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y, std::size_t M,
                                std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* row0 = A + (m0 * K);
  const auto* row1 = (m1 < M) ? A + (m1 * K) : nullptr;
  const auto* row0_vec = reinterpret_cast<const uint4*>(row0);
  const auto* row1_vec = reinterpret_cast<const uint4*>(row1);
  const std::size_t K_vec = K / 8;

  float dot0 = 0.0F;
  float dot1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 v0 = row0_vec[vi];
    const auto* vals0 = reinterpret_cast<const hip_bfloat16*>(&v0);
    dot0 += (static_cast<float>(vals0[0]) * x0.x) +
            (static_cast<float>(vals0[1]) * x0.y) +
            (static_cast<float>(vals0[2]) * x0.z) +
            (static_cast<float>(vals0[3]) * x0.w) +
            (static_cast<float>(vals0[4]) * x1.x) +
            (static_cast<float>(vals0[5]) * x1.y) +
            (static_cast<float>(vals0[6]) * x1.z) +
            (static_cast<float>(vals0[7]) * x1.w);

    if (row1 != nullptr) {
      const uint4 v1 = row1_vec[vi];
      const auto* vals1 = reinterpret_cast<const hip_bfloat16*>(&v1);
      dot1 += (static_cast<float>(vals1[0]) * x0.x) +
              (static_cast<float>(vals1[1]) * x0.y) +
              (static_cast<float>(vals1[2]) * x0.z) +
              (static_cast<float>(vals1[3]) * x0.w) +
              (static_cast<float>(vals1[4]) * x1.x) +
              (static_cast<float>(vals1[5]) * x1.y) +
              (static_cast<float>(vals1[6]) * x1.z) +
              (static_cast<float>(vals1[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot0 += static_cast<float>(row0[k]) * xv;
    if (row1 != nullptr) {
      dot1 += static_cast<float>(row1[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot0 += __shfl_xor(dot0, offset);
    dot1 += __shfl_xor(dot1, offset);
  }

  if (lane_id == 0) {
    y[m0] = dot0;
    if (m1 < M) {
      y[m1] = dot1;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_4Rows(const hip_bfloat16* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y, std::size_t M,
                                std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_quad = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_quad * 4;
  const std::size_t m1 = m0 + 1;
  const std::size_t m2 = m0 + 2;
  const std::size_t m3 = m0 + 3;
  if (m0 >= M) {
    return;
  }

  const auto* row0 = A + (m0 * K);
  const auto* row1 = (m1 < M) ? A + (m1 * K) : nullptr;
  const auto* row2 = (m2 < M) ? A + (m2 * K) : nullptr;
  const auto* row3 = (m3 < M) ? A + (m3 * K) : nullptr;
  const auto* row0_vec = reinterpret_cast<const uint4*>(row0);
  const auto* row1_vec = reinterpret_cast<const uint4*>(row1);
  const auto* row2_vec = reinterpret_cast<const uint4*>(row2);
  const auto* row3_vec = reinterpret_cast<const uint4*>(row3);
  const std::size_t K_vec = K / 8;

  float dot0 = 0.0F;
  float dot1 = 0.0F;
  float dot2 = 0.0F;
  float dot3 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 v0 = row0_vec[vi];
    const auto* vals0 = reinterpret_cast<const hip_bfloat16*>(&v0);
    dot0 += (static_cast<float>(vals0[0]) * x0.x) +
            (static_cast<float>(vals0[1]) * x0.y) +
            (static_cast<float>(vals0[2]) * x0.z) +
            (static_cast<float>(vals0[3]) * x0.w) +
            (static_cast<float>(vals0[4]) * x1.x) +
            (static_cast<float>(vals0[5]) * x1.y) +
            (static_cast<float>(vals0[6]) * x1.z) +
            (static_cast<float>(vals0[7]) * x1.w);

    if (row1 != nullptr) {
      const uint4 v1 = row1_vec[vi];
      const auto* vals1 = reinterpret_cast<const hip_bfloat16*>(&v1);
      dot1 += (static_cast<float>(vals1[0]) * x0.x) +
              (static_cast<float>(vals1[1]) * x0.y) +
              (static_cast<float>(vals1[2]) * x0.z) +
              (static_cast<float>(vals1[3]) * x0.w) +
              (static_cast<float>(vals1[4]) * x1.x) +
              (static_cast<float>(vals1[5]) * x1.y) +
              (static_cast<float>(vals1[6]) * x1.z) +
              (static_cast<float>(vals1[7]) * x1.w);
    }
    if (row2 != nullptr) {
      const uint4 v2 = row2_vec[vi];
      const auto* vals2 = reinterpret_cast<const hip_bfloat16*>(&v2);
      dot2 += (static_cast<float>(vals2[0]) * x0.x) +
              (static_cast<float>(vals2[1]) * x0.y) +
              (static_cast<float>(vals2[2]) * x0.z) +
              (static_cast<float>(vals2[3]) * x0.w) +
              (static_cast<float>(vals2[4]) * x1.x) +
              (static_cast<float>(vals2[5]) * x1.y) +
              (static_cast<float>(vals2[6]) * x1.z) +
              (static_cast<float>(vals2[7]) * x1.w);
    }
    if (row3 != nullptr) {
      const uint4 v3 = row3_vec[vi];
      const auto* vals3 = reinterpret_cast<const hip_bfloat16*>(&v3);
      dot3 += (static_cast<float>(vals3[0]) * x0.x) +
              (static_cast<float>(vals3[1]) * x0.y) +
              (static_cast<float>(vals3[2]) * x0.z) +
              (static_cast<float>(vals3[3]) * x0.w) +
              (static_cast<float>(vals3[4]) * x1.x) +
              (static_cast<float>(vals3[5]) * x1.y) +
              (static_cast<float>(vals3[6]) * x1.z) +
              (static_cast<float>(vals3[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot0 += static_cast<float>(row0[k]) * xv;
    if (row1 != nullptr) {
      dot1 += static_cast<float>(row1[k]) * xv;
    }
    if (row2 != nullptr) {
      dot2 += static_cast<float>(row2[k]) * xv;
    }
    if (row3 != nullptr) {
      dot3 += static_cast<float>(row3[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot0 += __shfl_xor(dot0, offset);
    dot1 += __shfl_xor(dot1, offset);
    dot2 += __shfl_xor(dot2, offset);
    dot3 += __shfl_xor(dot3, offset);
  }

  if (lane_id == 0) {
    y[m0] = dot0;
    if (m1 < M) {
      y[m1] = dot1;
    }
    if (m2 < M) {
      y[m2] = dot2;
    }
    if (m3 < M) {
      y[m3] = dot3;
    }
  }
}

__global__ void FastGEMVBlockKernel(const void* __restrict__ A,
                                    core::GgmlType a_type,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const float* __restrict__ residual,
                                    std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (a_type == core::GgmlType::kBF16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

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
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[16];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      const float r = (residual != nullptr) ? residual[m] : 0.0F;
      y[m] = block_sum + r;
    }
  }
}

void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream) {
  // Quantized weights (kQ8_K now; more types added later) run the Q row-dot
  // path, which has no residual epilogue.
  if (a_type != core::GgmlType::kBF16 && a_type != core::GgmlType::kF32) {
    LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
    return;
  }
  const bool is_bf16 = (a_type == core::GgmlType::kBF16);
  const auto strategy = detail::SelectGemvStrategy(M, K, is_bf16);
  detail::EmitGemvDispatch(M, K, is_bf16, detail::GemvStrategyName(strategy));
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  switch (strategy) {
    case detail::GemvStrategy::kWave32QuadRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t num_quads = (M + 3) / 4;
      const std::size_t grid =
          (num_quads + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_4Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
      return;
    }
    case detail::GemvStrategy::kWave32DualRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t num_pairs = (M + 1) / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
      return;
    }
    case detail::GemvStrategy::kWave32SingleRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_1Row<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, nullptr, M, K);
      return;
    }
    case detail::GemvStrategy::kBaselineBlock:
    default:
      break;
  }

  const std::size_t block_size = K >= kLargeGemvInput ? 512 : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, nullptr,
                                                    M, K);
}

// opt-c010-ssm-gate-residual: GEMV with an optional residual-add epilogue
// (y = A*x + residual). The dispatch mirrors LaunchGEMV; the Wave32 single-row
// and block strategies gain the residual fold, and the remaining strategies
// fall back to the single-row residual kernel. Used to fold the post-SSM
// residual add into the ssm_out GEMV for decode.
void LaunchGEMVResidual(const void* A, core::GgmlType a_type, const float* x,
                        float* y, const float* residual, std::size_t M,
                        std::size_t K, hipStream_t stream) {
  // Quantized weights run the Q row-dot path. The residual add is not folded
  // into the Q8K block GEMV, so a residual pointer is not applied (this path is
  // only reached for the ssm_out residual, which is currently dead).
  if (a_type != core::GgmlType::kBF16 && a_type != core::GgmlType::kF32) {
    LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
    return;
  }
  const bool is_bf16 = (a_type == core::GgmlType::kBF16);
  const auto strategy = detail::SelectGemvStrategy(M, K, is_bf16);
  detail::EmitGemvDispatch(M, K, is_bf16, detail::GemvStrategyName(strategy));
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  if (strategy != detail::GemvStrategy::kBaselineBlock) {
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32GEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, residual, M, K);
    return;
  }

  const std::size_t block_size = K >= kLargeGemvInput ? 512 : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, residual,
                                                    M, K);
}

}  // namespace strix::hip
#endif
