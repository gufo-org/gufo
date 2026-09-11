#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace gufo::hip {

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_1Row(const hip_bfloat16* __restrict__ A,
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
    const float r = 0.0F;
    y[m] = dot + r;
  }
}

__global__ void FastGEMVBlockKernel(const void* __restrict__ A,
                                    core::GgmlType a_type,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, std::size_t M,
                                    std::size_t K) {
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
      const float r = 0.0F;
      y[m] = block_sum + r;
    }
  }
}

namespace {

void ExecuteDenseGemv(models::qwen::QwenGemmRoute route, const void* A,
                      core::GgmlType a_type, const float* x, float* y,
                      std::size_t M, std::size_t K, hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
  if (route == models::qwen::QwenGemmRoute::kHipBf16Wave32Single) {
    detail::EmitGemvDispatch(M, K, true,
                             models::qwen::QwenGemmRouteName(route));
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32GEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
    return;
  }

  detail::EmitGemvDispatch(M, K, a_type == core::GgmlType::kBF16,
                           models::qwen::QwenGemmRouteName(route));
  const std::size_t block_size =
      route == models::qwen::QwenGemmRoute::kHipF32Baseline512 ||
              route == models::qwen::QwenGemmRoute::kHipBf16Baseline512
          ? 512
          : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, M, K);
}

}  // namespace

void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream,
                models::qwen::QwenGemmMode mode) {
  const auto resolution = models::qwen::ResolveQwenGemmRoute(
      {.type = a_type, .batch_size = 1, .m = M, .k = K, .mode = mode});
  switch (resolution.route) {
    case models::qwen::QwenGemmRoute::kHipQuantDirect:
      LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
      return;
    case models::qwen::QwenGemmRoute::kHipF32Baseline256:
    case models::qwen::QwenGemmRoute::kHipF32Baseline512:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline256:
    case models::qwen::QwenGemmRoute::kHipBf16Baseline512:
    case models::qwen::QwenGemmRoute::kHipBf16Wave32Single:
      ExecuteDenseGemv(resolution.route, A, a_type, x, y, M, K, stream);
      return;
    default:
      std::abort();
  }
}

}  // namespace gufo::hip
#endif
