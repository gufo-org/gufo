#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

__global__ void BatchedResidualAddKernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         std::size_t total_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < total_elements) {
    out[idx] = a[idx] + b[idx];
  }
}

void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream) {
  const std::size_t total = batch_size * dim;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total + block_size - 1) / block_size;
  BatchedResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out,
                                                                 total);
}

// opt-c010-residual-rmsnorm: batched variant of FusedResidualAddRMSNormKernel.
// Block b handles token b; writes the residual-updated sum row, the FP32
// normed row, and the optional BF16 normed row. Reduction order matches the
// unfused chain (BatchedResidualAddKernel + BatchedRMSNormKernel).
__launch_bounds__(256, 4) __global__ void BatchedFusedResidualAddRMSNormKernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out_sum, const float* __restrict__ weight,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t batch_size, std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const float* a_row = a + (row * dim);
  const float* b_row = b + (row * dim);
  float* sum_row = (out_sum != nullptr) ? out_sum + (row * dim) : nullptr;
  float* out_row = (out != nullptr) ? out + (row * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (row * dim) : nullptr;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = a_row[i] + b_row[i];
    if (sum_row != nullptr) {
      sum_row[i] = val;
    }
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    const float normed = (sum_row[i] * inv_rms) * w;
    if (out_row != nullptr) {
      out_row[i] = normed;
    }
    if (out_bf16_row != nullptr) {
      out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

void LaunchBatchedFusedResidualAddRMSNorm(const float* a, const float* b,
                                          float* out_sum, const float* weight,
                                          float* out, void* out_bf16,
                                          std::size_t batch_size,
                                          std::size_t dim, float eps,
                                          hipStream_t stream) {
  BatchedFusedResidualAddRMSNormKernel<<<batch_size, 256, 0, stream>>>(
      a, b, out_sum, weight, out, static_cast<hip_bfloat16*>(out_bf16),
      batch_size, dim, eps);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
