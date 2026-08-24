#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/norm_residual.hpp"

namespace strix::hip {

__global__ void ResidualAddKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, std::size_t dim) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    out[idx] = a[idx] + b[idx];
  }
}

// opt-c010-residual-rmsnorm: fuses the residual add (hidden += attn_out) with
// the subsequent RMSNorm into one launch. Each input element is read once and
// both the residual-updated sum and the normed output are written. The
// reduction order matches the unfused chain (ResidualAddKernel +
// RMSNormKernel) so the normed output is bit-identical.
__launch_bounds__(256, 4) __global__ void FusedResidualAddRMSNormKernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out_sum, const float* __restrict__ weight,
    float* __restrict__ out, std::size_t dim, float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = a[i] + b[i];
    out_sum[i] = val;
    sum_sq += val * val;
  }
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
    out[i] = (out_sum[i] * inv_rms) * w;
  }
}

void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (dim + block_size - 1) / block_size;
  ResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out, dim);
}

void LaunchFusedResidualAddRMSNorm(const float* a, const float* b,
                                   float* out_sum, const float* weight,
                                   float* out, std::size_t dim, float eps,
                                   hipStream_t stream) {
  FusedResidualAddRMSNormKernel<<<1, 256, 0, stream>>>(a, b, out_sum, weight,
                                                       out, dim, eps);
}

}  // namespace strix::hip
#endif
