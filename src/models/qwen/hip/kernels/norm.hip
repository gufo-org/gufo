#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {

__global__ void RMSNormKernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out, std::size_t dim,
                              float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = x[i];
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
    out[i] = (x[i] * inv_rms) * w;
  }
}
__global__ void PerHeadRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     std::uint32_t num_heads,
                                     std::uint32_t head_dim, float eps) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const float* x_head = x + (static_cast<std::size_t>(h) * head_dim);
  float* out_head = out + (static_cast<std::size_t>(h) * head_dim);

  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
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

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps, hipStream_t stream) {
  RMSNormKernel<<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
}
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(x, weight, out, num_heads,
                                                      head_dim, eps);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
