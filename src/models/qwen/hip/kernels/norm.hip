#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/norm_residual.hpp"

namespace gufo::hip {

template<std::size_t CachedDim = 0>
__global__ void RMSNormKernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              float* __restrict__ out, std::size_t dim,
                              float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;
  static_assert(CachedDim % 256 == 0);
  constexpr std::size_t cached_items = CachedDim == 0 ? 1 : CachedDim / 256;
  float values[cached_items];
  float weights[cached_items];
  float sum_sq = 0.0F;
  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      const std::size_t i = tid + item * 256;
      const float val = x[i];
      values[item] = val;
      weights[item] = weight != nullptr ? weight[i] : 1.0F;
      // Keep the generic loop's FMA when the compiler unrolls the row.
      sum_sq = __fmaf_rn(val, val, sum_sq);
    }
  } else {
    for (std::size_t i = tid; i < dim; i += blockDim.x) {
      const float val = x[i];
      sum_sq += val * val;
    }
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

  if constexpr (CachedDim != 0) {
#pragma unroll
    for (std::size_t item = 0; item < cached_items; ++item) {
      out[tid + item * 256] = (values[item] * inv_rms) * weights[item];
    }
  } else {
    for (std::size_t i = tid; i < dim; i += blockDim.x) {
      const float w = (weight != nullptr) ? weight[i] : 1.0F;
      out[i] = (x[i] * inv_rms) * w;
    }
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
  // Qwen27B decode reuses this complete row after the reduction. Retain it
  // in registers, including its weights, without changing the sum tree.
  if (dim == 5120) {
    RMSNormKernel<5120><<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
  } else {
    RMSNormKernel<><<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
  }
}
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(x, weight, out, num_heads,
                                                      head_dim, eps);
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
