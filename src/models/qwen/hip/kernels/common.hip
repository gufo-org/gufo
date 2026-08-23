#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strix::hip {
// opt-c010-rmsnorm-projection: computes the layer RMSNorm scale exactly like
// RMSNormKernel (256-thread strided partition over dim, tree reduction with
// stride 128..1 over s_sum[256]) so inv_rms is bit-identical to the unfused
// chain. All 256 threads must reach the embedded __syncthreads() calls before
// any divergent per-warp exit.
__device__ inline float FusedRMSNormInvScale(const float* __restrict__ x,
                                             const float* __restrict__ weight,
                                             std::size_t dim, float eps,
                                             float* __restrict__ s_sum) {
  const std::size_t tid = threadIdx.x;
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += 256U) {
    const float val = x[i];
    sum_sq += val * val;
  }
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = 128U; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  return rsqrtf(mean_sq + eps);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
