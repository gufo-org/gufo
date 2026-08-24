#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/token.hpp"

namespace strix::hip {

__global__ void ArgmaxKernel(const float* __restrict__ logits,
                             std::uint32_t* __restrict__ out_token,
                             std::size_t vocab_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;

  float max_val = -1e30F;
  std::uint32_t max_idx = 0;

  for (std::size_t i = tid; i < vocab_size; i += block_size) {
    const float val = logits[i];
    if (val > max_val) {
      max_val = val;
      max_idx = static_cast<std::uint32_t>(i);
    }
  }

  __shared__ float s_max[256];
  __shared__ std::uint32_t s_idx[256];
  s_max[tid] = max_val;
  s_idx[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      if (s_max[tid + stride] > s_max[tid]) {
        s_max[tid] = s_max[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out_token = s_idx[0];
  }
}

void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream) {
  ArgmaxKernel<<<1, 256, 0, stream>>>(logits, out_token, vocab_size);
}

}  // namespace strix::hip
#endif
