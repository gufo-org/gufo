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

__global__ void BatchedArgmaxKernel(const float* __restrict__ logits,
                                    std::uint32_t* __restrict__ out_tokens,
                                    std::size_t vocab_size) {
  const std::size_t row = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const float* row_logits = logits + (row * vocab_size);

  float max_val = -1e30F;
  std::uint32_t max_idx = 0;
  for (std::size_t index = tid; index < vocab_size; index += blockDim.x) {
    const float value = row_logits[index];
    if (value > max_val) {
      max_val = value;
      max_idx = static_cast<std::uint32_t>(index);
    }
  }

  __shared__ float shared_max[256];
  __shared__ std::uint32_t shared_index[256];
  shared_max[tid] = max_val;
  shared_index[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
      shared_max[tid] = shared_max[tid + stride];
      shared_index[tid] = shared_index[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_tokens[row] = shared_index[0];
  }
}

void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            hipStream_t stream) {
  if (batch_size == 0 || vocab_size == 0) {
    return;
  }
  BatchedArgmaxKernel<<<static_cast<unsigned int>(batch_size), 256, 0,
                        stream>>>(logits, out_tokens, vocab_size);
}

}  // namespace strix::hip
#endif
