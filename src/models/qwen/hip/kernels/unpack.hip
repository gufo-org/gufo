#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {

__global__ void UnpackQGKernel(const float* __restrict__ qg_interleaved,
                               float* __restrict__ q_out,
                               float* __restrict__ gate_out,
                               std::uint32_t num_heads,
                               std::uint32_t head_dim) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(num_heads) * head_dim;
  if (tid >= total)
    return;

  const std::uint32_t h = static_cast<std::uint32_t>(tid / head_dim);
  const std::uint32_t d = static_cast<std::uint32_t>(tid % head_dim);

  const std::size_t src_base = static_cast<std::size_t>(h) * head_dim * 2;
  q_out[tid] = qg_interleaved[src_base + d];
  gate_out[tid] = qg_interleaved[src_base + head_dim + d];
}

void LaunchUnpackQG(const float* qg_interleaved, float* q_out, float* gate_out,
                    std::uint32_t num_heads, std::uint32_t head_dim,
                    hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total + block_size - 1) / block_size;
  UnpackQGKernel<<<grid_size, block_size, 0, stream>>>(
      qg_interleaved, q_out, gate_out, num_heads, head_dim);
}

}  // namespace strix::hip
#endif
