#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

__global__ void BatchedUnpackQGKernel(const float* __restrict__ qg,
                                      float* __restrict__ q_out,
                                      float* __restrict__ gate_out,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;

  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  if (idx >= total_h)
    return;

  const std::size_t head_idx = idx / head_dim;
  const std::size_t dim_idx = idx % head_dim;
  const std::size_t src_idx =
      (b * total_h * 2) + (head_idx * 2 * head_dim) + dim_idx;

  q_out[(b * total_h) + idx] = qg[src_idx];
  gate_out[(b * total_h) + idx] = qg[src_idx + head_dim];
}

void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream) {
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_h + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedUnpackQGKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                          stream>>>(qg_interleaved, q_out, gate_out, batch_size,
                                    num_heads, head_dim);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
