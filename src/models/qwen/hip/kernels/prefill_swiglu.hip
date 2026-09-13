#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

__global__ void BatchedSwiGLUActivationKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    const std::size_t source = idx + blockIdx.y * 2 * num_elements;
    const std::size_t destination = idx + blockIdx.y * num_elements;
    const float g = gate[source];
    const float u = up[source];
    const float sig_g = 1.0F / (1.0F + expf(-g));
    const float val = (g * sig_g) * u;
    if (out != nullptr) {
      out[destination] = val;
    }
    if (out_bf16 != nullptr) {
      out_bf16[destination] = hip_bfloat16(val);
    }
  }
}

void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  BatchedSwiGLUActivationKernel<<<num_blocks, block_size, 0, stream>>>(
      gate, up, out, static_cast<hip_bfloat16*>(out_bf16), num_elements);
}

void LaunchPackedSwiGLUActivation(const float* gate_up, float* out,
                                   std::size_t batch_size,
                                   std::size_t intermediate_size,
                                   hipStream_t stream) {
  if (batch_size == 0 || intermediate_size == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>((intermediate_size + 255) / 256),
                  static_cast<unsigned int>(batch_size));
  BatchedSwiGLUActivationKernel<<<grid, 256, 0, stream>>>(
      gate_up, gate_up + intermediate_size, out, nullptr, intermediate_size);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
