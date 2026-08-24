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

__global__ void BatchedSwiGLUActivationKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    const float g = gate[idx];
    const float u = up[idx];
    const float sig_g = 1.0F / (1.0F + expf(-g));
    const float val = (g * sig_g) * u;
    if (out != nullptr) {
      out[idx] = val;
    }
    if (out_bf16 != nullptr) {
      out_bf16[idx] = hip_bfloat16(val);
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

__global__ void BatchedFusedSwiGLUGEMMKernel(
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ up_w, bool up_is_bf16, const float* __restrict__ X,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t batch_size, std::size_t intermediate_size,
    std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= intermediate_size || b >= batch_size)
    return;

  const float* x_row = X + (b * hidden_size);
  float* out_row = out + (b * intermediate_size);
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * intermediate_size) : nullptr;
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_is_bf16 && up_is_bf16) {
    const auto* g_row =
        static_cast<const hip_bfloat16*>(gate_w) + (m * hidden_size);
    const auto* u_row =
        static_cast<const hip_bfloat16*>(up_w) + (m * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[8];
  __shared__ float s_up_sums[8];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      const float val = (block_gate * sig_g) * block_up;
      out_row[m] = val;
      if (out_bf16_row != nullptr) {
        out_bf16_row[m] = hip_bfloat16(val);
      }
    }
  }
}

void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out, void* out_bf16,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(intermediate_size),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedSwiGLUGEMMKernel<<<grid, 128, 0, stream>>>(
      gate_w, gate_is_bf16, up_w, up_is_bf16, X, out,
      static_cast<hip_bfloat16*>(out_bf16), batch_size, intermediate_size,
      hidden_size);
}
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
