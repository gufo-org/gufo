#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/token.hpp"

namespace strix::hip {

__global__ void RoPEKernel(float* __restrict__ q, float* __restrict__ k,
                           std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           std::uint32_t head_dim, std::uint32_t rotary_dim,
                           std::uint32_t pos, float rope_theta) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_q_pairs =
      static_cast<std::size_t>(num_heads) * half_rot;
  const std::size_t total_kv_pairs =
      static_cast<std::size_t>(num_kv_heads) * half_rot;

  // Process Q heads
  if (tid < total_q_pairs) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

    const float q0 = q[idx0];
    const float q1 = q[idx1];
    q[idx0] = q0 * cos_val - q1 * sin_val;
    q[idx1] = q0 * sin_val + q1 * cos_val;
  }

  // Process K heads
  if (tid < total_kv_pairs && k != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

    const float k0 = k[idx0];
    const float k1 = k[idx1];
    k[idx0] = k0 * cos_val - k1 * sin_val;
    k[idx1] = k0 * sin_val + k1 * cos_val;
  }
}

__global__ void RoPEPtrKernel(float* __restrict__ q, float* __restrict__ k,
                              std::uint32_t num_heads,
                              std::uint32_t num_kv_heads,
                              std::uint32_t head_dim, std::uint32_t rotary_dim,
                              const std::uint32_t* __restrict__ d_pos,
                              float rope_theta) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_q_pairs =
      static_cast<std::size_t>(num_heads) * half_rot;
  const std::size_t total_kv_pairs =
      static_cast<std::size_t>(num_kv_heads) * half_rot;

  const std::uint32_t pos = *d_pos;

  // Process Q heads
  if (tid < total_q_pairs) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

    const float q0 = q[idx0];
    const float q1 = q[idx1];
    q[idx0] = q0 * cos_val - q1 * sin_val;
    q[idx1] = q0 * sin_val + q1 * cos_val;
  }

  // Process K heads
  if (tid < total_kv_pairs && k != nullptr) {
    const std::uint32_t h = static_cast<std::uint32_t>(tid / half_rot);
    const std::uint32_t pair = static_cast<std::uint32_t>(tid % half_rot);

    const float exponent =
        static_cast<float>(2 * pair) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t idx0 = static_cast<std::size_t>(h) * head_dim + pair;
    const std::size_t idx1 = idx0 + half_rot;

    const float k0 = k[idx0];
    const float k1 = k[idx1];
    k[idx0] = k0 * cos_val - k1 * sin_val;
    k[idx1] = k0 * sin_val + k1 * cos_val;
  }
}

void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float rope_theta,
                hipStream_t stream) {
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * half_rot,
               static_cast<std::size_t>(num_kv_heads) * half_rot);
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total_pairs + block_size - 1) / block_size;
  RoPEKernel<<<grid_size, block_size, 0, stream>>>(
      q, k, num_heads, num_kv_heads, head_dim, rotary_dim, pos, rope_theta);
}

void LaunchRoPE(float* q, float* k, std::uint32_t num_heads,
                std::uint32_t num_kv_heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, const std::uint32_t* d_pos,
                float rope_theta, hipStream_t stream) {
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * half_rot,
               static_cast<std::size_t>(num_kv_heads) * half_rot);
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total_pairs + block_size - 1) / block_size;
  RoPEPtrKernel<<<grid_size, block_size, 0, stream>>>(
      q, k, num_heads, num_kv_heads, head_dim, rotary_dim, d_pos, rope_theta);
}

}  // namespace strix::hip
#endif
