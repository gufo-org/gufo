#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/core/hip/qwen_gpu_quant_ops.hpp"
#include "src/core/gguf_reader.hpp"

namespace strix::hip {

namespace {

constexpr std::size_t kLargeGemvInput = 16384;

// Quant block layouts (Q8_0 / Q5_K / Q6_K) and quant row-dot helpers are
// shared via src/core/hip/qwen_gpu_quant_ops.hpp.

__global__ void EmbeddingLookupKernel(const void* __restrict__ table,
                                      core::GgmlType type, std::uint32_t token_id,
                                      float* __restrict__ out_hidden,
                                      std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;

  const std::size_t offset =
      static_cast<std::size_t>(token_id) * hidden_size + idx;
  if (type == core::GgmlType::kBF16) {
    const auto* bf_ptr = static_cast<const hip_bfloat16*>(table);
    out_hidden[idx] = static_cast<float>(bf_ptr[offset]);
  } else if (type == core::GgmlType::kF32) {
    const auto* f_ptr = static_cast<const float*>(table);
    out_hidden[idx] = f_ptr[offset];
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[static_cast<std::size_t>(token_id) * (hidden_size / kQ8_0BlockSize) +
           idx / kQ8_0BlockSize];
    out_hidden[idx] = __half2float(wblk.d) *
                      static_cast<float>(wblk.qs[idx % kQ8_0BlockSize]);
  }
}

__global__ void EmbeddingLookupPtrKernel(
    const void* __restrict__ table, core::GgmlType type,
    const std::uint32_t* __restrict__ d_token_id,
    float* __restrict__ out_hidden, std::size_t hidden_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hidden_size)
    return;

  const std::uint32_t token_id = *d_token_id;
  const std::size_t offset =
      static_cast<std::size_t>(token_id) * hidden_size + idx;
  if (type == core::GgmlType::kBF16) {
    const auto* bf_ptr = static_cast<const hip_bfloat16*>(table);
    out_hidden[idx] = static_cast<float>(bf_ptr[offset]);
  } else if (type == core::GgmlType::kF32) {
    const auto* f_ptr = static_cast<const float*>(table);
    out_hidden[idx] = f_ptr[offset];
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[static_cast<std::size_t>(token_id) * (hidden_size / kQ8_0BlockSize) +
           idx / kQ8_0BlockSize];
    out_hidden[idx] = __half2float(wblk.d) *
                      static_cast<float>(wblk.qs[idx % kQ8_0BlockSize]);
  }
}

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

__global__ void ResidualAddKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, std::size_t dim) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    out[idx] = a[idx] + b[idx];
  }
}

// opt-c010-residual-rmsnorm: fuses the residual add (hidden += attn_out) with
// the subsequent RMSNorm into one launch. Each input element is read once and
// both the residual-updated sum and the normed output are written. The
// reduction order matches the unfused chain (ResidualAddKernel +
// RMSNormKernel) so the normed output is bit-identical.
__launch_bounds__(256, 4) __global__ void FusedResidualAddRMSNormKernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out_sum, const float* __restrict__ weight,
    float* __restrict__ out, std::size_t dim, float eps) {
  __shared__ float s_sum[256];
  const std::size_t tid = threadIdx.x;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = a[i] + b[i];
    out_sum[i] = val;
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
    out[i] = (out_sum[i] * inv_rms) * w;
  }
}

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

__global__ void SwiGLUKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out,
                             std::size_t intermediate_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < intermediate_size) {
    const float g = gate[idx];
    const float silu_g = g / (1.0F + expf(-g));
    out[idx] = silu_g * up[idx];
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_1Row(const hip_bfloat16* __restrict__ A,
                               const float* __restrict__ x,
                               float* __restrict__ y,
                               const float* __restrict__ residual,
                               std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* row = A + (m * K);
  const auto* row_vec = reinterpret_cast<const uint4*>(row);
  const std::size_t K_vec = K / 8;

  float dot = 0.0F;
  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const uint4 v = row_vec[vi];
    const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    dot += (static_cast<float>(vals[0]) * x0.x) +
           (static_cast<float>(vals[1]) * x0.y) +
           (static_cast<float>(vals[2]) * x0.z) +
           (static_cast<float>(vals[3]) * x0.w) +
           (static_cast<float>(vals[4]) * x1.x) +
           (static_cast<float>(vals[5]) * x1.y) +
           (static_cast<float>(vals[6]) * x1.z) +
           (static_cast<float>(vals[7]) * x1.w);
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    dot += static_cast<float>(row[k]) * x[k];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    const float r = (residual != nullptr) ? residual[m] : 0.0F;
    y[m] = dot + r;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_2Rows(const hip_bfloat16* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y, std::size_t M,
                                std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* row0 = A + (m0 * K);
  const auto* row1 = (m1 < M) ? A + (m1 * K) : nullptr;
  const auto* row0_vec = reinterpret_cast<const uint4*>(row0);
  const auto* row1_vec = reinterpret_cast<const uint4*>(row1);
  const std::size_t K_vec = K / 8;

  float dot0 = 0.0F;
  float dot1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 v0 = row0_vec[vi];
    const auto* vals0 = reinterpret_cast<const hip_bfloat16*>(&v0);
    dot0 += (static_cast<float>(vals0[0]) * x0.x) +
            (static_cast<float>(vals0[1]) * x0.y) +
            (static_cast<float>(vals0[2]) * x0.z) +
            (static_cast<float>(vals0[3]) * x0.w) +
            (static_cast<float>(vals0[4]) * x1.x) +
            (static_cast<float>(vals0[5]) * x1.y) +
            (static_cast<float>(vals0[6]) * x1.z) +
            (static_cast<float>(vals0[7]) * x1.w);

    if (row1 != nullptr) {
      const uint4 v1 = row1_vec[vi];
      const auto* vals1 = reinterpret_cast<const hip_bfloat16*>(&v1);
      dot1 += (static_cast<float>(vals1[0]) * x0.x) +
              (static_cast<float>(vals1[1]) * x0.y) +
              (static_cast<float>(vals1[2]) * x0.z) +
              (static_cast<float>(vals1[3]) * x0.w) +
              (static_cast<float>(vals1[4]) * x1.x) +
              (static_cast<float>(vals1[5]) * x1.y) +
              (static_cast<float>(vals1[6]) * x1.z) +
              (static_cast<float>(vals1[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot0 += static_cast<float>(row0[k]) * xv;
    if (row1 != nullptr) {
      dot1 += static_cast<float>(row1[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot0 += __shfl_xor(dot0, offset);
    dot1 += __shfl_xor(dot1, offset);
  }

  if (lane_id == 0) {
    y[m0] = dot0;
    if (m1 < M) {
      y[m1] = dot1;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32GEMVKernel_4Rows(const hip_bfloat16* __restrict__ A,
                                const float* __restrict__ x,
                                float* __restrict__ y, std::size_t M,
                                std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_quad = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_quad * 4;
  const std::size_t m1 = m0 + 1;
  const std::size_t m2 = m0 + 2;
  const std::size_t m3 = m0 + 3;
  if (m0 >= M) {
    return;
  }

  const auto* row0 = A + (m0 * K);
  const auto* row1 = (m1 < M) ? A + (m1 * K) : nullptr;
  const auto* row2 = (m2 < M) ? A + (m2 * K) : nullptr;
  const auto* row3 = (m3 < M) ? A + (m3 * K) : nullptr;
  const auto* row0_vec = reinterpret_cast<const uint4*>(row0);
  const auto* row1_vec = reinterpret_cast<const uint4*>(row1);
  const auto* row2_vec = reinterpret_cast<const uint4*>(row2);
  const auto* row3_vec = reinterpret_cast<const uint4*>(row3);
  const std::size_t K_vec = K / 8;

  float dot0 = 0.0F;
  float dot1 = 0.0F;
  float dot2 = 0.0F;
  float dot3 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 v0 = row0_vec[vi];
    const auto* vals0 = reinterpret_cast<const hip_bfloat16*>(&v0);
    dot0 += (static_cast<float>(vals0[0]) * x0.x) +
            (static_cast<float>(vals0[1]) * x0.y) +
            (static_cast<float>(vals0[2]) * x0.z) +
            (static_cast<float>(vals0[3]) * x0.w) +
            (static_cast<float>(vals0[4]) * x1.x) +
            (static_cast<float>(vals0[5]) * x1.y) +
            (static_cast<float>(vals0[6]) * x1.z) +
            (static_cast<float>(vals0[7]) * x1.w);

    if (row1 != nullptr) {
      const uint4 v1 = row1_vec[vi];
      const auto* vals1 = reinterpret_cast<const hip_bfloat16*>(&v1);
      dot1 += (static_cast<float>(vals1[0]) * x0.x) +
              (static_cast<float>(vals1[1]) * x0.y) +
              (static_cast<float>(vals1[2]) * x0.z) +
              (static_cast<float>(vals1[3]) * x0.w) +
              (static_cast<float>(vals1[4]) * x1.x) +
              (static_cast<float>(vals1[5]) * x1.y) +
              (static_cast<float>(vals1[6]) * x1.z) +
              (static_cast<float>(vals1[7]) * x1.w);
    }
    if (row2 != nullptr) {
      const uint4 v2 = row2_vec[vi];
      const auto* vals2 = reinterpret_cast<const hip_bfloat16*>(&v2);
      dot2 += (static_cast<float>(vals2[0]) * x0.x) +
              (static_cast<float>(vals2[1]) * x0.y) +
              (static_cast<float>(vals2[2]) * x0.z) +
              (static_cast<float>(vals2[3]) * x0.w) +
              (static_cast<float>(vals2[4]) * x1.x) +
              (static_cast<float>(vals2[5]) * x1.y) +
              (static_cast<float>(vals2[6]) * x1.z) +
              (static_cast<float>(vals2[7]) * x1.w);
    }
    if (row3 != nullptr) {
      const uint4 v3 = row3_vec[vi];
      const auto* vals3 = reinterpret_cast<const hip_bfloat16*>(&v3);
      dot3 += (static_cast<float>(vals3[0]) * x0.x) +
              (static_cast<float>(vals3[1]) * x0.y) +
              (static_cast<float>(vals3[2]) * x0.z) +
              (static_cast<float>(vals3[3]) * x0.w) +
              (static_cast<float>(vals3[4]) * x1.x) +
              (static_cast<float>(vals3[5]) * x1.y) +
              (static_cast<float>(vals3[6]) * x1.z) +
              (static_cast<float>(vals3[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot0 += static_cast<float>(row0[k]) * xv;
    if (row1 != nullptr) {
      dot1 += static_cast<float>(row1[k]) * xv;
    }
    if (row2 != nullptr) {
      dot2 += static_cast<float>(row2[k]) * xv;
    }
    if (row3 != nullptr) {
      dot3 += static_cast<float>(row3[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot0 += __shfl_xor(dot0, offset);
    dot1 += __shfl_xor(dot1, offset);
    dot2 += __shfl_xor(dot2, offset);
    dot3 += __shfl_xor(dot3, offset);
  }

  if (lane_id == 0) {
    y[m0] = dot0;
    if (m1 < M) {
      y[m1] = dot1;
    }
    if (m2 < M) {
      y[m2] = dot2;
    }
    if (m3 < M) {
      y[m3] = dot3;
    }
  }
}

__global__ void FastGEMVBlockKernel(const void* __restrict__ A,
                                    core::GgmlType a_type,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const float* __restrict__ residual,
                                    std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (a_type == core::GgmlType::kBF16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[16];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      const float r = (residual != nullptr) ? residual[m] : 0.0F;
      y[m] = block_sum + r;
    }
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_1Row(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  const auto* g_row = gate_w + (m * K);
  const auto* u_row = up_w + (m * K);
  const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
  const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
  const std::size_t K_vec = K / 8;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv = g_vec[vi];
    const uint4 uv = u_vec[vi];
    const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
    const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);

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
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_gate += static_cast<float>(g_row[k]) * xv;
    dot_up += static_cast<float>(u_row[k]) * xv;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  if (lane_id == 0) {
    const float sig_g = 1.0F / (1.0F + expf(-dot_gate));
    out[m] = (dot_gate * sig_g) * dot_up;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSwiGLUGEMVKernel_2Rows(
        const hip_bfloat16* __restrict__ gate_w,
        const hip_bfloat16* __restrict__ up_w, const float* __restrict__ x,
        float* __restrict__ out, std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t row_pair = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* g_row0 = gate_w + (m0 * K);
  const auto* u_row0 = up_w + (m0 * K);
  const auto* g_row1 = (m1 < M) ? gate_w + (m1 * K) : nullptr;
  const auto* u_row1 = (m1 < M) ? up_w + (m1 * K) : nullptr;

  const auto* g_vec0 = reinterpret_cast<const uint4*>(g_row0);
  const auto* u_vec0 = reinterpret_cast<const uint4*>(u_row0);
  const auto* g_vec1 = reinterpret_cast<const uint4*>(g_row1);
  const auto* u_vec1 = reinterpret_cast<const uint4*>(u_row1);
  const std::size_t K_vec = K / 8;

  float dot_g0 = 0.0F, dot_u0 = 0.0F;
  float dot_g1 = 0.0F, dot_u1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

    const uint4 gv0 = g_vec0[vi];
    const uint4 uv0 = u_vec0[vi];
    const auto* g_vals0 = reinterpret_cast<const hip_bfloat16*>(&gv0);
    const auto* u_vals0 = reinterpret_cast<const hip_bfloat16*>(&uv0);

    dot_g0 += (static_cast<float>(g_vals0[0]) * x0.x) +
              (static_cast<float>(g_vals0[1]) * x0.y) +
              (static_cast<float>(g_vals0[2]) * x0.z) +
              (static_cast<float>(g_vals0[3]) * x0.w) +
              (static_cast<float>(g_vals0[4]) * x1.x) +
              (static_cast<float>(g_vals0[5]) * x1.y) +
              (static_cast<float>(g_vals0[6]) * x1.z) +
              (static_cast<float>(g_vals0[7]) * x1.w);

    dot_u0 += (static_cast<float>(u_vals0[0]) * x0.x) +
              (static_cast<float>(u_vals0[1]) * x0.y) +
              (static_cast<float>(u_vals0[2]) * x0.z) +
              (static_cast<float>(u_vals0[3]) * x0.w) +
              (static_cast<float>(u_vals0[4]) * x1.x) +
              (static_cast<float>(u_vals0[5]) * x1.y) +
              (static_cast<float>(u_vals0[6]) * x1.z) +
              (static_cast<float>(u_vals0[7]) * x1.w);

    if (g_row1 != nullptr) {
      const uint4 gv1 = g_vec1[vi];
      const uint4 uv1 = u_vec1[vi];
      const auto* g_vals1 = reinterpret_cast<const hip_bfloat16*>(&gv1);
      const auto* u_vals1 = reinterpret_cast<const hip_bfloat16*>(&uv1);

      dot_g1 += (static_cast<float>(g_vals1[0]) * x0.x) +
                (static_cast<float>(g_vals1[1]) * x0.y) +
                (static_cast<float>(g_vals1[2]) * x0.z) +
                (static_cast<float>(g_vals1[3]) * x0.w) +
                (static_cast<float>(g_vals1[4]) * x1.x) +
                (static_cast<float>(g_vals1[5]) * x1.y) +
                (static_cast<float>(g_vals1[6]) * x1.z) +
                (static_cast<float>(g_vals1[7]) * x1.w);

      dot_u1 += (static_cast<float>(u_vals1[0]) * x0.x) +
                (static_cast<float>(u_vals1[1]) * x0.y) +
                (static_cast<float>(u_vals1[2]) * x0.z) +
                (static_cast<float>(u_vals1[3]) * x0.w) +
                (static_cast<float>(u_vals1[4]) * x1.x) +
                (static_cast<float>(u_vals1[5]) * x1.y) +
                (static_cast<float>(u_vals1[6]) * x1.z) +
                (static_cast<float>(u_vals1[7]) * x1.w);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float xv = x[k];
    dot_g0 += static_cast<float>(g_row0[k]) * xv;
    dot_u0 += static_cast<float>(u_row0[k]) * xv;
    if (g_row1 != nullptr) {
      dot_g1 += static_cast<float>(g_row1[k]) * xv;
      dot_u1 += static_cast<float>(u_row1[k]) * xv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_g0 += __shfl_xor(dot_g0, offset);
    dot_u0 += __shfl_xor(dot_u0, offset);
    dot_g1 += __shfl_xor(dot_g1, offset);
    dot_u1 += __shfl_xor(dot_u1, offset);
  }

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

// Q8_K block layout is defined in the shared qwen_gpu_quant_ops.hpp header.

// Serial (per-thread) row dot for a Q8_K weight row: iterates every 256-wide
// block, computes the activation scale from max|x|, quantizes x to int8 and
// accumulates an integer MAC with qs, then applies d_w * scale. Correctness
// path used by the fused SwiGLU fallback for Q8_K weight rows.
__device__ float Q8KRowDot(const Q8KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ8KBlockSize;
  float sumf = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const Q8KBlock& wblk = row_blocks[b];
    const float d_w = wblk.d;
    const float* xb = x + (b * kQ8KBlockSize);

    float max = 0.0F;
    for (std::size_t k = 0; k < kQ8KBlockSize; ++k) {
      max = fmaxf(max, fabsf(xb[k]));
    }
    const float scale = (max > 0.0F) ? (max / 127.0F) : 0.0F;

    int acc = 0;
    for (std::size_t k = 0; k < kQ8KBlockSize; ++k) {
      const float xv = (scale > 0.0F) ? (xb[k] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      acc += static_cast<int>(wblk.qs[k]) * xq;
    }

    sumf += d_w * scale * static_cast<float>(acc);
  }
  return sumf;
}

// Q5KValue / Q6KValue (and the GetQKScaleMin unpacking they use) are defined
// in the shared qwen_gpu_quant_ops.hpp header.
// opt-r4-q5k-q6k: serial (per-thread) row dot for a Q5_K weight row: dequant
// each element to fp and accumulate a fp MAC against x. Used by RowDot and the
// fused kernels' fallback. Requires K % 256 == 0.
__device__ float Q5KRowDot(const Q5KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ5KBlockSize;
  float sum = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const float* xb = x + (b * kQ5KBlockSize);
    for (std::size_t i = 0; i < kQ5KBlockSize; ++i) {
      sum += Q5KValue(row_blocks[b], i) * xb[i];
    }
  }
  return sum;
}

// opt-r4-q5k-q6k: serial (per-thread) row dot for a Q6_K weight row: dequant
// each element to fp and accumulate a fp MAC against x. Requires K % 256 == 0.
__device__ float Q6KRowDot(const Q6KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ6KBlockSize;
  float sum = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const float* xb = x + (b * kQ6KBlockSize);
    for (std::size_t i = 0; i < kQ6KBlockSize; ++i) {
      sum += Q6KValue(row_blocks[b], i) * xb[i];
    }
  }
  return sum;
}

// opt-r2-q8_0: Q8_0 row dot. Runs AT Q8 (integer MAC) exactly like Q8KRowDot
// but per-32-element block with a half weight scale read via __half2float.
// Requires K % 32 == 0.
__device__ float Q8_0RowDot(const Q8_0Block* __restrict__ row_blocks,
                            const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ8_0BlockSize;
  float sumf = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const Q8_0Block& wblk = row_blocks[b];
    const float d_w = __half2float(wblk.d);
    const float* xb = x + (b * kQ8_0BlockSize);

    float max = 0.0F;
    for (std::size_t k = 0; k < kQ8_0BlockSize; ++k) {
      max = fmaxf(max, fabsf(xb[k]));
    }
    const float scale = (max > 0.0F) ? (max / 127.0F) : 0.0F;

    int acc = 0;
    for (std::size_t k = 0; k < kQ8_0BlockSize; ++k) {
      const float xv = (scale > 0.0F) ? (xb[k] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      acc += static_cast<int>(wblk.qs[k]) * xq;
    }

    sumf += d_w * scale * static_cast<float>(acc);
  }
  return sumf;
}

// QuantBlockBytes / QuantBlockQK are defined in the shared
// qwen_gpu_quant_ops.hpp header.
// Central dispatcher for a single quantized weight row's dot product. Routes
// on the weight's GgmlType and OWNS the type-aware row stride: given the
// weight base pointer and the output row index, it computes
//   base + row_idx * (K / QuantBlockQK(type) * QuantBlockBytes(type))
// then calls the type-specific row dot. Q8_0/Q8_K run at Q8 (integer MAC);
// Q5_K/Q6_K dequant to fp and accumulate a fp MAC (both mirror the CPU
// dequant oracles). bf16/fp32 are handled by the vectorized kernels and never
// reach here, so unsupported types return 0 (acceptable until those types are
// wired and the model is run).
__device__ inline float RowDot(core::GgmlType type, const void* __restrict__ base,
                               std::size_t row_idx, const float* __restrict__ x,
                               std::size_t K) {
  const std::size_t qk = QuantBlockQK(type);
  const std::size_t block_bytes = QuantBlockBytes(type);
  const char* row = static_cast<const char*>(base) + (row_idx * (K / qk * block_bytes));
  switch (type) {
    case core::GgmlType::kQ8_0:
      return Q8_0RowDot(reinterpret_cast<const Q8_0Block*>(row), x, K);
    case core::GgmlType::kQ8_K:
      return Q8KRowDot(reinterpret_cast<const Q8KBlock*>(row), x, K);
    case core::GgmlType::kQ5_K:
      return Q5KRowDot(reinterpret_cast<const Q5KBlock*>(row), x, K);
    case core::GgmlType::kQ6_K:
      return Q6KRowDot(reinterpret_cast<const Q6KBlock*>(row), x, K);
    default:
      return 0.0F;  // remaining K-typed rows unwired
  }
}

// QuantWarpBlockDot (warp-parallel quant row-dot) is defined as a template on
// the input element type in the shared qwen_gpu_quant_ops.hpp header.

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedSSMInputProjectionsKernel_1Row(
        const void* __restrict__ qkv_w, core::GgmlType qkv_type,
        const void* __restrict__ gate_w, core::GgmlType gate_type,
        const void* __restrict__ alpha_w, core::GgmlType alpha_type,
        const void* __restrict__ beta_w, core::GgmlType beta_type,
        const float* __restrict__ x, float* __restrict__ qkv_out,
        float* __restrict__ gate_out, float* __restrict__ alpha_out,
        float* __restrict__ beta_out, std::size_t hidden_size,
        std::size_t qkv_size, std::size_t inner_size,
        std::size_t time_step_rank) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    weight_type = qkv_type;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    weight_type = gate_type;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    weight_type = alpha_type;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    weight_type = beta_type;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + lane_id; k < hidden_size; k += 32) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type != core::GgmlType::kF32) {
    // Quantized weight: route via QuantWarpBlockDot (block_q8_K now; Q8_0/Q5_K/
    // Q6_K wired in later tasks). All 32 lanes cooperate on the row's blocks;
    // lane 0 returns the warp's partial row-dot and the warp reduction below
    // (num_warps == 1) broadcasts the single correct value.
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id, 0, 1);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < hidden_size; k += 32) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Wave32FusedQKVProjectionsKernel_1Row(
        const void* __restrict__ q_w, core::GgmlType q_type,
        const void* __restrict__ k_w, core::GgmlType k_type,
        const void* __restrict__ v_w, core::GgmlType v_type,
        const float* __restrict__ x, float* __restrict__ q_out,
        float* __restrict__ k_out, float* __restrict__ v_out, std::size_t q_dim,
        std::size_t kv_dim, std::size_t hidden_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    weight_type = q_type;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    weight_type = k_type;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    weight_type = v_type;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
  }

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + lane_id; k < hidden_size; k += 32) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type != core::GgmlType::kF32) {
    // Quantized weight: route via QuantWarpBlockDot (block_q8_K now; Q8_0/Q5_K/
    // Q6_K wired in later tasks). All 32 lanes cooperate on the row's blocks;
    // lane 0 returns the warp's partial row-dot and the warp reduction below
    // (num_warps == 1) broadcasts the single correct value.
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id, 0, 1);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < hidden_size; k += 32) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

__global__ void FastFusedSwiGLUGEMVBlockKernel(
    const void* __restrict__ gate_w, core::GgmlType gate_type,
    const void* __restrict__ up_w, core::GgmlType up_type,
    const float* __restrict__ x, float* __restrict__ out, std::size_t M,
    std::size_t K) {
  const std::size_t m = blockIdx.x;
  if (m >= M)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16) {
    const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
    const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

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
  } else {
    // Per-weight fallback for the non-(both-bf16) route: each of gate/up is
    // computed independently — Q8 (integer dot) or bf16 — based on its own
    // flags. A weight that is neither (legacy fp32) keeps its initialized
    // 0.0 dot so the block reduction collapses to 0, matching the pre-fallback
    // (HEAD) semantics. For Q8 the full row dot is computed in thread 0 only;
    // every other lane contributes 0 so the block reduction below collapses to
    // exactly the row dot (not multiplied by block/thread count).
    if (gate_type == core::GgmlType::kBF16) {
      const auto* g_row = static_cast<const hip_bfloat16*>(gate_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 gv = g_vec[vi];
        const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                    (static_cast<float>(g_vals[1]) * x0.y) +
                    (static_cast<float>(g_vals[2]) * x0.z) +
                    (static_cast<float>(g_vals[3]) * x0.w) +
                    (static_cast<float>(g_vals[4]) * x1.x) +
                    (static_cast<float>(g_vals[5]) * x1.y) +
                    (static_cast<float>(g_vals[6]) * x1.z) +
                    (static_cast<float>(g_vals[7]) * x1.w);
      }
    } else if (gate_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_gate = QuantWarpBlockDot(gate_type, gate_w, m, x, K, lane_id, warp_id,
                                   block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_gate == 0.0, matching HEAD semantics.

    if (up_type == core::GgmlType::kBF16) {
      const auto* u_row = static_cast<const hip_bfloat16*>(up_w) + (m * K);
      const std::size_t K_vec = K / 8;
      const auto* u_vec = reinterpret_cast<const uint4*>(u_row);
      for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
        const uint4 uv = u_vec[vi];
        const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
        const std::size_t k = vi * 8;
        const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
        const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
        dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                  (static_cast<float>(u_vals[1]) * x0.y) +
                  (static_cast<float>(u_vals[2]) * x0.z) +
                  (static_cast<float>(u_vals[3]) * x0.w) +
                  (static_cast<float>(u_vals[4]) * x1.x) +
                  (static_cast<float>(u_vals[5]) * x1.y) +
                  (static_cast<float>(u_vals[6]) * x1.z) +
                  (static_cast<float>(u_vals[7]) * x1.w);
      }
    } else if (up_type != core::GgmlType::kF32) {
      // Quantized weight (Q8_K now; Q8_0/Q5_K/Q6_K wired later via
      // QuantWarpBlockDot). All lanes cooperate; lane 0 returns this warp's
      // partial row-dot and the block reduction below combines the warps.
      dot_up = QuantWarpBlockDot(up_type, up_w, m, x, K, lane_id, warp_id,
                                 block_threads >> 5u);
    }
    // else (kF32 / legacy fp32) keeps dot_up == 0.0, matching HEAD semantics.
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[16];
  __shared__ float s_up_sums[16];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = static_cast<int>(num_warps / 2); offset > 0;
         offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      out[m] = (block_gate * sig_g) * block_up;
    }
  }
}

__global__ void FusedSSMInputProjectionsKernel(
    const void* __restrict__ qkv_w, core::GgmlType qkv_type,
    const void* __restrict__ gate_w, core::GgmlType gate_type,
    const void* __restrict__ alpha_w, core::GgmlType alpha_type,
    const void* __restrict__ beta_w, core::GgmlType beta_type,
    const float* __restrict__ x, float* __restrict__ qkv_out,
    float* __restrict__ gate_out, float* __restrict__ alpha_out,
    float* __restrict__ beta_out, std::size_t hidden_size,
    std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank) {
  const std::size_t m = blockIdx.x;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    weight_type = qkv_type;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    weight_type = gate_type;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    weight_type = alpha_type;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    weight_type = beta_type;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type != core::GgmlType::kF32) {
    // Quantized weight: route via QuantWarpBlockDot (block_q8_K now; Q8_0/Q5_K/
    // Q6_K wired in later tasks). All lanes cooperate; lane 0 returns this
    // warp's partial row-dot and the warp + cross-warp block reduction below
    // combines the warps (num_warps = block_threads >> 5u).
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id,
                            warp_id, block_threads >> 5u);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

__global__ void FusedQKVProjectionsKernel(
    const void* __restrict__ q_w, core::GgmlType q_type,
    const void* __restrict__ k_w, core::GgmlType k_type,
    const void* __restrict__ v_w, core::GgmlType v_type,
    const float* __restrict__ x, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows)
    return;

  const void* A = nullptr;
  core::GgmlType weight_type = core::GgmlType::kF32;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    weight_type = q_type;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    weight_type = k_type;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    weight_type = v_type;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (weight_type == core::GgmlType::kBF16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x[k];
    }
  } else if (weight_type != core::GgmlType::kF32) {
    // Quantized weight: route via QuantWarpBlockDot (block_q8_K now; Q8_0/Q5_K/
    // Q6_K wired in later tasks). All lanes cooperate; lane 0 returns this
    // warp's partial row-dot and the warp + cross-warp block reduction below
    // combines the warps (num_warps = block_threads >> 5u).
    dot = QuantWarpBlockDot(weight_type, A, row_idx, x, hidden_size, lane_id,
                            warp_id, block_threads >> 5u);
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
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

// opt-c010-qk-rope-kv: fuses per-head Q/K RMSNorm, RoPE, and the KV-cache write
// for a single decode token into one launch. Blocks [0, num_heads) normalize +
// rotate the Q heads in place; blocks [num_heads, num_heads + num_kv_heads)
// normalize + rotate the K heads, write K/V into the FP32+FP16 KV caches, and
// store the normed+roped K back for downstream consumers.
__launch_bounds__(256, 4) __global__ void FusedQKNormRoPEKvWriteKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ k_cache,
    float* __restrict__ v_cache, __half* __restrict__ k_cache_f16,
    __half* __restrict__ v_cache_f16, std::uint32_t layer_idx,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps) {
  const std::uint32_t block = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const std::uint32_t half_rot = rotary_dim / 2;
  const std::uint32_t pos = *d_pos;
  __shared__ float s_norm[256];

  const std::uint32_t num_q_blocks = num_heads;
  if (block < num_q_blocks) {
    const std::size_t head_base = static_cast<std::size_t>(block) * head_dim;
    const float* x_head = q + head_base;

    float sum_sq = 0.0F;
    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float val = x_head[i];
      sum_sq += val * val;
    }
    s_norm[tid] = sum_sq;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_norm[tid] += s_norm[tid + stride];
      }
      __syncthreads();
    }
    const float inv_rms =
        rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
    __syncthreads();

    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float w = (q_weight != nullptr) ? q_weight[i] : 1.0F;
      s_norm[i] = (x_head[i] * inv_rms) * w;
    }
    __syncthreads();

    for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
      const float q0 = s_norm[p];
      const float q1 = s_norm[p + half_rot];
      const float exponent =
          static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
      const float inv_freq = 1.0F / powf(rope_theta, exponent);
      const float angle = static_cast<float>(pos) * inv_freq;
      const float cos_val = cosf(angle);
      const float sin_val = sinf(angle);
      q_out[head_base + p] = (q0 * cos_val) - (q1 * sin_val);
      q_out[head_base + p + half_rot] = (q0 * sin_val) + (q1 * cos_val);
    }
    for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
      q_out[head_base + i] = s_norm[i];
    }
    return;
  }

  const std::uint32_t kv_h = block - num_q_blocks;
  if (kv_h >= num_kv_heads) {
    return;
  }

  const std::size_t head_base = static_cast<std::size_t>(kv_h) * head_dim;
  const float* x_head = k + head_base;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }
  s_norm[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_norm[tid] += s_norm[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
  __syncthreads();

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (k_weight != nullptr) ? k_weight[i] : 1.0F;
    s_norm[i] = (x_head[i] * inv_rms) * w;
  }
  __syncthreads();

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t f16_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      head_base;

  for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
    const float k0 = s_norm[p];
    const float k1 = s_norm[p + half_rot];
    const float exponent =
        static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float k0_rot = (k0 * cos_val) - (k1 * sin_val);
    const float k1_rot = (k0 * sin_val) + (k1 * cos_val);
    k_out[head_base + p] = k0_rot;
    k_out[head_base + p + half_rot] = k1_rot;
    k_cache[kv_offset + p] = k0_rot;
    k_cache[kv_offset + p + half_rot] = k1_rot;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + p] = __float2half_rn(k0_rot);
      k_cache_f16[f16_offset + p + half_rot] = __float2half_rn(k1_rot);
    }
  }
  for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
    const float key = s_norm[i];
    k_out[head_base + i] = key;
    k_cache[kv_offset + i] = key;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + i] = __float2half_rn(key);
    }
  }

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = v[head_base + i];
    v_cache[kv_offset + i] = val;
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset + i] = __float2half_rn(val);
    }
  }
}

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

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + Q/K/V projections.
// 8 rows per block (256 threads); the norm is computed once per block and the
// projection dots consume (x * inv_rms * w) per element, matching the unfused
// chain (RMSNormKernel then Wave32FusedQKVProjectionsKernel_1Row) bit-for-bit.
__launch_bounds__(256, 4) __global__ void FusedRMSNormQKVProjectionsKernel(
    const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
    const void* __restrict__ q_w, bool q_is_bf16, const void* __restrict__ k_w,
    bool k_is_bf16, const void* __restrict__ v_w, bool v_is_bf16,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  __shared__ float s_sum[256];
  const float inv_rms =
      FusedRMSNormInvScale(x, norm_w, hidden_size, eps, s_sum);

  const std::size_t m = (blockIdx.x * 8U) + warp_id;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows) {
    return;
  }

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    is_bf16 = q_is_bf16;
    out_target = q_out;
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    is_bf16 = k_is_bf16;
    out_target = k_out;
    row_idx = m - q_dim;
  } else {
    A = v_w;
    is_bf16 = v_is_bf16;
    out_target = v_out;
    row_idx = m - q_dim - kv_dim;
  }

  const float* nw = (norm_w != nullptr) ? norm_w : nullptr;
  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
      const float n0 = (x0.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
      const float n1 = (x0.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
      const float n2 = (x0.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
      const float n3 = (x0.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
      const float n4 = (x1.x * inv_rms) * ((nw != nullptr) ? nw[k + 4] : 1.0F);
      const float n5 = (x1.y * inv_rms) * ((nw != nullptr) ? nw[k + 5] : 1.0F);
      const float n6 = (x1.z * inv_rms) * ((nw != nullptr) ? nw[k + 6] : 1.0F);
      const float n7 = (x1.w * inv_rms) * ((nw != nullptr) ? nw[k + 7] : 1.0F);

      dot += (static_cast<float>(vals[0]) * n0) +
             (static_cast<float>(vals[1]) * n1) +
             (static_cast<float>(vals[2]) * n2) +
             (static_cast<float>(vals[3]) * n3) +
             (static_cast<float>(vals[4]) * n4) +
             (static_cast<float>(vals[5]) * n5) +
             (static_cast<float>(vals[6]) * n6) +
             (static_cast<float>(vals[7]) * n7);
    }
    for (std::size_t k = (K_vec * 8) + lane_id; k < hidden_size; k += 32) {
      const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
      dot += static_cast<float>(row[k]) * nv;
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      const float n0 = (xv.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
      const float n1 = (xv.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
      const float n2 = (xv.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
      const float n3 = (xv.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
      dot += (v.x * n0) + (v.y * n1) + (v.z * n2) + (v.w * n3);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < hidden_size; k += 32) {
      const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
      dot += row[k] * nv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + SSM input projections
// (QKV, gate, alpha, beta). Mirrors Wave32FusedSSMInputProjectionsKernel_1Row
// row mapping with 8 rows per block.
__launch_bounds__(256, 4) __global__ void FusedRMSNormSSMInputProjectionsKernel(
    const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
    const void* __restrict__ qkv_w, bool qkv_is_bf16,
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ alpha_w, bool alpha_is_bf16,
    const void* __restrict__ beta_w, bool beta_is_bf16,
    float* __restrict__ qkv_out, float* __restrict__ gate_out,
    float* __restrict__ alpha_out, float* __restrict__ beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  __shared__ float s_sum[256];
  const float inv_rms =
      FusedRMSNormInvScale(x, norm_w, hidden_size, eps, s_sum);

  const std::size_t m = (blockIdx.x * 8U) + warp_id;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows) {
    return;
  }

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    is_bf16 = qkv_is_bf16;
    out_target = qkv_out;
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    is_bf16 = gate_is_bf16;
    out_target = gate_out;
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    is_bf16 = alpha_is_bf16;
    out_target = alpha_out;
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    is_bf16 = beta_is_bf16;
    out_target = beta_out;
    row_idx = m - alpha_end;
  }

  const float* nw = (norm_w != nullptr) ? norm_w : nullptr;
  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
      const float n0 = (x0.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
      const float n1 = (x0.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
      const float n2 = (x0.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
      const float n3 = (x0.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
      const float n4 = (x1.x * inv_rms) * ((nw != nullptr) ? nw[k + 4] : 1.0F);
      const float n5 = (x1.y * inv_rms) * ((nw != nullptr) ? nw[k + 5] : 1.0F);
      const float n6 = (x1.z * inv_rms) * ((nw != nullptr) ? nw[k + 6] : 1.0F);
      const float n7 = (x1.w * inv_rms) * ((nw != nullptr) ? nw[k + 7] : 1.0F);

      dot += (static_cast<float>(vals[0]) * n0) +
             (static_cast<float>(vals[1]) * n1) +
             (static_cast<float>(vals[2]) * n2) +
             (static_cast<float>(vals[3]) * n3) +
             (static_cast<float>(vals[4]) * n4) +
             (static_cast<float>(vals[5]) * n5) +
             (static_cast<float>(vals[6]) * n6) +
             (static_cast<float>(vals[7]) * n7);
    }
    for (std::size_t k = (K_vec * 8) + lane_id; k < hidden_size; k += 32) {
      const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
      dot += static_cast<float>(row[k]) * nv;
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x[k]);
      const float n0 = (xv.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
      const float n1 = (xv.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
      const float n2 = (xv.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
      const float n3 = (xv.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
      dot += (v.x * n0) + (v.y * n1) + (v.z * n2) + (v.w * n3);
    }
    for (std::size_t k = (K_vec * 4) + lane_id; k < hidden_size; k += 32) {
      const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
      dot += row[k] * nv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  if (lane_id == 0) {
    out_target[row_idx] = dot;
  }
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + FFN SwiGLU gate/up
// projections. Mirrors Wave32FusedSwiGLUGEMVKernel_2Rows with 8 row pairs per
// block.
__launch_bounds__(256, 4) __global__ void FusedRMSNormSwiGLUGEMVKernel(
    const float* __restrict__ x, const float* __restrict__ norm_w, float eps,
    const hip_bfloat16* __restrict__ gate_w,
    const hip_bfloat16* __restrict__ up_w, float* __restrict__ out,
    std::size_t M, std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  __shared__ float s_sum[256];
  const float inv_rms = FusedRMSNormInvScale(x, norm_w, K, eps, s_sum);

  const std::size_t row_pair = (blockIdx.x * 8U) + warp_id;
  const std::size_t m0 = row_pair * 2;
  const std::size_t m1 = m0 + 1;
  if (m0 >= M) {
    return;
  }

  const auto* g_row0 = gate_w + (m0 * K);
  const auto* u_row0 = up_w + (m0 * K);
  const auto* g_row1 = (m1 < M) ? gate_w + (m1 * K) : nullptr;
  const auto* u_row1 = (m1 < M) ? up_w + (m1 * K) : nullptr;

  const auto* g_vec0 = reinterpret_cast<const uint4*>(g_row0);
  const auto* u_vec0 = reinterpret_cast<const uint4*>(u_row0);
  const auto* g_vec1 = reinterpret_cast<const uint4*>(g_row1);
  const auto* u_vec1 = reinterpret_cast<const uint4*>(u_row1);
  const std::size_t K_vec = K / 8;

  const float* nw = (norm_w != nullptr) ? norm_w : nullptr;
  float dot_g0 = 0.0F, dot_u0 = 0.0F;
  float dot_g1 = 0.0F, dot_u1 = 0.0F;

  for (std::size_t vi = lane_id; vi < K_vec; vi += 32) {
    const std::size_t k = vi * 8;
    const float4 x0 = *reinterpret_cast<const float4*>(&x[k + 0]);
    const float4 x1 = *reinterpret_cast<const float4*>(&x[k + 4]);
    const float n0 = (x0.x * inv_rms) * ((nw != nullptr) ? nw[k + 0] : 1.0F);
    const float n1 = (x0.y * inv_rms) * ((nw != nullptr) ? nw[k + 1] : 1.0F);
    const float n2 = (x0.z * inv_rms) * ((nw != nullptr) ? nw[k + 2] : 1.0F);
    const float n3 = (x0.w * inv_rms) * ((nw != nullptr) ? nw[k + 3] : 1.0F);
    const float n4 = (x1.x * inv_rms) * ((nw != nullptr) ? nw[k + 4] : 1.0F);
    const float n5 = (x1.y * inv_rms) * ((nw != nullptr) ? nw[k + 5] : 1.0F);
    const float n6 = (x1.z * inv_rms) * ((nw != nullptr) ? nw[k + 6] : 1.0F);
    const float n7 = (x1.w * inv_rms) * ((nw != nullptr) ? nw[k + 7] : 1.0F);

    const uint4 gv0 = g_vec0[vi];
    const uint4 uv0 = u_vec0[vi];
    const auto* g_vals0 = reinterpret_cast<const hip_bfloat16*>(&gv0);
    const auto* u_vals0 = reinterpret_cast<const hip_bfloat16*>(&uv0);

    dot_g0 += (static_cast<float>(g_vals0[0]) * n0) +
              (static_cast<float>(g_vals0[1]) * n1) +
              (static_cast<float>(g_vals0[2]) * n2) +
              (static_cast<float>(g_vals0[3]) * n3) +
              (static_cast<float>(g_vals0[4]) * n4) +
              (static_cast<float>(g_vals0[5]) * n5) +
              (static_cast<float>(g_vals0[6]) * n6) +
              (static_cast<float>(g_vals0[7]) * n7);

    dot_u0 += (static_cast<float>(u_vals0[0]) * n0) +
              (static_cast<float>(u_vals0[1]) * n1) +
              (static_cast<float>(u_vals0[2]) * n2) +
              (static_cast<float>(u_vals0[3]) * n3) +
              (static_cast<float>(u_vals0[4]) * n4) +
              (static_cast<float>(u_vals0[5]) * n5) +
              (static_cast<float>(u_vals0[6]) * n6) +
              (static_cast<float>(u_vals0[7]) * n7);

    if (g_row1 != nullptr) {
      const uint4 gv1 = g_vec1[vi];
      const uint4 uv1 = u_vec1[vi];
      const auto* g_vals1 = reinterpret_cast<const hip_bfloat16*>(&gv1);
      const auto* u_vals1 = reinterpret_cast<const hip_bfloat16*>(&uv1);

      dot_g1 += (static_cast<float>(g_vals1[0]) * n0) +
                (static_cast<float>(g_vals1[1]) * n1) +
                (static_cast<float>(g_vals1[2]) * n2) +
                (static_cast<float>(g_vals1[3]) * n3) +
                (static_cast<float>(g_vals1[4]) * n4) +
                (static_cast<float>(g_vals1[5]) * n5) +
                (static_cast<float>(g_vals1[6]) * n6) +
                (static_cast<float>(g_vals1[7]) * n7);

      dot_u1 += (static_cast<float>(u_vals1[0]) * n0) +
                (static_cast<float>(u_vals1[1]) * n1) +
                (static_cast<float>(u_vals1[2]) * n2) +
                (static_cast<float>(u_vals1[3]) * n3) +
                (static_cast<float>(u_vals1[4]) * n4) +
                (static_cast<float>(u_vals1[5]) * n5) +
                (static_cast<float>(u_vals1[6]) * n6) +
                (static_cast<float>(u_vals1[7]) * n7);
    }
  }
  for (std::size_t k = (K_vec * 8) + lane_id; k < K; k += 32) {
    const float nv = (x[k] * inv_rms) * ((nw != nullptr) ? nw[k] : 1.0F);
    dot_g0 += static_cast<float>(g_row0[k]) * nv;
    dot_u0 += static_cast<float>(u_row0[k]) * nv;
    if (g_row1 != nullptr) {
      dot_g1 += static_cast<float>(g_row1[k]) * nv;
      dot_u1 += static_cast<float>(u_row1[k]) * nv;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_g0 += __shfl_xor(dot_g0, offset);
    dot_u0 += __shfl_xor(dot_u0, offset);
    dot_g1 += __shfl_xor(dot_g1, offset);
    dot_u1 += __shfl_xor(dot_u1, offset);
  }

  if (lane_id == 0) {
    const float sig_g0 = 1.0F / (1.0F + expf(-dot_g0));
    out[m0] = (dot_g0 * sig_g0) * dot_u0;
    if (m1 < M) {
      const float sig_g1 = 1.0F / (1.0F + expf(-dot_g1));
      out[m1] = (dot_g1 * sig_g1) * dot_u1;
    }
  }
}

// opt-c014-layer-prefetch: touch up to 16 bytes per 4KiB page of a weight
// region so the page mappings are resident before the real projection kernels
// read them. Reads only; the guarded store can never fire (its value depends
// on the loaded data) and exists to defeat dead-code elimination. Reads are
// clamped to `bytes` so the last partial page never runs past the region.
__launch_bounds__(256, 4) __global__
    void LayerWeightPrefetchKernel(const void* __restrict__ data,
                                   std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  constexpr std::size_t kPageBytes = 4096U;
  constexpr std::size_t kTouchBytes = 16U;
  const std::size_t num_pages = (bytes + kPageBytes - 1U) / kPageBytes;
  float acc = 0.0F;
  for (std::size_t page = threadIdx.x; page < num_pages; page += blockDim.x) {
    const std::size_t base = page * kPageBytes;
    const std::size_t n = std::min<std::size_t>(kTouchBytes, bytes - base);
    for (std::size_t i = 0; i < n; ++i) {
      acc += static_cast<float>(p[base + i]);
    }
  }
  if (acc == 1.234567e30F) {
    const_cast<unsigned char*>(p)[0] = 0U;
  }
}

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void Q8KBlockGEMVKernel(const void* __restrict__ A, core::GgmlType type,
                            const float* __restrict__ x,
                            float* __restrict__ y, std::size_t M,
                            std::size_t K) {
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t m = (blockIdx.x * WavesPerBlock) + warp_id;
  if (m >= M) {
    return;
  }

  float sumf = 0.0F;
  if (type == core::GgmlType::kQ8_K) {
    const std::size_t num_blocks = K / kQ8KBlockSize;
    const auto* row_blocks =
        static_cast<const Q8KBlock*>(A) + (m * num_blocks);

    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlock& wblk = row_blocks[b];
      const float d_w = wblk.d;
      const float* xb = x + (b * kQ8KBlockSize);

      // Pass 1: per-lane max|x| over the 8 x values this lane owns, then a warp
      // reduction to get the block max and derive the activation scale.
      float local_max = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        local_max = fmaxf(local_max, fabsf(xb[lane_id + (32 * k)]));
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;

      // Pass 2: quantize x to int8 and integer MAC with the weight qs.
      int acc = 0;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32 * k);
        const float xv = (scale > 0.0F) ? (xb[idx] / scale) : 0.0F;
        int xq = static_cast<int>(roundf(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[idx]) * xq;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }

      sumf += d_w * scale * static_cast<float>(acc);
    }
  } else if (type == core::GgmlType::kQ8_0) {
    // Q8_0 path (QK = 32 = warp width): one element per lane. Scale derivation,
    // quantize/clamp, int-MAC and end-scale mirror the Q8_K path AND
    // Q8_0RowDot exactly (verified by the Q8_0 GEMV equivalence test).
    const std::size_t num_blocks = K / kQ8_0BlockSize;
    const auto* row_blocks =
        static_cast<const Q8_0Block*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8_0Block& wblk = row_blocks[b];
      const float d_w = __half2float(wblk.d);
      const float* xb = x + (b * kQ8_0BlockSize);

      float local_max = fabsf(xb[lane_id]);
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;

      const float xv = (scale > 0.0F) ? (xb[lane_id] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      int acc = static_cast<int>(wblk.qs[lane_id]) * xq;
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }

      sumf += d_w * scale * static_cast<float>(acc);
    }
  } else if (type == core::GgmlType::kQ5_K) {
    // Q5_K path (QK = 256): dequant each 256-wide element to fp via Q5KValue
    // and accumulate a fp MAC against x. One warp per row; each lane owns 8
    // elements (lane_id + 32*k) then a warp reduction. Mirrors CPU
    // DequantizeQ5_K + dot exactly modulo fp accumulation order.
    const std::size_t num_blocks = K / kQ5KBlockSize;
    const auto* row_blocks =
        static_cast<const Q5KBlock*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q5KBlock& wblk = row_blocks[b];
      const float* xb = x + (b * kQ5KBlockSize);
      float acc = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32u * k);
        acc += Q5KValue(wblk, idx) * xb[idx];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }
      sumf += acc;
    }
  } else if (type == core::GgmlType::kQ6_K) {
    // Q6_K path (QK = 256): dequant each 256-wide element to fp via Q6KValue
    // and accumulate a fp MAC against x. Mirrors CPU DequantizeQ6_K + dot.
    const std::size_t num_blocks = K / kQ6KBlockSize;
    const auto* row_blocks =
        static_cast<const Q6KBlock*>(A) + (m * num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q6KBlock& wblk = row_blocks[b];
      const float* xb = x + (b * kQ6KBlockSize);
      float acc = 0.0F;
#pragma unroll
      for (std::size_t k = 0; k < 8; ++k) {
        const std::size_t idx = lane_id + (32u * k);
        acc += Q6KValue(wblk, idx) * xb[idx];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
      }
      sumf += acc;
    }
  }

  if (lane_id == 0) {
    y[m] = sumf;
  }
}
}  // namespace

// opt-c014-layer-prefetch: touch one 16B chunk per 4KiB page of a weight
// region on the given stream (async). Reads only; the guarded store inside the
// kernel can never fire.
void LaunchLayerWeightPrefetch(const void* data, std::size_t bytes,
                               hipStream_t stream) {
  if (data == nullptr || bytes == 0) {
    return;
  }
  const std::size_t num_pages = (bytes + 4095U) / 4096U;
  const std::size_t grid = (num_pages + 255U) / 256U;
  LayerWeightPrefetchKernel<<<grid, 256, 0, stream>>>(data, bytes);
}

// opt-c1xx-q8k-gemv: launches the Q8_K on-the-fly integer-dot GEMV. One warp
// per output row (4 warps per block). Requires K % 256 == 0 (Q8_K block size).
void LaunchQ8KBlockGEMV(const void* A, core::GgmlType type, const float* x,
                        float* y, std::size_t M, std::size_t K,
                        hipStream_t stream) {
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
  const std::size_t grid = (M + kWavesPerBlock - 1U) / kWavesPerBlock;
  Q8KBlockGEMVKernel<kWavesPerBlock>
      <<<grid, kBlockThreads, 0, stream>>>(A, type, x, y, M, K);
}

__global__ void SSMConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out, std::size_t qkv_dim,
                              std::uint32_t layer_idx,
                              SsmReplayCapture replay_capture) {
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim)
    return;

  if (replay_capture.qkv != nullptr && replay_capture.position != nullptr &&
      replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
    const std::size_t slot =
        static_cast<std::size_t>(replay_capture.position[0]) %
        kSsmReplayCapacity;
    const std::size_t offset =
        ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
         qkv_dim) +
        c;
    replay_capture.qkv[offset] = qkv_in[c];
  }

  const std::size_t c_off = c * 4;
  conv_state[c_off + 0] = conv_state[c_off + 1];
  conv_state[c_off + 1] = conv_state[c_off + 2];
  conv_state[c_off + 2] = conv_state[c_off + 3];
  conv_state[c_off + 3] = qkv_in[c];

  float dot = 0.0F;
  if (conv_weights != nullptr) {
    for (std::size_t k = 0; k < 4; ++k) {
      dot += conv_state[c_off + k] * conv_weights[c * 4 + k];
    }
  } else {
    dot = qkv_in[c];
  }

  const float sig = 1.0F / (1.0F + expf(-dot));
  conv_out[c] = dot * sig;
}

__global__ void DeltaNetRecurrenceKernel(
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    const float* __restrict__ ssm_norm, const float* __restrict__ gate,
    float* __restrict__ out_buf, std::uint32_t layer_idx,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, SsmReplayCapture replay_capture) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;

  const float* q_ptr = conv_out;
  const float* k_ptr = conv_out + (num_key_heads * key_dim);
  const float* v_ptr = conv_out + (num_key_heads * key_dim * 2);

  const std::uint32_t kh_idx = h % num_key_heads;
  const float* q_h = q_ptr + (kh_idx * key_dim);
  const float* k_h = k_ptr + (kh_idx * key_dim);
  const float* v_h = v_ptr + (h * val_dim);
  float* o_h = out_buf + (h * val_dim);

  float* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta =
  // sigmoid(beta)
  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
  const float alpha_in = (alpha_buf != nullptr) ? alpha_buf[h] : 0.0F;
  const float beta_in = (beta_buf != nullptr) ? beta_buf[h] : 0.0F;

  if (tid == 0 && replay_capture.alpha != nullptr &&
      replay_capture.beta != nullptr && replay_capture.position != nullptr &&
      replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
    const std::size_t slot =
        static_cast<std::size_t>(replay_capture.position[0]) %
        kSsmReplayCapacity;
    const std::size_t offset =
        ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
         num_heads) +
        h;
    replay_capture.alpha[offset] = alpha_in;
    replay_capture.beta[offset] = beta_in;
  }

  const float alpha_biased = alpha_in + dt;
  const float alpha_softplus =
      (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
  const float alpha_h = expf(alpha_softplus * a_val);
  const float beta_h = 1.0F / (1.0F + expf(-beta_in));

  // 2. Normalize key ||k_h||_2 and query ||q_h||_2
  __shared__ float s_norm[2];
  if (tid == 0) {
    float k_sq = 0.0F;
    float q_sq = 0.0F;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      k_sq += k_h[i] * k_h[i];
      q_sq += q_h[i] * q_h[i];
    }
    s_norm[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    s_norm[1] = 1.0F / sqrtf(q_sq + 1e-6F);
  }
  __syncthreads();
  const float q_scale = (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1];
  const float inv_k_norm = s_norm[0];

  // 3. Associative retrieval: u_j = sum_i S[j, i] * (k[i] * inv_k_norm)
  if (tid < val_dim) {
    const std::size_t j = tid;
    float* s_row = s_matrix + (j * key_dim);

    // Decay state while computing the retrieval. Keeping the decayed value
    // live avoids a second full read of the state row.
    float u_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] * alpha_h;
      s_row[i] = state;
      u_j += state * (k_h[i] * inv_k_norm);
    }
    const float d_j = (v_h[j] - u_j) * beta_h;

    // Apply the delta and compute the query readout from the same updated
    // value, avoiding another full read of the state row.
    float o_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] + d_j * (k_h[i] * inv_k_norm);
      s_row[i] = state;
      o_j += state * (q_h[i] * q_scale);
    }
    o_h[j] = o_j;
  }
  __syncthreads();

  // 4. Per-head RMSNorm on o_h (val_dim = 256 or 128 elements)
  __shared__ float s_sum[256];
  if (tid < val_dim) {
    s_sum[tid] = o_h[tid] * o_h[tid];
  } else {
    s_sum[tid] = 0.0F;
  }
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s)
      s_sum[tid] += s_sum[tid + s];
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(val_dim);
  const float rms = rsqrtf(mean_sq + 1e-6F);
  __syncthreads();

  if (tid < val_dim) {
    const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
    float val = o_h[tid] * rms * w;
    if (gate != nullptr) {
      const float g = gate[h * val_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      val *= (g * sig);  // SiLU gating
    }
    o_h[tid] = val;
  }
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           std::uint32_t token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (hidden_size + block_size - 1) / block_size;
  EmbeddingLookupKernel<<<grid_size, block_size, 0, stream>>>(
      table, type, token_id, out_hidden, hidden_size);
}

void LaunchEmbeddingLookup(const void* table, core::GgmlType type,
                           const std::uint32_t* d_token_id, float* out_hidden,
                           std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (hidden_size + block_size - 1) / block_size;
  EmbeddingLookupPtrKernel<<<grid_size, block_size, 0, stream>>>(
      table, type, d_token_id, out_hidden, hidden_size);
}

void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps, hipStream_t stream) {
  RMSNormKernel<<<1, 256, 0, stream>>>(x, weight, out, dim, eps);
}

void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (dim + block_size - 1) / block_size;
  ResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out, dim);
}

void LaunchFusedResidualAddRMSNorm(const float* a, const float* b,
                                   float* out_sum, const float* weight,
                                   float* out, std::size_t dim, float eps,
                                   hipStream_t stream) {
  FusedResidualAddRMSNormKernel<<<1, 256, 0, stream>>>(a, b, out_sum, weight,
                                                       out, dim, eps);
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

void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size =
      (intermediate_size + block_size - 1) / block_size;
  SwiGLUKernel<<<grid_size, block_size, 0, stream>>>(gate, up, out,
                                                     intermediate_size);
}

void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream) {
  // Quantized weights (kQ8_K now; more types added later) run the Q row-dot
  // path, which has no residual epilogue.
  if (a_type != core::GgmlType::kBF16 && a_type != core::GgmlType::kF32) {
    LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
    return;
  }
  const bool is_bf16 = (a_type == core::GgmlType::kBF16);
  const auto strategy = detail::SelectGemvStrategy(M, K, is_bf16);
  detail::EmitGemvDispatch(M, K, is_bf16, detail::GemvStrategyName(strategy));
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  switch (strategy) {
    case detail::GemvStrategy::kWave32QuadRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t num_quads = (M + 3) / 4;
      const std::size_t grid =
          (num_quads + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_4Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
      return;
    }
    case detail::GemvStrategy::kWave32DualRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t num_pairs = (M + 1) / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, M, K);
      return;
    }
    case detail::GemvStrategy::kWave32SingleRow: {
      const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
      const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32GEMVKernel_1Row<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, nullptr, M, K);
      return;
    }
    case detail::GemvStrategy::kBaselineBlock:
    default:
      break;
  }

  const std::size_t block_size = K >= kLargeGemvInput ? 512 : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, nullptr,
                                                    M, K);
}

// opt-c010-ssm-gate-residual: GEMV with an optional residual-add epilogue
// (y = A*x + residual). The dispatch mirrors LaunchGEMV; the Wave32 single-row
// and block strategies gain the residual fold, and the remaining strategies
// fall back to the single-row residual kernel. Used to fold the post-SSM
// residual add into the ssm_out GEMV for decode.
void LaunchGEMVResidual(const void* A, core::GgmlType a_type, const float* x,
                        float* y, const float* residual, std::size_t M,
                        std::size_t K, hipStream_t stream) {
  // Quantized weights run the Q row-dot path. The residual add is not folded
  // into the Q8K block GEMV, so a residual pointer is not applied (this path is
  // only reached for the ssm_out residual, which is currently dead).
  if (a_type != core::GgmlType::kBF16 && a_type != core::GgmlType::kF32) {
    LaunchQ8KBlockGEMV(A, a_type, x, y, M, K, stream);
    return;
  }
  const bool is_bf16 = (a_type == core::GgmlType::kBF16);
  const auto strategy = detail::SelectGemvStrategy(M, K, is_bf16);
  detail::EmitGemvDispatch(M, K, is_bf16, detail::GemvStrategyName(strategy));
  constexpr std::uint32_t kWavesPerBlock = 4;
  constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

  if (strategy != detail::GemvStrategy::kBaselineBlock) {
    const auto* A_bf16 = static_cast<const hip_bfloat16*>(A);
    const std::size_t grid = (M + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32GEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(A_bf16, x, y, residual, M, K);
    return;
  }

  const std::size_t block_size = K >= kLargeGemvInput ? 512 : 256;
  FastGEMVBlockKernel<<<M, block_size, 0, stream>>>(A, a_type, x, y, residual,
                                                    M, K);
}

void LaunchFusedSSMInputProjections(
    const void* qkv_w, core::GgmlType qkv_type, const void* gate_w,
    core::GgmlType gate_type, const void* alpha_w, core::GgmlType alpha_type,
    const void* beta_w, core::GgmlType beta_type, const float* x,
    float* qkv_out, float* gate_out, float* alpha_out, float* beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  if (hidden_size % 8 == 0) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedSSMInputProjectionsKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            qkv_w, qkv_type, gate_w, gate_type, alpha_w, alpha_type, beta_w,
            beta_type, x, qkv_out, gate_out, alpha_out, beta_out, hidden_size,
            qkv_size, inner_size, time_step_rank);
    return;
  }

  FusedSSMInputProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      qkv_w, qkv_type, gate_w, gate_type, alpha_w, alpha_type, beta_w, beta_type,
      x, qkv_out, gate_out, alpha_out, beta_out, hidden_size, qkv_size,
      inner_size, time_step_rank);
}

void LaunchFusedQKVProjections(const void* q_w, core::GgmlType q_type,
                               const void* k_w, core::GgmlType k_type,
                               const void* v_w, core::GgmlType v_type,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (hidden_size % 8 == 0) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;
    const std::size_t grid = (total_rows + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedQKVProjectionsKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            q_w, q_type, k_w, k_type, v_w, v_type, x, q_out, k_out, v_out,
            q_dim, kv_dim, hidden_size);
    return;
  }

  FusedQKVProjectionsKernel<<<total_rows, 128, 0, stream>>>(
      q_w, q_type, k_w, k_type, v_w, v_type, x, q_out, k_out, v_out, q_dim,
      kv_dim, hidden_size);
}

void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size, hipStream_t stream) {
  if (gate_type == core::GgmlType::kBF16 && up_type == core::GgmlType::kBF16 &&
      (hidden_size % 8 == 0)) {
    constexpr std::uint32_t kWavesPerBlock = 4;
    constexpr std::uint32_t kBlockThreads = kWavesPerBlock * 32;

    if (intermediate_size >= 2 && (intermediate_size % 2 == 0)) {
      const std::size_t num_pairs = intermediate_size / 2;
      const std::size_t grid =
          (num_pairs + kWavesPerBlock - 1) / kWavesPerBlock;
      Wave32FusedSwiGLUGEMVKernel_2Rows<kWavesPerBlock>
          <<<grid, kBlockThreads, 0, stream>>>(
              static_cast<const hip_bfloat16*>(gate_w),
              static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
              hidden_size);
      return;
    }

    const std::size_t grid =
        (intermediate_size + kWavesPerBlock - 1) / kWavesPerBlock;
    Wave32FusedSwiGLUGEMVKernel_1Row<kWavesPerBlock>
        <<<grid, kBlockThreads, 0, stream>>>(
            static_cast<const hip_bfloat16*>(gate_w),
            static_cast<const hip_bfloat16*>(up_w), x, out, intermediate_size,
            hidden_size);
    return;
  }

  FastFusedSwiGLUGEMVBlockKernel<<<intermediate_size, 512, 0, stream>>>(
      gate_w, gate_type, up_w, up_type, x, out, intermediate_size, hidden_size);
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + Q/K/V projections.
// 256-thread blocks (8 rows each); requires hidden_size % 8 == 0 for the
// Wave32 BF16 path. The norm scale is computed once per block, bit-identical
// to RMSNormKernel.
void LaunchFusedRMSNormQKVProjections(
    const float* x, const float* norm_w, float eps, const void* q_w,
    bool q_is_bf16, const void* k_w, bool k_is_bf16, const void* v_w,
    bool v_is_bf16, float* q_out, float* k_out, float* v_out, std::size_t q_dim,
    std::size_t kv_dim, std::size_t hidden_size, hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  const std::size_t grid = (total_rows + 7U) / 8U;
  FusedRMSNormQKVProjectionsKernel<<<grid, 256, 0, stream>>>(
      x, norm_w, eps, q_w, q_is_bf16, k_w, k_is_bf16, v_w, v_is_bf16, q_out,
      k_out, v_out, q_dim, kv_dim, hidden_size);
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + SSM input projections
// (QKV, gate, alpha, beta).
void LaunchFusedRMSNormSSMInputProjections(
    const float* x, const float* norm_w, float eps, const void* qkv_w,
    bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, float* qkv_out, float* gate_out, float* alpha_out,
    float* beta_out, std::size_t hidden_size, std::size_t qkv_size,
    std::size_t inner_size, std::size_t time_step_rank, hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  const std::size_t grid = (total_rows + 7U) / 8U;
  FusedRMSNormSSMInputProjectionsKernel<<<grid, 256, 0, stream>>>(
      x, norm_w, eps, qkv_w, qkv_is_bf16, gate_w, gate_is_bf16, alpha_w,
      alpha_is_bf16, beta_w, beta_is_bf16, qkv_out, gate_out, alpha_out,
      beta_out, hidden_size, qkv_size, inner_size, time_step_rank);
}

// opt-c010-rmsnorm-projection: fused layer pre-RMSNorm + FFN SwiGLU gate/up
// projections (BF16 weights only, matching the production decode route).
void LaunchFusedRMSNormSwiGLUGEMV(const float* x, const float* norm_w,
                                  float eps, const void* gate_w,
                                  const void* up_w, float* out,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const std::size_t num_pairs = intermediate_size / 2;
  const std::size_t grid = (num_pairs + 7U) / 8U;
  FusedRMSNormSwiGLUGEMVKernel<<<grid, 256, 0, stream>>>(
      x, norm_w, eps, static_cast<const hip_bfloat16*>(gate_w),
      static_cast<const hip_bfloat16*>(up_w), out, intermediate_size,
      hidden_size);
}

void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps, hipStream_t stream) {
  PerHeadRMSNormKernel<<<num_heads, 256, 0, stream>>>(x, weight, out, num_heads,
                                                      head_dim, eps);
}

void LaunchFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps, hipStream_t stream) {
  FusedQKNormRoPEKvWriteKernel<<<num_heads + num_kv_heads, 256, 0, stream>>>(
      q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
      static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16),
      layer_idx, d_pos, max_context, num_heads, num_kv_heads, head_dim,
      rotary_dim, rope_theta, eps);
}

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, SsmReplayCapture replay_capture) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_size + block_size - 1) / block_size;
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  SSMConvKernel<<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_layer_state, conv_out, qkv_size, layer_idx,
      replay_capture);

  DeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      conv_out, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm,
      gate, out_buf, layer_idx, num_key_heads, num_heads, key_dim, val_dim,
      replay_capture);
}

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

// =========================================================================
// Batched GPU Operators for High-Throughput Prompt Processing (Prefill)
// =========================================================================

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
