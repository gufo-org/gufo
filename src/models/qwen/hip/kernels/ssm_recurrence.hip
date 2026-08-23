#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

// Batched Causal SSM Conv1D + token-serial DeltaNet recurrence (pure recurrence
// unit). The fused per-head post-RMSNorm + gate epilogue lives in
// qwen_gpu_batched_ssm.hip (Option B: composition layer owns the fusion); the
// non-fused launcher references it through this forward declaration.
__global__ void BatchedSSMPostNormGateKernel(
    const float* __restrict__ raw_out, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, float* __restrict__ out_buf,
    std::size_t batch_size, std::size_t inner_size, std::uint32_t num_heads,
    std::uint32_t val_dim);

__global__ void BatchedSSMConvKernel(const float* __restrict__ qkv_in,
                                     const float* __restrict__ conv_weights,
                                     float* __restrict__ conv_state,
                                     float* __restrict__ conv_out,
                                     std::size_t batch_size,
                                     std::size_t qkv_dim) {
  const std::size_t c = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t t = blockIdx.y;
  if (c >= qkv_dim || t >= batch_size) {
    return;
  }

  const std::size_t c_off = c * 4;
  const float w0 = (conv_weights != nullptr) ? conv_weights[c * 4 + 0] : 0.0F;
  const float w1 = (conv_weights != nullptr) ? conv_weights[c * 4 + 1] : 0.0F;
  const float w2 = (conv_weights != nullptr) ? conv_weights[c * 4 + 2] : 0.0F;
  const float w3 = (conv_weights != nullptr) ? conv_weights[c * 4 + 3] : 1.0F;

  const float s3 = qkv_in[(t * qkv_dim) + c];
  const float s2 =
      (t >= 1) ? qkv_in[((t - 1) * qkv_dim) + c] : conv_state[c_off + 3];
  const float s1 = (t >= 2)   ? qkv_in[((t - 2) * qkv_dim) + c]
                   : (t == 1) ? conv_state[c_off + 3]
                              : conv_state[c_off + 2];
  const float s0 = (t >= 3)   ? qkv_in[((t - 3) * qkv_dim) + c]
                   : (t == 2) ? conv_state[c_off + 3]
                   : (t == 1) ? conv_state[c_off + 2]
                              : conv_state[c_off + 1];

  float dot = 0.0F;
  if (conv_weights != nullptr) {
    dot = (s0 * w0) + (s1 * w1) + (s2 * w2) + (s3 * w3);
  } else {
    dot = s3;
  }

  const float sig = 1.0F / (1.0F + expf(-dot));
  conv_out[(t * qkv_dim) + c] = dot * sig;

  // Final token updates state for future decode steps
  if (t == batch_size - 1) {
    conv_state[c_off + 0] = s0;
    conv_state[c_off + 1] = s1;
    conv_state[c_off + 2] = s2;
    conv_state[c_off + 3] = s3;
  }
}

__global__ void BatchedDeltaNetRecurrenceKernel(
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ raw_out, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::uint32_t kh_idx = h % num_key_heads;

  float* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;

  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];
  __shared__ float s_warp_kq[8];
  __shared__ float s_scales[3];  // inv_k_norm, q_scale, kq_dot

  // Two adjacent lanes own one state row, cutting the live register tile in
  // half while preserving the token-serial recurrence.
  constexpr std::size_t row_vectors_per_thread = 16;
  constexpr std::size_t row_values_per_thread =
      row_vectors_per_thread * sizeof(float4) / sizeof(float);
  const std::size_t j = tid >> 1U;
  const std::size_t row_segment = tid & 1U;
  const std::size_t row_offset = row_segment * row_values_per_thread;

  float4 s_reg[row_vectors_per_thread];
  if (j < val_dim) {
    const float* s_row = s_matrix + (j * key_dim) + row_offset;
    const auto* s_row_vec = reinterpret_cast<const float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_reg[vi] = s_row_vec[vi];
    }
  }

  for (std::size_t t = 0; t < batch_size; ++t) {
    const float* conv_t = conv_out + (t * qkv_size);
    const float* q_h = conv_t + (kh_idx * key_dim);
    const float* k_h = conv_t + (num_key_heads * key_dim) + (kh_idx * key_dim);
    const float* v_h = conv_t + (2 * num_key_heads * key_dim) + (h * val_dim);
    const std::size_t inner_size =
        static_cast<std::size_t>(num_heads) * val_dim;
    float* o_h = raw_out + (t * inner_size) + (h * val_dim);

    const float alpha_in =
        (alpha_buf != nullptr) ? (alpha_buf + (t * num_heads))[h] : 0.0F;
    const float beta_in =
        (beta_buf != nullptr) ? (beta_buf + (t * num_heads))[h] : 0.0F;

    const float alpha_biased = alpha_in + dt;
    const float alpha_softplus =
        (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
    const float alpha_h = expf(alpha_softplus * a_val);
    const float beta_h = 1.0F / (1.0F + expf(-beta_in));

    // 1. Normalize k and q. The upper four waves contribute zeros.
    const float raw_k = (tid < key_dim) ? k_h[tid] : 0.0F;
    const float raw_q = (tid < key_dim) ? q_h[tid] : 0.0F;
    float k_sq = raw_k * raw_k;
    float q_sq = raw_q * raw_q;

    for (int offset = 16; offset > 0; offset >>= 1) {
      k_sq += __shfl_xor(k_sq, offset);
      q_sq += __shfl_xor(q_sq, offset);
    }
    if (lane_id == 0) {
      s_warp_k[warp_id] = k_sq;
      s_warp_q[warp_id] = q_sq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_k_sq = 0.0F;
      float total_q_sq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_k_sq += s_warp_k[warp];
        total_q_sq += s_warp_q[warp];
      }
      s_scales[0] = 1.0F / sqrtf(total_k_sq + 1e-6F);
      s_scales[1] = (1.0F / sqrtf(static_cast<float>(key_dim))) *
                    (1.0F / sqrtf(total_q_sq + 1e-6F));
    }
    __syncthreads();

    const float inv_k_norm = s_scales[0];
    const float q_scale = s_scales[1];
    const float norm_k = raw_k * inv_k_norm;
    const float norm_q = raw_q * q_scale;

    if (tid < 128) {
      s_k[tid] = norm_k;
      s_q[tid] = norm_q;
    }

    // 2. Parallel reduction for kq_dot
    float local_kq = norm_k * norm_q;
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_kq += __shfl_xor(local_kq, offset);
    }
    if (lane_id == 0) {
      s_warp_kq[warp_id] = local_kq;
    }
    __syncthreads();

    if (tid == 0) {
      float total_kq = 0.0F;
#pragma unroll
      for (std::size_t warp = 0; warp < 8; ++warp) {
        total_kq += s_warp_kq[warp];
      }
      s_scales[2] = total_kq;
    }
    __syncthreads();
    const float kq_dot = s_scales[2];

    // 3. Each lane processes half a row, then exchanges its two partial dots
    // with the adjacent lane that owns the other half.
    if (j < val_dim) {
      const auto* k_vec = reinterpret_cast<const float4*>(s_k) +
                          (row_segment * row_vectors_per_thread);
      const auto* q_vec = reinterpret_cast<const float4*>(s_q) +
                          (row_segment * row_vectors_per_thread);

      float u_j = 0.0F;
      float p_j = 0.0F;

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[vi] = s4;

        const float4 kv = k_vec[vi];
        const float4 qv = q_vec[vi];

        u_j += (s4.x * kv.x) + (s4.y * kv.y) + (s4.z * kv.z) + (s4.w * kv.w);
        p_j += (s4.x * qv.x) + (s4.y * qv.y) + (s4.z * qv.z) + (s4.w * qv.w);
      }

      u_j += __shfl_xor(u_j, 1);
      p_j += __shfl_xor(p_j, 1);
      const float d_j = (v_h[j] - u_j) * beta_h;
      if (row_segment == 0) {
        o_h[j] = p_j + (d_j * kq_dot);
      }

#pragma unroll
      for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
        float4 s4 = s_reg[vi];
        const float4 kv = k_vec[vi];
        s4.x += (d_j * kv.x);
        s4.y += (d_j * kv.y);
        s4.z += (d_j * kv.z);
        s4.w += (d_j * kv.w);
        s_reg[vi] = s4;
      }
    }
  }

  // Write final row state back to global memory ONCE
  if (j < val_dim) {
    float* s_row = s_matrix + (j * key_dim) + row_offset;
    auto* s_row_vec = reinterpret_cast<float4*>(s_row);
#pragma unroll
    for (std::size_t vi = 0; vi < row_vectors_per_thread; ++vi) {
      s_row_vec[vi] = s_reg[vi];
    }
  }
}

void LaunchBatchedSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_x = (qkv_size + block_size - 1) / block_size;
  const dim3 conv_grid(static_cast<unsigned int>(grid_x),
                       static_cast<unsigned int>(batch_size));
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  BatchedSSMConvKernel<<<conv_grid, static_cast<unsigned int>(block_size), 0,
                         stream>>>(qkv_in, conv_weights, conv_layer_state,
                                   conv_out, batch_size, qkv_size);

  BatchedDeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      conv_out, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, out_buf,
      layer_idx, batch_size, qkv_size, num_key_heads, num_heads, key_dim,
      val_dim);

  const dim3 post_grid(num_heads, static_cast<unsigned int>(batch_size));
  const std::size_t inner_size = static_cast<std::size_t>(num_heads) * val_dim;
  BatchedSSMPostNormGateKernel<<<post_grid, 128, 0, stream>>>(
      out_buf, ssm_norm, gate, out_buf, batch_size, inner_size, num_heads,
      val_dim);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
