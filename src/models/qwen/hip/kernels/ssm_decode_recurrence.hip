#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/ssm.hpp"

namespace strix::hip {

__global__ void CaptureBatchedSsmReplayKernel(
    const float* __restrict__ qkv, const float* __restrict__ alpha,
    const float* __restrict__ beta, SsmReplayCapture replay_capture,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::size_t qkv_size, std::size_t time_step_rank) {
  if (replay_capture.qkv == nullptr || replay_capture.alpha == nullptr ||
      replay_capture.beta == nullptr || replay_capture.enabled == nullptr ||
      replay_capture.enabled[0] == 0U) {
    return;
  }

  const std::size_t row_width = qkv_size + (2 * time_step_rank);
  const std::size_t flat_index =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x)) + threadIdx.x;
  if (flat_index >= batch_size * row_width) {
    return;
  }

  const std::size_t row = flat_index / row_width;
  const std::size_t element = flat_index - (row * row_width);
  const std::size_t slot =
      (static_cast<std::size_t>(start_pos) + row) % kSsmReplayCapacity;
  const std::size_t layer_slot =
      (static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity) + slot;

  if (element < qkv_size) {
    replay_capture.qkv[(layer_slot * qkv_size) + element] =
        qkv[(row * qkv_size) + element];
    return;
  }

  const std::size_t projection_element = element - qkv_size;
  if (projection_element < time_step_rank) {
    replay_capture.alpha[(layer_slot * time_step_rank) + projection_element] =
        alpha[(row * time_step_rank) + projection_element];
  } else {
    const std::size_t beta_element = projection_element - time_step_rank;
    replay_capture.beta[(layer_slot * time_step_rank) + beta_element] =
        beta[(row * time_step_rank) + beta_element];
  }
}

void LaunchCaptureBatchedSsmReplay(
    const float* qkv, const float* alpha, const float* beta,
    SsmReplayCapture replay_capture, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::size_t qkv_size,
    std::size_t time_step_rank, hipStream_t stream) {
  if (batch_size == 0 || qkv_size == 0 || time_step_rank == 0 ||
      replay_capture.qkv == nullptr) {
    return;
  }
  const std::size_t elements = batch_size * (qkv_size + (2 * time_step_rank));
  constexpr unsigned int kBlockSize = 256;
  const auto grid_size =
      static_cast<unsigned int>((elements + kBlockSize - 1) / kBlockSize);
  CaptureBatchedSsmReplayKernel<<<grid_size, kBlockSize, 0, stream>>>(
      qkv, alpha, beta, replay_capture, layer_idx, start_pos, batch_size,
      qkv_size, time_step_rank);
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

  // 2. Normalize key ||k_h||_2 and query ||q_h||_2 in parallel
  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_norm[2];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];

  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  float local_k_sq = 0.0F;
  float local_q_sq = 0.0F;
  if (tid < key_dim) {
    const float kv = k_h[tid];
    const float qv = q_h[tid];
    s_k[tid] = kv;
    s_q[tid] = qv;
    local_k_sq = kv * kv;
    local_q_sq = qv * qv;
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_k_sq += __shfl_xor(local_k_sq, off);
    local_q_sq += __shfl_xor(local_q_sq, off);
  }
  if (lane_id == 0) {
    s_warp_k[warp_id] = local_k_sq;
    s_warp_q[warp_id] = local_q_sq;
  }
  __syncthreads();

  if (tid == 0) {
    float k_sq = 0.0F, q_sq = 0.0F;
    const std::size_t num_warps = blockDim.x >> 5u;
    for (std::size_t w = 0; w < num_warps; ++w) {
      k_sq += s_warp_k[w];
      q_sq += s_warp_q[w];
    }
    s_norm[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    s_norm[1] = 1.0F / sqrtf(q_sq + 1e-6F);
  }
  __syncthreads();

  const float inv_k_norm = s_norm[0];
  const float q_scale = (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1];
  if (tid < key_dim) {
    s_k[tid] *= inv_k_norm;
    s_q[tid] *= q_scale;
  }
  __syncthreads();

  // 3. Associative retrieval & state update with vectorized float4
  if (tid < val_dim) {
    const std::size_t j = tid;
    float* s_row = s_matrix + (j * key_dim);
    const float4* s_row4 = reinterpret_cast<const float4*>(s_row);
    float4* s_row4_out = reinterpret_cast<float4*>(s_row);
    const float4* k4 = reinterpret_cast<const float4*>(s_k);
    const float4* q4 = reinterpret_cast<const float4*>(s_q);

    const std::size_t key_vec = key_dim / 4;

    float u_j = 0.0F;
#pragma unroll
    for (std::size_t vi = 0; vi < key_vec; ++vi) {
      const float4 st = s_row4[vi];
      const float4 kv = k4[vi];
      u_j += (st.x * alpha_h) * kv.x + (st.y * alpha_h) * kv.y +
             (st.z * alpha_h) * kv.z + (st.w * alpha_h) * kv.w;
    }
    const float d_j = (v_h[j] - u_j) * beta_h;

    float o_j = 0.0F;
#pragma unroll
    for (std::size_t vi = 0; vi < key_vec; ++vi) {
      float4 st = s_row4[vi];
      const float4 kv = k4[vi];
      const float4 qv = q4[vi];
      st.x = (st.x * alpha_h) + (d_j * kv.x);
      st.y = (st.y * alpha_h) + (d_j * kv.y);
      st.z = (st.z * alpha_h) + (d_j * kv.z);
      st.w = (st.w * alpha_h) + (d_j * kv.w);
      s_row4_out[vi] = st;
      o_j += st.x * qv.x + st.y * qv.y + st.z * qv.z + st.w * qv.w;
    }
    o_h[j] = o_j;
  }
  __syncthreads();

  // 4. Per-head RMSNorm on o_h with warp reductions
  float val_sq = (tid < val_dim) ? (o_h[tid] * o_h[tid]) : 0.0F;
  for (int off = 16; off > 0; off >>= 1) {
    val_sq += __shfl_xor(val_sq, off);
  }
  __shared__ float s_val_warp[8];
  if (lane_id == 0) {
    s_val_warp[warp_id] = val_sq;
  }
  __syncthreads();

  if (tid == 0) {
    float total_sq = 0.0F;
    const std::size_t num_warps = blockDim.x >> 5u;
    for (std::size_t w = 0; w < num_warps; ++w) {
      total_sq += s_val_warp[w];
    }
    const float mean_sq = total_sq / static_cast<float>(val_dim);
    s_norm[0] = rsqrtf(mean_sq + 1e-6F);
  }
  __syncthreads();

  const float rms = s_norm[0];
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

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
