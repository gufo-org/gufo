#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/models/qwen/hip/detail/recurrent_state.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

template<detail::QwenRecurrentStateElement StateT, bool Resident = false,
         bool WriteOutput = true>
__launch_bounds__(Resident ? 128 : 1024, 1) __global__
    void DeltaNetRecurrenceKernel(
        const float* __restrict__ conv_out, StateT* __restrict__ deltanet_state,
        const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
        const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
        const float* __restrict__ ssm_norm, const float* __restrict__ gate,
        float* __restrict__ out_buf, std::uint32_t layer_idx,
        std::uint32_t num_key_heads, std::uint32_t num_heads,
        std::uint32_t key_dimension, std::uint32_t value_dimension,
        SsmReplayCapture replay_capture, std::uint32_t rows,
        std::size_t conv_row_stride, std::size_t projection_row_stride,
        std::size_t inner_row_stride) {
  static_assert(!Resident || std::is_same_v<StateT, float>);
  const std::uint32_t key_dim = Resident ? 128 : key_dimension;
  const std::uint32_t val_dim = Resident ? 128 : value_dimension;
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;

  // opt-c191-ssm-rows: block `h` owns head h's recurrent state exclusively, so
  // walking `rows` consecutive verification rows here reproduces exactly the
  // sequence of updates that `rows` separate launches produced.
  __shared__ float s_k[128];
  __shared__ float s_q[128];
  __shared__ float s_norm[2];
  __shared__ float s_warp_k[8];
  __shared__ float s_warp_q[8];
  __shared__ float s_val_warp[8];

  StateT* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  // Keep each FP32 state row across the verification/replay block. Generic
  // shapes and BF16 storage retain their per-token storage transitions.
  float4 resident[Resident ? 32 : 1];
  if constexpr (Resident) {
#pragma unroll
    for (std::size_t vi = 0; vi < 32; ++vi)
      resident[vi] = detail::LoadRecurrentState4(s_matrix + tid * 128, vi);
  }

  for (std::uint32_t row = 0; row < rows; ++row) {
    const float* q_ptr =
        conv_out + (static_cast<std::size_t>(row) * conv_row_stride);
    const float* k_ptr = q_ptr + (num_key_heads * key_dim);
    const float* v_ptr = q_ptr + (num_key_heads * key_dim * 2);

    const std::uint32_t kh_idx = h % num_key_heads;
    const float* q_h = q_ptr + (kh_idx * key_dim);
    const float* k_h = k_ptr + (kh_idx * key_dim);
    const float* v_h = v_ptr + (h * val_dim);
    float* o_h = nullptr;
    if constexpr (WriteOutput) {
      o_h = out_buf + (static_cast<std::size_t>(row) * inner_row_stride) +
            (h * val_dim);
    }
    const float* row_alpha = (alpha_buf != nullptr)
                                 ? alpha_buf + (static_cast<std::size_t>(row) *
                                                projection_row_stride)
                                 : nullptr;
    const float* row_beta =
        (beta_buf != nullptr)
            ? beta_buf + (static_cast<std::size_t>(row) * projection_row_stride)
            : nullptr;
    const float* row_gate =
        (gate != nullptr)
            ? gate + (static_cast<std::size_t>(row) * inner_row_stride)
            : nullptr;

    // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta
    // = sigmoid(beta)
    const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
    const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
    const float alpha_in = (row_alpha != nullptr) ? row_alpha[h] : 0.0F;
    const float beta_in = (row_beta != nullptr) ? row_beta[h] : 0.0F;

    if (tid == 0 && replay_capture.alpha != nullptr &&
        replay_capture.beta != nullptr && replay_capture.position != nullptr &&
        replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
      const std::size_t slot =
          static_cast<std::size_t>(replay_capture.position[row]) %
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
    const std::size_t lane_id = tid & 31u;
    const std::size_t warp_id = tid >> 5u;

    float local_k_sq = 0.0F;
    float local_q_sq = 0.0F;
    if (tid < key_dim) {
      const float kv = k_h[tid];
      const float qv = q_h[tid];
      s_k[tid] = kv;
      if constexpr (WriteOutput)
        s_q[tid] = qv;
      local_k_sq = kv * kv;
      if constexpr (WriteOutput)
        local_q_sq = qv * qv;
    }
    for (int off = 16; off > 0; off >>= 1) {
      local_k_sq += __shfl_xor(local_k_sq, off);
      if constexpr (WriteOutput)
        local_q_sq += __shfl_xor(local_q_sq, off);
    }
    if (lane_id == 0) {
      s_warp_k[warp_id] = local_k_sq;
      if constexpr (WriteOutput)
        s_warp_q[warp_id] = local_q_sq;
    }
    __syncthreads();

    if (tid == 0) {
      float k_sq = 0.0F, q_sq = 0.0F;
      const std::size_t num_warps = blockDim.x >> 5u;
      for (std::size_t w = 0; w < num_warps; ++w) {
        k_sq += s_warp_k[w];
        if constexpr (WriteOutput)
          q_sq += s_warp_q[w];
      }
      s_norm[0] = 1.0F / sqrtf(k_sq + 1e-6F);
      if constexpr (WriteOutput)
        s_norm[1] = 1.0F / sqrtf(q_sq + 1e-6F);
    }
    __syncthreads();

    const float inv_k_norm = s_norm[0];
    const float q_scale =
        WriteOutput ? (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1]
                    : 0.0F;
    if (tid < key_dim) {
      s_k[tid] *= inv_k_norm;
      if constexpr (WriteOutput)
        s_q[tid] *= q_scale;
    }
    __syncthreads();

    // 3. Associative retrieval & state update with vectorized float4
    if (tid < val_dim) {
      const std::size_t j = tid;
      StateT* s_row = s_matrix + (j * key_dim);
      const float4* k4 = reinterpret_cast<const float4*>(s_k);
      const float4* q4 = reinterpret_cast<const float4*>(s_q);

      const std::size_t key_vec = Resident ? 32 : key_dim / 4;

      float u_j = 0.0F;
#pragma unroll
      for (std::size_t vi = 0; vi < key_vec; ++vi) {
        float4 st;
        if constexpr (Resident)
          st = resident[vi];
        else
          st = detail::LoadRecurrentState4(s_row, vi);
        const float4 kv = k4[vi];
        u_j += (st.x * alpha_h) * kv.x + (st.y * alpha_h) * kv.y +
               (st.z * alpha_h) * kv.z + (st.w * alpha_h) * kv.w;
        if constexpr (Resident) {
          // Bound shared-memory load hoisting so the 128 state values remain
          // in registers. This compiler barrier emits no GPU instruction.
          if (vi % 4 == 3)
            asm volatile("" ::: "memory");
        }
      }
      const float d_j = (v_h[j] - u_j) * beta_h;

      float o_j = 0.0F;
#pragma unroll
      for (std::size_t vi = 0; vi < key_vec; ++vi) {
        float4 st;
        if constexpr (Resident)
          st = resident[vi];
        else
          st = detail::LoadRecurrentState4(s_row, vi);
        const float4 kv = k4[vi];
        const float4 qv = WriteOutput ? q4[vi] : float4{};
        if constexpr (Resident) {
          // Match decode's rounding: delta*key rounds before the state-decay
          // FMA. Reusing the rounded state*alpha here changes the recurrence.
          st.x = fmaf(st.x, alpha_h, d_j * kv.x);
          st.y = fmaf(st.y, alpha_h, d_j * kv.y);
          st.z = fmaf(st.z, alpha_h, d_j * kv.z);
          st.w = fmaf(st.w, alpha_h, d_j * kv.w);
        } else {
          st.x = (st.x * alpha_h) + (d_j * kv.x);
          st.y = (st.y * alpha_h) + (d_j * kv.y);
          st.z = (st.z * alpha_h) + (d_j * kv.z);
          st.w = (st.w * alpha_h) + (d_j * kv.w);
        }
        if constexpr (Resident)
          resident[vi] = st;
        else
          detail::StoreRecurrentState4(s_row, vi, st);
        if constexpr (WriteOutput)
          o_j += st.x * qv.x + st.y * qv.y + st.z * qv.z + st.w * qv.w;
        if constexpr (Resident) {
          // Bound shared-memory load hoisting so the 128 state values remain
          // in registers. This compiler barrier emits no GPU instruction.
          if (vi % 4 == 3)
            asm volatile("" ::: "memory");
        }
      }
      if constexpr (WriteOutput)
        o_h[j] = o_j;
    }
    __syncthreads();

    // Replay only consumes recurrent state. The barrier above prevents the
    // next row from overwriting shared keys while another wave uses them.
    if constexpr (!WriteOutput)
      continue;

    // 4. Per-head RMSNorm on o_h with warp reductions
    float val_sq = (tid < val_dim) ? (o_h[tid] * o_h[tid]) : 0.0F;
    for (int off = 16; off > 0; off >>= 1) {
      val_sq += __shfl_xor(val_sq, off);
    }
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
      if (row_gate != nullptr) {
        const float g = row_gate[h * val_dim + tid];
        const float sig = 1.0F / (1.0F + expf(-g));
        val *= (g * sig);  // SiLU gating
      }
      o_h[tid] = val;
    }
    __syncthreads();
  }
  if constexpr (Resident) {
#pragma unroll
    for (std::size_t vi = 0; vi < 32; ++vi)
      detail::StoreRecurrentState4(s_matrix + tid * 128, vi, resident[vi]);
  }
}

}  // namespace gufo::hip
#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_DELTANET_DECODE_HPP_
