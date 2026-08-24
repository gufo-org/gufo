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
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
#include "src/core/gguf_reader.hpp"

#include "src/models/qwen/hip/kernels/common.hip"

namespace strix::hip {

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
