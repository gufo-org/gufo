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
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen/hip/qwen_gpu_quant_ops.hpp"
#include "src/core/gguf_reader.hpp"

namespace strix::hip {

namespace {

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
}  // namespace

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
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

