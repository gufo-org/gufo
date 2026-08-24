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
}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
