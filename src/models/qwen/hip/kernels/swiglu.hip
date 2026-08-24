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
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
#include "src/core/gguf_reader.hpp"


namespace strix::hip {

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

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
