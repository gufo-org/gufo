// Token-serial DeltaNet recurrence with independent state-row groups.
// Prologues compute per-head K/Q norms, the raw K.Q dot product, alpha and beta.
// Convolution history advances after all readers of the old history finish.
// The recurrence applies the shared scales to its dot products and rank-one
// update, then the epilogue normalizes and gates output.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/detail/recurrent_state.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

// Defined in kernels/ssm_recurrence.hip and hip/batched_ssm.hip respectively;
// the row-split route shares the convolution and post-normalization kernels.
__global__ void BatchedSSMConvKernel(const float* __restrict__ qkv_in,
                                     const float* __restrict__ conv_weights,
                                     const float* __restrict__ conv_state,
                                     float* __restrict__ conv_out,
                                     std::size_t batch_size,
                                     std::size_t qkv_dim);

__global__ void BatchedSSMPostNormGateKernel(
    const float* __restrict__ raw_out, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, float* __restrict__ out_buf,
    std::size_t batch_size, std::size_t inner_size, std::uint32_t num_heads,
    std::uint32_t val_dim);

// The row-split tile geometry is fixed at key_dim == val_dim == 128, which is
// what every shipped Qwen3-Next linear-attention configuration uses. Other
// shapes fall back to the previous kernel.
constexpr int kRowSplitDim = 128;

// Crossover between the two register tiles, measured in
// tools/qwen27b/prefill_deltanet_bench.hip.
constexpr std::size_t kRowSplitSmallBatch = 2048;

// One wave per (token, key head). 128 keys is exactly four values per lane, so
// the three reductions are a single butterfly with no LDS and no barrier.
__launch_bounds__(32) __global__
    void BatchedDeltaNetPrepKqKernel(const float* __restrict__ conv_out,
                                     float* __restrict__ kq_scales,
                                     std::size_t qkv_size,
                                     std::uint32_t num_key_heads) {
  const std::uint32_t kh = blockIdx.x;
  const std::size_t t = blockIdx.y;
  const int lane = static_cast<int>(threadIdx.x);

  const float* conv_t = conv_out + (t * qkv_size);
  const auto* qp = reinterpret_cast<const float4*>(
      conv_t + (static_cast<std::size_t>(kh) * kRowSplitDim));
  const auto* kp = reinterpret_cast<const float4*>(
      conv_t + (static_cast<std::size_t>(num_key_heads) * kRowSplitDim) +
      (static_cast<std::size_t>(kh) * kRowSplitDim));
  const float4 q4 = qp[lane];
  const float4 k4 = kp[lane];

  float k_sq = (k4.x * k4.x) + (k4.y * k4.y) + (k4.z * k4.z) + (k4.w * k4.w);
  float q_sq = (q4.x * q4.x) + (q4.y * q4.y) + (q4.z * q4.z) + (q4.w * q4.w);
  float kq = (k4.x * q4.x) + (k4.y * q4.y) + (k4.z * q4.z) + (k4.w * q4.w);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    k_sq += __shfl_xor(k_sq, offset);
    q_sq += __shfl_xor(q_sq, offset);
    kq += __shfl_xor(kq, offset);
  }
  if (lane == 0) {
    float* out = kq_scales + (((t * num_key_heads) + kh) * 3);
    out[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    out[1] = (1.0F / sqrtf(static_cast<float>(kRowSplitDim))) *
             (1.0F / sqrtf(q_sq + 1e-6F));
    out[2] = kq;
  }
}

// Softplus gate and beta sigmoid, per (token, value head). Uniform across the
// state rows, so hoisting it out of the recurrence removes it from every one of
// the 16 waves that now share a head.
__launch_bounds__(256) __global__ void BatchedDeltaNetPrepAlphaBetaKernel(
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    float* __restrict__ ab, std::size_t count, std::uint32_t num_heads,
    const float* __restrict__ qkv_in, float* __restrict__ conv_state,
    std::size_t batch_size, std::size_t qkv_size) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < qkv_size) {
    detail::StoreSSMConvHistory(qkv_in, conv_state, i, batch_size, qkv_size);
  }
  if (i >= count) {
    return;
  }
  const std::uint32_t h = static_cast<std::uint32_t>(i % num_heads);
  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
  const float alpha_in = (alpha_buf != nullptr) ? alpha_buf[i] : 0.0F;
  const float beta_in = (beta_buf != nullptr) ? beta_buf[i] : 0.0F;

  const float alpha_biased = alpha_in + dt;
  const float alpha_softplus =
      (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
  ab[i * 2] = expf(alpha_softplus * a_val);
  ab[(i * 2) + 1] = 1.0F / (1.0F + expf(-beta_in));
}

namespace {

// __shfl_xor lowers to ds_bpermute_b32, which takes the LDS path. For xor masks
// inside a 16-lane DPP row the same exchange is an ALU-path DPP modifier
// (ROW_XMASK, gfx10 and later).
template<int kMask>
__device__ __forceinline__ float XorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

template<int kLanesPerRow>
__device__ __forceinline__ void RowReduce2(float& u, float& p) {
  if (kLanesPerRow > 1) {
    u = XorAddDpp<1>(u);
    p = XorAddDpp<1>(p);
  }
  if (kLanesPerRow > 2) {
    u = XorAddDpp<2>(u);
    p = XorAddDpp<2>(p);
  }
  if (kLanesPerRow > 4) {
    u = XorAddDpp<4>(u);
    p = XorAddDpp<4>(p);
  }
  if (kLanesPerRow > 8) {
    u = XorAddDpp<8>(u);
    p = XorAddDpp<8>(p);
  }
}

}  // namespace

// kKeysPerLane lanes cooperate on one state row and each lane carries
// kRowsPerLane independent rows, so the resident wave count is
// num_heads * 16384 / (32 * kKeysPerLane * kRowsPerLane). Every base pointer is
// block-uniform and advances by one scalar add per token, and every
// lane-dependent index is loop-invariant, which keeps the token loop free of
// 64-bit address arithmetic.
template<detail::QwenRecurrentStateElement StateT, int kKeysPerLane,
         int kRowsPerLane, bool kPrefetch>
__launch_bounds__(256) __global__ void BatchedDeltaNetRowSplitKernel(
    const float* __restrict__ conv_out, const float* __restrict__ kq_scales,
    const float* __restrict__ ab, StateT* __restrict__ deltanet_state,
    float* __restrict__ raw_out, std::uint32_t layer_idx, int batch_size,
    int qkv_size, int inner_size, int num_key_heads, int num_heads) {
  constexpr int kLanesPerRow = kRowSplitDim / kKeysPerLane;
  constexpr int kRowGroups = 32 / kLanesPerRow;
  constexpr int kRowsPerWave = kRowGroups * kRowsPerLane;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  constexpr int kVec = kKeysPerLane / 4;
  static_assert(kKeysPerLane % 4 == 0, "float4 state and k/q loads");
  static_assert(kLanesPerRow <= 32, "a row group must fit in one wave");

  const int h = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int seg = lane % kLanesPerRow;
  const int grp = lane / kLanesPerRow;
  const int row0 = (static_cast<int>(blockIdx.x) * kRowsPerBlock) +
                   (wave * kRowsPerWave) + grp;
  const int kh = h % num_key_heads;
  const int kq_idx = seg * kVec;

  StateT* s_head = deltanet_state +
                   ((((static_cast<std::size_t>(layer_idx) * num_heads) + h)) *
                    kRowSplitDim * kRowSplitDim);

  float4 s_reg[kRowsPerLane][kVec];
#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    const StateT* src =
        s_head +
        (static_cast<std::size_t>(row0 + (r * kRowGroups)) * kRowSplitDim);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      s_reg[r][vi] = detail::LoadRecurrentState4(src, kq_idx + vi);
    }
  }

  // Block-uniform bases; only these advance in the token loop.
  const float* q_base = conv_out + (kh * kRowSplitDim);
  const float* k_base = conv_out + ((num_key_heads + kh) * kRowSplitDim);
  const float* v_base =
      conv_out + ((2 * num_key_heads * kRowSplitDim) + (h * kRowSplitDim));
  const float* kq_base = kq_scales + (kh * 3);
  const float* ab_base = ab + (h * 2);
  float* o_base = raw_out + (h * kRowSplitDim);

  float4 k_next[kVec];
  float4 q_next[kVec];
  if (kPrefetch) {
    const auto* kp = reinterpret_cast<const float4*>(k_base);
    const auto* qp = reinterpret_cast<const float4*>(q_base);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      k_next[vi] = kp[kq_idx + vi];
      q_next[vi] = qp[kq_idx + vi];
    }
  }

  for (int t = 0; t < batch_size; ++t) {
    float4 kv[kVec];
    float4 qv[kVec];
    if (kPrefetch) {
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = k_next[vi];
        qv[vi] = q_next[vi];
      }
      if (t + 1 < batch_size) {
        const auto* kp = reinterpret_cast<const float4*>(k_base + qkv_size);
        const auto* qp = reinterpret_cast<const float4*>(q_base + qkv_size);
#pragma unroll
        for (int vi = 0; vi < kVec; ++vi) {
          k_next[vi] = kp[kq_idx + vi];
          q_next[vi] = qp[kq_idx + vi];
        }
      }
    } else {
      const auto* kp = reinterpret_cast<const float4*>(k_base);
      const auto* qp = reinterpret_cast<const float4*>(q_base);
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        kv[vi] = kp[kq_idx + vi];
        qv[vi] = qp[kq_idx + vi];
      }
    }

    const float inv_k_norm = kq_base[0];
    const float q_scale = kq_base[1];
    const float kq_dot = inv_k_norm * q_scale * kq_base[2];
    const float alpha_h = ab_base[0];
    const float beta_h = ab_base[1];

#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      float u = 0.0F;
      float p = 0.0F;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        s4.x *= alpha_h;
        s4.y *= alpha_h;
        s4.z *= alpha_h;
        s4.w *= alpha_h;
        s_reg[r][vi] = s4;
        const float4 kk = kv[vi];
        const float4 qq = qv[vi];
        u += (s4.x * kk.x) + (s4.y * kk.y) + (s4.z * kk.z) + (s4.w * kk.w);
        p += (s4.x * qq.x) + (s4.y * qq.y) + (s4.z * qq.z) + (s4.w * qq.w);
      }
      RowReduce2<kLanesPerRow>(u, p);
      const int row = row0 + (r * kRowGroups);
      const float d = (v_base[row] - (u * inv_k_norm)) * beta_h;
      if (seg == 0) {
        o_base[row] = (p * q_scale) + (d * kq_dot);
      }
      const float dk = d * inv_k_norm;
#pragma unroll
      for (int vi = 0; vi < kVec; ++vi) {
        float4 s4 = s_reg[r][vi];
        const float4 kk = kv[vi];
        s4.x += dk * kk.x;
        s4.y += dk * kk.y;
        s4.z += dk * kk.z;
        s4.w += dk * kk.w;
        s_reg[r][vi] = s4;
      }
    }

    q_base += qkv_size;
    k_base += qkv_size;
    v_base += qkv_size;
    kq_base += num_key_heads * 3;
    ab_base += num_heads * 2;
    o_base += inner_size;
  }

#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    StateT* dst = s_head + (static_cast<std::size_t>(row0 + (r * kRowGroups)) *
                            kRowSplitDim);
#pragma unroll
    for (int vi = 0; vi < kVec; ++vi) {
      detail::StoreRecurrentState4(dst, kq_idx + vi, s_reg[r][vi]);
    }
  }
}

// Rows of the state covered by one 256-thread block for a given register tile.
template<int kKeysPerLane, int kRowsPerLane>
constexpr unsigned int RowSplitRowsPerBlock() {
  return static_cast<unsigned int>(
      ((32 / (kRowSplitDim / kKeysPerLane)) * kRowsPerLane) * 8);
}

bool IsDeltaNetRowSplitSupported(std::uint32_t key_dim,
                                 std::uint32_t val_dim) noexcept {
  return key_dim == static_cast<std::uint32_t>(kRowSplitDim) &&
         val_dim == static_cast<std::uint32_t>(kRowSplitDim);
}

namespace {
// One wave per head retains the four independent reductions from the FP32
// epilogue. Writing FP16 directly avoids materializing and rereading its FP32
// output; the recurrence still computes and stores FP32.
__global__ void BatchedSSMPostNormGateFp16Kernel(const float* raw,
                                                 const float* weight,
                                                 const float* gate, __half* out,
                                                 std::size_t head_count) {
  const unsigned lane = threadIdx.x & 31U;
  const std::size_t head = std::size_t(blockIdx.x) * 8 + threadIdx.x / 32;
  if (head >= head_count) {
    return;
  }
  const std::size_t base = head * kRowSplitDim;
  float values[4], squares[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    values[j] = raw[base + lane + j * 32];
    squares[j] = __fmul_rn(values[j], values[j]);
  }
  for (int shift = 16; shift > 0; shift >>= 1) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      squares[j] += __shfl_xor(squares[j], shift);
    }
  }
  const float sum =
      __shfl(__fadd_rn(__fadd_rn(__fadd_rn(squares[0], squares[1]), squares[2]),
                       squares[3]),
             0);
  const float inv =
      rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0F / kRowSplitDim), 1e-6F));
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned column = lane + j * 32;
    float result = (values[j] * inv) * (weight ? weight[column] : 1.0F);
    if (gate) {
      const float g = gate[base + column];
      const float sig = 1.0F / (1.0F + expf(-g));
      result *= g * sig;
    }
    // Preserve the standalone FP32 producer's rounding before conversion.
    asm volatile("" : "+v"(result));
    out[base + column] = __float2half_rn(result);
  }
}
}  // namespace

void LaunchBatchedSSMConvRecurrenceRowSplit(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf, void* q8_out,
    float* kq_scales, float* alpha_beta, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, QwenRecurrentStateStorage state_storage,
    void* fp16_out) {
  if (!IsDeltaNetRowSplitSupported(key_dim, val_dim)) {
    throw std::invalid_argument(
        "row-split Qwen recurrent state requires 128x128 matrices");
  }
  const std::size_t block_size = 256;
  const std::size_t grid_x = (qkv_size + block_size - 1) / block_size;
  const dim3 conv_grid(static_cast<unsigned int>(grid_x),
                       static_cast<unsigned int>(batch_size));
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  BatchedSSMConvKernel<<<conv_grid, static_cast<unsigned int>(block_size), 0,
                         stream>>>(qkv_in, conv_weights, conv_layer_state,
                                   conv_out, batch_size, qkv_size);

  const dim3 kq_grid(num_key_heads, static_cast<unsigned int>(batch_size));
  BatchedDeltaNetPrepKqKernel<<<kq_grid, 32, 0, stream>>>(
      conv_out, kq_scales, qkv_size, num_key_heads);

  const std::size_t ab_count = batch_size * num_heads;
  const std::size_t prep_count = std::max(ab_count, qkv_size);
  BatchedDeltaNetPrepAlphaBetaKernel<<<
      static_cast<unsigned int>((prep_count + 255) / 256), 256, 0, stream>>>(
      alpha_buf, beta_buf, ssm_a, ssm_dt, alpha_beta, ab_count, num_heads,
      qkv_in, conv_layer_state, batch_size, qkv_size);

  const std::size_t inner_size = static_cast<std::size_t>(num_heads) * val_dim;
  const auto batch_i = static_cast<int>(batch_size);
  const auto qkv_i = static_cast<int>(qkv_size);
  const auto inner_i = static_cast<int>(inner_size);
  const auto kh_i = static_cast<int>(num_key_heads);
  const auto h_i = static_cast<int>(num_heads);

  // Two register tiles cover the useful range. The 32-key / one-row tile keeps
  // the shortest reduction and wins while the live k/q working set still fits
  // the vector caches; past a few thousand tokens the blocks drift far enough
  // apart in the token index that halving the k/q traffic with a second row
  // matters more than the extra registers.
  if (batch_size <= kRowSplitSmallBatch) {
    const dim3 grid(val_dim / RowSplitRowsPerBlock<32, 1>(), num_heads);
    if (state_storage == QwenRecurrentStateStorage::kBf16) {
      BatchedDeltaNetRowSplitKernel<hip_bfloat16, 32, 1, false>
          <<<grid, 256, 0, stream>>>(conv_out, kq_scales, alpha_beta,
                                     static_cast<hip_bfloat16*>(deltanet_state),
                                     out_buf, layer_idx, batch_i, qkv_i,
                                     inner_i, kh_i, h_i);
    } else {
      BatchedDeltaNetRowSplitKernel<float, 32, 1, false>
          <<<grid, 256, 0, stream>>>(conv_out, kq_scales, alpha_beta,
                                     static_cast<float*>(deltanet_state),
                                     out_buf, layer_idx, batch_i, qkv_i,
                                     inner_i, kh_i, h_i);
    }
  } else {
    const dim3 grid(val_dim / RowSplitRowsPerBlock<32, 2>(), num_heads);
    if (state_storage == QwenRecurrentStateStorage::kBf16) {
      BatchedDeltaNetRowSplitKernel<hip_bfloat16, 32, 2, true>
          <<<grid, 256, 0, stream>>>(conv_out, kq_scales, alpha_beta,
                                     static_cast<hip_bfloat16*>(deltanet_state),
                                     out_buf, layer_idx, batch_i, qkv_i,
                                     inner_i, kh_i, h_i);
    } else {
      BatchedDeltaNetRowSplitKernel<float, 32, 2, true>
          <<<grid, 256, 0, stream>>>(conv_out, kq_scales, alpha_beta,
                                     static_cast<float*>(deltanet_state),
                                     out_buf, layer_idx, batch_i, qkv_i,
                                     inner_i, kh_i, h_i);
    }
  }

  // Emit the activation consumed by ssm_out directly. Other callers retain the
  // FP32 in-place epilogue.
  if (fp16_out != nullptr) {
    BatchedSSMPostNormGateFp16Kernel<<<(ab_count + 7) / 8, 256, 0, stream>>>(
        out_buf, ssm_norm, gate, static_cast<__half*>(fp16_out), ab_count);
  } else if (q8_out != nullptr) {
    LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
        out_buf, ssm_norm, gate, q8_out, batch_size, inner_size, num_heads,
        val_dim, stream);
  } else {
    const dim3 post_grid(num_heads, static_cast<unsigned int>(batch_size));
    BatchedSSMPostNormGateKernel<<<post_grid, 128, 0, stream>>>(
        out_buf, ssm_norm, gate, out_buf, batch_size, inner_size, num_heads,
        val_dim);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
