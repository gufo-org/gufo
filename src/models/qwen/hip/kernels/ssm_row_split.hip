// Row-split batched DeltaNet prefill recurrence (opt-c170-deltanet-rowsplit).
//
// The gated delta rule carries a [val_dim][key_dim] state per head and is
// serial in the token index, but it is *not* serial in the state row: row j of
// the state only ever needs its own row plus the per-token vectors k, q, v and
// the scalars alpha, beta, k.q. The previous kernel nevertheless pinned all 128
// rows of a head to one block so it could compute the k/q L2 norms and the
// post-RMSNorm inline, which cost four block-wide barriers and two
// single-thread cross-warp reductions per token, and left only 48 blocks
// (384 waves) of work on a 20-WGP part.
//
// Here everything uniform across rows is hoisted into two tiny prologue kernels
// and the recurrence itself becomes barrier-free, so the 128 rows spread over
// as many blocks as the per-lane register tile allows:
//
//   BatchedDeltaNetPrepKqKernel        per (token, key head): inv_k_norm,
//                                      q_scale, and the *unnormalized* k . q
//   BatchedDeltaNetPrepAlphaBetaKernel per (token, value head): alpha_h, beta_h
//   BatchedDeltaNetRowSplitKernel      the recurrence, one row group per lane
//
// Working on raw k and q instead of normalized copies is what makes the split
// cheap: every scale is uniform over the row, so it can be applied to the two
// reduced dot products and to the rank-1 update coefficient rather than to the
// 128-wide vectors.
//
//   u = inv_k_norm * (S' . k_raw)      p = q_scale * (S' . q_raw)
//   k . q = inv_k_norm * q_scale * (k_raw . q_raw)
//   S += (d * inv_k_norm) * k_raw
//
// This is the same arithmetic in a different order, so results differ from the
// previous kernel only by FP32 rounding (measured 5e-7 relative on the output
// and 3e-7 on the carried state at batch 512 through 8192).
//
// Measured against the previous kernel at 48 heads / 16 key heads / dk = dv =
// 128 (tools/bench/deltanet_bench.hip, one layer pass):
//
//   batch    before    after   speedup   cycles/token after
//     512    1.913    0.428     4.47x    2209
//    2048    7.401    1.991     3.72x    2567
//    4096   14.804    4.617     3.21x    2976
//    8192   29.686    9.362     3.17x    3017
//
// The floor is 1229 cycles/token: the state is touched four times per element
// per token (decay, two dot accumulations, rank-1 update) and gfx1151 retires
// 2560 FP32 lanes/cycle. A chunkwise-parallel reformulation would reach the
// matrix cores but needs 1.2x (chunk 16) to 1.9x (chunk 64) more MACs, which
// cancels the 2x BF16 WMMA rate, so this stays a VALU kernel.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/detail/recurrent_state.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

// Defined in kernels/ssm_recurrence.hip and hip/batched_ssm.hip respectively;
// the row-split route reuses both unchanged.
__global__ void BatchedSSMConvKernel(const float* __restrict__ qkv_in,
                                     const float* __restrict__ conv_weights,
                                     float* __restrict__ conv_state,
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
// tools/bench/deltanet_bench.hip.
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
    float* __restrict__ ab, std::size_t count, std::uint32_t num_heads) {
  const std::size_t i =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
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

void LaunchBatchedSSMConvRecurrenceRowSplit(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf, void* q8_out,
    float* kq_scales, float* alpha_beta, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, QwenRecurrentStateStorage state_storage) {
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
  BatchedDeltaNetPrepAlphaBetaKernel<<<
      static_cast<unsigned int>((ab_count + 255) / 256), 256, 0, stream>>>(
      alpha_buf, beta_buf, ssm_a, ssm_dt, alpha_beta, ab_count, num_heads);

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

  // opt-c174-ssm-epilogue-quant: when the gated row's only consumer is the Q8_0
  // ssm_out projection, the epilogue writes the quantized activation directly
  // and the FP32 store plus its read-back disappear. Otherwise the post-norm
  // gate runs in place, which is safe because it reads its input into registers
  // before the block reduction.
  if (q8_out != nullptr) {
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
