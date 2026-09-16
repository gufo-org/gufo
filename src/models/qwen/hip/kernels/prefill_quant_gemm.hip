#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

namespace gufo::hip {

template<bool StoreActivationSum, typename T>
__global__ void QuantizeActivationToQ8_1Kernel(const T* __restrict__ x,
                                               void* __restrict__ y,
                                               std::size_t batch,
                                               std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float val = static_cast<float>(x[elem_idx]);
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  StoreQ8ActLane<StoreActivationSum>(y, batch, num_blocks, tok, blk, lane_id, d,
                                     q);
}

// opt-c173-norm-quant: optional residual add, RMSNorm, and the tiled Q8_1
// quantization in one pass.
//
// Every projection reading the SSM pre-norm and the FFN norm is Q8_0, so the
// FP32 normed row and the BF16 staging copy the old chain produced were both
// dead: RMSNorm wrote 42 MB + 21 MB per layer at batch 2048 and the quantizer
// read the 21 MB straight back. Staging the row in LDS removes the rest of the
// traffic -- the reduction and the normalization share one global read, and
// with `residual` the preceding ResidualAdd pass disappears as well. Per norm
// at batch 2048 that is 221 MB of round trips down to 137 MB.
//
// The sum-of-squares reduction is the same loop and the same tree as
// BatchedRMSNormKernel, and folding the residual in registers gives the same
// FP32 value the separate add would have stored, so `inv_rms` and every normed
// value are bit-identical to the unfused chain. Only the quantization input
// changes, from BF16 to the FP32 value it was rounded from, which strictly
// narrows the quantization error -- the same trade `opt-c164-swiglu-quant` and
// `opt-c172-fp32-quant` made.
template<bool StoreActivationSum>
__global__ void BatchedFusedRMSNormQuantizeQ8_1Kernel(
    const float* __restrict__ x, const float* __restrict__ residual,
    const float* __restrict__ weight, float* __restrict__ sum_out,
    void* __restrict__ q8_out, std::size_t batch_size, std::size_t dim,
    float eps) {
  extern __shared__ float s_row[];
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  const float* res_row = (residual != nullptr) ? residual + (b * dim) : nullptr;
  float* sum_row = (sum_out != nullptr) ? sum_out + (b * dim) : nullptr;

  // Single pass over global memory: the row is staged in LDS so neither the
  // reduction nor the normalization re-reads it.
  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += block_size) {
    float val = in_row[i];
    if (res_row != nullptr) {
      val += res_row[i];
      if (sum_row != nullptr) {
        sum_row[i] = val;
      }
    }
    s_row[i] = val;
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += block_size) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    s_row[i] = (s_row[i] * inv_rms) * w;
  }
  __syncthreads();

  // One wave per 32-element quantization block, matching
  // QuantizeActivationToQ8_1Kernel's per-block scale and rounding exactly.
  const std::size_t num_blocks = dim / 32;
  const std::size_t lane_id = tid & 31u;
  const std::size_t wave_id = tid >> 5u;
  const std::size_t waves = block_size >> 5u;
  for (std::size_t blk = wave_id; blk < num_blocks; blk += waves) {
    const float val = s_row[(blk * 32) + lane_id];
    float max_abs = fabsf(val);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float d = max_abs / 127.0F;
    const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
    const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
    StoreQ8ActLane<StoreActivationSum>(q8_out, batch_size, num_blocks, b, blk,
                                       lane_id, d, q);
  }
}

// opt-c174-ssm-epilogue-quant: the SSM per-head post-RMSNorm + SiLU gate
// writing the tiled Q8_1 activation directly.
//
// The gated output's only consumer is the Q8_0 `ssm_out` projection, so the
// FP32 row was written and read straight back by the quantizer: 214 MB of round
// trips per layer at batch 2048 against 114 MB for one pass. The head's value
// dimension is a multiple of 32, so a block that owns one (head, token) already
// owns whole quantization blocks and needs no extra communication.
//
// The RMS reduction is the same warp butterfly, the same four-way cross-warp
// sum in the same order, and the same epsilon as BatchedSSMPostNormGateKernel,
// so every gated value is bit-identical to that kernel; only the quantization
// input changes from a re-read FP32 store to the register it came from, which
// is the same value.
template<bool StoreActivationSum>
__global__ void BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel(
    const float* __restrict__ raw_out, const float* __restrict__ ssm_norm,
    const float* __restrict__ gate, void* __restrict__ q8_out,
    std::size_t batch_size, std::size_t inner_size, std::uint32_t num_heads,
    std::uint32_t val_dim) {
  const std::size_t h = blockIdx.x;
  const std::size_t t = blockIdx.y;
  if (h >= num_heads || t >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const float* in_h = raw_out + (t * inner_size) + (h * val_dim);

  const float in_val = (tid < val_dim) ? in_h[tid] : 0.0F;
  float o_sq = in_val * in_val;
  for (int offset = 16; offset > 0; offset >>= 1) {
    o_sq += __shfl_xor(o_sq, offset);
  }

  __shared__ float s_warp_rms[4];
  if (lane_id == 0) {
    s_warp_rms[warp_id] = o_sq;
  }
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    const float total_o_sq =
        s_warp_rms[0] + s_warp_rms[1] + s_warp_rms[2] + s_warp_rms[3];
    const float mean_sq = total_o_sq / static_cast<float>(val_dim);
    s_inv_rms = rsqrtf(mean_sq + 1e-6F);
  }
  __syncthreads();

  if (tid >= val_dim) {
    return;
  }
  const float normed = in_val * s_inv_rms;
  const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
  float val = normed * w;
  if (gate != nullptr) {
    const float g = (gate + (t * inner_size))[(h * val_dim) + tid];
    const float sig = 1.0F / (1.0F + expf(-g));
    val *= (g * sig);
  }

  // Each wave of the block owns one whole 32-element quantization block.
  const std::size_t num_blocks = inner_size / 32;
  const std::size_t blk = ((h * val_dim) + tid) / 32;
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
  StoreQ8ActLane<StoreActivationSum>(q8_out, batch_size, num_blocks, t, blk,
                                     lane_id, d, q);
}

/// Zeroes the tail of the last token tile so a batch that is not a multiple of
/// 16 cannot feed uninitialized scales into the GEMM epilogue.
template<bool StoreActivationSum>
__global__ void ZeroQ8ActTailKernel(void* __restrict__ y, std::size_t batch,
                                    std::size_t num_blocks) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t tt = batch / kQ8ActTileTokens;
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t lanes = kQ8ActTileTokens - tail;
  if (idx >= num_blocks * lanes) {
    return;
  }
  const std::size_t kb = idx / lanes;
  const std::size_t tl = tail + (idx % lanes);
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, kb);
  reinterpret_cast<int32x4_t*>(tile + (tl * 16))[0] = {};
  reinterpret_cast<int32x4_t*>(tile + 256 + (tl * 16))[0] = {};
  *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
      0.0F;
  if constexpr (StoreActivationSum) {
    float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
    sums[(((tt * num_blocks) + kb) * kQ8ActTileTokens) + tl] = 0.0F;
  }
}

template<bool StoreActivationSum>
__global__ void BatchedFusedSwiGLUQuantizeQ8_1Kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    void* __restrict__ y, std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float g = gate[elem_idx];
  const float u = up[elem_idx];
  const float silu_g = g / (1.0F + expf(-g));
  const float val = silu_g * u;

  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const int8_t q = static_cast<int8_t>(roundf(val * id));

  StoreQ8ActLane<StoreActivationSum>(y, batch, num_blocks, tok, blk, lane_id, d,
                                     q);
}

// opt-c192-swiglu-epilogue: SwiGLU + tiled Q8_1 quantization straight out of a
// blocked GEMM's accumulator.
//
// The FFN up projection's only consumer is SwiGLU, whose only consumer is the
// Q8_1 activation ffn_down reads. Writing the FP32 intermediate therefore costs
// 143 MB of stores per layer at batch 2048 and another 143 MB of loads in a
// separate bandwidth-bound pass, to reproduce a value the GEMM already had in a
// register.
//
// Two shape facts make the fused form possible without any cross-wave traffic:
//
//   * A wave's two row subtiles are rows [32*wave_row, 32*wave_row+32) of the
//     macro tile, which is exactly one 32-element quantization block of the
//     *next* GEMM's K dimension. So one wave owns every value a block scale
//     depends on.
//   * Staging both subtiles through the wave's LDS slot puts one token's
//     sixteen rows in sixteen consecutive lanes, so the gate load is one
//     64-byte segment, the code stores are one 16-byte segment, and the max
//     reduction is four `__shfl_xor` inside a half-wave.
//
// Bit-identical to LaunchBatchedFusedSwiGLUQuantizeQ8_1 over the same inputs:
// `acc` is the float the FP32 store would have round-tripped exactly, `gate`
// comes from the same buffer, and `fmaxf` and integer addition are
// order-independent so the changed reduction shape cannot move a result.
template<int TokTiles, bool StoreActivationSum>
__device__ __forceinline__ void BlockedSwiGluQuantEpilogue(
    const float (*acc_lo)[8], const float (*acc_hi)[8], float* scratch_lo,
    float* scratch_hi, const float* __restrict__ gate,
    void* __restrict__ q8_out, std::size_t batch, std::size_t m,
    std::size_t r_base, std::size_t t_base, int lane_id) {
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const std::size_t num_blocks = m / 32;
  const std::size_t blk = r_base / 32;
#pragma unroll
  for (int j = 0; j < TokTiles; ++j) {
#pragma unroll
    for (int l = 0; l < 8; ++l) {
      scratch_lo[(sub_lane * 16) + (2 * l) + half_id] = acc_lo[j][l];
      scratch_hi[(sub_lane * 16) + (2 * l) + half_id] = acc_hi[j][l];
    }
    __builtin_amdgcn_wave_barrier();
    const std::size_t t0 = t_base + static_cast<std::size_t>(j * 16);
    std::int8_t* tile =
        Q8ActTile(q8_out, num_blocks, t0 / kQ8ActTileTokens, blk);
    // Every gate load for this tile is issued before any of the dependent math.
    // This epilogue runs after the K loop, so unlike the GEMM body there is no
    // other work left to hide memory latency behind: one load per step would
    // serialize sixteen round trips per tile, which measured as most of the
    // epilogue's cost. Sixteen floats per lane is sixteen registers, and the
    // block's whole gate panel (64 KiB) is far too large to stage instead.
    // A token past the batch contributes nothing and must not read `gate` out
    // of bounds; its tile slot is zeroed by ZeroQ8ActTailKernel, exactly as on
    // the separate-pass route.
    float g_lo[8];
    float g_hi[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
      const std::size_t row_base =
          (tok * m) + r_base + static_cast<std::size_t>(flat & 15);
      const bool live = tok < batch;
      g_lo[s] = live ? gate[row_base] : 0.0F;
      g_hi[s] = live ? gate[row_base + 16] : 0.0F;
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int flat = (s * 32) + lane_id;
      const int tl = flat >> 4;
      const int row = flat & 15;
      const std::size_t tok = t0 + static_cast<std::size_t>(tl);
      const bool live = tok < batch;
      float v_lo = 0.0F;
      float v_hi = 0.0F;
      if (live) {
        v_lo = (g_lo[s] / (1.0F + expf(-g_lo[s]))) * scratch_lo[flat];
        v_hi = (g_hi[s] / (1.0F + expf(-g_hi[s]))) * scratch_hi[flat];
      }
      float max_abs = fmaxf(fabsf(v_lo), fabsf(v_hi));
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off, 16));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q_lo = static_cast<std::int8_t>(roundf(v_lo * id));
      const auto q_hi = static_cast<std::int8_t>(roundf(v_hi * id));
      if (live) {
        tile[(tl * 16) + row] = q_lo;
        tile[256 + (tl * 16) + row] = q_hi;
        if (row == 0) {
          *reinterpret_cast<float*>(
              tile + kQ8ActScaleOffset +
              (static_cast<std::size_t>(tl) * sizeof(float))) = d;
        }
      }
      if constexpr (StoreActivationSum) {
        int qsum = static_cast<int>(q_lo) + static_cast<int>(q_hi);
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          qsum += __shfl_xor(qsum, off, 16);
        }
        if (live && row == 0) {
          float* sums = Q8ActSumSidecar(q8_out, batch, m);
          sums[(((t0 / kQ8ActTileTokens * num_blocks) + blk) *
                kQ8ActTileTokens) +
               static_cast<std::size_t>(tl)] = d * static_cast<float>(qsum);
        }
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
}

template<bool SubOpsel>
__device__ __forceinline__ int32x8_t Wmma_16x16x16_iu8(int32x4_t a, int32x4_t b,
                                                       int32x8_t c) {
#if defined(__has_builtin) && \
    __has_builtin(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32)
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c,
                                                    SubOpsel);
#else
  return c;
#endif
}

// Two-dimensionally blocked W8A8 WMMA GEMM (opt-c163-blocked-w8a8).
//
// Traffic per multiply-accumulate is 34/(32*BN) bytes of weight and
// 36/(32*BM) bytes of activation. The previous 16-row macro tile therefore
// re-read the whole activation panel m/16 times (1088x, ~12.8 GB for the FFN
// shape at batch 2048), which dominated everything else. Blocking both
// dimensions at 128 cuts that to m/128 and lifts the kernel from 35% to 58% of
// the measured 55.07 TOPS WMMA int8 ceiling.
//
//   block = BM rows x BN tokens, BK 32-element K blocks per LDS stage
//   waves = WM row groups x WN token groups (WM*WN == 8)
//   wave  = (BM/WM) rows x (BN/WN) tokens
//
// Weights are read from the production row-major block_q8_0 layout, so no
// second copy of the model is needed; activations arrive in the tiled layout
// the quantize kernel emits. Accumulation order over K blocks is unchanged from
// the 16-row kernel, so results are bit-identical.
template<int BM, int BN, int BK, int WM, int WN, bool FusedSwiGluQuant = false,
         bool StoreActivationSum = false>
__launch_bounds__(256) __global__ void W8A8BlockedWmmaGEMMKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const float* __restrict__ swiglu_gate = nullptr,
    void* __restrict__ q8_out = nullptr) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage. Fetching one stage ahead lets the
  // latency of the scattered 34-byte block_q8_0 reads overlap the WMMA work
  // instead of gating the LDS fill.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  // opt-c179-gemm-addressing. An ISA dump of this kernel put only 64 of its
  // 2752 instructions in WMMA: ~309 were exec-mask manipulation
  // (`s_and_saveexec_b32` / `s_or_b32` / `s_and_b32`) from the per-thread
  // bounds guards below, and ~265 were 64-bit address arithmetic
  // (`v_mad_u64_u32`, `v_add_co_u32`) from recomputing `(r * num_blocks) + kb`
  // per stage. Both go away here:
  //
  //   * Every 64-bit base is hoisted out of the K loop, so a stage advance is a
  //     32-bit add.
  //   * Out-of-range rows and K blocks are CLAMPED rather than branched, and
  //     their *scale* is zeroed with a multiply. The epilogue then multiplies
  //     by dw = 0, so an out-of-range row contributes exactly zero with no
  //     divergence -- bit-identical to zero-filling the operands, because
  //     0 * dx * float(c) is 0 for any finite c.
  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = reinterpret_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // The token-tile operands are read one tile at a time, inside the j loop,
      // rather than all kWaveTokTiles of them up front. Hoisting them held
      // 4 x (2 int32x4 + 1 float) = 36 extra VGPRs live across the whole WMMA
      // block, which is what pushed a BK=2 stage into 500 bytes/lane of
      // scratch.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = Wmma_16x16x16_iu8<true>(a0[i], b0, c);
          c = Wmma_16x16x16_iu8<true>(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
    }
    __syncthreads();
  }

  // Transpose each 16x16 result tile through LDS so the global stores cover 16
  // consecutive rows of one token instead of 16 separate cache lines.
  __syncthreads();
  if constexpr (FusedSwiGluQuant) {
    // The fused epilogue owns two 256-float slots per wave and never touches
    // `y`, so it takes the whole of both staging arrays. The conditions are
    // written against the flag because clang checks a static_assert in a
    // discarded `if constexpr` branch once the enclosing template is
    // instantiated, and the narrow tiles do not satisfy them.
    static_assert(!FusedSwiGluQuant || kWaveRowTiles == 2,
                  "the fused SwiGLU epilogue needs a wave to own one whole "
                  "32-element quantization block");
    static_assert(
        !FusedSwiGluQuant || (sizeof(s_a) >= WM * WN * 256 * sizeof(float) &&
                              sizeof(s_b) >= WM * WN * 256 * sizeof(float)),
        "the fused epilogue stages two output tiles per wave");
    BlockedSwiGluQuantEpilogue<kWaveTokTiles, StoreActivationSum>(
        acc[0], acc[1],
        reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256),
        reinterpret_cast<float*>(&s_b[0][0][0]) + (wave_id * 256), swiglu_gate,
        q8_out, batch, m, r_block + static_cast<std::size_t>(wave_row * 32),
        t_block + static_cast<std::size_t>(wave_tok * kWaveTokTiles * 16),
        lane_id);
    return;
  }
  float* tile_scratch =
      reinterpret_cast<float*>(&s_a[0][0][0]) + (wave_id * 256);
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[i][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // One uniform branch per 16x16 tile instead of eight per-lane predicates,
      // which is where most of the kernel's exec-mask traffic came from.
      if (t0 + 16 <= batch && r0 + 16 <= m) {
        const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
            static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
        }
      } else {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int flat = (s * 32) + lane_id;
          const std::size_t tok = t0 + static_cast<std::size_t>(flat >> 4);
          const std::size_t r = r0 + static_cast<std::size_t>(flat & 15);
          if (tok < batch && r < m) {
            y[(tok * m) + r] = tile_scratch[flat];
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

template<std::uint32_t WavesPerBlock, std::uint32_t TokensPerWave>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void TiledBatchedQ8_0GEMMKernel(const void* __restrict__ w,
                                    const hip_bfloat16* __restrict__ x,
                                    float* __restrict__ y, std::size_t batch,
                                    std::size_t m, std::size_t k) {
  constexpr std::size_t kRowsPerWave = 4;
  static_assert(TokensPerWave == 2 || TokensPerWave == 4 || TokensPerWave == 8);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;

  const std::size_t row_base =
      (blockIdx.x * WavesPerBlock + warp_id) * kRowsPerWave;
  const std::size_t tok_base = blockIdx.y * TokensPerWave;
  if (row_base >= m || tok_base >= batch) {
    return;
  }

  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* w_blocks = static_cast<const Q8_0Block*>(w);

  const auto* row0 = w_blocks + ((row_base + 0) * num_blocks);
  const auto* row1 =
      (row_base + 1 < m) ? (w_blocks + ((row_base + 1) * num_blocks)) : nullptr;
  const auto* row2 =
      (row_base + 2 < m) ? (w_blocks + ((row_base + 2) * num_blocks)) : nullptr;
  const auto* row3 =
      (row_base + 3 < m) ? (w_blocks + ((row_base + 3) * num_blocks)) : nullptr;

  const std::size_t group_id = lane_id >> 3u;       // 0..3
  const std::size_t lane_in_group = lane_id & 7u;   // 0..7
  const std::size_t elem_off = lane_in_group * 4u;  // 0, 4, 8..28

  float acc[4][TokensPerWave] = {};

  const std::size_t num_steps = (num_blocks + 3) / 4;
  for (std::size_t step = 0; step < num_steps; ++step) {
    const std::size_t b = (step * 4) + group_id;

    if (b < num_blocks) {
      // Load four bfloat16 activation values for each token column.
      hip_bfloat16 xv[TokensPerWave][4];
#pragma unroll
      for (std::size_t t = 0; t < TokensPerWave; ++t) {
        if (tok_base + t < batch) {
          *reinterpret_cast<uint64_t*>(&xv[t][0]) =
              *reinterpret_cast<const uint64_t*>(
                  &x[((tok_base + t) * k) + (b * kQ8_0BlockSize) + elem_off]);
        }
      }

      // Process row 0
      {
        const Q8_0Block& b0 = row0[b];
        const float d0 = __half2float(b0.d);
        const auto* qs0 = b0.qs + elem_off;
        float dot0[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d0 * static_cast<float>(qs0[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot0[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[0][t] += dot0[t];
        }
      }

      // Process row 1
      if (row1 != nullptr) {
        const Q8_0Block& b1 = row1[b];
        const float d1 = __half2float(b1.d);
        const auto* qs1 = b1.qs + elem_off;
        float dot1[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d1 * static_cast<float>(qs1[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot1[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[1][t] += dot1[t];
        }
      }

      // Process row 2
      if (row2 != nullptr) {
        const Q8_0Block& b2 = row2[b];
        const float d2 = __half2float(b2.d);
        const auto* qs2 = b2.qs + elem_off;
        float dot2[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d2 * static_cast<float>(qs2[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot2[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[2][t] += dot2[t];
        }
      }

      // Process row 3
      if (row3 != nullptr) {
        const Q8_0Block& b3 = row3[b];
        const float d3 = __half2float(b3.d);
        const auto* qs3 = b3.qs + elem_off;
        float dot3[TokensPerWave] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float q = d3 * static_cast<float>(qs3[i]);
#pragma unroll
          for (std::size_t t = 0; t < TokensPerWave; ++t) {
            dot3[t] += q * static_cast<float>(xv[t][i]);
          }
        }
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          acc[3][t] += dot3[t];
        }
      }
    }
  }

  // Reduce within group (8 lanes per block) and across groups (4 groups in
  // wave)
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
#pragma unroll
      for (std::size_t t = 0; t < TokensPerWave; ++t) {
        acc[r][t] += __shfl_xor(acc[r][t], off);
      }
    }
  }

  if (lane_id == 0) {
#pragma unroll
    for (std::size_t r = 0; r < 4; ++r) {
      const std::size_t row = row_base + r;
      if (row < m) {
#pragma unroll
        for (std::size_t t = 0; t < TokensPerWave; ++t) {
          const std::size_t tok = tok_base + t;
          if (tok < batch) {
            y[(tok * m) + row] = acc[r][t];
          }
        }
      }
    }
  }
}

template<typename Input>
__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const Input* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot =
      QuantWarpBlockDot<Input>(type, w, row, x + (b * k), k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

template<std::uint32_t WavesPerBlock, std::size_t TokensPerWave>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void SmallBatchQ8KFp32GEMMKernel(const void* __restrict__ w,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, std::size_t batch,
                                     std::size_t m, std::size_t k) {
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + warp_id;
  if (row >= m) {
    return;
  }

  const std::size_t num_blocks = k / kQ8KBlockSize;
  const auto* row_blocks = static_cast<const Q8KBlock*>(w) + (row * num_blocks);
  float sums[TokensPerWave] = {};

  for (std::size_t block = 0; block < num_blocks; ++block) {
    const Q8KBlock& weights = row_blocks[block];
    float weight_values[8];
#pragma unroll
    for (std::size_t element = 0; element < 8; ++element) {
      const std::size_t index = lane + (32 * element);
      weight_values[element] = static_cast<float>(weights.qs[index]);
    }

#pragma unroll
    for (std::size_t token = 0; token < TokensPerWave; ++token) {
      float dot = 0.0F;
      if (token < batch) {
        const float* input = x + (token * k) + (block * kQ8KBlockSize);
#pragma unroll
        for (std::size_t element = 0; element < 8; ++element) {
          const std::size_t index = lane + (32 * element);
          dot += weight_values[element] * input[index];
        }
      }
      sums[token] += weights.d * dot;
    }
  }

#pragma unroll
  for (std::size_t token = 0; token < TokensPerWave; ++token) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sums[token] += __shfl_xor(sums[token], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (std::size_t token = 0; token < TokensPerWave; ++token) {
      if (token < batch) {
        y[(token * m) + row] = sums[token];
      }
    }
  }
}

namespace {

template<bool StoreActivationSum>
void LaunchZeroQ8ActTail(void* q8_1_out, std::size_t batch, std::size_t k,
                         hipStream_t stream) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t work = num_blocks * (kQ8ActTileTokens - tail);
  ZeroQ8ActTailKernel<StoreActivationSum>
      <<<static_cast<unsigned int>((work + 255) / 256), 256, 0, stream>>>(
          q8_1_out, batch, num_blocks);
}

}  // namespace

void LaunchQuantizeActivationQ8_1(const void* bf16_x, void* q8_1_out,
                                  std::size_t batch, std::size_t k,
                                  hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          reinterpret_cast<const hip_bfloat16*>(bf16_x), q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

bool IsFusedSSMEpilogueQuantizeQ8_1Supported(std::uint32_t val_dim,
                                             std::size_t inner_size) noexcept {
  // One block owns one (head, token) and quantizes its own whole blocks, so the
  // value dimension has to be a multiple of 32 and fit the 128-thread block the
  // RMS reduction is written for.
  return val_dim != 0 && val_dim <= 128 && (val_dim % 32) == 0 &&
         (inner_size % 32) == 0;
}

void LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1(
    const float* raw_out, const float* ssm_norm, const float* gate,
    void* q8_1_out, std::size_t batch_size, std::size_t inner_size,
    std::uint32_t num_heads, std::uint32_t val_dim, hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || val_dim == 0) {
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel<true>
      <<<grid, 128, 0, stream>>>(raw_out, ssm_norm, gate, q8_1_out, batch_size,
                                 inner_size, num_heads, val_dim);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, inner_size, stream);
}

bool IsFusedRMSNormQuantizeQ8_1Supported(std::size_t dim) noexcept {
  // The normed row is staged in LDS, so the row has to fit the 32 KiB a block
  // can hold alongside the reduction scratch, and the quantizer needs whole
  // 32-element blocks.
  return dim != 0 && (dim % 32) == 0 && (dim * sizeof(float)) <= 32768;
}

void LaunchBatchedFusedRMSNormQuantizeQ8_1(
    const float* x, const float* residual, const float* weight, float* sum_out,
    void* q8_1_out, std::size_t batch_size, std::size_t dim, float eps,
    hipStream_t stream) {
  if (batch_size == 0 || dim == 0) {
    return;
  }
  const unsigned int lds_bytes = static_cast<unsigned int>(dim * sizeof(float));
  BatchedFusedRMSNormQuantizeQ8_1Kernel<true>
      <<<static_cast<unsigned int>(batch_size), 256, lds_bytes, stream>>>(
          x, residual, weight, sum_out, q8_1_out, batch_size, dim, eps);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, dim, stream);
}

void LaunchQuantizeActivationQ8_1FromFp32(const float* fp32_x, void* q8_1_out,
                                          std::size_t batch, std::size_t k,
                                          hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  QuantizeActivationToQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          fp32_x, q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

void LaunchBatchedFusedSwiGLUQuantizeQ8_1(const float* gate, const float* up,
                                          void* q8_1_out, std::size_t batch,
                                          std::size_t k, hipStream_t stream) {
  if (batch == 0 || k == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t total_x_blocks = batch * num_blocks;
  constexpr unsigned int quant_threads = 256;
  constexpr unsigned int warps_per_tb = quant_threads / 32;
  const std::size_t quant_grid =
      (total_x_blocks + warps_per_tb - 1) / warps_per_tb;
  BatchedFusedSwiGLUQuantizeQ8_1Kernel<true>
      <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
          gate, up, q8_1_out, batch, k);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
}

std::size_t QuantizedActivationBytes(std::size_t batch, std::size_t k) {
  return Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k);
}

// opt-q4kxl: two-level dispatch for SmallBatchKQuantExactFp32GEMMKernel --
// exact batch width (a padded width would stage and multiply activations no
// output consumes) crossed with weight format.
// Sixteen warps per block, not eight: the staged activation tile is shared by
// every warp in the block, so doubling the warps halves the staging cost per
// output row. This is the same shape opt-c18x-q8-vec-smallbatch settled on for
// the Q8_0 vec kernel.
constexpr std::size_t kKQuantSmallBatchWarps = 16;
constexpr std::size_t kKQuantSmallBatchRowsPerWave = 2;

static_assert(FitsSmallBatch32BitIndices(8, 17408, 5120, 3520));
static_assert(FitsSmallBatch32BitIndices(8, 248320, 5120, 4200));
static_assert(FitsSmallBatch32BitIndices(2, 17408, 5120, 3520));
static_assert(FitsSmallBatch32BitIndices(2, 248320, 5120, 4200));
static_assert(!FitsSmallBatch32BitIndices(2, 1, 1ULL << 31, 1));
static_assert(!FitsSmallBatch32BitIndices(2, 1ULL << 31, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1ULL << 29, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 1ULL << 29, 1, 1));
static_assert(!FitsSmallBatch32BitIndices(8, 65536, 131072, 90112));
static_assert(!FitsSmallBatch32BitIndices(8, 1, 1, 0));

template<std::size_t RowsPerWave, core::GgmlType WType>
void LaunchKQuantSmallBatchTyped(const void* w, const float* fp32_x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, dim3 grid, dim3 block,
                                 hipStream_t stream) {
  const auto launch = [&]<std::size_t Batch>() {
    // Adjacent gate/up tensors use the same kernel with twice as many rows.
    const bool ffn_up = (m == 17408 || (Batch >= 2 && m == 34816)) && k == 5120;
    if constexpr (Batch >= 3 && (WType == core::GgmlType::kIQ4_NL ||
                                 WType == core::GgmlType::kIQ3_S ||
                                 WType == core::GgmlType::kQ6_K ||
                                 WType == core::GgmlType::kIQ4_XS)) {
      const bool ffn_down = m == 5120 && k == 17408;
      // The remaining mixed formats benefit from four-row activation reuse
      // at these shapes and widths. Keep narrower batches where reuse loses.
      const bool reuse_four_rows =
          (WType == core::GgmlType::kIQ4_NL &&
           ((m == 17408 && k == 5120) || (Batch >= 5 && ffn_down))) ||
          (WType == core::GgmlType::kIQ3_S && (Batch == 3 || Batch >= 5) &&
           ffn_down) ||
          (WType == core::GgmlType::kQ6_K &&
           ((Batch >= 6 && m == 5120 && k == 6144) ||
            (Batch == 8 && m == 10240 && k == 5120))) ||
          (WType == core::GgmlType::kIQ4_XS &&
           ((m == 5120 && k == 6144) ||
            (Batch == 8 && k == 5120 && (m == 10240 || m == 12288))));
      if (reuse_four_rows) {
        SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, 1, 4, 1, true>
            <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    if constexpr (WType == core::GgmlType::kIQ4_XS &&
                  (Batch == 2 || Batch == 3 || Batch == 4 || Batch == 5 ||
                   Batch == 7)) {
      const bool up = ffn_up;
      const bool down = m == 5120 && k == 17408;
      if (up || (Batch != 4 && down)) {
        const auto launch_rows = [&]<std::size_t Tiles>() {
          SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, Tiles, 4, 1,
                                              true>
              <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
        };
        if constexpr (Batch == 3) {
          if (up) {
            launch_rows.template operator()<1>();
          } else {
            launch_rows.template operator()<2>();
          }
        } else {
          launch_rows.template operator()<2>();
        }
        return;
      }
    }
    if constexpr (WType == core::GgmlType::kIQ4_XS && Batch == 8) {
      if ((ffn_up) || (m == 5120 && k == 17408)) {
        SmallBatchKQuantExactFp32GEMMKernel<8, Batch, 4, WType, 1, 4, 1, true>
            <<<(m + 31) / 32, 256, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    if constexpr (Batch >= 2 && (WType == core::GgmlType::kQ4_K ||
                                 WType == core::GgmlType::kQ5_K)) {
      if (FitsSmallBatch32BitIndices(Batch, m, k,
                                     quant::QuantizedRowBytes(WType, k))) {
        if constexpr (WType == core::GgmlType::kQ4_K) {
          const bool up = ffn_up;
          const bool down = m == 5120 && k == 17408;
          // Four rows share two staged tiles. Widths five and six benefit
          // only for the longer down projection.
          if (down || (up && Batch != 5 && Batch != 6)) {
            SmallBatchKQuantExactFp32GEMMKernel<16, Batch, 4, WType, 2, 4, 1,
                                                true>
                <<<(m + 63) / 64, 512, 0, stream>>>(w, fp32_x, y, m, k);
            return;
          }
        }
        if constexpr (WType == core::GgmlType::kQ5_K && Batch <= 6) {
          const auto launch_rows = [&]<std::size_t Tiles,
                                       std::size_t TokensPerStep = Batch,
                                       std::uint32_t Waves = 16>() {
            SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, WType, Tiles,
                                                4, 1, true, TokensPerStep>
                <<<(m + Waves * 4 - 1) / (Waves * 4), Waves * 32, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          // Short adaptive blocks frequently use these medium projections.
          // Reuse four rows only at widths/shapes with measured gains.
          if constexpr (Batch == 5) {
            if (m == 6144 && k == 5120) {
              launch_rows.template operator()<1, Batch, 8>();
              return;
            }
            if (m == 5120 && k == 6144) {
              launch_rows.template operator()<1>();
              return;
            }
          }
          if constexpr (Batch == 5 || Batch == 6) {
            if ((m == 5120 && k == 6144) ||
                (k == 5120 && (m == 6144 || m == 10240 || m == 12288))) {
              launch_rows.template operator()<2>();
              return;
            }
          }
          // Reuse four output rows for the FFNs and selected smaller
          // projections, including the shortest speculative blocks.
          if constexpr (Batch == 2) {
            if ((m == 5120 && k == 6144) || (m == 10240 && k == 5120)) {
              launch_rows.template operator()<2>();
              return;
            }
          }
          if constexpr (Batch == 4) {
            if (m == 5120 && k == 6144) {
              launch_rows.template operator()<1, 2>();
              return;
            }
            if (m == 6144 && k == 5120) {
              launch_rows.template operator()<1>();
              return;
            }
          }
          if (m == 5120 && k == 17408) {
            if constexpr (Batch == 4) {
              launch_rows.template operator()<1, 2>();
            } else if constexpr (Batch == 5) {
              launch_rows.template operator()<1>();
            } else {
              launch_rows.template operator()<2>();
            }
            return;
          }
          if constexpr (Batch == 2 || Batch == 3 || Batch == 5) {
            if (ffn_up) {
              launch_rows.template operator()<2>();
              return;
            }
          }
        }
        if constexpr (WType == core::GgmlType::kQ5_K && Batch >= 7) {
          const auto launch_rows = [&]<std::uint32_t Waves,
                                       std::size_t TokensPerStep = Batch>() {
            SmallBatchKQuantExactFp32GEMMKernel<Waves, Batch, 4, WType, 1, 4, 1,
                                                true, TokensPerStep>
                <<<(m + Waves * 4 - 1) / (Waves * 4), Waves * 32, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          // These batch-8 projections finish one token's dots at a time.
          // The long FFN down projection shares staging across more waves.
          if constexpr (Batch == 8) {
            if (m == 5120 && k == 17408) {
              launch_rows.template operator()<16, 1>();
              return;
            }
            if ((m == 5120 && k == 6144) ||
                (k == 5120 && (m == 1024 || m == 6144 || m == 10240 ||
                               m == 12288 || m == 17408 || m == 34816))) {
              launch_rows.template operator()<8, 1>();
              return;
            }
          }
          launch_rows.template operator()<8>();
          return;
        } else {
          constexpr bool kDoubleStage =
              (WType == core::GgmlType::kQ4_K && Batch >= 5) ||
              (WType == core::GgmlType::kQ5_K && Batch >= 5 && Batch <= 6);
          if constexpr (kDoubleStage) {
            if ((ffn_up) || (m == 5120 && k == 17408)) {
              SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                                  RowsPerWave, WType, 2, 12, 1,
                                                  true>
                  <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
              return;
            }
          }
          SmallBatchKQuantExactFp32GEMMKernel<
              kKQuantSmallBatchWarps, Batch, RowsPerWave, WType, 1, 12, 1, true>
              <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        }
      }
    }
    // At the large FFN shapes, two tiles halve staging barriers. Select only
    // the widths/formats that benefit: notably Q4 batch 7 also avoids spills.
    constexpr bool kDoubleStage =
        (WType == core::GgmlType::kQ4_K && Batch >= 5) ||
        (WType == core::GgmlType::kQ5_K && (Batch == 6 || Batch == 7)) ||
        (WType == core::GgmlType::kIQ4_XS &&
         (Batch == 5 || Batch == 6 || Batch == 8));
    if constexpr (kDoubleStage) {
      if ((ffn_up) || (m == 5120 && k == 17408)) {
        SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                            RowsPerWave, WType, 2>
            <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
    }
    SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                        RowsPerWave, WType>
        <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);
  };
#define GUFO_KQUANT_SMALL_BATCH(N)   \
  case N:                            \
    launch.template operator()<N>(); \
    return;
  switch (batch) {
    GUFO_KQUANT_SMALL_BATCH(1)
    GUFO_KQUANT_SMALL_BATCH(2)
    GUFO_KQUANT_SMALL_BATCH(3)
    GUFO_KQUANT_SMALL_BATCH(4)
    GUFO_KQUANT_SMALL_BATCH(5)
    GUFO_KQUANT_SMALL_BATCH(6)
    GUFO_KQUANT_SMALL_BATCH(7)
    GUFO_KQUANT_SMALL_BATCH(8)
    default:
      return;
  }
#undef GUFO_KQUANT_SMALL_BATCH
}

void LaunchKQuantSmallBatchByType(core::GgmlType type, const void* w,
                                  const float* fp32_x, float* y,
                                  std::size_t batch, std::size_t m,
                                  std::size_t k, dim3 grid, dim3 block,
                                  hipStream_t stream) {
  if (detail::TryLaunchKQuantSmallBatchWave64(type, w, fp32_x, y, batch, m, k,
                                              stream)) {
    return;
  }
  constexpr std::size_t RowsPerWave = 3;
  switch (type) {
    case core::GgmlType::kQ4_K:
      if (batch == 8 && k == 5120 && m == 1280) {
        SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, 4,
                                            RowsPerWave, core::GgmlType::kQ4_K,
                                            1, 12, 2, true>
            <<<((m + 47) / 48) * 2, block, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ4_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ5_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kQ6_K: {
      // The large FFN projections reuse activations across three output rows.
      // Smaller batches keep two rows to avoid paying for unused capacity.
      if (batch >= 5 &&
          ((m == 17408 && k == 5120) || (m == 5120 && k == 17408))) {
        const auto launch_ffn = [&]<std::size_t Batch>() {
          SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch, 3,
                                              core::GgmlType::kQ6_K, 2, 1, 1,
                                              true>
              <<<(m + 47) / 48, block, 0, stream>>>(w, fp32_x, y, m, k);
        };
        switch (batch) {
          case 5:
            launch_ffn.template operator()<5>();
            return;
          case 6:
            launch_ffn.template operator()<6>();
            return;
          case 7:
            launch_ffn.template operator()<7>();
            return;
          case 8:
            launch_ffn.template operator()<8>();
            return;
          default:
            break;
        }
      }
      // The vocabulary projection reuses activations across three output
      // rows and two consecutive tiles, retaining the scalar sum order.
      // The Qwen27B head also benefits at shorter verification widths.
      if (((batch >= 2 && batch <= 6) && m == 248320 && k == 5120) ||
          ((batch == 7 || batch == 8) && m >= 65536)) {
        constexpr std::size_t kHeadRows = kKQuantSmallBatchWarps * 3;
        const auto launch_head = [&]<std::size_t Batch>() {
          const auto launch_indices = [&]<bool Narrow>() {
            SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, Batch,
                                                3, core::GgmlType::kQ6_K, 2, 1,
                                                1, Narrow>
                <<<(m + kHeadRows - 1) / kHeadRows, block, 0, stream>>>(
                    w, fp32_x, y, m, k);
          };
          if constexpr (Batch <= 6) {
            // These widths enter only for the bounded Qwen27B head above.
            launch_indices.template operator()<true>();
          } else if (FitsSmallBatch32BitIndices(
                         Batch, m, k,
                         quant::QuantizedRowBytes(core::GgmlType::kQ6_K, k))) {
            launch_indices.template operator()<true>();
          } else {
            launch_indices.template operator()<false>();
          }
        };
        switch (batch) {
          case 2:
            launch_head.template operator()<2>();
            break;
          case 3:
            launch_head.template operator()<3>();
            break;
          case 4:
            launch_head.template operator()<4>();
            break;
          case 5:
            launch_head.template operator()<5>();
            break;
          case 6:
            launch_head.template operator()<6>();
            break;
          case 7:
            launch_head.template operator()<7>();
            break;
          case 8:
            launch_head.template operator()<8>();
            break;
          default:
            break;
        }
        return;
      }
      // Other shapes retain two rows to bound register pressure.
      constexpr std::size_t kRowsPerBlock =
          kKQuantSmallBatchWarps * kKQuantSmallBatchRowsPerWave;
      const dim3 q6_grid(
          static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
      LaunchKQuantSmallBatchTyped<kKQuantSmallBatchRowsPerWave,
                                  core::GgmlType::kQ6_K>(
          w, fp32_x, y, batch, m, k, q6_grid, block, stream);
    }
      return;
    case core::GgmlType::kQ3_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ3_K>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ4_NL:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_NL>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ4_XS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_XS>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    case core::GgmlType::kIQ3_S:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ3_S>(
          w, fp32_x, y, batch, m, k, grid, block, stream);
      return;
    default:
      return;
  }
}

// opt-q4kxl: tile-configuration dispatch for WKQuantA8BlockedWmmaGEMMKernel.
// The configurations mirror the Q8_0 launcher exactly so the two paths share
// the same occupancy tuning.
template<core::GgmlType WType>
void LaunchKQuantWmmaGEMMTyped(const void* w, const void* q8_1_x, float* y,
                               std::size_t batch, std::size_t m, std::size_t k,
                               hipStream_t stream) {
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1, WType>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // WM=4/WN=2 (2 row subtiles and 4 token subtiles per wave). The mirror
    // shape WM=2/WN=4 halves the per-token activation sums the offset term
    // needs, but measured far worse -- pp2048 464 -> 372 -- because four row
    // subtiles per wave hold twice the A-fragment and scale state.
    WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    return;
  }
  constexpr int kBM = 128;
  constexpr int kBN = 64;
  const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                  static_cast<unsigned int>((m + kBM - 1) / kBM));
  WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1, WType>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

void LaunchKQuantWmmaGEMM(core::GgmlType type, const void* w,
                          const void* q8_1_x, float* y, std::size_t batch,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ4_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ5_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ6_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ6_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kQ3_K:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kQ3_K>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case core::GgmlType::kIQ4_NL:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ4_NL>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
    case core::GgmlType::kIQ4_XS:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ4_XS>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
    case core::GgmlType::kIQ3_S:
      LaunchKQuantWmmaGEMMTyped<core::GgmlType::kIQ3_S>(w, q8_1_x, y, batch, m,
                                                        k, stream);
      return;
    default:
      return;
  }
}

void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (detail::TryLaunchQuantPrefillWave64(type, w, q8_1_x, y, batch, m, k,
                                          stream)) {
    return;
  }
  // opt-q4kxl: every K-quant/IQ format in the UD-Q4_K_XL shard goes through the
  // generalized WMMA kernel. Without this the only prefill route for them was
  // BatchedQuantGEMVKernel, one warp per (row, token), which re-reads the whole
  // weight matrix once per token and is unusable at prefill batch.
  if (type != core::GgmlType::kQ8_0) {
    if (::gufo::quant::QuantizedBlockElements(type) != 0) {
      LaunchKQuantWmmaGEMM(type, w, q8_1_x, y, batch, m, k, stream);
    }
    return;
  }
  if (batch <= 8) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      W8A8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      W8A8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  // A 128-token macro tile is the throughput configuration; short prompt chunks
  // would leave most of it idle, so they take the 64-token variant.
  if (batch >= 96) {
    constexpr int kBM = 128;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // Two K blocks per LDS stage, not four: the stage buffers drop from 36864
    // to 18432 bytes and occupancy rises from 6 to 8 waves per SIMD. This is
    // only reachable with the register pressure the addressing rewrite and the
    // one-token-tile-at-a-time compute loop freed up -- at BK=2 the previous
    // body needed 500 bytes/lane of scratch and ran 4.3x slower.
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  } else {
    constexpr int kBM = 128;
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
  }
}

void LaunchBatchedDualQuantGEMMPreQuantized(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, float* y_up, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  // The fused 16-row dual kernel shared one activation stage between the two
  // weight matrices, which mattered when a block owned only 16 rows. With the
  // 128-row blocked macro tile the activation panel is already cheap, and two
  // blocked launches measured faster than a blocked dual (23.0 vs 24.4 ms for
  // the FFN shape at batch 2048) because the dual doubles LDS and halves
  // occupancy. See opt-c163-blocked-dual in benchmarks/qwen3.8-27b/README.md.
  LaunchBatchedQuantGEMMPreQuantized(type, w_gate, q8_1_x, y_gate, batch, m, k,
                                     stream);
  LaunchBatchedQuantGEMMPreQuantized(type, w_up, q8_1_x, y_up, batch, m, k,
                                     stream);
}

namespace {

/// Dispatches the fused-epilogue instantiation of whichever blocked prefill
/// kernel `type` uses. The row-block base has to be a whole 32-element
/// quantization block for every block in the grid, which is what pins BM here.
template<int BN>
void LaunchFusedSwiGluTile(const void* w_up, const void* q8_1_x,
                           const float* gate, void* q8_1_out, std::size_t batch,
                           std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  const dim3 grid(static_cast<unsigned int>((batch + BN - 1) / BN),
                  static_cast<unsigned int>(m / kBM));
  W8A8BlockedWmmaGEMMKernel<kBM, BN, 2, 4, 2, true, true>
      <<<grid, 256, 0, stream>>>(w_up, q8_1_x, nullptr, batch, m, k, gate,
                                 q8_1_out);
}

}  // namespace

bool IsFusedSwiGluGemmEpilogueSupported(core::GgmlType type, std::size_t batch,
                                        std::size_t m,
                                        std::size_t q8_out_bytes) noexcept {
  return type == core::GgmlType::kQ8_0 && batch >= 96 && m != 0 &&
         m % 128 == 0 &&
         Q8ActTiledBytes(batch, m) + Q8ActSumBytes(batch, m) <= q8_out_bytes;
}

void LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, float* y_gate, void* q8_1_out, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  LaunchBatchedQuantGEMMPreQuantized(type, w_gate, q8_1_x, y_gate, batch, m, k,
                                     stream);
  LaunchFusedSwiGluTile<128>(w_up, q8_1_x, y_gate, q8_1_out, batch, m, k,
                             stream);
  LaunchZeroQ8ActTail<true>(q8_1_out, batch, m, stream);
}

void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    static thread_local void* s_act_q8_scratch = nullptr;
    static thread_local std::size_t s_act_q8_scratch_bytes = 0;

    const std::size_t req_bytes = Q8ActTiledBytes(batch, k);
    if (s_act_q8_scratch_bytes < req_bytes) {
      if (s_act_q8_scratch != nullptr) {
        (void)hipFree(s_act_q8_scratch);
      }
      HIP_CHECK(hipMalloc(&s_act_q8_scratch, req_bytes));
      s_act_q8_scratch_bytes = req_bytes;
    }

    LaunchQuantizeActivationQ8_1(bf16_x, s_act_q8_scratch, batch, k, stream);
    LaunchBatchedQuantGEMMPreQuantized(type, w, s_act_q8_scratch, y, batch, m,
                                       k, stream);
    return;
  }

  constexpr std::size_t kWarp = 32;
  const std::size_t warps_per_block = 8;
  const std::size_t rows_per_block = warps_per_block;
  const std::size_t num_blocks_x = (m + rows_per_block - 1) / rows_per_block;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * warps_per_block));
  BatchedQuantGEMVKernel<hip_bfloat16><<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

void LaunchBatchedQuantGEMMBf16(core::GgmlType type, const void* w,
                                const void* bf16_x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || type == core::GgmlType::kBF16 ||
      type == core::GgmlType::kF32) {
    return;
  }

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  if (type == core::GgmlType::kQ8_0 && batch <= 8) {
    constexpr std::size_t kRowsPerWave = 4;
    const std::size_t tokens_per_wave = batch <= 2 ? 2 : (batch <= 4 ? 4 : 8);
    const dim3 grid(
        static_cast<unsigned int>((m + (kWarpsPerBlock * kRowsPerWave) - 1) /
                                  (kWarpsPerBlock * kRowsPerWave)),
        1);
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    if (tokens_per_wave == 2) {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 2><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    } else if (tokens_per_wave == 4) {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 4><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    } else {
      TiledBatchedQ8_0GEMMKernel<kWarpsPerBlock, 8><<<grid, block, 0, stream>>>(
          w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, batch, m, k);
    }
    return;
  }

  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<hip_bfloat16><<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || type == core::GgmlType::kBF16 ||
      type == core::GgmlType::kF32) {
    return;
  }

  if (batch > 8) {
    // Paired seven-row blocks avoid a separate tail on these projections.
    const bool iq4_groups = type == core::GgmlType::kIQ4_XS &&
                            ((batch == 28 && ((m == 34816 && k == 5120) ||
                                              (m == 5120 && k == 17408))) ||
                             (batch == 42 && m == 5120 && k == 17408));
    if ((type == core::GgmlType::kQ6_K ||
         (type == core::GgmlType::kQ5_K && m == 5120 && k == 6144) ||
         iq4_groups) &&
        batch <= 56 && batch % 14 == 0 &&
        detail::TryLaunchKQuantSmallBatchWave64(type, w, fp32_x, y, 14, m, k,
                                                stream, batch / 14)) {
      return;
    }
    const bool ffn =
        ((m == 17408 || m == 34816) && k == 5120) || (m == 5120 && k == 17408);
    const bool q8_projection =
        type == core::GgmlType::kQ8_0 &&
        (ffn || (k == 5120 && (m == 6144 || m == 10240)) ||
         (m == 5120 && k == 6144));
    const auto launch_q8 = [&]<std::size_t Batch>(const float* input,
                                                  float* output) {
      if constexpr (Batch == 12) {
        if (detail::TryLaunchQ8SmallBatchWave64(w, input, output, Batch, m, k,
                                                stream))
          return;
      }
      constexpr std::size_t rows = Batch == 11 || Batch == 13 || Batch == 14 ? 4
                                   : Batch == 12                             ? 2
                                                 : 3;
      SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, rows, true, true>
          <<<(m + 16 * rows - 1) / (16 * rows), 512, 0, stream>>>(w, input,
                                                                  output, m, k);
    };
    // Cross-request verification can contain up to 64 rows. Reuse weights in
    // wider exact groups where measured, and keep the established small-batch
    // arithmetic for other projections and the final unpadded group.
    const bool exact_width = q8_projection || (m == 34816 && k == 5120) ||
                             (m == 5120 && k == 17408) ||
                             (type == core::GgmlType::kQ6_K &&
                              ((k == 5120 && (m == 248320 || m == 17408)) ||
                               (m == 5120 && k == 6144)));
    for (std::size_t offset = 0; offset < batch;) {
      const std::size_t remaining = batch - offset;
      const auto groups = std::min<std::size_t>(4, remaining / 16);
      if (groups > 1) {
        const bool launched =
            q8_projection ? detail::TryLaunchQ8SmallBatchWave64(
                                w, fp32_x + offset * k, y + offset * m, 16, m,
                                k, stream, groups)
                          : detail::TryLaunchKQuantSmallBatchWave64(
                                type, w, fp32_x + offset * k, y + offset * m,
                                16, m, k, stream, groups);
        if (launched) {
          offset += groups * 16;
          continue;
        }
      }
      std::size_t width = exact_width && remaining > 8
                              ? std::min<std::size_t>(16, remaining)
                          : remaining >= 16 ? 16
                          : remaining >= 12 ? 12
                                            : 0;
      const float* input = fp32_x + offset * k;
      float* output = y + offset * m;
      if (width != 0 && q8_projection) {
        switch (width) {
          case 9:
            launch_q8.template operator()<9>(input, output);
            break;
          case 10:
            launch_q8.template operator()<10>(input, output);
            break;
          case 11:
            launch_q8.template operator()<11>(input, output);
            break;
          case 12:
            launch_q8.template operator()<12>(input, output);
            break;
          case 13:
            launch_q8.template operator()<13>(input, output);
            break;
          case 14:
            launch_q8.template operator()<14>(input, output);
            break;
          case 15:
            launch_q8.template operator()<15>(input, output);
            break;
          case 16:
            launch_q8.template operator()<16>(input, output);
            break;
        }
      } else if (width != 0 &&
                 detail::TryLaunchKQuantSmallBatchWave64(type, w, input, output,
                                                         width, m, k, stream)) {
        // The native dispatcher owns the qualified K-quant shapes.
      } else {
        width = std::min<std::size_t>(8, remaining);
        LaunchBatchedQuantGEMMFp32(type, w, input, output, width, m, k, stream);
      }
      offset += width;
    }
    return;
  }

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  // opt-q4kxl: K-quant/IQ formats share one weight stream across the whole
  // draft block instead of re-reading it per token.
  if (::gufo::hip::detail::IsNativeWmmaQuant(type) && batch <= 8) {
    // Exact width instantiations: a padded Batch would stage and multiply
    // activations that no output consumes.
    constexpr std::size_t kKWarpsPerBlock = kKQuantSmallBatchWarps;
    const dim3 block(static_cast<unsigned int>(kWarp * kKWarpsPerBlock));
    // Three rows with twelve waves is the retained geometry. Q6_K dispatches
    // two rows below because its extra scale state spills with three.
    constexpr std::size_t kRowsPerBlock = kKWarpsPerBlock * 3;
    const dim3 grid(
        static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
    LaunchKQuantSmallBatchByType(type, w, fp32_x, y, batch, m, k, grid, block,
                                 stream);
    return;
  }
  if ((type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K) &&
      batch >= 1 && batch <= 8) {
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    // Large projections share each weight row across the batch. Narrow SSM
    // projections instead distribute token rows across more workgroups.
    if (type == core::GgmlType::kQ8_0) {
      if (batch == 6 && detail::TryLaunchQ8SmallBatchWave64(w, fp32_x, y, batch,
                                                            m, k, stream))
        return;
      // The 48-output SSM controls, or their joined 96-output pair, fit in
      // cache. Independent token rows expose more workgroups than the
      // large-matrix shared-weight geometry.
      if ((m == 48 || m == 96) && k == 5120) {
        SmallBatchQ8_0ExactFp32VecGEMMKernel<8, 1, 1>
            <<<dim3((m + 7) / 8, static_cast<unsigned>(batch)), 256, 0,
               stream>>>(w, fp32_x, y, m, k);
        return;
      }
      constexpr std::size_t kVecWarpsPerBlock = 16;
      // Joined Qwen27B gate/up matrices benefit from three-row reuse at
      // these widths. Their bounded dimensions allow 32-bit device indexing;
      // each output keeps the scalar block-dot and reduction order.
      if (m == 34816 && k == 5120) {
        const auto launch_packed = [&]<std::size_t Batch>() {
          SmallBatchQ8_0ExactFp32VecGEMMKernel<16, Batch, 3, true>
              <<<(m + 47) / 48, 512, 0, stream>>>(w, fp32_x, y, m, k);
        };
        switch (batch) {
          case 2:
            launch_packed.template operator()<2>();
            return;
          case 4:
            launch_packed.template operator()<4>();
            return;
          case 8:
            launch_packed.template operator()<8>();
            return;
          default:
            break;
        }
      }
      // Batch 7 shares its 31.5 KiB activation stage across twice as many
      // waves. Batch 8 instead benefits from three rows per wave below.
      if (batch == 7 && m == 5120 && k == 17408) {
        SmallBatchQ8_0ExactFp32VecGEMMKernel<32, 7, 2>
            <<<(m + 63) / 64, 1024, 0, stream>>>(w, fp32_x, y, m, k);
        return;
      }
      // Reuse the long FFN-down input across three output rows.
      if (batch == 8 && m == 5120 && k == 17408) {
        constexpr std::size_t kDownRows = kVecWarpsPerBlock * 3;
        SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 8, 3>
            <<<(m + kDownRows - 1) / kDownRows, kVecWarpsPerBlock * 32, 0,
               stream>>>(w, fp32_x, y, m, k);
        return;
      }
      constexpr std::size_t kVecRowsPerWave = 2;
      constexpr std::size_t kVecRowsPerBlock =
          kVecWarpsPerBlock * kVecRowsPerWave;
      const unsigned int vec_blocks = static_cast<unsigned int>(
          (m + kVecRowsPerBlock - 1) / kVecRowsPerBlock);
      const dim3 vec_block(
          static_cast<unsigned int>(kWarp * kVecWarpsPerBlock));
      switch (batch) {
        case 1:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 1,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 2:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 2,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 3:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 3,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 4:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 4,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 5:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 5,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 6:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 6,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        case 7:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 7,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
        default:
          SmallBatchQ8_0ExactFp32VecGEMMKernel<kVecWarpsPerBlock, 8,
                                               kVecRowsPerWave>
              <<<vec_blocks, vec_block, 0, stream>>>(w, fp32_x, y, m, k);
          return;
      }
    }
    // Only Q8_K reaches here: Q8_0 always takes the width-templated route
    // above.
    if (batch <= 1) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 1>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else if (batch <= 2) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 2>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else if (batch <= 4) {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 4>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    } else {
      SmallBatchQ8KFp32GEMMKernel<kWarpsPerBlock, 8>
          <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
              w, fp32_x, y, batch, m, k);
    }
    return;
  }

  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<float>
      <<<grid, block, 0, stream>>>(type, w, fp32_x, y, m, k);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
