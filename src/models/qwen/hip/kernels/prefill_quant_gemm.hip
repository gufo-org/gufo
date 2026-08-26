#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace strix::hip {

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

[[nodiscard]] bool UseCompactBf16SmallBatchTile() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("STRIX_PREFILL_SMALL_BATCH_BF16_TILE");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "fixed8" && setting != "baseline" && setting != "0" &&
           setting != "false" && setting != "off";
  }();
  return enabled;
}

[[nodiscard]] bool UseCompactW8A8SmallBatchTile() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("STRIX_PREFILL_SMALL_BATCH_W8A8_TILE");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "fixed64" && setting != "baseline" && setting != "0" &&
           setting != "false" && setting != "off";
  }();
  return enabled;
}

// Tiled Q8_1 activation layout (opt-c163-blocked-w8a8).
//
// Activations are written straight into WMMA B-fragment order so both the LDS
// staging copy and the fragment reads are contiguous:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block. The 576-byte stride keeps
// every fragment and scale vector 16-byte aligned. For a token count that is a
// multiple of 16 this is exactly the same total size as the previous row-major
// block_q8_1 buffer (36 bytes per 32 elements), so arena sizing is unchanged.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;

/// Byte size of the tiled Q8_1 activation buffer for a [batch, k] tensor.
__host__ __device__ inline std::size_t Q8ActTiledBytes(std::size_t batch,
                                                       std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return tiles * (k / 32) * kQ8ActTileBytes;
}

__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

__device__ __forceinline__ const std::int8_t* Q8ActTile(const void* base,
                                                        std::size_t num_blocks,
                                                        std::size_t tt,
                                                        std::size_t kb) {
  return static_cast<const std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// Writes one lane's quantized byte and (lane 0) the block scale into the tiled
/// layout. `lane_id` is the element index inside the 32-element block.
__device__ __forceinline__ void StoreQ8ActLane(void* y, std::size_t num_blocks,
                                               std::size_t tok, std::size_t blk,
                                               std::size_t lane_id, float d,
                                               std::int8_t q) {
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

template<typename T>
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

  StoreQ8ActLane(y, num_blocks, tok, blk, lane_id, d, q);
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
    StoreQ8ActLane(q8_out, num_blocks, b, blk, lane_id, d, q);
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
  StoreQ8ActLane(q8_out, num_blocks, t, blk, lane_id, d, q);
}

/// Zeroes the tail of the last token tile so a batch that is not a multiple of
/// 16 cannot feed uninitialized scales into the GEMM epilogue.
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
}

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

  StoreQ8ActLane(y, num_blocks, tok, blk, lane_id, d, q);
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
template<int BM, int BN, int BK, int WM, int WN>
__launch_bounds__(256) __global__
    void W8A8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                   const void* __restrict__ x_blocks,
                                   float* __restrict__ y, std::size_t batch,
                                   std::size_t m, std::size_t k) {
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

template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void SmallBatchQ8_0Fp32GEMMKernel(const void* __restrict__ w,
                                      const float* __restrict__ x,
                                      float* __restrict__ y, std::size_t batch,
                                      std::size_t m, std::size_t k) {
  constexpr std::size_t kMaxBatch = 8;
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + warp_id;
  if (row >= m) {
    return;
  }

  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* row_blocks =
      static_cast<const Q8_0Block*>(w) + (row * num_blocks);
  float sums[kMaxBatch] = {0.0F};

  // This is the Q8_0 branch of QuantWarpBlockDot<float>, with its exact
  // per-block reduction and accumulation order retained for every token.
  // Keeping the candidate loop inside one warp lets all candidates share the
  // same weight load, which is the dominant traffic at these small batches.
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const Q8_0Block& wblk = row_blocks[b];
    const float d_w = __half2float(wblk.d);
    const float q = static_cast<float>(wblk.qs[lane]);
#pragma unroll
    for (std::size_t tok = 0; tok < kMaxBatch; ++tok) {
      if (tok < batch) {
        float dot = q * x[(tok * k) + (b * kQ8_0BlockSize) + lane];
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor(dot, off);
        }
        sums[tok] += d_w * dot;
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (std::size_t tok = 0; tok < kMaxBatch; ++tok) {
      if (tok < batch) {
        y[(tok * m) + row] = sums[tok];
      }
    }
  }
}

namespace {

void LaunchZeroQ8ActTail(void* q8_1_out, std::size_t batch, std::size_t k,
                         hipStream_t stream) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t work = num_blocks * (kQ8ActTileTokens - tail);
  ZeroQ8ActTailKernel<<<static_cast<unsigned int>((work + 255) / 256), 256, 0,
                        stream>>>(q8_1_out, batch, num_blocks);
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
  QuantizeActivationToQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                   quant_threads, 0, stream>>>(
      reinterpret_cast<const hip_bfloat16*>(bf16_x), q8_1_out, batch, k);
  LaunchZeroQ8ActTail(q8_1_out, batch, k, stream);
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
  BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel<<<grid, 128, 0, stream>>>(
      raw_out, ssm_norm, gate, q8_1_out, batch_size, inner_size, num_heads,
      val_dim);
  LaunchZeroQ8ActTail(q8_1_out, batch_size, inner_size, stream);
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
  BatchedFusedRMSNormQuantizeQ8_1Kernel<<<static_cast<unsigned int>(batch_size),
                                          256, lds_bytes, stream>>>(
      x, residual, weight, sum_out, q8_1_out, batch_size, dim, eps);
  LaunchZeroQ8ActTail(q8_1_out, batch_size, dim, stream);
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
  QuantizeActivationToQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                   quant_threads, 0, stream>>>(fp32_x, q8_1_out,
                                                               batch, k);
  LaunchZeroQ8ActTail(q8_1_out, batch, k, stream);
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
  BatchedFusedSwiGLUQuantizeQ8_1Kernel<<<static_cast<unsigned int>(quant_grid),
                                         quant_threads, 0, stream>>>(
      gate, up, q8_1_out, batch, k);
  LaunchZeroQ8ActTail(q8_1_out, batch, k, stream);
}

void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType type, const void* w,
                                        const void* q8_1_x, float* y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type != core::GgmlType::kQ8_0) {
    return;
  }
  if (batch <= 8 && UseCompactW8A8SmallBatchTile()) {
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
    const std::size_t tokens_per_wave =
        UseCompactBf16SmallBatchTile() ? (batch <= 2 ? 2 : (batch <= 4 ? 4 : 8))
                                       : 8;
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

  constexpr std::size_t kWarp = 32;
  constexpr std::size_t kWarpsPerBlock = 8;
  const std::size_t num_blocks_x = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (type == core::GgmlType::kQ8_0 && batch <= 8) {
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    SmallBatchQ8_0Fp32GEMMKernel<kWarpsPerBlock>
        <<<static_cast<unsigned int>(num_blocks_x), block, 0, stream>>>(
            w, fp32_x, y, batch, m, k);
    return;
  }

  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
  BatchedQuantGEMVKernel<float>
      <<<grid, block, 0, stream>>>(type, w, fp32_x, y, m, k);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
