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
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

[[nodiscard]] bool UseCompactBf16SmallBatchTile() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_PREFILL_SMALL_BATCH_BF16_TILE");
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
    const char* value = std::getenv("GUFO_PREFILL_SMALL_BATCH_W8A8_TILE");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "fixed64" && setting != "baseline" && setting != "0" &&
           setting != "false" && setting != "off";
  }();
  return enabled;
}

[[nodiscard]] bool UseExactSharedQ8SmallBatchFp32() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_Q8_SMALL_BATCH_EXACT_SHARED");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "0" && setting != "false" && setting != "off";
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
constexpr std::size_t kQ8ActSumTileBytes = kQ8ActTileTokens * sizeof(float);

/// Byte size of the tiled Q8_1 activation buffer for a [batch, k] tensor.
__host__ __device__ inline std::size_t Q8ActTiledBytes(std::size_t batch,
                                                       std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return tiles * (k / 32) * kQ8ActTileBytes;
}

/// Byte size of the optional per-token activation-sum sidecar. Keeping it
/// after the tiled payload leaves the Q8_1 layout and every Q8 kernel
/// unchanged.
__host__ __device__ inline std::size_t Q8ActSumBytes(std::size_t batch,
                                                     std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return tiles * (k / 32) * kQ8ActSumTileBytes;
}

__device__ __forceinline__ float* Q8ActSumSidecar(void* base, std::size_t batch,
                                                  std::size_t k) {
  return reinterpret_cast<float*>(static_cast<std::uint8_t*>(base) +
                                  Q8ActTiledBytes(batch, k));
}

__device__ __forceinline__ const float* Q8ActSumSidecar(const void* base,
                                                        std::size_t batch,
                                                        std::size_t k) {
  return reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(base) +
                                        Q8ActTiledBytes(batch, k));
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
template<bool StoreActivationSum>
__device__ __forceinline__ void StoreQ8ActLane(void* y, std::size_t batch,
                                               std::size_t num_blocks,
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
  if constexpr (StoreActivationSum) {
    int qsum = static_cast<int>(q);
    for (int off = 16; off > 0; off >>= 1) {
      qsum += __shfl_xor(qsum, off);
    }
    if (lane_id == 0) {
      float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
      sums[(((tt * num_blocks) + blk) * kQ8ActTileTokens) + tl] =
          d * static_cast<float>(qsum);
    }
  }
}

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
  const int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

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
    const std::int8_t q = static_cast<std::int8_t>(__float2int_rn(val * id));
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
  const std::int8_t q = static_cast<std::int8_t>(__float2int_rn(val * id));
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
  const int8_t q = static_cast<int8_t>(__float2int_rn(val * id));

  StoreQ8ActLane<StoreActivationSum>(y, batch, num_blocks, tok, blk, lane_id, d,
                                     q);
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

// opt-q4kxl: blocked W(K-quant)A8 WMMA GEMM.
//
// The Q8_0 kernel above stages 32 int8 weight codes and one fp32 scale per
// (row, K block) into LDS and feeds them to `wmma_16x16x16_iu8`. Every format
// in the UD-Q4_K_XL shard reduces to exactly that shape once
// DecodeQuantSub16 has run:
//
//     w[j] = scale * code[j] - offset
//
// so this kernel keeps the whole staging/compute/epilogue structure and only
// replaces the fetch. Two properties of the K-quant families force the two
// template flags:
//
//   PerHalfScale - Q6_K and Q3_K carry one scale per SIXTEEN elements, so the
//                  two halves of a 32-element K block need separate scales.
//                  Sixteen is exactly one WMMA fragment's K depth, so the two
//                  halves simply stop sharing an accumulator: c0 and c1 are
//                  scaled independently instead of being summed first. Every
//                  other format leaves both scales equal and takes the cheaper
//                  single-accumulator path.
//   HasOffset    - Q4_K and Q5_K encode an explicit minimum, giving
//                  sum_j w_j x_j = scale * sum_j q_j x_j - offset * sum_j x_j.
//                  The second term needs the per-token activation sum, which
//                  is recovered in-register from the staged Q8 codes
//                  (sum_j x_j = dx * sum_j qx_j) with four v_dot4 per fragment
//                  rather than by widening the activation layout. Formats
//                  without a minimum compile the term out entirely.
//
// Weight type is a TEMPLATE parameter, not a runtime argument. It is
// block-uniform either way, so a runtime argument looks free -- but
// DecodeQuantSub16 is a switch over every supported format, and inlining it
// twice per fetch put all eight decode bodies in the instruction stream. An
// ISA dump of the runtime-typed version measured 6711 instructions against the
// Q8_0 kernel's 3049, with 120 global-load instructions against 14, and the
// register allocator had to size for the worst branch: 192-208 VGPRs against
// 184. Only one branch ever executed, but every branch cost occupancy. With
// the type known at compile time each instantiation carries exactly one decode
// path.
//
// Left at the default occupancy target deliberately. Q5_K, Q4_K and Q6_K settle
// at 197-206 VGPRs, which is seven wave32s per SIMD rather than the eight the
// Q8_0 kernel gets at 184. Forcing eight with `__launch_bounds__(256, 8)` does
// reach 8 waves with no meaningful scratch, and measured WORSE: pp2048 fell
// from 462.7 to 412.9 t/s. The rematerialization the allocator inserts into the
// inner loop to hit the register target costs more than the extra wave returns,
// so this kernel is latency-bound rather than occupancy-bound.
template<int BM, int BN, int BK, int WM, int WN, core::GgmlType WType,
         bool UsePrecomputedSums, bool StagePrecomputedSums,
         bool ForceOccupancy = false, bool DropOffsetProbe = false,
         bool FuseOffsetTerm = false>
// `ForceOccupancy` asks the allocator for eight wave32s per SIMD instead of
// letting it settle. It exists only so the 512-thread tile can be measured at
// two workgroups per CU as well as one; see the opt-q4kxl-wide write-up.
__launch_bounds__(WM * WN * 32, ForceOccupancy ? 8 : 1) __global__
    void WKQuantA8BlockedWmmaGEMMKernel(const void* __restrict__ w,
                                        const void* __restrict__ x_blocks,
                                        float* __restrict__ y,
                                        std::size_t batch, std::size_t m,
                                        std::size_t k) {
  // Q6_K and Q3_K carry one scale per sixteen elements; Q4_K and Q5_K carry an
  // explicit minimum. The two properties never co-occur.
  constexpr bool PerHalfScale =
      WType == core::GgmlType::kQ6_K || WType == core::GgmlType::kQ3_K;
  // `DropOffsetProbe` deletes the Q4_K/Q5_K minimum correction outright. The
  // result is WRONG -- `w = scale*q - offset` is not `scale*q` -- and it exists
  // only to measure what a hypothetical free correction would be worth, which
  // is the upper bound on any exact reformulation. It is never a production
  // route and the correctness gate rejects it.
  constexpr bool HasOffset =
      (WType == core::GgmlType::kQ4_K || WType == core::GgmlType::kQ5_K) &&
      !DropOffsetProbe;
  static_assert(!StagePrecomputedSums || UsePrecomputedSums);
  // The block is WM*WN waves of 32 lanes. Eight waves (256 threads) is the
  // production shape; sixteen waves (512 threads) is what makes BN=256
  // reachable at all, because the accumulator array is BM*BN/threads registers
  // and BM*BN is otherwise pinned at 128*128.
  static_assert(WM * WN == 8 || WM * WN == 16, "8 or 16 wave32s per block");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kThreads = WM * WN * 32;
  constexpr int kWaves = WM * WN;
  static_assert(BN / 16 <= kWaves || (BN / 16) % kWaves == 0,
                "token subtiles must divide evenly across the waves");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  constexpr int kScaleHalves = PerHalfScale ? 2 : 1;
  constexpr int kOffRowTiles = HasOffset ? kRowTiles : 1;
  // Token subtiles are staged one per wave. A macro tile with more subtiles
  // than waves gives each wave several, which is what lets BN grow without
  // growing the block: BM=64/BN=256 keeps the same BM*BN/threads accumulator
  // and the same eight-wave workgroup as BM=128/BN=128.
  constexpr int kStageTiles = (kTokTiles + kWaves - 1) / kWaves;

  __shared__ int32x4_t s_a[BK][kRowTiles][32];
  __shared__ float s_dw[BK][kScaleHalves][kRowTiles][16];
  __shared__ float s_off[BK][kOffRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];
  __shared__ float s_sx[StagePrecomputedSums && HasOffset ? BK : 1][kTokTiles]
                       [16];

  const std::size_t num_blocks = k / 32;
  const std::size_t row_bytes = QuantRowBytes(WType, k);

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

  // One fetch unit is one (row, K-block) pair, decoded as two sub-16 blocks.
  // At 256 threads every configuration has at least one unit per thread and the
  // indexing below reduces to what it was. At 512 threads with BK=2 there are
  // only BM*BK = 256 units, so half the block sits out the weight half of the
  // stage; keeping BK=2 rather than growing it to 4 is what holds LDS at 30 KB
  // and therefore two resident workgroups.
  constexpr int kFetchUnits = BM * BK;
  constexpr int kPrefetch = (kFetchUnits + kThreads - 1) / kThreads;
  constexpr bool kPartialFetch = (kPrefetch * kThreads) != kFetchUnits;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw0[kPrefetch];
  float r_dw1[kPrefetch];
  float r_off[kPrefetch];
  int32x4_t r_b[kStageTiles][BK];
  float r_dx[kStageTiles][BK];
  float r_sx[kStageTiles][StagePrecomputedSums && HasOffset ? BK : 1];

  // Row bases and liveness are hoisted exactly as in the Q8_0 kernel: an
  // out-of-range row is clamped to a valid address and its scale (and offset)
  // zeroed, so it contributes zero without any divergent branch.
  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const std::uint8_t* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int idx = (p * kThreads) + tid;
    const bool unit_live = !kPartialFetch || idx < kFetchUnits;
    const int idx_clamped = unit_live ? idx : 0;
    const int r = static_cast<int>(r_block) + (idx_clamped / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = static_cast<const std::uint8_t*>(w) +
               (static_cast<std::size_t>(r_clamped) * row_bytes);
    row_live[p] = (r < m_i && unit_live) ? 1.0F : 0.0F;
  }
  // The staged activation buffer holds exactly ceil(batch / 16) token tiles.
  // A macro tile wider than the tail reaches past it, so an out-of-range tile
  // is clamped to tile 0 and its scale zeroed -- the same trick the row bases
  // use, and the reason a 256-token tile can be launched on any batch at all.
  const std::size_t num_act_tiles =
      (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  int b_tile[kStageTiles];
  const std::int8_t* b_base[kStageTiles];
  std::size_t act_tile[kStageTiles];
  float tile_scale[kStageTiles];
  bool b_live[kStageTiles];
#pragma unroll
  for (int t = 0; t < kStageTiles; ++t) {
    b_tile[t] = wave_id + (t * kWaves);
    const std::size_t wanted = tt_block + static_cast<std::size_t>(b_tile[t]);
    const bool live = (b_tile[t] < kTokTiles) && (wanted < num_act_tiles);
    act_tile[t] = live ? wanted : 0;
    tile_scale[t] = live ? 1.0F : 0.0F;
    b_live[t] = b_tile[t] < kTokTiles;
    b_base[t] = reinterpret_cast<const std::int8_t*>(x_blocks) +
                (act_tile[t] * num_blocks * kQ8ActTileBytes);
  }
  const float* activation_sums = Q8ActSumSidecar(x_blocks, batch, k);

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      const int idx_clamped = (!kPartialFetch || idx < kFetchUnits) ? idx : 0;
      const int kb = kb0 + (idx_clamped % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const float live = (kb < num_kb) ? row_live[p] : 0.0F;
      QuantSub16 lo;
      QuantSub16 hi;
      DecodeQuantSub16(WType, w_row[p],
                       static_cast<std::size_t>(kb_clamped) * 2, lo);
      DecodeQuantSub16(WType, w_row[p],
                       (static_cast<std::size_t>(kb_clamped) * 2) + 1, hi);
      __builtin_memcpy(&r_q0[p], lo.q, 16);
      __builtin_memcpy(&r_q1[p], hi.q, 16);
      r_dw0[p] = lo.scale * live;
      r_dw1[p] = hi.scale * live;
      if constexpr (HasOffset) {
        r_off[p] = lo.offset * live;
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile = b_base[t] + (static_cast<std::size_t>(kb_clamped) *
                                        kQ8ActTileBytes);
        r_b[t][i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                     reinterpret_cast<const float*>(
                         tile + kQ8ActScaleOffset)[lane_id & 15];
        if constexpr (StagePrecomputedSums && HasOffset) {
          r_sx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                       activation_sums[(((act_tile[t] * num_blocks) +
                                         static_cast<std::size_t>(kb_clamped)) *
                                        kQ8ActTileTokens) +
                                       static_cast<std::size_t>(lane_id & 15)];
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      if constexpr (kPartialFetch) {
        if (idx >= kFetchUnits) {
          continue;
        }
      }
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      const int slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][0][rs][slot] = r_dw0[p];
      if constexpr (PerHalfScale) {
        s_dw[kk][1][rs][slot] = r_dw1[p];
      }
      if constexpr (HasOffset) {
        s_off[kk][rs][slot] = r_off[p];
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile[t]][lane_id] = r_b[t][i];
        if (lane_id < 16) {
          s_dx[i][b_tile[t]][lane_id] = r_dx[t][i];
          if constexpr (StagePrecomputedSums && HasOffset) {
            s_sx[i][b_tile[t]][lane_id] = r_sx[t][i];
          }
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
      float dw0[kWaveRowTiles][8];
      // Sized away when the format does not use them: holding dead
      // per-accumulator arrays live across the whole WMMA block is exactly what
      // the Q8_0 kernel's addressing rewrite found to push a BK=2 stage into
      // scratch.
      float dw1[PerHalfScale ? kWaveRowTiles : 1][8];
      float off[HasOffset ? kWaveRowTiles : 1][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][0][rs][half_id * 8]);
        const float4 up = *reinterpret_cast<const float4*>(
            &s_dw[kb][0][rs][(half_id * 8) + 4]);
        dw0[i][0] = lo.x;
        dw0[i][1] = lo.y;
        dw0[i][2] = lo.z;
        dw0[i][3] = lo.w;
        dw0[i][4] = up.x;
        dw0[i][5] = up.y;
        dw0[i][6] = up.z;
        dw0[i][7] = up.w;
        if constexpr (PerHalfScale) {
          const float4 hlo =
              *reinterpret_cast<const float4*>(&s_dw[kb][1][rs][half_id * 8]);
          const float4 hup = *reinterpret_cast<const float4*>(
              &s_dw[kb][1][rs][(half_id * 8) + 4]);
          dw1[i][0] = hlo.x;
          dw1[i][1] = hlo.y;
          dw1[i][2] = hlo.z;
          dw1[i][3] = hlo.w;
          dw1[i][4] = hup.x;
          dw1[i][5] = hup.y;
          dw1[i][6] = hup.z;
          dw1[i][7] = hup.w;
        }
        if constexpr (HasOffset) {
          const float4 olo =
              *reinterpret_cast<const float4*>(&s_off[kb][rs][half_id * 8]);
          const float4 oup = *reinterpret_cast<const float4*>(
              &s_off[kb][rs][(half_id * 8) + 4]);
          off[i][0] = olo.x;
          off[i][1] = olo.y;
          off[i][2] = olo.z;
          off[i][3] = olo.w;
          off[i][4] = oup.x;
          off[i][5] = oup.y;
          off[i][6] = oup.z;
          off[i][7] = oup.w;
        }
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
        // Activation sum for this token over the 32-element K block. b0/b1
        // hold exactly token `sub_lane`'s codes, which is the same token dx
        // belongs to, so this needs no cross-lane traffic.
        float sx = 0.0F;
        if constexpr (HasOffset) {
          if constexpr (UsePrecomputedSums) {
            if constexpr (StagePrecomputedSums) {
              sx = s_sx[kb][ts][sub_lane];
            } else {
              const std::size_t wanted_tile =
                  tt_block + static_cast<std::size_t>(ts);
              const bool sum_live = wanted_tile < num_act_tiles;
              const std::size_t token_tile = sum_live ? wanted_tile : 0;
              const int global_kb = kb0 + kb;
              const std::size_t activation_block = static_cast<std::size_t>(
                  global_kb < num_kb ? global_kb : num_kb - 1);
              sx = (sum_live ? 1.0F : 0.0F) *
                   activation_sums[(((token_tile * num_blocks) +
                                     activation_block) *
                                    kQ8ActTileTokens) +
                                   static_cast<std::size_t>(sub_lane)];
            }
          } else {
            // gfx1151 exposes V_DOT4_I32_IU8 (`sudot4`) but not the dot1-insts
            // `sdot4`; the activation codes are signed and the ones vector is
            // unsigned, which is exactly the mixed form sudot4 takes.
            int qsum = 0;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              qsum = __builtin_amdgcn_sudot4(true, b0[v], false, 0x01010101,
                                             qsum, false);
              qsum = __builtin_amdgcn_sudot4(true, b1[v], false, 0x01010101,
                                             qsum, false);
            }
            sx = dx * static_cast<float>(qsum);
          }
        }
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          if constexpr (PerHalfScale) {
            int32x8_t c0 = {0, 0, 0, 0, 0, 0, 0, 0};
            int32x8_t c1 = {0, 0, 0, 0, 0, 0, 0, 0};
            c0 = Wmma_16x16x16_iu8<true>(a0[i], b0, c0);
            c1 = Wmma_16x16x16_iu8<true>(a1[i], b1, c1);
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              acc[i][j][l] += (dw0[i][l] * dx) * static_cast<float>(c0[l]) +
                              (dw1[i][l] * dx) * static_cast<float>(c1[l]);
            }
          } else {
            int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
            c = Wmma_16x16x16_iu8<true>(a0[i], b0, c);
            c = Wmma_16x16x16_iu8<true>(a1[i], b1, c);
            // opt-q4kxl-fuse: with the minimum correction applied as a second
            // accumulate, `acc` takes two dependent FMAs per K block and the
            // scheduler pays for it -- an ISA dump puts the correction at +131
            // fp32 FMAs and +115 `s_delay_alu` against the same kernel with the
            // term compiled out. Folding the correction into the main term
            // first leaves one accumulate, so the chain on `acc` is half as
            // long, at the cost of one extra multiply per output.
            if constexpr (HasOffset && FuseOffsetTerm) {
#pragma unroll
              for (int l = 0; l < 8; ++l) {
                const float term =
                    ((dw0[i][l] * dx) * static_cast<float>(c[l])) -
                    (off[i][l] * sx);
                acc[i][j][l] += term;
              }
            } else {
#pragma unroll
              for (int l = 0; l < 8; ++l) {
                acc[i][j][l] += (dw0[i][l] * dx) * static_cast<float>(c[l]);
              }
            }
          }
          if constexpr (HasOffset && !(FuseOffsetTerm && !PerHalfScale)) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              acc[i][j][l] -= off[i][l] * sx;
            }
          }
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  // Each wave stages its 16x16 output tile through 256 floats of LDS. `s_a` is
  // BK*kRowTiles*512 bytes, which is exactly kWaves*1024 at 256 threads; the
  // 512-thread shape needs 16 KB, which is what `s_b` is at BK=2 with BN=256.
  // Pick whichever staging array actually covers the requirement -- both are
  // dead by this point, and the __syncthreads above orders the reuse.
  // A stage buffer big enough for every wave at once is not guaranteed: at
  // BK=1 each of `s_a` and `s_b` is only 4 KB while eight waves want 8 KB. Take
  // the larger array, work out how many waves it holds, and run the epilogue in
  // that many passes. The extra barriers are per block, not per K stage.
  constexpr int kScratchBytes =
      static_cast<int>(sizeof(s_a) > sizeof(s_b) ? sizeof(s_a) : sizeof(s_b));
  constexpr int kWavesPerPass = kScratchBytes / (256 * 4);
  static_assert(kWavesPerPass >= 1,
                "no staging array holds even one wave's output tile");
  constexpr int kEpiloguePasses = (kWaves + kWavesPerPass - 1) / kWavesPerPass;
  float* const scratch_base =
      (sizeof(s_a) > sizeof(s_b) ? reinterpret_cast<float*>(&s_a[0][0][0])
                                 : reinterpret_cast<float*>(&s_b[0][0][0]));
  float* tile_scratch = scratch_base + ((wave_id % kWavesPerPass) * 256);
#pragma unroll
  for (int pass = 0; pass < kEpiloguePasses; ++pass) {
    if constexpr (kEpiloguePasses > 1) {
      __syncthreads();
      if ((wave_id / kWavesPerPass) != pass) {
        continue;
      }
    }
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
        if (t0 + 16 <= batch && r0 + 16 <= m) {
          const std::size_t out_base = (t0 * m) + r0;
#pragma unroll
          for (int st = 0; st < 8; ++st) {
            const int flat = (st * 32) + lane_id;
            y[out_base + (static_cast<std::size_t>(flat >> 4) * m) +
              static_cast<std::size_t>(flat & 15)] = tile_scratch[flat];
          }
        } else {
#pragma unroll
          for (int st = 0; st < 8; ++st) {
            const int flat = (st * 32) + lane_id;
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

// opt-c18x-q8-vec-smallbatch: exact Q8_0 GEMM for widths 2..8, staging the
// whole activation batch in LDS.
//
// Two layout choices carry this kernel, and the version it replaced got both
// wrong. Its activation stride was 33 floats, which is not 16-byte aligned, so
// each group of four activations cost four `ds_read_b32`; and with one output
// row per warp those reads were amortized over only 34 bytes of weight. That
// made the kernel LDS-instruction bound rather than DRAM bound, and it read
// 134 GB/s at width 8 against 228 GB/s at width 2. Padding the stride to 36
// floats makes every group one `ds_read_b128`, and two output rows per warp
// halves the reads per weight byte again: width 8 reaches 220 GB/s on the FFN
// shapes, the same bandwidth as width 2 and as batch-1 decode. See
// tools/bench/q8_small_batch_gemm_bench.hip.
//
// The accumulation order for a given (row, token) does not depend on the batch
// width -- blocks in lane, lane+32, ... order, groups ascending, four products
// per group in ascending element order -- so every width is bit-exact with the
// decode GEMV, which is what speculative verification requires.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave>
__launch_bounds__(WavesPerBlock * 32, 1) __global__
    void SmallBatchQ8_0ExactFp32VecGEMMKernel(const void* __restrict__ w,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,
                                              std::size_t m, std::size_t k) {
  constexpr std::size_t kBlocksPerTile = 32;
  constexpr std::size_t kVectorsPerBlock = kQ8_0BlockSize / 4;
  constexpr std::size_t kStride = kQ8_0BlockSize + 4;
  constexpr std::size_t kTileStride = kBlocksPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_blocks = k / kQ8_0BlockSize;
  const auto* base = static_cast<const Q8_0Block*>(w);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_blocks;
       tile_base += kBlocksPerTile) {
    constexpr std::size_t kTileVectors =
        Batch * kBlocksPerTile * kVectorsPerBlock;
    for (std::size_t flat = threadIdx.x; flat < kTileVectors;
         flat += blockDim.x) {
      const std::size_t token = flat / (kBlocksPerTile * kVectorsPerBlock);
      const std::size_t within = flat % (kBlocksPerTile * kVectorsPerBlock);
      const std::size_t block = within / kVectorsPerBlock;
      const std::size_t vector = within % kVectorsPerBlock;
      const std::size_t source_block = tile_base + block;
      float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
      if (source_block < num_blocks) {
        value = *reinterpret_cast<const float4*>(
            x + (token * k) + (source_block * kQ8_0BlockSize) + (vector * 4));
      }
      *reinterpret_cast<float4*>(staged_x + (token * kTileStride) +
                                 (block * kStride) + (vector * 4)) = value;
    }
    __syncthreads();

    const std::size_t block = tile_base + lane;
    if (block < num_blocks) {
      const Q8_0Block* rows[RowsPerWave];
      float scale[RowsPerWave];
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
        const std::size_t row = row_base + r;
        const std::size_t safe_row = row < m ? row : m - 1;
        rows[r] = &base[(safe_row * num_blocks) + block];
        scale[r] = __half2float(rows[r]->d);
      }
      float block_dots[RowsPerWave][Batch] = {};
#pragma unroll
      for (std::size_t group = 0; group < kVectorsPerBlock; ++group) {
        std::uint32_t packed[RowsPerWave];
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          __builtin_memcpy(&packed[r], rows[r]->qs + (group * 4),
                           sizeof(std::uint32_t));
        }
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          const float4 xv = *reinterpret_cast<const float4*>(
              staged_x + (token * kTileStride) + (lane * kStride) +
              (group * 4));
#pragma unroll
          for (std::size_t r = 0; r < RowsPerWave; ++r) {
            const std::uint32_t p = packed[r];
            block_dots[r][token] +=
                static_cast<float>(static_cast<std::int8_t>(p & 0xFFU)) * xv.x;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 8U) & 0xFFU)) *
                                    xv.y;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 16U) & 0xFFU)) *
                                    xv.z;
            block_dots[r][token] += static_cast<float>(static_cast<std::int8_t>(
                                        (p >> 24U) & 0xFFU)) *
                                    xv.w;
          }
        }
      }
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          sums[r][token] += scale[r] * block_dots[r][token];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], offset);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
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
  if (detail::ShouldStoreKQuantActivationSums()) {
    QuantizeActivationToQ8_1Kernel<true>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            reinterpret_cast<const hip_bfloat16*>(bf16_x), q8_1_out, batch, k);
    LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
  } else {
    QuantizeActivationToQ8_1Kernel<false>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            reinterpret_cast<const hip_bfloat16*>(bf16_x), q8_1_out, batch, k);
    LaunchZeroQ8ActTail<false>(q8_1_out, batch, k, stream);
  }
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
  if (detail::ShouldStoreKQuantActivationSums()) {
    BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel<true>
        <<<grid, 128, 0, stream>>>(raw_out, ssm_norm, gate, q8_1_out,
                                   batch_size, inner_size, num_heads, val_dim);
    LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, inner_size, stream);
  } else {
    BatchedFusedSSMPostNormGateQuantizeQ8_1Kernel<false>
        <<<grid, 128, 0, stream>>>(raw_out, ssm_norm, gate, q8_1_out,
                                   batch_size, inner_size, num_heads, val_dim);
    LaunchZeroQ8ActTail<false>(q8_1_out, batch_size, inner_size, stream);
  }
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
  if (detail::ShouldStoreKQuantActivationSums()) {
    BatchedFusedRMSNormQuantizeQ8_1Kernel<true>
        <<<static_cast<unsigned int>(batch_size), 256, lds_bytes, stream>>>(
            x, residual, weight, sum_out, q8_1_out, batch_size, dim, eps);
    LaunchZeroQ8ActTail<true>(q8_1_out, batch_size, dim, stream);
  } else {
    BatchedFusedRMSNormQuantizeQ8_1Kernel<false>
        <<<static_cast<unsigned int>(batch_size), 256, lds_bytes, stream>>>(
            x, residual, weight, sum_out, q8_1_out, batch_size, dim, eps);
    LaunchZeroQ8ActTail<false>(q8_1_out, batch_size, dim, stream);
  }
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
  if (detail::ShouldStoreKQuantActivationSums()) {
    QuantizeActivationToQ8_1Kernel<true>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            fp32_x, q8_1_out, batch, k);
    LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
  } else {
    QuantizeActivationToQ8_1Kernel<false>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            fp32_x, q8_1_out, batch, k);
    LaunchZeroQ8ActTail<false>(q8_1_out, batch, k, stream);
  }
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
  if (detail::ShouldStoreKQuantActivationSums()) {
    BatchedFusedSwiGLUQuantizeQ8_1Kernel<true>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            gate, up, q8_1_out, batch, k);
    LaunchZeroQ8ActTail<true>(q8_1_out, batch, k, stream);
  } else {
    BatchedFusedSwiGLUQuantizeQ8_1Kernel<false>
        <<<static_cast<unsigned int>(quant_grid), quant_threads, 0, stream>>>(
            gate, up, q8_1_out, batch, k);
    LaunchZeroQ8ActTail<false>(q8_1_out, batch, k, stream);
  }
}

std::size_t QuantizedActivationBytes(std::size_t batch, std::size_t k) {
  return Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k);
}

// opt-q4kxl: exact small-batch K-quant GEMM for speculative verification.
//
// Without this, every K-quant projection at draft width fell through to
// BatchedQuantGEMVKernel, whose grid carries the batch on the y axis and so
// re-reads the entire weight matrix once per token. At the Q4_K_XL shard's
// 17.5 GB that is catastrophic: tg128-dflash2 measured 1.32 t/s against 11.6
// unspeculated.
//
// The shape is the one opt-c18x-q8-vec-smallbatch arrived at for Q8_0, for the
// same reasons. A tile of 32 sub-blocks (one per lane) of the activation batch
// is staged in LDS so the batch is read from global once per block rather than
// once per warp; the staging stride is padded from 16 to 20 floats so every
// group of four activations is one 16-byte `ds_read_b128` instead of four
// 32-bit reads; and each warp owns two output rows so those LDS reads are
// amortized over twice the weight bytes.
//
// Lane `l` owns sub-block `tile_base + l`, walking tiles in ascending order,
// which is exactly the `sb = lane_id, lane_id + 32, ...` sequence
// QuantWarpBlockDot uses. Within a sub-block the products and the activation
// sum are accumulated strictly left to right, matching the GEMV's scalar loop
// term for term. The per-(row, token) accumulation order therefore does not
// depend on the batch width, and the kernel is bit-exact with the single-token
// decode GEMV -- which is what speculative verification requires, since a
// drafted token is accepted only if the verifier reproduces exactly what the
// unspeculated decode would have emitted.
//
// Weight type is a template parameter for the same reason it is in the blocked
// WMMA kernel: DecodeQuantSub16 is a switch over every supported format, and a
// runtime type puts all eight decode bodies in the instruction stream and makes
// the allocator size for the worst one. The activation sum is accumulated for
// every format even though only Q4_K and Q5_K consume it: see the note in the
// epilogue on why specializing it away breaks bit-exactness.
//
// The occupancy hint is load bearing, and note that it points the OPPOSITE way
// from the blocked WMMA kernel above. Left unhinted this kernel settles at
// 128-176 VGPRs against its Q8_0 counterpart's 80-104; asking for ten wave32s
// per SIMD brings it to ~100 with no scratch and measured tg128-dflash2
// 17.2 -> 18.2. The two kernels differ because this one streams weights at
// draft width and is memory-latency bound, where more waves in flight is the
// whole game, while the blocked GEMM is latency bound inside its own
// dependency chain and lost 11% when the same kind of hint was applied to it.
template<std::uint32_t WavesPerBlock, std::size_t Batch,
         std::size_t RowsPerWave, core::GgmlType WType,
         std::uint32_t MinWavesPerSimd = 10, bool HoistWeightDecode = false,
         bool DropOffsetProbe = false>
__launch_bounds__(WavesPerBlock * 32, MinWavesPerSimd) __global__
    void SmallBatchKQuantExactFp32GEMMKernel(const void* __restrict__ w,
                                             const float* __restrict__ x,
                                             float* __restrict__ y,
                                             std::size_t m, std::size_t k) {
  constexpr std::size_t kSubElems = 16;
  constexpr std::size_t kSubsPerTile = 32;
  constexpr std::size_t kVectorsPerSub = kSubElems / 4;
  constexpr std::size_t kStride = kSubElems + 4;
  constexpr std::size_t kTileStride = kSubsPerTile * kStride;
  __shared__ float staged_x[Batch * kTileStride];
  // Per-(token, sub-block) activation sums, which only Q4_K and Q5_K consume
  // but every warp in the block used to recompute identically: all eight warps
  // walk the same sub-blocks and differ only in which rows they own, so the sum
  // was being derived eight times over. Computing it once during staging, in
  // the same left-to-right order, keeps it bit-exact and removes a third of the
  // inner loop's arithmetic.
  __shared__ float staged_sums[Batch * kSubsPerTile];

  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp_id = threadIdx.x >> 5u;
  const std::size_t row_base =
      ((blockIdx.x * WavesPerBlock) + warp_id) * RowsPerWave;
  const std::size_t num_sub = k / kSubElems;
  const std::size_t row_bytes = QuantRowBytes(WType, k);
  float sums[RowsPerWave][Batch] = {};

  for (std::size_t tile_base = 0; tile_base < num_sub;
       tile_base += kSubsPerTile) {
    const std::size_t sub = tile_base + lane;
    // opt-q4kxl-hoist: the weight fetch reads nothing the activation stage
    // produces, so it need not wait behind the stage's barrier. This branch
    // issues it first, putting the lane's weight loads in flight while the
    // block is still reading activations. It is a rejected route kept
    // selectable; it lives in its own array so that with the flag off the
    // retained ordering below compiles exactly as it did before the flag
    // existed.
    QuantSub16 hoisted[HoistWeightDecode ? RowsPerWave : 1];
    if constexpr (HoistWeightDecode) {
      if (sub < num_sub) {
#pragma unroll
        for (std::size_t r = 0; r < RowsPerWave; ++r) {
          const std::size_t row = row_base + r;
          const std::size_t safe_row = row < m ? row : m - 1;
          DecodeQuantSub16(
              WType,
              static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes), sub,
              hoisted[r]);
        }
      }
    }
    // One thread stages a whole (token, sub-block) so it can accumulate that
    // sub-block's activation sum sequentially while it has the values.
    for (std::size_t flat = threadIdx.x; flat < Batch * kSubsPerTile;
         flat += blockDim.x) {
      const std::size_t token = flat / kSubsPerTile;
      const std::size_t sub = flat % kSubsPerTile;
      const std::size_t source_sub = tile_base + sub;
      float* dst = staged_x + (token * kTileStride) + (sub * kStride);
      float total = 0.0F;
#pragma unroll
      for (std::size_t vector = 0; vector < kVectorsPerSub; ++vector) {
        float4 value = {0.0F, 0.0F, 0.0F, 0.0F};
        if (source_sub < num_sub) {
          value = *reinterpret_cast<const float4*>(
              x + (token * k) + (source_sub * kSubElems) + (vector * 4));
        }
        *reinterpret_cast<float4*>(dst + (vector * 4)) = value;
        // Left to right, term by term, matching the GEMV's scalar loop.
        total += value.x;
        total += value.y;
        total += value.z;
        total += value.w;
      }
      staged_sums[(token * kSubsPerTile) + sub] = total;
    }
    __syncthreads();

    if (sub < num_sub) {
      QuantSub16 decoded[RowsPerWave];
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
        if constexpr (HoistWeightDecode) {
          decoded[r] = hoisted[r];
        } else {
          const std::size_t row = row_base + r;
          const std::size_t safe_row = row < m ? row : m - 1;
          DecodeQuantSub16(
              WType,
              static_cast<const std::uint8_t*>(w) + (safe_row * row_bytes), sub,
              decoded[r]);
        }
      }
      float dots[RowsPerWave][Batch] = {};
#pragma unroll
      for (std::size_t group = 0; group < kVectorsPerSub; ++group) {
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          const float4 xv = *reinterpret_cast<const float4*>(
              staged_x + (token * kTileStride) + (lane * kStride) +
              (group * 4));
#pragma unroll
          for (std::size_t r = 0; r < RowsPerWave; ++r) {
            const std::int8_t* q = decoded[r].q + (group * 4);
            dots[r][token] += static_cast<float>(q[0]) * xv.x;
            dots[r][token] += static_cast<float>(q[1]) * xv.y;
            dots[r][token] += static_cast<float>(q[2]) * xv.z;
            dots[r][token] += static_cast<float>(q[3]) * xv.w;
          }
        }
      }
#pragma unroll
      for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
        for (std::size_t token = 0; token < Batch; ++token) {
          // The subtraction stays even for the symmetric formats, where
          // `offset` is zero. Dropping it lets the compiler contract
          // `sums += scale * dot` into a single-rounding FMA, while the decode
          // GEMV rounds the product and the sum separately -- which broke
          // bit-exactness for all five symmetric formats (33 of 64 rows
          // differed). Exactness is the contract here; the saved adds are not
          // worth it.
          //
          // opt-dflash2-verify-marginal: `DropOffsetProbe` removes the term
          // outright, which is numerically WRONG for Q4_K and Q5_K. It bounds
          // what an exact cheaper correction could be worth per verified row.
          if constexpr (DropOffsetProbe) {
            sums[r][token] += decoded[r].scale * dots[r][token];
          } else {
            sums[r][token] += (decoded[r].scale * dots[r][token]) -
                              (decoded[r].offset *
                               staged_sums[(token * kSubsPerTile) + lane]);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (std::size_t r = 0; r < RowsPerWave; ++r) {
#pragma unroll
    for (std::size_t token = 0; token < Batch; ++token) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        sums[r][token] += __shfl_xor(sums[r][token], offset);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < RowsPerWave; ++r) {
      const std::size_t row = row_base + r;
      if (row >= m) {
        continue;
      }
#pragma unroll
      for (std::size_t token = 0; token < Batch; ++token) {
        y[(token * m) + row] = sums[r][token];
      }
    }
  }
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

template<std::size_t RowsPerWave, core::GgmlType WType,
         std::uint32_t MinWavesPerSimd, bool HoistWeightDecode,
         bool DropOffsetProbe>
void LaunchKQuantSmallBatchTyped(const void* w, const float* fp32_x, float* y,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, dim3 grid, dim3 block,
                                 hipStream_t stream) {
#define GUFO_KQUANT_SMALL_BATCH(N)                                           \
  case N:                                                                    \
    SmallBatchKQuantExactFp32GEMMKernel<kKQuantSmallBatchWarps, N,           \
                                        RowsPerWave, WType, MinWavesPerSimd, \
                                        HoistWeightDecode, DropOffsetProbe>  \
        <<<grid, block, 0, stream>>>(w, fp32_x, y, m, k);                    \
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

template<std::size_t RowsPerWave, std::uint32_t MinWavesPerSimd,
         bool HoistWeightDecode, bool DropOffsetProbe>
void LaunchKQuantSmallBatchByType(core::GgmlType type, const void* w,
                                  const float* fp32_x, float* y,
                                  std::size_t batch, std::size_t m,
                                  std::size_t k, dim3 grid, dim3 block,
                                  hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ4_K,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    case core::GgmlType::kQ5_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ5_K,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    case core::GgmlType::kQ6_K:
      if constexpr (RowsPerWave == 3) {
        // Q6_K's extra per-half scale state spills 12 bytes/lane with three
        // rows. Keep it on the zero-scratch two-row reference while the other
        // formats use the candidate geometry.
        constexpr std::size_t kRowsPerBlock =
            kKQuantSmallBatchWarps * kKQuantSmallBatchRowsPerWave;
        const dim3 reference_grid(
            static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
        LaunchKQuantSmallBatchTyped<kKQuantSmallBatchRowsPerWave,
                                    core::GgmlType::kQ6_K, MinWavesPerSimd,
                                    HoistWeightDecode, DropOffsetProbe>(
            w, fp32_x, y, batch, m, k, reference_grid, block, stream);
      } else {
        LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ6_K,
                                    MinWavesPerSimd, HoistWeightDecode,
                                    DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                     grid, block, stream);
      }
      return;
    case core::GgmlType::kQ3_K:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kQ3_K,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    case core::GgmlType::kIQ4_NL:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_NL,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    case core::GgmlType::kIQ4_XS:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ4_XS,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    case core::GgmlType::kIQ3_S:
      LaunchKQuantSmallBatchTyped<RowsPerWave, core::GgmlType::kIQ3_S,
                                  MinWavesPerSimd, HoistWeightDecode,
                                  DropOffsetProbe>(w, fp32_x, y, batch, m, k,
                                                   grid, block, stream);
      return;
    default:
      return;
  }
}

// opt-q4kxl: tile-configuration dispatch for WKQuantA8BlockedWmmaGEMMKernel.
// The configurations mirror the Q8_0 launcher exactly so the two paths share
// the same occupancy tuning.
template<core::GgmlType WType, bool UsePrecomputedSums,
         bool StagePrecomputedSums>
void LaunchKQuantWmmaGEMMTypedImpl(const void* w, const void* q8_1_x, float* y,
                                   std::size_t batch, std::size_t m,
                                   std::size_t k, hipStream_t stream) {
  if (batch <= 8 && UseCompactW8A8SmallBatchTile()) {
    constexpr int kBM = 128;
    const dim3 grid(1, static_cast<unsigned int>((m + kBM - 1) / kBM));
    if (m >= (3 * k)) {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 32, 4, 4, 2, WType,
                                     UsePrecomputedSums, StagePrecomputedSums>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    } else {
      WKQuantA8BlockedWmmaGEMMKernel<kBM, 16, 4, 8, 1, WType,
                                     UsePrecomputedSums, StagePrecomputedSums>
          <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    }
    return;
  }
  if (batch >= 96) {
    constexpr int kBM = 128;
    // opt-q4kxl-wide: 128x256 over sixteen waves. Same 64-register accumulator
    // as the 256-thread tile, half the weight re-decode count.
    const auto tile = detail::KQuantPrefillTileFromEnv();
    if (tile == detail::KQuantPrefillTile::kNarrowRows) {
      // opt-q4kxl-wide, second shape: BN=256 with BM=64 keeps BM*BN/256 = 64
      // accumulator registers and the eight-wave workgroup, so the CU can
      // still hold three blocks. BK has to drop to 1 because a 256-token
      // activation stage is 8 KB per K block.
      constexpr int kNarrowBM = 64;
      constexpr int kNarrowBN = 256;
      const dim3 narrow_grid(
          static_cast<unsigned int>((batch + kNarrowBN - 1) / kNarrowBN),
          static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
      WKQuantA8BlockedWmmaGEMMKernel<kNarrowBM, kNarrowBN, 1, 2, 4, WType,
                                     UsePrecomputedSums, StagePrecomputedSums>
          <<<narrow_grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
      return;
    }
    if (tile == detail::KQuantPrefillTile::kFusedOffset) {
      constexpr int kBN = 128;
      const dim3 fused_grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                            static_cast<unsigned int>((m + kBM - 1) / kBM));
      WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType,
                                     UsePrecomputedSums, StagePrecomputedSums,
                                     false, false, true>
          <<<fused_grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
      return;
    }
    if (tile == detail::KQuantPrefillTile::kSingleKBlock) {
      constexpr int kBN = 128;
      const dim3 bk1_grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                          static_cast<unsigned int>((m + kBM - 1) / kBM));
      WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 1, 4, 2, WType,
                                     UsePrecomputedSums, StagePrecomputedSums>
          <<<bk1_grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
      return;
    }
    if (tile == detail::KQuantPrefillTile::kDropOffsetProbe) {
      constexpr int kBN = 128;
      const dim3 probe_grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                            static_cast<unsigned int>((m + kBM - 1) / kBM));
      WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType,
                                     UsePrecomputedSums, StagePrecomputedSums,
                                     false, true>
          <<<probe_grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
      return;
    }
    if (tile != detail::KQuantPrefillTile::kDefault) {
      constexpr int kWideBN = 256;
      const dim3 wide_grid(
          static_cast<unsigned int>((batch + kWideBN - 1) / kWideBN),
          static_cast<unsigned int>((m + kBM - 1) / kBM));
      if (tile == detail::KQuantPrefillTile::kWideForcedOccupancy) {
        WKQuantA8BlockedWmmaGEMMKernel<kBM, kWideBN, 2, 4, 4, WType,
                                       UsePrecomputedSums, StagePrecomputedSums,
                                       true>
            <<<wide_grid, 512, 0, stream>>>(w, q8_1_x, y, batch, m, k);
        return;
      }
      WKQuantA8BlockedWmmaGEMMKernel<kBM, kWideBN, 2, 4, 4, WType,
                                     UsePrecomputedSums, StagePrecomputedSums>
          <<<wide_grid, 512, 0, stream>>>(w, q8_1_x, y, batch, m, k);
      return;
    }
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    // WM=4/WN=2 (2 row subtiles and 4 token subtiles per wave). The mirror
    // shape WM=2/WN=4 halves the per-token activation sums the offset term
    // needs, but measured far worse -- pp2048 464 -> 372 -- because four row
    // subtiles per wave hold twice the A-fragment and scale state.
    WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2, WType, UsePrecomputedSums,
                                   StagePrecomputedSums>
        <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
    return;
  }
  constexpr int kBM = 128;
  constexpr int kBN = 64;
  const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                  static_cast<unsigned int>((m + kBM - 1) / kBM));
  WKQuantA8BlockedWmmaGEMMKernel<kBM, kBN, 4, 8, 1, WType, UsePrecomputedSums,
                                 StagePrecomputedSums>
      <<<grid, 256, 0, stream>>>(w, q8_1_x, y, batch, m, k);
}

template<core::GgmlType WType>
void LaunchKQuantWmmaGEMMTyped(const void* w, const void* q8_1_x, float* y,
                               std::size_t batch, std::size_t m, std::size_t k,
                               hipStream_t stream) {
  switch (detail::KQuantActivationSumModeFromEnv()) {
    case detail::KQuantActivationSumMode::kStaged:
      LaunchKQuantWmmaGEMMTypedImpl<WType, true, true>(w, q8_1_x, y, batch, m,
                                                       k, stream);
      return;
    case detail::KQuantActivationSumMode::kDirect:
      LaunchKQuantWmmaGEMMTypedImpl<WType, true, false>(w, q8_1_x, y, batch, m,
                                                        k, stream);
      return;
    case detail::KQuantActivationSumMode::kReference:
      LaunchKQuantWmmaGEMMTypedImpl<WType, false, false>(w, q8_1_x, y, batch, m,
                                                         k, stream);
      return;
  }
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
  // opt-q4kxl: K-quant/IQ formats share one weight stream across the whole
  // draft block instead of re-reading it per token.
  if (::gufo::hip::detail::IsNativeWmmaQuant(type) && batch <= 8) {
    // Exact width instantiations: a padded Batch would stage and multiply
    // activations that no output consumes.
    constexpr std::size_t kKWarpsPerBlock = kKQuantSmallBatchWarps;
    const dim3 block(static_cast<unsigned int>(kWarp * kKWarpsPerBlock));
    // The four-row route stays on the ten-wave hint. It is a rejected geometry
    // kept selectable, and at twelve waves the allocator caps it at 120 VGPR
    // and spills 12-88 private bytes/lane for every format; the two- and
    // three-row geometries reach 96-112 VGPR with no scratch.
    const bool twelve_waves = detail::KQuantSmallBatchOccupancyFromEnv() ==
                              detail::KQuantSmallBatchOccupancy::kTwelveWaves;
    const bool hoist = detail::KQuantSmallBatchFetchFromEnv() ==
                       detail::KQuantSmallBatchFetch::kHoisted;
    // Measurement-only: deletes the Q4_K/Q5_K minimum correction, which makes
    // the result wrong. It bounds what an exact cheaper correction is worth in
    // the speculative verifier.
    const bool drop_offset = detail::KQuantSmallBatchOffsetFromEnv() ==
                             detail::KQuantSmallBatchOffset::kDropProbe;
    const auto dispatch_rows = [&]<std::size_t RowsPerWave>() {
      constexpr std::size_t kRowsPerBlock = kKWarpsPerBlock * RowsPerWave;
      const dim3 grid(
          static_cast<unsigned int>((m + kRowsPerBlock - 1) / kRowsPerBlock));
      const auto launch = [&]<std::uint32_t Waves, bool Hoist>() {
        if (drop_offset) {
          LaunchKQuantSmallBatchByType<RowsPerWave, Waves, Hoist, true>(
              type, w, fp32_x, y, batch, m, k, grid, block, stream);
          return;
        }
        LaunchKQuantSmallBatchByType<RowsPerWave, Waves, Hoist, false>(
            type, w, fp32_x, y, batch, m, k, grid, block, stream);
      };
      const bool waves12 = twelve_waves && RowsPerWave != 4;
      if (waves12 && hoist) {
        launch.template operator()<12, true>();
      } else if (waves12) {
        launch.template operator()<12, false>();
      } else if (hoist) {
        launch.template operator()<10, true>();
      } else {
        launch.template operator()<10, false>();
      }
    };
    if (detail::KQuantSmallBatchRowsFromEnv() ==
        detail::KQuantSmallBatchRows::kFour) {
      dispatch_rows.template operator()<4>();
    } else if (detail::KQuantSmallBatchRowsFromEnv() ==
               detail::KQuantSmallBatchRows::kThree) {
      dispatch_rows.template operator()<3>();
    } else {
      dispatch_rows.template operator()<kKQuantSmallBatchRowsPerWave>();
    }
    return;
  }
  if ((type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K) &&
      batch >= 1 && batch <= 8 && UseExactSharedQ8SmallBatchFp32()) {
    const dim3 block(static_cast<unsigned int>(kWarp * kWarpsPerBlock));
    // The LDS-staged route streams each weight row once for the whole batch, so
    // it is the fastest option at every width. Narrow batches used to fall back
    // to per-width kernels that re-read the weight stream, which made adaptive
    // draft widths far more expensive than width 8.
    if (type == core::GgmlType::kQ8_0) {
      constexpr std::size_t kVecWarpsPerBlock = 16;
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

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
