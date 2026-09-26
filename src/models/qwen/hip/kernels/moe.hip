#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kMoeWave = 32;

__device__ inline float MoeBlockSum(float v, float* shared) {
  for (int offset = kMoeWave / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int lane = threadIdx.x % kMoeWave;
  const int warp = threadIdx.x / kMoeWave;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0F;
  for (std::uint32_t w = 0; w < blockDim.x / kMoeWave; ++w) {
    total += shared[w];
  }
  return total;
}

__device__ inline float MoeBlockMax(float v, float* shared) {
  for (int offset = kMoeWave / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  const int lane = threadIdx.x % kMoeWave;
  const int warp = threadIdx.x / kMoeWave;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (std::uint32_t w = 0; w < blockDim.x / kMoeWave; ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

/// Softmax over n_experts router logits followed by top-k selection. One block
/// per token; the first wave runs the k selection rounds. Mirrors the CPU
/// reference: full-set softmax, lowest-index tie-break, renormalized weights
/// with the sum floored at 2^-14.
template<std::uint32_t MaxExperts>
__global__ void MoeRouterTopKKernel(const float* __restrict__ logits,
                                    std::uint32_t stride,
                                    std::int32_t* __restrict__ ids,
                                    float* __restrict__ weights,
                                    std::uint32_t n_experts,
                                    std::uint32_t k) {
  __shared__ float probs[MaxExperts];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = MoeBlockMax(local_max, shared);
  float local_sum = 0.0F;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = MoeBlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();

  if (threadIdx.x >= kMoeWave) {
    return;
  }
  constexpr std::uint32_t kItems = MaxExperts / kMoeWave;
  float values[kItems];
#pragma unroll
  for (std::uint32_t j = 0; j < kItems; ++j) {
    const std::uint32_t e = threadIdx.x + j * kMoeWave;
    values[j] = e < n_experts ? probs[e] : -1.0F;
  }
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float best = -1.0F;
    std::uint32_t index = 0xFFFFFFFFU;
#pragma unroll
    for (std::uint32_t j = 0; j < kItems; ++j) {
      const std::uint32_t e = threadIdx.x + j * kMoeWave;
      if (values[j] > best) {
        best = values[j];
        index = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor(best, offset);
      const std::uint32_t oi = __shfl_xor(index, offset);
      if (other > best || (other == best && oi < index)) {
        best = other;
        index = oi;
      }
    }
    if (threadIdx.x == 0) {
      chosen[slot] = index;
      chosen_p[slot] = best;
    }
#pragma unroll
    for (std::uint32_t j = 0; j < kItems; ++j) {
      if (threadIdx.x + j * kMoeWave == index) {
        values[j] = -1.0F;
      }
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5F);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}

__device__ inline float MoeSigmoid(float x) {
  return 1.0F / (1.0F + __expf(-x));
}

/// Four output lanes per thread; grid (tokens, dim / (threads * 4)).
__global__ void MoeEpilogueKernel(const float* __restrict__ expert_out,
                                  const float* __restrict__ weights,
                                  const float* __restrict__ shared_out,
                                  const float* __restrict__ shared_gate,
                                  float* __restrict__ out, std::uint32_t k,
                                  std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  const float* rows =
      expert_out + static_cast<std::size_t>(t) * k * dim + i;
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const float4 v = *reinterpret_cast<const float4*>(
        rows + static_cast<std::size_t>(s) * dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float g = MoeSigmoid(shared_gate[t]);
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  acc.x += g * sh.x;
  acc.y += g * sh.y;
  acc.z += g * sh.z;
  acc.w += g * sh.w;
  *reinterpret_cast<float4*>(out + idx) = acc;
}

/// Byte size of one expert matrix inside a stacked [cols, rows, experts]
/// payload for the formats this fallback decodes.
__device__ inline std::size_t MoeExpertBytes(core::GgmlType type,
                                             std::size_t m, std::size_t k) {
  if (type == core::GgmlType::kBF16) {
    return m * k * sizeof(hip_bfloat16);
  }
  if (type == core::GgmlType::kF32) {
    return m * k * sizeof(float);
  }
  return m * QuantRowBytes(type, k);
}

/// Warp-cooperative row dot for BF16/F32 weights (the quant formats go through
/// QuantWarpBlockDot). The result is valid in every lane.
template<typename TX>
__device__ inline float MoeWarpDenseRowDot(core::GgmlType type,
                                           const void* __restrict__ row,
                                           const TX* __restrict__ x,
                                           std::size_t k,
                                           std::uint32_t lane) {
  float sum = 0.0F;
  if (type == core::GgmlType::kBF16) {
    const auto* r = static_cast<const hip_bfloat16*>(row);
    for (std::size_t i = lane; i < k; i += kMoeWave) {
      sum += static_cast<float>(r[i]) * static_cast<float>(x[i]);
    }
  } else {
    const auto* r = static_cast<const float*>(row);
    for (std::size_t i = lane; i < k; i += kMoeWave) {
      sum += r[i] * static_cast<float>(x[i]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, offset);
  }
  return sum;
}

/// Row m of expert e's matrix dotted with x, valid in every lane.
template<typename TX>
__device__ inline float MoeWarpRowDot(core::GgmlType type,
                                      const void* __restrict__ expert_base,
                                      std::size_t m, const TX* __restrict__ x,
                                      std::size_t k, std::uint32_t lane) {
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    const std::size_t row_bytes =
        k * (type == core::GgmlType::kBF16 ? sizeof(hip_bfloat16)
                                           : sizeof(float));
    const auto* row =
        static_cast<const std::uint8_t*>(expert_base) + m * row_bytes;
    return MoeWarpDenseRowDot(type, row, x, k, lane);
  }
  return QuantWarpBlockDot<TX>(type, expert_base, m, x, k, lane, 0, 1);
}

/// Routed per-slot GEMV: grid.y = slot, one warp per output row.
__global__ void MoeSlotGemvKernel(const void* __restrict__ w,
                                  core::GgmlType type,
                                  const float* __restrict__ x,
                                  const std::int32_t* __restrict__ ids,
                                  float* __restrict__ out, std::size_t m,
                                  std::size_t k, std::uint32_t x_divisor) {
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % kMoeWave;
  const std::uint32_t warp = threadIdx.x / kMoeWave;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * (blockDim.x / kMoeWave)) + warp;
  if (row >= m) {
    return;
  }
  const std::uint32_t expert = static_cast<std::uint32_t>(ids[slot]);
  const auto* base = static_cast<const std::uint8_t*>(w) +
                     static_cast<std::size_t>(expert) *
                         MoeExpertBytes(type, m, k);
  const float* x_row =
      x + static_cast<std::size_t>(x_divisor == 0 ? slot : slot / x_divisor) *
              k;
  const float sum = MoeWarpRowDot(type, base, row, x_row, k, lane);
  if (lane == 0) {
    out[static_cast<std::size_t>(slot) * m + row] = sum;
  }
}

/// Routed fused SwiGLU GEMV: grid.y = slot, one warp per intermediate row.
__global__ void MoeSlotSwigluGemvKernel(const void* __restrict__ gate_w,
                                        core::GgmlType gate_type,
                                        const void* __restrict__ up_w,
                                        core::GgmlType up_type,
                                        const float* __restrict__ x,
                                        const std::int32_t* __restrict__ ids,
                                        float* __restrict__ act,
                                        std::size_t expert_ff,
                                        std::size_t hidden,
                                        std::uint32_t x_divisor) {
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % kMoeWave;
  const std::uint32_t warp = threadIdx.x / kMoeWave;
  const std::size_t row =
      (static_cast<std::size_t>(blockIdx.x) * (blockDim.x / kMoeWave)) + warp;
  if (row >= expert_ff) {
    return;
  }
  const std::uint32_t expert = static_cast<std::uint32_t>(ids[slot]);
  const auto* gate_base =
      static_cast<const std::uint8_t*>(gate_w) +
      static_cast<std::size_t>(expert) *
          MoeExpertBytes(gate_type, expert_ff, hidden);
  const auto* up_base =
      static_cast<const std::uint8_t*>(up_w) +
      static_cast<std::size_t>(expert) *
          MoeExpertBytes(up_type, expert_ff, hidden);
  const float* x_row =
      x + static_cast<std::size_t>(x_divisor == 0 ? slot : slot / x_divisor) *
              hidden;
  const float g = MoeWarpRowDot(gate_type, gate_base, row, x_row, hidden, lane);
  const float u = MoeWarpRowDot(up_type, up_base, row, x_row, hidden, lane);
  if (lane == 0) {
    act[static_cast<std::size_t>(slot) * expert_ff + row] =
        (g / (1.0F + __expf(-g))) * u;
  }
}

constexpr std::uint32_t kMoeGemvWavesPerBlock = 4;

// Grouped BF16 expert GEMM for prefill. The per-slot GEMV above re-reads a
// whole expert matrix for every (token, expert) slot, which is right for one
// decode token and ruinous for a prefill chunk (UD quants keep a few layers'
// experts in BF16). Here the slots are first grouped by expert, then each
// block multiplies up to kGroupedTileRows slots of one expert against a
// kGroupedTileCols-row slice of that expert's weights on the WMMA matrix
// cores, so each weight tile is read once per group of slots.
//
// WMMA wave32 fragment layout (as in the prefill attention kernel): A holds
// row L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous k, and C
// element i is row 2i + L/16, column L%16. A is the activation rows, B the
// weight rows (one per output column), so both are staged k-contiguous.
constexpr std::uint32_t kGroupedTileRows = 32;
constexpr std::uint32_t kGroupedTileCols = 64;
constexpr std::uint32_t kGroupedTileK = 64;
constexpr std::uint32_t kGroupedLdsStride = kGroupedTileK + 8;
constexpr std::uint32_t kGroupedThreads = 256;
constexpr std::uint32_t kGroupedMaxExperts = 1024;
// Tile entries pack (expert << kGroupedTileShift) | first row in the expert.
constexpr std::uint32_t kGroupedTileShift = 20;

using GroupedFragment = __attribute__((ext_vector_type(16))) unsigned short;
using GroupedAccumulator = __attribute__((ext_vector_type(8))) float;

__device__ inline GroupedFragment GroupedLoadFragment(
    const unsigned short* p) {
  union {
    GroupedFragment f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// Round-to-nearest-even FP32 -> BF16 bits (activations are finite).
__device__ inline unsigned short GroupedBf16Bits(float v) {
  std::uint32_t u = __float_as_uint(v);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<unsigned short>(u >> 16);
}

/// One block. Counts slots per expert, writes the expert-major slot order,
/// each expert's [begin, end) range, and one tile entry per kGroupedTileRows
/// slots of an expert; entries past the last tile are -1 up to grid_tiles.
__global__ void MoeGroupSlotsKernel(const std::int32_t* __restrict__ ids,
                                    std::uint32_t slots,
                                    std::uint32_t n_experts,
                                    std::int32_t* __restrict__ sorted_slots,
                                    std::int32_t* __restrict__ tiles,
                                    std::int32_t* __restrict__ expert_bounds,
                                    std::uint32_t grid_tiles) {
  __shared__ std::uint32_t count[kGroupedMaxExperts];
  __shared__ std::uint32_t cursor[kGroupedMaxExperts];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    count[e] = 0;
  }
  __syncthreads();
  for (std::uint32_t s = threadIdx.x; s < slots; s += blockDim.x) {
    atomicAdd(&count[ids[s]], 1U);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint32_t offset = 0;
    std::uint32_t tile = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      expert_bounds[e] = static_cast<std::int32_t>(offset);
      cursor[e] = offset;
      for (std::uint32_t r = 0; r < count[e]; r += kGroupedTileRows) {
        tiles[tile++] =
            static_cast<std::int32_t>((e << kGroupedTileShift) | r);
      }
      offset += count[e];
    }
    expert_bounds[n_experts] = static_cast<std::int32_t>(offset);
    for (; tile < grid_tiles; ++tile) {
      tiles[tile] = -1;
    }
  }
  __syncthreads();
  // Order within an expert is arbitrary; each output row depends only on its
  // own activation row, so the result does not.
  for (std::uint32_t s = threadIdx.x; s < slots; s += blockDim.x) {
    const std::uint32_t pos = atomicAdd(&cursor[ids[s]], 1U);
    sorted_slots[pos] = static_cast<std::int32_t>(s);
  }
}

/// grid (tiles, m / kGroupedTileCols); eight waves as 2 row x 4 column 16x16
/// output tiles. kDense multiplies one [m, k] matrix by dense_rows activation
/// rows instead, with row tile blockIdx.x and no grouping.
template<bool kDense>
__launch_bounds__(kGroupedThreads) __global__ void MoeGroupedBf16GemmKernel(
    const unsigned short* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ out, const std::int32_t* __restrict__ sorted_slots,
    const std::int32_t* __restrict__ tiles,
    const std::int32_t* __restrict__ expert_bounds, std::uint32_t m,
    std::uint32_t k, std::uint32_t x_divisor, std::uint32_t dense_rows) {
  std::uint32_t expert = 0;
  std::uint32_t first = 0;
  std::uint32_t end = 0;
  if constexpr (kDense) {
    first = blockIdx.x * kGroupedTileRows;
    end = dense_rows;
  } else {
    const std::int32_t entry = tiles[blockIdx.x];
    if (entry < 0) {
      return;
    }
    expert = static_cast<std::uint32_t>(entry) >> kGroupedTileShift;
    first = static_cast<std::uint32_t>(expert_bounds[expert]) +
            (static_cast<std::uint32_t>(entry) &
             ((1U << kGroupedTileShift) - 1U));
    end = static_cast<std::uint32_t>(expert_bounds[expert + 1]);
  }
  const std::uint32_t col0 = blockIdx.y * kGroupedTileCols;

  __shared__ unsigned short a_lds[kGroupedTileRows][kGroupedLdsStride];
  __shared__ unsigned short b_lds[kGroupedTileCols][kGroupedLdsStride];
  __shared__ std::int32_t slot_of[kGroupedTileRows];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid % 32U;
  const std::uint32_t wave = tid / 32U;
  if (tid < kGroupedTileRows) {
    if (first + tid >= end) {
      slot_of[tid] = -1;
    } else if constexpr (kDense) {
      slot_of[tid] = static_cast<std::int32_t>(first + tid);
    } else {
      slot_of[tid] = sorted_slots[first + tid];
    }
  }
  __syncthreads();

  // A staging: 32 rows x 64 k, eight floats per thread.
  const std::uint32_t a_row = tid / 8U;
  const std::uint32_t a_col = (tid % 8U) * 8U;
  const std::int32_t a_slot = slot_of[a_row];
  const float* a_src =
      a_slot < 0 ? nullptr
                 : x + static_cast<std::size_t>(
                           x_divisor == 0 ? static_cast<std::uint32_t>(a_slot)
                                          : static_cast<std::uint32_t>(a_slot) /
                                                x_divisor) *
                           k;
  const unsigned short* w_tile =
      w + (static_cast<std::size_t>(expert) * m + col0) * k;

  const std::uint32_t row_block = wave / 4U;
  const std::uint32_t col_block = wave % 4U;
  GroupedAccumulator acc = {};
  for (std::uint32_t k0 = 0; k0 < k; k0 += kGroupedTileK) {
    {
      union {
        uint4 u;
        unsigned short h[8];
      } packed;
      if (a_src != nullptr) {
        const auto* src = reinterpret_cast<const float4*>(a_src + k0 + a_col);
        const float4 lo = src[0];
        const float4 hi = src[1];
        packed.h[0] = GroupedBf16Bits(lo.x);
        packed.h[1] = GroupedBf16Bits(lo.y);
        packed.h[2] = GroupedBf16Bits(lo.z);
        packed.h[3] = GroupedBf16Bits(lo.w);
        packed.h[4] = GroupedBf16Bits(hi.x);
        packed.h[5] = GroupedBf16Bits(hi.y);
        packed.h[6] = GroupedBf16Bits(hi.z);
        packed.h[7] = GroupedBf16Bits(hi.w);
      } else {
        packed.u = make_uint4(0U, 0U, 0U, 0U);
      }
      *reinterpret_cast<uint4*>(&a_lds[a_row][a_col]) = packed.u;
    }
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i) {
      const std::uint32_t idx = tid + i * kGroupedThreads;
      const std::uint32_t b_row = idx / 8U;
      const std::uint32_t b_col = (idx % 8U) * 8U;
      *reinterpret_cast<uint4*>(&b_lds[b_row][b_col]) =
          *reinterpret_cast<const uint4*>(
              w_tile + static_cast<std::size_t>(b_row) * k + k0 + b_col);
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t ks = 0; ks < kGroupedTileK; ks += 16) {
      const GroupedFragment a =
          GroupedLoadFragment(&a_lds[row_block * 16 + lane % 16U][ks]);
      const GroupedFragment b =
          GroupedLoadFragment(&b_lds[col_block * 16 + lane % 16U][ks]);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    }
    __syncthreads();
  }

  const std::uint32_t col = col0 + col_block * 16 + lane % 16U;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::int32_t slot = slot_of[row_block * 16 + 2 * i + lane / 16U];
    if (slot >= 0) {
      out[static_cast<std::size_t>(slot) * m + col] = acc[i];
    }
  }
}

}  // namespace

bool IsMoeGroupedBf16GemmSupported(std::size_t m, std::size_t k,
                                   std::uint32_t n_experts) noexcept {
  return m != 0 && k != 0 && m % kGroupedTileCols == 0 &&
         k % kGroupedTileK == 0 && n_experts != 0 &&
         n_experts <= kGroupedMaxExperts;
}

namespace {

[[nodiscard]] std::uint32_t GroupedGridTiles(std::uint32_t slots,
                                             std::uint32_t n_experts) {
  // Every tile holds at least one slot, and each expert adds at most one
  // partial tile, so this bounds the tile count from above.
  const std::uint32_t bound =
      (slots + kGroupedTileRows - 1) / kGroupedTileRows + n_experts;
  return bound < slots ? bound : slots;
}

}  // namespace

void LaunchMoeGroupSlots(const std::int32_t* ids, std::uint32_t slots,
                         std::uint32_t n_experts,
                         const MoeGroupedScratch& scratch,
                         hipStream_t stream) {
  MoeGroupSlotsKernel<<<1, 1024, 0, stream>>>(
      ids, slots, n_experts, scratch.sorted_slots, scratch.tiles,
      scratch.expert_bounds, GroupedGridTiles(slots, n_experts));
}

void LaunchMoeGroupedBf16Gemm(const void* w, const float* x, float* out,
                              std::size_t m, std::size_t k,
                              std::uint32_t slots, std::uint32_t n_experts,
                              std::uint32_t x_token_divisor,
                              const MoeGroupedScratch& scratch,
                              hipStream_t stream) {
  if (slots == 0) {
    return;
  }
  const dim3 grid(GroupedGridTiles(slots, n_experts),
                  static_cast<std::uint32_t>(m / kGroupedTileCols));
  MoeGroupedBf16GemmKernel<false><<<grid, kGroupedThreads, 0, stream>>>(
      static_cast<const unsigned short*>(w), x, out, scratch.sorted_slots,
      scratch.tiles, scratch.expert_bounds, static_cast<std::uint32_t>(m),
      static_cast<std::uint32_t>(k), x_token_divisor, 0);
}

void LaunchBf16WmmaGemm(const void* w, const float* x, float* out,
                        std::size_t batch_size, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (batch_size == 0) {
    return;
  }
  const dim3 grid(static_cast<std::uint32_t>(
                      (batch_size + kGroupedTileRows - 1) / kGroupedTileRows),
                  static_cast<std::uint32_t>(m / kGroupedTileCols));
  MoeGroupedBf16GemmKernel<true><<<grid, kGroupedThreads, 0, stream>>>(
      static_cast<const unsigned short*>(w), x, out, nullptr, nullptr, nullptr,
      static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k), 0,
      static_cast<std::uint32_t>(batch_size));
}

void LaunchMoeRouterTopK(const float* logits, std::uint32_t logits_stride,
                         std::int32_t* ids, float* weights,
                         std::uint32_t n_tokens, std::uint32_t n_experts,
                         std::uint32_t n_used, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  if (n_experts <= 256) {
    MoeRouterTopKKernel<256><<<n_tokens, kThreads, 0, stream>>>(
        logits, logits_stride, ids, weights, n_experts, n_used);
  } else {
    MoeRouterTopKKernel<1024><<<n_tokens, kThreads, 0, stream>>>(
        logits, logits_stride, ids, weights, n_experts, n_used);
  }
}

void LaunchMoeEpilogue(const float* expert_out, const float* weights,
                       const float* shared_out, const float* shared_gate,
                       float* out, std::uint32_t n_tokens, std::uint32_t n_used,
                       std::uint32_t dim, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  const std::uint32_t lanes = dim / 4;
  const dim3 grid(n_tokens, (lanes + kThreads - 1) / kThreads);
  MoeEpilogueKernel<<<grid, kThreads, 0, stream>>>(
      expert_out, weights, shared_out, shared_gate, out, n_used, dim);
}

void LaunchMoeSlotGemv(const void* w, core::GgmlType type, const float* x,
                       const std::int32_t* ids, float* out, std::size_t m,
                       std::size_t k, std::uint32_t slots,
                       std::uint32_t x_token_divisor, hipStream_t stream) {
  const dim3 grid(
      static_cast<std::uint32_t>((m + kMoeGemvWavesPerBlock - 1) /
                                 kMoeGemvWavesPerBlock),
      slots);
  MoeSlotGemvKernel<<<grid, kMoeGemvWavesPerBlock * kMoeWave, 0, stream>>>(
      w, type, x, ids, out, m, k, x_token_divisor);
}

void LaunchMoeSlotSwigluGemv(const void* gate_w, core::GgmlType gate_type,
                             const void* up_w, core::GgmlType up_type,
                             const float* x, const std::int32_t* ids,
                             float* act, std::size_t expert_ff,
                             std::size_t hidden, std::uint32_t slots,
                             std::uint32_t x_token_divisor,
                             hipStream_t stream) {
  const dim3 grid(
      static_cast<std::uint32_t>((expert_ff + kMoeGemvWavesPerBlock - 1) /
                                 kMoeGemvWavesPerBlock),
      slots);
  MoeSlotSwigluGemvKernel<<<grid, kMoeGemvWavesPerBlock * kMoeWave, 0,
                            stream>>>(gate_w, gate_type, up_w, up_type, x, ids,
                                      act, expert_ff, hidden, x_token_divisor);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
