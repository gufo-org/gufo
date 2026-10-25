// Decode and speculative-verification attention on the WMMA matrix cores for
// eight query heads per KV head (Qwen3.6-35B-A3B: 16 query heads over two KV
// heads, head_dim 256).
//
// The scalar split-K kernel spends its time on per-position dot products and
// lane reductions, not on reading the cache. Here a block takes one KV head
// and one key partition and treats its eight query heads at up to eight
// positions as a 64-row query tile, the same tile the prefill kernel
// (attention_wmma.hip) uses: eight waves split four 16-row blocks by two
// halves of the head dimension, K is staged per 16-key tile and V transposed
// in LDS, with the next tile prefetched into registers. Row j is query head
// (8 * kv_head + j % 8) at position start_pos + j / 8.
//
// A row's result depends on that row alone: its query, its causal limit and
// the absolute key tiling. Partitions are fixed runs of keys chosen from the
// last position, empty partitions contribute nothing, and the tile that holds
// a row's own position is summed over that row's visible keys with scalar
// FMAs, so a verification row and the same position decoded on its own get
// the same bits.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGroup = 8;
#ifndef GUFO_WMMA_DECODE_POSITIONS
#define GUFO_WMMA_DECODE_POSITIONS 4
#endif
constexpr std::uint32_t kPositions = GUFO_WMMA_DECODE_POSITIONS;
constexpr std::uint32_t kRows = kGroup * kPositions;
constexpr std::uint32_t kRowBlocks = kRows / 16;
// Eight waves: every row block's score tile is split over the head dimension.
constexpr std::uint32_t kDimSplits = 8 / kRowBlocks;
constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kKSteps = kHeadDim / 16;
constexpr std::uint32_t kKStepsPerWave = kKSteps / kDimSplits;
constexpr std::uint32_t kKStride = kHeadDim + 8;
constexpr std::uint32_t kVtStride = kKeys + 8;
constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
constexpr std::uint32_t kMinPartition = 256;
static_assert(kRowBlocks * kDimSplits == 8 && kKRegs == 2 && kVRegs == 2 &&
              kKeys % kSoftmaxLanes == 0);

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

// Keys per partition for a context of `length` positions: at least 256 and a
// power of two times 256, at most 32 partitions. Fewer, longer partitions
// stream the cache faster than many short ones: at 104k positions 26
// partitions read it at 227 GB/s, 102 at 122 GB/s and 405 at 105 GB/s. Constant
// over long runs of lengths, so the rows of one verification step normally
// share it.
std::uint32_t DecodePartition(std::size_t length) {
  std::uint32_t partition = kMinPartition;
  while ((length + partition - 1) / partition >
         detail::kSplitKDecodeAttentionMaxSplits)
    partition *= 2;
  return partition;
}

}  // namespace

// Writes, per row and partition, the running maximum, the running sum and the
// unnormalized output in the split-K scratch layout that
// QwenDecodeSplitKAttentionReduceKernel merges.
__launch_bounds__(256, 2) __global__ void WmmaDecodeAttentionPartials(
    const float* __restrict__ q, const __half* __restrict__ k_layer,
    const __half* __restrict__ v_layer, float* __restrict__ scratch,
    std::uint32_t start_pos, std::uint32_t positions, std::uint32_t partition,
    std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t split = blockIdx.x;
  const std::uint32_t kv_head = blockIdx.y;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * kHeadDim;
  const std::size_t attn_width = static_cast<std::size_t>(num_heads) * kHeadDim;
  constexpr float attention_scale = 1.0F / 16.0F;

  const std::uint32_t visible_end = start_pos + positions;
  const std::uint32_t key_begin = split * partition;
  const std::uint32_t key_end = min(key_begin + partition, visible_end);
  const auto row_position = [](std::uint32_t row) { return row / kGroup; };
  const auto row_head = [&](std::uint32_t row) {
    return kv_head * kGroup + (row % kGroup);
  };

  __shared__ __half kv_lds[kHeadDim * kVtStride];
  __shared__ float s_lds[kDimSplits][kRowBlocks][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];
  static_assert(kHeadDim * kVtStride >= kKeys * kKStride);

  // Wave w computes the score tile of row block w % kRowBlocks over part
  // w / kRowBlocks of the head dimension.
  const std::uint32_t s_rb = wave % kRowBlocks;
  const std::uint32_t s_kh = wave / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t row = s_rb * 16 + sub;
    const bool live = row_position(row) < positions;
    const float* q_row =
        q + static_cast<std::size_t>(row_position(row)) * attn_width +
        static_cast<std::size_t>(row_head(row)) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dimension tiles `wave` and `wave + 8` of every row block.
  v8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* k_base = k_layer + static_cast<std::size_t>(kv_head) * kHeadDim;
  const auto* v_base =
      v_layer + static_cast<std::size_t>(kv_head) * kHeadDim + v_slice;
  // Keys at or past key_end are never read: positions past the last row may
  // not be written yet.
  const auto load_v = [&](std::uint32_t key_start, uint4* dst) {
    const std::uint32_t key = key_start + v_key;
    const auto* src = v_base + static_cast<std::size_t>(key) * kv_width;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j)
      dst[j] = key < key_end ? *reinterpret_cast<const uint4*>(src + j * 8)
                             : make_uint4(0u, 0u, 0u, 0u);
  };
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + n * 256;
      const std::uint32_t key = key_start + idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] = key < key_end
                   ? *reinterpret_cast<const uint4*>(
                         k_base + static_cast<std::size_t>(key) * kv_width + d8)
                   : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs] = {};
  uint4 v_pre[kVRegs] = {};
  if (key_begin < key_end) {
    load_k(key_begin, k_cur);
    load_v(key_begin, v_cur);
  }

  for (std::uint32_t key_start = key_begin; key_start < key_end;
       key_start += kKeys) {
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + n * 256;
      *reinterpret_cast<uint4*>(&kv_lds[(idx / (kHeadDim / 8)) * kKStride +
                                        (idx % (kHeadDim / 8)) * 8]) = k_cur[n];
    }
    __syncthreads();

    const std::uint32_t next_start = key_start + kKeys;
    if (next_start < key_end) {
      load_k(next_start, k_pre);
      load_v(next_start, v_pre);
    }

    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
        s_acc = Wmma(q_frag[ks], LoadFrag(&kv_lds[sub * kKStride + d0]), s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[s_kh][s_rb][(2 * i) + half_id][sub] = s_acc[i];
    }
    __syncthreads();

    // Online softmax: four threads per row, each over four keys.
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const bool live = row_position(rg) < positions;
      const std::uint32_t limit = start_pos + row_position(rg);
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = seg * kPerLane + m;
        const std::uint32_t key = key_start + col;
        const bool valid = live && key <= limit && key < key_end;
        float score = s_lds[0][rb][row][col];
#pragma unroll
        for (std::uint32_t h = 1; h < kDimSplits; ++h)
          score += s_lds[h][rb][row][col];
        vals[m] = valid ? score : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][seg * kPerLane + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_sum += __shfl_xor(part_sum, off);
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const float scale = row_scale[(rb * 16) + (2 * i) + half_id];
        o_acc[rb][0][i] *= scale;
        o_acc[rb][1][i] *= scale;
      }
    }

    // V^T: lane L owns key L % 16 and each thread a 16-dimension slice, so
    // consecutive lanes write consecutive columns (conflict-free).
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        kv_lds[(v_slice + j * 8 + i) * kVtStride + v_key] = packed[i];
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + t * 8;
      const v16h v_frag = LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride]);
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][0]);
        v8f next = Wmma(p_frag, v_frag, o_acc[rb][t]);
        // Row block rb holds positions 2 rb and 2 rb + 1. A row whose own
        // position falls inside this tile sums only its visible keys, in
        // order, so it does not depend on what later keys hold (a single
        // decoded row has not written them yet). A row whose position
        // precedes the tile keeps its accumulator: its probabilities are all
        // zero there, and a single decoded row never visits the tile.
        if (key_start + kKeys - 1 > start_pos + 2 * rb) {
          v8f tail = o_acc[rb][t];
#pragma unroll
          for (std::uint32_t key = 0; key < kKeys; ++key) {
            const float value = static_cast<float>(v_frag[key]);
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t row = rb * 16 + 2 * i + half_id;
              if (key_start + key <= start_pos + row_position(row))
                tail[i] = fmaf(__half2float(p_lds[row][key]), value, tail[i]);
            }
          }
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t limit =
                start_pos + row_position(rb * 16 + 2 * i + half_id);
            if (key_start > limit)
              next[i] = o_acc[rb][t][i];
            else if (key_start + kKeys - 1 > limit)
              next[i] = tail[i];
          }
        }
        o_acc[rb][t] = next;
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n)
      k_cur[n] = k_pre[n];
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n)
      v_cur[n] = v_pre[n];
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row_scratch =
      static_cast<std::size_t>(num_heads) * max_splits * (kHeadDim + 2);
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim = (wave + t * 8) * 16 + sub;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = rb * 16 + 2 * i + half_id;
        if (row_position(row) >= positions)
          continue;
        float* base = scratch + row_position(row) * row_scratch;
        const std::size_t partition_index =
            static_cast<std::size_t>(row_head(row)) * max_splits + split;
        base[stats_elements + partition_index * kHeadDim + dim] =
            o_acc[rb][t][i];
      }
    }
  }
  if (tid < kRows && row_position(tid) < positions) {
    float* base = scratch + row_position(tid) * row_scratch;
    const std::size_t partition_index =
        static_cast<std::size_t>(row_head(tid)) * max_splits + split;
    base[partition_index * 2 + 0] = row_max[tid];
    base[partition_index * 2 + 1] = row_sum[tid];
  }
}

bool LaunchQwenWmmaDecodeAttention(
    const float* q, const float* gate, const void* k_cache_f16,
    const void* v_cache_f16, float* out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t positions, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream, std::span<float> scratch) {
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(num_heads, head_dim);
  if (num_heads != kGroup * num_kv_heads || head_dim != kHeadDim ||
      k_cache_f16 == nullptr || v_cache_f16 == nullptr || positions == 0 ||
      scratch.size() / positions < row_scratch)
    return false;
  const std::size_t length = static_cast<std::size_t>(start_pos) + positions;
  const std::uint32_t partition = DecodePartition(length);
  // Every row must use the partitions it would use on its own: rows on both
  // sides of a partition-size step run one at a time.
  if (DecodePartition(static_cast<std::size_t>(start_pos) + 1) != partition) {
    const std::size_t attn_width =
        static_cast<std::size_t>(num_heads) * kHeadDim;
    for (std::size_t row = 0; row < positions; ++row) {
      if (!LaunchQwenWmmaDecodeAttention(
              q + row * attn_width,
              gate == nullptr ? nullptr : gate + row * attn_width, k_cache_f16,
              v_cache_f16, out_context + row * attn_width, layer_idx,
              start_pos + static_cast<std::uint32_t>(row), 1, max_context,
              num_heads, num_kv_heads, head_dim, stream,
              scratch.subspan(row * row_scratch, row_scratch)))
        return false;
    }
    return true;
  }
  const auto splits =
      static_cast<std::uint32_t>((length + partition - 1) / partition);
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * kHeadDim;
  const std::size_t layer_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  const auto* k = static_cast<const __half*>(k_cache_f16) + layer_offset;
  const auto* v = static_cast<const __half*>(v_cache_f16) + layer_offset;
  const std::size_t attn_width = static_cast<std::size_t>(num_heads) * kHeadDim;
  for (std::size_t first = 0; first < positions; first += kPositions) {
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(kPositions, positions - first));
    WmmaDecodeAttentionPartials<<<dim3(splits, num_kv_heads), 256, 0, stream>>>(
        q + first * attn_width, k, v, scratch.data() + first * row_scratch,
        start_pos + static_cast<std::uint32_t>(first), count, partition,
        num_heads, num_kv_heads);
  }
  LaunchSplitKAttentionReduce(gate, scratch.data(), out_context, num_heads,
                              static_cast<std::uint32_t>(positions), splits,
                              stream);
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
