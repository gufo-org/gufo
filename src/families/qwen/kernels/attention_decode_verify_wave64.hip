#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/attention_decode_split_k.hpp"
#include "src/families/qwen/kernels/attention_decode_verify_wave64.hpp"

namespace gufo::hip::detail {
namespace {

// Verification rows > 8 in wave64: two KV heads, twelve query heads of
// sixteen lanes, four heads per wave, 192 threads per block. The per-head
// arithmetic and scratch layout are SplitKAttentionHead's, as in
// QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16>; the score tree
// runs on DPP row exchanges because a 16-lane head is one DPP row.
__launch_bounds__(192) __global__
    void QwenVerifySplitKAttentionHalfPartialsWave64Kernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count) {
  constexpr std::uint32_t lanes = 16;
  // Mixed-precision FMAs take the fp16 cache operands directly instead of
  // converting first (see DotHalf4); the products are unchanged.
  using Head = SplitKAttentionHead<lanes, true>;
  constexpr std::uint32_t head_dim = Head::kHeadDim;
  constexpr std::uint32_t heads_per_kv = 6;
  constexpr std::uint32_t kv_heads_per_block = 2;
  constexpr std::uint32_t heads_per_block = heads_per_kv * kv_heads_per_block;
  constexpr unsigned tile = 4;

  const std::size_t row = blockIdx.y;
  q += row * num_heads * head_dim;
  scratch += row * num_heads * kSplitKDecodeAttentionMaxSplits * (head_dim + 2);
  pos += blockIdx.y;

  const std::uint32_t local_head = threadIdx.x / lanes;
  const std::uint32_t head = blockIdx.x * heads_per_block + local_head;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x % lanes;
  if (head >= num_heads || split >= split_count) {
    return;
  }

  const std::uint32_t local_kv = local_head / heads_per_kv;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  // The block's two KV heads are adjacent inside every token row.
  const std::size_t cache_block_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(blockIdx.x) * kv_heads_per_block * head_dim);
  const auto [partition_begin, partition_end] =
      SplitKDecodePartition(pos, split, split_count);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  Head state;
  state.LoadQuery(query, lane);
  const auto accumulate = [&](const __half* key, const __half* value) {
    const float score = RowSumDpp(state.Partial(key, lane)) * scale;
    state.Accumulate(score, value, lane);
  };

  // Stage both KV heads of a four-token tile once for the twelve heads.
  __shared__ __align__(16) __half keys[kv_heads_per_block][tile][head_dim];
  __shared__ __align__(16) __half values[kv_heads_per_block][tile][head_dim];
  constexpr unsigned vectors_per_token = kv_heads_per_block * head_dim / 8;
  const auto token_stride = static_cast<std::uint32_t>(kv_width);
  for (std::size_t begin = partition_begin; begin < partition_end;
       begin += tile) {
    const unsigned count =
        static_cast<unsigned>(min(std::size_t{tile}, partition_end - begin));
    // Tiles are consecutive tokens: one wave-uniform base per tile keeps
    // each thread's offset at 32 bits.
    const std::size_t tile_offset = cache_block_offset + begin * kv_width;
    const __half* key_tile = k_cache + tile_offset;
    const __half* value_tile = v_cache + tile_offset;
    for (unsigned vector = threadIdx.x; vector < count * vectors_per_token;
         vector += heads_per_block * lanes) {
      const unsigned token = vector / vectors_per_token;
      const unsigned element = (vector % vectors_per_token) * 8;
      const std::uint32_t offset = token * token_stride + element;
      __half* key_slot = keys[element / head_dim][token] + (element % head_dim);
      __half* value_slot =
          values[element / head_dim][token] + (element % head_dim);
      *reinterpret_cast<uint4*>(key_slot) =
          *reinterpret_cast<const uint4*>(key_tile + offset);
      *reinterpret_cast<uint4*>(value_slot) =
          *reinterpret_cast<const uint4*>(value_tile + offset);
    }
    __syncthreads();
#pragma unroll 1
    for (unsigned token = 0; token < count; ++token) {
      accumulate(keys[local_kv][token], values[local_kv][token]);
    }
    __syncthreads();
  }
  state.Store(scratch, num_heads, head, split, lane);
}

// Verification rows 2..8 in one block per (KV head, split): all rows' query
// heads read the same staged KV tile, so the cache streams once instead of
// once per row. Each row keeps its own causal partition, visits its tokens
// in order and runs SplitKAttentionHead's arithmetic, so the scratch matches
// the per-row kernels bit for bit. The block walks the union of its rows'
// partitions, loads the next tile into registers while the current one is
// consumed, and scores two tokens before their in-order softmax steps.
template<unsigned Rows, unsigned Lanes, unsigned Tile>
__launch_bounds__(Rows * 6 * Lanes) __global__
    void QwenVerifySharedRowsAttentionPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count) {
  using Head = SplitKAttentionHead<Lanes, true>;
  constexpr std::uint32_t head_dim = Head::kHeadDim;
  constexpr unsigned threads_per_row = 6 * Lanes;
  constexpr unsigned vectors_per_token = head_dim / 8;
  // One uint4 of K and one of V per staging thread and tile.
  static_assert(Tile * vectors_per_token <= Rows * threads_per_row);

  // The launch has one row group (blockIdx.z == 0). Reading the first row
  // from the grid keeps the row bounds out of constant folding; with them
  // folded, gfx1151 code for six to eight rows ran 5-10% slower.
  const unsigned first_row = blockIdx.z * Rows;
  const unsigned row = first_row + threadIdx.x / threads_per_row;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t head =
      (kv_head * 6) + ((threadIdx.x % threads_per_row) / Lanes);
  const std::uint32_t lane = threadIdx.x % Lanes;
  const std::uint32_t split = blockIdx.y;

  const auto [row_begin, row_end] =
      SplitKDecodePartition(pos + row, split, split_count);
  const std::size_t union_begin =
      SplitKDecodePartition(pos + first_row, split, split_count).begin;
  const std::size_t union_end =
      SplitKDecodePartition(pos + first_row + Rows - 1, split, split_count).end;

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  Head state;
  const std::size_t query_offset =
      ((static_cast<std::size_t>(row) * num_heads) + head) * head_dim;
  state.LoadQuery(reinterpret_cast<const float4*>(q + query_offset), lane);
  const auto score_of = [&](const __half* key) {
    const float partial = state.Partial(key, lane);
    if constexpr (Lanes == 8) {
      return HalfRowSumDpp(partial) * scale;
    } else {
      return RowSumDpp(partial) * scale;
    }
  };

  __shared__ __align__(16) __half keys[Tile][head_dim];
  __shared__ __align__(16) __half values[Tile][head_dim];
  const unsigned stage_token = threadIdx.x / vectors_per_token;
  const unsigned stage_element = (threadIdx.x % vectors_per_token) * 8;
  const bool stages = stage_token < Tile;
  uint4 next_key, next_value;
  // A tail tile restages the last token in its unused slots; no row reads
  // them, and every load stays inside the union.
  const auto fetch = [&](std::size_t begin) {
    if (stages) {
      const std::size_t token = min(begin + stage_token, union_end - 1);
      const std::size_t offset =
          cache_head_offset + (token * kv_width) + stage_element;
      next_key = *reinterpret_cast<const uint4*>(k_cache + offset);
      next_value = *reinterpret_cast<const uint4*>(v_cache + offset);
    }
  };

  fetch(union_begin);
  for (std::size_t begin = union_begin; begin < union_end; begin += Tile) {
    const std::size_t tile_end = min(begin + Tile, union_end);
    if (stages) {
      *reinterpret_cast<uint4*>(keys[stage_token] + stage_element) = next_key;
      *reinterpret_cast<uint4*>(values[stage_token] + stage_element) =
          next_value;
    }
    __syncthreads();
    if (tile_end < union_end) {
      fetch(tile_end);
    }
    // This row's tokens inside the tile.
    const auto first = static_cast<unsigned>(
        (row_begin > begin ? min(row_begin, tile_end) : begin) - begin);
    const auto last = static_cast<unsigned>(
        (row_end < tile_end ? max(row_end, begin) : tile_end) - begin);
    unsigned token = first;
#pragma unroll 1
    for (; token + 2 <= last; token += 2) {
      const float first_score = score_of(keys[token]);
      const float second_score = score_of(keys[token + 1]);
      state.AccumulateOneExp(first_score, values[token], lane);
      state.AccumulateOneExp(second_score, values[token + 1], lane);
    }
#pragma unroll 1
    for (; token < last; ++token) {
      state.AccumulateOneExp(score_of(keys[token]), values[token], lane);
    }
    __syncthreads();
  }
  state.Store(scratch + (static_cast<std::size_t>(row) * num_heads *
                         kSplitKDecodeAttentionMaxSplits * (head_dim + 2)),
              num_heads, head, split, lane);
}

template<unsigned Rows, unsigned Lanes, unsigned Tile>
void LaunchVerifySharedRows(const float* q, const __half* k_cache,
                            const __half* v_cache, float* scratch,
                            std::uint32_t layer_idx, std::uint32_t start_pos,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t splits,
                            hipStream_t stream) {
  QwenVerifySharedRowsAttentionPartialsKernel<Rows, Lanes, Tile>
      <<<dim3(num_kv_heads, splits), Rows * 6 * Lanes, 0, stream>>>(
          q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
          num_heads, num_kv_heads, splits);
}

}  // namespace

void LaunchVerifySplitKAttentionHalfPartialsWave64(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* scratch, std::uint32_t layer_idx, std::uint32_t start_pos,
    unsigned rows, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t splits, hipStream_t stream) {
  assert(num_kv_heads % 2 == 0 && "wave64 verification pairs KV heads");
  assert(num_heads == 6 * num_kv_heads && "twelve query heads per block");
  assert(rows > 1 && "single rows keep the 32-lane kernel");
  // Shared-row geometries measured on gfx1151 (four KV heads, 0.8K to 48K
  // context): 16-lane heads up to five rows, 8-lane heads from six, and as
  // many tile tokens as one uint4 per thread stages.
  const auto launch = [&](auto kernel) {
    kernel(q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
           num_heads, num_kv_heads, splits, stream);
  };
  switch (rows) {
    case 2:
      return launch(LaunchVerifySharedRows<2, 16, 4>);
    case 3:
      return launch(LaunchVerifySharedRows<3, 16, 8>);
    case 4:
      return launch(LaunchVerifySharedRows<4, 16, 8>);
    case 5:
      return launch(LaunchVerifySharedRows<5, 16, 8>);
    case 6:
      return launch(LaunchVerifySharedRows<6, 8, 8>);
    case 7:
      return launch(LaunchVerifySharedRows<7, 8, 8>);
    case 8:
      return launch(LaunchVerifySharedRows<8, 8, 8>);
    default:
      break;
  }
  QwenVerifySplitKAttentionHalfPartialsWave64Kernel<<<
      dim3(num_kv_heads / 2, rows, splits), 192, 0, stream>>>(
      q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
      num_heads, num_kv_heads, splits);
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)
