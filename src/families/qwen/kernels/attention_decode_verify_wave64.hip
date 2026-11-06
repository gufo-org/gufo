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

// Verification rows > 1 in wave64: two KV heads, twelve query heads of
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

}  // namespace

void LaunchVerifySplitKAttentionHalfPartialsWave64(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* scratch, std::uint32_t layer_idx, std::uint32_t start_pos,
    unsigned rows, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t splits, hipStream_t stream) {
  assert(num_kv_heads % 2 == 0 && "wave64 verification pairs KV heads");
  assert(num_heads == 6 * num_kv_heads && "twelve query heads per block");
  assert(rows > 1 && "single rows keep the 32-lane kernel");
  QwenVerifySplitKAttentionHalfPartialsWave64Kernel<<<
      dim3(num_kv_heads / 2, rows, splits), 192, 0, stream>>>(
      q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
      num_heads, num_kv_heads, splits);
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)
