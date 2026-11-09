#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "src/families/qwen/hip_common/quant_ops.hpp"
#include "src/families/qwen/kernels/dflash_kernels.hpp"

namespace gufo::hip::kernels {

// The DFlash-2 draft head is the target LM head, which the Q8 target keeps in
// BF16 at 2.54 GB -- half of everything the draft graph reads. Drafting is
// purely weight-bandwidth bound, and the draft only steers which tokens the
// target verifies, so a draft-private Q8_0 copy cannot change an emitted token:
// the verifier still runs the target's BF16 head.
__global__ void dflash_quantize_bf16_to_q8_0_kernel(
    const hip_bfloat16* __restrict__ source,
    Q8_0Block* __restrict__ destination, uint64_t total_blocks) {
  const uint64_t block =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= total_blocks) {
    return;
  }
  const hip_bfloat16* values = source + (block * kQ8_0BlockSize);
  float absolute_max = 0.0F;
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    absolute_max =
        fmaxf(absolute_max, fabsf(static_cast<float>(values[index])));
  }
  const float scale = absolute_max / 127.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  Q8_0Block& out = destination[block];
  out.d = __float2half(scale);
  for (uint32_t index = 0; index < kQ8_0BlockSize; ++index) {
    const float quantized = roundf(static_cast<float>(values[index]) * inverse);
    out.qs[index] =
        static_cast<int8_t>(fminf(fmaxf(quantized, -127.0F), 127.0F));
  }
}

void LaunchDFlashQuantizeBf16ToQ8_0(const void* bf16_source,
                                    void* q8_destination,
                                    std::size_t total_elements,
                                    hipStream_t stream) {
  if (bf16_source == nullptr || q8_destination == nullptr ||
      total_elements % kQ8_0BlockSize != 0) {
    return;
  }
  const std::uint64_t total_blocks = total_elements / kQ8_0BlockSize;
  constexpr std::uint32_t kThreads = 256;
  const std::uint64_t grid = (total_blocks + kThreads - 1) / kThreads;
  dflash_quantize_bf16_to_q8_0_kernel<<<static_cast<unsigned int>(grid),
                                        kThreads, 0, stream>>>(
      static_cast<const hip_bfloat16*>(bf16_source),
      static_cast<Q8_0Block*>(q8_destination), total_blocks);
}

__global__ void dflash_grouped_dynamic_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ dynamic_coefficients,
    const float* __restrict__ base_kernel, float* __restrict__ output,
    uint32_t num_tokens, uint32_t hidden_size, uint32_t kernel_size,
    uint32_t group_size, uint32_t side) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  if (index >= total) {
    return;
  }

  const uint32_t token = index / hidden_size;
  const uint32_t channel = index % hidden_size;
  const uint32_t groups = hidden_size / group_size;
  const uint32_t group = channel / group_size;
  const uint32_t coefficients_per_token = 2U * kernel_size * groups;

  float sum = 0.0F;
  for (uint32_t tap = 0; tap < kernel_size; ++tap) {
    if (token < tap) {
      continue;
    }
    const uint64_t dynamic_index =
        static_cast<uint64_t>(token) * coefficients_per_token +
        (side * kernel_size + tap) * groups + group;
    const uint64_t base_index =
        static_cast<uint64_t>(side * kernel_size + tap) * hidden_size + channel;
    const uint64_t input_index =
        static_cast<uint64_t>(token - tap) * hidden_size + channel;
    sum += (base_kernel[base_index] + dynamic_coefficients[dynamic_index]) *
           input[input_index];
  }
  output[index] = sum;
}

__global__ void dflash_silu_mul_kernel(float* __restrict__ gate,
                                       const float* __restrict__ up,
                                       uint32_t total_elements) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

// Coalesced QK loads and shared K/V reads for adjacent query heads. Each
// head retains its original dot-product and 256-thread softmax reduction.
// Causal (DSpine) masks block keys after the query row: exp(-1e20 - max) is
// exactly zero, so a row's output does not depend on the rows after it.
template<uint32_t QueryHeads, bool Causal = false>
__global__ void dflash_noncausal_attention_wave_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  const uint32_t tok = blockIdx.x, h = blockIdx.y * QueryHeads;
  if (tok >= block_count || h + QueryHeads > num_q_heads)
    return;
  const uint32_t kv_h = h / (num_q_heads / num_kv_heads),
                 q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count = history_length > history_start
                                     ? history_length - history_start
                                     : 0U,
                 total_k = history_count + block_count;
  const float4* q_vec = reinterpret_cast<const float4*>(
      q + uint64_t(tok) * q_dim + uint64_t(h) * head_dim);
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  extern __shared__ float scores[];
  __shared__ float reduce[QueryHeads][256];
  for (uint32_t local = wave; local < history_count; local += waves) {
    const uint32_t position = history_start + local;
    const float4* key = reinterpret_cast<const float4*>(
        injected_k + uint64_t(position % history_capacity) * kv_dim +
        uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + local] = query_position - position < sliding_window
                                          ? dot[g] * scale
                                          : -1e20F;
    }
  }
  for (uint32_t p = wave; p < block_count; p += waves) {
    const float4* key = reinterpret_cast<const float4*>(
        block_k + uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim);
    float dot[QueryHeads] = {};
    for (uint32_t d = lane; d < num_vec; d += 32U) {
      const float4 kv = key[d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g) {
        const float4 qv = q_vec[g * num_vec + d];
        dot[g] += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        dot[g] += __shfl_xor(dot[g], offset);
    }
    if (lane == 0) {
      const uint32_t key_position = current_pos + p;
      const uint32_t distance = query_position > key_position
                                    ? query_position - key_position
                                    : key_position - query_position;
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        scores[g * total_k + history_count + p] =
            distance < sliding_window && (!Causal || p <= tok) ? dot[g] * scale
                                                               : -1e20F;
    }
  }
  __syncthreads();
  float maxima[QueryHeads];
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      maxima[g] = fmaxf(maxima[g], scores[g * total_k + p]);
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = maxima[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] =
            fmaxf(reduce[g][threadIdx.x], reduce[g][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    maxima[g] = reduce[g][0];
  __syncthreads();
  float sums[QueryHeads] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g) {
      const float e = expf(scores[g * total_k + p] - maxima[g]);
      scores[g * total_k + p] = e;
      sums[g] += e;
    }
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    reduce[g][threadIdx.x] = sums[g];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        reduce[g][threadIdx.x] += reduce[g][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned g = 0; g < QueryHeads; ++g)
    sums[g] = 1.f / (reduce[g][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      scores[g * total_k + p] *= sums[g];
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc[QueryHeads] = {};
    uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
    // Issue independent value loads together, then retain the original
    // ascending-key FMA sequence. Advancing the ring slot avoids a division
    // for every value without changing the persistent KV layout.
    uint32_t local = 0;
    for (; history_count - local >= kPrefetch; local += kPrefetch) {
      float values[kPrefetch];
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
        values[i] =
            injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
        slot = slot + 1 == history_capacity ? 0 : slot + 1;
      }
#pragma unroll
      for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
        for (unsigned g = 0; g < QueryHeads; ++g)
          acc[g] += scores[g * total_k + local + i] * values[i];
      }
    }
    for (; local < history_count; ++local) {
      const float value =
          injected_v[uint64_t(slot) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + local] * value;
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float value =
          block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
      for (unsigned g = 0; g < QueryHeads; ++g)
        acc[g] += scores[g * total_k + history_count + p] * value;
    }
#pragma unroll
    for (unsigned g = 0; g < QueryHeads; ++g)
      out[uint64_t(tok) * q_dim + uint64_t(h + g) * head_dim + d] = acc[g];
  }
}

__global__ void dflash_noncausal_attention_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ injected_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, float* __restrict__ out,
    uint32_t current_pos, uint32_t history_length, uint32_t block_count,
    uint32_t sliding_window, uint32_t num_q_heads, uint32_t num_kv_heads,
    uint32_t head_dim, float scale, uint32_t history_capacity) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= block_count || h >= num_q_heads)
    return;

  const uint32_t gqa_group = num_q_heads / num_kv_heads;
  const uint32_t kv_h = h / gqa_group;
  const uint32_t q_dim = num_q_heads * head_dim;
  const uint32_t kv_dim = num_kv_heads * head_dim;
  const uint32_t query_position = current_pos + tok;
  const uint32_t history_start = current_pos + 1U > sliding_window
                                     ? current_pos + 1U - sliding_window
                                     : 0U;
  const uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  const uint32_t total_k = history_count + block_count;

  const float* q_ptr = q + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);
  float* out_ptr = out + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);

  extern __shared__ float s_scores[];

  for (uint32_t local = threadIdx.x; local < history_count;
       local += blockDim.x) {
    const uint32_t position = history_start + local;
    const float* k_ptr =
        injected_k + ((uint64_t)(position % history_capacity) * kv_dim +
                      (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[local] =
        query_position - position < sliding_window ? dot * scale : -1e20F;
  }

  for (uint32_t p = threadIdx.x; p < block_count; p += blockDim.x) {
    const float* k_ptr =
        block_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    const uint32_t key_position = current_pos + p;
    const uint32_t distance = query_position > key_position
                                  ? query_position - key_position
                                  : key_position - query_position;
    s_scores[history_count + p] =
        distance < sliding_window ? dot * scale : -1e20F;
  }
  __syncthreads();

  // Find max score
  __shared__ float s_reduce[256];
  float local_max = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] =
          fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_val = s_reduce[0];
  __syncthreads();

  // Exponentiate & sum denominator
  float local_sum = 0.0F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    const float exp_score = expf(s_scores[p] - max_val);
    s_scores[p] = exp_score;
    local_sum += exp_score;
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / (s_reduce[0] + 1e-9F);
  __syncthreads();

  // Normalize scores
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    s_scores[p] *= inv_sum;
  }
  __syncthreads();

  // Multiply by V
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (uint32_t local = 0; local < history_count; ++local) {
      const uint32_t position = history_start + local;
      const float* v_ptr =
          injected_v + ((uint64_t)(position % history_capacity) * kv_dim +
                        (uint64_t)kv_h * head_dim);
      acc += s_scores[local] * v_ptr[d];
    }
    for (uint32_t p = 0; p < block_count; ++p) {
      const float* v_ptr =
          block_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[history_count + p] * v_ptr[d];
    }
    out_ptr[d] = acc;
  }
}

constexpr uint32_t kSelectorThreads = 128;
constexpr uint32_t kMaxSelectorTopK = 16;
constexpr uint32_t kSelectorTokensPerBlock = 2048;
/// Threads that cooperate on one candidate's low-rank transition score.
constexpr uint32_t kSelectorTransitionLanes =
    kSelectorThreads / kMaxSelectorTopK;

__device__ __forceinline__ void InsertSelectorCandidate(
    float score, uint32_t id, float* scores, uint32_t* ids,
    uint32_t selector_top_k) {
  if (score <= scores[selector_top_k - 1U]) {
    return;
  }
  uint32_t insertion = selector_top_k - 1U;
  while (insertion > 0U && score > scores[insertion - 1U]) {
    scores[insertion] = scores[insertion - 1U];
    ids[insertion] = ids[insertion - 1U];
    --insertion;
  }
  scores[insertion] = score;
  ids[insertion] = id;
}

/// Tree-merges the per-thread descending top-k lists held in shared memory into
/// slot zero. The previous implementation had thread zero fold all
/// kSelectorThreads lists serially, which dominated both selector kernels; a
/// pairwise merge does the same work in log2(threads) rounds. Ties keep the
/// lower slot, which is the same preference the serial fold had.
__device__ void MergeSelectorLists(float* scores, uint32_t* ids,
                                   uint32_t selector_top_k) {
  for (uint32_t stride = kSelectorThreads / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      float* mine_scores = scores + (threadIdx.x * kMaxSelectorTopK);
      uint32_t* mine_ids = ids + (threadIdx.x * kMaxSelectorTopK);
      const float* other_scores =
          scores + ((threadIdx.x + stride) * kMaxSelectorTopK);
      const uint32_t* other_ids =
          ids + ((threadIdx.x + stride) * kMaxSelectorTopK);
      float merged_scores[kMaxSelectorTopK];
      uint32_t merged_ids[kMaxSelectorTopK];
      uint32_t mine = 0;
      uint32_t other = 0;
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        const bool take_mine =
            other >= selector_top_k ||
            (mine < selector_top_k && mine_scores[mine] >= other_scores[other]);
        if (take_mine) {
          merged_scores[index] = mine_scores[mine];
          merged_ids[index] = mine_ids[mine];
          ++mine;
        } else {
          merged_scores[index] = other_scores[other];
          merged_ids[index] = other_ids[other];
          ++other;
        }
      }
      for (uint32_t index = 0; index < selector_top_k; ++index) {
        mine_scores[index] = merged_scores[index];
        mine_ids[index] = merged_ids[index];
      }
    }
  }
  __syncthreads();
}

/// Row ranges of the batch's draft-vocabulary sequences; rows of other
/// sequences score the whole vocabulary.
struct SelectorVocabRows {
  DFlashSelectorVocab vocab[8];
  uint32_t row_begin[8];
  uint32_t row_end[8];
  uint32_t count;
};

__global__ void dflash_selector_partial_topk_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_scores,
    uint32_t* __restrict__ partial_ids, uint32_t vocab_size,
    uint32_t selector_top_k, uint64_t logits_stride = 0,
    SelectorVocabRows vocab_rows = {}) {
  logits += uint64_t(blockIdx.y) *
            (logits_stride != 0 ? logits_stride : uint64_t(vocab_size));
  partial_scores += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  partial_ids += uint64_t(blockIdx.y) * gridDim.x * selector_top_k;
  DFlashSelectorVocab vocab{};
  uint32_t vocab_row = 0;
  for (uint32_t index = 0; index < vocab_rows.count; ++index) {
    if (blockIdx.y >= vocab_rows.row_begin[index] &&
        blockIdx.y < vocab_rows.row_end[index]) {
      vocab = vocab_rows.vocab[index];
      vocab_row = blockIdx.y - vocab_rows.row_begin[index];
    }
  }
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }

  const uint32_t begin = blockIdx.x * kSelectorTokensPerBlock;
  const uint32_t requested_end = begin + kSelectorTokensPerBlock;
  if (vocab.prefix == 0) {
    const uint32_t end =
        requested_end < vocab_size ? requested_end : vocab_size;
    for (uint32_t token = begin + threadIdx.x; token < end;
         token += blockDim.x) {
      InsertSelectorCandidate(logits[token], token, local_scores, local_ids,
                              selector_top_k);
    }
  } else {
    // Column c < prefix is ID c; the table follows, and its empty slots
    // score -inf, so they never enter the list.
    const uint32_t columns = vocab.prefix + vocab.rows;
    const uint32_t end = requested_end < columns ? requested_end : columns;
    const float* table = vocab.logits + uint64_t(vocab_row) * vocab.rows;
    for (uint32_t column = begin + threadIdx.x; column < end;
         column += blockDim.x) {
      if (column < vocab.prefix) {
        InsertSelectorCandidate(logits[column], column, local_scores, local_ids,
                                selector_top_k);
      } else {
        const uint32_t slot = column - vocab.prefix;
        InsertSelectorCandidate(table[slot], uint32_t(vocab.ids[slot]),
                                local_scores, local_ids, selector_top_k);
      }
    }
  }

  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists(shared_scores, shared_ids, selector_top_k);

  const uint32_t output_offset = blockIdx.x * selector_top_k;
  for (uint32_t index = threadIdx.x; index < selector_top_k;
       index += blockDim.x) {
    partial_scores[output_offset + index] = shared_scores[index];
    partial_ids[output_offset + index] = shared_ids[index];
  }
}

__device__ __forceinline__ void FinalizeSelector(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    const float* __restrict__ projected_hidden,
    const hip_bfloat16* __restrict__ predecessor_codebook,
    const hip_bfloat16* __restrict__ successor_codebook,
    const uint32_t* __restrict__ predecessor_token, uint32_t* output_token,
    float* __restrict__ output_confidence, float temperature,
    const float* __restrict__ sample_uniform,
    uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k,
    DFlashSampling sampling = {},
    // Earlier iterations of this kernel write this history. Reload it for
    // each conditional q; cached read-only loads break sampled replay.
    const volatile uint32_t* previous_tokens = nullptr,
    uint32_t previous_count = 0) {
  __shared__ float shared_scores[kSelectorThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kSelectorThreads * kMaxSelectorTopK];
  __shared__ float
      shared_transition[kMaxSelectorTopK * kSelectorTransitionLanes];

  float local_scores[kMaxSelectorTopK];
  uint32_t local_ids[kMaxSelectorTopK];
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }
  const uint32_t candidate_count = partial_count * selector_top_k;
  for (uint32_t index = threadIdx.x; index < candidate_count;
       index += blockDim.x) {
    InsertSelectorCandidate(partial_scores[index], partial_ids[index],
                            local_scores, local_ids, selector_top_k);
  }

  const uint32_t shared_offset = threadIdx.x * kMaxSelectorTopK;
#pragma unroll
  for (uint32_t index = 0; index < kMaxSelectorTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }

  MergeSelectorLists(shared_scores, shared_ids, selector_top_k);

  const uint32_t predecessor = *predecessor_token;
  const hip_bfloat16* predecessor_row =
      predecessor_codebook + (uint64_t)predecessor * selector_rank;

  // Score the low-rank transitions in parallel: kSelectorTransitionLanes
  // threads share each candidate and reduce through shared memory. Thread zero
  // used to walk all selector_top_k x selector_rank products by itself.
  {
    const uint32_t candidate_index = threadIdx.x / kSelectorTransitionLanes;
    const uint32_t part = threadIdx.x % kSelectorTransitionLanes;
    if (candidate_index < selector_top_k) {
      const uint32_t candidate = shared_ids[candidate_index];
      const hip_bfloat16* successor_row =
          successor_codebook + (uint64_t)candidate * selector_rank;
      float partial = 0.0F;
      for (uint32_t rank = part; rank < selector_rank;
           rank += kSelectorTransitionLanes) {
        partial += static_cast<float>(predecessor_row[rank]) *
                   projected_hidden[rank] *
                   static_cast<float>(successor_row[rank]);
      }
      shared_transition[(candidate_index * kSelectorTransitionLanes) + part] =
          partial;
    }
  }
  __syncthreads();

  if (threadIdx.x != 0U) {
    return;
  }

  float top_scores[kMaxSelectorTopK];
  uint32_t top_ids[kMaxSelectorTopK];
  for (uint32_t index = 0; index < selector_top_k; ++index) {
    top_scores[index] = shared_scores[index];
    top_ids[index] = shared_ids[index];
  }

  float best_score = -INFINITY;
  uint32_t best_index = 0;
  float path_scores[kMaxSelectorTopK];
  for (uint32_t candidate_index = 0; candidate_index < selector_top_k;
       ++candidate_index) {
    float transition = 0.0F;
    for (uint32_t part = 0; part < kSelectorTransitionLanes; ++part) {
      transition +=
          shared_transition[(candidate_index * kSelectorTransitionLanes) +
                            part];
    }
    float score = top_scores[candidate_index] + transition;
    const auto id = top_ids[candidate_index];
    if (sampling.frequency_penalty != 0.0F ||
        sampling.presence_penalty != 0.0F || sampling.repeat_penalty != 1.0F) {
      uint32_t generated = 0;
      uint32_t low = 0, high = sampling.penalty_count;
      while (low < high) {
        const auto middle = low + (high - low) / 2;
        if (sampling.penalties[middle].token < id)
          low = middle + 1;
        else
          high = middle;
      }
      if (low < sampling.penalty_count && sampling.penalties[low].token == id)
        generated = sampling.penalties[low].generated_count;
      for (uint32_t i = 0; i < previous_count; ++i)
        generated += previous_tokens[i] == id;
      bool repeated = false;
      if (sampling.repeat_penalty != 1.0F && sampling.repeat_last_n > 0) {
        const auto retained = min(sampling.history_count,
                                  sampling.repeat_last_n > previous_count
                                      ? sampling.repeat_last_n - previous_count
                                      : 0U);
        for (uint32_t i = sampling.history_count - retained;
             i < sampling.history_count; ++i)
          repeated |= sampling.history[i] == id;
        const auto first = previous_count > sampling.repeat_last_n
                               ? previous_count - sampling.repeat_last_n
                               : 0U;
        for (uint32_t i = first; i < previous_count; ++i)
          repeated |= previous_tokens[i] == id;
      }
      if (repeated)
        score = score <= 0.0F ? score * sampling.repeat_penalty
                              : score / sampling.repeat_penalty;
      score -= sampling.frequency_penalty * static_cast<float>(generated);
      if (generated != 0)
        score -= sampling.presence_penalty;
    }
    path_scores[candidate_index] = score;
    if (score > best_score) {
      best_score = score;
      best_index = candidate_index;
    }
  }

  if (candidate_ids != nullptr) {
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      candidate_ids[index] = top_ids[index];
    }
  }

  if (temperature > 0.0F) {
    const float divisor = fmaxf(temperature, FLT_MIN);
    float denominator = 0.0F;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      // Subtract before division so a finite, very small temperature cannot
      // overflow both logits and their maximum to infinity.
      const float probability_weight =
          expf((path_scores[index] - best_score) / divisor);
      path_scores[index] = probability_weight;
      denominator += probability_weight;
    }
    const float inverse_denominator = 1.0F / fmaxf(denominator, 1e-30F);
    float threshold =
        fminf(fmaxf(*sample_uniform, 0.0F), 0.99999994F) * denominator;
    uint32_t sampled_index = best_index;
    bool selected = false;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      const float probability = path_scores[index] * inverse_denominator;
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[index] = probability;
      }
      if (!selected && probability > 0.0F)
        sampled_index = index;
      threshold -= path_scores[index];
      if (threshold < 0.0F && !selected) {
        sampled_index = index;
        selected = true;
      }
    }
    *output_token = top_ids[sampled_index];
    if (output_confidence != nullptr) {
      *output_confidence = path_scores[sampled_index] * inverse_denominator;
    }
  } else {
    *output_token = top_ids[best_index];
    if (candidate_probabilities != nullptr) {
      for (uint32_t index = 0; index < selector_top_k; ++index)
        candidate_probabilities[index] = index == best_index ? 1.0F : 0.0F;
    }
    if (output_confidence != nullptr) {
      *output_confidence = 1.0F;
    }
  }
}

__global__ void dflash_selector_finalize_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, const uint32_t* predecessor_token,
    uint32_t* output_token, float* output_confidence, float temperature,
    const float* sample_uniform, uint32_t* candidate_ids,
    float* candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  FinalizeSelector(partial_scores, partial_ids, projected_hidden,
                   predecessor_codebook, successor_codebook, predecessor_token,
                   output_token, output_confidence, temperature, sample_uniform,
                   candidate_ids, candidate_probabilities, partial_count,
                   selector_rank, selector_top_k);
}

struct SelectorBatch {
  DFlashSelectorSequence sequences[8];
  uint32_t row_offsets[8];
  uint32_t token_offsets[8];
};

__global__ void dflash_selector_finalize_batch_kernel(
    const float* partial_scores, const uint32_t* partial_ids,
    const float* projected_hidden, const hip_bfloat16* predecessor_codebook,
    const hip_bfloat16* successor_codebook, uint32_t* tokens,
    float* confidences, const float* uniforms, uint32_t* candidate_ids,
    float* candidate_probabilities, SelectorBatch batch, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
  const auto sequence = batch.sequences[blockIdx.x];
  __shared__ uint32_t predecessor;
  if (threadIdx.x == 0U)
    predecessor = tokens[batch.token_offsets[blockIdx.x]];
  __syncthreads();
  for (uint32_t step = 0; step < sequence.count; ++step) {
    const uint32_t row = batch.row_offsets[blockIdx.x] + step;
    const uint32_t token = batch.token_offsets[blockIdx.x] + step;
    const uint64_t partial = uint64_t(row) * partial_count * selector_top_k;
    const uint64_t candidate = uint64_t(row) * selector_top_k;
    FinalizeSelector(
        partial_scores + partial, partial_ids + partial,
        projected_hidden + uint64_t(row) * selector_rank, predecessor_codebook,
        successor_codebook, &predecessor, tokens + token + 1U,
        confidences == nullptr ? nullptr : confidences + row,
        sequence.temperature,
        sequence.temperature > 0.0F ? uniforms + row : nullptr,
        candidate_ids != nullptr ? candidate_ids + candidate : nullptr,
        candidate_probabilities != nullptr ? candidate_probabilities + candidate
                                           : nullptr,
        partial_count, selector_rank, selector_top_k, sequence.sampling,
        tokens + batch.token_offsets[blockIdx.x] + 1U, step);
    // Thread zero publishes the predecessor for this sequence's next step.
    // Other sequences own separate token, probability and partial buffers.
    if (threadIdx.x == 0U)
      predecessor = tokens[token + 1U];
    __syncthreads();
  }
}

__global__ void dflash_gather_rows_kernel(const uint32_t* __restrict__ source,
                                          uint32_t row_words,
                                          const int32_t* __restrict__ ids,
                                          uint32_t* __restrict__ destination) {
  const uint32_t* row =
      source + uint64_t(uint32_t(ids[blockIdx.x])) * row_words;
  uint32_t* out = destination + uint64_t(blockIdx.x) * row_words;
  for (uint32_t word = threadIdx.x; word < row_words; word += blockDim.x)
    out[word] = row[word];
}

__global__ void dflash_draft_table_logits_kernel(
    const float* __restrict__ source, uint32_t stride,
    const int32_t* __restrict__ ids, float* __restrict__ table_logits,
    uint32_t table_rows) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= table_rows)
    return;
  const int32_t id = ids[slot];
  float* out = table_logits + uint64_t(blockIdx.y) * table_rows + slot;
  if (id < 0)
    *out = -INFINITY;
  else if (source != nullptr)
    *out = source[uint64_t(blockIdx.y) * stride + uint32_t(id)];
}

void LaunchDFlashGroupedDynamicConv(
    const float* input, const float* dynamic_coefficients,
    const float* base_kernel, float* output, std::uint32_t num_tokens,
    std::uint32_t hidden_size, std::uint32_t kernel_size,
    std::uint32_t group_size, std::uint32_t side, hipStream_t stream) {
  const uint64_t total = static_cast<uint64_t>(num_tokens) * hidden_size;
  const uint32_t blocks = static_cast<uint32_t>((total + 255U) / 256U);
  dflash_grouped_dynamic_conv_kernel<<<blocks, 256, 0, stream>>>(
      input, dynamic_coefficients, base_kernel, output, num_tokens, hidden_size,
      kernel_size, group_size, side);
}

void LaunchDFlashSiLUMul(float* gate, const float* up,
                         std::uint32_t total_elements, hipStream_t stream) {
  const uint32_t blocks = (total_elements + 255u) / 256u;
  dflash_silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, total_elements);
}

namespace {

/// Shared launch geometry for both non-causal attention routes: one workgroup
/// per (draft token, query head) and one shared score slot per attended key.
struct DFlashAttentionLaunch {
  dim3 grid;
  std::size_t shared_bytes;
};

[[nodiscard]] DFlashAttentionLaunch DFlashAttentionGeometry(
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads) noexcept {
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  return {.grid = dim3(block_count, num_q_heads),
          .shared_bytes = (history_count + block_count) * sizeof(float)};
}

}  // namespace

void LaunchDFlashNonCausalAttentionScalar(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  dflash_noncausal_attention_kernel<<<geometry.grid, 256, geometry.shared_bytes,
                                      stream>>>(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, history_capacity);
}

void LaunchDFlashNonCausalAttentionWave(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, bool causal) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  // Wider blocks benefit from two query heads per CTA. Keep narrow blocks
  // independent and leave enough LDS for both 256-thread reduction rows.
  const auto heads_per_kv = num_q_heads / num_kv_heads;
  if (block_count >= 4 && (heads_per_kv % 2U) == 0U &&
      geometry.shared_bytes <= (64U * 1024U - 2U * 256U * sizeof(float)) / 2U) {
    auto grid = geometry.grid;
    grid.y /= 2U;
    if (causal) {
      dflash_noncausal_attention_wave_kernel<2, true>
          <<<grid, 256, 2U * geometry.shared_bytes, stream>>>(
              q, injected_k, injected_v, block_k, block_v, out, current_pos,
              history_length, block_count, sliding_window, num_q_heads,
              num_kv_heads, head_dim, scale, history_capacity);
      return;
    }
    dflash_noncausal_attention_wave_kernel<2>
        <<<grid, 256, 2U * geometry.shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    return;
  }
  if (causal) {
    dflash_noncausal_attention_wave_kernel<1, true>
        <<<geometry.grid, 256, geometry.shared_bytes, stream>>>(
            q, injected_k, injected_v, block_k, block_v, out, current_pos,
            history_length, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity);
    return;
  }
  dflash_noncausal_attention_wave_kernel<1>
      <<<geometry.grid, 256, geometry.shared_bytes, stream>>>(
          q, injected_k, injected_v, block_k, block_v, out, current_pos,
          history_length, block_count, sliding_window, num_q_heads,
          num_kv_heads, head_dim, scale, history_capacity);
}

void LaunchDFlashNonCausalAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, bool causal) {
  // Coalesced float4 reads cover all shipped DFlash2 head dimensions.
  if ((head_dim % 4U) == 0U) {
    LaunchDFlashNonCausalAttentionWave(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity, causal);
    return;
  }
  // The DSpine loader requires head_dim % 4 == 0: only the wave route masks.
  if (causal)
    throw std::invalid_argument("causal DFlash attention needs head_dim % 4");
  LaunchDFlashNonCausalAttentionScalar(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
      head_dim, scale, stream, history_capacity);
}

std::size_t DFlashSelectorScratchElements(std::uint32_t vocab_size) noexcept {
  const std::size_t partial_count =
      (static_cast<std::size_t>(vocab_size) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock;
  return partial_count * kMaxSelectorTopK;
}

void LaunchDFlashSelectorStep(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    const std::uint32_t* predecessor_token, std::uint32_t* output_token,
    float* output_confidence, float* partial_scores, std::uint32_t* partial_ids,
    float temperature, const float* sample_uniform,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::uint32_t vocab_size, std::uint32_t selector_rank,
    std::uint32_t selector_top_k, hipStream_t stream) {
  if (partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_top_k == 0 || selector_top_k > kMaxSelectorTopK ||
      selector_top_k > vocab_size) {
    throw std::invalid_argument("invalid DFlash selector shape or scratch");
  }
  const std::uint32_t partial_count =
      (vocab_size + kSelectorTokensPerBlock - 1U) / kSelectorTokensPerBlock;
  dflash_selector_partial_topk_kernel<<<partial_count, kSelectorThreads, 0,
                                        stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  dflash_selector_finalize_kernel<<<1, kSelectorThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16),
      predecessor_token, output_token, output_confidence, temperature,
      sample_uniform, candidate_ids, candidate_probabilities, partial_count,
      selector_rank, selector_top_k);
}

void LaunchDFlashSelectorBatch(
    const float* logits, const float* projected_hidden,
    const void* predecessor_codebook_bf16, const void* successor_codebook_bf16,
    std::uint32_t* tokens, float* confidences, float* partial_scores,
    std::uint32_t* partial_ids, const float* uniforms,
    std::uint32_t* candidate_ids, float* candidate_probabilities,
    std::span<const DFlashSelectorSequence> sequences, std::uint32_t vocab_size,
    std::uint32_t selector_rank, std::uint32_t selector_top_k,
    hipStream_t stream, std::uint32_t logits_stride) {
  if (sequences.empty() || sequences.size() > 8 || logits == nullptr ||
      projected_hidden == nullptr || predecessor_codebook_bf16 == nullptr ||
      successor_codebook_bf16 == nullptr || tokens == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr || vocab_size == 0 ||
      selector_rank == 0 || selector_top_k == 0 ||
      selector_top_k > kMaxSelectorTopK || selector_top_k > vocab_size)
    throw std::invalid_argument("invalid DFlash selector batch or scratch");
  SelectorBatch batch{};
  SelectorVocabRows vocab_rows{};
  std::uint32_t rows = 0;
  // The widest row decides the partial count; a draft-vocabulary row reads
  // fewer columns, and its remaining partial lists stay empty.
  std::uint32_t columns = 0;
  for (std::size_t index = 0; index < sequences.size(); ++index) {
    const auto sequence = sequences[index];
    const auto& vocab = sequence.vocab;
    if (sequence.count == 0 ||
        sequence.count >
            std::numeric_limits<std::uint32_t>::max() - 8U - rows ||
        !std::isfinite(sequence.temperature) || sequence.temperature < 0.0F ||
        (sequence.temperature > 0.0F && uniforms == nullptr) ||
        (vocab.prefix != 0 &&
         (vocab.prefix > vocab_size || vocab.logits == nullptr ||
          vocab.ids == nullptr || vocab.rows == 0 ||
          vocab.rows > vocab_size - vocab.prefix ||
          vocab.prefix > (logits_stride != 0 ? logits_stride : vocab_size))))
      throw std::invalid_argument("invalid DFlash selector sequence");
    batch.sequences[index] = sequence;
    batch.row_offsets[index] = rows;
    batch.token_offsets[index] = rows + static_cast<std::uint32_t>(index);
    if (vocab.prefix != 0) {
      vocab_rows.vocab[vocab_rows.count] = vocab;
      vocab_rows.row_begin[vocab_rows.count] = rows;
      vocab_rows.row_end[vocab_rows.count] = rows + sequence.count;
      ++vocab_rows.count;
      columns = std::max(columns, vocab.prefix + vocab.rows);
    } else {
      columns = vocab_size;
    }
    rows += sequence.count;
  }
  if (logits_stride != 0 && vocab_rows.count != sequences.size() &&
      logits_stride < vocab_size)
    throw std::invalid_argument("invalid DFlash selector logits stride");
  const auto partial_count = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(columns) + kSelectorTokensPerBlock - 1U) /
      kSelectorTokensPerBlock);
  dflash_selector_partial_topk_kernel<<<dim3(partial_count, rows),
                                        kSelectorThreads, 0, stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k,
      logits_stride, vocab_rows);
  dflash_selector_finalize_batch_kernel<<<sequences.size(), kSelectorThreads, 0,
                                          stream>>>(
      partial_scores, partial_ids, projected_hidden,
      static_cast<const hip_bfloat16*>(predecessor_codebook_bf16),
      static_cast<const hip_bfloat16*>(successor_codebook_bf16), tokens,
      confidences, uniforms, candidate_ids, candidate_probabilities, batch,
      partial_count, selector_rank, selector_top_k);
}

void LaunchDFlashGatherRows(const void* source, std::size_t row_bytes,
                            const std::int32_t* ids, void* destination,
                            std::uint32_t count, hipStream_t stream) {
  if (source == nullptr || ids == nullptr || destination == nullptr ||
      row_bytes == 0 || row_bytes % sizeof(std::uint32_t) != 0 ||
      row_bytes / sizeof(std::uint32_t) >
          std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("invalid DFlash row gather");
  if (count == 0)
    return;
  dflash_gather_rows_kernel<<<count, 256, 0, stream>>>(
      static_cast<const std::uint32_t*>(source),
      static_cast<std::uint32_t>(row_bytes / sizeof(std::uint32_t)), ids,
      static_cast<std::uint32_t*>(destination));
}

void LaunchDFlashDraftTableLogits(const float* source, std::uint32_t stride,
                                  const std::int32_t* ids, float* table_logits,
                                  std::uint32_t rows, std::uint32_t table_rows,
                                  hipStream_t stream) {
  if (ids == nullptr || table_logits == nullptr || rows == 0 || table_rows == 0)
    throw std::invalid_argument("invalid DFlash draft table");
  constexpr std::uint32_t kThreads = 256;
  dflash_draft_table_logits_kernel<<<
      dim3((table_rows + kThreads - 1U) / kThreads, rows), kThreads, 0,
      stream>>>(source, stride, ids, table_logits, table_rows);
}

// DSpine (tencent/Qwen3.8-27B-DSpine-b16). The maths follows Tencent's
// reference, sglang_dspine/kernels/dspine.py at 321b0fc ("dspine.py" below),
// in FP32; ties go to the lower token ID. Row 0 of a block is the anchor and
// rows 1..n are proposals; proposal t (0-based) is block row t + 1.

constexpr uint32_t kDSpineThreads = 256;

/// Workgroup sum; every thread receives it. `scratch` holds one float per
/// wave and may be reused by the next call.
__device__ float DSpineBlockSum(float value, float* scratch) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    value += __shfl_xor(value, offset);
  __syncthreads();
  if (threadIdx.x % warpSize == 0)
    scratch[threadIdx.x / warpSize] = value;
  __syncthreads();
  float total = 0.0F;
  for (uint32_t wave = 0; wave < blockDim.x / warpSize; ++wave)
    total += scratch[wave];
  return total;
}

/// rms_t = sqrt(mean(h_t^2)), normalized_t = h_t / (rms_t + eps)
/// (dspine.py:12-32, prepare_read).
__global__ void dspine_row_stats_kernel(const float* __restrict__ hidden,
                                        float* __restrict__ normalized,
                                        float* __restrict__ rms,
                                        uint32_t hidden_size, float eps) {
  __shared__ float scratch[32];
  const float* row = hidden + uint64_t(blockIdx.x) * hidden_size;
  float sum = 0.0F;
  for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x)
    sum += row[i] * row[i];
  const float row_rms =
      sqrtf(fmaxf(DSpineBlockSum(sum, scratch) / float(hidden_size), FLT_MIN));
  float* out = normalized + uint64_t(blockIdx.x) * hidden_size;
  for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x)
    out[i] = row[i] / (row_rms + eps);
  if (threadIdx.x == 0)
    rms[blockIdx.x] = row_rms;
}

/// Row 0: sqrt(C) * codebook[anchor]; row t: sqrt(C) * z_t / (|z_t| + eps)
/// (dspine.py:35-57, normalize_codes).
__global__ void dspine_codes_kernel(float* __restrict__ codes,
                                    const float* __restrict__ anchor_code,
                                    uint32_t code_size, float eps) {
  __shared__ float scratch[32];
  float* row = codes + uint64_t(blockIdx.x) * code_size;
  const float scale = sqrtf(float(code_size));
  if (blockIdx.x == 0) {
    for (uint32_t i = threadIdx.x; i < code_size; i += blockDim.x)
      row[i] = anchor_code[i] * scale;
    return;
  }
  float sum = 0.0F;
  for (uint32_t i = threadIdx.x; i < code_size; i += blockDim.x)
    sum += row[i] * row[i];
  const float norm = sqrtf(DSpineBlockSum(sum, scratch));
  for (uint32_t i = threadIdx.x; i < code_size; i += blockDim.x)
    row[i] = scale * row[i] / (norm + eps);
}

/// message = sigmoid(query + gate + bias) * value. lanes == 0: the bridge
/// (dspine.py:60-73); `packed` rows are [G c; P c; A c], row r's gate and
/// value come from row r - 1 and row 0 sends nothing. lanes > 0: the
/// refinement (dspine.py:571-579); row r = t * lanes + j pairs proposal t's
/// query (packed row t + 1) with sender lane j, whose `sender` row is
/// [P e_j; A e_j] for an unscaled codebook row e_j (scaled here).
__global__ void dspine_message_kernel(const float* __restrict__ packed,
                                      const float* __restrict__ sender,
                                      const float* __restrict__ bias,
                                      float* __restrict__ message,
                                      uint32_t rank, uint32_t lanes,
                                      float sender_scale) {
  const uint32_t row = blockIdx.x;
  float* out = message + uint64_t(row) * rank;
  const float* query = nullptr;
  const float* gate = nullptr;
  if (lanes == 0) {
    if (row == 0) {
      for (uint32_t k = threadIdx.x; k < rank; k += blockDim.x)
        out[k] = 0.0F;
      return;
    }
    query = packed + uint64_t(row) * 3U * rank;
    gate = packed + uint64_t(row - 1U) * 3U * rank + rank;
  } else {
    query = packed + uint64_t(row / lanes + 1U) * 3U * rank;
    gate = sender + uint64_t(row) * 2U * rank;
  }
  const float* value = gate + rank;
  for (uint32_t k = threadIdx.x; k < rank; k += blockDim.x) {
    const float logit = query[k] + sender_scale * gate[k] + bias[k];
    out[k] = (sender_scale * value[k]) / (1.0F + expf(-logit));
  }
}

/// out_t = h_t + rms_t * delta_t for t >= 1; row 0 passes through
/// (dspine.py:76-108, feature_finish). `out` may alias `hidden`.
__global__ void dspine_write_kernel(const float* hidden,
                                    const float* __restrict__ delta,
                                    const float* __restrict__ rms, float* out,
                                    uint32_t hidden_size) {
  const uint32_t row = blockIdx.y;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden_size)
    return;
  const uint64_t at = uint64_t(row) * hidden_size + i;
  out[at] = row == 0 ? hidden[at] : hidden[at] + rms[row] * delta[at];
}

__device__ __forceinline__ bool DSpineBetter(float score, uint32_t id,
                                             float other_score,
                                             uint32_t other_id) {
  return score > other_score || (score == other_score && id < other_id);
}

/// Each proposal row's top_k logits by (score desc, ID asc), stored in
/// ascending ID order as dspine.py:111-208 (topk_ids) returns them. `ids`
/// has rows + 1 rows of top_k: row 0 repeats the anchor (the refinement's
/// sender for proposal 0), row t + 1 holds proposal t's candidates.
__global__ void dspine_topk_kernel(const float* __restrict__ logits,
                                   uint32_t vocab_size, uint32_t top_k,
                                   const uint32_t* __restrict__ anchor,
                                   uint32_t* __restrict__ ids) {
  __shared__ float shared_scores[kDSpineThreads * kMaxSelectorTopK];
  __shared__ uint32_t shared_ids[kDSpineThreads * kMaxSelectorTopK];
  float scores[kMaxSelectorTopK];
  uint32_t lane_ids[kMaxSelectorTopK];
  for (uint32_t i = 0; i < kMaxSelectorTopK; ++i) {
    scores[i] = -INFINITY;
    lane_ids[i] = UINT32_MAX;
  }
  const float* row = logits + uint64_t(blockIdx.x) * vocab_size;
  for (uint32_t id = threadIdx.x; id < vocab_size; id += blockDim.x) {
    const float score = row[id];
    if (!DSpineBetter(score, id, scores[top_k - 1U], lane_ids[top_k - 1U]))
      continue;
    uint32_t at = top_k - 1U;
    while (at > 0U &&
           DSpineBetter(score, id, scores[at - 1U], lane_ids[at - 1U])) {
      scores[at] = scores[at - 1U];
      lane_ids[at] = lane_ids[at - 1U];
      --at;
    }
    scores[at] = score;
    lane_ids[at] = id;
  }
  float* mine_scores = shared_scores + threadIdx.x * kMaxSelectorTopK;
  uint32_t* mine_ids = shared_ids + threadIdx.x * kMaxSelectorTopK;
  for (uint32_t i = 0; i < top_k; ++i) {
    mine_scores[i] = scores[i];
    mine_ids[i] = lane_ids[i];
  }
  // Pairwise merges of the sorted lists; (score, ID) pairs are distinct, so
  // the merged prefix is the row's exact top_k.
  for (uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      const float* other_scores = mine_scores + stride * kMaxSelectorTopK;
      const uint32_t* other_ids = mine_ids + stride * kMaxSelectorTopK;
      uint32_t mine = 0, other = 0;
      for (uint32_t i = 0; i < top_k; ++i) {
        if (DSpineBetter(mine_scores[mine], mine_ids[mine], other_scores[other],
                         other_ids[other])) {
          scores[i] = mine_scores[mine];
          lane_ids[i] = mine_ids[mine++];
        } else {
          scores[i] = other_scores[other];
          lane_ids[i] = other_ids[other++];
        }
      }
      for (uint32_t i = 0; i < top_k; ++i) {
        mine_scores[i] = scores[i];
        mine_ids[i] = lane_ids[i];
      }
    }
  }
  if (threadIdx.x != 0U)
    return;
  for (uint32_t i = 1; i < top_k; ++i) {
    const uint32_t id = lane_ids[i];
    uint32_t at = i;
    for (; at > 0U && lane_ids[at - 1U] > id; --at)
      lane_ids[at] = lane_ids[at - 1U];
    lane_ids[at] = id;
  }
  uint32_t* out = ids + uint64_t(blockIdx.x + 1U) * top_k;
  for (uint32_t i = 0; i < top_k; ++i)
    out[i] = lane_ids[i];
  if (blockIdx.x == 0U) {
    for (uint32_t i = 0; i < top_k; ++i)
      ids[i] = *anchor;
  }
}

/// table[t][j][k] = head[cand_t,k] . RMSNorm_out(raw_t + rms_t * delta_tj):
/// proposal t, predecessor lane j, candidate lane k (dspine.py:571-585,
/// serial_rs_table). One workgroup per (t, j); `head_rows` row t * top_k + k
/// is candidate k's head row.
__global__ void dspine_table_kernel(const float* __restrict__ raw,
                                    const float* __restrict__ rms,
                                    const float* __restrict__ delta,
                                    const float* __restrict__ norm_weight,
                                    const float* __restrict__ head_rows,
                                    float* __restrict__ table, uint32_t top_k,
                                    uint32_t hidden_size, float eps) {
  extern __shared__ float refined[];
  __shared__ float scratch[32];
  __shared__ float partial[kMaxSelectorTopK][32];
  const uint32_t pair = blockIdx.x, proposal = pair / top_k;
  const float* base = raw + uint64_t(proposal + 1U) * hidden_size;
  const float* update = delta + uint64_t(pair) * hidden_size;
  const float scale = rms[proposal + 1U];
  float sum = 0.0F;
  for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float value = base[i] + scale * update[i];
    refined[i] = value;
    sum += value * value;
  }
  const float inverse =
      rsqrtf(DSpineBlockSum(sum, scratch) / float(hidden_size) + eps);
  const float* heads = head_rows + uint64_t(proposal) * top_k * hidden_size;
  float dots[kMaxSelectorTopK] = {};
  // Each thread reads back only the elements it wrote above.
  for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float value = refined[i] * inverse * norm_weight[i];
#pragma unroll
    for (uint32_t k = 0; k < kMaxSelectorTopK; ++k)
      if (k < top_k)
        dots[k] += heads[uint64_t(k) * hidden_size + i] * value;
  }
  const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
#pragma unroll
  for (uint32_t k = 0; k < kMaxSelectorTopK; ++k) {
    float value = dots[k];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
      value += __shfl_xor(value, offset);
    if (lane == 0U)
      partial[k][wave] = value;
  }
  __syncthreads();
  if (threadIdx.x < top_k) {
    float total = 0.0F;
    for (uint32_t w = 0; w < blockDim.x / warpSize; ++w)
      total += partial[threadIdx.x][w];
    table[uint64_t(pair) * top_k + threadIdx.x] = total;
  }
}

/// Walks the tables from the anchor (dspine.py:287-320 greedy, 589-641
/// sampled): proposal 0 reads lane 0 (all its senders are the anchor), then
/// k_t = argmax_k table[t][j_t][k] (ties: lower lane, so lower ID) or a draw
/// from softmax(table[t][j_t] / temperature), and j_(t+1) = k_t. The sampled
/// q is the softmax actually drawn from, conditional on the previous pick,
/// with FinalizeSelector's inverse-CDF convention.
__global__ void dspine_chain_kernel(
    const float* __restrict__ table, const uint32_t* __restrict__ ids,
    uint32_t count, uint32_t top_k, float temperature,
    const float* __restrict__ uniforms, uint32_t* __restrict__ tokens,
    float* __restrict__ confidences, uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities) {
  if (threadIdx.x != 0U)
    return;
  uint32_t lane = 0;
  for (uint32_t t = 0; t < count; ++t) {
    const float* scores = table + (uint64_t(t) * top_k + lane) * top_k;
    const uint32_t* candidates = ids + uint64_t(t + 1U) * top_k;
    uint32_t best = 0;
    for (uint32_t k = 1; k < top_k; ++k)
      if (scores[k] > scores[best])
        best = k;
    float weights[kMaxSelectorTopK];
    float denominator = 1.0F;
    uint32_t pick = best;
    if (temperature > 0.0F) {
      const float divisor = fmaxf(temperature, FLT_MIN);
      denominator = 0.0F;
      for (uint32_t k = 0; k < top_k; ++k) {
        weights[k] = expf((scores[k] - scores[best]) / divisor);
        denominator += weights[k];
      }
      const float inverse = 1.0F / fmaxf(denominator, 1e-30F);
      float threshold =
          fminf(fmaxf(uniforms[t], 0.0F), 0.99999994F) * denominator;
      bool selected = false;
      for (uint32_t k = 0; k < top_k; ++k) {
        const float weight = weights[k];
        weights[k] = weight * inverse;
        if (!selected && weights[k] > 0.0F)
          pick = k;
        threshold -= weight;
        if (threshold < 0.0F && !selected) {
          pick = k;
          selected = true;
        }
      }
    } else {
      for (uint32_t k = 0; k < top_k; ++k)
        weights[k] = k == best ? 1.0F : 0.0F;
    }
    tokens[t] = candidates[pick];
    if (confidences != nullptr)
      confidences[t] = weights[pick];
    for (uint32_t k = 0; k < top_k; ++k) {
      if (candidate_ids != nullptr)
        candidate_ids[uint64_t(t) * top_k + k] = candidates[k];
      if (candidate_probabilities != nullptr)
        candidate_probabilities[uint64_t(t) * top_k + k] = weights[k];
    }
    lane = pick;
  }
}

void LaunchDSpineRowStats(const float* hidden, float* normalized, float* rms,
                          std::uint32_t rows, std::uint32_t hidden_size,
                          float eps, hipStream_t stream) {
  dspine_row_stats_kernel<<<rows, kDSpineThreads, 0, stream>>>(
      hidden, normalized, rms, hidden_size, eps);
}

void LaunchDSpineCodes(float* codes, const float* anchor_code,
                       std::uint32_t rows, std::uint32_t code_size, float eps,
                       hipStream_t stream) {
  dspine_codes_kernel<<<rows, kDSpineThreads, 0, stream>>>(codes, anchor_code,
                                                           code_size, eps);
}

void LaunchDSpineMessage(const float* packed, const float* sender,
                         const float* bias, float* message, std::uint32_t rows,
                         std::uint32_t rank, std::uint32_t lanes,
                         float sender_scale, hipStream_t stream) {
  dspine_message_kernel<<<rows, kDSpineThreads, 0, stream>>>(
      packed, sender, bias, message, rank, lanes, sender_scale);
}

void LaunchDSpineWrite(const float* hidden, const float* delta,
                       const float* rms, float* out, std::uint32_t rows,
                       std::uint32_t hidden_size, hipStream_t stream) {
  dspine_write_kernel<<<
      dim3((hidden_size + kDSpineThreads - 1U) / kDSpineThreads, rows),
      kDSpineThreads, 0, stream>>>(hidden, delta, rms, out, hidden_size);
}

void LaunchDSpineTopK(const float* logits, std::uint32_t rows,
                      std::uint32_t vocab_size, std::uint32_t top_k,
                      const std::uint32_t* anchor, std::uint32_t* ids,
                      hipStream_t stream) {
  if (rows == 0 || top_k < 2 || top_k > kMaxSelectorTopK || top_k > vocab_size)
    throw std::invalid_argument("invalid DSpine top-k shape");
  dspine_topk_kernel<<<rows, kDSpineThreads, 0, stream>>>(logits, vocab_size,
                                                          top_k, anchor, ids);
}

void LaunchDSpineTable(const float* raw, const float* rms, const float* delta,
                       const float* norm_weight, const float* head_rows,
                       float* table, std::uint32_t rows, std::uint32_t top_k,
                       std::uint32_t hidden_size, float eps,
                       hipStream_t stream) {
  if (top_k > kMaxSelectorTopK || hidden_size * sizeof(float) > 48U * 1024U)
    throw std::invalid_argument("invalid DSpine table shape");
  dspine_table_kernel<<<rows * top_k, kDSpineThreads,
                        hidden_size * sizeof(float), stream>>>(
      raw, rms, delta, norm_weight, head_rows, table, top_k, hidden_size, eps);
}

void LaunchDSpineChain(const float* table, const std::uint32_t* ids,
                       std::uint32_t rows, std::uint32_t top_k,
                       float temperature, const float* uniforms,
                       std::uint32_t* tokens, float* confidences,
                       std::uint32_t* candidate_ids,
                       float* candidate_probabilities, hipStream_t stream) {
  if (top_k > kMaxSelectorTopK || (temperature > 0.0F && uniforms == nullptr))
    throw std::invalid_argument("invalid DSpine chain");
  dspine_chain_kernel<<<1, 32, 0, stream>>>(
      table, ids, rows, top_k, temperature, uniforms, tokens, confidences,
      candidate_ids, candidate_probabilities);
}

// DSpine draft attention in three phases (commit 3). The wave kernel above
// gives every (block row, head pair) its own CTA, and each CTA reads its KV
// head's whole window: at 16 rows and a 4k context that is 256 CTAs x 4 MB.
// Here the scores are split over KEYS (each K row is read once per KV head
// and held in registers while all rows x heads of that KV head reuse it), the
// softmax runs one 256-thread CTA per four score rows on a global score array,
// and PV is split over DIM. Ported from our gufo fork's GUFO_DEV_DRAFTK=2
// route (0a1521c1) with the causal block mask and partial PV groups added.
// Every float chain stays in one thread in the wave kernel's order (the
// per-lane K float4 dot and xor butterfly, the strided 256-thread softmax
// partials and LDS tree, the ascending-key PV sum with the same 32-key
// prefetch), so the output equals the causal wave route bitwise.
// Scores are laid out [block row * num_q_heads + head][total_k].

/// `KRegs` = ceil(head_dim / 4 / 32), the K float4s one lane keeps; a
/// template parameter because a runtime-indexed local array is promoted to LDS.
template<uint32_t KRegs>
__global__ __launch_bounds__(256) void dspine_attention_scores_kernel(
    const float* __restrict__ q, const float* __restrict__ injected_k,
    const float* __restrict__ block_k, float* __restrict__ scores,
    uint32_t current_pos, uint32_t history_start, uint32_t history_count,
    uint32_t block_count, uint32_t sliding_window, uint32_t num_q_heads,
    uint32_t num_kv_heads, uint32_t head_dim, float scale,
    uint32_t history_capacity, uint32_t keys_per_block) {
  const uint32_t kv_h = blockIdx.y;
  const uint32_t heads_per_kv = num_q_heads / num_kv_heads;
  const uint32_t q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t total_k = history_count + block_count;
  const uint32_t num_vec = head_dim / 4U, lane = threadIdx.x & 31U,
                 wave = threadIdx.x >> 5U, waves = blockDim.x >> 5U;
  const uint32_t key_begin = blockIdx.x * keys_per_block;
  const uint32_t key_limit = key_begin + keys_per_block;
  const uint32_t key_end = key_limit < total_k ? key_limit : total_k;

  for (uint32_t key = key_begin + wave; key < key_end; key += waves) {
    const bool in_block = key >= history_count;
    const uint32_t block_index = in_block ? key - history_count : 0U;
    const uint32_t key_position =
        in_block ? current_pos + block_index : history_start + key;
    const float4* source =
        in_block ? reinterpret_cast<const float4*>(
                       block_k + uint64_t(block_index) * kv_dim +
                       uint64_t(kv_h) * head_dim)
                 : reinterpret_cast<const float4*>(
                       injected_k +
                       uint64_t(key_position % history_capacity) * kv_dim +
                       uint64_t(kv_h) * head_dim);
    float4 held[KRegs];
#pragma unroll
    for (uint32_t i = 0; i < KRegs; ++i) {
      const uint32_t d = lane + (i * 32U);
      if (d < num_vec)
        held[i] = source[d];
    }
    for (uint32_t row = 0; row < block_count; ++row) {
      const uint32_t query_position = current_pos + row;
      bool visible;
      if (in_block) {
        const uint32_t distance = query_position > key_position
                                      ? query_position - key_position
                                      : key_position - query_position;
        visible = distance < sliding_window && block_index <= row;
      } else {
        visible = query_position - key_position < sliding_window;
      }
      const float4* q_vec = reinterpret_cast<const float4*>(
          q + uint64_t(row) * q_dim + uint64_t(kv_h * heads_per_kv) * head_dim);
      float* out_row =
          scores +
          (uint64_t(row) * num_q_heads + (kv_h * heads_per_kv)) * total_k + key;
      for (uint32_t r = 0; r < heads_per_kv; ++r) {
        float dot = 0.f;
#pragma unroll
        for (uint32_t i = 0; i < KRegs; ++i) {
          const uint32_t d = lane + (i * 32U);
          if (d >= num_vec)
            break;
          const float4 kv = held[i];
          const float4 qv = q_vec[d];
          dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
          dot += __shfl_xor(dot, offset);
        if (lane == 0)
          *out_row = visible ? dot * scale : -1e20F;
        q_vec += num_vec;
        out_row += total_k;
      }
    }
  }
}

/// The wave kernel's exact softmax, on `Combos` score rows per CTA. The
/// strided partials and the LDS tree pin blockDim.x at 256.
template<uint32_t Combos>
__global__ __launch_bounds__(256) void dspine_attention_softmax_kernel(
    float* __restrict__ scores, uint32_t total_k, uint32_t num_q_heads) {
  __shared__ float reduce[Combos][256];
  float* row[Combos];
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    row[c] = scores +
             (uint64_t(blockIdx.x) * num_q_heads + (blockIdx.y * Combos) + c) *
                 total_k;
  float maxima[Combos];
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    maxima[c] = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c)
      maxima[c] = fmaxf(maxima[c], row[c][p]);
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    reduce[c][threadIdx.x] = maxima[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < Combos; ++c)
        reduce[c][threadIdx.x] =
            fmaxf(reduce[c][threadIdx.x], reduce[c][threadIdx.x + stride]);
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    maxima[c] = reduce[c][0];
  __syncthreads();
  float sums[Combos] = {};
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c) {
      const float e = expf(row[c][p] - maxima[c]);
      row[c][p] = e;
      sums[c] += e;
    }
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    reduce[c][threadIdx.x] = sums[c];
  __syncthreads();
  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (unsigned c = 0; c < Combos; ++c)
        reduce[c][threadIdx.x] += reduce[c][threadIdx.x + stride];
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned c = 0; c < Combos; ++c)
    sums[c] = 1.f / (reduce[c][0] + 1e-9F);
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
#pragma unroll
    for (unsigned c = 0; c < Combos; ++c)
      row[c][p] *= sums[c];
  }
}

/// PV over one KV head: grid (combo groups, dim tiles, kv heads). A wave owns
/// one 32-wide dim tile and `Accumulators` combos (block row x head of this KV
/// head); combos past the end of the last group compute nothing they store.
template<uint32_t Accumulators>
__global__ __launch_bounds__(256) void dspine_attention_pv_kernel(
    const float* __restrict__ injected_v, const float* __restrict__ block_v,
    const float* __restrict__ scores, float* __restrict__ out,
    uint32_t history_start, uint32_t history_count, uint32_t block_count,
    uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
    uint32_t history_capacity) {
  constexpr uint32_t kPrefetch = 32;
  constexpr uint32_t kDimTile = 32;
  const uint32_t kv_h = blockIdx.z;
  const uint32_t heads_per_kv = num_q_heads / num_kv_heads;
  const uint32_t q_dim = num_q_heads * head_dim,
                 kv_dim = num_kv_heads * head_dim;
  const uint32_t total_k = history_count + block_count;
  const uint32_t combos = block_count * heads_per_kv;
  const uint32_t lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U,
                 waves = blockDim.x >> 5U;
  const uint32_t d = (blockIdx.y * kDimTile) + lane;
  const uint32_t combo_base = (blockIdx.x * waves * Accumulators) + wave;

  const float* srow[Accumulators];
#pragma unroll
  for (unsigned j = 0; j < Accumulators; ++j) {
    const uint32_t combo = min(combo_base + (j * waves), combos - 1U);
    const uint32_t row = combo / heads_per_kv;
    const uint32_t head = (kv_h * heads_per_kv) + (combo % heads_per_kv);
    srow[j] = scores + (uint64_t(row) * num_q_heads + head) * total_k;
  }
  float acc[Accumulators] = {};
  uint32_t slot = history_count > 0 ? history_start % history_capacity : 0;
  const float* vbase = injected_v + (uint64_t(kv_h) * head_dim) + d;
  uint32_t local = 0;
  for (; history_count - local >= kPrefetch; local += kPrefetch) {
    float values[kPrefetch];
#pragma unroll
    for (unsigned i = 0; i < kPrefetch; ++i) {
      values[i] = vbase[uint64_t(slot) * kv_dim];
      slot = slot + 1 == history_capacity ? 0 : slot + 1;
    }
#pragma unroll
    for (unsigned i = 0; i < kPrefetch; ++i) {
#pragma unroll
      for (unsigned j = 0; j < Accumulators; ++j)
        acc[j] += srow[j][local + i] * values[i];
    }
  }
  for (; local < history_count; ++local) {
    const float value = vbase[uint64_t(slot) * kv_dim];
#pragma unroll
    for (unsigned j = 0; j < Accumulators; ++j)
      acc[j] += srow[j][local] * value;
    slot = slot + 1 == history_capacity ? 0 : slot + 1;
  }
  for (uint32_t p = 0; p < block_count; ++p) {
    const float value =
        block_v[uint64_t(p) * kv_dim + uint64_t(kv_h) * head_dim + d];
#pragma unroll
    for (unsigned j = 0; j < Accumulators; ++j)
      acc[j] += srow[j][history_count + p] * value;
  }
#pragma unroll
  for (unsigned j = 0; j < Accumulators; ++j) {
    const uint32_t combo = combo_base + (j * waves);
    if (combo >= combos)
      continue;
    const uint32_t row = combo / heads_per_kv;
    const uint32_t head = (kv_h * heads_per_kv) + (combo % heads_per_kv);
    out[(uint64_t(row) * q_dim) + (uint64_t(head) * head_dim) + d] = acc[j];
  }
}

std::size_t DSpineAttentionScoreElements(std::uint32_t block_count,
                                         std::uint32_t num_q_heads,
                                         std::uint32_t history_capacity) {
  return static_cast<std::size_t>(block_count) * num_q_heads *
         (static_cast<std::size_t>(history_capacity) + block_count);
}

bool LaunchDSpineAttention(
    const float* q, const float* injected_k, const float* injected_v,
    const float* block_k, const float* block_v, float* out,
    std::uint32_t current_pos, std::uint32_t history_length,
    std::uint32_t block_count, std::uint32_t sliding_window,
    std::uint32_t num_q_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim, float scale, hipStream_t stream,
    std::uint32_t history_capacity, float* scores) {
  // The softmax tree pins 256 threads; a wave reads one 32-float V line.
  constexpr std::uint32_t kThreads = 256, kWaves = kThreads / 32U;
  constexpr std::uint32_t kDimTile = 32, kCombos = 4, kAccumulators = 2;
  constexpr std::uint32_t kTargetCtas = 160;  // 40 CUs, a little over 4x
  if (scores == nullptr || block_count == 0 || num_kv_heads == 0 ||
      num_q_heads % num_kv_heads != 0 || num_q_heads % kCombos != 0 ||
      head_dim % kDimTile != 0 || head_dim / 4U > 32U * 4U)
    return false;
  const std::uint32_t history_start = current_pos + 1U > sliding_window
                                          ? current_pos + 1U - sliding_window
                                          : 0U;
  const std::uint32_t history_count =
      history_length > history_start ? history_length - history_start : 0U;
  if (history_count > history_capacity)
    return false;
  const std::uint32_t total_k = history_count + block_count;
  const std::uint32_t splits = (kTargetCtas + num_kv_heads - 1U) / num_kv_heads;
  std::uint32_t keys_per_block = (total_k + splits - 1U) / splits;
  keys_per_block = (keys_per_block + kWaves - 1U) / kWaves * kWaves;
  const std::uint32_t score_blocks =
      (total_k + keys_per_block - 1U) / keys_per_block;
  const auto launch_scores = [&](auto regs) {
    dspine_attention_scores_kernel<decltype(regs)::value>
        <<<dim3(score_blocks, num_kv_heads), kThreads, 0, stream>>>(
            q, injected_k, block_k, scores, current_pos, history_start,
            history_count, block_count, sliding_window, num_q_heads,
            num_kv_heads, head_dim, scale, history_capacity, keys_per_block);
  };
  if (head_dim / 4U <= 32U)
    launch_scores(std::integral_constant<std::uint32_t, 1U>{});
  else
    launch_scores(std::integral_constant<std::uint32_t, 4U>{});
  dspine_attention_softmax_kernel<kCombos>
      <<<dim3(block_count, num_q_heads / kCombos), kThreads, 0, stream>>>(
          scores, total_k, num_q_heads);
  const std::uint32_t combos = block_count * (num_q_heads / num_kv_heads);
  const std::uint32_t per_group = kWaves * kAccumulators;
  dspine_attention_pv_kernel<kAccumulators>
      <<<dim3((combos + per_group - 1U) / per_group, head_dim / kDimTile,
              num_kv_heads),
         kThreads, 0, stream>>>(injected_v, block_v, scores, out, history_start,
                                history_count, block_count, num_q_heads,
                                num_kv_heads, head_dim, history_capacity);
  return true;
}

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
