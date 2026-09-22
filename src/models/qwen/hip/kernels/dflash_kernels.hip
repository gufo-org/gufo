#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

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
template<uint32_t QueryHeads>
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
            distance < sliding_window ? dot[g] * scale : -1e20F;
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

__global__ void dflash_selector_partial_topk_kernel(
    const float* __restrict__ logits, float* __restrict__ partial_scores,
    uint32_t* __restrict__ partial_ids, uint32_t vocab_size,
    uint32_t selector_top_k) {
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
  const uint32_t end = requested_end < vocab_size ? requested_end : vocab_size;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    InsertSelectorCandidate(logits[token], token, local_scores, local_ids,
                            selector_top_k);
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

__global__ void dflash_selector_finalize_kernel(
    const float* __restrict__ partial_scores,
    const uint32_t* __restrict__ partial_ids,
    const float* __restrict__ projected_hidden,
    const hip_bfloat16* __restrict__ predecessor_codebook,
    const hip_bfloat16* __restrict__ successor_codebook,
    const uint32_t* __restrict__ predecessor_token,
    uint32_t* __restrict__ output_token, float* __restrict__ output_confidence,
    float temperature, const float* __restrict__ sample_uniform,
    uint32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_probabilities, uint32_t partial_count,
    uint32_t selector_rank, uint32_t selector_top_k) {
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
    const float score = top_scores[candidate_index] + transition;
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
    uint32_t sampled_index = selector_top_k - 1U;
    for (uint32_t index = 0; index < selector_top_k; ++index) {
      const float probability = path_scores[index] * inverse_denominator;
      if (candidate_probabilities != nullptr) {
        candidate_probabilities[index] = probability;
      }
      threshold -= path_scores[index];
      if (threshold < 0.0F && sampled_index == selector_top_k - 1U) {
        sampled_index = index;
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
    std::uint32_t history_capacity) {
  const auto geometry = DFlashAttentionGeometry(
      current_pos, history_length, block_count, sliding_window, num_q_heads);
  // Wider blocks benefit from two query heads per CTA. Keep narrow blocks
  // independent and leave enough LDS for both 256-thread reduction rows.
  const auto heads_per_kv = num_q_heads / num_kv_heads;
  if (block_count >= 4 && (heads_per_kv % 2U) == 0U &&
      geometry.shared_bytes <= (64U * 1024U - 2U * 256U * sizeof(float)) / 2U) {
    auto grid = geometry.grid;
    grid.y /= 2U;
    dflash_noncausal_attention_wave_kernel<2>
        <<<grid, 256, 2U * geometry.shared_bytes, stream>>>(
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
    std::uint32_t history_capacity) {
  // Coalesced float4 reads cover all shipped DFlash2 head dimensions.
  if ((head_dim % 4U) == 0U) {
    LaunchDFlashNonCausalAttentionWave(
        q, injected_k, injected_v, block_k, block_v, out, current_pos,
        history_length, block_count, sliding_window, num_q_heads, num_kv_heads,
        head_dim, scale, stream, history_capacity);
    return;
  }
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

}  // namespace gufo::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
