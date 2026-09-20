#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/models/qwen3_asr/hip/text_ops.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {

// Preserve the descending shared-memory sum tree. Once only the first wave
// remains, register shuffles replace five workgroup barriers and LDS rounds.
__device__ __forceinline__ void FinishNormSum(float* sums) {
  const auto thread = threadIdx.x;
  for (unsigned int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (thread < stride)
      sums[thread] += sums[thread + stride];
    __syncthreads();
  }
  if (thread < 32) {
    float sum = sums[thread];
    for (int offset = 16; offset != 0; offset /= 2)
      sum += __shfl_down(sum, offset, 32);
    if (thread == 0)
      sums[0] = sum;
  }
  __syncthreads();
}

// Sixteen-byte load granule. A native vector type is used rather than HIP's
// uint4 so the compiler emits a single global_load_dwordx4 and so the same type
// can back the LDS staging buffer.
using Vec16 = __attribute__((__vector_size__(16))) unsigned int;

/// Number of row loads a wave issues before consuming the first. Eight is the
/// measured optimum on gfx1151: four leaves the memory pipeline short of the
/// DRAM read ceiling and sixteen costs enough registers to lose occupancy.
constexpr std::uint32_t kDecodeGemvUnroll = 8;
constexpr std::uint32_t kDecodeGemvWavesPerBlock = 4;
constexpr std::uint32_t kDecodeGemvBlockThreads = kDecodeGemvWavesPerBlock * 32;

struct DecodeGemvTensors {
  const hip_bfloat16* weights[kTextDecodeGemvMaxTensors];
  float* outputs[kTextDecodeGemvMaxTensors];
  std::uint32_t first_row[kTextDecodeGemvMaxTensors];
  std::uint32_t count;
  std::uint32_t total_rows;
};

__device__ __forceinline__ float Bf16BitsToFloat(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

/// Eight BF16 products summed left to right. This is the accumulation order the
/// unfused decode GEMV uses, and the generated token IDs depend on it.
__device__ __forceinline__ float DecodeDot8(Vec16 weights, Vec16 activations) {
  const auto* w = reinterpret_cast<const std::uint16_t*>(&weights);
  const auto* a = reinterpret_cast<const std::uint16_t*>(&activations);
  return (Bf16BitsToFloat(w[0]) * Bf16BitsToFloat(a[0])) +
         (Bf16BitsToFloat(w[1]) * Bf16BitsToFloat(a[1])) +
         (Bf16BitsToFloat(w[2]) * Bf16BitsToFloat(a[2])) +
         (Bf16BitsToFloat(w[3]) * Bf16BitsToFloat(a[3])) +
         (Bf16BitsToFloat(w[4]) * Bf16BitsToFloat(a[4])) +
         (Bf16BitsToFloat(w[5]) * Bf16BitsToFloat(a[5])) +
         (Bf16BitsToFloat(w[6]) * Bf16BitsToFloat(a[6])) +
         (Bf16BitsToFloat(w[7]) * Bf16BitsToFloat(a[7]));
}

__launch_bounds__(kDecodeGemvBlockThreads, 4) __global__
    void TextDecodeGemvKernel(DecodeGemvTensors tensors,
                              const hip_bfloat16* __restrict__ input,
                              std::uint32_t k_vectors) {
  // Staged before the row-range test so every thread in the block reaches the
  // barrier.
  extern __shared__ Vec16 shared_input[];
  const auto* input_vec = reinterpret_cast<const Vec16*>(input);
  for (std::uint32_t index = threadIdx.x; index < k_vectors;
       index += blockDim.x) {
    shared_input[index] = input_vec[index];
  }
  __syncthreads();

  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t row =
      (blockIdx.x * kDecodeGemvWavesPerBlock) + (threadIdx.x >> 5U);
  if (row >= tensors.total_rows) {
    return;
  }
  std::uint32_t tensor = 0U;
#pragma unroll
  for (std::uint32_t candidate = 1U; candidate < kTextDecodeGemvMaxTensors;
       ++candidate) {
    if (candidate < tensors.count && row >= tensors.first_row[candidate]) {
      tensor = candidate;
    }
  }
  const std::uint32_t local_row = row - tensors.first_row[tensor];
  const auto* row_vec = reinterpret_cast<const Vec16*>(
      tensors.weights[tensor] +
      (static_cast<std::size_t>(local_row) * k_vectors * 8U));

  constexpr std::uint32_t kStride = 32U * kDecodeGemvUnroll;
  float dot = 0.0F;
  std::uint32_t index = lane;
  for (; index + (kStride - 32U) < k_vectors; index += kStride) {
    Vec16 weights[kDecodeGemvUnroll];
#pragma unroll
    for (std::uint32_t step = 0U; step < kDecodeGemvUnroll; ++step) {
      weights[step] = row_vec[index + (step * 32U)];
    }
#pragma unroll
    for (std::uint32_t step = 0U; step < kDecodeGemvUnroll; ++step) {
      dot += DecodeDot8(weights[step], shared_input[index + (step * 32U)]);
    }
  }
  for (; index < k_vectors; index += 32U) {
    dot += DecodeDot8(row_vec[index], shared_input[index]);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane == 0U) {
    tensors.outputs[tensor][local_row] = dot;
  }
}

// ---------------------------------------------------------------------------
// Kernels that were previously reached through src/models/qwen. They are
// reproduced here so this model's sources stand alone, restricted to the BF16
// weights and the single route this checkpoint uses, and with the arithmetic of
// the shared originals preserved -- the generated token IDs depend on it.
// ---------------------------------------------------------------------------

__launch_bounds__(256) __global__
    void TextEmbeddingLookupKernel(const hip_bfloat16* __restrict__ table,
                                   const std::uint32_t* __restrict__ token_ids,
                                   float* __restrict__ output,
                                   std::size_t batch_size,
                                   std::size_t hidden_size) {
  const std::size_t batch = blockIdx.y;
  if (batch >= batch_size) {
    return;
  }
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= hidden_size) {
    return;
  }
  const std::size_t row =
      static_cast<std::size_t>(token_ids[batch]) * hidden_size;
  output[(batch * hidden_size) + index] =
      static_cast<float>(table[row + index]);
}

/// One wave per output row, lane-strided over eight-element groups, reduced by
/// a butterfly. Matches the shared wave32 single-row GEMV.
template<std::uint32_t WavesPerBlock>
__launch_bounds__(WavesPerBlock * 32, 4) __global__
    void TextRowGemvKernel(const hip_bfloat16* __restrict__ weights,
                           const float* __restrict__ input,
                           float* __restrict__ output, std::size_t rows,
                           std::size_t columns) {
  const std::size_t lane = threadIdx.x & 31U;
  const std::size_t row = (blockIdx.x * WavesPerBlock) + (threadIdx.x >> 5U);
  if (row >= rows) {
    return;
  }
  const auto* weight_row = weights + (row * columns);
  const auto* weight_vec = reinterpret_cast<const Vec16*>(weight_row);
  const std::size_t groups = columns / 8U;

  float dot = 0.0F;
  for (std::size_t group = lane; group < groups; group += 32U) {
    const Vec16 packed = weight_vec[group];
    const auto* values = reinterpret_cast<const std::uint16_t*>(&packed);
    const std::size_t base = group * 8U;
    const float4 low = *reinterpret_cast<const float4*>(&input[base]);
    const float4 high = *reinterpret_cast<const float4*>(&input[base + 4U]);
    dot += (Bf16BitsToFloat(values[0]) * low.x) +
           (Bf16BitsToFloat(values[1]) * low.y) +
           (Bf16BitsToFloat(values[2]) * low.z) +
           (Bf16BitsToFloat(values[3]) * low.w) +
           (Bf16BitsToFloat(values[4]) * high.x) +
           (Bf16BitsToFloat(values[5]) * high.y) +
           (Bf16BitsToFloat(values[6]) * high.z) +
           (Bf16BitsToFloat(values[7]) * high.w);
  }
  for (std::size_t column = (groups * 8U) + lane; column < columns;
       column += 32U) {
    dot += static_cast<float>(weight_row[column]) * input[column];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }
  if (lane == 0U) {
    output[row] = dot;
  }
}

/// Shortest sequence scored with the wave-parallel branch. Below it a whole row
/// is summed in one thread, which is a different accumulation order; both
/// branches are reproduced so behaviour matches the shared kernel at every
/// context length.
constexpr std::size_t kBatchedAttentionWaveMinSequence = 64;

__launch_bounds__(256) __global__ void TextBatchedAttentionKernel(
    const float* __restrict__ query, const float* __restrict__ key_cache,
    const float* __restrict__ value_cache, float* __restrict__ out_context,
    hip_bfloat16* __restrict__ out_context_bfloat16, std::uint32_t layer_index,
    std::uint32_t start_position, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t token = blockIdx.y;
  if (head >= num_heads || token >= batch_size) {
    return;
  }
  const std::size_t thread = threadIdx.x;
  const std::uint32_t key_value_head = head / (num_heads / num_key_value_heads);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  __shared__ float reduction[256];
  extern __shared__ float scores[];

  const std::uint32_t position = start_position + token;
  const std::size_t sequence = static_cast<std::size_t>(position) + 1U;
  const float* query_head = query + (token * num_heads * head_dim) +
                            (static_cast<std::size_t>(head) * head_dim);
  const auto* query_vec = reinterpret_cast<const float4*>(query_head);
  const std::size_t vectors = head_dim / 4U;

  const auto key_offset = [&](std::size_t key) {
    return (((static_cast<std::size_t>(layer_index) * num_key_value_heads) +
             key_value_head) *
                max_context +
            key) *
           head_dim;
  };

  if (sequence >= kBatchedAttentionWaveMinSequence) {
    const std::size_t lane = thread & 31U;
    const std::size_t wave = thread >> 5U;
    const std::size_t waves = blockDim.x >> 5U;
    for (std::size_t key = wave; key < sequence; key += waves) {
      const auto* key_vec =
          reinterpret_cast<const float4*>(key_cache + key_offset(key));
      float dot = 0.0F;
      for (std::size_t vector = lane; vector < vectors; vector += 32U) {
        const float4 q = query_vec[vector];
        const float4 k = key_vec[vector];
        dot += (q.x * k.x) + (q.y * k.y) + (q.z * k.z) + (q.w * k.w);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor(dot, offset);
      }
      if (lane == 0U) {
        scores[key] = static_cast<float>(
            hip_bfloat16(static_cast<float>(hip_bfloat16(dot)) * scale));
      }
    }
  } else {
    for (std::size_t key = thread; key < sequence; key += blockDim.x) {
      const auto* key_vec =
          reinterpret_cast<const float4*>(key_cache + key_offset(key));
      float dot = 0.0F;
      for (std::size_t vector = 0; vector < vectors; ++vector) {
        const float4 q = query_vec[vector];
        const float4 k = key_vec[vector];
        dot += (q.x * k.x) + (q.y * k.y) + (q.z * k.z) + (q.w * k.w);
      }
      scores[key] = static_cast<float>(
          hip_bfloat16(static_cast<float>(hip_bfloat16(dot)) * scale));
    }
  }
  __syncthreads();

  float local_maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    local_maximum = fmaxf(local_maximum, scores[key]);
  }
  reduction[thread] = local_maximum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    const float value = expf(scores[key] - maximum);
    scores[key] = value;
    local_sum += value;
  }
  reduction[thread] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      reduction[thread] += reduction[thread + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  __syncthreads();
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    scores[key] = static_cast<float>(hip_bfloat16(scores[key] * inverse_sum));
  }
  __syncthreads();

  if (thread < head_dim) {
    float context = 0.0F;
    for (std::size_t key = 0; key < sequence; ++key) {
      context += scores[key] * value_cache[key_offset(key) + thread];
    }
    const std::size_t index = (token * num_heads * head_dim) +
                              (static_cast<std::size_t>(head) * head_dim) +
                              thread;
    out_context[index] = context;
    if (out_context_bfloat16 != nullptr) {
      out_context_bfloat16[index] = hip_bfloat16(context);
    }
  }
}

__device__ void NormalizeAndRotateHead(float* values, const float* weight,
                                       std::uint32_t head_dim, float epsilon,
                                       std::uint32_t position, float rope_theta,
                                       float* shared_sum, float* normalized) {
  const std::uint32_t thread = threadIdx.x;
  float sum_squared = 0.0F;
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(values[index]));
    sum_squared += value * value;
  }
  shared_sum[thread] = sum_squared;
  __syncthreads();
  FinishNormSum(shared_sum);
  const float inverse_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(head_dim)) + epsilon);
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(values[index]));
    const float normalized_value =
        static_cast<float>(hip_bfloat16(value * inverse_rms));
    normalized[index] =
        static_cast<float>(hip_bfloat16(normalized_value * weight[index]));
  }
  __syncthreads();

  const std::uint32_t half = head_dim / 2U;
  for (std::uint32_t dimension = thread; dimension < half;
       dimension += blockDim.x) {
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2U * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float cosine = static_cast<float>(hip_bfloat16(cosf(angle)));
    const float sine = static_cast<float>(hip_bfloat16(sinf(angle)));
    const float first = normalized[dimension];
    const float second = normalized[dimension + half];
    const float first_cosine = static_cast<float>(hip_bfloat16(first * cosine));
    const float negative_second_sine =
        static_cast<float>(hip_bfloat16(-second * sine));
    const float second_cosine =
        static_cast<float>(hip_bfloat16(second * cosine));
    const float first_sine = static_cast<float>(hip_bfloat16(first * sine));
    values[dimension] =
        static_cast<float>(hip_bfloat16(first_cosine + negative_second_sine));
    values[dimension + half] =
        static_cast<float>(hip_bfloat16(second_cosine + first_sine));
  }
  __syncthreads();
}

template<std::uint32_t Threads>
__launch_bounds__(Threads) __global__ void TextQkNormRoPEKernel(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    float* key_cache, float* value_cache, std::uint32_t layer_index,
    std::uint32_t max_context) {
  const std::size_t batch = blockIdx.y;
  const std::uint32_t head = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  __shared__ float shared_sum[Threads];
  __shared__ float normalized[Threads];
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(batch);

  if (head < num_heads) {
    const std::size_t stride = static_cast<std::size_t>(num_heads) * head_dim;
    NormalizeAndRotateHead(
        query + (batch * stride) + (static_cast<std::size_t>(head) * head_dim),
        query_weight, head_dim, epsilon, position, rope_theta, shared_sum,
        normalized);
  }
  if (head < num_key_value_heads) {
    const std::size_t stride =
        static_cast<std::size_t>(num_key_value_heads) * head_dim;
    const std::size_t offset =
        (batch * stride) + (static_cast<std::size_t>(head) * head_dim);
    NormalizeAndRotateHead(key + offset, key_weight, head_dim, epsilon,
                           position, rope_theta, shared_sum, normalized);
    const std::size_t cache_offset =
        ((((static_cast<std::size_t>(layer_index) * num_key_value_heads) +
           head) *
              max_context +
          position) *
         head_dim);
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x) {
      const float rounded =
          static_cast<float>(hip_bfloat16(value[offset + index]));
      value[offset + index] = rounded;
      if (key_cache != nullptr) {
        key_cache[cache_offset + index] = key[offset + index];
        value_cache[cache_offset + index] = rounded;
      }
    }
  }
}

// Batch-one decode attention.
//
// The shared batched kernel launches one block per head -- sixteen blocks on a
// forty-CU part -- and its score loop issues a single 16-byte key load per wave
// per position, then immediately reduces it. With that little in flight the
// kernel runs at roughly an eighth of the DRAM read rate and is bound by load
// latency, not bandwidth.
//
// This kernel keeps every arithmetic step of the shared one and only raises the
// number of loads in flight: each wave scores `Unroll` positions at a time and
// the value accumulation prefetches `Unroll` rows before consuming the first.
// Per-position score reductions are independent, and the context accumulator
// stays single and in ascending position order, so every result is
// bit-identical and greedy decode still reproduces the official token IDs.
constexpr std::uint32_t kDecodeAttentionThreads = 256;
constexpr std::uint32_t kDecodeAttentionUnroll = 8;

/// Shortest sequence the shared kernel scores with its wave-parallel branch.
/// Below this it sums a whole row in one thread, which is a different
/// accumulation order, so short contexts must keep using the shared kernel.
constexpr std::uint32_t kDecodeAttentionMinSequence = 64;

__launch_bounds__(kDecodeAttentionThreads) __global__
    void TextDecodeAttentionKernel(
        const float* __restrict__ query, const float* __restrict__ key_cache,
        const float* __restrict__ value_cache, float* __restrict__ out_context,
        hip_bfloat16* __restrict__ out_context_bfloat16,
        std::uint32_t layer_index, std::uint32_t position,
        std::uint32_t max_context, std::uint32_t num_heads,
        std::uint32_t num_key_value_heads, std::uint32_t head_dim) {
  const std::uint32_t head = blockIdx.x;
  if (head >= num_heads) {
    return;
  }
  const std::uint32_t thread = threadIdx.x;
  const std::uint32_t lane = thread & 31U;
  const std::uint32_t wave = thread >> 5U;
  constexpr std::uint32_t kWaves = kDecodeAttentionThreads / 32U;
  const std::uint32_t key_value_head = head / (num_heads / num_key_value_heads);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t sequence = static_cast<std::size_t>(position) + 1U;
  const std::size_t cache_head_stride =
      static_cast<std::size_t>(max_context) * head_dim;
  const float* cache_head =
      key_cache +
      (((static_cast<std::size_t>(layer_index) * num_key_value_heads) +
        key_value_head) *
       cache_head_stride);
  const float* value_head =
      value_cache +
      (((static_cast<std::size_t>(layer_index) * num_key_value_heads) +
        key_value_head) *
       cache_head_stride);

  __shared__ float reduction[kDecodeAttentionThreads];
  extern __shared__ float scores[];

  const auto* query_vec = reinterpret_cast<const float4*>(
      query + (static_cast<std::size_t>(head) * head_dim));
  const std::uint32_t vectors = head_dim / 4U;

  // Scores. Each wave takes `Unroll` positions per step so that many key rows
  // are in flight before the first reduction consumes one.
  const std::uint32_t step = kWaves * kDecodeAttentionUnroll;
  std::size_t base = wave;
  for (; base + (step - kWaves) < sequence; base += step) {
    float dot[kDecodeAttentionUnroll];
#pragma unroll
    for (std::uint32_t slot = 0U; slot < kDecodeAttentionUnroll; ++slot) {
      const std::size_t key = base + (slot * kWaves);
      dot[slot] = 0.0F;
      for (std::uint32_t vector = lane; vector < vectors; vector += 32U) {
        const float4 q = query_vec[vector];
        const float4 k = reinterpret_cast<const float4*>(
            cache_head + (key * head_dim))[vector];
        dot[slot] += (q.x * k.x) + (q.y * k.y) + (q.z * k.z) + (q.w * k.w);
      }
    }
#pragma unroll
    for (std::uint32_t slot = 0U; slot < kDecodeAttentionUnroll; ++slot) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot[slot] += __shfl_xor(dot[slot], offset);
      }
      if (lane == 0U) {
        scores[base + (slot * kWaves)] = static_cast<float>(
            hip_bfloat16(static_cast<float>(hip_bfloat16(dot[slot])) * scale));
      }
    }
  }
  for (std::size_t key = base; key < sequence; key += kWaves) {
    float dot = 0.0F;
    for (std::uint32_t vector = lane; vector < vectors; vector += 32U) {
      const float4 q = query_vec[vector];
      const float4 k = reinterpret_cast<const float4*>(
          cache_head + (key * head_dim))[vector];
      dot += (q.x * k.x) + (q.y * k.y) + (q.z * k.z) + (q.w * k.w);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_xor(dot, offset);
    }
    if (lane == 0U) {
      scores[key] = static_cast<float>(
          hip_bfloat16(static_cast<float>(hip_bfloat16(dot)) * scale));
    }
  }
  __syncthreads();

  // Eager attention retains BF16 QK/scaling and probability boundaries.
  float local_maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    local_maximum = fmaxf(local_maximum, scores[key]);
  }
  reduction[thread] = local_maximum;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset /= 2U) {
    if (thread < offset) {
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + offset]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    const float value = expf(scores[key] - maximum);
    scores[key] = value;
    local_sum += value;
  }
  reduction[thread] = local_sum;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset > 0U; offset /= 2U) {
    if (thread < offset) {
      reduction[thread] += reduction[thread + offset];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  __syncthreads();
  for (std::size_t key = thread; key < sequence; key += blockDim.x) {
    scores[key] = static_cast<float>(hip_bfloat16(scores[key] * inverse_sum));
  }
  __syncthreads();

  // Values. One thread per head dimension, prefetching `Unroll` rows while the
  // single accumulator stays in ascending position order.
  if (thread < head_dim) {
    float context = 0.0F;
    std::size_t key = 0U;
    for (; key + kDecodeAttentionUnroll <= sequence;
         key += kDecodeAttentionUnroll) {
      float values[kDecodeAttentionUnroll];
#pragma unroll
      for (std::uint32_t slot = 0U; slot < kDecodeAttentionUnroll; ++slot) {
        values[slot] = value_head[((key + slot) * head_dim) + thread];
      }
#pragma unroll
      for (std::uint32_t slot = 0U; slot < kDecodeAttentionUnroll; ++slot) {
        context += scores[key + slot] * values[slot];
      }
    }
    for (; key < sequence; ++key) {
      context += scores[key] * value_head[(key * head_dim) + thread];
    }
    const std::size_t index =
        (static_cast<std::size_t>(head) * head_dim) + thread;
    out_context[index] = context;
    if (out_context_bfloat16 != nullptr) {
      out_context_bfloat16[index] = hip_bfloat16(context);
    }
  }
}

__launch_bounds__(256) __global__
    void TextRMSNormKernel(const float* input, const float* weight,
                           hip_bfloat16* output, std::size_t batch_size,
                           std::size_t dimension, float epsilon) {
  const std::size_t batch = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  const std::size_t thread = threadIdx.x;
  const float* row = input + (batch * dimension);
  hip_bfloat16* output_row = output + (batch * dimension);
  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(row[index]));
    sum_squared += value * value;
  }
  __shared__ float shared_sum[256];
  shared_sum[thread] = sum_squared;
  __syncthreads();
  FinishNormSum(shared_sum);
  const float inverse_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(dimension)) + epsilon);
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(row[index]));
    const float normalized =
        static_cast<float>(hip_bfloat16(value * inverse_rms));
    output_row[index] = hip_bfloat16(normalized * weight[index]);
  }
}

__launch_bounds__(256) __global__ void TextResidualAddRMSNormKernel(
    float* __restrict__ hidden, const float* __restrict__ update,
    const float* __restrict__ weight, hip_bfloat16* __restrict__ output,
    std::size_t batch_size, std::size_t dimension, float epsilon) {
  const std::size_t batch = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  const std::size_t thread = threadIdx.x;
  float* row = hidden + (batch * dimension);
  const float* update_row = update + (batch * dimension);
  hip_bfloat16* output_row = output + (batch * dimension);
  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float left = static_cast<float>(hip_bfloat16(row[index]));
    const float right = static_cast<float>(hip_bfloat16(update_row[index]));
    const float sum = static_cast<float>(hip_bfloat16(left + right));
    row[index] = sum;
    sum_squared += sum * sum;
  }
  __shared__ float shared_sum[256];
  shared_sum[thread] = sum_squared;
  __syncthreads();
  FinishNormSum(shared_sum);
  const float inverse_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(dimension)) + epsilon);
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float normalized =
        static_cast<float>(hip_bfloat16(row[index] * inverse_rms));
    output_row[index] = hip_bfloat16(normalized * weight[index]);
  }
}

__global__ void TextSwiGLUKernel(const float* gate, const float* up,
                                 hip_bfloat16* output, std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float gate_bfloat16 = static_cast<float>(hip_bfloat16(gate[index]));
  const float up_bfloat16 = static_cast<float>(hip_bfloat16(up[index]));
  const float activated = static_cast<float>(
      hip_bfloat16(gate_bfloat16 / (1.0F + expf(-gate_bfloat16))));
  output[index] = hip_bfloat16(activated * up_bfloat16);
}

__global__ void RoundBfloat16Kernel(float* values, std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    values[index] = static_cast<float>(hip_bfloat16(values[index]));
  }
}

__global__ void Bfloat16ToFloatKernel(const hip_bfloat16* input, float* output,
                                      std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(input[index]);
  }
}

constexpr std::uint32_t kArgmaxBlocks = 1024;
constexpr std::uint32_t kArgmaxThreads = 256;
struct ArgmaxValue {
  float value;
  std::uint32_t token;
};
__device__ ArgmaxValue BestArgmax(ArgmaxValue left, ArgmaxValue right) {
  if (isnan(left.value) || isnan(right.value))
    return {__int_as_float(0x7fc00000), UINT32_MAX};
  return right.value > left.value ||
                 (right.value == left.value && right.token < left.token)
             ? right
             : left;
}

template<bool Finish>
__launch_bounds__(kArgmaxThreads) __global__
    void TextArgmaxKernel(const float* logits, ArgmaxValue* partials,
                          std::uint32_t* output, std::size_t count) {
  ArgmaxValue best{-std::numeric_limits<float>::infinity(), UINT32_MAX};
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    ArgmaxValue candidate;
    if constexpr (Finish)
      candidate = partials[i];
    else
      candidate = isfinite(logits[i])
                      ? ArgmaxValue{logits[i], static_cast<std::uint32_t>(i)}
                      : ArgmaxValue{__int_as_float(0x7fc00000), UINT32_MAX};
    best = BestArgmax(best, candidate);
  }
  __shared__ ArgmaxValue shared[kArgmaxThreads];
  shared[threadIdx.x] = best;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] =
          BestArgmax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if constexpr (Finish)
      *output = shared[0].token;
    else if (gridDim.x == 1)
      *output = shared[0].token;
    else
      partials[blockIdx.x] = shared[0];
  }
}

dim3 Grid(std::size_t count) {
  constexpr unsigned int kBlock = 256;
  return dim3(static_cast<unsigned int>((count + kBlock - 1U) / kBlock));
}

}  // namespace

bool LaunchTextQkNormRoPE(float* query, float* key, float* value,
                          const float* query_weight, const float* key_weight,
                          std::size_t batch_size, std::uint32_t num_heads,
                          std::uint32_t num_key_value_heads,
                          std::uint32_t head_dim, std::uint32_t start_position,
                          float rope_theta, float epsilon, float* key_cache,
                          float* value_cache, std::uint32_t layer_index,
                          std::uint32_t max_context, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  if (batch_size == 0U || num_heads == 0U || num_key_value_heads == 0U ||
      head_dim == 0U || head_dim > kThreads || head_dim % 2U != 0U ||
      (key_cache != nullptr &&
       (value_cache == nullptr || start_position + batch_size > max_context))) {
    return false;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  TextQkNormRoPEKernel<kThreads><<<grid, kThreads, 0, stream>>>(
      query, key, value, query_weight, key_weight, batch_size, num_heads,
      num_key_value_heads, head_dim, start_position, rope_theta, epsilon,
      key_cache, value_cache, layer_index, max_context);
  return true;
}

void LaunchTextRMSNorm(const float* input, const float* weight,
                       void* output_bfloat16, std::size_t batch_size,
                       std::size_t dimension, float epsilon,
                       hipStream_t stream) {
  if (batch_size == 0U || dimension == 0U) {
    return;
  }
  TextRMSNormKernel<<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
      input, weight, static_cast<hip_bfloat16*>(output_bfloat16), batch_size,
      dimension, epsilon);
}

void LaunchTextResidualAddRMSNorm(float* hidden, const float* update,
                                  const float* weight, void* output_bfloat16,
                                  std::size_t batch_size, std::size_t dimension,
                                  float epsilon, hipStream_t stream) {
  if (batch_size == 0U || dimension == 0U) {
    return;
  }
  TextResidualAddRMSNormKernel<<<static_cast<unsigned int>(batch_size), 256, 0,
                                 stream>>>(
      hidden, update, weight, static_cast<hip_bfloat16*>(output_bfloat16),
      batch_size, dimension, epsilon);
}

void LaunchTextSwiGLU(const float* gate, const float* up, void* output_bfloat16,
                      std::size_t count, hipStream_t stream) {
  if (count == 0U) {
    return;
  }
  TextSwiGLUKernel<<<Grid(count), 256, 0, stream>>>(
      gate, up, static_cast<hip_bfloat16*>(output_bfloat16), count);
}

void LaunchTextRoundBfloat16(float* values, std::size_t count,
                             hipStream_t stream) {
  if (count != 0U) {
    RoundBfloat16Kernel<<<Grid(count), 256, 0, stream>>>(values, count);
  }
}

void LaunchTextBfloat16ToFloat(const void* input_bfloat16, float* output,
                               std::size_t count, hipStream_t stream) {
  if (count != 0U) {
    Bfloat16ToFloatKernel<<<Grid(count), 256, 0, stream>>>(
        static_cast<const hip_bfloat16*>(input_bfloat16), output, count);
  }
}

void LaunchTextEmbeddingLookup(const void* table_bfloat16,
                               const std::uint32_t* token_ids, float* output,
                               std::size_t batch_size, std::size_t hidden_size,
                               hipStream_t stream) {
  if (batch_size == 0U || hidden_size == 0U) {
    return;
  }
  constexpr unsigned int kBlock = 256;
  const dim3 grid(
      static_cast<unsigned int>((hidden_size + kBlock - 1U) / kBlock),
      static_cast<unsigned int>(batch_size));
  TextEmbeddingLookupKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const hip_bfloat16*>(table_bfloat16), token_ids, output,
      batch_size, hidden_size);
}

void LaunchTextRowGemv(const void* weights_bfloat16, const float* input,
                       float* output, std::size_t rows, std::size_t columns,
                       hipStream_t stream) {
  if (rows == 0U || columns == 0U) {
    return;
  }
  constexpr std::uint32_t kWaves = 4;
  const auto blocks = static_cast<unsigned int>((rows + kWaves - 1U) / kWaves);
  TextRowGemvKernel<kWaves><<<blocks, kWaves * 32, 0, stream>>>(
      static_cast<const hip_bfloat16*>(weights_bfloat16), input, output, rows,
      columns);
}

void LaunchTextBatchedAttention(
    const float* query, const float* key_cache, const float* value_cache,
    float* out_context, void* out_context_bfloat16, std::uint32_t layer_index,
    std::uint32_t start_position, std::size_t tokens, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, hipStream_t stream) {
  if (tokens == 0U || num_heads == 0U || num_key_value_heads == 0U) {
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(tokens));
  const std::size_t sequence =
      static_cast<std::size_t>(start_position) + tokens;
  TextBatchedAttentionKernel<<<grid, 256, sequence * sizeof(float), stream>>>(
      query, key_cache, value_cache, out_context,
      static_cast<hip_bfloat16*>(out_context_bfloat16), layer_index,
      start_position, tokens, max_context, num_heads, num_key_value_heads,
      head_dim);
}

bool LaunchTextDecodeAttention(
    const float* query, const float* key_cache, const float* value_cache,
    float* out_context, void* out_context_bfloat16, std::uint32_t layer_index,
    std::uint32_t position, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    hipStream_t stream) {
  const std::size_t sequence = static_cast<std::size_t>(position) + 1U;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      out_context == nullptr || num_heads == 0U || num_key_value_heads == 0U ||
      head_dim == 0U || head_dim % 4U != 0U ||
      head_dim > kDecodeAttentionThreads ||
      num_heads % num_key_value_heads != 0U || sequence > max_context ||
      sequence < kDecodeAttentionMinSequence) {
    return false;
  }
  TextDecodeAttentionKernel<<<num_heads, kDecodeAttentionThreads,
                              sequence * sizeof(float), stream>>>(
      query, key_cache, value_cache, out_context,
      static_cast<hip_bfloat16*>(out_context_bfloat16), layer_index, position,
      max_context, num_heads, num_key_value_heads, head_dim);
  return true;
}

bool LaunchTextDecodeGemv(const TextDecodeGemvTensor* tensors,
                          std::uint32_t count, const void* input_bfloat16,
                          std::size_t columns, hipStream_t stream) {
  // The kernel loads sixteen bytes at a time and stages the activation vector
  // in LDS, so it needs an eight-element column multiple, aligned bases, and a
  // vector that fits the 64 KB shared-memory limit for one workgroup.
  constexpr std::size_t kAlignment = 16;
  constexpr std::size_t kMaxSharedBytes = 64U * 1024U;
  if (tensors == nullptr || count == 0U || count > kTextDecodeGemvMaxTensors ||
      input_bfloat16 == nullptr || columns == 0U || columns % 8U != 0U) {
    return false;
  }
  const std::size_t k_vectors = columns / 8U;
  if (k_vectors * kAlignment > kMaxSharedBytes) {
    return false;
  }
  if (reinterpret_cast<std::uintptr_t>(input_bfloat16) % kAlignment != 0U) {
    return false;
  }

  DecodeGemvTensors packed{};
  packed.count = count;
  std::uint64_t total_rows = 0U;
  for (std::uint32_t index = 0U; index < count; ++index) {
    const TextDecodeGemvTensor& tensor = tensors[index];
    if (tensor.weights_bfloat16 == nullptr || tensor.output == nullptr ||
        tensor.rows == 0U ||
        reinterpret_cast<std::uintptr_t>(tensor.weights_bfloat16) %
                kAlignment !=
            0U) {
      return false;
    }
    packed.weights[index] =
        static_cast<const hip_bfloat16*>(tensor.weights_bfloat16);
    packed.outputs[index] = tensor.output;
    packed.first_row[index] = static_cast<std::uint32_t>(total_rows);
    total_rows += tensor.rows;
  }
  if (total_rows > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  packed.total_rows = static_cast<std::uint32_t>(total_rows);

  const unsigned int blocks = static_cast<unsigned int>(
      (total_rows + kDecodeGemvWavesPerBlock - 1U) / kDecodeGemvWavesPerBlock);
  TextDecodeGemvKernel<<<blocks, kDecodeGemvBlockThreads,
                         k_vectors * kAlignment, stream>>>(
      packed, static_cast<const hip_bfloat16*>(input_bfloat16),
      static_cast<std::uint32_t>(k_vectors));
  return true;
}

void LaunchTextArgmax(const float* logits, std::uint32_t* output_token,
                      std::size_t count, float* scratch, hipStream_t stream) {
  const auto blocks = scratch == nullptr
                          ? 1U
                          : static_cast<std::uint32_t>(std::min<std::size_t>(
                                kArgmaxBlocks, std::max<std::size_t>(
                                                   1, (count + 4095) / 4096)));
  auto* partials = reinterpret_cast<ArgmaxValue*>(scratch);
  TextArgmaxKernel<false><<<blocks, kArgmaxThreads, 0, stream>>>(
      logits, partials, output_token, count);
  if (blocks > 1)
    TextArgmaxKernel<true><<<1, kArgmaxThreads, 0, stream>>>(
        nullptr, partials, output_token, blocks);
}

std::size_t TextArgmaxScratchElements() {
  return 2 * kArgmaxBlocks;
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
