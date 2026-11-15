#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/families/qwen/eager/ops.hpp"

namespace gufo::qwen::eager {
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

__device__ __forceinline__ float2 RotateBfloat16(float first, float second,
                                                 float angle) {
  const float cosine = static_cast<float>(hip_bfloat16(cosf(angle)));
  const float sine = static_cast<float>(hip_bfloat16(sinf(angle)));
  const float first_cos = static_cast<float>(hip_bfloat16(first * cosine));
  const float second_sin = static_cast<float>(hip_bfloat16(second * sine));
  const float second_cos = static_cast<float>(hip_bfloat16(second * cosine));
  const float first_sin = static_cast<float>(hip_bfloat16(first * sine));
  return make_float2(static_cast<float>(hip_bfloat16(first_cos - second_sin)),
                     static_cast<float>(hip_bfloat16(second_cos + first_sin)));
}

__global__ void Bfloat16CacheWriteKernel(
    const float* key, const float* value, float* key_cache, float* value_cache,
    std::size_t layer, std::size_t position, std::size_t capacity,
    std::size_t kv_heads, std::size_t head_dim) {
  const std::size_t row = blockIdx.x;
  const std::size_t head = blockIdx.y;
  for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const auto source = (row * kv_heads + head) * head_dim + d;
    const auto destination =
        ((layer * kv_heads + head) * capacity + position + row) * head_dim + d;
    key_cache[destination] = key[source];
    value_cache[destination] = value[source];
  }
}

template<unsigned int Threads>
__launch_bounds__(Threads) __global__
    void Bfloat16AttentionKernel(const float* query, const float* key_cache,
                                 const float* value_cache, hip_bfloat16* output,
                                 std::size_t layer, std::size_t position,
                                 std::size_t capacity, std::size_t heads,
                                 std::size_t kv_heads, std::size_t head_dim) {
  const std::size_t head = blockIdx.x;
  const std::size_t row = blockIdx.y;
  const std::size_t tid = threadIdx.x;
  const std::size_t visible = position + row + 1;
  const std::size_t kv_head = head / (heads / kv_heads);
  const std::size_t cache_base =
      (layer * kv_heads + kv_head) * capacity * head_dim;
  const auto* q =
      reinterpret_cast<const float4*>(query + (row * heads + head) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  __shared__ float reduction[Threads];
  extern __shared__ float scores[];
  const auto dot_at = [&](std::size_t token, std::size_t begin,
                          std::size_t step) {
    const auto* key = reinterpret_cast<const float4*>(key_cache + cache_base +
                                                      token * head_dim);
    float dot = 0;
    for (std::size_t d = begin; d < head_dim / 4; d += step) {
      const float4 a = q[d], b = key[d];
      dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return dot;
  };
  if (visible >= 64) {
    const std::size_t lane = tid & 31;
    for (std::size_t p = tid / 32; p < visible; p += blockDim.x / 32) {
      float dot = dot_at(p, lane, 32);
      for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_xor(dot, offset);
      if (lane == 0)
        scores[p] = static_cast<float>(
            hip_bfloat16(static_cast<float>(hip_bfloat16(dot)) * scale));
    }
  } else {
    for (std::size_t p = tid; p < visible; p += blockDim.x)
      scores[p] = static_cast<float>(hip_bfloat16(
          static_cast<float>(hip_bfloat16(dot_at(p, 0, 1))) * scale));
  }
  __syncthreads();
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t p = tid; p < visible; p += blockDim.x)
    maximum = fmaxf(maximum, scores[p]);
  reduction[tid] = maximum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (tid < stride)
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float total = 0;
  for (std::size_t p = tid; p < visible; p += blockDim.x) {
    scores[p] = expf(scores[p] - maximum);
    total += scores[p];
  }
  reduction[tid] = total;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (tid < stride)
      reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  const float inverse = 1.0F / reduction[0];
  __syncthreads();
  // PyTorch eager attention casts the FP32 softmax back to BF16 before P @ V.
  for (std::size_t p = tid; p < visible; p += blockDim.x)
    scores[p] = static_cast<float>(hip_bfloat16(scores[p] * inverse));
  __syncthreads();
  for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
    float result = 0;
    for (std::size_t p = 0; p < visible; ++p)
      result += scores[p] * value_cache[cache_base + p * head_dim + d];
    output[(row * heads + head) * head_dim + d] = hip_bfloat16(result);
  }
}

template<std::uint32_t ThreadsPerBlock>
__launch_bounds__(ThreadsPerBlock, 4) __global__
    void Bfloat16PerHeadRMSNormKernel(float* values, const float* weight,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim, float epsilon) {
  const std::size_t batch = blockIdx.y;
  const std::uint32_t head = blockIdx.x;
  if (batch >= batch_size || head >= num_heads) {
    return;
  }

  const std::size_t thread = threadIdx.x;
  const std::size_t head_stride =
      static_cast<std::size_t>(num_heads) * head_dim;
  float* head_values = values + (batch * head_stride) +
                       (static_cast<std::size_t>(head) * head_dim);

  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    sum_squared += value * value;
  }

  __shared__ float shared_sum[ThreadsPerBlock];
  shared_sum[thread] = sum_squared;
  __syncthreads();

  FinishNormSum(shared_sum);

  const float mean_squared = shared_sum[0] / static_cast<float>(head_dim);
  const float inverse_rms = rsqrtf(mean_squared + epsilon);
  for (std::size_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    const float normalized =
        static_cast<float>(hip_bfloat16(value * inverse_rms));
    head_values[index] = static_cast<float>(hip_bfloat16(normalized * scale));
  }
}

__launch_bounds__(256, 4) __global__ void Bfloat16RoPEAndRoundVKernel(
    float* query, float* key, float* value, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta) {
  const std::size_t batch = blockIdx.y;
  if (batch >= batch_size) {
    return;
  }

  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t half_rotary = head_dim / 2;
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(batch);
  if (index < static_cast<std::size_t>(num_heads) * half_rotary) {
    const std::size_t head = index / half_rotary;
    const std::size_t dimension = index % half_rotary;
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const std::size_t stride = static_cast<std::size_t>(num_heads) * head_dim;
    float* head_values = query + (batch * stride) + (head * head_dim);
    const float first = head_values[dimension];
    const float second = head_values[dimension + half_rotary];
    const float2 rotated = RotateBfloat16(first, second, angle);
    head_values[dimension] = rotated.x;
    head_values[dimension + half_rotary] = rotated.y;
  }

  if (index < static_cast<std::size_t>(num_key_value_heads) * half_rotary) {
    const std::size_t head = index / half_rotary;
    const std::size_t dimension = index % half_rotary;
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const std::size_t stride =
        static_cast<std::size_t>(num_key_value_heads) * head_dim;
    float* head_values = key + (batch * stride) + (head * head_dim);
    const float first = head_values[dimension];
    const float second = head_values[dimension + half_rotary];
    const float2 rotated = RotateBfloat16(first, second, angle);
    head_values[dimension] = rotated.x;
    head_values[dimension + half_rotary] = rotated.y;
  }

  const std::size_t value_elements =
      static_cast<std::size_t>(num_key_value_heads) * head_dim;
  if (index < value_elements) {
    float* batch_values = value + (batch * value_elements);
    batch_values[index] = static_cast<float>(hip_bfloat16(batch_values[index]));
  }
}

/// Normalizes one attention head and rotates it without a round trip through
/// memory between the two steps. Reproduces the standalone per-head RMSNorm and
/// RoPE kernels exactly, including every intermediate BF16 rounding.
template<bool kRoundScale = false>
__device__ void NormalizeAndRotateHead(float* head_values, const float* weight,
                                       std::uint32_t head_dim, float epsilon,
                                       std::uint32_t position, float rope_theta,
                                       float* shared_sum, float* normalized) {
  const std::uint32_t thread = threadIdx.x;
  float sum_squared = 0.0F;
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    sum_squared += value * value;
  }
  shared_sum[thread] = sum_squared;
  __syncthreads();
  FinishNormSum(shared_sum);
  const float mean_squared = shared_sum[0] / static_cast<float>(head_dim);
  float inverse_rms = rsqrtf(mean_squared + epsilon);
  if constexpr (kRoundScale)
    inverse_rms = static_cast<float>(hip_bfloat16(inverse_rms));
  for (std::uint32_t index = thread; index < head_dim; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(head_values[index]));
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    const float unit = static_cast<float>(hip_bfloat16(value * inverse_rms));
    normalized[index] = static_cast<float>(hip_bfloat16(unit * scale));
  }
  __syncthreads();

  const std::uint32_t half_rotary = head_dim / 2;
  for (std::uint32_t dimension = thread; dimension < half_rotary;
       dimension += blockDim.x) {
    const float frequency =
        1.0F / powf(rope_theta, static_cast<float>(2 * dimension) /
                                    static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * frequency;
    const float first = normalized[dimension];
    const float second = normalized[dimension + half_rotary];
    const float2 rotated = RotateBfloat16(first, second, angle);
    head_values[dimension] = rotated.x;
    head_values[dimension + half_rotary] = rotated.y;
  }
  __syncthreads();
}

template<std::uint32_t ThreadsPerBlock>
__launch_bounds__(ThreadsPerBlock) __global__ void Bfloat16QkNormRoPEKernel(
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
  __shared__ float shared_sum[ThreadsPerBlock];
  __shared__ float normalized[ThreadsPerBlock];
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(batch);

  // Both branches are uniform across the workgroup, so the barriers inside the
  // helper stay collective.
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

/// A projection output at its BF16 boundary: FP32 results are rounded, BF16
/// GEMM outputs already sit there.
__device__ __forceinline__ float RoundedInput(float value) {
  return static_cast<float>(hip_bfloat16(value));
}
__device__ __forceinline__ float RoundedInput(hip_bfloat16 value) {
  return static_cast<float>(value);
}

/// Fuses residual addition and RMSNorm while retaining the upstream BF16
/// boundaries at the addition, normalized value and weighted output.
template<bool kRoundScale, typename Input = float>
__launch_bounds__(256) __global__ void Bfloat16ResidualAddRMSNormKernel(
    float* __restrict__ hidden, const Input* __restrict__ update,
    const float* __restrict__ weight, hip_bfloat16* __restrict__ output,
    std::size_t batch_size, std::size_t dimension, float epsilon) {
  const std::size_t batch = blockIdx.x;
  if (batch >= batch_size) {
    return;
  }
  const std::size_t thread = threadIdx.x;
  float* row = hidden + (batch * dimension);
  const Input* update_row =
      update == nullptr ? nullptr : update + (batch * dimension);
  hip_bfloat16* output_row = output + (batch * dimension);

  float sum_squared = 0.0F;
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float left = static_cast<float>(hip_bfloat16(row[index]));
    const float sum = update_row == nullptr
                          ? left
                          : static_cast<float>(hip_bfloat16(
                                left + RoundedInput(update_row[index])));
    row[index] = sum;
    sum_squared += sum * sum;
  }

  __shared__ float shared_sum[256];
  shared_sum[thread] = sum_squared;
  __syncthreads();
  FinishNormSum(shared_sum);
  const float mean_squared = shared_sum[0] / static_cast<float>(dimension);
  float inverse_rms = rsqrtf(mean_squared + epsilon);
  if constexpr (kRoundScale)
    inverse_rms = static_cast<float>(hip_bfloat16(inverse_rms));
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float scale = weight != nullptr ? weight[index] : 1.0F;
    const float unit =
        static_cast<float>(hip_bfloat16(row[index] * inverse_rms));
    output_row[index] = hip_bfloat16(unit * scale);
  }
}

__global__ void RoundBfloat16InPlaceKernel(float* values, std::size_t count) {
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

__global__ void Bfloat16ResidualAddKernel(const float* residual,
                                          const float* update, float* output,
                                          std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float left = static_cast<float>(hip_bfloat16(residual[index]));
    const float right = static_cast<float>(hip_bfloat16(update[index]));
    output[index] = static_cast<float>(hip_bfloat16(left + right));
  }
}

template<typename Input>
__global__ void Bfloat16SwiGluKernel(const Input* gate, const Input* up,
                                     float* output,
                                     hip_bfloat16* output_bfloat16,
                                     std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float gate_bfloat16 = RoundedInput(gate[index]);
    const float up_bfloat16 = RoundedInput(up[index]);
    const float activated = static_cast<float>(
        hip_bfloat16(gate_bfloat16 / (1.0F + expf(-gate_bfloat16))));
    const hip_bfloat16 result(activated * up_bfloat16);
    if (output != nullptr) {
      output[index] = static_cast<float>(result);
    }
    if (output_bfloat16 != nullptr) {
      output_bfloat16[index] = result;
    }
  }
}

__global__ void Bfloat16BiasSiluKernel(const float* input,
                                       const hip_bfloat16* bias, float* output,
                                       hip_bfloat16* output_bfloat16,
                                       std::size_t count, std::size_t columns) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const float biased = static_cast<float>(
        hip_bfloat16(input[index] + static_cast<float>(bias[index % columns])));
    const hip_bfloat16 result(biased / (1.0F + expf(-biased)));
    if (output != nullptr) {
      output[index] = static_cast<float>(result);
    }
    if (output_bfloat16 != nullptr) {
      output_bfloat16[index] = result;
    }
  }
}

__global__ void Bfloat16BiasKernel(const float* input, const hip_bfloat16* bias,
                                   float* output, std::size_t count,
                                   std::size_t columns) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = static_cast<float>(
        hip_bfloat16(input[index] + static_cast<float>(bias[index % columns])));
  }
}

__global__ void BatchedCodecEmbeddingSumKernel(
    const hip_bfloat16* const* embedding_tables, const std::uint32_t* codes,
    const float* text_embedding, float* output, std::size_t elements,
    std::size_t groups, std::size_t hidden_size) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t frame = index / hidden_size;
  const std::size_t dimension = index % hidden_size;
  float sum = 0.0F;
  for (std::size_t group = 0; group < groups; ++group) {
    const std::uint32_t code = codes[(frame * groups) + group];
    sum += static_cast<float>(
        embedding_tables[group][(static_cast<std::size_t>(code) * hidden_size) +
                                dimension]);
  }
  if (text_embedding != nullptr) {
    // Upstream sums BF16 codec embeddings before a separate BF16 text add.
    sum = static_cast<float>(hip_bfloat16(sum));
    sum += text_embedding[dimension];
  }
  output[index] = static_cast<float>(hip_bfloat16(sum));
}

__device__ __forceinline__ float LowBfloat16(std::uint32_t word) {
  return __uint_as_float(word << 16U);
}

__device__ __forceinline__ float HighBfloat16(std::uint32_t word) {
  return __uint_as_float(word & 0xFFFF0000U);
}

/// Dot product of the BF16 lanes packed into one vectorized load.
__device__ __forceinline__ float PackedDot(uint4 left, uint4 right) {
  return (LowBfloat16(left.x) * LowBfloat16(right.x)) +
         (HighBfloat16(left.x) * HighBfloat16(right.x)) +
         (LowBfloat16(left.y) * LowBfloat16(right.y)) +
         (HighBfloat16(left.y) * HighBfloat16(right.y)) +
         (LowBfloat16(left.z) * LowBfloat16(right.z)) +
         (HighBfloat16(left.z) * HighBfloat16(right.z)) +
         (LowBfloat16(left.w) * LowBfloat16(right.w)) +
         (HighBfloat16(left.w) * HighBfloat16(right.w));
}

__device__ __forceinline__ float PackedDot(uint2 left, uint2 right) {
  return (LowBfloat16(left.x) * LowBfloat16(right.x)) +
         (HighBfloat16(left.x) * HighBfloat16(right.x)) +
         (LowBfloat16(left.y) * LowBfloat16(right.y)) +
         (HighBfloat16(left.y) * HighBfloat16(right.y));
}

constexpr unsigned int kGemvWavefront = 32;
constexpr unsigned int kGemvWaves = 8;
constexpr std::size_t kGemvMaxBatch = 4;

/// Reduces one output row per workgroup. Eight wavefronts each walk the row
/// with their own coalesced stream, which is what lets a single-token
/// projection saturate DRAM. The reduction order is fixed by wavefront index,
/// so the result is reproducible.
///
/// `kVectors` input columns share one pass over the weight row, so the few
/// two-token predictor steps cost the same memory traffic as a single token.
template<typename Packed, unsigned int kVectors>
__launch_bounds__(kGemvWaves* kGemvWavefront) __global__
    void Bfloat16GemvKernel(const Packed* __restrict__ weights,
                            const Packed* __restrict__ inputs,
                            float* __restrict__ output,
                            std::uint32_t packed_columns, std::uint32_t rows) {
  __shared__ float wave_sums[kVectors][kGemvWaves];
  const unsigned int lane = threadIdx.x & (kGemvWavefront - 1U);
  const unsigned int wave = threadIdx.x >> 5U;
  const Packed* row =
      weights + (static_cast<std::size_t>(blockIdx.x) * packed_columns);

  float sums[kVectors];
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    sums[vector] = 0.0F;
  }
  for (std::uint32_t index = (wave * kGemvWavefront) + lane;
       index < packed_columns; index += kGemvWaves * kGemvWavefront) {
    const Packed weight = row[index];
#pragma unroll
    for (unsigned int vector = 0; vector < kVectors; ++vector) {
      sums[vector] +=
          PackedDot(weight, inputs[(vector * packed_columns) + index]);
    }
  }
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    float sum = sums[vector];
#pragma unroll
    for (int offset = kGemvWavefront / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down(sum, offset, kGemvWavefront);
    }
    if (lane == 0) {
      wave_sums[vector][wave] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x < kVectors) {
    float total = 0.0F;
#pragma unroll
    for (unsigned int index = 0; index < kGemvWaves; ++index) {
      total += wave_sums[threadIdx.x][index];
    }
    output[(threadIdx.x * rows) + blockIdx.x] = total;
  }
}

/// Weight matrices and destinations of a grouped GEMV, passed by value so the
/// kernel resolves its group from registers instead of a second memory load.
template<typename Packed, unsigned int kGroups>
struct GemvGroupedArgs {
  const Packed* weights[kGroups];
  float* outputs[kGroups];
  // Exclusive prefix sums of the per-group row counts, so `row_end[kGroups -
  // 1]` is the grid width.
  std::uint32_t row_end[kGroups];
  std::uint32_t rows[kGroups];
};

/// Grouped form of `Bfloat16GemvKernel`. The per-row reduction is character for
/// character the same, which is what keeps the fused launch bit-identical to
/// the separate ones; only the row-to-matrix mapping is new.
template<typename Packed, unsigned int kGroups, unsigned int kVectors>
__launch_bounds__(kGemvWaves* kGemvWavefront) __global__
    void Bfloat16GemvGroupedKernel(GemvGroupedArgs<Packed, kGroups> args,
                                   const Packed* __restrict__ inputs,
                                   std::uint32_t packed_columns) {
  unsigned int group = 0;
#pragma unroll
  for (unsigned int boundary = 1; boundary < kGroups; ++boundary) {
    group += blockIdx.x >= args.row_end[boundary - 1] ? 1U : 0U;
  }
  const std::uint32_t row =
      blockIdx.x - (group == 0U ? 0U : args.row_end[group - 1]);

  __shared__ float wave_sums[kVectors][kGemvWaves];
  const unsigned int lane = threadIdx.x & (kGemvWavefront - 1U);
  const unsigned int wave = threadIdx.x >> 5U;
  const Packed* weight_row =
      args.weights[group] + (static_cast<std::size_t>(row) * packed_columns);

  float sums[kVectors];
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    sums[vector] = 0.0F;
  }
  for (std::uint32_t index = (wave * kGemvWavefront) + lane;
       index < packed_columns; index += kGemvWaves * kGemvWavefront) {
    const Packed weight = weight_row[index];
#pragma unroll
    for (unsigned int vector = 0; vector < kVectors; ++vector) {
      sums[vector] +=
          PackedDot(weight, inputs[(vector * packed_columns) + index]);
    }
  }
#pragma unroll
  for (unsigned int vector = 0; vector < kVectors; ++vector) {
    float sum = sums[vector];
#pragma unroll
    for (int offset = kGemvWavefront / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down(sum, offset, kGemvWavefront);
    }
    if (lane == 0) {
      wave_sums[vector][wave] = sum;
    }
  }
  __syncthreads();
  if (threadIdx.x < kVectors) {
    float total = 0.0F;
#pragma unroll
    for (unsigned int index = 0; index < kGemvWaves; ++index) {
      total += wave_sums[threadIdx.x][index];
    }
    args.outputs[group][(threadIdx.x * args.rows[group]) + row] = total;
  }
}

template<bool kRoundScale, typename Input = float>
__launch_bounds__(128) __global__ void Bfloat16QkNormRoPEHeadMajorKernel(
    const Input* query, const Input* key, const Input* value,
    const float* query_weight, const float* key_weight, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    hip_bfloat16* query_out, std::size_t query_stride, std::size_t query_row,
    hip_bfloat16* key_out, hip_bfloat16* value_out, std::size_t cache_stride,
    std::size_t cache_row) {
  const std::size_t row = blockIdx.y;
  const std::uint32_t head = blockIdx.x;
  __shared__ float shared_sum[128];
  __shared__ float normalized[128];
  __shared__ float rotated[128];
  const std::uint32_t position =
      start_position + static_cast<std::uint32_t>(row);
  // Both branches are uniform across the workgroup, so the barriers inside
  // stay collective.
  const auto rotate = [&](const Input* source, const float* weight,
                          hip_bfloat16* destination) {
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x)
      rotated[index] = static_cast<float>(source[index]);
    __syncthreads();
    NormalizeAndRotateHead<kRoundScale>(rotated, weight, head_dim, epsilon,
                                        position, rope_theta, shared_sum,
                                        normalized);
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x)
      destination[index] = hip_bfloat16(rotated[index]);
    __syncthreads();
  };
  if (head < num_heads) {
    rotate(query + (row * num_heads + head) * head_dim, query_weight,
           query_out + (head * query_stride + query_row + row) * head_dim);
  }
  if (head < num_key_value_heads) {
    const std::size_t source = (row * num_key_value_heads + head) * head_dim;
    const std::size_t target =
        (head * cache_stride + cache_row + row) * head_dim;
    rotate(key + source, key_weight, key_out + target);
    for (std::uint32_t index = threadIdx.x; index < head_dim;
         index += blockDim.x)
      value_out[target + index] =
          hip_bfloat16(RoundedInput(value[source + index]));
  }
}

constexpr unsigned int kDecodeKeys = 256;
constexpr unsigned int kDecodeMaxGroup = 4;
// Unnormalized output channels, then the split's maximum and sum.
constexpr std::size_t kDecodePartial = 130;

__device__ __forceinline__ float BfloatBits(unsigned short bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

/// One workgroup scores `kDecodeKeys` keys of one KV head for every query
/// head sharing it and writes the unnormalized partial output with its
/// maximum and sum, which `Bfloat16DecodeMergeKernel` combines.
template<unsigned int kGroup>
__launch_bounds__(256) __global__ void Bfloat16DecodeAttentionKernel(
    const unsigned short* __restrict__ query,
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache, std::size_t stride,
    std::uint32_t keys, std::uint32_t kv_heads, float* __restrict__ partial) {
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t first = split * kDecodeKeys;
  const std::uint32_t count = min(kDecodeKeys, keys - first);
  const std::uint32_t heads = kv_heads * kGroup;
  __shared__ float q[kGroup][128];
  __shared__ float scores[kGroup][kDecodeKeys];
  __shared__ float stats[kGroup][2];
  for (unsigned int i = threadIdx.x; i < kGroup * 128; i += blockDim.x)
    q[i / 128][i % 128] =
        BfloatBits(query[(kv_head * kGroup + i / 128) * 128 + i % 128]);
  __syncthreads();
  const unsigned short* keys_base =
      key_cache + static_cast<std::size_t>(kv_head) * stride * 128;
  const unsigned short* values_base =
      value_cache + static_cast<std::size_t>(kv_head) * stride * 128;
  // One key per thread: each lane streams its own contiguous 256-byte row.
  if (threadIdx.x < count) {
    const auto* row = reinterpret_cast<const uint4*>(
        keys_base + static_cast<std::size_t>(first + threadIdx.x) * 128);
    float dot[kGroup] = {};
#pragma unroll 4
    for (unsigned int chunk = 0; chunk < 16; ++chunk) {
      const uint4 packed = row[chunk];
      const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (unsigned int w = 0; w < 4; ++w) {
        const float low = __uint_as_float(words[w] << 16U);
        const float high = __uint_as_float(words[w] & 0xFFFF0000U);
        const unsigned int d = chunk * 8 + w * 2;
#pragma unroll
        for (unsigned int g = 0; g < kGroup; ++g)
          dot[g] = fmaf(high, q[g][d + 1], fmaf(low, q[g][d], dot[g]));
      }
    }
#pragma unroll
    for (unsigned int g = 0; g < kGroup; ++g)
      scores[g][threadIdx.x] = dot[g] * rsqrtf(128.0F);
  }
  __syncthreads();
  // Wave g reduces query g's maximum and sum.
  const unsigned int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (wave < kGroup) {
    float maximum = -INFINITY;
    for (unsigned int k = lane; k < count; k += 32)
      maximum = fmaxf(maximum, scores[wave][k]);
    for (int offset = 16; offset != 0; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
    float sum = 0.0F;
    for (unsigned int k = lane; k < count; k += 32) {
      const float e = expf(scores[wave][k] - maximum);
      sum += e;
      scores[wave][k] = static_cast<float>(hip_bfloat16(e));
    }
    for (int offset = 16; offset != 0; offset /= 2)
      sum += __shfl_xor(sum, offset, 32);
    if (lane == 0) {
      stats[wave][0] = maximum;
      stats[wave][1] = sum;
    }
  }
  __syncthreads();
  // Each thread owns one output channel of every second query.
  const unsigned int d = threadIdx.x % 128;
  for (unsigned int g = threadIdx.x / 128; g < kGroup; g += 2) {
    float accumulator = 0.0F;
    for (unsigned int k = 0; k < count; ++k)
      accumulator =
          fmaf(scores[g][k],
               BfloatBits(
                   values_base[static_cast<std::size_t>(first + k) * 128 + d]),
               accumulator);
    const std::uint32_t head = kv_head * kGroup + g;
    float* out = partial + (static_cast<std::size_t>(split) * heads + head) *
                               kDecodePartial;
    out[d] = accumulator;
    if (d == 0) {
      out[128] = stats[g][0];
      out[129] = stats[g][1];
    }
  }
}

__launch_bounds__(128) __global__
    void Bfloat16DecodeMergeKernel(const float* __restrict__ partial,
                                   std::uint32_t splits, std::uint32_t heads,
                                   hip_bfloat16* __restrict__ output) {
  const std::uint32_t head = blockIdx.x;
  const unsigned int d = threadIdx.x;
  float maximum = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s)
    maximum = fmaxf(
        maximum,
        partial[(static_cast<std::size_t>(s) * heads + head) * kDecodePartial +
                128]);
  float numerator = 0.0F;
  float denominator = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* p =
        partial + (static_cast<std::size_t>(s) * heads + head) * kDecodePartial;
    const float weight = expf(p[128] - maximum);
    numerator = fmaf(weight, p[d], numerator);
    denominator = fmaf(weight, p[129], denominator);
  }
  output[head * 128 + d] = hip_bfloat16(numerator / denominator);
}

dim3 Grid(std::size_t count) {
  constexpr unsigned int block_size = 256;
  return dim3(static_cast<unsigned int>((count + block_size - 1) / block_size));
}

}  // namespace

void LaunchBfloat16Attention(const float* query, const float* key,
                             const float* value, float* key_cache,
                             float* value_cache, void* output_bfloat16,
                             std::size_t layer, std::size_t position,
                             std::size_t tokens, std::size_t capacity,
                             std::size_t heads, std::size_t kv_heads,
                             std::size_t head_dim, hipStream_t stream,
                             bool cache_written) {
  if (tokens == 0 || position > capacity || tokens > capacity - position ||
      kv_heads == 0 || heads == 0 || heads % kv_heads != 0 || head_dim == 0 ||
      head_dim > 256 || head_dim % 4 != 0)
    throw std::invalid_argument("unsupported eager attention shape");
  if (!cache_written)
    Bfloat16CacheWriteKernel<<<dim3(tokens, kv_heads), 256, 0, stream>>>(
        key, value, key_cache, value_cache, layer, position, capacity, kv_heads,
        head_dim);
  const auto launch = [&]<unsigned int Threads>() {
    Bfloat16AttentionKernel<Threads>
        <<<dim3(heads, tokens), Threads, (position + tokens) * sizeof(float),
           stream>>>(query, key_cache, value_cache,
                     static_cast<hip_bfloat16*>(output_bfloat16), layer,
                     position, capacity, heads, kv_heads, head_dim);
  };
  // Predictor contexts contain at most 16 keys. The omitted reduction
  // partners are all zero/-infinity, so active additions retain their order.
  if (position + tokens <= 32)
    launch.template operator()<128>();
  else
    launch.template operator()<256>();
}

void LaunchBfloat16PerHeadRMSNorm(float* values, const float* weight,
                                  std::size_t batch_size,
                                  std::uint32_t num_heads,
                                  std::uint32_t head_dim, float epsilon,
                                  hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || head_dim == 0) {
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  Bfloat16PerHeadRMSNormKernel<128><<<grid, 128, 0, stream>>>(
      values, weight, batch_size, num_heads, head_dim, epsilon);
}

void LaunchBfloat16RoPEAndRoundV(float* query, float* key, float* value,
                                 std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t num_key_value_heads,
                                 std::uint32_t head_dim,
                                 std::uint32_t start_position, float rope_theta,
                                 hipStream_t stream) {
  if (batch_size == 0 || num_heads == 0 || num_key_value_heads == 0 ||
      head_dim == 0) {
    return;
  }
  const std::size_t half_rotary = head_dim / 2;
  const std::size_t query_pairs =
      static_cast<std::size_t>(num_heads) * half_rotary;
  const std::size_t value_elements =
      static_cast<std::size_t>(num_key_value_heads) * head_dim;
  constexpr std::size_t kBlockSize = 256;
  const std::size_t blocks =
      (std::max(query_pairs, value_elements) + kBlockSize - 1) / kBlockSize;
  const dim3 grid(static_cast<unsigned int>(blocks),
                  static_cast<unsigned int>(batch_size));
  Bfloat16RoPEAndRoundVKernel<<<grid, kBlockSize, 0, stream>>>(
      query, key, value, batch_size, num_heads, num_key_value_heads, head_dim,
      start_position, rope_theta);
}

bool LaunchBfloat16QkNormRoPE(
    float* query, float* key, float* value, const float* query_weight,
    const float* key_weight, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim,
    std::uint32_t start_position, float rope_theta, float epsilon,
    float* key_cache, float* value_cache, std::uint32_t layer_index,
    std::uint32_t max_context, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 128;
  if (batch_size == 0 || num_heads == 0 || num_key_value_heads == 0 ||
      head_dim == 0 || head_dim > kThreads || head_dim % 2 != 0) {
    return false;
  }
  if (key_cache != nullptr &&
      (value_cache == nullptr || start_position + batch_size > max_context)) {
    return false;
  }
  const dim3 grid(std::max(num_heads, num_key_value_heads),
                  static_cast<unsigned int>(batch_size));
  Bfloat16QkNormRoPEKernel<kThreads><<<grid, kThreads, 0, stream>>>(
      query, key, value, query_weight, key_weight, batch_size, num_heads,
      num_key_value_heads, head_dim, start_position, rope_theta, epsilon,
      key_cache, value_cache, layer_index, max_context);
  return true;
}

void LaunchBfloat16ResidualAddRMSNorm(float* hidden, const float* update,
                                      const float* weight,
                                      void* output_bfloat16,
                                      std::size_t batch_size,
                                      std::size_t dimension, float epsilon,
                                      hipStream_t stream, RmsScale scale) {
  if (batch_size == 0 || dimension == 0) {
    return;
  }
  const auto launch = [&]<bool kRoundScale>() {
    Bfloat16ResidualAddRMSNormKernel<kRoundScale>
        <<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
            hidden, update, weight, static_cast<hip_bfloat16*>(output_bfloat16),
            batch_size, dimension, epsilon);
  };
  if (scale == RmsScale::kBfloat16)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
}

void LaunchBfloat16ResidualAddRMSNormFromBfloat16(
    float* hidden, const void* update_bfloat16, const float* weight,
    void* output_bfloat16, std::size_t batch_size, std::size_t dimension,
    float epsilon, hipStream_t stream, RmsScale scale) {
  if (batch_size == 0 || dimension == 0) {
    return;
  }
  const auto launch = [&]<bool kRoundScale>() {
    Bfloat16ResidualAddRMSNormKernel<kRoundScale, hip_bfloat16>
        <<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
            hidden, static_cast<const hip_bfloat16*>(update_bfloat16), weight,
            static_cast<hip_bfloat16*>(output_bfloat16), batch_size, dimension,
            epsilon);
  };
  if (scale == RmsScale::kBfloat16)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
}

void LaunchRoundBfloat16InPlace(float* values, std::size_t count,
                                hipStream_t stream) {
  if (count == 0) {
    return;
  }
  RoundBfloat16InPlaceKernel<<<Grid(count), 256, 0, stream>>>(values, count);
}

void LaunchBfloat16ToFloat(const void* input_bfloat16, float* output,
                           std::size_t count, hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16ToFloatKernel<<<Grid(count), 256, 0, stream>>>(
      static_cast<const hip_bfloat16*>(input_bfloat16), output, count);
}

void LaunchBfloat16ResidualAdd(const float* residual, const float* update,
                               float* output, std::size_t count,
                               hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16ResidualAddKernel<<<Grid(count), 256, 0, stream>>>(residual, update,
                                                             output, count);
}

void LaunchBfloat16SwiGlu(const float* gate, const float* up, float* output,
                          void* output_bfloat16, std::size_t count,
                          hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16SwiGluKernel<<<Grid(count), 256, 0, stream>>>(
      gate, up, output, static_cast<hip_bfloat16*>(output_bfloat16), count);
}

void LaunchBfloat16SwiGluFromBfloat16(const void* gate_bfloat16,
                                      const void* up_bfloat16,
                                      void* output_bfloat16, std::size_t count,
                                      hipStream_t stream) {
  if (count == 0) {
    return;
  }
  Bfloat16SwiGluKernel<<<Grid(count), 256, 0, stream>>>(
      static_cast<const hip_bfloat16*>(gate_bfloat16),
      static_cast<const hip_bfloat16*>(up_bfloat16), nullptr,
      static_cast<hip_bfloat16*>(output_bfloat16), count);
}

void LaunchBfloat16BiasSilu(const float* input, const void* bias_bfloat16,
                            float* output, void* output_bfloat16,
                            std::size_t rows, std::size_t columns,
                            hipStream_t stream) {
  const std::size_t count = rows * columns;
  if (count == 0) {
    return;
  }
  Bfloat16BiasSiluKernel<<<Grid(count), 256, 0, stream>>>(
      input, static_cast<const hip_bfloat16*>(bias_bfloat16), output,
      static_cast<hip_bfloat16*>(output_bfloat16), count, columns);
}

void LaunchBfloat16Bias(const float* input, const void* bias_bfloat16,
                        float* output, std::size_t rows, std::size_t columns,
                        hipStream_t stream) {
  const std::size_t count = rows * columns;
  if (count == 0) {
    return;
  }
  Bfloat16BiasKernel<<<Grid(count), 256, 0, stream>>>(
      input, static_cast<const hip_bfloat16*>(bias_bfloat16), output, count,
      columns);
}

void LaunchBatchedCodecEmbeddingSum(
    const void* const* embedding_tables_bfloat16, const std::uint32_t* codes,
    const float* text_embedding, float* output, std::size_t frames,
    std::size_t groups, std::size_t hidden_size, hipStream_t stream) {
  const std::size_t elements = frames * hidden_size;
  if (elements == 0 || groups == 0) {
    return;
  }
  BatchedCodecEmbeddingSumKernel<<<Grid(elements), 256, 0, stream>>>(
      reinterpret_cast<const hip_bfloat16* const*>(embedding_tables_bfloat16),
      codes, text_embedding, output, elements, groups, hidden_size);
}

bool LaunchBfloat16Gemv(const void* weights_bfloat16,
                        const void* inputs_bfloat16, float* output,
                        std::size_t batch, std::size_t rows,
                        std::size_t columns, hipStream_t stream) {
  if (batch == 0 || batch > kGemvMaxBatch || rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const auto aligned = [&](std::size_t bytes) {
    return reinterpret_cast<std::uintptr_t>(weights_bfloat16) % bytes == 0 &&
           reinterpret_cast<std::uintptr_t>(inputs_bfloat16) % bytes == 0 &&
           (columns * sizeof(std::uint16_t)) % bytes == 0;
  };
  const unsigned int block = kGemvWaves * kGemvWavefront;
  const auto grid = static_cast<unsigned int>(rows);
  const auto row_count = static_cast<std::uint32_t>(rows);

  // Prefer 16-byte loads. Safetensors place every tensor on an 8-byte boundary,
  // so the narrower path is what runs against memory-mapped weights.
  const std::size_t lanes = aligned(sizeof(uint4))   ? 8
                            : aligned(sizeof(uint2)) ? 4
                                                     : 0;
  if (lanes == 0) {
    return false;
  }
  const auto packed = static_cast<std::uint32_t>(columns / lanes);

  const auto launch = [&]<unsigned int kVectors>() {
    if (lanes == 8) {
      Bfloat16GemvKernel<uint4, kVectors><<<grid, block, 0, stream>>>(
          static_cast<const uint4*>(weights_bfloat16),
          static_cast<const uint4*>(inputs_bfloat16), output, packed,
          row_count);
    } else {
      Bfloat16GemvKernel<uint2, kVectors><<<grid, block, 0, stream>>>(
          static_cast<const uint2*>(weights_bfloat16),
          static_cast<const uint2*>(inputs_bfloat16), output, packed,
          row_count);
    }
  };
  switch (batch) {
    case 1:
      launch.operator()<1>();
      return true;
    case 2:
      launch.operator()<2>();
      return true;
    case 3:
      launch.operator()<3>();
      return true;
    case 4:
      launch.operator()<4>();
      return true;
    default:
      return false;
  }
}

bool LaunchBfloat16GemvGrouped(const Bfloat16GemvGroup* groups,
                               std::size_t group_count,
                               const void* inputs_bfloat16, std::size_t batch,
                               std::size_t columns, hipStream_t stream) {
  if (groups == nullptr || group_count < 2 ||
      group_count > kBfloat16GemvMaxGroups || batch == 0 ||
      batch > kGemvMaxBatch || columns == 0) {
    return false;
  }
  std::size_t total_rows = 0;
  for (std::size_t group = 0; group < group_count; ++group) {
    if (groups[group].weights_bfloat16 == nullptr ||
        groups[group].output == nullptr || groups[group].rows == 0) {
      return false;
    }
    total_rows += groups[group].rows;
  }
  if (total_rows > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }

  // Every group has to use the same load width, so the narrowest alignment in
  // the set decides for all of them.
  const auto aligned = [&](std::size_t bytes) {
    if (reinterpret_cast<std::uintptr_t>(inputs_bfloat16) % bytes != 0 ||
        (columns * sizeof(std::uint16_t)) % bytes != 0) {
      return false;
    }
    for (std::size_t group = 0; group < group_count; ++group) {
      if (reinterpret_cast<std::uintptr_t>(groups[group].weights_bfloat16) %
              bytes !=
          0) {
        return false;
      }
    }
    return true;
  };
  const std::size_t lanes = aligned(sizeof(uint4))   ? 8
                            : aligned(sizeof(uint2)) ? 4
                                                     : 0;
  if (lanes == 0) {
    return false;
  }

  const unsigned int block = kGemvWaves * kGemvWavefront;
  const auto grid = static_cast<unsigned int>(total_rows);
  const auto packed = static_cast<std::uint32_t>(columns / lanes);

  const auto launch =
      [&]<typename Packed, unsigned int kGroups, unsigned int kVectors>() {
        GemvGroupedArgs<Packed, kGroups> args{};
        std::uint32_t offset = 0;
        for (unsigned int group = 0; group < kGroups; ++group) {
          args.weights[group] =
              static_cast<const Packed*>(groups[group].weights_bfloat16);
          args.outputs[group] = groups[group].output;
          args.rows[group] = static_cast<std::uint32_t>(groups[group].rows);
          offset += args.rows[group];
          args.row_end[group] = offset;
        }
        Bfloat16GemvGroupedKernel<Packed, kGroups, kVectors>
            <<<grid, block, 0, stream>>>(
                args, static_cast<const Packed*>(inputs_bfloat16), packed);
      };
  const auto dispatch_vectors = [&]<typename Packed, unsigned int kGroups>() {
    switch (batch) {
      case 1:
        launch.template operator()<Packed, kGroups, 1>();
        return true;
      case 2:
        launch.template operator()<Packed, kGroups, 2>();
        return true;
      case 3:
        launch.template operator()<Packed, kGroups, 3>();
        return true;
      case 4:
        launch.template operator()<Packed, kGroups, 4>();
        return true;
      default:
        return false;
    }
  };
  const auto dispatch_groups = [&]<typename Packed>() {
    return group_count == 2 ? dispatch_vectors.template operator()<Packed, 2>()
                            : dispatch_vectors.template operator()<Packed, 3>();
  };
  return lanes == 8 ? dispatch_groups.template operator()<uint4>()
                    : dispatch_groups.template operator()<uint2>();
}

template<typename Input>
void QkNormRoPEHeadMajor(
    const Input* query, const Input* key, const Input* value,
    const float* query_weight, const float* key_weight, std::size_t rows,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta,
    float epsilon, RmsScale scale, void* query_out, std::size_t query_stride,
    std::size_t query_row, void* key_out, void* value_out,
    std::size_t cache_stride, std::size_t cache_row, hipStream_t stream) {
  if (rows == 0)
    return;
  if (head_dim == 0 || head_dim > 128 || head_dim % 2 != 0 || num_heads == 0 ||
      num_key_value_heads == 0 ||
      rows > std::numeric_limits<unsigned int>::max())
    throw std::invalid_argument("unsupported eager head-major norm shape");
  const dim3 grid(std::max(num_heads, num_key_value_heads),
                  static_cast<unsigned int>(rows));
  const auto launch = [&]<bool kRoundScale>() {
    Bfloat16QkNormRoPEHeadMajorKernel<kRoundScale, Input>
        <<<grid, 128, 0, stream>>>(
            query, key, value, query_weight, key_weight, num_heads,
            num_key_value_heads, head_dim, start_position, rope_theta, epsilon,
            static_cast<hip_bfloat16*>(query_out), query_stride, query_row,
            static_cast<hip_bfloat16*>(key_out),
            static_cast<hip_bfloat16*>(value_out), cache_stride, cache_row);
  };
  if (scale == RmsScale::kBfloat16)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
}

void LaunchBfloat16QkNormRoPEHeadMajor(
    const float* query, const float* key, const float* value,
    const float* query_weight, const float* key_weight, std::size_t rows,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta,
    float epsilon, RmsScale scale, void* query_out, std::size_t query_stride,
    std::size_t query_row, void* key_out, void* value_out,
    std::size_t cache_stride, std::size_t cache_row, hipStream_t stream) {
  QkNormRoPEHeadMajor(query, key, value, query_weight, key_weight, rows,
                      num_heads, num_key_value_heads, head_dim, start_position,
                      rope_theta, epsilon, scale, query_out, query_stride,
                      query_row, key_out, value_out, cache_stride, cache_row,
                      stream);
}

void LaunchBfloat16QkNormRoPEHeadMajorFromBfloat16(
    const void* query, const void* key, const void* value,
    const float* query_weight, const float* key_weight, std::size_t rows,
    std::uint32_t num_heads, std::uint32_t num_key_value_heads,
    std::uint32_t head_dim, std::uint32_t start_position, float rope_theta,
    float epsilon, RmsScale scale, void* query_out, std::size_t query_stride,
    std::size_t query_row, void* key_out, void* value_out,
    std::size_t cache_stride, std::size_t cache_row, hipStream_t stream) {
  QkNormRoPEHeadMajor(static_cast<const hip_bfloat16*>(query),
                      static_cast<const hip_bfloat16*>(key),
                      static_cast<const hip_bfloat16*>(value), query_weight,
                      key_weight, rows, num_heads, num_key_value_heads,
                      head_dim, start_position, rope_theta, epsilon, scale,
                      query_out, query_stride, query_row, key_out, value_out,
                      cache_stride, cache_row, stream);
}

std::size_t DecodeAttentionScratch(std::size_t capacity, std::size_t heads) {
  const std::size_t splits = (capacity + kDecodeKeys - 1) / kDecodeKeys;
  return splits * heads * kDecodePartial;
}

void LaunchBfloat16DecodeAttention(const void* query, const void* key_cache,
                                   const void* value_cache, std::size_t stride,
                                   std::size_t keys, std::size_t heads,
                                   std::size_t kv_heads, float* scratch,
                                   void* output, hipStream_t stream) {
  if (keys == 0 || keys > stride || kv_heads == 0 || heads % kv_heads != 0 ||
      heads / kv_heads > kDecodeMaxGroup ||
      keys > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("unsupported eager decode attention shape");
  const auto splits =
      static_cast<unsigned int>((keys + kDecodeKeys - 1) / kDecodeKeys);
  const dim3 grid(static_cast<unsigned int>(kv_heads), splits);
  const auto* q = static_cast<const unsigned short*>(query);
  const auto* k = static_cast<const unsigned short*>(key_cache);
  const auto* v = static_cast<const unsigned short*>(value_cache);
  const auto launch = [&]<unsigned int kGroup>() {
    Bfloat16DecodeAttentionKernel<kGroup><<<grid, 256, 0, stream>>>(
        q, k, v, stride, static_cast<std::uint32_t>(keys),
        static_cast<std::uint32_t>(kv_heads), scratch);
  };
  switch (heads / kv_heads) {
    case 1:
      launch.template operator()<1>();
      break;
    case 2:
      launch.template operator()<2>();
      break;
    case 3:
      launch.template operator()<3>();
      break;
    default:
      launch.template operator()<4>();
      break;
  }
  Bfloat16DecodeMergeKernel<<<static_cast<unsigned int>(heads), 128, 0,
                              stream>>>(scratch, splits,
                                        static_cast<std::uint32_t>(heads),
                                        static_cast<hip_bfloat16*>(output));
}

}  // namespace gufo::qwen::eager

#endif  // defined(ENGINE_ENABLE_HIP)
