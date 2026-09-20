#include "src/models/qwen3_asr/hip/audio_ops.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gufo::models::qwen3_asr::hip {
namespace {

constexpr std::size_t kThreads = 256;
constexpr std::size_t kAttentionThreads = 64;
constexpr std::size_t kMelBins = 128;
constexpr std::size_t kChunkFrames = 100;
constexpr std::size_t kConvChannels = 480;
constexpr std::size_t kConvFrequency = 16;
constexpr std::size_t kConvTime = 13;
constexpr std::size_t kConvFlattened = kConvChannels * kConvFrequency;
constexpr std::size_t kHidden = 1024;

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

__global__ void ChunkLogMelKernel(const float* input, hip_bfloat16* output,
                                  std::size_t frames, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t time = index % kChunkFrames;
  const std::size_t remaining = index / kChunkFrames;
  const std::size_t mel = remaining % kMelBins;
  const std::size_t chunk = remaining / kMelBins;
  const std::size_t source_time = chunk * kChunkFrames + time;
  output[index] = hip_bfloat16(
      source_time < frames ? input[mel * frames + source_time] : 0.0F);
}

__global__ void BiasGeluKernel(hip_bfloat16* values, const hip_bfloat16* bias,
                               std::size_t elements, std::size_t channels,
                               std::size_t spatial_size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t channel = (index / spatial_size) % channels;
  const float value = static_cast<float>(hip_bfloat16(
      static_cast<float>(values[index]) + static_cast<float>(bias[channel])));
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  const float activated = 0.5F * value * (1.0F + erff(value * kInverseSqrtTwo));
  values[index] = hip_bfloat16(activated);
}

__global__ void ConvOutputLayoutKernel(const hip_bfloat16* input,
                                       hip_bfloat16* output,
                                       std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t flattened = index % kConvFlattened;
  const std::size_t row = index / kConvFlattened;
  const std::size_t time = row % kConvTime;
  const std::size_t chunk = row / kConvTime;
  const std::size_t frequency = flattened % kConvFrequency;
  const std::size_t channel = flattened / kConvFrequency;
  const std::size_t source =
      (((chunk * kConvChannels + channel) * kConvFrequency + frequency) *
           kConvTime +
       time);
  output[index] = input[source];
}

__device__ float Position(std::size_t time, std::size_t column) {
  constexpr float kLogIncrement = 9.210340371976184F / 511.0F;
  const std::size_t frequency = column % (kHidden / 2U);
  const float inverse_timescale =
      expf(-kLogIncrement * static_cast<float>(frequency));
  const float phase = static_cast<float>(time) * inverse_timescale;
  return column < kHidden / 2U ? sinf(phase) : cosf(phase);
}

__global__ void AddPositionAndCompactKernel(const float* projected,
                                            float* compact_output,
                                            std::size_t feature_frames,
                                            std::size_t chunks,
                                            std::size_t output_tokens) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_tokens * kHidden) {
    return;
  }
  const std::size_t column = index % kHidden;
  const std::size_t output_token = index / kHidden;

  std::size_t remaining = output_token;
  std::size_t selected_chunk = 0;
  std::size_t selected_time = 0;
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    const std::size_t chunk_frames =
        min(kChunkFrames, feature_frames - chunk * kChunkFrames);
    const std::size_t valid = (chunk_frames + 7U) / 8U;
    if (remaining < valid) {
      selected_chunk = chunk;
      selected_time = remaining;
      break;
    }
    remaining -= valid;
  }

  const std::size_t source =
      (selected_chunk * kConvTime + selected_time) * kHidden + column;
  const float linear = static_cast<float>(hip_bfloat16(projected[source]));
  const float position =
      static_cast<float>(hip_bfloat16(Position(selected_time, column)));
  compact_output[index] = static_cast<float>(hip_bfloat16(linear + position));
}

__global__ void LayerNormToBfloat16Kernel(const float* input,
                                          const hip_bfloat16* weight,
                                          const hip_bfloat16* bias,
                                          hip_bfloat16* output,
                                          std::size_t columns, float epsilon) {
  const std::size_t row = blockIdx.x;
  extern __shared__ float scratch[];
  float sum = 0.0F;
  float square_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float value = input[row * columns + column];
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  scratch[threadIdx.x] = sum;
  scratch[blockDim.x + threadIdx.x] = square_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      scratch[blockDim.x + threadIdx.x] +=
          scratch[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(columns);
  const float variance = fmaxf(
      scratch[blockDim.x] / static_cast<float>(columns) - mean * mean, 0.0F);
  const float inverse_std = rsqrtf(variance + epsilon);
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const std::size_t index = row * columns + column;
    output[index] = hip_bfloat16(((input[index] - mean) * inverse_std) *
                                     static_cast<float>(weight[column]) +
                                 static_cast<float>(bias[column]));
  }
}

__global__ void BiasRoundBfloat16Kernel(float* values, const hip_bfloat16* bias,
                                        std::size_t elements,
                                        std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = static_cast<float>(hip_bfloat16(
        values[index] + static_cast<float>(bias[index % columns])));
  }
}

__global__ void ResidualAddBfloat16Kernel(float* residual, const float* update,
                                          std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    residual[index] =
        static_cast<float>(hip_bfloat16(residual[index] + update[index]));
  }
}

__global__ void BiasGeluToBfloat16Kernel(const float* input,
                                         const hip_bfloat16* bias,
                                         hip_bfloat16* output,
                                         std::size_t elements,
                                         std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const float biased = static_cast<float>(
      hip_bfloat16(input[index] + static_cast<float>(bias[index % columns])));
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  output[index] =
      hip_bfloat16(0.5F * biased * (1.0F + erff(biased * kInverseSqrtTwo)));
}

// Largest head dimension the register-resident query path covers. The audio
// tower is 1024 hidden over 16 heads, so 64.
constexpr std::size_t kAttentionMaxHeadDim = 64;

/// Number of value rows prefetched before the first is accumulated.
constexpr std::size_t kAttentionValueUnroll = 8;

// `Vectorized` holds the query row in registers and reads both score operands
// sixteen bytes at a time instead of one float at a time, and prefetches value
// rows. Every fmaf chain keeps its ascending order and single accumulator, so
// both instantiations produce bit-identical results; the generic one remains
// for head dimensions the fast path does not cover.
template<bool Vectorized>
__global__ void SegmentedSelfAttentionKernel(
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, float* __restrict__ output,
    hip_bfloat16* __restrict__ output_bfloat16, std::size_t tokens,
    std::size_t segment_tokens, std::size_t heads, std::size_t head_dim) {
  const std::size_t query_token = blockIdx.x;
  const std::size_t head = blockIdx.y;
  const std::size_t segment_begin =
      (query_token / segment_tokens) * segment_tokens;
  const std::size_t segment_end = min(segment_begin + segment_tokens, tokens);
  const std::size_t segment_length = segment_end - segment_begin;
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = shared + segment_length;
  const std::size_t query_base = (query_token * heads + head) * head_dim;
  // The query row is the same for every key this block scores, so it is read
  // once rather than once per key.
  float4 query_row[kAttentionMaxHeadDim / 4];
  if (Vectorized) {
    const auto* query_vec = reinterpret_cast<const float4*>(query + query_base);
    for (std::size_t vector = 0; vector < head_dim / 4U; ++vector) {
      query_row[vector] = query_vec[vector];
    }
  }

  float local_max = -INFINITY;
  for (std::size_t offset = threadIdx.x; offset < segment_length;
       offset += blockDim.x) {
    const std::size_t key_token = segment_begin + offset;
    float dot = 0.0F;
    const std::size_t key_base = (key_token * heads + head) * head_dim;
    if (Vectorized) {
      const auto* key_vec = reinterpret_cast<const float4*>(key + key_base);
      for (std::size_t vector = 0; vector < head_dim / 4U; ++vector) {
        const float4 k = key_vec[vector];
        const float4 q = query_row[vector];
        dot = fmaf(q.x, k.x, dot);
        dot = fmaf(q.y, k.y, dot);
        dot = fmaf(q.z, k.z, dot);
        dot = fmaf(q.w, k.w, dot);
      }
    } else {
      for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
        dot =
            fmaf(query[query_base + dimension], key[key_base + dimension], dot);
      }
    }
    const float product = static_cast<float>(hip_bfloat16(dot));
    const float score = static_cast<float>(
        hip_bfloat16(product / sqrtf(static_cast<float>(head_dim))));
    scores[offset] = score;
    local_max = fmaxf(local_max, score);
  }

  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  // Every wave must read the maximum before wave zero can overwrite its
  // shared slot with the exponential sum (especially for short tail windows).
  __syncthreads();
  float local_sum = 0.0F;
  for (std::size_t offset = threadIdx.x; offset < segment_length;
       offset += blockDim.x) {
    const float probability = expf(scores[offset] - maximum);
    scores[offset] = probability;
    local_sum += probability;
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  for (std::size_t offset = threadIdx.x; offset < segment_length;
       offset += blockDim.x) {
    scores[offset] =
        static_cast<float>(hip_bfloat16(scores[offset] / reduction[0]));
  }
  __syncthreads();

  if (threadIdx.x < head_dim) {
    const float* value_column =
        value + ((segment_begin * heads + head) * head_dim) + threadIdx.x;
    const std::size_t row_stride = heads * head_dim;
    float context = 0.0F;
    std::size_t offset = 0;
    if (Vectorized) {
      for (; offset + kAttentionValueUnroll <= segment_length;
           offset += kAttentionValueUnroll) {
        float rows[kAttentionValueUnroll];
#pragma unroll
        for (std::size_t slot = 0; slot < kAttentionValueUnroll; ++slot) {
          rows[slot] = value_column[(offset + slot) * row_stride];
        }
#pragma unroll
        for (std::size_t slot = 0; slot < kAttentionValueUnroll; ++slot) {
          context = fmaf(scores[offset + slot], rows[slot], context);
        }
      }
    }
    for (; offset < segment_length; ++offset) {
      context =
          fmaf(scores[offset], value_column[offset * row_stride], context);
    }
    const std::size_t output_index =
        (query_token * heads + head) * head_dim + threadIdx.x;
    const hip_bfloat16 rounded(context);
    output[output_index] = static_cast<float>(rounded);
    output_bfloat16[output_index] = rounded;
  }
}

__global__ void AllFiniteKernel(const float* values, std::size_t count,
                                std::uint32_t* finite) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && !isfinite(values[index])) {
    atomicExch(finite, 0U);
  }
}

}  // namespace

void LaunchChunkLogMel(const float* input, void* output_bfloat16,
                       std::size_t frames, std::size_t chunks,
                       hipStream_t stream) {
  const std::size_t elements = chunks * kMelBins * kChunkFrames;
  hipLaunchKernelGGL(
      ChunkLogMelKernel, dim3(Blocks(elements)), dim3(kThreads), 0, stream,
      input, static_cast<hip_bfloat16*>(output_bfloat16), frames, elements);
}

void LaunchBiasGelu(void* values_bfloat16, const void* bias_bfloat16,
                    std::size_t elements, std::size_t channels,
                    std::size_t spatial_size, hipStream_t stream) {
  hipLaunchKernelGGL(BiasGeluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, static_cast<hip_bfloat16*>(values_bfloat16),
                     static_cast<const hip_bfloat16*>(bias_bfloat16), elements,
                     channels, spatial_size);
}

void LaunchConvOutputLayout(const void* input_bfloat16, void* output_bfloat16,
                            std::size_t chunks, hipStream_t stream) {
  const std::size_t elements = chunks * kConvTime * kConvFlattened;
  hipLaunchKernelGGL(ConvOutputLayoutKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream,
                     static_cast<const hip_bfloat16*>(input_bfloat16),
                     static_cast<hip_bfloat16*>(output_bfloat16), elements);
}

void LaunchAddPositionAndCompact(const float* projected, float* compact_output,
                                 std::size_t feature_frames, std::size_t chunks,
                                 hipStream_t stream) {
  const std::size_t tail_frames = feature_frames - (chunks - 1U) * kChunkFrames;
  const std::size_t output_tokens =
      (chunks - 1U) * kConvTime + (tail_frames + 7U) / 8U;
  const std::size_t elements = output_tokens * kHidden;
  hipLaunchKernelGGL(AddPositionAndCompactKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, projected, compact_output,
                     feature_frames, chunks, output_tokens);
}

void LaunchLayerNormToBfloat16(const float* input, const void* weight_bfloat16,
                               const void* bias_bfloat16, void* output_bfloat16,
                               std::size_t rows, std::size_t columns,
                               float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormToBfloat16Kernel, dim3(rows), dim3(kThreads),
                     2U * kThreads * sizeof(float), stream, input,
                     static_cast<const hip_bfloat16*>(weight_bfloat16),
                     static_cast<const hip_bfloat16*>(bias_bfloat16),
                     static_cast<hip_bfloat16*>(output_bfloat16), columns,
                     epsilon);
}

void LaunchBiasRoundBfloat16(float* values, const void* bias_bfloat16,
                             std::size_t rows, std::size_t columns,
                             hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(BiasRoundBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, values,
                     static_cast<const hip_bfloat16*>(bias_bfloat16), elements,
                     columns);
}

void LaunchResidualAddBfloat16(float* residual, const float* update,
                               std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(ResidualAddBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, residual, update, elements);
}

void LaunchBiasGeluToBfloat16(const float* input, const void* bias_bfloat16,
                              void* output_bfloat16, std::size_t rows,
                              std::size_t columns, hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(
      BiasGeluToBfloat16Kernel, dim3(Blocks(elements)), dim3(kThreads), 0,
      stream, input, static_cast<const hip_bfloat16*>(bias_bfloat16),
      static_cast<hip_bfloat16*>(output_bfloat16), elements, columns);
}

void LaunchSegmentedSelfAttention(const float* query, const float* key,
                                  const float* value, float* output,
                                  void* output_bfloat16, std::size_t tokens,
                                  std::size_t segment_tokens, std::size_t heads,
                                  std::size_t head_dim, hipStream_t stream) {
  const dim3 grid(tokens, heads);
  const std::size_t shared =
      (segment_tokens + kAttentionThreads) * sizeof(float);
  const bool vectorized =
      head_dim % 4U == 0U && head_dim <= kAttentionMaxHeadDim;
  if (vectorized) {
    hipLaunchKernelGGL(SegmentedSelfAttentionKernel<true>, grid,
                       dim3(kAttentionThreads), shared, stream, query, key,
                       value, output,
                       static_cast<hip_bfloat16*>(output_bfloat16), tokens,
                       segment_tokens, heads, head_dim);
    return;
  }
  hipLaunchKernelGGL(SegmentedSelfAttentionKernel<false>, grid,
                     dim3(kAttentionThreads), shared, stream, query, key, value,
                     output, static_cast<hip_bfloat16*>(output_bfloat16),
                     tokens, segment_tokens, heads, head_dim);
}

void LaunchAllFinite(const float* values, std::size_t count,
                     std::uint32_t* finite, hipStream_t stream) {
  if (count == 0U) {
    return;
  }
  hipLaunchKernelGGL(AllFiniteKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, values, count, finite);
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
