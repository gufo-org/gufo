#include "src/models/qwen3_tts/hip/speech_decoder_ops.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::models::qwen3_tts::hip {
namespace {

constexpr std::size_t kThreads = 256;
constexpr float kCodebookEpsilon = 1.0e-5F;

__global__ void DecodeCodebookKernel(const std::uint32_t* codes,
                                     std::uint32_t code_groups,
                                     std::uint32_t group_index,
                                     const float* embedding_sum,
                                     const float* cluster_usage, float* output,
                                     std::size_t frames, std::size_t dimension,
                                     bool initialize) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t elements = frames * dimension;
  if (index >= elements) {
    return;
  }
  const std::size_t frame = index / dimension;
  const std::size_t column = index % dimension;
  const std::uint32_t code = codes[frame * code_groups + group_index];
  const float denominator = fmaxf(cluster_usage[code], kCodebookEpsilon);
  const float decoded =
      embedding_sum[static_cast<std::size_t>(code) * dimension + column] /
      denominator;
  output[index] = initialize ? decoded : output[index] + decoded;
}

__global__ void AddKernel(const float* left, const float* right, float* output,
                          std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = left[index] + right[index];
  }
}

__global__ void AddBiasKernel(float* values, const float* bias,
                              std::size_t elements, std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] += bias[index % columns];
  }
}

__global__ void AddScaledChannelsKernel(float* residual, const float* update,
                                        const float* scale,
                                        std::size_t elements,
                                        std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    residual[index] =
        fmaf(update[index], scale[index % columns], residual[index]);
  }
}

__global__ void RmsNormKernel(const float* input, const float* weight,
                              float* output, std::size_t rows,
                              std::size_t columns, float epsilon) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reduction[256];
  float sum = 0.0F;
  const std::size_t base = row * columns;
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float value = input[base + column];
    sum = fmaf(value, value, sum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    output[base + column] = input[base + column] * inverse * weight[column];
  }
}

__global__ void LayerNormKernel(const float* input, const float* weight,
                                const float* bias, float* output,
                                std::size_t rows, std::size_t columns,
                                float epsilon) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reduction[256];
  const std::size_t base = row * columns;
  float sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    sum += input[base + column];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = reduction[0] / static_cast<float>(columns);
  float squared = 0.0F;
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const float centered = input[base + column] - mean;
    squared = fmaf(centered, centered, squared);
  }
  reduction[threadIdx.x] = squared;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reduction[0] / static_cast<float>(columns) + epsilon);
  for (std::size_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    output[base + column] =
        (input[base + column] - mean) * inverse * weight[column] + bias[column];
  }
}

__global__ void RopeKernel(float* query, float* key, std::size_t frames,
                           std::size_t heads, std::size_t head_dimension,
                           float rope_theta, std::size_t start_position) {
  const std::size_t half = head_dimension / 2;
  const std::size_t pairs = frames * heads * half;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) {
    return;
  }
  const std::size_t pair = index % half;
  const std::size_t head_and_frame = index / half;
  const std::size_t head = head_and_frame % heads;
  const std::size_t frame = head_and_frame / heads;
  const float exponent =
      -2.0F * static_cast<float>(pair) / static_cast<float>(head_dimension);
  const float angle =
      static_cast<float>(frame + start_position) * powf(rope_theta, exponent);
  float sine = 0.0F;
  float cosine = 0.0F;
  sincosf(angle, &sine, &cosine);
  const std::size_t base = (frame * heads + head) * head_dimension + pair;
  const float query_left = query[base];
  const float query_right = query[base + half];
  const float key_left = key[base];
  const float key_right = key[base + half];
  query[base] = query_left * cosine - query_right * sine;
  query[base + half] = query_right * cosine + query_left * sine;
  key[base] = key_left * cosine - key_right * sine;
  key[base + half] = key_right * cosine + key_left * sine;
}

__global__ void SlidingCausalAttentionKernel(
    const float* query, const float* key, const float* value, float* output,
    std::size_t frames, std::size_t heads, std::size_t head_dimension,
    std::size_t sliding_window, std::size_t query_offset) {
  const std::size_t local_query = blockIdx.x / heads;
  const std::size_t query_frame = local_query + query_offset;
  const std::size_t head = blockIdx.x % heads;
  if (query_frame >= frames) {
    return;
  }
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = shared + frames;
  const std::size_t first_key =
      query_frame + 1 > sliding_window ? query_frame + 1 - sliding_window : 0;
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  const std::size_t query_base = (local_query * heads + head) * head_dimension;

  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t key_frame = threadIdx.x; key_frame < frames;
       key_frame += blockDim.x) {
    float score = -std::numeric_limits<float>::infinity();
    if (key_frame >= first_key && key_frame <= query_frame) {
      const std::size_t key_base = (key_frame * heads + head) * head_dimension;
      float dot = 0.0F;
      for (std::size_t column = 0; column < head_dimension; ++column) {
        dot = fmaf(query[query_base + column], key[key_base + column], dot);
      }
      score = dot * scale;
    }
    scores[key_frame] = score;
    local_max = fmaxf(local_max, score);
  }
  reduction[threadIdx.x] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  __syncthreads();
  float local_sum = 0.0F;
  for (std::size_t key_frame = threadIdx.x; key_frame < frames;
       key_frame += blockDim.x) {
    const float probability =
        isfinite(scores[key_frame]) ? expf(scores[key_frame] - maximum) : 0.0F;
    scores[key_frame] = probability;
    local_sum += probability;
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  for (std::size_t column = threadIdx.x; column < head_dimension;
       column += blockDim.x) {
    float result = 0.0F;
    for (std::size_t key_frame = first_key; key_frame <= query_frame;
         ++key_frame) {
      const std::size_t value_index =
          (key_frame * heads + head) * head_dimension + column;
      result =
          fmaf(scores[key_frame] * inverse_sum, value[value_index], result);
    }
    output[query_base + column] = result;
  }
}

__global__ void SwiGluKernel(const float* gate, const float* up, float* output,
                             std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = gate[index];
    output[index] = value / (1.0F + expf(-value)) * up[index];
  }
}

__global__ void GeluKernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    constexpr float inverse_sqrt_two = 0.70710678118654752440F;
    const float value = values[index];
    values[index] = 0.5F * value * (1.0F + erff(value * inverse_sqrt_two));
  }
}

__global__ void SnakeBetaKernel(const float* input, const float* alpha,
                                const float* beta, float* output,
                                std::size_t elements, std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t channel = index % columns;
    const float alpha_exp = expf(alpha[channel]);
    const float beta_exp = expf(beta[channel]);
    const float sine = sinf(input[index] * alpha_exp);
    output[index] = input[index] + sine * sine / (beta_exp + 1.0e-9F);
  }
}

__global__ void CausalConv1dIm2ColKernel(const float* input, float* columns,
                                         std::size_t input_length,
                                         std::size_t input_channels,
                                         std::size_t output_length,
                                         std::size_t kernel,
                                         std::size_t dilation) {
  const std::size_t reduction = input_channels * kernel;
  const std::size_t elements = output_length * reduction;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t tap = index % kernel;
  const std::size_t remaining = index / kernel;
  const std::size_t channel = remaining % input_channels;
  const std::size_t output_time = remaining / input_channels;
  const std::size_t padding = (kernel - 1) * dilation;
  const std::int64_t input_time = static_cast<std::int64_t>(output_time) -
                                  static_cast<std::int64_t>(padding) +
                                  static_cast<std::int64_t>(tap * dilation);
  columns[index] =
      input_time < 0 || input_time >= static_cast<std::int64_t>(input_length)
          ? 0.0F
          : input[static_cast<std::size_t>(input_time) * input_channels +
                  channel];
}

__global__ void CausalDepthwiseConv1dKernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::size_t length, std::size_t channels, std::size_t kernel,
    std::size_t dilation) {
  const std::size_t elements = length * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t time = index / channels;
  const std::size_t channel = index % channels;
  const std::size_t padding = (kernel - 1) * dilation;
  float result = bias == nullptr ? 0.0F : bias[channel];
  for (std::size_t tap = 0; tap < kernel; ++tap) {
    const std::int64_t input_time = static_cast<std::int64_t>(time) -
                                    static_cast<std::int64_t>(padding) +
                                    static_cast<std::int64_t>(tap * dilation);
    if (input_time >= 0 && input_time < static_cast<std::int64_t>(length)) {
      result =
          fmaf(input[static_cast<std::size_t>(input_time) * channels + channel],
               weight[channel * kernel + tap], result);
    }
  }
  output[index] = result;
}

__global__ void TransposeConvWeightKernel(const float* source, float* target,
                                          std::size_t input_channels,
                                          std::size_t output_channels,
                                          std::size_t kernel) {
  const std::size_t elements = input_channels * output_channels * kernel;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t tap = index % kernel;
  const std::size_t remaining = index / kernel;
  const std::size_t output_channel = remaining % output_channels;
  const std::size_t input_channel = remaining / output_channels;
  const std::size_t target_index =
      (output_channel * kernel + tap) * input_channels + input_channel;
  target[target_index] = source[index];
}

__global__ void AssembleCausalConvTranspose1dKernel(
    const float* expanded, const float* bias, float* output,
    std::size_t input_length, std::size_t output_length,
    std::size_t output_channels, std::size_t kernel, std::size_t stride) {
  const std::size_t elements = output_length * output_channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t output_time = index / output_channels;
  const std::size_t output_channel = index % output_channels;
  float result = bias == nullptr ? 0.0F : bias[output_channel];
  for (std::size_t tap = 0; tap < kernel; ++tap) {
    const std::int64_t numerator =
        static_cast<std::int64_t>(output_time) - static_cast<std::int64_t>(tap);
    if (numerator < 0 || numerator % static_cast<std::int64_t>(stride) != 0) {
      continue;
    }
    const std::size_t input_time =
        static_cast<std::size_t>(numerator / static_cast<std::int64_t>(stride));
    if (input_time < input_length) {
      const std::size_t expanded_index =
          (input_time * output_channels + output_channel) * kernel + tap;
      result += expanded[expanded_index];
    }
  }
  output[index] = result;
}

__global__ void ClampKernel(float* values, std::size_t elements, float minimum,
                            float maximum) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = fminf(maximum, fmaxf(minimum, values[index]));
  }
}

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

}  // namespace

void LaunchDecodeCodebook(const std::uint32_t* codes, std::uint32_t code_groups,
                          std::uint32_t group_index, const float* embedding_sum,
                          const float* cluster_usage, float* output,
                          std::size_t frames, std::size_t dimension,
                          bool initialize, hipStream_t stream) {
  const std::size_t elements = frames * dimension;
  hipLaunchKernelGGL(DecodeCodebookKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, codes, code_groups, group_index,
                     embedding_sum, cluster_usage, output, frames, dimension,
                     initialize);
}

void LaunchAdd(const float* left, const float* right, float* output,
               std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, left, right, output, elements);
}

void LaunchAddBias(float* values, const float* bias, std::size_t rows,
                   std::size_t columns, hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(AddBiasKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, bias, elements, columns);
}

void LaunchAddScaledChannels(float* residual, const float* update,
                             const float* scale, std::size_t rows,
                             std::size_t columns, hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(AddScaledChannelsKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, residual, update, scale,
                     elements, columns);
}

void LaunchRmsNorm(const float* input, const float* weight, float* output,
                   std::size_t rows, std::size_t columns, float epsilon,
                   hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, output, rows, columns, epsilon);
}

void LaunchLayerNorm(const float* input, const float* weight, const float* bias,
                     float* output, std::size_t rows, std::size_t columns,
                     float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, bias, output, rows, columns, epsilon);
}

void LaunchRope(float* query, float* key, std::size_t frames, std::size_t heads,
                std::size_t head_dimension, float rope_theta,
                hipStream_t stream, std::size_t start_position) {
  const std::size_t pairs = frames * heads * (head_dimension / 2);
  hipLaunchKernelGGL(RopeKernel, dim3(Blocks(pairs)), dim3(kThreads), 0, stream,
                     query, key, frames, heads, head_dimension, rope_theta,
                     start_position);
}

void LaunchSlidingCausalAttention(const float* query, const float* key,
                                  const float* value, float* output,
                                  std::size_t frames, std::size_t heads,
                                  std::size_t head_dimension,
                                  std::size_t sliding_window,
                                  hipStream_t stream, std::size_t query_offset,
                                  std::size_t key_frames) {
  if (key_frames == 0) key_frames = frames;
  constexpr std::size_t threads = 128;
  const std::size_t shared_bytes = (key_frames + threads) * sizeof(float);
  hipLaunchKernelGGL(SlidingCausalAttentionKernel, dim3(frames * heads),
                     dim3(threads), shared_bytes, stream, query, key, value,
                     output, key_frames, heads, head_dimension, sliding_window,
                     query_offset);
}

void LaunchSwiGlu(const float* gate, const float* up, float* output,
                  std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(SwiGluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, gate, up, output, elements);
}

void LaunchGelu(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(GeluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchSnakeBeta(const float* input, const float* alpha, const float* beta,
                     float* output, std::size_t rows, std::size_t columns,
                     hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(SnakeBetaKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, input, alpha, beta, output, elements, columns);
}

void LaunchCausalConv1dIm2Col(const float* input, float* columns,
                              std::size_t input_length,
                              std::size_t input_channels,
                              std::size_t output_length, std::size_t kernel,
                              std::size_t dilation, hipStream_t stream) {
  const std::size_t elements = output_length * input_channels * kernel;
  hipLaunchKernelGGL(CausalConv1dIm2ColKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input, columns, input_length,
                     input_channels, output_length, kernel, dilation);
}

void LaunchCausalDepthwiseConv1d(const float* input, const float* weight,
                                 const float* bias, float* output,
                                 std::size_t length, std::size_t channels,
                                 std::size_t kernel, std::size_t dilation,
                                 hipStream_t stream) {
  const std::size_t elements = length * channels;
  hipLaunchKernelGGL(CausalDepthwiseConv1dKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     length, channels, kernel, dilation);
}

void LaunchTransposeConvWeight(const float* source, float* target,
                               std::size_t input_channels,
                               std::size_t output_channels, std::size_t kernel,
                               hipStream_t stream) {
  const std::size_t elements = input_channels * output_channels * kernel;
  hipLaunchKernelGGL(TransposeConvWeightKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, source, target, input_channels,
                     output_channels, kernel);
}

void LaunchAssembleCausalConvTranspose1d(const float* expanded,
                                         const float* bias, float* output,
                                         std::size_t input_length,
                                         std::size_t output_length,
                                         std::size_t output_channels,
                                         std::size_t kernel, std::size_t stride,
                                         hipStream_t stream) {
  const std::size_t elements = output_length * output_channels;
  hipLaunchKernelGGL(AssembleCausalConvTranspose1dKernel,
                     dim3(Blocks(elements)), dim3(kThreads), 0, stream,
                     expanded, bias, output, input_length, output_length,
                     output_channels, kernel, stride);
}

void LaunchClamp(float* values, std::size_t elements, float minimum,
                 float maximum, hipStream_t stream) {
  hipLaunchKernelGGL(ClampKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements, minimum, maximum);
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
