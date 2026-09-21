#include "src/models/qwen3_tts/hip/encoder_ops.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::models::qwen3_tts::hip {
namespace {

constexpr std::size_t kThreads = 256;

__device__ float RoundBfloat16(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) != 0x7F800000U) {
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  }
  return __uint_as_float(bits & 0xFFFF0000U);
}

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1) / kThreads;
}

__device__ std::int64_t ReflectIndex(std::int64_t index, std::int64_t length) {
  if (length <= 1) {
    return 0;
  }
  while (index < 0 || index >= length) {
    index = index < 0 ? -index : (2 * length - index - 2);
  }
  return index;
}

__global__ void Conv1dIm2ColKernel(
    const float* input, float* columns, std::size_t input_length,
    std::size_t input_channels, std::size_t output_length, std::size_t kernel,
    std::size_t stride, std::size_t dilation, std::size_t left_padding,
    EncoderPaddingMode padding_mode) {
  const std::size_t reduction = input_channels * kernel;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_length * reduction) {
    return;
  }
  const std::size_t tap = index % kernel;
  const std::size_t remaining = index / kernel;
  const std::size_t channel = remaining % input_channels;
  const std::size_t output_time = remaining / input_channels;
  std::int64_t input_time =
      static_cast<std::int64_t>(output_time * stride + tap * dilation) -
      static_cast<std::int64_t>(left_padding);
  if (input_time < 0 || input_time >= static_cast<std::int64_t>(input_length)) {
    if (padding_mode == EncoderPaddingMode::kZero) {
      columns[index] = 0.0F;
      return;
    }
    if (padding_mode == EncoderPaddingMode::kReplicate) {
      input_time = max<std::int64_t>(
          0, min<std::int64_t>(input_time,
                               static_cast<std::int64_t>(input_length) - 1));
    } else {
      input_time =
          ReflectIndex(input_time, static_cast<std::int64_t>(input_length));
    }
  }
  columns[index] =
      input[static_cast<std::size_t>(input_time) * input_channels + channel];
}

__global__ void EluKernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements && values[index] < 0.0F) {
    values[index] = expm1f(values[index]);
  }
}

__global__ void ReluKernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = fmaxf(values[index], 0.0F);
  }
}

__global__ void TanhKernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = tanhf(values[index]);
  }
}

__global__ void RoundBfloat16Kernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = RoundBfloat16(values[index]);
  }
}

__global__ void ExtractChannelsKernel(const float* input, float* output,
                                      std::size_t elements,
                                      std::size_t input_channels,
                                      std::size_t channel_offset,
                                      std::size_t output_channels) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t row = index / output_channels;
    const std::size_t channel = index % output_channels;
    output[index] = input[row * input_channels + channel_offset + channel];
  }
}

__global__ void InsertChannelsKernel(const float* input, float* output,
                                     std::size_t elements,
                                     std::size_t output_channels,
                                     std::size_t channel_offset,
                                     std::size_t input_channels) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t row = index / input_channels;
    const std::size_t channel = index % input_channels;
    output[row * output_channels + channel_offset + channel] = input[index];
  }
}

__global__ void ChannelMeanKernel(const float* input, float* output,
                                  std::size_t rows, std::size_t channels) {
  const std::size_t channel = blockIdx.x;
  extern __shared__ float shared[];
  float sum = 0.0F;
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    sum += input[row * channels + channel];
  }
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[channel] = shared[0] / static_cast<float>(rows);
  }
}

__global__ void SigmoidMultiplyChannelsKernel(float* values, const float* scale,
                                              std::size_t elements,
                                              std::size_t channels) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = scale[index % channels];
    values[index] *= 1.0F / (1.0F + expf(-value));
  }
}

__global__ void ConcatenateThreeKernel(const float* first, const float* second,
                                       const float* third, float* output,
                                       std::size_t elements,
                                       std::size_t channels) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t row = index / (channels * 3);
    const std::size_t destination = index % (channels * 3);
    const std::size_t channel = destination % channels;
    const float* source = destination < channels
                              ? first
                              : (destination < channels * 2 ? second : third);
    output[index] = source[row * channels + channel];
  }
}

__global__ void AttentiveStatsInputKernel(const float* input, float* output,
                                          std::size_t rows,
                                          std::size_t channels, float epsilon) {
  const std::size_t channel = blockIdx.x;
  extern __shared__ float shared[];
  float sum = 0.0F;
  float square_sum = 0.0F;
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float value = input[row * channels + channel];
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  shared[threadIdx.x] = sum;
  shared[blockDim.x + threadIdx.x] = square_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
      shared[blockDim.x + threadIdx.x] +=
          shared[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float mean = shared[0] / static_cast<float>(rows);
    const float variance = fmaxf(
        shared[blockDim.x] / static_cast<float>(rows) - mean * mean, 0.0F);
    const float standard_deviation = sqrtf(variance + epsilon);
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t base = row * channels * 3;
      output[base + channel] = input[row * channels + channel];
      output[base + channels + channel] = mean;
      output[base + channels * 2 + channel] = standard_deviation;
    }
  }
}

__global__ void SoftmaxOverRowsKernel(float* values, std::size_t rows,
                                      std::size_t channels) {
  const std::size_t channel = blockIdx.x;
  extern __shared__ float shared[];
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    maximum = fmaxf(maximum, values[row * channels + channel]);
  }
  shared[threadIdx.x] = maximum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  maximum = shared[0];
  // Every wave must read the maximum before thread zero reuses shared[0]
  // for its partial exponential sum.
  __syncthreads();
  float sum = 0.0F;
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::size_t index = row * channels + channel;
    const float value = expf(values[index] - maximum);
    values[index] = value;
    sum += value;
  }
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse = 1.0F / shared[0];
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    values[row * channels + channel] *= inverse;
  }
}

__global__ void WeightedStatsKernel(const float* input, const float* weights,
                                    float* output, std::size_t rows,
                                    std::size_t channels, float epsilon) {
  const std::size_t channel = blockIdx.x;
  extern __shared__ float shared[];
  float mean = 0.0F;
  float second = 0.0F;
  for (std::size_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const std::size_t index = row * channels + channel;
    const float weight = weights[index];
    const float value = input[index];
    mean = fmaf(weight, value, mean);
    second = fmaf(weight, value * value, second);
  }
  shared[threadIdx.x] = mean;
  shared[blockDim.x + threadIdx.x] = second;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
      shared[blockDim.x + threadIdx.x] +=
          shared[blockDim.x + threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float reduced_mean = shared[0];
    output[channel] = reduced_mean;
    output[channels + channel] =
        sqrtf(fmaxf(shared[blockDim.x] - reduced_mean * reduced_mean, 0.0F) +
              epsilon);
  }
}

__global__ void QuantizeCodebookKernel(
    float* residual, const float* embedding, std::uint32_t* codes,
    std::size_t dimension, std::size_t codebook_size, std::size_t code_groups,
    std::size_t group_index, bool update_residual) {
  const std::size_t frame = blockIdx.x;
  extern __shared__ unsigned char storage[];
  auto* distances = reinterpret_cast<float*>(storage);
  auto* indices = reinterpret_cast<std::uint32_t*>(distances + blockDim.x);
  float best_distance = std::numeric_limits<float>::infinity();
  std::uint32_t best_code = 0;
  for (std::size_t code = threadIdx.x; code < codebook_size;
       code += blockDim.x) {
    float distance = 0.0F;
    for (std::size_t dim = 0; dim < dimension; ++dim) {
      const float centroid = embedding[dim * codebook_size + code];
      const float difference = residual[frame * dimension + dim] - centroid;
      distance = fmaf(difference, difference, distance);
    }
    if (distance < best_distance) {
      best_distance = distance;
      best_code = static_cast<std::uint32_t>(code);
    }
  }
  distances[threadIdx.x] = best_distance;
  indices[threadIdx.x] = best_code;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float candidate = distances[threadIdx.x + stride];
      const std::uint32_t candidate_index = indices[threadIdx.x + stride];
      if (candidate < distances[threadIdx.x] ||
          (candidate == distances[threadIdx.x] &&
           candidate_index < indices[threadIdx.x])) {
        distances[threadIdx.x] = candidate;
        indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }
  const std::uint32_t code = indices[0];
  if (threadIdx.x == 0) {
    codes[frame * code_groups + group_index] = code;
  }
  if (update_residual) {
    for (std::size_t dim = threadIdx.x; dim < dimension; dim += blockDim.x) {
      const std::size_t index = frame * dimension + dim;
      residual[index] = RoundBfloat16(residual[index] -
                                      embedding[dim * codebook_size + code]);
    }
  }
}

}  // namespace

void LaunchEncoderConv1dIm2Col(
    const float* input, float* columns, std::size_t input_length,
    std::size_t input_channels, std::size_t output_length, std::size_t kernel,
    std::size_t stride, std::size_t dilation, std::size_t left_padding,
    EncoderPaddingMode padding_mode, hipStream_t stream) {
  const std::size_t elements = output_length * input_channels * kernel;
  hipLaunchKernelGGL(Conv1dIm2ColKernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, input, columns, input_length, input_channels,
                     output_length, kernel, stride, dilation, left_padding,
                     padding_mode);
}

void LaunchElu(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(EluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchRelu(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(ReluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchTanh(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(TanhKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchRoundBfloat16(float* values, std::size_t elements,
                         hipStream_t stream) {
  hipLaunchKernelGGL(RoundBfloat16Kernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, values, elements);
}

void LaunchExtractChannels(const float* input, float* output, std::size_t rows,
                           std::size_t input_channels,
                           std::size_t channel_offset,
                           std::size_t output_channels, hipStream_t stream) {
  const std::size_t elements = rows * output_channels;
  hipLaunchKernelGGL(ExtractChannelsKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input, output, elements,
                     input_channels, channel_offset, output_channels);
}

void LaunchInsertChannels(const float* input, float* output, std::size_t rows,
                          std::size_t output_channels,
                          std::size_t channel_offset,
                          std::size_t input_channels, hipStream_t stream) {
  const std::size_t elements = rows * input_channels;
  hipLaunchKernelGGL(InsertChannelsKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input, output, elements,
                     output_channels, channel_offset, input_channels);
}

void LaunchChannelMean(const float* input, float* output, std::size_t rows,
                       std::size_t channels, hipStream_t stream) {
  hipLaunchKernelGGL(ChannelMeanKernel, dim3(channels), dim3(kThreads),
                     kThreads * sizeof(float), stream, input, output, rows,
                     channels);
}

void LaunchSigmoidMultiplyChannels(float* values, const float* scale,
                                   std::size_t rows, std::size_t channels,
                                   hipStream_t stream) {
  const std::size_t elements = rows * channels;
  hipLaunchKernelGGL(SigmoidMultiplyChannelsKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, values, scale, elements,
                     channels);
}

void LaunchConcatenateThree(const float* first, const float* second,
                            const float* third, float* output, std::size_t rows,
                            std::size_t channels, hipStream_t stream) {
  const std::size_t elements = rows * channels * 3;
  hipLaunchKernelGGL(ConcatenateThreeKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, first, second, third, output,
                     elements, channels);
}

void LaunchAttentiveStatsInput(const float* input, float* output,
                               std::size_t rows, std::size_t channels,
                               float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(AttentiveStatsInputKernel, dim3(channels), dim3(kThreads),
                     2 * kThreads * sizeof(float), stream, input, output, rows,
                     channels, epsilon);
}

void LaunchSoftmaxOverRows(float* values, std::size_t rows,
                           std::size_t channels, hipStream_t stream) {
  hipLaunchKernelGGL(SoftmaxOverRowsKernel, dim3(channels), dim3(kThreads),
                     kThreads * sizeof(float), stream, values, rows, channels);
}

void LaunchWeightedStats(const float* input, const float* weights,
                         float* output, std::size_t rows, std::size_t channels,
                         float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(WeightedStatsKernel, dim3(channels), dim3(kThreads),
                     2 * kThreads * sizeof(float), stream, input, weights,
                     output, rows, channels, epsilon);
}

void LaunchQuantizeCodebook(float* residual, const float* embedding,
                            std::uint32_t* codes, std::size_t frames,
                            std::size_t dimension, std::size_t codebook_size,
                            std::size_t code_groups, std::size_t group_index,
                            bool update_residual, hipStream_t stream) {
  const std::size_t shared = kThreads * (sizeof(float) + sizeof(std::uint32_t));
  hipLaunchKernelGGL(QuantizeCodebookKernel, dim3(frames), dim3(kThreads),
                     shared, stream, residual, embedding, codes, dimension,
                     codebook_size, code_groups, group_index, update_residual);
}

}  // namespace gufo::models::qwen3_tts::hip

#endif
