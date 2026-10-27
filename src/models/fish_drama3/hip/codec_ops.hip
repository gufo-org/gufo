#include "src/models/fish_drama3/hip/codec_ops.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::models::fish_drama3::hip {
namespace {

constexpr std::size_t kThreads = 256;
constexpr std::size_t kCodebookDim = 8;

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

__global__ void AddKernel(const float* left, const float* right, float* output,
                          std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = left[index] + right[index];
  }
}

__global__ void SubInPlaceKernel(float* target, const float* subtrahend,
                                 std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    target[index] -= subtrahend[index];
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

__global__ void InterleavedRopeKernel(float* query, float* key,
                                      std::size_t frames, std::size_t heads,
                                      std::size_t head_dimension,
                                      float rope_theta,
                                      std::size_t start_position) {
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
  const std::size_t base = (frame * heads + head) * head_dimension + 2U * pair;
  const float q0 = query[base];
  const float q1 = query[base + 1U];
  const float k0 = key[base];
  const float k1 = key[base + 1U];
  query[base] = q0 * cosine - q1 * sine;
  query[base + 1U] = q1 * cosine + q0 * sine;
  key[base] = k0 * cosine - k1 * sine;
  key[base + 1U] = k1 * cosine + k0 * sine;
}

__global__ void SlidingCausalAttentionKernel(
    const float* query, const float* key, const float* value, float* output,
    std::size_t key_frames, std::size_t heads, std::size_t head_dimension,
    std::size_t sliding_window, std::size_t query_offset) {
  const std::size_t local_query = blockIdx.x / heads;
  const std::size_t query_frame = local_query + query_offset;
  const std::size_t head = blockIdx.x % heads;
  if (query_frame >= key_frames) {
    return;
  }
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduction = shared + key_frames;
  const std::size_t first_key =
      (sliding_window > 0 && query_frame + 1 > sliding_window)
          ? query_frame + 1 - sliding_window
          : 0;
  const float scale = rsqrtf(static_cast<float>(head_dimension));
  const std::size_t query_base = (local_query * heads + head) * head_dimension;

  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t key_frame = threadIdx.x; key_frame < key_frames;
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
  for (std::size_t key_frame = threadIdx.x; key_frame < key_frames;
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
    output[index] = (value / (1.0F + expf(-value))) * up[index];
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

__global__ void Snake1dKernel(const float* input, const float* alpha,
                              float* output, std::size_t elements,
                              std::size_t columns) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t channel = index % columns;
    const float a = alpha[channel];
    const float sine = sinf(input[index] * a);
    output[index] = input[index] + (sine * sine) / (a + 1.0e-9F);
  }
}

__global__ void TanhKernel(float* values, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = tanhf(values[index]);
  }
}

// Coalesced Im2Col with layout [output_length, kernel, input_channels]:
// Adjacent threads in a wavefront read contiguous `channel` values from `input`
// and write contiguous `channel` values to `columns` with zero 64-bit integer
// divisions!
__global__ void CausalConv1dIm2ColKernel(const float* __restrict__ input,
                                         float* __restrict__ columns,
                                         std::size_t input_length,
                                         std::size_t input_channels,
                                         std::size_t output_length,
                                         std::size_t kernel, std::size_t stride,
                                         std::size_t dilation) {
  const std::size_t channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tap = blockIdx.y;
  const std::size_t output_time = blockIdx.z;
  if (channel >= input_channels || tap >= kernel ||
      output_time >= output_length) {
    return;
  }
  const std::int64_t effective_kernel =
      static_cast<std::int64_t>((kernel - 1U) * dilation + 1U);
  const std::int64_t left_pad =
      effective_kernel - static_cast<std::int64_t>(stride);
  const std::int64_t input_time =
      static_cast<std::int64_t>(output_time * stride) - left_pad +
      static_cast<std::int64_t>(tap * dilation);
  const std::size_t out_idx =
      (output_time * kernel + tap) * input_channels + channel;
  columns[out_idx] =
      (input_time < 0 || input_time >= static_cast<std::int64_t>(input_length))
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
  const std::size_t padding = (kernel - 1U) * dilation;
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
  // Layout: [kernel, output_channels, input_channels] so `output_channel` is
  // contiguous!
  const std::size_t target_index =
      (tap * output_channels + output_channel) * input_channels + input_channel;
  target[target_index] = source[index];
}

__global__ void AssembleCausalConvTranspose1dKernel(
    const float* __restrict__ expanded, const float* __restrict__ bias,
    float* __restrict__ output, std::size_t input_length,
    std::size_t output_length, std::size_t output_channels, std::size_t kernel,
    std::size_t stride) {
  const std::size_t output_channel =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_time = blockIdx.y;
  if (output_channel >= output_channels || output_time >= output_length) {
    return;
  }
  float result = bias == nullptr ? 0.0F : bias[output_channel];
  const std::size_t first_tap = output_time % stride;
  for (std::size_t tap = first_tap; tap < kernel; tap += stride) {
    if (output_time < tap) {
      break;
    }
    const std::size_t input_time = (output_time - tap) / stride;
    if (input_time < input_length) {
      const std::size_t expanded_index =
          (input_time * kernel + tap) * output_channels + output_channel;
      result += expanded[expanded_index];
    }
  }
  output[output_time * output_channels + output_channel] = result;
}

__global__ void DequantizeCodebookKernel(
    const std::int32_t* codes_row, std::size_t codebook_size,
    const float* codebook_weight, const float* out_proj_weight,
    const float* out_proj_bias, float* output, std::size_t frames,
    std::size_t latent_dim, bool initialize) {
  const std::size_t elements = frames * latent_dim;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t frame = index / latent_dim;
  const std::size_t col = index % latent_dim;
  std::int32_t raw_code = codes_row[frame];
  if (raw_code < 0) {
    raw_code = 0;
  } else if (static_cast<std::size_t>(raw_code) >= codebook_size) {
    raw_code = static_cast<std::int32_t>(codebook_size - 1U);
  }
  const float* emb =
      codebook_weight + static_cast<std::size_t>(raw_code) * kCodebookDim;
  const float* proj_row = out_proj_weight + col * kCodebookDim;
  float sum = out_proj_bias != nullptr ? out_proj_bias[col] : 0.0F;
#pragma unroll
  for (std::size_t d = 0; d < kCodebookDim; ++d) {
    sum = fmaf(emb[d], proj_row[d], sum);
  }
  output[index] = initialize ? sum : output[index] + sum;
}

__global__ void QuantizeCodebookStepKernel(
    float* residual, const float* in_proj_weight, const float* in_proj_bias,
    const float* codebook_weight, const float* normalized_codebook,
    const float* out_proj_weight, const float* out_proj_bias,
    std::int32_t* codes_row, std::size_t frames, std::size_t latent_dim,
    std::size_t codebook_size) {
  const std::size_t frame = blockIdx.x;
  if (frame >= frames) {
    return;
  }
  __shared__ float partial_proj[kCodebookDim][256];
  __shared__ float z_norm[kCodebookDim];
  __shared__ float best_scores[256];
  __shared__ std::int32_t best_indices[256];

  const float* frame_res = residual + frame * latent_dim;
#pragma unroll
  for (std::size_t d = 0; d < kCodebookDim; ++d) {
    float acc = 0.0F;
    const float* w_row = in_proj_weight + d * latent_dim;
    for (std::size_t c = threadIdx.x; c < latent_dim; c += blockDim.x) {
      acc = fmaf(frame_res[c], w_row[c], acc);
    }
    partial_proj[d][threadIdx.x] = acc;
  }
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
#pragma unroll
      for (std::size_t d = 0; d < kCodebookDim; ++d) {
        partial_proj[d][threadIdx.x] += partial_proj[d][threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float norm_sq = 0.0F;
    for (std::size_t d = 0; d < kCodebookDim; ++d) {
      const float val = partial_proj[d][0] +
                        (in_proj_bias != nullptr ? in_proj_bias[d] : 0.0F);
      z_norm[d] = val;
      norm_sq = fmaf(val, val, norm_sq);
    }
    const float inv_norm = rsqrtf(norm_sq + 1.0e-12F);
    for (std::size_t d = 0; d < kCodebookDim; ++d) {
      z_norm[d] *= inv_norm;
    }
  }
  __syncthreads();

  float local_best_score = -std::numeric_limits<float>::infinity();
  std::int32_t local_best_idx = 0;
  for (std::size_t code = threadIdx.x; code < codebook_size;
       code += blockDim.x) {
    const float* cb_row = normalized_codebook + code * kCodebookDim;
    float dot = 0.0F;
#pragma unroll
    for (std::size_t d = 0; d < kCodebookDim; ++d) {
      dot = fmaf(z_norm[d], cb_row[d], dot);
    }
    if (dot > local_best_score ||
        (dot == local_best_score &&
         static_cast<std::int32_t>(code) < local_best_idx)) {
      local_best_score = dot;
      local_best_idx = static_cast<std::int32_t>(code);
    }
  }
  best_scores[threadIdx.x] = local_best_score;
  best_indices[threadIdx.x] = local_best_idx;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const float other_score = best_scores[threadIdx.x + stride];
      const std::int32_t other_idx = best_indices[threadIdx.x + stride];
      const float cur_score = best_scores[threadIdx.x];
      const std::int32_t cur_idx = best_indices[threadIdx.x];
      if (other_score > cur_score ||
          (other_score == cur_score && other_idx < cur_idx)) {
        best_scores[threadIdx.x] = other_score;
        best_indices[threadIdx.x] = other_idx;
      }
    }
    __syncthreads();
  }

  const std::int32_t chosen_code = best_indices[0];
  if (threadIdx.x == 0) {
    codes_row[frame] = chosen_code;
  }

  const float* chosen_emb =
      codebook_weight + static_cast<std::size_t>(chosen_code) * kCodebookDim;
  for (std::size_t c = threadIdx.x; c < latent_dim; c += blockDim.x) {
    const float* proj_row = out_proj_weight + c * kCodebookDim;
    float q_val = out_proj_bias != nullptr ? out_proj_bias[c] : 0.0F;
#pragma unroll
    for (std::size_t d = 0; d < kCodebookDim; ++d) {
      q_val = fmaf(chosen_emb[d], proj_row[d], q_val);
    }
    residual[frame * latent_dim + c] -= q_val;
  }
}

}  // namespace

void LaunchCodecAdd(const float* left, const float* right, float* output,
                    std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(AddKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, left, right, output, elements);
}

void LaunchCodecSubInPlace(float* target, const float* subtrahend,
                           std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(SubInPlaceKernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, target, subtrahend, elements);
}

void LaunchCodecAddBias(float* values, const float* bias, std::size_t rows,
                        std::size_t columns, hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(AddBiasKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, bias, elements, columns);
}

void LaunchCodecAddScaledChannels(float* residual, const float* update,
                                  const float* scale, std::size_t rows,
                                  std::size_t columns, hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(AddScaledChannelsKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, residual, update, scale,
                     elements, columns);
}

void LaunchCodecRmsNorm(const float* input, const float* weight, float* output,
                        std::size_t rows, std::size_t columns, float epsilon,
                        hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, output, rows, columns, epsilon);
}

void LaunchCodecLayerNorm(const float* input, const float* weight,
                          const float* bias, float* output, std::size_t rows,
                          std::size_t columns, float epsilon,
                          hipStream_t stream) {
  hipLaunchKernelGGL(LayerNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, bias, output, rows, columns, epsilon);
}

void LaunchCodecInterleavedRope(float* query, float* key, std::size_t frames,
                                std::size_t heads, std::size_t head_dimension,
                                float rope_theta, hipStream_t stream,
                                std::size_t start_position) {
  const std::size_t pairs = frames * heads * (head_dimension / 2U);
  hipLaunchKernelGGL(InterleavedRopeKernel, dim3(Blocks(pairs)), dim3(kThreads),
                     0, stream, query, key, frames, heads, head_dimension,
                     rope_theta, start_position);
}

void LaunchCodecSlidingCausalAttention(
    const float* query, const float* key, const float* value, float* output,
    std::size_t frames, std::size_t heads, std::size_t head_dimension,
    std::size_t sliding_window, hipStream_t stream, std::size_t query_offset,
    std::size_t key_frames) {
  if (key_frames == 0) {
    key_frames = frames;
  }
  constexpr std::size_t threads = 128;
  const std::size_t shared_bytes = (key_frames + threads) * sizeof(float);
  hipLaunchKernelGGL(SlidingCausalAttentionKernel, dim3(frames * heads),
                     dim3(threads), shared_bytes, stream, query, key, value,
                     output, key_frames, heads, head_dimension, sliding_window,
                     query_offset);
}

void LaunchCodecSwiGlu(const float* gate, const float* up, float* output,
                       std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(SwiGluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, gate, up, output, elements);
}

void LaunchCodecGelu(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(GeluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchCodecSnake1d(const float* input, const float* alpha, float* output,
                        std::size_t rows, std::size_t columns,
                        hipStream_t stream) {
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(Snake1dKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, input, alpha, output, elements, columns);
}

void LaunchCodecTanh(float* values, std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(TanhKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, values, elements);
}

void LaunchCodecCausalConv1dIm2Col(const float* input, float* columns,
                                   std::size_t input_length,
                                   std::size_t input_channels,
                                   std::size_t output_length,
                                   std::size_t kernel, std::size_t stride,
                                   std::size_t dilation, hipStream_t stream) {
  if (output_length == 0 || input_channels == 0 || kernel == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(Blocks(input_channels)),
                  static_cast<unsigned int>(kernel),
                  static_cast<unsigned int>(output_length));
  hipLaunchKernelGGL(CausalConv1dIm2ColKernel, grid, dim3(kThreads), 0, stream,
                     input, columns, input_length, input_channels,
                     output_length, kernel, stride, dilation);
}

void LaunchCodecCausalDepthwiseConv1d(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      std::size_t length, std::size_t channels,
                                      std::size_t kernel, std::size_t dilation,
                                      hipStream_t stream) {
  const std::size_t elements = length * channels;
  hipLaunchKernelGGL(CausalDepthwiseConv1dKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     length, channels, kernel, dilation);
}

void LaunchCodecTransposeConvWeight(const float* source, float* target,
                                    std::size_t input_channels,
                                    std::size_t output_channels,
                                    std::size_t kernel, hipStream_t stream) {
  const std::size_t elements = input_channels * output_channels * kernel;
  hipLaunchKernelGGL(TransposeConvWeightKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, source, target, input_channels,
                     output_channels, kernel);
}

void LaunchCodecAssembleCausalConvTranspose1d(
    const float* expanded, const float* bias, float* output,
    std::size_t input_length, std::size_t output_length,
    std::size_t output_channels, std::size_t kernel, std::size_t stride,
    hipStream_t stream) {
  if (output_length == 0 || output_channels == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(Blocks(output_channels)),
                  static_cast<unsigned int>(output_length));
  hipLaunchKernelGGL(AssembleCausalConvTranspose1dKernel, grid, dim3(kThreads),
                     0, stream, expanded, bias, output, input_length,
                     output_length, output_channels, kernel, stride);
}

void LaunchCodecDequantizeCodebook(const std::int32_t* codes_row,
                                   std::size_t codebook_size,
                                   const float* codebook_weight,
                                   const float* out_proj_weight,
                                   const float* out_proj_bias, float* output,
                                   std::size_t frames, std::size_t latent_dim,
                                   bool initialize, hipStream_t stream) {
  const std::size_t elements = frames * latent_dim;
  hipLaunchKernelGGL(DequantizeCodebookKernel, dim3(Blocks(elements)),
                     dim3(kThreads), 0, stream, codes_row, codebook_size,
                     codebook_weight, out_proj_weight, out_proj_bias, output,
                     frames, latent_dim, initialize);
}

void LaunchCodecQuantizeCodebookStep(
    float* residual, const float* in_proj_weight, const float* in_proj_bias,
    const float* codebook_weight, const float* normalized_codebook,
    const float* out_proj_weight, const float* out_proj_bias,
    std::int32_t* codes_row, std::size_t frames, std::size_t latent_dim,
    std::size_t codebook_size, hipStream_t stream) {
  hipLaunchKernelGGL(QuantizeCodebookStepKernel, dim3(frames), dim3(kThreads),
                     0, stream, residual, in_proj_weight, in_proj_bias,
                     codebook_weight, normalized_codebook, out_proj_weight,
                     out_proj_bias, codes_row, frames, latent_dim,
                     codebook_size);
}

}  // namespace gufo::models::fish_drama3::hip

#endif
