#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/models/qwen3_asr/hip/text_ops.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {

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
  for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }
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
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(dimension)) + epsilon);
  for (std::size_t index = thread; index < dimension; index += blockDim.x) {
    const float value = static_cast<float>(hip_bfloat16(row[index]));
    const float normalized =
        static_cast<float>(hip_bfloat16(value * inverse_rms));
    output_row[index] = hip_bfloat16(normalized * weight[index]);
  }
}

__global__ void PackQkvKernel(
    const float* query, const float* key, const float* value,
    hip_bfloat16* query_bfloat16, hip_bfloat16* key_cache_bfloat16,
    hip_bfloat16* value_cache_bfloat16, std::size_t batch_size,
    std::uint32_t layer_index, std::uint32_t start_position,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_key_value_heads, std::uint32_t head_dim) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t query_elements = batch_size * num_heads * head_dim;
  if (index < query_elements) {
    const std::size_t dimension = index % head_dim;
    const std::size_t head_token = index / head_dim;
    const std::size_t head = head_token % num_heads;
    const std::size_t token = head_token / num_heads;
    const std::size_t packed =
        (head * batch_size + token) * head_dim + dimension;
    query_bfloat16[packed] = hip_bfloat16(query[index]);
  }

  const std::size_t key_value_elements =
      batch_size * num_key_value_heads * head_dim;
  if (index < key_value_elements) {
    const std::size_t dimension = index % head_dim;
    const std::size_t head_token = index / head_dim;
    const std::size_t head = head_token % num_key_value_heads;
    const std::size_t token = head_token / num_key_value_heads;
    const std::size_t cache =
        ((((static_cast<std::size_t>(layer_index) * num_key_value_heads) +
           head) *
              max_context +
          start_position + token) *
         head_dim) +
        dimension;
    key_cache_bfloat16[cache] = hip_bfloat16(key[index]);
    value_cache_bfloat16[cache] = hip_bfloat16(value[index]);
  }
}

__launch_bounds__(256) __global__
    void CausalSoftmaxKernel(hip_bfloat16* scores, float* probabilities,
                             std::uint32_t start_position,
                             std::size_t batch_size,
                             std::uint32_t context_length,
                             std::uint32_t num_heads, float scale) {
  const std::size_t row = blockIdx.x;
  const std::size_t rows = static_cast<std::size_t>(num_heads) * batch_size;
  if (row >= rows) {
    return;
  }
  const std::size_t token = row % batch_size;
  const std::size_t visible =
      static_cast<std::size_t>(start_position) + token + 1U;
  hip_bfloat16* row_scores = scores + (row * context_length);
  float* row_probabilities = probabilities + (row * context_length);
  float local_maximum = -INFINITY;
  for (std::size_t key = threadIdx.x; key < visible; key += blockDim.x) {
    const float scaled = static_cast<float>(
        hip_bfloat16(static_cast<float>(row_scores[key]) * scale));
    row_scores[key] = hip_bfloat16(scaled);
    local_maximum = fmaxf(local_maximum, scaled);
  }
  __shared__ float reduction[256];
  reduction[threadIdx.x] = local_maximum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_sum = 0.0F;
  for (std::size_t key = threadIdx.x; key < visible; key += blockDim.x) {
    const float probability =
        expf(static_cast<float>(row_scores[key]) - maximum);
    row_probabilities[key] = probability;
    local_sum += probability;
  }
  reduction[threadIdx.x] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float sum = reduction[0];
  for (std::size_t key = threadIdx.x; key < visible; key += blockDim.x) {
    row_scores[key] = hip_bfloat16(row_probabilities[key] / sum);
  }
  for (std::size_t key = visible + threadIdx.x; key < context_length;
       key += blockDim.x) {
    row_scores[key] = hip_bfloat16(0.0F);
  }
}

__global__ void UnpackAttentionKernel(const hip_bfloat16* input,
                                      hip_bfloat16* output,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t elements = batch_size * num_heads * head_dim;
  if (index >= elements) {
    return;
  }
  const std::size_t dimension = index % head_dim;
  const std::size_t head_token = index / head_dim;
  const std::size_t head = head_token % num_heads;
  const std::size_t token = head_token / num_heads;
  const std::size_t packed = (head * batch_size + token) * head_dim + dimension;
  output[index] = input[packed];
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
  for (std::size_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      shared_sum[thread] += shared_sum[thread + stride];
    }
    __syncthreads();
  }
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

__global__ void TextArgmaxKernel(const float* logits,
                                 std::uint32_t* output_token, std::size_t count,
                                 float tie_tolerance) {
  const std::uint32_t thread = threadIdx.x;
  float maximum = -INFINITY;
  for (std::size_t index = thread; index < count; index += blockDim.x) {
    maximum = fmaxf(maximum, logits[index]);
  }
  __shared__ float maxima[256];
  __shared__ std::uint32_t tokens[256];
  maxima[thread] = maximum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      maxima[thread] = fmaxf(maxima[thread], maxima[thread + stride]);
    }
    __syncthreads();
  }
  const float threshold = maxima[0] - tie_tolerance;
  std::uint32_t token = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t index = thread; index < count; index += blockDim.x) {
    if (logits[index] >= threshold) {
      token = static_cast<std::uint32_t>(index);
      break;
    }
  }
  tokens[thread] = token;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (thread < stride) {
      tokens[thread] = min(tokens[thread], tokens[thread + stride]);
    }
    __syncthreads();
  }
  if (thread == 0U) {
    *output_token = tokens[0];
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

void LaunchTextPackQkv(const float* query, const float* key, const float* value,
                       void* query_bfloat16, void* key_cache_bfloat16,
                       void* value_cache_bfloat16, std::uint32_t layer_index,
                       std::uint32_t start_position, std::size_t batch_size,
                       std::uint32_t max_context, std::uint32_t num_heads,
                       std::uint32_t num_key_value_heads,
                       std::uint32_t head_dim, hipStream_t stream) {
  const std::size_t elements = batch_size * num_heads * head_dim;
  if (elements == 0U) {
    return;
  }
  PackQkvKernel<<<Grid(elements), 256, 0, stream>>>(
      query, key, value, static_cast<hip_bfloat16*>(query_bfloat16),
      static_cast<hip_bfloat16*>(key_cache_bfloat16),
      static_cast<hip_bfloat16*>(value_cache_bfloat16), batch_size, layer_index,
      start_position, max_context, num_heads, num_key_value_heads, head_dim);
}

void LaunchTextCausalSoftmax(void* scores_bfloat16, float* probabilities,
                             std::uint32_t start_position,
                             std::size_t batch_size,
                             std::uint32_t context_length,
                             std::uint32_t num_heads, float scale,
                             hipStream_t stream) {
  const std::size_t rows = static_cast<std::size_t>(num_heads) * batch_size;
  if (rows == 0U || context_length == 0U) {
    return;
  }
  CausalSoftmaxKernel<<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
      static_cast<hip_bfloat16*>(scores_bfloat16), probabilities,
      start_position, batch_size, context_length, num_heads, scale);
}

void LaunchTextUnpackAttention(const void* input_bfloat16,
                               void* output_bfloat16, std::size_t batch_size,
                               std::uint32_t num_heads, std::uint32_t head_dim,
                               hipStream_t stream) {
  const std::size_t elements = batch_size * num_heads * head_dim;
  if (elements == 0U) {
    return;
  }
  UnpackAttentionKernel<<<Grid(elements), 256, 0, stream>>>(
      static_cast<const hip_bfloat16*>(input_bfloat16),
      static_cast<hip_bfloat16*>(output_bfloat16), batch_size, num_heads,
      head_dim);
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

void LaunchTextArgmax(const float* logits, std::uint32_t* output_token,
                      std::size_t count, float tie_tolerance,
                      hipStream_t stream) {
  if (count != 0U) {
    TextArgmaxKernel<<<1, 256, 0, stream>>>(logits, output_token, count,
                                            tie_tolerance);
  }
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
