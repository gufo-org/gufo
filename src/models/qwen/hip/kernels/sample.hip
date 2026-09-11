#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hipcub/hipcub.hpp>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kSamplingThreads = 256;

__device__ float SparseDraftProbability(
    std::uint32_t token,
    const std::uint32_t* __restrict__ draft_candidate_ids,
    const float* __restrict__ draft_candidate_probabilities,
    std::size_t draft_candidate_count) {
  float probability = 0.0F;
  for (std::size_t index = 0; index < draft_candidate_count; ++index) {
    if (draft_candidate_ids[index] == token) {
      probability += draft_candidate_probabilities[index];
    }
  }
  return probability;
}

__global__ void PrepareSamplingKernel(const float* __restrict__ logits,
                                      float* __restrict__ adjusted_logits,
                                      std::uint32_t* __restrict__ token_ids,
                                      std::size_t vocab_size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= vocab_size) {
    return;
  }
  adjusted_logits[index] = logits[index];
  token_ids[index] = static_cast<std::uint32_t>(index);
}

__global__ void ApplySparsePenaltiesKernel(
    float* __restrict__ logits,
    const std::uint32_t* __restrict__ penalty_tokens,
    const std::uint32_t* __restrict__ penalty_counts, std::size_t penalty_count,
    std::size_t vocab_size, float repeat_penalty, float frequency_penalty,
    float presence_penalty) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= penalty_count) {
    return;
  }
  const std::uint32_t token = penalty_tokens[index];
  if (token >= vocab_size) {
    return;
  }
  float adjusted = logits[token];
  if (repeat_penalty != 1.0F) {
    adjusted = adjusted <= 0.0F ? adjusted * repeat_penalty
                                : adjusted / repeat_penalty;
  }
  adjusted -= frequency_penalty * static_cast<float>(penalty_counts[index]);
  adjusted -= presence_penalty;
  logits[token] = adjusted;
}

__global__ void LinearSamplingKernel(const float* __restrict__ logits,
                                     std::uint32_t* __restrict__ out_token,
                                     std::size_t vocab_size, float temperature,
                                     float min_p, float uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t begin = (vocab_size * tid) / blockDim.x;
  const std::size_t end = (vocab_size * (tid + 1U)) / blockDim.x;

  __shared__ float shared_max[kSamplingThreads];
  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ std::uint32_t selected_segment;
  __shared__ float selected_threshold;

  float local_max = -INFINITY;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = logits[index];
    if (isfinite(value)) {
      local_max = fmaxf(local_max, value);
    }
  }
  shared_max[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0; stride /= 2U) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  const float maximum = shared_max[0];
  const float min_p_threshold =
      min_p > 0.0F ? maximum + logf(min_p) : -INFINITY;
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = logits[index];
    if (isfinite(value) && value >= min_p_threshold) {
      local_sum += expf((value - maximum) * inverse_temperature);
    }
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    float threshold = fminf(fmaxf(uniform, 0.0F), 0.99999994F) * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    float threshold = selected_threshold;
    std::uint32_t selected =
        static_cast<std::uint32_t>(end > begin ? end - 1U : 0U);
    for (std::size_t index = begin; index < end; ++index) {
      const float value = logits[index];
      if (!isfinite(value) || value < min_p_threshold) {
        continue;
      }
      threshold -= expf((value - maximum) * inverse_temperature);
      selected = static_cast<std::uint32_t>(index);
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}

__global__ void SortedSamplingKernel(
    const float* __restrict__ sorted_logits,
    const std::uint32_t* __restrict__ sorted_token_ids,
    std::uint32_t* __restrict__ out_token, std::size_t vocab_size,
    float temperature, std::int32_t top_k, float top_p, float min_p,
    std::size_t min_keep, float uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t minimum = min(vocab_size, max(min_keep, std::size_t{1}));
  const std::size_t initial_count =
      top_k > 0 ? min(vocab_size, max(static_cast<std::size_t>(top_k), minimum))
                : vocab_size;

  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ std::size_t top_p_keep;
  __shared__ std::size_t final_keep;
  __shared__ std::uint32_t selected_segment;
  __shared__ float selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
    local_sum += expf(sorted_logits[index] - maximum);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    top_p_keep = initial_count;
    if (top_p < 1.0F) {
      const float target = top_p * total;
      selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        if (target <= segment_prefix[segment] + segment_sums[segment]) {
          selected_segment = static_cast<std::uint32_t>(segment);
          break;
        }
      }
      selected_threshold = target - segment_prefix[selected_segment];
      top_p_keep = 0;
    }
  }
  __syncthreads();

  if (top_p < 1.0F && tid == selected_segment) {
    float threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf(sorted_logits[index] - maximum);
      if (threshold < 0.0F) {
        keep = index + 1U;
        break;
      }
    }
    top_p_keep = max(keep, minimum);
  }
  __syncthreads();

  if (tid == 0) {
    final_keep = top_p_keep;
    if (min_p > 0.0F && final_keep > 1U) {
      const float threshold = maximum + logf(min_p);
      std::size_t low = 0;
      std::size_t high = final_keep;
      while (low < high) {
        const std::size_t middle = low + ((high - low) / 2U);
        if (sorted_logits[middle] >= threshold) {
          low = middle + 1U;
        } else {
          high = middle;
        }
      }
      final_keep = min(final_keep, max(low, minimum));
    }
    if (temperature <= 0.0F || final_keep == 1U) {
      *out_token = sorted_token_ids[0];
    }
  }
  __syncthreads();
  if (temperature <= 0.0F || final_keep == 1U) {
    return;
  }

  const std::size_t sample_begin = (final_keep * tid) / blockDim.x;
  const std::size_t sample_end = (final_keep * (tid + 1U)) / blockDim.x;
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  local_sum = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    local_sum += expf((sorted_logits[index] - maximum) * inverse_temperature);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    const float threshold = fminf(fmaxf(uniform, 0.0F), 0.99999994F) * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    float threshold = selected_threshold;
    std::uint32_t selected = sorted_token_ids[sample_end - 1U];
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      selected = sorted_token_ids[index];
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}

__global__ void LinearSpeculativeSamplingKernel(
    const float* __restrict__ logits,
    std::uint32_t* __restrict__ out_token,
    std::uint32_t* __restrict__ out_accepted, std::size_t vocab_size,
    float temperature, float min_p, std::uint32_t draft_token,
    float draft_token_probability,
    const std::uint32_t* __restrict__ draft_candidate_ids,
    const float* __restrict__ draft_candidate_probabilities,
    std::size_t draft_candidate_count, float acceptance_uniform,
    float residual_uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t begin = (vocab_size * tid) / blockDim.x;
  const std::size_t end = (vocab_size * (tid + 1U)) / blockDim.x;

  __shared__ float shared_max[kSamplingThreads];
  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ float target_total;
  __shared__ float draft_target_probability;
  __shared__ bool accepted;
  __shared__ bool use_residual;
  __shared__ std::uint32_t selected_segment;
  __shared__ float selected_threshold;

  float local_max = -INFINITY;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = logits[index];
    if (isfinite(value)) {
      local_max = fmaxf(local_max, value);
    }
  }
  shared_max[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0; stride /= 2U) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  const float maximum = shared_max[0];
  const float min_p_threshold =
      min_p > 0.0F ? maximum + logf(min_p) : -INFINITY;
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  float local_draft_weight = 0.0F;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = logits[index];
    if (!isfinite(value) || value < min_p_threshold) {
      continue;
    }
    const float weight = expf((value - maximum) * inverse_temperature);
    local_sum += weight;
    if (index == draft_token) {
      local_draft_weight = weight;
    }
  }
  segment_sums[tid] = local_sum;
  shared_max[tid] = local_draft_weight;
  __syncthreads();

  if (tid == 0) {
    target_total = 0.0F;
    draft_target_probability = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      target_total += segment_sums[segment];
      draft_target_probability += shared_max[segment];
    }
    draft_target_probability /= target_total;
    accepted =
        acceptance_uniform * draft_token_probability <
        draft_target_probability;
    *out_accepted = accepted ? 1U : 0U;
    if (accepted) {
      *out_token = draft_token;
    }
  }
  __syncthreads();
  if (accepted) {
    return;
  }

  local_sum = 0.0F;
  for (std::size_t index = begin; index < end; ++index) {
    const float value = logits[index];
    if (!isfinite(value) || value < min_p_threshold) {
      continue;
    }
    const float target_weight =
        expf((value - maximum) * inverse_temperature);
    const float draft_probability = SparseDraftProbability(
        static_cast<std::uint32_t>(index), draft_candidate_ids,
        draft_candidate_probabilities, draft_candidate_count);
    local_sum += fmaxf(target_weight - draft_probability * target_total, 0.0F);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    use_residual = total > 0.0F && isfinite(total);
  }
  __syncthreads();

  if (!use_residual) {
    local_sum = 0.0F;
    for (std::size_t index = begin; index < end; ++index) {
      const float value = logits[index];
      if (isfinite(value) && value >= min_p_threshold) {
        local_sum += expf((value - maximum) * inverse_temperature);
      }
    }
    segment_sums[tid] = local_sum;
    __syncthreads();
    if (tid == 0) {
      float total = 0.0F;
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        segment_prefix[segment] = total;
        total += segment_sums[segment];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float total =
        segment_prefix[blockDim.x - 1U] + segment_sums[blockDim.x - 1U];
    const float threshold =
        fminf(fmaxf(residual_uniform, 0.0F), 0.99999994F) * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (segment_sums[segment] > 0.0F &&
          threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    float threshold = selected_threshold;
    std::uint32_t selected =
        static_cast<std::uint32_t>(end > begin ? end - 1U : 0U);
    for (std::size_t index = begin; index < end; ++index) {
      const float value = logits[index];
      if (!isfinite(value) || value < min_p_threshold) {
        continue;
      }
      const float target_weight =
          expf((value - maximum) * inverse_temperature);
      float weight = target_weight;
      if (use_residual) {
        const float draft_probability = SparseDraftProbability(
            static_cast<std::uint32_t>(index), draft_candidate_ids,
            draft_candidate_probabilities, draft_candidate_count);
        weight =
            fmaxf(target_weight - draft_probability * target_total, 0.0F);
      }
      if (weight <= 0.0F) {
        continue;
      }
      threshold -= weight;
      selected = static_cast<std::uint32_t>(index);
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}

__global__ void SortedSpeculativeSamplingKernel(
    const float* __restrict__ sorted_logits,
    const std::uint32_t* __restrict__ sorted_token_ids,
    std::uint32_t* __restrict__ out_token,
    std::uint32_t* __restrict__ out_accepted, std::size_t vocab_size,
    float temperature, std::int32_t top_k, float top_p, float min_p,
    std::size_t min_keep, std::uint32_t draft_token,
    float draft_token_probability,
    const std::uint32_t* __restrict__ draft_candidate_ids,
    const float* __restrict__ draft_candidate_probabilities,
    std::size_t draft_candidate_count, float acceptance_uniform,
    float residual_uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t minimum = min(vocab_size, max(min_keep, std::size_t{1}));
  const std::size_t initial_count =
      top_k > 0 ? min(vocab_size, max(static_cast<std::size_t>(top_k), minimum))
                : vocab_size;

  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ float draft_weights[kSamplingThreads];
  __shared__ std::size_t top_p_keep;
  __shared__ std::size_t final_keep;
  __shared__ float target_total;
  __shared__ bool accepted;
  __shared__ bool use_residual;
  __shared__ std::uint32_t selected_segment;
  __shared__ float selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
    local_sum += expf(sorted_logits[index] - maximum);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    top_p_keep = initial_count;
    if (top_p < 1.0F) {
      const float target = top_p * total;
      selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        if (target <= segment_prefix[segment] + segment_sums[segment]) {
          selected_segment = static_cast<std::uint32_t>(segment);
          break;
        }
      }
      selected_threshold = target - segment_prefix[selected_segment];
      top_p_keep = 0;
    }
  }
  __syncthreads();

  if (top_p < 1.0F && tid == selected_segment) {
    float threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf(sorted_logits[index] - maximum);
      if (threshold < 0.0F) {
        keep = index + 1U;
        break;
      }
    }
    top_p_keep = max(keep, minimum);
  }
  __syncthreads();

  if (tid == 0) {
    final_keep = top_p_keep;
    if (min_p > 0.0F && final_keep > 1U) {
      const float threshold = maximum + logf(min_p);
      std::size_t low = 0;
      std::size_t high = final_keep;
      while (low < high) {
        const std::size_t middle = low + ((high - low) / 2U);
        if (sorted_logits[middle] >= threshold) {
          low = middle + 1U;
        } else {
          high = middle;
        }
      }
      final_keep = min(final_keep, max(low, minimum));
    }
  }
  __syncthreads();

  const std::size_t sample_begin = (final_keep * tid) / blockDim.x;
  const std::size_t sample_end = (final_keep * (tid + 1U)) / blockDim.x;
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  local_sum = 0.0F;
  float local_draft_weight = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    const float weight =
        expf((sorted_logits[index] - maximum) * inverse_temperature);
    local_sum += weight;
    if (sorted_token_ids[index] == draft_token) {
      local_draft_weight = weight;
    }
  }
  segment_sums[tid] = local_sum;
  draft_weights[tid] = local_draft_weight;
  __syncthreads();

  if (tid == 0) {
    target_total = 0.0F;
    float draft_target_weight = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      target_total += segment_sums[segment];
      draft_target_weight += draft_weights[segment];
    }
    const float draft_target_probability = draft_target_weight / target_total;
    accepted =
        acceptance_uniform * draft_token_probability <
        draft_target_probability;
    *out_accepted = accepted ? 1U : 0U;
    if (accepted) {
      *out_token = draft_token;
    }
  }
  __syncthreads();
  if (accepted) {
    return;
  }

  local_sum = 0.0F;
  for (std::size_t index = sample_begin; index < sample_end; ++index) {
    const float target_weight =
        expf((sorted_logits[index] - maximum) * inverse_temperature);
    const float draft_probability = SparseDraftProbability(
        sorted_token_ids[index], draft_candidate_ids,
        draft_candidate_probabilities, draft_candidate_count);
    local_sum += fmaxf(target_weight - draft_probability * target_total, 0.0F);
  }
  segment_sums[tid] = local_sum;
  __syncthreads();

  if (tid == 0) {
    float total = 0.0F;
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      segment_prefix[segment] = total;
      total += segment_sums[segment];
    }
    use_residual = total > 0.0F && isfinite(total);
  }
  __syncthreads();

  if (!use_residual) {
    segment_sums[tid] = 0.0F;
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      segment_sums[tid] +=
          expf((sorted_logits[index] - maximum) * inverse_temperature);
    }
    __syncthreads();
    if (tid == 0) {
      float total = 0.0F;
      for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
        segment_prefix[segment] = total;
        total += segment_sums[segment];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float total =
        segment_prefix[blockDim.x - 1U] + segment_sums[blockDim.x - 1U];
    const float threshold =
        fminf(fmaxf(residual_uniform, 0.0F), 0.99999994F) * total;
    selected_segment = static_cast<std::uint32_t>(blockDim.x - 1U);
    for (std::size_t segment = 0; segment < blockDim.x; ++segment) {
      if (segment_sums[segment] > 0.0F &&
          threshold < segment_prefix[segment] + segment_sums[segment]) {
        selected_segment = static_cast<std::uint32_t>(segment);
        break;
      }
    }
    selected_threshold = threshold - segment_prefix[selected_segment];
  }
  __syncthreads();

  if (tid == selected_segment) {
    float threshold = selected_threshold;
    std::uint32_t selected = sorted_token_ids[sample_end - 1U];
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      const float target_weight =
          expf((sorted_logits[index] - maximum) * inverse_temperature);
      float weight = target_weight;
      if (use_residual) {
        const float draft_probability = SparseDraftProbability(
            sorted_token_ids[index], draft_candidate_ids,
            draft_candidate_probabilities, draft_candidate_count);
        weight =
            fmaxf(target_weight - draft_probability * target_total, 0.0F);
      }
      if (weight <= 0.0F) {
        continue;
      }
      threshold -= weight;
      selected = sorted_token_ids[index];
      if (threshold < 0.0F) {
        break;
      }
    }
    *out_token = selected;
  }
}

}  // namespace

__global__ void ArgmaxKernel(const float* __restrict__ logits,
                             std::uint32_t* __restrict__ out_token,
                             std::size_t vocab_size) {
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;

  float max_val = -1e30F;
  std::uint32_t max_idx = 0;

  for (std::size_t i = tid; i < vocab_size; i += block_size) {
    const float val = logits[i];
    if (val > max_val) {
      max_val = val;
      max_idx = static_cast<std::uint32_t>(i);
    }
  }

  __shared__ float s_max[256];
  __shared__ std::uint32_t s_idx[256];
  s_max[tid] = max_val;
  s_idx[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      if (s_max[tid + stride] > s_max[tid]) {
        s_max[tid] = s_max[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out_token = s_idx[0];
  }
}

void LaunchGPUArgmax(const float* logits, std::uint32_t* out_token,
                     std::size_t vocab_size, hipStream_t stream) {
  ArgmaxKernel<<<1, 256, 0, stream>>>(logits, out_token, vocab_size);
}

__global__ void BatchedArgmaxKernel(const float* __restrict__ logits,
                                    std::uint32_t* __restrict__ out_tokens,
                                    std::size_t vocab_size) {
  const std::size_t row = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const float* row_logits = logits + (row * vocab_size);

  float max_val = -1e30F;
  std::uint32_t max_idx = 0;
  for (std::size_t index = tid; index < vocab_size; index += blockDim.x) {
    const float value = row_logits[index];
    if (value > max_val) {
      max_val = value;
      max_idx = static_cast<std::uint32_t>(index);
    }
  }

  __shared__ float shared_max[256];
  __shared__ std::uint32_t shared_index[256];
  shared_max[tid] = max_val;
  shared_index[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
      shared_max[tid] = shared_max[tid + stride];
      shared_index[tid] = shared_index[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_tokens[row] = shared_index[0];
  }
}

void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            hipStream_t stream) {
  if (batch_size == 0 || vocab_size == 0) {
    return;
  }
  BatchedArgmaxKernel<<<static_cast<unsigned int>(batch_size), 256, 0,
                        stream>>>(logits, out_tokens, vocab_size);
}

void AllocateGpuSamplingWorkspace(GpuSamplingWorkspace* workspace,
                                  std::size_t vocab_size,
                                  std::size_t penalty_capacity) {
  if (workspace == nullptr || vocab_size == 0) {
    throw std::invalid_argument("invalid GPU sampling workspace request");
  }
  FreeGpuSamplingWorkspace(workspace);
  workspace->vocab_size = vocab_size;
  workspace->penalty_capacity = penalty_capacity;
  try {
    HIP_CHECK(
        hipMalloc(&workspace->adjusted_logits, vocab_size * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&workspace->sorted_logits, vocab_size * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&workspace->token_ids, vocab_size * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&workspace->sorted_token_ids,
                        vocab_size * sizeof(std::uint32_t)));
    if (penalty_capacity > 0) {
      HIP_CHECK(hipMalloc(&workspace->penalty_tokens,
                          penalty_capacity * sizeof(std::uint32_t)));
      HIP_CHECK(hipMalloc(&workspace->penalty_counts,
                          penalty_capacity * sizeof(std::uint32_t)));
      HIP_CHECK(hipMalloc(&workspace->draft_candidate_ids,
                          penalty_capacity * sizeof(std::uint32_t)));
      HIP_CHECK(hipMalloc(&workspace->draft_candidate_probabilities,
                          penalty_capacity * sizeof(float)));
    }
    HIP_CHECK(
        hipMalloc(&workspace->speculative_accepted, sizeof(std::uint32_t)));
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
        nullptr, workspace->sort_temp_storage_bytes,
        workspace->adjusted_logits, workspace->sorted_logits,
        workspace->token_ids, workspace->sorted_token_ids, vocab_size));
    HIP_CHECK(hipMalloc(&workspace->sort_temp_storage,
                        workspace->sort_temp_storage_bytes));
  } catch (...) {
    FreeGpuSamplingWorkspace(workspace);
    throw;
  }
}

void FreeGpuSamplingWorkspace(GpuSamplingWorkspace* workspace) noexcept {
  if (workspace == nullptr) {
    return;
  }
  const auto free_buffer = [](auto*& pointer) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
      pointer = nullptr;
    }
  };
  free_buffer(workspace->adjusted_logits);
  free_buffer(workspace->sorted_logits);
  free_buffer(workspace->token_ids);
  free_buffer(workspace->sorted_token_ids);
  free_buffer(workspace->penalty_tokens);
  free_buffer(workspace->penalty_counts);
  free_buffer(workspace->draft_candidate_ids);
  free_buffer(workspace->draft_candidate_probabilities);
  free_buffer(workspace->speculative_accepted);
  free_buffer(workspace->sort_temp_storage);
  workspace->sort_temp_storage_bytes = 0;
  workspace->vocab_size = 0;
  workspace->penalty_capacity = 0;
}

void LaunchGPUSampling(const float* logits, std::uint32_t* out_token,
                       std::size_t vocab_size,
                       const GpuSamplingParameters& parameters,
                       const std::uint32_t* penalty_tokens,
                       const std::uint32_t* penalty_counts,
                       std::size_t penalty_count,
                       GpuSamplingWorkspace* workspace, hipStream_t stream) {
  if (logits == nullptr || out_token == nullptr || workspace == nullptr ||
      vocab_size == 0 || workspace->vocab_size != vocab_size ||
      penalty_count > workspace->penalty_capacity) {
    throw std::invalid_argument("invalid GPU sampling launch");
  }

  const bool has_penalties = penalty_count > 0;
  const bool needs_ordering =
      parameters.top_k > 0 || parameters.top_p < 1.0F ||
      (parameters.min_p > 0.0F && parameters.min_keep > 1);
  const float* sampling_logits = logits;
  if (has_penalties || needs_ordering) {
    const std::size_t blocks =
        (vocab_size + kSamplingThreads - 1U) / kSamplingThreads;
    PrepareSamplingKernel<<<static_cast<unsigned int>(blocks), kSamplingThreads,
                            0, stream>>>(logits, workspace->adjusted_logits,
                                         workspace->token_ids, vocab_size);
    sampling_logits = workspace->adjusted_logits;
  }
  if (has_penalties) {
    HIP_CHECK(hipMemcpyAsync(workspace->penalty_tokens, penalty_tokens,
                             penalty_count * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(workspace->penalty_counts, penalty_counts,
                             penalty_count * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream));
    const std::size_t blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalty_tokens,
        workspace->penalty_counts, penalty_count, vocab_size,
        parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }

  if (parameters.temperature <= 0.0F) {
    LaunchGPUArgmax(sampling_logits, out_token, vocab_size, stream);
    return;
  }
  if (parameters.top_k == 1) {
    LaunchGPUArgmax(sampling_logits, out_token, vocab_size, stream);
    return;
  }
  if (!needs_ordering) {
    LinearSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
        sampling_logits, out_token, vocab_size, parameters.temperature,
        parameters.min_p, parameters.uniform);
    return;
  }

  HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
      workspace->sort_temp_storage, workspace->sort_temp_storage_bytes,
      workspace->adjusted_logits, workspace->sorted_logits,
      workspace->token_ids, workspace->sorted_token_ids, vocab_size, 0,
      static_cast<int>(sizeof(float) * 8U), stream));
  SortedSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      vocab_size, parameters.temperature, parameters.top_k, parameters.top_p,
      parameters.min_p, parameters.min_keep, parameters.uniform);
}

void LaunchGPUSpeculativeSampling(
    const float* logits, std::uint32_t* out_token,
    std::uint32_t* out_accepted, std::size_t vocab_size,
    const GpuSamplingParameters& parameters, std::uint32_t draft_token,
    float draft_token_probability,
    const std::uint32_t* draft_candidate_ids,
    const float* draft_candidate_probabilities,
    std::size_t draft_candidate_count, float acceptance_uniform,
    float residual_uniform, const std::uint32_t* penalty_tokens,
    const std::uint32_t* penalty_counts, std::size_t penalty_count,
    GpuSamplingWorkspace* workspace, hipStream_t stream) {
  if (logits == nullptr || out_token == nullptr || out_accepted == nullptr ||
      workspace == nullptr || vocab_size == 0 ||
      workspace->vocab_size != vocab_size ||
      draft_candidate_count == 0 ||
      draft_candidate_count > workspace->penalty_capacity ||
      penalty_count > workspace->penalty_capacity ||
      draft_candidate_ids == nullptr ||
      draft_candidate_probabilities == nullptr ||
      !std::isfinite(draft_token_probability) ||
      draft_token_probability <= 0.0F ||
      parameters.temperature <= 0.0F || parameters.top_k == 1) {
    throw std::invalid_argument("invalid GPU speculative sampling launch");
  }

  HIP_CHECK(hipMemcpyAsync(
      workspace->draft_candidate_ids, draft_candidate_ids,
      draft_candidate_count * sizeof(std::uint32_t), hipMemcpyHostToDevice,
      stream));
  HIP_CHECK(hipMemcpyAsync(
      workspace->draft_candidate_probabilities,
      draft_candidate_probabilities, draft_candidate_count * sizeof(float),
      hipMemcpyHostToDevice, stream));

  const bool has_penalties = penalty_count > 0;
  const bool needs_ordering =
      parameters.top_k > 0 || parameters.top_p < 1.0F ||
      (parameters.min_p > 0.0F && parameters.min_keep > 1);
  const float* sampling_logits = logits;
  if (has_penalties || needs_ordering) {
    const std::size_t blocks =
        (vocab_size + kSamplingThreads - 1U) / kSamplingThreads;
    PrepareSamplingKernel<<<static_cast<unsigned int>(blocks), kSamplingThreads,
                            0, stream>>>(logits, workspace->adjusted_logits,
                                         workspace->token_ids, vocab_size);
    sampling_logits = workspace->adjusted_logits;
  }
  if (has_penalties) {
    HIP_CHECK(hipMemcpyAsync(workspace->penalty_tokens, penalty_tokens,
                             penalty_count * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(workspace->penalty_counts, penalty_counts,
                             penalty_count * sizeof(std::uint32_t),
                             hipMemcpyHostToDevice, stream));
    const std::size_t blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalty_tokens,
        workspace->penalty_counts, penalty_count, vocab_size,
        parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }

  if (!needs_ordering) {
    LinearSpeculativeSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
        sampling_logits, out_token, out_accepted, vocab_size,
        parameters.temperature, parameters.min_p, draft_token,
        draft_token_probability, workspace->draft_candidate_ids,
        workspace->draft_candidate_probabilities, draft_candidate_count,
        acceptance_uniform, residual_uniform);
    return;
  }

  HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
      workspace->sort_temp_storage, workspace->sort_temp_storage_bytes,
      workspace->adjusted_logits, workspace->sorted_logits,
      workspace->token_ids, workspace->sorted_token_ids, vocab_size, 0,
      static_cast<int>(sizeof(float) * 8U), stream));
  SortedSpeculativeSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      out_accepted, vocab_size, parameters.temperature, parameters.top_k,
      parameters.top_p, parameters.min_p, parameters.min_keep, draft_token,
      draft_token_probability, workspace->draft_candidate_ids,
      workspace->draft_candidate_probabilities, draft_candidate_count,
      acceptance_uniform, residual_uniform);
}

}  // namespace gufo::hip
#endif
