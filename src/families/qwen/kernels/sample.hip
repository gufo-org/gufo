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
#include "src/families/qwen/kernels/ops/token.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kSamplingThreads = 256;
constexpr std::size_t kArgmaxTile = 4096;
constexpr std::size_t kCandidateTile = 1024;

// As in Flash-Next, a token outside its tile's top Keep cannot enter the
// global top Keep. Reduce bounded tile winners instead of sorting the whole
// vocabulary. The ordered float bits and original ID preserve radix ordering,
// lowest-ID ties and signed-zero equivalence, including after penalties.
template<unsigned Keep>
__global__ void SamplingCandidateTileKernel(const float* logits,
                                            const std::uint32_t* input,
                                            std::uint32_t* output,
                                            float* scores, std::size_t size,
                                            std::size_t vocab_size) {
  using Sort = hipcub::BlockRadixSort<std::uint64_t, kSamplingThreads, 4>;
  __shared__ typename Sort::TempStorage scratch;
  std::uint64_t keys[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const auto index = blockIdx.x * kCandidateTile + threadIdx.x * 4 + j;
    const auto id = index < size ? (input ? input[index] : index) : UINT32_MAX;
    keys[j] = 0;
    if (id < vocab_size) {
      const float value = logits[id];
      const auto bits = __float_as_uint(value == 0.0F ? 0.0F : value);
      const auto ordered = bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
      keys[j] = (static_cast<std::uint64_t>(ordered) << 32) |
                (UINT32_MAX - static_cast<std::uint32_t>(id));
    }
  }
  Sort(scratch).SortDescending(keys);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const auto rank = threadIdx.x * 4 + j;
    if (rank < min(std::size_t{Keep}, size)) {
      const auto id = UINT32_MAX - static_cast<std::uint32_t>(keys[j]);
      output[blockIdx.x * Keep + rank] = id;
      if (scores)
        scores[rank] = id < vocab_size ? logits[id] : -INFINITY;
    }
  }
}

template<unsigned Keep>
void SelectSamplingCandidates(GpuSamplingWorkspace* workspace,
                              hipStream_t stream) {
  const auto tiles = [](std::size_t size) {
    return (size + kCandidateTile - 1) / kCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = workspace->vocab_size; size > kCandidateTile;
       size = tiles(size) * Keep)
    ++passes;
  auto* destination =
      passes % 2 != 0 ? workspace->sorted_token_ids : workspace->token_ids;
  const std::uint32_t* source = nullptr;
  auto size = workspace->vocab_size;
  for (;;) {
    const auto blocks = tiles(size);
    SamplingCandidateTileKernel<Keep>
        <<<static_cast<unsigned>(blocks), kSamplingThreads, 0, stream>>>(
            workspace->adjusted_logits, source, destination,
            blocks == 1 ? workspace->sorted_logits : nullptr, size,
            workspace->vocab_size);
    if (blocks == 1)
      break;
    size = blocks * Keep;
    source = destination;
    destination = destination == workspace->sorted_token_ids
                      ? workspace->token_ids
                      : workspace->sorted_token_ids;
  }
}

void SortSamplingLogits(GpuSamplingWorkspace* workspace, std::size_t keep,
                        hipStream_t stream) {
  if (keep > 0 && keep <= 64 && workspace->vocab_size > kCandidateTile) {
    if (keep <= 32)
      SelectSamplingCandidates<32>(workspace, stream);
    else
      SelectSamplingCandidates<64>(workspace, stream);
    return;
  }
  HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
      workspace->sort_temp_storage, workspace->sort_temp_storage_bytes,
      workspace->adjusted_logits, workspace->sorted_logits,
      workspace->token_ids, workspace->sorted_token_ids, workspace->vocab_size,
      0, static_cast<int>(sizeof(float) * 8U), stream));
}

std::size_t SamplingTopCount(const GpuSamplingParameters& parameters,
                             std::size_t vocab_size) {
  return parameters.top_k > 0
             ? std::min(vocab_size,
                        std::max(static_cast<std::size_t>(parameters.top_k),
                                 parameters.min_keep))
             : 0;
}

__global__ void ScatterDraftProbabilitiesKernel(
    const std::uint32_t* __restrict__ draft_candidate_ids,
    const float* __restrict__ draft_candidate_probabilities,
    std::size_t draft_candidate_count, float* __restrict__ dense,
    std::size_t vocab_size) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= draft_candidate_count)
    return;
  const auto token = draft_candidate_ids[index];
  if (token >= vocab_size)
    return;
  // Only the first occurrence writes. Sum duplicates in their original order
  // so the lookup retains the sparse verifier's exact F32 probability.
  for (std::size_t previous = 0; previous < index; ++previous) {
    if (draft_candidate_ids[previous] == token)
      return;
  }
  float probability = 0.0F;
  for (std::size_t next = index; next < draft_candidate_count; ++next) {
    if (draft_candidate_ids[next] == token)
      probability += draft_candidate_probabilities[next];
  }
  dense[token] = probability;
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
  adjusted_logits[index] = isfinite(logits[index]) ? logits[index] : -INFINITY;
  token_ids[index] = static_cast<std::uint32_t>(index);
}

__global__ void PrepareCandidateLogitsKernel(const float* logits, float* scores,
                                             std::uint32_t* ids,
                                             std::size_t size) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    scores[i] = isfinite(logits[i]) ? logits[i] : -INFINITY;
    ids[i] = static_cast<std::uint32_t>(i);
  }
}

__global__ void ApplySparsePenaltiesKernel(
    float* __restrict__ logits,
    const sampling::TokenPenalty* __restrict__ penalties,
    std::size_t penalty_count, std::size_t vocab_size, float repeat_penalty,
    float frequency_penalty, float presence_penalty) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= penalty_count) {
    return;
  }
  const auto penalty = penalties[index];
  const std::uint32_t token = penalty.token;
  if (token >= vocab_size) {
    return;
  }
  float adjusted = logits[token];
  if (penalty.repeated && repeat_penalty != 1.0F) {
    adjusted = adjusted <= 0.0F ? adjusted * repeat_penalty
                                : adjusted / repeat_penalty;
  }
  adjusted -= frequency_penalty * static_cast<float>(penalty.generated_count);
  if (penalty.generated_count != 0)
    adjusted -= presence_penalty;
  logits[token] = adjusted;
}

__global__ void LinearSamplingKernel(const float* __restrict__ logits,
                                     std::uint32_t* __restrict__ out_token,
                                     std::size_t vocab_size, float temperature,
                                     float min_p, double uniform) {
  const std::size_t tid = threadIdx.x;
  const std::size_t begin = (vocab_size * tid) / blockDim.x;
  const std::size_t end = (vocab_size * (tid + 1U)) / blockDim.x;

  __shared__ float shared_max[kSamplingThreads];
  __shared__ float segment_sums[kSamplingThreads];
  __shared__ float segment_prefix[kSamplingThreads];
  __shared__ std::uint32_t selected_segment;
  __shared__ double selected_threshold;

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
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
    }
    return;
  }
  const float min_p_threshold =
      min_p > 0.0F ? maximum + temperature * logf(min_p) : -INFINITY;
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
    double threshold = uniform * total;
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
    double threshold = selected_threshold;
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
    std::size_t min_keep, double uniform) {
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
  __shared__ double selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
    }
    return;
  }
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
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
    double threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      if (threshold <= 0.0F) {
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
      const float threshold = maximum + temperature * logf(min_p);
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
    const double threshold = uniform * total;
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
    double threshold = selected_threshold;
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
    const float* __restrict__ logits, std::uint32_t* __restrict__ out_token,
    std::uint32_t* __restrict__ out_accepted, std::size_t vocab_size,
    float temperature, float min_p, std::uint32_t draft_token,
    float draft_token_probability,
    const float* __restrict__ draft_probabilities, double acceptance_uniform,
    double residual_uniform) {
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
  __shared__ double selected_threshold;

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
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
      *out_accepted = 2;
    }
    return;
  }
  const float min_p_threshold =
      min_p > 0.0F ? maximum + temperature * logf(min_p) : -INFINITY;
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
        acceptance_uniform * draft_token_probability < draft_target_probability;
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
    const float target_weight = expf((value - maximum) * inverse_temperature);
    const float draft_probability = draft_probabilities[index];
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
    const double threshold = residual_uniform * total;
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
    double threshold = selected_threshold;
    std::uint32_t selected =
        static_cast<std::uint32_t>(end > begin ? end - 1U : 0U);
    for (std::size_t index = begin; index < end; ++index) {
      const float value = logits[index];
      if (!isfinite(value) || value < min_p_threshold) {
        continue;
      }
      const float target_weight = expf((value - maximum) * inverse_temperature);
      float weight = target_weight;
      if (use_residual) {
        const float draft_probability = draft_probabilities[index];
        weight = fmaxf(target_weight - draft_probability * target_total, 0.0F);
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

// Preserve the linear sampler's 256 contiguous segments and sequential F32
// additions, while distributing loads and exponentials across the GPU.
// This workspace reuses the unused sorted token IDs on the unordered route.
struct SpeculativeSegments {
  float maxima[kSamplingThreads];
  float target_sums[kSamplingThreads];
  float residual_sums[kSamplingThreads];
  float target_total;
};

constexpr std::size_t kSpeculativeSegmentThreads = 128;

__global__ void SpeculativeSegmentMaxKernel(
    const float* __restrict__ logits, SpeculativeSegments* __restrict__ scratch,
    std::size_t vocab_size) {
  __shared__ float maxima[kSpeculativeSegmentThreads];
  const std::size_t begin = (vocab_size * blockIdx.x) / kSamplingThreads;
  const std::size_t end = (vocab_size * (blockIdx.x + 1U)) / kSamplingThreads;
  float maximum = -INFINITY;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    if (isfinite(logits[i]))
      maximum = fmaxf(maximum, logits[i]);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0; stride /= 2U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scratch->maxima[blockIdx.x] = maxima[0];
}

__global__ void SpeculativeSegmentWeightsKernel(
    const float* __restrict__ logits, float* __restrict__ weights,
    SpeculativeSegments* __restrict__ scratch, std::size_t vocab_size,
    float temperature, float min_p) {
  extern __shared__ float values[];
  __shared__ float maxima[kSpeculativeSegmentThreads];
  maxima[threadIdx.x] =
      fmaxf(scratch->maxima[threadIdx.x],
            scratch->maxima[threadIdx.x + kSpeculativeSegmentThreads]);
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2U; stride > 0; stride /= 2U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const float maximum = maxima[0];
  const float min_p_threshold =
      min_p > 0.0F ? maximum + temperature * logf(min_p) : -INFINITY;
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  const std::size_t begin = (vocab_size * blockIdx.x) / kSamplingThreads;
  const std::size_t end = (vocab_size * (blockIdx.x + 1U)) / kSamplingThreads;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float value = logits[i];
    const float weight = isfinite(value) && value >= min_p_threshold
                             ? expf((value - maximum) * inverse_temperature)
                             : 0.0F;
    weights[i] = weight;
    values[i - begin] = weight;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < end - begin; ++i)
      sum += values[i];
    scratch->target_sums[blockIdx.x] = sum;
  }
}

__global__ void SpeculativeSegmentResidualKernel(
    const float* __restrict__ weights,
    const float* __restrict__ draft_probabilities,
    SpeculativeSegments* __restrict__ scratch, std::size_t vocab_size,
    std::uint32_t draft_token, float draft_token_probability,
    double acceptance_uniform, std::uint32_t* __restrict__ out_token,
    std::uint32_t* __restrict__ out_accepted) {
  extern __shared__ float values[];
  __shared__ float sums[kSamplingThreads];
  __shared__ float target_total;
  __shared__ bool accepted;
  sums[threadIdx.x] = scratch->target_sums[threadIdx.x];
  sums[threadIdx.x + kSpeculativeSegmentThreads] =
      scratch->target_sums[threadIdx.x + kSpeculativeSegmentThreads];
  __syncthreads();
  if (threadIdx.x == 0) {
    target_total = 0.0F;
    for (std::size_t i = 0; i < kSamplingThreads; ++i)
      target_total += sums[i];
    if (!(target_total > 0) || !isfinite(target_total)) {
      accepted = true;  // All threads/blocks skip residual work on failure.
      if (blockIdx.x == 0) {
        *out_token = kInvalidSampleToken;
        *out_accepted = 2;
      }
    } else {
      const float draft_weight =
          draft_token < vocab_size ? weights[draft_token] : 0.0F;
      accepted = acceptance_uniform * draft_token_probability <
                 draft_weight / target_total;
      if (blockIdx.x == 0) {
        *out_accepted = accepted ? 1U : 0U;
        scratch->target_total = target_total;
        if (accepted)
          *out_token = draft_token;
      }
    }
  }
  __syncthreads();
  if (accepted)
    return;
  const std::size_t begin = (vocab_size * blockIdx.x) / kSamplingThreads;
  const std::size_t end = (vocab_size * (blockIdx.x + 1U)) / kSamplingThreads;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x)
    values[i - begin] =
        fmaxf(weights[i] - draft_probabilities[i] * target_total, 0.0F);
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::size_t i = 0; i < end - begin; ++i)
      sum += values[i];
    scratch->residual_sums[blockIdx.x] = sum;
  }
}

__global__ void SpeculativeSegmentSelectKernel(
    const float* __restrict__ weights,
    const float* __restrict__ draft_probabilities,
    const SpeculativeSegments* __restrict__ scratch, std::size_t vocab_size,
    double residual_uniform, std::uint32_t* __restrict__ out_token,
    const std::uint32_t* __restrict__ out_accepted) {
  if (*out_accepted != 0)
    return;
  extern __shared__ float values[];
  __shared__ float sums[kSamplingThreads];
  __shared__ float prefix[kSamplingThreads];
  __shared__ bool use_residual;
  __shared__ double selected_threshold;
  __shared__ std::uint32_t selected_segment;
  sums[threadIdx.x] = scratch->residual_sums[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
    for (std::size_t i = 0; i < kSamplingThreads; ++i) {
      prefix[i] = total;
      total += sums[i];
    }
    use_residual = total > 0.0F && isfinite(total);
  }
  __syncthreads();
  if (!use_residual) {
    sums[threadIdx.x] = scratch->target_sums[threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0F;
      for (std::size_t i = 0; i < kSamplingThreads; ++i) {
        prefix[i] = total;
        total += sums[i];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float total =
        prefix[kSamplingThreads - 1U] + sums[kSamplingThreads - 1U];
    const double threshold = residual_uniform * total;
    selected_segment = kSamplingThreads - 1U;
    for (std::size_t i = 0; i < kSamplingThreads; ++i) {
      if (sums[i] > 0.0F && threshold < prefix[i] + sums[i]) {
        selected_segment = static_cast<std::uint32_t>(i);
        break;
      }
    }
    selected_threshold = threshold - prefix[selected_segment];
  }
  __syncthreads();
  const std::size_t begin = (vocab_size * selected_segment) / kSamplingThreads;
  const std::size_t end =
      (vocab_size * (selected_segment + 1U)) / kSamplingThreads;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x)
    values[i - begin] =
        use_residual
            ? fmaxf(weights[i] - draft_probabilities[i] * scratch->target_total,
                    0.0F)
            : weights[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    double threshold = selected_threshold;
    std::uint32_t selected =
        static_cast<std::uint32_t>(end > begin ? end - 1U : 0U);
    for (std::size_t i = begin; i < end; ++i) {
      const float weight = values[i - begin];
      if (weight <= 0.0F)
        continue;
      threshold -= weight;
      selected = static_cast<std::uint32_t>(i);
      if (threshold < 0.0F)
        break;
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
    const float* __restrict__ draft_probabilities, double acceptance_uniform,
    double residual_uniform) {
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
  __shared__ double selected_threshold;

  const std::size_t top_p_begin = (initial_count * tid) / blockDim.x;
  const std::size_t top_p_end = (initial_count * (tid + 1U)) / blockDim.x;
  const float maximum = sorted_logits[0];
  if (!isfinite(maximum)) {
    if (tid == 0) {
      *out_token = kInvalidSampleToken;
      *out_accepted = 2;
    }
    return;
  }
  const float inverse_temperature = 1.0F / fmaxf(temperature, FLT_MIN);
  float local_sum = 0.0F;
  for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
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
    double threshold = selected_threshold;
    std::size_t keep = top_p_end;
    for (std::size_t index = top_p_begin; index < top_p_end; ++index) {
      threshold -= expf((sorted_logits[index] - maximum) * inverse_temperature);
      if (threshold <= 0.0F) {
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
      const float threshold = maximum + temperature * logf(min_p);
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
        acceptance_uniform * draft_token_probability < draft_target_probability;
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
    const float draft_probability =
        draft_probabilities[sorted_token_ids[index]];
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
    const double threshold = residual_uniform * total;
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
    double threshold = selected_threshold;
    std::uint32_t selected = sorted_token_ids[sample_end - 1U];
    for (std::size_t index = sample_begin; index < sample_end; ++index) {
      const float target_weight =
          expf((sorted_logits[index] - maximum) * inverse_temperature);
      float weight = target_weight;
      if (use_residual) {
        const float draft_probability =
            draft_probabilities[sorted_token_ids[index]];
        weight = fmaxf(target_weight - draft_probability * target_total, 0.0F);
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

  float max_val = -INFINITY;
  std::uint32_t max_idx = kInvalidSampleToken;

  for (std::size_t i = tid; i < vocab_size; i += block_size) {
    const float val = logits[i];
    if (isfinite(val) && val > max_val) {
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
      if (s_max[tid + stride] > s_max[tid] ||
          (s_max[tid + stride] == s_max[tid] &&
           s_idx[tid + stride] < s_idx[tid])) {
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

template<bool Partial = false, bool MapIndices = false>
__global__ void BatchedArgmaxKernel(const float* __restrict__ logits,
                                    std::uint32_t* __restrict__ out_tokens,
                                    std::size_t vocab_size,
                                    float* partial_values = nullptr,
                                    std::uint32_t* partial_tokens = nullptr) {
  const std::size_t row = Partial ? blockIdx.y : blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const float* row_logits = logits + (row * vocab_size);
  const std::size_t begin = Partial ? blockIdx.x * kArgmaxTile : 0;
  const std::size_t end = Partial && begin + kArgmaxTile < vocab_size
                              ? begin + kArgmaxTile
                              : vocab_size;

  float max_val = -INFINITY;
  std::uint32_t max_idx = kInvalidSampleToken;
  for (std::size_t index = begin + tid; index < end; index += blockDim.x) {
    const float value = row_logits[index];
    if constexpr (MapIndices) {
      const auto candidate = partial_tokens[row * vocab_size + index];
      if (isfinite(value) &&
          (value > max_val || (value == max_val && candidate < max_idx))) {
        max_val = value;
        max_idx = candidate;
      }
    } else {
      if (isfinite(value) && value > max_val) {
        max_val = value;
        max_idx = static_cast<std::uint32_t>(index);
      }
    }
  }

  __shared__ float shared_max[256];
  __shared__ std::uint32_t shared_index[256];
  shared_max[tid] = max_val;
  shared_index[tid] = max_idx;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride && (shared_max[tid + stride] > shared_max[tid] ||
                         (shared_max[tid + stride] == shared_max[tid] &&
                          shared_index[tid + stride] < shared_index[tid]))) {
      shared_max[tid] = shared_max[tid + stride];
      shared_index[tid] = shared_index[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if constexpr (Partial) {
      const auto offset = row * gridDim.x + blockIdx.x;
      partial_values[offset] = shared_max[0];
      partial_tokens[offset] = shared_index[0];
    } else {
      out_tokens[row] = shared_index[0];
    }
  }
}

hipError_t PrepareBatchedGPUArgmax() {
  for (const void* kernel :
       {reinterpret_cast<const void*>(BatchedArgmaxKernel<true>),
        reinterpret_cast<const void*>(BatchedArgmaxKernel<false, true>),
        reinterpret_cast<const void*>(BatchedArgmaxKernel<>)}) {
    hipFuncAttributes attributes{};
    const auto status = hipFuncGetAttributes(&attributes, kernel);
    if (status != hipSuccess)
      return status;
  }
  return hipSuccess;
}

void LaunchBatchedGPUArgmax(const float* logits, std::uint32_t* out_tokens,
                            std::size_t batch_size, std::size_t vocab_size,
                            std::span<float> scratch, hipStream_t stream) {
  if (batch_size == 0 || vocab_size == 0) {
    return;
  }
  const auto partials = (vocab_size - 1) / kArgmaxTile + 1;
  if (partials > 1 && batch_size <= scratch.size() / (2 * partials)) {
    // FFN output scratch is free after the head projection. Parallel vocabulary
    // tiles keep large rows from serializing their reads in one workgroup.
    auto* const indices = reinterpret_cast<std::uint32_t*>(
        scratch.data() + batch_size * partials);
    BatchedArgmaxKernel<true><<<dim3(static_cast<unsigned>(partials),
                                     static_cast<unsigned>(batch_size)),
                                256, 0, stream>>>(
        logits, out_tokens, vocab_size, scratch.data(), indices);
    BatchedArgmaxKernel<false, true>
        <<<static_cast<unsigned>(batch_size), 256, 0, stream>>>(
            scratch.data(), out_tokens, partials, nullptr, indices);
  } else {
    BatchedArgmaxKernel<>
        <<<static_cast<unsigned int>(batch_size), 256, 0, stream>>>(
            logits, out_tokens, vocab_size);
  }
}

std::size_t EstimateGpuSamplingWorkspaceBytes(std::size_t vocab_size,
                                              std::size_t penalty_capacity) {
  GpuSamplingWorkspace workspace;
  workspace.vocab_size = vocab_size;
  workspace.penalty_capacity = penalty_capacity;
  HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
      nullptr, workspace.sort_temp_storage_bytes, workspace.adjusted_logits,
      workspace.sorted_logits, workspace.token_ids, workspace.sorted_token_ids,
      vocab_size));
  return workspace.SizeBytes();
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
    HIP_CHECK(hipMalloc(&workspace->sorted_logits, vocab_size * sizeof(float)));
    HIP_CHECK(
        hipMalloc(&workspace->token_ids, vocab_size * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&workspace->sorted_token_ids,
                        vocab_size * sizeof(std::uint32_t)));
    if (penalty_capacity > 0) {
      HIP_CHECK(hipMalloc(&workspace->penalties,
                          penalty_capacity * sizeof(sampling::TokenPenalty)));
      HIP_CHECK(hipMalloc(&workspace->draft_candidate_ids,
                          penalty_capacity * sizeof(std::uint32_t)));
      HIP_CHECK(hipMalloc(&workspace->draft_candidate_probabilities,
                          penalty_capacity * sizeof(float)));
    }
    HIP_CHECK(
        hipMalloc(&workspace->speculative_accepted, sizeof(std::uint32_t)));
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(
        nullptr, workspace->sort_temp_storage_bytes, workspace->adjusted_logits,
        workspace->sorted_logits, workspace->token_ids,
        workspace->sorted_token_ids, vocab_size));
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
  free_buffer(workspace->penalties);
  free_buffer(workspace->draft_candidate_ids);
  free_buffer(workspace->draft_candidate_probabilities);
  free_buffer(workspace->speculative_accepted);
  free_buffer(workspace->sort_temp_storage);
  workspace->sort_temp_storage_bytes = 0;
  workspace->vocab_size = 0;
  workspace->penalty_capacity = 0;
}

void LaunchGPUSortLogits(const float* logits, GpuSamplingWorkspace* workspace,
                         hipStream_t stream, std::size_t keep) {
  if (logits == nullptr || workspace == nullptr || workspace->vocab_size == 0) {
    throw std::invalid_argument("invalid GPU candidate sort");
  }
  const auto size = workspace->vocab_size;
  if (keep > size)
    throw std::invalid_argument("candidate count exceeds vocabulary");
  PrepareCandidateLogitsKernel<<<
      static_cast<unsigned>((size + kSamplingThreads - 1) / kSamplingThreads),
      kSamplingThreads, 0, stream>>>(logits, workspace->adjusted_logits,
                                     workspace->token_ids, size);
  SortSamplingLogits(workspace, keep, stream);
}

void LaunchGPUSampling(const float* logits, std::uint32_t* out_token,
                       std::size_t vocab_size,
                       const GpuSamplingParameters& parameters,
                       const sampling::TokenPenalty* penalties,
                       std::size_t penalty_count,
                       GpuSamplingWorkspace* workspace, hipStream_t stream) {
  if (!std::isfinite(parameters.uniform) || parameters.uniform < 0 ||
      parameters.uniform >= 1)
    throw std::invalid_argument("sampling draw must be in [0, 1)");
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
    HIP_CHECK(hipMemcpyAsync(workspace->penalties, penalties,
                             penalty_count * sizeof(sampling::TokenPenalty),
                             hipMemcpyHostToDevice, stream));
    const std::size_t blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalties, penalty_count,
        vocab_size, parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }

  if (parameters.temperature <= 0.0F) {
    LaunchGPUArgmax(sampling_logits, out_token, vocab_size, stream);
    return;
  }
  if (parameters.top_k == 1 && parameters.min_keep <= 1) {
    LaunchGPUArgmax(sampling_logits, out_token, vocab_size, stream);
    return;
  }
  if (!needs_ordering) {
    LinearSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
        sampling_logits, out_token, vocab_size, parameters.temperature,
        parameters.min_p, parameters.uniform);
    return;
  }

  SortSamplingLogits(workspace, SamplingTopCount(parameters, vocab_size),
                     stream);
  SortedSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      vocab_size, parameters.temperature, parameters.top_k, parameters.top_p,
      parameters.min_p, parameters.min_keep, parameters.uniform);
}

void LaunchGPUSpeculativeSampling(
    const float* logits, std::uint32_t* out_token, std::uint32_t* out_accepted,
    std::size_t vocab_size, const GpuSamplingParameters& parameters,
    std::uint32_t draft_token, float draft_token_probability,
    const std::uint32_t* draft_candidate_ids,
    const float* draft_candidate_probabilities,
    std::size_t draft_candidate_count, double acceptance_uniform,
    double residual_uniform, const sampling::TokenPenalty* penalties,
    std::size_t penalty_count, GpuSamplingWorkspace* workspace,
    hipStream_t stream) {
  if (!std::isfinite(acceptance_uniform) || acceptance_uniform < 0 ||
      acceptance_uniform >= 1 || !std::isfinite(residual_uniform) ||
      residual_uniform < 0 || residual_uniform >= 1)
    throw std::invalid_argument("speculative draws must be in [0, 1)");
  if (logits == nullptr || out_token == nullptr || out_accepted == nullptr ||
      workspace == nullptr || vocab_size == 0 ||
      workspace->vocab_size != vocab_size || draft_candidate_count == 0 ||
      draft_candidate_count > workspace->penalty_capacity ||
      penalty_count > workspace->penalty_capacity ||
      draft_candidate_ids == nullptr ||
      draft_candidate_probabilities == nullptr ||
      !std::isfinite(draft_token_probability) ||
      draft_token_probability <= 0.0F || parameters.temperature <= 0.0F ||
      (parameters.top_k == 1 && parameters.min_keep <= 1)) {
    throw std::invalid_argument("invalid GPU speculative sampling launch");
  }

  HIP_CHECK(hipMemcpyAsync(workspace->draft_candidate_ids, draft_candidate_ids,
                           draft_candidate_count * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(
      workspace->draft_candidate_probabilities, draft_candidate_probabilities,
      draft_candidate_count * sizeof(float), hipMemcpyHostToDevice, stream));

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
    HIP_CHECK(hipMemcpyAsync(workspace->penalties, penalties,
                             penalty_count * sizeof(sampling::TokenPenalty),
                             hipMemcpyHostToDevice, stream));
    const std::size_t blocks =
        (penalty_count + kSamplingThreads - 1U) / kSamplingThreads;
    ApplySparsePenaltiesKernel<<<static_cast<unsigned int>(blocks),
                                 kSamplingThreads, 0, stream>>>(
        workspace->adjusted_logits, workspace->penalties, penalty_count,
        vocab_size, parameters.repeat_penalty, parameters.frequency_penalty,
        parameters.presence_penalty);
  }

  if (needs_ordering) {
    SortSamplingLogits(workspace, SamplingTopCount(parameters, vocab_size),
                       stream);
  }
  // The unsorted token IDs are dead after sorting. Reuse that buffer for q;
  // residual evaluation then makes one lookup per vocabulary token.
  auto* draft_probabilities = reinterpret_cast<float*>(workspace->token_ids);
  HIP_CHECK(hipMemsetAsync(draft_probabilities, 0, vocab_size * sizeof(float),
                           stream));
  ScatterDraftProbabilitiesKernel<<<
      static_cast<unsigned>((draft_candidate_count + kSamplingThreads - 1) /
                            kSamplingThreads),
      kSamplingThreads, 0, stream>>>(
      workspace->draft_candidate_ids, workspace->draft_candidate_probabilities,
      draft_candidate_count, draft_probabilities, vocab_size);
  if (!needs_ordering) {
    // Avoid multi-kernel overhead for small vocabularies and cap per-block
    // shared memory at 16 KiB. Production vocabularies use this parallel path.
    if (vocab_size >= 16384 && vocab_size <= kSamplingThreads * 4096U) {
      auto* scratch =
          reinterpret_cast<SpeculativeSegments*>(workspace->sorted_token_ids);
      static_assert(sizeof(SpeculativeSegments) <=
                    16384 * sizeof(std::uint32_t));
      auto* weights = workspace->sorted_logits;
      const std::size_t shared_bytes =
          ((vocab_size + kSamplingThreads - 1U) / kSamplingThreads) *
          sizeof(float);
      SpeculativeSegmentMaxKernel<<<kSamplingThreads,
                                    kSpeculativeSegmentThreads, 0, stream>>>(
          sampling_logits, scratch, vocab_size);
      SpeculativeSegmentWeightsKernel<<<
          kSamplingThreads, kSpeculativeSegmentThreads, shared_bytes, stream>>>(
          sampling_logits, weights, scratch, vocab_size, parameters.temperature,
          parameters.min_p);
      SpeculativeSegmentResidualKernel<<<
          kSamplingThreads, kSpeculativeSegmentThreads, shared_bytes, stream>>>(
          weights, draft_probabilities, scratch, vocab_size, draft_token,
          draft_token_probability, acceptance_uniform, out_token, out_accepted);
      SpeculativeSegmentSelectKernel<<<1, kSamplingThreads, shared_bytes,
                                       stream>>>(
          weights, draft_probabilities, scratch, vocab_size, residual_uniform,
          out_token, out_accepted);
      return;
    }
    LinearSpeculativeSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
        sampling_logits, out_token, out_accepted, vocab_size,
        parameters.temperature, parameters.min_p, draft_token,
        draft_token_probability, draft_probabilities, acceptance_uniform,
        residual_uniform);
    return;
  }

  SortedSpeculativeSamplingKernel<<<1, kSamplingThreads, 0, stream>>>(
      workspace->sorted_logits, workspace->sorted_token_ids, out_token,
      out_accepted, vocab_size, parameters.temperature, parameters.top_k,
      parameters.top_p, parameters.min_p, parameters.min_keep, draft_token,
      draft_token_probability, draft_probabilities, acceptance_uniform,
      residual_uniform);
}

}  // namespace gufo::hip
#endif
