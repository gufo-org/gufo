#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_query_attention_forward_wmma.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace strix::hip {

__global__ void WriteKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const float key = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  const float val = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  k_cache[kv_offset + tid] = key;
  v_cache[kv_offset + tid] = val;

  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t kv_width =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t f16_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
        (static_cast<std::size_t>(kv_h) * head_dim) + tid;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = __float2half_rn(val);
    }
  }
}

__global__ void AttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];

  // Step 1: Compute scores for each position p = 0..seq_len-1
  for (std::size_t p = 0; p < seq_len; ++p) {
    const std::size_t p_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          p) *
         head_dim);
    const float* k_p = k_cache + p_offset;

    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * k_p[d];
    }
    s_scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_scratch[tid] += s_scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p] = s_scratch[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..seq_len-1]
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_scratch[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
  __syncthreads();

  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    s_scores[p] *= inv_sum_exp;
  }
  __syncthreads();

  // Step 3: Weighted sum of V
  if (tid < head_dim) {
    float ctx = 0.0F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const float* v_p = v_cache + p_offset;
      ctx += s_scores[p] * v_p[tid];
    }

    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionPartialsKernel(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || split >= split_count || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t partition_length =
      (sequence_length + split_count - 1) / split_count;
  const std::size_t partition_begin =
      static_cast<std::size_t>(split) * partition_length;
  const std::size_t partition_end =
      min(partition_begin + partition_length, sequence_length);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  for (std::size_t position = partition_begin; position < partition_end;
       ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * max_splits) + split;
  if (lane == 0) {
    scratch[(partition_index * 2) + 0] = running_max;
    scratch[(partition_index * 2) + 1] = running_sum;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + (partition_index * vector_width);
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    partial_output[lane + (vector * 32)] = output_values[vector];
  }
}

__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionReduceKernel(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max =
        fmaxf(global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  float global_sum = 0.0F;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t stats_offset = (head_partition_offset + split) * 2;
    const float partition_scale = expf(scratch[stats_offset] - global_max);
    global_sum += scratch[stats_offset + 1] * partition_scale;
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / global_sum;

#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result{};
    for (std::uint32_t split = 0; split < split_count; ++split) {
      const std::size_t partition_index = head_partition_offset + split;
      const float partition_scale =
          expf(scratch[partition_index * 2] - global_max);
      const float4 partial = partial_outputs[(partition_index * vector_width) +
                                             lane + (vector * 32)];
      result.x += partial.x * partition_scale;
      result.y += partial.y * partition_scale;
      result.z += partial.z * partition_scale;
      result.w += partial.w * partition_scale;
    }

    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__global__ void WriteKVCachePtrKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* __restrict__ d_pos,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::uint32_t pos = *d_pos;
  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const float key = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  const float val = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  k_cache[kv_offset + tid] = key;
  v_cache[kv_offset + tid] = val;

  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t kv_width =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t f16_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
        (static_cast<std::size_t>(kv_h) * head_dim) + tid;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = __float2half_rn(val);
    }
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionPtrKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    const std::uint32_t* __restrict__ d_pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t pos = *d_pos;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, float* split_k_scratch,
                     bool skip_kv_write) {
  // 1. Write K and V into KV cache across all num_kv_heads (updating FP32 and
  // FP16 coherently). May be skipped when a fused kernel already wrote the
  // cache (opt-c010-qk-rope-kv).
  if (!skip_kv_write) {
    WriteKVCacheKernel<<<num_kv_heads, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, pos, max_context,
        num_kv_heads, head_dim);
  }

  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool split_k_supported = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, num_heads, num_kv_heads, head_dim);
  if (split_k_supported && split_k_scratch != nullptr) {
    const std::uint32_t split_count =
        detail::SelectDecodeAttentionSplitCount(sequence_length);
    detail::EmitAttentionDispatch("decode_split_k_fp32", "");
    const dim3 partial_grid(num_heads, split_count);
    QwenDecodeSplitKAttentionPartialsKernel<<<partial_grid, 32, 0, stream>>>(
        q, k_cache, v_cache, split_k_scratch, layer_idx, pos, max_context,
        num_heads, num_kv_heads, split_count);
    QwenDecodeSplitKAttentionReduceKernel<<<num_heads, 32, 0, stream>>>(
        gate, split_k_scratch, out_context, num_heads, split_count);
  } else if (head_dim == 256 && num_kv_heads > 0 &&
             (num_heads % num_kv_heads) == 0) {
    detail::EmitAttentionDispatch("decode_online_fp32",
                                  split_k_supported
                                      ? "decode_split_k_fp32: missing_scratch"
                                      : "decode_split_k_fp32: below_threshold");
    QwenDecodeOnlineAttentionKernel<<<num_heads, 32, 0, stream>>>(
        q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
        num_heads, num_kv_heads);
  } else {
    detail::EmitAttentionDispatch(
        "decode_baseline",
        "decode_split_k_fp32: unsupported_shape; decode_online_fp32: "
        "unsupported_shape");
    const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
    const std::size_t shared_scores_bytes = seq_len * sizeof(float);
    AttentionKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
        q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
        num_heads, num_kv_heads, head_dim);
  }
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, bool skip_kv_write) {
  if (!skip_kv_write) {
    WriteKVCachePtrKernel<<<num_kv_heads, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, d_pos, max_context,
        num_kv_heads, head_dim);
  }

  if (head_dim == 256 && num_kv_heads > 0 && (num_heads % num_kv_heads) == 0) {
    detail::EmitAttentionDispatch("decode_online_fp32_graph", "");
    QwenDecodeOnlineAttentionPtrKernel<<<num_heads, 32, 0, stream>>>(
        q, gate, k_cache, v_cache, out_context, layer_idx, d_pos, max_context,
        num_heads, num_kv_heads);
  } else {
    detail::EmitAttentionDispatch(
        "none", "decode_online_fp32_graph: unsupported_shape");
  }
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
