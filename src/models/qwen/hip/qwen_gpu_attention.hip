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
#include "src/models/qwen/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"

namespace strix::hip {

namespace {

constexpr ck::index_t kCkNumKvHeads = detail::kCkAttentionKvHeads;

template<ck::index_t... Values>
using CkSequence = ck::Sequence<Values...>;

using CkPassThrough = ck::tensor_operation::element_wise::PassThrough;
using CkScale = ck::tensor_operation::element_wise::Scale;
using CkGqaOperation =
    ck::tensor_operation::device::DeviceGroupedQueryAttentionForward_Wmma<
        2, 1, 1, 1, 1, ck::half_t, ck::half_t, ck::half_t, float, ck::Tuple<>,
        float, ck::Tuple<>, float, float, CkPassThrough, CkPassThrough, CkScale,
        CkPassThrough, CkPassThrough,
        ck::tensor_operation::device::GemmSpecialization::MNKOPadding,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default, 1,
        kCkNumKvHeads, 256,
        // QK: 128x128 output tile, K tile 64.
        128, 128, 64, 8, 8,
        // PV: output width 64, reduction tile 64.
        64, 64, 8, 16, 16, 16, 1, 8, 4,
        // Q and K transfers.
        CkSequence<2, 128, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>, 2, 8,
        8, true, CkSequence<8, 32, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>,
        2, 8, 8, true,
        // V and output transfers.
        CkSequence<2, 16, 8>, CkSequence<0, 2, 1>, CkSequence<0, 2, 1>, 1, 1, 1,
        false, 1, 1, CkSequence<1, 128, 1, 2>, 8,
        ck::tensor_operation::device::MaskingSpecialization::
            MaskOutUpperTriangle>;

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

}  // namespace

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

__global__ void WriteBatchedKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t kv_head = blockIdx.x;
  if (b >= batch_size || kv_head >= num_kv_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::uint32_t pos = start_pos + static_cast<std::uint32_t>(b);
  const std::size_t k_src_offset =
      (b * num_kv_heads * head_dim) +
      (static_cast<std::size_t>(kv_head) * head_dim);

  const std::size_t cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
            max_context +
        pos) *
       head_dim);

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float key = k[k_src_offset + i];
    const float val = v[k_src_offset + i];
    k_cache[cache_offset + i] = key;
    v_cache[cache_offset + i] = val;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_cache_offset + i] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_cache_offset + i] = __float2half_rn(val);
    }
  }
}

__global__ void BatchedAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (h >= num_heads || t >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::uint32_t kv_h = h / (num_heads / num_kv_heads);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));

  __shared__ float s_scratch[256];
  extern __shared__ float s_scores[];

  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(t);
  const std::size_t seq_len = static_cast<std::size_t>(cur_pos) + 1;
  const float* q_h =
      q + (t * num_heads * head_dim) + (static_cast<std::size_t>(h) * head_dim);

  const auto* q_h_vec = reinterpret_cast<const float4*>(q_h);
  const std::size_t num_vec = head_dim / 4;
  constexpr std::size_t wave_dot_min_sequence = 64;
  if (seq_len >= wave_dot_min_sequence) {
    const std::size_t lane_id = tid & 31U;
    const std::size_t wave_id = tid >> 5U;
    const std::size_t waves_per_block = blockDim.x >> 5U;
    for (std::size_t p = wave_id; p < seq_len; p += waves_per_block) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const auto* k_p_vec = reinterpret_cast<const float4*>(k_cache + p_offset);

      float dot = 0.0F;
      for (std::size_t d = lane_id; d < num_vec; d += 32) {
        const float4 qv = q_h_vec[d];
        const float4 kv = k_p_vec[d];
        dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_xor(dot, offset);
      }
      if (lane_id == 0) {
        s_scores[p] = dot * scale;
      }
    }
  } else {
    for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const auto* k_p_vec = reinterpret_cast<const float4*>(k_cache + p_offset);

      float dot = 0.0F;
      for (std::size_t d = 0; d < num_vec; ++d) {
        const float4 qv = q_h_vec[d];
        const float4 kv = k_p_vec[d];
        dot += (qv.x * kv.x) + (qv.y * kv.y) + (qv.z * kv.z) + (qv.w * kv.w);
      }
      s_scores[p] = dot * scale;
    }
  }
  __syncthreads();

  // Step 2: Softmax
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
      const float g =
          (gate + (t * num_heads *
                   head_dim))[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[(t * num_heads * head_dim) +
                (static_cast<std::size_t>(h) * head_dim) + tid] = ctx;
  }
}

__global__ void SyncTiledAttentionKvPrefixKernel(
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t prefix_length,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t pos = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (pos >= prefix_length || index >= kv_width) {
    return;
  }
  const std::size_t kv_head = index / head_dim;
  const std::size_t dimension = index % head_dim;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
            max_context +
        pos) *
       head_dim) +
      dimension;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      index;
  k_cache_f16[f16_cache_offset] = __float2half_rn(k_cache[f32_cache_offset]);
  v_cache_f16[f16_cache_offset] = __float2half_rn(v_cache[f32_cache_offset]);
}

void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            void* k_cache_f16, void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream, bool skip_kv_write) {
  const dim3 kv_grid(num_kv_heads, static_cast<unsigned int>(batch_size));
  if (!skip_kv_write) {
    WriteBatchedKVCacheKernel<<<kv_grid, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, start_pos, batch_size,
        max_context, num_kv_heads, head_dim);
  }

  const dim3 attn_grid(num_heads, static_cast<unsigned int>(batch_size));
  const std::size_t max_seq_len =
      static_cast<std::size_t>(start_pos) + batch_size;
  const std::size_t shared_scores_bytes = max_seq_len * sizeof(float);
  BatchedAttentionKernel<<<attn_grid, 256, shared_scores_bytes, stream>>>(
      q, gate, k_cache, v_cache, out_context, layer_idx, start_pos, batch_size,
      max_context, num_heads, num_kv_heads, head_dim);
}

__global__ void PackCkAttentionInputsKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, __half* __restrict__ q_f16,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (token >= batch_size) {
    return;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  if (index < attention_width) {
    const std::size_t offset = (token * attention_width) + index;
    q_f16[offset] = __float2half_rn(q[offset]);
  }

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (index < kv_width) {
    const std::size_t source_offset = (token * kv_width) + index;
    const std::size_t position = start_pos + token;
    const std::size_t f16_cache_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + position) *
         kv_width) +
        index;
    const std::size_t kv_head = index / head_dim;
    const std::size_t dimension = index % head_dim;
    const std::size_t f32_cache_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
              max_context +
          position) *
         head_dim) +
        dimension;
    const float key = k[source_offset];
    const float value = v[source_offset];
    k_cache_f16[f16_cache_offset] = __float2half_rn(key);
    v_cache_f16[f16_cache_offset] = __float2half_rn(value);
    k_cache[f32_cache_offset] = key;
    v_cache[f32_cache_offset] = value;
  }
}

// This Qwen-specific tile follows the causal online-softmax structure used by
// llama.cpp's MIT-licensed ggml-cuda/fattn-tile.cuh (commit e9fa078). The data
// movement is native HIP and specialized for gfx1151, D=256 and GQA=6.
constexpr std::uint32_t kTileAttentionHeadDim = detail::kTiledAttentionHeadDim;
constexpr std::uint32_t kTileAttentionQueryHeads =
    detail::kTiledAttentionQueryHeads;
constexpr std::uint32_t kTileAttentionKvHeads = detail::kTiledAttentionKvHeads;
constexpr std::uint32_t kTileAttentionGqa = 6;
constexpr std::uint32_t kTileAttentionQueryRows = 16;
constexpr std::uint32_t kTileAttentionHeadsPerBlock = 2;
constexpr std::uint32_t kTileAttentionColumns =
    kTileAttentionQueryRows * kTileAttentionHeadsPerBlock;
static_assert((kTileAttentionGqa % kTileAttentionHeadsPerBlock) == 0,
              "GQA heads must divide evenly across tiled blocks");
constexpr std::uint32_t kTileAttentionKeyRows = 64;
constexpr std::uint32_t kTileAttentionHalf2Dim = kTileAttentionHeadDim / 2;
constexpr std::uint32_t kTileAttentionKeyHalf2Dim = kTileAttentionHalf2Dim / 2;
constexpr std::uint32_t kTileAttentionValueRows = 32;
constexpr std::uint32_t kTileAttentionHalf2Padding = 1;
constexpr std::uint32_t kTileAttentionKeyRowStride =
    kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding;
static_assert((kTileAttentionKeyRowStride % 2) == 1,
              "Wave32 key rows require an odd stride across 32 LDS banks");

__device__ __forceinline__ float TileWaveReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float TileWaveReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ void TileHalf2Dot(float& accumulator, __half2 lhs,
                                             __half2 rhs) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
               : "+v"(accumulator)
               : "v"(lhs), "v"(rhs));
}

__global__ void PackTiledAttentionKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  if (token >= batch_size || index >= kv_width) {
    return;
  }

  const std::size_t source_offset = (token * kv_width) + index;
  const std::size_t position = start_pos + token;
  const std::size_t kv_head = index / kTileAttentionHeadDim;
  const std::size_t dimension = index % kTileAttentionHeadDim;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + position) *
       kv_width) +
      index;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * kTileAttentionKvHeads +
         kv_head) *
            max_context +
        position) *
       kTileAttentionHeadDim) +
      dimension;
  const float key = k[source_offset];
  const float value = v[source_offset];
  k_cache_f16[f16_cache_offset] = __float2half_rn(key);
  v_cache_f16[f16_cache_offset] = __float2half_rn(value);
  k_cache[f32_cache_offset] = key;
  v_cache[f32_cache_offset] = value;
}

__launch_bounds__(256, 3) __global__ void QwenTiledAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache_f16,
    const __half* __restrict__ v_cache_f16, float* __restrict__ out_context,
    std::uint32_t start_pos, std::size_t batch_size) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t wave = threadIdx.y;
  const std::uint32_t query_start = blockIdx.x * kTileAttentionQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kTileAttentionGqa / 2);
  const std::uint32_t pair_in_group = head_pair % (kTileAttentionGqa / 2);
  const std::uint32_t first_query_head =
      (kv_head * kTileAttentionGqa) +
      (pair_in_group * kTileAttentionHeadsPerBlock);
  constexpr std::size_t attention_width =
      kTileAttentionQueryHeads * kTileAttentionHeadDim;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  constexpr std::size_t kv_half2_stride = kv_width / 2;
  constexpr float attention_scale = 1.0F / 16.0F;

  __shared__ __half2 q_tile[kTileAttentionColumns][kTileAttentionHalf2Dim];
  __shared__ __half2
      kv_tile[kTileAttentionKeyRows * kTileAttentionKeyRowStride];
  __shared__ __half weights[kTileAttentionColumns][kTileAttentionKeyRows];

  constexpr std::uint32_t columns_per_wave = kTileAttentionColumns / 8;
  float running_max[columns_per_wave];
  float running_sum[columns_per_wave];
  float2 output[columns_per_wave][kTileAttentionHalf2Dim / 32];
#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    running_max[column] = -INFINITY;
    running_sum[column] = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      output[column][slot] = make_float2(0.0F, 0.0F);
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
#pragma unroll
    for (std::uint32_t half2_index = lane; half2_index < kTileAttentionHalf2Dim;
         half2_index += 32) {
      __half2 packed =
          __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      if (local_query_position < batch_size) {
        const std::size_t source =
            (static_cast<std::size_t>(local_query_position) * attention_width) +
            (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
            (2 * half2_index);
        packed =
            __halves2half2(__float2half_rn(q[source] * attention_scale),
                           __float2half_rn(q[source + 1] * attention_scale));
      }
      q_tile[tiled_column][half2_index] = packed;
    }
  }
  __syncthreads();

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kTileAttentionQueryRows);
  for (std::uint32_t key_start = 0; key_start < max_visible;
       key_start += kTileAttentionKeyRows) {
    float scores[columns_per_wave][2] = {};
#pragma unroll
    for (std::uint32_t key_dimension = 0; key_dimension < 2; ++key_dimension) {
      for (std::uint32_t key_row = wave; key_row < kTileAttentionKeyRows;
           key_row += 8) {
        const std::uint32_t key_position = key_start + key_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionKeyHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_key =
                reinterpret_cast<const __half2*>(k_cache_f16);
            packed = layer_key
                [(static_cast<std::size_t>(key_position) * kv_half2_stride) +
                 (static_cast<std::size_t>(kv_head) * kTileAttentionHalf2Dim) +
                 (key_dimension * kTileAttentionKeyHalf2Dim) + half2_index];
          }
          kv_tile[(key_row * kTileAttentionKeyRowStride) + half2_index] =
              packed;
        }
      }
      __syncthreads();

#pragma unroll 4
      for (std::uint32_t half2_index = 0;
           half2_index < kTileAttentionKeyHalf2Dim; ++half2_index) {
        const __half2 key_values[2] = {
            kv_tile[(lane * kTileAttentionKeyRowStride) + half2_index],
            kv_tile[((lane + 32) * kTileAttentionKeyRowStride) + half2_index]};
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          const __half2 query_value =
              q_tile[tiled_column]
                    [(key_dimension * kTileAttentionKeyHalf2Dim) + half2_index];
          TileHalf2Dot(scores[column][0], key_values[0], query_value);
          TileHalf2Dot(scores[column][1], key_values[1], query_value);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
      const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
      const std::uint32_t query_offset =
          tiled_column / kTileAttentionHeadsPerBlock;
      const std::uint32_t local_query_position = query_start + query_offset;
      const std::uint32_t absolute_query_position =
          start_pos + local_query_position;
      const std::uint32_t key_positions[2] = {key_start + lane,
                                              key_start + lane + 32};
      const bool valid[2] = {local_query_position < batch_size &&
                                 key_positions[0] <= absolute_query_position &&
                                 key_positions[0] < context_end,
                             local_query_position < batch_size &&
                                 key_positions[1] <= absolute_query_position &&
                                 key_positions[1] < context_end};
      const float score_values[2] = {valid[0] ? scores[column][0] : -INFINITY,
                                     valid[1] ? scores[column][1] : -INFINITY};
      const float tile_max =
          TileWaveReduceMax(fmaxf(score_values[0], score_values[1]));
      const float next_max = fmaxf(running_max[column], tile_max);
      const float prior_scale = isfinite(running_max[column])
                                    ? expf(running_max[column] - next_max)
                                    : 0.0F;
      const float weight_values[2] = {
          valid[0] ? expf(score_values[0] - next_max) : 0.0F,
          valid[1] ? expf(score_values[1] - next_max) : 0.0F};
      running_max[column] = next_max;
      running_sum[column] = (running_sum[column] * prior_scale) +
                            weight_values[0] + weight_values[1];
#pragma unroll
      for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
        output[column][slot].x *= prior_scale;
        output[column][slot].y *= prior_scale;
      }
      weights[tiled_column][lane] = __float2half_rn(weight_values[0]);
      weights[tiled_column][lane + 32] = __float2half_rn(weight_values[1]);
    }
    __syncthreads();

    for (std::uint32_t value_group = 0; value_group < 2; ++value_group) {
      for (std::uint32_t value_row = wave; value_row < kTileAttentionValueRows;
           value_row += 8) {
        const std::uint32_t key_position =
            key_start + (value_group * kTileAttentionValueRows) + value_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const auto* layer_value =
                reinterpret_cast<const __half2*>(v_cache_f16);
            packed = layer_value
                [(static_cast<std::size_t>(key_position) * kv_half2_stride) +
                 (static_cast<std::size_t>(kv_head) * kTileAttentionHalf2Dim) +
                 half2_index];
          }
          kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll
      for (std::uint32_t value_row = 0; value_row < kTileAttentionValueRows;
           value_row += 2) {
        __half2 weight_values[columns_per_wave];
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          weight_values[column] = __halves2half2(
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row],
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row + 1]);
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32;
             ++slot) {
          const std::uint32_t half2_index = lane + (slot * 32);
          const __half2 value0 =
              kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index];
          const __half2 value1 =
              kv_tile[((value_row + 1) * kTileAttentionHalf2Dim) + half2_index];
          const __half2 values_x =
              __halves2half2(__low2half(value0), __low2half(value1));
          const __half2 values_y =
              __halves2half2(__high2half(value0), __high2half(value1));
#pragma unroll
          for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
            TileHalf2Dot(output[column][slot].x, values_x,
                         weight_values[column]);
            TileHalf2Dot(output[column][slot].y, values_y,
                         weight_values[column]);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
    const float denominator = TileWaveReduceSum(running_sum[column]);
    if (local_query_position >= batch_size || denominator == 0.0F) {
      continue;
    }
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      const std::uint32_t half2_index = lane + (slot * 32);
      const std::uint32_t dimension = 2 * half2_index;
      const std::size_t output_offset =
          (static_cast<std::size_t>(local_query_position) * attention_width) +
          (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
          dimension;
      const float2 result = output[column][slot];
      const float gate0 =
          gate == nullptr ? 1.0F : 1.0F / (1.0F + expf(-gate[output_offset]));
      const float gate1 = gate == nullptr
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset + 1]));
      out_context[output_offset] = (result.x / denominator) * gate0;
      out_context[output_offset + 1] = (result.y / denominator) * gate1;
    }
  }
}

bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = k_cache_f16 != nullptr,
      .has_v_cache_f16 = v_cache_f16 != nullptr,
      .has_scratch_f16 = false,
  };
  if (!detail::IsTiledAttentionSupported(support_params)) {
    return false;
  }

  constexpr std::size_t block_size = 256;
  constexpr std::size_t kv_width =
      kTileAttentionKvHeads * kTileAttentionHeadDim;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  auto* layer_k = static_cast<__half*>(k_cache_f16) + layer_cache_offset;
  auto* layer_v = static_cast<__half*>(v_cache_f16) + layer_cache_offset;
  const dim3 pack_grid(
      static_cast<unsigned int>((kv_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  if (start_pos > 0 && k_cache != nullptr && v_cache != nullptr) {
    const dim3 sync_grid(
        static_cast<unsigned int>((kv_width + block_size - 1) / block_size),
        static_cast<unsigned int>(start_pos));
    SyncTiledAttentionKvPrefixKernel<<<sync_grid, block_size, 0, stream>>>(
        k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, start_pos, max_context,
        num_kv_heads, head_dim);
  }
  PackTiledAttentionKvKernel<<<pack_grid, block_size, 0, stream>>>(
      k, v, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx, start_pos,
      batch_size, max_context);

  const dim3 attention_grid(
      static_cast<unsigned int>((batch_size + kTileAttentionQueryRows - 1) /
                                kTileAttentionQueryRows),
      kTileAttentionKvHeads *
          (kTileAttentionGqa / kTileAttentionHeadsPerBlock));
  const dim3 attention_block(32, 8);
  QwenTiledAttentionKernel<<<attention_grid, attention_block, 0, stream>>>(
      q, gate, layer_k, layer_v, out_context, start_pos, batch_size);
  return true;
}

__global__ void ApplyCkAttentionGateKernel(const float* __restrict__ gate,
                                           float* __restrict__ output,
                                           std::size_t num_elements) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float gate_value = gate[index];
  output[index] *= 1.0F / (1.0F + expf(-gate_value));
}

bool LaunchBatchedAttentionCk(const float* q, const float* k, const float* v,
                              const float* gate, float* k_cache, float* v_cache,
                              void* k_cache_f16, void* v_cache_f16,
                              void* scratch_f16, float* out_context,
                              std::uint32_t layer_idx, std::uint32_t start_pos,
                              std::size_t batch_size, std::uint32_t max_context,
                              std::uint32_t num_heads,
                              std::uint32_t num_kv_heads,
                              std::uint32_t head_dim, hipStream_t stream) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = k_cache_f16 != nullptr,
      .has_v_cache_f16 = v_cache_f16 != nullptr,
      .has_scratch_f16 = scratch_f16 != nullptr,
  };
  if (!detail::IsCkAttentionSupported(support_params)) {
    return false;
  }

  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  auto* q_f16 = static_cast<__half*>(scratch_f16);
  constexpr std::size_t block_size = 256;
  const std::size_t pack_width = std::max(attention_width, kv_width);
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  const auto* layer_k =
      static_cast<const ck::half_t*>(k_cache_f16) + layer_cache_offset;
  const auto* layer_v =
      static_cast<const ck::half_t*>(v_cache_f16) + layer_cache_offset;
  CkGqaOperation operation;
  const auto argument = operation.MakeArgument(
      reinterpret_cast<const ck::half_t*>(q_f16), layer_k, layer_v, out_context,
      static_cast<ck::index_t>(batch_size),
      static_cast<ck::index_t>(batch_size), static_cast<ck::index_t>(head_dim),
      static_cast<ck::index_t>(head_dim), 1,
      static_cast<ck::index_t>(num_heads),
      1.0F / sqrtf(static_cast<float>(head_dim)), true, true);
  if (!operation.IsSupportedArgument(argument)) {
    return false;
  }

  const dim3 pack_grid(
      static_cast<unsigned int>((pack_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  PackCkAttentionInputsKernel<<<pack_grid, block_size, 0, stream>>>(
      q, k, v, q_f16, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), k_cache, v_cache, layer_idx, start_pos,
      batch_size, max_context, num_heads, num_kv_heads, head_dim);

  auto invoker = operation.MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});

  const std::size_t num_elements = batch_size * attention_width;
  if (gate != nullptr) {
    const std::size_t output_grid =
        (num_elements + block_size - 1) / block_size;
    ApplyCkAttentionGateKernel<<<output_grid, block_size, 0, stream>>>(
        gate, out_context, num_elements);
  }
  return true;
}

__global__ void CausalSoftmaxKernel(float* __restrict__ scores,
                                    std::uint32_t start_pos,
                                    std::size_t batch_size,
                                    std::size_t sequence_length,
                                    std::size_t matrix_stride) {
  const std::size_t query = blockIdx.x;
  const std::size_t matrix = blockIdx.y;
  const std::size_t tid = threadIdx.x;
  if (query >= batch_size) {
    return;
  }

  float* column = scores + (matrix * matrix_stride) + (query * sequence_length);
  const std::size_t visible = static_cast<std::size_t>(start_pos) + query + 1;
  __shared__ float scratch[256];

  float local_max = -INFINITY;
  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    local_max = fmaxf(local_max, column[key]);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];

  float local_sum = 0.0F;
  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    const float value = expf(column[key] - max_score);
    column[key] = value;
    local_sum += value;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / (scratch[0] + 1e-9F);

  for (std::size_t key = tid; key < visible; key += blockDim.x) {
    column[key] *= inverse_sum;
  }
  for (std::size_t key = visible + tid; key < sequence_length;
       key += blockDim.x) {
    column[key] = 0.0F;
  }
}

__global__ void ApplyAttentionGateKernel(const float* __restrict__ gate,
                                         float* __restrict__ out,
                                         std::size_t num_elements) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= num_elements) {
    return;
  }
  const float value = gate[index];
  out[index] *= 1.0F / (1.0F + expf(-value));
}

void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* scores, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream) {
  if (batch_size == 0) {
    return;
  }
  const std::size_t sequence_length =
      static_cast<std::size_t>(start_pos) + batch_size;
  if (sequence_length > max_context) {
    return;
  }
  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const dim3 kv_grid(num_kv_heads, static_cast<unsigned int>(batch_size));
  WriteBatchedKVCacheKernel<<<kv_grid, 256, 0, stream>>>(
      k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
      static_cast<__half*>(v_cache_f16), layer_idx, start_pos, batch_size,
      max_context, num_kv_heads, head_dim);

  const float attention_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  constexpr float one = 1.0F;
  constexpr float zero = 0.0F;
  const std::size_t head_stride =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t cache_head_stride =
      static_cast<std::size_t>(max_context) * head_dim;
  const std::uint32_t heads_per_kv = num_heads / num_kv_heads;
  const std::size_t score_matrix_stride = sequence_length * batch_size;
  constexpr std::size_t grouped_attention_min_batch = 2048;

  if (batch_size < grouped_attention_min_batch) {
    for (std::uint32_t head = 0; head < num_heads; ++head) {
      const std::uint32_t kv_head = head / heads_per_kv;
      const std::size_t cache_head =
          (static_cast<std::size_t>(layer_idx) * num_kv_heads) + kv_head;
      const float* key_head = k_cache + (cache_head * cache_head_stride);
      const float* value_head = v_cache + (cache_head * cache_head_stride);
      const float* query_head = q + (static_cast<std::size_t>(head) * head_dim);
      float* output_head =
          out_context + (static_cast<std::size_t>(head) * head_dim);

      HIPBLAS_CHECK(hipblasGemmEx(
          handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(sequence_length),
          static_cast<int>(batch_size), static_cast<int>(head_dim),
          &attention_scale, key_head, HIP_R_32F, static_cast<int>(head_dim),
          query_head, HIP_R_32F, static_cast<int>(head_stride), &zero, scores,
          HIP_R_32F, static_cast<int>(sequence_length), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));

      const dim3 softmax_grid(static_cast<unsigned int>(batch_size), 1);
      CausalSoftmaxKernel<<<softmax_grid, 256, 0, stream>>>(
          scores, start_pos, batch_size, sequence_length, score_matrix_stride);

      HIPBLAS_CHECK(hipblasGemmEx(
          handle, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
          static_cast<int>(batch_size), static_cast<int>(sequence_length), &one,
          value_head, HIP_R_32F, static_cast<int>(head_dim), scores, HIP_R_32F,
          static_cast<int>(sequence_length), &zero, output_head, HIP_R_32F,
          static_cast<int>(head_stride), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));
    }
  } else {
    for (std::uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
      const std::uint32_t first_head = kv_head * heads_per_kv;
      const std::size_t cache_head =
          (static_cast<std::size_t>(layer_idx) * num_kv_heads) + kv_head;
      const float* key_head = k_cache + (cache_head * cache_head_stride);
      const float* value_head = v_cache + (cache_head * cache_head_stride);
      const float* query_group =
          q + (static_cast<std::size_t>(first_head) * head_dim);
      float* output_group =
          out_context + (static_cast<std::size_t>(first_head) * head_dim);

      HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
          handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(sequence_length),
          static_cast<int>(batch_size), static_cast<int>(head_dim),
          &attention_scale, key_head, HIP_R_32F, static_cast<int>(head_dim), 0,
          query_group, HIP_R_32F, static_cast<int>(head_stride),
          static_cast<hipblasStride>(head_dim), &zero, scores, HIP_R_32F,
          static_cast<int>(sequence_length),
          static_cast<hipblasStride>(score_matrix_stride),
          static_cast<int>(heads_per_kv), HIPBLAS_COMPUTE_32F,
          HIPBLAS_GEMM_DEFAULT));

      const dim3 softmax_grid(static_cast<unsigned int>(batch_size),
                              heads_per_kv);
      CausalSoftmaxKernel<<<softmax_grid, 256, 0, stream>>>(
          scores, start_pos, batch_size, sequence_length, score_matrix_stride);

      HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
          handle, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(head_dim),
          static_cast<int>(batch_size), static_cast<int>(sequence_length), &one,
          value_head, HIP_R_32F, static_cast<int>(head_dim), 0, scores,
          HIP_R_32F, static_cast<int>(sequence_length),
          static_cast<hipblasStride>(score_matrix_stride), &zero, output_group,
          HIP_R_32F, static_cast<int>(head_stride),
          static_cast<hipblasStride>(head_dim), static_cast<int>(heads_per_kv),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
  }

  if (gate != nullptr) {
    const std::size_t num_elements = batch_size * head_stride;
    constexpr std::size_t block_size = 256;
    const std::size_t grid_size = (num_elements + block_size - 1) / block_size;
    ApplyAttentionGateKernel<<<grid_size, block_size, 0, stream>>>(
        gate, out_context, num_elements);
  }
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
