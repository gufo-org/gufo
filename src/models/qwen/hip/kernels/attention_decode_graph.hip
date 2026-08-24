#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace strix::hip {

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
