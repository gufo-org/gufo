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


}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
