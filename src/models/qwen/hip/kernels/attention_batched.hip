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
