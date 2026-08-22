#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/qwen_gpu_ops.hpp"
#include "src/core/hip/qwen_gpu_quant_ops.hpp"

namespace strix::hip {

// Quant block layouts (Q8_0 / Q5_K / Q6_K / Q8_K) and the quant helpers are
// shared via src/core/hip/qwen_gpu_quant_ops.hpp.

__global__ void BatchedEmbeddingLookupKernel(
    const void* __restrict__ table, core::GgmlType type,
    const std::uint32_t* __restrict__ token_ids, float* __restrict__ out_hidden,
    std::size_t batch_size, std::size_t hidden_size) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;
  const std::size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= hidden_size)
    return;

  const std::uint32_t tok = token_ids[b];
  float* out_row = out_hidden + (b * hidden_size);
  const std::size_t row_off = static_cast<std::size_t>(tok) * hidden_size;

  if (type == core::GgmlType::kBF16) {
    const auto* table_bf16 = static_cast<const hip_bfloat16*>(table);
    out_row[tid] = static_cast<float>(table_bf16[row_off + tid]);
  } else if (type == core::GgmlType::kF32) {
    const auto* table_f32 = static_cast<const float*>(table);
    out_row[tid] = table_f32[row_off + tid];
  } else {
    const auto* q8 = static_cast<const Q8_0Block*>(table);
    const Q8_0Block& wblk =
        q8[row_off / kQ8_0BlockSize + tid / kQ8_0BlockSize];
    out_row[tid] = __half2float(wblk.d) *
                   static_cast<float>(wblk.qs[tid % kQ8_0BlockSize]);
  }
}

void LaunchBatchedEmbeddingLookup(const void* table, core::GgmlType type,
                                  const std::uint32_t* token_ids,
                                  float* out_hidden, std::size_t batch_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_x = (hidden_size + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedEmbeddingLookupKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                                 stream>>>(table, type, token_ids, out_hidden,
                                           batch_size, hidden_size);
}

__global__ void BatchedRMSNormKernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ out,
                                     hip_bfloat16* __restrict__ out_bf16,
                                     std::size_t batch_size, std::size_t dim,
                                     float eps) {
  const std::size_t b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;
  const float* in_row = x + (b * dim);
  float* out_row = (out != nullptr) ? out + (b * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * dim) : nullptr;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += block_size) {
    const float val = in_row[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = block_size / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += block_size) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    const float normed = (in_row[i] * inv_rms) * w;
    if (out_row != nullptr) {
      out_row[i] = normed;
    }
    if (out_bf16_row != nullptr) {
      out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps, hipStream_t stream) {
  BatchedRMSNormKernel<<<batch_size, 256, 0, stream>>>(
      x, weight, out, static_cast<hip_bfloat16*>(out_bf16), batch_size, dim,
      eps);
}

__global__ void BatchedPerHeadRMSNormKernel(const float* __restrict__ x,
                                            const float* __restrict__ weight,
                                            float* __restrict__ out,
                                            std::size_t batch_size,
                                            std::uint32_t num_heads,
                                            std::uint32_t head_dim, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  if (b >= batch_size || h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t head_total = static_cast<std::size_t>(num_heads) * head_dim;
  const float* x_head =
      x + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);
  float* out_head =
      out + (b * head_total) + (static_cast<std::size_t>(h) * head_dim);

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(head_dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    out_head[i] = (x_head[i] * inv_rms) * w;
  }
}

void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps,
                                 hipStream_t stream) {
  const dim3 grid(num_heads, static_cast<unsigned int>(batch_size));
  BatchedPerHeadRMSNormKernel<<<grid, 256, 0, stream>>>(
      x, weight, out, batch_size, num_heads, head_dim, eps);
}
// opt-c010-qk-rope-kv: batched variant of FusedQKNormRoPEKvWriteKernel.
// Block (h, b) handles token b; blocks [0, num_heads) process Q heads in place,
// blocks [num_heads, num_heads + num_kv_heads) process K heads, write K/V into
// the FP32+FP16 KV caches, and store the normed+roped K back for downstream
// consumers (tiled/CK packing reads the K buffer).
__launch_bounds__(256, 4) __global__ void BatchedFusedQKNormRoPEKvWriteKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ k_cache,
    float* __restrict__ v_cache, __half* __restrict__ k_cache_f16,
    __half* __restrict__ v_cache_f16, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps) {
  const std::size_t b = blockIdx.y;
  const std::uint32_t block = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  if (b >= batch_size) {
    return;
  }

  const std::uint32_t half_rot = rotary_dim / 2;
  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(b);
  __shared__ float s_norm[256];

  const std::size_t head_total = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_total =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::uint32_t num_q_blocks = num_heads;

  if (block < num_q_blocks) {
    const std::size_t head_base =
        (b * head_total) + (static_cast<std::size_t>(block) * head_dim);
    const float* x_head = q + head_base;

    float sum_sq = 0.0F;
    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float val = x_head[i];
      sum_sq += val * val;
    }
    s_norm[tid] = sum_sq;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_norm[tid] += s_norm[tid + stride];
      }
      __syncthreads();
    }
    const float inv_rms =
        rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
    __syncthreads();

    for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
      const float w = (q_weight != nullptr) ? q_weight[i] : 1.0F;
      s_norm[i] = (x_head[i] * inv_rms) * w;
    }
    __syncthreads();

    for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
      const float q0 = s_norm[p];
      const float q1 = s_norm[p + half_rot];
      const float exponent =
          static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
      const float inv_freq = 1.0F / powf(rope_theta, exponent);
      const float angle = static_cast<float>(cur_pos) * inv_freq;
      const float cos_val = cosf(angle);
      const float sin_val = sinf(angle);
      q_out[head_base + p] = (q0 * cos_val) - (q1 * sin_val);
      q_out[head_base + p + half_rot] = (q0 * sin_val) + (q1 * cos_val);
    }
    for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
      q_out[head_base + i] = s_norm[i];
    }
    return;
  }

  const std::uint32_t kv_h = block - num_q_blocks;
  if (kv_h >= num_kv_heads) {
    return;
  }

  const std::size_t head_base =
      (b * kv_total) + (static_cast<std::size_t>(kv_h) * head_dim);
  const float* x_head = k + head_base;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = x_head[i];
    sum_sq += val * val;
  }
  s_norm[tid] = sum_sq;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_norm[tid] += s_norm[tid + stride];
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(s_norm[0] / static_cast<float>(head_dim) + eps);
  __syncthreads();

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float w = (k_weight != nullptr) ? k_weight[i] : 1.0F;
    s_norm[i] = (x_head[i] * inv_rms) * w;
  }
  __syncthreads();

  const std::size_t cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        cur_pos) *
       head_dim);
  const std::size_t f16_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + cur_pos) *
       kv_total) +
      (static_cast<std::size_t>(kv_h) * head_dim);

  for (std::size_t p = tid; p < half_rot; p += blockDim.x) {
    const float k0 = s_norm[p];
    const float k1 = s_norm[p + half_rot];
    const float exponent =
        static_cast<float>(2 * p) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0F / powf(rope_theta, exponent);
    const float angle = static_cast<float>(cur_pos) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float k0_rot = (k0 * cos_val) - (k1 * sin_val);
    const float k1_rot = (k0 * sin_val) + (k1 * cos_val);
    k_out[head_base + p] = k0_rot;
    k_out[head_base + p + half_rot] = k1_rot;
    k_cache[cache_offset + p] = k0_rot;
    k_cache[cache_offset + p + half_rot] = k1_rot;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + p] = __float2half_rn(k0_rot);
      k_cache_f16[f16_offset + p + half_rot] = __float2half_rn(k1_rot);
    }
  }
  for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
    const float key = s_norm[i];
    k_out[head_base + i] = key;
    k_cache[cache_offset + i] = key;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + i] = __float2half_rn(key);
    }
  }

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = v[head_base + i];
    v_cache[cache_offset + i] = val;
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset + i] = __float2half_rn(val);
    }
  }
}

void LaunchBatchedFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps, hipStream_t stream) {
  const dim3 grid(num_heads + num_kv_heads,
                  static_cast<unsigned int>(batch_size));
  BatchedFusedQKNormRoPEKvWriteKernel<<<grid, 256, 0, stream>>>(
      q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
      static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16),
      layer_idx, start_pos, batch_size, max_context, num_heads, num_kv_heads,
      head_dim, rotary_dim, rope_theta, eps);
}

__global__ void BatchedResidualAddKernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         std::size_t total_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < total_elements) {
    out[idx] = a[idx] + b[idx];
  }
}

void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream) {
  const std::size_t total = batch_size * dim;
  const std::size_t block_size = 256;
  const std::size_t grid_size = (total + block_size - 1) / block_size;
  BatchedResidualAddKernel<<<grid_size, block_size, 0, stream>>>(a, b, out,
                                                                 total);
}

// opt-c010-residual-rmsnorm: batched variant of FusedResidualAddRMSNormKernel.
// Block b handles token b; writes the residual-updated sum row, the FP32
// normed row, and the optional BF16 normed row. Reduction order matches the
// unfused chain (BatchedResidualAddKernel + BatchedRMSNormKernel).
__launch_bounds__(256, 4) __global__ void BatchedFusedResidualAddRMSNormKernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out_sum, const float* __restrict__ weight,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t batch_size, std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= batch_size) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const float* a_row = a + (row * dim);
  const float* b_row = b + (row * dim);
  float* sum_row = (out_sum != nullptr) ? out_sum + (row * dim) : nullptr;
  float* out_row = (out != nullptr) ? out + (row * dim) : nullptr;
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (row * dim) : nullptr;

  float sum_sq = 0.0F;
  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float val = a_row[i] + b_row[i];
    if (sum_row != nullptr) {
      sum_row[i] = val;
    }
    sum_sq += val * val;
  }

  __shared__ float s_sum[256];
  s_sum[tid] = sum_sq;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float mean_sq = s_sum[0] / static_cast<float>(dim);
  const float inv_rms = rsqrtf(mean_sq + eps);

  for (std::size_t i = tid; i < dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    const float normed = (sum_row[i] * inv_rms) * w;
    if (out_row != nullptr) {
      out_row[i] = normed;
    }
    if (out_bf16_row != nullptr) {
      out_bf16_row[i] = hip_bfloat16(normed);
    }
  }
}

void LaunchBatchedFusedResidualAddRMSNorm(const float* a, const float* b,
                                          float* out_sum, const float* weight,
                                          float* out, void* out_bf16,
                                          std::size_t batch_size,
                                          std::size_t dim, float eps,
                                          hipStream_t stream) {
  BatchedFusedResidualAddRMSNormKernel<<<batch_size, 256, 0, stream>>>(
      a, b, out_sum, weight, out, static_cast<hip_bfloat16*>(out_bf16),
      batch_size, dim, eps);
}

__global__ void BatchedUnpackQGKernel(const float* __restrict__ qg,
                                      float* __restrict__ q_out,
                                      float* __restrict__ gate_out,
                                      std::size_t batch_size,
                                      std::uint32_t num_heads,
                                      std::uint32_t head_dim) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;

  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  if (idx >= total_h)
    return;

  const std::size_t head_idx = idx / head_dim;
  const std::size_t dim_idx = idx % head_dim;
  const std::size_t src_idx =
      (b * total_h * 2) + (head_idx * 2 * head_dim) + dim_idx;

  q_out[(b * total_h) + idx] = qg[src_idx];
  gate_out[(b * total_h) + idx] = qg[src_idx + head_dim];
}

void LaunchBatchedUnpackQG(const float* qg_interleaved, float* q_out,
                           float* gate_out, std::size_t batch_size,
                           std::uint32_t num_heads, std::uint32_t head_dim,
                           hipStream_t stream) {
  const std::size_t total_h = static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_h + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedUnpackQGKernel<<<grid, static_cast<unsigned int>(block_size), 0,
                          stream>>>(qg_interleaved, q_out, gate_out, batch_size,
                                    num_heads, head_dim);
}

__global__ void BatchedRoPEKernel(
    float* __restrict__ q, float* __restrict__ k, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t start_pos, float rope_theta) {
  const std::size_t b = blockIdx.y;
  if (b >= batch_size)
    return;

  const std::size_t pair_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t half_rot = rotary_dim / 2;
  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(b);

  if (pair_idx < static_cast<std::size_t>(num_heads) * half_rot) {
    const std::size_t h = pair_idx / half_rot;
    const std::size_t i = pair_idx % half_rot;
    const float freq =
        1.0F / powf(rope_theta,
                    static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(cur_pos) * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t q_stride = static_cast<std::size_t>(num_heads) * head_dim;
    float* q_head = q + (b * q_stride) + (h * head_dim);
    const float q0 = q_head[i];
    const float q1 = q_head[i + half_rot];
    q_head[i] = (q0 * cos_val) - (q1 * sin_val);
    q_head[i + half_rot] = (q0 * sin_val) + (q1 * cos_val);
  }

  if (pair_idx < static_cast<std::size_t>(num_kv_heads) * half_rot) {
    const std::size_t h = pair_idx / half_rot;
    const std::size_t i = pair_idx % half_rot;
    const float freq =
        1.0F / powf(rope_theta,
                    static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    const float angle = static_cast<float>(cur_pos) * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const std::size_t k_stride =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    float* k_head = k + (b * k_stride) + (h * head_dim);
    const float k0 = k_head[i];
    const float k1 = k_head[i + half_rot];
    k_head[i] = (k0 * cos_val) - (k1 * sin_val);
    k_head[i + half_rot] = (k0 * sin_val) + (k1 * cos_val);
  }
}

void LaunchBatchedRoPE(float* q, float* k, std::size_t batch_size,
                       std::uint32_t num_heads, std::uint32_t num_kv_heads,
                       std::uint32_t head_dim, std::uint32_t rotary_dim,
                       std::uint32_t start_pos, float rope_theta,
                       hipStream_t stream) {
  const std::size_t half_rot = rotary_dim / 2;
  const std::size_t total_pairs =
      std::max(static_cast<std::size_t>(num_heads) * half_rot,
               static_cast<std::size_t>(num_kv_heads) * half_rot);
  const std::size_t block_size = 256;
  const std::size_t grid_x = (total_pairs + block_size - 1) / block_size;
  const dim3 grid(static_cast<unsigned int>(grid_x),
                  static_cast<unsigned int>(batch_size));
  BatchedRoPEKernel<<<grid, static_cast<unsigned int>(block_size), 0, stream>>>(
      q, k, batch_size, num_heads, num_kv_heads, head_dim, rotary_dim,
      start_pos, rope_theta);
}

__global__ void BatchedGEMMKernel(const void* __restrict__ A, bool is_bf16,
                                  const float* __restrict__ X,
                                  float* __restrict__ Y, std::size_t batch_size,
                                  std::size_t M, std::size_t K) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= M || b >= batch_size)
    return;

  const std::size_t tid = threadIdx.x;
  const std::size_t block_threads = blockDim.x;
  const float* x_row = X + (b * K);
  float* y_row = Y + (b * M);

  float dot = 0.0F;
  if (is_bf16) {
    const auto* row = static_cast<const hip_bfloat16*>(A) + (m * K);
    const std::size_t K_vec = K / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < K; k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (m * K);
    const std::size_t K_vec = K / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < K; k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      y_row[m] = block_sum;
    }
  }
}

void LaunchBatchedGEMM(const void* A, bool is_bf16, const float* X, float* Y,
                       std::size_t batch_size, std::size_t M, std::size_t K,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }
  const dim3 grid(static_cast<unsigned int>(M),
                  static_cast<unsigned int>(batch_size));
  BatchedGEMMKernel<<<grid, 128, 0, stream>>>(A, is_bf16, X, Y, batch_size, M,
                                              K);
}

__global__ void FloatToBfloat16Kernel(const float* __restrict__ in,
                                      hip_bfloat16* __restrict__ out,
                                      std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    out[idx] = hip_bfloat16(in[idx]);
  }
}

void LaunchFloatToBfloat16(const float* in, void* out, std::size_t num_elements,
                           hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  FloatToBfloat16Kernel<<<num_blocks, block_size, 0, stream>>>(
      in, static_cast<hip_bfloat16*>(out), num_elements);
}

__global__ void BatchedSwiGLUActivationKernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t num_elements) {
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < num_elements) {
    const float g = gate[idx];
    const float u = up[idx];
    const float sig_g = 1.0F / (1.0F + expf(-g));
    const float val = (g * sig_g) * u;
    if (out != nullptr) {
      out[idx] = val;
    }
    if (out_bf16 != nullptr) {
      out_bf16[idx] = hip_bfloat16(val);
    }
  }
}

void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream) {
  if (num_elements == 0) {
    return;
  }
  const unsigned int block_size = 256;
  const unsigned int num_blocks =
      static_cast<unsigned int>((num_elements + block_size - 1) / block_size);
  BatchedSwiGLUActivationKernel<<<num_blocks, block_size, 0, stream>>>(
      gate, up, out, static_cast<hip_bfloat16*>(out_bf16), num_elements);
}

__global__ void BatchedFusedSSMInputProjectionsKernel(
    const void* __restrict__ qkv_w, bool qkv_is_bf16,
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ alpha_w, bool alpha_is_bf16,
    const void* __restrict__ beta_w, bool beta_is_bf16,
    const float* __restrict__ X, float* __restrict__ qkv_out,
    float* __restrict__ gate_out, float* __restrict__ alpha_out,
    float* __restrict__ beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  const std::size_t gate_end = qkv_size + inner_size;
  const std::size_t alpha_end = gate_end + time_step_rank;
  const std::size_t total_rows = alpha_end + time_step_rank;
  if (m >= total_rows || b >= batch_size)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < qkv_size) {
    A = qkv_w;
    is_bf16 = qkv_is_bf16;
    out_target = qkv_out + (b * qkv_size);
    row_idx = m;
  } else if (m < gate_end) {
    A = gate_w;
    is_bf16 = gate_is_bf16;
    out_target = gate_out + (b * inner_size);
    row_idx = m - qkv_size;
  } else if (m < alpha_end) {
    A = alpha_w;
    is_bf16 = alpha_is_bf16;
    out_target = alpha_out + (b * time_step_rank);
    row_idx = m - gate_end;
  } else {
    A = beta_w;
    is_bf16 = beta_is_bf16;
    out_target = beta_out + (b * time_step_rank);
    row_idx = m - alpha_end;
  }

  const float* x_row = X + (b * hidden_size);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

void LaunchBatchedFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* X, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream) {
  const std::size_t total_rows = qkv_size + inner_size + (2 * time_step_rank);
  const dim3 grid(static_cast<unsigned int>(total_rows),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedSSMInputProjectionsKernel<<<grid, 128, 0, stream>>>(
      qkv_w, qkv_is_bf16, gate_w, gate_is_bf16, alpha_w, alpha_is_bf16, beta_w,
      beta_is_bf16, X, qkv_out, gate_out, alpha_out, beta_out, batch_size,
      hidden_size, qkv_size, inner_size, time_step_rank);
}

__global__ void BatchedFusedQKVProjectionsKernel(
    const void* __restrict__ q_w, bool q_is_bf16, const void* __restrict__ k_w,
    bool k_is_bf16, const void* __restrict__ v_w, bool v_is_bf16,
    const float* __restrict__ X, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ v_out,
    std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  if (m >= total_rows || b >= batch_size)
    return;

  const void* A = nullptr;
  bool is_bf16 = false;
  float* out_target = nullptr;
  std::size_t row_idx = 0;

  if (m < q_dim) {
    A = q_w;
    is_bf16 = q_is_bf16;
    out_target = q_out + (b * q_dim);
    row_idx = m;
  } else if (m < q_dim + kv_dim) {
    A = k_w;
    is_bf16 = k_is_bf16;
    out_target = k_out + (b * kv_dim);
    row_idx = m - q_dim;
  } else {
    A = v_w;
    is_bf16 = v_is_bf16;
    out_target = v_out + (b * kv_dim);
    row_idx = m - q_dim - kv_dim;
  }

  const float* x_row = X + (b * hidden_size);
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot = 0.0F;

  if (is_bf16) {
    const auto* row =
        static_cast<const hip_bfloat16*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* row_vec = reinterpret_cast<const uint4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 v = row_vec[vi];
      const auto* vals = reinterpret_cast<const hip_bfloat16*>(&v);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot += (static_cast<float>(vals[0]) * x0.x) +
             (static_cast<float>(vals[1]) * x0.y) +
             (static_cast<float>(vals[2]) * x0.z) +
             (static_cast<float>(vals[3]) * x0.w) +
             (static_cast<float>(vals[4]) * x1.x) +
             (static_cast<float>(vals[5]) * x1.y) +
             (static_cast<float>(vals[6]) * x1.z) +
             (static_cast<float>(vals[7]) * x1.w);
    }
    for (std::size_t k = (K_vec * 8) + tid; k < hidden_size;
         k += block_threads) {
      dot += static_cast<float>(row[k]) * x_row[k];
    }
  } else {
    const auto* row = static_cast<const float*>(A) + (row_idx * hidden_size);
    const std::size_t K_vec = hidden_size / 4;
    const auto* row_vec = reinterpret_cast<const float4*>(row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const float4 v = row_vec[vi];
      const std::size_t k = vi * 4;
      const float4 xv = *reinterpret_cast<const float4*>(&x_row[k]);
      dot += (v.x * xv.x) + (v.y * xv.y) + (v.z * xv.z) + (v.w * xv.w);
    }
    for (std::size_t k = (K_vec * 4) + tid; k < hidden_size;
         k += block_threads) {
      dot += row[k] * x_row[k];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_xor(dot, offset);
  }

  __shared__ float s_warp_sums[8];
  if (lane_id == 0) {
    s_warp_sums[warp_id] = dot;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_sum = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor(block_sum, offset);
    }
    if (lane_id == 0) {
      out_target[row_idx] = block_sum;
    }
  }
}

void LaunchBatchedFusedQKVProjections(
    const void* q_w, bool q_is_bf16, const void* k_w, bool k_is_bf16,
    const void* v_w, bool v_is_bf16, const float* X, float* q_out, float* k_out,
    float* v_out, std::size_t batch_size, std::size_t q_dim, std::size_t kv_dim,
    std::size_t hidden_size, hipStream_t stream) {
  const std::size_t total_rows = q_dim + (2 * kv_dim);
  const dim3 grid(static_cast<unsigned int>(total_rows),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedQKVProjectionsKernel<<<grid, 128, 0, stream>>>(
      q_w, q_is_bf16, k_w, k_is_bf16, v_w, v_is_bf16, X, q_out, k_out, v_out,
      batch_size, q_dim, kv_dim, hidden_size);
}

__global__ void BatchedFusedSwiGLUGEMMKernel(
    const void* __restrict__ gate_w, bool gate_is_bf16,
    const void* __restrict__ up_w, bool up_is_bf16, const float* __restrict__ X,
    float* __restrict__ out, hip_bfloat16* __restrict__ out_bf16,
    std::size_t batch_size, std::size_t intermediate_size,
    std::size_t hidden_size) {
  const std::size_t m = blockIdx.x;
  const std::size_t b = blockIdx.y;
  if (m >= intermediate_size || b >= batch_size)
    return;

  const float* x_row = X + (b * hidden_size);
  float* out_row = out + (b * intermediate_size);
  hip_bfloat16* out_bf16_row =
      (out_bf16 != nullptr) ? out_bf16 + (b * intermediate_size) : nullptr;
  const std::size_t tid = threadIdx.x;
  const std::size_t lane_id = tid & 31u;
  const std::size_t warp_id = tid >> 5u;
  const std::size_t block_threads = blockDim.x;

  float dot_gate = 0.0F;
  float dot_up = 0.0F;

  if (gate_is_bf16 && up_is_bf16) {
    const auto* g_row =
        static_cast<const hip_bfloat16*>(gate_w) + (m * hidden_size);
    const auto* u_row =
        static_cast<const hip_bfloat16*>(up_w) + (m * hidden_size);
    const std::size_t K_vec = hidden_size / 8;
    const auto* g_vec = reinterpret_cast<const uint4*>(g_row);
    const auto* u_vec = reinterpret_cast<const uint4*>(u_row);

    for (std::size_t vi = tid; vi < K_vec; vi += block_threads) {
      const uint4 gv = g_vec[vi];
      const uint4 uv = u_vec[vi];
      const auto* g_vals = reinterpret_cast<const hip_bfloat16*>(&gv);
      const auto* u_vals = reinterpret_cast<const hip_bfloat16*>(&uv);
      const std::size_t k = vi * 8;
      const float4 x0 = *reinterpret_cast<const float4*>(&x_row[k + 0]);
      const float4 x1 = *reinterpret_cast<const float4*>(&x_row[k + 4]);

      dot_gate += (static_cast<float>(g_vals[0]) * x0.x) +
                  (static_cast<float>(g_vals[1]) * x0.y) +
                  (static_cast<float>(g_vals[2]) * x0.z) +
                  (static_cast<float>(g_vals[3]) * x0.w) +
                  (static_cast<float>(g_vals[4]) * x1.x) +
                  (static_cast<float>(g_vals[5]) * x1.y) +
                  (static_cast<float>(g_vals[6]) * x1.z) +
                  (static_cast<float>(g_vals[7]) * x1.w);

      dot_up += (static_cast<float>(u_vals[0]) * x0.x) +
                (static_cast<float>(u_vals[1]) * x0.y) +
                (static_cast<float>(u_vals[2]) * x0.z) +
                (static_cast<float>(u_vals[3]) * x0.w) +
                (static_cast<float>(u_vals[4]) * x1.x) +
                (static_cast<float>(u_vals[5]) * x1.y) +
                (static_cast<float>(u_vals[6]) * x1.z) +
                (static_cast<float>(u_vals[7]) * x1.w);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_xor(dot_gate, offset);
    dot_up += __shfl_xor(dot_up, offset);
  }

  __shared__ float s_gate_sums[8];
  __shared__ float s_up_sums[8];
  if (lane_id == 0) {
    s_gate_sums[warp_id] = dot_gate;
    s_up_sums[warp_id] = dot_up;
  }
  __syncthreads();

  if (warp_id == 0) {
    const std::size_t num_warps = block_threads >> 5u;
    float block_gate = (lane_id < num_warps) ? s_gate_sums[lane_id] : 0.0F;
    float block_up = (lane_id < num_warps) ? s_up_sums[lane_id] : 0.0F;
    for (int offset = 4; offset > 0; offset >>= 1) {
      block_gate += __shfl_xor(block_gate, offset);
      block_up += __shfl_xor(block_up, offset);
    }
    if (lane_id == 0) {
      const float sig_g = 1.0F / (1.0F + expf(-block_gate));
      const float val = (block_gate * sig_g) * block_up;
      out_row[m] = val;
      if (out_bf16_row != nullptr) {
        out_bf16_row[m] = hip_bfloat16(val);
      }
    }
  }
}

void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out, void* out_bf16,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(intermediate_size),
                  static_cast<unsigned int>(batch_size));
  BatchedFusedSwiGLUGEMMKernel<<<grid, 128, 0, stream>>>(
      gate_w, gate_is_bf16, up_w, up_is_bf16, X, out,
      static_cast<hip_bfloat16*>(out_bf16), batch_size, intermediate_size,
      hidden_size);
}

// opt-c162-q8k-prefill-dequant: device Q8_K weight dequantization to BF16 for
// the prefill hipblas GEMM path. hipBLAS cannot consume block_q8_K weights, so
// prefill first dequantizes the weight region into a BF16 scratch buffer and
// runs the GF16 GEMM on that scratch. One block per 256-wide Q8_K block, one
// thread per element; matches the CPU DequantizeQ8_K oracle
// (out[j] = d * qs[j] for j in [0,256) per block). Decode is unaffected: it
// already computes at Q8 via Q8KRowDot.
// Q8_K block layout is defined in the shared qwen_gpu_quant_ops.hpp header.

// Q5_K / Q6_K block layouts are defined in the shared qwen_gpu_quant_ops.hpp
// header; dequant to BF16 uses Q5KValueFp / Q6KValueFp below.

__global__ void DequantizeQ8KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  const auto* blk = static_cast<const Q8KBlock*>(w) + b;
  out[(b * kQ8KBlockSize) + j] =
      hip_bfloat16(blk->d * static_cast<float>(blk->qs[j]));
}

void LaunchDequantizeQ8KToBf16(const void* w, hip_bfloat16* out,
                               std::size_t n_elems, hipStream_t stream) {
  if (n_elems == 0) {
    return;
  }
  // Requires the weight region to be a whole number of 256-wide Q8_K blocks.
  if (n_elems % kQ8KBlockSize != 0) {
    return;
  }
  const std::size_t n_blocks = n_elems / kQ8KBlockSize;
  DequantizeQ8KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                              static_cast<unsigned int>(kQ8KBlockSize), 0,
                              stream>>>(w, out, n_blocks);
}

// Per-value dequant helpers (fp32), matching the CPU oracles
// Q5Value/Q6Value/DequantizeQ8_0 in ggml_dequant.cpp. GetQKScaleMin (used by
// Q5KValueFp) is defined in the shared qwen_gpu_quant_ops.hpp header.

__device__ float Q5KValueFp(const Q5KBlock& block,
                            std::size_t index) noexcept {
  const std::size_t gg = index / 64;   // 0..3
  const std::size_t wv = index % 64;   // 0..63
  const std::size_t lane = wv % 32;    // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((block.qh[lane] >> bit) & 1U) ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);  // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQKScaleMin(sis, block.scales, sc, m);
  return __half2float(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         __half2float(block.dmin) * static_cast<float>(m);
}

__device__ float Q6KValueFp(const Q6KBlock& block,
                            std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t segment = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh_byte = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = (qh_byte >> 0U) & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh_byte >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh_byte >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh_byte >> 6U) & 0x03U;
      break;
  }
  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return __half2float(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

__device__ float Q8_0ValueFp(const Q8_0Block& block,
                             std::size_t index) noexcept {
  return __half2float(block.d) * static_cast<float>(block.qs[index]);
}

// Type-dispatched dequant-to-BF16 kernels. One block per quant block, one
// thread per element; out[b*qk + j] = (hip_bfloat16)(per_value_dequant).
__global__ void DequantizeQ8_0ToBf16Kernel(const void* __restrict__ w,
                                           hip_bfloat16* __restrict__ out,
                                           std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ8_0BlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q8_0Block*>(w) + b;
  out[(b * kQ8_0BlockSize) + j] = hip_bfloat16(Q8_0ValueFp(*blk, j));
}

__global__ void DequantizeQ5KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ5KBlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q5KBlock*>(w) + b;
  out[(b * kQ5KBlockSize) + j] = hip_bfloat16(Q5KValueFp(*blk, j));
}

__global__ void DequantizeQ6KToBf16Kernel(const void* __restrict__ w,
                                          hip_bfloat16* __restrict__ out,
                                          std::size_t n_blocks) {
  const std::size_t b = blockIdx.x;
  if (b >= n_blocks) {
    return;
  }
  const std::size_t j = threadIdx.x;
  if (j >= kQ6KBlockSize) {
    return;
  }
  const auto* blk = static_cast<const Q6KBlock*>(w) + b;
  out[(b * kQ6KBlockSize) + j] = hip_bfloat16(Q6KValueFp(*blk, j));
}

// opt-c162-q8k-prefill-dequant (extended): type-dispatched entry point that
// converts a quantized weight region (Q8_0/Q5_K/Q6_K/Q8_K) into a BF16 scratch
// for the prefill hipblas GEMM path. Q8_0 uses a 32-thread block (QK=32);
// Q5_K/Q6_K/Q8_K use a 256-thread block (QK=256). Requires n_elems to be a
// whole number of blocks. Unsupported types and non-multiple sizes are no-ops.
void LaunchDequantizeToBf16(core::GgmlType type, const void* w,
                            hip_bfloat16* out, std::size_t n_elems,
                            hipStream_t stream) {
  if (n_elems == 0) {
    return;
  }
  if (type == core::GgmlType::kQ8_0) {
    if (n_elems % kQ8_0BlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ8_0BlockSize;
    DequantizeQ8_0ToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                 static_cast<unsigned int>(kQ8_0BlockSize), 0,
                                 stream>>>(w, out, n_blocks);
    return;
  }
  if (type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
      type == core::GgmlType::kQ8_K) {
    if (n_elems % kQ5KBlockSize != 0) {
      return;
    }
    const std::size_t n_blocks = n_elems / kQ5KBlockSize;
    if (type == core::GgmlType::kQ5_K) {
      DequantizeQ5KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                  static_cast<unsigned int>(kQ5KBlockSize), 0,
                                  stream>>>(w, out, n_blocks);
    } else if (type == core::GgmlType::kQ6_K) {
      DequantizeQ6KToBf16Kernel<<<static_cast<unsigned int>(n_blocks),
                                  static_cast<unsigned int>(kQ6KBlockSize), 0,
                                  stream>>>(w, out, n_blocks);
    } else {
      // kQ8_K: reuse the existing single-type kernel (one implementation).
      LaunchDequantizeQ8KToBf16(w, out, n_elems, stream);
    }
    return;
  }
  // Unsupported quant type: no-op.
}

// opt-c162-prefill-quant-direct: batched prefill projection that computes the
// matmul DIRECTLY on quantized weights (block_q8_0 / block_q8_K / block_q5_K /
// block_q6_K) with NO dequantize-to-BF16 and NO BF16 GEMM. One warp per output
// row; each warp reads the row's quant blocks and the bf16 activation and
// reduces via QuantWarpBlockDot (num_warps=1 => full row dot in lane 0).
// Output y[b*m + row] = qdot(weights[row], x[b]).
__global__ void BatchedQuantGEMVKernel(core::GgmlType type,
                                       const void* __restrict__ w,
                                       const hip_bfloat16* __restrict__ x,
                                       float* __restrict__ y, std::size_t m,
                                       std::size_t k) {
  constexpr std::size_t kWarp = 32;
  const std::size_t warp_id = threadIdx.x / kWarp;
  const std::size_t lane = threadIdx.x % kWarp;
  const std::size_t warps_per_block = blockDim.x / kWarp;

  const std::size_t b = blockIdx.y;
  const std::size_t row = blockIdx.x * warps_per_block + warp_id;
  if (row >= m) {
    return;
  }

  const float dot = QuantWarpBlockDot<hip_bfloat16>(type, w, row, x + (b * k),
                                                    k, lane, 0, 1);
  if (lane == 0) {
    y[(b * m) + row] = dot;
  }
}

void LaunchBatchedQuantGEMM(core::GgmlType type, const void* w,
                            const void* bf16_x, float* y, std::size_t batch,
                            std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0) {
    return;
  }
  if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF32) {
    return;
  }
  constexpr std::size_t kWarp = 32;
  const std::size_t warps_per_block = 8;
  const std::size_t rows_per_block = warps_per_block;  // one row per warp
  const std::size_t num_blocks_x = (m + rows_per_block - 1) / rows_per_block;
  const dim3 grid(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(batch));
  const dim3 block(static_cast<unsigned int>(kWarp * warps_per_block));
  BatchedQuantGEMVKernel<<<grid, block, 0, stream>>>(
      type, w, reinterpret_cast<const hip_bfloat16*>(bf16_x), y, m, k);
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
