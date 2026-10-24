#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {

// opt-c010-qk-rope-kv: batched variant of FusedQKNormRoPEKvWriteKernel.
// Block (h, b) handles token b; blocks [0, num_heads) process Q heads in place,
// blocks [num_heads, num_heads + num_kv_heads) process K heads, write K/V into
// every non-null selected KV representation, and store the normed+roped K back
// for downstream consumers (tiled/CK packing reads the K buffer).
__launch_bounds__(256, 4) __global__ void BatchedFusedQKNormRoPEKvWriteKernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, float* __restrict__ q_out,
    float* __restrict__ k_out, float* __restrict__ k_cache,
    float* __restrict__ v_cache, __half* __restrict__ k_cache_f16,
    __half* __restrict__ v_cache_f16, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps,
    const models::qwen::vision::DeviceRope* rope) {
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
      const float angle =
          models::qwen::vision::RopePosition(rope, cur_pos, p) * inv_freq;
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
    const float angle =
        models::qwen::vision::RopePosition(rope, cur_pos, p) * inv_freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float k0_rot = (k0 * cos_val) - (k1 * sin_val);
    const float k1_rot = (k0 * sin_val) + (k1 * cos_val);
    k_out[head_base + p] = k0_rot;
    k_out[head_base + p + half_rot] = k1_rot;
    if (k_cache != nullptr) {
      k_cache[cache_offset + p] = k0_rot;
      k_cache[cache_offset + p + half_rot] = k1_rot;
    }
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + p] = __float2half_rn(k0_rot);
      k_cache_f16[f16_offset + p + half_rot] = __float2half_rn(k1_rot);
    }
  }
  for (std::size_t i = rotary_dim + tid; i < head_dim; i += blockDim.x) {
    const float key = s_norm[i];
    k_out[head_base + i] = key;
    if (k_cache != nullptr) {
      k_cache[cache_offset + i] = key;
    }
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + i] = __float2half_rn(key);
    }
  }

  for (std::size_t i = tid; i < head_dim; i += blockDim.x) {
    const float val = v[head_base + i];
    if (v_cache != nullptr) {
      v_cache[cache_offset + i] = val;
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset + i] = __float2half_rn(val);
    }
  }
}

// One wave per (head, token) for 256-wide heads with a 64-wide rotary span.
// Lane l holds elements l + 32 * v: register v is what the threads of wave v
// held in BatchedFusedQKNormRoPEKvWriteKernel, the rotary pair (p, p + 32) is
// lane p of registers 0 and 1, and the reduction is the same descending tree.
// Every expression is unchanged, so the outputs are bit-identical; the block
// needs no LDS and no barrier. The inputs may alias the outputs: a wave reads
// its whole head before writing it.
__launch_bounds__(32) __global__ void BatchedFusedQKNormRoPEKvWriteWaveKernel(
    const float* q, const float* k, const float* __restrict__ v,
    const float* __restrict__ q_weight, const float* __restrict__ k_weight,
    float* q_out, float* k_out, float* __restrict__ k_cache,
    float* __restrict__ v_cache, __half* __restrict__ k_cache_f16,
    __half* __restrict__ v_cache_f16, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, float rope_theta,
    float eps, const models::qwen::vision::DeviceRope* rope) {
  constexpr int kWaves = 8;
  constexpr std::uint32_t kHeadDim = 256;
  constexpr std::uint32_t kRotaryDim = 64;
  constexpr std::uint32_t kHalfRot = kRotaryDim / 2;
  const std::size_t b = blockIdx.y;
  const std::uint32_t block = blockIdx.x;
  const std::size_t lane = threadIdx.x;
  if (b >= batch_size) {
    return;
  }
  const std::uint32_t cur_pos = start_pos + static_cast<std::uint32_t>(b);
  const bool is_q = block < num_heads;
  const std::uint32_t kv_h = block - num_heads;
  if (!is_q && kv_h >= num_kv_heads) {
    return;
  }
  const std::size_t head_total = static_cast<std::size_t>(num_heads) * kHeadDim;
  const std::size_t kv_total =
      static_cast<std::size_t>(num_kv_heads) * kHeadDim;
  const std::size_t head_base =
      is_q ? (b * head_total) + (static_cast<std::size_t>(block) * kHeadDim)
           : (b * kv_total) + (static_cast<std::size_t>(kv_h) * kHeadDim);
  const float* x_head = (is_q ? q : k) + head_base;
  const float* weight = is_q ? q_weight : k_weight;

  float values[kWaves];
  float sums[kWaves];
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const float val = x_head[(32 * w) + lane];
    values[w] = val;
    float square = val * val;
    // The 256-thread kernel stores each square before the tree adds them;
    // keep the product rounded on its own instead of fused into the add.
    asm volatile("" : "+v"(square));
    sums[w] = square;
  }
  // Strides 128, 64 and 32 of the 256-thread tree pair whole waves; the
  // remaining strides run inside the first wave.
  sums[0] += sums[4];
  sums[1] += sums[5];
  sums[2] += sums[6];
  sums[3] += sums[7];
  sums[0] += sums[2];
  sums[1] += sums[3];
  float total = sums[0] + sums[1];
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset /= 2) {
    total += __shfl_down(total, offset);
  }
  total = __shfl(total, 0);
  const float inv_rms = rsqrtf(total / static_cast<float>(kHeadDim) + eps);

#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const std::size_t i = (32 * w) + lane;
    const float wt = (weight != nullptr) ? weight[i] : 1.0F;
    float normed = (values[w] * inv_rms) * wt;
    // Likewise the normed value is a stored operand of the rotation.
    asm volatile("" : "+v"(normed));
    values[w] = normed;
  }

  const std::size_t p = lane;
  const float x0 = values[0];
  const float x1 = values[1];
  const float exponent =
      static_cast<float>(2 * p) / static_cast<float>(kRotaryDim);
  const float inv_freq = 1.0F / powf(rope_theta, exponent);
  const float angle =
      models::qwen::vision::RopePosition(rope, cur_pos, p) * inv_freq;
  const float cos_val = cosf(angle);
  const float sin_val = sinf(angle);
  values[0] = (x0 * cos_val) - (x1 * sin_val);
  values[1] = (x0 * sin_val) + (x1 * cos_val);

  if (is_q) {
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
      q_out[head_base + (32 * w) + lane] = values[w];
    }
    return;
  }

  const std::size_t cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        cur_pos) *
       kHeadDim);
  const std::size_t f16_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + cur_pos) *
       kv_total) +
      (static_cast<std::size_t>(kv_h) * kHeadDim);
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const std::size_t i = (32 * w) + lane;
    const float key = values[w];
    k_out[head_base + i] = key;
    if (k_cache != nullptr) {
      k_cache[cache_offset + i] = key;
    }
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset + i] = __float2half_rn(key);
    }
  }
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    const std::size_t i = (32 * w) + lane;
    const float val = v[head_base + i];
    if (v_cache != nullptr) {
      v_cache[cache_offset + i] = val;
    }
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
    std::uint32_t rotary_dim, float rope_theta, float eps, hipStream_t stream,
    const models::qwen::vision::DeviceRope* rope) {
  const dim3 grid(num_heads + num_kv_heads,
                  static_cast<unsigned int>(batch_size));
  if (head_dim == 256 && rotary_dim == 64) {
    BatchedFusedQKNormRoPEKvWriteWaveKernel<<<grid, 32, 0, stream>>>(
        q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
        static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16),
        layer_idx, start_pos, batch_size, max_context, num_heads, num_kv_heads,
        rope_theta, eps, rope);
    return;
  }
  BatchedFusedQKNormRoPEKvWriteKernel<<<grid, 256, 0, stream>>>(
      q, k, v, q_weight, k_weight, q_out, k_out, k_cache, v_cache,
      static_cast<__half*>(k_cache_f16), static_cast<__half*>(v_cache_f16),
      layer_idx, start_pos, batch_size, max_context, num_heads, num_kv_heads,
      head_dim, rotary_dim, rope_theta, eps, rope);
}

__global__ void BatchedRoPEKernel(
    float* __restrict__ q, float* __restrict__ k, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t start_pos, float rope_theta,
    const models::qwen::vision::DeviceRope* rope) {
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
    const float angle =
        models::qwen::vision::RopePosition(rope, cur_pos, i) * freq;
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
    const float angle =
        models::qwen::vision::RopePosition(rope, cur_pos, i) * freq;
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
                       hipStream_t stream,
                       const models::qwen::vision::DeviceRope* rope) {
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
      start_pos, rope_theta, rope);
}
}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
