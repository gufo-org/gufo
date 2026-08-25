#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdint>

namespace strix::hip::kernels {

__global__ void dflash_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    uint32_t dim,
    float eps) {
  const uint32_t row = blockIdx.x;
  const float* row_in = input + (uint64_t)row * dim;
  float* row_out = output + (uint64_t)row * dim;

  __shared__ float s_sum[256];
  const uint32_t tid = threadIdx.x;

  float local_sum_sq = 0.0F;
  for (uint32_t i = tid; i < dim; i += blockDim.x) {
    const float val = row_in[i];
    local_sum_sq += val * val;
  }
  s_sum[tid] = local_sum_sq;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    __syncthreads();
  }

  const float rms = rsqrtf((s_sum[0] / (float)dim) + eps);

  for (uint32_t i = tid; i < dim; i += blockDim.x) {
    const float w = (weight != nullptr) ? weight[i] : 1.0F;
    row_out[i] = row_in[i] * rms * w;
  }
}

__global__ void dflash_dynamic_conv_2tap_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    uint32_t num_tokens,
    uint32_t hidden_size) {
  const uint32_t tok = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (tok >= num_tokens) return;

  for (uint32_t c = blockIdx.y * blockDim.x + tid; c < hidden_size;
       c += gridDim.y * blockDim.x) {
    const float w0 = weight[c * 2u + 0u];
    const float w1 = weight[c * 2u + 1u];
    const float b = (bias != nullptr) ? bias[c] : 0.0F;

    const float x_curr = input[(uint64_t)tok * hidden_size + c];
    const float x_prev = (tok > 0) ? input[(uint64_t)(tok - 1u) * hidden_size + c] : x_curr;

    output[(uint64_t)tok * hidden_size + c] = (w0 * x_curr) + (w1 * x_prev) + b;
  }
}

__global__ void dflash_silu_mul_kernel(
    float* __restrict__ gate,
    const float* __restrict__ up,
    uint32_t total_elements) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

__global__ void dflash_noncausal_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ injected_k,
    const float* __restrict__ injected_v,
    const float* __restrict__ block_k,
    const float* __restrict__ block_v,
    float* __restrict__ out,
    uint32_t current_pos,
    uint32_t draft_count,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    float scale) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= draft_count || h >= num_q_heads) return;

  const uint32_t gqa_group = num_q_heads / num_kv_heads;
  const uint32_t kv_h = h / gqa_group;
  const uint32_t q_dim = num_q_heads * head_dim;
  const uint32_t kv_dim = num_kv_heads * head_dim;
  const uint32_t total_k = current_pos + draft_count;

  const float* q_ptr = q + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);
  float* out_ptr = out + ((uint64_t)tok * q_dim + (uint64_t)h * head_dim);

  extern __shared__ float s_scores[];

  // 1. Compute dot products against historical injected KV
  for (uint32_t p = threadIdx.x; p < current_pos; p += blockDim.x) {
    const float* k_ptr = injected_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[p] = dot * scale;
  }

  // 2. Compute non-causal dot products against noise block KV
  for (uint32_t p = threadIdx.x; p < draft_count; p += blockDim.x) {
    const float* k_ptr = block_k + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
    float dot = 0.0F;
    for (uint32_t d = 0; d < head_dim; ++d) {
      dot += q_ptr[d] * k_ptr[d];
    }
    s_scores[current_pos + p] = dot * scale;
  }
  __syncthreads();

  // Find max score
  __shared__ float s_reduce[256];
  float local_max = -1e20F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_reduce[threadIdx.x] = local_max;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_val = s_reduce[0];
  __syncthreads();

  // Exponentiate & sum denominator
  float local_sum = 0.0F;
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    const float exp_score = expf(s_scores[p] - max_val);
    s_scores[p] = exp_score;
    local_sum += exp_score;
  }
  s_reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / (s_reduce[0] + 1e-9F);
  __syncthreads();

  // Normalize scores
  for (uint32_t p = threadIdx.x; p < total_k; p += blockDim.x) {
    s_scores[p] *= inv_sum;
  }
  __syncthreads();

  // Multiply by V
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (uint32_t p = 0; p < current_pos; ++p) {
      const float* v_ptr = injected_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[p] * v_ptr[d];
    }
    for (uint32_t p = 0; p < draft_count; ++p) {
      const float* v_ptr = block_v + ((uint64_t)p * kv_dim + (uint64_t)kv_h * head_dim);
      acc += s_scores[current_pos + p] * v_ptr[d];
    }
    out_ptr[d] = acc;
  }
}

void LaunchDFlashRMSNorm(
    const float* input,
    const float* weight,
    float* output,
    std::uint32_t dim,
    float eps,
    hipStream_t stream) {
  dflash_rmsnorm_kernel<<<1, 256, 0, stream>>>(input, weight, output, dim, eps);
}

void LaunchDFlashDynamicConv2Tap(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint32_t num_tokens,
    std::uint32_t hidden_size,
    hipStream_t stream) {
  dim3 grid(num_tokens, (hidden_size + 255u) / 256u);
  dflash_dynamic_conv_2tap_kernel<<<grid, 256, 0, stream>>>(
      input, weight, bias, output, num_tokens, hidden_size);
}

void LaunchDFlashSiLUMul(
    float* gate,
    const float* up,
    std::uint32_t total_elements,
    hipStream_t stream) {
  const uint32_t blocks = (total_elements + 255u) / 256u;
  dflash_silu_mul_kernel<<<blocks, 256, 0, stream>>>(gate, up, total_elements);
}

void LaunchDFlashNonCausalAttention(
    const float* q,
    const float* injected_k,
    const float* injected_v,
    const float* block_k,
    const float* block_v,
    float* out,
    std::uint32_t current_pos,
    std::uint32_t draft_count,
    std::uint32_t num_q_heads,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    float scale,
    hipStream_t stream) {
  const std::size_t shared_bytes = (current_pos + draft_count) * sizeof(float);
  dim3 grid(draft_count, num_q_heads);
  dflash_noncausal_attention_kernel<<<grid, 256, shared_bytes, stream>>>(
      q, injected_k, injected_v, block_k, block_v, out, current_pos,
      draft_count, num_q_heads, num_kv_heads, head_dim, scale);
}

}  // namespace strix::hip::kernels
#endif  // defined(ENGINE_ENABLE_HIP)
