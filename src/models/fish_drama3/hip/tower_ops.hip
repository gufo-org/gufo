#include "src/models/fish_drama3/hip/tower_ops.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::models::fish_drama3::hip {
namespace {

constexpr std::size_t kThreads = 256;
constexpr std::size_t kWaveSize = 32;
constexpr std::size_t kWavesPerBlock = kThreads / kWaveSize;

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16U);
}

__device__ __forceinline__ float WaveReduceSum(float val) {
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset, kWaveSize);
  }
  return val;
}

std::size_t Blocks(std::size_t elements) {
  return (elements + kThreads - 1U) / kThreads;
}

__global__ void FloatToBf16Kernel(const float* input, std::uint16_t* output,
                                  std::size_t elements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = FloatToBf16(input[idx]);
  }
}

__global__ void RmsNormKernel(const float* input, const float* weight,
                              float* output, std::size_t rows,
                              std::size_t columns, float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reduction[256];
  const float* in_row = input + row * columns;
  float* out_row = output + row * columns;
  float sum = 0.0F;
  for (std::size_t c = threadIdx.x; c < columns; c += blockDim.x) {
    const float v = in_row[c];
    sum = fmaf(v, v, sum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      reduction[threadIdx.x] += reduction[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(reduction[0] / static_cast<float>(columns) + eps);
  for (std::size_t c = threadIdx.x; c < columns; c += blockDim.x) {
    out_row[c] = in_row[c] * inv * weight[c];
  }
}

__global__ void ResidualAddRmsNormKernel(float* x_inout, const float* update,
                                         const float* weight, float* normed_out,
                                         std::size_t rows, std::size_t columns,
                                         float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reduction[256];
  float* x_row = x_inout + row * columns;
  const float* u_row = update + row * columns;
  float* n_row = normed_out + row * columns;
  float sum = 0.0F;
  for (std::size_t c = threadIdx.x; c < columns; c += blockDim.x) {
    const float v = x_row[c] + u_row[c];
    x_row[c] = v;
    sum = fmaf(v, v, sum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      reduction[threadIdx.x] += reduction[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(reduction[0] / static_cast<float>(columns) + eps);
  for (std::size_t c = threadIdx.x; c < columns; c += blockDim.x) {
    n_row[c] = x_row[c] * inv * weight[c];
  }
}

__global__ void EmbedAndConcatNormKernel(
    const std::uint16_t* embedding_table_bf16, const std::int32_t* codes,
    const float* label_mask, const float* extra, const float* hidden,
    const float* enorm, const float* hnorm, float* concat_out,
    std::size_t steps, std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= steps) {
    return;
  }
  __shared__ float red_e[256];
  __shared__ float red_h[256];

  const std::int32_t code = codes[row];
  const float mask = label_mask != nullptr ? label_mask[row] : 1.0F;
  const std::uint16_t* emb_row =
      embedding_table_bf16 + static_cast<std::size_t>(code) * dim;
  const float* extra_row = extra != nullptr ? (extra + row * dim) : nullptr;
  const float* h_row = hidden + row * dim;
  float* out_e = concat_out + row * (2U * dim);
  float* out_h = out_e + dim;

  float sum_e = 0.0F;
  float sum_h = 0.0F;
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    float e_val = Bf16ToFloat(emb_row[c]) * mask;
    if (extra_row != nullptr) {
      e_val += extra_row[c];
    }
    out_e[c] = e_val;
    const float h_val = h_row[c];
    sum_e = fmaf(e_val, e_val, sum_e);
    sum_h = fmaf(h_val, h_val, sum_h);
  }
  red_e[threadIdx.x] = sum_e;
  red_h[threadIdx.x] = sum_h;
  __syncthreads();

  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      red_e[threadIdx.x] += red_e[threadIdx.x + s];
      red_h[threadIdx.x] += red_h[threadIdx.x + s];
    }
    __syncthreads();
  }

  const float inv_e = rsqrtf(red_e[0] / static_cast<float>(dim) + eps);
  const float inv_h = rsqrtf(red_h[0] / static_cast<float>(dim) + eps);
  for (std::size_t c = threadIdx.x; c < dim; c += blockDim.x) {
    out_e[c] = out_e[c] * inv_e * enorm[c];
    out_h[c] = h_row[c] * inv_h * hnorm[c];
  }
}

__global__ void SplitQkvNormRopeCacheKernel(
    const float* qkv, const float* q_norm, const float* k_norm, float* q_out,
    float* key_cache, float* value_cache, std::size_t steps,
    std::size_t start_pos, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t rotary_dim, float rope_theta, float eps) {
  const std::size_t total_qk_heads = heads + kv_heads;
  const std::size_t step = blockIdx.x / total_qk_heads;
  const std::size_t h_idx = blockIdx.x % total_qk_heads;
  if (step >= steps) {
    return;
  }

  const std::size_t q_width = heads * head_dim;
  const std::size_t kv_width = kv_heads * head_dim;
  const std::size_t qkv_stride = 2U * q_width + 2U * kv_width;
  const float* qkv_row = qkv + step * qkv_stride;

  const bool is_q = h_idx < heads;
  const std::size_t local_h = is_q ? h_idx : (h_idx - heads);
  const float* src_head = is_q ? (qkv_row + local_h * head_dim)
                               : (qkv_row + 2U * q_width + local_h * head_dim);
  const float* norm_w = is_q ? q_norm : k_norm;

  __shared__ float head_vals[256];
  __shared__ float reduction[256];

  const std::size_t d = threadIdx.x;
  float val = d < head_dim ? src_head[d] : 0.0F;
  reduction[d] = val * val;
  __syncthreads();

  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (d < s) {
      reduction[d] += reduction[d + s];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(reduction[0] / static_cast<float>(head_dim) + eps);
  if (d < head_dim) {
    head_vals[d] = val * inv * norm_w[d];
  }
  __syncthreads();

  if (d < head_dim) {
    float final_val = head_vals[d];
    if (d < rotary_dim) {
      const std::size_t pair = d / 2U;
      const float exponent =
          -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle =
          static_cast<float>(start_pos + step) * powf(rope_theta, exponent);
      float s_val = 0.0F;
      float c_val = 0.0F;
      sincosf(angle, &s_val, &c_val);
      const float v0 = head_vals[2U * pair];
      const float v1 = head_vals[2U * pair + 1U];
      final_val = (d & 1U) == 0U ? (v0 * c_val - v1 * s_val)
                                 : (v1 * c_val + v0 * s_val);
    }
    if (is_q) {
      q_out[(step * heads + local_h) * head_dim + d] = final_val;
    } else {
      const std::size_t cache_pos = start_pos + step;
      key_cache[(cache_pos * kv_heads + local_h) * head_dim + d] = final_val;
      const float* v_src =
          qkv_row + 2U * q_width + kv_width + local_h * head_dim;
      value_cache[(cache_pos * kv_heads + local_h) * head_dim + d] = v_src[d];
    }
  }
}

__global__ void GqaAttentionAndGateKernel(
    const float* q, const float* key_cache, const float* value_cache,
    const float* qkv, float* attended_out, std::size_t steps,
    std::size_t start_pos, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim, std::size_t qkv_stride) {
  const std::size_t step = blockIdx.x / heads;
  const std::size_t head = blockIdx.x % heads;
  if (step >= steps) {
    return;
  }

  const std::size_t kv_group = heads / kv_heads;
  const std::size_t kv_head = head / kv_group;
  const std::size_t valid_keys = start_pos + step + 1U;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float* q_vec = q + (step * heads + head) * head_dim;

  extern __shared__ float shared[];
  float* scores = shared;
  float* red = shared + valid_keys;

  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;

  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t k_idx = wave_id; k_idx < valid_keys;
       k_idx += kWavesPerBlock) {
    const float* k_vec = key_cache + (k_idx * kv_heads + kv_head) * head_dim;
    float partial = 0.0F;
    for (std::size_t d = lane; d < head_dim; d += kWaveSize) {
      partial = fmaf(q_vec[d], k_vec[d], partial);
    }
    const float dot = WaveReduceSum(partial) * scale;
    if (lane == 0) {
      scores[k_idx] = dot;
      local_max = fmaxf(local_max, dot);
    }
  }
  red[threadIdx.x] =
      lane == 0 ? local_max : -std::numeric_limits<float>::infinity();
  __syncthreads();

  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    }
    __syncthreads();
  }
  const float max_score = red[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t k_idx = threadIdx.x; k_idx < valid_keys;
       k_idx += blockDim.x) {
    const float p = expf(scores[k_idx] - max_score);
    scores[k_idx] = p;
    local_sum += p;
  }
  red[threadIdx.x] = local_sum;
  __syncthreads();

  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / red[0];

  const std::size_t q_width = heads * head_dim;
  const float* gate_vec = qkv + step * qkv_stride + q_width + head * head_dim;
  float* out_vec = attended_out + (step * heads + head) * head_dim;

  for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0F;
    for (std::size_t k_idx = 0; k_idx < valid_keys; ++k_idx) {
      const float* v_vec =
          value_cache + (k_idx * kv_heads + kv_head) * head_dim;
      acc = fmaf(scores[k_idx] * inv_sum, v_vec[d], acc);
    }
    const float g = gate_vec[d];
    const float sig_g = 1.0F / (1.0F + expf(-g));
    out_vec[d] = acc * sig_g;
  }
}

__device__ __forceinline__ void TowerAccumBf16Vec8(const uint4& w_pack,
                                                   const float4& x0,
                                                   const float4& x1,
                                                   float& acc) {
  const std::uint16_t* w16 = reinterpret_cast<const std::uint16_t*>(&w_pack);
  acc = fmaf(Bf16ToFloat(w16[0]), x0.x, acc);
  acc = fmaf(Bf16ToFloat(w16[1]), x0.y, acc);
  acc = fmaf(Bf16ToFloat(w16[2]), x0.z, acc);
  acc = fmaf(Bf16ToFloat(w16[3]), x0.w, acc);
  acc = fmaf(Bf16ToFloat(w16[4]), x1.x, acc);
  acc = fmaf(Bf16ToFloat(w16[5]), x1.y, acc);
  acc = fmaf(Bf16ToFloat(w16[6]), x1.z, acc);
  acc = fmaf(Bf16ToFloat(w16[7]), x1.w, acc);
}

__launch_bounds__(kThreads) __global__
    void Bf16GemvKernel(const std::uint16_t* __restrict__ weight_bf16,
                        const float* __restrict__ input_f32,
                        float* __restrict__ output_f32, std::size_t batch,
                        std::size_t rows, std::size_t cols) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const std::size_t vec_cols = cols / 8U;
  const uint4* w_vec = reinterpret_cast<const uint4*>(weight_bf16 + row * cols);

  for (std::size_t b = 0; b < batch; ++b) {
    const float4* x_vec = reinterpret_cast<const float4*>(input_f32 + b * cols);
    float acc = 0.0F;
    for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
      const uint4 w_pack = w_vec[idx];
      const float4 x0 = x_vec[2U * idx];
      const float4 x1 = x_vec[2U * idx + 1U];
      TowerAccumBf16Vec8(w_pack, x0, x1, acc);
    }
    const float w_sum = WaveReduceSum(acc);
    if (lane == 0) {
      wave_sums[wave_id] = w_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total = 0.0F;
#pragma unroll
      for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
        total += wave_sums[w];
      }
      output_f32[b * rows + row] = total;
    }
    if (batch > 1U) {
      __syncthreads();
    }
  }
}

__launch_bounds__(kThreads) __global__
    void Bf16SwiGluGemvKernel(const std::uint16_t* __restrict__ w1_bf16,
                              const std::uint16_t* __restrict__ w3_bf16,
                              const float* __restrict__ input_f32,
                              float* __restrict__ output_f32, std::size_t rows,
                              std::size_t cols) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float gate_sums[kWavesPerBlock];
  __shared__ float up_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const uint4* w1_vec = reinterpret_cast<const uint4*>(w1_bf16 + row * cols);
  const uint4* w3_vec = reinterpret_cast<const uint4*>(w3_bf16 + row * cols);
  const float4* x_vec = reinterpret_cast<const float4*>(input_f32);
  const std::size_t vec_cols = cols / 8U;

  float acc_gate = 0.0F;
  float acc_up = 0.0F;
  for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
    const uint4 w1_pack = w1_vec[idx];
    const uint4 w3_pack = w3_vec[idx];
    const float4 x0 = x_vec[2U * idx];
    const float4 x1 = x_vec[2U * idx + 1U];
    TowerAccumBf16Vec8(w1_pack, x0, x1, acc_gate);
    TowerAccumBf16Vec8(w3_pack, x0, x1, acc_up);
  }
  const float g_sum = WaveReduceSum(acc_gate);
  const float u_sum = WaveReduceSum(acc_up);
  if (lane == 0) {
    gate_sums[wave_id] = g_sum;
    up_sums[wave_id] = u_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float gate = 0.0F;
    float up = 0.0F;
#pragma unroll
    for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
      gate += gate_sums[w];
      up += up_sums[w];
    }
    const float silu_gate = gate / (1.0F + expf(-gate));
    output_f32[row] = silu_gate * up;
  }
}

__global__ void SwiGluKernel(const float* gate, const float* up, float* output,
                             std::size_t elements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    const float g = gate[idx];
    output[idx] = (g / (1.0F + expf(-g))) * up[idx];
  }
}

__global__ void SampleTopKTopPKernel(
    const float* logits, std::size_t vocab_size, float temperature, int top_k,
    float top_p, float uniform_sample, std::int32_t* out_code_dev,
    std::int32_t* out_codes_array, std::size_t book_idx) {
  __shared__ float s_vals[4096];
  __shared__ std::int32_t s_ids[4096];
  __shared__ float red[256];
  __shared__ std::int32_t red_idx[256];

  const bool greedy = (temperature <= 0.0F || top_k <= 1);
  if (greedy) {
    float best_val = -std::numeric_limits<float>::infinity();
    std::int32_t best_id = 0;
    for (std::size_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
      const float v = logits[i];
      if (v > best_val ||
          (v == best_val && static_cast<std::int32_t>(i) < best_id)) {
        best_val = v;
        best_id = static_cast<std::int32_t>(i);
      }
    }
    red[threadIdx.x] = best_val;
    red_idx[threadIdx.x] = best_id;
    __syncthreads();
    for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
      if (threadIdx.x < s) {
        const float v_other = red[threadIdx.x + s];
        const std::int32_t i_other = red_idx[threadIdx.x + s];
        if (v_other > red[threadIdx.x] ||
            (v_other == red[threadIdx.x] && i_other < red_idx[threadIdx.x])) {
          red[threadIdx.x] = v_other;
          red_idx[threadIdx.x] = i_other;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const std::int32_t chosen = red_idx[0];
      if (out_code_dev != nullptr) {
        *out_code_dev = chosen;
      }
      if (out_codes_array != nullptr) {
        out_codes_array[book_idx] = chosen;
      }
    }
    return;
  }

  const float inv_temp = 1.0F / temperature;
  float local_max = -std::numeric_limits<float>::infinity();
  for (std::size_t i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    local_max = fmaxf(local_max, logits[i] * inv_temp);
  }
  red[threadIdx.x] = local_max;
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    }
    __syncthreads();
  }
  const float max_scaled = red[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t i = 0; i < 4096; i += blockDim.x) {
    const std::size_t idx = i + threadIdx.x;
    if (idx < vocab_size) {
      const float w = expf(logits[idx] * inv_temp - max_scaled);
      s_vals[idx] = w;
      s_ids[idx] = static_cast<std::int32_t>(idx);
      local_sum += w;
    } else if (idx < 4096) {
      s_vals[idx] = -1.0F;
      s_ids[idx] = 0x7fffffff;
    }
  }
  red[threadIdx.x] = local_sum;
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s != 0; s >>= 1U) {
    if (threadIdx.x < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  const float inv_total = 1.0F / red[0];
  for (std::size_t idx = threadIdx.x; idx < vocab_size; idx += blockDim.x) {
    s_vals[idx] *= inv_total;
  }
  __syncthreads();

  for (std::size_t k = 2; k <= 4096; k <<= 1U) {
    for (std::size_t j = k >> 1U; j > 0; j >>= 1U) {
      for (std::size_t tid = threadIdx.x; tid < 2048; tid += blockDim.x) {
        const std::size_t i = ((tid & ~(j - 1U)) << 1U) | (tid & (j - 1U));
        const std::size_t partner = i ^ j;
        const bool descending = ((i & k) == 0);
        const float vi = s_vals[i];
        const float vp = s_vals[partner];
        const std::int32_t idi = s_ids[i];
        const std::int32_t idp = s_ids[partner];
        const bool i_before_p = (vi > vp) || (vi == vp && idi < idp);
        if (i_before_p != descending) {
          s_vals[i] = vp;
          s_vals[partner] = vi;
          s_ids[i] = idp;
          s_ids[partner] = idi;
        }
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    const std::size_t limit = std::min<std::size_t>(
        vocab_size, static_cast<std::size_t>(std::max(1, top_k)));
    float cumulative = 0.0F;
    std::size_t kept = 0;
    for (std::size_t r = 0; r < limit; ++r) {
      cumulative += s_vals[r];
      if (r > 0 && cumulative > top_p) {
        cumulative -= s_vals[r];
        break;
      }
      kept = r + 1U;
    }
    if (kept == 0) {
      kept = 1;
      cumulative = s_vals[0];
    }
    const float target = uniform_sample * cumulative;
    float running = 0.0F;
    std::int32_t chosen = s_ids[kept - 1U];
    for (std::size_t r = 0; r < kept; ++r) {
      running += s_vals[r];
      if (target < running) {
        chosen = s_ids[r];
        break;
      }
    }
    if (out_code_dev != nullptr) {
      *out_code_dev = chosen;
    }
    if (out_codes_array != nullptr) {
      out_codes_array[book_idx] = chosen;
    }
  }
}

}  // namespace

void LaunchTowerFloatToBf16(const float* input, std::uint16_t* output,
                            std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(FloatToBf16Kernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, input, output, elements);
}

void LaunchTowerRmsNorm(const float* input, const float* weight, float* output,
                        std::size_t rows, std::size_t columns, float eps,
                        hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, output, rows, columns, eps);
}

void LaunchTowerResidualAddRmsNorm(float* x_inout, const float* update,
                                   const float* weight, float* normed_out,
                                   std::size_t rows, std::size_t columns,
                                   float eps, hipStream_t stream) {
  hipLaunchKernelGGL(ResidualAddRmsNormKernel, dim3(rows), dim3(kThreads), 0,
                     stream, x_inout, update, weight, normed_out, rows, columns,
                     eps);
}

void LaunchTowerEmbedAndConcatNorm(const std::uint16_t* embedding_table_bf16,
                                   const std::int32_t* codes,
                                   const float* label_mask, const float* extra,
                                   const float* hidden, const float* enorm,
                                   const float* hnorm, float* concat_out,
                                   std::size_t steps, std::size_t dim,
                                   float eps, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedAndConcatNormKernel, dim3(steps), dim3(kThreads), 0,
                     stream, embedding_table_bf16, codes, label_mask, extra,
                     hidden, enorm, hnorm, concat_out, steps, dim, eps);
}

void LaunchTowerSplitQkvNormRopeCache(const float* qkv, const float* q_norm,
                                      const float* k_norm, float* q_out,
                                      float* key_cache, float* value_cache,
                                      std::size_t steps, std::size_t start_pos,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim,
                                      std::size_t rotary_dim, float rope_theta,
                                      float eps, hipStream_t stream) {
  const std::size_t total_blocks = steps * (heads + kv_heads);
  hipLaunchKernelGGL(SplitQkvNormRopeCacheKernel, dim3(total_blocks),
                     dim3(kThreads), 0, stream, qkv, q_norm, k_norm, q_out,
                     key_cache, value_cache, steps, start_pos, heads, kv_heads,
                     head_dim, rotary_dim, rope_theta, eps);
}

void LaunchTowerGqaAttentionAndGate(const float* q, const float* key_cache,
                                    const float* value_cache, const float* qkv,
                                    float* attended_out, std::size_t steps,
                                    std::size_t start_pos, std::size_t heads,
                                    std::size_t kv_heads, std::size_t head_dim,
                                    std::size_t qkv_stride,
                                    hipStream_t stream) {
  const std::size_t max_keys = start_pos + steps;
  const std::size_t shared_bytes = (max_keys + kThreads) * sizeof(float);
  hipLaunchKernelGGL(GqaAttentionAndGateKernel, dim3(steps * heads),
                     dim3(kThreads), shared_bytes, stream, q, key_cache,
                     value_cache, qkv, attended_out, steps, start_pos, heads,
                     kv_heads, head_dim, qkv_stride);
}

void LaunchTowerBf16Gemv(const std::uint16_t* weight_bf16,
                         const float* input_f32, float* output_f32,
                         std::size_t batch, std::size_t rows, std::size_t cols,
                         hipStream_t stream) {
  hipLaunchKernelGGL(Bf16GemvKernel, dim3(rows), dim3(kThreads), 0, stream,
                     weight_bf16, input_f32, output_f32, batch, rows, cols);
}

void LaunchTowerBf16SwiGluGemv(const std::uint16_t* w1_bf16,
                               const std::uint16_t* w3_bf16,
                               const float* input_f32, float* output_f32,
                               std::size_t rows, std::size_t cols,
                               hipStream_t stream) {
  hipLaunchKernelGGL(Bf16SwiGluGemvKernel, dim3(rows), dim3(kThreads), 0,
                     stream, w1_bf16, w3_bf16, input_f32, output_f32, rows,
                     cols);
}

void LaunchTowerSwiGlu(const float* gate, const float* up, float* output,
                       std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(SwiGluKernel, dim3(Blocks(elements)), dim3(kThreads), 0,
                     stream, gate, up, output, elements);
}

void LaunchTowerSampleTopKTopP(const float* logits, std::size_t vocab_size,
                               float temperature, int top_k, float top_p,
                               float uniform_sample, std::int32_t* out_code_dev,
                               std::int32_t* out_codes_array,
                               std::size_t book_idx, hipStream_t stream) {
  hipLaunchKernelGGL(SampleTopKTopPKernel, dim3(1), dim3(kThreads), 0, stream,
                     logits, vocab_size, temperature, top_k, top_p,
                     uniform_sample, out_code_dev, out_codes_array, book_idx);
}

}  // namespace gufo::models::fish_drama3::hip

#endif
