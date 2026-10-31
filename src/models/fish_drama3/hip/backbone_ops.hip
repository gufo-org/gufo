#include "src/models/fish_drama3/hip/backbone_ops.hpp"

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

// Exact IEEE / PyTorch float8_e4m3fn -> float32 conversion in registers.
__device__ __forceinline__ float Fp8E4m3ToFloat(std::uint8_t b) {
  const std::uint32_t sign = static_cast<std::uint32_t>(b & 0x80U) << 24U;
  const std::uint32_t exp_mant = b & 0x7fU;
  if (exp_mant < 8U) {
    const float val = static_cast<float>(exp_mant) * (1.0F / 512.0F);
    return (b & 0x80U) ? -val : val;
  }
  return __uint_as_float(sign | ((exp_mant << 20U) + 0x3c000000U));
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

__global__ void ResidualAddKernel(float* x_inout, const float* update,
                                  std::size_t elements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    x_inout[idx] += update[idx];
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
  __shared__ float reduction[kThreads];
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

__global__ void FloatToBf16Kernel(const float* input, std::uint16_t* output,
                                  std::size_t elements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < elements) {
    output[idx] = FloatToBf16(input[idx]);
  }
}

__device__ __forceinline__ void AccumulateBf16Vec8(const uint4& w_pack,
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

// One output row per 256-thread workgroup (8 wavefronts walk the row together
// in a single 4KB coalesced stream to saturate gfx1151 LPDDR5X bandwidth).
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
      AccumulateBf16Vec8(w_pack, x0, x1, acc);
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

__launch_bounds__(kThreads) __global__ void FullQkvProjFusedGemvKernel(
    const std::uint16_t* __restrict__ q_w,
    const std::uint16_t* __restrict__ k_w,
    const std::uint16_t* __restrict__ v_w, const float* __restrict__ input_f32,
    float* __restrict__ q_out, float* __restrict__ k_out,
    float* __restrict__ v_out, std::size_t q_rows, std::size_t kv_rows,
    std::size_t cols) {
  const std::size_t global_row = blockIdx.x;
  const std::size_t total_rows = q_rows + 2U * kv_rows;
  if (global_row >= total_rows) {
    return;
  }

  const std::uint16_t* w_row = nullptr;
  float* out_ptr = nullptr;
  if (global_row < q_rows) {
    w_row = q_w + global_row * cols;
    out_ptr = q_out + global_row;
  } else if (global_row < q_rows + kv_rows) {
    const std::size_t r = global_row - q_rows;
    w_row = k_w + r * cols;
    out_ptr = k_out + r;
  } else {
    const std::size_t r = global_row - q_rows - kv_rows;
    w_row = v_w + r * cols;
    out_ptr = v_out + r;
  }

  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const std::size_t vec_cols = cols / 8U;
  const uint4* w_vec = reinterpret_cast<const uint4*>(w_row);
  const float4* x_vec = reinterpret_cast<const float4*>(input_f32);

  float acc = 0.0F;
  for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
    const uint4 w_pack = w_vec[idx];
    const float4 x0 = x_vec[2U * idx];
    const float4 x1 = x_vec[2U * idx + 1U];
    AccumulateBf16Vec8(w_pack, x0, x1, acc);
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
    *out_ptr = total;
  }
}

__launch_bounds__(kThreads) __global__ void LinearProjFusedGemvKernel(
    const std::uint16_t* __restrict__ qkv_w,
    const std::uint16_t* __restrict__ z_w,
    const std::uint16_t* __restrict__ b_w,
    const std::uint16_t* __restrict__ a_w, const float* __restrict__ input_f32,
    float* __restrict__ qkv_out, float* __restrict__ z_out,
    float* __restrict__ b_out, float* __restrict__ a_out, std::size_t qkv_rows,
    std::size_t z_rows, std::size_t ba_rows, std::size_t cols) {
  const std::size_t global_row = blockIdx.x;
  const std::size_t total_rows = qkv_rows + z_rows + 2U * ba_rows;
  if (global_row >= total_rows) {
    return;
  }

  const std::uint16_t* w_row = nullptr;
  float* out_ptr = nullptr;
  if (global_row < qkv_rows) {
    w_row = qkv_w + global_row * cols;
    out_ptr = qkv_out + global_row;
  } else if (global_row < qkv_rows + z_rows) {
    const std::size_t r = global_row - qkv_rows;
    w_row = z_w + r * cols;
    out_ptr = z_out + r;
  } else if (global_row < qkv_rows + z_rows + ba_rows) {
    const std::size_t r = global_row - qkv_rows - z_rows;
    w_row = b_w + r * cols;
    out_ptr = b_out + r;
  } else {
    const std::size_t r = global_row - qkv_rows - z_rows - ba_rows;
    w_row = a_w + r * cols;
    out_ptr = a_out + r;
  }

  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const std::size_t vec_cols = cols / 8U;
  const uint4* w_vec = reinterpret_cast<const uint4*>(w_row);
  const float4* x_vec = reinterpret_cast<const float4*>(input_f32);

  float acc = 0.0F;
  for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
    const uint4 w_pack = w_vec[idx];
    const float4 x0 = x_vec[2U * idx];
    const float4 x1 = x_vec[2U * idx + 1U];
    AccumulateBf16Vec8(w_pack, x0, x1, acc);
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
    *out_ptr = total;
  }
}

// Full Attention Q/K norm + partial NeoX (half-split) RoPE + KV cache write.
// In Backbone Full Attention, q_full is [steps, heads, 2 * head_dim] where
// [0..head_dim-1] is Q and [head_dim..2*head_dim-1] is Gate. RoPE is
// GGML_ROPE_TYPE_NEOX on the first rotary_dim (64) dimensions: pairs are (d, d
// + rotary_dim/2) for d in [0, rotary_dim/2).
__global__ void FullAttnPrepKernel(const float* q_full, const float* k_in,
                                   const float* v_in, const float* q_norm,
                                   const float* k_norm, float* q_out,
                                   float* key_cache, float* value_cache,
                                   std::size_t steps, std::size_t start_pos,
                                   std::size_t heads, std::size_t kv_heads,
                                   std::size_t head_dim, std::size_t rotary_dim,
                                   float rope_theta, float eps) {
  const std::size_t total_qk_heads = heads + kv_heads;
  const std::size_t step = blockIdx.x / total_qk_heads;
  const std::size_t h_idx = blockIdx.x % total_qk_heads;
  if (step >= steps) {
    return;
  }

  const bool is_q = h_idx < heads;
  const std::size_t local_h = is_q ? h_idx : (h_idx - heads);
  const float* src_head =
      is_q ? (q_full + (step * heads + local_h) * (2U * head_dim))
           : (k_in + (step * kv_heads + local_h) * head_dim);
  const float* norm_w = is_q ? q_norm : k_norm;

  __shared__ float head_vals[256];
  __shared__ float reduction[256];

  const std::size_t d = threadIdx.x;
  const float val = d < head_dim ? src_head[d] : 0.0F;
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
      const std::size_t half_rot = rotary_dim / 2U;
      const std::size_t pair = d < half_rot ? d : (d - half_rot);
      const float exponent =
          -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
      const float angle =
          static_cast<float>(start_pos + step) * powf(rope_theta, exponent);
      float s_val = 0.0F;
      float c_val = 0.0F;
      sincosf(angle, &s_val, &c_val);
      const float v0 = head_vals[pair];
      const float v1 = head_vals[pair + half_rot];
      final_val =
          d < half_rot ? (v0 * c_val - v1 * s_val) : (v1 * c_val + v0 * s_val);
    }
    if (is_q) {
      q_out[(step * heads + local_h) * head_dim + d] = final_val;
    } else {
      const std::size_t cache_pos = start_pos + step;
      key_cache[(cache_pos * kv_heads + local_h) * head_dim + d] = final_val;
      const float* v_src = v_in + (step * kv_heads + local_h) * head_dim;
      value_cache[(cache_pos * kv_heads + local_h) * head_dim + d] = v_src[d];
    }
  }
}

__global__ void FullAttnGqaAndGateKernel(
    const float* q, const float* key_cache, const float* value_cache,
    const float* q_full, float* attended_out, std::size_t steps,
    std::size_t start_pos, std::size_t heads, std::size_t kv_heads,
    std::size_t head_dim) {
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

  const float* gate_vec =
      q_full + (step * heads + head) * (2U * head_dim) + head_dim;
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

// Depthwise Causal Conv1d (kernel = 4) + SiLU for Linear Attention.
__global__ void LinearConv1dSiluKernel(const float* qkv_in,
                                       const float* conv_weight,
                                       float* conv_state, float* qkv_conv_out,
                                       std::size_t steps,
                                       std::size_t channels) {
  const std::size_t c =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  float h0 = conv_state[0U * channels + c];
  float h1 = conv_state[1U * channels + c];
  float h2 = conv_state[2U * channels + c];
  const float w0 = conv_weight[c * 4U + 0U];
  const float w1 = conv_weight[c * 4U + 1U];
  const float w2 = conv_weight[c * 4U + 2U];
  const float w3 = conv_weight[c * 4U + 3U];

  for (std::size_t t = 0; t < steps; ++t) {
    const float x = qkv_in[t * channels + c];
    const float v = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x;
    qkv_conv_out[t * channels + c] = v / (1.0F + expf(-v));
    h0 = h1;
    h1 = h2;
    h2 = x;
  }
  conv_state[0U * channels + c] = h0;
  conv_state[1U * channels + c] = h1;
  conv_state[2U * channels + c] = h2;
}

// Register-resident Gated DeltaNet recurrent kernel:
// Grid: dim3(v_heads = 32, 4) where blockIdx.y tiles v_d (128 = 4 * 32).
// Block: dim3(256) where v_col = tid % 32, k_group = tid / 32 (0..7).
// Each thread holds 16 elements of ssm_state[hv, k_d, v_d] for k_d in
// [k_group*16, k_group*16+15].
__global__ void GatedDeltaNetStateKernel(
    const float* __restrict__ qkv_conv, const float* __restrict__ b,
    const float* __restrict__ a, const float* __restrict__ dt_bias,
    const float* __restrict__ neg_exp_a, float* __restrict__ ssm_state,
    float* __restrict__ attn_out, std::size_t steps, std::size_t k_heads,
    std::size_t v_heads, std::size_t head_dim, float eps) {
  const std::size_t hv = blockIdx.x;
  const std::size_t v_tile = blockIdx.y;
  if (hv >= v_heads || v_tile >= 4U) {
    return;
  }

  const std::size_t v_col = threadIdx.x % 32U;
  const std::size_t k_group = threadIdx.x / 32U;
  const std::size_t v_d = v_tile * 32U + v_col;
  const std::size_t k_base = k_group * 16U;

  const std::size_t group_ratio = v_heads / k_heads;
  const std::size_t kh = hv / group_ratio;
  const std::size_t q_dim = k_heads * head_dim;
  const std::size_t k_dim = k_heads * head_dim;
  const std::size_t v_dim = v_heads * head_dim;
  const std::size_t qkv_stride = q_dim + k_dim + v_dim;

  // Load 16 state elements into registers
  float s_reg[16];
  float* state_head = ssm_state + hv * (128U * 128U);
#pragma unroll
  for (std::size_t i = 0; i < 16U; ++i) {
    s_reg[i] = state_head[(k_base + i) * 128U + v_d];
  }

  const float bias_hv = dt_bias[hv];
  const float neg_a_hv = neg_exp_a[hv];
  const float q_head_scale = rsqrtf(static_cast<float>(head_dim));

  __shared__ float s_q[128];
  __shared__ float s_k[128];
  __shared__ float s_norm_red[256];
  __shared__ float s_kv_red[8][33];  // +1 padding to avoid LDS bank conflicts
  __shared__ float s_decay;
  __shared__ float s_beta;

  for (std::size_t t = 0; t < steps; ++t) {
    const float* row_qkv = qkv_conv + t * qkv_stride;
    const float* q_src = row_qkv + kh * 128U;
    const float* k_src = row_qkv + q_dim + kh * 128U;

    // 1. Load & L2-normalize Q (threads 0..127) and K (threads 128..255)
    const bool is_k_loader = threadIdx.x >= 128U;
    const std::size_t d_load = is_k_loader ? (threadIdx.x - 128U) : threadIdx.x;
    const float val_load = is_k_loader ? k_src[d_load] : q_src[d_load];
    if (is_k_loader) {
      s_k[d_load] = val_load;
    } else {
      s_q[d_load] = val_load;
    }
    s_norm_red[threadIdx.x] = val_load * val_load;
    if (threadIdx.x == 0) {
      const float a_val = a[t * v_heads + hv] + bias_hv;
      const float sp = a_val > 20.0F ? a_val : log1pf(expf(a_val));
      s_decay = expf(sp * neg_a_hv);
      s_beta = 1.0F / (1.0F + expf(-b[t * v_heads + hv]));
    }
    __syncthreads();

    for (std::size_t s = 64U; s != 0; s >>= 1U) {
      if (d_load < s) {
        s_norm_red[threadIdx.x] += s_norm_red[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (is_k_loader) {
      const float inv_k = rsqrtf(s_norm_red[128] + eps);
      s_k[d_load] *= inv_k;
    } else {
      const float inv_q = rsqrtf(s_norm_red[0] + eps) * q_head_scale;
      s_q[d_load] *= inv_q;
    }
    __syncthreads();

    // 2. Decay state and compute kv_mem = S^T * k
    const float decay = s_decay;
    float p_kv = 0.0F;
#pragma unroll
    for (std::size_t i = 0; i < 16U; ++i) {
      s_reg[i] *= decay;
      p_kv = fmaf(s_reg[i], s_k[k_base + i], p_kv);
    }
    s_kv_red[k_group][v_col] = p_kv;
    __syncthreads();

    float kv_mem = 0.0F;
#pragma unroll
    for (std::size_t g = 0; g < 8U; ++g) {
      kv_mem += s_kv_red[g][v_col];
    }

    // 3. Rank-1 delta update & output projection y = S^T * q
    const float v_val = row_qkv[q_dim + k_dim + hv * 128U + v_d];
    const float delta = (v_val - kv_mem) * s_beta;
    float p_y = 0.0F;
#pragma unroll
    for (std::size_t i = 0; i < 16U; ++i) {
      s_reg[i] = fmaf(s_k[k_base + i], delta, s_reg[i]);
      p_y = fmaf(s_reg[i], s_q[k_base + i], p_y);
    }
    __syncthreads();
    s_kv_red[k_group][v_col] = p_y;
    __syncthreads();

    if (k_group == 0) {
      float y_sum = 0.0F;
#pragma unroll
      for (std::size_t g = 0; g < 8U; ++g) {
        y_sum += s_kv_red[g][v_col];
      }
      attn_out[(t * v_heads + hv) * 128U + v_d] = y_sum;
    }
    __syncthreads();
  }

// Write updated state back to global memory
#pragma unroll
  for (std::size_t i = 0; i < 16U; ++i) {
    state_head[(k_base + i) * 128U + v_d] = s_reg[i];
  }
}

// Per-head RMSNorm (128-dim) * norm_weight * silu(z)
__global__ void GatedDeltaNetNormSiluKernel(float* attn_out, const float* z,
                                            const float* norm_weight,
                                            std::size_t total_heads,
                                            std::size_t head_dim, float eps) {
  const std::size_t idx = blockIdx.x;
  if (idx >= total_heads) {
    return;
  }
  __shared__ float red[128];
  const std::size_t d = threadIdx.x;
  float* y_head = attn_out + idx * head_dim;
  const float* z_head = z + idx * head_dim;

  const float val = d < head_dim ? y_head[d] : 0.0F;
  red[d] = val * val;
  __syncthreads();
  for (std::size_t s = 64U; s != 0; s >>= 1U) {
    if (d < s) {
      red[d] += red[d + s];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / static_cast<float>(head_dim) + eps);
  if (d < head_dim) {
    const float z_val = z_head[d];
    const float silu_z = z_val / (1.0F + expf(-z_val));
    y_head[d] = val * inv * norm_weight[d] * silu_z;
  }
}

// MoE Router Kernels (Full-GPU Parallel Logits GEMV + Warp-Parallel Top-8
// Selection): Device scratch for router logits up to 1024 tokens x 256 experts
// (1 MB static BSS).
__device__ float g_router_logits_scratch[1024U * 256U];

// Kernel 1: Grid dim3(num_experts + 1, steps), Block dim3(256).
// Blocks 0..num_experts-1 compute g_router_logits_scratch[s, e] with 8-wave
// coalesced 4KB row reads. Block num_experts computes shared_gate_scale[s] =
// sigmoid(shared_gate_weight . x).
__launch_bounds__(kThreads) __global__ void MoERouterLogitsKernel(
    const std::uint16_t* __restrict__ router_weight_bf16,
    const std::uint16_t* __restrict__ shared_gate_weight_bf16,
    const float* __restrict__ ffn_in, float* __restrict__ shared_gate_scale,
    std::size_t steps, std::size_t num_experts, std::size_t hidden_size) {
  const std::size_t row = blockIdx.x;
  const std::size_t s = blockIdx.y;
  if (row > num_experts || s >= steps) {
    return;
  }

  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const float4* x_vec =
      reinterpret_cast<const float4*>(ffn_in + s * hidden_size);
  const std::size_t vec_cols = hidden_size / 8U;

  const std::uint16_t* w_ptr = (row < num_experts)
                                   ? (router_weight_bf16 + row * hidden_size)
                                   : shared_gate_weight_bf16;
  const uint4* w_vec = reinterpret_cast<const uint4*>(w_ptr);

  float acc = 0.0F;
  for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
    const uint4 w_pack = w_vec[idx];
    const float4 x0 = x_vec[2U * idx];
    const float4 x1 = x_vec[2U * idx + 1U];
    AccumulateBf16Vec8(w_pack, x0, x1, acc);
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
    if (row < num_experts) {
      g_router_logits_scratch[s * num_experts + row] = total;
    } else {
      shared_gate_scale[s] = 1.0F / (1.0F + expf(-total));
    }
  }
}

// Kernel 2: Grid dim3(steps), Block dim3(256).
// Softmax over 256 expert logits + 8-wave parallel tournament argmax for top-8.
__launch_bounds__(kThreads) __global__
    void MoERouterTop8SelectKernel(std::int32_t* __restrict__ top_indices,
                                   float* __restrict__ top_weights,
                                   std::size_t steps, std::size_t num_experts,
                                   std::size_t top_k) {
  const std::size_t s = blockIdx.x;
  if (s >= steps) {
    return;
  }

  __shared__ float s_red[256];
  __shared__ float s_wave_p[kWavesPerBlock];
  __shared__ std::int32_t s_wave_e[kWavesPerBlock];
  __shared__ std::int32_t s_chosen_e;

  const std::size_t tid = threadIdx.x;
  const std::size_t wave_id = tid / kWaveSize;
  const std::size_t lane = tid % kWaveSize;

  const float my_logit = tid < num_experts
                             ? g_router_logits_scratch[s * num_experts + tid]
                             : -std::numeric_limits<float>::infinity();
  s_red[tid] = my_logit;
  __syncthreads();
  for (std::size_t stride = 128U; stride != 0; stride >>= 1U) {
    if (tid < stride) {
      s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
    }
    __syncthreads();
  }
  const float max_logit = s_red[0];
  __syncthreads();

  const float my_exp = tid < num_experts ? expf(my_logit - max_logit) : 0.0F;
  s_red[tid] = my_exp;
  __syncthreads();
  for (std::size_t stride = 128U; stride != 0; stride >>= 1U) {
    if (tid < stride) {
      s_red[tid] += s_red[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = 1.0F / s_red[0];
  float my_p = tid < num_experts ? (my_exp * inv_sum) : -1.0F;

  std::int32_t* out_idx = top_indices + s * top_k;
  float* out_w = top_weights + s * top_k;
  float weight_sum = 0.0F;

  for (std::size_t k = 0; k < top_k; ++k) {
    float cand_p = my_p;
    std::int32_t cand_e = static_cast<std::int32_t>(tid);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_p = __shfl_xor(cand_p, offset, 32);
      const std::int32_t other_e = __shfl_xor(cand_e, offset, 32);
      if (other_p > cand_p || (other_p == cand_p && other_e < cand_e)) {
        cand_p = other_p;
        cand_e = other_e;
      }
    }
    if (lane == 0) {
      s_wave_p[wave_id] = cand_p;
      s_wave_e[wave_id] = cand_e;
    }
    __syncthreads();

    if (tid == 0) {
      float best_p = -1.0F;
      std::int32_t best_e = 0;
#pragma unroll
      for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
        if (s_wave_p[w] > best_p) {
          best_p = s_wave_p[w];
          best_e = s_wave_e[w];
        }
      }
      out_idx[k] = best_e;
      out_w[k] = best_p;
      weight_sum += best_p;
      s_chosen_e = best_e;
    }
    __syncthreads();
    if (static_cast<std::int32_t>(tid) == s_chosen_e) {
      my_p = -1.0F;
    }
  }

  if (tid == 0) {
    const float inv_w_sum = weight_sum > 0.0F ? (1.0F / weight_sum) : 0.0F;
    for (std::size_t k = 0; k < top_k; ++k) {
      out_w[k] *= inv_w_sum;
    }
  }
}

// Fused Gate + Up + SwiGLU for top_k FP8 routed experts AND 1 BF16 shared
// expert: 1 output row per 256-thread workgroup (8 waves): For FP8 routed
// experts, col_blocks = 16 (each block is 128 FP8 bytes = 32 lanes * 4 bytes).
// Wave `wave_id` (0..7) processes `cb = wave_id` and `cb = wave_id + 8` in just
// 2 coalesced 1KB loads!
__device__ __forceinline__ float DotFp8Vec4(std::uint32_t pack,
                                            const float4& xp) {
  float dot = 0.0F;
  dot =
      fmaf(Fp8E4m3ToFloat(static_cast<std::uint8_t>(pack & 0xffU)), xp.x, dot);
  dot = fmaf(Fp8E4m3ToFloat(static_cast<std::uint8_t>((pack >> 8U) & 0xffU)),
             xp.y, dot);
  dot = fmaf(Fp8E4m3ToFloat(static_cast<std::uint8_t>((pack >> 16U) & 0xffU)),
             xp.z, dot);
  dot = fmaf(Fp8E4m3ToFloat(static_cast<std::uint8_t>(pack >> 24U)), xp.w, dot);
  return dot;
}

__launch_bounds__(kThreads) __global__ void MoEGateUpSwiGluKernel(
    const std::uint8_t* __restrict__ experts_gate_fp8,
    const float* __restrict__ experts_gate_scale,
    const std::uint8_t* __restrict__ experts_up_fp8,
    const float* __restrict__ experts_up_scale,
    const std::uint16_t* __restrict__ shared_gate_bf16,
    const std::uint16_t* __restrict__ shared_up_bf16,
    const float* __restrict__ ffn_in,
    const std::int32_t* __restrict__ top_indices,
    float* __restrict__ expert_inter, std::size_t steps, std::size_t top_k,
    std::size_t inter_size, std::size_t hidden_size) {
  const std::size_t row = blockIdx.x;
  const std::size_t slot = blockIdx.y;
  const std::size_t s = blockIdx.z;
  if (row >= inter_size || slot > top_k || s >= steps) {
    return;
  }

  __shared__ float gate_sums[kWavesPerBlock];
  __shared__ float up_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const float* x_row = ffn_in + s * hidden_size;

  if (slot == top_k) {
    // Shared expert (BF16) - all 8 waves coalesce across the 2048-col row
    const uint4* g_vec =
        reinterpret_cast<const uint4*>(shared_gate_bf16 + row * hidden_size);
    const uint4* u_vec =
        reinterpret_cast<const uint4*>(shared_up_bf16 + row * hidden_size);
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    const std::size_t vec_cols = hidden_size / 8U;

    float acc_g = 0.0F;
    float acc_u = 0.0F;
    for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
      const uint4 g_pack = g_vec[idx];
      const uint4 u_pack = u_vec[idx];
      const float4 x0 = x_vec[2U * idx];
      const float4 x1 = x_vec[2U * idx + 1U];
      AccumulateBf16Vec8(g_pack, x0, x1, acc_g);
      AccumulateBf16Vec8(u_pack, x0, x1, acc_u);
    }
    const float w_g = WaveReduceSum(acc_g);
    const float w_u = WaveReduceSum(acc_u);
    if (lane == 0) {
      gate_sums[wave_id] = w_g;
      up_sums[wave_id] = w_u;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float gate = 0.0F, up = 0.0F;
#pragma unroll
      for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
        gate += gate_sums[w];
        up += up_sums[w];
      }
      const float silu_g = gate / (1.0F + expf(-gate));
      expert_inter[(s * (top_k + 1U) + slot) * inter_size + row] = silu_g * up;
    }
    return;
  }

  // Routed expert (F8_E4M3 + 128x128 block scale_inv)
  // 8 waves coalesce across col_blocks = 16 (each wave handles cb = wave_id,
  // wave_id + 8)
  const std::size_t expert_id =
      static_cast<std::size_t>(top_indices[s * top_k + slot]);
  const std::size_t row_blocks = inter_size / 128U;   // 4
  const std::size_t col_blocks = hidden_size / 128U;  // 16
  const std::size_t rb = row >> 7U;

  const std::uint32_t* g_u32 = reinterpret_cast<const std::uint32_t*>(
      experts_gate_fp8 + (expert_id * inter_size + row) * hidden_size);
  const std::uint32_t* u_u32 = reinterpret_cast<const std::uint32_t*>(
      experts_up_fp8 + (expert_id * inter_size + row) * hidden_size);
  const float* g_scales =
      experts_gate_scale + (expert_id * row_blocks + rb) * col_blocks;
  const float* u_scales =
      experts_up_scale + (expert_id * row_blocks + rb) * col_blocks;
  const float4* x_f4 = reinterpret_cast<const float4*>(x_row);

  float acc_g = 0.0F;
  float acc_u = 0.0F;
  for (std::size_t cb = wave_id; cb < col_blocks; cb += kWavesPerBlock) {
    const std::size_t vec_idx = cb * kWaveSize + lane;
    const std::uint32_t gp = g_u32[vec_idx];
    const std::uint32_t up_pack = u_u32[vec_idx];
    const float4 xp = x_f4[vec_idx];
    acc_g = fmaf(DotFp8Vec4(gp, xp), g_scales[cb], acc_g);
    acc_u = fmaf(DotFp8Vec4(up_pack, xp), u_scales[cb], acc_u);
  }

  const float w_g = WaveReduceSum(acc_g);
  const float w_u = WaveReduceSum(acc_u);
  if (lane == 0) {
    gate_sums[wave_id] = w_g;
    up_sums[wave_id] = w_u;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float gate = 0.0F, up = 0.0F;
#pragma unroll
    for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
      gate += gate_sums[w];
      up += up_sums[w];
    }
    const float silu_g = gate / (1.0F + expf(-gate));
    expert_inter[(s * (top_k + 1U) + slot) * inter_size + row] = silu_g * up;
  }
}

// Fused Down Projection + Weighted Combination + Residual Add:
// Grid: dim3(hidden_size = 2048, steps), Block: dim3(256) = 8 waves.
// Wave `wave_id` (0..7) computes routed expert `k = wave_id` in parallel!
// All 8 waves also cooperatively compute the BF16 shared expert for `row`.
__launch_bounds__(kThreads) __global__ void MoEDownCombineKernel(
    const std::uint8_t* __restrict__ experts_down_fp8,
    const float* __restrict__ experts_down_scale,
    const std::uint16_t* __restrict__ shared_down_bf16,
    const float* __restrict__ expert_inter,
    const std::int32_t* __restrict__ top_indices,
    const float* __restrict__ top_weights,
    const float* __restrict__ shared_gate_scale,
    float* __restrict__ hidden_inout, std::size_t steps, std::size_t top_k,
    std::size_t inter_size, std::size_t hidden_size) {
  const std::size_t row = blockIdx.x;
  const std::size_t s = blockIdx.y;
  if (row >= hidden_size || s >= steps) {
    return;
  }

  __shared__ float wave_sums[kWavesPerBlock];
  const std::size_t wave_id = threadIdx.x / kWaveSize;
  const std::size_t lane = threadIdx.x % kWaveSize;
  const std::size_t row_blocks = hidden_size / 128U;  // 16
  const std::size_t col_blocks = inter_size / 128U;   // 4
  const std::size_t rb = row >> 7U;

  float wave_total = 0.0F;

  // 1. Wave `wave_id` handles routed expert `k = wave_id` (parallel across all
  // 8 experts!)
  for (std::size_t k = wave_id; k < top_k; k += kWavesPerBlock) {
    const std::size_t expert_id =
        static_cast<std::size_t>(top_indices[s * top_k + k]);
    const float w_k = top_weights[s * top_k + k];
    const std::uint32_t* d_u32 = reinterpret_cast<const std::uint32_t*>(
        experts_down_fp8 + (expert_id * hidden_size + row) * inter_size);
    const float* d_scales =
        experts_down_scale + (expert_id * row_blocks + rb) * col_blocks;
    const float4* inter_f4 = reinterpret_cast<const float4*>(
        expert_inter + (s * (top_k + 1U) + k) * inter_size);

    float exp_acc = 0.0F;
#pragma unroll
    for (std::size_t cb = 0; cb < 4U; ++cb) {
      const std::size_t vec_idx = cb * kWaveSize + lane;
      const std::uint32_t dp = d_u32[vec_idx];
      const float4 xp = inter_f4[vec_idx];
      exp_acc = fmaf(DotFp8Vec4(dp, xp), d_scales[cb], exp_acc);
    }
    wave_total = fmaf(exp_acc, w_k, wave_total);
  }

  // 2. All 8 waves cooperatively compute BF16 shared expert (64 uint4 elements
  // = 8 waves * 8 lanes)
  {
    const uint4* sd_vec =
        reinterpret_cast<const uint4*>(shared_down_bf16 + row * inter_size);
    const float4* inter_f4 = reinterpret_cast<const float4*>(
        expert_inter + (s * (top_k + 1U) + top_k) * inter_size);
    const std::size_t vec_cols = inter_size / 8U;  // 64
    float shared_acc = 0.0F;
    for (std::size_t idx = threadIdx.x; idx < vec_cols; idx += kThreads) {
      const uint4 w_pack = sd_vec[idx];
      const float4 x0 = inter_f4[2U * idx];
      const float4 x1 = inter_f4[2U * idx + 1U];
      AccumulateBf16Vec8(w_pack, x0, x1, shared_acc);
    }
    wave_total = fmaf(shared_acc, shared_gate_scale[s], wave_total);
  }

  const float w_sum = WaveReduceSum(wave_total);
  if (lane == 0) {
    wave_sums[wave_id] = w_sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
#pragma unroll
    for (std::size_t w = 0; w < kWavesPerBlock; ++w) {
      sum += wave_sums[w];
    }
    hidden_inout[s * hidden_size + row] += sum;
  }
}

}  // namespace

void LaunchBackboneRmsNorm(const float* input, const float* weight,
                           float* output, std::size_t rows, std::size_t columns,
                           float eps, hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(kThreads), 0, stream,
                     input, weight, output, rows, columns, eps);
}

void LaunchBackboneResidualAdd(float* x_inout, const float* update,
                               std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(ResidualAddKernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, x_inout, update, elements);
}

void LaunchBackboneResidualAddRmsNorm(float* x_inout, const float* update,
                                      const float* weight, float* normed_out,
                                      std::size_t rows, std::size_t columns,
                                      float eps, hipStream_t stream) {
  hipLaunchKernelGGL(ResidualAddRmsNormKernel, dim3(rows), dim3(kThreads), 0,
                     stream, x_inout, update, weight, normed_out, rows, columns,
                     eps);
}

void LaunchBackboneFullQkvProjFusedGemv(const std::uint16_t* q_w,
                                        const std::uint16_t* k_w,
                                        const std::uint16_t* v_w,
                                        const float* input_f32, float* q_out,
                                        float* k_out, float* v_out,
                                        std::size_t q_rows, std::size_t kv_rows,
                                        std::size_t cols, hipStream_t stream) {
  const std::size_t total_rows = q_rows + 2U * kv_rows;
  hipLaunchKernelGGL(FullQkvProjFusedGemvKernel, dim3(total_rows),
                     dim3(kThreads), 0, stream, q_w, k_w, v_w, input_f32, q_out,
                     k_out, v_out, q_rows, kv_rows, cols);
}

void LaunchBackboneLinearProjFusedGemv(
    const std::uint16_t* qkv_w, const std::uint16_t* z_w,
    const std::uint16_t* b_w, const std::uint16_t* a_w, const float* input_f32,
    float* qkv_out, float* z_out, float* b_out, float* a_out,
    std::size_t qkv_rows, std::size_t z_rows, std::size_t ba_rows,
    std::size_t cols, hipStream_t stream) {
  const std::size_t total_rows = qkv_rows + z_rows + 2U * ba_rows;
  hipLaunchKernelGGL(LinearProjFusedGemvKernel, dim3(total_rows),
                     dim3(kThreads), 0, stream, qkv_w, z_w, b_w, a_w, input_f32,
                     qkv_out, z_out, b_out, a_out, qkv_rows, z_rows, ba_rows,
                     cols);
}

void LaunchBackboneBf16Gemv(const std::uint16_t* weight_bf16,
                            const float* input_f32, float* output_f32,
                            std::size_t batch, std::size_t rows,
                            std::size_t cols, hipStream_t stream) {
  hipLaunchKernelGGL(Bf16GemvKernel, dim3(rows), dim3(kThreads), 0, stream,
                     weight_bf16, input_f32, output_f32, batch, rows, cols);
}

void LaunchBackboneFloatToBf16(const float* input, std::uint16_t* output,
                               std::size_t elements, hipStream_t stream) {
  hipLaunchKernelGGL(FloatToBf16Kernel, dim3(Blocks(elements)), dim3(kThreads),
                     0, stream, input, output, elements);
}

void LaunchBackboneFullAttnPrep(
    const float* q_full, const float* k_in, const float* v_in,
    const float* q_norm, const float* k_norm, float* q_out, float* key_cache,
    float* value_cache, std::size_t steps, std::size_t start_pos,
    std::size_t heads, std::size_t kv_heads, std::size_t head_dim,
    std::size_t rotary_dim, float rope_theta, float eps, hipStream_t stream) {
  const std::size_t total_blocks = steps * (heads + kv_heads);
  hipLaunchKernelGGL(FullAttnPrepKernel, dim3(total_blocks), dim3(kThreads), 0,
                     stream, q_full, k_in, v_in, q_norm, k_norm, q_out,
                     key_cache, value_cache, steps, start_pos, heads, kv_heads,
                     head_dim, rotary_dim, rope_theta, eps);
}

void LaunchBackboneFullAttnGqaAndGate(const float* q, const float* key_cache,
                                      const float* value_cache,
                                      const float* q_full, float* attended_out,
                                      std::size_t steps, std::size_t start_pos,
                                      std::size_t heads, std::size_t kv_heads,
                                      std::size_t head_dim,
                                      hipStream_t stream) {
  const std::size_t max_keys = start_pos + steps;
  const std::size_t shared_bytes = (max_keys + kThreads) * sizeof(float);
  hipLaunchKernelGGL(FullAttnGqaAndGateKernel, dim3(steps * heads),
                     dim3(kThreads), shared_bytes, stream, q, key_cache,
                     value_cache, q_full, attended_out, steps, start_pos, heads,
                     kv_heads, head_dim);
}

void LaunchBackboneLinearConv1dSilu(const float* qkv_in,
                                    const float* conv_weight, float* conv_state,
                                    float* qkv_conv_out, std::size_t steps,
                                    std::size_t channels, hipStream_t stream) {
  hipLaunchKernelGGL(LinearConv1dSiluKernel, dim3(Blocks(channels)),
                     dim3(kThreads), 0, stream, qkv_in, conv_weight, conv_state,
                     qkv_conv_out, steps, channels);
}

void LaunchBackboneGatedDeltaNetRecurrent(
    const float* qkv_conv, const float* z, const float* b, const float* a,
    const float* dt_bias, const float* neg_exp_a, const float* norm_weight,
    float* ssm_state, float* attn_out, std::size_t steps, std::size_t k_heads,
    std::size_t v_heads, std::size_t head_dim, float eps, hipStream_t stream) {
  hipLaunchKernelGGL(GatedDeltaNetStateKernel, dim3(v_heads, 4U),
                     dim3(kThreads), 0, stream, qkv_conv, b, a, dt_bias,
                     neg_exp_a, ssm_state, attn_out, steps, k_heads, v_heads,
                     head_dim, eps);
  hipLaunchKernelGGL(GatedDeltaNetNormSiluKernel, dim3(steps * v_heads),
                     dim3(128U), 0, stream, attn_out, z, norm_weight,
                     steps * v_heads, head_dim, eps);
}

void LaunchBackboneMoERouter(const std::uint16_t* router_weight_bf16,
                             const std::uint16_t* shared_gate_weight_bf16,
                             const float* ffn_in, std::int32_t* top_indices,
                             float* top_weights, float* shared_gate_scale,
                             std::size_t steps, std::size_t num_experts,
                             std::size_t top_k, std::size_t hidden_size,
                             hipStream_t stream) {
  hipLaunchKernelGGL(MoERouterLogitsKernel, dim3(num_experts + 1U, steps),
                     dim3(kThreads), 0, stream, router_weight_bf16,
                     shared_gate_weight_bf16, ffn_in, shared_gate_scale, steps,
                     num_experts, hidden_size);
  hipLaunchKernelGGL(MoERouterTop8SelectKernel, dim3(steps), dim3(kThreads), 0,
                     stream, top_indices, top_weights, steps, num_experts,
                     top_k);
}

void LaunchBackboneMoEGateUpSwiGlu(
    const std::uint8_t* experts_gate_fp8, const float* experts_gate_scale,
    const std::uint8_t* experts_up_fp8, const float* experts_up_scale,
    const std::uint16_t* shared_gate_bf16, const std::uint16_t* shared_up_bf16,
    const float* ffn_in, const std::int32_t* top_indices, float* expert_inter,
    std::size_t steps, std::size_t top_k, std::size_t inter_size,
    std::size_t hidden_size, hipStream_t stream) {
  hipLaunchKernelGGL(MoEGateUpSwiGluKernel, dim3(inter_size, top_k + 1U, steps),
                     dim3(kThreads), 0, stream, experts_gate_fp8,
                     experts_gate_scale, experts_up_fp8, experts_up_scale,
                     shared_gate_bf16, shared_up_bf16, ffn_in, top_indices,
                     expert_inter, steps, top_k, inter_size, hidden_size);
}

void LaunchBackboneMoEDownCombine(
    const std::uint8_t* experts_down_fp8, const float* experts_down_scale,
    const std::uint16_t* shared_down_bf16, const float* expert_inter,
    const std::int32_t* top_indices, const float* top_weights,
    const float* shared_gate_scale, float* hidden_inout, std::size_t steps,
    std::size_t top_k, std::size_t inter_size, std::size_t hidden_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(MoEDownCombineKernel, dim3(hidden_size, steps),
                     dim3(kThreads), 0, stream, experts_down_fp8,
                     experts_down_scale, shared_down_bf16, expert_inter,
                     top_indices, top_weights, shared_gate_scale, hidden_inout,
                     steps, top_k, inter_size, hidden_size);
}

}  // namespace gufo::models::fish_drama3::hip

#endif
