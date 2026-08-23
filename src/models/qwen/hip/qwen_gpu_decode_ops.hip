#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/models/qwen/hip/qwen_gpu_quant_ops.hpp"
#include "src/core/gguf_reader.hpp"

namespace strix::hip {

namespace {

// Quant block layouts (Q8_0 / Q5_K / Q6_K) and quant row-dot helpers are
// shared via src/models/qwen/hip/qwen_gpu_quant_ops.hpp.

__global__ void SwiGLUKernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             float* __restrict__ out,
                             std::size_t intermediate_size) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < intermediate_size) {
    const float g = gate[idx];
    const float silu_g = g / (1.0F + expf(-g));
    out[idx] = silu_g * up[idx];
  }
}

// Q8_K block layout is defined in the shared qwen_gpu_quant_ops.hpp header.

// Serial (per-thread) row dot for a Q8_K weight row: iterates every 256-wide
// block, computes the activation scale from max|x|, quantizes x to int8 and
// accumulates an integer MAC with qs, then applies d_w * scale. Correctness
// path used by the fused SwiGLU fallback for Q8_K weight rows.
__device__ float Q8KRowDot(const Q8KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ8KBlockSize;
  float sumf = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const Q8KBlock& wblk = row_blocks[b];
    const float d_w = wblk.d;
    const float* xb = x + (b * kQ8KBlockSize);

    float max = 0.0F;
    for (std::size_t k = 0; k < kQ8KBlockSize; ++k) {
      max = fmaxf(max, fabsf(xb[k]));
    }
    const float scale = (max > 0.0F) ? (max / 127.0F) : 0.0F;

    int acc = 0;
    for (std::size_t k = 0; k < kQ8KBlockSize; ++k) {
      const float xv = (scale > 0.0F) ? (xb[k] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      acc += static_cast<int>(wblk.qs[k]) * xq;
    }

    sumf += d_w * scale * static_cast<float>(acc);
  }
  return sumf;
}

// Q5KValue / Q6KValue (and the GetQKScaleMin unpacking they use) are defined
// in the shared qwen_gpu_quant_ops.hpp header.
// opt-r4-q5k-q6k: serial (per-thread) row dot for a Q5_K weight row: dequant
// each element to fp and accumulate a fp MAC against x. Used by RowDot and the
// fused kernels' fallback. Requires K % 256 == 0.
__device__ float Q5KRowDot(const Q5KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ5KBlockSize;
  float sum = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const float* xb = x + (b * kQ5KBlockSize);
    for (std::size_t i = 0; i < kQ5KBlockSize; ++i) {
      sum += Q5KValue(row_blocks[b], i) * xb[i];
    }
  }
  return sum;
}

// opt-r4-q5k-q6k: serial (per-thread) row dot for a Q6_K weight row: dequant
// each element to fp and accumulate a fp MAC against x. Requires K % 256 == 0.
__device__ float Q6KRowDot(const Q6KBlock* __restrict__ row_blocks,
                           const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ6KBlockSize;
  float sum = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const float* xb = x + (b * kQ6KBlockSize);
    for (std::size_t i = 0; i < kQ6KBlockSize; ++i) {
      sum += Q6KValue(row_blocks[b], i) * xb[i];
    }
  }
  return sum;
}

// opt-r2-q8_0: Q8_0 row dot. Runs AT Q8 (integer MAC) exactly like Q8KRowDot
// but per-32-element block with a half weight scale read via __half2float.
// Requires K % 32 == 0.
__device__ float Q8_0RowDot(const Q8_0Block* __restrict__ row_blocks,
                            const float* __restrict__ x, std::size_t K) {
  const std::size_t num_blocks = K / kQ8_0BlockSize;
  float sumf = 0.0F;
  for (std::size_t b = 0; b < num_blocks; ++b) {
    const Q8_0Block& wblk = row_blocks[b];
    const float d_w = __half2float(wblk.d);
    const float* xb = x + (b * kQ8_0BlockSize);

    float max = 0.0F;
    for (std::size_t k = 0; k < kQ8_0BlockSize; ++k) {
      max = fmaxf(max, fabsf(xb[k]));
    }
    const float scale = (max > 0.0F) ? (max / 127.0F) : 0.0F;

    int acc = 0;
    for (std::size_t k = 0; k < kQ8_0BlockSize; ++k) {
      const float xv = (scale > 0.0F) ? (xb[k] / scale) : 0.0F;
      int xq = static_cast<int>(roundf(xv));
      xq = (xq > 127) ? 127 : xq;
      xq = (xq < -127) ? -127 : xq;
      acc += static_cast<int>(wblk.qs[k]) * xq;
    }

    sumf += d_w * scale * static_cast<float>(acc);
  }
  return sumf;
}

// QuantBlockBytes / QuantBlockQK are defined in the shared
// qwen_gpu_quant_ops.hpp header.
// Central dispatcher for a single quantized weight row's dot product. Routes
// on the weight's GgmlType and OWNS the type-aware row stride: given the
// weight base pointer and the output row index, it computes
//   base + row_idx * (K / QuantBlockQK(type) * QuantBlockBytes(type))
// then calls the type-specific row dot. Q8_0/Q8_K run at Q8 (integer MAC);
// Q5_K/Q6_K dequant to fp and accumulate a fp MAC (both mirror the CPU
// dequant oracles). bf16/fp32 are handled by the vectorized kernels and never
// reach here, so unsupported types return 0 (acceptable until those types are
// wired and the model is run).
__device__ inline float RowDot(core::GgmlType type, const void* __restrict__ base,
                               std::size_t row_idx, const float* __restrict__ x,
                               std::size_t K) {
  const std::size_t qk = QuantBlockQK(type);
  const std::size_t block_bytes = QuantBlockBytes(type);
  const char* row = static_cast<const char*>(base) + (row_idx * (K / qk * block_bytes));
  switch (type) {
    case core::GgmlType::kQ8_0:
      return Q8_0RowDot(reinterpret_cast<const Q8_0Block*>(row), x, K);
    case core::GgmlType::kQ8_K:
      return Q8KRowDot(reinterpret_cast<const Q8KBlock*>(row), x, K);
    case core::GgmlType::kQ5_K:
      return Q5KRowDot(reinterpret_cast<const Q5KBlock*>(row), x, K);
    case core::GgmlType::kQ6_K:
      return Q6KRowDot(reinterpret_cast<const Q6KBlock*>(row), x, K);
    default:
      return 0.0F;  // remaining K-typed rows unwired
  }
}

// opt-c014-layer-prefetch: touch up to 16 bytes per 4KiB page of a weight
// region so the page mappings are resident before the real projection kernels
// read them. Reads only; the guarded store can never fire (its value depends
// on the loaded data) and exists to defeat dead-code elimination. Reads are
// clamped to `bytes` so the last partial page never runs past the region.
__launch_bounds__(256, 4) __global__
    void LayerWeightPrefetchKernel(const void* __restrict__ data,
                                   std::size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  constexpr std::size_t kPageBytes = 4096U;
  constexpr std::size_t kTouchBytes = 16U;
  const std::size_t num_pages = (bytes + kPageBytes - 1U) / kPageBytes;
  float acc = 0.0F;
  for (std::size_t page = threadIdx.x; page < num_pages; page += blockDim.x) {
    const std::size_t base = page * kPageBytes;
    const std::size_t n = std::min<std::size_t>(kTouchBytes, bytes - base);
    for (std::size_t i = 0; i < n; ++i) {
      acc += static_cast<float>(p[base + i]);
    }
  }
  if (acc == 1.234567e30F) {
    const_cast<unsigned char*>(p)[0] = 0U;
  }
}

}  // namespace

// opt-c014-layer-prefetch: touch one 16B chunk per 4KiB page of a weight
// region on the given stream (async). Reads only; the guarded store inside the
// kernel can never fire.
void LaunchLayerWeightPrefetch(const void* data, std::size_t bytes,
                               hipStream_t stream) {
  if (data == nullptr || bytes == 0) {
    return;
  }
  const std::size_t num_pages = (bytes + 4095U) / 4096U;
  const std::size_t grid = (num_pages + 255U) / 256U;
  LayerWeightPrefetchKernel<<<grid, 256, 0, stream>>>(data, bytes);
}

__global__ void SSMConvKernel(const float* __restrict__ qkv_in,
                              const float* __restrict__ conv_weights,
                              float* __restrict__ conv_state,
                              float* __restrict__ conv_out, std::size_t qkv_dim,
                              std::uint32_t layer_idx,
                              SsmReplayCapture replay_capture) {
  const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= qkv_dim)
    return;

  if (replay_capture.qkv != nullptr && replay_capture.position != nullptr &&
      replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
    const std::size_t slot =
        static_cast<std::size_t>(replay_capture.position[0]) %
        kSsmReplayCapacity;
    const std::size_t offset =
        ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
         qkv_dim) +
        c;
    replay_capture.qkv[offset] = qkv_in[c];
  }

  const std::size_t c_off = c * 4;
  conv_state[c_off + 0] = conv_state[c_off + 1];
  conv_state[c_off + 1] = conv_state[c_off + 2];
  conv_state[c_off + 2] = conv_state[c_off + 3];
  conv_state[c_off + 3] = qkv_in[c];

  float dot = 0.0F;
  if (conv_weights != nullptr) {
    for (std::size_t k = 0; k < 4; ++k) {
      dot += conv_state[c_off + k] * conv_weights[c * 4 + k];
    }
  } else {
    dot = qkv_in[c];
  }

  const float sig = 1.0F / (1.0F + expf(-dot));
  conv_out[c] = dot * sig;
}

__global__ void DeltaNetRecurrenceKernel(
    const float* __restrict__ conv_out, float* __restrict__ deltanet_state,
    const float* __restrict__ alpha_buf, const float* __restrict__ beta_buf,
    const float* __restrict__ ssm_a, const float* __restrict__ ssm_dt,
    const float* __restrict__ ssm_norm, const float* __restrict__ gate,
    float* __restrict__ out_buf, std::uint32_t layer_idx,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, SsmReplayCapture replay_capture) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::size_t tid = threadIdx.x;

  const float* q_ptr = conv_out;
  const float* k_ptr = conv_out + (num_key_heads * key_dim);
  const float* v_ptr = conv_out + (num_key_heads * key_dim * 2);

  const std::uint32_t kh_idx = h % num_key_heads;
  const float* q_h = q_ptr + (kh_idx * key_dim);
  const float* k_h = k_ptr + (kh_idx * key_dim);
  const float* v_h = v_ptr + (h * val_dim);
  float* o_h = out_buf + (h * val_dim);

  float* s_matrix =
      deltanet_state + ((static_cast<std::size_t>(layer_idx) * num_heads + h) *
                        key_dim * val_dim);

  // 1. Decay & learning rate: alpha = exp(ssm_a * softplus(alpha + dt)), beta =
  // sigmoid(beta)
  const float dt = (ssm_dt != nullptr) ? ssm_dt[h] : 0.0F;
  const float a_val = (ssm_a != nullptr) ? ssm_a[h] : -0.05F;
  const float alpha_in = (alpha_buf != nullptr) ? alpha_buf[h] : 0.0F;
  const float beta_in = (beta_buf != nullptr) ? beta_buf[h] : 0.0F;

  if (tid == 0 && replay_capture.alpha != nullptr &&
      replay_capture.beta != nullptr && replay_capture.position != nullptr &&
      replay_capture.enabled != nullptr && replay_capture.enabled[0] != 0U) {
    const std::size_t slot =
        static_cast<std::size_t>(replay_capture.position[0]) %
        kSsmReplayCapacity;
    const std::size_t offset =
        ((static_cast<std::size_t>(layer_idx) * kSsmReplayCapacity + slot) *
         num_heads) +
        h;
    replay_capture.alpha[offset] = alpha_in;
    replay_capture.beta[offset] = beta_in;
  }

  const float alpha_biased = alpha_in + dt;
  const float alpha_softplus =
      (alpha_biased > 20.0F) ? alpha_biased : log1pf(expf(alpha_biased));
  const float alpha_h = expf(alpha_softplus * a_val);
  const float beta_h = 1.0F / (1.0F + expf(-beta_in));

  // 2. Normalize key ||k_h||_2 and query ||q_h||_2
  __shared__ float s_norm[2];
  if (tid == 0) {
    float k_sq = 0.0F;
    float q_sq = 0.0F;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      k_sq += k_h[i] * k_h[i];
      q_sq += q_h[i] * q_h[i];
    }
    s_norm[0] = 1.0F / sqrtf(k_sq + 1e-6F);
    s_norm[1] = 1.0F / sqrtf(q_sq + 1e-6F);
  }
  __syncthreads();
  const float q_scale = (1.0F / sqrtf(static_cast<float>(key_dim))) * s_norm[1];
  const float inv_k_norm = s_norm[0];

  // 3. Associative retrieval: u_j = sum_i S[j, i] * (k[i] * inv_k_norm)
  if (tid < val_dim) {
    const std::size_t j = tid;
    float* s_row = s_matrix + (j * key_dim);

    // Decay state while computing the retrieval. Keeping the decayed value
    // live avoids a second full read of the state row.
    float u_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] * alpha_h;
      s_row[i] = state;
      u_j += state * (k_h[i] * inv_k_norm);
    }
    const float d_j = (v_h[j] - u_j) * beta_h;

    // Apply the delta and compute the query readout from the same updated
    // value, avoiding another full read of the state row.
    float o_j = 0.0F;
    for (std::size_t i = 0; i < key_dim; ++i) {
      const float state = s_row[i] + d_j * (k_h[i] * inv_k_norm);
      s_row[i] = state;
      o_j += state * (q_h[i] * q_scale);
    }
    o_h[j] = o_j;
  }
  __syncthreads();

  // 4. Per-head RMSNorm on o_h (val_dim = 256 or 128 elements)
  __shared__ float s_sum[256];
  if (tid < val_dim) {
    s_sum[tid] = o_h[tid] * o_h[tid];
  } else {
    s_sum[tid] = 0.0F;
  }
  __syncthreads();
  for (std::size_t s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s)
      s_sum[tid] += s_sum[tid + s];
    __syncthreads();
  }
  const float mean_sq = s_sum[0] / static_cast<float>(val_dim);
  const float rms = rsqrtf(mean_sq + 1e-6F);
  __syncthreads();

  if (tid < val_dim) {
    const float w = (ssm_norm != nullptr) ? ssm_norm[tid] : 1.0F;
    float val = o_h[tid] * rms * w;
    if (gate != nullptr) {
      const float g = gate[h * val_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      val *= (g * sig);  // SiLU gating
    }
    o_h[tid] = val;
  }
}

void LaunchSwiGLU(const float* gate, const float* up, float* out,
                  std::size_t intermediate_size, hipStream_t stream) {
  const std::size_t block_size = 256;
  const std::size_t grid_size =
      (intermediate_size + block_size - 1) / block_size;
  SwiGLUKernel<<<grid_size, block_size, 0, stream>>>(gate, up, out,
                                                     intermediate_size);
}

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream, SsmReplayCapture replay_capture) {
  const std::size_t block_size = 256;
  const std::size_t grid_size = (qkv_size + block_size - 1) / block_size;
  float* conv_layer_state =
      conv_state + (static_cast<std::size_t>(layer_idx) * qkv_size * 4);

  SSMConvKernel<<<grid_size, block_size, 0, stream>>>(
      qkv_in, conv_weights, conv_layer_state, conv_out, qkv_size, layer_idx,
      replay_capture);

  DeltaNetRecurrenceKernel<<<num_heads, 256, 0, stream>>>(
      conv_out, deltanet_state, alpha_buf, beta_buf, ssm_a, ssm_dt, ssm_norm,
      gate, out_buf, layer_idx, num_key_heads, num_heads, key_dim, val_dim,
      replay_capture);
}

// =========================================================================
// Batched GPU Operators for High-Throughput Prompt Processing (Prefill)
// =========================================================================

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)

