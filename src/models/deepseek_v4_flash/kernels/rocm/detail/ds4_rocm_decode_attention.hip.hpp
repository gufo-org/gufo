#pragma once

#include <hip/hip_runtime.h>

inline constexpr uint32_t DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP = 1024u;

// DSpark support attention. The launcher guarantees r < n_raw <= raw_cap and
// raw_start < raw_cap, so ring indexing needs at most one subtraction. Keep the
// floating-point accumulation order unchanged; integer division in the value
// loop was particularly expensive at long contexts.
__global__ static void attention_noncausal_raw_batch_heads_kernel(
    float* heads, const float* sinks, const float* q, const float* raw_kv,
    uint32_t n_tokens, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
    uint32_t n_head, uint32_t head_dim) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= n_tokens || h >= n_head)
    return;

  extern __shared__ float sh_scores[];
  const float* qh = q + ((uint64_t)tok * n_head + h) * head_dim;
  const float scale = rsqrtf((float)head_dim);
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    const uint32_t row =
        (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++)
      dot += qh[d] * kv[d];
    sh_scores[r] = dot * scale;
  }
  __syncthreads();

  __shared__ float partial[256];
  __shared__ float max_s;
  __shared__ float denom;
  float local_max = sinks[h];
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    local_max = fmaxf(local_max, sh_scores[r]);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    max_s = partial[0];
  __syncthreads();

  float den_local = 0.0f;
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    sh_scores[r] = expf(sh_scores[r] - max_s);
    den_local += sh_scores[r];
  }
  partial[threadIdx.x] = den_local;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    denom = partial[0] + expf(sinks[h] - max_s);
  __syncthreads();

  float* oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row =
          (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
      acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
    }
    oh[d] = acc / denom;
  }
}

// Target decode and verification preserve the same scalar reductions.
__device__ static float attention_warp_sum_oldhip_w32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v += __shfl_down(v, offset, 32);
#else
    v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
  }
  return v;
}

__device__ static float attention_warp_max_oldhip_w32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v = fmaxf(v, __shfl_down(v, offset, 32));
#else
    v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
  }
  return v;
}

__device__ static float attention_block_sum_oldhip_w32(float v) {
  __shared__ float sh[32];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wid = tid >> 5u;
  const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
  v = attention_warp_sum_oldhip_w32(v);
  if (lane == 0u)
    sh[wid] = v;
  __syncthreads();
  v = (tid < nwarp) ? sh[lane] : 0.0f;
  if (wid == 0u)
    v = attention_warp_sum_oldhip_w32(v);
  if (tid == 0u)
    sh[0] = v;
  __syncthreads();
  return sh[0];
}

__device__ static float attention_block_max_oldhip_w32(float v) {
  __shared__ float sh[32];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wid = tid >> 5u;
  const uint32_t nwarp = (blockDim.x + 31u) >> 5u;
  v = attention_warp_max_oldhip_w32(v);
  if (lane == 0u)
    sh[wid] = v;
  __syncthreads();
  v = (tid < nwarp) ? sh[lane] : -3.4e38f;
  if (wid == 0u)
    v = attention_warp_max_oldhip_w32(v);
  if (tid == 0u)
    sh[0] = v;
  __syncthreads();
  return sh[0];
}

__device__ __forceinline__ static float attention_dot_f32_vec4_oldhip(
    const float* a, const float* b, uint32_t n) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  const uint32_t n4 = n >> 2u;
  const float4* a4 = (const float4*)a;
  const float4* b4 = (const float4*)b;
  for (uint32_t i = 0; i < n4; i++) {
    const float4 av = a4[i];
    const float4 bv = b4[i];
    s0 += av.x * bv.x;
    s1 += av.y * bv.y;
    s2 += av.z * bv.z;
    s3 += av.w * bv.w;
  }
  float s = (s0 + s1) + (s2 + s3);
  for (uint32_t i = n4 << 2u; i < n; i++)
    s += a[i] * b[i];
  return s;
}

__global__ static void attention_decode_mixed_one_fast_oldhip_kernel(
    float* heads, const float* q, const float* raw_kv, const float* comp_kv,
    const float* comp_mask, const float* sinks, uint32_t n_raw,
    uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp, uint32_t use_mask,
    uint32_t n_head, uint32_t head_dim, uint32_t use_vec4) {
  const uint32_t h = (uint32_t)blockIdx.x;
  if (h >= n_head)
    return;
  extern __shared__ float scores[];
  const uint32_t tid = threadIdx.x;
  const uint32_t n_rows = n_raw + n_comp;
  const float* qh = q + (uint64_t)h * head_dim;
  const float scale = rsqrtf((float)head_dim);

  float local_max = sinks[h];
  for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
    const uint32_t row =
        (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        s += qh[i] * kv[i];
    }
    s *= scale;
    scores[r] = s;
    local_max = fmaxf(local_max, s);
  }
  for (uint32_t c = tid; c < n_comp; c += blockDim.x) {
    float s = -3.4e38f;
    if (!(use_mask && comp_mask && comp_mask[c] <= -5.0e29f)) {
      const float* kv = comp_kv + (uint64_t)c * head_dim;
      float dot =
          use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
      if (!use_vec4) {
        for (uint32_t i = 0; i < head_dim; i++)
          dot += qh[i] * kv[i];
      }
      s = dot * scale;
      if (use_mask && comp_mask)
        s += comp_mask[c];
    }
    scores[n_raw + c] = s;
    local_max = fmaxf(local_max, s);
  }
  const float max_score = attention_block_max_oldhip_w32(local_max);

  float local_sum = 0.0f;
  for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
    const float w = expf(scores[r] - max_score);
    scores[r] = w;
    local_sum += w;
  }
  if (tid == 0u)
    local_sum += expf(sinks[h] - max_score);
  const float denom = attention_block_sum_oldhip_w32(local_sum);
  const float inv_denom = 1.0f / denom;

  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row =
          (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
      acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
    }
    for (uint32_t c = 0; c < n_comp; c++) {
      acc += scores[n_raw + c] * comp_kv[(uint64_t)c * head_dim + d];
    }
    heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
  }
}

__global__ static void attention_decode_indexed_mixed_one_fast_oldhip_kernel(
    float* heads, const float* q, const float* raw_kv, const float* comp_kv,
    const int32_t* topk, const float* sinks, uint32_t n_raw, uint32_t raw_cap,
    uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t pos0,
    uint32_t ratio, uint32_t n_head, uint32_t head_dim, uint32_t use_vec4) {
  const uint32_t h = (uint32_t)blockIdx.x;
  if (h >= n_head)
    return;
  extern __shared__ float scores[];
  __shared__ uint32_t comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
  __shared__ uint32_t comp_count_s;
  const uint32_t tid = threadIdx.x;
  const float* qh = q + (uint64_t)h * head_dim;
  const float scale = rsqrtf((float)head_dim);

  uint32_t visible_comp = n_comp;
  if (ratio != 0u) {
    visible_comp = (pos0 + 1u) / ratio;
    if (visible_comp > n_comp)
      visible_comp = n_comp;
  }
  if (tid == 0u) {
    comp_count_s = 0;
    for (uint32_t i = 0;
         i < top_k && comp_count_s < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP; i++) {
      const int32_t ci = topk[i];
      if (ci < 0)
        continue;
      const uint32_t c = (uint32_t)ci;
      if (c < n_comp && c < visible_comp)
        comp_rows[comp_count_s++] = c;
    }
  }
  __syncthreads();
  const uint32_t comp_count = comp_count_s;
  const uint32_t n_rows = n_raw + comp_count;

  float local_max = sinks[h];
  for (uint32_t r = tid; r < n_raw; r += blockDim.x) {
    const uint32_t row =
        (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float s = use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        s += qh[i] * kv[i];
    }
    s *= scale;
    scores[r] = s;
    local_max = fmaxf(local_max, s);
  }
  for (uint32_t c = tid; c < comp_count; c += blockDim.x) {
    const uint32_t row = comp_rows[c];
    const float* kv = comp_kv + (uint64_t)row * head_dim;
    float dot =
        use_vec4 ? attention_dot_f32_vec4_oldhip(qh, kv, head_dim) : 0.0f;
    if (!use_vec4) {
      for (uint32_t i = 0; i < head_dim; i++)
        dot += qh[i] * kv[i];
    }
    const float s = dot * scale;
    scores[n_raw + c] = s;
    local_max = fmaxf(local_max, s);
  }
  const float max_score = attention_block_max_oldhip_w32(local_max);

  float local_sum = 0.0f;
  for (uint32_t r = tid; r < n_rows; r += blockDim.x) {
    const float w = expf(scores[r] - max_score);
    scores[r] = w;
    local_sum += w;
  }
  if (tid == 0u)
    local_sum += expf(sinks[h] - max_score);
  const float denom = attention_block_sum_oldhip_w32(local_sum);
  const float inv_denom = 1.0f / denom;

  for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row =
          (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
      acc += scores[r] * raw_kv[(uint64_t)row * head_dim + d];
    }
    for (uint32_t c = 0; c < comp_count; c++) {
      const uint32_t row = comp_rows[c];
      acc += scores[n_raw + c] * comp_kv[(uint64_t)row * head_dim + d];
    }
    heads[(uint64_t)h * head_dim + d] = acc * inv_denom;
  }
}
