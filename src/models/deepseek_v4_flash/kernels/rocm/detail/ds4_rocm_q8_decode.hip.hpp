// Narrow Q8 projections shared by the backend and its numerical oracle.
#pragma once
#include "ds4_rocm_device.hip.hpp"

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(
    const int8_t* p) {
  return *(const int32_t*)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(
    const int8_t* p) {
  const uint8_t* u = (const uint8_t*)p;
  return (int32_t)((uint32_t)u[0] | ((uint32_t)u[1] << 8) |
                   ((uint32_t)u[2] << 16) | ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t* a,
                                                         const int8_t* b) {
  int32_t dot = 0;
#pragma unroll
  for (uint32_t i = 0; i < 32u; i += 4u) {
    dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i),
                 dot);
  }
  return dot;
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a_loaded(
    int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4, int32_t a5,
    int32_t a6, int32_t a7, const int8_t* b) {
  int32_t dot = 0;
  dot = __dp4a(a0, load_i8x4_i32_aligned(b + 0u), dot);
  dot = __dp4a(a1, load_i8x4_i32_aligned(b + 4u), dot);
  dot = __dp4a(a2, load_i8x4_i32_aligned(b + 8u), dot);
  dot = __dp4a(a3, load_i8x4_i32_aligned(b + 12u), dot);
  dot = __dp4a(a4, load_i8x4_i32_aligned(b + 16u), dot);
  dot = __dp4a(a5, load_i8x4_i32_aligned(b + 20u), dot);
  dot = __dp4a(a6, load_i8x4_i32_aligned(b + 24u), dot);
  dot = __dp4a(a7, load_i8x4_i32_aligned(b + 28u), dot);
  return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t* a,
                                                       const int8_t* b,
                                                       uint64_t n,
                                                       int use_dp4a) {
  if (use_dp4a && n == 32u)
    return dot_i8x32_dp4a(a, b);
  int32_t dot = 0;
  for (uint64_t i = 0; i < n; i++)
    dot += (int32_t)a[i] * (int32_t)b[i];
  return dot;
}

__global__ static void quantize_q8_0_f32_kernel(int8_t* xq, float* xscale,
                                                const float* x, uint64_t in_dim,
                                                uint64_t blocks) {
  uint64_t b = blockIdx.x;
  uint64_t tok = blockIdx.y;
  if (b >= blocks)
    return;
  uint64_t i0 = b * 32;
  uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
  const float* xr = x + tok * in_dim + i0;

  float a = 0.0f;
  if (threadIdx.x < bn)
    a = fabsf(xr[threadIdx.x]);
  a = warp_max_f32(a);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  const float d = __shfl(a, 0, 32) / 127.0f;
#else
  const float d = __shfl_sync(FULL_WARP_MASK, a, 0, 32) / 127.0f;
#endif
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  if (threadIdx.x == 0)
    xscale[tok * blocks + b] = d;
  int8_t* dst = xq + (tok * blocks + b) * 32;
  if (threadIdx.x < bn) {
    int v = (int)lrintf(xr[threadIdx.x] * id);
    v = v > 127 ? 127 : (v < -128 ? -128 : v);
    dst[threadIdx.x] = (int8_t)v;
  } else {
    dst[threadIdx.x] = 0;
  }
}

__global__ static void matmul_q8_0_preq_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, uint64_t blocks,
    int use_dp4a) {
  uint64_t row = (uint64_t)blockIdx.x;
  uint64_t tok = (uint64_t)blockIdx.y;
  if (row >= out_dim || tok >= n_tok)
    return;
  const unsigned char* wr = w + row * blocks * 34;
  const int8_t* xqr = xq + tok * blocks * 32;
  const float* xsr = xscale + tok * blocks;
  float acc = 0.0f;
  for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const __half* scale_h = (const __half*)(wr + b * 34);
    const int8_t* qs = (const int8_t*)(wr + b * 34 + 2);
    const int8_t* xqb = xqr + b * 32;
    int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
    acc += __half2float(*scale_h) * xsr[b] * (float)dot;
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_rows_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks, uint32_t rows_per_block,
    int use_dp4a) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;
  const unsigned char* wr = w + row * blocks * 34u;
  float acc = 0.0f;
  for (uint64_t b = lane; b < blocks; b += 32u) {
    const uint64_t i0 = b * 32u;
    const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
    const __half* scale_h = (const __half*)(wr + b * 34u);
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int8_t* xqb = xq + b * 32u;
    const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
    acc += __half2float(*scale_h) * xscale[b] * (float)dot;
  }
  acc = warp_sum_f32(acc);
  if (lane == 0u)
    out[row] = acc;
}

// Match the scalar kernel's compiled multiply/FMA order under fast-math.
// Grouping the two scales first changes over half the projection outputs.
__device__ __forceinline__ static float q8_0_decode_accumulate(
    float acc, float weight_scale, float input_scale, int dot) {
  return fmaf(input_scale, __fmul_rn(weight_scale, (float)dot), acc);
}

/*
 * Narrow-batch Q8_0 projection: read each weight block once, apply it to every
 * row in the batch.
 *
 * The prompt-chunk batch kernel stages float activations and reaches only about
 * a quarter of DRAM bandwidth at these widths, while the prequantized batch
 * kernel re-reads the whole weight matrix per row. This one keeps the
 * single-row decode kernel's shape - one warp per output row, lane-strided
 * 34-byte blocks, dp4a dots - and adds an inner loop over the batch, so the
 * weights move once and the activations are already quantized.
 *
 * The accumulation order per (output row, row index) is deliberately identical
 * to matmul_q8_0_preq_rows_w32_kernel: same block sequence per lane, same
 * multiply order, same warp reduction. That is what keeps a verified
 * speculative row bitwise equal to what ordinary decode would have produced.
 */
template<uint32_t MAXT, bool EXACT, bool PAIR = false>
__global__ static void matmul_q8_0_preq_batch_reuse_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks, uint32_t n_tok,
    uint32_t rows_per_block, const unsigned char* w1 = nullptr,
    float* out1 = nullptr) {
  if constexpr (PAIR) {
    if (blockIdx.y) {
      w = w1;
      out = out1;
    }
  }
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  /* row depends only on threadIdx.x >> 5, so a warp exits as a whole and the
   * reductions below always run with a full warp. */
  if (row >= out_dim)
    return;
  const unsigned char* wr = w + row * blocks * 34u;

  float acc[MAXT];
#pragma unroll
  for (uint32_t t = 0; t < MAXT; t++)
    acc[t] = 0.0f;

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float wscale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t t = 0; t < MAXT; t++) {
      if constexpr (!EXACT) {
        if (t >= n_tok)
          break;
      }
      const int8_t* xqb = xq + (uint64_t)t * blocks * 32u + b * 32u;
      const int dot =
          dot_i8x32_dp4a_loaded(q0, q1, q2, q3, q4, q5, q6, q7, xqb);
      if constexpr (PAIR) {
        // The paired scalar projection rounds the activation product first.
        acc[t] = fmaf(wscale,
                      __fmul_rn(xscale[(uint64_t)t * blocks + b], (float)dot),
                      acc[t]);
      } else {
        acc[t] = q8_0_decode_accumulate(acc[t], wscale,
                                        xscale[(uint64_t)t * blocks + b], dot);
      }
    }
  }

#pragma unroll
  for (uint32_t t = 0; t < MAXT; t++) {
    if constexpr (!EXACT) {
      if (t >= n_tok)
        break;
    }
    const float sum = warp_sum_f32(acc[t]);
    if (lane == 0u)
      out[(uint64_t)t * out_dim + row] = sum;
  }
}

/*
 * Two independent verification blocks share one Q8 weight stream while each
 * block keeps the exact accumulation order of the original narrow-batch
 * kernel. DSpark pairs C2/C4/C6/C8 requests, so this halves the dominant
 * projection traffic without flattening request boundaries.
 */
template<uint32_t NT>
__global__ static void matmul_q8_0_preq_group_pair_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks, uint32_t row0,
    uint32_t row1, uint32_t rows_per_block) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;
  const unsigned char* wr = w + row * blocks * 34u;

  float acc0[NT];
  float acc1[NT];
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float wscale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
      const uint64_t token0 = (uint64_t)row0 + t;
      const uint64_t token1 = (uint64_t)row1 + t;
      const int dot0 = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token0 * blocks * 32u + b * 32u);
      const int dot1 = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token1 * blocks * 32u + b * 32u);
      acc0[t] = q8_0_decode_accumulate(acc0[t], wscale,
                                       xscale[token0 * blocks + b], dot0);
      acc1[t] = q8_0_decode_accumulate(acc1[t], wscale,
                                       xscale[token1 * blocks + b], dot1);
    }
  }

#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    const float sum0 = warp_sum_f32(acc0[t]);
    const float sum1 = warp_sum_f32(acc1[t]);
    if (lane == 0u) {
      out[((uint64_t)row0 + t) * out_dim + row] = sum0;
      out[((uint64_t)row1 + t) * out_dim + row] = sum1;
    }
  }
}

/*
 * Ragged request pairs retain the qualified per-row accumulation order while
 * sharing each Q8 weight block across two independently sized draft prefixes.
 */
template<uint32_t NT0, uint32_t NT1>
__global__ static void matmul_q8_0_preq_ragged_group_pair_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks, uint32_t row0,
    uint32_t row1, uint32_t rows_per_block) {
  static_assert(NT0 >= 1u && NT0 <= 6u);
  static_assert(NT1 >= 1u && NT1 <= 6u);
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;
  const unsigned char* wr = w + row * blocks * 34u;

  float acc0[NT0] = {};
  float acc1[NT1] = {};
  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float wscale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t t = 0u; t < NT0; ++t) {
      const uint64_t token = (uint64_t)row0 + t;
      const int dot = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token * blocks * 32u + b * 32u);
      acc0[t] = q8_0_decode_accumulate(acc0[t], wscale,
                                       xscale[token * blocks + b], dot);
    }
#pragma unroll
    for (uint32_t t = 0u; t < NT1; ++t) {
      const uint64_t token = (uint64_t)row1 + t;
      const int dot = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token * blocks * 32u + b * 32u);
      acc1[t] = q8_0_decode_accumulate(acc1[t], wscale,
                                       xscale[token * blocks + b], dot);
    }
  }

#pragma unroll
  for (uint32_t t = 0u; t < NT0; ++t) {
    const float sum = warp_sum_f32(acc0[t]);
    if (lane == 0u) {
      out[((uint64_t)row0 + t) * out_dim + row] = sum;
    }
  }
#pragma unroll
  for (uint32_t t = 0u; t < NT1; ++t) {
    const float sum = warp_sum_f32(acc1[t]);
    if (lane == 0u) {
      out[((uint64_t)row1 + t) * out_dim + row] = sum;
    }
  }
}

/*
 * Equal-width request pairs use the same arithmetic as the single-pair kernel
 * above, but place the pair index in grid.y. This keeps every request's
 * accumulator independent while replacing C/2 launches per projection with
 * one launch for C2/C4/C6/C8.
 */
template<uint32_t NT>
__global__ static void matmul_q8_0_preq_equal_group_pairs_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks,
    uint32_t rows_per_block) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;

  const uint32_t row0 = (uint32_t)blockIdx.y * (2u * NT);
  const uint32_t row1 = row0 + NT;
  const unsigned char* wr = w + row * blocks * 34u;

  float acc0[NT];
  float acc1[NT];
#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float wscale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t t = 0; t < NT; ++t) {
      const uint64_t token0 = (uint64_t)row0 + t;
      const uint64_t token1 = (uint64_t)row1 + t;
      const int dot0 = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token0 * blocks * 32u + b * 32u);
      const int dot1 = dot_i8x32_dp4a_loaded(
          q0, q1, q2, q3, q4, q5, q6, q7, xq + token1 * blocks * 32u + b * 32u);
      acc0[t] = q8_0_decode_accumulate(acc0[t], wscale,
                                       xscale[token0 * blocks + b], dot0);
      acc1[t] = q8_0_decode_accumulate(acc1[t], wscale,
                                       xscale[token1 * blocks + b], dot1);
    }
  }

#pragma unroll
  for (uint32_t t = 0; t < NT; ++t) {
    const float sum0 = warp_sum_f32(acc0[t]);
    const float sum1 = warp_sum_f32(acc1[t]);
    if (lane == 0u) {
      out[((uint64_t)row0 + t) * out_dim + row] = sum0;
      out[((uint64_t)row1 + t) * out_dim + row] = sum1;
    }
  }
}

/*
 * A DSpark multi-request decode is one logical row batch. Request boundaries
 * only select KV-cache slices; dense projections do not depend on them. Keep
 * one accumulator per flattened row so every Q8 weight block is loaded once
 * for the whole C2/C4/C6/C8 batch while preserving the single-row fmaf order.
 */
template<uint32_t NT>
__global__ static void matmul_q8_0_preq_all_rows_exact_w32_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t blocks,
    uint32_t rows_per_block) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim)
    return;

  const unsigned char* wr = w + row * blocks * 34u;
  float acc[NT];
#pragma unroll
  for (uint32_t token = 0; token < NT; ++token) {
    acc[token] = 0.0f;
  }

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float wscale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t token = 0; token < NT; ++token) {
      const int dot =
          dot_i8x32_dp4a_loaded(q0, q1, q2, q3, q4, q5, q6, q7,
                                xq + ((uint64_t)token * blocks + b) * 32u);
      acc[token] = q8_0_decode_accumulate(
          acc[token], wscale, xscale[(uint64_t)token * blocks + b], dot);
    }
  }

#pragma unroll
  for (uint32_t token = 0; token < NT; ++token) {
    const float sum = warp_sum_f32(acc[token]);
    if (lane == 0u) {
      out[(uint64_t)token * out_dim + row] = sum;
    }
  }
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
    float* out0, float* out1, const unsigned char* w0, const unsigned char* w1,
    const int8_t* xq, const float* xscale, uint64_t in_dim, uint64_t out0_dim,
    uint64_t out1_dim, uint64_t blocks, int use_dp4a) {
  uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
  uint32_t lane = threadIdx.x & 31u;
  if (row >= out0_dim && row >= out1_dim)
    return;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  const unsigned char* wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
  const unsigned char* wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
  for (uint64_t b = lane; b < blocks; b += 32u) {
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const int8_t* xqb = xq + b * 32;
    const float xs = xscale[b];
    if (wr0) {
      const __half* scale_h = (const __half*)(wr0 + b * 34);
      const int8_t* qs = (const int8_t*)(wr0 + b * 34 + 2);
      int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
      acc0 += __half2float(*scale_h) * xs * (float)dot;
    }
    if (wr1) {
      const __half* scale_h = (const __half*)(wr1 + b * 34);
      const int8_t* qs = (const int8_t*)(wr1 + b * 34 + 2);
      int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
      acc1 += __half2float(*scale_h) * xs * (float)dot;
    }
  }
  acc0 = warp_sum_f32(acc0);
  acc1 = warp_sum_f32(acc1);
  if (lane == 0) {
    if (row < out0_dim)
      out0[row] = acc0;
    if (row < out1_dim)
      out1[row] = acc1;
  }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
    float* out, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t in_dim, uint64_t out_dim, uint64_t n_tok, uint64_t blocks,
    int use_dp4a) {
  const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
  const uint64_t tok = (uint64_t)blockIdx.y;
  const uint32_t lane = threadIdx.x & 31u;
  if (row >= out_dim || tok >= n_tok)
    return;

  const unsigned char* wr = w + row * blocks * 34;
  const int8_t* xqr = xq + tok * blocks * 32;
  const float* xsr = xscale + tok * blocks;
  float acc = 0.0f;
  for (uint64_t b = lane; b < blocks; b += 32u) {
    const uint64_t i0 = b * 32;
    const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const __half* scale_h = (const __half*)(wr + b * 34);
    const int8_t* qs = (const int8_t*)(wr + b * 34 + 2);
    const int8_t* xqb = xqr + b * 32;
    int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
    acc += __half2float(*scale_h) * xsr[b] * (float)dot;
  }
  acc = warp_sum_f32(acc);
  if (lane == 0)
    out[tok * out_dim + row] = acc;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
    float* low, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
    uint64_t blocks, int use_dp4a) {
  const uint32_t rows_per_block = blockDim.x >> 5u;
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint64_t tok = (uint64_t)blockIdx.y;
  const uint32_t lane = threadIdx.x & 31u;
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  if (row >= low_dim || tok >= n_tokens)
    return;

  const uint64_t group = row / rank;
  const uint64_t row_in_group = row - group * rank;
  const unsigned char* wr = w + (group * rank + row_in_group) * blocks * 34;
  const uint64_t xrow = tok * (uint64_t)n_groups + group;
  const int8_t* xqr = xq + xrow * blocks * 32;
  const float* xsr = xscale + xrow * blocks;
  float acc = 0.0f;

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const uint64_t i0 = b * 32;
    const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
    const __half* scale_h = (const __half*)(wr + b * 34);
    const int8_t* qs = (const int8_t*)(wr + b * 34 + 2);
    const int8_t* xqb = xqr + b * 32;
    int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
    acc += __half2float(*scale_h) * xsr[b] * (float)dot;
  }
  acc = warp_sum_f32(acc);
  if (lane == 0)
    low[tok * low_dim + row] = acc;
}

/*
 * Session-batch attention output-A projection.
 *
 * Each warp owns one output row and walks the Q8 blocks in the same lane order
 * as grouped_q8_0_a_preq_warp8_kernel. Keeping one accumulator per concurrent
 * session therefore preserves C1's quantization, multiply order, and warp
 * reduction while reusing every weight block across W2-W8.
 */
template<uint32_t MAXT, bool EXACT>
__global__ static void grouped_q8_0_a_preq_batch_reuse_w32_kernel(
    float* low, const unsigned char* w, const int8_t* xq, const float* xscale,
    uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
    uint64_t blocks, uint32_t rows_per_block) {
  const uint64_t row =
      (uint64_t)blockIdx.x * rows_per_block + (threadIdx.x >> 5u);
  const uint32_t lane = threadIdx.x & 31u;
  const uint64_t low_dim = (uint64_t)n_groups * rank;
  if (row >= low_dim)
    return;

  const uint64_t group = row / rank;
  const uint64_t row_in_group = row - group * rank;
  const unsigned char* wr = w + (group * rank + row_in_group) * blocks * 34u;
  float acc[MAXT];
#pragma unroll
  for (uint32_t t = 0; t < MAXT; ++t)
    acc[t] = 0.0f;

  for (uint64_t b = lane; b < blocks; b += 32u) {
    const float weight_scale = __half2float(*(const __half*)(wr + b * 34u));
    const int8_t* qs = (const int8_t*)(wr + b * 34u + 2u);
    const int32_t q0 = load_i8x4_i32_unaligned(qs + 0u);
    const int32_t q1 = load_i8x4_i32_unaligned(qs + 4u);
    const int32_t q2 = load_i8x4_i32_unaligned(qs + 8u);
    const int32_t q3 = load_i8x4_i32_unaligned(qs + 12u);
    const int32_t q4 = load_i8x4_i32_unaligned(qs + 16u);
    const int32_t q5 = load_i8x4_i32_unaligned(qs + 20u);
    const int32_t q6 = load_i8x4_i32_unaligned(qs + 24u);
    const int32_t q7 = load_i8x4_i32_unaligned(qs + 28u);
#pragma unroll
    for (uint32_t t = 0; t < MAXT; ++t) {
      if constexpr (!EXACT) {
        if (t >= n_tokens)
          break;
      }
      const uint64_t xrow = (uint64_t)t * n_groups + group;
      const int8_t* xqb = xq + (xrow * blocks + b) * 32u;
      const float activation_scale = xscale[xrow * blocks + b];
      const int dot =
          dot_i8x32_dp4a_loaded(q0, q1, q2, q3, q4, q5, q6, q7, xqb);
      acc[t] =
          q8_0_decode_accumulate(acc[t], weight_scale, activation_scale, dot);
    }
  }

#pragma unroll
  for (uint32_t t = 0; t < MAXT; ++t) {
    if constexpr (!EXACT) {
      if (t >= n_tokens)
        break;
    }
    const float sum = warp_sum_f32(acc[t]);
    if (lane == 0u)
      low[(uint64_t)t * low_dim + row] = sum;
  }
}
