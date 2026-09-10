#pragma once
#include "ds4_rocm_device.hip.hpp"

__global__ static void matmul_f16_ordered_chunks_kernel(
    float* out, const __half* w, const float* x, uint64_t in_dim,
    uint64_t out_dim, uint64_t n_tok) {
  uint64_t row = (uint64_t)blockIdx.x;
  uint64_t tok = (uint64_t)blockIdx.y;
  if (row >= out_dim || tok >= n_tok)
    return;

  __shared__ float partial[32];
  const uint32_t tid = threadIdx.x;
  float sum = 0.0f;
  const uint64_t chunk = (in_dim + 31u) / 32u;
  const uint64_t k0 = (uint64_t)tid * chunk;
  uint64_t k1 = k0 + chunk;
  if (k1 > in_dim)
    k1 = in_dim;
  const __half* wr = w + row * in_dim;
  const float* xr = x + tok * in_dim;
  for (uint64_t i = k0; i < k1; i++) {
    sum += __half2float(wr[i]) * xr[i];
  }
  partial[tid] = sum;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (uint32_t i = 0; i < 32u; i++)
      total += partial[i];
    out[tok * out_dim + row] = total;
  }
}

template<uint32_t BATCH>
__global__ static void matmul_f16_ordered_batch_reuse_kernel(float* out,
                                                             const __half* w,
                                                             const float* x,
                                                             uint64_t in_dim,
                                                             uint64_t out_dim) {
  const uint64_t row = (uint64_t)blockIdx.x;
  if (row >= out_dim)
    return;

  __shared__ float partial[BATCH][32];
  const uint32_t tid = threadIdx.x;
  float sum[BATCH];
#pragma unroll
  for (uint32_t token = 0; token < BATCH; token++) {
    sum[token] = 0.0f;
  }
  const uint64_t chunk = (in_dim + 31u) / 32u;
  const uint64_t k0 = (uint64_t)tid * chunk;
  uint64_t k1 = k0 + chunk;
  if (k1 > in_dim)
    k1 = in_dim;
  const __half* wr = w + row * in_dim;
  uint64_t i = k0;
  if ((in_dim & 3u) == 0u && (i & 3u) == 0u) {
    for (; i + 3u < k1; i += 4u) {
      const float2 weight01 =
          __half22float2(*reinterpret_cast<const __half2*>(wr + i));
      const float2 weight23 =
          __half22float2(*reinterpret_cast<const __half2*>(wr + i + 2u));
#pragma unroll
      for (uint32_t token = 0; token < BATCH; token++) {
        const float4 activation =
            *reinterpret_cast<const float4*>(x + (uint64_t)token * in_dim + i);
        sum[token] = fmaf(weight01.x, activation.x, sum[token]);
        sum[token] = fmaf(weight01.y, activation.y, sum[token]);
        sum[token] = fmaf(weight23.x, activation.z, sum[token]);
        sum[token] = fmaf(weight23.y, activation.w, sum[token]);
      }
    }
  }
  if ((in_dim & 1u) == 0u && (i & 1u) == 0u) {
    for (; i + 1u < k1; i += 2u) {
      const float2 weight =
          __half22float2(*reinterpret_cast<const __half2*>(wr + i));
#pragma unroll
      for (uint32_t token = 0; token < BATCH; token++) {
        const float2 activation =
            *reinterpret_cast<const float2*>(x + (uint64_t)token * in_dim + i);
        sum[token] = fmaf(weight.x, activation.x, sum[token]);
        sum[token] = fmaf(weight.y, activation.y, sum[token]);
      }
    }
  }
  for (; i < k1; i++) {
    const float weight = __half2float(wr[i]);
#pragma unroll
    for (uint32_t token = 0; token < BATCH; token++) {
      sum[token] = fmaf(weight, x[(uint64_t)token * in_dim + i], sum[token]);
    }
  }
#pragma unroll
  for (uint32_t token = 0; token < BATCH; token++) {
    partial[token][tid] = sum[token];
  }
  __syncthreads();
  if (tid == 0u) {
#pragma unroll
    for (uint32_t token = 0; token < BATCH; token++) {
      float total = 0.0f;
#pragma unroll
      for (uint32_t lane = 0; lane < 32u; lane++) {
        total += partial[token][lane];
      }
      out[(uint64_t)token * out_dim + row] = total;
    }
  }
}
