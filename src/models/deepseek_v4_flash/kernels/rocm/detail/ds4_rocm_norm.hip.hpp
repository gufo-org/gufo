#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

/* Plain RMS norm with the row held in registers.
 *
 * The generic kernel below reads the row twice from global memory: once to
 * square and once to scale. At `n == blockDim.x * PER_THREAD` each thread can
 * keep its own strided slice, so the second read disappears and a 4,096-wide
 * row costs 16 VGPRs. The accumulation order and the reduction tree are
 * unchanged, so the output is bit-identical. */
template<uint32_t PER_THREAD>
__global__ static void rms_norm_plain_regs_kernel(float* out, const float* x,
                                                  uint32_t n, uint32_t rows,
                                                  float eps) {
  const uint32_t row = blockIdx.x;
  if (row >= rows)
    return;
  const float* xr = x + (uint64_t)row * n;
  float* orow = out + (uint64_t)row * n;
  float v[PER_THREAD];
  float sum = 0.0f;
#pragma unroll
  for (uint32_t k = 0; k < PER_THREAD; k++) {
    v[k] = xr[threadIdx.x + k * blockDim.x];
    sum += v[k] * v[k];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
  for (uint32_t k = 0; k < PER_THREAD; k++) {
    const float y = v[k] * scale;
    const uint32_t idx = threadIdx.x + k * blockDim.x;
    orow[idx] = y;
  }
}

__global__ static void rms_norm_plain_kernel(float* out, const float* x,
                                             uint32_t n, uint32_t rows,
                                             float eps) {
  uint32_t row = blockIdx.x;
  if (row >= rows)
    return;
  const float* xr = x + (uint64_t)row * n;
  float* orow = out + (uint64_t)row * n;
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    float v = xr[i];
    sum += v * v;
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  float scale = rsqrtf(partial[0] / (float)n + eps);
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    orow[i] = xr[i] * scale;
  }
}
