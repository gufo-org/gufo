// The activation pack of the WMMA verification projections (wmma_gemv.hpp),
// shared by their PackKernel and by the kernels that produce an input and
// pack it in place of a separate launch.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_PACK_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_PACK_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/wmma_gemv.hpp"

namespace gufo::models::gemma4::rocm {

/// Packs the 32 values at `src` as block b of row t of a width-k pack: the
/// scale 2^e putting their maximum in [2^13, 2^14), x 2^e as binary16 hi +
/// lo, 2^-e, and the sequential FP32 sum of x.
__device__ inline void PackWmmaBlock(const float* src, std::uint32_t t,
                                     std::uint32_t b, std::uint32_t k,
                                     __half* hi, __half* lo, float* inv,
                                     float* sums) {
  const auto* src4 = reinterpret_cast<const float4*>(src);
  float v[32];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 f = src4[j];
    v[4 * j] = f.x;
    v[4 * j + 1] = f.y;
    v[4 * j + 2] = f.z;
    v[4 * j + 3] = f.w;
  }
  float peak = 0.0F;
  float sum = 0.0F;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    peak = fmaxf(peak, fabsf(v[j]));
    sum += v[j];
  }
  int e = 0;
  if (peak > 0.0F) {
    int exponent = 0;
    (void)frexpf(peak, &exponent);
    e = 14 - exponent;
  }
  const float scale = ldexpf(1.0F, e);
  __half h[32];
  __half l[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const float s = v[j] * scale;
    h[j] = __float2half_rn(s);
    l[j] = __float2half_rn(s - __half2float(h[j]));
  }
  auto* dh = reinterpret_cast<uint4*>(hi + std::size_t{t} * k + b * 32);
  auto* dl = reinterpret_cast<uint4*>(lo + std::size_t{t} * k + b * 32);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint4 a;
    uint4 c;
    __builtin_memcpy(&a, &h[8 * j], 16);
    __builtin_memcpy(&c, &l[8 * j], 16);
    dh[j] = a;
    dl[j] = c;
  }
  inv[b * kWmmaGemvRows + t] = ldexpf(1.0F, -e);
  sums[b * kWmmaGemvRows + t] = sum;
}

/// Packs `blocks` 32-value blocks a kernel staged in shared memory, as
/// blocks b0, b0 + 1, ... of pack row t, after a barrier: one thread per
/// block, as PackKernel. Every thread of the kernel block must call it.
__device__ inline void PackStaged(const float* stage, std::uint32_t blocks,
                                  std::uint32_t t, std::uint32_t b0,
                                  const WmmaPackOut& pack) {
  __syncthreads();
  for (std::uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
    PackWmmaBlock(stage + std::size_t{b} * 32, t, b0 + b, pack.k,
                  static_cast<__half*>(pack.hi), static_cast<__half*>(pack.lo),
                  pack.inv, pack.sums);
  }
}

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_PACK_HPP_
