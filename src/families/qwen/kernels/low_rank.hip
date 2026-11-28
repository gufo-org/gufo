#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/families/qwen/kernels/ops/low_rank.hpp"

namespace gufo::hip {
namespace {

// A wide block shortens the serial loop and reduction of a single row, which
// is latency bound in decode: 1024 threads measured 4.8 and 5.8 us per graph
// node for 6144 and 17408 inputs, against 10.1 and 13.8 us with 256.
constexpr unsigned kThreads = 1024;
// Ranks reduced per pass over x; rank 3 takes one pass.
constexpr unsigned kRankChunk = 4;

// Four consecutive inputs, widened to float32.
__device__ inline float4 Load4(const float* x, unsigned i) {
  return reinterpret_cast<const float4*>(x)[i];
}
__device__ inline float4 Load4(const __half* x, unsigned i) {
  const auto* pair = reinterpret_cast<const __half2*>(x) + 2 * i;
  const float2 low = __half22float2(pair[0]);
  const float2 high = __half22float2(pair[1]);
  return {low.x, low.y, high.x, high.y};
}

// One block per row: t = A * x with a fixed shared-memory tree reduction,
// then y += multiplier * (B * t).
template<typename Input>
__global__ __launch_bounds__(kThreads) void LowRankAddKernel(
    const Input* __restrict__ x, const float* __restrict__ a,
    const float* __restrict__ b, float multiplier, float* __restrict__ y,
    unsigned in, unsigned out, unsigned rank) {
  __shared__ float partial[kRankChunk][kThreads];
  __shared__ float t[kLowRankMaxRank];
  const unsigned tid = threadIdx.x;
  const unsigned in4 = in / 4;
  const Input* xr = x + static_cast<std::size_t>(blockIdx.x) * in;
  for (unsigned k0 = 0; k0 < rank; k0 += kRankChunk) {
    float s[kRankChunk] = {};
    for (unsigned i = tid; i < in4; i += kThreads) {
      const float4 v = Load4(xr, i);
#pragma unroll
      for (unsigned j = 0; j < kRankChunk; ++j) {
        if (k0 + j < rank) {
          const float4 w = reinterpret_cast<const float4*>(
              a + static_cast<std::size_t>(k0 + j) * in)[i];
          s[j] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
      }
    }
#pragma unroll
    for (unsigned j = 0; j < kRankChunk; ++j) {
      partial[j][tid] = s[j];
    }
    __syncthreads();
    for (unsigned stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
#pragma unroll
        for (unsigned j = 0; j < kRankChunk; ++j) {
          partial[j][tid] += partial[j][tid + stride];
        }
      }
      __syncthreads();
    }
    if (tid < kRankChunk && k0 + tid < rank) {
      t[k0 + tid] = partial[tid][0];
    }
    __syncthreads();
  }
  float* yr = y + static_cast<std::size_t>(blockIdx.x) * out;
  for (unsigned o = tid; o < out; o += kThreads) {
    const float* bo = b + static_cast<std::size_t>(o) * rank;
    float acc = 0.0F;
    for (unsigned k = 0; k < rank; ++k) {
      acc += bo[k] * t[k];
    }
    yr[o] += multiplier * acc;
  }
}

template<typename Input>
void Launch(const Input* x, const float* a, const float* b, float multiplier,
            float* y, std::uint32_t rows, std::uint32_t in, std::uint32_t out,
            std::uint32_t rank, hipStream_t stream) {
  if (rows == 0) {
    return;
  }
  LowRankAddKernel<Input>
      <<<rows, kThreads, 0, stream>>>(x, a, b, multiplier, y, in, out, rank);
}

}  // namespace

void LaunchLowRankAdd(const float* x, const float* a, const float* b,
                      float multiplier, float* y, std::uint32_t rows,
                      std::uint32_t in, std::uint32_t out, std::uint32_t rank,
                      hipStream_t stream) {
  Launch(x, a, b, multiplier, y, rows, in, out, rank, stream);
}

void LaunchLowRankAddFp16(const void* x, const float* a, const float* b,
                          float multiplier, float* y, std::uint32_t rows,
                          std::uint32_t in, std::uint32_t out,
                          std::uint32_t rank, hipStream_t stream) {
  Launch(static_cast<const __half*>(x), a, b, multiplier, y, rows, in, out,
         rank, stream);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
