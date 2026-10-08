// Binary16 copies of FP32 results for the binary16 prefill GEMMs.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_HALF_STORE_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_HALF_STORE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace gufo::models::gemma4::rocm {

/// The FP32 value v rounded to binary16 (nearest), as NarrowActivations
/// rounds a stored FP32 row. The empty asm pins v as an FP32 register:
/// otherwise the compiler may fold the product that made v into a mixed
/// FMA rounding once to binary16, which differs from rounding the FP32
/// result by an ulp now and then.
__device__ __forceinline__ __half HalfOf(float v) {
  asm volatile("" : "+v"(v));
  return __float2half(v);
}

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_HALF_STORE_HPP_
