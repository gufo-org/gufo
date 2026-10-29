#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_IQ2XXS_HIP_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_IQ2XXS_HIP_HPP_

// IQ2_XXS decoding shared by the vector products (gemv.hip) and the prefill
// matrix products (gemm.hip): BF16 grid words with their signs, the group
// scale, and BF16 rounding.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::nemotron35_bolt::rocm {

/// Bytes of one IQ2_XXS super-block of 256 values: F16 d, then eight 32-value
/// groups of four grid indices and a word of signs and scale.
inline constexpr int kIq2xxsBlockBytes = 66;

using Bf16x2 = __bf16 __attribute__((ext_vector_type(2)));

/// Round to nearest even; NaN stays NaN.
__device__ __forceinline__ std::uint32_t Bf16Bits(float f) {
  const std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7FFFFFFFU) > 0x7F800000U)
    return (u >> 16) | 0x40U;
  return (u + 0x7FFFU + ((u >> 16) & 1U)) >> 16;
}

__device__ __forceinline__ std::uint32_t PackBf16(float lo, float hi) {
  return Bf16Bits(lo) | (Bf16Bits(hi) << 16);
}

/// c + a.lo * b.lo + a.hi * b.hi with F32 accumulation.
__device__ __forceinline__ float Dot2(std::uint32_t a, std::uint32_t b,
                                      float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(
      __builtin_bit_cast(Bf16x2, a), __builtin_bit_cast(Bf16x2, b), c, false);
}

/// Seven stored sign bits plus the even-parity eighth, moved pairwise onto
/// the BF16 sign positions 15 and 31 of the grid entry's four words.
struct SignedGrid {
  std::uint32_t w[4];

  __device__ SignedGrid(const uint4& g, std::uint32_t signs7) {
    const std::uint32_t s8 = signs7 | ((__popc(signs7) & 1U) << 7);
    const std::uint32_t spread = s8 | ((s8 >> 1) << 16);
    w[0] = g.x ^ ((spread << 15) & 0x80008000U);
    w[1] = g.y ^ ((spread << 13) & 0x80008000U);
    w[2] = g.z ^ ((spread << 11) & 0x80008000U);
    w[3] = g.w ^ ((spread << 9) & 0x80008000U);
  }
  /// c + the eight products with x.
  __device__ float Dot(const uint4& x, float c) const {
    c = Dot2(w[0], x.x, c);
    c = Dot2(w[1], x.y, c);
    c = Dot2(w[2], x.z, c);
    return Dot2(w[3], x.w, c);
  }
};

/// Group scale of a 32-value group: d * (0.5 + s) / 4.
__device__ __forceinline__ float GroupScale(const std::uint8_t* block,
                                            std::uint32_t aux) {
  const float d = __half2float(*reinterpret_cast<const __half*>(block));
  return d * (0.5F + static_cast<float>(aux >> 28)) * 0.25F;
}

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_IQ2XXS_HIP_HPP_
