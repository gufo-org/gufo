#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_LANES_HIP_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_LANES_HIP_HPP_

// Wave32 lane exchanges that stay in the VALU: DPP row_xmask within a row of
// 16 lanes and permlanex16 across the two rows, instead of the LDS round
// trip that __shfl_xor compiles to.

#include <hip/hip_runtime.h>

namespace gufo::models::nemotron35_bolt::rocm {

/// The value of lane (lane ^ kOffset).
template<int kOffset>
__device__ __forceinline__ float XorLane(float v) {
  static_assert(kOffset > 0 && kOffset <= 16 && (kOffset & (kOffset - 1)) == 0);
  const int bits = __builtin_bit_cast(int, v);
  int other;
  if constexpr (kOffset < 16) {
    other =
        __builtin_amdgcn_update_dpp(0, bits, 0x160 | kOffset, 0xF, 0xF, false);
  } else {
    other = __builtin_amdgcn_permlanex16(0, bits, 0x76543210, 0xFEDCBA98, false,
                                         false);
  }
  return __builtin_bit_cast(float, other);
}

/// v + the value of lane (lane ^ kOffset).
template<int kOffset>
__device__ __forceinline__ float AddXorLane(float v) {
  return v + XorLane<kOffset>(v);
}

/// Sum over the 32 lanes, adding neighbours first; every lane receives it.
__device__ __forceinline__ float WaveSum(float v) {
  v = AddXorLane<1>(v);
  v = AddXorLane<2>(v);
  v = AddXorLane<4>(v);
  v = AddXorLane<8>(v);
  return AddXorLane<16>(v);
}

}  // namespace gufo::models::nemotron35_bolt::rocm

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_LANES_HIP_HPP_
