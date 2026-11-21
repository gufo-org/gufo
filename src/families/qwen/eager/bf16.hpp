#ifndef GUFO_FAMILIES_QWEN_EAGER_BF16_HPP_
#define GUFO_FAMILIES_QWEN_EAGER_BF16_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace gufo::qwen::eager {

// Match hip_bfloat16's round-to-nearest-even conversion, including infinities
// and NaN payloads, without divergent branches in a matrix epilogue.
__device__ __forceinline__ hip_bfloat16 Bf16(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned special = 0U - unsigned((bits & 0x7f800000U) == 0x7f800000U);
  const unsigned rounded = bits + 32767U + ((bits >> 16) & 1U);
  const unsigned preserved = bits | (unsigned((bits & 65535U) != 0) << 16);
  hip_bfloat16 result;
  result.data = ((rounded & ~special) | (preserved & special)) >> 16;
  return result;
}

}  // namespace gufo::qwen::eager

#endif  // GUFO_FAMILIES_QWEN_EAGER_BF16_HPP_
