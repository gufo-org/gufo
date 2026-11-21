#ifndef GUFO_MODELS_QWEN_IMAGE_21_HIP_BF16_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_HIP_BF16_HPP_

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/families/qwen/eager/bf16.hpp"

namespace gufo::models::qwen_image_21::hip {

using qwen::eager::Bf16;

// The rounded result matches expf for every BF16 input bit pattern on gfx1151.
// Keep the BF16 input type: this equivalence does not hold for arbitrary FP32.
__device__ __forceinline__ hip_bfloat16 SiluBf16(hip_bfloat16 input) {
  const float value = float(input);
  return Bf16(value / (1.0F + __expf(-value)));
}

}  // namespace gufo::models::qwen_image_21::hip

#endif
