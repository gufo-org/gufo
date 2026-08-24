#ifndef STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
#define STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"

namespace strix::test {

[[nodiscard]] inline bool HasHipDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  return device_count > 0;
}

}  // namespace strix::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
