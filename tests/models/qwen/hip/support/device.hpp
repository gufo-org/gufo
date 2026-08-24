#ifndef STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
#define STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

namespace strix::test {

[[nodiscard]] inline bool HasHipDevice() {
  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  if (status != hipSuccess) {
    std::cerr << "HIP device discovery failed: " << hipGetErrorString(status)
              << '\n';
    std::abort();
  }
  return device_count > 0;
}

}  // namespace strix::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
