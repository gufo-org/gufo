#ifndef STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
#define STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <hip/hip_runtime.h>

namespace strix::test {

enum class HipDeviceRequirement { kOptional, kRequired };

inline constexpr int kHipTestSuccess = EXIT_SUCCESS;
inline constexpr int kHipTestFailure = EXIT_FAILURE;
inline constexpr int kCtestSkipReturnCode = 77;

[[nodiscard]] inline int GateHipDevice(HipDeviceRequirement requirement,
                                       std::string_view test_name) {
  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  if (status != hipSuccess) {
    std::cerr << test_name << ": HIP device discovery failed: "
              << hipGetErrorString(status) << "; failing test\n";
    return kHipTestFailure;
  }
  if (device_count > 0) {
    return kHipTestSuccess;
  }
  if (requirement == HipDeviceRequirement::kOptional) {
    std::cout << test_name
              << ": no HIP device found; skipping optional test\n";
    return kCtestSkipReturnCode;
  }
  std::cerr << test_name << ": no HIP device found; failing required test\n";
  return kHipTestFailure;
}

}  // namespace strix::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
