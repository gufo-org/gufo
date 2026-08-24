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

[[nodiscard]] constexpr int ResolveHipDeviceGate(
    hipError_t status, int device_count,
    HipDeviceRequirement requirement) noexcept {
  if (status == hipSuccess && device_count > 0) {
    return kHipTestSuccess;
  }
  if (status == hipSuccess || status == hipErrorNoDevice) {
    return requirement == HipDeviceRequirement::kOptional
               ? kCtestSkipReturnCode
               : kHipTestFailure;
  }
  return kHipTestFailure;
}

[[nodiscard]] inline int GateHipDevice(HipDeviceRequirement requirement,
                                       std::string_view test_name) {
  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  const int result = ResolveHipDeviceGate(status, device_count, requirement);
  if (result == kHipTestSuccess) {
    return result;
  }
  if (status != hipSuccess && status != hipErrorNoDevice) {
    std::cerr << test_name << ": HIP device discovery failed: "
              << hipGetErrorString(status) << "; failing test\n";
    return result;
  }
  if (result == kCtestSkipReturnCode) {
    std::cout << test_name
              << ": no HIP device found; skipping optional test\n";
    return result;
  }
  std::cerr << test_name << ": no HIP device found; failing required test\n";
  return result;
}

}  // namespace strix::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_HPP_
