#include <cstdlib>
#include <iostream>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "tests/models/qwen/hip/support/device.hpp"

namespace {

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "Qwen HIP device gate test failure: " << message << '\n';
    std::abort();
  }
}

void TestHipDeviceGateDecisions() {
  using strix::test::HipDeviceRequirement;
  using strix::test::ResolveHipDeviceGate;

  Check(ResolveHipDeviceGate(hipSuccess, 1,
                             HipDeviceRequirement::kOptional) ==
            strix::test::kHipTestSuccess,
        "visible device must pass");
  Check(ResolveHipDeviceGate(hipSuccess, 0,
                             HipDeviceRequirement::kOptional) ==
            strix::test::kCtestSkipReturnCode,
        "zero optional devices must skip");
  Check(ResolveHipDeviceGate(hipErrorNoDevice, 0,
                             HipDeviceRequirement::kOptional) ==
            strix::test::kCtestSkipReturnCode,
        "hipErrorNoDevice must skip an optional test");
  Check(ResolveHipDeviceGate(hipErrorNoDevice, 0,
                             HipDeviceRequirement::kRequired) ==
            strix::test::kHipTestFailure,
        "hipErrorNoDevice must fail a required test");
  Check(ResolveHipDeviceGate(hipErrorInvalidValue, 0,
                             HipDeviceRequirement::kOptional) ==
            strix::test::kHipTestFailure,
        "other HIP discovery errors must fail");
}

}  // namespace
#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  TestHipDeviceGateDecisions();
  std::cout << "Qwen HIP device gate tests passed.\n";
  return EXIT_SUCCESS;
#else
  std::cerr << "HIP disabled; Qwen HIP device gate contract unavailable.\n";
  return EXIT_FAILURE;
#endif
}
