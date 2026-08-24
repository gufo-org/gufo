#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

void TestGpuRMSNorm() {
  constexpr std::size_t dim = 256;
  const std::vector<float> host_input(dim, 1.0F);
  const std::vector<float> host_weight(dim, 2.0F);
  const std::vector<float> expected(dim, 2.0F);
  strix::test::DeviceBuffer<float> input(host_input);
  strix::test::DeviceBuffer<float> weight(host_weight);
  strix::test::DeviceBuffer<float> output(dim);

  strix::hip::LaunchRMSNorm(input.data(), weight.data(), output.data(), dim,
                            1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  const auto actual = output.CopyToHost();
  strix::test::ExpectSpanNear(expected, actual, 1e-4F, "GPU RMSNorm output");
}

void TestGpuResidualAdd() {
  constexpr std::size_t dim = 128;
  const std::vector<float> host_lhs(dim, 3.5F);
  const std::vector<float> host_rhs(dim, 1.5F);
  const std::vector<float> expected(dim, 5.0F);
  strix::test::DeviceBuffer<float> lhs(host_lhs);
  strix::test::DeviceBuffer<float> rhs(host_rhs);
  strix::test::DeviceBuffer<float> output(dim);

  strix::hip::LaunchResidualAdd(lhs.data(), rhs.data(), output.data(), dim);
  HIP_CHECK(hipDeviceSynchronize());

  const auto actual = output.CopyToHost();
  strix::test::ExpectSpanNear(expected, actual, 1e-5F,
                              "GPU residual-add output");
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      strix::test::GateHipDevice(strix::test::HipDeviceRequirement::kOptional,
                                 "Qwen elementwise GPU ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  TestGpuRMSNorm();
  TestGpuResidualAdd();
  std::cout << "Qwen elementwise GPU ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen elementwise GPU ops test.\n";
  return 77;
#endif
}
