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
  // Cover the real decode width and both neighboring generic-path tails.
  for (const std::size_t dim : {256U, 5119U, 5120U, 5121U}) {
    for (const float scale : {0.001F, 1.0F, 100.0F}) {
      std::vector<float> host_input(dim), host_weight(dim), expected(dim);
      double sum_squared = 0.0;
      for (std::size_t i = 0; i < dim; ++i) {
        host_input[i] =
            scale * static_cast<float>(static_cast<int>(i % 127) - 63) / 64.0F;
        host_weight[i] = 0.5F + static_cast<float>(i % 31) / 32.0F;
        sum_squared += static_cast<double>(host_input[i]) * host_input[i];
      }
      const double inverse_rms =
          1.0 / std::sqrt(sum_squared / static_cast<double>(dim) + 1e-6F);
      gufo::test::DeviceBuffer<float> input(host_input);
      gufo::test::DeviceBuffer<float> weight(host_weight);
      gufo::test::DeviceBuffer<float> output(dim);
      for (const bool weighted : {false, true}) {
        for (std::size_t i = 0; i < dim; ++i) {
          expected[i] = static_cast<float>(host_input[i] * inverse_rms *
                                           (weighted ? host_weight[i] : 1.0F));
        }
        gufo::hip::LaunchRMSNorm(input.data(),
                                 weighted ? weight.data() : nullptr,
                                 output.data(), dim, 1e-6F);
        const auto actual = output.CopyToHost();
        gufo::test::ExpectSpanNear(expected, actual, 1e-4F,
                                   "GPU RMSNorm output");
      }
    }
  }
}

void TestGpuResidualAdd() {
  constexpr std::size_t dim = 128;
  const std::vector<float> host_lhs(dim, 3.5F);
  const std::vector<float> host_rhs(dim, 1.5F);
  const std::vector<float> expected(dim, 5.0F);
  gufo::test::DeviceBuffer<float> lhs(host_lhs);
  gufo::test::DeviceBuffer<float> rhs(host_rhs);
  gufo::test::DeviceBuffer<float> output(dim);

  gufo::hip::LaunchResidualAdd(lhs.data(), rhs.data(), output.data(), dim);
  HIP_CHECK(hipDeviceSynchronize());

  const auto actual = output.CopyToHost();
  gufo::test::ExpectSpanNear(expected, actual, 1e-5F,
                             "GPU residual-add output");
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen elementwise GPU ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
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
