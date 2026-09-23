#include <algorithm>
#include <bit>
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

void TestGpuBatchedRMSNorm() {
  std::uint32_t rng = 33;
  const auto draw = [&]() {
    rng = rng * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<int>(rng >> 8U) - 8388608) /
           8388608.0F;
  };
  for (const std::size_t dim : {5119U, 5120U, 5121U}) {
    for (const std::size_t rows : {1U, 7U, 33U, 64U, 65U}) {
      std::vector<float> values(rows * dim), weights(dim), expected(rows * dim);
      for (auto& value : values)
        value = draw();
      for (auto& weight : weights)
        weight = draw();
      gufo::test::DeviceBuffer<float> input(values), weight(weights);
      gufo::test::DeviceBuffer<float> output(values.size());
      gufo::test::DeviceBuffer<std::uint16_t> bf16(values.size());
      for (const float scale : {0.001F, 0.3F, 30.0F}) {
        auto scaled = values;
        for (auto& value : scaled)
          value *= scale;
        input.CopyFrom(scaled);
        for (const bool weighted : {false, true}) {
          for (std::size_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
              const double value = scaled[row * dim + i];
              sum += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sum / dim + 1e-6F);
            for (std::size_t i = 0; i < dim; ++i)
              expected[row * dim + i] =
                  static_cast<float>(scaled[row * dim + i] * inverse *
                                     (weighted ? weights[i] : 1.0F));
          }
          const auto* norm_weight = weighted ? weight.data() : nullptr;
          gufo::hip::LaunchBatchedRMSNorm(input.data(), norm_weight,
                                          output.data(), bf16.data(), rows, dim,
                                          1e-6F);
          const auto actual = output.CopyToHost();
          const auto actual_bf16 = bf16.CopyToHost();
          gufo::test::ExpectSpanNear(expected, actual, 1e-5F,
                                     "batched RMSNorm FP64 formula");
          for (std::size_t row = 0; row < rows; ++row)
            gufo::hip::LaunchRMSNorm(input.data() + row * dim, norm_weight,
                                     output.data() + row * dim, dim, 1e-6F);
          const auto scalar = output.CopyToHost();
          gufo::test::Expect(
              std::memcmp(actual.data(), scalar.data(),
                          actual.size() * sizeof(float)) == 0,
              "scalar and batched RMSNorm must be byte identical");
          for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto bits = std::bit_cast<std::uint32_t>(actual[i]);
            const auto rounded = static_cast<std::uint16_t>(
                (bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
            gufo::test::Expect(actual_bf16[i] == rounded,
                               "batched RMSNorm BF16 rounding");
          }
          gufo::hip::LaunchBatchedRMSNorm(input.data(), norm_weight, nullptr,
                                          bf16.data(), rows, dim, 1e-6F);
          const auto bf16_only = bf16.CopyToHost();
          gufo::test::Expect(
              std::memcmp(actual_bf16.data(), bf16_only.data(),
                          actual_bf16.size() * sizeof(std::uint16_t)) == 0,
              "batched RMSNorm optional FP32 output");
        }
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
  TestGpuBatchedRMSNorm();
  TestGpuResidualAdd();
  std::cout << "Qwen elementwise GPU ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen elementwise GPU ops test.\n";
  return 77;
#endif
}
