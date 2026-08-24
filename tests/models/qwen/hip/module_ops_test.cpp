#include <algorithm>
#include <cassert>
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
#include "tests/models/qwen/hip/support/device.hpp"

void TestGpuNormForwardModule() {
  constexpr std::size_t dim = 256;
  std::vector<float> h_x(dim, 1.0F);
  std::vector<float> h_w(dim, 2.0F);
  std::vector<float> h_out(dim, 0.0F);

  float* d_x = nullptr;
  float* d_w = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_x, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, dim * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  const strix::models::qwen::HipModuleContext ctx;  // default HIP stream

  strix::models::qwen::NormLayerView view;
  view.weight.data = d_w;
  view.weight.type = strix::core::GgmlType::kF32;
  view.weight.num_elements = dim;
  view.eps = 1e-6F;

  std::span<const float> x_span(d_x, dim);
  std::span<float> out_span(d_out, dim);
  strix::models::qwen::NormForward(ctx, view, x_span, out_span);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  // mean(x^2)=1, rms=1 => out = x * w = 2.0
  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 2.0F) < 1e-3F);
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

void TestGpuResidualAddModule() {
  constexpr std::size_t dim = 128;
  std::vector<float> h_a(dim, 3.5F);
  std::vector<float> h_b(dim, 1.5F);
  std::vector<float> h_out(dim, 0.0F);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_dst, dim * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  const strix::models::qwen::HipModuleContext ctx;

  // ResidualAdd is in-place on dst: dst = a, src = b => dst = a + b.
  std::span<float> dst(d_dst, dim);
  std::span<const float> src(d_b, dim);
  // Seed dst on device with a (the module reads dst as the accumulator).
  std::vector<float> h_dst(h_a);
  HIP_CHECK(hipMemcpy(d_dst, h_dst.data(), dim * sizeof(float),
                      hipMemcpyHostToDevice));
  strix::models::qwen::ResidualAdd(ctx, dst, src);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(h_out.data(), d_dst, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < dim; ++i) {
    assert(std::abs(h_out[i] - 5.0F) < 1e-4F);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_dst));
}

void TestGpuQuantGemmModule() {
  constexpr std::size_t M = 4;
  constexpr std::size_t K = 8;
  std::vector<float> h_A(M * K, 1.0F);  // all ones
  std::vector<float> h_x(K, 2.0F);      // all twos
  std::vector<float> h_y(M, 0.0F);

  float* d_A = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  const strix::models::qwen::HipModuleContext ctx;

  strix::models::QwenTensorRef A;
  A.data = d_A;
  A.type = strix::core::GgmlType::kF32;
  A.num_elements = M * K;

  std::span<const float> x(d_x, K);
  std::span<float> y(d_y, M);
  strix::models::qwen::QuantGemm(ctx, A, x, M, K, y);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(
      hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t m = 0; m < M; ++m) {
    assert(std::abs(h_y[m] - 16.0F) < 1e-3F);  // K=8 ones * 2.0
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = strix::test::GateHipDevice(
      strix::test::HipDeviceRequirement::kOptional, "Qwen module ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  TestGpuNormForwardModule();
  TestGpuResidualAddModule();
  TestGpuQuantGemmModule();
  std::cout << "Qwen module ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen module ops test.\n";
  return 77;
#endif
}
