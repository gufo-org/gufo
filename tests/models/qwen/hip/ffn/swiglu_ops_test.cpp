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

void TestBatchedFusedSwiGLUEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 2560;
  constexpr std::size_t intermediate_size = 9216;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_gate_w(intermediate_size * hidden_size, 0.01F);
  std::vector<float> h_up_w(intermediate_size * hidden_size, 0.02F);

  float *d_x = nullptr, *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_gate_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_up_w, intermediate_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * intermediate_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchFusedSwiGLUGEMV(
        d_gate_w, strix::core::GgmlType::kF32, d_up_w,
        strix::core::GgmlType::kF32, d_x + t * hidden_size,
        d_out_seq + t * intermediate_size, intermediate_size, hidden_size);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSwiGLUGEMM(d_gate_w, false, d_up_w, false, d_x,
                                           d_out_batch, nullptr, batch,
                                           intermediate_size, hidden_size);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * intermediate_size),
      res_batch(batch * intermediate_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SwiGLU Seq vs Batch max diff: " << max_diff << "\n";
  strix::test::Expect(max_diff < 1e-4F, "batched fused SwiGLU result mismatch");

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

void TestBatchedFusedSwiGLUProductionEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 512;
  constexpr std::size_t intermediate_size = 1024;

  std::vector<float> h_x(batch * hidden_size);
  std::vector<std::uint16_t> h_gate_w(intermediate_size * hidden_size);
  std::vector<std::uint16_t> h_up_w(intermediate_size * hidden_size);
  for (std::size_t i = 0; i < batch * hidden_size; ++i) {
    h_x[i] = 0.5F * std::sin(static_cast<float>(i + 1) * 0.037F);
  }
  for (std::size_t i = 0; i < intermediate_size * hidden_size; ++i) {
    h_gate_w[i] = strix::test::FloatToBf16Bits(
        0.012F * std::cos(static_cast<float>(i + 1) * 0.0021F));
    h_up_w[i] = strix::test::FloatToBf16Bits(
        0.017F * std::sin(static_cast<float>(i + 1) * 0.0017F));
  }

  float *d_x = nullptr, *d_x_bf16 = nullptr;
  void *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_gate = nullptr, *d_up = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  void *d_out_bf16_ref = nullptr, *d_out_bf16_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x_bf16, batch * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_gate_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_up_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_gate, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16_ref,
                      batch * intermediate_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_bf16_fus,
                      batch * intermediate_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchFloatToBfloat16(d_x, d_x_bf16, batch * hidden_size);

  hipblasHandle_t handle = nullptr;
  HIPBLAS_CHECK(hipblasCreate(&handle));

  // Unfused production chain: gate GEMM, up GEMM, SwiGLU activation.
  strix::hip::LaunchHipblasGEMMBF16(handle, d_gate_w, d_x_bf16, d_gate, batch,
                                    intermediate_size, hidden_size);
  strix::hip::LaunchHipblasGEMMBF16(handle, d_up_w, d_x_bf16, d_up, batch,
                                    intermediate_size, hidden_size);
  strix::hip::LaunchBatchedSwiGLUActivation(
      d_gate, d_up, d_out_ref, d_out_bf16_ref, batch * intermediate_size);

  // Fused kernel (consumes the FP32 normed input).
  strix::hip::LaunchBatchedFusedSwiGLUGEMM(d_gate_w, true, d_up_w, true, d_x,
                                           d_out_fus, d_out_bf16_fus, batch,
                                           intermediate_size, hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_ref(batch * intermediate_size);
  std::vector<float> res_fus(batch * intermediate_size);
  std::vector<std::uint16_t> res_bf16_ref(batch * intermediate_size);
  std::vector<std::uint16_t> res_bf16_fus(batch * intermediate_size);
  HIP_CHECK(hipMemcpy(res_ref.data(), d_out_ref,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_fus.data(), d_out_fus,
                      batch * intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_ref.data(), d_out_bf16_ref,
                      batch * intermediate_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_fus.data(), d_out_bf16_fus,
                      batch * intermediate_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F, max_bf16_diff = 0.0F, mean_diff = 0.0F;
  for (std::size_t i = 0; i < batch * intermediate_size; ++i) {
    const float d = std::abs(res_ref[i] - res_fus[i]);
    max_diff = std::max(max_diff, d);
    mean_diff += d;
    const float db = std::abs(strix::test::Bf16BitsToFloat(res_bf16_ref[i]) -
                              strix::test::Bf16BitsToFloat(res_bf16_fus[i]));
    max_bf16_diff = std::max(max_bf16_diff, db);
  }
  mean_diff /= static_cast<float>(batch * intermediate_size);
  std::cout << "BatchedFusedSwiGLU vs production chain: max_diff=" << max_diff
            << " mean_diff=" << mean_diff << " max_bf16_diff=" << max_bf16_diff
            << "\n";
  if (max_diff >= 1e-2F || max_bf16_diff >= 1e-2F) {
    std::cerr << "Fused SwiGLU GEMM does not match the unfused production "
                 "chain (max_diff "
              << max_diff << ", max_bf16_diff " << max_bf16_diff << ")\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(handle));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_x_bf16));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_out_bf16_ref));
  HIP_CHECK(hipFree(d_out_bf16_fus));
}

void TestFusedRMSNormSwiGLUEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t intermediate_size = 2048;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_gate_w(intermediate_size * hidden_size);
  std::vector<std::uint16_t> h_up_w(intermediate_size * hidden_size);
  for (std::size_t i = 0; i < intermediate_size * hidden_size; ++i) {
    h_gate_w[i] = strix::test::FloatToBf16Bits(
        0.012F * std::cos(static_cast<float>(i) * 0.0021F));
    h_up_w[i] = strix::test::FloatToBf16Bits(
        0.017F * std::sin(static_cast<float>(i) * 0.0017F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_gate_w = nullptr, *d_up_w = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_up_w,
                      intermediate_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, intermediate_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, intermediate_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_up_w, h_up_w.data(),
                      intermediate_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  strix::hip::LaunchFusedSwiGLUGEMV(d_gate_w, strix::core::GgmlType::kBF16,
                                    d_up_w, strix::core::GgmlType::kBF16,
                                    d_normed, d_out_ref, intermediate_size,
                                    hidden_size);
  strix::hip::LaunchFusedRMSNormSwiGLUGEMV(d_x, d_w, eps, d_gate_w, d_up_w,
                                           d_out_fus, intermediate_size,
                                           hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(intermediate_size);
  std::vector<float> fus(intermediate_size);
  HIP_CHECK(hipMemcpy(ref.data(), d_out_ref, intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fus.data(), d_out_fus, intermediate_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < intermediate_size; ++i) {
    max_diff = std::max(max_diff, std::abs(ref[i] - fus[i]));
  }
  std::cout << "Fused RMSNorm+SwiGLU vs unfused max diff: " << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+SwiGLU mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = strix::test::GateHipDevice(
      strix::test::HipDeviceRequirement::kOptional, "Qwen SwiGLU ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedFusedSwiGLUEquivalence();
  TestBatchedFusedSwiGLUProductionEquivalence();
  TestFusedRMSNormSwiGLUEquivalence();
  std::cout << "Qwen SwiGLU ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen SwiGLU ops test.\n";
  return 77;
#endif
}
