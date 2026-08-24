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

void TestFusedResidualAddRMSNormEquivalence() {
  constexpr std::size_t dim = 5120;
  std::vector<float> h_a(dim), h_b(dim), h_w(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    h_a[i] = 0.17F * std::sin(static_cast<float>(i + 1) * 0.013F);
    h_b[i] = 0.09F * std::cos(static_cast<float>(i + 1) * 0.007F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_w = nullptr;
  float *d_sum_ref = nullptr, *d_sum_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_a, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_ref, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_fus, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, dim * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_a, h_a.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_b, h_b.data(), dim * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  // Unfused reference chain.
  strix::hip::LaunchResidualAdd(d_a, d_b, d_sum_ref, dim);
  strix::hip::LaunchRMSNorm(d_sum_ref, d_w, d_out_ref, dim, 1e-6F);

  // Fused kernel.
  strix::hip::LaunchFusedResidualAddRMSNorm(d_a, d_b, d_sum_fus, d_w, d_out_fus,
                                            dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_sum_ref(dim), res_sum_fus(dim);
  std::vector<float> res_out_ref(dim), res_out_fus(dim);
  HIP_CHECK(hipMemcpy(res_sum_ref.data(), d_sum_ref, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_sum_fus.data(), d_sum_fus, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < dim; ++i) {
    if (res_sum_ref[i] != res_sum_fus[i]) {
      std::cerr << "Fused residual sum mismatch at " << i << ": "
                << res_sum_ref[i] << " vs " << res_sum_fus[i] << "\n";
      std::abort();
    }
    if (res_out_ref[i] != res_out_fus[i]) {
      std::cerr << "Fused residual RMSNorm mismatch at " << i << ": "
                << res_out_ref[i] << " vs " << res_out_fus[i] << "\n";
      std::abort();
    }
  }
  std::cout << "FusedResidualAddRMSNorm decode: exact sum and normed match\n";

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_sum_ref));
  HIP_CHECK(hipFree(d_sum_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

void TestBatchedFusedResidualAddRMSNormEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t dim = 5120;
  std::vector<float> h_a(batch * dim), h_b(batch * dim), h_w(dim);
  for (std::size_t i = 0; i < batch * dim; ++i) {
    h_a[i] = 0.23F * std::cos(static_cast<float>(i + 1) * 0.011F);
    h_b[i] = 0.11F * std::sin(static_cast<float>(i + 1) * 0.019F);
  }
  for (std::size_t i = 0; i < dim; ++i) {
    h_w[i] = 1.1F - 0.04F * static_cast<float>(i % 19);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_w = nullptr;
  float *d_sum_ref = nullptr, *d_sum_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  void *d_out_bf16_ref = nullptr, *d_out_bf16_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_a, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_ref, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_sum_fus, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16_ref, batch * dim * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_bf16_fus, batch * dim * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), batch * dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), batch * dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_w, h_w.data(), dim * sizeof(float), hipMemcpyHostToDevice));

  // Unfused reference chain.
  strix::hip::LaunchBatchedResidualAdd(d_a, d_b, d_sum_ref, batch, dim);
  strix::hip::LaunchBatchedRMSNorm(d_sum_ref, d_w, d_out_ref, d_out_bf16_ref,
                                   batch, dim, 1e-6F);

  // Fused kernel.
  strix::hip::LaunchBatchedFusedResidualAddRMSNorm(
      d_a, d_b, d_sum_fus, d_w, d_out_fus, d_out_bf16_fus, batch, dim, 1e-6F);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_sum_ref(batch * dim), res_sum_fus(batch * dim);
  std::vector<float> res_out_ref(batch * dim), res_out_fus(batch * dim);
  std::vector<std::uint16_t> res_bf16_ref(batch * dim),
      res_bf16_fus(batch * dim);
  HIP_CHECK(hipMemcpy(res_sum_ref.data(), d_sum_ref,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_sum_fus.data(), d_sum_fus,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus,
                      batch * dim * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_ref.data(), d_out_bf16_ref,
                      batch * dim * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16_fus.data(), d_out_bf16_fus,
                      batch * dim * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < batch * dim; ++i) {
    if (res_sum_ref[i] != res_sum_fus[i]) {
      std::cerr << "Batched fused residual sum mismatch at " << i << ": "
                << res_sum_ref[i] << " vs " << res_sum_fus[i] << "\n";
      std::abort();
    }
    if (res_out_ref[i] != res_out_fus[i]) {
      std::cerr << "Batched fused residual RMSNorm mismatch at " << i << ": "
                << res_out_ref[i] << " vs " << res_out_fus[i] << "\n";
      std::abort();
    }
    if (res_bf16_ref[i] != res_bf16_fus[i]) {
      std::cerr << "Batched fused residual BF16 mismatch at " << i << "\n";
      std::abort();
    }
  }
  std::cout << "BatchedFusedResidualAddRMSNorm prefill: exact sum, normed, and "
               "BF16 match\n";

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_sum_ref));
  HIP_CHECK(hipFree(d_sum_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_out_bf16_ref));
  HIP_CHECK(hipFree(d_out_bf16_fus));
}

void TestGEMVResidualEquivalence() {
  // Wave32 single-row: BF16 weights, M=5120, K=6144 (decode ssm_out shape).
  constexpr std::size_t M = 5120;
  constexpr std::size_t K = 6144;
  std::vector<std::uint16_t> h_A(M * K);
  std::vector<float> h_x(K);
  std::vector<float> h_res(M);
  for (std::size_t i = 0; i < M * K; ++i) {
    h_A[i] = strix::test::FloatToBf16Bits(
        0.01F * std::sin(static_cast<float>(i) * 0.0007F));
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.3F * std::cos(static_cast<float>(i) * 0.011F);
  }
  for (std::size_t i = 0; i < M; ++i) {
    h_res[i] = 0.25F * std::sin(static_cast<float>(i) * 0.017F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_res = nullptr;
  float *d_y_ref = nullptr, *d_y_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_res, M * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_ref, M * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_fus, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_res, h_res.data(), M * sizeof(float), hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A, strix::core::GgmlType::kBF16, d_x, d_y_ref, M, K);
  strix::hip::LaunchResidualAdd(d_res, d_y_ref, d_y_ref, M);
  strix::hip::LaunchGEMVResidual(d_A, strix::core::GgmlType::kBF16, d_x,
                                 d_y_fus, d_res, M, K);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref(M);
  std::vector<float> fus(M);
  HIP_CHECK(
      hipMemcpy(ref.data(), d_y_ref, M * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(fus.data(), d_y_fus, M * sizeof(float), hipMemcpyDeviceToHost));
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < M; ++i) {
    const float d = std::abs(ref[i] - fus[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Wave32 GEMV+residual fused vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Wave32 GEMV+residual mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_res));
  HIP_CHECK(hipFree(d_y_ref));
  HIP_CHECK(hipFree(d_y_fus));

  // Block strategy: FP32 weights, K >= 8192.
  constexpr std::size_t M2 = 2048;
  constexpr std::size_t K2 = 12288;
  std::vector<float> h_A2(M2 * K2);
  std::vector<float> h_x2(K2);
  std::vector<float> h_res2(M2);
  for (std::size_t i = 0; i < M2 * K2; ++i) {
    h_A2[i] = 0.008F * std::sin(static_cast<float>(i) * 0.0003F);
  }
  for (std::size_t i = 0; i < K2; ++i) {
    h_x2[i] = 0.2F * std::cos(static_cast<float>(i) * 0.007F);
  }
  for (std::size_t i = 0; i < M2; ++i) {
    h_res2[i] = 0.15F * std::sin(static_cast<float>(i) * 0.019F);
  }

  float *d_A2 = nullptr, *d_x2 = nullptr, *d_res2 = nullptr;
  float *d_y_ref2 = nullptr, *d_y_fus2 = nullptr;
  HIP_CHECK(hipMalloc(&d_A2, M2 * K2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x2, K2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_res2, M2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_ref2, M2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y_fus2, M2 * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A2, h_A2.data(), M2 * K2 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x2, h_x2.data(), K2 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_res2, h_res2.data(), M2 * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchGEMV(d_A2, strix::core::GgmlType::kF32, d_x2, d_y_ref2,
                         M2, K2);
  strix::hip::LaunchResidualAdd(d_res2, d_y_ref2, d_y_ref2, M2);
  strix::hip::LaunchGEMVResidual(d_A2, strix::core::GgmlType::kF32, d_x2,
                                 d_y_fus2, d_res2, M2, K2);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> ref2(M2);
  std::vector<float> fus2(M2);
  HIP_CHECK(hipMemcpy(ref2.data(), d_y_ref2, M2 * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(fus2.data(), d_y_fus2, M2 * sizeof(float),
                      hipMemcpyDeviceToHost));
  max_diff = 0.0F;
  for (std::size_t i = 0; i < M2; ++i) {
    const float d = std::abs(ref2[i] - fus2[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Block GEMV+residual fused vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Block GEMV+residual mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A2));
  HIP_CHECK(hipFree(d_x2));
  HIP_CHECK(hipFree(d_res2));
  HIP_CHECK(hipFree(d_y_ref2));
  HIP_CHECK(hipFree(d_y_fus2));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (!strix::test::HasHipDevice()) {
    std::cout
        << "No HIP device found, skipping Qwen FFN residual ops test.\n";
    return 0;
  }

  TestFusedResidualAddRMSNormEquivalence();
  TestBatchedFusedResidualAddRMSNormEquivalence();
  TestGEMVResidualEquivalence();
  std::cout << "Qwen FFN residual and epilogue ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen FFN residual and epilogue ops test.\n";
  return 0;
#endif
}
