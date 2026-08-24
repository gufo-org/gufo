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
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_up_w));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
}

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
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping Qwen ffn fusion ops test.\n";
    return 0;
  }

  TestBatchedFusedSwiGLUEquivalence();
  TestFusedResidualAddRMSNormEquivalence();
  TestBatchedFusedResidualAddRMSNormEquivalence();
  TestBatchedFusedSwiGLUProductionEquivalence();
  TestGEMVResidualEquivalence();
  TestFusedRMSNormSwiGLUEquivalence();
  std::cout << "Qwen ffn fusion ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen ffn fusion ops test.\n";
  return 0;
#endif
}
