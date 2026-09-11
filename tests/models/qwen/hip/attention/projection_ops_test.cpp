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

void TestBatchedFusedProjectionsEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t qkv_size = 10240;
  constexpr std::size_t inner_size = 6144;
  constexpr std::size_t time_step_rank = 48;

  std::vector<float> h_x(batch * hidden_size, 0.5F);
  std::vector<float> h_qkv_w(qkv_size * hidden_size, 0.01F);
  std::vector<float> h_gate_w(inner_size * hidden_size, 0.02F);
  std::vector<float> h_alpha_w(time_step_rank * hidden_size, 0.03F);
  std::vector<float> h_beta_w(time_step_rank * hidden_size, 0.04F);

  float *d_x = nullptr, *d_qkv_w = nullptr, *d_gate_w = nullptr,
        *d_alpha_w = nullptr, *d_beta_w = nullptr;
  float *d_qkv_seq = nullptr, *d_qkv_batch = nullptr;
  float *d_gate_seq = nullptr, *d_gate_batch = nullptr;
  float *d_alpha_seq = nullptr, *d_alpha_batch = nullptr;
  float *d_beta_seq = nullptr, *d_beta_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_x, batch * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_w, qkv_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_w, inner_size * hidden_size * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_alpha_w, time_step_rank * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_w, time_step_rank * hidden_size * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_qkv_seq, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_batch, batch * qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_batch, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_batch, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_seq, batch * time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_batch, batch * time_step_rank * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), batch * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv_w, h_qkv_w.data(),
                      qkv_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate_w, h_gate_w.data(),
                      inner_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha_w, h_alpha_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta_w, h_beta_w.data(),
                      time_step_rank * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    gufo::hip::LaunchFusedSSMInputProjections(
        d_qkv_w, gufo::core::GgmlType::kF32, d_gate_w,
        gufo::core::GgmlType::kF32, d_alpha_w, gufo::core::GgmlType::kF32,
        d_beta_w, gufo::core::GgmlType::kF32, d_x + t * hidden_size,
        d_qkv_seq + t * qkv_size, d_gate_seq + t * inner_size,
        d_alpha_seq + t * time_step_rank, d_beta_seq + t * time_step_rank,
        hidden_size, qkv_size, inner_size, time_step_rank);
  }

  // Batched
  gufo::hip::LaunchBatchedFusedSSMInputProjections(
      d_qkv_w, false, d_gate_w, false, d_alpha_w, false, d_beta_w, false, d_x,
      d_qkv_batch, d_gate_batch, d_alpha_batch, d_beta_batch, batch,
      hidden_size, qkv_size, inner_size, time_step_rank);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_seq(batch * qkv_size);
  std::vector<float> qkv_batch(batch * qkv_size);
  std::vector<float> gate_seq(batch * inner_size);
  std::vector<float> gate_batch(batch * inner_size);
  std::vector<float> alpha_seq(batch * time_step_rank);
  std::vector<float> alpha_batch(batch * time_step_rank);
  std::vector<float> beta_seq(batch * time_step_rank);
  std::vector<float> beta_batch(batch * time_step_rank);
  HIP_CHECK(hipMemcpy(qkv_seq.data(), d_qkv_seq, qkv_seq.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_batch.data(), d_qkv_batch,
                      qkv_batch.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_seq.data(), d_gate_seq,
                      gate_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_batch.data(), d_gate_batch,
                      gate_batch.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_seq.data(), d_alpha_seq,
                      alpha_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_batch.data(), d_alpha_batch,
                      alpha_batch.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_seq.data(), d_beta_seq,
                      beta_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_batch.data(), d_beta_batch,
                      beta_batch.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  gufo::test::ExpectSpanNear(qkv_seq, qkv_batch, 1e-4F,
                             "batched SSM QKV projection mismatch");
  gufo::test::ExpectSpanNear(gate_seq, gate_batch, 1e-4F,
                             "batched SSM gate projection mismatch");
  gufo::test::ExpectSpanNear(alpha_seq, alpha_batch, 1e-4F,
                             "batched SSM alpha projection mismatch");
  gufo::test::ExpectSpanNear(beta_seq, beta_batch, 1e-4F,
                             "batched SSM beta projection mismatch");

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_qkv_w));
  HIP_CHECK(hipFree(d_gate_w));
  HIP_CHECK(hipFree(d_alpha_w));
  HIP_CHECK(hipFree(d_beta_w));
  HIP_CHECK(hipFree(d_qkv_seq));
  HIP_CHECK(hipFree(d_qkv_batch));
  HIP_CHECK(hipFree(d_gate_seq));
  HIP_CHECK(hipFree(d_gate_batch));
  HIP_CHECK(hipFree(d_alpha_seq));
  HIP_CHECK(hipFree(d_alpha_batch));
  HIP_CHECK(hipFree(d_beta_seq));
  HIP_CHECK(hipFree(d_beta_batch));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen attention projection ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedFusedProjectionsEquivalence();
  std::cout << "Qwen attention projection ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention projection ops test.\n";
  return 77;
#endif
}
