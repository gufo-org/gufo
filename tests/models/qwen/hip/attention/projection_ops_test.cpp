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
    strix::hip::LaunchFusedSSMInputProjections(
        d_qkv_w, strix::core::GgmlType::kF32, d_gate_w,
        strix::core::GgmlType::kF32, d_alpha_w, strix::core::GgmlType::kF32,
        d_beta_w, strix::core::GgmlType::kF32, d_x + t * hidden_size,
        d_qkv_seq + t * qkv_size, d_gate_seq + t * inner_size,
        d_alpha_seq + t * time_step_rank, d_beta_seq + t * time_step_rank,
        hidden_size, qkv_size, inner_size, time_step_rank);
  }

  // Batched
  strix::hip::LaunchBatchedFusedSSMInputProjections(
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
  HIP_CHECK(hipMemcpy(qkv_seq.data(), d_qkv_seq,
                      qkv_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_batch.data(), d_qkv_batch,
                      qkv_batch.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_seq.data(), d_gate_seq,
                      gate_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_batch.data(), d_gate_batch,
                      gate_batch.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_seq.data(), d_alpha_seq,
                      alpha_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_batch.data(), d_alpha_batch,
                      alpha_batch.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_seq.data(), d_beta_seq,
                      beta_seq.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_batch.data(), d_beta_batch,
                      beta_batch.size() * sizeof(float), hipMemcpyDeviceToHost));

  strix::test::ExpectSpanNear(qkv_seq, qkv_batch, 1e-4F,
                              "batched SSM QKV projection mismatch");
  strix::test::ExpectSpanNear(gate_seq, gate_batch, 1e-4F,
                              "batched SSM gate projection mismatch");
  strix::test::ExpectSpanNear(alpha_seq, alpha_batch, 1e-4F,
                              "batched SSM alpha projection mismatch");
  strix::test::ExpectSpanNear(beta_seq, beta_batch, 1e-4F,
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

void TestFusedRMSNormQKVProjectionsEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t q_dim = 512;
  constexpr std::size_t kv_dim = 128;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_qw(q_dim * hidden_size);
  std::vector<std::uint16_t> h_kw(kv_dim * hidden_size);
  std::vector<std::uint16_t> h_vw(kv_dim * hidden_size);
  for (std::size_t i = 0; i < q_dim * hidden_size; ++i) {
    h_qw[i] = strix::test::FloatToBf16Bits(
        0.01F * std::sin(static_cast<float>(i) * 0.0021F));
  }
  for (std::size_t i = 0; i < kv_dim * hidden_size; ++i) {
    h_kw[i] = strix::test::FloatToBf16Bits(
        0.013F * std::cos(static_cast<float>(i) * 0.0017F));
    h_vw[i] = strix::test::FloatToBf16Bits(
        0.011F * std::sin(static_cast<float>(i) * 0.0013F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_qw = nullptr, *d_kw = nullptr, *d_vw = nullptr;
  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qw, q_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kw, kv_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vw, kv_dim * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_q_ref, q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_dim * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qw, h_qw.data(),
                      q_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_kw, h_kw.data(),
                      kv_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_vw, h_vw.data(),
                      kv_dim * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  strix::hip::LaunchFusedQKVProjections(
      d_qw, strix::core::GgmlType::kBF16, d_kw, strix::core::GgmlType::kBF16,
      d_vw, strix::core::GgmlType::kBF16, d_normed, d_q_ref, d_k_ref, d_v_ref,
      q_dim, kv_dim, hidden_size);
  strix::hip::LaunchFusedRMSNormQKVProjections(
      d_x, d_w, eps, d_qw, true, d_kw, true, d_vw, true, d_q_fus, d_k_fus,
      d_v_fus, q_dim, kv_dim, hidden_size);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> q_ref(q_dim), k_ref(kv_dim), v_ref(kv_dim);
  std::vector<float> q_fus(q_dim), k_fus(kv_dim), v_fus(kv_dim);
  HIP_CHECK(hipMemcpy(q_ref.data(), d_q_ref, q_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_ref.data(), d_k_ref, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_ref.data(), d_v_ref, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(q_fus.data(), d_q_fus, q_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k_fus.data(), d_k_fus, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(v_fus.data(), d_v_fus, kv_dim * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < q_dim; ++i) {
    max_diff = std::max(max_diff, std::abs(q_ref[i] - q_fus[i]));
  }
  for (std::size_t i = 0; i < kv_dim; ++i) {
    max_diff = std::max(max_diff, std::abs(k_ref[i] - k_fus[i]));
    max_diff = std::max(max_diff, std::abs(v_ref[i] - v_fus[i]));
  }
  std::cout << "Fused RMSNorm+QKV vs unfused max diff: " << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+QKV mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_qw));
  HIP_CHECK(hipFree(d_kw));
  HIP_CHECK(hipFree(d_vw));
  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (!strix::test::HasHipDevice()) {
    std::cout << "No HIP device found, skipping Qwen attention projection ops test.\n";
    return 0;
  }

  TestBatchedFusedProjectionsEquivalence();
  TestFusedRMSNormQKVProjectionsEquivalence();
  std::cout << "Qwen attention projection ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention projection ops test.\n";
  return 0;
#endif
}
