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

  std::vector<float> res_seq(batch * qkv_size);
  std::vector<float> res_batch(batch * qkv_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_qkv_seq,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_qkv_batch,
                      batch * qkv_size * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Fused SSM Projections Seq vs Batch max diff: " << max_diff
            << "\n";
  assert(max_diff < 1e-4F);

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

void TestBatchedRoPEEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 16;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;

  std::vector<float> h_q(q_size), h_k(kv_size);
  for (std::size_t i = 0; i < q_size; ++i)
    h_q[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < kv_size; ++i)
    h_k[i] = std::cos(static_cast<float>(i + 1));

  float *d_q_seq = nullptr, *d_k_seq = nullptr;
  float *d_q_batch = nullptr, *d_k_batch = nullptr;

  HIP_CHECK(hipMalloc(&d_q_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_seq, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_batch, kv_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_seq, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_seq, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_batch, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_batch, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchRoPE(d_q_seq + t * (num_heads * head_dim),
                           d_k_seq + t * (num_kv_heads * head_dim), num_heads,
                           num_kv_heads, head_dim, rotary_dim,
                           static_cast<std::uint32_t>(t), rope_theta);
  }

  // Batched
  strix::hip::LaunchBatchedRoPE(d_q_batch, d_k_batch, batch, num_heads,
                                num_kv_heads, head_dim, rotary_dim, 0,
                                rope_theta);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_seq(q_size), res_q_batch(q_size);
  std::vector<float> res_k_seq(kv_size), res_k_batch(kv_size);
  HIP_CHECK(hipMemcpy(res_q_seq.data(), d_q_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_batch.data(), d_q_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_seq.data(), d_k_seq, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_batch.data(), d_k_batch, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_q_diff = 0.0F, max_k_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    const float d = std::abs(res_q_seq[i] - res_q_batch[i]);
    if (d > max_q_diff)
      max_q_diff = d;
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    const float d = std::abs(res_k_seq[i] - res_k_batch[i]);
    if (d > max_k_diff)
      max_k_diff = d;
  }
  std::cout << "RoPE Seq vs Batch max Q diff: " << max_q_diff
            << " K diff: " << max_k_diff << "\n";
  assert(max_q_diff < 1e-4F);
  assert(max_k_diff < 1e-4F);

  HIP_CHECK(hipFree(d_q_seq));
  HIP_CHECK(hipFree(d_k_seq));
  HIP_CHECK(hipFree(d_q_batch));
  HIP_CHECK(hipFree(d_k_batch));
}

void TestBatchedPerHeadRMSNormEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr float eps = 1e-6F;

  const std::size_t total_size = batch * num_heads * head_dim;
  std::vector<float> h_x(total_size), h_w(head_dim);
  for (std::size_t i = 0; i < total_size; ++i)
    h_x[i] = std::sin(static_cast<float>(i + 1));
  for (std::size_t i = 0; i < head_dim; ++i)
    h_w[i] = 1.0F + 0.1F * std::cos(static_cast<float>(i + 1));

  float *d_x_seq = nullptr, *d_x_batch = nullptr, *d_w = nullptr;
  HIP_CHECK(hipMalloc(&d_x_seq, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_x_batch, total_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, head_dim * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x_seq, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x_batch, h_x.data(), total_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));

  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchPerHeadRMSNorm(d_x_seq + t * (num_heads * head_dim), d_w,
                                     d_x_seq + t * (num_heads * head_dim),
                                     num_heads, head_dim, eps);
  }

  strix::hip::LaunchBatchedPerHeadRMSNorm(d_x_batch, d_w, d_x_batch, batch,
                                          num_heads, head_dim, eps);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(total_size), res_batch(total_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_x_seq, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_x_batch, total_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < total_size; ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "PerHeadRMSNorm Seq vs Batch max diff: " << max_diff << "\n";
  assert(max_diff < 1e-4F);

  HIP_CHECK(hipFree(d_x_seq));
  HIP_CHECK(hipFree(d_x_batch));
  HIP_CHECK(hipFree(d_w));
}

void TestFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 1;
  constexpr std::uint32_t max_context = 16;
  constexpr std::uint32_t pos = 3;

  const std::size_t q_size = num_heads * head_dim;
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.18F * std::sin(static_cast<float>(i + 1) * 0.05F);
    h_gate[i] = 0.3F * std::cos(static_cast<float>(i + 1) * 0.03F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.11F * std::cos(static_cast<float>(i + 1) * 0.07F);
    h_v[i] = 0.24F * std::sin(static_cast<float>(i + 1) * 0.04F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.9F + 0.04F * static_cast<float>(i % 17);
    h_k_w[i] = 1.1F - 0.03F * static_cast<float>(i % 13);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  std::uint32_t* d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_pos, &pos, sizeof(std::uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Unfused reference chain
  strix::hip::LaunchPerHeadRMSNorm(d_q_ref, d_wq, d_q_ref, num_heads, head_dim,
                                   eps);
  strix::hip::LaunchPerHeadRMSNorm(d_k_ref, d_wk, d_k_ref, num_kv_heads,
                                   head_dim, eps);
  strix::hip::LaunchRoPE(d_q_ref, d_k_ref, num_heads, num_kv_heads, head_dim,
                         rotary_dim, d_pos, rope_theta);
  strix::hip::LaunchAttention(d_q_ref, d_k_ref, d_v_ref, d_gate, d_kc_ref,
                              d_vc_ref, d_kc16_ref, d_vc16_ref, d_out_ref,
                              layer_idx, d_pos, max_context, num_heads,
                              num_kv_heads, head_dim, nullptr,
                              /*skip_kv_write=*/false);

  // Fused chain
  strix::hip::LaunchFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, d_pos, max_context,
      num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta, eps);
  strix::hip::LaunchAttention(d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus,
                              d_vc_fus, d_kc16_fus, d_vc16_fus, d_out_fus,
                              layer_idx, d_pos, max_context, num_heads,
                              num_kv_heads, head_dim, nullptr,
                              /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Fused Q mismatch at " << i << ": " << res_q_ref[i] << " vs "
                << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Fused K mismatch at " << i << ": " << res_k_ref[i] << " vs "
                << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "FusedQKNormRoPEKvWrite decode: exact Q/K/cache match, max out "
               "diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Fused decode attention output mismatch: " << max_out_diff
              << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_pos));
}

void TestBatchedFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::size_t batch = 8;
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 2;
  constexpr std::uint32_t max_context = 64;
  constexpr std::uint32_t start_pos = 5;

  const std::size_t head_total = num_heads * head_dim;
  const std::size_t kv_total = num_kv_heads * head_dim;
  const std::size_t q_size = batch * head_total;
  const std::size_t kv_size = batch * kv_total;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.16F * std::sin(static_cast<float>(i + 1) * 0.04F);
    h_gate[i] = 0.25F * std::cos(static_cast<float>(i + 1) * 0.06F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.13F * std::cos(static_cast<float>(i + 1) * 0.05F);
    h_v[i] = 0.21F * std::sin(static_cast<float>(i + 1) * 0.08F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.95F + 0.03F * static_cast<float>(i % 19);
    h_k_w[i] = 1.05F - 0.02F * static_cast<float>(i % 11);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Per-token unfused reference chain
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchPerHeadRMSNorm(d_q_ref + t * head_total, d_wq,
                                     d_q_ref + t * head_total, num_heads,
                                     head_dim, eps);
    strix::hip::LaunchPerHeadRMSNorm(d_k_ref + t * kv_total, d_wk,
                                     d_k_ref + t * kv_total, num_kv_heads,
                                     head_dim, eps);
    strix::hip::LaunchRoPE(d_q_ref + t * head_total, d_k_ref + t * kv_total,
                           num_heads, num_kv_heads, head_dim, rotary_dim,
                           static_cast<std::uint32_t>(start_pos + t),
                           rope_theta);
    strix::hip::LaunchAttention(
        d_q_ref + t * head_total, d_k_ref + t * kv_total,
        d_v_ref + t * kv_total, d_gate + t * head_total, d_kc_ref, d_vc_ref,
        d_kc16_ref, d_vc16_ref, d_out_ref + t * head_total, layer_idx,
        static_cast<std::uint32_t>(start_pos + t), max_context, num_heads,
        num_kv_heads, head_dim, nullptr, nullptr, /*skip_kv_write=*/false);
  }

  // Batched fused chain
  strix::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, start_pos, batch,
      max_context, num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta,
      eps);
  strix::hip::LaunchBatchedAttention(
      d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus, d_vc_fus, d_kc16_fus,
      d_vc16_fus, d_out_fus, layer_idx, start_pos, batch, max_context,
      num_heads, num_kv_heads, head_dim, nullptr,
      /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Batched fused Q mismatch at " << i << ": " << res_q_ref[i]
                << " vs " << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Batched fused K mismatch at " << i << ": " << res_k_ref[i]
                << " vs " << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Batched fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Batched fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "BatchedFusedQKNormRoPEKvWrite prefill: exact Q/K/cache match, "
               "max out diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Batched fused prefill attention output mismatch: "
              << max_out_diff << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
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
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout
        << "No HIP device found, skipping Qwen attention fusion ops test.\n";
    return 0;
  }

  TestBatchedFusedProjectionsEquivalence();
  TestBatchedRoPEEquivalence();
  TestBatchedPerHeadRMSNormEquivalence();
  TestFusedQKNormRoPEKvWriteEquivalence();
  TestBatchedFusedQKNormRoPEKvWriteEquivalence();
  TestFusedRMSNormQKVProjectionsEquivalence();
  std::cout << "Qwen attention fusion ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention fusion ops test.\n";
  return 0;
#endif
}
