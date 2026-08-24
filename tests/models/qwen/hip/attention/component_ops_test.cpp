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
  strix::test::Expect(max_q_diff < 1e-4F,
                      "batched RoPE Q result mismatch");
  strix::test::Expect(max_k_diff < 1e-4F,
                      "batched RoPE K result mismatch");

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
  strix::test::Expect(max_diff < 1e-4F,
                      "batched per-head RMSNorm result mismatch");

  HIP_CHECK(hipFree(d_x_seq));
  HIP_CHECK(hipFree(d_x_batch));
  HIP_CHECK(hipFree(d_w));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (!strix::test::HasHipDevice()) {
    std::cout << "No HIP device found, skipping Qwen attention component ops test.\n";
    return 0;
  }

  TestBatchedRoPEEquivalence();
  TestBatchedPerHeadRMSNormEquivalence();
  std::cout << "Qwen attention component ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention component ops test.\n";
  return 0;
#endif
}
