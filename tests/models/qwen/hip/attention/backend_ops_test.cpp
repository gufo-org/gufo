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

void TestBatchedAttentionEquivalence() {
  constexpr std::size_t batch = 2048;
  constexpr std::uint32_t num_heads = 2;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 32;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t q_size = batch * num_heads * head_dim;
  const std::size_t kv_size = batch * num_kv_heads * head_dim;
  const std::size_t cache_size = 8 * num_kv_heads * max_context * head_dim * 2;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] = 0.08F * std::sin(static_cast<float>(index % 257) * 0.07F);
    h_gate[index] = 0.4F * std::cos(static_cast<float>(index % 193) * 0.05F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] = 0.07F * std::cos(static_cast<float>(index % 251) * 0.06F);
    h_v[index] = 0.2F * std::sin(static_cast<float>(index % 239) * 0.04F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_batch = nullptr,
        *d_cache_gemm = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr, *d_out_gemm = nullptr;
  float* d_scores = nullptr;
  hipblasHandle_t hipblas_handle = nullptr;

  HIPBLAS_CHECK(hipblasCreate(&hipblas_handle));
  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_batch, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_gemm, cache_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_gemm, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scores, (num_heads / num_kv_heads) * batch *
                                     max_context * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_batch, 0, cache_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_gemm, 0, cache_size * sizeof(float)));

  const std::size_t total_k = 8 * num_kv_heads * max_context * head_dim;

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    strix::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k, nullptr, nullptr,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  strix::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache_batch,
                                     d_cache_batch + total_k, nullptr, nullptr,
                                     d_out_batch, 0, 0, batch, max_context,
                                     num_heads, num_kv_heads, head_dim);
  strix::hip::LaunchBatchedAttentionGemm(
      hipblas_handle, d_q, d_k, d_v, d_gate, d_cache_gemm,
      d_cache_gemm + total_k, nullptr, nullptr, d_scores, d_out_gemm, 0, 0,
      batch, max_context, num_heads, num_kv_heads, head_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(q_size), res_batch(q_size), res_gemm(q_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_gemm.data(), d_out_gemm, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  float max_gemm_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
    const float gemm_diff = std::abs(res_seq[i] - res_gemm[i]);
    if (gemm_diff > max_gemm_diff)
      max_gemm_diff = gemm_diff;
  }
  std::cout << "Attention Seq vs Batch max diff: " << max_diff << "\n";
  std::cout << "Attention Seq vs GEMM max diff: " << max_gemm_diff << "\n";
  if (max_diff >= 1e-4F || max_gemm_diff >= 1e-4F) {
    std::cerr << "Batched attention mismatch\n";
    std::abort();
  }

  HIPBLAS_CHECK(hipblasDestroy(hipblas_handle));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_batch));
  HIP_CHECK(hipFree(d_cache_gemm));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_gemm));
  HIP_CHECK(hipFree(d_scores));
}

void TestAttentionBackendEquivalence() {
  constexpr std::size_t batch = 128;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 128;
  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = batch * attention_width;
  const std::size_t kv_size = batch * kv_width;
  const std::size_t cache_elements = num_kv_heads * max_context * head_dim;

  std::vector<float> h_q(q_size);
  std::vector<float> h_k(kv_size);
  std::vector<float> h_v(kv_size);
  std::vector<float> h_gate(q_size);
  for (std::size_t index = 0; index < q_size; ++index) {
    h_q[index] =
        0.15F * std::sin(static_cast<float>((index % 257) + 1) * 0.017F);
    h_gate[index] =
        0.5F * std::cos(static_cast<float>((index % 193) + 1) * 0.013F);
  }
  for (std::size_t index = 0; index < kv_size; ++index) {
    h_k[index] =
        0.2F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.25F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_cache_seq = nullptr, *d_cache_tile = nullptr, *d_cache_ck = nullptr;
  float *d_out_seq = nullptr, *d_out_tile = nullptr, *d_out_ck = nullptr;
  void *d_cache_tile_f16 = nullptr, *d_cache_ck_f16 = nullptr,
       *d_scratch_ck_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_tile, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_ck, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_cache_tile_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_cache_ck_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_scratch_ck_f16, 2 * q_size * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_tile, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ck, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_ck, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMemset(d_cache_ck_f16, 0, 2 * cache_elements * sizeof(hip_bfloat16)));

  for (std::size_t token = 0; token < batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }
  const bool tile_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!tile_launched) {
    std::cerr << "Tiled attention rejected the Qwen shape\n";
    std::abort();
  }
  const bool launched = strix::hip::LaunchBatchedAttentionCk(
      d_q, d_k, d_v, d_gate, d_cache_ck, d_cache_ck + cache_elements,
      d_cache_ck_f16,
      static_cast<std::uint16_t*>(d_cache_ck_f16) + cache_elements,
      d_scratch_ck_f16, d_out_ck, 0, 0, batch, max_context, num_heads,
      num_kv_heads, head_dim);
  if (!launched) {
    std::cerr << "Composable Kernel attention rejected the Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> sequential(q_size), tiled(q_size), ck(q_size);
  std::vector<float> sequential_cache(2 * cache_elements);
  std::vector<float> tile_cache(2 * cache_elements);
  std::vector<float> ck_cache(2 * cache_elements);
  HIP_CHECK(hipMemcpy(sequential.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck.data(), d_out_ck, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sequential_cache.data(), d_cache_seq,
                      sequential_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck_cache.data(), d_cache_ck,
                      ck_cache.size() * sizeof(float), hipMemcpyDeviceToHost));
  float max_tile_diff = 0.0F;
  float max_ck_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_tile_diff =
        std::max(max_tile_diff, std::abs(sequential[index] - tiled[index]));
    max_ck_diff =
        std::max(max_ck_diff, std::abs(sequential[index] - ck[index]));
  }
  float max_tile_cache_diff = 0.0F;
  float max_cache_diff = 0.0F;
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_tile_cache_diff =
        std::max(max_tile_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
    max_cache_diff = std::max(
        max_cache_diff, std::abs(sequential_cache[index] - ck_cache[index]));
  }
  std::cout << "Attention Seq vs tile max diff: " << max_tile_diff << "\n";
  std::cout << "Attention Seq vs tile cache max diff: " << max_tile_cache_diff
            << "\n";
  std::cout << "Attention Seq vs CK max diff: " << max_ck_diff << "\n";
  std::cout << "Attention Seq vs CK cache max diff: " << max_cache_diff << "\n";
  if (max_tile_diff >= 5e-3F || max_tile_cache_diff != 0.0F) {
    std::cerr << "Tiled attention mismatch\n";
    std::abort();
  }
  if (max_ck_diff >= 2e-3F || max_cache_diff != 0.0F) {
    std::cerr << "Composable Kernel attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_out_tile, 0, q_size * sizeof(float)));
  constexpr std::size_t chunk_size = batch / 2;
  const bool first_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, chunk_size, max_context, num_heads, num_kv_heads,
      head_dim);
  const bool second_chunk_launched = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk_size * attention_width, d_k + chunk_size * kv_width,
      d_v + chunk_size * kv_width, d_gate + chunk_size * attention_width,
      d_cache_tile, d_cache_tile + cache_elements, d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile + chunk_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk_size), chunk_size, max_context,
      num_heads, num_kv_heads, head_dim);
  if (!first_chunk_launched || !second_chunk_launched) {
    std::cerr << "Tiled attention rejected a chunked Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  float max_chunked_diff = 0.0F;
  float max_chunked_cache_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_chunked_diff =
        std::max(max_chunked_diff, std::abs(sequential[index] - tiled[index]));
  }
  for (std::size_t index = 0; index < sequential_cache.size(); ++index) {
    max_chunked_cache_diff =
        std::max(max_chunked_cache_diff,
                 std::abs(sequential_cache[index] - tile_cache[index]));
  }
  std::cout << "Attention Seq vs chunked tile max diff: " << max_chunked_diff
            << "\n";
  std::cout << "Attention Seq vs chunked tile cache max diff: "
            << max_chunked_cache_diff << "\n";
  if (max_chunked_diff >= 5e-3F || max_chunked_cache_diff != 0.0F) {
    std::cerr << "Chunked tiled attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_cache_tile));
  HIP_CHECK(hipFree(d_cache_ck));
  HIP_CHECK(hipFree(d_cache_tile_f16));
  HIP_CHECK(hipFree(d_cache_ck_f16));
  HIP_CHECK(hipFree(d_scratch_ck_f16));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_tile));
  HIP_CHECK(hipFree(d_out_ck));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  if (!strix::test::HasHipDevice()) {
    std::cout << "No HIP device found, skipping Qwen attention backend ops test.\n";
    return 0;
  }

  TestBatchedAttentionEquivalence();
  TestAttentionBackendEquivalence();
  std::cout << "Qwen attention backend ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention backend ops test.\n";
  return 0;
#endif
}
