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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/ops.hpp"
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
    gufo::hip::LaunchAttention(
        d_q + t * num_heads * head_dim, d_k + t * num_kv_heads * head_dim,
        d_v + t * num_kv_heads * head_dim, d_gate + t * num_heads * head_dim,
        d_cache_seq, d_cache_seq + total_k, nullptr, nullptr,
        d_out_seq + t * num_heads * head_dim, 0, static_cast<std::uint32_t>(t),
        max_context, num_heads, num_kv_heads, head_dim);
  }

  // Batched
  gufo::hip::LaunchBatchedAttention(d_q, d_k, d_v, d_gate, d_cache_batch,
                                    d_cache_batch + total_k, nullptr, nullptr,
                                    d_out_batch, 0, 0, batch, max_context,
                                    num_heads, num_kv_heads, head_dim);
  gufo::hip::LaunchBatchedAttentionGemm(
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
  float *d_cache_seq = nullptr, *d_cache_tile = nullptr,
        *d_cache_wmma = nullptr;
  float *d_out_seq = nullptr, *d_out_tile = nullptr, *d_out_wmma = nullptr;
  void *d_cache_tile_f16 = nullptr, *d_cache_wmma_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_tile, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_wmma, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_cache_tile_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(
      hipMalloc(&d_cache_wmma_f16, 2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_tile, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_wmma, q_size * sizeof(float)));

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
  HIP_CHECK(hipMemset(d_cache_wmma, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_cache_wmma_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));

  for (std::size_t token = 0; token < batch; ++token) {
    gufo::hip::LaunchAttention(d_q + token * attention_width,
                               d_k + token * kv_width, d_v + token * kv_width,
                               d_gate + token * attention_width, d_cache_seq,
                               d_cache_seq + cache_elements, nullptr, nullptr,
                               d_out_seq + token * attention_width, 0,
                               static_cast<std::uint32_t>(token), max_context,
                               num_heads, num_kv_heads, head_dim);
  }
  const bool tile_launched = gufo::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!tile_launched) {
    std::cerr << "Tiled attention rejected the Qwen shape\n";
    std::abort();
  }
  const bool launched = gufo::hip::LaunchQwenWmmaAttention(
      d_q, d_k, d_v, d_gate, d_cache_wmma, d_cache_wmma + cache_elements,
      d_cache_wmma_f16,
      static_cast<std::uint16_t*>(d_cache_wmma_f16) + cache_elements,
      d_out_wmma, 0, 0, batch, max_context, num_heads, num_kv_heads, head_dim);
  if (!launched) {
    std::cerr << "Native WMMA attention rejected the Qwen shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> sequential(q_size), tiled(q_size), wmma(q_size);
  std::vector<float> sequential_cache(2 * cache_elements);
  std::vector<float> tile_cache(2 * cache_elements);
  std::vector<float> ck_cache(2 * cache_elements);
  HIP_CHECK(hipMemcpy(sequential.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tiled.data(), d_out_tile, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wmma.data(), d_out_wmma, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sequential_cache.data(), d_cache_seq,
                      sequential_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tile_cache.data(), d_cache_tile,
                      tile_cache.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ck_cache.data(), d_cache_wmma,
                      ck_cache.size() * sizeof(float), hipMemcpyDeviceToHost));
  float max_tile_diff = 0.0F;
  float max_wmma_diff = 0.0F;
  for (std::size_t index = 0; index < q_size; ++index) {
    max_tile_diff =
        std::max(max_tile_diff, std::abs(sequential[index] - tiled[index]));
    max_wmma_diff =
        std::max(max_wmma_diff, std::abs(sequential[index] - wmma[index]));
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
  std::cout << "Attention Seq vs WMMA max diff: " << max_wmma_diff << "\n";
  std::cout << "Attention Seq vs WMMA cache max diff: " << max_cache_diff
            << "\n";
  if (max_tile_diff >= 5e-3F || max_tile_cache_diff != 0.0F) {
    std::cerr << "Tiled attention mismatch\n";
    std::abort();
  }
  if (max_wmma_diff >= 2e-3F || max_cache_diff != 0.0F) {
    std::cerr << "Native WMMA attention mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipMemset(d_cache_tile, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_tile_f16, 0,
                      2 * cache_elements * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemset(d_out_tile, 0, q_size * sizeof(float)));
  constexpr std::size_t chunk_size = batch / 2;
  const bool first_chunk_launched = gufo::hip::LaunchBatchedAttentionTile(
      d_q, d_k, d_v, d_gate, d_cache_tile, d_cache_tile + cache_elements,
      d_cache_tile_f16,
      static_cast<std::uint16_t*>(d_cache_tile_f16) + cache_elements,
      d_out_tile, 0, 0, chunk_size, max_context, num_heads, num_kv_heads,
      head_dim);
  const bool second_chunk_launched = gufo::hip::LaunchBatchedAttentionTile(
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
  HIP_CHECK(hipFree(d_cache_wmma));
  HIP_CHECK(hipFree(d_cache_tile_f16));
  HIP_CHECK(hipFree(d_cache_wmma_f16));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_tile));
  HIP_CHECK(hipFree(d_out_wmma));
}

// Verification rows over the fp16 cache in the Qwen3.8-27B shape: 2..8 rows
// share KV tiles in one block, other counts keep the per-pair kernel. Every
// count must match one scalar decode per row bit for bit, on both sides of
// the 512-token shared-KV threshold and across partial tiles.
void TestVerificationAttentionMatchesScalarRows() {
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 3072;
  constexpr std::uint32_t layer = 1;
  constexpr std::uint32_t max_rows = 10;
  constexpr std::size_t width = std::size_t{num_heads} * head_dim;
  constexpr std::size_t cache_elements =
      std::size_t{2} * max_context * num_kv_heads * head_dim;

  std::vector<float> h_q(max_rows * width), h_gate(max_rows * width);
  for (std::size_t index = 0; index < h_q.size(); ++index) {
    h_q[index] = 1.5F * std::sin(static_cast<float>(index) * 0.073F);
    h_gate[index] = std::cos(static_cast<float>(index) * 0.019F);
  }
  // Large keys on some heads keep the running maximum moving.
  std::vector<__half> h_k(cache_elements), h_v(cache_elements);
  for (std::size_t index = 0; index < cache_elements; ++index) {
    const float key_scale = (index / head_dim) % 3 == 0 ? 2.5F : 0.5F;
    h_k[index] =
        __float2half(key_scale * std::sin(static_cast<float>(index) * 0.011F));
    h_v[index] = __float2half(std::cos(static_cast<float>(index) * 0.007F));
  }

  float *d_q = nullptr, *d_gate = nullptr, *d_scalar = nullptr,
        *d_batched = nullptr, *d_scratch = nullptr;
  __half *d_k = nullptr, *d_v = nullptr;
  const std::size_t scratch_elements =
      max_rows *
      gufo::hip::detail::DecodeAttentionScratchElements(num_heads, head_dim);
  HIP_CHECK(hipMalloc(&d_q, h_q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, h_gate.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scalar, h_q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_batched, h_q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scratch, scratch_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, cache_elements * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_v, cache_elements * sizeof(__half)));
  HIP_CHECK(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), cache_elements * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), cache_elements * sizeof(__half),
                      hipMemcpyHostToDevice));

  std::vector<float> scalar(h_q.size()), batched(h_q.size());
  std::size_t checked = 0;
  for (const std::uint32_t start : {100U, 509U, 511U, 512U, 777U, 2047U, 3000U})
    for (std::uint32_t rows = 1; rows <= max_rows; ++rows) {
      if (start + rows > max_context)
        continue;
      HIP_CHECK(hipMemset(d_scalar, 0xff, h_q.size() * sizeof(float)));
      HIP_CHECK(hipMemset(d_batched, 0x7f, h_q.size() * sizeof(float)));
      for (std::uint32_t row = 0; row < rows; ++row) {
        gufo::hip::LaunchAttention(
            d_q + row * width, nullptr, nullptr, d_gate + row * width, nullptr,
            nullptr, d_k, d_v, d_scalar + row * width, layer, start + row,
            max_context, num_heads, num_kv_heads, head_dim, nullptr, d_scratch,
            true);
      }
      gufo::hip::LaunchCausalDecodeAttention(
          d_q, d_gate, nullptr, nullptr, d_k, d_v, d_batched, layer, start,
          rows, max_context, num_heads, num_kv_heads, head_dim, nullptr,
          {d_scratch, scratch_elements});
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(scalar.data(), d_scalar, rows * width * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(batched.data(), d_batched,
                          rows * width * sizeof(float), hipMemcpyDeviceToHost));
      if (std::memcmp(scalar.data(), batched.data(),
                      rows * width * sizeof(float)) != 0) {
        std::cerr << "Verification attention differs from scalar rows at start "
                  << start << " with " << rows << " rows\n";
        std::abort();
      }
      ++checked;
    }
  std::cout << "Verification attention bit-identical to scalar rows: "
            << checked << " shapes\n";

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_scalar));
  HIP_CHECK(hipFree(d_batched));
  HIP_CHECK(hipFree(d_scratch));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen attention backend ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedAttentionEquivalence();
  TestAttentionBackendEquivalence();
  TestVerificationAttentionMatchesScalarRows();
  std::cout << "Qwen attention backend ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention backend ops test.\n";
  return 77;
#endif
}
