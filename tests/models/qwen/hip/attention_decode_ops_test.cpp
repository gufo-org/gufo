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

void TestLongContextDecodeAttention() {
  constexpr std::uint32_t max_position = 32767;
  constexpr std::uint32_t max_context = max_position + 129;
  constexpr std::uint32_t num_heads = 6;
  constexpr std::uint32_t num_kv_heads = 1;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t positions[] = {4095, 8191, 16383, max_position};
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * head_dim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(num_kv_heads) * max_context * head_dim;

  std::vector<float> h_q(attention_width);
  std::vector<float> h_k(kv_width);
  std::vector<float> h_v(kv_width);
  std::vector<float> h_gate(attention_width);
  std::vector<float> h_cache(cache_elements);
  for (std::size_t index = 0; index < attention_width; ++index) {
    h_q[index] =
        0.1F * std::sin(static_cast<float>((index % 257) + 1) * 0.013F);
    h_gate[index] =
        0.4F * std::cos(static_cast<float>((index % 193) + 1) * 0.017F);
  }
  for (std::size_t index = 0; index < kv_width; ++index) {
    h_k[index] =
        0.08F * std::cos(static_cast<float>((index % 251) + 1) * 0.019F);
    h_v[index] =
        0.2F * std::sin(static_cast<float>((index % 239) + 1) * 0.023F);
  }
  for (std::size_t index = 0; index < cache_elements; ++index) {
    h_cache[index] =
        0.03F * std::sin(static_cast<float>((index % 509) + 1) * 0.011F);
  }

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_gate = nullptr;
  float *d_k_cache = nullptr, *d_v_cache = nullptr, *d_out = nullptr;
  float* d_split_k_scratch = nullptr;
  HIP_CHECK(hipMalloc(&d_q, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_cache, cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, attention_width * sizeof(float)));
  HIP_CHECK(hipMalloc(
      &d_split_k_scratch,
      strix::hip::detail::DecodeAttentionScratchElements(num_heads, head_dim) *
          sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), attention_width * sizeof(float),
                      hipMemcpyHostToDevice));
  std::vector<float> output(attention_width);
  std::vector<float> reference(attention_width);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  for (const std::uint32_t position : positions) {
    HIP_CHECK(hipMemcpy(d_k_cache, h_cache.data(),
                        cache_elements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v_cache, h_cache.data(),
                        cache_elements * sizeof(float), hipMemcpyHostToDevice));

    strix::hip::LaunchAttention(d_q, d_k, d_v, d_gate, d_k_cache, d_v_cache,
                                nullptr, nullptr, d_out, 0, position,
                                max_context, num_heads, num_kv_heads, head_dim,
                                nullptr, d_split_k_scratch);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output.data(), d_out, attention_width * sizeof(float),
                        hipMemcpyDeviceToHost));

    std::vector<double> scores(static_cast<std::size_t>(position) + 1);
    for (std::uint32_t head = 0; head < num_heads; ++head) {
      const std::size_t query_offset =
          static_cast<std::size_t>(head) * head_dim;
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::uint32_t token = 0; token <= position; ++token) {
        double dot = 0.0;
        const std::size_t cache_offset =
            static_cast<std::size_t>(token) * head_dim;
        for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
          const float key =
              token == position ? h_k[dim] : h_cache[cache_offset + dim];
          dot += static_cast<double>(h_q[query_offset + dim]) * key;
        }
        scores[token] = dot * scale;
        max_score = std::max(max_score, scores[token]);
      }

      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - max_score);
        denominator += score;
      }
      for (std::uint32_t dim = 0; dim < head_dim; ++dim) {
        double weighted_value = 0.0;
        for (std::uint32_t token = 0; token <= position; ++token) {
          const std::size_t cache_offset =
              static_cast<std::size_t>(token) * head_dim;
          const float value =
              token == position ? h_v[dim] : h_cache[cache_offset + dim];
          weighted_value += scores[token] * value;
        }
        const double sigmoid =
            1.0 / (1.0 + std::exp(-h_gate[query_offset + dim]));
        reference[query_offset + dim] =
            static_cast<float>((weighted_value / denominator) * sigmoid);
      }
    }

    bool has_nonzero = false;
    float max_reference_diff = 0.0F;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const float value = output[index];
      max_reference_diff =
          std::max(max_reference_diff, std::abs(value - reference[index]));
      if (!std::isfinite(value)) {
        std::cerr
            << "Long-context decode attention produced non-finite output\n";
        std::abort();
      }
      has_nonzero = has_nonzero || std::abs(value) > 1e-8F;
    }
    std::cout << "Split-K decode attention context " << (position + 1)
              << " max reference diff: " << max_reference_diff << "\n";
    if (max_reference_diff >= 5e-4F) {
      std::cerr << "Long-context decode attention mismatch\n";
      std::abort();
    }
    if (!has_nonzero) {
      std::cerr << "Long-context decode attention produced only zeroes\n";
      std::abort();
    }
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_k_cache));
  HIP_CHECK(hipFree(d_v_cache));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_split_k_scratch));
}

void TestBaselineToTiledKvCacheTransition() {
  constexpr std::size_t chunk0_size = 512;
  constexpr std::size_t chunk1_size = 1024;
  constexpr std::size_t total_batch = chunk0_size + chunk1_size;
  constexpr std::uint32_t num_heads = 24;
  constexpr std::uint32_t num_kv_heads = 4;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t max_context = 2048;

  const std::size_t attention_width = num_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t q_size = total_batch * attention_width;
  const std::size_t kv_size = total_batch * kv_width;
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
  float *d_cache_seq = nullptr, *d_out_seq = nullptr;
  float *d_cache_trans = nullptr, *d_out_trans = nullptr;
  void* d_cache_trans_f16 = nullptr;

  HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_seq, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_trans, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cache_trans_f16,
                      2 * cache_elements * sizeof(std::uint16_t)));

  HIP_CHECK(hipMemcpy(d_q, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMemset(d_cache_seq, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_seq, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans, 0, 2 * cache_elements * sizeof(float)));
  HIP_CHECK(hipMemset(d_out_trans, 0, q_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_cache_trans_f16, 0,
                      2 * cache_elements * sizeof(std::uint16_t)));

  // 1. Golden sequential reference token-by-token
  for (std::size_t token = 0; token < total_batch; ++token) {
    strix::hip::LaunchAttention(d_q + token * attention_width,
                                d_k + token * kv_width, d_v + token * kv_width,
                                d_gate + token * attention_width, d_cache_seq,
                                d_cache_seq + cache_elements, nullptr, nullptr,
                                d_out_seq + token * attention_width, 0,
                                static_cast<std::uint32_t>(token), max_context,
                                num_heads, num_kv_heads, head_dim);
  }

  // 2. Incremental transition:
  // Chunk 0 (0..512): LaunchBatchedAttention (Baseline)
  strix::hip::LaunchBatchedAttention(
      d_q, d_k, d_v, d_gate, d_cache_trans, d_cache_trans + cache_elements,
      d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans, 0, 0, chunk0_size, max_context, num_heads, num_kv_heads,
      head_dim);

  // Chunk 1 (512..1536): LaunchBatchedAttentionTile (Tiled FP16 at start_pos =
  // 512)
  const bool chunk1_ok = strix::hip::LaunchBatchedAttentionTile(
      d_q + chunk0_size * attention_width, d_k + chunk0_size * kv_width,
      d_v + chunk0_size * kv_width, d_gate + chunk0_size * attention_width,
      d_cache_trans, d_cache_trans + cache_elements, d_cache_trans_f16,
      static_cast<std::uint16_t*>(d_cache_trans_f16) + cache_elements,
      d_out_trans + chunk0_size * attention_width, 0,
      static_cast<std::uint32_t>(chunk0_size), chunk1_size, max_context,
      num_heads, num_kv_heads, head_dim);
  assert(chunk1_ok);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> golden(q_size);
  std::vector<float> transition(q_size);
  HIP_CHECK(hipMemcpy(golden.data(), d_out_seq, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(transition.data(), d_out_trans, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t token = 0; token < total_batch; ++token) {
    for (std::size_t i = 0; i < attention_width; ++i) {
      const float g = golden[token * attention_width + i];
      const float tr = transition[token * attention_width + i];
      const float diff = std::abs(g - tr);
      if (diff > 1e-3F || !std::isfinite(tr)) {
        std::cout << "Token " << token << " index " << i << " head "
                  << (i / head_dim) << " dim " << (i % head_dim)
                  << " golden=" << g << " trans=" << tr << " diff=" << diff
                  << "\n";
        goto done_diff;
      }
    }
  }
done_diff:
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_diff = std::max(max_diff, std::abs(golden[i] - transition[i]));
  }
  std::cout << "Baseline-to-Tiled transition max diff: " << max_diff << "\n";
  assert(max_diff < 5e-3F);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_cache_seq));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_cache_trans));
  HIP_CHECK(hipFree(d_out_trans));
  HIP_CHECK(hipFree(d_cache_trans_f16));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout
        << "No HIP device found, skipping Qwen attention decode ops test.\n";
    return 0;
  }

  TestBatchedAttentionEquivalence();
  TestAttentionBackendEquivalence();
  TestLongContextDecodeAttention();
  TestBaselineToTiledKvCacheTransition();
  std::cout << "Qwen attention decode ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen attention decode ops test.\n";
  return 0;
#endif
}
