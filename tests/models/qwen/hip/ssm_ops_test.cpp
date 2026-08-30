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
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"

void TestBf16RecurrentMemoryAndSnapshot() {
  gufo::core::ModelConfig production;
  production.model_name = "qwen3.8-27b";
  production.num_layers = 64;
  production.hidden_size = 5120;
  production.intermediate_size = 17408;
  production.num_attention_heads = 24;
  production.num_key_value_heads = 4;
  production.head_dim = 256;
  production.vocab_size = 248320;
  production.context_length = 262144;
  production.full_attention_interval = 4;
  production.mtp_num_layers = 1;
  production.ssm_conv_kernel = 4;
  production.ssm_state_size = 128;
  production.ssm_group_count = 16;
  production.ssm_time_step_rank = 48;
  production.ssm_inner_size = 6144;
  production.rotary_dim = 64;
  production.rope_theta = 10000000.0F;

  auto fp32_policy = gufo::hip::QwenExecutionPolicy::Production();
  auto bf16_policy = fp32_policy;
  bf16_policy.recurrent_state_storage =
      gufo::hip::QwenRecurrentStateStorage::kBf16;

  constexpr std::uint32_t context = 4096;
  const auto fp32_usage = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
      production, context, fp32_policy);
  const auto bf16_usage = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
      production, context, bf16_policy);
  const std::size_t recurrent_elements =
      static_cast<std::size_t>(production.num_layers) *
      production.ssm_time_step_rank * production.ssm_state_size *
      production.SsmValueSize();
  const std::size_t expected_savings =
      recurrent_elements * (sizeof(float) - sizeof(std::uint16_t));
  if (fp32_usage.request_state_bytes - bf16_usage.request_state_bytes !=
      expected_savings) {
    throw std::runtime_error(
        "BF16 recurrent accounting did not remove exactly two bytes per "
        "state element");
  }
  if (fp32_usage.temporary_scratch_bytes !=
      bf16_usage.temporary_scratch_bytes) {
    throw std::runtime_error(
        "recurrent storage precision changed shared scratch accounting");
  }

  std::cout << "Qwen27B BF16 recurrent fp32_state_bytes="
            << fp32_usage.request_state_bytes
            << " bf16_state_bytes=" << bf16_usage.request_state_bytes
            << " saved_per_state=" << expected_savings << "\n";
  for (const std::size_t concurrency : {1U, 2U, 4U}) {
    std::cout << "  C=" << concurrency
              << " fp32_reserved=" << concurrency * fp32_usage.TotalBytes()
              << " bf16_reserved=" << concurrency * bf16_usage.TotalBytes()
              << " saved=" << concurrency * expected_savings << "\n";
  }

  const auto small = gufo::models::qwen::make_small_qwen_config();
  constexpr std::uint32_t small_context = 32;
  gufo::hip::QwenGpuArena fp32_arena(small, small_context, fp32_policy);
  gufo::hip::QwenGpuArena bf16_arena(small, small_context, bf16_policy);
  const auto bf16_small_usage = gufo::hip::QwenGpuArena::EstimateMemoryUsage(
      small, small_context, bf16_policy);
  auto fp32_snapshot = fp32_arena.SaveSnapshot(7);
  auto bf16_snapshot = bf16_arena.SaveSnapshot(7);
  if (bf16_snapshot->RecurrentStateStorage() !=
      gufo::hip::QwenRecurrentStateStorage::kBf16) {
    throw std::runtime_error(
        "snapshot did not record canonical BF16 recurrent storage");
  }
  if (bf16_snapshot->PayloadBytes() != bf16_small_usage.request_state_bytes) {
    throw std::runtime_error(
        "BF16 snapshot payload disagrees with state-byte accounting");
  }

  bool rejected = false;
  try {
    bf16_arena.RestoreSnapshot(*fp32_snapshot);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error(
        "BF16 arena accepted an FP32 recurrent-state snapshot");
  }
}

void TestBatchedSSMConvEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim, 1.0F);
  std::vector<float> h_gate(batch * inner_size, 0.5F);

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_seq = nullptr, *d_state_batch = nullptr;
  float* d_state_bf16 = nullptr;
  float *d_conv_out_seq = nullptr, *d_conv_out_batch = nullptr;
  float* d_conv_out_bf16 = nullptr;
  float *d_delta_seq = nullptr, *d_delta_batch = nullptr;
  void* d_delta_bf16 = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_seq = nullptr, *d_out_batch = nullptr;
  float* d_out_bf16 = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_seq, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_batch, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_bf16, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_seq, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_batch, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_bf16, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_seq, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_batch, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_bf16, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_seq, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_batch, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_seq, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_bf16, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_seq, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_batch, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_bf16, 0, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Sequential
  for (std::size_t t = 0; t < batch; ++t) {
    gufo::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w, d_state_seq, d_conv_out_seq + t * qkv_dim,
        d_delta_seq, d_alpha + t * num_heads, d_beta + t * num_heads, d_ssm_a,
        d_ssm_dt, d_ssm_norm, d_gate + t * inner_size,
        d_out_seq + t * inner_size, 0, qkv_dim, num_key_heads, num_heads,
        key_dim, val_dim);
    gufo::hip::LaunchSSMConvRecurrence(
        d_qkv + t * qkv_dim, d_w, d_state_bf16, d_conv_out_bf16 + t * qkv_dim,
        d_delta_bf16, d_alpha + t * num_heads, d_beta + t * num_heads, d_ssm_a,
        d_ssm_dt, d_ssm_norm, d_gate + t * inner_size,
        d_out_bf16 + t * inner_size, 0, qkv_dim, num_key_heads, num_heads,
        key_dim, val_dim, nullptr, {},
        gufo::hip::QwenRecurrentStateStorage::kBf16);
  }

  // Batched
  gufo::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_batch, d_conv_out_batch, d_delta_batch, d_alpha,
      d_beta, d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_batch, 0, batch,
      qkv_dim, num_key_heads, num_heads, key_dim, val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_seq(batch * inner_size);
  std::vector<float> res_batch(batch * inner_size);
  std::vector<float> res_bf16(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_seq.data(), d_out_seq,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_batch.data(), d_out_batch,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_bf16.data(), d_out_bf16,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    const float d = std::abs(res_seq[i] - res_batch[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "DeltaNet Seq vs Batch max diff: " << max_diff << "\n";
  if (max_diff >= 1e-4F) {
    std::cerr << "DeltaNet batched recurrence mismatch\n";
    std::abort();
  }

  float bf16_output_max_diff = 0.0F;
  for (std::size_t i = 0; i < res_seq.size(); ++i) {
    if (!std::isfinite(res_bf16[i])) {
      std::cerr << "BF16 DeltaNet decode produced non-finite output\n";
      std::abort();
    }
    bf16_output_max_diff =
        std::max(bf16_output_max_diff, std::abs(res_seq[i] - res_bf16[i]));
  }

  std::vector<float> delta_fp32(delta_size);
  std::vector<std::uint16_t> delta_bf16(delta_size);
  HIP_CHECK(hipMemcpy(delta_fp32.data(), d_delta_seq,
                      delta_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(delta_bf16.data(), d_delta_bf16,
                      delta_size * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  float bf16_state_max_diff = 0.0F;
  for (std::size_t i = 0; i < delta_size; ++i) {
    const float value = gufo::test::Bf16BitsToFloat(delta_bf16[i]);
    if (!std::isfinite(value)) {
      std::cerr << "BF16 DeltaNet decode produced non-finite state\n";
      std::abort();
    }
    bf16_state_max_diff =
        std::max(bf16_state_max_diff, std::abs(delta_fp32[i] - value));
  }
  std::cout << "DeltaNet BF16 decode max output diff: " << bf16_output_max_diff
            << " state diff: " << bf16_state_max_diff << "\n";
  if (bf16_output_max_diff >= 2e-2F || bf16_state_max_diff >= 1e-2F) {
    std::cerr << "BF16 DeltaNet decode exceeded the storage envelope\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_seq));
  HIP_CHECK(hipFree(d_state_batch));
  HIP_CHECK(hipFree(d_state_bf16));
  HIP_CHECK(hipFree(d_conv_out_seq));
  HIP_CHECK(hipFree(d_conv_out_batch));
  HIP_CHECK(hipFree(d_conv_out_bf16));
  HIP_CHECK(hipFree(d_delta_seq));
  HIP_CHECK(hipFree(d_delta_batch));
  HIP_CHECK(hipFree(d_delta_bf16));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_seq));
  HIP_CHECK(hipFree(d_out_batch));
  HIP_CHECK(hipFree(d_out_bf16));
}

void TestBatchedSSMRecurrenceNormGateEquivalence() {
  constexpr std::size_t batch = 4;
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = std::sin(static_cast<float>(i) * 0.01F);
  }
  std::vector<float> h_weights(qkv_dim * 4, 0.25F);
  std::vector<float> h_ssm_a(num_heads, -0.05F);
  std::vector<float> h_ssm_dt(num_heads, 0.01F);
  std::vector<float> h_ssm_norm(val_dim);
  std::vector<float> h_gate(batch * inner_size);
  for (std::size_t i = 0; i < val_dim; ++i) {
    h_ssm_norm[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  for (std::size_t i = 0; i < batch * inner_size; ++i) {
    h_gate[i] = 0.5F * std::sin(static_cast<float>(i + 1) * 0.013F);
  }

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_ref = nullptr, *d_state_fus = nullptr;
  float *d_conv_out_ref = nullptr, *d_conv_out_fus = nullptr;
  float *d_delta_ref = nullptr, *d_delta_fus = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;

  const std::size_t delta_size = num_heads * key_dim * val_dim;
  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_ref, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_fus, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_ref, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_out_fus, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_ref, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_fus, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, batch * inner_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(), batch * qkv_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_weights.data(), qkv_dim * 4 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_ref, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_fus, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_ref, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_delta_fus, 0, delta_size * sizeof(float)));
  HIP_CHECK(hipMemset(d_alpha, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemset(d_beta, 0, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_ssm_a, h_ssm_a.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_dt, h_ssm_dt.data(), num_heads * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_ssm_norm, h_ssm_norm.data(), val_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), batch * inner_size * sizeof(float),
                      hipMemcpyHostToDevice));

  // Unfused reference chain: conv + recurrence + post-norm gate.
  gufo::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_ref, d_conv_out_ref, d_delta_ref, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_ref, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);

  // Fused: conv + recurrence with the norm+gate in the epilogue.
  gufo::hip::LaunchBatchedSSMConvRecurrenceNormGate(
      d_qkv, d_w, d_state_fus, d_conv_out_fus, d_delta_fus, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_fus, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_ref(batch * inner_size);
  std::vector<float> res_fus(batch * inner_size);
  HIP_CHECK(hipMemcpy(res_ref.data(), d_out_ref,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_fus.data(), d_out_fus,
                      batch * inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < res_ref.size(); ++i) {
    const float d = std::abs(res_ref[i] - res_fus[i]);
    if (d > max_diff)
      max_diff = d;
  }
  std::cout << "Batched SSM recurrence+norm+gate fused vs unfused max diff: "
            << max_diff << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused SSM recurrence+norm+gate mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_state_ref));
  HIP_CHECK(hipFree(d_state_fus));
  HIP_CHECK(hipFree(d_conv_out_ref));
  HIP_CHECK(hipFree(d_conv_out_fus));
  HIP_CHECK(hipFree(d_delta_ref));
  HIP_CHECK(hipFree(d_delta_fus));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_ssm_a));
  HIP_CHECK(hipFree(d_ssm_dt));
  HIP_CHECK(hipFree(d_ssm_norm));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// opt-c170-deltanet-rowsplit: the row-split recurrence spreads the 128 state
// rows of a head over 16 waves and applies the k/q normalization scales to the
// reduced dot products instead of to the 128-wide vectors. That is the same
// arithmetic reassociated, so it must agree with the single-block kernel to
// FP32 rounding over a full chunk -- including the carried state, which is what
// accumulates any real error across tokens.
//
// Both register tiles are exercised: the launcher picks the 32-key/one-row tile
// at or below 2048 tokens and the two-row prefetching tile above it.
void TestBatchedSSMRowSplitRecurrenceEquivalence(std::size_t batch) {
  constexpr std::uint32_t num_key_heads = 16;
  constexpr std::uint32_t num_heads = 48;
  constexpr std::uint32_t key_dim = 128;
  constexpr std::uint32_t val_dim = 128;
  constexpr std::size_t qkv_dim =
      (2 * num_key_heads * key_dim) + (num_heads * val_dim);
  constexpr std::size_t inner_size = num_heads * val_dim;
  const std::size_t delta_size =
      static_cast<std::size_t>(num_heads) * key_dim * val_dim;

  std::cout << "Batched SSM row-split recurrence: batch=" << batch << "\n";

  std::vector<float> h_qkv(batch * qkv_dim);
  for (std::size_t i = 0; i < h_qkv.size(); ++i) {
    h_qkv[i] = 0.6F * std::sin(0.017F * static_cast<float>(i % 997)) +
               0.2F * std::cos(0.003F * static_cast<float>(i % 31));
  }
  std::vector<float> h_weights(qkv_dim * 4);
  for (std::size_t i = 0; i < h_weights.size(); ++i) {
    h_weights[i] = 0.2F + 0.05F * static_cast<float>(i % 7);
  }
  // Non-trivial decay and beta gates: with alpha and beta pinned to zero the
  // recurrence degenerates to one decay value and the state carry is not
  // tested.
  std::vector<float> h_alpha(batch * num_heads);
  std::vector<float> h_beta(batch * num_heads);
  for (std::size_t i = 0; i < h_alpha.size(); ++i) {
    h_alpha[i] = 1.5F + 0.5F * std::sin(0.011F * static_cast<float>(i % 401));
    h_beta[i] = 0.3F * std::cos(0.007F * static_cast<float>(i % 211));
  }
  std::vector<float> h_ssm_a(num_heads);
  std::vector<float> h_ssm_dt(num_heads);
  for (std::uint32_t i = 0; i < num_heads; ++i) {
    h_ssm_a[i] = -0.04F - 0.01F * static_cast<float>(i % 5);
    h_ssm_dt[i] = 0.1F * static_cast<float>(i % 3);
  }
  std::vector<float> h_ssm_norm(val_dim);
  for (std::size_t i = 0; i < val_dim; ++i) {
    h_ssm_norm[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<float> h_gate(batch * inner_size);
  for (std::size_t i = 0; i < h_gate.size(); ++i) {
    h_gate[i] = 0.5F * std::sin(0.013F * static_cast<float>(i + 1));
  }
  // A non-zero starting state so the decay path is live from the first token.
  std::vector<float> h_delta0(delta_size);
  std::vector<std::uint16_t> h_delta0_bf16(delta_size);
  for (std::size_t i = 0; i < h_delta0.size(); ++i) {
    h_delta0[i] = 0.01F * std::sin(0.013F * static_cast<float>(i % 577));
    h_delta0_bf16[i] = gufo::test::FloatToBf16Bits(h_delta0[i]);
  }

  float *d_qkv = nullptr, *d_w = nullptr;
  float *d_state_ref = nullptr, *d_state_new = nullptr;
  float* d_state_bf16 = nullptr;
  float *d_conv_ref = nullptr, *d_conv_new = nullptr;
  float* d_conv_bf16 = nullptr;
  float *d_delta_ref = nullptr, *d_delta_new = nullptr;
  void* d_delta_bf16 = nullptr;
  float *d_alpha = nullptr, *d_beta = nullptr;
  float *d_ssm_a = nullptr, *d_ssm_dt = nullptr, *d_ssm_norm = nullptr;
  float* d_gate = nullptr;
  float *d_out_ref = nullptr, *d_out_new = nullptr;
  float* d_out_bf16 = nullptr;
  float *d_kq = nullptr, *d_ab = nullptr;

  HIP_CHECK(hipMalloc(&d_qkv, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_ref, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_new, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_state_bf16, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_ref, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_new, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_conv_bf16, batch * qkv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_ref, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_new, delta_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_delta_bf16, delta_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta, batch * num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_a, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_dt, num_heads * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_norm, val_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_ref, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_new, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_bf16, batch * inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kq, batch * num_key_heads * 3 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ab, batch * num_heads * 2 * sizeof(float)));

  const auto upload = [](float* dst, const std::vector<float>& src) {
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  };
  upload(d_qkv, h_qkv);
  upload(d_w, h_weights);
  upload(d_alpha, h_alpha);
  upload(d_beta, h_beta);
  upload(d_ssm_a, h_ssm_a);
  upload(d_ssm_dt, h_ssm_dt);
  upload(d_ssm_norm, h_ssm_norm);
  upload(d_gate, h_gate);
  upload(d_delta_ref, h_delta0);
  upload(d_delta_new, h_delta0);
  HIP_CHECK(hipMemcpy(d_delta_bf16, h_delta0_bf16.data(),
                      h_delta0_bf16.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_state_ref, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_new, 0, qkv_dim * 4 * sizeof(float)));
  HIP_CHECK(hipMemset(d_state_bf16, 0, qkv_dim * 4 * sizeof(float)));

  gufo::hip::LaunchBatchedSSMConvRecurrence(
      d_qkv, d_w, d_state_ref, d_conv_ref, d_delta_ref, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_ref, 0, batch, qkv_dim,
      num_key_heads, num_heads, key_dim, val_dim);

  if (!gufo::hip::IsDeltaNetRowSplitSupported(key_dim, val_dim)) {
    std::cerr << "row-split recurrence rejected the production state shape\n";
    std::abort();
  }
  gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
      d_qkv, d_w, d_state_new, d_conv_new, d_delta_new, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_new, /*q8_out=*/nullptr,
      d_kq, d_ab, 0, batch, qkv_dim, num_key_heads, num_heads, key_dim,
      val_dim);
  gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
      d_qkv, d_w, d_state_bf16, d_conv_bf16, d_delta_bf16, d_alpha, d_beta,
      d_ssm_a, d_ssm_dt, d_ssm_norm, d_gate, d_out_bf16,
      /*q8_out=*/nullptr, d_kq, d_ab, 0, batch, qkv_dim, num_key_heads,
      num_heads, key_dim, val_dim, nullptr,
      gufo::hip::QwenRecurrentStateStorage::kBf16);

  HIP_CHECK(hipDeviceSynchronize());

  const auto download = [](std::vector<float>& dst, const float* src) {
    HIP_CHECK(hipMemcpy(dst.data(), src, dst.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
  };
  std::vector<float> out_ref(batch * inner_size);
  std::vector<float> out_new(batch * inner_size);
  std::vector<float> out_bf16(batch * inner_size);
  std::vector<float> delta_ref(delta_size);
  std::vector<float> delta_new(delta_size);
  std::vector<std::uint16_t> delta_bf16_bits(delta_size);
  std::vector<float> delta_bf16(delta_size);
  download(out_ref, d_out_ref);
  download(out_new, d_out_new);
  download(out_bf16, d_out_bf16);
  download(delta_ref, d_delta_ref);
  download(delta_new, d_delta_new);
  HIP_CHECK(hipMemcpy(delta_bf16_bits.data(), d_delta_bf16,
                      delta_bf16_bits.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < delta_bf16.size(); ++i) {
    delta_bf16[i] = gufo::test::Bf16BitsToFloat(delta_bf16_bits[i]);
  }

  const auto compare = [](const char* label, const std::vector<float>& want,
                          const std::vector<float>& got) {
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                             static_cast<double>(want[i])));
      magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
    }
    const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
    std::cout << "  " << label << ": max_abs=" << max_abs
              << " magnitude=" << magnitude << " rel=" << rel
              << " non_finite=" << non_finite << "\n";
    if (non_finite != 0 || rel > 1e-5) {
      std::cerr << label
                << ": row-split recurrence disagrees with the single-block "
                   "kernel\n";
      std::abort();
    }
  };
  compare("gated output", out_ref, out_new);
  compare("carried state", delta_ref, delta_new);

  const auto compare_bf16 = [](const char* label,
                               const std::vector<float>& want,
                               const std::vector<float>& got,
                               double relative_tolerance) {
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                             static_cast<double>(want[i])));
      magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
    }
    const double relative = magnitude > 0.0 ? max_abs / magnitude : max_abs;
    std::cout << "  BF16 " << label << ": max_abs=" << max_abs
              << " magnitude=" << magnitude << " rel=" << relative
              << " non_finite=" << non_finite << "\n";
    if (non_finite != 0 || relative > relative_tolerance) {
      std::cerr << "BF16 " << label
                << " exceeded the recurrent-storage envelope\n";
      std::abort();
    }
  };
  compare_bf16("gated output", out_ref, out_bf16, 3e-2);
  compare_bf16("carried state", delta_ref, delta_bf16, 3e-2);

  // opt-c174-ssm-epilogue-quant: with a Q8_1 destination the epilogue quantizes
  // the gated row in the same pass instead of storing FP32 for the quantizer to
  // read back. That has to be byte-for-byte identical to the FP32 path followed
  // by an FP32 quantize, including the per-block scales and the tail-tile
  // zeros.
  if (gufo::hip::IsFusedSSMEpilogueQuantizeQ8_1Supported(val_dim, inner_size)) {
    const std::size_t num_blocks = inner_size / 32;
    const std::size_t q8_bytes =
        ((((batch + 15) / 16) * num_blocks * 576)) + 4096;
    void* d_q8_ref = nullptr;
    void* d_q8_got = nullptr;
    float* d_state_q8 = nullptr;
    float* d_conv_q8 = nullptr;
    float* d_delta_q8 = nullptr;
    float* d_out_q8 = nullptr;
    HIP_CHECK(hipMalloc(&d_q8_ref, q8_bytes));
    HIP_CHECK(hipMalloc(&d_q8_got, q8_bytes));
    HIP_CHECK(hipMalloc(&d_state_q8, qkv_dim * 4 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_conv_q8, batch * qkv_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_delta_q8, delta_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_q8, batch * inner_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_q8_ref, 0xA5, q8_bytes));
    HIP_CHECK(hipMemset(d_q8_got, 0xA5, q8_bytes));

    // Reference: the FP32 epilogue that just ran, quantized separately.
    gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(d_out_new, d_q8_ref, batch,
                                                    inner_size);

    // Candidate: the same recurrence from the same starting state, but with the
    // epilogue writing Q8_1.
    HIP_CHECK(hipMemset(d_state_q8, 0, qkv_dim * 4 * sizeof(float)));
    upload(d_delta_q8, h_delta0);
    gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
        d_qkv, d_w, d_state_q8, d_conv_q8, d_delta_q8, d_alpha, d_beta, d_ssm_a,
        d_ssm_dt, d_ssm_norm, d_gate, d_out_q8, d_q8_got, d_kq, d_ab, 0, batch,
        qkv_dim, num_key_heads, num_heads, key_dim, val_dim);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint8_t> q8_ref(q8_bytes);
    std::vector<std::uint8_t> q8_got(q8_bytes);
    HIP_CHECK(
        hipMemcpy(q8_ref.data(), d_q8_ref, q8_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(q8_got.data(), d_q8_got, q8_bytes, hipMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < q8_bytes; ++i) {
      if (q8_ref[i] != q8_got[i]) {
        ++mismatches;
      }
    }
    std::cout << "  fused Q8_1 epilogue mismatching bytes: " << mismatches
              << " of " << q8_bytes << "\n";
    if (mismatches != 0) {
      std::cerr << "fused SSM Q8_1 epilogue differs from the FP32 epilogue "
                   "plus a separate quantize\n";
      std::abort();
    }

    HIP_CHECK(hipFree(d_q8_ref));
    HIP_CHECK(hipFree(d_q8_got));
    HIP_CHECK(hipFree(d_state_q8));
    HIP_CHECK(hipFree(d_conv_q8));
    HIP_CHECK(hipFree(d_delta_q8));
    HIP_CHECK(hipFree(d_out_q8));
  }

  HIP_CHECK(hipFree(d_delta_bf16));
  for (float* p :
       {d_qkv,      d_w,        d_state_ref, d_state_new, d_state_bf16,
        d_conv_ref, d_conv_new, d_conv_bf16, d_delta_ref, d_delta_new,
        d_alpha,    d_beta,     d_ssm_a,     d_ssm_dt,    d_ssm_norm,
        d_gate,     d_out_ref,  d_out_new,   d_out_bf16,  d_kq,
        d_ab}) {
    HIP_CHECK(hipFree(p));
  }
}

void TestFusedRMSNormSSMInputProjectionsEquivalence() {
  constexpr std::size_t hidden_size = 1024;
  constexpr std::size_t qkv_size = 2048;
  constexpr std::size_t inner_size = 512;
  constexpr std::size_t time_step_rank = 16;
  constexpr float eps = 1e-6F;

  std::vector<float> h_x(hidden_size);
  std::vector<float> h_w(hidden_size);
  for (std::size_t i = 0; i < hidden_size; ++i) {
    h_x[i] = 0.3F * std::sin(static_cast<float>(i) * 0.017F);
    h_w[i] = 0.9F + 0.05F * static_cast<float>(i % 23);
  }
  std::vector<std::uint16_t> h_qkv(qkv_size * hidden_size);
  std::vector<std::uint16_t> h_gate(inner_size * hidden_size);
  std::vector<std::uint16_t> h_alpha(time_step_rank * hidden_size);
  std::vector<std::uint16_t> h_beta(time_step_rank * hidden_size);
  for (std::size_t i = 0; i < qkv_size * hidden_size; ++i) {
    h_qkv[i] = gufo::test::FloatToBf16Bits(
        0.009F * std::cos(static_cast<float>(i) * 0.0019F));
  }
  for (std::size_t i = 0; i < inner_size * hidden_size; ++i) {
    h_gate[i] = gufo::test::FloatToBf16Bits(
        0.012F * std::sin(static_cast<float>(i) * 0.0011F));
  }
  for (std::size_t i = 0; i < time_step_rank * hidden_size; ++i) {
    h_alpha[i] = gufo::test::FloatToBf16Bits(
        0.007F * std::cos(static_cast<float>(i) * 0.0023F));
    h_beta[i] = gufo::test::FloatToBf16Bits(
        0.006F * std::sin(static_cast<float>(i) * 0.0029F));
  }

  float *d_x = nullptr, *d_w = nullptr, *d_normed = nullptr;
  void *d_qkv = nullptr, *d_gate = nullptr, *d_alpha = nullptr;
  void* d_beta = nullptr;
  float *d_qkv_ref = nullptr, *d_gate_ref = nullptr, *d_alpha_ref = nullptr;
  float* d_beta_ref = nullptr;
  float *d_qkv_fus = nullptr, *d_gate_fus = nullptr, *d_alpha_fus = nullptr;
  float* d_beta_fus = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv, qkv_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(
      hipMalloc(&d_gate, inner_size * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_alpha,
                      time_step_rank * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(
      hipMalloc(&d_beta, time_step_rank * hidden_size * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_qkv_ref, qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_ref, inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_ref, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_ref, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qkv_fus, qkv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_fus, inner_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_alpha_fus, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_beta_fus, time_step_rank * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qkv, h_qkv.data(),
                      qkv_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(),
                      inner_size * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_alpha, h_alpha.data(),
                      time_step_rank * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta, h_beta.data(),
                      time_step_rank * hidden_size * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));

  gufo::hip::LaunchRMSNorm(d_x, d_w, d_normed, hidden_size, eps);
  gufo::hip::LaunchFusedSSMInputProjections(
      d_qkv, gufo::core::GgmlType::kBF16, d_gate, gufo::core::GgmlType::kBF16,
      d_alpha, gufo::core::GgmlType::kBF16, d_beta, gufo::core::GgmlType::kBF16,
      d_normed, d_qkv_ref, d_gate_ref, d_alpha_ref, d_beta_ref, hidden_size,
      qkv_size, inner_size, time_step_rank);
  gufo::hip::LaunchFusedRMSNormSSMInputProjections(
      d_x, d_w, eps, d_qkv, true, d_gate, true, d_alpha, true, d_beta, true,
      d_qkv_fus, d_gate_fus, d_alpha_fus, d_beta_fus, hidden_size, qkv_size,
      inner_size, time_step_rank);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> qkv_ref(qkv_size), gate_ref(inner_size);
  std::vector<float> alpha_ref(time_step_rank), beta_ref(time_step_rank);
  std::vector<float> qkv_fus(qkv_size), gate_fus(inner_size);
  std::vector<float> alpha_fus(time_step_rank), beta_fus(time_step_rank);
  HIP_CHECK(hipMemcpy(qkv_ref.data(), d_qkv_ref, qkv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_ref.data(), d_gate_ref, inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_ref.data(), d_alpha_ref,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_ref.data(), d_beta_ref,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qkv_fus.data(), d_qkv_fus, qkv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gate_fus.data(), d_gate_fus, inner_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(alpha_fus.data(), d_alpha_fus,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(beta_fus.data(), d_beta_fus,
                      time_step_rank * sizeof(float), hipMemcpyDeviceToHost));

  float max_diff = 0.0F;
  for (std::size_t i = 0; i < qkv_size; ++i) {
    max_diff = std::max(max_diff, std::abs(qkv_ref[i] - qkv_fus[i]));
  }
  for (std::size_t i = 0; i < inner_size; ++i) {
    max_diff = std::max(max_diff, std::abs(gate_ref[i] - gate_fus[i]));
  }
  for (std::size_t i = 0; i < time_step_rank; ++i) {
    max_diff = std::max(max_diff, std::abs(alpha_ref[i] - alpha_fus[i]));
    max_diff = std::max(max_diff, std::abs(beta_ref[i] - beta_fus[i]));
  }
  std::cout << "Fused RMSNorm+SSM-input vs unfused max diff: " << max_diff
            << "\n";
  if (max_diff != 0.0F) {
    std::cerr << "Fused RMSNorm+SSM-input mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_qkv_ref));
  HIP_CHECK(hipFree(d_gate_ref));
  HIP_CHECK(hipFree(d_alpha_ref));
  HIP_CHECK(hipFree(d_beta_ref));
  HIP_CHECK(hipFree(d_qkv_fus));
  HIP_CHECK(hipFree(d_gate_fus));
  HIP_CHECK(hipFree(d_alpha_fus));
  HIP_CHECK(hipFree(d_beta_fus));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen SSM ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestBatchedSSMConvEquivalence();
  TestBf16RecurrentMemoryAndSnapshot();
  TestBatchedSSMRecurrenceNormGateEquivalence();
  TestBatchedSSMRowSplitRecurrenceEquivalence(96);
  // Above the launcher's 2048-token crossover, so the two-row prefetching tile
  // runs too.
  TestBatchedSSMRowSplitRecurrenceEquivalence(2080);
  TestFusedRMSNormSSMInputProjectionsEquivalence();
  std::cout << "Qwen ssm ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen ssm ops test.\n";
  return 77;
#endif
}
