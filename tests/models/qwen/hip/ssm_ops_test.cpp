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
  constexpr std::size_t batch = 8;
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

  // Verification and rollback replay must preserve every state bit. Reuse the
  // existing buffers to cover both storage formats, per-row and whole-block
  // dispatch, and omission of outputs that replay never consumes.
  const auto expect_device_equal = [](const void* expected, const void* actual,
                                      std::size_t bytes) {
    std::vector<std::uint8_t> reference(bytes), candidate(bytes);
    HIP_CHECK(
        hipMemcpy(reference.data(), expected, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(candidate.data(), actual, bytes, hipMemcpyDeviceToHost));
    if (reference != candidate) {
      throw std::runtime_error("SSM verification/replay changed stored bits");
    }
  };
  using Storage = gufo::hip::QwenRecurrentStateStorage;
  for (const auto storage : {Storage::kFp32, Storage::kBf16}) {
    const std::size_t state_bytes =
        delta_size * gufo::hip::QwenRecurrentStateElementBytes(storage);
    const void* reference_state = storage == Storage::kFp32
                                      ? static_cast<void*>(d_delta_seq)
                                      : d_delta_bf16;
    const float* reference_output =
        storage == Storage::kFp32 ? d_out_seq : d_out_bf16;
    for (const bool write_output : {true, false}) {
      for (const auto rows_per_launch : {std::size_t{1}, batch}) {
        HIP_CHECK(hipMemset(d_state_batch, 0, qkv_dim * 4 * sizeof(float)));
        HIP_CHECK(hipMemset(d_delta_batch, 0, state_bytes));
        for (std::size_t offset = 0; offset < batch;
             offset += rows_per_launch) {
          gufo::hip::LaunchSSMConvRecurrenceRows(
              d_qkv + offset * qkv_dim, d_w, d_state_batch, d_conv_out_batch,
              d_delta_batch, d_alpha + offset * num_heads,
              d_beta + offset * num_heads, d_ssm_a, d_ssm_dt, d_ssm_norm,
              d_gate + offset * inner_size,
              write_output ? d_out_batch + offset * inner_size : nullptr, 0,
              qkv_dim, num_key_heads, num_heads, key_dim, val_dim,
              static_cast<std::uint32_t>(rows_per_launch), num_heads,
              inner_size, nullptr, {}, storage);
        }
        expect_device_equal(reference_state, d_delta_batch, state_bytes);
        expect_device_equal(d_state_seq, d_state_batch,
                            qkv_dim * 4 * sizeof(float));
        if (write_output) {
          expect_device_equal(reference_output, d_out_batch,
                              batch * inner_size * sizeof(float));
        }
      }
    }
  }
  std::cout << "SSM verification and state-only replay: bit-exact\n";

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
    const std::size_t q8_bytes =
        gufo::hip::QuantizedActivationBytes(batch, inner_size) + 4096;
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
  TestBatchedSSMRowSplitRecurrenceEquivalence(96);
  // Above the launcher's 2048-token crossover, so the two-row prefetching tile
  // runs too.
  TestBatchedSSMRowSplitRecurrenceEquivalence(2080);
  std::cout << "Qwen ssm ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen ssm ops test.\n";
  return 77;
#endif
}
