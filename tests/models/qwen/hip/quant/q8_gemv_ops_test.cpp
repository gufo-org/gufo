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
#include "tests/models/qwen/hip/support/device.hpp"

void TestQ8KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 512;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8KBlockTest = gufo::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };

  // Synthetic Q8_K weights: qs in [-127,127], d in ~[-2,2]; x is fp32 in
  // [-1,1].
  std::vector<Q8KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = rnd_float(-2.0F, 2.0F);
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
    for (std::size_t i = 0; i < 16; ++i) {
      blk.bsums[i] = 0;
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ8_K, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Primary check: CPU DotProductQ8_K (exact fp dequant dot of weights against
  // x).
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = gufo::quant::DotProductQ8_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_diff = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_diff =
        abs_diff / std::max(1e-4F, std::abs(y_dequant_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_diff);
    primary_max_abs = std::max(primary_max_abs, abs_diff);
  }

  std::cout << "Q8K GEMV vs CPU oracle: max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8K GEMV mismatch against CPU oracle\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestQ8KSmallBatchFp32GEMMEquivalence() {
  constexpr std::size_t kRows = 16;
  constexpr std::size_t kBlockSize = 256;
  constexpr std::size_t kColumns = 512;
  constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
  constexpr std::size_t kMaximumBatch = 8;

  using Q8KBlockTest = gufo::quant::block_q8_K;
  std::uint32_t seed = 98431U;
  auto random = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto random_float = [&random](float low, float high) -> float {
    const float unit = static_cast<float>(random() & 0xFFFFU) / 65535.0F;
    return low + unit * (high - low);
  };

  std::vector<Q8KBlockTest> weights(kRows * kBlocksPerRow);
  std::vector<float> inputs(kMaximumBatch * kColumns);
  for (auto& block : weights) {
    block.d = random_float(-0.25F, 0.25F);
    for (auto& value : block.qs) {
      value = static_cast<std::int8_t>(static_cast<int>(random() % 255U) - 127);
    }
    std::fill(std::begin(block.bsums), std::end(block.bsums), 0);
  }
  for (auto& input : inputs) {
    input = random_float(-1.0F, 1.0F);
  }

  void* device_weights = nullptr;
  float* device_inputs = nullptr;
  float* device_reference = nullptr;
  float* device_batched = nullptr;
  HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(&device_inputs, inputs.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&device_reference, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_batched, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_weights, weights.data(),
                      weights.size() * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_inputs, inputs.data(),
                      inputs.size() * sizeof(float), hipMemcpyHostToDevice));

  for (const std::size_t batch :
       {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5},
        std::size_t{6}, std::size_t{7}, std::size_t{8}}) {
    for (std::size_t token = 0; token < batch; ++token) {
      gufo::hip::LaunchQ8KBlockGEMV(device_weights, gufo::core::GgmlType::kQ8_K,
                                    device_inputs + (token * kColumns),
                                    device_reference + (token * kRows), kRows,
                                    kColumns, nullptr);
    }
    gufo::hip::LaunchBatchedQuantGEMMFp32(
        gufo::core::GgmlType::kQ8_K, device_weights, device_inputs,
        device_batched, batch, kRows, kColumns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> reference(batch * kRows);
    std::vector<float> batched(batch * kRows);
    HIP_CHECK(hipMemcpy(reference.data(), device_reference,
                        reference.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched.data(), device_batched,
                        batched.size() * sizeof(float), hipMemcpyDeviceToHost));

    for (std::size_t index = 0; index < reference.size(); ++index) {
      if (reference[index] != batched[index]) {
        std::cerr << "shared Q8_K small-batch mismatch at batch " << batch
                  << " index " << index << ": expected " << reference[index]
                  << ", got " << batched[index] << "\n";
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(device_weights));
  HIP_CHECK(hipFree(device_inputs));
  HIP_CHECK(hipFree(device_reference));
  HIP_CHECK(hipFree(device_batched));
}

// `opt-c206-q8-small-batch` selects among physical kernel widths by shape, so a
// 16x512 toy matrix does not exercise the routes a real projection reaches. The
// shapes below are the ones the Q8 shard actually dispatches: attention output
// (5120x5120), FFN gate/up (17408x5120) and the tied LM head (248320x5120),
// truncated in rows where the full tensor would not fit but keeping each
// distinct row-to-K ratio that the selector branches on.
void TestQ8_0SmallBatchFp32GEMMEquivalence(std::size_t kRows,
                                           std::size_t kColumns) {
  constexpr std::size_t kBlockSize = 32;
  const std::size_t kBlocksPerRow = kColumns / kBlockSize;
  constexpr std::size_t kMaximumBatch = 8;

  using Q8_0BlockTest = gufo::quant::block_q8_0;
  std::uint32_t seed = 31789U;
  auto random = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto random_float = [&random](float low, float high) -> float {
    const float unit = static_cast<float>(random() & 0xFFFFU) / 65535.0F;
    return low + unit * (high - low);
  };
  auto float_to_half_bits = [](float value) -> std::uint16_t {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    std::int32_t exponent =
        static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    std::uint32_t mantissa = (bits >> 13) & 0x3FFU;
    if (exponent <= 0) {
      if (exponent < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mantissa |= 0x400U;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
      return static_cast<std::uint16_t>(sign | (mantissa >> shift));
    }
    if (exponent >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00U |
                                        (mantissa != 0 ? 0x200U : 0U));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10) | mantissa);
  };

  std::vector<Q8_0BlockTest> weights(kRows * kBlocksPerRow);
  std::vector<float> inputs(kMaximumBatch * kColumns);
  for (auto& block : weights) {
    block.d = float_to_half_bits(random_float(-0.25F, 0.25F));
    for (auto& value : block.qs) {
      value = static_cast<std::int8_t>(static_cast<int>(random() % 255U) - 127);
    }
  }
  for (auto& input : inputs) {
    input = random_float(-1.0F, 1.0F);
  }

  void* device_weights = nullptr;
  float* device_inputs = nullptr;
  float* device_reference = nullptr;
  float* device_batched = nullptr;
  HIP_CHECK(hipMalloc(&device_weights, weights.size() * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(&device_inputs, inputs.size() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&device_reference, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_batched, kMaximumBatch * kRows * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_weights, weights.data(),
                      weights.size() * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_inputs, inputs.data(),
                      inputs.size() * sizeof(float), hipMemcpyHostToDevice));

  // Every width the speculative verifier can dispatch, not a sample of them.
  // DFlash-2 runs a fixed draft width of 7, so verification batches are
  // routinely 8 but drop to 7 and below whenever a block is truncated or the
  // controller narrows, and a width with no exact route silently stops
  // reproducing the decode GEMV. Width 1 is excluded because
  // `ForwardTokenBatch` rejects a batch below two, so it is not a verification
  // width at all.
  for (const std::size_t batch :
       {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5},
        std::size_t{6}, std::size_t{7}, std::size_t{8}}) {
    for (std::size_t token = 0; token < batch; ++token) {
      gufo::hip::LaunchQ8KBlockGEMV(device_weights, gufo::core::GgmlType::kQ8_0,
                                    device_inputs + (token * kColumns),
                                    device_reference + (token * kRows), kRows,
                                    kColumns, nullptr);
    }
    gufo::hip::LaunchBatchedQuantGEMMFp32(
        gufo::core::GgmlType::kQ8_0, device_weights, device_inputs,
        device_batched, batch, kRows, kColumns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> reference(batch * kRows);
    std::vector<float> batched(batch * kRows);
    HIP_CHECK(hipMemcpy(reference.data(), device_reference,
                        reference.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(batched.data(), device_batched,
                        batched.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (std::size_t index = 0; index < reference.size(); ++index) {
      if (reference[index] != batched[index]) {
        std::cerr << "exact Q8_0 small-batch mismatch at " << kRows << "x"
                  << kColumns << " batch " << batch << " index " << index
                  << ": expected " << reference[index] << ", got "
                  << batched[index] << "\n";
        std::abort();
      }
    }
  }

  HIP_CHECK(hipFree(device_weights));
  HIP_CHECK(hipFree(device_inputs));
  HIP_CHECK(hipFree(device_reference));
  HIP_CHECK(hipFree(device_batched));
}

void TestQ8_0BlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 32;
  constexpr std::size_t K = 512;  // K % 32 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8_0BlockTest = gufo::quant::block_q8_0;
  static_assert(sizeof(Q8_0BlockTest) == 34, "Q8_0 block must be 34 bytes");

  // Deterministic pseudo-random generator (same values on every run).
  std::uint32_t seed = 12345U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };
  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) {
      if (exp < -10) {
        return static_cast<std::uint16_t>(sign);
      }
      mant |= 0x400u;
      const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
      return static_cast<std::uint16_t>(sign | (mant >> shift));
    }
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
  };

  // Synthetic Q8_0 weights: qs in [-127,127], d (as fp16) in ~[-2,2]; x is fp32
  // in [-1,1].
  std::vector<Q8_0BlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = float_to_half_bits(rnd_float(-2.0F, 2.0F));
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = rnd_float(-1.0F, 1.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ8_0, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8_0 GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Primary check: CPU DotProductQ8_0 (exact fp dequant dot of weights against
  // x).
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = gufo::quant::DotProductQ8_0(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_diff = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_diff =
        abs_diff / std::max(1e-4F, std::abs(y_dequant_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_diff);
    primary_max_abs = std::max(primary_max_abs, abs_diff);
  }

  std::cout << "Q8_0 GEMV vs CPU oracle: max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8_0 GEMV mismatch against CPU oracle\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen Q8 GEMV ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestQ8KBlockGEMVEquivalence();
  TestQ8KSmallBatchFp32GEMMEquivalence();
  TestQ8_0SmallBatchFp32GEMMEquivalence(16, 512);
  TestQ8_0SmallBatchFp32GEMMEquivalence(5120, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(17408, 5120);
  TestQ8_0SmallBatchFp32GEMMEquivalence(4096, 17408);
  TestQ8_0BlockGEMVEquivalence();
  std::cout << "Qwen Q8 GEMV ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen Q8 GEMV ops test.\n";
  return 77;
#endif
}
