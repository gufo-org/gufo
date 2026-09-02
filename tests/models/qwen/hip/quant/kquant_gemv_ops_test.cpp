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
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

void TestQ5KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 256;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q5KBlockTest = gufo::quant::block_q5_K;
  static_assert(sizeof(Q5KBlockTest) == 176, "Q5_K block must be 176 bytes");

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

  // Deterministic Q5_K weights: d=0.25, dmin=0.125; scales set so
  // GetQKScaleMin(0..7) = (4,1) via the high-bit encoding; qh full (0xFF) on
  // even rows / zero (0x00) on odd rows; qs nibbles 0x11/0x22/0x33/0x44 across
  // the four 64-element groups.
  std::vector<Q5KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  const std::uint16_t d16 = float_to_half_bits(0.25F);
  const std::uint16_t dmin16 = float_to_half_bits(0.125F);
  const std::uint8_t scales[12] = {0x04, 0x04, 0x04, 0x04, 0x01, 0x01,
                                   0x01, 0x01, 0x14, 0x14, 0x14, 0x14};
  const std::uint8_t nibbles[4] = {0x11, 0x22, 0x33, 0x44};
  for (std::size_t r = 0; r < M * num_blocks; ++r) {
    Q5KBlockTest& blk = h_A[r];
    blk.d = d16;
    blk.dmin = dmin16;
    std::memcpy(blk.scales, scales, sizeof(scales));
    std::memset(blk.qh, (r % 2 == 0) ? 0xFF : 0x00, sizeof(blk.qh));
    for (std::size_t g = 0; g < 4; ++g) {
      std::memset(blk.qs + (g * 32), nibbles[g], 32);
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.1F + (static_cast<float>((i * 37 + 11) % 1000) / 2000.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q5KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q5KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ5_K, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = gufo::quant::DotProductQ5_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float max_rel = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q5_K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
    const float rel =
        std::abs(y_gpu[m] - y_ref[m]) / std::max(1e-6F, std::abs(y_ref[m]));
    max_rel = std::max(max_rel, rel);
  }

  std::cout << "Q5_K GEMV (dequant-dot vs GPU): max_rel=" << max_rel << "\n";
  if (max_rel >= 1e-3F) {
    std::cerr << "Q5_K GEMV dequant-dot mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestQ6KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 256;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q6KBlockTest = gufo::quant::block_q6_K;
  static_assert(sizeof(Q6KBlockTest) == 210, "Q6_K block must be 210 bytes");

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

  // Deterministic Q6_K weights: d=0.3; ql varies per byte, qh=0xFF (every
  // 2-bit field = 3 -> quant = 16 + ql nibble, always positive), scales are
  // positive int8 [7..4]. No cancellation -> robust fp reordering comparison.
  std::vector<Q6KBlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  const std::uint16_t d16 = float_to_half_bits(0.3F);
  for (std::size_t r = 0; r < M * num_blocks; ++r) {
    Q6KBlockTest& blk = h_A[r];
    blk.d = d16;
    for (std::size_t i = 0; i < 128; ++i) {
      blk.ql[i] = static_cast<std::uint8_t>((i * 13 + 5) & 0xFF);
    }
    std::memset(blk.qh, 0xFF, sizeof(blk.qh));
    for (std::size_t i = 0; i < 16; ++i) {
      blk.scales[i] = static_cast<std::int8_t>(7 - (i % 4));
    }
  }
  for (std::size_t i = 0; i < K; ++i) {
    h_x[i] = 0.1F + (static_cast<float>((i * 37 + 11) % 1000) / 2000.0F);
  }

  void* d_A = nullptr;
  float *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * num_blocks * sizeof(Q6KBlockTest)));
  HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * num_blocks * sizeof(Q6KBlockTest),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchQ8KBlockGEMV(d_A, gufo::core::GgmlType::kQ6_K, d_x, d_y, M,
                                K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(
      hipMemcpy(y_gpu.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = gufo::quant::DotProductQ6_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float max_rel = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q6_K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
    const float rel =
        std::abs(y_gpu[m] - y_ref[m]) / std::max(1e-6F, std::abs(y_ref[m]));
    max_rel = std::max(max_rel, rel);
  }

  std::cout << "Q6_K GEMV (dequant-dot vs GPU): max_rel=" << max_rel << "\n";
  if (max_rel >= 1e-3F) {
    std::cerr << "Q6_K GEMV dequant-dot mismatch\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

// The single-token decode computes an SSM layer's four input projections with
// one fused kernel, while the speculative verification chunk computes them as
// separate batched projections. A drafted token is only acceptable if the
// verifier reproduces what unspeculated decode would have emitted, so the two
// have to agree for every weight format an SSM layer can carry. Q8_0 and BF16
// have hand-written paths inside the fused kernel; every other quantized
// format falls through to `QuantWarpBlockDot`, and that is the half this test
// exists to pin. The Unsloth Q8 shard carries Q6_K in `attn_qkv` on some
// layers and Q8_0 on others, so both reach this code on one model.
template<typename FillWeight>
void TestFusedSSMInputProjectionMatchesGemv(gufo::core::GgmlType type,
                                            const char* label,
                                            FillWeight fill) {
  // Production dimensions. The shard's SSM layers are hidden 5120 with a
  // 10240-row qkv, 6144-row gate and 48-row alpha/beta, and the disagreement
  // this test exists to catch does not reproduce at toy sizes.
  constexpr std::size_t kHidden = 5120;
  constexpr std::size_t kQkvSize = 10240;
  constexpr std::size_t kInnerSize = 6144;
  constexpr std::size_t kTimeStepRank = 48;

  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, kHidden);
  gufo::test::Expect(row_bytes != 0, "row bytes for the tested format");

  std::uint32_t seed = 991U;
  auto next = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };

  const auto make_weight = [&](std::size_t rows) {
    std::vector<std::uint8_t> bytes(rows * row_bytes);
    fill(bytes, rows, row_bytes, next);
    return bytes;
  };

  std::vector<std::uint8_t> qkv = make_weight(kQkvSize);
  std::vector<std::uint8_t> gate = make_weight(kInnerSize);
  std::vector<std::uint8_t> alpha = make_weight(kTimeStepRank);
  std::vector<std::uint8_t> beta = make_weight(kTimeStepRank);
  std::vector<float> x(kHidden);
  for (auto& value : x) {
    value = (static_cast<float>(next() & 0xFFFFU) / 65535.0F) - 0.5F;
  }

  const auto upload = [](const std::vector<std::uint8_t>& host) {
    void* device = nullptr;
    HIP_CHECK(hipMalloc(&device, host.size()));
    HIP_CHECK(
        hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice));
    return device;
  };
  void* d_qkv = upload(qkv);
  void* d_gate = upload(gate);
  void* d_alpha = upload(alpha);
  void* d_beta = upload(beta);
  float* d_x = nullptr;
  HIP_CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const std::size_t total = kQkvSize + kInnerSize + (2 * kTimeStepRank);
  float* d_fused = nullptr;
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_fused, total * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_split, total * sizeof(float)));
  HIP_CHECK(hipMemset(d_fused, 0, total * sizeof(float)));
  HIP_CHECK(hipMemset(d_split, 0, total * sizeof(float)));

  float* const fused_qkv = d_fused;
  float* const fused_gate = fused_qkv + kQkvSize;
  float* const fused_alpha = fused_gate + kInnerSize;
  float* const fused_beta = fused_alpha + kTimeStepRank;
  gufo::hip::LaunchFusedSSMInputProjections(
      d_qkv, type, d_gate, type, d_alpha, type, d_beta, type, d_x, fused_qkv,
      fused_gate, fused_alpha, fused_beta, kHidden, kQkvSize, kInnerSize,
      kTimeStepRank, nullptr);

  float* const split_qkv = d_split;
  float* const split_gate = split_qkv + kQkvSize;
  float* const split_alpha = split_gate + kInnerSize;
  float* const split_beta = split_alpha + kTimeStepRank;
  const auto mode = gufo::models::qwen::QwenGemmMode::kHipDecode;
  gufo::hip::LaunchGEMV(d_qkv, type, d_x, split_qkv, kQkvSize, kHidden, nullptr,
                        mode);
  gufo::hip::LaunchGEMV(d_gate, type, d_x, split_gate, kInnerSize, kHidden,
                        nullptr, mode);
  gufo::hip::LaunchGEMV(d_alpha, type, d_x, split_alpha, kTimeStepRank, kHidden,
                        nullptr, mode);
  gufo::hip::LaunchGEMV(d_beta, type, d_x, split_beta, kTimeStepRank, kHidden,
                        nullptr, mode);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> fused(total);
  std::vector<float> split(total);
  HIP_CHECK(hipMemcpy(fused.data(), d_fused, total * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(split.data(), d_split, total * sizeof(float),
                      hipMemcpyDeviceToHost));

  // The verification chunk does not call LaunchGEMV: it calls the batched
  // projection, which is required to reproduce the decode GEMV bit for bit.
  // Compare that route too, at the narrowest batch it can dispatch.
  float* d_batched = nullptr;
  HIP_CHECK(hipMalloc(&d_batched, 2 * kQkvSize * sizeof(float)));
  HIP_CHECK(hipMemset(d_batched, 0, 2 * kQkvSize * sizeof(float)));
  float* d_x2 = nullptr;
  HIP_CHECK(hipMalloc(&d_x2, 2 * kHidden * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_x2, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x2 + kHidden, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  gufo::hip::LaunchBatchedQuantGEMMFp32(type, d_qkv, d_x2, d_batched, 2,
                                        kQkvSize, kHidden, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> batched(kQkvSize);
  HIP_CHECK(hipMemcpy(batched.data(), d_batched, kQkvSize * sizeof(float),
                      hipMemcpyDeviceToHost));
  std::size_t batched_mismatches = 0;
  for (std::size_t index = 0; index < kQkvSize; ++index) {
    if (batched[index] != fused[index]) {
      ++batched_mismatches;
    }
  }
  std::cout << "[ " << (batched_mismatches == 0 ? "OK" : "FAIL")
            << " ] batched projection vs fused SSM input " << label << ": "
            << batched_mismatches << " of " << kQkvSize << " rows differ"
            << std::endl;
  HIP_CHECK(hipFree(d_batched));
  HIP_CHECK(hipFree(d_x2));

  std::size_t mismatches = 0;
  float worst = 0.0F;
  bool finite = true;
  for (std::size_t index = 0; index < total; ++index) {
    finite =
        finite && std::isfinite(fused[index]) && std::isfinite(split[index]);
    if (fused[index] != split[index]) {
      ++mismatches;
      worst = std::max(worst, std::abs(fused[index] - split[index]));
    }
  }
  std::cout << "[ " << (mismatches == 0 ? "OK" : "FAIL")
            << " ] fused SSM input projections vs decode GEMV " << label << ": "
            << mismatches << " of " << total << " rows differ, max abs "
            << worst << std::endl;
  gufo::test::Expect(finite, "fused and split SSM projections must be finite");
  gufo::test::Expect(mismatches == 0,
                     "fused SSM input projections must reproduce the decode "
                     "GEMV bit for bit");

  HIP_CHECK(hipFree(d_qkv));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_fused));
  HIP_CHECK(hipFree(d_split));
}

/// Half-precision bits for a finite positive scale, so generated blocks decode
/// to real numbers instead of the NaNs random bytes would produce. A power of
/// two would be worse than useless here: it makes every product exactly
/// representable, so any summation order gives the same answer and a test built
/// on it cannot see two kernels accumulate K in different orders. These scales
/// have a full mantissa and vary per block, which is what real weights look
/// like.
constexpr std::uint16_t kTestScaleHalves[4] = {0x3555U, 0x2E67U, 0x39ABU,
                                               0x3123U};

[[nodiscard]] std::uint16_t TestScaleHalf(std::size_t block) noexcept {
  return kTestScaleHalves[block % 4];
}

void FillQ8_0(std::vector<std::uint8_t>& bytes, std::size_t rows,
              std::size_t row_bytes, auto&& next) {
  for (std::size_t row = 0; row < rows; ++row) {
    auto* blocks = reinterpret_cast<gufo::quant::block_q8_0*>(
        bytes.data() + (row * row_bytes));
    const std::size_t count = row_bytes / sizeof(gufo::quant::block_q8_0);
    for (std::size_t b = 0; b < count; ++b) {
      blocks[b].d = TestScaleHalf(b);
      for (auto& value : blocks[b].qs) {
        value = static_cast<std::int8_t>(static_cast<int>(next() % 255U) - 127);
      }
    }
  }
}

void FillQ6_K(std::vector<std::uint8_t>& bytes, std::size_t rows,
              std::size_t row_bytes, auto&& next) {
  for (std::size_t row = 0; row < rows; ++row) {
    auto* blocks = reinterpret_cast<gufo::quant::block_q6_K*>(
        bytes.data() + (row * row_bytes));
    const std::size_t count = row_bytes / sizeof(gufo::quant::block_q6_K);
    for (std::size_t b = 0; b < count; ++b) {
      blocks[b].d = TestScaleHalf(b);
      for (auto& value : blocks[b].ql) {
        value = static_cast<std::uint8_t>(next() & 0xFFU);
      }
      for (auto& value : blocks[b].qh) {
        value = static_cast<std::uint8_t>(next() & 0xFFU);
      }
      for (auto& value : blocks[b].scales) {
        value = static_cast<std::int8_t>((next() % 15U) + 1U);
      }
    }
  }
}

/// The verification chunk computes every FFN down-projection with the batched
/// K-quant kernel while unspeculated decode computes it with the block GEMV,
/// and a drafted token is only acceptable if the two agree bit for bit. The
/// Unsloth Q8 shard carries `ffn_down` as Q6_K on some layers, and that tensor
/// is 5120x17408 -- a K far longer than any other projection in the model, and
/// the one shape where the two routes were never compared.
void TestBatchedFfnDownMatchesGemv(gufo::core::GgmlType type, const char* label,
                                   std::size_t rows, std::size_t columns,
                                   auto&& fill) {
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, columns);
  gufo::test::Expect(row_bytes != 0, "row bytes for the tested format");

  std::uint32_t seed = 4242U;
  auto next = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };

  std::vector<std::uint8_t> weights(rows * row_bytes);
  fill(weights, rows, row_bytes, next);
  // Real post-SwiGLU activations span several orders of magnitude, and a
  // uniform vector hides order-dependent rounding: partial sums stay the same
  // size so any accumulation order rounds identically. Spread the exponents so
  // that reordering actually changes the result.
  std::vector<float> x(columns);
  for (std::size_t index = 0; index < columns; ++index) {
    const float unit = (static_cast<float>(next() & 0xFFFFU) / 65535.0F) - 0.5F;
    const int exponent = static_cast<int>(next() % 12U) - 6;
    x[index] = std::ldexp(unit, exponent);
  }

  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_reference = nullptr;
  float* d_batched = nullptr;
  HIP_CHECK(hipMalloc(&d_w, weights.size()));
  HIP_CHECK(hipMalloc(&d_x, columns * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_reference, rows * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_batched, 8 * rows * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_w, weights.data(), weights.size(), hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_x, x.data(), columns * sizeof(float), hipMemcpyHostToDevice));

  gufo::hip::LaunchGEMV(d_w, type, d_x, d_reference, rows, columns, nullptr,
                        gufo::models::qwen::QwenGemmMode::kHipDecode);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> reference(rows);
  HIP_CHECK(hipMemcpy(reference.data(), d_reference, rows * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Every batch row gets its own activation vector. Filling the batch with one
  // repeated vector would hide any cross-row contamination in the staged
  // kernel, which is precisely the class of bug a batched route can have and a
  // per-row route cannot.
  std::vector<std::vector<float>> batch_x(8);
  std::vector<std::vector<float>> batch_reference(8);
  float* d_batched_x = nullptr;
  HIP_CHECK(hipMalloc(&d_batched_x, 8 * columns * sizeof(float)));
  for (std::size_t row = 0; row < 8; ++row) {
    batch_x[row].resize(columns);
    for (std::size_t index = 0; index < columns; ++index) {
      const float unit =
          (static_cast<float>(next() & 0xFFFFU) / 65535.0F) - 0.5F;
      const int exponent = static_cast<int>(next() % 12U) - 6;
      batch_x[row][index] = std::ldexp(unit, exponent);
    }
    HIP_CHECK(hipMemcpy(d_batched_x + (row * columns), batch_x[row].data(),
                        columns * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, batch_x[row].data(), columns * sizeof(float),
                        hipMemcpyHostToDevice));
    gufo::hip::LaunchGEMV(d_w, type, d_x, d_reference, rows, columns, nullptr,
                          gufo::models::qwen::QwenGemmMode::kHipDecode);
    HIP_CHECK(hipDeviceSynchronize());
    batch_reference[row].resize(rows);
    HIP_CHECK(hipMemcpy(batch_reference[row].data(), d_reference,
                        rows * sizeof(float), hipMemcpyDeviceToHost));
  }

  for (std::size_t batch = 2; batch <= 8; ++batch) {
    HIP_CHECK(hipMemset(d_batched, 0, 8 * rows * sizeof(float)));
    gufo::hip::LaunchBatchedQuantGEMMFp32(type, d_w, d_batched_x, d_batched,
                                          batch, rows, columns, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> batched(batch * rows);
    HIP_CHECK(hipMemcpy(batched.data(), d_batched,
                        batched.size() * sizeof(float), hipMemcpyDeviceToHost));
    std::size_t mismatches = 0;
    for (std::size_t token = 0; token < batch; ++token) {
      for (std::size_t row = 0; row < rows; ++row) {
        if (batched[(token * rows) + row] != batch_reference[token][row]) {
          ++mismatches;
        }
      }
    }
    std::cout << "[ " << (mismatches == 0 ? "OK" : "FAIL") << " ] " << label
              << " " << rows << "x" << columns << " batch " << batch << ": "
              << mismatches << " of " << (batch * rows) << " differ"
              << std::endl;
    gufo::test::Expect(mismatches == 0,
                       "batched projection must reproduce the decode GEMV");
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_reference));
  HIP_CHECK(hipFree(d_batched));
  HIP_CHECK(hipFree(d_batched_x));
}

/// Unspeculated decode computes an FFN's gate and up projections inside one
/// fused SwiGLU GEMV; the speculative verification chunk computes them as two
/// batched projections followed by a standalone SwiGLU. A drafted token is
/// only acceptable if the verifier reproduces decode exactly, so those two
/// spellings have to agree for every weight format a layer can carry -- and
/// this seam had no test. The Unsloth Q8 shard mixes formats per layer, so one
/// model reaches both the K-quant and the Q8_0 spelling of the fused kernel.
void TestFusedSwiGLUMatchesSplitProjections(gufo::core::GgmlType type,
                                            const char* label,
                                            std::size_t intermediate,
                                            std::size_t hidden, auto&& fill) {
  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, hidden);
  gufo::test::Expect(row_bytes != 0, "row bytes for the tested format");

  std::uint32_t seed = 20250902U;
  auto next = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };

  std::vector<std::uint8_t> gate(intermediate * row_bytes);
  std::vector<std::uint8_t> up(intermediate * row_bytes);
  fill(gate, intermediate, row_bytes, next);
  fill(up, intermediate, row_bytes, next);
  std::vector<float> x(hidden);
  for (auto& value : x) {
    value = (static_cast<float>(next() & 0xFFFFU) / 65535.0F) - 0.5F;
  }

  const auto upload = [](const std::vector<std::uint8_t>& host) {
    void* device = nullptr;
    HIP_CHECK(hipMalloc(&device, host.size()));
    HIP_CHECK(
        hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice));
    return device;
  };
  void* d_gate = upload(gate);
  void* d_up = upload(up);
  float* d_x = nullptr;
  HIP_CHECK(hipMalloc(&d_x, hidden * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_x, x.data(), hidden * sizeof(float), hipMemcpyHostToDevice));

  float* d_fused = nullptr;
  float* d_gate_out = nullptr;
  float* d_up_out = nullptr;
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_fused, intermediate * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate_out, intermediate * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up_out, intermediate * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_split, intermediate * sizeof(float)));

  gufo::hip::LaunchFusedSwiGLUGEMV(d_gate, type, d_up, type, d_x, d_fused,
                                   intermediate, hidden, nullptr);

  // Batch two, not one: the verification chunk never dispatches a single row
  // (`ForwardTokenBatch` rejects it) and the Q8_0 exact shared route is gated
  // on `batch > 1`, so a batch-one comparison would exercise a kernel the
  // verifier never reaches.
  float* d_x2 = nullptr;
  HIP_CHECK(hipMalloc(&d_x2, 2 * hidden * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_x2, x.data(), hidden * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_x2 + hidden, x.data(), hidden * sizeof(float),
                      hipMemcpyHostToDevice));
  float* d_gate2 = nullptr;
  float* d_up2 = nullptr;
  HIP_CHECK(hipMalloc(&d_gate2, 2 * intermediate * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_up2, 2 * intermediate * sizeof(float)));
  gufo::hip::LaunchBatchedQuantGEMMFp32(type, d_gate, d_x2, d_gate2, 2,
                                        intermediate, hidden, nullptr);
  gufo::hip::LaunchBatchedQuantGEMMFp32(type, d_up, d_x2, d_up2, 2,
                                        intermediate, hidden, nullptr);
  HIP_CHECK(hipMemcpyAsync(d_gate_out, d_gate2, intermediate * sizeof(float),
                           hipMemcpyDeviceToDevice, nullptr));
  HIP_CHECK(hipMemcpyAsync(d_up_out, d_up2, intermediate * sizeof(float),
                           hipMemcpyDeviceToDevice, nullptr));
  gufo::hip::LaunchBatchedSwiGLUActivation(d_gate_out, d_up_out, d_split,
                                           nullptr, intermediate, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> fused(intermediate);
  std::vector<float> split(intermediate);
  HIP_CHECK(hipMemcpy(fused.data(), d_fused, intermediate * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(split.data(), d_split, intermediate * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::size_t mismatches = 0;
  float worst = 0.0F;
  for (std::size_t index = 0; index < intermediate; ++index) {
    if (fused[index] != split[index]) {
      ++mismatches;
      worst = std::max(worst, std::abs(fused[index] - split[index]));
    }
  }
  // When they disagree, a double-precision host oracle decides which spelling
  // is the accurate one, because the fix direction depends on it.
  double fused_error = 0.0;
  double split_error = 0.0;
  if (mismatches != 0 && type == gufo::core::GgmlType::kQ8_0) {
    constexpr std::size_t kOracleRows = 64;
    for (std::size_t row = 0; row < kOracleRows && row < intermediate; ++row) {
      const auto dot = [&](const std::vector<std::uint8_t>& w) {
        const auto* blocks = reinterpret_cast<const gufo::quant::block_q8_0*>(
            w.data() + (row * row_bytes));
        const std::size_t count = hidden / 32;
        double total = 0.0;
        for (std::size_t b = 0; b < count; ++b) {
          const double d = gufo::quant::Fp16ToFloat(blocks[b].d);
          for (std::size_t i = 0; i < 32; ++i) {
            total += d * static_cast<double>(blocks[b].qs[i]) *
                     static_cast<double>(x[(b * 32) + i]);
          }
        }
        return total;
      };
      const double g = dot(gate);
      const double u = dot(up);
      const double reference = (g / (1.0 + std::exp(-g))) * u;
      fused_error = std::max(
          fused_error, std::abs(static_cast<double>(fused[row]) - reference));
      split_error = std::max(
          split_error, std::abs(static_cast<double>(split[row]) - reference));
    }
    std::cout << "        oracle max abs error: fused " << fused_error
              << ", split " << split_error << std::endl;
  }

  std::cout << "[ " << (mismatches == 0 ? "OK" : "FAIL")
            << " ] fused SwiGLU vs split projections " << label << " "
            << intermediate << "x" << hidden << ": " << mismatches << " of "
            << intermediate << " differ, max abs " << worst << std::endl;
  gufo::test::Expect(mismatches == 0,
                     "the fused SwiGLU GEMV and the split projections must "
                     "agree bit for bit");

  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_up));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_fused));
  HIP_CHECK(hipFree(d_gate_out));
  HIP_CHECK(hipFree(d_up_out));
  HIP_CHECK(hipFree(d_split));
  HIP_CHECK(hipFree(d_x2));
  HIP_CHECK(hipFree(d_gate2));
  HIP_CHECK(hipFree(d_up2));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen K-quant GEMV ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestQ5KBlockGEMVEquivalence();
  TestQ6KBlockGEMVEquivalence();
  TestFusedSSMInputProjectionMatchesGemv(
      gufo::core::GgmlType::kQ8_0, "Q8_0",
      [](std::vector<std::uint8_t>& bytes, std::size_t rows,
         std::size_t row_bytes,
         auto&& next) { FillQ8_0(bytes, rows, row_bytes, next); });
  TestFusedSSMInputProjectionMatchesGemv(
      gufo::core::GgmlType::kQ6_K, "Q6_K",
      [](std::vector<std::uint8_t>& bytes, std::size_t rows,
         std::size_t row_bytes,
         auto&& next) { FillQ6_K(bytes, rows, row_bytes, next); });
  TestBatchedFfnDownMatchesGemv(
      gufo::core::GgmlType::kQ6_K, "Q6_K ffn_down", 5120, 17408,
      [](std::vector<std::uint8_t>& bytes, std::size_t rows,
         std::size_t row_bytes,
         auto&& next) { FillQ6_K(bytes, rows, row_bytes, next); });
  TestFusedSwiGLUMatchesSplitProjections(
      gufo::core::GgmlType::kQ8_0, "Q8_0", 17408, 5120,
      [](std::vector<std::uint8_t>& bytes, std::size_t rows,
         std::size_t row_bytes,
         auto&& next) { FillQ8_0(bytes, rows, row_bytes, next); });
  TestFusedSwiGLUMatchesSplitProjections(
      gufo::core::GgmlType::kQ6_K, "Q6_K", 17408, 5120,
      [](std::vector<std::uint8_t>& bytes, std::size_t rows,
         std::size_t row_bytes,
         auto&& next) { FillQ6_K(bytes, rows, row_bytes, next); });
  std::cout << "Qwen K-quant GEMV ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen K-quant GEMV ops test.\n";
  return 77;
#endif
}
