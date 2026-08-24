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

void TestQ5KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 256;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q5KBlockTest = strix::quant::block_q5_K;
  static_assert(sizeof(Q5KBlockTest) == 176, "Q5_K block must be 176 bytes");

  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
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
      return static_cast<std::uint16_t>(sign | 0x7C00u |
                                        (mant ? 0x200u : 0u));
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
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ5_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = strix::quant::DotProductQ5_K(
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

  using Q6KBlockTest = strix::quant::block_q6_K;
  static_assert(sizeof(Q6KBlockTest) == 210, "Q6_K block must be 210 bytes");

  auto float_to_half_bits = [](float f) -> std::uint16_t {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp =
        static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
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
      return static_cast<std::uint16_t>(sign | 0x7C00u |
                                        (mant ? 0x200u : 0u));
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
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ6_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // CPU oracle: exact fp dequant + dot (no activation quantization).
  std::vector<float> y_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_ref[m] = strix::quant::DotProductQ6_K(
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

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = strix::test::GateHipDevice(
      strix::test::HipDeviceRequirement::kOptional,
      "Qwen K-quant GEMV ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  TestQ5KBlockGEMVEquivalence();
  TestQ6KBlockGEMVEquivalence();
  std::cout << "Qwen K-quant GEMV ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen K-quant GEMV ops test.\n";
  return 0;
#endif
}
