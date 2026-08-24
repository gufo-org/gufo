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

void TestQ8KBlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 256;
  constexpr std::size_t K = 512;  // K % 256 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8KBlockTest = strix::quant::block_q8_K;
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
      blk.qs[i] =
          static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
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
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ8_K, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8K GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Check 1 (PRIMARY, tight): CPU reference replicating the integer-Q8
  // arithmetic (max|x| -> scale, xq clamp, int32 MAC, end scale).
  std::vector<float> y_q8_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    float sumf = 0.0F;
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8KBlockTest& wblk = h_A[m * num_blocks + b];
      const float d_w = wblk.d;
      const float* xb = h_x.data() + b * QK;
      float local_max = 0.0F;
      for (std::size_t i = 0; i < QK; ++i) {
        local_max = std::max(local_max, std::abs(xb[i]));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
      int acc = 0;
      for (std::size_t i = 0; i < QK; ++i) {
        const float xv = (scale > 0.0F) ? (xb[i] / scale) : 0.0F;
        int xq = static_cast<int>(std::round(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[i]) * xq;
      }
      sumf += d_w * scale * static_cast<float>(acc);
    }
    y_q8_ref[m] = sumf;
  }

  // Check 2 (SANITY, loose): CPU DotProductQ8_K (exact fp dequant dot of the
  // weights against the raw x, no activation quant) vs GPU. The difference is
  // the activation-quantization error, expected ~1%, not a sign/index bug.
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = strix::quant::DotProductQ8_K(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  float sanity_max_rel = 0.0F, sanity_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_q8 = std::abs(y_gpu[m] - y_q8_ref[m]);
    const float rel_q8 = abs_q8 / std::max(1e-6F, std::abs(y_q8_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_q8);
    primary_max_abs = std::max(primary_max_abs, abs_q8);

    const float abs_san = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_san =
        abs_san / std::max(1e-3F, std::abs(y_dequant_ref[m]));
    sanity_max_rel = std::max(sanity_max_rel, rel_san);
    sanity_max_abs = std::max(sanity_max_abs, abs_san);
  }

  std::cout << "Q8K GEMV PRIMARY (int-Q8 vs GPU): max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  std::cout << "Q8K GEMV SANITY (dequant vs GPU): max_rel=" << sanity_max_rel
            << " max_abs=" << sanity_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8K GEMV integer-dot mismatch (primary)\n";
    std::abort();
  }
  if (sanity_max_rel >= 2e-2F) {
    std::cerr
        << "Q8K GEMV dequant delta out of the expected quantization band\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

void TestQ8_0BlockGEMVEquivalence() {
  constexpr std::size_t M = 4;
  constexpr std::size_t QK = 32;
  constexpr std::size_t K = 512;  // K % 32 == 0 required by the kernel
  constexpr std::size_t num_blocks = K / QK;

  using Q8_0BlockTest = strix::quant::block_q8_0;
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
      return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    return static_cast<std::uint16_t>(sign |
                                      (static_cast<std::uint32_t>(exp) << 10) |
                                      mant);
  };
  auto half_to_float = [](std::uint16_t h) -> float {
    const std::uint32_t sign = (h >> 15) & 1u;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    const std::uint32_t mant = h & 0x3FFu;
    std::uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign << 31;
      } else {
        std::uint32_t e = 0;
        std::uint32_t m = mant;
        while ((m & 0x400u) == 0) {
          m <<= 1;
          ++e;
        }
        m &= 0x3FFu;
        f = (sign << 31) | ((127 - 15 - e) << 23) | (m << 13);
      }
    } else if (exp == 31) {
      f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
      f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
  };

  // Synthetic Q8_0 weights: qs in [-127,127], d (as fp16) in ~[-2,2]; x is fp32
  // in [-1,1].
  std::vector<Q8_0BlockTest> h_A(M * num_blocks);
  std::vector<float> h_x(K);
  for (auto& blk : h_A) {
    blk.d = float_to_half_bits(rnd_float(-2.0F, 2.0F));
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] =
          static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
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
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), K * sizeof(float),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchQ8KBlockGEMV(d_A, strix::core::GgmlType::kQ8_0, d_x, d_y,
                                 M, K, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> y_gpu(M);
  HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, M * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Check 3: finite, no NaN/Inf.
  for (std::size_t m = 0; m < M; ++m) {
    if (!std::isfinite(y_gpu[m])) {
      std::cerr << "Q8_0 GEMV produced non-finite output at row " << m << ": "
                << y_gpu[m] << "\n";
      std::abort();
    }
  }

  // Check 1 (PRIMARY, tight): CPU reference replicating the integer-Q8
  // arithmetic (max|x| -> scale, xq clamp, int32 MAC, end scale).
  std::vector<float> y_q8_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    float sumf = 0.0F;
    for (std::size_t b = 0; b < num_blocks; ++b) {
      const Q8_0BlockTest& wblk = h_A[m * num_blocks + b];
      const float d_w = half_to_float(wblk.d);
      const float* xb = h_x.data() + b * QK;
      float local_max = 0.0F;
      for (std::size_t i = 0; i < QK; ++i) {
        local_max = std::max(local_max, std::abs(xb[i]));
      }
      const float scale = (local_max > 0.0F) ? (local_max / 127.0F) : 0.0F;
      int acc = 0;
      for (std::size_t i = 0; i < QK; ++i) {
        const float xv = (scale > 0.0F) ? (xb[i] / scale) : 0.0F;
        int xq = static_cast<int>(std::round(xv));
        xq = (xq > 127) ? 127 : xq;
        xq = (xq < -127) ? -127 : xq;
        acc += static_cast<int>(wblk.qs[i]) * xq;
      }
      sumf += d_w * scale * static_cast<float>(acc);
    }
    y_q8_ref[m] = sumf;
  }

  // Check 2 (SANITY, loose): CPU DotProductQ8_0 (exact fp dequant dot, no
  // activation quant) vs GPU. The difference is the activation-quantization
  // error, expected ~1%.
  std::vector<float> y_dequant_ref(M, 0.0F);
  for (std::size_t m = 0; m < M; ++m) {
    y_dequant_ref[m] = strix::quant::DotProductQ8_0(
        &h_A[m * num_blocks], std::span<const float>(h_x.data(), K), K);
  }

  float primary_max_rel = 0.0F, primary_max_abs = 0.0F;
  float sanity_max_rel = 0.0F, sanity_max_abs = 0.0F;
  for (std::size_t m = 0; m < M; ++m) {
    const float abs_q8 = std::abs(y_gpu[m] - y_q8_ref[m]);
    const float rel_q8 = abs_q8 / std::max(1e-6F, std::abs(y_q8_ref[m]));
    primary_max_rel = std::max(primary_max_rel, rel_q8);
    primary_max_abs = std::max(primary_max_abs, abs_q8);

    const float abs_san = std::abs(y_gpu[m] - y_dequant_ref[m]);
    const float rel_san =
        abs_san / std::max(1e-3F, std::abs(y_dequant_ref[m]));
    sanity_max_rel = std::max(sanity_max_rel, rel_san);
    sanity_max_abs = std::max(sanity_max_abs, abs_san);
  }

  std::cout << "Q8_0 GEMV PRIMARY (int-Q8 vs GPU): max_rel=" << primary_max_rel
            << " max_abs=" << primary_max_abs << "\n";
  std::cout << "Q8_0 GEMV SANITY (dequant vs GPU): max_rel=" << sanity_max_rel
            << " max_abs=" << sanity_max_abs << "\n";
  if (primary_max_rel >= 1e-3F || primary_max_abs >= 1e-3F) {
    std::cerr << "Q8_0 GEMV integer-dot mismatch (primary)\n";
    std::abort();
  }
  if (sanity_max_rel >= 2e-2F) {
    std::cerr
        << "Q8_0 GEMV dequant delta out of the expected quantization band\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
}

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
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No HIP device found, skipping Qwen quant gemv ops test.\n";
    return 0;
  }

  TestQ8KBlockGEMVEquivalence();
  TestQ8_0BlockGEMVEquivalence();
  TestQ5KBlockGEMVEquivalence();
  TestQ6KBlockGEMVEquivalence();
  std::cout << "Qwen quant gemv ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen quant gemv ops test.\n";
  return 0;
#endif
}
