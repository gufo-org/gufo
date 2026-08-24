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

void TestDequantizeQ8KToBf16Equivalence() {
  constexpr std::size_t QK = 256;
  constexpr std::size_t num_blocks = 4;
  constexpr std::size_t n_elems = num_blocks * QK;

  using Q8KBlockTest = strix::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  // Deterministic pseudo-random weights (same values every run).
  std::uint32_t seed = 777U;
  auto rnd = [&seed]() -> std::uint32_t {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  auto rnd_float = [&rnd](float lo, float hi) -> float {
    const float u = static_cast<float>(rnd() & 0xFFFFU) / 65535.0F;
    return lo + u * (hi - lo);
  };

  std::vector<Q8KBlockTest> h_w(num_blocks);
  for (auto& blk : h_w) {
    blk.d = rnd_float(-1.0F, 1.0F);
    for (std::size_t i = 0; i < QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>(static_cast<int>(rnd() % 255) - 127);
    }
    for (std::size_t i = 0; i < 16; ++i) {
      blk.bsums[i] = 0;
    }
  }

  // CPU reference oracle (full-precision float dequant).
  std::vector<float> h_ref(n_elems);
  strix::quant::DequantizeQ8_K(h_w.data(), h_ref.data(), n_elems);

  void* d_w = nullptr;
  hip_bfloat16* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_w, num_blocks * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      n_elems * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), num_blocks * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));

  strix::hip::LaunchDequantizeQ8KToBf16(d_w, d_out, n_elems, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<hip_bfloat16> h_gpu(n_elems);
  HIP_CHECK(hipMemcpy(h_gpu.data(), d_out, n_elems * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));

  float max_rel = 0.0F, max_abs = 0.0F;
  for (std::size_t i = 0; i < n_elems; ++i) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &h_gpu[i], sizeof(bits));
    const float gpu = strix::test::Bf16BitsToFloat(bits);
    const float ref = h_ref[i];
    if (!std::isfinite(gpu)) {
      std::cerr << "Q8K dequant produced non-finite output at " << i << ": "
                << gpu << "\n";
      std::abort();
    }
    const float abs_d = std::abs(gpu - ref);
    const float rel_d = abs_d / std::max(1e-3F, std::abs(ref));
    max_rel = std::max(max_rel, rel_d);
    max_abs = std::max(max_abs, abs_d);
  }
  std::cout << "Q8K dequant (GPU BF16 vs CPU): max_rel=" << max_rel
            << " max_abs=" << max_abs << "\n";
  if (max_rel >= 1e-2F) {
    std::cerr << "Q8K dequant BF16 mismatch (relative)\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

static void CheckDequantToBf16(const char* name, const std::vector<float>& ref,
                               const std::vector<hip_bfloat16>& gpu) {
  float max_rel = 0.0F, max_abs = 0.0F;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &gpu[i], sizeof(bits));
    const float g = strix::test::Bf16BitsToFloat(bits);
    const float r = ref[i];
    if (!std::isfinite(g)) {
      std::cerr << name << " dequant produced non-finite output at " << i
                << ": " << g << "\n";
      std::abort();
    }
    const float abs_d = std::abs(g - r);
    const float rel_d = abs_d / std::max(1e-3F, std::abs(r));
    max_rel = std::max(max_rel, rel_d);
    max_abs = std::max(max_abs, abs_d);
  }
  std::cout << name << " dequant (GPU BF16 vs CPU): max_rel=" << max_rel
            << " max_abs=" << max_abs << "\n";
  if (max_rel >= 1e-2F) {
    std::cerr << name << " dequant BF16 mismatch (relative)\n";
    std::abort();
  }
}

void TestDequantizeToBf16Equivalence() {
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

  // ---- Q8_0: 2 blocks (64 elems), d=0.5, deterministic qs across [-127,127].
  constexpr std::size_t Q80_QK = 32;
  constexpr std::size_t Q80_BLOCKS = 2;
  constexpr std::size_t Q80_ELEMS = Q80_BLOCKS * Q80_QK;
  using Q8_0BlockTest = strix::quant::block_q8_0;
  static_assert(sizeof(Q8_0BlockTest) == 34, "Q8_0 block must be 34 bytes");

  std::vector<Q8_0BlockTest> h_q80(Q80_BLOCKS);
  for (auto& blk : h_q80) {
    blk.d = float_to_half_bits(0.5F);
    for (std::size_t i = 0; i < Q80_QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>((i * 13 + 7) % 255 - 127);
    }
  }
  std::vector<float> h_ref_q80(Q80_ELEMS);
  strix::quant::DequantizeQ8_0(h_q80.data(), h_ref_q80.data(), Q80_ELEMS);

  void* d_w = nullptr;
  hip_bfloat16* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_w, Q80_BLOCKS * sizeof(Q8_0BlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      Q80_ELEMS * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_q80.data(), Q80_BLOCKS * sizeof(Q8_0BlockTest),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ8_0, d_w, d_out,
                                     Q80_ELEMS, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q80(Q80_ELEMS);
  HIP_CHECK(hipMemcpy(h_gpu_q80.data(), d_out, Q80_ELEMS * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q8_0", h_ref_q80, h_gpu_q80);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q5_K: 1 block (256 elems), d=0.25 dmin=0.125, high-bit scales,
  // qh=0xFF, qs nibbles across the four 64-element groups.
  constexpr std::size_t QK = 256;
  using Q5KBlockTest = strix::quant::block_q5_K;
  static_assert(sizeof(Q5KBlockTest) == 176, "Q5_K block must be 176 bytes");

  Q5KBlockTest q5k{};
  q5k.d = float_to_half_bits(0.25F);
  q5k.dmin = float_to_half_bits(0.125F);
  const std::uint8_t scales[12] = {0x04, 0x04, 0x04, 0x04, 0x01, 0x01,
                                   0x01, 0x01, 0x14, 0x14, 0x14, 0x14};
  std::memcpy(q5k.scales, scales, sizeof(scales));
  std::memset(q5k.qh, 0xFF, sizeof(q5k.qh));
  const std::uint8_t nibbles[4] = {0x11, 0x22, 0x33, 0x44};
  for (std::size_t g = 0; g < 4; ++g) {
    std::memset(q5k.qs + (g * 32), nibbles[g], 32);
  }
  std::vector<float> h_ref_q5k(QK);
  strix::quant::DequantizeQ5_K(&q5k, h_ref_q5k.data(), QK);

  HIP_CHECK(hipMalloc(&d_w, sizeof(Q5KBlockTest)));
  HIP_CHECK(
      hipMalloc(reinterpret_cast<void**>(&d_out), QK * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, &q5k, sizeof(Q5KBlockTest), hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ5_K, d_w, d_out,
                                     QK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q5k(QK);
  HIP_CHECK(hipMemcpy(h_gpu_q5k.data(), d_out, QK * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q5_K", h_ref_q5k, h_gpu_q5k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q6_K: 1 block, d=0.3, ql varies, qh=0xFF, scales [7..4] (positive).
  using Q6KBlockTest = strix::quant::block_q6_K;
  static_assert(sizeof(Q6KBlockTest) == 210, "Q6_K block must be 210 bytes");

  Q6KBlockTest q6k{};
  q6k.d = float_to_half_bits(0.3F);
  for (std::size_t i = 0; i < 128; ++i) {
    q6k.ql[i] = static_cast<std::uint8_t>((i * 13 + 5) & 0xFF);
  }
  std::memset(q6k.qh, 0xFF, sizeof(q6k.qh));
  for (std::size_t i = 0; i < 16; ++i) {
    q6k.scales[i] = static_cast<std::int8_t>(7 - (i % 4));
  }
  std::vector<float> h_ref_q6k(QK);
  strix::quant::DequantizeQ6_K(&q6k, h_ref_q6k.data(), QK);

  HIP_CHECK(hipMalloc(&d_w, sizeof(Q6KBlockTest)));
  HIP_CHECK(
      hipMalloc(reinterpret_cast<void**>(&d_out), QK * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, &q6k, sizeof(Q6KBlockTest), hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ6_K, d_w, d_out,
                                     QK, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q6k(QK);
  HIP_CHECK(hipMemcpy(h_gpu_q6k.data(), d_out, QK * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q6_K", h_ref_q6k, h_gpu_q6k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));

  // ---- Q8_K via the dispatcher (delegates to the existing Q8_K kernel):
  // 2 blocks (512 elems), deterministic d/qs.
  constexpr std::size_t Q8K_QK = 256;
  constexpr std::size_t Q8K_BLOCKS = 2;
  constexpr std::size_t Q8K_ELEMS = Q8K_BLOCKS * Q8K_QK;
  using Q8KBlockTest = strix::quant::block_q8_K;
  static_assert(sizeof(Q8KBlockTest) == 292, "Q8_K block must be 292 bytes");

  std::vector<Q8KBlockTest> h_q8k(Q8K_BLOCKS);
  for (std::size_t blk_idx = 0; blk_idx < Q8K_BLOCKS; ++blk_idx) {
    Q8KBlockTest& blk = h_q8k[blk_idx];
    blk.d = 0.5F + static_cast<float>(blk_idx);
    for (std::size_t i = 0; i < Q8K_QK; ++i) {
      blk.qs[i] = static_cast<std::int8_t>((i * 7 + 13) % 255 - 127);
    }
    std::memset(blk.bsums, 0, sizeof(blk.bsums));
  }
  std::vector<float> h_ref_q8k(Q8K_ELEMS);
  strix::quant::DequantizeQ8_K(h_q8k.data(), h_ref_q8k.data(), Q8K_ELEMS);

  HIP_CHECK(hipMalloc(&d_w, Q8K_BLOCKS * sizeof(Q8KBlockTest)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_out),
                      Q8K_ELEMS * sizeof(hip_bfloat16)));
  HIP_CHECK(hipMemcpy(d_w, h_q8k.data(), Q8K_BLOCKS * sizeof(Q8KBlockTest),
                      hipMemcpyHostToDevice));
  strix::hip::LaunchDequantizeToBf16(strix::core::GgmlType::kQ8_K, d_w, d_out,
                                     Q8K_ELEMS, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<hip_bfloat16> h_gpu_q8k(Q8K_ELEMS);
  HIP_CHECK(hipMemcpy(h_gpu_q8k.data(), d_out, Q8K_ELEMS * sizeof(hip_bfloat16),
                      hipMemcpyDeviceToHost));
  CheckDequantToBf16("Q8_K", h_ref_q8k, h_gpu_q8k);
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_out));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = strix::test::GateHipDevice(
      strix::test::HipDeviceRequirement::kOptional, "Qwen dequant ops test");
  if (device_status != strix::test::kHipTestSuccess) {
    return device_status;
  }

  TestDequantizeQ8KToBf16Equivalence();
  TestDequantizeToBf16Equivalence();
  std::cout << "Qwen dequant ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen dequant ops test.\n";
  return 77;
#endif
}
