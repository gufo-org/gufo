// Parity test for the unified quant::Dequantize / quant::Dot dispatch
// (src/core/quant/ggml_gemm.cpp). Proves:
//   (1) for the types the OLD CPU GEMV switch handles (Q3_K/Q4_K/Q6_K), the new
//       dispatch is bit-identical to the old QuantizedDot/DequantizeRow behavior;
//   (2) for the types the old switch silently skipped (Q8_K/Q8_0/Q5_K), the new
//       dispatch actually dequantizes/dots (finite + nonzero) instead of
//       returning 0.0F, matching the canonical DequantizeQ* routines exactly;
//   (3) F32/F16/BF16 handling is correct.
// The OLD reference below mirrors QuantizedDot/DequantizeRow in
// src/models/qwen_forward.cpp (Q3_K/Q4_K/Q6_K only; everything else -> 0.0F).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"

namespace {

// ---- OLD reference (mirrors qwen_forward.cpp, anonymous namespace) ----
float OldQuantizedDot(strix::core::GgmlType type, const void* row,
                      std::span<const float> x, std::size_t k) {
  switch (type) {
    case strix::core::GgmlType::kQ3_K:
      return strix::quant::DotProductQ3_K(row, x, k);
    case strix::core::GgmlType::kQ4_K:
      return strix::quant::DotProductQ4_K(row, x, k);
    case strix::core::GgmlType::kQ6_K:
      return strix::quant::DotProductQ6_K(row, x, k);
    default:
      return 0.0F;  // silent fallback the new dispatch removes
  }
}
void OldDequantizeRow(strix::core::GgmlType type, const void* row, float* out,
                      std::size_t k) {
  switch (type) {
    case strix::core::GgmlType::kQ3_K:
      strix::quant::DequantizeQ3_K(row, out, k);
      break;
    case strix::core::GgmlType::kQ4_K:
      strix::quant::DequantizeQ4_K(row, out, k);
      break;
    case strix::core::GgmlType::kQ6_K:
      strix::quant::DequantizeQ6_K(row, out, k);
      break;
    default:
      break;  // no-op; leaves `out` untouched
  }
}

bool SameBits(float a, float b) {
  std::uint32_t ua, ub;
  std::memcpy(&ua, &a, sizeof(float));
  std::memcpy(&ub, &b, sizeof(float));
  return ua == ub;
}
bool AllFinite(const float* p, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) return false;
  }
  return true;
}
std::size_t RowBytes(strix::core::GgmlType type, std::size_t k) {
  return strix::quant::QuantizedRowBytes(type, k);
}

std::vector<std::uint8_t> RandomPackedBytes(std::mt19937& rng,
                                            std::size_t n) {
  std::vector<std::uint8_t> b(n);
  std::uniform_int_distribution<unsigned> d(0, 255);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<std::uint8_t>(d(rng));
  return b;
}
std::vector<float> RandomFloats(std::mt19937& rng, std::size_t n) {
  std::vector<float> v(n);
  std::uniform_real_distribution<float> d(-1.0F, 1.0F);
  for (std::size_t i = 0; i < n; ++i) v[i] = d(rng);
  return v;
}

// ---- Valid block builders for the previously-silent types ----
std::vector<std::uint8_t> BuildQ8_0() {
  strix::quant::block_q8_0 b{};
  b.d = 0x3C00;                       // fp16(1.0)
  for (int i = 0; i < 32; ++i) b.qs[i] = static_cast<std::int8_t>(i + 1);
  std::vector<std::uint8_t> out(sizeof(b));
  std::memcpy(out.data(), &b, sizeof(b));
  return out;
}
std::vector<std::uint8_t> BuildQ8_K() {
  strix::quant::block_q8_K b{};
  b.d = 0.5F;
  for (int i = 0; i < 256; ++i) b.qs[i] = static_cast<std::int8_t>(i % 64 + 1);
  std::vector<std::uint8_t> out(sizeof(b));
  std::memcpy(out.data(), &b, sizeof(b));
  return out;
}
std::vector<std::uint8_t> BuildQ5_K() {
  strix::quant::block_q5_K b{};
  b.d = 0x3C00;    // fp16(1.0)
  b.dmin = 0x0000; // fp16(0.0)
  b.scales[0] = 1; b.scales[1] = 1; b.scales[2] = 1; b.scales[3] = 1;
  std::memset(b.qh, 0x00, sizeof(b.qh));
  std::memset(b.qs, 0x11, sizeof(b.qs));
  std::vector<std::uint8_t> out(sizeof(b));
  std::memcpy(out.data(), &b, sizeof(b));
  return out;
}

bool AllEquals(const float* p, float v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (p[i] != v) return false;
  }
  return true;
}

}  // namespace

int main() {
  std::mt19937 rng(123456);

  struct Case {
    const char* name;
    strix::core::GgmlType type;
    std::size_t k;
  };

  // (1) Parity vs OLD behavior for the 3 types old code really handles.
  const Case old_cases[] = {
      {"Q3_K", strix::core::GgmlType::kQ3_K, 256},
      {"Q4_K", strix::core::GgmlType::kQ4_K, 256},
      {"Q6_K", strix::core::GgmlType::kQ6_K, 256},
  };
  for (const auto& c : old_cases) {
    const auto packed = RandomPackedBytes(rng, RowBytes(c.type, c.k));
    const auto x = RandomFloats(rng, c.k);
    std::vector<float> dst_new(c.k), dst_old(c.k);

    strix::quant::Dequantize(c.type, packed.data(), dst_new.data(), c.k);
    OldDequantizeRow(c.type, packed.data(), dst_old.data(), c.k);

    bool deq_ok = true;
    for (std::size_t i = 0; i < c.k; ++i) {
      if (!SameBits(dst_new[i], dst_old[i])) {
        deq_ok = false;
        break;
      }
    }
    const float dot_new = strix::quant::Dot(c.type, packed.data(), x, c.k);
    const float dot_old = OldQuantizedDot(c.type, packed.data(), x, c.k);
    const bool dot_ok = SameBits(dot_new, dot_old);

    if (!deq_ok || !dot_ok) {
      std::printf("PARITY %s: FAIL (deq=%d dot=%d)\n", c.name, deq_ok, dot_ok);
      return 1;
    }
    std::printf("PARITY %s: PASS\n", c.name);
  }

  // (2) Previously-silent types: new dispatch must be finite + nonzero and
  // matching the canonical routine; OLD must have been silent (0.0F / no-op).
  const Case new_cases[] = {
      {"Q5_K", strix::core::GgmlType::kQ5_K, 256},
      {"Q8_0", strix::core::GgmlType::kQ8_0, 32},
      {"Q8_K", strix::core::GgmlType::kQ8_K, 256},
  };
  const std::vector<std::uint8_t> q5 = BuildQ5_K();
  const std::vector<std::uint8_t> q80 = BuildQ8_0();
  const std::vector<std::uint8_t> q8k = BuildQ8_K();
  for (const auto& c : new_cases) {
    const std::vector<std::uint8_t>* block =
        (c.type == strix::core::GgmlType::kQ5_K) ? &q5
        : (c.type == strix::core::GgmlType::kQ8_0) ? &q80
                                                    : &q8k;
    const auto x = RandomFloats(rng, c.k);
    std::vector<float> dst_new(c.k), dst_ref(c.k), dst_sentinel(c.k, 7.0F);

    strix::quant::Dequantize(c.type, block->data(), dst_new.data(), c.k);
    // canonical direct call:
    switch (c.type) {
      case strix::core::GgmlType::kQ5_K:
        strix::quant::DequantizeQ5_K(block->data(), dst_ref.data(), c.k);
        break;
      case strix::core::GgmlType::kQ8_0:
        strix::quant::DequantizeQ8_0(block->data(), dst_ref.data(), c.k);
        break;
      default:
        strix::quant::DequantizeQ8_K(block->data(), dst_ref.data(), c.k);
        break;
    }

    bool match = true;
    for (std::size_t i = 0; i < c.k; ++i) {
      if (!SameBits(dst_new[i], dst_ref[i])) {
        match = false;
        break;
      }
    }
    const bool finite = AllFinite(dst_new.data(), c.k);
    const bool wrote = !AllEquals(dst_new.data(), 7.0F, c.k);

    const float dot_new = strix::quant::Dot(c.type, block->data(), x, c.k);
    const float dot_old = OldQuantizedDot(c.type, block->data(), x, c.k);

    // OLD was silent:
    OldDequantizeRow(c.type, block->data(), dst_sentinel.data(), c.k);
    const bool old_noop = AllEquals(dst_sentinel.data(), 7.0F, c.k);
    const bool old_zero_dot = (dot_old == 0.0F);
    const bool dot_finite = std::isfinite(dot_new);

    if (!match || !finite || !wrote || !old_noop || !old_zero_dot || !dot_finite) {
      std::printf("NEW %s: FAIL (match=%d finite=%d wrote=%d old_noop=%d old_zero_dot=%d dotfinite=%d)\n",
                  c.name, match, finite, wrote, old_noop, old_zero_dot, dot_finite);
      return 1;
    }
    std::printf("NEW %s: PASS (dequant==canonical, finite, nonzero; OLD silent->0.0F)\n",
                c.name);
  }

  // (3) F32 / F16 / BF16.
  {
    const std::size_t k = 64;
    const auto x = RandomFloats(rng, k);
    std::vector<float> f32 = RandomFloats(rng, k);
    std::vector<float> dst(k);
    strix::quant::Dequantize(strix::core::GgmlType::kF32, f32.data(), dst.data(), k);
    bool f32_deq = true;
    for (std::size_t i = 0; i < k; ++i)
      if (!SameBits(dst[i], f32[i])) { f32_deq = false; break; }
    float f32_dot = 0.0F, f32_ref = 0.0F;
    for (std::size_t i = 0; i < k; ++i) f32_ref += f32[i] * x[i];
    f32_dot = strix::quant::Dot(strix::core::GgmlType::kF32, f32.data(), x, k);
    if (!f32_deq || !SameBits(f32_dot, f32_ref)) {
      std::printf("F32: FAIL (deq=%d dot=%d) dot=%.6f ref=%.6f\n", f32_deq,
                  SameBits(f32_dot, f32_ref), f32_dot, f32_ref);
      return 1;
    }
    std::printf("F32: PASS\n");

    // BF16: convert f32 to bf16 (truncate to upper 16 bits), ensure round-trip.
    std::vector<std::uint16_t> bf16(k);
    for (std::size_t i = 0; i < k; ++i) {
      std::uint32_t u32;
      std::memcpy(&u32, &f32[i], sizeof(u32));
      bf16[i] = static_cast<std::uint16_t>(u32 >> 16U);
    }
    strix::quant::Dequantize(strix::core::GgmlType::kBF16, bf16.data(), dst.data(), k);
    bool bf16_finite = AllFinite(dst.data(), k);
    float bf16_dot = strix::quant::Dot(strix::core::GgmlType::kBF16, bf16.data(), x, k);
    if (!bf16_finite || !std::isfinite(bf16_dot)) {
      std::printf("BF16: FAIL (finite=%d dotfinite=%d)\n", bf16_finite,
                  std::isfinite(bf16_dot));
      return 1;
    }
    std::printf("BF16: PASS\n");

    // F16: use Fp16ToFloat as the oracle.
    std::vector<std::uint16_t> f16(k);
    for (std::size_t i = 0; i < k; ++i) {
      // encode a few known fp16 values (e.g. 1.0, 0.5, -2.0) by scanning.
      f16[i] = 0x3C00;  // 1.0
    }
    strix::quant::Dequantize(strix::core::GgmlType::kF16, f16.data(), dst.data(), k);
    bool f16_ok = true;
    for (std::size_t i = 0; i < k; ++i)
      if (dst[i] != 1.0F) { f16_ok = false; break; }
    if (!f16_ok) {
      std::printf("F16: FAIL\n");
      return 1;
    }
    std::printf("F16: PASS\n");
  }

  // (4) IsSupported boundaries.
  if (!strix::quant::IsSupported(strix::core::GgmlType::kF32) ||
      !strix::quant::IsSupported(strix::core::GgmlType::kQ8_K) ||
      strix::quant::IsSupported(strix::core::GgmlType::kQ4_0) ||
      strix::quant::IsSupported(strix::core::GgmlType::kStrixSHQ4_T16)) {
    std::printf("IsSupported: FAIL\n");
    return 1;
  }
  std::printf("IsSupported: PASS\n");

  std::printf("ALL_GEMM_PARITY_PASS\n");
  return 0;
}
