// Parity proof for the canonical quantized block layouts exported from
// src/core/quant/ggml_dequant.hpp.
//
// Goal (R2 risk): static_assert(sizeof(...)) proves byte SIZE but NOT field
// ORDER. If the canonical header structs had the wrong field order, dequant
// via the struct would silently diverge from the true byte-level format. This
// test dequantizes identical packed bytes TWO independent ways:
//    (A) reference  = the existing production DequantizeQ* functions
//                     (which consume the header structs);
//    (B) parity     = a byte-offset dequant that reads raw bytes at explicit
//                     offsets per the spec, WITHOUT using any struct type.
// If (A)==(B) for every position, the header struct field order matches the
// byte layout. If the canonical layout were wrong, (A) would diverge from (B).

#include "src/core/quant/ggml_dequant.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using strix::quant::Fp16ToFloat;

namespace {

struct Fp16Bits {
  std::uint16_t bits;
};

float ReadF32Le(const std::uint8_t* p, std::size_t off) {
  float v = 0.0f;
  std::memcpy(&v, p + off, sizeof(float));
  return v;
}

std::uint16_t ReadU16Le(const std::uint8_t* p, std::size_t off) {
  return static_cast<std::uint16_t>(p[off]) |
         (static_cast<std::uint16_t>(p[off + 1]) << 8U);
}

auto Fp16(const std::uint8_t* p, std::size_t off) {
  return Fp16ToFloat(ReadU16Le(p, off));
}

// ---- byte-offset dequant references (no struct type) ----

void GetQ4ScaleMin(std::size_t index, const std::uint8_t* scales,
                   std::uint8_t& scale, std::uint8_t& minimum) {
  if (index < 4) {
    scale = scales[index] & 0x3FU;
    minimum = scales[index + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((scales[index + 4] & 0x0FU) |
                                    ((scales[index - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((scales[index + 4] >> 4U) |
                                      ((scales[index] >> 6U) << 4U));
}

float Q4Val(const std::uint8_t* p, std::size_t index) {
  const std::size_t group = index / 32;
  const std::size_t pair = group / 2;
  const bool high = (group & 1U) != 0;
  const std::size_t lane = index % 32;
  const std::uint8_t packed = p[16 + (pair * 32) + lane];  // qs at offset 16
  const std::uint8_t quant = high ? packed >> 4U : packed & 0x0FU;
  std::uint8_t scale = 0;
  std::uint8_t min = 0;
  GetQ4ScaleMin(group, p + 4, scale, min);  // scales at offset 4
  return Fp16(p, 0) * static_cast<float>(scale) * static_cast<float>(quant) -
         Fp16(p, 2) * static_cast<float>(min);  // d/dmin at offsets 0/2
}

float Q5Val(const std::uint8_t* p, std::size_t index) {
  const std::uint8_t* qs = p + 48;  // qs at offset 48
  const std::uint8_t* qh = p + 16;  // qh at offset 16
  const std::size_t gg = index / 64;
  const std::size_t wv = index % 64;
  const std::size_t lane = wv % 32;
  const bool lohalf = wv < 32;
  const std::uint8_t qb = qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const std::uint8_t qhb = qh[lane];
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((qhb >> bit) & 1U) ? 16U : 0U));
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQ4ScaleMin(sis, p + 4, sc, m);
  return Fp16(p, 0) * static_cast<float>(sc) * static_cast<float>(quant) -
         Fp16(p, 2) * static_cast<float>(m);
}

float Q6Val(const std::uint8_t* p, std::size_t index) {
  const std::uint8_t* ql = p;        // ql at offset 0
  const std::uint8_t* qh = p + 128;  // qh at offset 128
  const std::int8_t* scales = reinterpret_cast<const std::int8_t*>(p + 192);
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t segment = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh0 = qh[(half * 32) + lane];
  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = ql[ql_base + lane] & 0x0FU;
      high = qh0 & 0x03U;
      break;
    case 1:
      low = ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh0 >> 2U) & 0x03U;
      break;
    case 2:
      low = ql[ql_base + lane] >> 4U;
      high = (qh0 >> 4U) & 0x03U;
      break;
    default:
      low = ql[ql_base + 32 + lane] >> 4U;
      high = (qh0 >> 6U) & 0x03U;
      break;
  }
  const std::size_t si = (half * 8) + (lane / 16) + (segment * 2);
  const auto quant =
      static_cast<std::int8_t>(static_cast<int>((high << 4U) | low) - 32);
  return Fp16(p, 208) * static_cast<float>(scales[si]) *
         static_cast<float>(quant);  // d at offset 208
}

std::int8_t UnpackQ3Scale(const std::uint8_t* packed, std::size_t index) {
  const auto low = index < 8
                       ? packed[index] & 0x0FU
                       : (packed[index - 8] >> 4U) & 0x0FU;
  const auto high = (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
  return static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
}

float Q3Val(const std::uint8_t* p, std::size_t index) {
  const std::uint8_t* hmask = p;        // hmask at offset 0
  const std::uint8_t* qs = p + 32;      // qs at offset 32
  const std::uint8_t* scales = p + 96;  // scales at offset 96
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t sp = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t qi = (half * 32) + lane;
  const std::uint8_t shift = static_cast<std::uint8_t>(2U * sp);
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>(1U << ((half * 4) + sp));
  const auto low = static_cast<std::int8_t>((qs[qi] >> shift) & 0x03U);
  const auto quant = static_cast<std::int8_t>(
      static_cast<int>(low) - ((hmask[lane] & high_mask) != 0 ? 0 : 4));
  const std::size_t si = (half * 8) + (sp * 2) + (lane / 16);
  const std::int8_t scale = UnpackQ3Scale(scales, si);
  return Fp16(p, 108) * static_cast<float>(scale) *
         static_cast<float>(quant);  // d at offset 108
}

float Q8_0Val(const std::uint8_t* p, std::size_t index) {
  const float d = Fp16(p, 0);
  return d * static_cast<float>(
                 static_cast<std::int8_t>(p[2 + index]));  // qs at offset 2
}

float Q8_KVal(const std::uint8_t* p, std::size_t index) {
  const float d = ReadF32Le(p, 0);
  return d * static_cast<float>(
                 static_cast<std::int8_t>(p[4 + index]));  // qs at offset 4
}

// ---- test-vector helpers ----

void FillPattern(std::vector<std::uint8_t>& bytes, std::uint64_t seed) {
  std::uint64_t x = seed;
  for (auto& b : bytes) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    b = static_cast<std::uint8_t>(x >> 33U);
  }
}

int Compare(const char* name, std::size_t k, std::vector<float>& ref,
            std::vector<float>& ind) {
  int worst = 0;
  for (std::size_t i = 0; i < k; ++i) {
    if (ref[i] != ind[i]) {
      const float d = std::fabs(ref[i] - ind[i]);
      if (d > 1e-6f) {
        std::printf("  %s[%zu]: ref=%.9g parity=%.9g diff=%.3g\n", name, i,
                    ref[i], ind[i], d);
        ++worst;
      }
    }
  }
  std::printf("%-6s k=%zu  %s\n", name, k, worst == 0 ? "PASS" : "FAIL");
  return worst == 0 ? 0 : 1;
}

}  // namespace

int main() {
  int failures = 0;

  // Q8_0 (k=32, block 34B).
  {
    constexpr std::size_t k = 32;
    std::vector<std::uint8_t> bytes(34);
    FillPattern(bytes, 0x1234);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ8_0(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q8_0Val(bytes.data(), i);
    failures += Compare("Q8_0", k, ref, ind);
  }

  // Q8_K (k=256, block 292B).
  {
    constexpr std::size_t k = 256;
    std::vector<std::uint8_t> bytes(292);
    FillPattern(bytes, 0x5678);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ8_K(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q8_KVal(bytes.data(), i);
    failures += Compare("Q8_K", k, ref, ind);
  }

  // Q4_K (k=256, block 144B).
  {
    constexpr std::size_t k = 256;
    std::vector<std::uint8_t> bytes(144);
    FillPattern(bytes, 0x9abc);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ4_K(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q4Val(bytes.data(), i);
    failures += Compare("Q4_K", k, ref, ind);
  }

  // Q5_K (k=256, block 176B).
  {
    constexpr std::size_t k = 256;
    std::vector<std::uint8_t> bytes(176);
    FillPattern(bytes, 0xdef0);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ5_K(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q5Val(bytes.data(), i);
    failures += Compare("Q5_K", k, ref, ind);
  }

  // Q6_K (k=256, block 210B).
  {
    constexpr std::size_t k = 256;
    std::vector<std::uint8_t> bytes(210);
    FillPattern(bytes, 0x1357);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ6_K(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q6Val(bytes.data(), i);
    failures += Compare("Q6_K", k, ref, ind);
  }

  // Q3_K (k=256, block 110B).
  {
    constexpr std::size_t k = 256;
    std::vector<std::uint8_t> bytes(110);
    FillPattern(bytes, 0x2468);
    std::vector<float> ref(k), ind(k);
    strix::quant::DequantizeQ3_K(bytes.data(), ref.data(), k);
    for (std::size_t i = 0; i < k; ++i) ind[i] = Q3Val(bytes.data(), i);
    failures += Compare("Q3_K", k, ref, ind);
  }

  if (failures == 0) {
    std::printf("ALL_BLOCK_PARITY_PASS\n");
  } else {
    std::printf("BLOCK_PARITY_FAILED (%d)\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
