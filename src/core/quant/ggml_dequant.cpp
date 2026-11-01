#include "src/core/quant/ggml_dequant.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace gufo::quant {

// Standard half-precision float to single-precision float conversion
float Fp16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      float zero = 0.0f;
      std::memcpy(&zero, &sign, sizeof(float));
      return zero;
    }
    const float f =
        (static_cast<float>(mant) / 1024.0f) * std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    const std::uint32_t val = sign | 0x7F800000U | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
  }

  const std::uint32_t val = sign | ((exp + 112) << 23) | (mant << 13);
  float f = 0.0f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

// Block layouts (block_q4_K, block_q5_K, block_q6_K, block_q3_K,
// block_q8_K, block_q8_0) now live in the header; they are the canonical
// layout source. Definitions removed here to avoid ODR redefinition.

namespace {

void GetQ4ScaleMin(std::size_t index, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) noexcept {
  if (index < 4) {
    scale = packed[index] & 0x3FU;
    minimum = packed[index + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                    ((packed[index - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                      ((packed[index] >> 6U) << 4U));
}

std::array<std::int8_t, 16> UnpackQ3Scales(
    const std::uint8_t* packed) noexcept {
  std::array<std::int8_t, 16> scales{};
  for (std::size_t index = 0; index < scales.size(); ++index) {
    const auto low =
        index < 8 ? packed[index] & 0x0FU : (packed[index - 8] >> 4U) & 0x0FU;
    const auto high = (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
    scales[index] =
        static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
  }
  return scales;
}

float Q4Value(const block_q4_K& block, std::size_t index) noexcept {
  const std::size_t group = index / 32;
  const std::size_t pair = group / 2;
  const bool high_nibble = (group & 1U) != 0;
  const std::size_t lane = index % 32;
  const std::uint8_t packed = block.qs[(pair * 32) + lane];
  const std::uint8_t quant = high_nibble ? packed >> 4U : packed & 0x0FU;
  std::uint8_t scale = 0;
  std::uint8_t minimum = 0;
  GetQ4ScaleMin(group, block.scales, scale, minimum);
  return Fp16ToFloat(block.d) * static_cast<float>(scale) *
             static_cast<float>(quant) -
         Fp16ToFloat(block.dmin) * static_cast<float>(minimum);
}

float Q5Value(const block_q5_K& block, std::size_t index) noexcept {
  const std::size_t gg = index / 64;  // 0..3
  const std::size_t wv = index % 64;  // 0..63
  const std::size_t lane = wv % 32;   // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const std::uint8_t qhb = block.qh[lane];
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((qhb >> bit) & 1U) ? 16U : 0U));       // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);  // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQ4ScaleMin(sis, block.scales, sc, m);
  return Fp16ToFloat(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         Fp16ToFloat(block.dmin) * static_cast<float>(m);
}

float Q6Value(const block_q6_K& block, std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t segment = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }

  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const auto quant =
      static_cast<std::int8_t>(static_cast<int>((high << 4U) | low) - 32);
  return Fp16ToFloat(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

float Q3Value(const block_q3_K& block,
              const std::array<std::int8_t, 16>& scales,
              std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t scale_pair = within_half / 32;
  const std::size_t lane = within_half % 32;
  const std::size_t quant_index = (half * 32) + lane;
  const std::uint8_t shift = static_cast<std::uint8_t>(2U * scale_pair);
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>(1U << ((half * 4) + scale_pair));
  const auto low =
      static_cast<std::int8_t>((block.qs[quant_index] >> shift) & 0x03U);
  const auto quant = static_cast<std::int8_t>(
      static_cast<int>(low) - ((block.hmask[lane] & high_mask) != 0 ? 0 : 4));
  const std::size_t scale_index = (half * 8) + (scale_pair * 2) + (lane / 16);
  return Fp16ToFloat(block.d) * static_cast<float>(scales[scale_index]) *
         static_cast<float>(quant);
}

}  // namespace

std::size_t QuantizedRowBytes(core::GgmlType type,
                              std::size_t elements) noexcept {
  const std::size_t block_qk = QuantizedBlockElements(type);
  std::size_t block_bytes = 0;
  switch (type) {
    // GGML storage layouts also used by the DeepSeek runtime. Storage support
    // here does not imply that every model has a compute kernel for the type.
    case core::GgmlType::kQ4_0:
      block_bytes = 18;
      break;
    case core::GgmlType::kQ4_1:
      block_bytes = 20;
      break;
    case core::GgmlType::kQ5_0:
      block_bytes = 22;
      break;
    case core::GgmlType::kQ5_1:
      block_bytes = 24;
      break;
    case core::GgmlType::kQ8_1:
      block_bytes = 36;
      break;
    case core::GgmlType::kQ2_K:
      block_bytes = 84;
      break;
    case core::GgmlType::kIQ2_XXS:
      block_bytes = 66;
      break;
    case core::GgmlType::kIQ2_XS:
      block_bytes = 74;
      break;
    case core::GgmlType::kIQ2_S:
      block_bytes = 82;
      break;
    case core::GgmlType::kIQ3_XXS:
      block_bytes = 98;
      break;
    case core::GgmlType::kIQ1_S:
      block_bytes = 50;
      break;
    case core::GgmlType::kIQ1_M:
      block_bytes = 56;
      break;
    case core::GgmlType::kQ3_K:
      block_bytes = sizeof(block_q3_K);
      break;
    case core::GgmlType::kQ4_K:
      block_bytes = sizeof(block_q4_K);
      break;
    case core::GgmlType::kQ5_K:
      block_bytes = sizeof(block_q5_K);
      break;
    case core::GgmlType::kQ6_K:
      block_bytes = sizeof(block_q6_K);
      break;
    case core::GgmlType::kQ8_K:
      block_bytes = sizeof(block_q8_K);
      break;
    case core::GgmlType::kQ8_0:
      block_bytes = sizeof(block_q8_0);
      break;
    case core::GgmlType::kIQ4_NL:
      block_bytes = sizeof(block_iq4_nl);
      break;
    case core::GgmlType::kIQ4_XS:
      block_bytes = sizeof(block_iq4_xs);
      break;
    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;
    case core::GgmlType::kQ2_0:
      block_bytes = sizeof(block_q2_0);
      break;
    default:
      return 0;
  }
  if ((elements % block_qk) != 0) {
    return 0;
  }
  const std::size_t blocks = elements / block_qk;
  if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
    return 0;
  }
  return blocks * block_bytes;
}

std::size_t EncodedSizeBytes(core::GgmlType type,
                             std::size_t elements) noexcept {
  switch (type) {
    case core::GgmlType::kI32:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0;
      }
      return elements * sizeof(std::int32_t);
    case core::GgmlType::kF32:
      if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return 0;
      }
      return elements * sizeof(float);
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return 0;
      }
      return elements * sizeof(std::uint16_t);
    default:
      return QuantizedRowBytes(type, elements);
  }
}

void DequantizeQ4_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q4Value(blocks[b], i);
    }
  }
}

void DequantizeQ5_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q5_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q5Value(blocks[b], i);
    }
  }
}

void DequantizeQ6_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q6Value(blocks[b], i);
    }
  }
}

void DequantizeQ3_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Q3Value(blocks[b], scales, i);
    }
  }
}

void DequantizeQ8_K(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q8_K*>(src);
  const std::size_t nb = k / 256;

  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = blocks[b].d * static_cast<float>(blocks[b].qs[i]);
    }
  }
}

float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q4_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q4Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ5_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q5_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q5Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q6_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q6Value(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q3_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const auto scales = UnpackQ3Scales(blocks[b].scales);
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += Q3Value(blocks[b], scales, i) * v[i];
    }
  }
  return sum;
}

float DotProductQ8_K(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q8_K*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + b * 256;

    for (std::size_t i = 0; i < 256; ++i) {
      sum += blocks[b].d * static_cast<float>(blocks[b].qs[i]) * v[i];
    }
  }
  return sum;
}

void DequantizeQ8_0(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q8_0*>(src);
  const std::size_t nb = k / 32;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    for (std::size_t i = 0; i < 32; ++i) {
      dst[(b * 32) + i] = d * static_cast<float>(blocks[b].qs[i]);
    }
  }
}

void DequantizeQ5_1(const void* src, float* dst, std::size_t k) {
  // d, m (F16), 32 high bits, then 16 bytes of low nibbles per 32 elements.
  constexpr std::size_t kBlockBytes = 24;
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 32; ++b) {
    const std::uint8_t* blk = bytes + b * kBlockBytes;
    std::uint16_t dh = 0;
    std::uint16_t mh = 0;
    std::uint32_t qh = 0;
    std::memcpy(&dh, blk, 2);
    std::memcpy(&mh, blk + 2, 2);
    std::memcpy(&qh, blk + 4, 4);
    const std::uint8_t* qs = blk + 8;
    const float d = Fp16ToFloat(dh);
    const float m = Fp16ToFloat(mh);
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      const int xh0 = ((qh >> j) & 1U) << 4;
      const int xh1 = ((qh >> (j + 16)) & 1U) << 4;
      y[j] = static_cast<float>((qs[j] & 0x0F) | xh0) * d + m;
      y[j + 16] = static_cast<float>((qs[j] >> 4) | xh1) * d + m;
    }
  }
}

float DotProductQ8_0(const void* row_data, std::span<const float> vec,
                     std::size_t k) {
  const auto* blocks = static_cast<const block_q8_0*>(row_data);
  const std::size_t nb = k / 32;
  float sum = 0.0F;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    const float* v = vec.data() + b * 32;

    for (std::size_t i = 0; i < 32; ++i) {
      sum += d * static_cast<float>(blocks[b].qs[i]) * v[i];
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// IQ4_NL / IQ4_XS / IQ3_S (opt-q4kxl): the Unsloth UD-Q4_K_XL Qwen3.8-27B shard
// mixes these three with Q3_K/Q4_K/Q5_K/Q6_K/Q8_0, so every one of them needs a
// CPU oracle here before the HIP path can be validated against it. All three
// mirror ggml-quants.c (dequantize_row_iq4_nl / _iq4_xs / _iq3_s) element for
// element.
// ---------------------------------------------------------------------------

namespace {

// 512-entry IQ3_S grid, verbatim from ggml-common.h. Each entry packs four
// uint8 magnitudes for one group of four elements.

constexpr std::uint8_t kSignMaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
}  // namespace

float Iq4NlValue(const block_iq4_nl& block, std::size_t index) noexcept {
  const std::size_t lane = index % 16;
  const bool high_nibble = index >= 16;
  const std::uint8_t packed = block.qs[lane];
  const std::uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  return Fp16ToFloat(block.d) * static_cast<float>(kValuesIq4Nl[code]);
}

float Iq4XsValue(const block_iq4_xs& block, std::size_t index) noexcept {
  const std::size_t ib = index / 32;  // 0..7 sub-block
  const std::size_t within = index % 32;
  const std::size_t lane = within % 16;
  const bool high_nibble = within >= 16;
  const std::uint8_t packed = block.qs[(ib * 16) + lane];
  const std::uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  const int ls =
      static_cast<int>((block.scales_l[ib / 2] >> (4U * (ib % 2))) & 0x0FU) |
      static_cast<int>(((block.scales_h >> (2U * ib)) & 0x03U) << 4U);
  return Fp16ToFloat(block.d) * static_cast<float>(ls - 32) *
         static_cast<float>(kValuesIq4Nl[code]);
}

float Iq3sValue(const block_iq3_s& block, std::size_t index) noexcept {
  const std::size_t ib32 = index / 32;    // 0..7 sub-block
  const std::size_t within = index % 32;  // 0..31
  const std::size_t l = within / 8;       // 0..3 group of eight
  const std::size_t j = within % 8;       // 0..7 element in the group
  // Each group of eight is two grid lookups of four values each.
  const std::size_t half = j / 4;  // 0 -> grid1, 1 -> grid2
  const std::size_t jj = j % 4;

  const std::uint8_t qh_byte = block.qh[ib32];
  const std::size_t qs_index = (ib32 * 8) + (2 * l) + half;
  const int shift = static_cast<int>(8 - (2 * l) - half);
  const std::uint32_t grid_index =
      static_cast<std::uint32_t>(block.qs[qs_index]) |
      ((static_cast<std::uint32_t>(qh_byte) << shift) & 256U);
  const auto* grid =
      reinterpret_cast<const std::uint8_t*>(&kIq3sGrid[grid_index]);

  const std::uint8_t sign_byte = block.signs[(ib32 * 4) + l];
  const bool negate = (sign_byte & kSignMaskIq2xs[j]) != 0;

  const std::uint8_t scale_byte = block.scales[ib32 / 2];
  const int scale_nibble = static_cast<int>(
      (ib32 % 2 == 0) ? (scale_byte & 0x0FU)
                      : static_cast<unsigned>(scale_byte >> 4U));
  const float db =
      Fp16ToFloat(block.d) * static_cast<float>(1 + (2 * scale_nibble));

  const float magnitude = static_cast<float>(grid[jj]);
  return negate ? -db * magnitude : db * magnitude;
}

const std::uint32_t* Iq3sGrid() noexcept {
  return kIq3sGrid;
}

void DequantizeIQ4_NL(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_nl*>(src);
  const std::size_t nb = k / 32;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 32; ++i) {
      dst[(b * 32) + i] = Iq4NlValue(blocks[b], i);
    }
  }
}

void DequantizeIQ4_XS(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_xs*>(src);
  const std::size_t nb = k / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Iq4XsValue(blocks[b], i);
    }
  }
}

void DequantizeIQ3_S(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_iq3_s*>(src);
  const std::size_t nb = k / 256;
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::size_t i = 0; i < 256; ++i) {
      dst[(b * 256) + i] = Iq3sValue(blocks[b], i);
    }
  }
}

float DotProductIQ4_NL(const void* row_data, std::span<const float> vec,
                       std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_nl*>(row_data);
  const std::size_t nb = k / 32;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 32);
    for (std::size_t i = 0; i < 32; ++i) {
      sum += Iq4NlValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductIQ4_XS(const void* row_data, std::span<const float> vec,
                       std::size_t k) {
  const auto* blocks = static_cast<const block_iq4_xs*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 256);
    for (std::size_t i = 0; i < 256; ++i) {
      sum += Iq4XsValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

float DotProductIQ3_S(const void* row_data, std::span<const float> vec,
                      std::size_t k) {
  const auto* blocks = static_cast<const block_iq3_s*>(row_data);
  const std::size_t nb = k / 256;
  float sum = 0.0F;
  for (std::size_t b = 0; b < nb; ++b) {
    const float* v = vec.data() + (b * 256);
    for (std::size_t i = 0; i < 256; ++i) {
      sum += Iq3sValue(blocks[b], i) * v[i];
    }
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Q4_0 / Q4_1 / Q5_0 / Q2_K and the IQ1/IQ2/IQ3 importance formats, element
// for element as ggml-quants.c's dequantize_row_* (the same products in the
// same order, so an exact-arithmetic consumer reproduces them bit for bit).
// ---------------------------------------------------------------------------

namespace {

#include "src/core/quant/ggml_iq_grids.inc"

constexpr float kIq1Delta = 0.125F;

std::uint16_t LoadU16(const std::uint8_t* p) noexcept {
  std::uint16_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

std::uint32_t LoadU32(const std::uint8_t* p) noexcept {
  std::uint32_t v = 0;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

float SignOf(std::uint8_t signs, int j) noexcept {
  return (signs & kSignMaskIq2xs[j]) != 0 ? -1.0F : 1.0F;
}

/// Element j of iq1s_grid entry `index`, from the GPU packing (value + 1).
float Iq1sGrid(std::uint32_t index, int j) noexcept {
  const int nibble = 2 * (j % 4) + j / 4;
  return static_cast<float>(
      static_cast<int>((kIq1sGridGpu[index] >> (4 * nibble)) & 0x0FU) - 1);
}

}  // namespace

void DequantizeQ4_0(const void* src, float* dst, std::size_t k) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 32; ++b) {
    const std::uint8_t* blk = bytes + b * 18;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint8_t* qs = blk + 2;
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      y[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
      y[j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
  }
}

void DequantizeQ4_1(const void* src, float* dst, std::size_t k) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 32; ++b) {
    const std::uint8_t* blk = bytes + b * 20;
    const float d = Fp16ToFloat(LoadU16(blk));
    const float m = Fp16ToFloat(LoadU16(blk + 2));
    const std::uint8_t* qs = blk + 4;
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      y[j] = static_cast<float>(qs[j] & 0x0F) * d + m;
      y[j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
    }
  }
}

void DequantizeQ5_0(const void* src, float* dst, std::size_t k) {
  // d (F16), 32 high bits, then 16 bytes of low nibbles per 32 elements.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 32; ++b) {
    const std::uint8_t* blk = bytes + b * 22;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint32_t qh = LoadU32(blk + 2);
    const std::uint8_t* qs = blk + 6;
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      const int xh0 = static_cast<int>(((qh >> j) << 4) & 0x10U);
      const int xh1 = static_cast<int>((qh >> (j + 12)) & 0x10U);
      y[j] = static_cast<float>(((qs[j] & 0x0F) | xh0) - 16) * d;
      y[j + 16] = static_cast<float>(((qs[j] >> 4) | xh1) - 16) * d;
    }
  }
}

void DequantizeQ2_K(const void* src, float* dst, std::size_t k) {
  // scales[16] (low nibble scale, high nibble minimum), qs[64], d, dmin.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 84;
    const std::uint8_t* scales = blk;
    const std::uint8_t* q = blk + 16;
    const float d = Fp16ToFloat(LoadU16(blk + 80));
    const float min = Fp16ToFloat(LoadU16(blk + 82));
    float* y = dst + b * 256;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        for (int half = 0; half < 2; ++half) {
          const std::uint8_t sc = scales[is++];
          const float dl = d * static_cast<float>(sc & 0x0F);
          const float ml = min * static_cast<float>(sc >> 4);
          for (int l = 0; l < 16; ++l) {
            *y++ =
                dl * static_cast<float>((q[l + 16 * half] >> shift) & 3) - ml;
          }
        }
        shift += 2;
      }
      q += 32;
    }
  }
}

void DequantizeIQ2_XXS(const void* src, float* dst, std::size_t k) {
  // d, then per 32 elements two words: four grid indices, and seven-bit
  // sign groups with a four-bit scale in the top bits.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 66;
    const float d = Fp16ToFloat(LoadU16(blk));
    float* y = dst + b * 256;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const std::uint8_t* q = blk + 2 + 8 * ib32;
      const std::uint32_t aux1 = LoadU32(q + 4);
      const float db = d * (0.5F + static_cast<float>(aux1 >> 28)) * 0.25F;
      for (int l = 0; l < 4; ++l) {
        const auto* grid =
            reinterpret_cast<const std::uint8_t*>(&kIq2xxsGrid[q[l]]);
        const std::uint8_t signs = kKsignsIq2xs[(aux1 >> (7 * l)) & 127U];
        for (int j = 0; j < 8; ++j)
          y[j] = db * static_cast<float>(grid[j]) * SignOf(signs, j);
        y += 8;
      }
    }
  }
}

void DequantizeIQ2_XS(const void* src, float* dst, std::size_t k) {
  // d, qs[32] (nine-bit grid index, seven-bit sign group), scales[8].
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 74;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint8_t* scales = blk + 66;
    float* y = dst + b * 256;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const float db[2] = {
          d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F,
          d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F};
      for (int l = 0; l < 4; ++l) {
        const std::uint16_t qs = LoadU16(blk + 2 + 2 * (4 * ib32 + l));
        const auto* grid =
            reinterpret_cast<const std::uint8_t*>(&kIq2xsGrid[qs & 511U]);
        const std::uint8_t signs = kKsignsIq2xs[qs >> 9];
        for (int j = 0; j < 8; ++j)
          y[j] = db[l / 2] * static_cast<float>(grid[j]) * SignOf(signs, j);
        y += 8;
      }
    }
  }
}

void DequantizeIQ2_S(const void* src, float* dst, std::size_t k) {
  // d, qs[32] grid lows, signs[32], qh[8] grid highs, scales[8].
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 82;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint8_t* qs = blk + 2;
    const std::uint8_t* signs = qs + 32;
    const std::uint8_t* qh = blk + 66;
    const std::uint8_t* scales = blk + 74;
    float* y = dst + b * 256;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const float db[2] = {
          d * (0.5F + static_cast<float>(scales[ib32] & 0x0F)) * 0.25F,
          d * (0.5F + static_cast<float>(scales[ib32] >> 4)) * 0.25F};
      for (int l = 0; l < 4; ++l) {
        const float dl = db[l / 2];
        const std::uint32_t index =
            qs[l] |
            ((static_cast<std::uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300U);
        const auto* grid =
            reinterpret_cast<const std::uint8_t*>(&gufo::quant::kIq2sGrid[index]);
        for (int j = 0; j < 8; ++j)
          y[j] = dl * static_cast<float>(grid[j]) * SignOf(signs[l], j);
        y += 8;
      }
      qs += 4;
      signs += 4;
    }
  }
}

void DequantizeIQ3_XXS(const void* src, float* dst, std::size_t k) {
  // d, qs[64] grid indices, then per 32 elements a word of sign groups
  // with a four-bit scale in the top bits.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 98;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint8_t* qs = blk + 2;
    const std::uint8_t* scales_and_signs = blk + 2 + 64;
    float* y = dst + b * 256;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
      const std::uint32_t aux = LoadU32(scales_and_signs + 4 * ib32);
      const float db = d * (0.5F + static_cast<float>(aux >> 28)) * 0.5F;
      for (int l = 0; l < 4; ++l) {
        const std::uint8_t signs = kKsignsIq2xs[(aux >> (7 * l)) & 127U];
        const auto* grid1 =
            reinterpret_cast<const std::uint8_t*>(&gufo::quant::kIq3xxsGrid[qs[2 * l]]);
        const auto* grid2 =
            reinterpret_cast<const std::uint8_t*>(&gufo::quant::kIq3xxsGrid[qs[2 * l + 1]]);
        for (int j = 0; j < 4; ++j) {
          y[j] = db * static_cast<float>(grid1[j]) * SignOf(signs, j);
          y[j + 4] = db * static_cast<float>(grid2[j]) * SignOf(signs, j + 4);
        }
        y += 8;
      }
      qs += 8;
    }
  }
}

void DequantizeIQ1_S(const void* src, float* dst, std::size_t k) {
  // d, qs[32] grid lows, qh[8] words: three grid highs per group, a
  // three-bit scale and the delta's sign.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 50;
    const float d = Fp16ToFloat(LoadU16(blk));
    const std::uint8_t* qs = blk + 2;
    float* y = dst + b * 256;
    for (int ib = 0; ib < 8; ++ib) {
      const std::uint16_t qh = LoadU16(blk + 34 + 2 * ib);
      const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
      const float delta = (qh & 0x8000U) != 0 ? -kIq1Delta : kIq1Delta;
      for (int l = 0; l < 4; ++l) {
        const std::uint32_t index = qs[l] | (((qh >> (3 * l)) & 7U) << 8);
        for (int j = 0; j < 8; ++j)
          y[j] = dl * (Iq1sGrid(index, j) + delta);
        y += 8;
      }
      qs += 4;
    }
  }
}

void DequantizeIQ1_M(const void* src, float* dst, std::size_t k) {
  // qs[32] grid lows, qh[16] (grid highs and delta signs per group of
  // eight), scales[8]: twelve three-bit scales and the F16 d in the top
  // nibbles of the four scale words.
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / 256; ++b) {
    const std::uint8_t* blk = bytes + b * 56;
    const std::uint8_t* qs = blk;
    const std::uint8_t* qh = blk + 32;
    std::uint16_t sc[4];
    for (int i = 0; i < 4; ++i)
      sc[i] = LoadU16(blk + 48 + 2 * i);
    const auto packed =
        static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                   ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = Fp16ToFloat(packed);
    float* y = dst + b * 256;
    for (int ib = 0; ib < 8; ++ib) {
      const float dl1 =
          d *
          static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
      const float dl2 =
          d *
          static_cast<float>(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
      const std::uint32_t idx[4] = {
          qs[0] | ((static_cast<std::uint32_t>(qh[0]) << 8) & 0x700U),
          qs[1] | ((static_cast<std::uint32_t>(qh[0]) << 4) & 0x700U),
          qs[2] | ((static_cast<std::uint32_t>(qh[1]) << 8) & 0x700U),
          qs[3] | ((static_cast<std::uint32_t>(qh[1]) << 4) & 0x700U)};
      const float delta[4] = {(qh[0] & 0x08) != 0 ? -kIq1Delta : kIq1Delta,
                              (qh[0] & 0x80) != 0 ? -kIq1Delta : kIq1Delta,
                              (qh[1] & 0x08) != 0 ? -kIq1Delta : kIq1Delta,
                              (qh[1] & 0x80) != 0 ? -kIq1Delta : kIq1Delta};
      for (int l = 0; l < 4; ++l) {
        const float dl = l < 2 ? dl1 : dl2;
        for (int j = 0; j < 8; ++j)
          y[j] = dl * (Iq1sGrid(idx[l], j) + delta[l]);
        y += 8;
      }
      qs += 4;
      qh += 2;
    }
  }
}

bool DequantizeRow(core::GgmlType type, const void* src, float* dst,
                   std::size_t k) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  switch (type) {
    case core::GgmlType::kF32:
      std::memcpy(dst, src, k * sizeof(float));
      return true;
    case core::GgmlType::kF16:
      for (std::size_t i = 0; i < k; ++i)
        dst[i] = Fp16ToFloat(LoadU16(bytes + 2 * i));
      return true;
    case core::GgmlType::kBF16:
      for (std::size_t i = 0; i < k; ++i) {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(LoadU16(bytes + 2 * i)) << 16;
        std::memcpy(dst + i, &bits, sizeof(float));
      }
      return true;
    default:
      break;
  }
  const std::size_t block = QuantizedBlockElements(type);
  if (block == 0 || k % block != 0) {
    return false;
  }
  switch (type) {
    case core::GgmlType::kQ4_0:
      DequantizeQ4_0(src, dst, k);
      return true;
    case core::GgmlType::kQ4_1:
      DequantizeQ4_1(src, dst, k);
      return true;
    case core::GgmlType::kQ5_0:
      DequantizeQ5_0(src, dst, k);
      return true;
    case core::GgmlType::kQ5_1:
      DequantizeQ5_1(src, dst, k);
      return true;
    case core::GgmlType::kQ8_0:
      DequantizeQ8_0(src, dst, k);
      return true;
    case core::GgmlType::kQ2_K:
      DequantizeQ2_K(src, dst, k);
      return true;
    case core::GgmlType::kQ3_K:
      DequantizeQ3_K(src, dst, k);
      return true;
    case core::GgmlType::kQ4_K:
      DequantizeQ4_K(src, dst, k);
      return true;
    case core::GgmlType::kQ5_K:
      DequantizeQ5_K(src, dst, k);
      return true;
    case core::GgmlType::kQ6_K:
      DequantizeQ6_K(src, dst, k);
      return true;
    case core::GgmlType::kQ8_K:
      DequantizeQ8_K(src, dst, k);
      return true;
    case core::GgmlType::kIQ2_XXS:
      DequantizeIQ2_XXS(src, dst, k);
      return true;
    case core::GgmlType::kIQ2_XS:
      DequantizeIQ2_XS(src, dst, k);
      return true;
    case core::GgmlType::kIQ2_S:
      DequantizeIQ2_S(src, dst, k);
      return true;
    case core::GgmlType::kIQ3_XXS:
      DequantizeIQ3_XXS(src, dst, k);
      return true;
    case core::GgmlType::kIQ3_S:
      DequantizeIQ3_S(src, dst, k);
      return true;
    case core::GgmlType::kIQ1_S:
      DequantizeIQ1_S(src, dst, k);
      return true;
    case core::GgmlType::kIQ1_M:
      DequantizeIQ1_M(src, dst, k);
      return true;
    case core::GgmlType::kIQ4_NL:
      DequantizeIQ4_NL(src, dst, k);
      return true;
    case core::GgmlType::kIQ4_XS:
      DequantizeIQ4_XS(src, dst, k);
      return true;
    default:
      return false;
  }
}

// IQ3_XXS / IQ2_S / Q2_0 per-element reference dequantizers

float Iq3XxsValue(const block_iq3_xxs& block, std::size_t index) noexcept {
  const std::size_t ib32 = index / 32;
  const std::size_t l = (index / 8) % 4;
  const std::size_t j = index % 8;
  std::uint32_t aux32 = 0;
  std::memcpy(&aux32, block.scales_and_signs + 4 * ib32, sizeof(aux32));
  const float d = Fp16ToFloat(block.d);
  const float db = d * (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
  const std::uint8_t grid_index = block.qs[8 * ib32 + 2 * l + (j >= 4 ? 1 : 0)];
  const std::uint8_t magnitude =
      static_cast<std::uint8_t>(gufo::quant::kIq3xxsGrid[grid_index] >> (8 * (j % 4)));
  const std::uint8_t signs = kSignsIq2xs[(aux32 >> (7 * l)) & 127];
  return db * static_cast<float>(magnitude) *
         ((signs & kSignMaskIq2xs[j]) != 0 ? -1.0f : 1.0f);
}

float Iq2SValue(const block_iq2_s& block, std::size_t index) noexcept {
  const std::size_t ib32 = index / 32;
  const std::size_t l = (index / 8) % 4;
  const std::size_t j = index % 8;
  const float d = Fp16ToFloat(block.d);
  const std::uint8_t scales_byte = block.scales[ib32];
  const float db =
      d * (0.5f + ((l / 2) == 0 ? (scales_byte & 0xF) : (scales_byte >> 4))) *
      0.25f;
  const std::uint8_t* bytes =
      reinterpret_cast<const std::uint8_t*>(&block);  // NOLINT
  const std::uint8_t qs_byte = bytes[2 + 4 * ib32 + l];
  const std::size_t grid_index =
      static_cast<std::size_t>(qs_byte) |
      ((static_cast<std::size_t>(bytes[66 + ib32]) << (8 - 2 * l)) & 0x300);
  const std::uint8_t magnitude = static_cast<std::uint8_t>(
      gufo::quant::kIq2sGrid[grid_index] >> (8 * static_cast<std::size_t>(j)));
  // Signs overlap qs at offset QK_K/8 (byte 34 of the block); one bit per
  // element (1 = negative), no sign-table indirection.
  const std::uint8_t signs = bytes[34 + 4 * ib32 + l];
  return db * static_cast<float>(magnitude) *
         ((signs & kSignMaskIq2xs[j]) != 0 ? -1.0f : 1.0f);
}

float Q2_0Value(const block_q2_0& block, std::size_t index) noexcept {
  const std::uint8_t q = (block.qs[index / 4] >> (2 * (index % 4))) & 0x03U;
  return (static_cast<float>(static_cast<int>(q)) - 1.0f) *
         Fp16ToFloat(block.d);
}

float Q6KValue(const block_q6_K& block, std::size_t index) noexcept {
  // Two half-blocks of 128; four rows of 32 per half. The 6-bit value is a
  // 4-bit low nibble from ql (column-interleaved) and a 2-bit high piece
  // from qh, biased by -32 and scaled per 32-element row.
  const std::size_t h = (index / 128) & 1U;
  const std::size_t within = index % 128;
  const std::size_t seg = within / 32;
  const std::size_t lane = within % 32;
  const std::uint8_t qbyte = block.ql[(seg & 1U) ? h * 64 + 32 + lane
                                                 : h * 64 + lane];
  const std::uint8_t low = (seg & 2U) ? (qbyte >> 4U) : (qbyte & 0x0FU);
  const std::uint8_t high = (block.qh[h * 32 + lane] >> (2 * seg)) & 0x3U;
  const std::int8_t s = block.scales[h * 8 + lane / 16 + seg * 2];
  return Fp16ToFloat(block.d) * static_cast<float>(s) *
         static_cast<float>(static_cast<int>((high << 4U) | low) - 32);
}

}  // namespace gufo::quant
