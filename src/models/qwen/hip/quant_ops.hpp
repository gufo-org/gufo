#ifndef GUFO_MODELS_QWEN_HIP_QUANT_OPS_HPP_
#define GUFO_MODELS_QWEN_HIP_QUANT_OPS_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>

namespace gufo::hip {

// Shared quantized block layouts + quant row-dot helpers for the decode and
// prefill GPU paths, matching the CPU oracles in ggml_dequant.cpp.

// block_q8_0 layout: {__half d; int8_t qs[32];}, 34 bytes, QK=32. Dominant
// quant in the Q8_K_L model.
constexpr std::size_t kQ8_0BlockSize = 32;

struct Q8_0Block {
  __half d;
  std::int8_t qs[kQ8_0BlockSize];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

// opt-r4-q5k-q6k: block_q5_K layout ({half d; half dmin; uint8 scales[12];
// uint8 qh[32]; uint8 qs[128]; }, 176 bytes, QK_K=256). Dequant-to-fp path:
// Q5KValue read a fp16 d/dmin and the packed 4-bit qs + qh sign bits, then
// apply the (scale, min) pair from GetQKScaleMin. Requires K % 256 == 0.
constexpr std::size_t kQ5KBlockSize = 256;

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

// opt-r4-q5k-q6k: block_q6_K layout ({uint8 ql[128]; uint8 qh[64]; int8
// scales[16]; half d; }, 210 bytes, QK_K=256). Dequant-to-fp path: Q6KValue
// reads a fp16 d and the packed 4-bit ql + 2-bit qh, then applies the int8
// scale. Requires K % 256 == 0.
constexpr std::size_t kQ6KBlockSize = 256;

struct Q6KBlock {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  __half d;
};
static_assert(sizeof(Q6KBlock) == 210, "block_q6_K must be 210 bytes");

// opt-c1xx-q8k-gemv: block_q8_K layout ({ float d; int8_t qs[256];
// int16_t bsums[16]; }, 292 bytes, QK_K=256). The dot runs AT Q8, not by
// casting weights to fp16: each 256-wide fp32 x block is quantized in-register
// to int8 (d_x = max|x|, scale = d_x/127) and accumulated as an int32 integer
// MAC with qs; the single fp scale (d_w * scale_x) is applied at block end.
// Requires K % 256 == 0.
constexpr std::size_t kQ8KBlockSize = 256;

struct Q8KBlock {
  float d;
  std::int8_t qs[kQ8KBlockSize];
  std::int16_t bsums[16];
};
static_assert(sizeof(Q8KBlock) == 292, "block_q8_K must be 292 bytes");

// opt-q4kxl: the Unsloth UD-Q4_K_XL Qwen3.8-27B shard mixes Q5_K, IQ4_XS,
// Q4_K, Q6_K, IQ4_NL, Q3_K, IQ3_S and Q8_0. Pre-dequantizing the K-quants to
// BF16 the way the Q8_K_XL loader does would turn a 17.5 GB shard back into
// 54 GB and throw away the whole reason for using Q4, so every one of these
// types is decoded in-kernel from its packed form.

// block_q4_K layout ({half d; half dmin; uint8 scales[12]; uint8 qs[128];},
// 144 bytes, QK_K=256). Eight 32-wide sub-blocks, each with a 6-bit scale and
// a 6-bit minimum: value = d*sc*q - dmin*m.
constexpr std::size_t kQ4KBlockSize = 256;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

// block_q3_K layout ({uint8 hmask[32]; uint8 qs[64]; uint8 scales[12];
// half d;}, 110 bytes, QK_K=256). Sixteen 16-wide sub-blocks; the scale is a
// 6-bit biased value and the quant is 2 low bits plus an inverted high bit.
constexpr std::size_t kQ3KBlockSize = 256;

struct Q3KBlock {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  __half d;
};
static_assert(sizeof(Q3KBlock) == 110, "block_q3_K must be 110 bytes");

// block_iq4_nl layout ({half d; uint8 qs[16];}, 18 bytes, QK=32). Non-linear
// 4-bit codebook, one scale per 32 elements.
constexpr std::size_t kIQ4NLBlockSize = 32;

struct IQ4NLBlock {
  __half d;
  std::uint8_t qs[16];
};
static_assert(sizeof(IQ4NLBlock) == 18, "block_iq4_nl must be 18 bytes");

// block_iq4_xs layout ({half d; uint16 scales_h; uint8 scales_l[4];
// uint8 qs[128];}, 136 bytes, QK_K=256). Same codebook as IQ4_NL with eight
// 6-bit sub-block scales split across scales_l (low 4) and scales_h (high 2).
constexpr std::size_t kIQ4XSBlockSize = 256;

struct IQ4XSBlock {
  __half d;
  std::uint16_t scales_h;
  std::uint8_t scales_l[4];
  std::uint8_t qs[128];
};
static_assert(sizeof(IQ4XSBlock) == 136, "block_iq4_xs must be 136 bytes");

// block_iq3_s layout ({half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32];
// uint8 scales[4];}, 110 bytes, QK_K=256). Each group of four elements is one
// entry of the 512-word grid; signs are carried per group of eight.
constexpr std::size_t kIQ3SBlockSize = 256;

struct IQ3SBlock {
  __half d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};
static_assert(sizeof(IQ3SBlock) == 110, "block_iq3_s must be 110 bytes");

// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS. Mirrors
// quant::kValuesIq4Nl (kvalues_iq4nl in ggml-common.h).
__device__ inline constexpr std::int8_t kDeviceValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// opt-r2-q8_0: type-aware quant block geometry helpers so RowDot can own the
// per-type row stride (bytes per block and elements per block).
__device__ inline std::size_t QuantBlockBytes(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return sizeof(Q8_0Block);
    case core::GgmlType::kQ8_K:
      return sizeof(Q8KBlock);
    case core::GgmlType::kQ5_K:
      return sizeof(Q5KBlock);
    case core::GgmlType::kQ6_K:
      return sizeof(Q6KBlock);
    case core::GgmlType::kQ4_K:
      return sizeof(Q4KBlock);
    case core::GgmlType::kQ3_K:
      return sizeof(Q3KBlock);
    case core::GgmlType::kIQ4_NL:
      return sizeof(IQ4NLBlock);
    case core::GgmlType::kIQ4_XS:
      return sizeof(IQ4XSBlock);
    case core::GgmlType::kIQ3_S:
      return sizeof(IQ3SBlock);
    default:
      return 0;
  }
}

__device__ inline std::size_t QuantBlockQK(core::GgmlType t) {
  switch (t) {
    case core::GgmlType::kQ8_0:
      return kQ8_0BlockSize;
    case core::GgmlType::kQ8_K:
      return kQ8KBlockSize;
    case core::GgmlType::kQ5_K:
      return kQ5KBlockSize;
    case core::GgmlType::kQ6_K:
      return kQ6KBlockSize;
    case core::GgmlType::kQ4_K:
      return kQ4KBlockSize;
    case core::GgmlType::kQ3_K:
      return kQ3KBlockSize;
    case core::GgmlType::kIQ4_NL:
      return kIQ4NLBlockSize;
    case core::GgmlType::kIQ4_XS:
      return kIQ4XSBlockSize;
    case core::GgmlType::kIQ3_S:
      return kIQ3SBlockSize;
    default:
      return 0;
  }
}

// opt-r4-q5k-q6k: unpack the (scale, minimum) pair for Q5_K from the packed
// 12-byte scales array. index 0..7; mirrors CPU GetQ4ScaleMin in
// ggml_dequant.cpp (Q5_K uses the same scale/min encoding as Q4_K).
template<bool SharedScales = false>
__device__ inline void GetQKScaleMin(std::size_t index,
                                     const std::uint8_t* packed,
                                     std::uint8_t& sc,
                                     std::uint8_t& m) noexcept {
  // Batched projections amortize one extra byte load across token rows and
  // benefit from avoiding divergent execution of the two encodings. Scalar
  // GEMV keeps the fewer-load spelling.
  if constexpr (SharedScales) {
    const std::size_t base = index & 3U;
    const unsigned low = packed[base];
    const unsigned middle = packed[base + 4];
    const unsigned high = packed[base + 8];
    const unsigned upper_mask = 0U - static_cast<unsigned>(index >> 2U);
    const unsigned lower_scale = low & 63U;
    const unsigned upper_scale = (high & 15U) | ((low >> 6U) << 4U);
    const unsigned lower_minimum = middle & 63U;
    const unsigned upper_minimum = (high >> 4U) | ((middle >> 6U) << 4U);
    sc = static_cast<std::uint8_t>(
        lower_scale ^ ((lower_scale ^ upper_scale) & upper_mask));
    m = static_cast<std::uint8_t>(
        lower_minimum ^ ((lower_minimum ^ upper_minimum) & upper_mask));
    return;
  }
  if (index < 4) {
    sc = packed[index] & 0x3FU;
    m = packed[index + 4] & 0x3FU;
    return;
  }
  sc = static_cast<std::uint8_t>((packed[index + 4] & 0x0FU) |
                                 ((packed[index - 4] >> 6U) << 4U));
  m = static_cast<std::uint8_t>((packed[index + 4] >> 4U) |
                                ((packed[index] >> 6U) << 4U));
}

// opt-r4-q5k-q6k: dequantize one Q5_K element (index i in [0,256)) to fp32.
// Mirrors CPU Q5Value in ggml_dequant.cpp exactly.
__device__ inline float Q5KValue(const Q5KBlock& block,
                                 std::size_t index) noexcept {
  const std::size_t gg = index / 64;  // 0..3
  const std::size_t wv = index % 64;  // 0..63
  const std::size_t lane = wv % 32;   // 0..31
  const bool lohalf = (wv < 32);
  const std::uint8_t qb = block.qs[(gg * 32) + lane];
  const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const std::uint8_t quant = static_cast<std::uint8_t>(
      quant4 + (((block.qh[lane] >> bit) & 1U) ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);        // 0..7
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQKScaleMin(sis, block.scales, sc, m);
  return __half2float(block.d) * static_cast<float>(sc) *
             static_cast<float>(quant) -
         __half2float(block.dmin) * static_cast<float>(m);
}

// opt-r4-q5k-q6k: dequantize one Q6_K element (index i in [0,256)) to fp32.
// Mirrors CPU Q6Value in ggml_dequant.cpp exactly.
__device__ inline float Q6KValue(const Q6KBlock& block,
                                 std::size_t index) noexcept {
  const std::size_t half = index / 128;
  const std::size_t within = index % 128;
  const std::size_t segment = within / 32;
  const std::size_t lane = within % 32;
  const std::size_t ql_base = half * 64;
  const std::uint8_t qh_byte = block.qh[(half * 32) + lane];

  std::uint8_t low = 0;
  std::uint8_t high = 0;
  switch (segment) {
    case 0:
      low = block.ql[ql_base + lane] & 0x0FU;
      high = (qh_byte >> 0U) & 0x03U;
      break;
    case 1:
      low = block.ql[ql_base + 32 + lane] & 0x0FU;
      high = (qh_byte >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[ql_base + lane] >> 4U;
      high = (qh_byte >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[ql_base + 32 + lane] >> 4U;
      high = (qh_byte >> 6U) & 0x03U;
      break;
  }
  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return __half2float(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

// opt-q4kxl: 512-entry IQ3_S grid, verbatim from ggml-common.h. Each word packs
// four uint8 magnitudes for one group of four elements.
__device__ inline constexpr std::uint32_t kDeviceIq3sGrid[512] = {
    0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU,
    0x01010301U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU,
    0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U,
    0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U,
    0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU,
    0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U,
    0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU,
    0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU,
    0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU,
    0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU,
    0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU,
    0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U,
    0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U,
    0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U,
    0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U,
    0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U,
    0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U,
    0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U,
    0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU,
    0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U,
    0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U,
    0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U,
    0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U,
    0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U,
    0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU,
    0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U,
    0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U,
    0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U,
    0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU,
    0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U,
    0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U,
    0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU,
    0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U,
    0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U,
    0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U,
    0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U,
    0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU,
    0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U,
    0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U,
    0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU,
    0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU,
    0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U,
    0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U,
    0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U,
    0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U,
    0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U,
    0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U,
    0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU,
    0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU,
    0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U,
    0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU,
    0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U,
    0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U,
    0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U,
    0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U,
    0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U,
    0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU,
    0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU,
    0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U,
    0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U,
    0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U,
    0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U,
    0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U,
    0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U,
    0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U,
    0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U,
    0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U,
    0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U,
    0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U,
    0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U,
    0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U,
    0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U,
    0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U,
    0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U,
    0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU,
    0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U,
    0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U,
    0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U,
    0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U,
    0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U,
    0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U,
    0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U,
    0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU,
    0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U,
    0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U,
    0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U,
    0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U,
    0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU,
    0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U,
    0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U,
    0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU,
    0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU,
    0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U,
    0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U,
    0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U,
    0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU,
    0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U,
    0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U,
    0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU,
    0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U,
    0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U,
    0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U,
    0x0f0d0703U, 0x0f0f0101U,
};

// opt-q4kxl: unpack the sixteen 6-bit biased Q3_K scales from the packed
// 12-byte array. Mirrors CPU UnpackQ3Scales in ggml_dequant.cpp.
__device__ inline std::int8_t GetQ3KScale(std::size_t index,
                                          const std::uint8_t* packed) noexcept {
  const unsigned low =
      index < 8 ? (packed[index] & 0x0FU) : ((packed[index - 8] >> 4U) & 0x0FU);
  const unsigned high = (packed[8 + (index % 4)] >> (2U * (index / 4))) & 0x03U;
  return static_cast<std::int8_t>(static_cast<int>(low | (high << 4U)) - 32);
}

// opt-q4kxl: unpack one IQ4_XS sub-block scale (index 0..7).
__device__ inline int GetIQ4XSScale(std::size_t index,
                                    const IQ4XSBlock& block) noexcept {
  const int lo = static_cast<int>(
      (block.scales_l[index / 2] >> (4U * (index % 2))) & 0x0FU);
  const int hi = static_cast<int>((block.scales_h >> (2U * index)) & 0x03U);
  return (lo | (hi << 4)) - 32;
}

/// Byte-wise subtraction of a small constant from four packed bytes.
/// Setting bit 7 of every byte first stops a borrow from crossing into the next
/// byte, and the final XOR removes it again, so each byte independently becomes
/// its two's-complement value. Every use here feeds a value that is known to
/// fit in int8 after the subtraction.
__device__ inline std::uint32_t SubBytes(std::uint32_t v,
                                         std::uint32_t bias) noexcept {
  return ((v | 0x80808080U) - bias) ^ 0x80808080U;
}

/// Applies the 16-entry IQ4 codebook to four 4-bit indices at once.
///
/// The naive form indexes kDeviceValuesIq4Nl per element, which the compiler
/// cannot keep in registers under a divergent index -- it becomes four
/// dependent loads per word, and IQ4_XS is 22% of the shard. Two V_PERM_B32
/// cover the two halves of the table (each perm is an 8-entry byte gather over
/// the {high, low} dword pair) and a byte-wide mask built from bit 3 of each
/// index selects between them.
__device__ inline std::uint32_t Iq4NlLookup4(std::uint32_t nibbles) noexcept {
  constexpr std::uint32_t kTable0 = 0xBFAD9881U;  // entries 0..3
  constexpr std::uint32_t kTable1 = 0xF6EADDCFU;  // entries 4..7
  constexpr std::uint32_t kTable2 = 0x26190D01U;  // entries 8..11
  constexpr std::uint32_t kTable3 = 0x71594535U;  // entries 12..15
  const std::uint32_t low_index = nibbles & 0x07070707U;
  const std::uint32_t lower =
      __builtin_amdgcn_perm(kTable1, kTable0, low_index);
  const std::uint32_t upper =
      __builtin_amdgcn_perm(kTable3, kTable2, low_index);
  const std::uint32_t select = ((nibbles & 0x08080808U) >> 3U) * 0xFFU;
  return (lower & ~select) | (upper & select);
}

// opt-q4kxl: one 16-element sub-block decoded into int8 codes plus an affine
// pair, so that
//
//     w[j] = scale * code[j] - offset      for j in [0, 16)
//
// This is the single primitive the decode GEMV, the prefill WMMA GEMM and the
// embedding lookup all build on. Sixteen is the smallest scale group any of
// these formats uses (Q3_K and Q6_K carry one scale per sixteen elements, the
// rest per thirty-two or per super-block) AND it is exactly the K depth of one
// `wmma_16x16x16_iu8` fragment, so the GEMM never needs a scale that changes
// inside a fragment. `offset` is non-zero only for the two formats that encode
// an explicit minimum (Q4_K and Q5_K); everything else is symmetric.
//
// A 32-element variant was tried, sharing the super-block header (and the
// branchy scale/min extraction) between the two halves, which Q4_K, Q5_K and
// IQ4_XS decode identically. It left prefill unchanged and cost decode 13%
// (tg128 8.25 -> 7.18): holding thirty-two decoded bytes plus three scales
// pushed the GEMV kernels from 88 to 120-136 VGPRs, and those kernels are
// DRAM-bound, so the halved occupancy cost more memory-level parallelism than
// the saved header work returned. The decode path wants the small footprint
// more than it wants the shared header.
struct QuantSub16 {
  union {
    std::int8_t q[16];
    std::uint32_t w[4];
  };
  float scale;
  float offset;
};

/// Loads sixteen consecutive weight bytes as four 32-bit words. Quant blocks
/// are not word aligned (block_q4_K is 144 bytes, block_q6_K 210), so this must
/// go through memcpy rather than a reinterpreting load.
__device__ inline void LoadQuantWords16(const std::uint8_t* __restrict__ src,
                                        std::uint32_t out[4]) noexcept {
  __builtin_memcpy(out, src, 16);
}

/// Decodes the `sub16`-th 16-element sub-block of the row starting at `row`.
/// `row` must point at the first block of the row for `type`.
template<bool SharedScales = false>
__device__ inline void DecodeQuantSub16(core::GgmlType type,
                                        const void* __restrict__ row,
                                        std::size_t sub16,
                                        QuantSub16& out) noexcept {
  out.offset = 0.0F;
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const auto& blk = static_cast<const Q8_0Block*>(row)[sub16 / 2];
      const std::size_t base = (sub16 % 2) * 16;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        out.q[j] = blk.qs[base + j];
      }
      out.scale = __half2float(blk.d);
      return;
    }
    case core::GgmlType::kQ4_K: {
      const auto& blk = static_cast<const Q4KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t base = ((sb32 / 2) * 32) + ((sub16 % 2) * 16);
      const unsigned shift = 4U * static_cast<unsigned>(sb32 & 1U);
      std::uint32_t packed[4];
      LoadQuantWords16(blk.qs + base, packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out.w[i] = (packed[i] >> shift) & 0x0F0F0F0FU;
      }
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      GetQKScaleMin<SharedScales>(sb32, blk.scales, sc, m);
      out.scale = __half2float(blk.d) * static_cast<float>(sc);
      out.offset = __half2float(blk.dmin) * static_cast<float>(m);
      return;
    }
    case core::GgmlType::kQ5_K: {
      const auto& blk = static_cast<const Q5KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t lane0 = (sub16 % 2) * 16;
      const std::size_t base = ((sb32 / 2) * 32) + lane0;
      const unsigned shift = ((sb32 % 2) == 0) ? 0U : 4U;
      std::uint32_t packed[4];
      std::uint32_t high[4];
      LoadQuantWords16(blk.qs + base, packed);
      LoadQuantWords16(blk.qh + lane0, high);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (packed[i] >> shift) & 0x0F0F0F0FU;
        const std::uint32_t bit =
            (high[i] >> static_cast<unsigned>(sb32)) & 0x01010101U;
        out.w[i] = low | (bit << 4U);
      }
      std::uint8_t sc = 0;
      std::uint8_t m = 0;
      GetQKScaleMin<SharedScales>(sb32, blk.scales, sc, m);
      out.scale = __half2float(blk.d) * static_cast<float>(sc);
      out.offset = __half2float(blk.dmin) * static_cast<float>(m);
      return;
    }
    case core::GgmlType::kQ6_K: {
      const auto& blk = static_cast<const Q6KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t half = sb32 / 4;
      const std::size_t segment = sb32 % 4;
      const std::size_t lane0 = (sub16 % 2) * 16;
      const std::size_t ql_base =
          (half * 64) + (((segment & 1U) != 0U) ? 32 : 0);
      const unsigned ql_shift = (segment >= 2) ? 4U : 0U;
      const unsigned qh_shift = 2U * static_cast<unsigned>(segment);
      std::uint32_t low_bits[4];
      std::uint32_t high_bits[4];
      LoadQuantWords16(blk.ql + ql_base + lane0, low_bits);
      LoadQuantWords16(blk.qh + (half * 32) + lane0, high_bits);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (low_bits[i] >> ql_shift) & 0x0F0F0F0FU;
        const std::uint32_t high = (high_bits[i] >> qh_shift) & 0x03030303U;
        // Each byte is 0..63 here, so the biased subtract lands in [-32, 31].
        out.w[i] = SubBytes(low | (high << 4U), 0x20202020U);
      }
      const std::size_t scale_index = (half * 8) + (segment * 2) + (sub16 % 2);
      out.scale =
          __half2float(blk.d) * static_cast<float>(blk.scales[scale_index]);
      return;
    }
    case core::GgmlType::kQ3_K: {
      const auto& blk = static_cast<const Q3KBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t half = sb32 / 4;
      const std::size_t scale_pair = sb32 % 4;
      const std::size_t lane0 = (sub16 % 2) * 16;
      const unsigned shift = 2U * static_cast<unsigned>(scale_pair);
      const unsigned high_shift =
          static_cast<unsigned>((half * 4) + scale_pair);
      std::uint32_t low_bits[4];
      std::uint32_t mask_bits[4];
      LoadQuantWords16(blk.qs + (half * 32) + lane0, low_bits);
      LoadQuantWords16(blk.hmask + lane0, mask_bits);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t low = (low_bits[i] >> shift) & 0x03030303U;
        const std::uint32_t bit = (mask_bits[i] >> high_shift) & 0x01010101U;
        // low is 0..3 and (bit << 2) is 0 or 4, so the sum stays inside its
        // byte and the biased subtract lands in [-4, 3].
        out.w[i] = SubBytes(low + (bit << 2U), 0x04040404U);
      }
      const std::size_t scale_index =
          (half * 8) + (scale_pair * 2) + (sub16 % 2);
      out.scale = __half2float(blk.d) *
                  static_cast<float>(GetQ3KScale(scale_index, blk.scales));
      return;
    }
    case core::GgmlType::kIQ4_NL: {
      const auto& blk = static_cast<const IQ4NLBlock*>(row)[sub16 / 2];
      const unsigned shift = 4U * static_cast<unsigned>(sub16 % 2);
      std::uint32_t packed[4];
      LoadQuantWords16(blk.qs, packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out.w[i] = Iq4NlLookup4((packed[i] >> shift) & 0x0F0F0F0FU);
      }
      out.scale = __half2float(blk.d);
      return;
    }
    case core::GgmlType::kIQ4_XS: {
      const auto& blk = static_cast<const IQ4XSBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t base = sb32 * 16;
      const unsigned shift = 4U * static_cast<unsigned>(sub16 % 2);
      std::uint32_t packed[4];
      LoadQuantWords16(blk.qs + base, packed);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        out.w[i] = Iq4NlLookup4((packed[i] >> shift) & 0x0F0F0F0FU);
      }
      out.scale =
          __half2float(blk.d) * static_cast<float>(GetIQ4XSScale(sb32, blk));
      return;
    }
    case core::GgmlType::kIQ3_S: {
      const auto& blk = static_cast<const IQ3SBlock*>(row)[sub16 / 16];
      const std::size_t sb32 = (sub16 / 2) % 8;
      const std::size_t within0 = (sub16 % 2) * 16;
      const std::uint8_t qh_byte = blk.qh[sb32];
      // Irregular by construction: every group of four elements is a separate
      // 512-entry grid lookup, so this one stays element-wise. IQ3_S is a
      // single tensor in the shard.
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const std::size_t within = within0 + static_cast<std::size_t>(j);
        const std::size_t l = within / 8;
        const std::size_t jj = within % 8;
        const std::size_t grid_half = jj / 4;
        const int shift = static_cast<int>(8 - (2 * l) - grid_half);
        const std::uint32_t grid_index =
            static_cast<std::uint32_t>(
                blk.qs[(sb32 * 8) + (2 * l) + grid_half]) |
            ((static_cast<std::uint32_t>(qh_byte) << shift) & 256U);
        const auto* grid =
            reinterpret_cast<const std::uint8_t*>(&kDeviceIq3sGrid[grid_index]);
        const int magnitude = static_cast<int>(grid[jj % 4]);
        const bool negate = (blk.signs[(sb32 * 4) + l] & (1U << jj)) != 0U;
        out.q[j] = static_cast<std::int8_t>(negate ? -magnitude : magnitude);
      }
      const std::uint8_t scale_byte = blk.scales[sb32 / 2];
      const int nibble =
          (sb32 % 2 == 0) ? (scale_byte & 0x0FU) : (scale_byte >> 4U);
      out.scale = __half2float(blk.d) * static_cast<float>(1 + (2 * nibble));
      return;
    }
    default:
      break;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out.w[i] = 0;
  }
  out.scale = 0.0F;
}

/// Bytes occupied by one logical row of `k` elements in `type`.
__device__ inline std::size_t QuantRowBytes(core::GgmlType type,
                                            std::size_t k) noexcept {
  const std::size_t qk = QuantBlockQK(type);
  return (qk == 0) ? 0 : ((k / qk) * QuantBlockBytes(type));
}

/// Single-element dequantization for any DecodeQuantSub16-backed format.
/// `row` points at the first block of the row that contains `index`.
__device__ inline float QuantBlockElement(core::GgmlType type,
                                          const void* __restrict__ row,
                                          std::size_t index) noexcept {
  QuantSub16 sub;
  DecodeQuantSub16(type, row, index / 16, sub);
  return (sub.scale * static_cast<float>(sub.q[index % 16])) - sub.offset;
}

/// True when `type` is decoded through DecodeQuantSub16 rather than one of the
/// hand-rolled per-type branches in QuantWarpBlockDot.
__device__ inline bool IsSub16DecodedQuant(core::GgmlType t) noexcept {
  return t == core::GgmlType::kQ4_K || t == core::GgmlType::kQ5_K ||
         t == core::GgmlType::kQ6_K || t == core::GgmlType::kQ3_K ||
         t == core::GgmlType::kIQ4_NL || t == core::GgmlType::kIQ4_XS ||
         t == core::GgmlType::kIQ3_S;
}

// opt-r7-decode-parallel: warp-parallel quant row-dot, templated on the input
// element type TX (float in decode, hip_bfloat16 for the prefill bf16 path).
// ONE warp cooperates: all 32 lanes process a slice of the row's quant blocks
// (b in [b0,b1)), each lane owning 8 elements per 256-wide block
// (8_K/5_K/6_K) or 1 element per 32-wide block (Q8_0), then reduce within each
// block via __shfl_xor. Returns this warp's partial row-dot ONLY IN LANE 0
// (other lanes return 0), so it is a drop-in for both the one-warp-per-row
// kernels (a) and the one-block-per-row kernels (b) whose existing __shfl_xor
// / shared-memory reductions then broadcast/combine it correctly. With
// num_warps == 1 the warp processes the entire row. Mirrors Q8KBlockGEMVKernel
// exactly. The only TX-dependent code is the x reads, wrapped in
// static_cast<float> so TX=float is identity and TX=hip_bfloat16 converts.
template<typename TX>
__device__ inline float QuantWarpBlockDot(
    core::GgmlType type, const void* __restrict__ base, std::size_t row_idx,
    const TX* __restrict__ x, std::size_t K, std::size_t lane_id,
    std::size_t warp_id, std::size_t num_warps) {
  const std::size_t qk = QuantBlockQK(type);
  const std::size_t block_bytes = QuantBlockBytes(type);
  const char* row =
      static_cast<const char*>(base) + (row_idx * (K / qk * block_bytes));
  const std::size_t num_blocks = K / qk;
  const std::size_t chunk = (num_blocks + num_warps - 1) / num_warps;
  const std::size_t b0 = warp_id * chunk;
  const std::size_t b1 = (b0 + chunk < num_blocks) ? (b0 + chunk) : num_blocks;
  float sumf = 0.0F;
  switch (type) {
    case core::GgmlType::kQ8_0: {
      const auto* row_blocks = reinterpret_cast<const Q8_0Block*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q8_0Block& wblk = row_blocks[b];
        const float d_w = __half2float(wblk.d);
        const TX* xb = x + (b * kQ8_0BlockSize);
        float dot = static_cast<float>(wblk.qs[lane_id]) *
                    static_cast<float>(xb[lane_id]);
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor(dot, off);
        }
        sumf += d_w * dot;
      }
      break;
    }
    case core::GgmlType::kQ8_K: {
      const auto* row_blocks = reinterpret_cast<const Q8KBlock*>(row);
      for (std::size_t b = b0; b < b1; ++b) {
        const Q8KBlock& wblk = row_blocks[b];
        const float d_w = wblk.d;
        const TX* xb = x + (b * kQ8KBlockSize);
        float dot = 0.0F;
#pragma unroll
        for (std::size_t k = 0; k < 8; ++k) {
          const std::size_t idx = lane_id + (32 * k);
          dot += static_cast<float>(wblk.qs[idx]) * static_cast<float>(xb[idx]);
        }
        for (int off = 16; off > 0; off >>= 1) {
          dot += __shfl_xor(dot, off);
        }
        sumf += d_w * dot;
      }
      break;
    }
    // opt-q4kxl: Q5_K and Q6_K used to have hand-rolled branches here that
    // dequantized one element at a time (eight per lane per 256-wide block) via
    // Q5KValue / Q6KValue. They now take the sub16 route below, which unpacks a
    // whole 16-element group with word-wide bit operations instead of per
    // element. Together they are 15.0G of the UD-Q4_K_XL shard's 27.2G
    // elements, so this is the dominant decode path for that model. The Q8_K_XL
    // model is unaffected: its Q5_K/Q6_K tensors are pre-dequantized to BF16 at
    // load time and never reach this kernel.
    default: {
      // opt-q4kxl: generic route for every format decoded through
      // DecodeQuantSub16 (Q4_K, Q3_K, IQ4_NL, IQ4_XS, IQ3_S). Instead of
      // splitting one 256-wide block across all 32 lanes eight elements at a
      // time, each lane owns whole 16-element sub-blocks. A sub-block is the
      // unit the decoder produces, so this pays the scale/nibble unpacking once
      // per sixteen values rather than once per value, and every lane's sixteen
      // weight bytes are contiguous.
      //
      // Sixteen and not thirty-two: these kernels are DRAM-bound, and the wider
      // decode raised them from 88 to 120-136 VGPRs, which cut occupancy and
      // measured 13% slower end to end.
      if (!IsSub16DecodedQuant(type)) {
        break;
      }
      const std::size_t s0 = b0 * (qk / 16);
      const std::size_t s1 = b1 * (qk / 16);
      QuantSub16 sub;
      for (std::size_t sb = s0 + lane_id; sb < s1; sb += 32) {
        DecodeQuantSub16(type, row, sb, sub);
        const TX* xb = x + (sb * 16);
        float dot = 0.0F;
        float xsum = 0.0F;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const float xv = static_cast<float>(xb[j]);
          dot += static_cast<float>(sub.q[j]) * xv;
          xsum += xv;
        }
        sumf += (sub.scale * dot) - (sub.offset * xsum);
      }
      for (int off = 16; off > 0; off >>= 1) {
        sumf += __shfl_xor(sumf, off);
      }
      break;
    }
  }
  if (lane_id != 0)
    return 0.0F;
  return sumf;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_QUANT_OPS_HPP_
