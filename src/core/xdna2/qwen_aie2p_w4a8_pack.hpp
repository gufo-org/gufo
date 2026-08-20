// Copyright (C) 2026 Strix Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pure-CPU W4A8 SHQ4-T16 packing and reference kernels for the Qwen3.8-27B
// MTP AIE2P (XDNA2) program family. Header-only so CPU oracle tests and the
// matrix bench can run without XRT; the host driver
// (src/core/xdna2/qwen_aie2p_w4a8.cpp) uses the same builders to fill the NPU
// input tensors.
//
// Record contracts (identical to the AIE kernel in
// programs/qwen_aie2p_w4a8/gemm.cc):
//
//   weight_record (16 lanes x 3072 B, one 256-K block):
//     [0,2048)   codes:  8 groups x 2 half-tiles x (16 k x 16 lanes) uint4,
//                        byte = k*8 + lane/2, low nibble = even lane
//     [2048,2560) scales: 8 groups x 16 lanes float32 (BF16 width expanded)
//     [2560,3072) zcorr:  8 groups x 16 lanes float32
//   input_record (5 x 256 int8, one 256-K block, one chunk of 4 M rows):
//     rows 0..3  codes:  group g, half h, row r, k: offset
//                        g*128 + h*64 + r*16 + k (int8, per-row codes)
//     row 4 [0,128)      activation scales: 4 rows x 8 groups float32
//     row 4 [128,256)    int32 sums:         4 rows x 8 groups
//   output_acc (4 x 16 float32, row-major), accumulated per block.
//
// Numerical contract (docs/QUANTIZATION.md "Dynamic A8 activation contract"
// and "W4A8 accumulation"):
//   a_scale = amax/127, qa = clamp(rne(x/a_scale), -127, 127), amax==0 -> 0
//   dot[r,n,g]   = sum_k int32(qa[r,k]) * int32(qw[n,k])   (exact INT32)
//   corrected    = dot - int32(z[n,g]) * sum_k int32(qa[r,k])
//   partial      = float(corrected) * float(a_scale[r,g]) * float(w_scale[n,g])
//   group partials accumulated in FP32 in (block, group) order.
//
// Weight interpretations (host packing only; one kernel ELF):
//   u4z:  codes = SHQ4 U4Z plane (uint4), wscale = BF16 scale,
//         zcorr = UINT4 zero point decoded to FP32
//   s4:   codes = SHQ4 S4 plane bytes read unsigned (u = s + 8),
//         wscale = BF16 scale, zcorr = 8.0  (dot_u - 8*asum == dot_s)
//   q4k:  Q4_K lossless: codes = Q4_K nibbles, wscale = d*scale6 FP32 exact,
//         zcorr = dmin*min6 FP32 exact

#ifndef STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_PACK_H_
#define STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_PACK_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace strix::xdna2::w4a8 {

// ---------------------------------------------------------------------------
// Geometry constants
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kBlockElements = 256;
inline constexpr std::uint32_t kGroupElements = 32;
inline constexpr std::uint32_t kGroupsPerBlock = kBlockElements / kGroupElements;
inline constexpr std::uint32_t kMmulM = 4;
inline constexpr std::uint32_t kMmulK = 16;
inline constexpr std::uint32_t kMmulN = 16;
inline constexpr std::uint32_t kMmulActivationElements = kMmulM * kMmulK;
inline constexpr std::uint32_t kMmulWeightElements = kMmulK * kMmulN;

inline constexpr std::uint32_t kWeightRecordCodesBytes = 2048;
inline constexpr std::uint32_t kWeightRecordScaleOffset = 2048;
inline constexpr std::uint32_t kWeightRecordZcorrOffset = 2560;
inline constexpr std::uint32_t kWeightRecordBytes =
    kWeightRecordZcorrOffset + (kGroupsPerBlock * kMmulN * sizeof(float));

inline constexpr std::uint32_t kInputRecordCodesBytes = 1024;
inline constexpr std::uint32_t kInputRecordScaleOffset = 1024;
inline constexpr std::uint32_t kInputRecordSumOffset = 1152;
inline constexpr std::uint32_t kInputRecordBytes = kInputRecordSumOffset +
    (kMmulM * kGroupsPerBlock * sizeof(std::int32_t));

inline constexpr std::uint32_t kOutputAccElements = kMmulM * kMmulN;

static_assert(kWeightRecordBytes == 3072);
static_assert(kInputRecordBytes == 1280);
static_assert(kInputRecordSumOffset +
                  (kMmulM * kGroupsPerBlock * sizeof(std::int32_t)) ==
              kInputRecordBytes);

// AIE2P array: 8 columns x 4 rows, 16 output lanes per core.
inline constexpr std::uint32_t kArrayColumns = 8;
inline constexpr std::uint32_t kArrayRows = 4;
inline constexpr std::uint32_t kLanesPerArrayLayer =
    kArrayColumns * kArrayRows * kMmulN;  // 512

enum class WeightMode {
  kU4Z,
  kS4,
  kQ4K,
};

struct Shape {
  std::uint32_t m{0};
  std::uint32_t n{0};
  std::uint32_t k{0};

  [[nodiscard]] std::uint32_t Blocks() const noexcept {
    return (k + kBlockElements - 1) / kBlockElements;
  }
  [[nodiscard]] std::uint32_t TilesPerColumn() const noexcept {
    return (n + kLanesPerArrayLayer - 1) / kLanesPerArrayLayer;
  }
  [[nodiscard]] std::uint32_t Rounds() const noexcept {
    return (m + kMmulM - 1) / kMmulM;
  }
};

// ---------------------------------------------------------------------------
// BF16 helpers (RNE, ties-to-even -- matches tools/strix/shq.py)
// ---------------------------------------------------------------------------

inline std::uint16_t RoundToBf16Rne(float value) noexcept {
  const std::uint32_t u = [&]() {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }();
  const std::uint32_t lsb = (u >> 16U) & 1U;
  const std::uint32_t rounded = (u + 0x7FFFU + lsb) >> 16U;
  return static_cast<std::uint16_t>(rounded);
}

inline float Bf16ToFloat32(std::uint16_t id) noexcept {
  std::uint32_t bits = static_cast<std::uint32_t>(id) << 16U;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// GGML Q4_K stores d/dmin as FP16 (10-bit mantissa). Exact for FP16-representable
// values; mirrors src/core/quant/ggml_dequant.cpp Fp16ToFloat. Subnormals: an FP16
// subnormal (exp==0, mant!=0) is 2^-24 * mant, representable in FP32.
inline float Fp16ToFloat32(std::uint16_t id) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(id & 0x8000U) << 16U;
  const std::uint32_t exp = (id >> 10U) & 0x1FU;
  const std::uint32_t mant = id & 0x03FFU;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant != 0) {
      // subnormal: 2^-24 * mant
      const float f = (static_cast<float>(mant) / 1024.0F) * std::ldexp(1.0F, -14);
      std::memcpy(&bits, &f, sizeof(bits));
      bits |= sign;
    } else {
      bits = sign;  // +/- zero
    }
  } else if (exp == 0x1FU) {
    bits = sign | 0x7F800000U | (mant << 13U);  // inf/NaN
  } else {
    bits = sign | ((exp + 112U) << 23U) | (mant << 13U);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// ---------------------------------------------------------------------------
// Weight record builders
// ---------------------------------------------------------------------------

// Decoded view of one 256-K block x 16 lanes, ready to be packed or compared
// against the source quantization dequantization.
struct WeightBlockSpec {
  std::uint8_t codes[kGroupsPerBlock][2][kMmulK][kMmulN]{};
  float wscale[kGroupsPerBlock][kMmulN]{};
  float zcorr[kGroupsPerBlock][kMmulN]{};
};

// Decoded value accessor shared by packers and the decoder.
inline std::uint8_t& WeightSpecCode(WeightBlockSpec& spec, std::uint32_t group,
                                    std::uint32_t half, std::uint32_t k,
                                    std::uint32_t lane) noexcept {
  return spec.codes[group][half][k][lane];
}

inline const std::uint8_t& WeightSpecCode(const WeightBlockSpec& spec,
                                          std::uint32_t group,
                                          std::uint32_t half,
                                          std::uint32_t k,
                                          std::uint32_t lane) noexcept {
  return spec.codes[group][half][k][lane];
}

inline std::uint8_t* WeightRecordNibble(std::uint8_t* record,
                                        std::uint32_t group,
                                        std::uint32_t half,
                                        std::uint32_t k,
                                        std::uint32_t lane) noexcept {
  const std::uint32_t element_index = (k * kMmulN) + lane;
  return record + ((group * 2U + half) * kMmulWeightElements / 2U) +
         (element_index / 2U);
}

inline void SetWeightRecordNibble(std::uint8_t* record, std::uint32_t group,
                                  std::uint32_t half, std::uint32_t k,
                                  std::uint32_t lane,
                                  std::uint8_t value) noexcept {
  assert(group < kGroupsPerBlock && half < 2 && k < kMmulK && lane < kMmulN);
  assert(value <= 0x0FU);
  auto* byte = WeightRecordNibble(record, group, half, k, lane);
  if ((lane & 1U) == 0U) {
    *byte = static_cast<std::uint8_t>((*byte & 0xF0U) | value);
  } else {
    *byte = static_cast<std::uint8_t>((*byte & 0x0FU) | (value << 4U));
  }
}

inline std::uint8_t GetWeightRecordNibble(const std::uint8_t* record,
                                          std::uint32_t group,
                                          std::uint32_t half,
                                          std::uint32_t k,
                                          std::uint32_t lane) noexcept {
  const std::uint8_t byte = *WeightRecordNibble(
      const_cast<std::uint8_t*>(record), group, half, k, lane);
  return (lane & 1U) == 0U ? static_cast<std::uint8_t>(byte & 0x0FU)
                           : static_cast<std::uint8_t>(byte >> 4U);
}

inline float* WeightRecordScale(std::uint8_t* record, std::uint32_t group,
                                std::uint32_t lane) noexcept {
  return reinterpret_cast<float*>(
      record + kWeightRecordScaleOffset + ((group * kMmulN) + lane) * 4U);
}

inline const float* WeightRecordScale(const std::uint8_t* record,
                                      std::uint32_t group,
                                      std::uint32_t lane) noexcept {
  return reinterpret_cast<const float*>(
      record + kWeightRecordScaleOffset + ((group * kMmulN) + lane) * 4U);
}

inline float* WeightRecordZcorr(std::uint8_t* record, std::uint32_t group,
                                std::uint32_t lane) noexcept {
  return reinterpret_cast<float*>(
      record + kWeightRecordZcorrOffset + ((group * kMmulN) + lane) * 4U);
}

inline const float* WeightRecordZcorr(const std::uint8_t* record,
                                      std::uint32_t group,
                                      std::uint32_t lane) noexcept {
  return reinterpret_cast<const float*>(
      record + kWeightRecordZcorrOffset + ((group * kMmulN) + lane) * 4U);
}

inline void PackWeightBlock(const WeightBlockSpec& spec,
                            std::uint8_t* record) noexcept {
  std::memset(record, 0, kWeightRecordBytes);
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          SetWeightRecordNibble(record, group, half, k, lane,
                                spec.codes[group][half][k][lane]);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      *WeightRecordScale(record, group, lane) = spec.wscale[group][lane];
      *WeightRecordZcorr(record, group, lane) = spec.zcorr[group][lane];
    }
  }
}

// ---------------------------------------------------------------------------
// Q4_K -> record (lossless: codes = Q4_K nibbles, wscale = d*scale6,
// zcorr = dmin*min6). Source layout is one GGML Q4_K block of 256 elements
// (144 bytes): u16 d, u16 dmin, u8 scales[12], u8 qs[128].
// ---------------------------------------------------------------------------

namespace detail {

#pragma pack(push, 1)
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
#pragma pack(pop)

inline void GetQ4ScaleMin(std::uint32_t group, const std::uint8_t* packed,
                          std::uint8_t& scale,
                          std::uint8_t& minimum) noexcept {
  if (group < 4) {
    scale = packed[group] & 0x3FU;
    minimum = packed[group + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[group + 4] & 0x0FU) |
                                    ((packed[group - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[group + 4] >> 4U) |
                                      ((packed[group] >> 6U) << 4U));
}

}  // namespace detail

static_assert(sizeof(detail::BlockQ4K) == 144);

inline WeightBlockSpec DecodeQ4KBlock(const std::uint8_t* src) noexcept {
  const auto& block = *reinterpret_cast<const detail::BlockQ4K*>(src);
  const float d = Fp16ToFloat32(block.d);
  const float dmin = Fp16ToFloat32(block.dmin);
  WeightBlockSpec spec;
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    const std::uint32_t pair = group / 2;
    const bool high_nibble = (group & 1U) != 0U;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    detail::GetQ4ScaleMin(group, block.scales, scale, minimum);
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        const std::uint32_t lane_in_group = (half * kMmulK) + k;
        const std::uint8_t source =
            block.qs[(pair * kGroupElements) + lane_in_group];
        const std::uint8_t quantized =
            high_nibble ? static_cast<std::uint8_t>(source >> 4U)
                        : static_cast<std::uint8_t>(source & 0x0FU);
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][k][lane] = quantized;
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = d * static_cast<float>(scale);
      spec.zcorr[group][lane] = dmin * static_cast<float>(minimum);
    }
  }
  return spec;
}

// Decode one Q4_K row block (256 K elements of output row `row`) into the
// `lane`-th 16-element K slice of a tile's WeightBlockSpec. The source tensor
// is row-major: block index within the row = `block`, so the caller supplies
// the pointer to row `lane`'s block.
inline void DecodeQ4KRowIntoSpec(const std::uint8_t* src_row_block,
                                 std::uint32_t lane,
                                 WeightBlockSpec& spec) noexcept {
  const auto& block = *reinterpret_cast<const detail::BlockQ4K*>(src_row_block);
  const float d = Fp16ToFloat32(block.d);
  const float dmin = Fp16ToFloat32(block.dmin);
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    const std::uint32_t pair = group / 2;
    const bool high_nibble = (group & 1U) != 0U;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    detail::GetQ4ScaleMin(group, block.scales, scale, minimum);
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        const std::uint32_t lane_in_group = (half * kMmulK) + k;
        const std::uint8_t source =
            block.qs[(pair * kGroupElements) + lane_in_group];
        const std::uint8_t quantized =
            high_nibble ? static_cast<std::uint8_t>(source >> 4U)
                        : static_cast<std::uint8_t>(source & 0x0FU);
        WeightSpecCode(spec, group, half, k, lane) = quantized;
      }
    }
    spec.wscale[group][lane] = d * static_cast<float>(scale);
    spec.zcorr[group][lane] = dmin * static_cast<float>(minimum);
  }
}

// Fast path: one Q4_K block -> one weight record (no intermediate decode).
// NOTE: one 256-element Q4_K block covers one output row's K range; a record
// holds 16 DISTINCT rows, so callers must pass the 16 row blocks for the
// (tile, block) pair (see DecodeQ4KRowIntoSpec for a lane-wise builder).
inline void PackQ4KBlockToRecord(const std::uint8_t* src_q4k_block,
                                 std::uint8_t* record) noexcept {
  // Convenience: broadcast the single row block across all 16 lanes (a tile
  // record normally holds 16 DISTINCT rows; use DecodeQ4KRowIntoSpec per lane
  // for real tensors).
  WeightBlockSpec spec;
  for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
    DecodeQ4KRowIntoSpec(src_q4k_block, lane, spec);
  }
  PackWeightBlock(spec, record);
}

// ---------------------------------------------------------------------------
// SHQ4 (U4Z/S4) planes -> record.
//
// SHQ4 plane layouts (docs/QUANTIZATION.md, tools/strix/shq.py):
//   codes[nt][kg][k16][lane][kp8]: byte = ((gidx*k16_per + k16)*16 + lane)*8
//                                  + kp, low nibble = k = kp*2, high = kp*2+1
//   scales u16 BF16[gidx*16 + lane]
//   zeros u8 packed[gidx*8 + lane/2], low nibble = even lane (U4Z only)
// where gidx = n_tile*k_groups_total + kg. S4 zero plane is absent and every
// lane uses zcorr = 8.0 (codes are stored two's-complement S4, so reading the
// nibble unsigned gives u = s + 8).
// ---------------------------------------------------------------------------

struct Shq4PlaneRefs {
  const std::uint8_t* codes{nullptr};
  const std::uint8_t* scales{nullptr};  // BF16 uint16 bytes
  const std::uint8_t* zeros{nullptr};   // nullptr for S4
  std::uint32_t k_groups_total{0};      // total K groups in source tensor
  std::uint32_t k16_per_group{2};       // group_size / 16
};

inline void PackShq4BlockToRecord(WeightMode mode, const Shq4PlaneRefs& planes,
                                  std::uint32_t n_tile,
                                  std::uint32_t group_offset,
                                  std::uint8_t* record) noexcept {
  assert(mode == WeightMode::kU4Z || mode == WeightMode::kS4);
  std::memset(record, 0, kWeightRecordBytes);
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    const std::uint32_t gidx = n_tile * planes.k_groups_total +
                               group_offset + group;
    for (std::uint32_t half = 0; half < 2; ++half) {
      const std::uint32_t k16 = half;
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        const std::uint32_t kp = k / 2;
        const bool high = (k & 1U) != 0U;
        const std::uint32_t lane_base =
            ((gidx * planes.k16_per_group + k16) * kMmulN) * 8U + kp;
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          const std::uint8_t source = planes.codes[lane_base + (lane * 8U)];
          const std::uint8_t value =
              high ? static_cast<std::uint8_t>(source >> 4U)
                   : static_cast<std::uint8_t>(source & 0x0FU);
          SetWeightRecordNibble(record, group, half, k, lane, value);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      const std::uint16_t scale_id =
          reinterpret_cast<const std::uint16_t*>(planes.scales)
              [gidx * kMmulN + lane];
      *WeightRecordScale(record, group, lane) = Bf16ToFloat32(scale_id);
      if (mode == WeightMode::kU4Z) {
        const std::uint8_t packed =
            planes.zeros[(gidx * kMmulN + lane) / 2U];
        const std::uint8_t zero =
            (lane & 1U) == 0U ? static_cast<std::uint8_t>(packed & 0x0FU)
                              : static_cast<std::uint8_t>(packed >> 4U);
        *WeightRecordZcorr(record, group, lane) = static_cast<float>(zero);
      } else {
        *WeightRecordZcorr(record, group, lane) = 8.0F;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Activation quantization + input record builder
// ---------------------------------------------------------------------------

// Quantize 4 rows (chunk) of `k` floats each into one input record. `k` must
// be <= kBlockElements; elements [k, 256) are zero-coded with zero scales
// (padded tail K, dequantizes to zero). Empty rows are treated as all-zero
// (padded tail M rows). Per (row, group):
//   a_scale = amax/127 (amax == 0 -> 0, codes 0, sum 0),
//   codes = clamp(rne(x/a_scale),-127,127)
//   scale stored BF16-RNE rounded FP32, sum stored INT32.
inline void QuantizeActivationBlock(
    const std::span<const float> rows[kMmulM], std::uint32_t k,
    std::uint8_t* record) {
  assert(k <= kBlockElements);
  std::memset(record, 0, kInputRecordBytes);
  const std::uint32_t groups_in_block = (k + kGroupElements - 1) / kGroupElements;
  for (std::uint32_t row = 0; row < kMmulM; ++row) {
    if (rows[row].empty()) {
      continue;
    }
    for (std::uint32_t group = 0; group < groups_in_block; ++group) {
      const std::uint32_t group_start = group * kGroupElements;
      const std::uint32_t group_end =
          std::min(std::uint32_t{group_start + kGroupElements}, k);
      float maximum = 0.0F;
      for (std::uint32_t lane = group_start; lane < group_end; ++lane) {
        maximum = std::max(maximum, std::abs(rows[row][lane]));
      }
      const float scale_real = maximum == 0.0F ? 0.0F : maximum / 127.0F;
      std::int32_t quantized_sum = 0;
      // amax == 0 -> scale 0, codes 0, sum 0 (skip the division by a zero
      // scale: 0/0 is NaN and (int)NaN is UB).
      const bool active = scale_real != 0.0F;
      for (std::uint32_t half = 0; half < 2; ++half) {
        auto* tile = record + ((group * 2U + half) * kMmulActivationElements);
        for (std::uint32_t kk = 0; kk < kMmulK; ++kk) {
          const std::uint32_t lane = (half * kMmulK) + kk;
          std::int8_t quantized = 0;
          if (active && (group_start + lane) < k) {
            const auto rounded =
                static_cast<int>(std::nearbyint(rows[row][group_start + lane] /
                                                scale_real));
            quantized = static_cast<std::int8_t>(
                std::clamp(rounded, -127, 127));
            quantized_sum += quantized;
          }
          tile[(row * kMmulK) + kk] = quantized;
        }
      }
      float* scale_slot =
          reinterpret_cast<float*>(record + kInputRecordScaleOffset +
                                   ((row * kGroupsPerBlock) + group) * 4U);
      *scale_slot = scale_real == 0.0F
                        ? 0.0F
                        : Bf16ToFloat32(RoundToBf16Rne(scale_real));
      auto* sum_slot =
          reinterpret_cast<std::int32_t*>(record + kInputRecordSumOffset +
                                          ((row * kGroupsPerBlock) + group) *
                                              sizeof(std::int32_t));
      *sum_slot = quantized_sum;
    }
  }
}

// ---------------------------------------------------------------------------
// AIE-mirror reference: one block's contribution accumulated into a 4x16
// FP32 output accumulator. Mirrors gemm.cc: INT32 dot over uint4 codes,
// weight-scale/zcorr/activation-scale epilogue, FP32 accumulation in
// (group, row) order.
// ---------------------------------------------------------------------------

inline void ReferenceBlockAccumulate(const std::uint8_t* weight_record,
                                     const std::uint8_t* input_record,
                                     float* output_acc) {
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    std::int32_t dot[kMmulM][kMmulN]{};
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t r = 0; r < kMmulM; ++r) {
        for (std::uint32_t k = 0; k < kMmulK; ++k) {
          const int a = static_cast<int>(static_cast<std::int8_t>(
              input_record[((group * 2U + half) * kMmulActivationElements) +
                           (r * kMmulK) + k]));
          for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
            dot[r][lane] +=
                a * static_cast<int>(GetWeightRecordNibble(weight_record,
                                                           group, half, k, lane));
          }
        }
      }
    }
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      const float activation_scale = reinterpret_cast<const float*>(
          input_record + kInputRecordScaleOffset +
          ((r * kGroupsPerBlock) + group) * 4U)[0];
      const float quantized_sum =
          static_cast<float>(reinterpret_cast<const std::int32_t*>(
              input_record + kInputRecordSumOffset +
              ((r * kGroupsPerBlock) + group) * sizeof(std::int32_t))[0]);
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        const float weight_scale =
            *WeightRecordScale(weight_record, group, lane);
        const float weight_zero = *WeightRecordZcorr(weight_record, group, lane);
        const float weighted_dot =
            static_cast<float>(dot[r][lane]) * weight_scale;
        const float zero_correction = weight_zero * quantized_sum;
        const float corrected = weighted_dot - zero_correction;
        output_acc[(r * kMmulN) + lane] += corrected * activation_scale;
      }
    }
  }
}

// Reference GEMM over the logical weight-record array (record `n_tile *
// blocks + block`, 3072 B each) and a row-major float activation matrix.
// `out` is row-major over the PADDED output grid (`rounds*kMmulM` rows x
// `n_tiles*kMmulN` columns); padded rows/lanes are zero. Rows outside the
// logical shape are activation-quantized to zero (all-zero codes/scales).
inline void ReferenceGemm(std::span<const float> activations, Shape shape,
                          const std::uint8_t* weight_records, float* out) {
  const std::uint32_t blocks = shape.Blocks();
  const std::uint32_t n_tiles = (shape.n + kMmulN - 1) / kMmulN;
  const std::uint32_t rounds = shape.Rounds();
  const std::uint32_t k_pad = blocks * kBlockElements;
  const std::uint32_t n_padded = n_tiles * kMmulN;
  std::memset(out, 0, static_cast<std::size_t>(rounds * kMmulM) * n_tiles *
                          kMmulN * sizeof(float));
  std::vector<std::uint8_t> input_record(kInputRecordBytes);
  std::vector<std::vector<float>> padded_rows(
      kMmulM, std::vector<float>(k_pad));
  // ReferenceBlockAccumulate mirrors the per-core 4x16 tile accumulator of
  // gemm.cc; scatter each tile's 64 floats into the row-major padded grid.
  std::array<float, kOutputAccElements> tile_acc{};
  for (std::uint32_t rounds_ix = 0; rounds_ix < rounds; ++rounds_ix) {
    const std::uint32_t m_base = rounds_ix * kMmulM;
    // Pad each activation row to k_pad (zero codes for tail K).
    std::span<const float> rows[kMmulM];
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      const std::uint32_t logical_row = m_base + r;
      if (logical_row < shape.m) {
        std::fill(padded_rows[r].begin(), padded_rows[r].end(), 0.0F);
        std::ranges::copy(
            activations.subspan(static_cast<std::size_t>(logical_row) *
                                    shape.k,
                                shape.k),
            padded_rows[r].begin());
        rows[r] = std::span<const float>(padded_rows[r].data(), k_pad);
      } else {
        rows[r] = {};
      }
    }
    for (std::uint32_t block = 0; block < blocks; ++block) {
      std::span<const float> block_rows[kMmulM];
      for (std::uint32_t r = 0; r < kMmulM; ++r) {
        block_rows[r] = rows[r].empty()
                            ? std::span<const float>{}
                            : rows[r].subspan(static_cast<std::size_t>(block) *
                                                  kBlockElements,
                                              kBlockElements);
      }
      QuantizeActivationBlock(block_rows, kBlockElements, input_record.data());
      for (std::uint32_t n_tile = 0; n_tile < n_tiles; ++n_tile) {
        const std::uint8_t* record =
            weight_records + (static_cast<std::size_t>(n_tile) * blocks +
                              block) * kWeightRecordBytes;
        tile_acc.fill(0.0F);
        ReferenceBlockAccumulate(record, input_record.data(),
                                 tile_acc.data());
        for (std::uint32_t r = 0; r < kMmulM; ++r) {
          float* row_out =
              out + (static_cast<std::size_t>(m_base + r) * n_padded);
          row_out += static_cast<std::size_t>(n_tile) * kMmulN;
          for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
            if (n_tile * kMmulN + lane >= shape.n) {
              continue;  // padded lanes stay zero
            }
            row_out[lane] += tile_acc[(r * kMmulN) + lane];
          }
        }
      }
    }
  }
}

}  // namespace strix::xdna2::w4a8

#endif  // STRIX_CORE_XDNA2_QWEN_AIE2P_W4A8_PACK_H_