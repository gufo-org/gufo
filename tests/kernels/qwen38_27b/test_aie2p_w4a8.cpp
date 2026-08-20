// Copyright (C) 2026 Strix Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIE2P W4A8 SHQ4-T16 kernel tests for the Qwen3.8-27B MTP family (issue #35,
// M004-C007).
//
// CPU oracle gates (no XRT required):
//   - BF16 RNE ties-to-even rounding (matches tools/strix/shq.py)
//   - exhaustive uint4 nibble decode and record round trip
//   - SHQ4 U4Z and S4 plane -> AIE weight-record transposition and semantics
//   - Q4_K lossless record packing vs DotProductQ4_K oracle
//   - dynamic group-32 INT8 activation quantization contract
//   - INT32 exact accumulation vs an int64 oracle
//   - padded tail K (float-exact) and tail M rows
//   - batch-1 GEMV vs packed GEMM row equivalence
//   - group-boundary quantization
//
// Hardware gates (ENGINE_ENABLE_XRT only; SKIP semantics when the XDNA2/XRT
// device or the STRIX_MTP_MODEL weights are unavailable; STRIX_REQUIRE_XDNA2=1
// turns the device skip into a hard failure):
//   - M=1 eh_proj on the real 27B MTP weights vs the CPU oracle
//     (ReferenceGemm over Q4_K-lossless records from pack.hpp)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/xdna2/qwen_aie2p_w4a8_pack.hpp"

#ifdef ENGINE_ENABLE_XRT
#include "src/core/xdna2/qwen_aie2p_w4a8.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/qwen_mtp_reference.hpp"
#include "src/core/xdna2/device.h"
#endif

namespace {

using namespace strix::xdna2::w4a8;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                std::string_view message) {
  if (!(std::abs(actual - expected) <= tolerance)) {
    throw std::runtime_error(std::string(message) + " (got " +
                             std::to_string(actual) + ", expected " +
                             std::to_string(expected) + ", diff " +
                             std::to_string(std::abs(actual - expected)) +
                             " > " + std::to_string(tolerance) + ")");
  }
}

// ---------------------------------------------------------------------------
// BF16 RNE rounding (datapoints from tools/strix/conformance.py)
// ---------------------------------------------------------------------------

void TestBf16Rne() {
  struct Case {
    float value;
    std::uint16_t expected;
  };
  constexpr Case kCases[] = {
      {0.5F, 0x3F00U},   {1.0F, 0x3F80U},
      {1.5F, 0x3FC0U},   {-2.0F, 0xC000U},
      {0.0F, 0x0000U},   {-0.0F, 0x8000U},
      // RNE rounds 65504 (exactly representable as BF16) to 2^16 = 0x4780;
      // the constant-row expectation is skipped (shq.py has no such case).
      {65504.0F, 0x4780U},
  };
  for (const auto& test : kCases) {
    Expect(RoundToBf16Rne(test.value) == test.expected,
           "BF16 RNE exact value");
  }
  // Ties-to-even: bit16 == 0 -> rounds down (even), bit16 == 1 -> rounds up.
  {
    std::uint32_t halfway_down = 0x3F800000U + 0x8000U;  // bit16 == 0
    float value;
    std::memcpy(&value, &halfway_down, sizeof(value));
    Expect(RoundToBf16Rne(value) == 0x3F80U, "BF16 tie rounds to even (down)");
    // Midpoint between BF16 0x3F81 and 0x3F82 (bit16 == 1) -> rounds to 0x3F82.
    std::uint32_t halfway_up = 0x3F818000U;  // bit16 == 1, bits below zero
    std::memcpy(&value, &halfway_up, sizeof(value));
    Expect(RoundToBf16Rne(value) == 0x3F82U, "BF16 tie rounds to even (up)");
  }
  for (std::uint32_t bits = 0; bits < 0x10000U; ++bits) {
    if ((bits & 0x7C00U) == 0x7C00U && (bits & 0x03FFU) != 0U) {
      continue;  // NaN payloads compare unequal as floats
    }
    Expect(Bf16ToFloat32(static_cast<std::uint16_t>(bits)) ==
               Bf16ToFloat32(RoundToBf16Rne(
                   Bf16ToFloat32(static_cast<std::uint16_t>(bits)))),
           "BF16 roundtrip is idempotent for exact BF16 values");
  }
}

// ---------------------------------------------------------------------------
// Exhaustive nibble decode and record round trip
// ---------------------------------------------------------------------------

void TestNibbleExhaustive() {
  std::array<std::uint8_t, kWeightRecordBytes> record{};
  for (std::uint32_t value = 0; value <= 0x0FU; ++value) {
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      SetWeightRecordNibble(record.data(), 0, 0, 0, lane,
                            static_cast<std::uint8_t>(value));
      Expect(GetWeightRecordNibble(record.data(), 0, 0, 0, lane) == value,
             "nibble round trip per lane");
    }
    Expect(record[0] == static_cast<std::uint8_t>((value << 4U) | value),
           "byte-level low/high nibble packing");
  }
}

// ---------------------------------------------------------------------------
// SHQ4 plane -> AIE record transposition
// ---------------------------------------------------------------------------

// Write an SHQ4-T16 plane from a WeightBlockSpec (inverse of the packer):
// codes [gidx][k16][lane][kp8] (low nibble = k = kp*2), scales u16 BF16, zeros
// u8 packed lane pairs (U4Z only).
struct Shq4Plane {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint8_t> zeros;
};

Shq4Plane MakeShq4Plane(const WeightBlockSpec& spec, std::uint32_t n_tiles,
                        std::uint32_t k_groups_total,
                        std::uint32_t group_offset, bool u4z) {
  Shq4Plane plane;
  // The packer indexes gidx = n_tile*k_groups_total + group_offset + group,
  // which ranges over [group_offset, group_offset + kGroupsPerBlock) for a
  // single tile. Allocate enough blocks to cover the max gidx actually used.
  const std::uint32_t total_blocks =
      std::max(kGroupsPerBlock, n_tiles * k_groups_total);
  plane.codes.assign(total_blocks * (kGroupElements / kMmulK) * kMmulN * 8U, 0);
  plane.scales.assign(total_blocks * kMmulN, 0);
  plane.zeros.assign(total_blocks * kMmulN / 2U, 0);
  for (std::uint32_t nt = 0; nt < n_tiles; ++nt) {
    for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
      const std::uint32_t gidx = nt * k_groups_total + group_offset + group;
      for (std::uint32_t half = 0; half < 2; ++half) {
        for (std::uint32_t k = 0; k < kMmulK; ++k) {
          const std::uint32_t kp = k / 2;
          const bool high = (k & 1U) != 0U;
          for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
            const std::size_t byte_index =
                (static_cast<std::size_t>(gidx) *
                     (kGroupElements / kMmulK) +
                 half) *
                    kMmulN * 8U +
                static_cast<std::size_t>(lane) * 8U + kp;
            const std::uint8_t nibble =
                WeightSpecCode(spec, group, half, k, lane);
            if (high) {
              plane.codes[byte_index] =
                  static_cast<std::uint8_t>((plane.codes[byte_index] & 0x0FU) |
                                            (nibble << 4U));
            } else {
              plane.codes[byte_index] =
                  static_cast<std::uint8_t>((plane.codes[byte_index] & 0xF0U) |
                                            nibble);
            }
          }
        }
      }
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        plane.scales[gidx * kMmulN + lane] =
            RoundToBf16Rne(spec.wscale[group][lane]);
        if (u4z) {
          const std::uint8_t zero = static_cast<std::uint8_t>(
              std::lround(spec.zcorr[group][lane]));
          const std::size_t zero_index = (gidx * kMmulN + lane) / 2U;
          if ((lane & 1U) == 0U) {
            plane.zeros[zero_index] =
                static_cast<std::uint8_t>((plane.zeros[zero_index] & 0xF0U) |
                                          zero);
          } else {
            plane.zeros[zero_index] =
                static_cast<std::uint8_t>((plane.zeros[zero_index] & 0x0FU) |
                                          (zero << 4U));
          }
        }
      }
    }
  }
  return plane;
}

void TestShq4U4zPacking() {
  WeightBlockSpec spec;
  // Every nibble value appears in lane <l> of group 0..7 half 0..1 k 0..15.
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][k][lane] =
              static_cast<std::uint8_t>((group + half + k + lane) & 0x0FU);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = 0.25F * static_cast<float>(lane + 1);
      spec.zcorr[group][lane] = static_cast<float>(lane % 8);
    }
  }
  const std::uint32_t k_groups_total = 2;  // gidx = nt*2 + group
  const auto plane = MakeShq4Plane(spec, /*n_tiles=*/1, k_groups_total,
                                   /*group_offset=*/0, /*u4z=*/true);
  const Shq4PlaneRefs refs{
      .codes = plane.codes.data(),
      .scales = reinterpret_cast<const std::uint8_t*>(plane.scales.data()),
      .zeros = plane.zeros.data(),
      .k_groups_total = k_groups_total,
      .k16_per_group = 2,
  };
  std::array<std::uint8_t, kWeightRecordBytes> record{};
  PackShq4BlockToRecord(WeightMode::kU4Z, refs, /*n_tile=*/0,
                        /*group_offset=*/0, record.data());
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          Expect(GetWeightRecordNibble(record.data(), group, half, k, lane) ==
                     spec.codes[group][half][k][lane],
                 "U4Z plane-to-record nibble transposition");
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      ExpectNear(*WeightRecordScale(record.data(), group, lane),
                 spec.wscale[group][lane], 1e-6F,
                 "U4Z BF16 scale transposition");
      ExpectNear(*WeightRecordZcorr(record.data(), group, lane),
                 spec.zcorr[group][lane], 1e-6F, "U4Z zero transposition");
    }
  }
}

void TestShq4S4Packing() {
  WeightBlockSpec spec;
  // Signed codes -8..7 stored unsigned (u = s + 8).
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][k][lane] =
              static_cast<std::uint8_t>((group + k + lane) & 0x0FU);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = 0.5F * static_cast<float>(lane + 1);
      spec.zcorr[group][lane] = 8.0F;
    }
  }
  const std::uint32_t k_groups_total = 2;
  const auto plane = MakeShq4Plane(spec, /*n_tiles=*/1, k_groups_total,
                                   /*group_offset=*/0, /*u4z=*/false);
  const Shq4PlaneRefs refs{
      .codes = plane.codes.data(),
      .scales = reinterpret_cast<const std::uint8_t*>(plane.scales.data()),
      .zeros = nullptr,
      .k_groups_total = k_groups_total,
      .k16_per_group = 2,
  };
  std::array<std::uint8_t, kWeightRecordBytes> record{};
  PackShq4BlockToRecord(WeightMode::kS4, refs, /*n_tile=*/0,
                        /*group_offset=*/0, record.data());
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      ExpectNear(*WeightRecordZcorr(record.data(), group, lane), 8.0F, 0.0F,
                 "S4 zero correction is 8");
      ExpectNear(*WeightRecordScale(record.data(), group, lane),
                 spec.wscale[group][lane], 1e-6F, "S4 BF16 scale");
    }
  }
}

// ---------------------------------------------------------------------------
// Reference block accumulation vs an independent int64 oracle
// ---------------------------------------------------------------------------

std::array<float, kOutputAccElements> Int64Oracle(
    const std::uint8_t* weight_record, const std::uint8_t* input_record) {
  std::array<float, kOutputAccElements> out{};
  const auto* scales = reinterpret_cast<const float*>(
      input_record + kInputRecordScaleOffset);
  const auto* sums = reinterpret_cast<const std::int32_t*>(
      input_record + kInputRecordSumOffset);
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        std::int64_t dot = 0;
        for (std::uint32_t half = 0; half < 2; ++half) {
          for (std::uint32_t k = 0; k < kMmulK; ++k) {
            const int a = static_cast<int>(
                input_record[((group * 2U + half) * kMmulActivationElements) +
                             (r * kMmulK) + k]);
            dot += static_cast<std::int64_t>(a) *
                   static_cast<int>(GetWeightRecordNibble(
                       weight_record, group, half, k, lane));
          }
        }
        const float correction =
            static_cast<float>(dot) * *WeightRecordScale(weight_record, group,
                                                         lane) -
            *WeightRecordZcorr(weight_record, group, lane) *
                static_cast<float>(sums[(r * kGroupsPerBlock) + group]);
        out[(r * kMmulN) + lane] += correction * scales[(r * kGroupsPerBlock) + group];
      }
    }
  }
  return out;
}

void TestReferenceVsInt64Oracle() {
  WeightBlockSpec spec;
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][k][lane] =
              static_cast<std::uint8_t>((group * 3U + half + k + lane) & 0x0FU);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = 0.125F * static_cast<float>(group + 1);
      spec.zcorr[group][lane] = static_cast<float>((group + lane) % 12);
    }
  }
  std::array<std::uint8_t, kWeightRecordBytes> record{};
  PackWeightBlock(spec, record.data());

  std::array<std::uint8_t, kInputRecordBytes> input{};
  std::array<std::vector<float>, kMmulM> rows;
  for (auto& row : rows) {
    row.resize(kBlockElements);
  }
  // Values spanning negative and positive ranges per (row, group).
  for (std::uint32_t r = 0; r < kMmulM; ++r) {
    for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
      for (std::uint32_t lane = 0; lane < kGroupElements; ++lane) {
        rows[r][group * kGroupElements + lane] =
            static_cast<float>(static_cast<int>(lane + r) * (group % 2 == 0 ? 1 : -1) - r);
      }
    }
    rows[r][0] = 127.0F;  // scale 1 for group 0
  }
  const std::span<const float> spans[kMmulM] = {rows[0], rows[1], rows[2],
                                                rows[3]};
  QuantizeActivationBlock(spans, kBlockElements, input.data());

  std::array<float, kOutputAccElements> acc{};
  ReferenceBlockAccumulate(record.data(), input.data(), acc.data());
  const auto expected = Int64Oracle(record.data(), input.data());
  for (std::uint32_t index = 0; index < kOutputAccElements; ++index) {
    ExpectNear(acc[index], expected[index], 1e-4F,
               "reference accumulation matches int64 oracle");
  }
}

// ---------------------------------------------------------------------------
// Q4_K lossless packing vs DotProductQ4_K
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct BlockQ4KTest {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
#pragma pack(pop)

void SetScaleMinTest(BlockQ4KTest& block, std::size_t group, std::uint8_t scale,
                     std::uint8_t minimum) {
  if (group < 4) {
    block.scales[group] =
        static_cast<std::uint8_t>((block.scales[group] & 0xC0U) | scale);
    block.scales[group + 4] =
        static_cast<std::uint8_t>((block.scales[group + 4] & 0xC0U) | minimum);
    return;
  }
  block.scales[group + 4] = static_cast<std::uint8_t>(
      (block.scales[group + 4] & 0xF0U) | (scale & 0x0FU));
  block.scales[group - 4] = static_cast<std::uint8_t>(
      (block.scales[group - 4] & 0x3FU) | ((scale >> 4U) << 6U));
  block.scales[group + 4] = static_cast<std::uint8_t>(
      (block.scales[group + 4] & 0x0FU) | ((minimum & 0x0FU) << 4U));
  block.scales[group] = static_cast<std::uint8_t>(
      (block.scales[group] & 0x3FU) | ((minimum >> 4U) << 6U));
}

void SetQuantTest(BlockQ4KTest& block, std::size_t index, std::uint8_t value) {
  const std::size_t group = index / kGroupElements;
  const std::size_t pair = group / 2;
  const std::size_t lane = index % kGroupElements;
  auto& packed = block.qs[(pair * kGroupElements) + lane];
  if ((group & 1U) == 0U) {
    packed = static_cast<std::uint8_t>((packed & 0xF0U) | value);
  } else {
    packed = static_cast<std::uint8_t>((packed & 0x0FU) | (value << 4U));
  }
}

std::vector<std::uint8_t> MakeQ4KTensor(std::uint32_t n, std::uint32_t k) {
  Expect(k % kBlockElements == 0, "Q4_K test K must be block aligned");
  const std::uint32_t blocks_per_row = k / kBlockElements;
  std::vector<std::uint8_t> tensor_bytes(
      n * blocks_per_row * sizeof(BlockQ4KTest));
  std::uint32_t seed = 0x5EEDU;
  auto next = [&seed]() {
    seed = seed * 1664525U + 1013904223U;
    return seed;
  };
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t block_ix = 0; block_ix < blocks_per_row; ++block_ix) {
      auto& block = *reinterpret_cast<BlockQ4KTest*>(
          tensor_bytes.data() +
          (static_cast<std::size_t>(row) * blocks_per_row + block_ix) *
              sizeof(BlockQ4KTest));
      std::memset(&block, 0, sizeof(block));
      block.d = 0x3C00U;           // 1.0
      block.dmin = 0x3800U;        // 0.5
      // Small magnitudes keep both accumulation orders FP32-exact.
      const std::uint8_t scale = 1 + static_cast<std::uint8_t>((next() % 6));
      const std::uint8_t minimum = static_cast<std::uint8_t>(next() % 8);
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        SetScaleMinTest(block, group, scale, minimum);
        for (std::uint32_t lane = 0; lane < kGroupElements; ++lane) {
          const std::uint8_t value = static_cast<std::uint8_t>(
              (row * 3U + block_ix + group + lane) & 0x0FU);
          SetQuantTest(block, group * kGroupElements + lane, value);
        }
      }
    }
  }
  return tensor_bytes;
}

std::uint32_t next_logical(std::uint32_t r, std::uint32_t block,
                           std::uint32_t group, std::uint32_t lane) {
  return (r * 131U + block * 17U + group * 29U + lane * 7U) %
         static_cast<std::uint32_t>(253);
}

void TestQ4kLosslessVsDotProduct() {
  const std::uint32_t n = 32;
  const std::uint32_t k = 1024;  // 4 blocks
  const std::uint32_t blocks = k / kBlockElements;
  const std::uint32_t n_tiles = n / kMmulN;
  const auto bytes = MakeQ4KTensor(n, k);

  // Pack every (n_tile, block) into a record: 16 rows per tile, one Q4_K
  // row-block per lane (row-major tensor layout).
  std::vector<std::uint8_t> records(
      static_cast<std::size_t>(n_tiles) * blocks * kWeightRecordBytes);
  for (std::uint32_t nt = 0; nt < n_tiles; ++nt) {
    for (std::uint32_t block = 0; block < blocks; ++block) {
      WeightBlockSpec spec;
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        const std::uint32_t row = nt * kMmulN + lane;
        const auto* src =
            bytes.data() + (static_cast<std::size_t>(row) * blocks + block) *
                               sizeof(BlockQ4KTest);
        DecodeQ4KRowIntoSpec(src, lane, spec);
      }
      PackWeightBlock(spec, records.data() + (static_cast<std::size_t>(nt) *
                                                   blocks +
                                               block) *
                                              kWeightRecordBytes);
    }
  }

  // Inputs: exact-representable integers with amax == 127 per group so the
  // dynamic quantization is lossless (scale == 1).
  std::vector<float> input(4 * k);
  for (std::uint32_t r = 0; r < 4; ++r) {
    for (std::uint32_t block = 0; block < blocks; ++block) {
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        input[r * k + block * kBlockElements + group * kGroupElements] =
            127.0F;
        for (std::uint32_t lane = 1; lane < kGroupElements; ++lane) {
          input[r * k + block * kBlockElements + group * kGroupElements + lane] =
              static_cast<float>(
                  (static_cast<int>(next_logical(r, block, group, lane) % 253)) -
                  126);
        }
      }
    }
  }

  Shape shape{.m = 4, .n = n, .k = k};
  std::vector<float> out(4 * n);
  ReferenceGemm(input, shape, records.data(), out.data());

  // Oracle: full Q4_K dequant dot per row.
  for (std::uint32_t row = 0; row < 4; ++row) {
    for (std::uint32_t n_row = 0; n_row < n; ++n_row) {
      std::vector<float> dequant(k);
      for (std::uint32_t block = 0; block < blocks; ++block) {
        strix::quant::DequantizeQ4_K(
            bytes.data() +
                (static_cast<std::size_t>(n_row) * blocks + block) *
                    sizeof(BlockQ4KTest),
            dequant.data() + static_cast<std::size_t>(block) * kBlockElements,
            kBlockElements);
      }
      float oracle = 0.0F;
      for (std::uint32_t i = 0; i < k; ++i) {
        oracle += dequant[i] * input[row * k + i];
      }
      ExpectNear(out[row * n + n_row], oracle, 1e-3F,
                 "Q4_K lossless record vs dequant dot");
    }
  }
}

// ---------------------------------------------------------------------------
// Dynamic A8 activation contract
// ---------------------------------------------------------------------------

void TestDynamicA8() {
  std::array<std::uint8_t, kInputRecordBytes> record{};

  // amax == 0 -> scale 0, codes 0, sum 0.
  {
    std::array<std::vector<float>, kMmulM> rows;
    for (auto& row : rows) {
      row.assign(kBlockElements, 0.0F);
    }
    const std::span<const float> spans[kMmulM] = {rows[0], rows[1], rows[2],
                                                  rows[3]};
    QuantizeActivationBlock(spans, kBlockElements, record.data());
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      const float* scales = reinterpret_cast<const float*>(
          record.data() + kInputRecordScaleOffset);
      const auto* sums = reinterpret_cast<const std::int32_t*>(
          record.data() + kInputRecordSumOffset);
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        Expect(scales[(r * kGroupsPerBlock) + group] == 0.0F,
               "zero group scale is 0");
        Expect(sums[(r * kGroupsPerBlock) + group] == 0, "zero group sum is 0");
      }
      for (std::uint32_t i = 0; i < kInputRecordCodesBytes; ++i) {
        Expect(record[i] == 0, "zero group codes are 0");
      }
    }
  }

  // amax == 127 -> scale == 1, exact codes, no -128 emitted anywhere.
  {
    std::array<std::vector<float>, kMmulM> rows;
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      rows[r].resize(kBlockElements);
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        rows[r][group * kGroupElements] = static_cast<float>(127 - r);
        for (std::uint32_t lane = 1; lane < kGroupElements; ++lane) {
          rows[r][group * kGroupElements + lane] =
              static_cast<float>((static_cast<int>(lane) *
                                  static_cast<int>(r + 1) % 254) -
                                 127);
        }
      }
    }
    const std::span<const float> spans[kMmulM] = {rows[0], rows[1], rows[2],
                                                  rows[3]};
    QuantizeActivationBlock(spans, kBlockElements, record.data());
    const bool has_min_128 =
        std::ranges::any_of(record.begin(), record.begin() + kInputRecordCodesBytes,
                            [](std::uint8_t b) { return b == 128; });
    Expect(!has_min_128, "dynamic A8 never emits -128");
    for (std::uint32_t r = 0; r < kMmulM; ++r) {
      const float scale = reinterpret_cast<const float*>(
          record.data() + kInputRecordScaleOffset)[r * kGroupsPerBlock];
      ExpectNear(scale, -1.0F + static_cast<float>(127 - r) / 127.0F + 1.0F,
                 // group 0 of row r has amax 127 -> scale 1 exactly.
                 0.01F, "A8 scale");
      const std::int8_t code = static_cast<std::int8_t>(
          record.data()[r * kMmulK]);  // k=0 group 0 half 0
      Expect(code != -128, "no -128 code in group 0");
    }
  }
}

// ---------------------------------------------------------------------------
// Tail K float-exactness, tail M, GEMV-vs-GEMM, group boundaries
// ---------------------------------------------------------------------------

void TestTailKExact() {
  // K = 257 (2 blocks, second block has a 1-element logical tail) must be
  // bitwise equal to K = 256 with the same prefix input.
  const std::uint32_t n = 16;
  WeightBlockSpec spec;
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t kk = 0; kk < kMmulK; ++kk) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][kk][lane] = static_cast<std::uint8_t>(
              (group * 5U + half + kk + lane) & 0x0FU);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = 0.0625F * static_cast<float>(lane + 1);
      spec.zcorr[group][lane] = static_cast<float>(lane % 4);
    }
  }
  std::array<std::uint8_t, kWeightRecordBytes> record{};
  PackWeightBlock(spec, record.data());
  std::vector<std::uint8_t> records(n / kMmulN * 2 * kWeightRecordBytes);
  std::memcpy(records.data(), record.data(), kWeightRecordBytes);
  std::memcpy(records.data() + kWeightRecordBytes, record.data(),
              kWeightRecordBytes);  // tail block records (identical plane)

  std::vector<float> input(257);
  for (std::uint32_t i = 0; i < 256; ++i) {
    input[i] = static_cast<float>((static_cast<int>(i * 13) % 253) - 126);
  }
  input[256] = 0.0F;  // the only real element of the tail block contributes 0
  // Padded outputs: rounds(1)*4 rows x n_tiles(1)*16 columns.
  std::vector<float> exact_out(64);
  std::vector<float> tail_out(64);
  ReferenceGemm(std::span<const float>(input.data(), 256),
                Shape{.m = 1, .n = n, .k = 256}, record.data(),
                exact_out.data());
  ReferenceGemm(input, Shape{.m = 1, .n = n, .k = 257}, records.data(),
                tail_out.data());
  for (std::uint32_t i = 0; i < 16; ++i) {
    Expect(exact_out[i] == tail_out[i],
           "padded tail K contributes exactly zero");
  }
}

void TestGemvMatchesGemmRow() {
  // Batch-1 GEMV result must be float-identical to the first row of the
  // same GEMM because per-row activation quantization is row-local.
  const std::uint32_t n = 32;
  const std::uint32_t k = 512;
  WeightBlockSpec spec;
  for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
    for (std::uint32_t half = 0; half < 2; ++half) {
      for (std::uint32_t kk = 0; kk < kMmulK; ++kk) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          spec.codes[group][half][kk][lane] = static_cast<std::uint8_t>(
              (group + half + kk + lane * 2U) & 0x0FU);
        }
      }
    }
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      spec.wscale[group][lane] = 0.5F * static_cast<float>(group + 1);
      spec.zcorr[group][lane] = static_cast<float>((group + lane) % 5);
    }
  }
  const std::uint32_t n_tiles = n / kMmulN;
  std::vector<std::uint8_t> records(
      static_cast<std::size_t>(n_tiles) * 2 * kWeightRecordBytes);
  for (std::uint32_t t = 0; t < n_tiles * 2; ++t) {
    PackWeightBlock(spec, records.data() + t * kWeightRecordBytes);
  }
  std::vector<float> input(8 * k);  // 8 rows > 4 for the GEMM test
  for (std::uint32_t r = 0; r < 8; ++r) {
    for (std::uint32_t i = 0; i < k; ++i) {
      input[r * k + i] =
          static_cast<float>((static_cast<int>((r + 1) * (i + 3)) % 253) - 126);
    }
  }
  std::vector<float> gemv_out(4 * n);
  std::vector<float> gemm_out(4 * n);
  ReferenceGemm(std::span<const float>(input.data(), k),
                Shape{.m = 1, .n = n, .k = k}, records.data(), gemv_out.data());
  ReferenceGemm(input, Shape{.m = 4, .n = n, .k = k}, records.data(),
                gemm_out.data());
  for (std::uint32_t i = 0; i < n; ++i) {
    Expect(gemv_out[i] == gemm_out[i],
           "GEMV row 0 equals GEMM chunk row 0 (row-local quantization)");
  }
}

void TestGroupBoundaries() {
  // K = 33: group 1 holds exactly one element; its scale must derive only
  // from that element and the padded tail must not leak into codes.
  std::array<std::uint8_t, kInputRecordBytes> record{};
  std::vector<float> row(kBlockElements, 0.0F);
  row[32] = 127.0F;
  const std::span<const float> empty{};
  const std::span<const float> spans[kMmulM] = {row, empty, empty, empty};
  QuantizeActivationBlock(spans, 33, record.data());
  const float* scales = reinterpret_cast<const float*>(
      record.data() + kInputRecordScaleOffset);
  const auto* sums =
      reinterpret_cast<const std::int32_t*>(record.data() + kInputRecordSumOffset);
  ExpectNear(scales[1], 1.0F, 0.0F, "group boundary scale from single element");
  Expect(sums[1] == 127, "group boundary sum");
  // Codes for group 1 half 0: k=0 holds 127. Layout: byte =
  // (group*2 + half)*kMmulActivationElements + row*kMmulK + kk, so group 1
  // half 0 row 0 k 0 sits at 2*kMmulActivationElements.
  Expect(record.data()[2 * kMmulActivationElements] == 127,
         "group boundary code");
  Expect(record.data()[kMmulActivationElements] == 0,
         "group boundary pad code zero");  // Group 0 must be all zero (amax 0).
  Expect(scales[0] == 0.0F && sums[0] == 0, "group 0 untouched");
}

#ifdef ENGINE_ENABLE_XRT

// ---------------------------------------------------------------------------
// Hardware gate: M=1 eh_proj on the real 27B MTP weights vs the CPU oracle
// ---------------------------------------------------------------------------

bool Xdna2Required() {
  const char* val = std::getenv("STRIX_REQUIRE_XDNA2");
  if (val == nullptr) {
    val = std::getenv("STRIX_REQUIRE_NPU");
  }
  return val != nullptr && std::string_view(val) != "0" &&
         std::string_view(val) != "false";
}

std::vector<float> MakeRepresentableInput() {
  std::vector<float> input(strix::xdna2::kQwenAie2pW4a8InputElements);
  for (std::size_t group_start = 0; group_start < input.size();
       group_start += kGroupElements) {
    for (std::size_t lane = 0; lane < kGroupElements; ++lane) {
      const int quantized =
          lane == kGroupElements - 1 ? 127 : static_cast<int>(lane) - 16;
      input[group_start + lane] = static_cast<float>(quantized) / 64.0F;
    }
  }
  return input;
}

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};
};

Comparison Compare(std::span<const float> actual,
                   std::span<const float> expected, std::string_view message) {
  Expect(actual.size() == expected.size(), std::string(message));
  double squared_error = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  double dot = 0.0;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double lhs = actual[index];
    const double rhs = expected[index];
    const double difference = lhs - rhs;
    squared_error += difference * difference;
    actual_squared += lhs * lhs;
    expected_squared += rhs * rhs;
    dot += lhs * rhs;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
  }
  return {
      .rmse = std::sqrt(squared_error / static_cast<double>(actual.size())),
      .cosine = dot / std::sqrt(actual_squared * expected_squared),
      .max_abs = max_abs,
  };
}

// Pack the real Q4_K eh_proj matrix into CPU-reference records (global 16-lane
// n_tiles, as consumed by ReferenceGemm).
std::vector<std::uint8_t> PackReferenceRecords(const void* q4k_data) {
  constexpr std::size_t blocks =
      strix::xdna2::kQwenAie2pW4a8InputElements / kBlockElements;
  constexpr std::size_t n_tiles =
      strix::xdna2::kQwenAie2pW4a8OutputElements / kMmulN;
  const auto* rows = static_cast<const std::uint8_t*>(q4k_data);
  const std::size_t row_bytes = strix::quant::QuantizedRowBytes(
      strix::core::GgmlType::kQ4_K, strix::xdna2::kQwenAie2pW4a8InputElements);
  std::vector<std::uint8_t> records(static_cast<std::size_t>(n_tiles) * blocks *
                                    kWeightRecordBytes);
  for (std::uint32_t nt = 0; nt < n_tiles; ++nt) {
    for (std::uint32_t block = 0; block < blocks; ++block) {
      WeightBlockSpec spec;
      for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
        const std::size_t row = nt * kMmulN + lane;
        const auto* src =
            rows + (row * row_bytes) + (block * sizeof(BlockQ4KTest));
        DecodeQ4KRowIntoSpec(src, lane, spec);
      }
      PackWeightBlock(spec, records.data() +
                                 (static_cast<std::size_t>(nt) * blocks +
                                  block) *
                                     kWeightRecordBytes);
    }
  }
  return records;
}

// Returns true when a real hardware run happened; prints a SKIP marker
// otherwise (missing model file only -- the caller skips on XDNA2 absence).
bool TestHardwareM1EhProj(const strix::xdna2::XrtDeviceInfo& device) {
  const char* model_path = std::getenv("STRIX_MTP_MODEL");
  if (model_path == nullptr || std::string_view(model_path).empty()) {
    if (Xdna2Required()) {
      throw std::runtime_error(
          "qwen_aie2p_w4a8 hardware gate requires STRIX_MTP_MODEL "
          "(STRIX_REQUIRE_XDNA2=1)");
    }
    std::cout << "qwen_aie2p_w4a8_hardware_m1: skipped "
                 "TODO(P2-hardware) (STRIX_MTP_MODEL not set)\n";
    return false;
  }
  std::string error;
  auto reader_owner = strix::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader_owner != nullptr, error);
  std::shared_ptr<const strix::core::GgufReader> reader(
      std::move(reader_owner));
  const auto weights =
      strix::speculative::QwenMtpWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), error);

  const auto& q4k = weights->fusion_projection;
  Expect(q4k.type == strix::core::GgmlType::kQ4_K,
         "eh_proj weight must be Q4_K");
  Expect(q4k.num_elements ==
             strix::xdna2::kQwenAie2pW4a8OutputElements *
                 strix::xdna2::kQwenAie2pW4a8InputElements,
         "eh_proj weight must be 5120x10240");

  const auto session_count =
      strix::xdna2::QwenAie2pW4a8Session::ActiveSessionCountForDiagnostics();
  const auto bo_count =
      strix::xdna2::QwenAie2pW4a8Session::ActiveBoCountForDiagnostics();
  strix::xdna2::QwenAie2pW4a8Failure failure;
  auto session = strix::xdna2::QwenAie2pW4a8Session::Create(
      {.program_dir = STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR}, device, q4k,
      &failure);
  Expect(session != nullptr,
         "W4A8 session opens: " + failure.category + ": " + failure.message);
  Expect(strix::xdna2::QwenAie2pW4a8Session::ActiveSessionCountForDiagnostics() ==
             session_count + 1,
         "session counter increments");
  Expect(strix::xdna2::QwenAie2pW4a8Session::ActiveBoCountForDiagnostics() ==
             bo_count + 3,
         "BO counter increments");
  const auto& info = session->ProgramInfo();
  Expect(info.partition_columns == 8, "program uses all eight NPU2 columns");
  Expect(info.blocks == 40 && info.tiles_per_column == 10 && info.rounds == 1,
         "baked M=1 eh_proj configuration");
  Expect(info.packed_weight_bytes == 39'321'600,
         "packed weight view size (8x10x40x12288)");
  Expect(info.packed_input_bytes == 40 * 1280,
         "packed input view size (blocks x record)");

  const auto input = MakeRepresentableInput();
  std::vector<float> output(strix::xdna2::kQwenAie2pW4a8OutputElements);
  strix::xdna2::QwenAie2pW4a8RunMetrics metrics;
  Expect(session->Run(input, output, &metrics, &failure),
         "W4A8 M=1 eh_proj command: " + failure.category + ": " +
             failure.message);
  Expect(!metrics.quarantined, "session remains usable");
  Expect(metrics.command_us > 0.0, "command timing is positive");

  const auto records = PackReferenceRecords(q4k.data);
  std::vector<float> expected(4 *
                              (strix::xdna2::kQwenAie2pW4a8OutputElements /
                               kMmulN) * kMmulN);
  ReferenceGemm(input, Shape{.m = 1,
                             .n = strix::xdna2::kQwenAie2pW4a8OutputElements,
                             .k = strix::xdna2::kQwenAie2pW4a8InputElements},
                records.data(), expected.data());
  const auto comparison =
      Compare(output, std::span<const float>(expected.data(), output.size()),
              "M=1 eh_proj comparison size");
  Expect(std::isfinite(comparison.rmse), "M=1 eh_proj RMSE is finite");
  Expect(comparison.rmse < 0.1, "M=1 eh_proj RMSE");
  Expect(comparison.cosine > 0.9999, "M=1 eh_proj cosine");
  Expect(comparison.max_abs < 1.0F, "M=1 eh_proj max absolute error");

  std::cout << "qwen_aie2p_w4a8_hardware_m1: setup_ms=" << info.setup_ms
            << " weight_pack_ms=" << info.weight_pack_ms
            << " weight_upload_ms=" << info.weight_upload_ms
            << " activation_pack_us=" << metrics.activation_pack_us
            << " input_upload_us=" << metrics.input_upload_us
            << " command_us=" << metrics.command_us
            << " output_download_us=" << metrics.output_download_us
            << " end_to_end_us=" << metrics.end_to_end_us
            << " rmse=" << comparison.rmse << " cosine=" << comparison.cosine
            << " max_abs=" << comparison.max_abs << '\n';
  session.reset();
  Expect(strix::xdna2::QwenAie2pW4a8Session::ActiveSessionCountForDiagnostics() ==
             session_count,
         "session counter returns to baseline");
  Expect(strix::xdna2::QwenAie2pW4a8Session::ActiveBoCountForDiagnostics() ==
             bo_count,
         "BO counter returns to baseline");
  return true;
}

#endif

}  // namespace

int main() {
  try {
    TestBf16Rne();
    TestNibbleExhaustive();
    TestShq4U4zPacking();
    TestShq4S4Packing();
    TestReferenceVsInt64Oracle();
    TestQ4kLosslessVsDotProduct();
    TestDynamicA8();
    TestTailKExact();
    TestGemvMatchesGemmRow();
    TestGroupBoundaries();
    std::cout << "qwen_aie2p_w4a8 CPU oracle gates: PASS\n";
#ifdef ENGINE_ENABLE_XRT
    const auto inventory = strix::diagnostics::CollectSystemInventory();
    const auto device = strix::xdna2::DiscoverXrtDevice(0, inventory);
    if (!device.available) {
      if (Xdna2Required()) {
        throw std::runtime_error(
            "qwen_aie2p_w4a8 hardware gate required but XDNA2 unavailable: " +
            device.error_category + " detected=" + device.detected +
            " required=" + device.required + " remediation=" +
            device.remediation);
      }
      std::cout << "qwen_aie2p_w4a8 hardware gates: SKIP "
                   "TODO(P2-hardware) (XDNA2 unavailable: "
                << device.error_category << ")\n";
      return 77;
    }
    const bool hardware_ran = TestHardwareM1EhProj(device);
    if (hardware_ran) {
      std::cout << "qwen_aie2p_w4a8 hardware gates: PASS\n";
    } else {
      std::cout << "qwen_aie2p_w4a8 hardware gates: SKIP "
                   "TODO(P2-hardware) (model weights unavailable)\n";
    }
#endif
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_aie2p_w4a8_test failed: " << exception.what() << '\n';
    return 1;
  }
}