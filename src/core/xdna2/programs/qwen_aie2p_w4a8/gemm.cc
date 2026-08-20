// Copyright (C) 2026 Strix Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AIE2P (XDNA2) W4A8 SHQ4-T16 GEMM microkernel for the Qwen3.8-27B MTP
// family. One kernel ELF serves every (K-blocks, N-tiles-per-column,
// M-rounds) configuration: the per-config data movement graph in
// qwen_aie2p_w4a8.py decides loop counts; this file only computes one
// 256-element K block's contribution with native 4x16x16 uint4 x int8
// matrix instructions and the fused zero/activation/weight-scale epilogue.
//
// Record layouts (host packing contract, see qwen_aie2p_w4a8.h):
//   weight_record (16 lanes x 3072 B, one 256-K block):
//     [0,2048)   codes:  8 groups x 2 half-tiles x (16 k x 16 lanes) uint4,
//                        byte = k*8 + lane/2, low nibble = even lane
//     [2048,2560) scales: 8 groups x 16 lanes float32 (BF16 width expanded)
//     [2560,3072) zcorr:  8 groups x 16 lanes float32
//   input_record (5 x 256 int8, one 256-K block, one chunk of 4 M rows):
//     rows 0..3  codes:  group g, half h, row r, k: offset
//                        g*128 + h*64 + r*16 + k (int8, broadcast over k)
//     row 4 [0,128)      activation scales: 4 rows x 8 groups float32
//     row 4 [128,256)    int32 sums:         4 rows x 8 groups
//   The kernel accumulates one block's contribution into the caller's
//   64-float output accumulator (4 rows x 16 lanes, row-major buckets).
#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_M
#define DIM_M 4
#endif
#ifndef DIM_K
#define DIM_K 16
#endif
#ifndef DIM_N
#define DIM_N 16
#endif

namespace {

constexpr int kBlockElements = 256;
constexpr int kGroupElements = 32;
constexpr int kGroupsPerBlock = kBlockElements / kGroupElements;
constexpr int kMmulK = DIM_K;
constexpr int kMmulM = DIM_M;
constexpr int kMmulN = DIM_N;
constexpr int kActivationRows = 4;
constexpr int kWeightRecordCodesBytes = 2048;
constexpr int kWeightScaleOffset = 2048;
constexpr int kWeightZcorrOffset = 2560;
constexpr int kWeightRecordBytes = 3072;
static_assert(kBlockElements == 256);
constexpr int kInputCodeBytes = kBlockElements * kActivationRows;
constexpr int kInputScaleOffset = 1024;
constexpr int kInputSumOffset = 1152;
constexpr int kScaleStride = 32;  // 8 groups x 4 bytes per activation row

using W4A8Mmul =
    aie::mmul<kMmulM, kMmulK, kMmulN, int8, uint4, acc32>;

static_assert(kMmulK == 16);
static_assert(kMmulM == 4);
static_assert(kMmulN == 16);
static_assert(kGroupsPerBlock * 2 * kMmulK == kBlockElements);

}  // namespace

extern "C" {

void qwen_aie2p_w4a8_zero_f32(float* output) {
  aie::store_v(output, aie::zeros<float, 64>());
}

void qwen_aie2p_w4a8_gemm(std::uint8_t* __restrict weight_record,
                          std::int8_t* __restrict input_record,
                          float* __restrict output) {
  const auto* weight_scales =
      reinterpret_cast<const float*>(weight_record + kWeightScaleOffset);
  const auto* weight_zcorr =
      reinterpret_cast<const float*>(weight_record + kWeightZcorrOffset);
  const auto* activation_scales =
      reinterpret_cast<const float*>(input_record + kInputScaleOffset);
  const auto* activation_sums =
      reinterpret_cast<const std::int32_t*>(input_record + kInputSumOffset);

  auto output_accumulator = aie::load_v<64>(output);

  for (int group = 0; group < kGroupsPerBlock; ++group) {
    const auto* activation_base =
        input_record + (group * 2 * kActivationRows * kMmulK);
    const auto* weight_base = weight_record + (group * 2 * kMmulK * kMmulN);

    const auto activation_lo =
        aie::load_v<kActivationRows * kMmulK>(activation_base);
    const auto activation_hi = aie::load_v<kActivationRows * kMmulK>(
        activation_base + (kActivationRows * kMmulK));
    const auto weight_lo = aie::load_v<kMmulK * kMmulN>(
        reinterpret_cast<const uint4*>(weight_base));
    const auto weight_hi = aie::load_v<kMmulK * kMmulN>(
        reinterpret_cast<const uint4*>(weight_base + (kMmulK * kMmulN)));

    W4A8Mmul dot;
    dot.mul(activation_lo, weight_lo);
    dot.mac(activation_hi, weight_hi);
    const auto dot_products = dot.template to_vector<std::int32_t>();

    const auto weight_scale = aie::load_v<kMmulN>(weight_scales + (group * kMmulN));
    const auto weight_zero = aie::load_v<kMmulN>(weight_zcorr + (group * kMmulN));

    for (int row = 0; row < kActivationRows; ++row) {
      const auto row_dot =
          aie::to_float(dot_products.template extract<kMmulN>(row));
      const auto activation_scale = aie::broadcast<float, kMmulN>(
          activation_scales[(row * 8) + group]);
      const auto quantized_sum = aie::broadcast<float, kMmulN>(
          static_cast<float>(activation_sums[(row * 8) + group]));
      const aie::vector<float, kMmulN> weighted_dot =
          aie::mul(row_dot, weight_scale);
      const aie::vector<float, kMmulN> zero_correction =
          aie::mul(weight_zero, quantized_sum);
      const aie::vector<float, kMmulN> corrected =
          aie::sub(weighted_dot, zero_correction);
      const aie::vector<float, kMmulN> scaled =
          aie::mul(corrected, activation_scale);
      const auto old_row = output_accumulator.template extract<kMmulN>(row);
      output_accumulator.insert(row, aie::add(old_row, scaled));
    }
  }

  aie::store_v(output, output_accumulator);
}

}  // extern "C"