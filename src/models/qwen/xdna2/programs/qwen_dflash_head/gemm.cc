// Copyright (C) 2026 Gufo Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_M
#define DIM_M 16
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

#ifndef BATCH_ROWS
#define BATCH_ROWS 8
#endif

namespace {

constexpr int kGroupElements = 32;
constexpr int kGroupsPerBlock = DIM_K / kGroupElements;
constexpr int kMmulK = 8;
constexpr int kMmulM = 4;
constexpr int kMmulN = DIM_M;
constexpr int kBatchGroups = BATCH_ROWS / kMmulM;
constexpr int kActivationTileElements = kMmulM * kMmulK;
constexpr int kWeightTileElements = kMmulK * kMmulN;
constexpr int kTilesPerGroup = kGroupElements / kMmulK;
constexpr int kWeightCodeBytes =
    kGroupsPerBlock * kTilesPerGroup * kWeightTileElements;
constexpr int kWeightScaleOffset = kWeightCodeBytes;
constexpr int kInputCodeBytes =
    kBatchGroups * kGroupsPerBlock * kTilesPerGroup * kActivationTileElements;
constexpr int kInputScaleOffset = kInputCodeBytes;

using W8A8Mmul = aie::mmul<kMmulM, kMmulK, kMmulN, int8, int8, acc32>;

static_assert(DIM_M == 8);
static_assert(DIM_K == 256);
static_assert(BATCH_ROWS == 8);
static_assert(kWeightCodeBytes == 2048);
static_assert(kInputCodeBytes == 2048);

}  // namespace

extern "C" {

void qwen_dflash_head_zero_f32(float* output) {
  for (int row = 0; row < BATCH_ROWS; ++row) {
    aie::store_v(output + (row * DIM_M), aie::zeros<float, DIM_M>());
  }
}

void qwen_dflash_head_gemm_q8_0_w8a8_f32(std::int8_t* __restrict weight_record,
                                         std::int8_t* __restrict input_record,
                                         float* __restrict output) {
  const auto* weight_scales =
      reinterpret_cast<const float*>(weight_record + kWeightScaleOffset);
  const auto* activation_scales =
      reinterpret_cast<const float*>(input_record + kInputScaleOffset);

  for (int group = 0; group < kGroupsPerBlock; ++group) {
    const auto* weight_base =
        weight_record + (group * kTilesPerGroup * kWeightTileElements);
    const auto weight_0 = aie::load_v<kWeightTileElements>(weight_base);
    const auto weight_1 =
        aie::load_v<kWeightTileElements>(weight_base + kWeightTileElements);
    const auto weight_2 = aie::load_v<kWeightTileElements>(
        weight_base + (2 * kWeightTileElements));
    const auto weight_3 = aie::load_v<kWeightTileElements>(
        weight_base + (3 * kWeightTileElements));
    const auto weight_scale =
        aie::load_v<DIM_M>(weight_scales + (group * DIM_M));

    for (int batch_group = 0; batch_group < kBatchGroups; ++batch_group) {
      const auto* activation_base =
          input_record +
          (((batch_group * kGroupsPerBlock + group) * kTilesPerGroup) *
           kActivationTileElements);
      const auto activation_0 =
          aie::load_v<kActivationTileElements>(activation_base);
      const auto activation_1 = aie::load_v<kActivationTileElements>(
          activation_base + kActivationTileElements);
      const auto activation_2 = aie::load_v<kActivationTileElements>(
          activation_base + (2 * kActivationTileElements));
      const auto activation_3 = aie::load_v<kActivationTileElements>(
          activation_base + (3 * kActivationTileElements));

      W8A8Mmul dot;
      dot.mul(activation_0, weight_0);
      dot.mac(activation_1, weight_1);
      dot.mac(activation_2, weight_2);
      dot.mac(activation_3, weight_3);
      const auto dot_products = dot.template to_vector<std::int32_t>();

      const auto dot_float = aie::to_float(dot_products);
      const auto weight_scale_group =
          aie::concat(weight_scale, weight_scale, weight_scale, weight_scale);
      const aie::vector<float, kMmulM * DIM_M> scaled_weights =
          aie::mul(dot_float, weight_scale_group);
      const int batch_row = batch_group * kMmulM;
      const auto activation_scale_group = aie::concat(
          aie::broadcast<float, DIM_M>(
              activation_scales[(batch_row * kGroupsPerBlock) + group]),
          aie::broadcast<float, DIM_M>(
              activation_scales[((batch_row + 1) * kGroupsPerBlock) + group]),
          aie::broadcast<float, DIM_M>(
              activation_scales[((batch_row + 2) * kGroupsPerBlock) + group]),
          aie::broadcast<float, DIM_M>(
              activation_scales[((batch_row + 3) * kGroupsPerBlock) + group]));
      const aie::vector<float, kMmulM * DIM_M> contribution =
          aie::mul(scaled_weights, activation_scale_group);
      auto accumulator =
          aie::load_v<kMmulM * DIM_M>(output + (batch_row * DIM_M));
      accumulator = aie::add(accumulator, contribution);
      aie::store_v(output + (batch_row * DIM_M), accumulator);
    }
  }
}

}  // extern "C"
