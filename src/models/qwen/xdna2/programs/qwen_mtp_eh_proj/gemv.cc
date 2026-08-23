// Copyright (C) 2026 Strix Engine contributors
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

namespace {

constexpr int kGroupElements = 32;
constexpr int kGroupsPerBlock = DIM_K / kGroupElements;
constexpr int kMmulK = 16;
constexpr int kMmulM = 4;
constexpr int kMmulN = DIM_M;
constexpr int kActivationRows = 4;
constexpr int kActivationTileElements = kActivationRows * kMmulK;
constexpr int kWeightTileElements = kMmulK * kMmulN;
constexpr int kWeightTileBytes = kWeightTileElements / 2;
constexpr int kWeightTilesPerGroup = kGroupElements / kMmulK;
constexpr int kWeightCodeBytes =
    kGroupsPerBlock * kWeightTilesPerGroup * kWeightTileBytes;
constexpr int kWeightScaleOffset = kWeightCodeBytes;
constexpr int kWeightMinOffset =
    kWeightScaleOffset + (kGroupsPerBlock * DIM_M * sizeof(float));
constexpr int kInputCodeBytes =
    kGroupsPerBlock * kWeightTilesPerGroup * kActivationTileElements;
constexpr int kInputScaleOffset = kInputCodeBytes;
constexpr int kInputSumOffset =
    kInputScaleOffset + (kGroupsPerBlock * sizeof(float));

using W4A8Mmul =
    aie::mmul<kMmulM, kMmulK, kMmulN, int8, uint4, acc32>;

static_assert(DIM_M == 16);
static_assert(DIM_K == 256);
static_assert(kWeightCodeBytes == 2048);
static_assert(kInputCodeBytes == 1024);

}  // namespace

extern "C" {

void qwen_mtp_zero_f32(float* output) {
  aie::store_v(output, aie::zeros<float, DIM_M>());
}

void qwen_mtp_gemv_q4k_w4a8_f32(std::uint8_t* __restrict weight_record,
                                std::int8_t* __restrict input_record,
                                float* __restrict output) {
  const auto* weight_scales =
      reinterpret_cast<const float*>(weight_record + kWeightScaleOffset);
  const auto* weight_mins =
      reinterpret_cast<const float*>(weight_record + kWeightMinOffset);
  const auto* activation_scales =
      reinterpret_cast<const float*>(input_record + kInputScaleOffset);
  const auto* activation_sums =
      reinterpret_cast<const std::int32_t*>(input_record + kInputSumOffset);

  auto output_accumulator = aie::load_v<DIM_M>(output);

  for (int group = 0; group < kGroupsPerBlock; ++group) {
    const auto* activation_base =
        input_record +
        (group * kWeightTilesPerGroup * kActivationTileElements);
    const auto* weight_base =
        weight_record +
        (group * kWeightTilesPerGroup * kWeightTileBytes);

    const auto activation_lo =
        aie::load_v<kActivationTileElements>(activation_base);
    const auto activation_hi = aie::load_v<kActivationTileElements>(
        activation_base + kActivationTileElements);
    const auto weight_lo = aie::load_v<kWeightTileElements>(
        reinterpret_cast<const uint4*>(weight_base));
    const auto weight_hi = aie::load_v<kWeightTileElements>(
        reinterpret_cast<const uint4*>(weight_base + kWeightTileBytes));

    W4A8Mmul dot;
    dot.mul(activation_lo, weight_lo);
    dot.mac(activation_hi, weight_hi);
    const auto dot_products = dot.template to_vector<std::int32_t>();

    const float activation_scale = activation_scales[group];
    const auto dot_float =
        aie::to_float(dot_products.template extract<DIM_M>(0));
    const auto weight_scale =
        aie::load_v<DIM_M>(weight_scales + (group * DIM_M));
    const auto weight_min =
        aie::load_v<DIM_M>(weight_mins + (group * DIM_M));
    const auto quantized_sum = aie::broadcast<float, DIM_M>(
        static_cast<float>(activation_sums[group]));
    const auto group_scale =
        aie::broadcast<float, DIM_M>(activation_scale);
    const aie::vector<float, DIM_M> weighted_dot =
        aie::mul(dot_float, weight_scale);
    const aie::vector<float, DIM_M> minimum_correction =
        aie::mul(weight_min, quantized_sum);
    const aie::vector<float, DIM_M> corrected =
        aie::sub(weighted_dot, minimum_correction);
    const aie::vector<float, DIM_M> scaled =
        aie::mul(corrected, group_scale);
    output_accumulator = aie::add(output_accumulator, scaled);
  }

  aie::store_v(output, output_accumulator);
}

}  // extern "C"
