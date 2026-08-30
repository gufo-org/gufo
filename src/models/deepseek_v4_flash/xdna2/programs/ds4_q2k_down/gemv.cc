// Copyright (C) 2026 Gufo Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_M
#define DIM_M 16
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

namespace {

constexpr int kGroupElements = 16;
constexpr int kGroupsPerBlock = DIM_K / kGroupElements;
constexpr int kMmulRows = 4;
constexpr int kMmulK = 16;
constexpr int kMmulN = DIM_M;
constexpr int kActivationTileElements = kMmulRows * kMmulK;
constexpr int kWeightTileElements = kMmulK * kMmulN;
constexpr int kWeightCodeBytes = kGroupsPerBlock * kWeightTileElements;
constexpr int kWeightScaleOffset = kWeightCodeBytes;
constexpr int kWeightMinimumOffset =
    kWeightScaleOffset + (kGroupsPerBlock * kMmulN * sizeof(float));
constexpr int kInputCodeBytes = kGroupsPerBlock * kActivationTileElements;
constexpr int kInputScaleOffset = kInputCodeBytes;
constexpr int kInputSumOffset =
    kInputScaleOffset + (kGroupsPerBlock * kMmulRows * sizeof(float));

using Q2A8Mmul = aie::mmul<kMmulRows, kMmulK, kMmulN, int8, uint8, acc32>;

static_assert(DIM_M == 8);
static_assert(DIM_K == 256);
static_assert(kWeightCodeBytes == 2048);
static_assert(kInputCodeBytes == 1024);

aie::vector<float, kMmulN> AccumulateRow(
    aie::vector<float, kMmulN> accumulator,
    const aie::vector<std::int32_t, kMmulRows * kMmulN>& dot_products, int row,
    const aie::vector<float, kMmulN>& weight_scale,
    const aie::vector<float, kMmulN>& weight_minimum,
    const float* activation_scales, const std::int32_t* activation_sums,
    int group) {
  const auto dot_float =
      aie::to_float(dot_products.template extract<kMmulN>(row));
  const auto quantized_sum = aie::broadcast<float, kMmulN>(
      static_cast<float>(activation_sums[group * kMmulRows + row]));
  const auto activation_scale =
      aie::broadcast<float, kMmulN>(activation_scales[group * kMmulRows + row]);
  const aie::vector<float, kMmulN> corrected =
      aie::sub(aie::mul(dot_float, weight_scale),
               aie::mul(weight_minimum, quantized_sum));
  const aie::vector<float, kMmulN> scaled =
      aie::mul(corrected, activation_scale);
  return aie::add(accumulator, scaled);
}

}  // namespace

extern "C" {

void ds4_q2k_down_zero_f32(float* output) {
  const auto zero = aie::zeros<float, kMmulN>();
  for (int row = 0; row < kMmulRows; ++row) {
    aie::store_v(output + row * kMmulN, zero);
  }
}

void ds4_q2k_down_gemm_a8q2_f32(std::uint8_t* __restrict weight_record,
                                std::int8_t* __restrict input_record,
                                float* __restrict output) {
  const auto* weight_scales =
      reinterpret_cast<const float*>(weight_record + kWeightScaleOffset);
  const auto* weight_minima =
      reinterpret_cast<const float*>(weight_record + kWeightMinimumOffset);
  const auto* activation_scales = reinterpret_cast<const float*>(
      reinterpret_cast<std::uint8_t*>(input_record) + kInputScaleOffset);
  const auto* activation_sums = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<std::uint8_t*>(input_record) + kInputSumOffset);

  auto output0 = aie::load_v<kMmulN>(output);
  auto output1 = aie::load_v<kMmulN>(output + kMmulN);
  auto output2 = aie::load_v<kMmulN>(output + 2 * kMmulN);
  auto output3 = aie::load_v<kMmulN>(output + 3 * kMmulN);

  for (int group = 0; group < kGroupsPerBlock; ++group) {
    const auto activation = aie::load_v<kActivationTileElements>(
        input_record + group * kActivationTileElements);
    const auto weight =
        aie::load_v<kWeightTileElements>(reinterpret_cast<const uint8*>(
            weight_record + group * kWeightTileElements));

    Q2A8Mmul dot;
    dot.mul(activation, weight);
    const auto dot_products = dot.template to_vector<std::int32_t>();
    const auto weight_scale =
        aie::load_v<kMmulN>(weight_scales + group * kMmulN);
    const auto weight_minimum =
        aie::load_v<kMmulN>(weight_minima + group * kMmulN);

    output0 =
        AccumulateRow(output0, dot_products, 0, weight_scale, weight_minimum,
                      activation_scales, activation_sums, group);
    output1 =
        AccumulateRow(output1, dot_products, 1, weight_scale, weight_minimum,
                      activation_scales, activation_sums, group);
    output2 =
        AccumulateRow(output2, dot_products, 2, weight_scale, weight_minimum,
                      activation_scales, activation_sums, group);
    output3 =
        AccumulateRow(output3, dot_products, 3, weight_scale, weight_minimum,
                      activation_scales, activation_sums, group);
  }

  aie::store_v(output, output0);
  aie::store_v(output + kMmulN, output1);
  aie::store_v(output + 2 * kMmulN, output2);
  aie::store_v(output + 3 * kMmulN, output3);
}

}  // extern "C"
