// Copyright (C) 2026 Strix Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace {

constexpr int kVectorWidth = 16;
constexpr float kEpsilon = 1.0e-6F;

}  // namespace

extern "C" void qwen_mtp_rmsnorm(const bfloat16* restrict input,
                                 const bfloat16* restrict weight,
                                 bfloat16* restrict output, int32_t columns) {
  aie::vector<float, kVectorWidth> partial = aie::zeros<float, kVectorWidth>();

  const int32_t vector_chunks = columns / kVectorWidth;
  for (int32_t chunk = 0; chunk < vector_chunks; ++chunk) {
    const auto values =
        aie::load_v<kVectorWidth>(input + (chunk * kVectorWidth));
    const aie::vector<float, kVectorWidth> squares = aie::mul_square(values);
    partial = aie::add(partial, squares);
  }

  float sum_squares = aie::reduce_add(partial);
  const int32_t tail_start = vector_chunks * kVectorWidth;
  for (int32_t index = tail_start; index < columns; ++index) {
    const float value = static_cast<float>(input[index]);
    sum_squares += value * value;
  }

  const float mean_square =
      (sum_squares / static_cast<float>(columns)) + kEpsilon;
  const bfloat16 inverse_rms = static_cast<bfloat16>(aie::invsqrt(mean_square));
  const auto inverse_rms_vector =
      aie::broadcast<bfloat16, kVectorWidth>(inverse_rms);

  for (int32_t chunk = 0; chunk < vector_chunks; ++chunk) {
    const int32_t offset = chunk * kVectorWidth;
    const auto values = aie::load_v<kVectorWidth>(input + offset);
    const auto weights = aie::load_v<kVectorWidth>(weight + offset);
    const aie::vector<bfloat16, kVectorWidth> normalized =
        aie::mul(values, inverse_rms_vector);
    const aie::vector<bfloat16, kVectorWidth> weighted =
        aie::mul(normalized, weights);
    aie::store_v(output + offset, weighted);
  }

  for (int32_t index = tail_start; index < columns; ++index) {
    const float normalized =
        static_cast<float>(input[index]) * static_cast<float>(inverse_rms);
    output[index] =
        static_cast<bfloat16>(normalized * static_cast<float>(weight[index]));
  }
}
