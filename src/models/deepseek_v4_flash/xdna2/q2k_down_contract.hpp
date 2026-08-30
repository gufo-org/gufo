// Copyright (C) 2026 Gufo Engine contributors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pure-CPU contract for a future DeepSeek V4 Flash XDNA2 Q2_K down
// projection. The record geometry follows the AIE2P-native M=4, K=16, N=8
// matrix tile while preserving the source GGUF Q2_K codes and FP16 scales.

#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_XDNA2_Q2K_DOWN_CONTRACT_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_XDNA2_Q2K_DOWN_CONTRACT_HPP_

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace gufo::models::deepseek_v4_flash::xdna2::q2k {

inline constexpr std::uint32_t kBlockElements = 256;
inline constexpr std::uint32_t kGroupElements = 16;
inline constexpr std::uint32_t kGroupsPerBlock =
    kBlockElements / kGroupElements;
inline constexpr std::uint32_t kMmulRows = 4;
inline constexpr std::uint32_t kMmulK = 16;
inline constexpr std::uint32_t kMmulN = 8;

#pragma pack(push, 1)
struct BlockQ2K {
  std::uint8_t scales[kGroupsPerBlock];
  std::uint8_t codes[kBlockElements / 4];
  std::uint16_t d;
  std::uint16_t dmin;
};
#pragma pack(pop)

static_assert(sizeof(BlockQ2K) == 84);

// One 256-K block for sixteen output rows:
//   codes  [group][k][output lane] as unpacked uint2 values in uint8 storage
//   scales [group][output lane] as FP32
//   minima [group][output lane] as FP32 zero-correction coefficients
inline constexpr std::uint32_t kWeightCodeBytes =
    kGroupsPerBlock * kMmulK * kMmulN;
inline constexpr std::uint32_t kWeightScaleOffset = kWeightCodeBytes;
inline constexpr std::uint32_t kWeightMinimumOffset =
    kWeightScaleOffset + kGroupsPerBlock * kMmulN * sizeof(float);
inline constexpr std::uint32_t kWeightRecordBytes =
    kWeightMinimumOffset + kGroupsPerBlock * kMmulN * sizeof(float);

// One 256-K block for four activation rows:
//   codes  [group][row][k] as dynamic signed INT8
//   scales [group][row] as FP32
//   sums   [group][row] as INT32, used by the Q2_K minimum correction
inline constexpr std::uint32_t kInputCodeBytes =
    kGroupsPerBlock * kMmulRows * kMmulK;
inline constexpr std::uint32_t kInputScaleOffset = kInputCodeBytes;
inline constexpr std::uint32_t kInputSumOffset =
    kInputScaleOffset + kGroupsPerBlock * kMmulRows * sizeof(float);
inline constexpr std::uint32_t kInputRecordBytes =
    kInputSumOffset + kGroupsPerBlock * kMmulRows * sizeof(std::int32_t);

static_assert(kWeightRecordBytes == 3072);
static_assert(kInputRecordBytes == 1536);

template<typename T>
T LoadScalar(const std::uint8_t* data, std::size_t offset) noexcept {
  T value{};
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

template<typename T>
void StoreScalar(std::uint8_t* data, std::size_t offset, T value) noexcept {
  std::memcpy(data + offset, &value, sizeof(value));
}

inline float Fp16ToFloat32(std::uint16_t id) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(id & 0x8000U) << 16U;
  const std::uint32_t exponent = (id >> 10U) & 0x1FU;
  const std::uint32_t mantissa = id & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      const float value =
          (static_cast<float>(mantissa) / 1024.0F) * std::ldexp(1.0F, -14);
      bits = std::bit_cast<std::uint32_t>(value);
      bits |= sign;
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

inline std::uint8_t SourceCode(const BlockQ2K& block,
                               std::uint32_t index) noexcept {
  const std::uint32_t half = index / 128U;
  const std::uint32_t within_half = index % 128U;
  const std::uint32_t shift = (within_half / 32U) * 2U;
  const std::uint32_t byte = half * 32U + within_half % 32U;
  return static_cast<std::uint8_t>((block.codes[byte] >> shift) & 0x03U);
}

inline float SourceScale(const BlockQ2K& block, std::uint32_t group) noexcept {
  return Fp16ToFloat32(block.d) *
         static_cast<float>(block.scales[group] & 0x0FU);
}

inline float SourceMinimum(const BlockQ2K& block,
                           std::uint32_t group) noexcept {
  return Fp16ToFloat32(block.dmin) *
         static_cast<float>(block.scales[group] >> 4U);
}

inline std::size_t WeightCodeOffset(std::uint32_t group, std::uint32_t k,
                                    std::uint32_t lane) noexcept {
  return (static_cast<std::size_t>(group) * kMmulK + k) * kMmulN + lane;
}

inline std::size_t WeightParameterOffset(std::uint32_t base,
                                         std::uint32_t group,
                                         std::uint32_t lane) noexcept {
  return base + static_cast<std::size_t>(group * kMmulN + lane) * sizeof(float);
}

inline void PackWeightRecord(
    const std::array<const BlockQ2K*, kMmulN>& source_rows,
    std::span<std::uint8_t, kWeightRecordBytes> output) {
  std::ranges::fill(output, std::uint8_t{0});
  for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
    const BlockQ2K& block = *source_rows[lane];
    for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
      StoreScalar(output.data(),
                  WeightParameterOffset(kWeightScaleOffset, group, lane),
                  SourceScale(block, group));
      StoreScalar(output.data(),
                  WeightParameterOffset(kWeightMinimumOffset, group, lane),
                  SourceMinimum(block, group));
      for (std::uint32_t k = 0; k < kMmulK; ++k) {
        output[WeightCodeOffset(group, k, lane)] =
            SourceCode(block, group * kGroupElements + k);
      }
    }
  }
}

inline std::size_t InputCodeOffset(std::uint32_t group, std::uint32_t row,
                                   std::uint32_t k) noexcept {
  return (static_cast<std::size_t>(group) * kMmulRows + row) * kMmulK + k;
}

inline std::size_t InputParameterOffset(std::uint32_t base, std::uint32_t group,
                                        std::uint32_t row) noexcept {
  return base +
         static_cast<std::size_t>(group * kMmulRows + row) * sizeof(float);
}

inline void PackInputRecord(std::span<const float> input,
                            std::uint32_t logical_rows,
                            std::uint32_t row_stride, std::uint32_t block_index,
                            std::span<std::uint8_t, kInputRecordBytes> output) {
  std::ranges::fill(output, std::uint8_t{0});
  const std::uint32_t active_rows = std::min(logical_rows, kMmulRows);
  for (std::uint32_t row = 0; row < active_rows; ++row) {
    for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
      const std::size_t begin =
          static_cast<std::size_t>(row) * row_stride +
          static_cast<std::size_t>(block_index) * kBlockElements +
          static_cast<std::size_t>(group) * kGroupElements;
      float maximum = 0.0F;
      for (std::uint32_t k = 0; k < kGroupElements; ++k) {
        maximum = std::max(maximum, std::abs(input[begin + k]));
      }
      const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
      std::int32_t sum = 0;
      for (std::uint32_t k = 0; k < kGroupElements; ++k) {
        std::int8_t quantized = 0;
        if (scale != 0.0F) {
          const auto rounded =
              static_cast<int>(std::nearbyint(input[begin + k] / scale));
          quantized = static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
        }
        output[InputCodeOffset(group, row, k)] =
            static_cast<std::uint8_t>(quantized);
        sum += quantized;
      }
      StoreScalar(output.data(),
                  InputParameterOffset(kInputScaleOffset, group, row), scale);
      StoreScalar(output.data(),
                  InputParameterOffset(kInputSumOffset, group, row), sum);
    }
  }
}

inline void AccumulatePackedRecord(
    std::span<const std::uint8_t, kWeightRecordBytes> weights,
    std::span<const std::uint8_t, kInputRecordBytes> input,
    std::span<float, kMmulRows * kMmulN> output) {
  for (std::uint32_t row = 0; row < kMmulRows; ++row) {
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      float value = output[row * kMmulN + lane];
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        std::int32_t dot = 0;
        for (std::uint32_t k = 0; k < kMmulK; ++k) {
          const auto activation =
              std::bit_cast<std::int8_t>(input[InputCodeOffset(group, row, k)]);
          const std::uint8_t weight = weights[WeightCodeOffset(group, k, lane)];
          dot += static_cast<std::int32_t>(activation) *
                 static_cast<std::int32_t>(weight);
        }
        const float activation_scale = LoadScalar<float>(
            input.data(), InputParameterOffset(kInputScaleOffset, group, row));
        const std::int32_t activation_sum = LoadScalar<std::int32_t>(
            input.data(), InputParameterOffset(kInputSumOffset, group, row));
        const float weight_scale = LoadScalar<float>(
            weights.data(),
            WeightParameterOffset(kWeightScaleOffset, group, lane));
        const float minimum = LoadScalar<float>(
            weights.data(),
            WeightParameterOffset(kWeightMinimumOffset, group, lane));
        value +=
            activation_scale * (static_cast<float>(dot) * weight_scale -
                                static_cast<float>(activation_sum) * minimum);
      }
      output[row * kMmulN + lane] = value;
    }
  }
}

inline void AccumulateSourceRecord(
    const std::array<const BlockQ2K*, kMmulN>& source_rows,
    std::span<const std::uint8_t, kInputRecordBytes> input,
    std::span<float, kMmulRows * kMmulN> output) {
  for (std::uint32_t row = 0; row < kMmulRows; ++row) {
    for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
      float value = output[row * kMmulN + lane];
      const BlockQ2K& block = *source_rows[lane];
      for (std::uint32_t group = 0; group < kGroupsPerBlock; ++group) {
        std::int32_t dot = 0;
        for (std::uint32_t k = 0; k < kMmulK; ++k) {
          const auto activation =
              std::bit_cast<std::int8_t>(input[InputCodeOffset(group, row, k)]);
          dot += static_cast<std::int32_t>(activation) *
                 SourceCode(block, group * kGroupElements + k);
        }
        const float activation_scale = LoadScalar<float>(
            input.data(), InputParameterOffset(kInputScaleOffset, group, row));
        const std::int32_t activation_sum = LoadScalar<std::int32_t>(
            input.data(), InputParameterOffset(kInputSumOffset, group, row));
        value +=
            activation_scale *
            (static_cast<float>(dot) * SourceScale(block, group) -
             static_cast<float>(activation_sum) * SourceMinimum(block, group));
      }
      output[row * kMmulN + lane] = value;
    }
  }
}

}  // namespace gufo::models::deepseek_v4_flash::xdna2::q2k

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_XDNA2_Q2K_DOWN_CONTRACT_HPP_
