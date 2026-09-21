#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/minimax_h3/attention.hip.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/dit_ops.hip.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace {

namespace h3 = gufo::minimax_h3;
namespace ops = gufo::minimax_h3::dit_ops;

int failures = 0;

#define CHECK(condition)                                          \
  do {                                                            \
    if (!(condition)) {                                           \
      std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << ": " \
                << #condition << '\n';                            \
      ++failures;                                                 \
    }                                                             \
  } while (false)

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template<typename T>
class DeviceArray {
public:
  explicit DeviceArray(std::size_t count) : count_(count) {
    if (hipMalloc(&data_, count * sizeof(T)) != hipSuccess) {
      data_ = nullptr;
    }
  }
  ~DeviceArray() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  bool CopyFrom(std::span<const T> input) {
    return data_ != nullptr && input.size() == count_ &&
           hipMemcpy(data_, input.data(), input.size_bytes(),
                     hipMemcpyHostToDevice) == hipSuccess;
  }
  bool CopyTo(std::span<T> output) const {
    return data_ != nullptr && output.size() == count_ &&
           hipMemcpy(output.data(), data_, output.size_bytes(),
                     hipMemcpyDeviceToHost) == hipSuccess;
  }
  T* get() noexcept { return data_; }
  const T* get() const noexcept { return data_; }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

void CheckClose(std::span<const std::uint16_t> got,
                std::span<const float> expected, float tolerance) {
  CHECK(got.size() == expected.size());
  for (std::size_t index = 0; index < std::min(got.size(), expected.size());
       ++index) {
    const float delta = std::abs(ToFloat(got[index]) - expected[index]);
    if (delta > tolerance) {
      std::cerr << "first mismatch index=" << index
                << " got=" << ToFloat(got[index])
                << " expected=" << expected[index] << " delta=" << delta
                << '\n';
      CHECK(false);
      return;
    }
  }
}

void CheckCastsElementwiseAndNorms() {
  const std::vector<float> f32 = {-4.25F, -1.0F, -0.125F, 0.0F,
                                  0.25F,  2.5F,  19.0F};
  DeviceArray<float> d_f32(f32.size());
  DeviceArray<float> d_roundtrip(f32.size());
  DeviceArray<std::uint16_t> d_bf16(f32.size());
  CHECK(d_f32.CopyFrom(f32));
  ops::LaunchF32ToBf16(d_f32.get(), d_bf16.get(), f32.size(), nullptr);
  ops::LaunchBf16ToF32(d_bf16.get(), d_roundtrip.get(), f32.size(), nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<float> roundtrip(f32.size());
  CHECK(d_roundtrip.CopyTo(roundtrip));
  for (std::size_t index = 0; index < f32.size(); ++index) {
    CHECK(roundtrip[index] == ToFloat(ToBf16(f32[index])));
  }

  const std::vector<std::uint16_t> left = {
      ToBf16(1.0F),   ToBf16(-2.0F), ToBf16(0.5F),  ToBf16(9.0F),
      ToBf16(-0.25F), ToBf16(7.0F),  ToBf16(0.125F)};
  const std::vector<std::uint16_t> right = {
      ToBf16(2.0F),  ToBf16(0.25F), ToBf16(-0.5F), ToBf16(-4.0F),
      ToBf16(0.75F), ToBf16(1.0F),  ToBf16(-0.25F)};
  DeviceArray<std::uint16_t> d_left(left.size());
  DeviceArray<std::uint16_t> d_right(right.size());
  DeviceArray<std::uint16_t> d_output(left.size());
  CHECK(d_left.CopyFrom(left));
  CHECK(d_right.CopyFrom(right));
  ops::LaunchAddSubtract(d_left.get(), d_right.get(), d_output.get(),
                         left.size(), false, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> got(left.size());
  CHECK(d_output.CopyTo(got));
  const std::vector<float> added = {3.0F, -1.75F, 0.0F,   5.0F,
                                    0.5F, 8.0F,   -0.125F};
  CheckClose(got, added, 0.01F);
  ops::LaunchAddSubtract(d_left.get(), d_right.get(), d_output.get(),
                         left.size(), true, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(d_output.CopyTo(got));
  const std::vector<float> subtracted = {-1.0F, -2.25F, 1.0F,  13.0F,
                                         -1.0F, 6.0F,   0.375F};
  CheckClose(got, subtracted, 0.01F);

  constexpr std::uint32_t kRows = 2;
  constexpr std::uint32_t kWidth = 5;
  const std::vector<std::uint16_t> norm_input = {
      ToBf16(1.0F),  ToBf16(2.0F), ToBf16(3.0F), ToBf16(4.0F),  ToBf16(5.0F),
      ToBf16(-2.0F), ToBf16(1.0F), ToBf16(0.5F), ToBf16(-1.0F), ToBf16(3.0F)};
  const std::vector<std::uint16_t> weight(kWidth, ToBf16(1.0F));
  const std::vector<std::uint16_t> bias(kWidth, ToBf16(0.125F));
  DeviceArray<std::uint16_t> d_norm_input(norm_input.size());
  DeviceArray<std::uint16_t> d_weight(weight.size());
  DeviceArray<std::uint16_t> d_bias(bias.size());
  DeviceArray<std::uint16_t> d_norm_output(norm_input.size());
  CHECK(d_norm_input.CopyFrom(norm_input));
  CHECK(d_weight.CopyFrom(weight));
  CHECK(d_bias.CopyFrom(bias));
  ops::LaunchRmsNorm(d_norm_input.get(), d_weight.get(), d_norm_output.get(),
                     kRows, kWidth, 1.0e-5F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  got.resize(norm_input.size());
  CHECK(d_norm_output.CopyTo(got));
  for (const std::uint16_t value : got) {
    CHECK(std::isfinite(ToFloat(value)));
  }
  ops::LaunchLayerNorm(d_norm_input.get(), d_weight.get(), d_bias.get(),
                       d_norm_output.get(), kRows, kWidth, 1.0e-5F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(d_norm_output.CopyTo(got));
  for (std::uint32_t row = 0; row < kRows; ++row) {
    float mean = 0.0F;
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      mean += ToFloat(norm_input[row * kWidth + column]);
    }
    mean /= static_cast<float>(kWidth);
    float variance = 0.0F;
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      const float delta = ToFloat(norm_input[row * kWidth + column]) - mean;
      variance += delta * delta;
    }
    variance /= static_cast<float>(kWidth);
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      const float expected =
          (ToFloat(norm_input[row * kWidth + column]) - mean) /
              std::sqrt(variance + 1.0e-5F) +
          0.125F;
      CHECK(std::abs(ToFloat(got[row * kWidth + column]) - expected) < 0.02F);
    }
  }
}

void CheckAdaLnGateAndSwiGlu() {
  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kWidth = 8;
  constexpr std::uint32_t kSlots = 6;
  constexpr std::uint32_t kModulationRows = 2;
  std::vector<std::uint16_t> input(kRows * kWidth);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = ToBf16((static_cast<float>(index) - 6.0F) / 7.0F);
  }
  const std::vector<std::uint16_t> branch(kRows * kWidth, ToBf16(0.25F));
  const std::vector<std::uint16_t> weight(kWidth, ToBf16(1.0F));
  std::vector<std::uint16_t> modulation(kModulationRows * kSlots * kWidth,
                                        ToBf16(0.0F));
  const std::vector<std::uint32_t> row_map = {0, 1, 0};
  for (std::uint32_t row = 0; row < kModulationRows; ++row) {
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      modulation[(row * kSlots + 0) * kWidth + column] =
          ToBf16(0.1F * static_cast<float>(row + 1));
      modulation[(row * kSlots + 1) * kWidth + column] =
          ToBf16(-0.05F * static_cast<float>(column));
      modulation[(row * kSlots + 2) * kWidth + column] = ToBf16(0.5F);
    }
  }
  DeviceArray<std::uint16_t> d_input(input.size());
  DeviceArray<std::uint16_t> d_branch(branch.size());
  DeviceArray<std::uint16_t> d_weight(weight.size());
  DeviceArray<std::uint16_t> d_modulation(modulation.size());
  DeviceArray<std::uint32_t> d_row_map(row_map.size());
  DeviceArray<float> d_inverse(kRows);
  DeviceArray<std::uint16_t> d_output(input.size());
  CHECK(d_input.CopyFrom(input));
  CHECK(d_branch.CopyFrom(branch));
  CHECK(d_weight.CopyFrom(weight));
  CHECK(d_modulation.CopyFrom(modulation));
  CHECK(d_row_map.CopyFrom(row_map));
  ops::LaunchAdaLnSplit(d_input.get(), d_weight.get(), d_modulation.get(),
                        d_row_map.get(), d_inverse.get(), d_output.get(), kRows,
                        kWidth, kSlots, 0, 1, 1.0e-5F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> got(input.size());
  CHECK(d_output.CopyTo(got));
  for (std::uint32_t row = 0; row < kRows; ++row) {
    float mean_square = 0.0F;
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      const float value = ToFloat(input[row * kWidth + column]);
      mean_square += value * value;
    }
    const float inverse =
        1.0F / std::sqrt(mean_square / static_cast<float>(kWidth) + 1.0e-5F);
    const std::uint32_t modulation_row = row_map[row];
    for (std::uint32_t column = 0; column < kWidth; ++column) {
      const float normalized = ToFloat(input[row * kWidth + column]) * inverse;
      const float shift =
          ToFloat(modulation[(modulation_row * kSlots + 0) * kWidth + column]);
      const float scale =
          ToFloat(modulation[(modulation_row * kSlots + 1) * kWidth + column]);
      const float expected = normalized * (1.0F + scale) + shift;
      CHECK(std::abs(ToFloat(got[row * kWidth + column]) - expected) < 0.02F);
    }
  }
  ops::LaunchGate(d_input.get(), d_branch.get(), d_modulation.get(),
                  d_row_map.get(), d_output.get(), kRows, kWidth, kSlots, 2,
                  nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(d_output.CopyTo(got));
  for (std::size_t index = 0; index < got.size(); ++index) {
    CHECK(std::abs(ToFloat(got[index]) - (ToFloat(input[index]) + 0.125F)) <
          0.015F);
  }

  constexpr std::uint32_t kSwiRows = 2;
  constexpr std::uint32_t kSwiWidth = 8;
  std::vector<std::uint16_t> fused(kSwiRows * kSwiWidth * 2U);
  for (std::uint32_t row = 0; row < kSwiRows; ++row) {
    for (std::uint32_t column = 0; column < kSwiWidth; ++column) {
      fused[(row * 2U) * kSwiWidth + column] =
          ToBf16((static_cast<float>(row * kSwiWidth + column) - 6.0F) / 4.0F);
      fused[(row * 2U + 1U) * kSwiWidth + column] =
          ToBf16((static_cast<float>(column) - 3.0F) / 5.0F);
    }
  }
  DeviceArray<std::uint16_t> d_fused(fused.size());
  DeviceArray<std::uint16_t> d_swiglu(kSwiRows * kSwiWidth);
  CHECK(d_fused.CopyFrom(fused));
  ops::LaunchSwiGlu(d_fused.get(), d_swiglu.get(), kSwiRows, kSwiWidth,
                    nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  got.resize(kSwiRows * kSwiWidth);
  CHECK(d_swiglu.CopyTo(got));
  for (std::uint32_t row = 0; row < kSwiRows; ++row) {
    for (std::uint32_t column = 0; column < kSwiWidth; ++column) {
      const float gate = ToFloat(fused[(row * 2U) * kSwiWidth + column]);
      const float up = ToFloat(fused[(row * 2U + 1U) * kSwiWidth + column]);
      const float expected = gate / (1.0F + std::exp(-gate)) * up;
      CHECK(std::abs(ToFloat(got[row * kSwiWidth + column]) - expected) <
            0.02F);
    }
  }

  constexpr std::uint32_t kLookupRows = 256;
  constexpr std::uint32_t kLookupWidth = 256;
  constexpr std::size_t kLookupEntries = 1ULL << 16U;
  std::vector<std::uint16_t> lookup_fused(kLookupEntries * 2U);
  for (std::size_t index = 0; index < kLookupEntries; ++index) {
    const std::size_t row = index / kLookupWidth;
    const std::size_t column = index % kLookupWidth;
    lookup_fused[(row * 2U) * kLookupWidth + column] =
        static_cast<std::uint16_t>(index);
    lookup_fused[(row * 2U + 1U) * kLookupWidth + column] =
        ToBf16((static_cast<float>(index % 31U) - 15.0F) / 7.0F);
  }
  DeviceArray<std::uint16_t> d_lookup_fused(lookup_fused.size());
  DeviceArray<std::uint16_t> d_lookup_reference(kLookupEntries);
  DeviceArray<std::uint16_t> d_lookup_output(kLookupEntries);
  DeviceArray<float> d_silu_lookup(kLookupEntries);
  CHECK(d_lookup_fused.CopyFrom(lookup_fused));
  ops::LaunchBf16SiluLookup(d_silu_lookup.get(), nullptr);
  ops::LaunchSwiGlu(d_lookup_fused.get(), d_lookup_reference.get(), kLookupRows,
                    kLookupWidth, nullptr);
  ops::LaunchSwiGluLookup(d_lookup_fused.get(), d_lookup_output.get(),
                          d_silu_lookup.get(), kLookupRows, kLookupWidth,
                          nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> lookup_reference(kLookupEntries);
  std::vector<std::uint16_t> lookup_output(kLookupEntries);
  CHECK(d_lookup_reference.CopyTo(lookup_reference));
  CHECK(d_lookup_output.CopyTo(lookup_output));
  CHECK(lookup_output == lookup_reference);
}

void CheckPatchAndFinalProjection() {
  constexpr std::uint32_t kRows = 2;
  constexpr std::uint32_t kPatchWidth = 3;
  constexpr std::uint32_t kHiddenWidth = 5;
  const std::vector<float> patch = {1.0F, -2.0F, 0.5F, 0.25F, 3.0F, -1.0F};
  std::vector<float> patch_weight(kHiddenWidth * kPatchWidth);
  for (std::size_t index = 0; index < patch_weight.size(); ++index) {
    patch_weight[index] = (static_cast<float>(index) - 5.0F) / 11.0F;
  }
  const std::vector<float> patch_bias = {0.1F, -0.2F, 0.3F, -0.4F, 0.5F};
  DeviceArray<float> d_patch(patch.size());
  DeviceArray<float> d_patch_weight(patch_weight.size());
  DeviceArray<float> d_patch_bias(patch_bias.size());
  DeviceArray<std::uint16_t> d_hidden(kRows * kHiddenWidth);
  CHECK(d_patch.CopyFrom(patch));
  CHECK(d_patch_weight.CopyFrom(patch_weight));
  CHECK(d_patch_bias.CopyFrom(patch_bias));
  ops::LaunchPatchProjection(d_patch.get(), d_patch_weight.get(),
                             d_patch_bias.get(), d_hidden.get(), kRows,
                             kPatchWidth, kHiddenWidth, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> hidden(kRows * kHiddenWidth);
  CHECK(d_hidden.CopyTo(hidden));
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kHiddenWidth; ++column) {
      float expected = patch_bias[column];
      for (std::uint32_t index = 0; index < kPatchWidth; ++index) {
        expected =
            std::fma(patch[row * kPatchWidth + index],
                     patch_weight[column * kPatchWidth + index], expected);
      }
      CHECK(std::abs(ToFloat(hidden[row * kHiddenWidth + column]) - expected) <
            0.02F);
    }
  }

  constexpr std::uint32_t kOutputWidth = 3;
  std::vector<float> final_weight(kOutputWidth * kHiddenWidth);
  for (std::size_t index = 0; index < final_weight.size(); ++index) {
    final_weight[index] = (7.0F - static_cast<float>(index)) / 13.0F;
  }
  const std::vector<float> final_bias = {-0.25F, 0.5F, 0.125F};
  DeviceArray<float> d_final_weight(final_weight.size());
  DeviceArray<float> d_final_bias(final_bias.size());
  DeviceArray<float> d_output(kRows * kOutputWidth);
  CHECK(d_final_weight.CopyFrom(final_weight));
  CHECK(d_final_bias.CopyFrom(final_bias));
  ops::LaunchFinalProjection(d_hidden.get(), d_final_weight.get(),
                             d_final_bias.get(), d_output.get(), kRows,
                             kHiddenWidth, kOutputWidth, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<float> output(kRows * kOutputWidth);
  CHECK(d_output.CopyTo(output));
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kOutputWidth; ++column) {
      float expected = final_bias[column];
      for (std::uint32_t index = 0; index < kHiddenWidth; ++index) {
        expected =
            std::fma(ToFloat(hidden[row * kHiddenWidth + index]),
                     final_weight[column * kHiddenWidth + index], expected);
      }
      CHECK(std::abs(output[row * kOutputWidth + column] - expected) < 1.0e-5F);
    }
  }
}

void CheckGroupedQkvAndAttention() {
  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kHeads = 2;
  constexpr std::uint32_t kDimension = 4;
  constexpr std::uint32_t kRopeHalf = 2;
  constexpr std::size_t kInner = kHeads * kDimension;
  std::vector<std::uint16_t> qkv(kRows * kInner * 3);
  for (std::size_t index = 0; index < qkv.size(); ++index) {
    qkv[index] = ToBf16((static_cast<float>(index % 17) - 8.0F) / 9.0F);
  }
  const std::vector<std::uint16_t> norm(kDimension, ToBf16(1.0F));
  std::vector<std::uint16_t> cosine(kRows * kRopeHalf, ToBf16(1.0F));
  std::vector<std::uint16_t> sine(kRows * kRopeHalf, ToBf16(0.0F));
  cosine[kRopeHalf] = ToBf16(std::cos(0.5F));
  sine[kRopeHalf] = ToBf16(std::sin(0.5F));

  DeviceArray<std::uint16_t> d_qkv(qkv.size());
  DeviceArray<std::uint16_t> d_norm(norm.size());
  DeviceArray<std::uint16_t> d_cosine(cosine.size());
  DeviceArray<std::uint16_t> d_sine(sine.size());
  DeviceArray<std::uint16_t> d_query(kRows * kInner);
  DeviceArray<std::uint16_t> d_key(kRows * kInner);
  DeviceArray<std::uint16_t> d_value(kRows * kInner);
  DeviceArray<std::uint16_t> d_attention(kRows * kInner);
  CHECK(d_qkv.CopyFrom(qkv));
  CHECK(d_norm.CopyFrom(norm));
  CHECK(d_cosine.CopyFrom(cosine));
  CHECK(d_sine.CopyFrom(sine));
  ops::LaunchGroupedQkvNormRope(d_qkv.get(), d_norm.get(), d_norm.get(),
                                d_cosine.get(), d_sine.get(), d_query.get(),
                                d_key.get(), d_value.get(), kRows, kHeads,
                                kDimension, kRopeHalf, 1.0e-5F, nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> query(kRows * kInner);
  std::vector<std::uint16_t> key(kRows * kInner);
  std::vector<std::uint16_t> value(kRows * kInner);
  CHECK(d_query.CopyTo(query));
  CHECK(d_key.CopyTo(key));
  CHECK(d_value.CopyTo(value));
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const std::size_t grouped =
          static_cast<std::size_t>(row) * kInner * 3U +
          static_cast<std::size_t>(head) * kDimension * 3U;
      const std::size_t flat =
          (static_cast<std::size_t>(row) * kHeads + head) * kDimension;
      for (std::uint32_t dimension = 0; dimension < kDimension; ++dimension) {
        CHECK(value[flat + dimension] ==
              qkv[grouped + kDimension * 2U + dimension]);
      }
    }
  }
  ops::LaunchFullAttention(d_query.get(), d_key.get(), d_value.get(),
                           d_attention.get(), kRows, kHeads, kDimension,
                           1.0F / std::sqrt(static_cast<float>(kDimension)),
                           nullptr);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<std::uint16_t> attention(kRows * kInner);
  CHECK(d_attention.CopyTo(attention));
  for (const std::uint16_t element : attention) {
    CHECK(std::isfinite(ToFloat(element)));
  }
}

void CheckNativeAttention(std::uint32_t rows, float query_scale = 1.7F) {
  constexpr std::uint32_t kHeads = 3, kDimension = 128;
  constexpr std::size_t kGuard = 16;
  const std::size_t elements = std::size_t(rows) * kHeads * kDimension;
  const auto random = [](std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return (static_cast<int>(x % 20001) - 10000) * 0.0001F;
  };
  std::vector<std::uint16_t> query(elements), key(elements), value(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    query[i] = ToBf16(random(i * 3) * query_scale);
    key[i] = ToBf16(random(i * 3 + 1) * query_scale);
    value[i] = ToBf16(random(i * 3 + 2));
  }
  DeviceArray<std::uint16_t> dq(elements), dk(elements), dv(elements);
  DeviceArray<std::uint16_t> output(elements + kGuard);
  CHECK(dq.CopyFrom(query));
  CHECK(dk.CopyFrom(key));
  CHECK(dv.CopyFrom(value));
  std::vector<std::uint16_t> actual(elements + kGuard, 0x7fc1U);
  CHECK(output.CopyFrom(actual));
  const auto run = [&] {
    ops::LaunchWmmaAttention(dq.get(), dk.get(), dv.get(), output.get(), rows,
                             kHeads, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess);
  };
  run();
  CHECK(output.CopyTo(actual));
  for (std::size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(ToFloat(actual[i]))) {
      CHECK(false);
      return;
    }
  }
  for (std::size_t i = elements; i < actual.size(); ++i)
    CHECK(actual[i] == 0x7fc1U);
  run();
  std::vector<std::uint16_t> replay(actual.size());
  CHECK(output.CopyTo(replay));
  CHECK(replay == actual);

  // Independent dense FP64 softmax/PV, including the final key and query tails.
  // Do not compare against another optimized Gufo attention implementation.
  double error_squared = 0, norm_squared = 0, max_error = 0, max_reference = 0;
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t row : {0U, rows / 2, rows - 1}) {
      std::vector<double> probabilities(rows);
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::uint32_t k = 0; k < rows; ++k) {
        double dot = 0;
        for (std::uint32_t d = 0; d < kDimension; ++d)
          dot +=
              double(ToFloat(
                  query[(std::size_t(head) * rows + row) * kDimension + d])) *
              ToFloat(key[(std::size_t(head) * rows + k) * kDimension + d]);
        probabilities[k] = dot / std::sqrt(double(kDimension));
        maximum = std::max(maximum, probabilities[k]);
      }
      double denominator = 0;
      for (double& probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (std::uint32_t d = 0; d < kDimension; ++d) {
        double reference = 0;
        for (std::uint32_t k = 0; k < rows; ++k)
          reference +=
              probabilities[k] / denominator *
              ToFloat(value[(std::size_t(head) * rows + k) * kDimension + d]);
        const double delta =
            ToFloat(
                actual[(std::size_t(row) * kHeads + head) * kDimension + d]) -
            reference;
        error_squared += delta * delta;
        norm_squared += reference * reference;
        max_error = std::max(max_error, std::abs(delta));
        max_reference = std::max(max_reference, std::abs(reference));
      }
    }
  }
  const double relative_l2 = std::sqrt(error_squared / norm_squared);
  const double relative_max = max_error / max_reference;
  std::cout << "H3 native attention FP64 teacher rows=" << rows
            << " scale=" << query_scale << " relative_l2=" << relative_l2
            << " relative_max=" << relative_max << '\n';
  CHECK(relative_l2 < 0.004);
  CHECK(relative_max < 0.01);
}

bool HasGfx1151() {
  int device = 0;
  hipDeviceProp_t properties{};
  return hipGetDevice(&device) == hipSuccess &&
         hipGetDeviceProperties(&properties, device) == hipSuccess &&
         std::string_view(properties.gcnArchName).starts_with("gfx1151") &&
         properties.integrated != 0;
}

int CheckAnalytic() {
  if (!HasGfx1151()) {
    std::cout << "SKIP: integrated gfx1151 is unavailable\n";
    return 77;
  }
  CheckCastsElementwiseAndNorms();
  CheckAdaLnGateAndSwiGlu();
  CheckPatchAndFinalProjection();
  CheckGroupedQkvAndAttention();
  for (std::uint32_t rows : {1U, 8U, 9U, 31U, 32U, 33U, 63U, 64U, 65U, 127U,
                             128U, 129U, 528U, 4096U, 4097U})
    CheckNativeAttention(rows);
  CheckNativeAttention(33, 0.0F);
  CheckNativeAttention(65, 6.0F);
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 DiT analytic checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 DiT analytic checks passed\n";
  return 0;
}

int CheckFocusedOps() {
  if (!HasGfx1151()) {
    std::cout << "SKIP: integrated gfx1151 is unavailable\n";
    return 77;
  }
  CheckAdaLnGateAndSwiGlu();
  if (failures != 0) {
    std::cerr << failures << " MiniMax H3 focused operation checks failed\n";
    return 1;
  }
  std::cout << "MiniMax H3 focused operation checks passed\n";
  return 0;
}

template<typename T>
bool ReadBinary(const std::filesystem::path& path, std::vector<T>* output) {
  std::error_code error;
  const std::uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error || bytes % sizeof(T) != 0 ||
      bytes / sizeof(T) > std::numeric_limits<std::size_t>::max()) {
    std::cerr << "cannot size " << path << '\n';
    return false;
  }
  output->resize(static_cast<std::size_t>(bytes / sizeof(T)));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(bytes));
  if (!input || input.gcount() != static_cast<std::streamsize>(bytes)) {
    std::cerr << "cannot read " << path << '\n';
    return false;
  }
  return true;
}

struct Metrics {
  double relative_l2{0.0};
  double relative_max{0.0};
  std::size_t first_mismatch{std::numeric_limits<std::size_t>::max()};
};

Metrics Compare(std::span<const std::uint16_t> got,
                std::span<const std::uint16_t> expected,
                double element_tolerance) {
  Metrics metrics;
  if (got.size() != expected.size()) {
    metrics.relative_l2 = std::numeric_limits<double>::infinity();
    metrics.relative_max = std::numeric_limits<double>::infinity();
    metrics.first_mismatch = 0;
    return metrics;
  }
  double error_squared = 0.0;
  double expected_squared = 0.0;
  double maximum_error = 0.0;
  double maximum_expected = 0.0;
  for (std::size_t index = 0; index < got.size(); ++index) {
    const double got_value = ToFloat(got[index]);
    const double expected_value = ToFloat(expected[index]);
    const double delta = got_value - expected_value;
    error_squared += delta * delta;
    expected_squared += expected_value * expected_value;
    maximum_error = std::max(maximum_error, std::abs(delta));
    maximum_expected = std::max(maximum_expected, std::abs(expected_value));
    if (metrics.first_mismatch == std::numeric_limits<std::size_t>::max() &&
        std::abs(delta) > element_tolerance) {
      metrics.first_mismatch = index;
    }
  }
  metrics.relative_l2 =
      std::sqrt(error_squared / std::max(expected_squared, 1.0e-24));
  metrics.relative_max = maximum_error / std::max(maximum_expected, 1.0e-12);
  return metrics;
}

bool CheckBoundary(std::string_view name, std::span<const std::uint16_t> got,
                   std::span<const std::uint16_t> expected) {
  const Metrics metrics = Compare(got, expected, 0.02);
  std::cout << name << " relative_l2=" << metrics.relative_l2
            << " relative_max=" << metrics.relative_max;
  if (metrics.first_mismatch != std::numeric_limits<std::size_t>::max()) {
    std::cout << " first_mismatch=" << metrics.first_mismatch
              << " got=" << ToFloat(got[metrics.first_mismatch])
              << " expected=" << ToFloat(expected[metrics.first_mismatch]);
  }
  std::cout << '\n';
  CHECK(metrics.relative_l2 < 1.0e-2);
  CHECK(metrics.relative_max < 1.0e-2);
  return metrics.relative_l2 < 1.0e-2 && metrics.relative_max < 1.0e-2;
}

int CheckReal() {
  if (!HasGfx1151()) {
    std::cout << "SKIP: integrated gfx1151 is unavailable\n";
    return 77;
  }
  const char* root_value = std::getenv("GUFO_H3_MODEL_ROOT");
  const char* golden_value = std::getenv("GUFO_H3_DIT_GOLDEN");
  if (root_value == nullptr || *root_value == '\0' || golden_value == nullptr ||
      *golden_value == '\0') {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT or GUFO_H3_DIT_GOLDEN unset\n";
    return 77;
  }
  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root_value, GUFO_H3_SOURCE_MANIFEST, &error);
  CHECK(inventory.has_value());
  if (!inventory.has_value()) {
    std::cerr << error << '\n';
    return 1;
  }
  const std::filesystem::path golden(golden_value);
  const std::array<std::pair<std::string_view, std::string_view>, 9>
      expected_hashes = {{
          {"hidden.bf16",
           "08516eb99b3c24c644eaf5040680ba1e149be773202223833f8c2e40ab1dd3ec"},
          {"modulation.bf16",
           "dbfa48daedca87c113a556bbc59d0364914efe25e8b0c8deb4c89c80024707ed"},
          {"row_map.u32",
           "816629a33e18daef4f94301aba7b54ba832c50db86fd8ae16a66f86f8d03f5d0"},
          {"rope_cos.bf16",
           "bcc490f17cd0a9d2df03909560bdaa48d6794a2ad1750010067b922fd5be661b"},
          {"rope_sin.bf16",
           "0acfacc27e06e8e9478502dd435135d2f7fadf272dc62a0444bcde8ea2fb7156"},
          {"modulation_attention.bf16",
           "e95dc9de8cddadb5eac131c5834f66a200489dd874b0178bfabf4c123b635bbe"},
          {"attention_output.bf16",
           "d97ccc9d5cd47555b39c5dc13849fcd2625af0381b81994976030fd72acbad11"},
          {"modulation_mlp.bf16",
           "6ba2f26f6324cb49b772636150280c57fc21d1c72a4d83d9078d35865091b8e8"},
          {"block_output.bf16",
           "f5c7c96a6b33586d0aa5c6d77177b11b0c52eda935c40a3e9b325089b60047f5"},
      }};
  for (const auto& [path, hash] : expected_hashes) {
    CHECK(h3::Sha256File(golden / path) == hash);
  }
  std::vector<std::uint16_t> hidden;
  std::vector<std::uint16_t> modulation;
  std::vector<std::uint32_t> row_map;
  std::vector<std::uint16_t> rope_cos;
  std::vector<std::uint16_t> rope_sin;
  std::vector<std::uint16_t> expected_mod_attention;
  std::vector<std::uint16_t> expected_attention;
  std::vector<std::uint16_t> expected_mod_mlp;
  std::vector<std::uint16_t> expected_output;
  CHECK(ReadBinary(golden / "hidden.bf16", &hidden));
  CHECK(ReadBinary(golden / "modulation.bf16", &modulation));
  CHECK(ReadBinary(golden / "row_map.u32", &row_map));
  CHECK(ReadBinary(golden / "rope_cos.bf16", &rope_cos));
  CHECK(ReadBinary(golden / "rope_sin.bf16", &rope_sin));
  CHECK(ReadBinary(golden / "modulation_attention.bf16",
                   &expected_mod_attention));
  CHECK(ReadBinary(golden / "attention_output.bf16", &expected_attention));
  CHECK(ReadBinary(golden / "modulation_mlp.bf16", &expected_mod_mlp));
  CHECK(ReadBinary(golden / "block_output.bf16", &expected_output));
  if (failures != 0 || hidden.size() % h3::kDitHiddenSize != 0 ||
      modulation.size() % (h3::kDitModulationSlots * h3::kDitHiddenSize) != 0) {
    return 1;
  }
  const std::size_t rows = hidden.size() / h3::kDitHiddenSize;
  const std::size_t modulation_rows =
      modulation.size() / (h3::kDitModulationSlots * h3::kDitHiddenSize);
  CHECK(rows == 528);
  CHECK(row_map.size() == rows);
  CHECK(rope_cos.size() == rows * h3::kDitRopeHalf);
  CHECK(rope_sin.size() == rope_cos.size());

  auto session = h3::DitBlockSession::Create(
      *inventory,
      {.block_index = 0, .rows = rows, .modulation_rows = modulation_rows},
      nullptr, &error);
  CHECK(session != nullptr);
  if (session == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(session->gemm_plan_validations() == 4);
  const std::uintptr_t scratch = session->scratch_address();
  h3::DitBlockRetained retained;
  h3::DitBlockTelemetry telemetry;
  const h3::DitBlockInput input = {
      .hidden = hidden,
      .modulation = modulation,
      .row_map = row_map,
      .rope_cos = rope_cos,
      .rope_sin = rope_sin,
  };
  CHECK(session->Run(input, nullptr, &retained, &telemetry, &error));
  if (retained.block_output.empty()) {
    std::cerr << error << '\n';
    return 1;
  }
  CHECK(telemetry.dispatches > 0);
  CHECK(telemetry.attention_backend == "ck_bf16_wmma_fused");
  CHECK(telemetry.weight_bytes > 700ULL * 1024ULL * 1024ULL);
  CHECK(telemetry.activation_bytes > 0);
  CHECK(telemetry.scratch_address == scratch);
  CHECK(telemetry.first_non_finite.empty());
  CheckBoundary("modulation_attention", retained.modulation_attention,
                expected_mod_attention);
  CheckBoundary("attention_output", retained.attention_output,
                expected_attention);
  CheckBoundary("modulation_mlp", retained.modulation_mlp, expected_mod_mlp);
  CheckBoundary("block_output", retained.block_output, expected_output);

  h3::CancellationToken cancellation;
  cancellation.Cancel();
  h3::DitBlockRetained cancelled;
  h3::DitBlockTelemetry cancelled_telemetry;
  CHECK(!session->Run(input, &cancellation, &cancelled, &cancelled_telemetry,
                      &error));
  CHECK(error.find("cancelled") != std::string::npos);
  std::cout << "H3 DiT block0 rows=" << rows << " load_ms=" << telemetry.load_ms
            << " gpu_ms=" << telemetry.gpu_ms << " weights_gib="
            << static_cast<double>(telemetry.weight_bytes) /
                   (1024.0 * 1024.0 * 1024.0)
            << " activation_mib="
            << static_cast<double>(telemetry.activation_bytes) /
                   (1024.0 * 1024.0)
            << '\n';
  return failures == 0 ? 0 : 1;
}

int ProfileBlock(std::size_t rows, bool row_parallel) {
  if (!HasGfx1151()) {
    std::cout << "SKIP: integrated gfx1151 is unavailable\n";
    return 77;
  }
  const char* root_value = std::getenv("GUFO_H3_MODEL_ROOT");
  if (root_value == nullptr || *root_value == '\0') {
    std::cout << "SKIP: GUFO_H3_MODEL_ROOT unset\n";
    return 77;
  }

  constexpr std::size_t kModulationRows = 6;
  std::vector<std::uint16_t> hidden(rows * h3::kDitHiddenSize);
  std::vector<std::uint16_t> modulation(
      kModulationRows * h3::kDitModulationSlots * h3::kDitHiddenSize);
  std::vector<std::uint32_t> row_map(rows);
  std::vector<std::uint16_t> rope_cos(rows * h3::kDitRopeHalf);
  std::vector<std::uint16_t> rope_sin(rows * h3::kDitRopeHalf);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    hidden[index] =
        ToBf16(std::sin(static_cast<float>(index) * 0.00013F) * 0.25F);
  }
  for (std::size_t index = 0; index < modulation.size(); ++index) {
    modulation[index] =
        ToBf16(std::cos(static_cast<float>(index) * 0.00031F) * 0.04F);
  }
  for (std::size_t row = 0; row < rows; ++row) {
    row_map[row] = static_cast<std::uint32_t>(row % kModulationRows);
    for (std::size_t frequency = 0; frequency < h3::kDitRopeHalf; ++frequency) {
      const float angle = static_cast<float>(row) *
                          static_cast<float>(frequency + 1U) * 0.000017F;
      rope_cos[row * h3::kDitRopeHalf + frequency] = ToBf16(std::cos(angle));
      rope_sin[row * h3::kDitRopeHalf + frequency] = ToBf16(std::sin(angle));
    }
  }

  std::string error;
  auto inventory =
      h3::ModelInventory::Inspect(root_value, GUFO_H3_SOURCE_MANIFEST, &error);
  CHECK(inventory.has_value());
  if (!inventory.has_value()) {
    std::cerr << error << '\n';
    return 1;
  }
  auto session =
      h3::DitBlockSession::Create(*inventory,
                                  {.block_index = 0,
                                   .rows = rows,
                                   .modulation_rows = kModulationRows,
                                   .row_parallel_attention = row_parallel},
                                  nullptr, &error);
  CHECK(session != nullptr);
  if (session == nullptr ||
      !session->PrepareConstants(modulation, rope_cos, rope_sin, nullptr,
                                 &error)) {
    std::cerr << error << '\n';
    return 1;
  }

  std::vector<std::uint16_t> output(hidden.size());
  h3::DitBlockTelemetry telemetry;
  const auto begin = std::chrono::steady_clock::now();
  CHECK(session->RunPrepared({hidden, row_map}, nullptr, output, &telemetry,
                             &error));
  const double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
  if (failures != 0) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto bytes = std::span<const unsigned char>(
      reinterpret_cast<const unsigned char*>(output.data()),
      output.size() * sizeof(output.front()));
  if (const char* output_path = std::getenv("GUFO_H3_DIT_PROFILE_OUTPUT");
      output_path != nullptr && *output_path != '\0') {
    std::ofstream output_file(output_path, std::ios::binary);
    output_file.write(
        reinterpret_cast<const char*>(output.data()),
        static_cast<std::streamsize>(output.size() * sizeof(output.front())));
    CHECK(output_file.good());
  }
  std::cout << "H3 DiT production block profile rows=" << rows
            << " gpu_ms=" << telemetry.gpu_ms << " wall_ms=" << wall_ms
            << " attention_backend=" << telemetry.attention_backend
            << " activation_mib="
            << static_cast<double>(telemetry.activation_bytes) /
                   (1024.0 * 1024.0)
            << " output_sha256=" << h3::Sha256(bytes) << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view(argv[1]) != "--analytic" &&
                    std::string_view(argv[1]) != "--ops" &&
                    std::string_view(argv[1]) != "--real" &&
                    std::string_view(argv[1]) != "--profile-1872" &&
                    std::string_view(argv[1]) != "--profile-7136-scalar" &&
                    std::string_view(argv[1]) != "--profile-7136-fused" &&
                    std::string_view(argv[1]) != "--profile-37716-fused")) {
    std::cerr << "usage: " << argv[0]
              << " --analytic|--ops|--real|--profile-1872|"
                 "--profile-7136-scalar|--profile-7136-fused|"
                 "--profile-37716-fused\n";
    return 2;
  }
  if (std::string_view(argv[1]) == "--ops") {
    return CheckFocusedOps();
  }
  if (std::string_view(argv[1]) == "--real") {
    return CheckReal();
  }
  if (std::string_view(argv[1]) == "--profile-1872") {
    return ProfileBlock(1872, true);
  }
  if (std::string_view(argv[1]) == "--profile-7136-scalar") {
    return ProfileBlock(7136, false);
  }
  if (std::string_view(argv[1]) == "--profile-7136-fused") {
    return ProfileBlock(7136, true);
  }
  if (std::string_view(argv[1]) == "--profile-37716-fused") {
    return ProfileBlock(37716, true);
  }
  return CheckAnalytic();
}
