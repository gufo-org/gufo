#include <hip/hip_runtime.h>

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "src/models/moss_image/hip/ops.hpp"
#include "src/models/qwen_image_21/weights.hpp"

namespace {
namespace ops = gufo::models::moss_image::hip;
namespace image = gufo::models::qwen_image_21;

// Round-to-nearest-even BF16, as torch.bfloat16 for finite values.
float Round(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits = (bits + 32767 + ((bits >> 16) & 1)) & 0xffff0000U;
  return std::bit_cast<float>(bits);
}

std::vector<float> Random(std::mt19937& rng, std::size_t count, float scale) {
  std::normal_distribution<float> normal(0, scale);
  std::vector<float> values(count);
  for (auto& value : values)
    value = Round(normal(rng));
  return values;
}

// Distance in BF16 units of least precision.
int Ulps(float a, float b) {
  const auto order = [](float v) {
    const auto bits = static_cast<std::int32_t>(std::bit_cast<std::uint32_t>(v));
    return bits < 0 ? std::int32_t(0x80000000U - std::uint32_t(bits)) : bits;
  };
  return std::abs(order(a) - order(b)) >> 16;
}

void Modulation(ops::Runtime& rt, std::mt19937& rng) {
  constexpr int kRows = 37, kCols = 96, kParts = 6;
  const auto x = Random(rng, kRows * kCols, 2.0F);
  const auto y = Random(rng, kRows * kCols, 1.5F);
  const auto mod = Random(rng, kParts * kCols, 0.7F);
  const auto dx = rt.Upload(x, kRows, kCols), dy = rt.Upload(y, kRows, kCols),
             dmod = rt.Upload(mod, 1, kParts * kCols);
  // Each PyTorch BF16 operation rounds: (1 + scale), x * it, then + shift.
  const auto shifted = rt.Download(ops::ShiftScale(rt, dx, dmod, 3, 4));
  const auto gated = rt.Download(ops::GatedResidual(rt, dx, dy, dmod, 5));
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c) {
      const auto i = std::size_t(r) * kCols + c;
      const float scale = Round(1.0F + mod[4 * kCols + c]);
      assert(shifted[i] == Round(Round(x[i] * scale) + mod[3 * kCols + c]));
      assert(gated[i] == Round(x[i] + Round(y[i] * mod[5 * kCols + c])));
    }
}

void Situ(ops::Runtime& rt, std::mt19937& rng) {
  // One fused up/gate matrix: hidden in the first half, gate in the second.
  // Wide values reach both tanh saturations.
  constexpr int kRows = 19, kWidth = 160;
  const auto fused = Random(rng, kRows * 2 * kWidth, 12.0F);
  const auto out = rt.Download(
      ops::Situ(rt, rt.Upload(fused, kRows, 2 * kWidth), 0,
                rt.Upload(fused, kRows, 2 * kWidth), kWidth, kWidth));
  int differing = 0;
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kWidth; ++c) {
      const float h = fused[std::size_t(r) * 2 * kWidth + c];
      const float g = fused[std::size_t(r) * 2 * kWidth + kWidth + c];
      const float left = Round(Round(4.0F * Round(std::tanh(Round(g / 4.0F)))) *
                               Round(1.0F / (1.0F + std::exp(-g))));
      const float right = Round(25.0F * Round(std::tanh(Round(h / 25.0F))));
      const float expected = Round(left * right);
      const float actual = out[std::size_t(r) * kWidth + c];
      // Device and host libm may differ by an FP32 ulp before rounding.
      assert(Ulps(actual, expected) <= 1);
      differing += actual != expected;
    }
  assert(differing * 100 < kRows * kWidth);
}

void GroupNorm(ops::Runtime& rt, std::mt19937& rng) {
  constexpr int kPixels = 300, kChannels = 64, kGroups = 32;
  auto x = Random(rng, kPixels * kChannels, 3.0F);
  for (auto& value : x)
    value = Round(value + 5.0F);  // a nonzero mean exercises centering
  const auto weight = Random(rng, kChannels, 1.0F);
  const auto bias = Random(rng, kChannels, 1.0F);
  for (const bool silu : {false, true}) {
    const auto out = rt.Download(ops::GroupNorm(
        rt, rt.Upload(x, kPixels, kChannels), rt.Upload(weight, 1, kChannels),
        rt.Upload(bias, 1, kChannels), kGroups, 1e-6F, silu));
    const int per_group = kChannels / kGroups;
    for (int group = 0; group < kGroups; ++group) {
      double sum = 0, square = 0;
      for (int p = 0; p < kPixels; ++p)
        for (int c = group * per_group; c < (group + 1) * per_group; ++c) {
          const double v = x[std::size_t(p) * kChannels + c];
          sum += v;
          square += v * v;
        }
      const double count = double(kPixels) * per_group;
      const double mean = sum / count;
      const double inverse = 1 / std::sqrt(square / count - mean * mean + 1e-6);
      for (int p = 0; p < kPixels; ++p)
        for (int c = group * per_group; c < (group + 1) * per_group; ++c) {
          const auto i = std::size_t(p) * kChannels + c;
          double v = (x[i] - mean) * inverse * weight[c] + bias[c];
          if (silu)
            v = v / (1 + std::exp(-v));
          assert(std::abs(out[i] - v) <= std::abs(v) / 128 + 1e-5);
        }
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
    return 77;
  const auto weights =
      image::Weights::FromFiles(std::span<const image::SafetensorsSource>{});
  ops::Runtime rt(weights);
  std::mt19937 rng(20261030);
  Modulation(rt, rng);
  Situ(rt, rng);
  GroupNorm(rt, rng);
  std::cout << "MOSS-Image kernel contracts passed\n";
}
