// YaRN in the scalar CPU oracle (cpu::Rope): bit-identical to the pre-YaRN
// loop when disabled, and matching the HF golden table (shared with
// rope_scaling_test.cpp via rope_scaling_golden.hpp) when enabled. This test
// links the model oracle library, so per docs/DEVELOPMENT.md it stays out of
// the hosted `check-pr` set; it is still a "cpu"-labeled CTest target run by
// the cpu-test preset and any explicit validation run.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"
#include "tests/models/qwen38_flash_next/rope_scaling_golden.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

qfn::RopeScaling Yarn(float factor) {
  return {.factor = factor, .original_context = 262144};
}

// Today's cpu::Rope loop, verbatim: the disabled path must equal it bitwise.
void LegacyRope(float* x, std::uint32_t heads, std::uint32_t head_dim,
                std::uint32_t rotary_dim, std::uint32_t pos, float theta) {
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t h = 0; h < heads; ++h) {
    float* v = x + static_cast<std::size_t>(h) * head_dim;
    for (std::uint32_t i = 0; i < half; ++i) {
      const float freq = std::pow(theta, -2.0F * static_cast<float>(i) /
                                             static_cast<float>(rotary_dim));
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const float a = v[i];
      const float b = v[i + half];
      v[i] = a * c - b * s;
      v[i + half] = a * s + b * c;
    }
  }
}

std::vector<float> Values(std::size_t count, std::uint32_t seed) {
  std::vector<float> out(count);
  for (float& v : out) {
    seed = seed * 1664525U + 1013904223U;
    v = static_cast<float>(static_cast<int>(seed >> 16) - 32768) / 32768.0F;
  }
  return out;
}

void CheckCpuOracle() {
  constexpr std::uint32_t kHeads = 4, kDim = 128, kRotary = 64;
  // Off: bitwise identical to the pre-YaRN loop at native and deep positions.
  for (const std::uint32_t pos : {0U, 1000U, 131069U, 262143U}) {
    auto a = Values(kHeads * kDim, pos + 7);
    auto b = a;
    gufo::models::qwen38_flash_next::cpu::Rope(a.data(), kHeads, kDim, kRotary,
                                               pos, 1e7F);
    LegacyRope(b.data(), kHeads, kDim, kRotary, pos, 1e7F);
    Require(a == b, "disabled YaRN changed cpu::Rope bits");
  }
  // On: every pair against the HF float32 inv_freq in double, mscale on
  // cos and sin, partial-rotary tail untouched.
  constexpr std::uint32_t kPos = 1000;
  for (const qfn::testing::Golden& g : qfn::testing::kGolden) {
    const auto yarn = qfn::MakeYarnRope(Yarn(g.factor), kRotary, 1e7F);
    const auto input = Values(kHeads * kDim, 99);
    auto out = input;
    gufo::models::qwen38_flash_next::cpu::Rope(out.data(), kHeads, kDim,
                                               kRotary, kPos, 1e7F, yarn);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const float* x = input.data() + h * kDim;
      const float* y = out.data() + h * kDim;
      for (std::uint32_t i = 0; i < kRotary / 2; ++i) {
        const double angle = kPos * g.inv_freq[i];
        const double c = std::cos(angle) * g.attention_factor;
        const double s = std::sin(angle) * g.attention_factor;
        const double lo = x[i] * c - x[i + 32] * s;
        const double hi = x[i] * s + x[i + 32] * c;
        // fp32 angle at position 1000: < 3e-4 rad from HF's fp32 product.
        Require(std::abs(y[i] - lo) <= 5e-4 && std::abs(y[i + 32] - hi) <= 5e-4,
                "YaRN cpu::Rope disagrees with the HF formula");
      }
      for (std::uint32_t i = kRotary; i < kDim; ++i)
        Require(y[i] == x[i], "YaRN touched the non-rotary tail");
    }
  }
  std::cout << "CPU oracle: off bit-identical, on matches HF at pos 1000\n";
}

}  // namespace

int main() {
  try {
    CheckCpuOracle();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
