// YaRN parameters against Hugging Face transformers 5.17.0
// `_compute_yarn_parameters` (Flash-Next: rotary 64 of 256, theta 1e7,
// original context 262144, beta 32/1, truncated correction range). The
// regeneration command is in the pi-lab plan 2026-09-26-gufo-flash-next-yarn.
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Golden {
  float factor;
  double attention_factor;
  std::array<double, 32> inv_freq;
};

constexpr std::array<Golden, 3> kGolden{{
    {1.5625F,
     1.044628710262842,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.997506621e-04,
         2.877672960e-04, 1.652974315e-04, 9.469212091e-05, 5.408186553e-05,
         3.078384179e-05, 1.745583177e-05, 9.855529242e-06, 5.955661436e-06,
         3.598984449e-06, 2.174853307e-06, 1.314255996e-06, 7.942002185e-07,
         4.799323392e-07, 2.900213474e-07, 1.752588616e-07, 1.059082990e-07,
     }},
    {2.5F,
     1.0916290731874154,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.840516776e-04,
         2.687936067e-04, 1.480988576e-04, 8.083473222e-05, 4.361441097e-05,
         2.319330997e-05, 1.210440860e-05, 6.159706118e-06, 3.722288511e-06,
         2.249365252e-06, 1.359283260e-06, 8.214099694e-07, 4.963750939e-07,
         2.999576907e-07, 1.812633457e-07, 1.095367850e-07, 6.619268333e-08,
     }},
    {4.0F,
     1.138629436111989,
     {
         1.000000000e+00, 6.042963862e-01, 3.651741445e-01, 2.206733972e-01,
         1.333521456e-01, 8.058421314e-02, 4.869675264e-02, 2.942727320e-02,
         1.778279431e-02, 1.074607857e-02, 6.493816618e-03, 3.924189601e-03,
         2.371373819e-03, 1.433012658e-03, 8.659643354e-04, 4.742398160e-04,
         2.569350763e-04, 1.373497507e-04, 7.217386883e-05, 3.707224823e-05,
         1.844922372e-05, 8.759770026e-06, 3.849816039e-06, 2.326430149e-06,
         1.405853368e-06, 8.495520660e-07, 5.133812238e-07, 3.102344408e-07,
         1.874735602e-07, 1.132895946e-07, 6.846049416e-08, 4.137042708e-08,
     }},
}};

qfn::RopeScaling Yarn(float factor) {
  return {.factor = factor, .original_context = 262144};
}

void CheckGoldenTable() {
  for (const Golden& g : kGolden) {
    const auto yarn = qfn::MakeYarnRope(Yarn(g.factor), 64, 1e7F);
    Require(yarn.enabled == 1, "YaRN must be enabled for factor > 1");
    Require(yarn.low == 14.0F && yarn.high == 22.0F,
            "correction range must be the truncated HF range [14, 22]");
    Require(std::abs(yarn.mscale - g.attention_factor) <= 1e-7,
            "mscale differs from HF attention_factor");
    for (std::uint32_t i = 0; i < 32; ++i) {
      const float freq = std::pow(1e7F, -2.0F * static_cast<float>(i) / 64.0F) *
                         qfn::YarnPairMultiplier(yarn, i);
      const double rel = std::abs(freq - g.inv_freq[i]) / g.inv_freq[i];
      if (rel > 1e-6) {
        throw std::runtime_error("inv_freq pair " + std::to_string(i) +
                                 " differs from HF by " + std::to_string(rel));
      }
    }
  }
  std::cout << "YaRN inv_freq and mscale match HF for 1.5625/2.5/4\n";
}

void CheckOffAndValidation() {
  const qfn::RopeScaling off;
  Require(!off.Enabled(), "default scaling must be off");
  const auto none = qfn::MakeYarnRope(off, 64, 1e7F);
  Require(none.enabled == 0 && none.mscale == 1.0F,
          "disabled YaRN must not scale");
  Require(qfn::ScaledContextLength(off, 262144) == 262144, "off keeps native");
  Require(qfn::ScaledContextLength(Yarn(1.5625F), 262144) == 409600,
          "1.5625 x 262144");
  Require(qfn::ScaledContextLength(Yarn(2.5F), 262144) == 655360,
          "2.5 x 262144");
  Require(qfn::ScaledContextLength(Yarn(4.0F), 262144) == 1048576,
          "4 x 262144");
  Require(qfn::ScaledContextLength(
              qfn::RopeScaling{.factor = NAN, .original_context = 262144},
              262144) == 262144,
          "a non-finite factor must keep native context, not invoke UB");
  std::string error;
  Require(qfn::ValidateRopeScaling(off, 262144, &error), "off is valid");
  Require(qfn::ValidateRopeScaling(Yarn(2.5F), 262144, &error), error);
  Require(qfn::ValidateRopeScaling(
              qfn::RopeScaling{.factor = 64.0F, .original_context = 262144},
              262144, &error),
          "64 x 262144 == 2^24 must be accepted at the exact boundary");
  for (const qfn::RopeScaling bad :
       {qfn::RopeScaling{.factor = 0.5F, .original_context = 262144},
        qfn::RopeScaling{.factor = NAN, .original_context = 262144},
        qfn::RopeScaling{.factor = 2.5F, .original_context = 0},
        qfn::RopeScaling{.factor = 2.5F, .original_context = 262145},
        qfn::RopeScaling{.factor = 2.5F,
                         .original_context = 262144,
                         .beta_fast = 1.0F,
                         .beta_slow = 32.0F},
        qfn::RopeScaling{
            .factor = 2.5F, .original_context = 262144, .beta_fast = NAN},
        qfn::RopeScaling{.factor = 65.0F, .original_context = 262144}}) {
    error.clear();
    Require(!qfn::ValidateRopeScaling(bad, 262144, &error) && !error.empty(),
            "invalid YaRN scaling accepted");
  }
  std::cout << "YaRN validation and scaled context checks passed\n";
}

void CheckInvertedCorrectionRange() {
  // original_context=1, factor=2 truncates to low=0, high=-3 (HF: max < min).
  // HF's linear_ramp_factor only special-cases min == max, so a negative
  // denominator clamps every ramp to 0: no interpolation at all (multiplier 1
  // for every pair), while attention_factor (mscale) still applies.
  const auto yarn = qfn::MakeYarnRope(
      qfn::RopeScaling{.factor = 2.0F, .original_context = 1}, 64, 1e7F);
  Require(yarn.enabled == 1, "YaRN must still be enabled");
  for (std::uint32_t i = 0; i < 32; ++i) {
    Require(qfn::YarnPairMultiplier(yarn, i) == 1.0F,
            "inverted correction range must fall back to no interpolation");
  }
  const double expected_mscale = 0.1 * std::log(2.0) + 1.0;
  Require(std::abs(yarn.mscale - expected_mscale) <= 1e-7,
          "mscale must still apply when the correction range is degenerate");
  std::cout << "YaRN inverted correction range falls back to HF's "
               "no-interpolation behavior\n";
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
  for (const Golden& g : kGolden) {
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

void CheckPositionPolicy() {
  Require(qfn::RopePositionPolicy({}) == "absolute-v1",
          "off must keep the existing disk-cache identity line");
  Require(qfn::RopePositionPolicy(Yarn(1.5625F)) ==
              "absolute-yarn-v1;factor=0x1.9p+0;original=262144;"
              "beta_fast=0x1p+5;beta_slow=0x1p+0",
          "YaRN policy must encode every parameter exactly");
  Require(qfn::RopePositionPolicy(Yarn(2.5F)) !=
              qfn::RopePositionPolicy(Yarn(1.5625F)),
          "a factor change must change the identity");
  std::cout << "position policy strings pinned\n";
}

}  // namespace

int main() {
  try {
    CheckGoldenTable();
    CheckOffAndValidation();
    CheckInvertedCorrectionRange();
    CheckCpuOracle();
    CheckPositionPolicy();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
