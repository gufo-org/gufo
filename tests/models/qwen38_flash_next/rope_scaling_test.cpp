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

}  // namespace

int main() {
  try {
    CheckGoldenTable();
    CheckOffAndValidation();
    CheckInvertedCorrectionRange();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
