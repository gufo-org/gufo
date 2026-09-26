// YaRN parameters against Hugging Face transformers 5.17.0
// `_compute_yarn_parameters` (Flash-Next: rotary 64 of 256, theta 1e7,
// original context 262144, beta 32/1, truncated correction range). The
// golden table and its regeneration command live in rope_scaling_golden.hpp
// (shared with the CPU-oracle check in rope_scaling_oracle_test.cpp, which
// links the model oracle and stays out of the hosted set; see
// docs/DEVELOPMENT.md).
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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

void CheckGoldenTable() {
  for (const qfn::testing::Golden& g : qfn::testing::kGolden) {
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

// The snapshot header uses this to decide whether two configs restore each
// other's disk cache; it must collapse every disabled config to one
// identity, since RopePositionPolicy already reports "absolute-v1" for all
// of them regardless of original_context/beta.
void CheckNormalizedRopeScaling() {
  const qfn::RopeScaling default_off;
  const qfn::RopeScaling leftover_off{
      .original_context = 262144, .beta_fast = 1.0F, .beta_slow = 32.0F};
  Require(qfn::NormalizedRopeScaling(default_off) ==
              qfn::NormalizedRopeScaling(leftover_off),
          "disabled scaling must normalize to one identity regardless of "
          "leftover original_context/beta fields");
  Require(qfn::NormalizedRopeScaling(default_off) == qfn::RopeScaling{},
          "normalizing an already-default off scaling must be a no-op");
  const auto on = Yarn(2.5F);
  Require(qfn::NormalizedRopeScaling(on) == on,
          "enabled scaling must pass through unchanged");
  std::cout << "normalized rope scaling collapses disabled configs to one "
               "identity\n";
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
    CheckNormalizedRopeScaling();
    CheckPositionPolicy();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
