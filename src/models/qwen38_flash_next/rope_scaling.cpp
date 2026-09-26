#include "src/models/qwen38_flash_next/rope_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <locale>
#include <numbers>
#include <sstream>

namespace gufo::models::qwen38_flash_next {

bool ValidateRopeScaling(const RopeScaling& scaling,
                         std::uint32_t native_context, std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr)
      *error = message;
    return false;
  };
  if (!std::isfinite(scaling.factor) || scaling.factor < 1.0F)
    return fail("YaRN factor must be finite and at least 1");
  if (!scaling.Enabled())
    return true;
  if (scaling.original_context == 0 ||
      scaling.original_context > native_context)
    return fail("YaRN original context must be within the native context");
  if (!std::isfinite(scaling.beta_fast) || !std::isfinite(scaling.beta_slow) ||
      scaling.beta_slow <= 0.0F || scaling.beta_fast <= scaling.beta_slow)
    return fail("YaRN betas must satisfy beta_fast > beta_slow > 0");
  if (static_cast<double>(scaling.original_context) * scaling.factor >
      static_cast<double>(kMaxRopePositions))
    return fail("YaRN context exceeds 16777216 exact rope positions");
  return true;
}

std::uint32_t ScaledContextLength(const RopeScaling& scaling,
                                  std::uint32_t native_context) noexcept {
  if (!scaling.Enabled() || !std::isfinite(scaling.factor))
    return native_context;
  const double scaled =
      std::floor(static_cast<double>(scaling.original_context) *
                 static_cast<double>(scaling.factor));
  const auto bounded = static_cast<std::uint32_t>(
      std::min(scaled, static_cast<double>(kMaxRopePositions)));
  return std::max(native_context, bounded);
}

YarnRope MakeYarnRope(const RopeScaling& scaling, std::uint32_t rotary_dim,
                      float theta) noexcept {
  if (!scaling.Enabled())
    return {};
  // HF find_correction_dim: the pair index completing `rotations` turns over
  // the original context.
  const auto dim = [&](double rotations) {
    return static_cast<double>(rotary_dim) *
           std::log(static_cast<double>(scaling.original_context) /
                    (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(static_cast<double>(theta)));
  };
  double low = std::max(std::floor(dim(scaling.beta_fast)), 0.0);
  double high = std::min(std::ceil(dim(scaling.beta_slow)),
                         static_cast<double>(rotary_dim - 1));
  if (high < low) {
    // HF linear_ramp_factor(min, max, dim) only special-cases min == max; a
    // negative (max - min) denominator makes every ramp clamp to 0, i.e. no
    // interpolation at all (multiplier 1 for every pair). Forcing low = high
    // past the last pair reproduces that with our clamped ramp formula.
    low = high = static_cast<double>(rotary_dim);
  }
  return {
      .enabled = 1,
      .low = static_cast<float>(low),
      .high = static_cast<float>(high),
      .inv_factor = 1.0F / scaling.factor,
      .mscale = static_cast<float>(
          0.1 * std::log(static_cast<double>(scaling.factor)) + 1.0),
  };
}

std::string RopePositionPolicy(const RopeScaling& scaling) {
  if (!scaling.Enabled())
    return "absolute-v1";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::hexfloat << "absolute-yarn-v1;factor=" << scaling.factor
      << ";original=" << scaling.original_context
      << ";beta_fast=" << scaling.beta_fast
      << ";beta_slow=" << scaling.beta_slow;
  return out.str();
}

}  // namespace gufo::models::qwen38_flash_next
