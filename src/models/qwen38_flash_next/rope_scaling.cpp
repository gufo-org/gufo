#include "src/models/qwen38_flash_next/rope_scaling.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <numbers>
#include <optional>
#include <sstream>
#include <utility>

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

std::optional<RopeScaling> RopeScalingForContext(
    std::uint32_t requested, std::uint32_t native) noexcept {
  // native == 0 means the caller could not read a native context (e.g. a
  // GGUF missing qwen4exp.context_length); dividing by it would be +-inf.
  // Leave YaRN off and let the caller's existing "context exceeds the
  // model's N tokens" check in Model::Load reject the request instead.
  if (native == 0 || requested <= native)
    return std::nullopt;
  RopeScaling scaling{
      .factor = static_cast<float>(static_cast<double>(requested) /
                                   static_cast<double>(native)),
      .original_context = native,
  };
  // The float ratio can round down from the exact double value and floor
  // `ScaledContextLength` one token below `requested`; nudge the factor up
  // by the smallest float steps until the scaled ceiling covers it. Only
  // fires for a non-power-of-two native context (the shipped 262,144 is
  // exact for every valid `requested`, so the loop is a zero-iteration
  // no-op there). Bounded by `ScaledContextLength`'s own kMaxRopePositions
  // clamp, so a pathological input still terminates.
  while (std::isfinite(scaling.factor) &&
         ScaledContextLength(scaling, native) < requested &&
         ScaledContextLength(scaling, native) < kMaxRopePositions) {
    scaling.factor =
        std::nextafter(scaling.factor, std::numeric_limits<float>::infinity());
  }
  return scaling;
}

bool ResolveRopeScaling(std::uint32_t requested, std::uint32_t native,
                        bool enable_yarn, std::optional<RopeScaling>* scaling,
                        std::string* error) {
  *scaling = std::nullopt;
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return false;
  };
  if (native == 0 && enable_yarn)
    return fail("--enable-yarn needs the model's native context length");
  if (native == 0)
    return true;
  const std::string native_text = std::to_string(native);
  if (requested > native && !enable_yarn) {
    return fail("context above the native " + native_text +
                " tokens needs static YaRN, which applies to every "
                "request; pass --enable-yarn (requested " +
                std::to_string(requested) + ")");
  }
  if (requested <= native && enable_yarn) {
    return fail("--enable-yarn has no effect: context " +
                std::to_string(requested) + " is within the native " +
                native_text + " tokens");
  }
  *scaling = RopeScalingForContext(requested, native);
  return true;
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
  // Hex floats go through to_chars: its shortest round-trip form is
  // specified, while iostream hexfloat padding differs per standard library.
  std::array<char, 64> digits{};
  const auto emit = [&](float value) {
    const auto result =
        std::to_chars(digits.data() + 2, digits.data() + digits.size(), value,
                      std::chars_format::hex);
    const auto length = static_cast<std::size_t>(result.ptr - digits.data());
    digits[0] = '0';
    digits[1] = 'x';
    return std::string_view(digits.data(), length);
  };
  std::string out = "absolute-yarn-v1;factor=";
  out += emit(scaling.factor);
  out += ";original=";
  out += std::to_string(scaling.original_context);
  out += ";beta_fast=";
  out += emit(scaling.beta_fast);
  out += ";beta_slow=";
  out += emit(scaling.beta_slow);
  return out;
}

}  // namespace gufo::models::qwen38_flash_next
