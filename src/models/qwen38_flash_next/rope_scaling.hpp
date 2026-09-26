#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_SCALING_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_SCALING_HPP_

#include <cstdint>
#include <string>

#if defined(__HIPCC__)
#define QFN_HOST_DEVICE __host__ __device__
#else
#define QFN_HOST_DEVICE
#endif

namespace gufo::models::qwen38_flash_next {

/// Static YaRN context extension as in Hugging Face `rope_type: yarn`:
/// truncated correction range, linear ramp between the beta bounds, and
/// attention factor 0.1 ln(factor) + 1 applied to cos and sin. `factor == 1`
/// is off and leaves every rope site on its original instruction sequence.
struct RopeScaling {
  float factor{1.0F};
  std::uint32_t original_context{0};  ///< Pretraining context (262144).
  float beta_fast{32.0F};
  float beta_slow{1.0F};

  [[nodiscard]] bool Enabled() const noexcept { return factor != 1.0F; }
  friend bool operator==(const RopeScaling&, const RopeScaling&) = default;
};

/// Rope angles use float positions, which are exact below 2^24.
inline constexpr std::uint32_t kMaxRopePositions = 1U << 24;

/// Launch constants shared by the CPU oracle and every HIP rope site.
struct YarnRope {
  std::uint32_t enabled{0};
  float low{0.0F};         ///< First ramped rotary pair.
  float high{0.0F};        ///< Pair from which frequencies are divided.
  float inv_factor{1.0F};  ///< 1 / factor.
  float mscale{1.0F};      ///< Multiplies cos and sin.
};

/// Rejects non-finite or sub-unit factors, an original context outside
/// (0, native], betas that are not beta_fast > beta_slow > 0, and scaled
/// contexts past kMaxRopePositions. `factor == 1` is always valid (off).
[[nodiscard]] bool ValidateRopeScaling(const RopeScaling& scaling,
                                       std::uint32_t native_context,
                                       std::string* error);

/// Native context when off; floor(original x factor), never below native.
[[nodiscard]] std::uint32_t ScaledContextLength(
    const RopeScaling& scaling, std::uint32_t native_context) noexcept;

/// Correction range and scale for a rotary dimension and base.
[[nodiscard]] YarnRope MakeYarnRope(const RopeScaling& scaling,
                                    std::uint32_t rotary_dim,
                                    float theta) noexcept;

/// Disk-cache position policy: "absolute-v1" when off (unchanged identity).
[[nodiscard]] std::string RopePositionPolicy(const RopeScaling& scaling);

/// Frequency multiplier of rotary pair `pair`: 1 below `low`, 1 / factor
/// from `high`, linear between (HF `linear_ramp_factor`).
QFN_HOST_DEVICE inline float YarnPairMultiplier(const YarnRope& yarn,
                                                std::uint32_t pair) noexcept {
  const float width = yarn.high - yarn.low;
  const float span = width > 0.001F ? width : 0.001F;
  float ramp = (static_cast<float>(pair) - yarn.low) / span;
  ramp = ramp < 0.0F ? 0.0F : (ramp > 1.0F ? 1.0F : ramp);
  return 1.0F - ramp * (1.0F - yarn.inv_factor);
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROPE_SCALING_HPP_
