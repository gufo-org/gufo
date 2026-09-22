#ifndef GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_
#define GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace gufo::speculative {

enum class DFlashDraftPolicy : std::uint32_t { kFixed, kAdaptive };

[[nodiscard]] inline DFlashDraftPolicy ParseDFlashDraftPolicy(
    std::string_view value) {
  if (value == "fixed")
    return DFlashDraftPolicy::kFixed;
  if (value.empty() || value == "adaptive")
    return DFlashDraftPolicy::kAdaptive;
  throw std::invalid_argument("DFlash2 draft policy must be fixed or adaptive");
}

[[nodiscard]] inline std::string_view DFlashDraftPolicyName(
    DFlashDraftPolicy policy) {
  return policy == DFlashDraftPolicy::kFixed ? "fixed" : "adaptive";
}

// Select the whole block before drawing any proposals. Decisions depend only
// on committed acceptance history, so scheduling and timing cannot change RNG
// consumption on a replay. Full acceptance is a censored observation: probe
// upward instead of treating the current block limit as the true stopping
// point.
class DFlashLengthController {
public:
  static constexpr std::uint32_t kMaxDraftTokens = 7;

  DFlashLengthController(DFlashDraftPolicy policy, std::uint32_t limit,
                         bool q8_target = false)
      : policy_(policy),
        limit_(std::clamp(limit, 1U, kMaxDraftTokens)),
        q8_target_(q8_target) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(
      std::uint32_t budget, std::uint32_t position = 0) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (policy_ == DFlashDraftPolicy::kFixed || cap == 0)
      return cap;

    // Weight reads impose a substantial cost even on a short verification.
    // gfx1151 profiles estimate draft-plus-verification cost at each width.
    // Use that offline estimate, never request timings, and maximize expected
    // emitted tokens (including the target correction) per unit of work.
    // At the cap, full acceptance is still censored: the actual accepted run
    // can be longer than our limit. Probe the wider profitable block instead
    // of imposing an artificial rejection probability (1/8 at limit seven).
    const float probability =
        mean_ == static_cast<float>(limit_) ? 1.0F : mean_ / (mean_ + 1.0F);
    float survival = 1.0F;
    float expected_tokens = 1.0F;
    float best_score = 0.0F;
    std::uint32_t best_length = 1;
    const float context_scale =
        static_cast<float>(position > 2048 ? position - 2048 : 0) / 30720.0F;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= probability;
      expected_tokens += survival;
      const float cost = q8_target_
                             ? 1.0F + 0.02F * static_cast<float>(length)
                             : kQ4RelativeCost[length - 1] +
                                   context_scale * kQ4ContextCost[length - 1];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best_length = length;
      }
    }
    return best_length;
  }

  void Observe(std::size_t accepted, std::size_t drafted) noexcept {
    if (policy_ == DFlashDraftPolicy::kFixed || drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    mean_ = accepted == drafted
                ? std::min(mean_ + 1.0F, static_cast<float>(limit_))
                : 0.75F * mean_ + 0.25F * static_cast<float>(accepted);
  }

  void Reset() noexcept { mean_ = std::min(3.0F, static_cast<float>(limit_)); }
  [[nodiscard]] float State() const noexcept { return mean_; }
  void Restore(float mean) {
    if (!std::isfinite(mean) || mean < 0.0F ||
        mean > static_cast<float>(limit_))
      throw std::invalid_argument("DFlash2 controller state is invalid");
    mean_ = mean;
  }

private:
  static constexpr std::array<float, kMaxDraftTokens> kQ4RelativeCost{
      1.0F, 1.02F, 1.045F, 1.08F, 1.14F, 1.20F, 1.29F};
  // Extra cost from 2K to 32K for the target's 16 full-attention layers,
  // relative to the roughly 100 ms shallow draft/verification cycle. These
  // cold-KV measurements include the anchor plus 1–7 proposals and preserve
  // the actual row-tile boundaries. Position is deterministic on replay;
  // request timings and sampled proposals never enter the decision.
  static constexpr std::array<float, kMaxDraftTokens> kQ4ContextCost{
      0.11F, 0.14F, 0.21F, 0.23F, 0.25F, 0.31F, 0.35F};
  DFlashDraftPolicy policy_;
  std::uint32_t limit_;
  bool q8_target_;
  float mean_{0.0F};
};

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_
