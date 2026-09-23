#ifndef GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_
#define GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
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

// Select the whole block before drawing any proposals. Sampled decisions use
// private acceptance history and position, preserving RNG consumption on
// replay. Greedy cohorts use their measured wider-verification cost.
// Full acceptance is censored: probe upward instead of treating the block limit
// as the true stopping point.
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
      std::uint32_t budget, std::uint32_t position = 0,
      std::size_t greedy_batch_size = 1) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (policy_ == DFlashDraftPolicy::kFixed || cap == 0)
      return cap;
    // Preserve the full-acceptance probe at C5-C8. Extrapolating attention
    // cost beyond the measured depths must not shorten a saturated block.
    if (greedy_batch_size >= 5 && greedy_batch_size <= 8 &&
        mean_ == static_cast<float>(limit_))
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
    const auto* relative_costs = &kQ4RelativeCost;
    float context_multiplier = 1.0F;
    switch (greedy_batch_size) {
      case 2:
        relative_costs = &kQ4PairedRelativeCost;
        context_multiplier = 1.92F;
        break;
      // A finishing request does not make the remaining weight reads cheap.
      // Use the next measured capacity for odd cohorts too.
      case 3:
      case 4:
        relative_costs = &kQ4FourRequestRelativeCost;
        context_multiplier = 3.20F;
        break;
      case 5:
      case 6:
        relative_costs = &kQ4SixRequestRelativeCost;
        context_multiplier = 3.50F;
        break;
      case 7:
      case 8:
        relative_costs = &kQ4EightRequestRelativeCost;
        context_multiplier = 4.42F;
        break;
    }
    if (q8_target_ && greedy_batch_size == 4) {
      relative_costs = &kQ8FourRequestRelativeCost;
      context_multiplier = 4.0F * 100.0F / 170.66F;
    }
    const float attention_cost_scale = context_scale * context_multiplier;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= probability;
      expected_tokens += survival;
      const float cost =
          q8_target_ && greedy_batch_size != 4
              ? 1.0F + 0.02F * static_cast<float>(length)
              : (*relative_costs)[length - 1] +
                    attention_cost_scale * kQ4ContextCost[length - 1];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best_length = length;
      }
    }
    return best_length;
  }

  // All-adaptive Q8 cohorts use a shared width. Their
  // target cost depends on the total row count: mixing individually cheap
  // widths can require two expensive projection launches. Keep each history
  // private, sum its expected tokens and choose one physical batch shape.
  [[nodiscard]] static std::optional<std::uint32_t> ChooseGreedyBatch(
      std::span<const DFlashLengthController* const> controllers,
      std::span<const std::uint32_t> budgets,
      std::span<const std::uint32_t> positions) noexcept {
    const auto users = controllers.size();
    if (budgets.size() != users || positions.size() != users)
      return std::nullopt;
    const auto* relative_costs = &kQ8FourRequestRelativeCost;
    double base_ms = 170.66;
    std::uint32_t probe_width = 3;
    // Odd cohorts use the next measured capacity's costs. A finishing request
    // must not send its remaining peers back to C1's cheap-wide-block estimate.
    switch (users) {
      case 2:
        relative_costs = &kQ8PairedRelativeCost;
        base_ms = 158.26;
        break;
      case 3:
      case 4:
        break;
      case 5:
      case 6:
        relative_costs = &kQ8SixRequestRelativeCost;
        base_ms = 189.32;
        probe_width = 1;
        break;
      case 7:
      case 8:
        relative_costs = &kQ8EightRequestRelativeCost;
        base_ms = 209.20;
        probe_width = 1;
        break;
      default:
        return std::nullopt;
    }
    std::uint32_t cap = kMaxDraftTokens;
    std::array<double, 8> probabilities{};
    double attention_cost_scale = 0.0;
    bool full_tile = true;
    bool neutral_or_full = users % 2 != 0;
    std::size_t full_histories = 0;
    for (std::size_t index = 0; index < controllers.size(); ++index) {
      const auto* controller = controllers[index];
      if (controller == nullptr || !controller->q8_target_ ||
          controller->policy_ != DFlashDraftPolicy::kAdaptive)
        return std::nullopt;
      cap = std::min(cap, std::min(budgets[index], controller->limit_));
      const double mean = controller->mean_;
      probabilities[index] =
          mean == controller->limit_ ? 1.0 : mean / (mean + 1.0);
      attention_cost_scale +=
          static_cast<double>(positions[index] > 2048 ? positions[index] - 2048
                                                      : 0) /
          30720.0 * 100.0 / base_ms;
      full_tile &= controller->last_full_width_ >= probe_width;
      const bool full_history = controller->last_full_width_ >= 3;
      // A single accepted short block is too weak to carry new peers into a
      // full probe. Two such blocks raise a neutral estimate from 3 to 4.5.
      full_histories += full_history && mean >= 4.5;
      neutral_or_full &=
          full_history ||
          (controller->last_full_width_ == 0 &&
           mean == std::min(3.0F, static_cast<float>(controller->limit_)));
    }
    if (cap == 0)
      return std::nullopt;
    // All requests fully accepted a short tile. Probe wider together; one lucky
    // request must not force its peers into a costly ragged verification.
    // Two established full blocks can carry a growing odd cohort through the
    // probe when its other estimates remain neutral. Other histories use the
    // ordinary shared cost decision.
    if (full_tile || (neutral_or_full && full_histories >= 2))
      return cap;
    std::array<double, 8> survival{};
    survival.fill(1.0);
    double expected_tokens = static_cast<double>(users), best_score = 0.0;
    std::uint32_t best_length = 1;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      for (std::size_t index = 0; index < controllers.size(); ++index) {
        survival[index] *= probabilities[index];
        expected_tokens += survival[index];
      }
      const double cost = (*relative_costs)[length - 1] +
                          attention_cost_scale * kQ4ContextCost[length - 1];
      const double score = expected_tokens / cost;
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
    last_full_width_ =
        accepted == drafted ? static_cast<std::uint32_t>(drafted) : 0U;
    // For a geometric accepted run with mean m, censoring at any block width
    // preserves E[accepted - m * rejected] == 0. Weight a completed block by
    // its accepted tokens too; a fixed increment biases short blocks upward
    // and long blocks downward.
    mean_ = accepted == drafted
                ? std::min(mean_ + 0.25F * static_cast<float>(accepted),
                           static_cast<float>(limit_))
                : 0.75F * mean_ + 0.25F * static_cast<float>(accepted);
  }

  void Reset() noexcept {
    mean_ = std::min(3.0F, static_cast<float>(limit_));
    last_full_width_ = 0;
  }
  [[nodiscard]] float State() const noexcept { return mean_; }
  [[nodiscard]] std::uint32_t LastFullWidth() const noexcept {
    return last_full_width_;
  }
  void Restore(float mean, std::uint32_t last_full_width = 0) {
    if (!std::isfinite(mean) || mean < 0.0F ||
        mean > static_cast<float>(limit_) || last_full_width > limit_)
      throw std::invalid_argument("DFlash2 controller state is invalid");
    mean_ = mean;
    last_full_width_ = last_full_width;
  }

private:
  // Q8 C2 complete cycles cost 158/161/161/172/178/211/187 ms.
  // The fourteen-row verification at six drafts is slower than sixteen rows.
  static constexpr std::array<float, kMaxDraftTokens> kQ8PairedRelativeCost{
      1.0F, 1.015F, 1.020F, 1.089F, 1.127F, 1.333F, 1.182F};
  // Q8_K_XL C4 complete cycles cost 171/181/197/327/325/357/316 ms.
  // Four through six drafts split verification into separate launches;
  // seven fills two adjacent sixteen-row groups and reuses their weights.
  // The same attention geometry uses the measured absolute per-request
  // context delta below, normalized to this cohort's one-draft cycle.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ8FourRequestRelativeCost{1.0F,   1.06F,  1.153F, 1.914F,
                                 1.907F, 2.089F, 1.853F};
  // C6 complete cycles cost 189/333/334/405/445/474/455 ms.
  // A short twelve-row tile avoids the wider target projection launches.
  static constexpr std::array<float, kMaxDraftTokens> kQ8SixRequestRelativeCost{
      1.0F, 1.760F, 1.763F, 2.141F, 2.352F, 2.505F, 2.404F};
  // C8 complete cycles cost 209/337/326/467/457/596/589 ms for 1–7 drafts.
  // Ragged 24/40/56-row projections cost more than their adjacent full tiles.
  // These selector-to-selector spans include target verification, context
  // injection, the next draft and host acceptance work.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ8EightRequestRelativeCost{1.0F,   1.610F, 1.561F, 2.231F,
                                  2.183F, 2.850F, 2.813F};
  static constexpr std::array<float, kMaxDraftTokens> kQ4RelativeCost{
      1.0F, 1.02F, 1.045F, 1.08F, 1.14F, 1.20F, 1.29F};
  // Two greedy requests verify 4–16 rows together. Crossing eight rows has a
  // large projection cost that the C1 estimate misses. Complete-cycle release
  // controls normalize to 104 ms at one draft; the paired context multiplier
  // is 2 * 100 / 104, using the existing per-request attention calibration.
  static constexpr std::array<float, kMaxDraftTokens> kQ4PairedRelativeCost{
      1.0F, 1.04F, 1.10F, 1.48F, 1.45F, 1.50F, 1.50F};
  // Four-request cycles cross another projection cliff above sixteen rows.
  // One fixed-width profile per point gives 125/163/167/246/265/286/289 ms.
  // The context multiplier is 4 * 100 / 125.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ4FourRequestRelativeCost{1.0F,  1.30F, 1.33F, 1.97F,
                                 2.12F, 2.29F, 2.32F};
  // Six requests verify 12–48 rows. Measured complete cycles cost
  // 171/247/273/323/378/416/423 ms; the context multiplier is 6 * 100 / 171.
  static constexpr std::array<float, kMaxDraftTokens> kQ4SixRequestRelativeCost{
      1.0F, 1.44F, 1.59F, 1.89F, 2.21F, 2.43F, 2.47F};
  // Eight requests verify 16–64 rows. Measured complete cycles cost
  // 181/276/304/400/430/523/558 ms; the context multiplier is 8 * 100 / 181.
  static constexpr std::array<float, kMaxDraftTokens>
      kQ4EightRequestRelativeCost{1.0F,  1.52F, 1.68F, 2.21F,
                                  2.38F, 2.89F, 3.08F};
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
  std::uint32_t last_full_width_{0};
};

}  // namespace gufo::speculative

#endif  // GUFO_MODELS_QWEN_DFLASH_POLICY_HPP_
