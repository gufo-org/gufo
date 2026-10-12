#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "src/models/qwen38_flash_next/mtp_costs.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

/// Top-1 mass below which another draft slot is not spent, or 0 to leave width
/// entirely to the controller. This is a property of the loaded target, not of
/// the predictor: 0.45 measures +9.7 % on UD-Q4_K_XL and -8.3 % on UD-IQ4_XS,
/// whose drafts are better calibrated. Comparing the same signal against the
/// controller's observed acceptance instead regresses both (-5 % and -20 %),
/// because candidate-set top-1 mass and verification acceptance are not on the
/// same scale. A per-target value therefore needs the signal mapped onto the
/// acceptance scale first; that mapping is the open part, not the gating.
/// 0.60 is tuned on the daily UD-Q5K_HCP target (won 3 of 4 interleaved
/// 32k+64k rounds against 0.70/0.80, and beat 0.45 in both rounds); UD-Q4_K_XL
/// was tuned at 0.45, which GUFO_MTP_MIN_CONF restores.
inline constexpr float kMinDraftConfidence = 0.60F;

/// Runtime override of kMinDraftConfidence via GUFO_MTP_MIN_CONF. Comparing
/// targets at one gate needs no rebuild; 0 disables the gate entirely. An
/// unparseable or out-of-range value falls back to kMinDraftConfidence.
[[nodiscard]] inline float MinDraftConfidence() noexcept {
  const char* v = std::getenv("GUFO_MTP_MIN_CONF");
  if (v == nullptr || v[0] == '\0') {
    return kMinDraftConfidence;
  }
  char* end = nullptr;
  const float parsed = std::strtof(v, &end);
  if (end == v || parsed < 0.0F || parsed > 1.0F) {
    return kMinDraftConfidence;
  }
  return parsed;
}

struct MtpLengthState {
  std::array<float, kMaxMtpDraftTokens> successes;
  std::array<float, kMaxMtpDraftTokens> failures;
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

// Length depends on committed acceptance history, never request timings.
// Only the first rejected draft is an observed failure; its unverified tail
// and a fully accepted, budget-limited chain are censored observations.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit,
                               std::uint32_t concurrency = 1)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)),
        concurrency_(std::clamp(concurrency, 1U, 8U)) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t context = 0) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0) {
      return 0;
    }
    // After a bounded AR interval, try one proposal to detect recovery.
    return std::min(
        cap, std::max({1U, BestLength(cap, context), state_.probe_depth}));
  }

  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const noexcept {
    float expected = 1.0F, survival = 1.0F;
    for (unsigned depth = 0; depth < std::min(drafts, limit_); ++depth) {
      survival *= Probability(depth);
      expected += survival;
    }
    return expected;
  }

  void Observe(std::size_t accepted, std::size_t drafted,
               std::uint32_t context = 0) noexcept {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      state_.successes[depth] =
          0.75F * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          0.75F * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    // A fully accepted chain supplies no observations for its tail. Probe
    // farther rather than permanently treating those depths as the prior.
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_, context) == 0)
      state_.retry_tokens = 16;
  }

  void ObserveArToken() noexcept {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }

  void Reset() noexcept {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }

  [[nodiscard]] MtpLengthState State() const noexcept { return state_; }

  [[nodiscard]] bool Restore(const MtpLengthState& state) noexcept {
    if (state.retry_tokens > 16 || state.probe_depth > limit_ ||
        state.explored_depth > limit_ || state.probe_delay > 16 ||
        state.failed_depths >= (1U << limit_))
      return false;
    for (std::size_t depth = 0; depth < kMaxMtpDraftTokens; ++depth) {
      const float success = state.successes[depth];
      const float failure = state.failures[depth];
      if (!std::isfinite(success) || !std::isfinite(failure) ||
          success < 0.0F || failure < 0.0F ||
          !std::isfinite(success + failure) || success + failure <= 0.0F)
        return false;
    }
    state_ = state;
    return true;
  }

private:
  [[nodiscard]] float Probability(unsigned depth) const noexcept {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }
  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap,
                                         std::uint32_t context) const noexcept {
    const auto costs = MtpCycleCosts(context, concurrency_);
    float survival = 1.0F;
    float expected_tokens = 1.0F;  // The sampled target anchor.
    float best_score = 1.0F;       // Ordinary target decoding.
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      const auto depth = length - 1;
      // An explored depth with no actual rejection remains optimistic.
      // Otherwise the prior alone shortens perfect chains and prevents
      // their less frequently sampled tails from collecting evidence.
      const float probability = Probability(depth);
      survival *= probability;
      expected_tokens += survival;
      const float cost = costs[length] / costs[0];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  std::uint32_t concurrency_;
  MtpLengthState state_{};
};

// Runtime costs belong to a physical batch, not an individual sequence.
// Separate occupancy/context bins avoid applying shallow C1 timings to C8.
// Greedy output is independent of the chosen width; sampled requests keep
// the deterministic per-session policy to preserve seeded replay.
class MtpBatchController {
public:
  struct Row {
    const MtpLengthController* acceptance;
    std::uint32_t budget;
  };

  [[nodiscard]] std::uint32_t Choose(std::span<const Row> rows,
                                     std::uint32_t context) const noexcept {
    if (rows.empty() || rows.size() > 8)
      return 0;
    const auto& cell = CellFor(rows.size(), context);
    std::uint32_t cap = kMaxMtpDraftTokens;
    for (const auto& row : rows)
      cap = std::min(cap, row.budget);
    if (cap == 0)
      return 0;
    // Periodic headless cycles measure the actual AR alternative, including
    // predictor catch-up. The next cycle can then resume speculation.
    const bool stable = last_concurrency_ == rows.size() &&
                        last_context_bin_ == ContextBin(context) &&
                        stable_cycles_ >= 2;
    if (stable && (cell.samples[0] == 0 || cell.cycles % 32 < 2))
      return 0;
    auto costs = MtpCycleCosts(context, rows.size());
    const float scale = cell.samples[0] ? cell.ms[0] / costs[0] : 1.0F;
    for (unsigned width = 0; width <= cap; ++width)
      costs[width] =
          cell.samples[width] >= 2 ? cell.ms[width] : costs[width] * scale;
    float best_score = rows.size() / costs[0];
    unsigned best = 0;
    for (unsigned width = 1; width <= cap; ++width) {
      float expected = 0;
      for (const auto& row : rows)
        expected += row.acceptance->ExpectedTokens(width);
      const float score = expected / costs[width];
      if (score > best_score * 1.01F) {
        best_score = score;
        best = width;
      }
    }
    // Fully accepted cohorts can qualify deeper shapes promptly. Stop
    // forcing exploration once that physical width has warmed measurements.
    unsigned probe = cap;
    for (const auto& row : rows)
      probe = std::min(probe, row.acceptance->State().probe_depth);
    if (probe > best && cell.samples[probe] < 2)
      return probe;
    // Bounded exploration learns wider costs and conditional acceptance;
    // it also lets an AR-only batch detect a change in the workload.
    if (cell.cycles % 16 == 2)
      return std::min(cap, std::max(1U, best + 1));
    return best;
  }

  void Observe(std::size_t concurrency, std::uint32_t context,
               std::uint32_t drafts, float ms) noexcept {
    if (concurrency == 0 || concurrency > 8 || drafts > kMaxMtpDraftTokens ||
        !std::isfinite(ms) || ms <= 0)
      return;
    const auto bin = ContextBin(context);
    const bool same_cohort =
        last_concurrency_ == concurrency && last_context_bin_ == bin;
    stable_cycles_ = same_cohort ? std::min(stable_cycles_ + 1, 1000000U) : 1;
    last_concurrency_ = concurrency;
    last_context_bin_ = bin;
    auto& cell = CellFor(concurrency, context);
    // Catch-up belongs to the preceding chain. A transition from seven
    // proposals to three still replays the longer chain, so it cannot
    // measure the steady cost of three proposals (or of plain decoding).
    if (!same_cohort || drafts != cell.last_drafts) {
      cell.last_drafts = drafts;
      ++cell.cycles;
      return;
    }
    cell.last_drafts = drafts;
    auto& count = cell.samples[drafts];
    auto& estimate = cell.ms[drafts];
    // Drop first-use graph/allocation cost once a warmed sample arrives.
    if (count < 2)
      estimate = ms;
    else
      estimate += 0.25F * (ms - estimate);
    count = std::min(count + 1, 1000000U);
    ++cell.cycles;
  }

private:
  struct Cell {
    std::array<float, 8> ms{};
    std::array<unsigned, 8> samples{};
    std::uint64_t cycles{0};
    std::uint32_t last_drafts{0};
  };
  static unsigned ContextBin(std::uint32_t context) noexcept {
    unsigned bin = 0;
    for (unsigned bound = 4096; context >= bound && bin < 7; bound *= 2)
      ++bin;
    return bin;
  }
  Cell& CellFor(std::size_t concurrency, std::uint32_t context) noexcept {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  const Cell& CellFor(std::size_t concurrency,
                      std::uint32_t context) const noexcept {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  std::array<std::array<Cell, 8>, 8> cells_{};
  std::size_t last_concurrency_{0};
  unsigned last_context_bin_{0};
  unsigned stable_cycles_{0};
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_POLICY_HPP_
