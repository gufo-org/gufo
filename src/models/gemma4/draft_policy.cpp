#include "src/models/gemma4/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gufo::models::gemma4 {
namespace {

// Costs on gfx1151 for the dense 31B family: drafter steps of the
// gemma4-assistant Q8_0 drafter measured in place 2026-10-20 (drafting time
// per drafter step over fixed-length cycles, --draft-trace), verification
// forwards (target rows with logits,
// median of seven) remeasured 2026-10-05 on UD-Q4_K_XL after its
// projections moved to the WMMA kernels, whose cost barely grows with rows (a
// row never costs less than a narrower forward: the medians differ by noise
// there); two to four rows from 4K on lowered 2026-10-17 by the measured
// global split-attention saving of one wave per head (ten layers). Only cost
// ratios steer the decision.
constexpr std::array<std::uint32_t, 5> kDepths = {0, 4096, 16384, 32768, 65536};
constexpr std::array<float, 5> kDraftStepMs = {1.93F, 2.39F, 2.68F, 3.13F,
                                               4.04F};
// Verification rows 2-16 at 4K-64K lowered 2026-10-19 by the measured
// global split-attention savings (ten layers).
constexpr std::array<std::array<float, 16>, 5> kVerifyMs = {{
    {99.5F, 99.8F, 100.6F, 101.4F, 102.0F, 102.8F, 103.3F, 103.9F, 109.0F,
     109.6F, 110.1F, 111.1F, 111.7F, 112.3F, 112.7F, 113.1F},
    {104.1F, 105.1F, 106.3F, 107.9F, 108.5F, 110.3F, 111.5F, 112.8F, 120.5F,
     121.8F, 122.6F, 123.7F, 125.2F, 126.7F, 126.8F, 129.9F},
    {108.2F, 109.6F, 111.2F, 113.3F, 116.4F, 118.8F, 121.4F, 123.8F, 134.6F,
     136.8F, 139.5F, 141.7F, 144.0F, 146.1F, 146.9F, 149.4F},
    {112.5F, 114.9F, 117.2F, 119.6F, 124.6F, 128.5F, 132.7F, 136.6F, 149.7F,
     153.8F, 157.9F, 161.1F, 164.9F, 167.6F, 169.2F, 173.8F},
    {121.0F, 124.3F, 128.3F, 132.4F, 140.2F, 145.8F, 153.8F, 160.0F, 180.0F,
     187.3F, 194.8F, 202.0F, 208.2F, 212.0F, 216.0F, 222.7F},
}};

// The 26B-A4B (UD-Q4_K_XL target, Unsloth Q8_0 drafter), measured
// 2026-09-29 on varied English prose (the rows' routing, and so the experts
// they read, follows the text) up to the 16 rows a batched forward verifies:
// every verified row adds the experts it routes to, so verification grows
// several times faster per row than on the dense family.
// Two to four rows lowered 2026-10-17 by the measured global split-attention
// saving of one wave per head (five layers).
// Drafter steps measured in place 2026-10-20 like the dense family's; the
// small head over frequent tokens drafts 95% of ~31K prose cycles (1.78 ms
// per step there, the full head 2.21 ms).
constexpr std::array<float, 5> kExpertDraftStepMs = {1.25F, 1.43F, 1.63F, 1.83F,
                                                     2.17F};
// Verification rows 2-16 at 4K-64K lowered 2026-10-19 by the measured
// global split-attention savings (five layers).
constexpr std::array<std::array<float, 16>, 5> kExpertVerifyMs = {{
    {18.3F, 21.1F, 23.1F, 25.4F, 27.3F, 29.1F, 30.8F, 31.8F, 33.2F, 34.4F,
     37.1F, 40.6F, 42.3F, 44.8F, 44.6F, 46.0F},
    {19.8F, 22.8F, 25.7F, 28.0F, 30.0F, 32.7F, 34.7F, 36.2F, 37.1F, 38.9F,
     42.0F, 45.5F, 48.6F, 50.9F, 50.8F, 52.4F},
    {20.8F, 24.0F, 26.6F, 29.3F, 31.9F, 34.8F, 37.5F, 39.2F, 40.7F, 42.8F,
     46.2F, 49.9F, 51.8F, 54.5F, 54.7F, 56.8F},
    {21.7F, 25.2F, 28.0F, 31.0F, 34.3F, 36.8F, 39.4F, 41.3F, 43.6F, 46.0F,
     49.6F, 54.0F, 56.6F, 59.7F, 59.9F, 61.8F},
    {23.6F, 25.7F, 29.1F, 32.4F, 36.8F, 40.5F, 43.9F, 46.2F, 49.2F, 52.2F,
     57.0F, 61.4F, 66.3F, 69.5F, 69.9F, 72.8F},
}};

}  // namespace

DraftPolicy ParseDraftPolicy(std::string_view name) {
  if (name.empty() || name == "calibrated") {
    return DraftPolicy::kCalibrated;
  }
  if (name == "confidence") {
    return DraftPolicy::kConfidence;
  }
  if (name == "fixed") {
    return DraftPolicy::kFixed;
  }
  throw std::invalid_argument(
      "Gemma 4 draft policy must be calibrated, confidence or fixed");
}

std::string_view DraftPolicyName(DraftPolicy policy) noexcept {
  switch (policy) {
    case DraftPolicy::kConfidence:
      return "confidence";
    case DraftPolicy::kFixed:
      return "fixed";
    case DraftPolicy::kCalibrated:
      break;
  }
  return "calibrated";
}

DraftCalibrationScope ParseDraftCalibrationScope(std::string_view name) {
  if (name.empty() || name == "shared") {
    return DraftCalibrationScope::kShared;
  }
  if (name == "request") {
    return DraftCalibrationScope::kRequest;
  }
  throw std::invalid_argument("draft calibration must be shared or request");
}

std::string_view DraftCalibrationScopeName(
    DraftCalibrationScope scope) noexcept {
  return scope == DraftCalibrationScope::kRequest ? "request" : "shared";
}

float DraftSignal(
    const qwen38_flash_next::MtpCandidateLogits& candidates) noexcept {
  if (candidates.size == 0) {
    return 0.0F;
  }
  float top = candidates.logits[0];
  for (std::size_t i = 1; i < candidates.size; ++i) {
    top = std::max(top, candidates.logits[i]);
  }
  double total = 0.0;
  double weighted = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    const double shifted = static_cast<double>(candidates.logits[i] - top);
    const double e = std::exp(shifted);
    total += e;
    weighted += e * shifted;
  }
  const double entropy = std::max(0.0, std::log(total) - weighted / total);
  const double signal = 1.0 - std::sqrt(0.2 * entropy);
  return static_cast<float>(std::clamp(signal, 0.0, 1.0));
}

namespace {

/// Costs from one table: linear between measured depths, past the deepest
/// one the last interval's slope continues; rows past the table keep its
/// last row step.
template<std::size_t kRows>
DraftCosts Interpolate(
    std::uint32_t context, const std::array<float, 5>& draft_step,
    const std::array<std::array<float, kRows>, 5>& verify) noexcept {
  std::size_t hi = 1;
  while (hi + 1 < kDepths.size() && context > kDepths[hi]) {
    ++hi;
  }
  const std::size_t lo = hi - 1;
  const float t = (static_cast<float>(context) - kDepths[lo]) /
                  static_cast<float>(kDepths[hi] - kDepths[lo]);
  const auto lerp = [t](float a, float b) {
    return std::max(a, a + t * (b - a));
  };
  DraftCosts costs;
  const float step = lerp(draft_step[lo], draft_step[hi]);
  for (std::size_t n = 0; n < costs.draft.size(); ++n) {
    costs.draft[n] = step * static_cast<float>(n);
  }
  for (std::size_t r = 1; r <= kRows && r < costs.verify.size(); ++r) {
    costs.verify[r] = lerp(verify[lo][r - 1], verify[hi][r - 1]);
  }
  for (std::size_t r = kRows + 1; r < costs.verify.size(); ++r) {
    costs.verify[r] =
        costs.verify[r - 1] + (costs.verify[kRows] - costs.verify[kRows - 1]);
  }
  costs.verify[0] = costs.verify[1];
  return costs;
}

}  // namespace

DraftCosts DraftCostsAt(std::uint32_t context, bool experts) noexcept {
  return experts ? Interpolate(context, kExpertDraftStepMs, kExpertVerifyMs)
                 : Interpolate(context, kDraftStepMs, kVerifyMs);
}

std::size_t DraftCalibration::Bin(float signal) noexcept {
  const float clamped = std::clamp(signal, 0.0F, 1.0F);
  return std::min(
      kBins - 1, static_cast<std::size_t>(clamped * static_cast<float>(kBins)));
}

float DraftCalibration::Estimate(float signal) const noexcept {
  const std::size_t bin = Bin(signal);
  const float prior =
      (static_cast<float>(bin) + 0.5F) / static_cast<float>(kBins);
  return (accepted_[bin] + kPriorWeight * prior) / (seen_[bin] + kPriorWeight);
}

void DraftCalibration::Observe(float signal, bool accepted) noexcept {
  const std::size_t bin = Bin(signal);
  accepted_[bin] += accepted ? 1.0F : 0.0F;
  seen_[bin] += 1.0F;
}

void DraftCalibration::Reset() noexcept {
  accepted_.fill(0.0F);
  seen_.fill(0.0F);
}

CalibratedChain::CalibratedChain(const DraftCalibration& calibration,
                                 const DraftCosts& costs,
                                 std::uint32_t min_drafts, std::uint32_t cap,
                                 DraftBatch others) noexcept
    : calibration_(calibration),
      costs_(costs),
      // Beside other sessions even the first draft must pay for its row.
      min_drafts_(others.rows > 0 ? 0U : std::max(1U, min_drafts)),
      cap_(std::min<std::uint32_t>(cap, costs.draft.size() - 1)),
      others_(others) {}

float CalibratedChain::Verify(std::uint32_t rows) const noexcept {
  return costs_.verify[std::min<std::size_t>(others_.rows + rows,
                                             costs_.verify.size() - 1)];
}

bool CalibratedChain::FirstDraftCanPay() const noexcept {
  if (min_drafts_ > 0 || cap_ == 0) {
    return cap_ > 0;
  }
  // Include's rule for the first draft with survival one, against a cycle
  // that runs no drafter step.
  const float without = others_.draft_ms + Verify(1);
  const float with = others_.draft_ms + costs_.draft[1] + Verify(2);
  return without >= (others_.expected + expected_) * (with - without);
}

bool CalibratedChain::Include(float signal) noexcept {
  const std::uint32_t index = kept_;  // drafter steps so far: index + 1
  decision_ = {};
  if (index >= cap_) {
    return false;
  }
  steps_ = index + 1;
  const float accept = calibration_.Estimate(signal);
  const float survival = survival_ * accept;
  if (index >= min_drafts_) {
    // Stopping here verifies `index` drafts after one more drafter step than
    // needed; verifying this one adds a row, and going on adds a step.
    const float current =
        others_.draft_ms + costs_.draft[index + 1] + Verify(index + 1);
    const float next = others_.draft_ms +
                       costs_.draft[std::min(index + 2, cap_)] +
                       Verify(index + 2);
    decision_ = {.compared = true,
                 .gain = survival * current,
                 .cost = (others_.expected + expected_) * (next - current)};
    if (decision_.gain < decision_.cost) {
      return false;
    }
  }
  reach_[kept_] = survival_;
  accept_[kept_] = accept;
  survival_ = survival;
  expected_ += survival;
  ++kept_;
  return true;
}

bool CalibratedChain::IncludeSibling(std::uint32_t depth, std::uint32_t rows,
                                     float hit) noexcept {
  if (depth >= kept_) {
    return false;
  }
  const float gain = reach_[depth] * (1.0F - accept_[depth]) * hit;
  // The drafter does not run again: only the row is extra.
  const float current = others_.draft_ms + costs_.draft[steps_] + Verify(rows);
  const float next = current - Verify(rows) + Verify(rows + 1);
  if (gain * current < (others_.expected + expected_) * (next - current)) {
    return false;
  }
  expected_ += gain;
  return true;
}

}  // namespace gufo::models::gemma4
