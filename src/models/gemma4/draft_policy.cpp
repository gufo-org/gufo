#include "src/models/gemma4/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gufo::models::gemma4 {
namespace {

// Costs on gfx1151 for the dense 31B family: drafter steps measured
// 2026-09-28 (QAT drafter), verification forwards (target rows with logits,
// median of seven) remeasured 2026-10-05 on UD-Q4_K_XL after its
// projections moved to the WMMA kernels, whose cost barely grows with rows (a
// row never costs less than a narrower forward: the medians differ by noise
// there). Only cost ratios steer the decision.
constexpr std::array<std::uint32_t, 5> kDepths = {0, 4096, 16384, 32768, 65536};
constexpr std::array<float, 5> kDraftStepMs = {1.95F, 2.05F, 2.40F, 2.82F,
                                               3.66F};
constexpr std::array<std::array<float, 16>, 5> kVerifyMs = {{
    {99.5F, 99.8F, 100.6F, 101.4F, 102.0F, 102.8F, 103.3F, 103.9F, 109.0F,
     109.6F, 110.1F, 111.1F, 111.7F, 112.3F, 112.7F, 113.1F},
    {104.1F, 105.8F, 107.0F, 108.0F, 108.7F, 110.6F, 111.9F, 113.3F, 122.7F,
     123.5F, 124.4F, 125.0F, 126.1F, 127.6F, 127.6F, 130.7F},
    {108.2F, 112.2F, 114.0F, 115.3F, 117.3F, 120.3F, 123.2F, 125.7F, 138.9F,
     140.0F, 142.2F, 144.4F, 146.8F, 149.1F, 150.0F, 153.2F},
    {112.5F, 119.1F, 121.7F, 123.1F, 126.3F, 130.6F, 136.0F, 140.0F, 158.3F,
     160.4F, 163.8F, 167.0F, 170.3F, 173.5F, 175.2F, 180.1F},
    {121.0F, 132.0F, 136.1F, 137.9F, 143.6F, 149.7F, 159.8F, 166.7F, 195.9F,
     199.8F, 206.1F, 213.5F, 218.0F, 223.6F, 228.0F, 235.5F},
}};

// The 26B-A4B (UD-Q4_K_XL target, Unsloth Q8_0 drafter), measured
// 2026-09-29 on varied English prose (the rows' routing, and so the experts
// they read, follows the text) up to the 16 rows a batched forward verifies:
// every verified row adds the experts it routes to, so verification grows
// several times faster per row than on the dense family.
constexpr std::array<float, 5> kExpertDraftStepMs = {1.81F, 2.03F, 2.25F, 2.51F,
                                                     3.04F};
constexpr std::array<std::array<float, 16>, 5> kExpertVerifyMs = {{
    {18.3F, 21.1F, 23.1F, 25.4F, 27.3F, 29.1F, 30.8F, 31.8F, 33.2F, 34.4F,
     37.1F, 40.6F, 42.3F, 44.8F, 44.6F, 46.0F},
    {19.8F, 22.9F, 25.8F, 28.1F, 30.0F, 32.8F, 34.9F, 36.4F, 37.8F, 39.5F,
     42.5F, 46.0F, 48.9F, 51.1F, 51.0F, 52.6F},
    {20.8F, 24.6F, 27.3F, 29.5F, 32.1F, 35.1F, 37.9F, 39.6F, 41.9F, 43.5F,
     46.8F, 50.5F, 52.6F, 55.3F, 55.5F, 57.6F},
    {21.7F, 26.7F, 29.7F, 32.0F, 34.7F, 37.5F, 40.2F, 42.3F, 45.8F, 47.6F,
     50.9F, 55.4F, 58.0F, 61.2F, 61.5F, 63.6F},
    {23.6F, 28.2F, 31.9F, 34.2F, 37.6F, 41.6F, 45.6F, 48.0F, 53.5F, 55.6F,
     60.0F, 64.7F, 69.0F, 72.4F, 73.0F, 76.0F},
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
    if (survival * current <
        (others_.expected + expected_) * (next - current)) {
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
