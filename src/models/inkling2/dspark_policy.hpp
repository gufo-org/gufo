// Adaptive DSpark verification width for Inkling 2 Mini (host only).
//
// A cycle drafts up to seven tokens (the drafter's block; cheap: about a tenth
// of a decode step for the block and a twentieth per chained token, stopping
// once a longer width cannot pay), then verifies the anchor plus the first W
// of them in one Engine::Verify step, whose cost grows with W.
// The width maximises the expected emitted tokens per cost: the drafter's own
// probability for each proposal, mapped through an acceptance calibration
// that keeps learning from the outcomes, gives the chance that a prefix
// survives. Requests whose drafts stop paying back fall back to plain decoding
// for a while and probe again. None of this changes any token: Verify is
// exact.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace gufo::models::inkling2 {

/// P(accepted | the drafter's probability), in ten bins, as running counts
/// that start from a measured prior (wikitext, contexts 64 to 12K).
class DsparkCalibration {
public:
  DsparkCalibration() {
    constexpr std::array<double, 10> prior{0.03, 0.08, 0.1,  0.15, 0.25,
                                           0.3,  0.32, 0.38, 0.5,  0.88};
    for (std::size_t b = 0; b < 10; ++b) {
      seen_[b] = kPriorWeight;
      hit_[b] = kPriorWeight * prior[b];
    }
  }
  [[nodiscard]] double Estimate(float p) const {
    const std::size_t b = Bin(p);
    return hit_[b] / seen_[b];
  }
  void Observe(float p, bool accepted) {
    const std::size_t b = Bin(p);
    seen_[b] += 1;
    hit_[b] += accepted ? 1 : 0;
    if (seen_[b] > kWindow) {  // forget slowly
      seen_[b] *= 0.5;
      hit_[b] *= 0.5;
    }
  }

private:
  static constexpr double kPriorWeight = 16, kWindow = 1024;
  static std::size_t Bin(float p) {
    return static_cast<std::size_t>(std::clamp(p * 10.f, 0.f, 9.f));
  }
  std::array<double, 10> seen_{}, hit_{};
};

/// Cycle costs in milliseconds, as running averages of the measured times.
class DsparkCosts {
public:
  static constexpr int kMaxRows = 8;
  [[nodiscard]] double Verify(int rows) const { return verify_[rows]; }
  [[nodiscard]] double Draft(int tokens) const {
    return draft_base_ + draft_step_ * tokens;
  }
  void ObserveVerify(int rows, double ms) { Ema(verify_[rows], ms); }
  void ObserveDraft(int tokens, double ms) {
    // the per-token part from the known base, the base from the rest
    if (tokens > 0)
      Ema(draft_step_, std::max(0.0, (ms - draft_base_) / tokens));
    Ema(draft_base_, std::max(0.0, ms - draft_step_ * tokens));
  }

private:
  static void Ema(double& v, double x) { v += 0.05 * (x - v); }
  // Strix Halo, short context (inkling2_bench --spec / --dspark)
  std::array<double, kMaxRows + 1> verify_{0,    22.0, 27.8, 32.9, 37.8,
                                           43.1, 47.8, 53.6, 58.9};
  double draft_base_ = 3.0, draft_step_ = 1.5;
};

/// The number of drafts (0 .. n) to verify after drafting n, given each
/// draft's probability: argmax over W of E[emitted] / cost, with E = 1 + the
/// survival of each prefix and cost = the drafting already spent + Verify(W+1).
[[nodiscard]] inline int ChooseDsparkWidth(const float* p, int n,
                                           const DsparkCalibration& cal,
                                           const DsparkCosts& cost) {
  double survive = 1, expect = 1;
  int best = 0;
  double best_rate = 1 / (cost.Draft(n) + cost.Verify(1));
  for (int w = 1; w <= n && w + 1 <= DsparkCosts::kMaxRows; ++w) {
    survive *= cal.Estimate(p[w - 1]);
    expect += survive;
    const double rate = expect / (cost.Draft(n) + cost.Verify(w + 1));
    if (rate > best_rate) {
      best_rate = rate;
      best = w;
    }
  }
  return best;
}

/// Per-request back-off: every eight drafted cycles, when they emitted fewer
/// tokens per millisecond than plain decoding steps of the request's kind
/// would, skip drafting for a while (8 steps, doubling to 64), then probe
/// again.
struct DsparkBackoff {
  int skip = 0;     // plain steps left before the next drafted cycle
  int penalty = 0;  // the last back-off length
  double tokens = 0, ms = 0;
  int cycles = 0;

  [[nodiscard]] bool Draft() {
    if (skip > 0) {
      --skip;
      return false;
    }
    return true;
  }
  void Observe(int emitted, double cycle_ms, double plain_ms) {
    tokens += emitted;
    ms += cycle_ms;
    if (++cycles < 8)
      return;
    const double gain = tokens * plain_ms / std::max(ms, 1e-3);
    tokens = ms = 0;
    cycles = 0;
    if (gain < 0.97) {
      penalty = std::clamp(penalty ? 2 * penalty : 8, 8, 64);
      skip = penalty;
    } else {
      penalty = 0;
    }
  }
};

}  // namespace gufo::models::inkling2
