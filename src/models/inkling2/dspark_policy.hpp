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
  static constexpr int kMaxRows = 16;
  [[nodiscard]] double Verify(int rows) const { return verify_[rows]; }
  /// Starting costs of steps whose rows belong to several sessions (their
  /// rows route to more distinct experts than one session's).
  static DsparkCosts Batched() {
    DsparkCosts c;
    c.start_ = {0,    21.9, 29.3, 35.5, 42.8, 47.8, 53.0, 58.0, 63.0,
                67.5, 72.0, 76.5, 81.0, 85.5, 90.0, 94.5, 99.0};
    c.verify_ = c.start_;
    return c;
  }
  [[nodiscard]] double Draft(int tokens) const {
    return draft_base_ + draft_step_ * tokens;
  }
  /// One sample moves an estimate by at most 5%: a step that something else
  /// stalled (a cache spill, a long prefill beside it) can't by itself make
  /// a width look unaffordable. A width the policy doesn't choose is never
  /// timed again, so a pessimistic estimate would stay for the server's life:
  /// estimates not refreshed for kStale steps relax towards the starting
  /// table scaled by the one-row cost (which follows the context depth), until
  /// the width is chosen and measured again.
  void ObserveVerify(int rows, double ms) {
    Ema(verify_[rows], std::min(ms, 2 * verify_[rows]));
    age_[rows] = 0;
    const double scale = verify_[1] / start_[1];
    for (int r = 2; r <= kMaxRows; ++r) {
      if (r == rows || ++age_[r] < kStale)
        continue;
      if (verify_[r] > start_[r] * scale)
        Ema(verify_[r], start_[r] * scale);
    }
  }
  void ObserveDraft(int tokens, double ms) {
    // the per-token part from the known base, the base from the rest
    if (tokens > 0)
      Ema(draft_step_, std::max(0.0, (ms - draft_base_) / tokens));
    Ema(draft_base_, std::max(0.0, ms - draft_step_ * tokens));
  }

private:
  static constexpr int kStale = 256;
  static void Ema(double& v, double x) { v += 0.05 * (x - v); }
  // Strix Halo, short context (inkling2_bench --spec / --dspark)
  std::array<double, kMaxRows + 1> start_{0,    20.2, 26.2, 31.0, 35.5, 40.5,
                                          45.3, 50.0, 54.9, 58.0, 60.5, 63.0,
                                          65.5, 67.5, 69.5, 71.5, 73.5};
  std::array<double, kMaxRows + 1> verify_ = start_;
  std::array<int, kMaxRows + 1> age_{};
  double draft_base_ = 2.8, draft_step_ = 0.25;
};

/// The number of drafts (0 .. n) to verify after drafting n, given each
/// draft's probability: argmax over W of E[emitted] / cost, with E = 1 + the
/// survival of each prefix and cost = the drafting already spent + Verify(W+1).
/// `draft_ms` is what the proposals cost (negative: the drafter's measured
/// cost of n tokens; copied proposals cost nothing).
[[nodiscard]] inline int ChooseDsparkWidth(const float* p, int n,
                                           const DsparkCalibration& cal,
                                           const DsparkCosts& cost,
                                           double draft_ms = -1) {
  if (draft_ms < 0)
    draft_ms = cost.Draft(n);
  double survive = 1, expect = 1;
  int best = 0;
  double best_rate = 1 / (draft_ms + cost.Verify(1));
  for (int w = 1; w <= n && w + 1 <= DsparkCosts::kMaxRows; ++w) {
    survive *= cal.Estimate(p[w - 1]);
    expect += survive;
    const double rate = expect / (draft_ms + cost.Verify(w + 1));
    if (rate > best_rate) {
      best_rate = rate;
      best = w;
    }
  }
  return best;
}

/// Widths of a batched cycle (several sessions in one verification step):
/// session i drafted n[i] tokens whose calibrated survivals are given by
/// `cal[i]` from the drafter's probabilities p[i][..]. Rows are added one at a
/// time to the session whose next draft is the most likely to survive (each
/// session's survival only falls along its drafts, so for every row count
/// that is the best allocation), and the count that maximises
/// E[emitted] / (drafting already spent + Verify(rows)) is kept. w[i]
/// receives each session's width (its rows minus the anchor).
inline void ChooseDsparkBatchWidths(int sessions, const float* const* p,
                                    const int* n,
                                    const DsparkCalibration* const* cal,
                                    const DsparkCosts& cost, double draft_ms,
                                    int* w) {
  double survive[DsparkCosts::kMaxRows] = {};
  int at[DsparkCosts::kMaxRows] = {};
  for (int i = 0; i < sessions; ++i) {
    survive[i] = 1;
    w[i] = 0;
  }
  int rows = sessions;
  double expect = sessions;
  double best_rate = expect / (draft_ms + cost.Verify(rows));
  int best_rows = rows;
  int order[DsparkCosts::kMaxRows] = {};  // the session each added row took
  while (rows < DsparkCosts::kMaxRows) {
    int pick = -1;
    double gain = 0;
    for (int i = 0; i < sessions; ++i) {
      if (at[i] >= n[i])
        continue;
      const double g = survive[i] * cal[i]->Estimate(p[i][at[i]]);
      if (pick < 0 || g > gain) {
        pick = i;
        gain = g;
      }
    }
    if (pick < 0)
      break;
    survive[pick] *= cal[pick]->Estimate(p[pick][at[pick]]);
    ++at[pick];
    order[rows] = pick;
    ++rows;
    expect += gain;
    const double rate = expect / (draft_ms + cost.Verify(rows));
    if (rate > best_rate) {
      best_rate = rate;
      best_rows = rows;
    }
  }
  for (int r = sessions; r < best_rows; ++r)
    ++w[order[r]];
}

/// Copy drafts (prompt lookup): per-request back-off. A copy that kept fewer
/// than half of its tokens skips the next copy opportunities (4, doubling to
/// 64; the drafter proposes meanwhile); a copy that kept at least half
/// clears it.
struct CopyBackoff {
  int skip = 0, penalty = 0;
  [[nodiscard]] bool Copy() {
    if (skip > 0) {
      --skip;
      return false;
    }
    return true;
  }
  void Observe(int proposed, int accepted) {
    if (2 * accepted < proposed) {
      penalty = std::clamp(penalty ? 2 * penalty : 4, 4, 64);
      skip = penalty;
    } else {
      penalty = 0;
    }
  }
};

/// Per-request back-off: every eight drafted cycles, when they emitted fewer
/// tokens per millisecond than plain decoding steps of the request's kind
/// would (times `min_gain`), skip drafting for a while (8 steps, doubling to
/// 64), then probe again. A lone request tolerates 3%: its eight-cycle rate is
/// noisy and drafting gains much more on average. Batched steps tolerate
/// nothing: with few rows left to draft in, the gain is small either way.
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
  void Observe(int emitted, double cycle_ms, double plain_ms,
               double min_gain = 0.97) {
    tokens += emitted;
    ms += cycle_ms;
    if (++cycles < 8)
      return;
    const double gain = tokens * plain_ms / std::max(ms, 1e-3);
    tokens = ms = 0;
    cycles = 0;
    if (gain < min_gain) {
      penalty = std::clamp(penalty ? 2 * penalty : 8, 8, 64);
      skip = penalty;
    } else {
      penalty = 0;
    }
  }
};

}  // namespace gufo::models::inkling2
