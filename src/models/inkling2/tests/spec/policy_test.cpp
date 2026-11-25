// DSpark width policy (host only): a verification step that something else
// stalled must not stop drafting for the rest of the server's life.
//
// The policy drafts a token only while verifying it (a 2-row step) beats a
// plain step, and only a 2-row step re-measures Verify(2). Before the fix one
// slow 2-row sample (400 ms and more) lifted that estimate past what one draft
// can pay at any acceptance, and it never came back (PR #1340 report).
#include <cstdio>

#include "src/models/inkling2/dspark_policy.hpp"

namespace {
using namespace gufo::models::inkling2;

bool DraftsOne(const DsparkCalibration& cal, const DsparkCosts& costs,
               float p) {
  return ChooseDsparkWidth(&p, 1, cal, costs) == 1;
}

// Cycles at the top confidence bin with the true costs (2-row 26.2 ms, plain
// 20.2 ms scaled by `depth`) after one slow 2-row sample; returns how many
// drafted.
int TwoRowSteps(double slow_ms, double depth, int cycles) {
  DsparkCalibration cal;
  DsparkCosts costs;
  costs.ObserveVerify(2, slow_ms);
  int drafted = 0;
  for (int i = 0; i < cycles; ++i) {
    if (DraftsOne(cal, costs, 0.95f)) {
      costs.ObserveVerify(2, 26.2 * depth);
      ++drafted;
    } else {
      costs.ObserveVerify(1, 20.2 * depth);
    }
  }
  return drafted;
}
}  // namespace

int main() {
  int failures = 0;
  for (double slow : {300.0, 400.0, 650.0, 5000.0}) {
    // one sample moves the estimate by at most 5%: still drafting at once
    DsparkCosts costs;
    costs.ObserveVerify(2, slow);
    const bool now = DraftsOne(DsparkCalibration{}, costs, 0.95f);
    const int steps = TwoRowSteps(slow, 1.0, 10000);
    std::printf("%6.0f ms: Verify(2) %.1f, drafts next %d, 2-row steps %d\n",
                slow, costs.Verify(2), now, steps);
    if (!now || steps < 9000)
      ++failures;
  }
  // A sustained slow phase (e.g. a neighbour's prefill) raises the estimate
  // past paying; once it ends, the stale estimate relaxes and drafting resumes.
  {
    DsparkCalibration cal;
    DsparkCosts costs;
    for (int i = 0; i < 200; ++i)
      costs.ObserveVerify(2, 400);
    const bool stuck = !DraftsOne(cal, costs, 0.95f);
    int drafted = 0;
    for (int i = 0; i < 2000; ++i) {
      if (DraftsOne(cal, costs, 0.95f)) {
        costs.ObserveVerify(2, 26.2);
        ++drafted;
      } else {
        costs.ObserveVerify(1, 20.2);
      }
    }
    std::printf(
        "sustained: stopped drafting %d, 2-row steps after %d, "
        "Verify(2) %.1f\n",
        stuck, drafted, costs.Verify(2));
    if (!stuck || drafted < 1000 || costs.Verify(2) > 30)
      ++failures;
  }
  // A 2-row step that really costs too much keeps losing: relaxation retries
  // it only now and then.
  {
    DsparkCalibration cal;
    DsparkCosts costs;
    int drafted = 0;
    for (int i = 0; i < 20000; ++i) {
      if (DraftsOne(cal, costs, 0.95f)) {
        costs.ObserveVerify(2, 60);
        ++drafted;
      } else {
        costs.ObserveVerify(1, 20.2);
      }
    }
    std::printf("expensive: 2-row steps %d of 20000\n", drafted);
    if (drafted > 2000)
      ++failures;
  }
  // Batched verification up to kMaxRows = 16: an 8-session batch with high
  // draft confidence allocates drafts beyond 8 total rows.
  {
    DsparkCalibration cal;
    const DsparkCosts costs = DsparkCosts::Batched();
    constexpr int kB = 8;
    float p_hi[kB][2];
    const float* pp[kB];
    int n[kB], w[kB];
    const DsparkCalibration* cp[kB];
    for (int i = 0; i < kB; ++i) {
      p_hi[i][0] = 0.95f;
      p_hi[i][1] = 0.95f;
      pp[i] = p_hi[i];
      n[i] = 2;
      cp[i] = &cal;
    }
    ChooseDsparkBatchWidths(kB, pp, n, cp, costs, 3.0, w);
    int total_drafts = 0;
    for (int i = 0; i < kB; ++i)
      total_drafts += w[i];
    std::printf("batch B=8 high-conf: %d drafts (total rows %d)\n",
                total_drafts, kB + total_drafts);
    if (total_drafts == 0 || kB + total_drafts > DsparkCosts::kMaxRows)
      ++failures;
  }
  std::printf("%s\n", failures ? "FAIL" : "ok");
  return failures ? 1 : 0;
}
