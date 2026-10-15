// Host contract for the Gemma 4 draft-length policy: option names, the
// drafter signal, calibration learning, the cost table and the calibrated
// chain decision.
#include "src/models/gemma4/draft_policy.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

gufo::models::qwen38_flash_next::MtpCandidateLogits Candidates(float spread) {
  gufo::models::qwen38_flash_next::MtpCandidateLogits c;
  c.size = 64;
  for (std::size_t i = 0; i < c.size; ++i) {
    c.ids[i] = static_cast<gufo::sampling::TokenId>(i);
    c.logits[i] = i == 0 ? spread : 0.0F;
  }
  return c;
}

void CheckNames() {
  Require(g4::ParseDraftPolicy("") == g4::DraftPolicy::kCalibrated,
          "empty policy is not calibrated");
  for (const auto policy :
       {g4::DraftPolicy::kCalibrated, g4::DraftPolicy::kConfidence,
        g4::DraftPolicy::kFixed}) {
    Require(g4::ParseDraftPolicy(g4::DraftPolicyName(policy)) == policy,
            "policy name does not round-trip");
  }
  bool threw = false;
  try {
    (void)g4::ParseDraftPolicy("adaptive");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Require(threw, "unknown policy accepted");
  Require(
      g4::ParseDraftCalibrationScope("") == g4::DraftCalibrationScope::kShared,
      "empty scope is not shared");
  Require(g4::ParseDraftCalibrationScope("request") ==
              g4::DraftCalibrationScope::kRequest,
          "request scope not parsed");
}

void CheckSignal() {
  // A flat top-64 has entropy ln 64: 1 - sqrt(0.2 ln 64) = 0.088.
  const float flat = g4::DraftSignal(Candidates(0.0F));
  Require(std::fabs(flat - 0.0880F) < 1e-3F,
          "flat signal " + std::to_string(flat));
  const float peaked = g4::DraftSignal(Candidates(30.0F));
  Require(peaked > 0.99F && peaked <= 1.0F,
          "peaked signal " + std::to_string(peaked));
  Require(g4::DraftSignal(Candidates(4.0F)) > g4::DraftSignal(Candidates(2.0F)),
          "signal does not grow with confidence");
}

void CheckCalibration() {
  g4::DraftCalibration calibration;
  Require(std::fabs(calibration.Estimate(0.95F) - 0.9375F) < 1e-6F,
          "prior is not the bin midpoint");
  for (int i = 0; i < 1000; ++i) {
    calibration.Observe(0.95F, i % 4 != 0);  // 75% accepted
  }
  const float learned = calibration.Estimate(0.9F);
  Require(std::fabs(learned - 0.75F) < 0.01F,
          "calibration did not learn: " + std::to_string(learned));
  Require(std::fabs(calibration.Estimate(0.3F) - 0.3125F) < 1e-6F,
          "another bin moved");
  calibration.Reset();
  Require(std::fabs(calibration.Estimate(0.95F) - 0.9375F) < 1e-6F,
          "reset kept observations");
}

void CheckCosts() {
  const g4::DraftCosts shallow = g4::DraftCostsAt(0);
  Require(shallow.verify[1] == 99.5F && shallow.verify[8] == 103.9F,
          "d0 verification costs");
  Require(std::fabs(shallow.draft[7] - 7 * 1.95F) < 1e-4F, "d0 draft steps");
  const g4::DraftCosts mid = g4::DraftCostsAt(2048);
  Require(std::fabs(mid.verify[3] - 0.5F * (100.6F + 107.0F)) < 1e-3F,
          "interpolation between depths");
  const g4::DraftCosts deep = g4::DraftCostsAt(131072);
  const g4::DraftCosts measured = g4::DraftCostsAt(65536);
  for (std::size_t r = 1; r < deep.verify.size(); ++r) {
    Require(deep.verify[r] > measured.verify[r], "extrapolation past 64K");
    Require(deep.verify[r] >= deep.verify[r - 1], "rows are not monotone");
  }
  // The 26B-A4B's own table: cheaper cycles whose rows cost relatively more.
  const g4::DraftCosts moe = g4::DraftCostsAt(0, true);
  Require(moe.verify[1] == 18.3F && moe.verify[8] == 31.8F &&
              std::fabs(moe.draft[1] - 1.81F) < 1e-4F,
          "expert-model d0 costs");
  Require(moe.verify[8] / moe.verify[1] > shallow.verify[8] / shallow.verify[1],
          "expert-model rows are not relatively dearer");
}

void CheckChain() {
  const g4::DraftCosts costs = g4::DraftCostsAt(0);
  g4::DraftCalibration sure;
  g4::DraftCalibration hopeless;
  for (int i = 0; i < 10000; ++i) {
    for (float s = 0.05F; s < 1.0F; s += 0.125F) {
      sure.Observe(s, true);
      hopeless.Observe(s, false);
    }
  }
  {
    g4::CalibratedChain chain(sure, costs, 1, 7);
    for (int i = 0; i < 7; ++i) {
      Require(chain.Include(0.5F), "a sure chain stopped");
    }
    Require(!chain.Include(0.5F), "the cap was exceeded");
  }
  {
    g4::CalibratedChain chain(hopeless, costs, 1, 7);
    Require(chain.Include(0.5F), "the first draft was not verified");
    Require(!chain.Include(0.5F), "a hopeless chain continued");
  }
  {
    g4::CalibratedChain chain(hopeless, costs, 3, 7);
    for (int i = 0; i < 3; ++i) {
      Require(chain.Include(0.5F), "the minimum was not verified");
    }
    Require(!chain.Include(0.5F), "a hopeless chain passed its minimum");
  }
  {
    // Prior only: signal s is taken as the acceptance probability of its bin
    // midpoint. Draft two is worth it while survival * T >= E * dT (at 64K
    // keys, where a row costs enough to put the threshold between bins).
    const g4::DraftCosts deep = g4::DraftCostsAt(65536);
    g4::DraftCalibration prior;
    g4::CalibratedChain chain(prior, deep, 1, 7);
    Require(chain.Include(0.95F), "first draft");  // survival 0.9375
    const float current = deep.draft[2] + deep.verify[2];
    const float next = deep.draft[3] + deep.verify[3];
    const float needed = (1.0F + 0.9375F) * (next - current) / current;
    // Bin midpoints 0.0625 (0.0-0.125) and 0.1875: survival 0.059 / 0.176.
    Require(0.9375F * 0.0625F < needed && 0.9375F * 0.1875F > needed,
            "test premise: threshold between two bins");
    g4::CalibratedChain again(prior, deep, 1, 7);
    Require(again.Include(0.95F), "first draft");
    Require(!again.Include(0.05F), "draft below the cost threshold kept");
    Require(chain.Include(0.15F), "draft above the cost threshold dropped");
  }
}

/// In a batch the other sessions' rows make each extra row dearer and their
/// tokens raise the rate a draft must match: a chain that goes on alone stops
/// beside seven sessions on the expert model's costs.
void CheckBatch() {
  const g4::DraftCosts costs = g4::DraftCostsAt(0, true);
  // Every signal accepted three times in five.
  g4::DraftCalibration likely;
  for (int i = 0; i < 10000; ++i) {
    likely.Observe(0.9F, i % 5 < 3);
  }
  g4::CalibratedChain alone(likely, costs, 1, 7);
  Require(alone.Include(0.9F) && alone.Include(0.9F),
          "a likely chain stopped alone");
  Require(std::fabs(alone.Expected() - (1.0F + 0.6F + 0.36F)) < 1e-2F &&
              std::fabs(alone.DraftMs() - costs.draft[2]) < 1e-5F,
          "chain expectation and drafter time");
  // Beside three sessions (a C4 batch) the second draft no longer pays.
  g4::CalibratedChain batched(likely, costs, 1, 7,
                              g4::DraftBatch{.rows = 3, .expected = 3.0F});
  Require(batched.Include(0.9F), "a likely first draft was dropped");
  Require(!batched.Include(0.9F), "a batched chain ignored its peers");
  // Beside seven sessions a doubtful first draft is not verified at all.
  g4::DraftCalibration doubtful;
  for (int i = 0; i < 10000; ++i) {
    doubtful.Observe(0.9F, i % 10 == 0);
  }
  g4::CalibratedChain lone(doubtful, costs, 1, 7);
  Require(lone.Include(0.9F), "a lone cycle skipped its first draft");
  g4::CalibratedChain crowded(doubtful, costs, 1, 7,
                              g4::DraftBatch{.rows = 7, .expected = 7.0F});
  Require(!crowded.Include(0.9F), "a doubtful batched draft was verified");
  // Beside ten rows even a certain draft cannot pay for its row; beside one
  // it can, and a lone cycle always drafts.
  Require(!g4::CalibratedChain(doubtful, costs, 1, 1,
                               g4::DraftBatch{.rows = 10, .expected = 10.0F})
               .FirstDraftCanPay(),
          "an unpayable draft would run the drafter");
  Require(g4::CalibratedChain(doubtful, costs, 1, 7,
                              g4::DraftBatch{.rows = 1, .expected = 1.0F})
                  .FirstDraftCanPay() &&
              lone.FirstDraftCanPay(),
          "a payable draft was skipped");
}

/// A sibling adds a token where the chain reaches its draft, the draft is
/// rejected and the sibling accepted; it pays for its row like a draft.
void CheckSiblings() {
  const g4::DraftCosts costs = g4::DraftCostsAt(0);
  // Every signal accepted three times in five.
  g4::DraftCalibration likely;
  for (int i = 0; i < 10000; ++i) {
    likely.Observe(0.9F, i % 5 < 3);
  }
  g4::CalibratedChain chain(likely, costs, 1, 7);
  Require(chain.Include(0.9F) && chain.Include(0.9F), "a likely chain");
  const float expected = chain.Expected();
  // Rows: pending and two drafts. The first sibling's gain is
  // reach 1 x rejection 0.4 x hit 0.5.
  Require(chain.IncludeSibling(0, 3, 0.5F), "a paying sibling was dropped");
  Require(std::fabs(chain.Expected() - (expected + 0.2F)) < 1e-3F,
          "sibling expectation");
  // A sibling that is never accepted, or past the chain, does not pay.
  Require(!chain.IncludeSibling(1, 4, 0.0F), "a hopeless sibling was kept");
  Require(!chain.IncludeSibling(2, 4, 1.0F), "a sibling beyond the chain");
  // Beside three sessions on the expert model a doubtful sibling's row costs
  // more than it adds.
  const g4::DraftCosts moe = g4::DraftCostsAt(0, true);
  g4::CalibratedChain crowded(likely, moe, 1, 7,
                              g4::DraftBatch{.rows = 3, .expected = 3.0F});
  Require(crowded.Include(0.9F), "first draft");
  Require(!crowded.IncludeSibling(0, 2, 0.05F),
          "a doubtful batched sibling was verified");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckNames();
    CheckSignal();
    CheckCalibration();
    CheckCosts();
    CheckChain();
    CheckBatch();
    CheckSiblings();
  });
}
