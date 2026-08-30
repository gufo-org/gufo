#ifndef GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_
#define GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_

#include <string_view>

namespace gufo::speculative {

/// Resolves the `auto` draft-policy default for a backend.
///
/// The adaptive controllers exist because a verification batch used to get
/// steeply more expensive as it got wider, so trimming the width bought back
/// time. For DFlash-2 that premise no longer holds: a batch now costs about
/// what a single decode step costs at every width in 2..8, so trimming only
/// emits fewer tokens for the same cost. Measured on both pinned suites, fixed
/// width 7 beats rolling and accepted-EMA while staying greedy-exact, and
/// acceptance *rate* is inversely correlated with throughput there. Backends
/// whose verifier cost still grows with width keep the rolling controller.
///
/// Re-measured on the 10-prompt corpus after porting the upstream adaptive
/// controller (see `benchmarks/qwen3.8-27b/README.md`, "Draft width and policy
/// on DFlash-2"). Two facts keep `fixed` here, and both are properties of this
/// drafter rather than of the controller:
///
///  - Accepted tokens per step rise monotonically with width on every prompt
///    class (1.40 at width 2 to 2.54 at width 7 on structured output). There is
///    no width at which the block drafter degrades enough to make a narrower
///    block accept *more*, so there is nothing for a controller to find.
///  - Throughput is a broad plateau over widths 4..7 (within 2%), so the best
///    available reward is inside the run-to-run band while the cost of guessing
///    low is not: prompts that accept deeply lose ~10% per trimmed token.
///
/// Any controller that targets the accepted count therefore trades throughput
/// for acceptance rate. `accepted-ema` reaches 50.7% acceptance against fixed
/// width 7's 37.6% and is 4.4% slower. Reports of large adaptive wins on this
/// hardware come from stacks whose drafter *does* collapse at full block width;
/// this one does not.
///
/// Only the synthetic repetitive prompt is clipped by the drafter's
/// `block_size - 1` ceiling, accepting 96.6% of seven drafted tokens and 100%
/// once a controller narrows it. No representative prompt class comes close --
/// reasoning, the next deepest, accepts 4.2 of 7 -- so drafting past one block
/// would buy almost nothing outside that one case and is not worth the draft
/// path rework it needs. Neither direction, trimming nor extending, is where
/// DFlash-2 throughput is left.
[[nodiscard]] inline std::string_view ResolveDraftPolicy(
    std::string_view requested, bool block_diffusion_draft) noexcept {
  if (requested != "auto") {
    return requested;
  }
  return block_diffusion_draft ? "fixed" : "rolling";
}

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_DRAFT_POLICY_HPP_
