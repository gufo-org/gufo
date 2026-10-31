#ifndef GUFO_MODELS_GEMMA4_CYCLE_TRACE_HPP_
#define GUFO_MODELS_GEMMA4_CYCLE_TRACE_HPP_

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::gemma4 {

/// One MTP verification cycle of one session: what the draft policy priced
/// and expected, what verification accepted and how long each stage took.
/// Times are steady-clock nanoseconds; a batch's sessions share the
/// drafting and verification stages.
struct CycleRecord {
  /// The forward this cycle verified in, counted per model, and the
  /// sessions it verified.
  std::uint64_t forward{0};
  std::uint32_t sessions{1};
  std::uint32_t position{0};
  bool sampled{false};
  /// Verified rows: the pending token, MTP drafts, copies, then siblings.
  std::uint32_t rows{0};
  std::uint32_t copied{0};
  std::uint32_t siblings{0};
  /// Drafter steps run, including a last proposal the policy declined, and
  /// the most the chain could run.
  std::uint32_t drafter_steps{0};
  std::uint32_t step_cap{0};
  /// The chain drafted on the small head over frequent tokens.
  bool small_head{false};
  /// Accepted drafts and copies, whether a sibling was accepted, and the
  /// tokens the cycle emitted.
  std::uint32_t accepted{0};
  bool sibling_accepted{false};
  std::uint32_t emitted{0};
  /// Calibrated policy: drafter and verification prices of the cycle and
  /// the tokens it expected (zero under other policies).
  float priced_draft_ms{0.0F};
  float priced_verify_ms{0.0F};
  float expected{0.0F};
  /// Calibrated policy: each verified MTP draft's signal, its acceptance
  /// estimate and its decision's gain minus cost (zero for the first
  /// `min_drafts`), and each sibling's hit estimate.
  std::vector<float> signals;
  std::vector<float> estimates;
  std::vector<float> margins;
  std::vector<float> sibling_estimates;
  /// Every drafter proposal's signal under any policy, a declined last one
  /// included.
  std::vector<float> proposal_signals;
  /// Per proposal, the drafter's top-1 share and top-1 minus top-2 share
  /// among its top-64 logits, and the proposal's own probability (sampled:
  /// q of the drawn token; greedy: the top-1 share).
  std::vector<float> proposal_top;
  std::vector<float> proposal_margin;
  std::vector<float> proposal_q;
  /// Sampled: the top probability and entropy (nats) of the request
  /// sampler's distribution the next pending token was drawn from (-1 when
  /// not recorded).
  float pending_top{-1.0F};
  float pending_entropy{-1.0F};
  /// Per verified chain row past the pending one (MTP drafts, then
  /// copies): the probability that verification accepts it given the rows
  /// before it accepted, whatever happened to them. Greedy: 1 when the
  /// target's argmax is the row's token, else 0; sampled: min(1, p/q) with
  /// p the request sampler's distribution at the cycle's start (exact
  /// without repetition penalties) and q the proposal's probability (1 for
  /// a copy).
  std::vector<float> row_acceptance;
  /// Per MTP draft, the drafter's alternative beside it (-1: none), its
  /// signal, and the probability that verification accepts it as the
  /// sibling once the draft is rejected (a hit adds one token). Greedy: 1
  /// when the target's argmax is the alternative; sampled: min(1, r/q')
  /// with r = norm(max(p - q, 0)) and q' = q without the draft.
  std::vector<std::int32_t> runner_ups;
  std::vector<float> runner_up_signals;
  std::vector<float> runner_up_acceptance;
  /// Sampled cycles, for offline proposal studies: the request sampler's
  /// temperature, top-k and top-p; per drafter proposal its top-64
  /// candidates (ids and raw logits); per verified MTP draft the proposal q
  /// it was drawn from and the target distribution p of its row (as for
  /// row_acceptance; ids with their probabilities).
  struct Sparse {
    std::vector<std::int32_t> ids;
    std::vector<float> values;
  };
  float temperature{0.0F};
  std::int32_t top_k{0};
  float top_p{1.0F};
  std::vector<Sparse> draft_candidates;
  std::vector<Sparse> proposal_dist;
  std::vector<Sparse> target_dist;
  /// What the calibrated policy learned: each judged MTP draft's outcome
  /// (signals[i] with observed[i]) and the sibling's, when one was judged.
  std::vector<bool> observed;
  std::optional<bool> sibling_observed;
  /// The proposal the calibrated policy declined: its signal, estimate,
  /// expected gain and cost (the chain stops when gain < cost).
  struct Stop {
    float signal{0.0F};
    float estimate{0.0F};
    float gain{0.0F};
    float cost{0.0F};
  };
  std::optional<Stop> stop;
  std::int64_t begin_ns{0};     ///< the cycle starts drafting
  std::int64_t drafted_ns{0};   ///< its rows are complete
  std::int64_t verified_ns{0};  ///< the forward's results are on the host
  std::int64_t end_ns{0};       ///< its tokens are emitted
};

/// The record as one line of JSON (no newline).
[[nodiscard]] std::string FormatCycleRecord(const CycleRecord& record);

/// Appends one JSON line per cycle to a file (`--draft-trace`).
class CycleTrace {
public:
  /// Opens `path` for appending; throws std::runtime_error if it cannot.
  explicit CycleTrace(const std::string& path);
  /// Writes and flushes one record; safe from concurrent sessions.
  void Write(const CycleRecord& record);
  /// Steady-clock nanoseconds.
  [[nodiscard]] static std::int64_t Now() noexcept;

private:
  std::mutex mutex_;
  std::ofstream out_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CYCLE_TRACE_HPP_
