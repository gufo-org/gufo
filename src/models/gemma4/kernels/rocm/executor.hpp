#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next {
struct MtpCandidateLogits;
}  // namespace gufo::models::qwen38_flash_next

namespace gufo::models::gemma4::rocm {

/// A step's draft token, or none to end the chain before it; `last` keeps
/// the token and ends the chain after it.
struct DraftProposal {
  std::optional<std::int32_t> token;
  bool last{false};
};

/// Picks the next draft from the drafter's top candidate logits.
using DraftProposer =
    std::function<DraftProposal(const qwen38_flash_next::MtpCandidateLogits&)>;

/// One session's attention state. Global layers keep every position in a
/// token-major binary16 cache; sliding layers keep a ring of `ring` slots,
/// at least window + the largest forward, so no forward overwrites a key
/// still inside a live window and rollback only moves the frontier.
struct KvCache {
  ~KvCache();
  void* allocation{nullptr};
  std::size_t bytes{0};
  std::uint32_t max_context{0};
  std::uint32_t ring{0};
  /// Per layer, binary16: whole keys, or their rotated dims where keys are
  /// derived from V (Executor::KeyWidths).
  std::vector<std::uint16_t*> k;
  std::vector<std::uint16_t*> v;  ///< Per layer, binary16.
  /// Target post-norm hidden state of the frontier token (MTP draft input).
  float* hidden{nullptr};
  /// The same for the last of the lookahead rows a prefill computed past the
  /// frontier (Session::Sync), adopted if the next Sync asks for them.
  float* ahead_hidden{nullptr};
  /// Positions [written_from, written_to) span every row Forward and MoveKey
  /// wrote since the owner last reset them (Session's shared snapshot
  /// blocks); a new cache counts as written everywhere.
  std::uint32_t written_from{0};
  std::uint32_t written_to{UINT32_MAX};
};

/// Image soft-token rows of one forward: rows [row, row + count) take
/// `embedding` (device FP32 [count][hidden]) unscaled in place of their token
/// embeddings and attend to each other bidirectionally in sliding layers.
struct ImageRows {
  std::uint32_t row;
  std::uint32_t count;
  const float* embedding;
};

/// Runs the Gemma 4 graph on device-resident weights. Scratch is shared by
/// every session; calls are serialized on one stream.
class Executor {
public:
  /// `max_rows` bounds one forward (prefill chunk); `max_logit_rows` bounds
  /// how many rows of one forward may request logits.
  Executor(const DeviceModel& model, std::uint32_t max_rows,
           std::uint32_t max_logit_rows, std::uint32_t max_context);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::size_t ScratchBytes(
      const Config& config, const Config* draft, std::uint32_t vocab,
      std::size_t max_cols, std::size_t max_half_cols, std::uint32_t max_rows,
      std::uint32_t max_logit_rows, std::uint32_t max_context);
  /// Per-session cache bytes for per-layer K row widths (KeyWidths).
  [[nodiscard]] static std::size_t CacheBytes(
      const Config& config, std::uint32_t max_context, std::uint32_t ring,
      const std::vector<std::uint32_t>& key_widths);
  /// Values per K cache row: the whole K, or only the rotated dims where one
  /// projection feeds K and V (K = rope(k_norm * V)).
  [[nodiscard]] static std::vector<std::uint32_t> KeyWidths(
      const DeviceModel& model);

  [[nodiscard]] std::unique_ptr<KvCache> CreateCache(
      std::uint32_t max_context, std::string* error_msg = nullptr) const;

  /// Evaluates `tokens` at positions [first_position, +tokens.size()) and
  /// writes softcapped logits of `logit_rows` (indices into `tokens`, in
  /// order) to the device logits buffer. Every earlier position must already
  /// be in `cache`.
  /// `images` (ordered, disjoint, inside `tokens`) replace those rows' inputs.
  /// `siblings`: see Segment.
  void Forward(KvCache& cache, std::span<const std::int32_t> tokens,
               std::uint32_t first_position,
               std::span<const std::uint32_t> logit_rows,
               std::span<const ImageRows> images = {},
               std::uint32_t siblings = 0);

  /// Consecutive rows of one session in a batched forward. The last
  /// `siblings` rows are alternatives to the drafts at depths 1..siblings:
  /// sibling i sits at position first_position + 1 + i, attends what the
  /// chain's draft there attends, and keeps its K/V at the spare key
  /// first_position + rows - siblings + i (MoveKey commits it).
  struct Segment {
    KvCache* cache;
    std::uint32_t first_position;
    std::uint32_t rows;
    std::uint32_t siblings{0};
  };
  /// Forward over several sessions' rows (`tokens` holds the segments' rows
  /// in order; logit and hidden rows index that concatenation). Row-wise work
  /// runs once for all rows, attention and cache writes per segment, so each
  /// row computes what its session's own forward of the same width would.
  void Forward(std::span<const Segment> segments,
               std::span<const std::int32_t> tokens,
               std::span<const std::uint32_t> logit_rows,
               std::span<const ImageRows> images = {});

  /// Keeps hidden row `row` of the last Forward as the cache's frontier
  /// state for drafting.
  void CommitHidden(KvCache& cache, std::uint32_t row);
  /// Copies row `row` of the last forward to cache.ahead_hidden, and that
  /// back to cache.hidden.
  void StashHidden(KvCache& cache, std::uint32_t row);
  void AdoptStashedHidden(KvCache& cache);
  /// Moves every layer's K/V of key `from` to key `to`: an accepted
  /// sibling's spare key into its position.
  void MoveKey(KvCache& cache, std::uint32_t from, std::uint32_t to);
  /// Enqueues device copies (CopyRun's alignment) as one launch.
  void CopyRuns(std::span<const CopyRun> runs);

  /// Drafts up to `steps` tokens with the MTP drafter after `token` at
  /// position `position` (the committed frontier), reading the target's KV
  /// strictly before it and the cache's frontier hidden state. Each step
  /// hands the drafter's top-64 logits to `propose` and embeds the token it
  /// returns; `drafts` ends where it returns nothing or a last token.
  void DraftChain(KvCache& cache, std::int32_t token, std::uint32_t position,
                  std::uint32_t steps, std::vector<std::int32_t>* drafts,
                  const DraftProposer& propose);

  /// One session's chain in DraftChains.
  struct DraftJob {
    KvCache* cache;
    std::int32_t token;
    std::uint32_t position;
    std::uint32_t steps;
    DraftProposer propose;
    std::vector<std::int32_t>* drafts;
  };
  /// DraftChain for up to kMaxDraftSessions sessions at once: every step is
  /// one drafter forward over all chains still running (each attending its
  /// own session's KV) and one host wait, then each job's proposer decides
  /// its row, in job order.
  void DraftChains(std::span<DraftJob> jobs);

  /// Receives, after each layer of every Forward, that layer's residual rows
  /// ([count][hidden] on the device, ready once `stream` reaches this point):
  /// the target features a DFlash drafter reads. Unset in serving.
  using TapSink = std::function<void(std::uint32_t layer, const float* rows,
                                     std::uint32_t count, hipStream_t stream)>;
  void SetTapSink(TapSink sink) { tap_sink_ = std::move(sink); }

  /// Copies the logits of the last Forward to the host, [rows][vocab].
  void CopyLogits(std::size_t rows, std::vector<float>* out) const;
  /// Post-norm hidden rows of the last Forward, [tokens][hidden].
  [[nodiscard]] const float* hidden() const noexcept { return h_; }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t max_rows() const noexcept { return max_rows_; }
  [[nodiscard]] std::uint32_t ring() const noexcept { return ring_; }
  [[nodiscard]] const std::vector<std::uint32_t>& key_widths() const noexcept {
    return key_widths_;
  }
  [[nodiscard]] const DeviceModel& model() const noexcept { return model_; }

private:
  /// y[rows][w.rows] = x[rows][w.cols] * W^T; `xq` is x staged for the
  /// prefill GEMMs when rows exceed the small-batch limit (Quantize).
  void Project(const DeviceTensor& w, const float* x, const void* xq,
               std::uint32_t rows, float* y);
  /// Project for binary16 weights: the batch-invariant small-batch GEMV up
  /// to kSplitRows rows, the binary16 WMMA GEMM past that.
  void ProjectHalf(const DeviceTensor& w, const float* x, const void* xq,
                   std::uint32_t rows, float* y);
  /// Stages x for the prefill GEMMs: the tiled Q8_1 encoding, or binary16
  /// rows on models whose prefill runs with binary16 activations
  /// (half_prefill_). Returns null at small-batch widths.
  const void* Quantize(const float* x, std::uint32_t rows, std::uint32_t cols);
  /// Routed experts of layer `l` for the n rows of the attention residual
  /// x_ and their expert input moe_h_: routing weights in moe_weights_,
  /// every slot's output in moe_out_ (mixed by MoeFinish).
  void Experts(const DeviceLayer& l, std::uint32_t n, hipStream_t stream);
  /// Prefill route of Experts: binary16 expert GEMMs over rows compacted
  /// by expert. Returns false when a weight format lacks that route.
  bool PrefillExperts(const DeviceLayer& l, std::uint32_t n,
                      hipStream_t stream);
  [[nodiscard]] bool DerivedKeys(std::uint32_t layer) const;
  /// Marks `att` as a derived-key layer.
  void DeriveKeys(AttentionArgs& att, std::uint32_t layer) const;

  const DeviceModel& model_;
  std::uint32_t max_rows_;
  std::uint32_t max_logit_rows_;
  TapSink tap_sink_;
  std::uint32_t max_context_;
  std::uint32_t ring_;
  std::vector<std::uint32_t> key_widths_;
  hipStream_t stream_{nullptr};
  /// Experts beside the dense MLP at grouped widths (MoE models).
  hipStream_t expert_stream_{nullptr};
  hipEvent_t expert_fork_{nullptr};
  hipEvent_t expert_join_{nullptr};
  hipblasHandle_t blas_{nullptr};  ///< Prefill router logits
  void* scratch_{nullptr};
  std::int32_t* tokens_{nullptr};
  std::uint32_t* logit_index_{nullptr};
  /// Per-row sliding-attention key ends of a forward with images.
  std::uint32_t* key_ends_{nullptr};
  std::vector<std::uint32_t> key_ends_host_;
  CopyRun* copy_runs_{nullptr};  ///< CopyRuns descriptors
  std::size_t copy_runs_capacity_{0};
  float* x_{nullptr};
  float* h_{nullptr};
  float* q_{nullptr};
  float* k_{nullptr};
  float* v_{nullptr};
  float* attn_{nullptr};
  float* o_{nullptr};
  float* gate_{nullptr};
  float* up_{nullptr};
  float* act_{nullptr};  ///< GeGLU rows of a fused gate/up projection
  float* hsel_{nullptr};
  float* logits_{nullptr};
  float* partials_{nullptr};
  void* q8_{nullptr};
  /// Activation pack and partial sums of the WMMA projections.
  void* wmma_{nullptr};
  /// Prefill activations as binary16 (Q8_1 costs this model's accuracy),
  /// written by the kernels producing them: each projection input in x_half_,
  /// the expert input, read after the dense MLP, in moe_x_half_.
  bool half_prefill_{false};
  void* x_half_{nullptr};
  void* moe_x_half_{nullptr};
  // Routed expert scratch (present on mixture-of-experts models).
  float* moe_h_{nullptr};  ///< Expert input rows
  float* moe_logits_{nullptr};
  std::int32_t* moe_ids_{nullptr};
  float* moe_weights_{nullptr};
  std::int32_t* moe_groups_{nullptr};
  void* moe_gu_{nullptr};   ///< gate/up rows: FP32, or binary16 in prefill
  void* moe_act_{nullptr};  ///< GeGLU rows: FP32, or binary16 in prefill
  float* moe_out_{nullptr};
  std::uint32_t* moe_counts_{nullptr};
  std::uint32_t* moe_sync_{nullptr};  ///< One-launch routing's block counter
  std::int32_t* moe_bounds_{nullptr};
  std::int32_t* moe_cursors_{nullptr};
  std::int32_t* moe_rows_token_{nullptr};
  std::int32_t* moe_rows_slot_{nullptr};
  std::int32_t* moe_tiles_{nullptr};
  // Drafter scratch (present with an MTP drafter).
  std::uint32_t* draft_tokens_{nullptr};
  float* draft_embed_{nullptr};
  float* draft_concat_{nullptr};
  float* draft_x_{nullptr};
  float* draft_h_{nullptr};
  float* draft_q_{nullptr};
  float* draft_attn_{nullptr};
  float* draft_o_{nullptr};
  float* draft_gate_{nullptr};
  float* draft_up_{nullptr};
  float* draft_logits_{nullptr};
  float* draft_next_{nullptr};
  std::uint32_t* draft_candidates_{nullptr};
  std::uint32_t* draft_candidate_scratch_{nullptr};
  qwen38_flash_next::MtpCandidateLogits* draft_candidates_host_{nullptr};
};

/// Longest draft chain one cycle may request; verification then carries
/// kMaxDraftTokens + 1 rows, within the batch-invariant projection width.
inline constexpr std::uint32_t kMaxDraftTokens = 15;
/// Sessions whose drafter chains share one forward per step.
inline constexpr std::uint32_t kMaxDraftSessions = 8;

/// Ring slots for sliding layers given the largest forward.
[[nodiscard]] std::uint32_t RingSlots(const Config& config,
                                      std::uint32_t max_rows) noexcept;

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_EXECUTOR_HPP_
