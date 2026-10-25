#ifndef GUFO_MODELS_QWEN_HIP_MTP_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/mtp_draft_vocab.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Row budget for one MTP prompt-prime slice. The prime scratch is sized by it,
/// so it trades a bounded scratch reservation against the number of batched
/// GEMM launches. `0` selects the legacy per-token serial prime.
inline constexpr std::uint32_t kMtpDefaultPrimeChunk = 1024;

/// Device slot budget for one batched draft chain. The chain stages every
/// proposed token on the device so that successive steps embed the previous
/// step's argmax without a host round trip; either `8` covers every supported
/// per-request cap today.
inline constexpr std::uint32_t kMtpMaxChainTokens = 8;

/// Immutable GPU-visible MTP layer weights. Current GGUF K-quant matrices are
/// expanded once to BF16; the tied embedding and LM head remain shared with
/// the target model.
class QwenMtpGpuModel final {
public:
  ~QwenMtpGpuModel();

  QwenMtpGpuModel(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel& operator=(const QwenMtpGpuModel&) = delete;
  QwenMtpGpuModel(QwenMtpGpuModel&&) = delete;
  QwenMtpGpuModel& operator=(QwenMtpGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenMtpGpuModel> Create(
      std::shared_ptr<const core::GgufReader> mtp_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenMtpWeights& GetWeights() const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }
  [[nodiscard]] double GetPackTimeSeconds() const noexcept {
    return pack_time_seconds_;
  }
  /// Whether proposals may score the small draft head (mtp_draft_vocab.hpp).
  [[nodiscard]] bool HasDraftVocab() const noexcept {
    return draft_specials_.has_value();
  }
  /// The special tokens that open every session's draft table.
  [[nodiscard]] std::span<const std::int32_t> GetDraftSpecials()
      const noexcept {
    return draft_specials_ ? std::span<const std::int32_t>(*draft_specials_)
                           : std::span<const std::int32_t>();
  }
  /// Bytes of one output-head row; nonzero when HasDraftVocab().
  [[nodiscard]] std::size_t GetOutputRowBytes() const noexcept {
    return output_row_bytes_;
  }

private:
  QwenMtpGpuModel(std::shared_ptr<const core::GgufReader> mtp_reader,
                  std::shared_ptr<const QwenGpuModel> target_model,
                  speculative::QwenMtpWeights weights,
                  std::vector<void*> allocations,
                  std::size_t packed_weight_bytes, double pack_time_seconds,
                  std::optional<std::vector<std::int32_t>> draft_specials,
                  std::size_t output_row_bytes);

  std::shared_ptr<const core::GgufReader> mtp_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenMtpWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  double pack_time_seconds_{0.0};
  std::optional<std::vector<std::int32_t>> draft_specials_;
  std::size_t output_row_bytes_{0};
};

/// Frozen MTP attention cache at a committed proposal boundary. The
/// authoritative cache is the FP32 plane, which is head-major: each key/value
/// head owns a run of `max_context` positions, so the valid region is compacted
/// per head instead of copied as a prefix of the plane.
class QwenMtpGpuSnapshot final {
public:
  ~QwenMtpGpuSnapshot();

  QwenMtpGpuSnapshot(const QwenMtpGpuSnapshot&) = delete;
  QwenMtpGpuSnapshot& operator=(const QwenMtpGpuSnapshot&) = delete;
  QwenMtpGpuSnapshot(QwenMtpGpuSnapshot&&) = delete;
  QwenMtpGpuSnapshot& operator=(QwenMtpGpuSnapshot&&) = delete;

  /// Packed key plane followed by the packed value plane.
  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  /// Draft context length this snapshot restores, one behind the target's
  /// position: the cache holds a hidden row for every token but the last.
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  QwenMtpGpuSnapshot() = default;

  void* d_kv_{nullptr};
  std::uint32_t valid_context_{0};
  std::uint32_t max_context_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t head_dim_{0};
  std::uint32_t num_kv_heads_{0};
  std::size_t payload_bytes_{0};

  friend class QwenMtpGpuExecutor;
};

/// Mutable single-session executor for the Qwen3.8 layer-64 MTP graph.
class QwenMtpGpuExecutor final {
public:
  ~QwenMtpGpuExecutor();

  QwenMtpGpuExecutor(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor& operator=(const QwenMtpGpuExecutor&) = delete;
  QwenMtpGpuExecutor(QwenMtpGpuExecutor&&) = delete;
  QwenMtpGpuExecutor& operator=(QwenMtpGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuExecutor> Create(
      std::shared_ptr<const QwenMtpGpuModel> model, std::uint32_t max_context,
      std::uint32_t prime_chunk = kMtpDefaultPrimeChunk,
      std::string* error_msg = nullptr);

  void SetVisionInput(models::qwen::vision::DeviceInput* input) noexcept {
    vision_input_ = input;
  }
  void Reset() noexcept;
  void Rewind(std::uint32_t position);
  /// Waits for queued work, including host-to-device copies whose source
  /// the caller may reuse once this returns.
  void Synchronize();

  /// `small_head` scores the prefix rows and this executor's draft table
  /// instead of the full output head; it needs HasDraftVocab().
  [[nodiscard]] tokenization::TokenId ForwardTargetHidden(
      tokenization::TokenId input_token, std::span<const float> target_hidden,
      std::uint32_t position, bool compute_logits = true,
      bool small_head = false);

  [[nodiscard]] tokenization::TokenId ForwardFeedback(
      tokenization::TokenId input_token, std::uint32_t position,
      bool compute_logits = true, bool small_head = false);

  /// Enqueues `count` draft steps that mirror the serial Propose loop without
  /// the per-token host round trip. Step 0 embeds `input_token` against the
  /// host `target_hidden` (staged once, as ForwardTargetHidden does); every
  /// later step embeds the device-resident argmax token of its predecessor.
  /// The per-step arithmetic, launch order and cursor semantics equal the
  /// serial path bit for bit, so batched and serial proposals agree exactly.
  /// `CollectChainTokens()` performs the single synchronization that lands the
  /// proposal tokens on the host, and nothing but `GetNextPosition()` may be
  /// called on this executor between the two calls.
  void StartDraftChain(tokenization::TokenId input_token,
                       std::span<const float> target_hidden,
                       std::uint32_t count, bool small_head = false);
  [[nodiscard]] std::vector<tokenization::TokenId> CollectChainTokens();

  /// Draft-table rows in use: the model's special tokens, then the rows
  /// appended since the last ResetDraftRows(). Zero without a draft table.
  [[nodiscard]] std::uint32_t GetDraftRows() const noexcept {
    return draft_rows_;
  }
  /// Keeps only the special-token rows. Later rows stay on the device
  /// unread until they are overwritten.
  void ResetDraftRows();
  /// Copies the output-head rows of `ids` into the table after the rows in
  /// use.
  void AppendDraftRows(std::span<const std::int32_t> ids);

  /// Drops a chain that is still between `StartDraftChain` and
  /// `CollectChainTokens`, so a failed batched proposal can release every
  /// session's executor before its rollback rewinds the cursor. Abandoning
  /// leaves no proposal behind: the slots already written become scratch for
  /// the next chain, and the stream is drained once so no kernel can land
  /// behind the caller's rewind. A chain that is not active is a no-op.
  void AbandonChain() noexcept;

  /// Batched, projection-only prompt prime (MTP priming plan, stage A). Writes
  /// KV rows for `count` consecutive positions beginning at `start_position`,
  /// from the trunk hidden rows that own those positions; `successor_tokens[i]`
  /// is the token that follows the position owning `hidden_rows[i]`. Attention,
  /// MoE, the output norm and the logits are not computed: nothing downstream
  /// of the KV write is consumed by the draft, so the reduction is exact rather
  /// than an approximation. On return the decode cursor is
  /// `start_position + count`; the caller owns sequencing and the cache base.
  void PrimeTargetHiddenBatch(
      std::uint32_t start_position,
      std::span<const tokenization::TokenId> successor_tokens,
      std::span<const float> hidden_rows);

  /// Device-resident form: the rows already live on the device, so only the KV
  /// write is enqueued. Unused until stage D (the device-resident hidden ring);
  /// it exists now so that D becomes a caller change rather than a rewrite.
  void PrimeTargetHiddenBatchDevice(
      std::uint32_t start_position,
      std::span<const tokenization::TokenId> successor_tokens,
      const float* hidden_rows_device, std::size_t row_stride);

  [[nodiscard]] std::span<const float> CopyLastHidden();
  /// The full head's logits; a small-head forward leaves them partial.
  [[nodiscard]] std::span<const float> CopyLastLogits();
  /// The exact top candidates of the full head's last logits, in descending
  /// order with lowest-ID ties. Requires a full-head forward.
  [[nodiscard]] models::qwen38_flash_next::MtpCandidateLogits TopCandidates();

  /// Captures the committed cache prefix. Requires a committed proposal
  /// boundary; the cursor is the draft's own next position.
  [[nodiscard]] std::unique_ptr<QwenMtpGpuSnapshot> SaveSnapshot() const;
  void RestoreSnapshot(const QwenMtpGpuSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);

  /// Resident request state: both attention cache planes.
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept;
  /// Per-session estimate for admission control, before a session exists.
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const QwenMtpGpuModel& model, std::uint32_t max_context,
      std::uint32_t prime_chunk = kMtpDefaultPrimeChunk);

  [[nodiscard]] std::uint32_t GetNextPosition() const noexcept {
    return next_position_;
  }
  [[nodiscard]] std::uint32_t GetMaxContext() const noexcept {
    return max_context_;
  }
  [[nodiscard]] const QwenMtpGpuModel& GetModel() const noexcept {
    return *model_;
  }
  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }

private:
  QwenMtpGpuExecutor(std::shared_ptr<const QwenMtpGpuModel> model,
                     std::uint32_t max_context, std::uint32_t prime_chunk);

  [[nodiscard]] tokenization::TokenId Run(tokenization::TokenId input_token,
                                          const float* hidden_input,
                                          std::uint32_t position,
                                          bool compute_logits, bool small_head);
  void Allocate();
  void Free() noexcept;
  friend class models::qwen::ContinuationAdapter;
  models::qwen::ContinuationHooks* continuation_hooks_{};
  void AllocatePrimeScratch();
  /// Shared body of both prime entry points. The two row sources are mutually
  /// exclusive: `device_rows` is read directly, `host_rows` is staged through
  /// `d_prime_hidden_` a slice at a time.
  void PrimeRowsBatched(std::uint32_t start_position,
                        std::span<const tokenization::TokenId> successor_tokens,
                        const float* device_rows, std::size_t device_row_stride,
                        const float* host_rows, std::size_t host_row_stride);
  /// Scatters a packed key plane/value plane pair into the head-major FP32
  /// cache; the caller owns the decode cursor.
  void RestoreCachePlanes(const std::uint8_t* packed,
                          std::uint32_t valid_context, hipMemcpyKind kind);

  /// Throws while a batched draft chain is between StartDraftChain and
  /// CollectChainTokens: nothing may touch the chain's device state or cursor.
  void RequireChainIdle() const;

  /// One full draft forward. `source.d_token` selects the device-token
  /// embedding path used by chain steps 1..n-1; otherwise `source.h_token` is
  /// embedded by value as today. `argmax_slot` below kMtpMaxChainTokens
  /// redirects the argmax into that proposal slot and skips the legacy
  /// synchronizing readback; kChainNoSlot keeps the serial behavior exactly.
  static constexpr std::uint32_t kChainNoSlot =
      std::numeric_limits<std::uint32_t>::max();
  void RunDraftForward(const std::uint32_t* source_d_token,
                       tokenization::TokenId source_h_token,
                       const float* hidden_input, std::uint32_t position,
                       std::uint32_t argmax_slot, bool compute_logits,
                       bool small_head);

  std::shared_ptr<const QwenMtpGpuModel> model_;
  std::uint32_t max_context_;
  std::uint32_t prime_chunk_;
  std::uint32_t next_position_{0};
  hipStream_t stream_{nullptr};

  float* d_target_hidden_{nullptr};
  float* d_embedding_{nullptr};
  float* d_fusion_{nullptr};
  float* d_hidden_{nullptr};
  float* d_normed_{nullptr};
  float* d_qg_{nullptr};
  float* d_q_{nullptr};
  float* d_k_{nullptr};
  float* d_v_{nullptr};
  float* d_gate_{nullptr};
  float* d_context_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_act_{nullptr};
  float* d_ffn_out_{nullptr};
  float* d_feedback_hidden_{nullptr};
  float* d_logits_{nullptr};
  // Candidate IDs, then their scores from offset kMtpCandidates on.
  std::uint32_t* d_candidates_{nullptr};
  std::uint32_t* d_candidate_scratch_{nullptr};
  float* d_kv_cache_{nullptr};
  void* d_kv_cache_f16_{nullptr};
  float* d_split_k_scratch_{nullptr};
  std::uint32_t* d_out_token_{nullptr};
  // Device proposal-token slots for batched draft chains; sized
  // kMtpMaxChainTokens so estimate and allocation stay exact.
  std::uint32_t* d_chain_tokens_{nullptr};
  std::uint32_t chain_slots_{0};
  bool chain_active_{false};
  // Draft table (mtp_draft_vocab.hpp): kMtpDraftExtraRows output-head rows
  // and their token IDs; allocated only when the model HasDraftVocab().
  std::uint8_t* d_draft_rows_{nullptr};
  std::int32_t* d_draft_ids_{nullptr};
  std::uint32_t draft_rows_{0};
  // One-token MoE FFN scratch; allocated only for MoE targets.
  std::uint8_t* d_moe_{nullptr};
  QwenMoeScratch moe_{};

  models::qwen::vision::DeviceInput* vision_input_{
      nullptr};  // owned by the target session
  // Batched-prime scratch, sized by prime_chunk_ and empty when it is zero.
  std::uint32_t* d_prime_tokens_{nullptr};
  float* d_prime_hidden_{nullptr};
  float* d_prime_embed_{nullptr};
  float* d_prime_fusion_{nullptr};
  float* d_prime_fusion_out_{nullptr};
  float* d_prime_normed_{nullptr};
  float* d_prime_k_{nullptr};
  float* d_prime_v_{nullptr};
  std::vector<std::uint32_t> h_prime_tokens_;
  std::vector<float> h_last_hidden_;
  std::vector<float> h_logits_;
  std::size_t scratch_bytes_{0};
};

struct QwenMtpGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{4};
  /// Rows per batched prime slice; `0` restores the legacy serial prime.
  std::uint32_t prime_chunk{kMtpDefaultPrimeChunk};
  models::qwen::vision::DeviceInput* vision_input{nullptr};
};

/// Stateful MTP draft provider backed by the production HIP executor.
class QwenMtpGpuDraftBackend final : public speculative::IDraftBackend {
public:
  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> Create(
      std::shared_ptr<const QwenMtpGpuModel> model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenMtpGpuDraftBackend> CreateFromGguf(
      std::string_view model_path,
      std::shared_ptr<const QwenGpuModel> target_model,
      QwenMtpGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenMtpGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;

  /// Draws each draft token from the full head's top candidates at
  /// `temperature` and returns that exact distribution with the proposal,
  /// so the verifier's rejection sampling stays lossless.
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }

  /// Batched greedy proposal across sessions, mirroring the DFlash2 backend
  /// contract: two to eight primed backends on one shared model, validated
  /// before any state changes, with per-session launch-free device chaining
  /// between steps. Greedy cohorts only in this first cut; mixed or sampled
  /// cohorts take the serial default so their semantics stay untouched.
  [[nodiscard]] std::vector<speculative::DraftProposal> ProposeBatch(
      std::span<const speculative::DraftProposalRequest> requests) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;
  void Reset() noexcept override;
  void BeginRequest() noexcept override;

  /// Drops an uncommitted proposal and rewinds the draft to the committed
  /// frontier, so a cancelled verification resumes from `position`. MTP's cache
  /// trails the target by one row, so `position` must equal
  /// `GetNextPosition() + 1` once the proposal is discarded.
  void DiscardPendingTargetContext(std::uint32_t position);

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept {
    return executor_->GetMemoryUsage();
  }

  [[nodiscard]] const std::string& GetLastError() const noexcept {
    return last_error_;
  }

private:
  QwenMtpGpuDraftBackend(std::unique_ptr<QwenMtpGpuExecutor> executor,
                         QwenMtpGpuDraftConfig config);
  friend class models::qwen::ContinuationAdapter;

  /// Brings the draft table up to `tokens` (the proposal's sequence, ending
  /// in its anchor) and returns whether the proposal scores the small head.
  [[nodiscard]] bool SyncDraftVocab(
      std::span<const tokenization::TokenId> tokens);

  std::unique_ptr<QwenMtpGpuExecutor> executor_;
  QwenMtpGpuDraftConfig config_;
  std::vector<float> target_hidden_;
  std::vector<float> proposal_target_hidden_;
  std::vector<float> committed_target_hidden_;
  std::vector<float> append_hidden_rows_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  tokenization::TokenId proposal_input_{0};
  std::uint32_t proposal_checkpoint_{0};
  std::optional<speculative::QwenMtpDraftTable> draft_table_;
  bool primed_{false};
  bool proposal_active_{false};
  std::string last_error_;
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_HPP_
