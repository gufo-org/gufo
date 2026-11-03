#ifndef GUFO_MODELS_QWEN_HIP_DFLASH_HPP_
#define GUFO_MODELS_QWEN_HIP_DFLASH_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gufo/ext/draft_backend.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/prompt_lookup.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/families/qwen/draft_vocab/draft_vocab.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/qwen/dflash_policy.hpp"
#include "src/models/qwen/dflash_weights.hpp"
#include "src/models/qwen/hip/executor.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

namespace kernels {
struct DFlashSampling;
}

/// Optional synchronous host observation for the independent reference test.
/// The span is valid only during the callback; production leaves it empty.
using DFlashTrace =
    std::function<void(std::string_view, std::span<const float>)>;

class QwenDFlashGpuExecutor;

struct QwenDFlashContextRequest {
  QwenDFlashGpuExecutor* executor{nullptr};
  std::span<const float> features;
  std::uint32_t position{0};
  DFlashTrace trace;
};

struct QwenDFlashBlockRequest {
  QwenDFlashGpuExecutor* executor{nullptr};
  tokenization::TokenId anchor{0};
  std::uint32_t position{0};
  std::uint32_t draft_count{0};
  float temperature{0.0F};
  std::span<const float> uniforms;
  DFlashTrace trace;
  const sampling::SamplerState* sampler{nullptr};
  /// Score the executor's draft vocabulary instead of the full head.
  bool draft_subset{false};
};

class QwenDFlashGpuSnapshot final {
public:
  ~QwenDFlashGpuSnapshot();

  QwenDFlashGpuSnapshot(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot& operator=(const QwenDFlashGpuSnapshot&) = delete;
  QwenDFlashGpuSnapshot(QwenDFlashGpuSnapshot&&) = delete;
  QwenDFlashGpuSnapshot& operator=(QwenDFlashGpuSnapshot&&) = delete;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint32_t ValidContext() const noexcept {
    return valid_context_;
  }
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  QwenDFlashGpuSnapshot() = default;

  void* d_k_{nullptr};
  void* d_v_{nullptr};
  std::size_t elements_per_layer_{0};
  std::uint32_t num_layers_{0};
  std::uint32_t kv_width_{0};
  std::uint32_t max_context_{0};
  std::uint32_t valid_context_{0};
  std::uint32_t history_capacity_{0};
  std::size_t payload_bytes_{0};

  friend class QwenDFlashGpuExecutor;
};

/// Immutable GPU-visible DFlash / DFlash-2 model weights and topology.
class QwenDFlashGpuModel final {
public:
  ~QwenDFlashGpuModel();

  QwenDFlashGpuModel(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel& operator=(const QwenDFlashGpuModel&) = delete;
  QwenDFlashGpuModel(QwenDFlashGpuModel&&) = delete;
  QwenDFlashGpuModel& operator=(QwenDFlashGpuModel&&) = delete;

  [[nodiscard]] static std::shared_ptr<const QwenDFlashGpuModel> Create(
      std::shared_ptr<const core::GgufReader> dflash_reader,
      std::shared_ptr<const QwenGpuModel> target_model,
      std::string* error_msg = nullptr);

  [[nodiscard]] const speculative::QwenDFlashWeights& GetWeights()
      const noexcept {
    return weights_;
  }
  [[nodiscard]] const core::ModelConfig& GetConfig() const noexcept {
    return weights_.config;
  }
  [[nodiscard]] const speculative::QwenDFlashConfig& GetDFlashConfig()
      const noexcept {
    return weights_.dflash_config;
  }
  [[nodiscard]] const tokenization::QwenTokenizer& GetTokenizer()
      const noexcept {
    return target_model_->GetTokenizer();
  }
  [[nodiscard]] std::size_t GetPackedWeightBytes() const noexcept {
    return packed_weight_bytes_;
  }

  /// Draft vocabulary (draft_vocab.hpp): the head's leading rows, scored in
  /// place. Zero when every draft scores the full head.
  [[nodiscard]] std::uint32_t DraftPrefixRows() const noexcept {
    return draft_prefix_rows_;
  }
  /// Special tokens above the prefix; they open every request's table.
  [[nodiscard]] std::span<const std::int32_t> DraftSpecials() const noexcept {
    return draft_specials_;
  }
  [[nodiscard]] std::size_t DraftRowBytes() const noexcept {
    return draft_row_bytes_;
  }
  /// In the prefix or special: never needs a table row.
  [[nodiscard]] bool InDraftSubset(tokenization::TokenId token) const;

private:
  QwenDFlashGpuModel(std::shared_ptr<const core::GgufReader> dflash_reader,
                     std::shared_ptr<const QwenGpuModel> target_model,
                     speculative::QwenDFlashWeights weights,
                     std::vector<void*> allocations,
                     std::size_t packed_weight_bytes);

  std::shared_ptr<const core::GgufReader> dflash_reader_;
  std::shared_ptr<const QwenGpuModel> target_model_;
  speculative::QwenDFlashWeights weights_;
  std::vector<void*> allocations_;
  std::size_t packed_weight_bytes_{0};
  std::uint32_t draft_prefix_rows_{0};
  std::vector<std::int32_t> draft_specials_;
  std::size_t draft_row_bytes_{0};
};

/// Session executor for GPU DFlash / DFlash-2 parallel block drafting.
class QwenDFlashGpuExecutor final {
public:
  ~QwenDFlashGpuExecutor();

  QwenDFlashGpuExecutor(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor& operator=(const QwenDFlashGpuExecutor&) = delete;
  QwenDFlashGpuExecutor(QwenDFlashGpuExecutor&&) = delete;
  QwenDFlashGpuExecutor& operator=(QwenDFlashGpuExecutor&&) = delete;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuExecutor> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      std::uint32_t max_context = 4096, std::string* error_msg = nullptr);

  void Reset() noexcept;

  /// Ingests target multi-layer hidden states and injects K/V into the draft
  /// cache.
  bool InjectTargetContext(std::span<const float> target_features,
                           std::uint32_t position, std::uint32_t num_tokens,
                           const DFlashTrace& trace = {});

  /// Projects committed features together and writes each private cache at
  /// its original absolute positions. Reuses the coordinator's scalar scratch.
  static void InjectTargetContextBatch(
      std::span<const QwenDFlashContextRequest> requests);

  /// Executes non-causal parallel block diffusion drafting.
  [[nodiscard]] std::vector<tokenization::TokenId> ForwardBlock(
      tokenization::TokenId anchor_token, std::uint32_t current_pos,
      std::uint32_t draft_count, float temperature = 0.0F,
      std::span<const float> sample_uniforms = {},
      std::vector<float>* out_confidences = nullptr,
      std::vector<tokenization::TokenId>* out_candidate_ids = nullptr,
      std::vector<float>* out_candidate_probabilities = nullptr,
      const DFlashTrace& trace = {},
      const sampling::SamplerState* sampler = nullptr,
      bool draft_subset = false);

  /// Clears the draft table from slot `first`: the slots score -inf.
  void ClearDraftRows(std::uint32_t first);
  /// Copies the head rows of `ids` into the draft table from slot `first`.
  void AppendDraftRows(std::uint32_t first, std::span<const std::int32_t> ids);

  /// Shared projections with independent attention, convolution and selector
  /// boundaries. A single request retains the ordinary block execution.
  [[nodiscard]] static std::vector<speculative::DraftProposal>
  ForwardBlockBatch(std::span<const QwenDFlashBlockRequest> requests);

  [[nodiscard]] std::unique_ptr<QwenDFlashGpuSnapshot> SaveSnapshot() const;
  void RestoreSnapshot(const QwenDFlashGpuSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);
  [[nodiscard]] std::size_t StateBytes() const noexcept;
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const noexcept;
  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept;
  [[nodiscard]] static QwenGpuMemoryUsage EstimateMemoryUsage(
      const QwenDFlashGpuModel& model, std::uint32_t max_context,
      std::size_t max_batch_width = 1);

  [[nodiscard]] std::size_t GetHiddenSize() const noexcept {
    return model_->GetConfig().hidden_size;
  }
  [[nodiscard]] std::size_t GetTargetFeaturesSize() const noexcept {
    return model_->GetDFlashConfig().target_layer_ids.size() * GetHiddenSize();
  }
  [[nodiscard]] std::uint32_t GetInjectedContextLength() const noexcept {
    return injected_context_len_;
  }
  [[nodiscard]] const QwenDFlashGpuModel& GetModel() const noexcept {
    return *model_;
  }

private:
  [[nodiscard]] kernels::DFlashSampling PrepareDraftSampling(
      const sampling::SamplerState* sampler, hipStream_t stream);
  QwenDFlashGpuExecutor(std::shared_ptr<const QwenDFlashGpuModel> model,
                        std::uint32_t max_context);
  friend class models::qwen::ContinuationAdapter;
  void CheckContinuation() const {
    if (continuation_hooks_)
      continuation_hooks_->BeforeExecution();
  }
  models::qwen::ContinuationHooks* continuation_hooks_{};

  bool InjectTargetContextChunk(std::span<const float> target_features,
                                std::uint32_t position,
                                std::uint32_t num_tokens,
                                const DFlashTrace& trace);
  void Allocate();
  void Free() noexcept;
  [[nodiscard]] static std::size_t BatchScratchBytes(
      const QwenDFlashGpuModel& model, std::size_t rows);
  void PrewarmBlockGemms();
  void RunBlockGemm(const models::QwenTensorRef& weight, const float* input,
                    float* output, std::size_t batch_size,
                    std::size_t output_size, std::size_t input_size);
  void RunInjectGemm(const models::QwenTensorRef& weight, const float* input,
                     float* output, std::size_t num_tokens,
                     std::size_t output_size, std::size_t input_size);
  /// The draft table as a head of speculative::kDraftVocabExtraRows rows.
  [[nodiscard]] models::QwenTensorRef DraftTable() const noexcept;

  std::shared_ptr<const QwenDFlashGpuModel> model_;
  std::uint32_t max_context_{4096};
  std::uint32_t injected_context_len_{0};
  std::uint32_t history_capacity_{0};
  std::uint32_t injection_capacity_{0};
  hipStream_t stream_{nullptr};
  hipblasHandle_t hipblas_handle_{nullptr};
  std::unique_ptr<HipblasLtGemm> hipblaslt_gemm_;

  // GPU Allocations for draft KV cache
  std::vector<float*> d_injected_k_;
  std::vector<float*> d_injected_v_;

  // GPU Scratch buffers
  float* d_target_features_{nullptr};
  float* d_fused_features_{nullptr};
  float* d_block_hidden_{nullptr};
  float* d_block_normed_{nullptr};
  float* d_conv_hidden_{nullptr};
  float* d_dynamic_coefficients_{nullptr};
  float* d_q_{nullptr};
  float* d_k_block_{nullptr};
  float* d_v_block_{nullptr};
  float* d_attn_out_{nullptr};
  float* d_ffn_gate_{nullptr};
  float* d_ffn_up_{nullptr};
  float* d_ffn_down_{nullptr};
  float* d_logits_{nullptr};
  float* d_selector_hidden_{nullptr};
  std::uint32_t* d_selector_candidate_ids_{nullptr};
  float* d_selector_candidate_probabilities_{nullptr};
  float* d_selector_uniforms_{nullptr};
  float* d_confidences_{nullptr};
  std::uint32_t* d_out_token_{nullptr};
  hip_bfloat16* d_bf16_input_{nullptr};
  // The request's draft table (draft_vocab.hpp): head rows, their IDs (-1:
  // empty) and one block of their scores. Rebuilt from the token history.
  std::uint8_t* d_draft_rows_{nullptr};
  std::int32_t* d_draft_ids_{nullptr};
  float* d_draft_logits_{nullptr};
  std::vector<std::int32_t> draft_upload_;
  std::size_t scratch_bytes_{0};
  void* d_batch_scratch_{nullptr};
  std::size_t batch_scratch_bytes_{0};
};

struct QwenDFlashGpuDraftConfig {
  std::uint32_t max_context{4096};
  std::uint32_t max_draft_tokens{7};
  speculative::DFlashDraftPolicy policy{
      speculative::DFlashDraftPolicy::kAdaptive};
  /// Verification rows (anchor included) a prompt-lookup copy of a request
  /// decoding alone may grow to; at most max_draft_tokens + 1 disables it.
  std::uint32_t max_copy_rows{0};
};

/// Adapts the GPU DFlash executor to the repository's IDraftBackend speculative
/// interface.
class QwenDFlashGpuDraftBackend final : public speculative::IDraftBackend {
public:
  /// Committed target rows the draft leaves pending for its next proposal,
  /// as continuation snapshots expect (an ordinary verification's rows).
  static constexpr std::size_t kMaxPendingRows = 8;

  ~QwenDFlashGpuDraftBackend() override = default;

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend> Create(
      std::shared_ptr<const QwenDFlashGpuModel> model,
      QwenDFlashGpuDraftConfig config = {}, std::string* error_msg = nullptr);

  [[nodiscard]] static std::unique_ptr<QwenDFlashGpuDraftBackend>
  CreateFromGguf(std::string_view dflash_model_path,
                 std::shared_ptr<const QwenGpuModel> target_model,
                 QwenDFlashGpuDraftConfig config = {},
                 std::string* error_msg = nullptr);

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "QwenDFlashGpuDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return executor_->GetModel().GetDFlashConfig().target_layer_ids;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;

  void UpdateTargetHidden(std::span<const float> hidden) override;

  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  [[nodiscard]] speculative::DraftProposal ProposeSampled(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state) override;
  [[nodiscard]] bool SupportsSampledProposals() const noexcept override {
    return true;
  }
  [[nodiscard]] speculative::DraftProposal ProposeForRequest(
      const speculative::DraftProposalRequest& request) override;
  [[nodiscard]] std::vector<speculative::DraftProposal> ProposeBatch(
      std::span<const speculative::DraftProposalRequest> requests) override;

  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override;
  [[nodiscard]] std::unique_ptr<speculative::IDraftBackendSnapshot> Snapshot()
      const override;
  void RestoreSnapshot(
      const speculative::IDraftBackendSnapshot& snapshot) override;
  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override;

  void Reset() noexcept override;
  void BeginRequest() noexcept override {
    try {
      CheckContinuation();
      controller_.Reset();
      lookup_.BeginRequest();
      ResetCopyWidth();
    } catch (...) {
      if (continuation_hooks_)
        continuation_hooks_->WriteFailed();
    }
  }
  /// The most recent verification has not yet injected these target rows.
  /// Discard them when the target returns to its saved verification frontier.
  void DiscardPendingTargetContext(std::uint32_t position);

  [[nodiscard]] QwenGpuMemoryUsage GetMemoryUsage() const noexcept {
    return executor_->GetMemoryUsage();
  }
  [[nodiscard]] std::uint32_t GetInjectedContextLength() const noexcept {
    return executor_->GetInjectedContextLength();
  }
  /// Neural drafts that scored the draft vocabulary (draft_vocab.hpp).
  [[nodiscard]] std::uint64_t DraftSubsetBlocks() const noexcept {
    return draft_subset_blocks_;
  }

private:
  void InjectPendingFeatures(std::uint32_t position);
  /// Brings the request's draft table up to its committed history and
  /// returns whether this block, anchored on `tokens.back()`, drafts through
  /// it. `allowed` is speculative::DraftSubsetAllowed for this step.
  [[nodiscard]] bool UseDraftSubset(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool allowed);
  friend class models::qwen::ContinuationAdapter;
  void CheckContinuation() const {
    if (continuation_hooks_)
      continuation_hooks_->BeforeExecution();
  }
  models::qwen::ContinuationHooks* continuation_hooks_{};
  [[nodiscard]] std::uint32_t PendingFeatureCount(std::uint32_t position) const;
  QwenDFlashGpuDraftBackend(std::unique_ptr<QwenDFlashGpuExecutor> executor,
                            QwenDFlashGpuDraftConfig config);
  /// `wide`: the proposal is verified alone (ProposeForRequest), so a copy
  /// may use the current copy width instead of max_draft_tokens.
  [[nodiscard]] speculative::DraftProposal ProposeImpl(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
      std::uint64_t* rng_state, const sampling::SamplerState* sampler = nullptr,
      bool wide = false);
  [[nodiscard]] static std::vector<speculative::DraftProposal> ProposeEach(
      std::span<const speculative::DraftProposalRequest> requests);
  [[nodiscard]] std::optional<speculative::DraftProposal> TryLookup(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      std::uint32_t limit, bool sampled, std::uint32_t cap);
  [[nodiscard]] std::uint32_t OrdinaryRows() const noexcept {
    return config_.max_draft_tokens + 1;
  }
  void ResetCopyWidth() noexcept {
    copy_rows_ = 0;
    copy_streak_ = 0;
    lookup_wide_ = false;
  }
  /// Applies the copy width rule after feedback on a wide-eligible copy.
  void ObserveCopyWidth(std::size_t rows, std::size_t kept) noexcept;
  /// Injects all but the last pending row once a wide keep leaves more than
  /// an ordinary verification's rows pending, as snapshots and batched
  /// cycles expect.
  void CatchUpPendingFeatures();

  std::unique_ptr<QwenDFlashGpuExecutor> executor_;
  QwenDFlashGpuDraftConfig config_;
  speculative::DFlashLengthController controller_;
  std::array<tokenization::TokenId, 3> lookup_boundaries_{};
  speculative::PromptLookup<tokenization::TokenId> lookup_;
  std::size_t lookup_start_{0};
  bool lookup_proposal_{false};
  /// The pending copy may verify wider than an ordinary block.
  bool lookup_wide_{false};
  /// Rows (anchor included) the next wide-eligible copy may verify; 0 means
  /// OrdinaryRows(). Two whole copies in a row at this width double it, up to
  /// max_copy_rows; a rejected copy returns it to OrdinaryRows().
  std::uint32_t copy_rows_{0};
  std::uint32_t copy_streak_{0};
  std::vector<float> pending_target_features_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  bool primed_{false};
  bool proposal_active_{false};
  /// This request's draft vocabulary; the executor's table follows it.
  speculative::DraftVocab draft_vocab_;
  /// After the token history is replaced: the table restarts from it.
  bool draft_vocab_stale_{true};
  std::vector<std::int32_t> draft_added_;
  std::uint64_t draft_subset_blocks_{0};
};

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_DFLASH_HPP_
