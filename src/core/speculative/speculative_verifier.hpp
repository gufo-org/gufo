#ifndef GUFO_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_
#define GUFO_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/models/qwen/dflash_policy.hpp"
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"
#endif

namespace gufo::speculative {

struct SpeculativeOptions {
  std::uint32_t max_draft_tokens{4};
  std::uint32_t min_draft_tokens{1};
  std::uint32_t initial_draft_tokens{3};
  std::size_t rolling_window{16};
  float target_acceptance_rate{0.70F};
  bool enable_adaptive_draft_length{true};
  bool use_batched_verification{false};
  bool retain_frontier_logits{false};
  DFlashDraftPolicy dflash_policy{DFlashDraftPolicy::kAdaptive};
};

struct SpeculativeStats {
  std::size_t total_draft_tokens{0};
  std::size_t total_accepted_tokens{0};
  std::size_t total_verification_steps{0};
  std::size_t total_emitted_tokens{0};
  [[nodiscard]] float AcceptanceRate() const noexcept {
    if (total_draft_tokens == 0)
      return 0.0F;
    return static_cast<float>(total_accepted_tokens) /
           static_cast<float>(total_draft_tokens);
  }
};

#if defined(ENGINE_ENABLE_HIP)

struct VerificationChunkResult {
  std::vector<tokenization::TokenId> predictions;
  std::vector<float> hidden_states;
  std::vector<float> logits;
  std::size_t hidden_width{0};
  std::size_t vocab_size{0};
};

struct SampledVerificationResult {
  tokenization::TokenId token{0};
  bool accepted{false};
};

class ISpeculativeTargetExecutor;

struct TargetVerificationItem {
  ISpeculativeTargetExecutor* executor;
  std::span<const tokenization::TokenId> tokens;
  std::uint32_t position;
  bool capture_hidden;
  bool capture_logits;
};

struct TargetVerificationBatch {
  std::vector<VerificationChunkResult> chunks;
  std::size_t physical_width{1};
};

class ISpeculativeTargetExecutor {
public:
  virtual ~ISpeculativeTargetExecutor() = default;

  virtual void Reset() noexcept = 0;
  [[nodiscard]] virtual tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) = 0;
  [[nodiscard]] virtual tokenization::TokenId ForwardPromptSuffix(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits = true) {
    (void)compute_logits;
    if (position == 0)
      return ForwardPromptBatch(tokens);
    throw std::logic_error("target does not support batched prompt extension");
  }
  [[nodiscard]] virtual tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits = true) = 0;
  virtual void SaveState(std::uint32_t valid_context) = 0;
  virtual void RestoreState() = 0;
  virtual void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids = {}) {
    (void)enabled;
    (void)target_layer_ids;
  }
  [[nodiscard]] virtual std::span<const float> GetPromptHiddenStates()
      const noexcept {
    return {};
  }
  [[nodiscard]] virtual std::span<const float> CopyLastHidden() { return {}; }
  [[nodiscard]] virtual std::span<const float> CopyLastLogits() { return {}; }
  [[nodiscard]] virtual std::span<const float> CopyVerificationLogits(
      std::size_t row) {
    (void)row;
    return {};
  }
  [[nodiscard]] virtual bool SupportsDeviceResidentSampling() const noexcept {
    return false;
  }
  /// Zero is reserved for synthetic executors without a fixed vocabulary.
  [[nodiscard]] virtual std::size_t VocabularySize() const noexcept {
    return 0;
  }
  [[nodiscard]] virtual tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler) {
    const auto logits = CopyLastLogits();
    if (logits.empty()) {
      throw std::runtime_error(
          "target executor did not retain sampled decode logits");
    }
    return sampler.Sample(logits);
  }
  [[nodiscard]] virtual tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler) {
    const auto logits = CopyVerificationLogits(row);
    if (logits.empty()) {
      throw std::runtime_error(
          "target executor did not retain verification logits");
    }
    return sampler.Sample(logits);
  }
  [[nodiscard]] virtual SampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability, sampling::SamplerState& sampler) {
    const auto logits = CopyVerificationLogits(row);
    if (logits.empty()) {
      throw std::runtime_error(
          "target executor did not retain verification logits");
    }
    const auto target_distribution = sampler.Distribution(logits);
    const double target_probability =
        target_distribution.probability(draft_token);
    if (sampler.Uniform() * draft_token_probability < target_probability) {
      sampler.Accept(draft_token);
      return {.token = draft_token, .accepted = true};
    }
    return {
        .token = target_distribution.SampleResidual(
            draft_candidate_ids, draft_candidate_probabilities,
            sampler.mutable_rng_state()),
        .accepted = false,
    };
  }
  [[nodiscard]] virtual VerificationChunkResult ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden, bool capture_logits) {
    VerificationChunkResult result;
    result.predictions.reserve(candidate_tokens.size());
    std::uint32_t pos = start_pos;
    for (auto tok : candidate_tokens) {
      result.predictions.push_back(ForwardToken(tok, pos++));
      if (!capture_hidden) {
      } else {
        const auto hidden = CopyLastHidden();
        if (hidden.empty()) {
          throw std::runtime_error(
              "target executor did not capture a verification hidden state");
        }
        if (result.hidden_width == 0) {
          result.hidden_width = hidden.size();
        } else if (result.hidden_width != hidden.size()) {
          throw std::runtime_error(
              "verification hidden-state width changed within a chunk");
        }
        result.hidden_states.insert(result.hidden_states.end(), hidden.begin(),
                                    hidden.end());
      }
      if (capture_logits) {
        const auto logits = CopyLastLogits();
        if (logits.empty()) {
          throw std::runtime_error(
              "target executor did not capture verification logits");
        }
        if (result.vocab_size == 0) {
          result.vocab_size = logits.size();
        } else if (result.vocab_size != logits.size()) {
          throw std::runtime_error(
              "verification vocabulary changed within a chunk");
        }
        result.logits.insert(result.logits.end(), logits.begin(), logits.end());
      }
    }
    return result;
  }
  /// Groups independent sequences while preserving each executor's state.
  [[nodiscard]] virtual TargetVerificationBatch ForwardVerificationBatch(
      std::span<const TargetVerificationItem> items) {
    TargetVerificationBatch result;
    result.chunks.reserve(items.size());
    for (const auto& item : items) {
      result.chunks.push_back(item.executor->ForwardVerificationChunk(
          item.tokens, item.position, item.capture_hidden,
          item.capture_logits));
    }
    return result;
  }
  virtual void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos) {
    std::uint32_t pos = start_pos;
    for (const auto token : committed_tokens) {
      (void)ForwardToken(token, pos++, false);
    }
  }
  [[nodiscard]] virtual tokenization::TokenId GetEosTokenId()
      const noexcept = 0;
  [[nodiscard]] virtual std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept = 0;
};

class SpeculativeVerifierSnapshot final {
public:
  SpeculativeVerifierSnapshot(const SpeculativeVerifierSnapshot&) = delete;
  SpeculativeVerifierSnapshot& operator=(const SpeculativeVerifierSnapshot&) =
      delete;
  SpeculativeVerifierSnapshot(SpeculativeVerifierSnapshot&&) = delete;
  SpeculativeVerifierSnapshot& operator=(SpeculativeVerifierSnapshot&&) =
      delete;
  ~SpeculativeVerifierSnapshot() = default;

  [[nodiscard]] std::size_t PayloadBytes() const noexcept;
  [[nodiscard]] std::size_t PersistentPayloadBytes() const;
  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const;

private:
  SpeculativeVerifierSnapshot() = default;

  std::unique_ptr<IDraftBackendSnapshot> draft_snapshot_;
  SpeculativeStats stats_;
  std::uint32_t current_draft_length_{0};
  std::deque<float> rolling_acceptance_;

  friend class SpeculativeVerifier;
};

/// High-throughput speculative decoding verifier with transactional state
/// management
class SpeculativeVerifier {
public:
  void BeginRequest() noexcept;
  SpeculativeVerifier(hip::QwenGpuExecutor& target_executor,
                      std::unique_ptr<IDraftBackend> draft_backend,
                      SpeculativeOptions options = {});
  SpeculativeVerifier(ISpeculativeTargetExecutor& target_executor,
                      std::unique_ptr<IDraftBackend> draft_backend,
                      SpeculativeOptions options = {});

  /// Runs speculative auto-regressive generation
  std::vector<tokenization::TokenId> Generate(
      std::span<const tokenization::TokenId> prompt_tokens,
      const models::GenerationOptions& options,
      const std::function<bool(tokenization::TokenId, std::string_view)>&
          on_token = nullptr);

  /// Resets target and draft state, prefills the prompt, and returns the first
  /// target token. This setup is outside benchmarked decode regions.
  [[nodiscard]] tokenization::TokenId Prime(
      std::span<const tokenization::TokenId> prompt_tokens,
      bool compute_logits = true);
  [[nodiscard]] tokenization::TokenId ExtendPrompt(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits = true);

  /// Commits one externally supplied continuation token through the target
  /// model and updates draft-provider state with the matching target features.
  [[nodiscard]] tokenization::TokenId AdvanceCommittedToken(
      tokenization::TokenId token, std::uint32_t position);

  /// Performs a single speculative verification step
  struct StepResult {
    std::vector<tokenization::TokenId> emitted_tokens;
    std::size_t accepted_count{0};
    std::size_t draft_count{0};
    tokenization::TokenId next_token{0};
    std::vector<float> next_token_logits;
    bool hit_eos{false};
    std::size_t physical_width{1};
  };

  struct StepRequest {
    SpeculativeVerifier& verifier;
    std::span<const tokenization::TokenId> sequence;
    std::uint32_t position;
    tokenization::TokenId current_token;
    tokenization::TokenId eos_id;
    std::uint32_t max_emitted_tokens;
    sampling::SamplerState& sampler;
  };

  [[nodiscard]] static std::vector<StepResult> VerifyBatch(
      std::span<const StepRequest> requests);

  [[nodiscard]] std::span<const float> CopyLastTargetLogits() {
    return target_executor_->CopyLastLogits();
  }

  StepResult VerifyStep(std::vector<tokenization::TokenId>& current_sequence,
                        std::uint32_t cur_pos,
                        tokenization::TokenId current_token,
                        tokenization::TokenId eos_id);
  StepResult VerifyStep(std::vector<tokenization::TokenId>& current_sequence,
                        std::uint32_t cur_pos,
                        tokenization::TokenId current_token,
                        tokenization::TokenId eos_id,
                        std::uint32_t max_emitted_tokens);
  StepResult VerifyStep(std::vector<tokenization::TokenId>& current_sequence,
                        std::uint32_t cur_pos,
                        tokenization::TokenId current_token,
                        tokenization::TokenId eos_id,
                        std::uint32_t max_emitted_tokens, float temperature,
                        std::uint64_t* rng_state);
  StepResult VerifyStep(std::vector<tokenization::TokenId>& current_sequence,
                        std::uint32_t cur_pos,
                        tokenization::TokenId current_token,
                        tokenization::TokenId eos_id,
                        std::uint32_t max_emitted_tokens,
                        sampling::SamplerState& sampler);

  [[nodiscard]] std::unique_ptr<SpeculativeVerifierSnapshot> Snapshot() const;
  void RestoreSnapshot(const SpeculativeVerifierSnapshot& snapshot);
  void RestorePersistentSnapshot(std::span<const std::uint8_t> payload);

  [[nodiscard]] const SpeculativeStats& GetStats() const noexcept {
    return stats_;
  }

  [[nodiscard]] std::uint32_t GetCurrentDraftLength() const noexcept {
    return current_draft_length_;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const;
  void Reset() noexcept;

private:
  void ConfigureAdaptiveDraftPolicy();
  void ResetAdaptiveDraftLength() noexcept;
  void UpdateAdaptiveDraftLength(std::size_t accepted, std::size_t drafted);
  struct PreparedStep;
  [[nodiscard]] PreparedStep PrepareStep(const StepRequest& request);
  [[nodiscard]] StepResult FinishStep(PreparedStep& prepared,
                                      VerificationChunkResult verification,
                                      sampling::SamplerState& sampler);
  [[nodiscard]] StepResult VerifySequentialStep(
      std::span<const tokenization::TokenId> current_sequence,
      std::uint32_t cur_pos, tokenization::TokenId current_token,
      tokenization::TokenId eos_id, std::uint32_t max_emitted_tokens);

  std::unique_ptr<ISpeculativeTargetExecutor> owned_target_executor_;
  ISpeculativeTargetExecutor* target_executor_;
  std::unique_ptr<IDraftBackend> draft_backend_;
  SpeculativeOptions options_;
  SpeculativeStats stats_;
  std::uint32_t current_draft_length_{3};
  std::deque<float> rolling_acceptance_;
  bool use_batched_verification_{false};
};

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_
