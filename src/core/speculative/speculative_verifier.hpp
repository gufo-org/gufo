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
#include "src/models/qwen/generator.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"
#endif

namespace gufo::speculative {

enum class AdaptiveDraftPolicy {
  kRollingAcceptanceRate,
  kAcceptedTokenEma,
};

struct SpeculativeOptions {
  std::uint32_t max_draft_tokens{4};
  std::uint32_t min_draft_tokens{1};
  std::uint32_t initial_draft_tokens{3};
  std::size_t rolling_window{16};
  float target_acceptance_rate{0.70F};
  bool enable_adaptive_draft_length{true};
  AdaptiveDraftPolicy adaptive_draft_policy{
      AdaptiveDraftPolicy::kRollingAcceptanceRate};
  bool use_batched_verification{false};
  bool use_batched_lm_head{false};
  bool retain_frontier_logits{false};
  int target_bf16_from_layer{-1};
  int target_fp32_from_layer{-1};
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

class ISpeculativeTargetExecutor {
public:
  virtual ~ISpeculativeTargetExecutor() = default;

  virtual void Reset() noexcept = 0;
  [[nodiscard]] virtual tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) = 0;
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

private:
  SpeculativeVerifierSnapshot() = default;

  std::unique_ptr<IDraftBackendSnapshot> draft_snapshot_;
  SpeculativeStats stats_;
  std::uint32_t current_draft_length_{0};
  std::deque<float> rolling_acceptance_;
  float accepted_token_ema_{0.0F};

  friend class SpeculativeVerifier;
};

/// High-throughput speculative decoding verifier with transactional state
/// management
class SpeculativeVerifier {
public:
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
      std::span<const tokenization::TokenId> prompt_tokens);

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
  };

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

  [[nodiscard]] std::unique_ptr<SpeculativeVerifierSnapshot> Snapshot() const;
  void RestoreSnapshot(const SpeculativeVerifierSnapshot& snapshot);

  [[nodiscard]] const SpeculativeStats& GetStats() const noexcept {
    return stats_;
  }

  [[nodiscard]] std::uint32_t GetCurrentDraftLength() const noexcept {
    return current_draft_length_;
  }

  void Reset() noexcept;

private:
  void ConfigureAdaptiveDraftPolicy();
  void ResetAdaptiveDraftLength() noexcept;
  void UpdateDraftTargetHidden();
  void UpdateAdaptiveDraftLength(std::size_t accepted, std::size_t drafted);
  [[nodiscard]] StepResult VerifySampledStep(
      std::vector<tokenization::TokenId>& current_sequence,
      std::uint32_t cur_pos, tokenization::TokenId current_token,
      tokenization::TokenId eos_id, std::uint32_t max_emitted_tokens,
      float temperature, std::uint64_t* rng_state);

  std::unique_ptr<ISpeculativeTargetExecutor> owned_target_executor_;
  ISpeculativeTargetExecutor* target_executor_;
  std::unique_ptr<IDraftBackend> draft_backend_;
  SpeculativeOptions options_;
  SpeculativeStats stats_;
  std::uint32_t current_draft_length_{3};
  std::deque<float> rolling_acceptance_;
  float accepted_token_ema_{2.0F};
  bool use_batched_verification_{false};
};

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::speculative

#endif  // GUFO_CORE_SPECULATIVE_SPECULATIVE_VERIFIER_HPP_
