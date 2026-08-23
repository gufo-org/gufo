#ifndef STRIX_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_
#define STRIX_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen/qwen_tokenizer.hpp"

namespace strix::speculative {

/// Represents a speculative draft proposal block
struct DraftProposal {
  std::vector<tokenization::TokenId> tokens;
  float confidence{1.0F};
  std::uint32_t start_pos{0};
};

struct DraftTargetContext {
  std::span<const tokenization::TokenId> prompt_tokens;
  std::span<const float> prompt_hidden_states;
  std::size_t hidden_size{0};
  tokenization::TokenId first_token{0};
};

/// Provider-neutral interface for draft token generators (NPU, MTP heads, small
/// model, heuristic)
class IDraftBackend {
public:
  virtual ~IDraftBackend() = default;

  /// Returns the name / type of the draft backend
  [[nodiscard]] virtual std::string_view Name() const noexcept = 0;

  /// Proposes up to max_tokens draft tokens given the current sequence
  [[nodiscard]] virtual DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) = 0;

  /// Returns true when the backend consumes target-model hidden states.
  [[nodiscard]] virtual bool RequiresTargetHiddenStates() const noexcept {
    return false;
  }

  /// Primes provider-specific state after target prompt prefill.
  [[nodiscard]] virtual bool PrimeTargetContext(
      const DraftTargetContext& context) {
    (void)context;
    return true;
  }

  /// Supplies the target hidden state paired with the next correction token.
  virtual void UpdateTargetHidden(std::span<const float> hidden) {
    (void)hidden;
  }

  /// Notifies the draft backend of which tokens were accepted and the target
  /// correction token
  virtual void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                              tokenization::TokenId correction_token) {
    (void)accepted;
    (void)correction_token;
  }

  /// Resets internal draft generator state
  virtual void Reset() noexcept {}
};

/// Mock / test draft backend for deterministic verification testing
class MockDraftBackend : public IDraftBackend {
public:
  explicit MockDraftBackend(std::vector<tokenization::TokenId> candidate_pool)
      : candidate_pool_(std::move(candidate_pool)) {}

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "MockDraftBackend";
  }

  [[nodiscard]] DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override {
    (void)prompt_tokens;
    DraftProposal proposal;
    proposal.start_pos = current_pos;
    const std::size_t count =
        std::min<std::size_t>(max_tokens, candidate_pool_.size());
    for (std::size_t i = 0; i < count; ++i) {
      proposal.tokens.push_back(candidate_pool_[i]);
    }
    return proposal;
  }

  void SetCandidates(std::vector<tokenization::TokenId> candidates) {
    candidate_pool_ = std::move(candidates);
  }

private:
  std::vector<tokenization::TokenId> candidate_pool_;
};

}  // namespace strix::speculative

#endif  // STRIX_CORE_SPECULATIVE_DRAFT_BACKEND_HPP_
