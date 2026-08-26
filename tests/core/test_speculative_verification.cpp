#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using strix::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

class ScriptedTargetExecutor final
    : public strix::speculative::ISpeculativeTargetExecutor {
public:
  ScriptedTargetExecutor(std::vector<TokenId> generated_tokens, TokenId eos_id)
      : generated_tokens_(std::move(generated_tokens)), eos_id_(eos_id) {}

  void Reset() noexcept override {
    prompt_.clear();
    prompt_hidden_.clear();
    last_hidden_.clear();
    state_.clear();
    saved_state_.clear();
    restore_count_ = 0;
  }

  TokenId ForwardPromptBatch(std::span<const TokenId> prompt_tokens) override {
    prompt_.assign(prompt_tokens.begin(), prompt_tokens.end());
    state_ = prompt_;
    prompt_hidden_.clear();
    for (const auto token : prompt_) {
      prompt_hidden_.push_back(static_cast<float>(token));
      prompt_hidden_.push_back(-static_cast<float>(token));
    }
    if (!prompt_.empty()) {
      last_hidden_ = {static_cast<float>(prompt_.back()),
                      -static_cast<float>(prompt_.back())};
    }
    if (generated_tokens_.empty()) {
      return eos_id_;
    }
    return generated_tokens_.front();
  }

  TokenId ForwardToken(TokenId token_id, std::uint32_t pos,
                       bool compute_logits) override {
    (void)compute_logits;
    Expect(pos == state_.size(), "target position must match committed state");
    Expect(pos >= prompt_.size(), "target position must follow the prompt");
    const std::size_t generated_index = pos - prompt_.size();
    state_.push_back(token_id);
    last_hidden_ = {static_cast<float>(token_id), static_cast<float>(pos)};
    if (generated_index + 1 < generated_tokens_.size()) {
      return generated_tokens_[generated_index + 1];
    }
    return eos_id_;
  }

  void SaveState(std::uint32_t valid_context) override {
    Expect(valid_context == state_.size(),
           "snapshot length must match committed target state");
    saved_state_ = state_;
  }

  void RestoreState() override {
    Expect(!saved_state_.empty(), "target state must be saved before restore");
    state_ = saved_state_;
    ++restore_count_;
  }

  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids) override {
    (void)target_layer_ids;
    hidden_capture_enabled_ = enabled;
  }

  std::span<const float> GetPromptHiddenStates() const noexcept override {
    return prompt_hidden_;
  }

  std::span<const float> CopyLastHidden() override { return last_hidden_; }

  TokenId GetEosTokenId() const noexcept override { return eos_id_; }

  std::string_view DecodeToken(TokenId token_id) const noexcept override {
    (void)token_id;
    return "token";
  }

  [[nodiscard]] const std::vector<TokenId>& State() const noexcept {
    return state_;
  }

  [[nodiscard]] std::size_t RestoreCount() const noexcept {
    return restore_count_;
  }

  [[nodiscard]] bool HiddenCaptureEnabled() const noexcept {
    return hidden_capture_enabled_;
  }

private:
  std::vector<TokenId> generated_tokens_;
  TokenId eos_id_;
  std::vector<TokenId> prompt_;
  std::vector<float> prompt_hidden_;
  std::vector<float> last_hidden_;
  std::vector<TokenId> state_;
  std::vector<TokenId> saved_state_;
  std::size_t restore_count_{0};
  bool hidden_capture_enabled_{false};
};

class HiddenAwareDraftBackend final : public strix::speculative::IDraftBackend {
public:
  [[nodiscard]] std::string_view Name() const noexcept override {
    return "HiddenAwareDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const strix::speculative::DraftTargetContext& context) override {
    primed_prompt_.assign(context.prompt_tokens.begin(),
                          context.prompt_tokens.end());
    primed_hidden_.assign(context.prompt_hidden_states.begin(),
                          context.prompt_hidden_states.end());
    hidden_size_ = context.hidden_size;
    first_token_ = context.first_token;
    return context.hidden_size == 2 &&
           context.prompt_hidden_states.size() ==
               context.prompt_tokens.size() * context.hidden_size;
  }

  [[nodiscard]] strix::speculative::DraftProposal Propose(
      std::span<const TokenId> prompt_tokens, std::uint32_t current_pos,
      std::uint32_t max_tokens) override {
    (void)prompt_tokens;
    strix::speculative::DraftProposal proposal;
    proposal.start_pos = current_pos;
    if (max_tokens > 0) {
      proposal.tokens.push_back(11);
    }
    return proposal;
  }

  void AcceptFeedback(std::span<const TokenId> accepted,
                      TokenId correction_token) override {
    accepted_.assign(accepted.begin(), accepted.end());
    correction_token_ = correction_token;
  }

  void UpdateTargetHidden(std::span<const float> hidden) override {
    updated_hidden_.assign(hidden.begin(), hidden.end());
  }

  [[nodiscard]] const std::vector<TokenId>& PrimedPrompt() const noexcept {
    return primed_prompt_;
  }
  [[nodiscard]] const std::vector<float>& PrimedHidden() const noexcept {
    return primed_hidden_;
  }
  [[nodiscard]] std::size_t HiddenSize() const noexcept { return hidden_size_; }
  [[nodiscard]] TokenId FirstToken() const noexcept { return first_token_; }
  [[nodiscard]] const std::vector<TokenId>& Accepted() const noexcept {
    return accepted_;
  }
  [[nodiscard]] TokenId CorrectionToken() const noexcept {
    return correction_token_;
  }
  [[nodiscard]] const std::vector<float>& UpdatedHidden() const noexcept {
    return updated_hidden_;
  }

private:
  std::vector<TokenId> primed_prompt_;
  std::vector<float> primed_hidden_;
  std::size_t hidden_size_{0};
  TokenId first_token_{0};
  std::vector<TokenId> accepted_;
  TokenId correction_token_{0};
  std::vector<float> updated_hidden_;
};

strix::models::GenerationOptions GenerationOptions(std::size_t max_tokens,
                                                   TokenId eos_id) {
  strix::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  options.eos_token_id = eos_id;
  return options;
}

void TestSpeculativeDraftBackendInterface() {
  std::vector<TokenId> pool = {101, 102, 103, 104};
  strix::speculative::MockDraftBackend backend(pool);

  Expect(backend.Name() == "MockDraftBackend", "mock backend name");

  const std::vector<TokenId> prefix = {1, 2, 3};
  const auto proposal = backend.Propose(prefix, 3, 3);
  Expect(proposal.tokens == std::vector<TokenId>({101, 102, 103}),
         "mock proposal tokens");
  Expect(proposal.start_pos == 3, "mock proposal position");

  const std::vector<TokenId> accepted = {101, 102};
  backend.AcceptFeedback(accepted, 999);
}

void TestSpeculativeStats() {
  strix::speculative::SpeculativeStats stats;
  Expect(stats.AcceptanceRate() == 0.0F, "empty acceptance rate");

  stats.total_draft_tokens = 10;
  stats.total_accepted_tokens = 8;
  Expect(std::abs(stats.AcceptanceRate() - 0.80F) < 1e-5F,
         "non-empty acceptance rate");
}

void TestFullAcceptanceProducesTargetBonusToken() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12, 13}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 12});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(4, eos_id));

  Expect(output == std::vector<TokenId>({10, 11, 12, 13}),
         "full acceptance must emit the target bonus token");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11, 12}),
         "full acceptance target state");
  Expect(target.RestoreCount() == 0, "full acceptance must not roll back");
  Expect(!target.HiddenCaptureEnabled(),
         "plain draft backend must not capture target hidden states");
  Expect(verifier.GetStats().total_accepted_tokens == 2,
         "full acceptance statistics");
}

void TestHiddenAwareBackendReceivesCommittedTargetState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12}, eos_id);
  auto backend = std::make_unique<HiddenAwareDraftBackend>();
  auto* backend_view = backend.get();
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(3, eos_id));

  Expect(output == std::vector<TokenId>({10, 11, 12}),
         "hidden-aware output must match greedy target");
  Expect(target.HiddenCaptureEnabled(),
         "hidden-aware backend enables target capture");
  Expect(backend_view->PrimedPrompt() == prompt,
         "hidden-aware backend receives prompt tokens");
  Expect(backend_view->PrimedHidden() ==
             std::vector<float>({1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}),
         "hidden-aware backend receives all prompt hidden states");
  Expect(backend_view->HiddenSize() == 2,
         "hidden-aware backend receives hidden width");
  Expect(backend_view->FirstToken() == 10,
         "hidden-aware backend receives first target token");
  Expect(backend_view->Accepted() == std::vector<TokenId>({11}),
         "hidden-aware backend receives accepted draft");
  Expect(backend_view->CorrectionToken() == 12,
         "hidden-aware backend receives target correction");
  Expect(backend_view->UpdatedHidden() == std::vector<float>({11.0F, 4.0F}),
         "hidden-aware backend receives latest committed target hidden");
}

void TestPartialRejectionRestoresAndReplaysState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 99});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(3, eos_id));

  Expect(output == std::vector<TokenId>({10, 11, 12}),
         "partial rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11}),
         "partial rejection target state");
  Expect(target.RestoreCount() == 1, "partial rejection must restore once");
}

void TestImmediateRejectionRestoresGreedyState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{99, 98});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto output = verifier.Generate(prompt, GenerationOptions(2, eos_id));

  Expect(output == std::vector<TokenId>({10, 11}),
         "immediate rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10}),
         "immediate rejection target state");
  Expect(target.RestoreCount() == 1, "immediate rejection must restore once");
}

void TestFirstPrefillTokenHonorsBudgetAndCallback() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{11});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};
  std::vector<TokenId> callback_tokens;

  const auto output =
      verifier.Generate(prompt, GenerationOptions(1, eos_id),
                        [&](TokenId token, std::string_view piece) {
                          callback_tokens.push_back(token);
                          Expect(piece == "token", "callback decoded token");
                          return true;
                        });

  Expect(output == std::vector<TokenId>({10}), "one-token output budget");
  Expect(callback_tokens == output, "first token callback");
  Expect(target.State() == prompt,
         "first token remains uncommitted until the next decode step");
}

void TestFirstPrefillEosIsNotEmitted() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({eos_id}, eos_id);
  auto backend = std::make_unique<strix::speculative::MockDraftBackend>(
      std::vector<TokenId>{1});
  strix::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};
  std::size_t callback_count = 0;

  const auto output = verifier.Generate(prompt, GenerationOptions(4, eos_id),
                                        [&](TokenId, std::string_view) {
                                          ++callback_count;
                                          return true;
                                        });

  Expect(output.empty(), "prefill EOS must not be emitted");
  Expect(callback_count == 0, "prefill EOS callback");
  Expect(target.State() == prompt, "prefill EOS target state");
}

}  // namespace

int main() {
  TestSpeculativeDraftBackendInterface();
  TestSpeculativeStats();
  TestFullAcceptanceProducesTargetBonusToken();
  TestPartialRejectionRestoresAndReplaysState();
  TestImmediateRejectionRestoresGreedyState();
  TestHiddenAwareBackendReceivesCommittedTargetState();
  TestFirstPrefillTokenHonorsBudgetAndCallback();
  TestFirstPrefillEosIsNotEmitted();
  std::cout << "All speculative verification tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping speculative GPU tests.\n";
  return 0;
}
#endif
