#include <array>
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

#include "src/core/sampling.hpp"
#include "src/core/speculative/draft_backend.hpp"
#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace {

using gufo::tokenization::TokenId;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

class ScriptedTargetExecutor final
    : public gufo::speculative::ISpeculativeTargetExecutor {
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

class HiddenAwareDraftBackend final : public gufo::speculative::IDraftBackend {
public:
  [[nodiscard]] std::string_view Name() const noexcept override {
    return "HiddenAwareDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] bool PrimeTargetContext(
      const gufo::speculative::DraftTargetContext& context) override {
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

  [[nodiscard]] gufo::speculative::DraftProposal Propose(
      std::span<const TokenId> prompt_tokens, std::uint32_t current_pos,
      std::uint32_t max_tokens) override {
    (void)prompt_tokens;
    gufo::speculative::DraftProposal proposal;
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

class SampledTargetExecutor final
    : public gufo::speculative::ISpeculativeTargetExecutor {
public:
  explicit SampledTargetExecutor(std::vector<float> target_logits)
      : target_logits_(std::move(target_logits)) {}

  void Reset() noexcept override {
    state_.clear();
    saved_state_.clear();
    last_logits_ = {0.0F, -INFINITY, -INFINITY};
  }

  TokenId ForwardPromptBatch(std::span<const TokenId> prompt_tokens) override {
    state_.assign(prompt_tokens.begin(), prompt_tokens.end());
    last_logits_ = {0.0F, -INFINITY, -INFINITY};
    return 0;
  }

  TokenId ForwardToken(TokenId token_id, std::uint32_t pos,
                       bool compute_logits) override {
    (void)compute_logits;
    Expect(pos == state_.size(), "sampled target position");
    state_.push_back(token_id);
    last_logits_ = state_.size() == 2
                       ? target_logits_
                       : std::vector<float>{0.0F, -INFINITY, -INFINITY};
    return gufo::sampling::SampleLogits(last_logits_, 0.0F, nullptr);
  }

  void SaveState(std::uint32_t valid_context) override {
    Expect(valid_context == state_.size(), "sampled target snapshot boundary");
    saved_state_ = state_;
  }

  void RestoreState() override { state_ = saved_state_; }

  std::span<const float> CopyLastLogits() override { return last_logits_; }

  std::size_t VocabularySize() const noexcept override {
    return target_logits_.size();
  }
  std::size_t StateSize() const noexcept { return state_.size(); }

  TokenId GetEosTokenId() const noexcept override { return 99; }

  std::string_view DecodeToken(TokenId) const noexcept override {
    return "token";
  }

private:
  std::vector<float> target_logits_;
  std::vector<float> last_logits_;
  std::vector<TokenId> state_;
  std::vector<TokenId> saved_state_;
};

class BinarySampledDraftBackend final
    : public gufo::speculative::IDraftBackend {
public:
  BinarySampledDraftBackend(float first_probability, float second_probability)
      : probabilities_{first_probability, second_probability} {}

  std::string_view Name() const noexcept override {
    return "BinarySampledDraftBackend";
  }

  gufo::speculative::DraftProposal Propose(std::span<const TokenId>,
                                           std::uint32_t current_pos,
                                           std::uint32_t max_tokens) override {
    gufo::speculative::DraftProposal proposal;
    proposal.start_pos = current_pos;
    if (max_tokens > 0) {
      proposal.tokens.push_back(probabilities_[0] >= probabilities_[1] ? 1 : 2);
    }
    return proposal;
  }

  gufo::speculative::DraftProposal ProposeSampled(
      std::span<const TokenId>, std::uint32_t current_pos,
      std::uint32_t max_tokens, float, std::uint64_t* rng_state) override {
    gufo::speculative::DraftProposal proposal;
    proposal.start_pos = current_pos;
    if (max_tokens == 0) {
      return proposal;
    }
    proposal.candidates_per_token = 2;
    proposal.candidate_ids = {1, 2};
    proposal.candidate_probabilities.assign(probabilities_.begin(),
                                            probabilities_.end());
    proposal.tokens.push_back(
        gufo::sampling::Uniform(rng_state) < probabilities_[0] ? 1 : 2);
    return proposal;
  }

  bool SupportsSampledProposals() const noexcept override { return true; }

private:
  std::array<float, 2> probabilities_;
};

class FixedSampledDraftBackend final : public gufo::speculative::IDraftBackend {
public:
  explicit FixedSampledDraftBackend(gufo::speculative::DraftProposal proposal)
      : proposal_(std::move(proposal)) {}
  std::string_view Name() const noexcept override {
    return "FixedSampledDraft";
  }
  gufo::speculative::DraftProposal Propose(std::span<const TokenId>,
                                           std::uint32_t,
                                           std::uint32_t) override {
    return proposal_;
  }
  gufo::speculative::DraftProposal ProposeSampled(std::span<const TokenId>,
                                                  std::uint32_t, std::uint32_t,
                                                  float,
                                                  std::uint64_t*) override {
    return proposal_;
  }
  bool SupportsSampledProposals() const noexcept override { return true; }

private:
  gufo::speculative::DraftProposal proposal_;
};

class PersistentDraftSnapshot final
    : public gufo::speculative::IDraftBackendSnapshot {
public:
  explicit PersistentDraftSnapshot(std::uint32_t marker) : marker_(marker) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return sizeof(marker_);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    return sizeof(marker_);
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    if (destination.size() != sizeof(marker_)) {
      throw std::invalid_argument(
          "persistent draft test destination size is invalid");
    }
    for (std::size_t byte = 0; byte < sizeof(marker_); ++byte) {
      destination[byte] = static_cast<std::uint8_t>(marker_ >> (byte * 8U));
    }
    return destination.size();
  }

  [[nodiscard]] std::uint32_t Marker() const noexcept { return marker_; }

private:
  std::uint32_t marker_;
};

class PersistentDraftBackend final : public gufo::speculative::IDraftBackend {
public:
  PersistentDraftBackend(std::vector<TokenId> target_tokens,
                         std::size_t prompt_size)
      : target_tokens_(std::move(target_tokens)), prompt_size_(prompt_size) {}

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "PersistentDraftBackend";
  }

  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }

  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    static constexpr std::array<std::uint32_t, 1> target_layers = {7};
    return target_layers;
  }

  [[nodiscard]] gufo::speculative::DraftProposal Propose(
      std::span<const TokenId>, std::uint32_t current_pos,
      std::uint32_t max_tokens) override {
    Expect(current_pos >= prompt_size_, "persistent draft position");
    ++marker_;
    gufo::speculative::DraftProposal proposal;
    proposal.start_pos = current_pos;
    const std::size_t target_start =
        static_cast<std::size_t>(current_pos) - prompt_size_ + 1U;
    if (target_start >= target_tokens_.size()) {
      return proposal;
    }
    const std::size_t count =
        std::min<std::size_t>(max_tokens, target_tokens_.size() - target_start);
    proposal.tokens.insert(
        proposal.tokens.end(),
        target_tokens_.begin() + static_cast<std::ptrdiff_t>(target_start),
        target_tokens_.begin() +
            static_cast<std::ptrdiff_t>(target_start + count));
    return proposal;
  }

  void AcceptFeedback(std::span<const TokenId> accepted,
                      TokenId correction_token) override {
    marker_ += static_cast<std::uint32_t>(accepted.size());
    marker_ ^= correction_token;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes() const override {
    return sizeof(marker_);
  }

  [[nodiscard]] std::unique_ptr<gufo::speculative::IDraftBackendSnapshot>
  Snapshot() const override {
    return std::make_unique<PersistentDraftSnapshot>(marker_);
  }

  void RestoreSnapshot(
      const gufo::speculative::IDraftBackendSnapshot& snapshot) override {
    const auto* persistent =
        dynamic_cast<const PersistentDraftSnapshot*>(&snapshot);
    if (persistent == nullptr) {
      throw std::invalid_argument("persistent draft test snapshot type");
    }
    marker_ = persistent->Marker();
  }

  void RestorePersistentSnapshot(
      std::span<const std::uint8_t> payload) override {
    if (payload.size() != sizeof(marker_)) {
      throw std::invalid_argument("persistent draft test payload size");
    }
    std::uint32_t restored = 0;
    for (std::size_t byte = 0; byte < sizeof(restored); ++byte) {
      restored |= static_cast<std::uint32_t>(payload[byte]) << (byte * 8U);
    }
    marker_ = restored;
  }

  [[nodiscard]] std::uint32_t Marker() const noexcept { return marker_; }

private:
  std::vector<TokenId> target_tokens_;
  std::size_t prompt_size_;
  std::uint32_t marker_{0};
};

gufo::models::GenerationOptions GenerationOptions(std::size_t max_tokens,
                                                  TokenId eos_id) {
  gufo::models::GenerationOptions options;
  options.max_new_tokens = max_tokens;
  options.eos_token_id = eos_id;
  return options;
}

void TestSpeculativeDraftBackendInterface() {
  std::vector<TokenId> pool = {101, 102, 103, 104};
  gufo::speculative::MockDraftBackend backend(pool);

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
  gufo::speculative::SpeculativeStats stats;
  Expect(stats.AcceptanceRate() == 0.0F, "empty acceptance rate");

  stats.total_draft_tokens = 10;
  stats.total_accepted_tokens = 8;
  Expect(std::abs(stats.AcceptanceRate() - 0.80F) < 1e-5F,
         "non-empty acceptance rate");
}

void TestFullAcceptanceProducesTargetBonusToken() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12, 13}, eos_id);
  auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 12});
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
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
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
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

void TestRetainedPrefixAdvanceUpdatesTargetAndDraftState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<HiddenAwareDraftBackend>();
  auto* backend_view = backend.get();
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  Expect(verifier.Prime(prompt) == 10,
         "retained-prefix test primes the target frontier");
  const auto next = verifier.AdvanceCommittedToken(42, 3);

  Expect(next == 11, "retained-prefix advance returns the next target token");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 42}),
         "retained-prefix advance commits the supplied suffix token");
  Expect(backend_view->UpdatedHidden() == std::vector<float>({42.0F, 3.0F}),
         "retained-prefix advance updates DFlash target features");
}

void TestPartialRejectionRestoresAndReplaysState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11, 12}, eos_id);
  auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
      std::vector<TokenId>{11, 99});
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto current = verifier.Prime(prompt);
  std::vector<TokenId> sequence{1, 2, 3, current};
  const auto step = verifier.VerifyStep(sequence, 3, current, eos_id, 3);

  Expect(step.emitted_tokens == std::vector<TokenId>({11, 12}) &&
             step.accepted_count == 1 && step.draft_count == 2,
         "partial rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11}),
         "partial rejection target state");
  Expect(target.RestoreCount() == 1, "partial rejection must restore once");
}

void TestImmediateRejectionRestoresGreedyState() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
      std::vector<TokenId>{99, 98});
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
  const std::vector<TokenId> prompt = {1, 2, 3};

  const auto current = verifier.Prime(prompt);
  std::vector<TokenId> sequence{1, 2, 3, current};
  const auto step = verifier.VerifyStep(sequence, 3, current, eos_id, 3);

  Expect(step.emitted_tokens == std::vector<TokenId>({11}) &&
             step.accepted_count == 0 && step.draft_count == 2,
         "immediate rejection must match greedy output");
  Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10}),
         "immediate rejection target state");
  Expect(target.RestoreCount() == 1, "immediate rejection must restore once");
}

void TestFirstPrefillTokenHonorsBudgetAndCallback() {
  constexpr TokenId eos_id = 900;
  ScriptedTargetExecutor target({10, 11}, eos_id);
  auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
      std::vector<TokenId>{11});
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
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
  auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
      std::vector<TokenId>{1});
  gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
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

void TestSampledSpeculationMatchesTargetDistribution() {
  constexpr std::size_t trials = 4096;
  constexpr double expected_second_probability = 0.75;
  std::size_t second_token_count = 0;
  std::size_t drafted_count = 0;

  for (std::uint64_t seed = 1; seed <= trials; ++seed) {
    SampledTargetExecutor target({-INFINITY, std::log(0.25F), std::log(0.75F)});
    auto backend = std::make_unique<BinarySampledDraftBackend>(0.80F, 0.20F);
    gufo::speculative::SpeculativeOptions options;
    options.max_draft_tokens = 1;
    options.min_draft_tokens = 1;
    options.initial_draft_tokens = 1;
    options.enable_adaptive_draft_length = false;
    gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend),
                                                    options);
    const std::vector<TokenId> prompt = {0};
    const TokenId current = verifier.Prime(prompt);
    std::vector<TokenId> sequence = {prompt.front(), current};
    std::uint64_t rng_state = seed;
    const auto result =
        verifier.VerifyStep(sequence, 1, current, 99, 2, 1.0F, &rng_state);
    Expect(!result.emitted_tokens.empty(),
           "sampled speculation emits a target-distributed token");
    second_token_count += result.emitted_tokens.front() == 2 ? 1 : 0;
    drafted_count += result.draft_count;
  }

  const double observed = static_cast<double>(second_token_count) / trials;
  Expect(std::abs(observed - expected_second_probability) < 0.035,
         "lossless rejection sampling preserves the target distribution");
  Expect(drafted_count == trials,
         "positive-temperature verification continues to use drafts");
}

void TestFilteredSampledSpeculationMatchesTargetDistribution() {
  constexpr std::size_t trials = 4096;
  const std::vector<float> target_logits = {-INFINITY, std::log(0.40F),
                                            std::log(0.60F)};
  gufo::sampling::SamplingConfig config;
  config.temperature = 0.8F;
  config.top_k = 2;
  config.top_p = 0.95F;
  config.min_p = 0.05F;
  config.min_keep = 1;
  config.repeat_penalty = 1.2F;
  config.repeat_last_n = 2;
  config.frequency_penalty = 0.1F;
  config.presence_penalty = 0.05F;

  const std::vector<TokenId> initial_sequence = {1, 0};
  const double expected_second_probability =
      gufo::sampling::BuildDistribution(target_logits, config, initial_sequence)
          .probability(2);
  std::size_t second_token_count = 0;
  std::size_t drafted_count = 0;

  for (std::uint64_t seed = 1; seed <= trials; ++seed) {
    SampledTargetExecutor target(target_logits);
    auto backend = std::make_unique<BinarySampledDraftBackend>(0.80F, 0.20F);
    gufo::speculative::SpeculativeOptions options;
    options.max_draft_tokens = 1;
    options.min_draft_tokens = 1;
    options.initial_draft_tokens = 1;
    options.enable_adaptive_draft_length = false;
    gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend),
                                                    options);
    const std::vector<TokenId> prompt = {1};
    const TokenId current = verifier.Prime(prompt);
    std::vector<TokenId> sequence = {prompt.front(), current};
    config.seed = static_cast<std::int64_t>(seed);
    gufo::sampling::SamplerState sampler(config, sequence);
    const auto result =
        verifier.VerifyStep(sequence, 1, current, 99, 2, sampler);
    Expect(!result.emitted_tokens.empty(),
           "filtered speculation emits a target-distributed token");
    second_token_count += result.emitted_tokens.front() == 2 ? 1 : 0;
    drafted_count += result.draft_count;
  }

  const double observed = static_cast<double>(second_token_count) / trials;
  Expect(std::abs(observed - expected_second_probability) < 0.035,
         "filtered rejection sampling preserves the target distribution");
  Expect(drafted_count == trials,
         "top-k/top-p/min-p/penalty requests continue to use drafts");
}

void TestMalformedProposalDoesNotAdvanceTarget() {
  for (int kind = 0; kind < 7; ++kind) {
    SampledTargetExecutor target({-INFINITY, std::log(0.25F), std::log(0.75F)});
    gufo::speculative::DraftProposal proposal{
        .tokens = {1, 2},
        .candidate_ids = {1, 2, 1, 2},
        .candidate_probabilities = {0.8F, 0.2F, 0.8F, 0.2F},
        .candidates_per_token = 2,
        .start_pos = 1,
    };
    // Corrupt the second row: validation must happen even when verification
    // could reject the first token and never consume this row.
    switch (kind) {
      case 0:
        proposal.candidate_ids[2] = 2;
        break;
      case 1:
        proposal.candidate_ids[2] = 3;
        break;
      case 2:
        proposal.candidate_probabilities[2] = NAN;
        break;
      case 3:
        proposal.candidate_probabilities[2] = 0.799F;
        break;
      case 4:
        proposal.candidate_probabilities[2] = 1.0F;
        proposal.candidate_probabilities[3] = 0.0F;
        break;
      case 5:
        proposal.start_pos = 2;
        break;
      case 6:
        proposal.candidate_ids.pop_back();
        break;
    }
    gufo::speculative::SpeculativeOptions options;
    options.max_draft_tokens = 2;
    options.initial_draft_tokens = 2;
    options.enable_adaptive_draft_length = false;
    gufo::speculative::SpeculativeVerifier verifier(
        target, std::make_unique<FixedSampledDraftBackend>(std::move(proposal)),
        options);
    const std::vector<TokenId> prompt{0};
    const auto current = verifier.Prime(prompt);
    std::vector<TokenId> sequence{0, current};
    std::uint64_t rng = 42;
    bool rejected = false;
    try {
      (void)verifier.VerifyStep(sequence, 1, current, 99, 3, 1.0F, &rng);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Expect(rejected && target.StateSize() == prompt.size(),
           "malformed sparse proposal must fail before target execution");
  }
}

void TestStopAndBudgetKeepExactFrontier() {
  for (const bool batched : {false, true}) {
    ScriptedTargetExecutor target({10, 11, 900, 12}, 900);
    auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
        std::vector<TokenId>{11, 900, 12});
    gufo::speculative::SpeculativeOptions options;
    options.use_batched_verification = batched;
    gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend),
                                                    options);
    const std::vector<TokenId> prompt{1, 2, 3};
    const auto current = verifier.Prime(prompt);
    std::vector<TokenId> sequence{1, 2, 3, current};
    const auto result = verifier.VerifyStep(sequence, 3, current, 900, 4);
    Expect(result.emitted_tokens == std::vector<TokenId>({11, 900}) &&
               result.next_token == 900 && result.hit_eos,
           "greedy speculation must end at the first stop token");
    Expect(target.State() == std::vector<TokenId>({1, 2, 3, 10, 11}),
           "a stop token and its suffix must remain uncommitted");
  }
  {
    SampledTargetExecutor target({-INFINITY, -INFINITY, 0});
    gufo::speculative::DraftProposal proposal{
        .tokens = {2, 1},
        .candidate_ids = {2, 1, 1, 2},
        .candidate_probabilities = {1, 0, 1, 0},
        .candidates_per_token = 2,
        .start_pos = 1,
    };
    gufo::speculative::SpeculativeOptions options;
    options.max_draft_tokens = options.initial_draft_tokens = 2;
    gufo::speculative::SpeculativeVerifier verifier(
        target, std::make_unique<FixedSampledDraftBackend>(proposal), options);
    const std::vector<TokenId> prompt{0};
    const auto current = verifier.Prime(prompt);
    std::vector<TokenId> sequence{0, current};
    std::uint64_t rng = 42;
    const auto result =
        verifier.VerifyStep(sequence, 1, current, 2, 3, 1.0F, &rng);
    Expect(result.emitted_tokens == std::vector<TokenId>({2}) &&
               result.next_token == 2 && result.hit_eos &&
               target.StateSize() == 2,
           "sampled speculation must stop before committing EOS or its suffix");
  }
  {
    ScriptedTargetExecutor target({10, 11, 12, 13, 14}, 900);
    auto backend = std::make_unique<gufo::speculative::MockDraftBackend>(
        std::vector<TokenId>{11, 12, 13});
    gufo::speculative::SpeculativeVerifier verifier(target, std::move(backend));
    const std::vector<TokenId> prompt{1, 2, 3};
    const auto output = verifier.Generate(prompt, GenerationOptions(2, 900));
    Expect(output == std::vector<TokenId>({10, 11}) &&
               target.State() == std::vector<TokenId>({1, 2, 3, 10}),
           "greedy generation must not commit beyond its output budget");
  }
}

void TestPersistentVerifierSnapshotRoundTrip() {
  constexpr TokenId eos_id = 900;
  const std::vector<TokenId> prompt = {1, 2, 3};
  const std::vector<TokenId> generated = {10, 11, 12, 13, 14, 15, 16};
  ScriptedTargetExecutor source_target(generated, eos_id);
  auto source_backend =
      std::make_unique<PersistentDraftBackend>(generated, prompt.size());
  auto* source_backend_view = source_backend.get();
  gufo::speculative::SpeculativeOptions options;
  options.max_draft_tokens = 4;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = 2;
  options.rolling_window = 4;
  gufo::speculative::SpeculativeVerifier source(
      source_target, std::move(source_backend), options);

  const auto generated_tokens =
      source.Generate(prompt, GenerationOptions(6, eos_id));
  Expect(generated_tokens ==
             std::vector<TokenId>(generated.begin(), generated.begin() + 6),
         "persistent verifier source output");
  const auto source_stats = source.GetStats();
  const auto source_draft_length = source.GetCurrentDraftLength();
  const auto source_marker = source_backend_view->Marker();
  auto snapshot = source.Snapshot();
  std::vector<std::uint8_t> payload(snapshot->PersistentPayloadBytes());
  Expect(snapshot->SerializePersistent(payload) == payload.size(),
         "persistent verifier byte count");

  ScriptedTargetExecutor restored_target(generated, eos_id);
  auto restored_backend =
      std::make_unique<PersistentDraftBackend>(generated, prompt.size());
  auto* restored_backend_view = restored_backend.get();
  gufo::speculative::SpeculativeVerifier restored(
      restored_target, std::move(restored_backend), options);

  auto corrupt = payload;
  corrupt.front() ^= 0xFFU;
  bool rejected_corruption = false;
  try {
    restored.RestorePersistentSnapshot(corrupt);
  } catch (const std::invalid_argument&) {
    rejected_corruption = true;
  }
  Expect(rejected_corruption, "persistent verifier rejects a malformed header");
  Expect(restored_backend_view->Marker() == 0,
         "malformed verifier payload does not mutate draft state");

  restored.RestorePersistentSnapshot(payload);
  const auto restored_stats = restored.GetStats();
  Expect(restored_target.HiddenCaptureEnabled(),
         "persistent verifier restore re-enables target hidden capture");
  Expect(restored_stats.total_draft_tokens == source_stats.total_draft_tokens &&
             restored_stats.total_accepted_tokens ==
                 source_stats.total_accepted_tokens &&
             restored_stats.total_verification_steps ==
                 source_stats.total_verification_steps &&
             restored_stats.total_emitted_tokens ==
                 source_stats.total_emitted_tokens,
         "persistent verifier restores exact counters");
  Expect(restored.GetCurrentDraftLength() == source_draft_length,
         "persistent verifier restores adaptive draft length");
  Expect(restored_backend_view->Marker() == source_marker,
         "persistent verifier restores provider state");

  auto restored_snapshot = restored.Snapshot();
  std::vector<std::uint8_t> restored_payload(
      restored_snapshot->PersistentPayloadBytes());
  Expect(restored_snapshot->SerializePersistent(restored_payload) ==
             restored_payload.size(),
         "restored persistent verifier byte count");
  Expect(restored_payload == payload,
         "persistent verifier round trip is byte exact");
}

}  // namespace

int main() {
  TestSpeculativeDraftBackendInterface();
  TestSpeculativeStats();
  TestFullAcceptanceProducesTargetBonusToken();
  TestPartialRejectionRestoresAndReplaysState();
  TestImmediateRejectionRestoresGreedyState();
  TestHiddenAwareBackendReceivesCommittedTargetState();
  TestRetainedPrefixAdvanceUpdatesTargetAndDraftState();
  TestFirstPrefillTokenHonorsBudgetAndCallback();
  TestFirstPrefillEosIsNotEmitted();
  TestSampledSpeculationMatchesTargetDistribution();
  TestFilteredSampledSpeculationMatchesTargetDistribution();
  TestMalformedProposalDoesNotAdvanceTarget();
  TestStopAndBudgetKeepExactFrontier();
  TestPersistentVerifierSnapshotRoundTrip();
  std::cout << "All speculative verification tests passed.\n";
  return 0;
}

#else
int main() {
  std::cout << "HIP disabled, skipping speculative GPU tests.\n";
  return 0;
}
#endif
