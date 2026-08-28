#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>

#include "src/core/sampling.hpp"

namespace gufo::speculative {
namespace {

[[nodiscard]] bool CheckBatchedVerification() noexcept;

class QwenGpuSpeculativeTarget final : public ISpeculativeTargetExecutor {
public:
  explicit QwenGpuSpeculativeTarget(hip::QwenGpuExecutor& executor)
      : executor_(executor) {}

  void Reset() noexcept override { executor_.Reset(); }

  tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) override {
    return executor_.ForwardPromptBatch(prompt_tokens);
  }

  tokenization::TokenId ForwardToken(tokenization::TokenId token_id,
                                     std::uint32_t pos,
                                     bool compute_logits) override {
    return executor_.ForwardToken(token_id, pos, compute_logits);
  }

  void SaveState(std::uint32_t valid_context) override {
    executor_.SaveState(valid_context);
  }

  void RestoreState() override { executor_.RestoreState(); }

  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids) override {
    executor_.SetPromptHiddenCapture(enabled, target_layer_ids);
  }

  std::span<const float> GetPromptHiddenStates() const noexcept override {
    return executor_.GetPromptHiddenStates();
  }

  std::span<const float> CopyLastHidden() override {
    return executor_.CopyLastHidden();
  }

  std::span<const float> CopyLastLogits() override {
    return executor_.CopyLastLogits();
  }

  std::span<const float> CopyVerificationLogits(std::size_t row) override {
    return executor_.CopyVerificationLogits(row);
  }
  VerificationChunkResult ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden,
      bool capture_logits) override {
    VerificationChunkResult result;
    result.predictions = executor_.ForwardVerificationChunk(
        candidate_tokens, start_pos, capture_logits);
    if (capture_hidden) {
      const auto hidden = executor_.GetVerificationHiddenStates();
      result.hidden_states.assign(hidden.begin(), hidden.end());
      if (!candidate_tokens.empty()) {
        if (result.hidden_states.size() % candidate_tokens.size() != 0) {
          throw std::runtime_error(
              "batched verification hidden-state capture is malformed");
        }
        result.hidden_width =
            result.hidden_states.size() / candidate_tokens.size();
      }
    }
    if (capture_logits) {
      const auto logits = executor_.GetVerificationLogits();
      result.logits.assign(logits.begin(), logits.end());
      result.vocab_size = executor_.GetModel().GetConfig().vocab_size;
    }
    if (CheckBatchedVerification()) {
      executor_.RestoreState();
      VerificationChunkResult reference;
      reference.predictions.reserve(candidate_tokens.size());
      if (capture_hidden) {
        reference.hidden_width = result.hidden_width;
        reference.hidden_states.reserve(candidate_tokens.size() *
                                        reference.hidden_width);
      }
      if (capture_logits) {
        reference.vocab_size = result.vocab_size;
        reference.logits.reserve(candidate_tokens.size() *
                                 reference.vocab_size);
      }
      for (std::size_t index = 0; index < candidate_tokens.size(); ++index) {
        reference.predictions.push_back(executor_.ForwardToken(
            candidate_tokens[index],
            start_pos + static_cast<std::uint32_t>(index)));
        if (capture_hidden) {
          const auto hidden = executor_.CopyLastHidden();
          if (hidden.size() != reference.hidden_width) {
            throw std::runtime_error(
                "sequential verification hidden-state capture is malformed");
          }
          reference.hidden_states.insert(reference.hidden_states.end(),
                                         hidden.begin(), hidden.end());
        }
        if (capture_logits) {
          const auto logits = executor_.CopyLastLogits();
          reference.logits.insert(reference.logits.end(), logits.begin(),
                                  logits.end());
        }
      }
      for (std::size_t index = 0; index < candidate_tokens.size(); ++index) {
        if (result.predictions[index] != reference.predictions[index]) {
          std::cerr << "[spec-batch-check] pos=" << (start_pos + index)
                    << " input=" << candidate_tokens[index]
                    << " batch=" << result.predictions[index]
                    << " decode=" << reference.predictions[index] << '\n';
        }
        if (capture_hidden) {
          float max_abs_error = 0.0F;
          const std::size_t row_offset = index * result.hidden_width;
          const std::size_t hidden_size =
              executor_.GetModel().GetConfig().hidden_size;
          const std::size_t tap_count = result.hidden_width / hidden_size;
          std::vector<float> tap_max_abs_error(tap_count, 0.0F);
          for (std::size_t element = 0; element < result.hidden_width;
               ++element) {
            const float error =
                std::abs(result.hidden_states[row_offset + element] -
                         reference.hidden_states[row_offset + element]);
            max_abs_error = std::max(max_abs_error, error);
            tap_max_abs_error[element / hidden_size] =
                std::max(tap_max_abs_error[element / hidden_size], error);
          }
          std::cerr << "[spec-batch-check] pos=" << (start_pos + index)
                    << " hidden_max_abs_error=" << max_abs_error << " taps=";
          for (std::size_t tap = 0; tap < tap_count; ++tap) {
            if (tap != 0) {
              std::cerr << ',';
            }
            std::cerr << tap_max_abs_error[tap];
          }
          std::cerr << '\n';
        }
      }
      return reference;
    }
    return result;
  }

  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos) override {
    executor_.CommitVerificationChunk(committed_tokens, start_pos);
  }

  tokenization::TokenId GetEosTokenId() const noexcept override {
    return executor_.GetTokenizer().GetEosTokenId();
  }

  std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept override {
    return executor_.GetTokenizer().DecodeToken(token_id);
  }

private:
  hip::QwenGpuExecutor& executor_;
};

bool IsStopToken(tokenization::TokenId token,
                 tokenization::TokenId eos_id) noexcept {
  return token == eos_id || token == tokenization::kDefaultQwenEndoftextId ||
         token == 248044U || token == 248046U;
}

[[nodiscard]] bool ResolveFlag(const char* name, bool fallback) noexcept {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  const std::string_view setting{value};
  return setting != "0" && setting != "false" && setting != "off";
}

[[nodiscard]] bool CheckBatchedVerification() noexcept {
  const char* value = std::getenv("GUFO_SPEC_BATCH_VERIFY_CHECK");
  if (value == nullptr) {
    return false;
  }
  const std::string_view setting{value};
  return setting != "0" && setting != "false" && setting != "off";
}

}  // namespace

SpeculativeVerifier::SpeculativeVerifier(
    hip::QwenGpuExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : owned_target_executor_(
          std::make_unique<QwenGpuSpeculativeTarget>(target_executor)),
      target_executor_(owned_target_executor_.get()),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens),
      use_batched_verification_(ResolveFlag(
          "GUFO_SPEC_BATCH_VERIFY", options_.use_batched_verification)) {
  target_executor.SetVerificationPolicy({
      .batched_lm_head = options_.use_batched_lm_head,
      .bf16_from_layer = options_.target_bf16_from_layer,
      .fp32_from_layer = options_.target_fp32_from_layer,
  });
  ConfigureAdaptiveDraftPolicy();
}

SpeculativeVerifier::SpeculativeVerifier(
    ISpeculativeTargetExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : target_executor_(&target_executor),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens),
      use_batched_verification_(ResolveFlag(
          "GUFO_SPEC_BATCH_VERIFY", options_.use_batched_verification)) {
  ConfigureAdaptiveDraftPolicy();
}

void SpeculativeVerifier::Reset() noexcept {
  stats_ = {};
  ResetAdaptiveDraftLength();
  if (draft_backend_ != nullptr) {
    draft_backend_->Reset();
  }
}

void SpeculativeVerifier::ConfigureAdaptiveDraftPolicy() {
  options_.max_draft_tokens = std::max(options_.max_draft_tokens, 1U);
  options_.min_draft_tokens =
      std::clamp(options_.min_draft_tokens, 1U, options_.max_draft_tokens);
  options_.initial_draft_tokens =
      std::clamp(options_.initial_draft_tokens, options_.min_draft_tokens,
                 options_.max_draft_tokens);
  ResetAdaptiveDraftLength();
}

void SpeculativeVerifier::ResetAdaptiveDraftLength() noexcept {
  constexpr float initial_accepted_token_ema = 2.0F;
  rolling_acceptance_.clear();
  accepted_token_ema_ = initial_accepted_token_ema;
  if (options_.enable_adaptive_draft_length &&
      options_.adaptive_draft_policy ==
          AdaptiveDraftPolicy::kAcceptedTokenEma) {
    current_draft_length_ =
        std::clamp(static_cast<std::uint32_t>(std::lround(accepted_token_ema_)),
                   options_.min_draft_tokens, options_.max_draft_tokens);
    return;
  }
  current_draft_length_ = options_.initial_draft_tokens;
}

void SpeculativeVerifier::UpdateAdaptiveDraftLength(std::size_t accepted,
                                                    std::size_t drafted) {
  if (!options_.enable_adaptive_draft_length || drafted == 0) {
    return;
  }

  if (options_.adaptive_draft_policy ==
      AdaptiveDraftPolicy::kAcceptedTokenEma) {
    constexpr float ema_alpha = 0.25F;
    constexpr float full_accept_probe = 1.0F;
    if (accepted >= drafted) {
      accepted_token_ema_ += full_accept_probe;
    } else {
      accepted_token_ema_ = ((1.0F - ema_alpha) * accepted_token_ema_) +
                            (ema_alpha * static_cast<float>(accepted));
    }
    accepted_token_ema_ = std::min(
        accepted_token_ema_, static_cast<float>(options_.max_draft_tokens));
    current_draft_length_ =
        std::clamp(static_cast<std::uint32_t>(std::lround(accepted_token_ema_)),
                   options_.min_draft_tokens, options_.max_draft_tokens);
    return;
  }

  const float rate = static_cast<float>(accepted) / static_cast<float>(drafted);
  rolling_acceptance_.push_back(rate);
  if (rolling_acceptance_.size() > options_.rolling_window) {
    rolling_acceptance_.pop_front();
  }

  const float avg_rate = std::accumulate(rolling_acceptance_.begin(),
                                         rolling_acceptance_.end(), 0.0F) /
                         static_cast<float>(rolling_acceptance_.size());

  if (avg_rate > options_.target_acceptance_rate &&
      current_draft_length_ < options_.max_draft_tokens) {
    ++current_draft_length_;
  } else if (avg_rate < (options_.target_acceptance_rate * 0.5F) &&
             current_draft_length_ > options_.min_draft_tokens) {
    --current_draft_length_;
  }
}

void SpeculativeVerifier::UpdateDraftTargetHidden() {
  if (draft_backend_ == nullptr ||
      !draft_backend_->RequiresTargetHiddenStates()) {
    return;
  }
  const auto hidden = target_executor_->CopyLastHidden();
  if (hidden.empty()) {
    throw std::runtime_error(
        "draft backend requires a committed target hidden state");
  }
  draft_backend_->UpdateTargetHidden(hidden);
}

tokenization::TokenId SpeculativeVerifier::AdvanceCommittedToken(
    tokenization::TokenId token, std::uint32_t position) {
  const auto next = target_executor_->ForwardToken(token, position);
  UpdateDraftTargetHidden();
  return next;
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id) {
  return VerifyStep(current_sequence, cur_pos, current_token, eos_id,
                    std::numeric_limits<std::uint32_t>::max());
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id,
    std::uint32_t max_emitted_tokens) {
  return VerifyStep(current_sequence, cur_pos, current_token, eos_id,
                    max_emitted_tokens, 0.0F, nullptr);
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id,
    std::uint32_t max_emitted_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!std::isfinite(temperature) || temperature < 0.0F) {
    throw std::invalid_argument(
        "speculative temperature must be finite and nonnegative");
  }
  if (temperature > 0.0F) {
    return VerifySampledStep(current_sequence, cur_pos, current_token, eos_id,
                             max_emitted_tokens, temperature, rng_state);
  }
  if (max_emitted_tokens == 0) {
    throw std::invalid_argument(
        "speculative verification must emit at least one token");
  }
  const std::uint32_t max_draft_tokens =
      max_emitted_tokens > 1
          ? std::min(current_draft_length_, max_emitted_tokens - 1)
          : 0;
  if (draft_backend_ == nullptr || max_draft_tokens == 0) {
    const auto next = target_executor_->ForwardToken(current_token, cur_pos);
    std::vector<float> next_logits;
    if (options_.retain_frontier_logits) {
      const auto logits = target_executor_->CopyLastLogits();
      next_logits.assign(logits.begin(), logits.end());
    }
    UpdateDraftTargetHidden();
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = IsStopToken(next, eos_id);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .next_token_logits = std::move(next_logits),
            .hit_eos = hit_eos};
  }

  // 1. Propose draft tokens
  const auto proposal =
      draft_backend_->Propose(current_sequence, cur_pos, max_draft_tokens);
  if (proposal.tokens.empty()) {
    const auto next = target_executor_->ForwardToken(current_token, cur_pos);
    std::vector<float> next_logits;
    if (options_.retain_frontier_logits) {
      const auto logits = target_executor_->CopyLastLogits();
      next_logits.assign(logits.begin(), logits.end());
    }
    UpdateDraftTargetHidden();
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = IsStopToken(next, eos_id);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .next_token_logits = std::move(next_logits),
            .hit_eos = hit_eos};
  }

  const std::size_t num_draft = proposal.tokens.size();
  const bool capture_target_hidden =
      draft_backend_->RequiresTargetHiddenStates();
  std::size_t accepted_count = 0;
  tokenization::TokenId correction_token = 0;
  std::vector<float> correction_logits;

  if (use_batched_verification_) {
    std::vector<tokenization::TokenId> verification_inputs;
    verification_inputs.reserve(num_draft + 1);
    verification_inputs.push_back(current_token);
    verification_inputs.insert(verification_inputs.end(),
                               proposal.tokens.begin(), proposal.tokens.end());

    // Candidate route: verify [current, draft...] in one target prefill batch.
    target_executor_->SaveState(cur_pos);
    auto verification = target_executor_->ForwardVerificationChunk(
        verification_inputs, cur_pos, capture_target_hidden, false);
    if (verification.predictions.size() != verification_inputs.size()) {
      throw std::runtime_error(
          "target executor returned an incomplete verification chunk");
    }
    if (capture_target_hidden &&
        (verification.hidden_width == 0 ||
         verification.hidden_states.size() !=
             verification_inputs.size() * verification.hidden_width)) {
      throw std::runtime_error(
          "target executor returned incomplete verification hidden states");
    }

    while (accepted_count < num_draft &&
           verification.predictions[accepted_count] ==
               proposal.tokens[accepted_count]) {
      ++accepted_count;
    }
    correction_token = verification.predictions[accepted_count];
    if (options_.retain_frontier_logits) {
      const auto logits =
          target_executor_->CopyVerificationLogits(accepted_count);
      correction_logits.assign(logits.begin(), logits.end());
    }

    // Full acceptance leaves the batch committed. A rejection restores the
    // recurrent snapshot and rebuilds only the committed input prefix.
    const std::size_t committed_input_count = accepted_count + 1;
    if (accepted_count != num_draft) {
      target_executor_->RestoreState();
      target_executor_->CommitVerificationChunk(
          std::span<const tokenization::TokenId>(verification_inputs.data(),
                                                 committed_input_count),
          cur_pos);
    }

    if (capture_target_hidden) {
      for (std::size_t row = 0; row < committed_input_count; ++row) {
        draft_backend_->UpdateTargetHidden(
            std::span<const float>(verification.hidden_states.data() +
                                       (row * verification.hidden_width),
                                   verification.hidden_width));
      }
    }
  } else {
    // Reference route: sequential transactional verification.
    std::vector<float> tentative_target_hidden;
    std::vector<float> committed_target_hidden;
    std::size_t target_hidden_width = 0;
    const auto append_target_hidden = [&](std::vector<float>& destination) {
      if (!capture_target_hidden) {
        return;
      }
      const auto hidden = target_executor_->CopyLastHidden();
      if (hidden.empty()) {
        throw std::runtime_error(
            "target executor did not capture a committed hidden state");
      }
      if (target_hidden_width == 0) {
        target_hidden_width = hidden.size();
      } else if (target_hidden_width != hidden.size()) {
        throw std::runtime_error(
            "target hidden-state width changed during verification");
      }
      destination.insert(destination.end(), hidden.begin(), hidden.end());
    };

    target_executor_->SaveState(cur_pos);
    std::vector<tokenization::TokenId> target_predictions;
    target_predictions.reserve(num_draft);

    tokenization::TokenId input_token = current_token;
    std::uint32_t eval_pos = cur_pos;
    for (std::size_t index = 0; index < num_draft; ++index) {
      const auto target_prediction =
          target_executor_->ForwardToken(input_token, eval_pos);
      append_target_hidden(tentative_target_hidden);
      target_predictions.push_back(target_prediction);
      if (target_prediction != proposal.tokens[index]) {
        break;
      }
      input_token = proposal.tokens[index];
      ++eval_pos;
    }

    while (accepted_count < target_predictions.size() &&
           target_predictions[accepted_count] ==
               proposal.tokens[accepted_count]) {
      ++accepted_count;
    }

    if (accepted_count == num_draft) {
      correction_token = target_executor_->ForwardToken(input_token, eval_pos);
      if (options_.retain_frontier_logits) {
        const auto logits = target_executor_->CopyLastLogits();
        correction_logits.assign(logits.begin(), logits.end());
      }
      append_target_hidden(tentative_target_hidden);
      committed_target_hidden = std::move(tentative_target_hidden);
    } else {
      target_executor_->RestoreState();
      tokenization::TokenId replay_input = current_token;
      std::uint32_t replay_pos = cur_pos;
      for (std::size_t index = 0; index < accepted_count; ++index) {
        (void)target_executor_->ForwardToken(replay_input, replay_pos, false);
        append_target_hidden(committed_target_hidden);
        replay_input = proposal.tokens[index];
        ++replay_pos;
      }
      correction_token =
          target_executor_->ForwardToken(replay_input, replay_pos, true);
      if (options_.retain_frontier_logits) {
        const auto logits = target_executor_->CopyLastLogits();
        correction_logits.assign(logits.begin(), logits.end());
      }
      append_target_hidden(committed_target_hidden);
    }

    if (capture_target_hidden) {
      if (target_hidden_width == 0 ||
          committed_target_hidden.size() % target_hidden_width != 0) {
        throw std::runtime_error(
            "committed target hidden-state capture is incomplete");
      }
      for (std::size_t offset = 0; offset < committed_target_hidden.size();
           offset += target_hidden_width) {
        draft_backend_->UpdateTargetHidden(std::span<const float>(
            committed_target_hidden.data() + offset, target_hidden_width));
      }
    }
  }

  StepResult result;
  result.draft_count = num_draft;
  result.accepted_count = accepted_count;

  // Build emitted tokens: accepted draft tokens + correction token
  for (std::size_t i = 0; i < accepted_count; ++i) {
    result.emitted_tokens.push_back(proposal.tokens[i]);
  }
  result.emitted_tokens.push_back(correction_token);
  result.next_token = correction_token;
  result.next_token_logits = std::move(correction_logits);

  // Check EOS in emitted tokens
  for (const auto tok : result.emitted_tokens) {
    if (IsStopToken(tok, eos_id)) {
      result.hit_eos = true;
      break;
    }
  }

  // 6. Update stats and notify backend
  stats_.total_draft_tokens += num_draft;
  stats_.total_accepted_tokens += accepted_count;
  stats_.total_verification_steps += 1;
  stats_.total_emitted_tokens += result.emitted_tokens.size();

  UpdateAdaptiveDraftLength(accepted_count, num_draft);

  draft_backend_->AcceptFeedback(std::span<const tokenization::TokenId>(
                                     proposal.tokens.data(), accepted_count),
                                 correction_token);

  return result;
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifySampledStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id,
    std::uint32_t max_emitted_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (max_emitted_tokens == 0) {
    throw std::invalid_argument(
        "speculative verification must emit at least one token");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("sampled speculation requires RNG state");
  }

  const auto target_only_step = [&]() {
    (void)target_executor_->ForwardToken(current_token, cur_pos);
    const auto logits = target_executor_->CopyLastLogits();
    if (logits.empty()) {
      throw std::runtime_error(
          "target executor did not retain sampled decode logits");
    }
    const auto next = sampling::SampleLogits(logits, temperature, rng_state);
    UpdateDraftTargetHidden();
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    return StepResult{
        .emitted_tokens = {next},
        .accepted_count = 0,
        .draft_count = 0,
        .next_token = next,
        .next_token_logits = {logits.begin(), logits.end()},
        .hit_eos = IsStopToken(next, eos_id),
    };
  };

  const std::uint32_t max_draft_tokens =
      max_emitted_tokens > 1
          ? std::min(current_draft_length_, max_emitted_tokens - 1)
          : 0;
  if (draft_backend_ == nullptr || max_draft_tokens == 0) {
    return target_only_step();
  }

  const auto proposal = draft_backend_->ProposeSampled(
      current_sequence, cur_pos, max_draft_tokens, temperature, rng_state);
  if (proposal.tokens.empty()) {
    return target_only_step();
  }
  const std::size_t num_draft = proposal.tokens.size();
  if (num_draft > max_draft_tokens || proposal.candidates_per_token == 0 ||
      proposal.candidate_ids.size() !=
          num_draft * proposal.candidates_per_token ||
      proposal.candidate_probabilities.size() !=
          proposal.candidate_ids.size()) {
    throw std::runtime_error(
        "draft backend returned a malformed sampled proposal");
  }

  const auto proposal_row = [&](std::size_t row) {
    const std::size_t offset = row * proposal.candidates_per_token;
    return std::pair{
        std::span<const tokenization::TokenId>(
            proposal.candidate_ids.data() + offset,
            proposal.candidates_per_token),
        std::span<const float>(proposal.candidate_probabilities.data() + offset,
                               proposal.candidates_per_token),
    };
  };
  const auto draft_token_probability = [&](std::size_t row) {
    const auto [ids, probabilities] = proposal_row(row);
    double row_sum = 0.0;
    double token_probability = 0.0;
    for (std::size_t index = 0; index < ids.size(); ++index) {
      const float probability = probabilities[index];
      if (!std::isfinite(probability) || probability < 0.0F) {
        throw std::runtime_error(
            "draft backend returned an invalid sampled probability");
      }
      row_sum += probability;
      if (ids[index] == proposal.tokens[row]) {
        token_probability += probability;
      }
    }
    constexpr double probability_tolerance = 1e-3;
    if (std::abs(row_sum - 1.0) > probability_tolerance ||
        !(token_probability > 0.0)) {
      throw std::runtime_error(
          "draft backend returned an unnormalized sampled proposal");
    }
    return token_probability;
  };

  std::vector<tokenization::TokenId> verification_inputs;
  verification_inputs.reserve(num_draft + 1);
  verification_inputs.push_back(current_token);
  verification_inputs.insert(verification_inputs.end(), proposal.tokens.begin(),
                             proposal.tokens.end());

  const bool capture_target_hidden =
      draft_backend_->RequiresTargetHiddenStates();
  target_executor_->SaveState(cur_pos);
  auto verification = target_executor_->ForwardVerificationChunk(
      verification_inputs, cur_pos, capture_target_hidden, true);
  if (verification.vocab_size == 0 ||
      verification.logits.size() !=
          verification_inputs.size() * verification.vocab_size) {
    throw std::runtime_error(
        "target executor returned incomplete verification logits");
  }
  if (capture_target_hidden &&
      (verification.hidden_width == 0 ||
       verification.hidden_states.size() !=
           verification_inputs.size() * verification.hidden_width)) {
    throw std::runtime_error(
        "target executor returned incomplete verification hidden states");
  }

  std::size_t accepted_count = 0;
  tokenization::TokenId correction_token = 0;
  for (; accepted_count < num_draft; ++accepted_count) {
    const auto target_row = std::span<const float>(
        verification.logits.data() + (accepted_count * verification.vocab_size),
        verification.vocab_size);
    const double target_probability = sampling::TokenProbability(
        target_row, temperature, proposal.tokens[accepted_count]);
    const double draft_probability = draft_token_probability(accepted_count);
    if (sampling::Uniform(rng_state) * draft_probability < target_probability) {
      continue;
    }
    const auto [candidate_ids, candidate_probabilities] =
        proposal_row(accepted_count);
    correction_token =
        sampling::SampleResidual(target_row, temperature, candidate_ids,
                                 candidate_probabilities, rng_state);
    break;
  }
  if (accepted_count == num_draft) {
    const auto bonus_row = std::span<const float>(
        verification.logits.data() + (num_draft * verification.vocab_size),
        verification.vocab_size);
    correction_token =
        sampling::SampleLogits(bonus_row, temperature, rng_state);
  }
  const auto correction_row = std::span<const float>(
      verification.logits.data() + (accepted_count * verification.vocab_size),
      verification.vocab_size);

  const std::size_t committed_input_count = accepted_count + 1;
  if (accepted_count != num_draft) {
    target_executor_->RestoreState();
    target_executor_->CommitVerificationChunk(
        std::span<const tokenization::TokenId>(verification_inputs.data(),
                                               committed_input_count),
        cur_pos);
  }
  if (capture_target_hidden) {
    for (std::size_t row = 0; row < committed_input_count; ++row) {
      draft_backend_->UpdateTargetHidden(std::span<const float>(
          verification.hidden_states.data() + (row * verification.hidden_width),
          verification.hidden_width));
    }
  }

  StepResult result;
  result.draft_count = num_draft;
  result.accepted_count = accepted_count;
  result.emitted_tokens.insert(
      result.emitted_tokens.end(), proposal.tokens.begin(),
      proposal.tokens.begin() + static_cast<std::ptrdiff_t>(accepted_count));
  result.emitted_tokens.push_back(correction_token);
  result.next_token = correction_token;
  result.next_token_logits.assign(correction_row.begin(), correction_row.end());
  result.hit_eos = std::any_of(
      result.emitted_tokens.begin(), result.emitted_tokens.end(),
      [&](tokenization::TokenId token) { return IsStopToken(token, eos_id); });

  stats_.total_draft_tokens += num_draft;
  stats_.total_accepted_tokens += accepted_count;
  ++stats_.total_verification_steps;
  stats_.total_emitted_tokens += result.emitted_tokens.size();
  UpdateAdaptiveDraftLength(accepted_count, num_draft);
  draft_backend_->AcceptFeedback(std::span<const tokenization::TokenId>(
                                     proposal.tokens.data(), accepted_count),
                                 correction_token);
  return result;
}

std::size_t SpeculativeVerifierSnapshot::PayloadBytes() const noexcept {
  return (draft_snapshot_ != nullptr ? draft_snapshot_->PayloadBytes() : 0) +
         rolling_acceptance_.size() * sizeof(float) + sizeof(stats_) +
         sizeof(current_draft_length_) + sizeof(accepted_token_ema_);
}

std::unique_ptr<SpeculativeVerifierSnapshot> SpeculativeVerifier::Snapshot()
    const {
  if (draft_backend_ == nullptr) {
    throw std::logic_error(
        "speculative verifier has no draft backend to snapshot");
  }
  auto snapshot = std::unique_ptr<SpeculativeVerifierSnapshot>(
      new SpeculativeVerifierSnapshot());
  snapshot->draft_snapshot_ = draft_backend_->Snapshot();
  snapshot->stats_ = stats_;
  snapshot->current_draft_length_ = current_draft_length_;
  snapshot->rolling_acceptance_ = rolling_acceptance_;
  snapshot->accepted_token_ema_ = accepted_token_ema_;
  return snapshot;
}

void SpeculativeVerifier::RestoreSnapshot(
    const SpeculativeVerifierSnapshot& snapshot) {
  if (draft_backend_ == nullptr || snapshot.draft_snapshot_ == nullptr) {
    throw std::invalid_argument("speculative verifier snapshot is incomplete");
  }
  draft_backend_->RestoreSnapshot(*snapshot.draft_snapshot_);
  stats_ = snapshot.stats_;
  current_draft_length_ = snapshot.current_draft_length_;
  rolling_acceptance_ = snapshot.rolling_acceptance_;
  accepted_token_ema_ = snapshot.accepted_token_ema_;
}

std::vector<tokenization::TokenId> SpeculativeVerifier::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }
  if (!std::isfinite(options.temperature) || options.temperature < 0.0F) {
    throw std::invalid_argument(
        "generation temperature must be finite and nonnegative");
  }

  tokenization::TokenId first_token = Prime(prompt_tokens);
  std::uint64_t rng_state = 0;
  if (options.temperature > 0.0F) {
    if (options.seed >= 0) {
      rng_state = static_cast<std::uint64_t>(options.seed);
    } else {
      std::random_device random_device;
      rng_state = (static_cast<std::uint64_t>(random_device()) << 32U) ^
                  static_cast<std::uint64_t>(random_device());
    }
    const auto logits = target_executor_->CopyLastLogits();
    first_token =
        sampling::SampleLogits(logits, options.temperature, &rng_state);
  }
  const auto eos_id = target_executor_->GetEosTokenId();
  if (options.max_new_tokens == 0 || IsStopToken(first_token, eos_id)) {
    return output_tokens;
  }

  std::vector<tokenization::TokenId> current_sequence(prompt_tokens.begin(),
                                                      prompt_tokens.end());
  current_sequence.push_back(first_token);
  std::uint32_t cur_pos = static_cast<std::uint32_t>(prompt_tokens.size());
  tokenization::TokenId next_token = first_token;

  output_tokens.push_back(first_token);
  if (on_token) {
    const auto piece = target_executor_->DecodeToken(first_token);
    if (!on_token(first_token, piece)) {
      return output_tokens;
    }
  }

  // Speculative decode generation loop.
  while (output_tokens.size() < options.max_new_tokens) {
    StepResult step_res;
    if (options.temperature > 0.0F) {
      const auto remaining = options.max_new_tokens - output_tokens.size();
      step_res =
          VerifyStep(current_sequence, cur_pos, next_token, eos_id,
                     static_cast<std::uint32_t>(std::min<std::size_t>(
                         remaining, std::numeric_limits<std::uint32_t>::max())),
                     options.temperature, &rng_state);
    } else {
      step_res = VerifyStep(current_sequence, cur_pos, next_token, eos_id);
    }

    bool should_stop = false;
    for (const auto tok : step_res.emitted_tokens) {
      if (IsStopToken(tok, eos_id)) {
        should_stop = true;
        break;
      }

      output_tokens.push_back(tok);
      current_sequence.push_back(tok);
      ++cur_pos;

      if (on_token) {
        const auto piece = target_executor_->DecodeToken(tok);
        if (!on_token(tok, piece)) {
          should_stop = true;
          break;
        }
      }

      if (output_tokens.size() >= options.max_new_tokens) {
        should_stop = true;
        break;
      }
    }

    if (should_stop || step_res.hit_eos) {
      break;
    }

    next_token = step_res.next_token;
  }

  return output_tokens;
}

tokenization::TokenId SpeculativeVerifier::Prime(
    std::span<const tokenization::TokenId> prompt_tokens) {
  if (prompt_tokens.empty()) {
    throw std::invalid_argument("speculative prompt must not be empty");
  }
  Reset();
  target_executor_->Reset();

  const bool capture_hidden =
      draft_backend_ != nullptr && draft_backend_->RequiresTargetHiddenStates();
  const auto target_layer_ids = capture_hidden
                                    ? draft_backend_->TargetHiddenLayerIds()
                                    : std::span<const std::uint32_t>{};
  target_executor_->SetPromptHiddenCapture(capture_hidden, target_layer_ids);
  const tokenization::TokenId first_token =
      target_executor_->ForwardPromptBatch(prompt_tokens);
  if (capture_hidden) {
    const auto prompt_hidden = target_executor_->GetPromptHiddenStates();
    if (prompt_hidden.empty() ||
        (prompt_hidden.size() % prompt_tokens.size()) != 0) {
      throw std::runtime_error(
          "target executor did not capture complete prompt hidden states");
    }
    const DraftTargetContext context{
        .prompt_tokens = prompt_tokens,
        .prompt_hidden_states = prompt_hidden,
        .hidden_size = prompt_hidden.size() / prompt_tokens.size(),
        .first_token = first_token,
    };
    if (!draft_backend_->PrimeTargetContext(context)) {
      throw std::runtime_error("draft backend failed to prime target context");
    }
  }
  return first_token;
}

}  // namespace gufo::speculative
#endif  // defined(ENGINE_ENABLE_HIP)
