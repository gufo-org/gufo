#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "src/core/sampling.hpp"

namespace gufo::speculative {
namespace {

constexpr std::array<std::uint8_t, 8> kVerifierPersistentMagic = {
    'G', 'S', 'P', 'V', 'E', 'R', '0', '1'};
constexpr std::uint32_t kVerifierPersistentVersion = 2;
constexpr std::size_t kVerifierPersistentHeaderBytes = 96;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error(
        "speculative verifier persistent header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
[[nodiscard]] T GetLittleEndian(std::span<const std::uint8_t> source,
                                std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument(
        "speculative verifier persistent header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

[[nodiscard]] std::size_t CheckedPersistentAdd(std::size_t left,
                                               std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("speculative verifier persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("speculative verifier persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

class QwenGpuSpeculativeTarget final : public ISpeculativeTargetExecutor {
public:
  explicit QwenGpuSpeculativeTarget(hip::QwenGpuExecutor& executor)
      : executor_(executor) {}

  void Reset() noexcept override { executor_.Reset(); }

  tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) override {
    return executor_.ForwardPromptBatch(prompt_tokens);
  }
  tokenization::TokenId ForwardPromptSuffix(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits) override {
    return executor_.ForwardPromptBatch(tokens, position, compute_logits);
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

  bool SupportsDeviceResidentSampling() const noexcept override { return true; }

  tokenization::TokenId SampleLastLogits(
      sampling::SamplerState& sampler) override {
    return executor_.SampleLastLogits(sampler);
  }

  tokenization::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler) override {
    return executor_.SampleVerificationLogits(row, sampler);
  }

  SampledVerificationResult VerifySampledToken(
      std::size_t row, tokenization::TokenId draft_token,
      std::span<const tokenization::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability,
      sampling::SamplerState& sampler) override {
    const auto result = executor_.VerifySampledToken(
        row, draft_token, draft_candidate_ids, draft_candidate_probabilities,
        draft_token_probability, sampler);
    return {.token = result.token, .accepted = result.accepted};
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

  std::size_t VocabularySize() const noexcept override {
    return executor_.GetConfig().vocab_size;
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
      use_batched_verification_(options_.use_batched_verification) {
  ConfigureAdaptiveDraftPolicy();
}

SpeculativeVerifier::SpeculativeVerifier(
    ISpeculativeTargetExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : target_executor_(&target_executor),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens),
      use_batched_verification_(options_.use_batched_verification) {
  ConfigureAdaptiveDraftPolicy();
}

void SpeculativeVerifier::Reset() noexcept {
  stats_ = {};
  ResetAdaptiveDraftLength();
  if (draft_backend_ != nullptr) {
    draft_backend_->Reset();
  }
}

void SpeculativeVerifier::BeginRequest() noexcept {
  stats_ = {};
  ResetAdaptiveDraftLength();
  if (draft_backend_ != nullptr)
    draft_backend_->BeginRequest();
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
  rolling_acceptance_.clear();
  current_draft_length_ = options_.initial_draft_tokens;
}

void SpeculativeVerifier::UpdateAdaptiveDraftLength(std::size_t accepted,
                                                    std::size_t drafted) {
  if (!options_.enable_adaptive_draft_length || drafted == 0) {
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

tokenization::TokenId SpeculativeVerifier::AdvanceCommittedToken(
    tokenization::TokenId token, std::uint32_t position) {
  const auto next = target_executor_->ForwardToken(token, position);
  if (draft_backend_ != nullptr &&
      draft_backend_->RequiresTargetHiddenStates()) {
    const auto hidden = target_executor_->CopyLastHidden();
    if (hidden.empty() || !draft_backend_->AppendTargetContext(
                              {.prompt_tokens = std::span(&token, 1),
                               .prompt_hidden_states = hidden,
                               .hidden_size = hidden.size(),
                               .first_token = next},
                              position)) {
      throw std::runtime_error("draft failed to append committed target token");
    }
  }
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
  if (temperature > 0.0F && rng_state == nullptr) {
    throw std::invalid_argument("sampled speculation requires RNG state");
  }
  sampling::SamplingConfig config;
  config.temperature = temperature;
  config.seed = 0;
  sampling::SamplerState sampler(config, current_sequence);
  if (rng_state != nullptr) {
    sampler.SetRngState(*rng_state);
  }
  auto result = VerifyStep(current_sequence, cur_pos, current_token, eos_id,
                           max_emitted_tokens, sampler);
  if (rng_state != nullptr) {
    *rng_state = sampler.rng_state();
  }
  return result;
}

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id,
    std::uint32_t max_emitted_tokens, sampling::SamplerState& sampler) {
  sampler.config().Validate();
  if (sampler.config().uses_random_sampling() ||
      sampler.config().penalties_enabled()) {
    return VerifySampledStep(current_sequence, cur_pos, current_token, eos_id,
                             max_emitted_tokens, sampler);
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
    const auto next = AdvanceCommittedToken(current_token, cur_pos);
    std::vector<float> next_logits;
    if (options_.retain_frontier_logits) {
      const auto logits = target_executor_->CopyLastLogits();
      next_logits.assign(logits.begin(), logits.end());
    }
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
  std::optional<speculative::DraftProposal> proposal_holder;
  {
    proposal_holder =
        draft_backend_->Propose(current_sequence, cur_pos, max_draft_tokens);
  }
  const auto proposal = std::move(*proposal_holder);
  if (proposal.tokens.empty()) {
    const auto next = AdvanceCommittedToken(current_token, cur_pos);
    std::vector<float> next_logits;
    if (options_.retain_frontier_logits) {
      const auto logits = target_executor_->CopyLastLogits();
      next_logits.assign(logits.begin(), logits.end());
    }
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
    {
      target_executor_->SaveState(cur_pos);
    }
    VerificationChunkResult verification;
    {
      verification = target_executor_->ForwardVerificationChunk(
          verification_inputs, cur_pos, capture_target_hidden, false);
    }
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
           !IsStopToken(proposal.tokens[accepted_count], eos_id) &&
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
      if (target_prediction != proposal.tokens[index] ||
          IsStopToken(target_prediction, eos_id)) {
        break;
      }
      input_token = proposal.tokens[index];
      ++eval_pos;
    }

    while (accepted_count < target_predictions.size() &&
           !IsStopToken(proposal.tokens[accepted_count], eos_id) &&
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
    std::uint32_t max_emitted_tokens, sampling::SamplerState& sampler) {
  if (max_emitted_tokens == 0) {
    throw std::invalid_argument(
        "speculative verification must emit at least one token");
  }
  sampling::SamplerState working_sampler = sampler;

  const auto target_only_step = [&]() {
    (void)AdvanceCommittedToken(current_token, cur_pos);
    const auto next = target_executor_->SampleLastLogits(working_sampler);
    std::vector<float> next_logits;
    if (options_.retain_frontier_logits) {
      const auto logits = target_executor_->CopyLastLogits();
      next_logits.assign(logits.begin(), logits.end());
    }
    sampler.SetRngState(working_sampler.rng_state());
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    return StepResult{
        .emitted_tokens = {next},
        .accepted_count = 0,
        .draft_count = 0,
        .next_token = next,
        .next_token_logits = std::move(next_logits),
        .hit_eos = IsStopToken(next, eos_id),
    };
  };

  const bool random_sampling = sampler.config().uses_random_sampling();
  const std::uint32_t max_draft_tokens =
      max_emitted_tokens > 1 && draft_backend_ != nullptr &&
              draft_backend_->SupportsSampledProposals()
          ? std::min(current_draft_length_, max_emitted_tokens - 1)
          : 0;
  if (draft_backend_ == nullptr || max_draft_tokens == 0) {
    return target_only_step();
  }

  const auto proposal = random_sampling
                            ? draft_backend_->ProposeSampled(
                                  current_sequence, cur_pos, max_draft_tokens,
                                  sampler.config().temperature,
                                  working_sampler.mutable_rng_state())
                            : draft_backend_->Propose(current_sequence, cur_pos,
                                                      max_draft_tokens);
  if (proposal.tokens.empty()) {
    return target_only_step();
  }
  const std::size_t num_draft = proposal.tokens.size();
  if (num_draft > max_draft_tokens || proposal.start_pos != cur_pos) {
    throw std::runtime_error("draft backend returned a malformed proposal");
  }
  if (random_sampling && (proposal.candidates_per_token == 0 ||
                          proposal.candidates_per_token >
                              proposal.candidate_ids.size() / num_draft ||
                          proposal.candidate_ids.size() !=
                              num_draft * proposal.candidates_per_token ||
                          proposal.candidate_probabilities.size() !=
                              proposal.candidate_ids.size())) {
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
    const auto vocab_size = target_executor_->VocabularySize();
    double row_sum = 0.0;
    double token_probability = 0.0;
    for (std::size_t index = 0; index < ids.size(); ++index) {
      const float probability = probabilities[index];
      if (!std::isfinite(probability) || probability < 0.0F ||
          probability > 1.0F || (vocab_size != 0 && ids[index] >= vocab_size) ||
          std::find(ids.begin(), ids.begin() + index, ids[index]) !=
              ids.begin() + index) {
        throw std::runtime_error(
            "draft backend returned an invalid sampled candidate");
      }
      row_sum += probability;
      if (ids[index] == proposal.tokens[row]) {
        token_probability += probability;
      }
    }
    constexpr double probability_tolerance = 1e-5;
    if (std::abs(row_sum - 1.0) > probability_tolerance ||
        !(token_probability > 0.0)) {
      throw std::runtime_error(
          "draft backend returned an unnormalized sampled proposal");
    }
    return token_probability;
  };
  // Validate every row before advancing target state, including rows that
  // would otherwise go unchecked after an early rejection.
  std::vector<double> token_probabilities(num_draft);
  if (random_sampling) {
    for (std::size_t row = 0; row < num_draft; ++row)
      token_probabilities[row] = draft_token_probability(row);
  }

  std::vector<tokenization::TokenId> verification_inputs;
  verification_inputs.reserve(num_draft + 1);
  verification_inputs.push_back(current_token);
  verification_inputs.insert(verification_inputs.end(), proposal.tokens.begin(),
                             proposal.tokens.end());

  const bool capture_target_hidden =
      draft_backend_->RequiresTargetHiddenStates();
  const bool device_resident_sampling =
      target_executor_->SupportsDeviceResidentSampling();
  target_executor_->SaveState(cur_pos);
  auto verification = target_executor_->ForwardVerificationChunk(
      verification_inputs, cur_pos, capture_target_hidden,
      !device_resident_sampling);
  if (!device_resident_sampling &&
      (verification.vocab_size == 0 ||
       verification.logits.size() !=
           verification_inputs.size() * verification.vocab_size)) {
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
    if (!random_sampling) {
      // Penalties change the target argmax after each committed token. Draft
      // proposals remain useful: verify against that same evolving AR history,
      // without drawing random numbers or constructing a proposal distribution.
      const auto target_token =
          device_resident_sampling
              ? target_executor_->SampleVerificationLogits(accepted_count,
                                                           working_sampler)
              : working_sampler.Sample(std::span<const float>(
                    verification.logits.data() +
                        accepted_count * verification.vocab_size,
                    verification.vocab_size));
      if (target_token != proposal.tokens[accepted_count] ||
          IsStopToken(target_token, eos_id)) {
        correction_token = target_token;
        break;
      }
      working_sampler.Accept(target_token);
      continue;
    }
    const double draft_probability = token_probabilities[accepted_count];
    const auto [candidate_ids, candidate_probabilities] =
        proposal_row(accepted_count);
    if (device_resident_sampling) {
      const auto decision = target_executor_->VerifySampledToken(
          accepted_count, proposal.tokens[accepted_count], candidate_ids,
          candidate_probabilities, draft_probability, working_sampler);
      if (decision.accepted) {
        if (IsStopToken(proposal.tokens[accepted_count], eos_id)) {
          correction_token = proposal.tokens[accepted_count];
          break;
        }
        continue;
      }
      correction_token = decision.token;
      break;
    }

    const auto target_row = std::span<const float>(
        verification.logits.data() + (accepted_count * verification.vocab_size),
        verification.vocab_size);
    const auto target_distribution = working_sampler.Distribution(target_row);
    const double target_probability =
        target_distribution.probability(proposal.tokens[accepted_count]);
    if (working_sampler.Uniform() * draft_probability < target_probability) {
      working_sampler.Accept(proposal.tokens[accepted_count]);
      if (IsStopToken(proposal.tokens[accepted_count], eos_id)) {
        correction_token = proposal.tokens[accepted_count];
        break;
      }
      continue;
    }
    correction_token = target_distribution.SampleResidual(
        candidate_ids, candidate_probabilities,
        working_sampler.mutable_rng_state());
    break;
  }
  if (accepted_count == num_draft) {
    if (device_resident_sampling) {
      correction_token = target_executor_->SampleVerificationLogits(
          num_draft, working_sampler);
    } else {
      const auto bonus_row = std::span<const float>(
          verification.logits.data() + (num_draft * verification.vocab_size),
          verification.vocab_size);
      correction_token = working_sampler.Sample(bonus_row);
    }
  }

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
  if (options_.retain_frontier_logits) {
    if (device_resident_sampling) {
      const auto correction_row =
          target_executor_->CopyVerificationLogits(accepted_count);
      result.next_token_logits.assign(correction_row.begin(),
                                      correction_row.end());
    } else {
      const auto correction_row =
          std::span<const float>(verification.logits.data() +
                                     (accepted_count * verification.vocab_size),
                                 verification.vocab_size);
      result.next_token_logits.assign(correction_row.begin(),
                                      correction_row.end());
    }
  }
  result.hit_eos = std::any_of(
      result.emitted_tokens.begin(), result.emitted_tokens.end(),
      [&](tokenization::TokenId token) { return IsStopToken(token, eos_id); });

  stats_.total_draft_tokens += num_draft;
  stats_.total_accepted_tokens += accepted_count;
  ++stats_.total_verification_steps;
  stats_.total_emitted_tokens += result.emitted_tokens.size();
  UpdateAdaptiveDraftLength(accepted_count, num_draft);
  sampler.SetRngState(working_sampler.rng_state());
  draft_backend_->AcceptFeedback(std::span<const tokenization::TokenId>(
                                     proposal.tokens.data(), accepted_count),
                                 correction_token);
  return result;
}

std::size_t SpeculativeVerifierSnapshot::PayloadBytes() const noexcept {
  return (draft_snapshot_ != nullptr ? draft_snapshot_->PayloadBytes() : 0) +
         rolling_acceptance_.size() * sizeof(float) + sizeof(stats_) +
         sizeof(current_draft_length_);
}

std::size_t SpeculativeVerifierSnapshot::PersistentPayloadBytes() const {
  if (draft_snapshot_ == nullptr) {
    throw std::logic_error(
        "speculative verifier snapshot has no draft payload");
  }
  if (rolling_acceptance_.size() >
      std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
    throw std::overflow_error("speculative verifier persistent size overflows");
  }
  const std::size_t rolling_bytes =
      rolling_acceptance_.size() * sizeof(std::uint32_t);
  return CheckedPersistentAdd(
      CheckedPersistentAdd(kVerifierPersistentHeaderBytes, rolling_bytes),
      draft_snapshot_->PersistentPayloadBytes());
}

std::size_t SpeculativeVerifierSnapshot::SerializePersistent(
    std::span<std::uint8_t> destination) const {
  const std::size_t expected_bytes = PersistentPayloadBytes();
  if (destination.size() != expected_bytes) {
    throw std::invalid_argument(
        "speculative verifier persistent destination size is invalid");
  }
  const std::size_t draft_payload_bytes =
      draft_snapshot_->PersistentPayloadBytes();
  const std::size_t rolling_bytes =
      rolling_acceptance_.size() * sizeof(std::uint32_t);
  const std::size_t draft_offset =
      CheckedPersistentAdd(kVerifierPersistentHeaderBytes, rolling_bytes);

  std::fill(destination.begin(), destination.end(), std::uint8_t{0});
  std::copy(kVerifierPersistentMagic.begin(), kVerifierPersistentMagic.end(),
            destination.begin());
  PutLittleEndian<std::uint32_t>(destination, 8, kVerifierPersistentVersion);
  PutLittleEndian<std::uint32_t>(
      destination, 12,
      static_cast<std::uint32_t>(kVerifierPersistentHeaderBytes));
  PutLittleEndian<std::uint64_t>(
      destination, 16, static_cast<std::uint64_t>(draft_payload_bytes));
  PutLittleEndian<std::uint64_t>(
      destination, 24, static_cast<std::uint64_t>(rolling_acceptance_.size()));
  PutLittleEndian<std::uint64_t>(
      destination, 32, static_cast<std::uint64_t>(stats_.total_draft_tokens));
  PutLittleEndian<std::uint64_t>(
      destination, 40,
      static_cast<std::uint64_t>(stats_.total_accepted_tokens));
  PutLittleEndian<std::uint64_t>(
      destination, 48,
      static_cast<std::uint64_t>(stats_.total_verification_steps));
  PutLittleEndian<std::uint64_t>(
      destination, 56, static_cast<std::uint64_t>(stats_.total_emitted_tokens));
  PutLittleEndian<std::uint32_t>(destination, 64, current_draft_length_);
  PutLittleEndian<std::uint64_t>(destination, 72,
                                 static_cast<std::uint64_t>(expected_bytes));

  std::size_t rolling_offset = kVerifierPersistentHeaderBytes;
  for (const float rate : rolling_acceptance_) {
    PutLittleEndian<std::uint32_t>(destination, rolling_offset,
                                   std::bit_cast<std::uint32_t>(rate));
    rolling_offset += sizeof(std::uint32_t);
  }
  const std::size_t written = draft_snapshot_->SerializePersistent(
      destination.subspan(draft_offset, draft_payload_bytes));
  if (written != draft_payload_bytes) {
    throw std::runtime_error(
        "draft persistent serializer returned the wrong byte count");
  }
  return destination.size();
}

std::size_t SpeculativeVerifier::SnapshotPayloadBytes() const {
  if (draft_backend_ == nullptr) {
    throw std::logic_error("speculative verifier has no draft backend to size");
  }
  std::size_t bytes = draft_backend_->SnapshotPayloadBytes();
  const auto checked_add = [&bytes](std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - bytes) {
      throw std::overflow_error("speculative snapshot size overflows");
    }
    bytes += value;
  };
  if (rolling_acceptance_.size() >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("speculative snapshot size overflows");
  }
  checked_add(rolling_acceptance_.size() * sizeof(float));
  checked_add(sizeof(stats_));
  checked_add(sizeof(current_draft_length_));
  return bytes;
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
  return snapshot;
}

void SpeculativeVerifier::RestoreSnapshot(
    const SpeculativeVerifierSnapshot& snapshot) {
  if (draft_backend_ == nullptr || snapshot.draft_snapshot_ == nullptr) {
    throw std::invalid_argument("speculative verifier snapshot is incomplete");
  }
  const bool capture_hidden = draft_backend_->RequiresTargetHiddenStates();
  const auto target_layer_ids = capture_hidden
                                    ? draft_backend_->TargetHiddenLayerIds()
                                    : std::span<const std::uint32_t>{};
  target_executor_->SetPromptHiddenCapture(capture_hidden, target_layer_ids);
  draft_backend_->RestoreSnapshot(*snapshot.draft_snapshot_);
  stats_ = snapshot.stats_;
  current_draft_length_ = snapshot.current_draft_length_;
  rolling_acceptance_ = snapshot.rolling_acceptance_;
}

void SpeculativeVerifier::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (draft_backend_ == nullptr) {
    throw std::logic_error(
        "speculative verifier has no draft backend to restore");
  }
  if (payload.size() < kVerifierPersistentHeaderBytes ||
      !std::equal(kVerifierPersistentMagic.begin(),
                  kVerifierPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kVerifierPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kVerifierPersistentHeaderBytes ||
      GetLittleEndian<std::uint32_t>(payload, 68) != 0 ||
      GetLittleEndian<std::uint64_t>(payload, 80) != 0 ||
      GetLittleEndian<std::uint64_t>(payload, 88) != 0) {
    throw std::invalid_argument(
        "speculative verifier persistent header is invalid");
  }

  const std::size_t draft_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 16));
  const std::size_t rolling_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 24));
  const auto total_draft_tokens = GetLittleEndian<std::uint64_t>(payload, 32);
  const auto total_accepted_tokens =
      GetLittleEndian<std::uint64_t>(payload, 40);
  const auto total_verification_steps =
      GetLittleEndian<std::uint64_t>(payload, 48);
  const auto total_emitted_tokens = GetLittleEndian<std::uint64_t>(payload, 56);
  const std::uint32_t current_draft_length =
      GetLittleEndian<std::uint32_t>(payload, 64);
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 72));

  if (rolling_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      rolling_count > options_.rolling_window ||
      total_accepted_tokens > total_draft_tokens ||
      total_draft_tokens > std::numeric_limits<std::size_t>::max() ||
      total_accepted_tokens > std::numeric_limits<std::size_t>::max() ||
      total_verification_steps > std::numeric_limits<std::size_t>::max() ||
      total_emitted_tokens > std::numeric_limits<std::size_t>::max() ||
      current_draft_length < options_.min_draft_tokens ||
      current_draft_length > options_.max_draft_tokens ||
      total_bytes != payload.size()) {
    throw std::invalid_argument(
        "speculative verifier persistent metadata is invalid");
  }

  const std::size_t rolling_bytes = rolling_count * sizeof(std::uint32_t);
  const std::size_t draft_offset =
      CheckedPersistentAdd(kVerifierPersistentHeaderBytes, rolling_bytes);
  if (CheckedPersistentAdd(draft_offset, draft_payload_bytes) !=
      payload.size()) {
    throw std::invalid_argument(
        "speculative verifier persistent payload size is invalid");
  }

  std::deque<float> rolling_acceptance;
  std::size_t rolling_offset = kVerifierPersistentHeaderBytes;
  for (std::size_t index = 0; index < rolling_count; ++index) {
    const float rate = std::bit_cast<float>(
        GetLittleEndian<std::uint32_t>(payload, rolling_offset));
    if (!std::isfinite(rate) || rate < 0.0F || rate > 1.0F) {
      throw std::invalid_argument(
          "speculative verifier rolling acceptance is invalid");
    }
    rolling_acceptance.push_back(rate);
    rolling_offset += sizeof(std::uint32_t);
  }

  const bool capture_hidden = draft_backend_->RequiresTargetHiddenStates();
  const auto target_layer_ids = capture_hidden
                                    ? draft_backend_->TargetHiddenLayerIds()
                                    : std::span<const std::uint32_t>{};
  target_executor_->SetPromptHiddenCapture(capture_hidden, target_layer_ids);
  draft_backend_->RestorePersistentSnapshot(
      payload.subspan(draft_offset, draft_payload_bytes));
  stats_ = {
      .total_draft_tokens = static_cast<std::size_t>(total_draft_tokens),
      .total_accepted_tokens = static_cast<std::size_t>(total_accepted_tokens),
      .total_verification_steps =
          static_cast<std::size_t>(total_verification_steps),
      .total_emitted_tokens = static_cast<std::size_t>(total_emitted_tokens),
  };
  current_draft_length_ = current_draft_length;
  rolling_acceptance_ = std::move(rolling_acceptance);
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
  options.sampling.Validate();

  tokenization::TokenId first_token = Prime(prompt_tokens);
  sampling::SamplerState sampler(options.sampling, prompt_tokens);
  if (!options.sampling.can_use_unmodified_argmax()) {
    first_token = target_executor_->SampleLastLogits(sampler);
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
  sampler.Accept(first_token);
  if (on_token) {
    const auto piece = target_executor_->DecodeToken(first_token);
    if (!on_token(first_token, piece)) {
      return output_tokens;
    }
  }

  // Speculative decode generation loop.
  while (output_tokens.size() < options.max_new_tokens) {
    const auto remaining = options.max_new_tokens - output_tokens.size();
    const auto verification_budget =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            remaining, std::numeric_limits<std::uint32_t>::max()));
    StepResult step_res = VerifyStep(current_sequence, cur_pos, next_token,
                                     eos_id, verification_budget, sampler);

    bool should_stop = false;
    for (const auto tok : step_res.emitted_tokens) {
      if (IsStopToken(tok, eos_id)) {
        should_stop = true;
        break;
      }

      output_tokens.push_back(tok);
      sampler.Accept(tok);
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

tokenization::TokenId SpeculativeVerifier::ExtendPrompt(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position,
    bool compute_logits) {
  if (tokens.empty() || position == 0) {
    throw std::invalid_argument(
        "prompt extension needs a nonempty retained prefix");
  }
  const bool capture =
      draft_backend_ != nullptr && draft_backend_->RequiresTargetHiddenStates();
  target_executor_->SetPromptHiddenCapture(
      capture, capture ? draft_backend_->TargetHiddenLayerIds()
                       : std::span<const std::uint32_t>{});
  const auto next =
      target_executor_->ForwardPromptSuffix(tokens, position, compute_logits);
  if (capture) {
    const auto hidden = target_executor_->GetPromptHiddenStates();
    if (hidden.empty() || hidden.size() % tokens.size() != 0 ||
        !draft_backend_->AppendTargetContext(
            {.prompt_tokens = tokens,
             .prompt_hidden_states = hidden,
             .hidden_size = hidden.size() / tokens.size(),
             .first_token = next},
            position)) {
      throw std::runtime_error("draft failed to append target prompt features");
    }
  }
  return next;
}

tokenization::TokenId SpeculativeVerifier::Prime(
    std::span<const tokenization::TokenId> prompt_tokens, bool compute_logits) {
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
      target_executor_->ForwardPromptSuffix(prompt_tokens, 0, compute_logits);
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
