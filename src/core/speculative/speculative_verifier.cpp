#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace strix::speculative {
namespace {

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

  void SetPromptHiddenCapture(bool enabled) override {
    executor_.SetPromptHiddenCapture(enabled);
  }

  std::span<const float> GetPromptHiddenStates() const noexcept override {
    return executor_.GetPromptHiddenStates();
  }

  std::span<const float> CopyLastHidden() override {
    return executor_.CopyLastHidden();
  }

  std::vector<tokenization::TokenId> ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos) override {
    return executor_.ForwardVerificationChunk(candidate_tokens, start_pos);
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

}  // namespace

SpeculativeVerifier::SpeculativeVerifier(
    hip::QwenGpuExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : owned_target_executor_(
          std::make_unique<QwenGpuSpeculativeTarget>(target_executor)),
      target_executor_(owned_target_executor_.get()),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens) {}

SpeculativeVerifier::SpeculativeVerifier(
    ISpeculativeTargetExecutor& target_executor,
    std::unique_ptr<IDraftBackend> draft_backend, SpeculativeOptions options)
    : target_executor_(&target_executor),
      draft_backend_(std::move(draft_backend)),
      options_(options),
      current_draft_length_(options_.initial_draft_tokens) {}

void SpeculativeVerifier::Reset() noexcept {
  stats_ = {};
  rolling_acceptance_.clear();
  current_draft_length_ = options_.initial_draft_tokens;
  if (draft_backend_ != nullptr) {
    draft_backend_->Reset();
  }
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

SpeculativeVerifier::StepResult SpeculativeVerifier::VerifyStep(
    std::vector<tokenization::TokenId>& current_sequence, std::uint32_t cur_pos,
    tokenization::TokenId current_token, tokenization::TokenId eos_id) {
  if (draft_backend_ == nullptr || current_draft_length_ == 0) {
    const auto next = target_executor_->ForwardToken(current_token, cur_pos);
    UpdateDraftTargetHidden();
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = IsStopToken(next, eos_id);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .hit_eos = hit_eos};
  }

  // 1. Propose draft tokens
  const auto proposal =
      draft_backend_->Propose(current_sequence, cur_pos, current_draft_length_);
  if (proposal.tokens.empty()) {
    const auto next = target_executor_->ForwardToken(current_token, cur_pos);
    UpdateDraftTargetHidden();
    ++stats_.total_verification_steps;
    ++stats_.total_emitted_tokens;
    const bool hit_eos = IsStopToken(next, eos_id);
    return {.emitted_tokens = {next},
            .accepted_count = 0,
            .draft_count = 0,
            .next_token = next,
            .hit_eos = hit_eos};
  }

  const std::size_t num_draft = proposal.tokens.size();

  // 2. Build candidate sequence: [current_token, D_0, ..., D_{num_draft-1}]
  std::vector<tokenization::TokenId> candidates;
  candidates.reserve(num_draft + 1);
  candidates.push_back(current_token);
  for (const auto tok : proposal.tokens) {
    candidates.push_back(tok);
  }

  // 3. Execute batched target forward verification in 1 single GPU prefill pass!
  const auto target_predictions =
      target_executor_->ForwardVerificationChunk(candidates, cur_pos);

  // 4. Determine acceptance prefix
  std::size_t accepted_count = 0;
  while (accepted_count < num_draft &&
         accepted_count < target_predictions.size() &&
         target_predictions[accepted_count] == proposal.tokens[accepted_count]) {
    ++accepted_count;
  }

  // 5. Authoritative correction token (computed in parallel by ForwardVerificationChunk)
  const tokenization::TokenId correction_token =
      (accepted_count < target_predictions.size())
          ? target_predictions[accepted_count]
          : target_executor_->ForwardToken(
                (accepted_count > 0) ? proposal.tokens[accepted_count - 1]
                                     : current_token,
                cur_pos + static_cast<std::uint32_t>(accepted_count));

  StepResult result;
  result.draft_count = num_draft;
  result.accepted_count = accepted_count;

  // Build emitted tokens: accepted draft tokens + correction token
  for (std::size_t i = 0; i < accepted_count; ++i) {
    result.emitted_tokens.push_back(proposal.tokens[i]);
  }
  result.emitted_tokens.push_back(correction_token);
  result.next_token = correction_token;

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
  UpdateDraftTargetHidden();

  return result;
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

  const tokenization::TokenId first_token = Prime(prompt_tokens);
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
    const auto step_res =
        VerifyStep(current_sequence, cur_pos, next_token, eos_id);

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
  target_executor_->SetPromptHiddenCapture(capture_hidden);
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

}  // namespace strix::speculative
#endif  // defined(ENGINE_ENABLE_HIP)
