#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/mtp.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::hip {

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::Create(
    std::shared_ptr<const QwenMtpGpuModel> model, QwenMtpGpuDraftConfig config,
    std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenMtpGpuExecutor::Create(
      std::move(model), config.max_context, error_msg, config.execution_mode);
  if (executor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<QwenMtpGpuDraftBackend>(
      new QwenMtpGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenMtpGpuDraftBackend> QwenMtpGpuDraftBackend::CreateFromGguf(
    std::string_view model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenMtpGpuDraftConfig config, std::string* error_msg) {
  if (model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "MTP GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenMtpGpuModel::Create(std::move(reader),
                                       std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenMtpGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty() || context.hidden_size == 0 ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    last_error_ = "MTP prompt hidden-state shape is invalid";
    return false;
  }

  try {
    for (std::size_t index = 0; index + 1 < context.prompt_tokens.size();
         ++index) {
      const auto hidden = context.prompt_hidden_states.subspan(
          index * context.hidden_size, context.hidden_size);
      (void)executor_->ForwardTargetHidden(
          context.prompt_tokens[index + 1], hidden,
          static_cast<std::uint32_t>(index), false);
    }
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    executor_->ResetHybridMetrics();
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    Reset();
    return false;
  }
}

speculative::DraftProposal QwenMtpGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("MTP GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("MTP GPU proposal feedback is pending");
  }
  if (executor_->GetNextPosition() + 1 != current_pos) {
    throw std::logic_error("MTP GPU draft position is inconsistent");
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = executor_->GetNextPosition();
  proposal_input_ = prompt_tokens.back();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  auto token = executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                              proposal_checkpoint_, true);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token =
        executor_->ForwardFeedback(token, proposal_checkpoint_ + index, true);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }

  executor_->Rewind(proposal_checkpoint_);
  (void)executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                       proposal_checkpoint_, false);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    (void)executor_->ForwardFeedback(
        accepted[index],
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
