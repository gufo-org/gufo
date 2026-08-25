#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::hip {

QwenDFlashGpuDraftBackend::QwenDFlashGpuDraftBackend(
    std::unique_ptr<QwenDFlashGpuExecutor> executor,
    QwenDFlashGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_accumulator_(executor_->GetTargetFeaturesSize()) {}

std::unique_ptr<QwenDFlashGpuDraftBackend> QwenDFlashGpuDraftBackend::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model,
    QwenDFlashGpuDraftConfig config, std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor =
      QwenDFlashGpuExecutor::Create(std::move(model), config.max_context, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<QwenDFlashGpuDraftBackend>(
      new QwenDFlashGpuDraftBackend(std::move(executor), config));
}

std::unique_ptr<QwenDFlashGpuDraftBackend>
QwenDFlashGpuDraftBackend::CreateFromGguf(
    std::string_view dflash_model_path,
    std::shared_ptr<const QwenGpuModel> target_model,
    QwenDFlashGpuDraftConfig config, std::string* error_msg) {
  if (dflash_model_path.empty() || target_model == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GGUF path and target GPU model are required";
    }
    return nullptr;
  }
  auto reader_owner =
      core::GgufReader::OpenFile(std::string(dflash_model_path), error_msg);
  if (reader_owner == nullptr) {
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  auto model = QwenDFlashGpuModel::Create(std::move(reader),
                                          std::move(target_model), error_msg);
  if (model == nullptr) {
    return nullptr;
  }
  return Create(std::move(model), config, error_msg);
}

bool QwenDFlashGpuDraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  Reset();
  if (context.prompt_tokens.empty()) {
    last_error_ = "DFlash prompt context is empty";
    return false;
  }

  const std::size_t num_tokens = context.prompt_tokens.size();
  const std::size_t hidden_size = executor_->GetHiddenSize();
  const std::size_t target_layers_count =
      executor_->GetModel().GetDFlashConfig().target_layer_ids.size();
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();

  if (context.prompt_hidden_states.empty()) {
    primed_ = true;
    proposal_input_ = context.first_token;
    return true;
  }

  std::vector<float> expanded_features;
  std::span<const float> features_to_inject;

  if (context.prompt_hidden_states.size() == num_tokens * enc_in_dim) {
    features_to_inject = context.prompt_hidden_states;
  } else if (context.prompt_hidden_states.size() == num_tokens * hidden_size) {
    expanded_features.resize(num_tokens * enc_in_dim);
    for (std::size_t t = 0; t < num_tokens; ++t) {
      const auto src_tok = std::span<const float>(
          context.prompt_hidden_states.data() + (t * hidden_size), hidden_size);
      for (std::size_t l = 0; l < target_layers_count; ++l) {
        std::copy_n(src_tok.data(), hidden_size,
                    expanded_features.data() + (t * enc_in_dim) +
                        (l * hidden_size));
      }
    }
    features_to_inject = expanded_features;
  } else {
    last_error_ = "DFlash target features dimension mismatch";
    return false;
  }

  try {
    const auto ok = executor_->InjectTargetContext(
        features_to_inject, 0, static_cast<std::uint32_t>(num_tokens));
    if (!ok) {
      last_error_ = "DFlash target context injection failed";
      return false;
    }
    proposal_input_ = context.prompt_tokens.back();
    primed_ = true;
    return true;
  } catch (const std::exception& ex) {
    last_error_ = ex.what();
    Reset();
    return false;
  }
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("DFlash GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("DFlash GPU proposal feedback is pending");
  }

  // Inject newly committed target tokens into DFlash draft KV cache
  if (current_pos > executor_->GetInjectedContextLength()) {
    const std::uint32_t start_p = executor_->GetInjectedContextLength();
    for (std::uint32_t p = start_p; p < current_pos; ++p) {
      executor_->InjectTargetContext(target_hidden_accumulator_, p, 1);
    }
  }

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t count = std::min(max_tokens, config_.max_draft_tokens);
  if (count == 0) {
    return proposal;
  }

  proposal_checkpoint_ = current_pos;
  proposal_input_ = prompt_tokens.back();
  proposed_tokens_.clear();

  proposed_tokens_ =
      executor_->ForwardBlock(proposal_input_, current_pos, count);
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

void QwenDFlashGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("DFlash GPU proposal feedback is invalid");
  }

  proposal_active_ = false;
  proposed_tokens_.clear();
}

void QwenDFlashGpuDraftBackend::UpdateTargetHidden(
    std::span<const float> hidden) {
  const std::size_t hidden_size = executor_->GetHiddenSize();
  const std::size_t target_layers_count =
      executor_->GetModel().GetDFlashConfig().target_layer_ids.size();
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();

  if (hidden.size() == enc_in_dim) {
    std::copy_n(hidden.data(), enc_in_dim, target_hidden_accumulator_.data());
  } else if (hidden.size() == hidden_size) {
    for (std::size_t l = 0; l < target_layers_count; ++l) {
      std::copy_n(hidden.data(), hidden_size,
                  target_hidden_accumulator_.data() + (l * hidden_size));
    }
  }
}

void QwenDFlashGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_accumulator_, 0.0F);
  proposed_tokens_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
