#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/dflash.hpp"

namespace gufo::hip {
namespace {

class QwenDFlashDraftSnapshot final
    : public speculative::IDraftBackendSnapshot {
public:
  QwenDFlashDraftSnapshot(std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot,
                          std::vector<float> pending_target_features,
                          bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        pending_target_features(std::move(pending_target_features)),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           pending_target_features.size() * sizeof(float);
  }

  std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot;
  std::vector<float> pending_target_features;
  bool primed{false};
};

}  // namespace

QwenDFlashGpuDraftBackend::QwenDFlashGpuDraftBackend(
    std::unique_ptr<QwenDFlashGpuExecutor> executor,
    QwenDFlashGpuDraftConfig config)
    : executor_(std::move(executor)), config_(config) {}

std::unique_ptr<QwenDFlashGpuDraftBackend> QwenDFlashGpuDraftBackend::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model,
    QwenDFlashGpuDraftConfig config, std::string* error_msg) {
  if (model == nullptr || config.max_draft_tokens == 0) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU draft configuration is invalid";
    }
    return nullptr;
  }
  auto executor = QwenDFlashGpuExecutor::Create(std::move(model),
                                                config.max_context, error_msg);
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
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();

  if (context.hidden_size != enc_in_dim ||
      context.prompt_hidden_states.size() != num_tokens * enc_in_dim) {
    last_error_ = "DFlash target features dimension mismatch";
    return false;
  }

  try {
    const auto ok =
        executor_->InjectTargetContext(context.prompt_hidden_states, 0,
                                       static_cast<std::uint32_t>(num_tokens));
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
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, 0.0F, nullptr);
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "DFlash sampled proposal temperature must be finite and positive");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("DFlash sampled proposal requires RNG state");
  }
  return ProposeImpl(prompt_tokens, current_pos, max_tokens, temperature,
                     rng_state);
}

speculative::DraftProposal QwenDFlashGpuDraftBackend::ProposeImpl(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("DFlash GPU draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("DFlash GPU proposal feedback is pending");
  }

  // Inject newly committed target tokens into DFlash draft KV cache
  if (current_pos > executor_->GetInjectedContextLength()) {
    const std::uint32_t start_p = executor_->GetInjectedContextLength();
    const std::uint32_t count = current_pos - start_p;
    const std::size_t expected =
        static_cast<std::size_t>(count) * executor_->GetTargetFeaturesSize();
    if (pending_target_features_.size() != expected) {
      throw std::logic_error(
          "DFlash committed target feature history is incomplete");
    }
    if (!executor_->InjectTargetContext(pending_target_features_, start_p,
                                        count)) {
      throw std::runtime_error(
          "DFlash committed target feature injection failed");
    }
    pending_target_features_.clear();
  } else if (!pending_target_features_.empty()) {
    throw std::logic_error(
        "DFlash has target features without a matching committed position");
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

  std::vector<float> sample_uniforms;
  if (temperature > 0.0F) {
    sample_uniforms.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      sample_uniforms.push_back(
          static_cast<float>(sampling::Uniform(rng_state)));
    }
    proposal.candidates_per_token =
        executor_->GetModel().GetDFlashConfig().selector_top_k;
  }
  proposed_tokens_ = executor_->ForwardBlock(
      proposal_input_, current_pos, count, temperature, sample_uniforms,
      nullptr, temperature > 0.0F ? &proposal.candidate_ids : nullptr,
      temperature > 0.0F ? &proposal.candidate_probabilities : nullptr);
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
  const std::size_t enc_in_dim = executor_->GetTargetFeaturesSize();
  if (hidden.size() != enc_in_dim) {
    throw std::invalid_argument(
        "DFlash committed target feature width is invalid");
  }
  pending_target_features_.insert(pending_target_features_.end(),
                                  hidden.begin(), hidden.end());
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenDFlashGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "DFlash snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenDFlashDraftSnapshot>(
      executor_->SaveSnapshot(), pending_target_features_, primed_);
}

void QwenDFlashGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* dflash_snapshot =
      dynamic_cast<const QwenDFlashDraftSnapshot*>(&snapshot);
  if (dflash_snapshot == nullptr || dflash_snapshot->gpu_snapshot == nullptr) {
    throw std::invalid_argument(
        "DFlash draft snapshot is incompatible with the backend");
  }
  const std::size_t feature_width = executor_->GetTargetFeaturesSize();
  if (feature_width == 0 ||
      dflash_snapshot->pending_target_features.size() % feature_width != 0) {
    throw std::invalid_argument(
        "DFlash draft snapshot has malformed pending target features");
  }
  executor_->RestoreSnapshot(*dflash_snapshot->gpu_snapshot);
  pending_target_features_ = dflash_snapshot->pending_target_features;
  proposed_tokens_.clear();
  proposal_checkpoint_ = executor_->GetInjectedContextLength() +
                         static_cast<std::uint32_t>(
                             pending_target_features_.size() / feature_width);
  proposal_input_ = 0;
  primed_ = dflash_snapshot->primed;
  proposal_active_ = false;
  last_error_.clear();
}

void QwenDFlashGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  pending_target_features_.clear();
  proposed_tokens_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
