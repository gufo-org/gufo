#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/hip/dflash.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kDFlashDraftPersistentMagic = {
    'G', 'D', 'F', 'D', 'R', 'F', '0', '1'};
constexpr std::uint32_t kDFlashDraftPersistentVersion = 1;
constexpr std::size_t kDFlashDraftPersistentHeaderBytes = 64;
constexpr std::uint32_t kDFlashDraftPrimedFlag = 1U << 0U;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("DFlash draft persistent header is truncated");
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
    throw std::invalid_argument("DFlash draft persistent header is truncated");
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
    throw std::overflow_error("DFlash draft persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("DFlash draft persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

class QwenDFlashDraftSnapshot final
    : public speculative::IDraftBackendSnapshot {
public:
  QwenDFlashDraftSnapshot(std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot,
                          std::vector<float> pending_target_features,
                          std::size_t target_feature_width,
                          std::uint32_t max_context, bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        pending_target_features(std::move(pending_target_features)),
        target_feature_width(target_feature_width),
        max_context(max_context),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           pending_target_features.size() * sizeof(float);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    if (gpu_snapshot == nullptr ||
        pending_target_features.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument(
          "DFlash draft snapshot persistent metadata is invalid");
    }
    return CheckedPersistentAdd(
        CheckedPersistentAdd(kDFlashDraftPersistentHeaderBytes,
                             gpu_snapshot->PersistentPayloadBytes()),
        pending_target_features.size() * sizeof(std::uint32_t));
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t expected_bytes = PersistentPayloadBytes();
    if (destination.size() != expected_bytes || target_feature_width == 0 ||
        target_feature_width > std::numeric_limits<std::uint32_t>::max() ||
        pending_target_features.size() % target_feature_width != 0) {
      throw std::invalid_argument(
          "DFlash draft persistent destination is invalid");
    }
    const std::size_t gpu_payload_bytes =
        gpu_snapshot->PersistentPayloadBytes();
    const std::size_t pending_bytes =
        pending_target_features.size() * sizeof(std::uint32_t);
    std::fill(destination.begin(), destination.end(), std::uint8_t{0});
    std::copy(kDFlashDraftPersistentMagic.begin(),
              kDFlashDraftPersistentMagic.end(), destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8,
                                   kDFlashDraftPersistentVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kDFlashDraftPersistentHeaderBytes));
    PutLittleEndian<std::uint32_t>(destination, 16,
                                   primed ? kDFlashDraftPrimedFlag : 0U);
    PutLittleEndian<std::uint32_t>(
        destination, 20, static_cast<std::uint32_t>(target_feature_width));
    PutLittleEndian<std::uint64_t>(
        destination, 24,
        static_cast<std::uint64_t>(pending_target_features.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 32, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 40,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint32_t>(destination, 48,
                                   gpu_snapshot->ValidContext());
    PutLittleEndian<std::uint32_t>(destination, 52, max_context);

    const std::size_t gpu_offset = kDFlashDraftPersistentHeaderBytes;
    const std::size_t pending_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t written = gpu_snapshot->SerializePersistent(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "DFlash GPU persistent serializer returned the wrong byte count");
    }
    for (std::size_t index = 0; index < pending_target_features.size();
         ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, pending_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(pending_target_features[index]));
    }
    if (CheckedPersistentAdd(pending_offset, pending_bytes) !=
        destination.size()) {
      throw std::logic_error(
          "DFlash draft persistent serializer size mismatch");
    }
    return destination.size();
  }

  std::unique_ptr<QwenDFlashGpuSnapshot> gpu_snapshot;
  std::vector<float> pending_target_features;
  std::size_t target_feature_width{0};
  std::uint32_t max_context{0};
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

void QwenDFlashGpuDraftBackend::InjectPendingFeatures(
    std::uint32_t current_pos) {
  if (current_pos < executor_->GetInjectedContextLength()) {
    throw std::logic_error(
        "DFlash committed position precedes injected history");
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
}

bool QwenDFlashGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      context.hidden_size != executor_->GetTargetFeaturesSize() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  InjectPendingFeatures(position);
  return executor_->InjectTargetContext(
      context.prompt_hidden_states, position,
      static_cast<std::uint32_t>(context.prompt_tokens.size()));
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

  InjectPendingFeatures(current_pos);

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
  proposal_active_ = !proposal.tokens.empty();
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

std::size_t QwenDFlashGpuDraftBackend::SnapshotPayloadBytes() const {
  const std::size_t gpu_bytes = executor_->SnapshotPayloadBytes();
  if (pending_target_features_.size() >
      (std::numeric_limits<std::size_t>::max() - gpu_bytes) / sizeof(float)) {
    throw std::overflow_error("DFlash snapshot size overflows");
  }
  return gpu_bytes + pending_target_features_.size() * sizeof(float);
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenDFlashGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "DFlash snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenDFlashDraftSnapshot>(
      executor_->SaveSnapshot(), pending_target_features_,
      executor_->GetTargetFeaturesSize(), config_.max_context, primed_);
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

void QwenDFlashGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kDFlashDraftPersistentHeaderBytes ||
      !std::equal(kDFlashDraftPersistentMagic.begin(),
                  kDFlashDraftPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kDFlashDraftPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kDFlashDraftPersistentHeaderBytes ||
      GetLittleEndian<std::uint64_t>(payload, 56) != 0) {
    throw std::invalid_argument("DFlash draft persistent header is invalid");
  }
  const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::size_t feature_width = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::size_t pending_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 24));
  const std::size_t gpu_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::uint32_t injected_context =
      GetLittleEndian<std::uint32_t>(payload, 48);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 52);
  const std::size_t expected_feature_width = executor_->GetTargetFeaturesSize();

  if ((flags & ~kDFlashDraftPrimedFlag) != 0 || feature_width == 0 ||
      feature_width != expected_feature_width ||
      pending_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      pending_count % feature_width != 0 ||
      max_context != config_.max_context ||
      injected_context > config_.max_context ||
      pending_count / feature_width >
          static_cast<std::size_t>(config_.max_context - injected_context) ||
      total_bytes != payload.size()) {
    throw std::invalid_argument(
        "DFlash draft persistent metadata is incompatible");
  }
  const bool primed = (flags & kDFlashDraftPrimedFlag) != 0;
  if (!primed && (injected_context != 0 || pending_count != 0)) {
    throw std::invalid_argument("DFlash draft unprimed payload contains state");
  }
  const std::size_t gpu_offset = kDFlashDraftPersistentHeaderBytes;
  const std::size_t pending_offset =
      CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
  const std::size_t pending_bytes = pending_count * sizeof(std::uint32_t);
  if (CheckedPersistentAdd(pending_offset, pending_bytes) != payload.size()) {
    throw std::invalid_argument(
        "DFlash draft persistent payload size is invalid");
  }

  std::vector<float> pending_target_features;
  pending_target_features.reserve(pending_count);
  for (std::size_t index = 0; index < pending_count; ++index) {
    pending_target_features.push_back(
        std::bit_cast<float>(GetLittleEndian<std::uint32_t>(
            payload, pending_offset + index * sizeof(std::uint32_t))));
  }

  executor_->RestorePersistentSnapshot(
      payload.subspan(gpu_offset, gpu_payload_bytes));
  if (executor_->GetInjectedContextLength() != injected_context) {
    throw std::invalid_argument(
        "DFlash draft persistent context length is inconsistent");
  }
  pending_target_features_ = std::move(pending_target_features);
  proposed_tokens_.clear();
  proposal_checkpoint_ = injected_context + static_cast<std::uint32_t>(
                                                pending_count / feature_width);
  proposal_input_ = 0;
  primed_ = primed;
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
