#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/models/qwen/hip/mtp.hpp"

namespace gufo::hip {
namespace {

constexpr std::array<std::uint8_t, 8> kMtpDraftPersistentMagic = {
    'G', 'M', 'T', 'D', 'R', 'F', '0', '1'};
constexpr std::uint32_t kMtpDraftPersistentVersion = 1;
constexpr std::size_t kMtpDraftPersistentHeaderBytes = 64;
constexpr std::uint32_t kMtpDraftPrimedFlag = 1U << 0U;

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("MTP draft persistent header is truncated");
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
    throw std::invalid_argument("MTP draft persistent header is truncated");
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
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return left + right;
}

[[nodiscard]] std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("MTP draft persistent size overflows");
  }
  return static_cast<std::size_t>(value);
}

/// Committed-boundary MTP draft state: the packed attention cache plus the last
/// committed target hidden row that seeds the next proposal.
class QwenMtpDraftSnapshot final : public speculative::IDraftBackendSnapshot {
public:
  QwenMtpDraftSnapshot(std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot,
                       std::vector<float> target_hidden,
                       QwenMtpGpuDraftConfig config, bool primed)
      : gpu_snapshot(std::move(gpu_snapshot)),
        target_hidden(std::move(target_hidden)),
        config(config),
        primed(primed) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (gpu_snapshot != nullptr ? gpu_snapshot->PayloadBytes() : 0) +
           target_hidden.size() * sizeof(float) + sizeof(std::uint32_t);
  }

  [[nodiscard]] std::size_t PersistentPayloadBytes() const override {
    if (gpu_snapshot == nullptr ||
        target_hidden.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::invalid_argument(
          "MTP draft snapshot persistent metadata is invalid");
    }
    return CheckedPersistentAdd(
        CheckedPersistentAdd(kMtpDraftPersistentHeaderBytes,
                             gpu_snapshot->PersistentPayloadBytes()),
        target_hidden.size() * sizeof(std::uint32_t));
  }

  [[nodiscard]] std::size_t SerializePersistent(
      std::span<std::uint8_t> destination) const override {
    const std::size_t expected_bytes = PersistentPayloadBytes();
    if (destination.size() != expected_bytes ||
        target_hidden.size() > std::numeric_limits<std::uint32_t>::max() ||
        primed == target_hidden.empty() ||
        (!primed && gpu_snapshot->ValidContext() != 0)) {
      throw std::invalid_argument(
          "MTP draft persistent destination is invalid");
    }
    const std::size_t gpu_payload_bytes =
        gpu_snapshot->PersistentPayloadBytes();
    std::fill(destination.begin(), destination.end(), std::uint8_t{0});
    std::copy(kMtpDraftPersistentMagic.begin(), kMtpDraftPersistentMagic.end(),
              destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8, kMtpDraftPersistentVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kMtpDraftPersistentHeaderBytes));
    PutLittleEndian<std::uint32_t>(destination, 16,
                                   primed ? kMtpDraftPrimedFlag : 0U);
    PutLittleEndian<std::uint32_t>(
        destination, 20, static_cast<std::uint32_t>(target_hidden.size()));
    PutLittleEndian<std::uint32_t>(destination, 24,
                                   gpu_snapshot->ValidContext());
    PutLittleEndian<std::uint32_t>(destination, 28, config.max_context);
    PutLittleEndian<std::uint32_t>(destination, 32, config.max_draft_tokens);
    PutLittleEndian<std::uint64_t>(
        destination, 40, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 48,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint64_t>(
        destination, 56, static_cast<std::uint64_t>(target_hidden.size()));

    const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
    const std::size_t hidden_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t written = gpu_snapshot->SerializePersistent(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "MTP GPU persistent serializer returned the wrong byte count");
    }
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      PutLittleEndian<std::uint32_t>(
          destination, hidden_offset + index * sizeof(std::uint32_t),
          std::bit_cast<std::uint32_t>(target_hidden[index]));
    }
    if (CheckedPersistentAdd(hidden_offset,
                             target_hidden.size() * sizeof(std::uint32_t)) !=
        destination.size()) {
      throw std::logic_error("MTP draft persistent serializer size mismatch");
    }
    return destination.size();
  }

  std::unique_ptr<QwenMtpGpuSnapshot> gpu_snapshot;
  std::vector<float> target_hidden;
  QwenMtpGpuDraftConfig config;
  bool primed{false};
};

}  // namespace

QwenMtpGpuDraftBackend::QwenMtpGpuDraftBackend(
    std::unique_ptr<QwenMtpGpuExecutor> executor, QwenMtpGpuDraftConfig config)
    : executor_(std::move(executor)),
      config_(config),
      target_hidden_(executor_->GetHiddenSize()) {
  const auto& model = executor_->GetModel();
  if (model.HasDraftVocab()) {
    const auto specials = model.GetDraftSpecials();
    draft_table_.emplace(
        std::vector<std::int32_t>(specials.begin(), specials.end()));
  }
}

bool QwenMtpGpuDraftBackend::SyncDraftVocab(
    std::span<const tokenization::TokenId> tokens) {
  if (!draft_table_) {
    return false;
  }
  // Table upkeep never fails a proposal: an inconsistent table drafts over
  // the full head and rebuilds next time. Device errors still propagate.
  const auto fall_back = [&](std::string_view reason) {
    draft_table_->Forget();
    last_error_ = reason;
    return false;
  };
  try {
    const auto update = draft_table_->Sync(tokens);
    if (update.restart) {
      executor_->ResetDraftRows();
    }
    executor_->AppendDraftRows(update.added);
  } catch (const std::logic_error& error) {
    return fall_back(error.what());
  } catch (...) {
    // The device table may hold part of the update; rebuild it next time.
    draft_table_->Forget();
    throw;
  }
  if (executor_->GetDraftRows() != draft_table_->rows()) {
    return fall_back("MTP draft table disagrees with its device rows");
  }
  return draft_table_->Fits(tokens.back());
}

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
      std::move(model), config.max_context, config.prime_chunk, error_msg);
  if (executor == nullptr) {
    return nullptr;
  }
  executor->SetVisionInput(config.vision_input);
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
    if (config_.prime_chunk != 0) {
      // Batched, projection-only prime: the same rows the loop below writes
      // (position `i` consumes token `i + 1` and the trunk hidden of `i`), with
      // no attention, MoE or logits, because nothing downstream of the KV write
      // is consumed by the draft.
      const std::size_t rows = context.prompt_tokens.size() - 1;
      executor_->PrimeTargetHiddenBatch(
          0, context.prompt_tokens.subspan(1),
          context.prompt_hidden_states.first(rows * context.hidden_size));
    } else {
      for (std::size_t index = 0; index + 1 < context.prompt_tokens.size();
           ++index) {
        const auto hidden = context.prompt_hidden_states.subspan(
            index * context.hidden_size, context.hidden_size);
        (void)executor_->ForwardTargetHidden(
            context.prompt_tokens[index + 1], hidden,
            static_cast<std::uint32_t>(index), false);
      }
    }
    // The prompt rows may be pinned and reused by the next target prefill.
    executor_->Synchronize();
    const std::size_t final_offset =
        (context.prompt_tokens.size() - 1) * context.hidden_size;
    const auto final_hidden =
        context.prompt_hidden_states.subspan(final_offset, context.hidden_size);
    std::ranges::copy(final_hidden, target_hidden_.begin());
    primed_ = true;
    return true;
  } catch (const std::exception& exception) {
    last_error_ = exception.what();
    // Earlier per-row uploads may still be reading the pinned prompt rows.
    try {
      executor_->Synchronize();
    } catch (...) {
    }
    Reset();
    return false;
  }
}

bool QwenMtpGpuDraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  if (!primed_ || proposal_active_ ||
      executor_->GetNextPosition() + 1 != position ||
      context.hidden_size != target_hidden_.size() ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * context.hidden_size) {
    return false;
  }
  if (context.prompt_tokens.empty()) {
    return true;
  }
  if (config_.prime_chunk != 0) {
    // Batched extension of the primed draft sequence: the row at draft
    // position `position - 1 + j` consumes suffix token `j` and the trunk
    // hidden of suffix row `j - 1` (the previously appended rolling hidden for
    // `j == 0`), which is exactly what the per-row loop below computes one
    // ForwardTargetHidden at a time. Projection-only like the prompt prime:
    // no attention, MoE or logits, because the proposal path consumes only the
    // draft KV write and the final trunk hidden.
    const std::size_t hidden_size = context.hidden_size;
    append_hidden_rows_.assign(target_hidden_.begin(), target_hidden_.end());
    append_hidden_rows_.insert(
        append_hidden_rows_.end(), context.prompt_hidden_states.begin(),
        context.prompt_hidden_states.begin() +
            static_cast<std::ptrdiff_t>((context.prompt_tokens.size() - 1) *
                                        hidden_size));
    executor_->PrimeTargetHiddenBatch(position - 1, context.prompt_tokens,
                                      append_hidden_rows_);
    std::ranges::copy(
        context.prompt_hidden_states.subspan(
            (context.prompt_tokens.size() - 1) * hidden_size, hidden_size),
        target_hidden_.begin());
    return true;
  }
  for (std::size_t row = 0; row < context.prompt_tokens.size(); ++row) {
    (void)executor_->ForwardTargetHidden(
        context.prompt_tokens[row], target_hidden_,
        position + static_cast<std::uint32_t>(row) - 1, false);
    UpdateTargetHidden(context.prompt_hidden_states.subspan(
        row * context.hidden_size, context.hidden_size));
  }
  return true;
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
  proposal_target_hidden_ = target_hidden_;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  const bool small_head = SyncDraftVocab(prompt_tokens);
  auto token = executor_->ForwardTargetHidden(
      proposal_input_, target_hidden_, proposal_checkpoint_, true, small_head);
  proposed_tokens_.push_back(token);
  for (std::uint32_t index = 1; index < count; ++index) {
    token = executor_->ForwardFeedback(token, proposal_checkpoint_ + index,
                                       true, small_head);
    proposed_tokens_.push_back(token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

speculative::DraftProposal QwenMtpGpuDraftBackend::ProposeSampled(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens, float temperature,
    std::uint64_t* rng_state) {
  namespace qfn = models::qwen38_flash_next;
  if (!std::isfinite(temperature) || temperature <= 0.0F) {
    throw std::invalid_argument(
        "MTP sampled proposal temperature must be finite and positive");
  }
  if (rng_state == nullptr) {
    throw std::invalid_argument("MTP sampled proposal requires RNG state");
  }
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
  proposal_target_hidden_ = target_hidden_;
  committed_target_hidden_.clear();
  proposed_tokens_.clear();
  proposed_tokens_.reserve(count);

  // q is the temperature softmax over the full head's top candidates. Rows
  // the distribution drops keep their IDs at probability zero, so every row
  // has the same width and distinct IDs.
  sampling::SamplingConfig config;
  config.temperature = temperature;
  const sampling::SamplerState sampler(config);
  std::size_t width = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    if (index == 0) {
      (void)executor_->ForwardTargetHidden(proposal_input_, target_hidden_,
                                           proposal_checkpoint_, true, false);
    } else {
      (void)executor_->ForwardFeedback(
          proposed_tokens_.back(), proposal_checkpoint_ + index, true, false);
    }
    const auto candidates = executor_->TopCandidates();
    if (width == 0) {
      width = candidates.size;
      proposal.candidates_per_token = width;
      proposal.candidate_ids.reserve(count * width);
      proposal.candidate_probabilities.reserve(count * width);
    } else if (candidates.size != width) {
      throw std::logic_error("MTP candidate width changed within a proposal");
    }
    const auto drawn = qfn::SampleMtpProposal(candidates, sampler, rng_state);
    const std::size_t row = proposal.candidate_ids.size();
    for (std::size_t i = 0; i < drawn.size; ++i) {
      proposal.candidate_ids.push_back(drawn.ids[i]);
      proposal.candidate_probabilities.push_back(drawn.probabilities[i]);
    }
    for (std::size_t i = 0;
         i < width && proposal.candidate_ids.size() < row + width; ++i) {
      const auto id = candidates.ids[i];
      if (std::find(proposal.candidate_ids.begin() + row,
                    proposal.candidate_ids.end(),
                    id) == proposal.candidate_ids.end()) {
        proposal.candidate_ids.push_back(id);
        proposal.candidate_probabilities.push_back(0.0F);
      }
    }
    proposed_tokens_.push_back(drawn.token);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = true;
  return proposal;
}

std::vector<speculative::DraftProposal> QwenMtpGpuDraftBackend::ProposeBatch(
    std::span<const speculative::DraftProposalRequest> requests) {
  if (requests.size() < 2 || requests.size() > kMtpMaxChainTokens)
    return IDraftBackend::ProposeBatch(requests);
  // First cut: greedy cohorts only. A sampled or mixed cohort keeps the
  // serial per-session path so its lossless semantics stay untouched.
  for (const auto& request : requests) {
    if (request.temperature != 0.0F)
      return IDraftBackend::ProposeBatch(requests);
  }
  std::array<QwenMtpGpuDraftBackend*, kMtpMaxChainTokens> backends{};
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* backend =
        dynamic_cast<QwenMtpGpuDraftBackend*>(requests[index].backend);
    if (backend == nullptr ||
        &backend->executor_->GetModel() != &executor_->GetModel())
      return IDraftBackend::ProposeBatch(requests);
    backends[index] = backend;
  }
  // Validate every request before any cache or cursor mutation. The checks
  // reproduce serial Propose's invariants so a rejected cohort matches what
  // the per-session path would have rejected.
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const auto& backend = *backends[index];
    if (!backend.primed_ || request.tokens.empty() || request.position == 0 ||
        backend.proposal_active_ ||
        backend.executor_->GetNextPosition() + 1 != request.position)
      throw std::invalid_argument("MTP proposal batch has an invalid request");
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (backends[previous] == backends[index])
        throw std::invalid_argument("MTP proposal batch repeats a session");
    }
  }
  struct SessionState {
    speculative::DraftProposal proposal;
    tokenization::TokenId input{0};
    std::uint32_t checkpoint{0};
    std::uint32_t count{0};
    bool small_head{false};
  };
  std::vector<SessionState> sessions(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const auto& backend = *backends[index];
    const auto count =
        std::min(request.max_tokens, backend.config_.max_draft_tokens);
    if (backend.executor_->GetNextPosition() + count >
        backend.executor_->GetMaxContext()) {
      throw std::invalid_argument(
          "MTP proposal batch exceeds a session context limit");
    }
    sessions[index].proposal.start_pos = request.position;
    sessions[index].input = request.tokens.back();
    sessions[index].checkpoint = backend.executor_->GetNextPosition();
    sessions[index].count = count;
  }
  // Choose every head, as serial Propose would from the same history, before
  // any chain starts: an error here leaves no chain in flight.
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (sessions[index].count != 0) {
      sessions[index].small_head =
          backends[index]->SyncDraftVocab(requests[index].tokens);
    }
  }
  // Stage the per-session serial Propose state, enqueue both chains, and only
  // then perform the single collect synchronization per executor.
  try {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      auto& backend = *backends[index];
      auto& session = sessions[index];
      backend.proposal_checkpoint_ = session.checkpoint;
      backend.proposal_input_ = session.input;
      backend.proposal_target_hidden_ = backend.target_hidden_;
      backend.committed_target_hidden_.clear();
      backend.proposed_tokens_.clear();
      if (session.count == 0) {
        continue;
      }
      backend.executor_->StartDraftChain(session.input, backend.target_hidden_,
                                         session.count, session.small_head);
    }
    for (std::size_t index = 0; index < requests.size(); ++index) {
      auto& backend = *backends[index];
      auto& session = sessions[index];
      if (session.count == 0) {
        continue;
      }
      backend.proposed_tokens_ = backend.executor_->CollectChainTokens();
      session.proposal.tokens = backend.proposed_tokens_;
      backend.proposal_active_ = true;
    }
  } catch (...) {
    // A failed batch must leave every touched session at its committed
    // frontier, exactly as a cancelled proposal discards its rows.
    // Abandon any chain still in flight first: a rollback Rewind with a live
    // chain throws RequireChainIdle, the rollback error lands in last_error_
    // in place of the true failure, and the executor stays wedged with
    // chain_active_ set until a noexcept Reset() would terminate on it.
    for (std::size_t index = 0; index < requests.size(); ++index) {
      backends[index]->executor_->AbandonChain();
    }
    for (std::size_t index = 0; index < requests.size(); ++index) {
      auto& backend = *backends[index];
      auto& session = sessions[index];
      try {
        if (backend.proposal_active_ ||
            backend.executor_->GetNextPosition() != session.checkpoint) {
          backend.executor_->Rewind(session.checkpoint);
        }
      } catch (const std::exception& rollback_error) {
        backend.last_error_ = rollback_error.what();
      }
      backend.committed_target_hidden_.clear();
      backend.proposal_target_hidden_.clear();
      backend.proposed_tokens_.clear();
      backend.proposal_active_ = false;
    }
    throw;
  }
  std::vector<speculative::DraftProposal> proposals;
  proposals.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    proposals.push_back(std::move(sessions[index].proposal));
  }
  return proposals;
}

void QwenMtpGpuDraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("MTP GPU proposal feedback is invalid");
  }
  if (committed_target_hidden_.size() !=
      (accepted.size() + 1) * target_hidden_.size()) {
    throw std::logic_error("MTP feedback requires each committed target row");
  }

  // Pair input t+1 with target hidden t, as during prompt priming. Verification
  // has already updated target_hidden_ to the final committed row; it cannot
  // be reused for the earlier anchor. Accepted rows also need target features,
  // rather than the tentative draft hidden states used while proposing.
  executor_->Rewind(proposal_checkpoint_);
  (void)executor_->ForwardTargetHidden(proposal_input_, proposal_target_hidden_,
                                       proposal_checkpoint_, false);
  for (std::size_t index = 0; index < accepted.size(); ++index) {
    (void)executor_->ForwardTargetHidden(
        accepted[index],
        std::span<const float>(committed_target_hidden_)
            .subspan(index * target_hidden_.size(), target_hidden_.size()),
        proposal_checkpoint_ + static_cast<std::uint32_t>(index) + 1, false);
  }
  proposal_active_ = false;
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
}

void QwenMtpGpuDraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != target_hidden_.size()) {
    throw std::invalid_argument("MTP target hidden-state shape is invalid");
  }
  if (proposal_active_) {
    committed_target_hidden_.insert(committed_target_hidden_.end(),
                                    hidden.begin(), hidden.end());
  }
  std::ranges::copy(hidden, target_hidden_.begin());
}

void QwenMtpGpuDraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_) {
    // A cancelled verification never feeds its proposal back, so the tentative
    // draft rows are dropped by rewinding to this step's checkpoint.
    if (proposal_checkpoint_ + 1 != position) {
      throw std::logic_error(
          "MTP cancellation frontier does not match the draft");
    }
    executor_->Rewind(proposal_checkpoint_);
    proposal_active_ = false;
    proposed_tokens_.clear();
    proposal_target_hidden_.clear();
    committed_target_hidden_.clear();
  }
  if (executor_->GetNextPosition() + 1 != position) {
    throw std::logic_error(
        "MTP cancelled draft is not at the committed frontier");
  }
}

std::size_t QwenMtpGpuDraftBackend::SnapshotPayloadBytes() const {
  // The primed flag travels with the payload but is not part of it; it is
  // reserved here so the reservation is an exact upper bound.
  return CheckedPersistentAdd(
      CheckedPersistentAdd(executor_->SnapshotPayloadBytes(),
                           target_hidden_.size() * sizeof(float)),
      sizeof(std::uint32_t));
}

std::unique_ptr<speculative::IDraftBackendSnapshot>
QwenMtpGpuDraftBackend::Snapshot() const {
  if (proposal_active_) {
    throw std::logic_error(
        "MTP snapshot requires a committed proposal boundary");
  }
  return std::make_unique<QwenMtpDraftSnapshot>(
      executor_->SaveSnapshot(), target_hidden_, config_, primed_);
}

void QwenMtpGpuDraftBackend::RestoreSnapshot(
    const speculative::IDraftBackendSnapshot& snapshot) {
  const auto* mtp_snapshot =
      dynamic_cast<const QwenMtpDraftSnapshot*>(&snapshot);
  if (mtp_snapshot == nullptr || mtp_snapshot->gpu_snapshot == nullptr) {
    throw std::invalid_argument(
        "MTP draft snapshot is incompatible with the backend");
  }
  if (mtp_snapshot->config.max_context != config_.max_context ||
      mtp_snapshot->config.max_draft_tokens != config_.max_draft_tokens ||
      mtp_snapshot->target_hidden.size() != target_hidden_.size() ||
      mtp_snapshot->primed != !mtp_snapshot->target_hidden.empty()) {
    throw std::invalid_argument(
        "MTP draft snapshot has a malformed target hidden row");
  }
  executor_->RestoreSnapshot(*mtp_snapshot->gpu_snapshot);
  target_hidden_ = mtp_snapshot->target_hidden;
  // The draft table is not saved; it rebuilds from the next proposal's
  // sequence.
  if (draft_table_) {
    draft_table_->Forget();
  }
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = mtp_snapshot->primed;
  proposal_active_ = false;
}

void QwenMtpGpuDraftBackend::RestorePersistentSnapshot(
    std::span<const std::uint8_t> payload) {
  if (payload.size() < kMtpDraftPersistentHeaderBytes ||
      !std::equal(kMtpDraftPersistentMagic.begin(),
                  kMtpDraftPersistentMagic.end(), payload.begin()) ||
      GetLittleEndian<std::uint32_t>(payload, 8) !=
          kMtpDraftPersistentVersion ||
      GetLittleEndian<std::uint32_t>(payload, 12) !=
          kMtpDraftPersistentHeaderBytes ||
      GetLittleEndian<std::uint32_t>(payload, 36) != 0) {
    throw std::invalid_argument("MTP draft persistent header is invalid");
  }
  const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 16);
  const std::size_t hidden_size = GetLittleEndian<std::uint32_t>(payload, 20);
  const std::uint32_t valid_context =
      GetLittleEndian<std::uint32_t>(payload, 24);
  const std::uint32_t max_context = GetLittleEndian<std::uint32_t>(payload, 28);
  const std::uint32_t max_draft_tokens =
      GetLittleEndian<std::uint32_t>(payload, 32);
  const std::size_t gpu_payload_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
  const std::size_t total_bytes =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
  const std::size_t hidden_count =
      PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56));
  const bool primed = (flags & kMtpDraftPrimedFlag) != 0;
  const std::size_t expected_hidden = target_hidden_.size();

  if ((flags & ~kMtpDraftPrimedFlag) != 0 || hidden_size != expected_hidden ||
      max_context != config_.max_context ||
      max_draft_tokens != config_.max_draft_tokens ||
      valid_context > config_.max_context ||
      hidden_count != (primed ? expected_hidden : 0) ||
      (!primed && valid_context != 0) ||
      hidden_count >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t) ||
      total_bytes != payload.size()) {
    throw std::invalid_argument(
        "MTP draft persistent metadata is incompatible");
  }
  const std::size_t gpu_offset = kMtpDraftPersistentHeaderBytes;
  const std::size_t hidden_offset =
      CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
  const std::size_t hidden_bytes = hidden_count * sizeof(std::uint32_t);
  if (CheckedPersistentAdd(hidden_offset, hidden_bytes) != payload.size()) {
    throw std::invalid_argument("MTP draft persistent payload size is invalid");
  }

  std::vector<float> target_hidden;
  target_hidden.reserve(hidden_count);
  for (std::size_t index = 0; index < hidden_count; ++index) {
    target_hidden.push_back(std::bit_cast<float>(GetLittleEndian<std::uint32_t>(
        payload, hidden_offset + index * sizeof(std::uint32_t))));
  }

  executor_->RestorePersistentSnapshot(
      payload.subspan(gpu_offset, gpu_payload_bytes));
  if (executor_->GetNextPosition() != valid_context) {
    throw std::invalid_argument(
        "MTP draft persistent context length is inconsistent");
  }
  if (primed) {
    target_hidden_ = std::move(target_hidden);
  } else {
    std::ranges::fill(target_hidden_, 0.0F);
  }
  if (draft_table_) {
    draft_table_->Forget();
  }
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = primed;
  proposal_active_ = false;
}

void QwenMtpGpuDraftBackend::Reset() noexcept {
  executor_->Reset();
  std::ranges::fill(target_hidden_, 0.0F);
  proposed_tokens_.clear();
  proposal_target_hidden_.clear();
  committed_target_hidden_.clear();
  proposal_input_ = 0;
  proposal_checkpoint_ = 0;
  primed_ = false;
  proposal_active_ = false;
  last_error_.clear();
  if (draft_table_) {
    draft_table_->Forget();
  }
}

void QwenMtpGpuDraftBackend::BeginRequest() noexcept {
  // A reused prefix can continue with other tokens than the ones scanned.
  if (draft_table_) {
    draft_table_->Forget();
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
