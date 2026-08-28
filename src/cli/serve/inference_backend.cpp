#include "src/cli/serve/inference_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#endif

namespace gufo::server {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)
std::uint64_t ClientLabel(std::string_view client_id) noexcept {
  constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  std::uint64_t hash = kFnvOffset;
  for (const unsigned char character : client_id) {
    hash ^= character;
    hash *= kFnvPrime;
  }
  return hash;
}

void EmitRequestMetrics(const InferenceBackend::Result& result,
                        std::string_view status) {
  static const bool enabled = (std::getenv("GUFO_DEBUG_METRICS") != nullptr);
  if (!enabled) {
    return;
  }
  static std::mutex output_mutex;
  std::ostringstream line;
  line << std::fixed << std::setprecision(3)
       << "{\"event\":\"http_inference\",\"status\":\"" << status
       << "\",\"client_label\":\"" << std::hex << ClientLabel(result.client_id)
       << std::dec << "\",\"prompt_tokens\":" << result.prompt_tokens
       << ",\"cache_hit\":" << (result.cache_hit ? "true" : "false")
       << ",\"cached_prompt_tokens\":" << result.cached_prompt_tokens
       << ",\"uncached_prompt_tokens\":"
       << (result.prompt_tokens - result.cached_prompt_tokens)
       << ",\"cache_restore_bytes\":" << result.cache_restore_bytes
       << ",\"cache_snapshot_bytes\":" << result.cache_snapshot_bytes
       << ",\"cache_shared_bytes\":" << result.cache_shared_bytes
       << ",\"cache_restore_ms\":" << result.cache_restore_ms
       << ",\"cache_snapshot_ms\":" << result.cache_snapshot_ms
       << ",\"prefill_tokens\":" << result.prefill_tokens
       << ",\"prefill_chunks\":" << result.prefill_chunks
       << ",\"active_decode_prefill_chunks\":"
       << result.active_decode_prefill_chunks
       << ",\"max_prefill_chunk_tokens\":" << result.max_prefill_chunk_tokens
       << ",\"max_consecutive_active_prefill_chunks\":"
       << result.max_consecutive_active_prefill_chunks
       << ",\"configured_active_prefill_tokens\":"
       << result.configured_active_prefill_tokens
       << ",\"queue_depth_at_submit\":" << result.queue_depth_at_submit
       << ",\"client_queue_depth_at_submit\":"
       << result.client_queue_depth_at_submit
       << ",\"resident_requests_at_admission\":"
       << result.resident_requests_at_admission
       << ",\"queue_ms\":" << result.queue_ms
       << ",\"requested_logical_concurrency\":"
       << result.requested_logical_concurrency
       << ",\"physical_execution_width\":" << result.physical_execution_width
       << ",\"execution_plan\":\"" << result.execution_plan << '"'
       << ",\"max_buffered_output_bytes\":" << result.max_buffered_output_bytes
       << ",\"incremental_prefill_supported\":"
       << (result.incremental_prefill_supported ? "true" : "false")
       << ",\"prefill_fallback_reason\":";
  if (result.prefill_fallback_reason.empty()) {
    line << "null";
  } else {
    line << '"' << result.prefill_fallback_reason << '"';
  }
  line << ",\"prefill_ms\":" << result.prefill_ms
       << ",\"completion_tokens\":" << result.completion_tokens
       << ",\"draft_tokens\":" << result.draft_tokens
       << ",\"draft_tokens_accepted\":" << result.draft_accepted_tokens
       << ",\"ttft_ms\":" << result.ttft_ms
       << ",\"mean_inter_token_ms\":" << result.mean_inter_token_ms
       << ",\"max_inter_token_ms\":" << result.max_inter_token_ms
       << ",\"decode_ms\":" << result.decode_ms
       << ",\"cancelled\":" << (result.cancelled ? "true" : "false") << "}";
  const std::lock_guard<std::mutex> lock(output_mutex);
  std::clog << line.str() << '\n';
}

bool IsQwenStopToken(const tokenization::QwenTokenizer& tokenizer,
                     TextRunnerToken token) noexcept {
  return token == tokenizer.GetEosTokenId() ||
         token == tokenization::kDefaultQwenEndoftextId || token == 248044U ||
         token == 248046U;
}

class QwenTextRunnerState final : public TextRunnerState {
public:
  QwenTextRunnerState(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
      speculative::SpeculativeOptions speculative_options)
      : model_(std::move(model)) {
    std::string error;
    executor_ = hip::QwenGpuExecutor::Create(model_, &error, max_context);
    if (executor_ == nullptr) {
      throw std::runtime_error("Failed to create GPU session: " + error);
    }
    if (dflash_model != nullptr) {
      auto draft_backend = hip::QwenDFlashGpuDraftBackend::Create(
          std::move(dflash_model),
          hip::QwenDFlashGpuDraftConfig{
              .max_context = max_context,
              .max_draft_tokens = speculative_options.max_draft_tokens,
          },
          &error);
      if (draft_backend == nullptr) {
        throw std::runtime_error("Failed to create DFlash session: " + error);
      }
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, std::move(draft_backend), speculative_options);
    }
  }

  void Invalidate() noexcept override {
    if (verifier_ != nullptr) {
      verifier_->Reset();
    }
    executor_->Reset();
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      const auto usage = executor_->GetMemoryUsage();
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    frontier_ = verifier_->Prime(prompt);
    const auto logits = verifier_->CopyLastTargetLogits();
    frontier_logits_.assign(logits.begin(), logits.end());
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (verifier_ == nullptr) {
      return;
    }
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen DFlash retained prefix does not match its target state");
    }
    sequence_.assign(prefix.begin(), prefix.end());
    frontier_published_ = false;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    for (const TextRunnerToken token : suffix) {
      frontier_ = verifier_->AdvanceCommittedToken(
          token, static_cast<std::uint32_t>(position_));
      sequence_.push_back(token);
      ++position_;
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(float temperature,
                                               std::uint64_t* rng_state) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    if (temperature == 0.0F) {
      return *frontier_;
    }
    const std::span<const float> logits =
        frontier_logits_.empty() ? executor_->CopyLastLogits()
                                 : std::span<const float>(frontier_logits_);
    return sampling::SampleLogits(logits, temperature, rng_state);
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(std::size_t max_tokens,
                                                 float temperature,
                                                 std::uint64_t* rng_state) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }

    TextDecodeStep result;
    const auto append_selection = [&](TextRunnerToken token) {
      if (IsQwenStopToken(model_->GetTokenizer(), token)) {
        result.stop = true;
        return false;
      }
      result.selections.push_back({
          .stop = false,
          .token = token,
          .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
      });
      sequence_.push_back(token);
      return true;
    };

    if (!frontier_published_) {
      if (temperature > 0.0F) {
        frontier_ = SelectFrontier(temperature, rng_state);
      }
      if (!append_selection(*frontier_)) {
        return result;
      }
      frontier_published_ = true;
      if (result.selections.size() == max_tokens) {
        return result;
      }
    }

    const auto stats_before = verifier_->GetStats();
    const std::size_t remaining = max_tokens - result.selections.size();
    const auto verification = verifier_->VerifyStep(
        sequence_, static_cast<std::uint32_t>(position_), *frontier_,
        model_->GetTokenizer().GetEosTokenId(),
        static_cast<std::uint32_t>(std::min<std::size_t>(
            remaining, std::numeric_limits<std::uint32_t>::max())),
        temperature, rng_state);
    const auto stats_after = verifier_->GetStats();
    result.draft_tokens =
        stats_after.total_draft_tokens - stats_before.total_draft_tokens;
    result.draft_accepted_tokens =
        stats_after.total_accepted_tokens - stats_before.total_accepted_tokens;

    for (const TextRunnerToken token : verification.emitted_tokens) {
      if (!append_selection(token)) {
        break;
      }
      ++position_;
    }
    frontier_ = verification.next_token;
    frontier_logits_ = verification.next_token_logits;
    frontier_published_ = !result.stop;
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
  SaveVerifierSnapshot() const {
    return verifier_ != nullptr ? verifier_->Snapshot() : nullptr;
  }
  void RestoreVerifierSnapshot(
      const speculative::SpeculativeVerifierSnapshot& snapshot) {
    if (verifier_ == nullptr) {
      throw std::invalid_argument(
          "cannot restore speculative state into a plain Qwen session");
    }
    verifier_->RestoreSnapshot(snapshot);
  }

private:
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken> frontier_;
  std::vector<float> frontier_logits_;
  bool frontier_published_{false};
};

class QwenTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenTextRunnerSnapshot(
      std::shared_ptr<const hip::QwenGpuModel> model,
      std::unique_ptr<hip::QwenGpuSnapshot> snapshot, std::size_t position,
      std::optional<TextRunnerToken> frontier,
      std::vector<float> frontier_logits,
      std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
          verifier_snapshot)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position),
        frontier(frontier),
        frontier_logits(std::move(frontier_logits)),
        verifier_snapshot(std::move(verifier_snapshot)) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (snapshot != nullptr ? snapshot->PayloadBytes() : 0) +
           frontier_logits.size() * sizeof(float) +
           (verifier_snapshot != nullptr ? verifier_snapshot->PayloadBytes()
                                         : 0);
  }

  std::shared_ptr<const hip::QwenGpuModel> model;
  std::unique_ptr<hip::QwenGpuSnapshot> snapshot;
  std::size_t position;
  std::optional<TextRunnerToken> frontier;
  std::vector<float> frontier_logits;
  std::unique_ptr<speculative::SpeculativeVerifierSnapshot> verifier_snapshot;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

class QwenTextRunner final : public TextModelRunner {
public:
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {})
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        max_context_(max_context),
        speculative_options_(speculative_options) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = dflash_model_ != nullptr;
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi = speculative_enabled ? "qwen-gfx1151-dflash-state-v2"
                                         : "qwen-gfx1151-state-v1",
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = !speculative_enabled,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = !speculative_enabled,
                .multi_token_decode = speculative_enabled,
                .prefix_reuse = true,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    const auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = model_->GetResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    if (dflash_model_ != nullptr) {
      return {{
          .kind = TextExecutionPlanKind::kSerial,
          .physical_width = 1,
      }};
    }
    return {
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 2,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 4,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 8,
        },
    };
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->GetTokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    tokenization::ChatTemplateOptions options;
    options.require_tool_call =
        request.tool_choice == ChatRequest::ToolChoice::kRequired;
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        options);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenTextRunnerState>(
        model_, max_context_, dflash_model_, speculative_options_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      qwen.PreparePrefixReuse(prefix);
    }
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    if (offset != qwen.position()) {
      throw std::logic_error(
          "Qwen prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Qwen prefill has no remaining input");
    }
    if (qwen.speculative()) {
      if (offset == 0) {
        if (max_input_tokens < prompt.size()) {
          throw std::logic_error(
              "Qwen DFlash cold prefill requires the complete prompt");
        }
        qwen.PrimeSpeculative(prompt);
        return {
            .consumed_tokens = prompt.size(),
            .decode_ready = true,
        };
      }
      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      if (consumed == 0) {
        throw std::logic_error(
            "Qwen DFlash retained prefix has no suffix to prefill");
      }
      qwen.ExtendSpeculative(prompt.subspan(offset, consumed));
      return {
          .consumed_tokens = consumed,
          .decode_ready = offset + consumed == prompt.size(),
      };
    }

    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const bool decode_ready = offset + consumed == prompt.size();
    const auto frontier = qwen.executor().ForwardPromptBatch(
        prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
        decode_ready);
    qwen.set_position(offset + consumed);
    if (decode_ready) {
      qwen.set_frontier(frontier);
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = decode_ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, float temperature,
      std::uint64_t* rng_state) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    const TextRunnerToken token = qwen.SelectFrontier(temperature, rng_state);
    if (IsQwenStopToken(model_->GetTokenizer(), token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens, float temperature,
      std::uint64_t* rng_state) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, temperature,
                                         rng_state);
    }
    return qwen.DecodeSpeculative(max_tokens, temperature, rng_state);
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (dflash_model_ != nullptr) {
      throw std::logic_error(
          "Qwen DFlash sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qwen = RequireQwenState(state);
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen state has no exact frontier to snapshot");
    }
    return std::make_unique<QwenTextRunnerSnapshot>(
        model_,
        qwen.executor().SaveSnapshot(
            static_cast<std::uint32_t>(qwen.position())),
        qwen.position(), qwen.frontier(), qwen.CopyFrontierLogits(),
        qwen.SaveVerifierSnapshot());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    if (qwen_snapshot == nullptr ||
        qwen_snapshot->model.get() != model_.get() ||
        qwen_snapshot->snapshot == nullptr ||
        !qwen_snapshot->frontier.has_value()) {
      throw std::invalid_argument(
          "Qwen snapshot does not belong to this model");
    }
    auto& restored = RequireQwenState(state);
    if (restored.speculative() !=
        (qwen_snapshot->verifier_snapshot != nullptr)) {
      throw std::invalid_argument(
          "Qwen snapshot speculative mode does not match the destination");
    }
    restored.executor().RestoreSnapshot(*qwen_snapshot->snapshot);
    restored.set_position(qwen_snapshot->position);
    restored.set_frontier(*qwen_snapshot->frontier);
    restored.RestoreFrontierLogits(qwen_snapshot->frontier_logits);
    if (qwen_snapshot->verifier_snapshot != nullptr) {
      restored.RestoreVerifierSnapshot(*qwen_snapshot->verifier_snapshot);
    }
  }

private:
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
};

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

std::string DeepSeekToolsPrompt(const ChatRequest& request) {
  if (request.tools.empty() ||
      request.tool_choice == ChatRequest::ToolChoice::kNone) {
    return {};
  }

  std::ostringstream prompt;
  prompt << "\n\n## Tools\n\n"
         << "You have access to tools. Invoke them with this exact syntax:\n"
         << "<｜DSML｜tool_calls｜>\n"
         << "<｜DS｜invoke name=\"$TOOL_NAME\">\n"
         << "<｜DS｜parameter name=\"$PARAMETER_NAME\" "
            "string=\"true|false\">$PARAMETER_VALUE"
            "</｜DS｜parameter>\n"
         << "</｜DS｜invoke>\n"
         << "</｜DSML｜tool_calls｜>\n\n"
         << "Available tool schemas:\n";
  for (const auto& tool : request.tools) {
    prompt << "{\"type\":\"function\",\"function\":{\"name\":"
           << std::quoted(tool.name)
           << ",\"description\":" << std::quoted(tool.description)
           << ",\"parameters\":"
           << (tool.parameters_json.empty() ? "{}" : tool.parameters_json)
           << "}}\n";
  }
  if (request.tool_choice == ChatRequest::ToolChoice::kRequired) {
    prompt << "\nYou must call at least one available tool.";
  }
  return prompt.str();
}

void AppendDeepSeekToolCalls(
    std::string& content,
    std::span<const tokenization::ChatMessage::ToolCall> calls) {
  if (calls.empty()) {
    return;
  }
  content.append("<｜DSML｜tool_calls｜>\n");
  for (const auto& call : calls) {
    content.append("<｜DS｜invoke name=\"");
    content.append(call.name);
    content.append("\">\n");
    for (const auto& argument : call.arguments) {
      content.append("<｜DS｜parameter name=\"");
      content.append(argument.name);
      content.append("\" string=\"");
      content.append(argument.is_string ? "true" : "false");
      content.append("\">");
      content.append(argument.value);
      content.append("</｜DS｜parameter>\n");
    }
    content.append("</｜DS｜invoke>\n");
  }
  content.append("</｜DSML｜tool_calls｜>");
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
  std::size_t position_{0};
};

class DeepSeekTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  DeepSeekTextRunnerSnapshot(
      std::shared_ptr<models::deepseek_v4_flash::Model> model,
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model;
  std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
  std::size_t position;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

class DeepSeekTextRunner final : public TextModelRunner {
public:
  DeepSeekTextRunner(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                     std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = "deepseek-v4-flash-gfx1151-state-v1",
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = std::nullopt,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    messages.reserve(request.messages.size() + 1);
    const std::string tools_prompt = DeepSeekToolsPrompt(request);
    bool tools_rendered = tools_prompt.empty();
    if (!tools_rendered &&
        (request.messages.empty() ||
         (request.messages.front().role != tokenization::ChatRole::kSystem &&
          request.messages.front().role !=
              tokenization::ChatRole::kDeveloper))) {
      messages.push_back({
          .role = "system",
          .content = tools_prompt,
      });
      tools_rendered = true;
    }
    for (const auto& message : request.messages) {
      std::string content = message.content;
      if (message.role == tokenization::ChatRole::kAssistant &&
          !message.thought.empty()) {
        content = "<think>\n" + message.thought + "\n</think>\n" + content;
      }
      if (!tools_rendered &&
          (message.role == tokenization::ChatRole::kSystem ||
           message.role == tokenization::ChatRole::kDeveloper)) {
        content += tools_prompt;
        tools_rendered = true;
      }
      if (message.role == tokenization::ChatRole::kAssistant) {
        AppendDeepSeekToolCalls(content, message.tool_calls);
      }
      messages.push_back({
          .role = std::string(ChatRoleName(message.role)),
          .content = std::move(content),
      });
    }
    auto tokens = DeepSeekRunnerTokens(model_->EncodeChat(messages));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<DeepSeekTextRunnerState>(model_, max_context_);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }

    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    std::string error;
    if (!deepseek.session().Sync(prefix, &error)) {
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, float temperature,
      std::uint64_t* rng_state) const override {
    auto& deepseek = RequireDeepSeekState(state);
    const int token =
        deepseek.session().SelectNext(temperature, rng_state, 0, 1.0F, 0.05F);
    if (token < 0) {
      throw std::runtime_error("DeepSeek token selection failed");
    }
    if (model_->IsStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    auto snapshot = deepseek.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("DeepSeek snapshot failed: " + error);
    }
    return std::make_unique<DeepSeekTextRunnerSnapshot>(
        model_, std::move(snapshot), deepseek.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek snapshot does not belong to this model");
    }
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*deepseek_snapshot->snapshot,
                                            &error)) {
      throw std::runtime_error("DeepSeek snapshot restore failed: " + error);
    }
    restored.set_position(deepseek_snapshot->position);
  }

private:
  std::shared_ptr<models::deepseek_v4_flash::Model> model_;
  std::uint32_t max_context_;
};

#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    SamplingDefaults sampling_defaults;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request, Result error_result)
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)),
          error_result_(std::move(error_result)) {}

    Result Wait(const TokenCallback& on_token) override {
      try {
        Result result = request_.Wait(on_token);
        EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
        return result;
      } catch (const TextGenerationError& exception) {
        EmitRequestMetrics(error_result_, exception.stable_code());
        throw;
      } catch (...) {
        EmitRequestMetrics(error_result_, "error");
        throw;
      }
    }

    void Cancel() noexcept override { request_.Cancel(); }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
    Result error_result_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(std::shared_ptr<const State> current,
                           std::vector<TextRunnerToken> prompt_tokens,
                           Clock::time_point request_start,
                           std::size_t max_tokens, float temperature,
                           const CancellationCheck& is_cancelled,
                           const TokenCallback& on_token,
                           std::string client_id) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      EmitRequestMetrics(result, "cancelled");
      return result;
    }

    try {
      auto request = current->scheduler->Submit(
          std::move(prompt_tokens), max_tokens, temperature, is_cancelled,
          static_cast<bool>(on_token),
          TextRequestMetadata{
              .client_id = std::move(client_id),
              .deadline = std::nullopt,
              .request_start = request_start,
          });
      result = request.Wait(on_token);
    } catch (...) {
      EmitRequestMetrics(result, "error");
      throw;
    }

    EmitRequestMetrics(result, result.cancelled ? "cancelled" : "ok");
    return result;
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config) {
#if defined(ENGINE_ENABLE_HIP)
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled) {
      SetError(error,
               "Speculative decoding is only supported by Qwen HTTP models");
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .prefill_chunk = 2048,
            .power_percent = 100,
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy);
  }
  auto model = hip::QwenGpuModel::CreateFromGguf(reader, &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  return load(std::move(model), error, max_context, session_count,
              prefill_policy, scheduler_policy, speculative_config);
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    speculative::SpeculativeOptions speculative_options;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        if (const char* environment = std::getenv("GUFO_DFLASH_MODEL");
            environment != nullptr && *environment != '\0') {
          speculative_config.draft_model_path = environment;
        }
      }
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      std::string dflash_error;
      auto dflash_reader_owner = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &dflash_error);
      if (dflash_reader_owner == nullptr) {
        SetError(error, "Failed to open DFlash GGUF: " + dflash_error);
        return false;
      }
      std::shared_ptr<const core::GgufReader> dflash_reader(
          std::move(dflash_reader_owner));
      dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                     model, &dflash_error);
      if (dflash_model == nullptr) {
        SetError(error, "Failed to create DFlash model: " + dflash_error);
        return false;
      }

      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification = true;
      speculative_options.use_batched_lm_head = true;
      speculative_options.retain_frontier_logits = true;
      speculative_options.target_bf16_from_layer = 48;
      switch (speculative_config.draft_policy) {
        case TextDraftPolicy::kFixed:
          speculative_options.enable_adaptive_draft_length = false;
          break;
        case TextDraftPolicy::kRollingAcceptance:
          break;
        case TextDraftPolicy::kAcceptedTokenEma:
          speculative_options.adaptive_draft_policy =
              speculative::AdaptiveDraftPolicy::kAcceptedTokenEma;
          break;
      }
    }

    auto new_state = std::make_shared<Impl::State>();
    auto runner = std::make_shared<QwenTextRunner>(
        std::move(model), max_context, std::move(dflash_model),
        speculative_options);
    new_state->model_id = runner->Descriptor().model_id;
    auto runner_pool =
        std::make_shared<TextRunnerPool>(std::move(runner), session_count);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    auto runner =
        std::make_shared<DeepSeekTextRunner>(std::move(model), max_context);
    new_state->model_id = runner->Descriptor().model_id;
    auto runner_pool =
        std::make_shared<TextRunnerPool>(std::move(runner), session_count);
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

void InferenceBackend::set_model_id(const std::string& model_id) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_id.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = model_id;
  impl_->state = std::move(updated);
#else
  (void)model_id;
#endif
}

void InferenceBackend::set_sampling_defaults(std::size_t max_tokens,
                                             float temperature) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults = {
      .max_tokens = max_tokens,
      .temperature = temperature,
  };
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)temperature;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(state, std::move(prompt_tokens),
                                  request_start, max_tokens, temperature,
                                  is_cancelled, on_token, "anonymous");
#else
  (void)prompt;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().RenderAndTokenize(request);
  if (!prompt_tokens.has_value() || prompt_tokens->empty()) {
    return {};
  }
  return impl_->GenerateScheduled(state, std::move(*prompt_tokens),
                                  request_start, max_tokens, temperature,
                                  is_cancelled, on_token, request.client_id);
#else
  (void)request;
  (void)max_tokens;
  (void)temperature;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             float temperature,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(request, max_tokens, temperature,
                                             is_cancelled, stream_output);
  }

  auto prompt_tokens = state->scheduler->runner().RenderAndTokenize(request);
  if (!prompt_tokens.has_value() || prompt_tokens->empty()) {
    return TextGenerationBackend::start_chat(request, max_tokens, temperature,
                                             is_cancelled, stream_output);
  }

  Result error_result;
  error_result.prompt_tokens = prompt_tokens->size();
  error_result.client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request =
      state->scheduler->Submit(std::move(*prompt_tokens), max_tokens,
                               temperature, is_cancelled, stream_output,
                               TextRequestMetadata{
                                   .client_id = error_result.client_id,
                                   .deadline = std::nullopt,
                                   .request_start = request_start,
                               });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request), std::move(error_result));
#else
  return TextGenerationBackend::start_chat(request, max_tokens, temperature,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, float temperature,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, temperature, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

}  // namespace gufo::server
