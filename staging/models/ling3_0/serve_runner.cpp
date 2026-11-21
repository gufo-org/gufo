#include "staging/models/ling3_0/serve_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "staging/models/ling3_0/chat_template.hpp"

namespace gufo::models::ling3_0 {
namespace {

using server::ChatRequest;
using server::TextModelRunner;
using server::TextRunnerState;
using server::TextRunnerToken;

constexpr std::string_view kLing3StateAbi = "ling3-flash-rocm-session-v1";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

ChatTemplateOptions Ling3ChatOptions(const ChatRequest& request) {
  return ResolveLing3ChatOptions(request.reasoning);
}

std::vector<std::int32_t> EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Ling 3.0 Flash token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class Ling3TextRunnerState final : public TextRunnerState {
public:
  Ling3TextRunnerState(const std::shared_ptr<Model>& model,
                       std::uint32_t max_context, bool use_mtp,
                       std::uint32_t min_drafts, std::uint32_t max_drafts) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Ling 3.0 Flash session: " +
                               error);
    }
    // A speculative session must fill the draft cache during prefill, so
    // enable MTP before the first Sync.
    session_->SetMtpEnabled(use_mtp);
    session_->SetDraftLimits(min_drafts, max_drafts);
  }

  void Invalidate() noexcept override {
    session_->Reset();
    position_ = 0;
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<Session> session_;
  std::size_t position_{0};
};

Ling3TextRunnerState& RequireLing3State(TextRunnerState& state) {
  auto* ling3 = dynamic_cast<Ling3TextRunnerState*>(&state);
  if (ling3 == nullptr) {
    throw std::logic_error("text runner state is not Ling 3.0 Flash");
  }
  return *ling3;
}

const Ling3TextRunnerState& RequireLing3State(const TextRunnerState& state) {
  auto* ling3 = dynamic_cast<const Ling3TextRunnerState*>(&state);
  if (ling3 == nullptr) {
    throw std::logic_error("text runner state is not Ling 3.0 Flash");
  }
  return *ling3;
}

/// Serial runner for Ling 3.0 Flash. The model owns one shared executor
/// (weights and scratch) plus one private KV/recurrent state per session;
/// the pool creates up to the configured session count, each with a divided
/// context window.
class Ling3TextRunner final : public TextModelRunner {
public:
  Ling3TextRunner(std::shared_ptr<Model> model, std::uint32_t max_context,
                  bool use_mtp, std::uint32_t min_drafts,
                  std::uint32_t max_drafts)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        min_drafts_(min_drafts),
        max_drafts_(max_drafts) {}

  [[nodiscard]] server::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kLing3StateAbi),
        .max_context = max_context_,
        .capabilities =
            server::TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0u,
                .prefix_reuse = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] server::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Each session owns its KV/recurrent state; the executor scratch is
    // shared and already allocated at model load, so reserve it once.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(max_context_),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = 0,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<server::TextExecutionPlan> SupportedPlans()
      const override {
    return {
        {.kind = server::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    // `tool_choice: "none"` hides the tools section, so the model is free to
    // answer in prose and its calls would have no advertised frame.
    const bool show_tools =
        request.tool_choice != ChatRequest::ToolChoice::kNone;
    const auto rendered = RenderChat(
        request.messages, Ling3ChatOptions(request), true,
        show_tools ? std::span<const tokenization::ChatTool>(request.tools)
                   : std::span<const tokenization::ChatTool>{});
    if (!rendered.has_value()) {
      throw std::invalid_argument(
          "Ling 3.0 Flash does not support this conversation");
    }
    return Tokenize(*rendered);
  }

  [[nodiscard]] sampling::JsonConstraint::ToolFormat ToolFormat()
      const override {
    return sampling::JsonConstraint::ToolFormat::kLing3;
  }

  [[nodiscard]] server::TextGenerationBackend::InitialOutputState
  InitialOutputState(const ChatRequest& request) const override {
    return Ling3ChatOptions(request).enable_thinking
               ? server::TextGenerationBackend::InitialOutputState::kReasoning
               : server::TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Ling3TextRunnerState>(
        model_, max_context_, use_mtp_, min_drafts_, max_drafts_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& ling3 = RequireLing3State(state);
    if (ling3.position() != prefix.size()) {
      throw std::logic_error(
          "Ling 3.0 Flash reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] server::TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& ling3 = RequireLing3State(state);
    if (offset != ling3.position()) {
      throw std::logic_error(
          "Ling 3.0 Flash prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Ling 3.0 Flash prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
    std::string error;
    if (!ling3.session().Sync(prefix, &error)) {
      ling3.set_position(0);
      throw std::runtime_error("Ling 3.0 Flash prefill failed: " + error);
    }
    ling3.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] server::TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& ling3 = RequireLing3State(state);
    if (ling3.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = ling3.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("Ling 3.0 Flash token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Ling 3.0 Flash token ID exceeds engine range");
    }
    auto& ling3 = RequireLing3State(state);
    std::string error;
    if (!ling3.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Ling 3.0 Flash decode failed: " + error);
    }
    ling3.set_position(ling3.position() + 1);
  }

  [[nodiscard]] std::optional<server::TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] server::TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Ling 3.0 Flash MTP decode budget must be at least one token");
    }
    auto& ling3 = RequireLing3State(state);
    if (ling3.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    Session::DecodeResult decoded;
    std::string error;
    if (!ling3.session().DecodeStep(max_tokens, sampler, &decoded, &error)) {
      throw std::runtime_error("Ling 3.0 Flash MTP decode failed: " + error);
    }
    // The engine never commits a stop token, so `decoded.tokens` is pure
    // content and Post() equals the prompt plus this content.
    server::TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    ling3.set_position(ling3.session().Position());
    step.draft_tokens = decoded.drafted;
    step.draft_accepted_tokens = decoded.accepted;
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireLing3State(state).position();
  }

private:
  std::shared_ptr<Model> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t min_drafts_;
  std::uint32_t max_drafts_;
};

}  // namespace

std::shared_ptr<Model> LoadServeModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context,
    const server::TextSpeculativeConfig& speculative_config,
    std::string* error) {
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Ling 3.0 Flash HTTP models support only MTP speculative "
             "decoding (--speculative mtp)");
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(model_path, ModelOptions{.max_context = max_context},
                           &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create Ling 3.0 Flash model: " + load_error);
    return nullptr;
  }
  return model;
}

std::shared_ptr<TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error) {
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return nullptr;
  }
  constexpr std::size_t kMaxLing3Sessions = 16;
  if (session_count > kMaxLing3Sessions) {
    SetError(error,
             "Ling 3.0 Flash supports at most 16 concurrent HTTP sessions "
             "(--sessions 16)");
    return nullptr;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error,
             "HTTP context exceeds the loaded Ling 3.0 Flash model context");
    return nullptr;
  }
  // Divide the configured context across the concurrent sessions so their
  // combined KV/recurrent state fits the device budget; a single session
  // keeps the full window.
  const auto per_session_context = static_cast<std::uint32_t>(
      std::max<std::size_t>(1, max_context / session_count));
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "Ling 3.0 Flash HTTP models support only MTP speculative "
             "decoding (--speculative mtp)");
    return nullptr;
  }
  const bool use_mtp =
      speculative_config.backend == server::TextSpeculativeBackend::kMtp;
  if (use_mtp && !model->HasMtp()) {
    SetError(error,
             "Ling 3.0 Flash MTP speculative decoding requires a GGUF with "
             "an MTP block");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error, "Ling 3.0 Flash HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<Ling3TextRunner>(
      std::move(model), per_session_context, use_mtp,
      speculative_config.min_draft_tokens, speculative_config.max_draft_tokens);
}

}  // namespace gufo::models::ling3_0

#endif  // defined(ENGINE_ENABLE_HIP)
