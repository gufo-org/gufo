#include "staging/models/glm5_next/serve_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "staging/models/glm5_next/chat_template.hpp"

namespace gufo::models::glm5_next {
namespace {

using server::ChatRequest;
using server::TextModelRunner;
using server::TextPreparedPrompt;
using server::TextPromptContext;
using server::TextRunnerState;
using server::TextRunnerToken;

constexpr std::string_view kGlm5StateAbi = "glm5-next-rocm-session-v1";

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::vector<std::int32_t> EngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("GLM-5 token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

/// Maps the provider-neutral effort onto the two labels the GLM-5 template
/// honors; anything else (or unset) renders the template default.
std::optional<std::string> RenderEffort(ReasoningEffort effort) {
  switch (effort) {
    case ReasoningEffort::kLow:
      return std::string("low");
    case ReasoningEffort::kHigh:
    case ReasoningEffort::kXHigh:
      return std::string("high");
    case ReasoningEffort::kMinimal:
    case ReasoningEffort::kMedium:
    case ReasoningEffort::kMax:
      break;
  }
  return std::nullopt;
}

class Glm5TextRunnerState final : public TextRunnerState {
 public:
  Glm5TextRunnerState(const std::shared_ptr<Model>& model,
                      std::uint32_t max_context) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create GLM-5 session: " + error);
    }
  }

  void Invalidate() noexcept override {
    session_->Reset();
    position_ = 0;
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

 private:
  std::shared_ptr<Session> session_;
  std::size_t position_{0};
};

Glm5TextRunnerState& RequireGlm5State(TextRunnerState& state) {
  auto* runner = dynamic_cast<Glm5TextRunnerState*>(&state);
  if (runner == nullptr) {
    throw std::logic_error("text runner state is not GLM-5");
  }
  return *runner;
}

const Glm5TextRunnerState& RequireGlm5State(const TextRunnerState& state) {
  const auto* runner = dynamic_cast<const Glm5TextRunnerState*>(&state);
  if (runner == nullptr) {
    throw std::logic_error("text runner state is not GLM-5");
  }
  return *runner;
}

/// Serial runner for GLM-5. The model owns one shared executor whose KDA
/// states, latent cache and indexer pool hold the only session state, so
/// the pool creates exactly one session.
class Glm5TextRunner final : public TextModelRunner {
 public:
  Glm5TextRunner(std::shared_ptr<Model> model, std::uint32_t max_context)
      : model_(std::move(model)), max_context_(max_context) {}

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->tokenizer().GetVocabSize(), [this](std::uint32_t id) {
          const auto& tokenizer = model_->tokenizer();
          return sampling::ConstraintVocabulary::Piece{
              tokenizer.DecodeTokenCopy(id), tokenizer.IsStopToken(id)};
        });
  }

  [[nodiscard]] server::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kGlm5StateAbi),
        .max_context = max_context_,
        .capabilities =
            server::TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = false,
                .fork = false,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
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
    ChatRenderOptions options;
    if (request.reasoning.effort.has_value()) {
      options.reasoning_effort = RenderEffort(*request.reasoning.effort);
    }
    options.clear_thinking =
        request.reasoning.preserve_thinking.value_or(true) == false;
    const auto prompt = RenderChat(
        request.messages, true,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        options);
    if (!prompt.has_value()) {
      return std::nullopt;
    }
    return model_->tokenizer().Encode(*prompt);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(EngineTokens(tokens));
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Glm5TextRunnerState>(model_, max_context_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& runner = RequireGlm5State(state);
    if (runner.position() != prefix.size()) {
      throw std::logic_error("GLM-5 reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] server::TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& runner = RequireGlm5State(state);
    if (offset != runner.position()) {
      throw std::logic_error(
          "GLM-5 prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("GLM-5 prefill has no remaining input");
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
    std::string error;
    if (!runner.session().Sync(prefix, &error)) {
      runner.set_position(0);
      throw std::runtime_error("GLM-5 prefill failed: " + error);
    }
    runner.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] server::TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& runner = RequireGlm5State(state);
    if (runner.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = runner.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error("GLM-5 token selection has no logits");
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
      throw std::invalid_argument("GLM-5 token ID exceeds engine range");
    }
    auto& runner = RequireGlm5State(state);
    std::string error;
    if (!runner.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("GLM-5 decode failed: " + error);
    }
    runner.set_position(runner.position() + 1);
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireGlm5State(state).position();
  }

 private:
  std::shared_ptr<Model> model_;
  std::uint32_t max_context_;
};

}  // namespace

std::shared_ptr<Model> LoadServeModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context,
    const server::TextSpeculativeConfig& speculative_config,
    std::string* error) {
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "GLM-5 HTTP models do not support speculative decoding");
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(model_path, ModelOptions{.max_context = max_context},
                           &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create GLM-5 model: " + load_error);
    return nullptr;
  }
  return model;
}

std::shared_ptr<TextModelRunner> CreateTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::size_t session_count,
    const server::TextSpeculativeConfig& speculative_config,
    const server::TextDiskCacheConfig& disk_cache_config, std::string* error) {
  if (session_count != 1) {
    SetError(
        error,
        "GLM-5 serves one session: its KDA states, latent cache and indexer "
        "pool live in the single shared device executor (--sessions 1)");
    return nullptr;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded GLM-5 model context");
    return nullptr;
  }
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled) {
    SetError(error,
             "GLM-5 HTTP models do not support speculative decoding");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error, "GLM-5 HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<Glm5TextRunner>(std::move(model), max_context);
}

}  // namespace gufo::models::glm5_next

#endif  // defined(ENGINE_ENABLE_HIP)
