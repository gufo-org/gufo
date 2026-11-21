#include "staging/models/glm5_next/serve_runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <algorithm>
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
#include "src/core/image.hpp"
#include "src/core/reasoning.hpp"
#include "staging/models/glm5_next/chat_template.hpp"
#include "staging/models/glm5_next/vision/prompt.hpp"

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

/// Builds the template knobs from the request: effort label mapping and the
/// clear-thinking rule shared by the text and vision prompt paths.
[[nodiscard]] ChatRenderOptions RenderOptionsFor(const ChatRequest& request) {
  ChatRenderOptions options;
  if (request.reasoning.effort.has_value()) {
    options.reasoning_effort = RenderEffort(*request.reasoning.effort);
  }
  options.clear_thinking =
      request.reasoning.preserve_thinking.value_or(true) == false;
  return options;
}

/// Prompt context carried by an image request: the prepared per-image slots
/// the prefill splices, plus the FNV identity that keys the KV cache.
struct Glm5ImageContext final : TextPromptContext {
  std::vector<vision::VisionSlot> slots;
};

class Glm5TextRunnerState final : public TextRunnerState {
 public:
  Glm5TextRunnerState(const std::shared_ptr<Model>& model,
                      std::uint32_t max_context, bool use_mtp,
                      std::uint32_t min_drafts, std::uint32_t max_drafts) {
    std::string error;
    session_ = model->CreateSession(max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create GLM-5 session: " + error);
    }
    // A speculative session must fill the draft cache during prefill, so
    // enable MTP before the first Sync.
    session_->SetMtpEnabled(use_mtp);
    session_->SetDraftLimits(min_drafts, max_drafts);
  }

  void Invalidate() noexcept override {
    session_->Reset();
    vision_context_.reset();
    position_ = 0;
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  void set_vision_context(
      std::shared_ptr<const Glm5ImageContext> context) noexcept {
    vision_context_ = std::move(context);
  }
  [[nodiscard]] const std::shared_ptr<const Glm5ImageContext>& vision_context()
      const noexcept {
    return vision_context_;
  }

 private:
  std::shared_ptr<Session> session_;
  std::shared_ptr<const Glm5ImageContext> vision_context_;
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
  Glm5TextRunner(std::shared_ptr<Model> model, std::uint32_t max_context,
                 bool use_mtp, std::uint32_t min_drafts,
                 std::uint32_t max_drafts)
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        min_drafts_(min_drafts),
        max_drafts_(max_drafts) {}

  // GLM frames thoughts with plain text tags and closes a thought by ending
  // the turn: no explicit close tag means the next-turn control token
  // terminates it, mirroring the reference template's reasoning split.
  [[nodiscard]] ext::ThoughtMarkers ThoughtMarkers() const override {
    ext::ThoughtMarkers markers;
    markers.open = kThinkOpenTag;
    markers.close = kThinkCloseTag;
    const auto turn_end = model_->tokenizer().GetTurnEndTokenId();
    if (turn_end != kInvalidTokenId)
      markers.turn_end = model_->tokenizer().DecodeToken(turn_end);
    return markers;
  }

  // The prompt framing appends the thought opener (matching the reference
  // template's generation prompt), so the completion always starts inside
  // the thought and the stream needs no opener to enter reasoning.
  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest&) const override {
    return ext::InitialOutputState::kReasoning;
  }

  // GLM writes calls as kToolCallOpenTag + name + kArgKeyOpenTag frames,
  // recognized in the output like Ling 3.0 Flash: constraining them with the
  // JSON grammar would rewrite the template's dialect and reject its own
  // tokens.
  [[nodiscard]] sampling::JsonConstraint::ToolFormat ToolFormat()
      const override {
    return sampling::JsonConstraint::ToolFormat::kLing3;
  }

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
    const auto prompt = RenderChat(
        request.messages, true,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        RenderOptionsFor(request));
    if (!prompt.has_value()) {
      return std::nullopt;
    }
    return model_->tokenizer().Encode(*prompt);
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    const bool has_images = std::ranges::any_of(
        request.messages, [](const auto& m) { return !m.images.empty(); });
    if (!has_images) {
      auto tokens = RenderAndTokenize(request);
      if (!tokens) {
        return std::nullopt;
      }
      return TextPreparedPrompt{std::move(*tokens), {}};
    }
    if (!model_->HasVision()) {
      throw std::invalid_argument(
          "image input requires a matching --mmproj sidecar");
    }
    std::vector<core::Image> images;
    for (const auto& message : request.messages) {
      for (const auto& part : message.images) {
        if (part.bytes == nullptr) {
          throw std::invalid_argument("image part carries no bytes");
        }
        images.push_back(
            core::DecodeImage(std::span<const std::uint8_t>(*part.bytes)));
      }
    }
    std::string error;
    const auto prepared = model_->EncodeChatVision(
        request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        RenderOptionsFor(request), images, &error);
    if (prepared.tokens.empty()) {
      throw std::invalid_argument("GLM-5 image prompt failed: " + error);
    }
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(prepared.tokens.size());
    for (const TokenId token : prepared.tokens) {
      tokens.push_back(static_cast<TextRunnerToken>(token));
    }
    auto context = std::make_shared<Glm5ImageContext>();
    context->slots = std::move(prepared.images);
    context->cache_identity = ImageCacheIdentity(request);
    return TextPreparedPrompt{std::move(tokens), std::move(context), 0};
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& runner = RequireGlm5State(state);
    if (context == nullptr) {
      runner.set_vision_context(nullptr);
      return;
    }
    if (dynamic_cast<const Glm5ImageContext*>(context.get()) == nullptr) {
      throw std::invalid_argument("invalid GLM-5 prompt context");
    }
    runner.set_vision_context(
        std::static_pointer_cast<const Glm5ImageContext>(std::move(context)));
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(EngineTokens(tokens));
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<Glm5TextRunnerState>(model_, max_context_,
                                                 use_mtp_, min_drafts_,
                                                 max_drafts_);
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
    std::string error;
    if (const auto vision = runner.vision_context(); vision != nullptr) {
      if (offset != 0) {
        throw std::logic_error(
            "GLM-5 image prefill cannot resume mid-prompt");
      }
      // Image rows cannot be reused from a text prefix, so the whole prompt
      // is replayed with the staged embeddings spliced over the token rows;
      // the executor chunks internally, so one call bounds peak memory.
      if (!runner.session().Sync(EngineTokens(prompt), vision->slots,
                                 &error)) {
        runner.set_position(0);
        throw std::runtime_error("GLM-5 image prefill failed: " + error);
      }
      runner.set_position(prompt.size());
      return {
          .consumed_tokens = prompt.size(),
          .decode_ready = true,
      };
    }
    const std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const std::size_t next_position = offset + consumed;
    const auto prefix = EngineTokens(prompt.first(next_position));
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

  [[nodiscard]] std::optional<server::TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] server::TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    // Route even a one-token budget through the engine: a round can end
    // with a correction deferred into the next round, whose token must
    // reach the trunk before any logits are sampled.
    if (!use_mtp_) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "GLM-5 MTP decode budget must be at least one token");
    }
    auto& runner = RequireGlm5State(state);
    if (runner.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    Session::DecodeResult decoded;
    std::string error;
    if (!runner.session().DecodeStep(max_tokens, sampler, &decoded, &error)) {
      throw std::runtime_error("GLM-5 MTP decode failed: " + error);
    }
    // The engine never commits a stop token: a mid-chain end-of-turn
    // truncates the accepted prefix and a rejected-row end-of-turn stays
    // out of the deferred correction. So `decoded.tokens` is pure content
    // and Post() equals the prompt plus this content, which is exactly the
    // committed history `Request::Commit` validates.
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
    runner.set_position(runner.session().Position());
    step.draft_tokens = decoded.drafted;
    step.draft_accepted_tokens = decoded.accepted;
    return step;
  }

 private:
  /// FNV-1a over every image's raw bytes so distinct image sets never share
  /// a cached KV prefix; the count is folded in first to separate ordering.
  [[nodiscard]] static std::vector<std::uint8_t> ImageCacheIdentity(
      const ChatRequest& request) {
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](std::uint64_t value) {
      for (int byte = 0; byte < 8; ++byte) {
        hash = (hash ^ ((value >> (byte * 8)) & 0xFFULL)) * 1099511628211ULL;
      }
    };
    std::size_t count = 0;
    for (const auto& message : request.messages) {
      count += message.images.size();
    }
    mix(count);
    for (const auto& message : request.messages) {
      for (const auto& part : message.images) {
        if (part.bytes == nullptr) {
          mix(0);
          continue;
        }
        for (const std::uint8_t byte : *part.bytes) {
          hash = (hash ^ byte) * 1099511628211ULL;
        }
        mix(part.bytes->size());
      }
    }
    std::vector<std::uint8_t> identity;
    identity.reserve(8);
    for (int byte = 0; byte < 8; ++byte) {
      identity.push_back(
          static_cast<std::uint8_t>((hash >> (byte * 8)) & 0xFF));
    }
    return identity;
  }

  std::shared_ptr<Model> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t min_drafts_;
  std::uint32_t max_drafts_;
};

}  // namespace

std::shared_ptr<Model> LoadServeModel(
    const std::string& model_path, const core::GgufReader& reader,
    std::uint32_t max_context, const std::string& vision_model_path,
    const server::TextSpeculativeConfig& speculative_config,
    std::string* error) {
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "GLM-5 HTTP models support only nextn MTP speculative decoding "
             "(--speculative mtp)");
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(
      model_path,
      ModelOptions{.max_context = max_context,
                   .vision_model_path = vision_model_path},
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
  if (speculative_config.backend != server::TextSpeculativeBackend::kDisabled &&
      speculative_config.backend != server::TextSpeculativeBackend::kMtp) {
    SetError(error,
             "GLM-5 HTTP models support only nextn MTP speculative decoding "
             "(--speculative mtp)");
    return nullptr;
  }
  const bool use_mtp =
      speculative_config.backend == server::TextSpeculativeBackend::kMtp;
  if (use_mtp && !model->HasMtp()) {
    SetError(error,
             "GLM-5 MTP speculative decoding requires a GGUF with a nextn "
             "block");
    return nullptr;
  }
  if (!disk_cache_config.directory.empty()) {
    SetError(error, "GLM-5 HTTP models do not support the disk cache");
    return nullptr;
  }
  return std::make_shared<Glm5TextRunner>(
      std::move(model), max_context, use_mtp,
      speculative_config.min_draft_tokens, speculative_config.max_draft_tokens);
}

}  // namespace gufo::models::glm5_next

#endif  // defined(ENGINE_ENABLE_HIP)
