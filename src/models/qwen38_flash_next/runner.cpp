#include "src/models/qwen38_flash_next/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/component_cache_resources.hpp"
#include "gufo/ext/experimental/admission_query.hpp"
#include "gufo/ext/gguf_identity.hpp"
#include "gufo/ext/host_budget.hpp"
#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/encoder.hpp"
#include "src/families/qwen/vision/image_cache.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "src/models/qwen38_flash_next/continuation_identity.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/probe.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using namespace gufo::ext;

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const auto token : tokens) {
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

template<typename State, typename Tokens>
bool EndsTurn(const State& state, const Tokens& tokens, int token) {
  return state.stop_at_eos() && tokens.IsStopToken(token);
}

[[nodiscard]] inline hip::ComponentCacheHooks HipComponentCacheHooks() {
  return {[] { return ext::HostSnapshotCeilingBytes(); },
          [](std::string_view) {}, [](std::string_view) {}};
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request,
                                                  std::uint32_t max_context) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  // gufo #285: the rendered prompt may be as long as the context can hold.
  options.max_output_bytes =
      tokenization::RenderedPromptBoundBytes(max_context);
  return options;
}

// The text before tokenization. An image renders as its one placeholder, not
// the visual tokens Prepare expands it into. Template errors are request
// errors, as in Prepare.
std::optional<std::string> RenderQwenPrompt(const ChatRequest& request,
                                            std::uint32_t max_context) {
  std::string error;
  auto rendered = tokenization::QwenChatTemplate::Render(
      request.messages,
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools},
      QwenChatOptions(request, max_context), &error);
  if (!rendered)
    throw std::invalid_argument(error);
  return rendered;
}

std::optional<std::size_t> CountQwenPromptTokens(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const auto options = QwenChatOptions(request, max_context);
  const auto tools =
      request.tool_choice == ChatRequest::ToolChoice::kNone
          ? std::span<const tokenization::ChatTool>{}
          : std::span<const tokenization::ChatTool>{request.tools};
  if (std::ranges::none_of(request.messages,
                           [](const auto& m) { return !m.images.empty(); })) {
    const auto tokens = tokenization::QwenChatTemplate::RenderAndTokenize(
        tokenizer, request.messages, tools, options);
    if (!tokens)
      return std::nullopt;
    return tokens->size();
  }
  return models::qwen::vision::Prepare(
             tokenizer, request.messages, tools, options,
             encoder ? encoder->identity() : std::string_view{},
             std::numeric_limits<std::int32_t>::max(),
             request.cache_prompt ? &image_cache : nullptr)
      .tokens.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context, models::qwen::vision::ImageCache& image_cache) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request, max_context);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context, request.cache_prompt ? &image_cache : nullptr));
  const auto cache_prefix = prompt->stable_prefix_tokens;
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  for (const auto& image : prompt->images) {
    const auto identity = prompt->IdentityForPrefix(image.grid.offset);
    context->cache_prefixes.push_back(
        {image.grid.offset, {identity.begin(), identity.end()}});
  }
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(QwenFlashNextSession& session,
                               cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] models::qwen38_flash_next::ContinuationAdapter& adapter()
      const {
    return dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(
        *adapter_);
  }
  [[nodiscard]] cache::Slot& slot() const { return *slot_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  QwenFlashNextSession* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

// Arms the model's layer hook from the host's yield for one prefill step.
class PrefillYieldScope {
public:
  PrefillYieldScope(const QwenFlashNextModel& model,
                    const TextRunnerState& state)
      : model_(model) {
    if (state.prefill_yield_groups() > 1)
      model_.SetPrefillYield(state.prefill_yield_groups(),
                             [&state] { state.YieldPrefill(); });
  }
  ~PrefillYieldScope() { model_.SetPrefillYield(1, {}); }
  PrefillYieldScope(const PrefillYieldScope&) = delete;
  PrefillYieldScope& operator=(const PrefillYieldScope&) = delete;

private:
  const QwenFlashNextModel& model_;
};

// Beside decoders, prompts keep the 4,096-row pass and split it into layer
// groups: one pass takes about 2.1 s, a decode step about 34 ms.
constexpr std::size_t kMaxPrefillLayerGroups = 8;
// Every batched plan runs inside a layer hook (Executor::kHookParkedRows),
// so each round decodes all decoders at once rather than one in turn.
constexpr std::size_t kMaxBatchWidth = 8;

class QwenFlashNextTextRunner final : public HipTextModelRunner {
public:
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    // The typed in-process API owns an immutable loaded model but need not
    // supply file fingerprints when disk caching is disabled. Its model
    // lifetime is an exact process-local identity; durable caches still
    // require the validated full artifact fingerprints below.
    if (!artifact_fingerprint.empty()) {
      component_identity_ = QwenFlashNextCompatibilityIdentity(
          artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
          use_mtp_, max_context_, max_draft_tokens_,
          model_->DecodeConcurrency(),
          models::qwen38_flash_next::RopePositionPolicy(
              model_->config().rope_scaling));
    } else {
      component_identity_ = QwenFlashNextProcessIdentity(
          model_.get(), max_context_, use_mtp_, max_draft_tokens_,
          model_->DecodeConcurrency());
    }
  }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    return hip::CreateHipComponentCacheResources(
        std::make_unique<models::qwen38_flash_next::ContinuationAdapter>(
            model_,
            use_mtp_ ? core::SessionMode::kSpeculative
                     : core::SessionMode::kAutoregressive,
            max_context_, component_identity_),
        ledger, HipComponentCacheHooks());
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& flash =
        dynamic_cast<models::qwen38_flash_next::ContinuationAdapter&>(adapter);
    return std::make_unique<QwenFlashNextTextRunnerState>(
        flash.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireQwenFlashNextState(state).set_position(position);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwenFlashNextStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .supports_vision = model_->VisionEncoder() != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width =
                    use_mtp_ ? kMaxBatchWidth : 0u,
                .prefix_reuse = true,
                .in_pass_checkpoint = true,
                .partial_prefill_retention = true,
                // Sparse prefill tiles group four queries from each pass
                // start: one selection block (compress ratio).
                .prefill_alignment =
                    std::max<std::size_t>(1, model_->config().compress_ratio),
                .max_in_pass_checkpoints = rocm::kMaxPassCheckpoints,
            },
        .sampling_defaults = FlashNextSamplingDefaults(),
        .scheduling =
            TextSchedulingProfile{
                .pass_tokens = model_->PrefillCapacity(),
                .max_layer_groups = kMaxPrefillLayerGroups,
                .max_yield_decode_width = kMaxBatchWidth,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    const auto session_bytes = model_->SessionBytes(
        use_mtp_ ? gufo::core::SessionMode::kSpeculative
                 : gufo::core::SessionMode::kAutoregressive,
        max_context_);
    std::optional<std::size_t> capacity;
    if (const auto memory = gufo::hip::QueryAdmissionMemory()) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity =
          memory->free_bytes > deferred ? memory->free_bytes - deferred : 0;
      gufo::hip::LogStateCapacity(*memory, *capacity, session_bytes, deferred);
    }
    // Admission charges complete snapshot payloads. Flash-Next can retain
    // mutable state and protected K/V rows in private device storage.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = session_bytes,
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= kMaxBatchWidth; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
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
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->tokenizer().GetVocabSize());
  }

  // llama.cpp's Qwen3-Coder format forces "\n</think>" and also ends
  // reasoning at a tool call.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "\n</think>", {"</think>", "<tool_call>"}};
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request, max_context_));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_,
                             image_cache_);
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    return CountQwenPromptTokens(request, model_->tokenizer(),
                                 model_->VisionEncoder(), max_context_,
                                 image_cache_);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderQwenPrompt(request, max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenFlashNextState(state).session().ConfigureVision(
        QwenPrompt(context));
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request, max_context_).enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t remaining = prompt.size() - offset;
    const std::size_t consumed =
        StepTokens(offset, max_input_tokens, remaining,
                   remaining <= model_->PrefillThroughCapacity()
                       ? model_->PrefillThroughCapacity()
                       : model_->PrefillCapacity());
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    // The next step most likely takes the same budget; its n-gram rows are
    // read during this one and used for the tokens it shares. A step ending
    // at a cache frontier continues into the pool's lookahead.
    const auto rest = next_position == prompt.size()
                          ? state.prefill_lookahead()
                          : prompt.subspan(next_position);
    const std::size_t after = rest.size();
    const auto next = QwenFlashNextEngineTokens(rest.subspan(
        0, std::min<std::size_t>({max_input_tokens, after,
                                  after <= model_->PrefillThroughCapacity()
                                      ? model_->PrefillThroughCapacity()
                                      : model_->PrefillCapacity()})));
    std::string error;
    const PrefillYieldScope yield(*model_, state);
    if (!qfn.session().Sync(prefix, &error, next)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    const auto consumed = StepTokens(offset, budget, prompt.size() - offset,
                                     model_->PrefillThroughCapacity());
    if (boundary <= offset + 1 || boundary >= offset + consumed)
      return std::nullopt;
    const auto bytes = qfn.session().PrefillCheckpointBytes(
        static_cast<std::uint32_t>(boundary));
    return bytes ? std::optional<std::size_t>(bytes) : std::nullopt;
  }

  [[nodiscard]] TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::span<const std::size_t> boundaries,
      const InPassConsumer& consume) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (!consume || boundaries.empty() || offset != qfn.position())
      throw std::logic_error("invalid component in-pass checkpoint request");
    const auto consumed = StepTokens(offset, budget, prompt.size() - offset,
                                     model_->PrefillThroughCapacity());
    const auto next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    // As in Prefill: the next step's n-gram rows are read during this one.
    const auto rest = next_position == prompt.size()
                          ? state.prefill_lookahead()
                          : prompt.subspan(next_position);
    const std::size_t after = rest.size();
    const auto next = QwenFlashNextEngineTokens(rest.subspan(
        0, std::min<std::size_t>({budget, after,
                                  after <= model_->PrefillThroughCapacity()
                                      ? model_->PrefillThroughCapacity()
                                      : model_->PrefillCapacity()})));
    std::vector<std::uint32_t> points(boundaries.begin(), boundaries.end());
    double capture_ms = 0;
    const PrefillYieldScope yield(*model_, state);
    try {
      qfn.adapter().PrefillThrough(
          qfn.slot(), prefix, points,
          [&](std::size_t index) {
            const auto started = std::chrono::steady_clock::now();
            consume(index);
            capture_ms += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
          },
          next);
    } catch (...) {
      qfn.set_position(0);
      throw;
    }
    qfn.set_position(next_position);
    return {.consumed_tokens = consumed,
            .decode_ready = next_position == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  // The engine stops before committing EOS when the request enables it.
  // Every returned token is committed work and is emitted exactly once.
  void ResolveStepTokens(
      QwenFlashNextTextRunnerState&, bool engine_stopped,
      std::span<const std::int32_t> tokens, TextDecodeStep* step,
      const QwenFlashNextSession::DecodeResult& decoded) const {
    step->stop = engine_stopped;
    step->terminal_logprobs = decoded.terminal_logprobs;
    step->selections.reserve(tokens.size());
    for (const std::int32_t token : tokens) {
      step->selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
          .logprobs =
              decoded.logprobs.empty()
                  ? std::nullopt
                  : std::optional(decoded.logprobs.at(step->selections.size())),
      });
    }
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    const bool stop = EndsTurn(qfn, *model_, token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : model_->TokenText(token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    // Use the same request policy as SelectNext and the batched MTP path.
    // The engine bounds the physical block and stops at EOS. Preserve the
    // remaining request budget so its policy can recognize a short tail.
    if (!qfn.session().DecodeStep(max_tokens, working_sampler, &decoded, &error,
                                  qfn.stop_at_eos())) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    ResolveStepTokens(qfn, decoded.stop, decoded.tokens, &step, decoded);
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_rounds = stats_after.cycles - stats_before.cycles;
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    step.lookup_tokens = stats_after.lookup - stats_before.lookup;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      auto& session = state.session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back({&session, decodes[i].max_tokens, &sampler,
                          &results[i], state.stop_at_eos(), &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      ResolveStepTokens(state, results[i].stop, results[i].tokens, &step,
                        results[i]);
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = TextExecutionPlan{
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = active_count,
        };
      }
      const auto stats = state.session().Statistics();
      step.draft_rounds = stats.cycles - before[i].cycles;
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
      // A request whose peers dropped out runs a single-session step,
      // which can use prompt lookup.
      step.lookup_tokens = stats.lookup - before[i].lookup;
      step.lookup_accepted_tokens =
          stats.lookup_accepted - before[i].lookup_accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] TextRunnerNgramRows NgramRows(
      const TextRunnerState& state) const override {
    const auto totals =
        RequireQwenFlashNextState(state).session().NgramRowTotals();
    return {totals.rows, totals.dedup_rows, totals.cache_hits,
            totals.disk_reads};
  }

private:
  /// Tokens one prefill step takes from `offset`. A step the pass capacity
  /// ends before the prompt stops on a selection block (prefill_alignment),
  /// as the serving runner ends budget-limited ones, so the next step
  /// continues the prompt exactly from an unaligned reused frontier too.
  [[nodiscard]] std::size_t StepTokens(std::size_t offset, std::size_t budget,
                                       std::size_t remaining,
                                       std::size_t capacity) const noexcept {
    auto count = std::min({budget, remaining, capacity});
    const std::size_t ratio =
        std::max<std::uint32_t>(1, model_->config().compress_ratio);
    if (const auto over = (offset + count) % ratio;
        count < remaining && count > over)
      count -= over;
    return count;
  }

  mutable models::qwen::vision::ImageCache image_cache_;
  std::shared_ptr<QwenFlashNextModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  cache::Identity component_identity_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config,
    std::string artifact_fingerprint, std::string draft_artifact_fingerprint) {
  const bool use_mtp =
      speculative_config.backend == ext::TextSpeculativeBackend::kMtp;
  return std::make_shared<QwenFlashNextTextRunner>(
      std::move(model), max_context, use_mtp,
      speculative_config.max_draft_tokens, std::move(artifact_fingerprint),
      std::move(draft_artifact_fingerprint));
}

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto reader_owner =
        core::GgufReader::OpenFile(options.model_path, &load_error);
    if (!reader_owner) {
      if (error)
        *error = "Failed to open GGUF: " + load_error;
      return nullptr;
    }
    reader = std::shared_ptr<const core::GgufReader>(std::move(reader_owner));
  }
  const auto architecture =
      reader->GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen4exp") {
    if (error)
      *error = "Flash-Next requires architecture qwen4exp";
    return nullptr;
  }
  const std::uint32_t qwen4exp_native_context =
      reader->GetMetadataUint32("qwen4exp.context_length").value_or(0);
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    max_context = qwen4exp_native_context;
    if (max_context < 2) {
      if (error)
        *error = "GGUF has no valid native context length; specify --context";
      return nullptr;
    }
  }
  std::optional<RopeScaling> rope_scaling;
  if (!ResolveRopeScaling(max_context, qwen4exp_native_context,
                          options.enable_yarn, &rope_scaling, error)) {
    return nullptr;
  }
  const bool use_mtp =
      options.speculative.backend == ext::TextSpeculativeBackend::kMtp;
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
      !use_mtp) {
    if (error)
      *error =
          "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
          "decoding";
    return nullptr;
  }
  if (use_mtp && options.speculative.draft_model_path.empty()) {
    if (error)
      *error = "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model";
    return nullptr;
  }
  std::string load_error;
  auto model = Model::Load(
      options.model_path,
      ModelOptions{
          .max_context = max_context,
          .mtp_model_path =
              use_mtp ? options.speculative.draft_model_path : std::string{},
          .vision_model_path = options.vision_model_path,
          .decode_concurrency =
              DecodeConcurrencyForSessions(options.session_count),
          .rope_scaling = rope_scaling,
      },
      &load_error);
  if (!model) {
    if (error)
      *error = "Failed to load Flash-Next model: " + load_error;
    return nullptr;
  }
  try {
    models::qwen::vision::PrepareForServing(model->VisionEncoder());
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
  // Each eight rows of wide copy verification past the first reserve
  // rollback rows beside the sessions (ResourceClaim). Narrow the width
  // until the configured sessions fit, rather than refusing a session count
  // the ordinary verifier width fits.
  if (std::size_t free_bytes = 0, total_bytes = 0;
      use_mtp && hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
    const std::size_t sessions =
        std::max<std::size_t>(options.session_count, 1);
    for (std::uint32_t rows = rocm::Executor::kWideRows;; rows -= 8) {
      model->LimitCopyRows(rows);
      if (rows == 8 ||
          sessions * model->SessionBytes(gufo::core::SessionMode::kSpeculative,
                                         max_context) +
                  model->DeferredScratchBytes() <=
              free_bytes)
        break;
    }
  }
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty() && model_fingerprint.empty()) {
    try {
      model_fingerprint = core::GgufIdentityHex(*reader);
    } catch (const std::exception& e) {
      if (error)
        *error = std::string("Flash-Next disk cache fingerprinting failed: ") +
                 e.what();
      return nullptr;
    }
  }
  std::string draft_fingerprint =
      options.disk_cache.draft_model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty() && use_mtp &&
      draft_fingerprint.empty()) {
    std::string mtp_error;
    auto mtp_reader = core::GgufReader::OpenFile(
        options.speculative.draft_model_path, &mtp_error);
    if (!mtp_reader) {
      if (error)
        *error = "Failed to open MTP model: " + mtp_error;
      return nullptr;
    }
    try {
      draft_fingerprint = core::GgufIdentityHex(*mtp_reader);
    } catch (const std::exception& e) {
      if (error)
        *error =
            std::string("MTP disk cache fingerprinting failed: ") + e.what();
      return nullptr;
    }
  }
  return CreateQwenFlashNextTextRunner(std::move(model), max_context,
                                       options.speculative, model_fingerprint,
                                       draft_fingerprint);
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // ENGINE_ENABLE_HIP
