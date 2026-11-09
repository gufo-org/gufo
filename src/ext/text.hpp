#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/adapter.hpp"
#include "gufo/ext/cache.hpp"
#include "gufo/ext/continuation.hpp"
#include "gufo/ext/device_probe.hpp"
#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/host_budget.hpp"
#include "gufo/ext/ledger.hpp"
#include "gufo/ext/options.hpp"
#include "gufo/ext/reasoning.hpp"
#include "gufo/ext/sampling.hpp"
#include "gufo/ext/scheduling.hpp"
#include "gufo/ext/text_contracts.hpp"
#include "src/cache/component_resources.hpp"
#include "src/core/text_sampling_defaults.hpp"

namespace gufo::ext {

using ComponentCacheResources = cache::ComponentCacheResources;
using TextSchedulingProfile = sched::TextSchedulingProfile;
using PrefillYield = sched::PrefillYield;
using TextRunnerToken = ContinuationToken;

/// Immutable model-owned input which travels with one scheduled request.
/// Cache identity supplements token IDs for inputs such as image embeddings.
struct TextPromptContext {
  virtual ~TextPromptContext() = default;
  std::vector<std::uint8_t> cache_identity;
  std::vector<ContinuationInputPrefix> cache_prefixes;
  [[nodiscard]] std::span<const std::uint8_t> CacheIdentity(
      std::size_t token_count) const {
    return PrefixInputIdentity(cache_identity, cache_prefixes, token_count);
  }
  /// Moves a candidate prefill stop, a token count, to a position where
  /// prefill may end: the nearest legal one at or before `position`, or at or
  /// after it when `up`. A model whose input rows must be evaluated in one
  /// forward, such as image rows attending each other, returns the first row
  /// of the span containing `position` (down) or the position after its last
  /// token (up). Positions at span edges and outside spans are legal and must
  /// be returned unchanged, and the result must be a fixed point of this
  /// function. The pool applies it to cache checkpoints, snapshot and shared
  /// prefix boundaries, and never restores a checkpoint inside a span. A step
  /// that starts at a span consumes the whole span even beyond its token
  /// budget, so the runner's scratch must fit its largest span, and a runner
  /// consuming less than a step must not stop inside one. The default declares
  /// no spans.
  [[nodiscard]] virtual std::size_t PrefillStop(std::size_t position,
                                                bool /*up*/) const {
    return position;
  }
};

struct TextPreparedPrompt {
  std::vector<TextRunnerToken> tokens;
  std::shared_ptr<const TextPromptContext> context;
  /// Snapshot before mutable assistant framing. Zero uses the complete prompt.
  std::size_t cache_prefix_tokens{0};
};

/// Optional hidden-state encoding, which lets another model use a text
/// runner's backbone as a frozen encoder (for example, an image model's text
/// conditioning). Discovered through `TextModelRunner::HiddenStateEncoding()`.
struct TextHiddenStateEncoding {
  /// Scalars per row: the model's hidden size.
  std::size_t width{0};
  /// Most tokens one `EncodeHiddenStates` call accepts.
  std::size_t max_tokens{0};
};

/// One complete prompt, framed and tokenized by the caller.
struct TextHiddenStateRequest {
  /// Encoded as a whole from position zero with the model's causal attention.
  std::span<const TextRunnerToken> tokens;
  /// Polled during execution; returning true stops the call with
  /// `TextHiddenStateStatus::kCancelled`.
  std::function<bool()> is_cancelled;
};

/// Hidden states of one request. `values` holds `rows * width` FP32 scalars
/// in row-major order; row `i` belongs to request token `i`, so rows follow
/// token order and `rows` equals the request's token count.
///
/// Each row is the residual stream after the last decoder layer, before the
/// model's final output norm. Runners apply no normalization: a consumer that
/// conditions on normalized states applies that final norm exactly once.
///
/// The result owns its storage; it never aliases runner memory and remains
/// valid after the runner is destroyed.
struct TextHiddenStates {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<float> values;
};

enum class TextHiddenStateStatus : std::uint8_t {
  kOk,
  /// `is_cancelled` returned true; no states are returned.
  kCancelled,
  /// Empty prompt, more than `max_tokens` tokens, or a token outside the
  /// model's vocabulary. Nothing was executed.
  kInvalidInput,
  /// Execution failed. A later call may succeed.
  kFailed,
};

struct TextRunnerRamCacheOptions {
  static constexpr std::size_t kAutomaticMaxBytes = std::size_t{32} << 30;
  std::size_t capacity_bytes{0};
  bool enabled{true};
};

[[nodiscard]] std::optional<TextRunnerRamCacheOptions> ParseRamCacheOption(
    std::string_view text, std::string* error);

struct TextRunnerDiskCacheOptions {
  static constexpr std::size_t kDefaultCapacityBytes =
      std::size_t{8} * 1024U * 1024U * 1024U;
  std::filesystem::path directory;
  std::size_t capacity_bytes{kDefaultCapacityBytes};
  std::size_t staging_capacity_bytes{0};
  std::size_t shared_prefix_min_tokens{128};
  std::size_t shared_prefix_max_boundaries{4};
  std::size_t min_checkpoint_step_tokens{2048};
};

enum class TextExecutionPlanKind : std::uint8_t {
  kSerial,
  kBatched,
};

struct TextExecutionPlan {
  TextExecutionPlanKind kind{TextExecutionPlanKind::kSerial};
  std::size_t physical_width{1};

  bool operator==(const TextExecutionPlan&) const = default;
};

struct TextRunnerCapabilities {
  bool supports_vision{false};
  bool incremental_prefill{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t batched_multi_token_decode_max_width{0};
  bool prefix_reuse{true};
  bool in_pass_checkpoint{false};
  bool partial_prefill_retention{false};
  bool prefill_rollback{false};
  /// Prefill continues a prompt exactly only from positions divisible by
  /// this. Intermediate pass ends and published prompt checkpoints are
  /// rounded down to it, except the complete prompt, which only an
  /// unchanged retry restores; one keeps every position.
  std::size_t prefill_alignment{1};
  /// Boundaries one PrefillThrough call may capture in place.
  std::size_t max_in_pass_checkpoints{1};
};

struct TextRunnerDescriptor {
  std::string model_id;
  std::string state_abi;
  std::uint32_t max_context{0};
  TextRunnerCapabilities capabilities;
  /// Provider-defined sampling defaults. They take precedence over
  /// `sampling_preset`; neither set leaves controls to the caller.
  std::optional<sampling::TextSamplingDefaults> sampling_defaults;
  /// Core preset kept for existing extensions.
  sampling::TextModelPreset sampling_preset{
      sampling::TextModelPreset::kUnspecified};
  /// Prompt scheduling beside other sessions' decoding. Empty keeps passes
  /// bounded by the host's decode-active budget.
  std::optional<TextSchedulingProfile> scheduling{};
};

struct TextRunnerResourceClaim {
  std::optional<std::size_t> resident_weights_bytes;
  std::optional<std::size_t> state_capacity_bytes;
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
  std::optional<std::size_t> retained_snapshot_capacity_bytes;
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;
  bool requires_device_runtime_lock{false};
  /// Host bytes a prompt token writes into model state, currently the KV row.
  /// Missing means the runner has no prompt-length state to charge.
  std::optional<std::size_t> prompt_state_bytes_per_token;
  /// Host bytes prompt state may claim before admission refuses. Missing
  /// disables prompt-state admission for this runner.
  std::optional<std::size_t> prompt_admission_budget_bytes;
  /// Optional live host headroom, sampled before each prompt reservation.
  /// The server subtracts outstanding reservations conservatively, including
  /// those released during the sample. Empty keeps the fixed budget.
  std::function<std::size_t()> prompt_admission_available_bytes;
};

struct TextRunnerMeasuredResources {
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
};

struct TextPrefillStep {
  std::size_t consumed_tokens{0};
  bool decode_ready{false};
  double checkpoint_ms{0};
  bool interrupted{false};
};

struct TextRunnerNgramRows {
  std::uint64_t rows{0};
  std::uint64_t dedup_rows{0};
  std::uint64_t cache_hits{0};
  std::uint64_t disk_reads{0};
};

struct TextDecodeSelection {
  bool stop{false};
  TextRunnerToken token{0};
  std::string piece;
  std::optional<sampling::TokenLogprobs> logprobs;
};

struct TextDecodeStep {
  std::vector<TextDecodeSelection> selections;
  std::size_t draft_rounds{0};
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
  std::size_t lookup_tokens{0};
  std::size_t lookup_accepted_tokens{0};
  bool stop{false};
  std::optional<sampling::TokenLogprobs> terminal_logprobs;
  TextExecutionPlan execution_plan{};
  std::exception_ptr failure{};
};

class TextRunnerState : public ContinuationState {
public:
  using CancellationCheck = std::function<bool()>;

  void SetStopAtEos(bool value) noexcept { stop_at_eos_ = value; }
  [[nodiscard]] bool stop_at_eos() const noexcept { return stop_at_eos_; }

  void SetPrefillLookahead(std::span<const TextRunnerToken> tokens) noexcept {
    prefill_lookahead_ = tokens;
  }
  [[nodiscard]] std::span<const TextRunnerToken> prefill_lookahead()
      const noexcept {
    return prefill_lookahead_;
  }
  /// The whole prompt's token count during a prefill step, 0 outside one. A
  /// step's prompt ends at its own frontier, which may be a cache boundary
  /// short of this. A caller driving the runner outside the serve cache
  /// leaves it 0; treat 0 as the step's own prompt end.
  void SetPrefillPromptEnd(std::size_t tokens) noexcept {
    prefill_prompt_end_ = tokens;
  }
  [[nodiscard]] std::size_t prefill_prompt_end() const noexcept {
    return prefill_prompt_end_;
  }

  virtual void SetCancellationCheck(const CancellationCheck&) {}

  /// Set by the host for the next Prefill or PrefillThrough call only. A
  /// runner whose descriptor has a scheduling profile splits each pass into
  /// about `groups` layer groups of similar duration and calls YieldPrefill()
  /// between them; it may add a boundary where the pass would otherwise
  /// stall, such as a wait for input rows. Every layer keeps the arithmetic
  /// of an unsplit pass. A runner may ignore it and run unsplit passes.
  /// While yielding, it must hold no lock or scratch its decode paths take.
  void SetPrefillYield(std::size_t groups, PrefillYield yield) {
    prefill_yield_groups_ = yield ? std::max<std::size_t>(groups, 1) : 1;
    prefill_yield_ = std::move(yield);
  }
  /// One when the host asked for unsplit passes.
  [[nodiscard]] std::size_t prefill_yield_groups() const noexcept {
    return prefill_yield_groups_;
  }
  /// Runs the host's decode round between two layer groups.
  void YieldPrefill() const {
    if (prefill_yield_)
      prefill_yield_();
  }

  [[nodiscard]] virtual TextRunnerMeasuredResources MeasuredResources()
      const noexcept {
    return {};
  }

private:
  bool stop_at_eos_{true};
  std::span<const TextRunnerToken> prefill_lookahead_;
  std::size_t prefill_prompt_end_{};
  std::size_t prefill_yield_groups_{1};
  PrefillYield prefill_yield_;
};

struct TextRunnerAdvance {
  std::reference_wrapper<TextRunnerState> state;
  TextRunnerToken token{0};
  std::exception_ptr* failure{nullptr};
};

struct TextRunnerDecode {
  std::reference_wrapper<TextRunnerState> state;
  std::size_t max_tokens{0};
  std::reference_wrapper<sampling::SamplerState> sampler;
};

enum class InitialOutputState : std::uint8_t {
  kAuto,
  kThought,
  kText,
  kReasoning = kThought,
  kContent = kText,
};

enum class ChatRole : std::uint8_t {
  kSystem = 0,
  kDeveloper = 1,
  kUser = 2,
  kAssistant = 3,
  kTool = 4,
};

struct ChatImageInput {
  std::vector<std::uint8_t> bytes;
  std::string media_type;
  std::optional<std::string> format;
};

struct ChatToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;
};

struct ChatMessage {
  ChatMessage() = default;
  ChatMessage(ChatRole role_in, std::string content_in,
              std::string name_in = {}, std::string thought_in = {})
      : role(role_in),
        content(std::move(content_in)),
        name(std::move(name_in)),
        thought(std::move(thought_in)) {}

  ChatRole role{ChatRole::kUser};
  std::string content;
  std::string framing_suffix;
  std::string name;
  std::string thought;
  std::string tool_call_id;

  enum class ImageDetail : std::uint8_t { kAuto, kLow };
  struct ImagePart {
    std::size_t offset{0};
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    ImageDetail detail{ImageDetail::kAuto};
  };
  std::vector<ImagePart> images;

  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };
  struct ToolCall {
    std::string id;
    std::string name;
    std::vector<ToolArgument> arguments;
    std::optional<std::string> partial_arguments;
  };
  std::vector<ToolCall> tool_calls;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  std::string definition_json;
};

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<ChatMessage> messages;
  std::vector<ChatTool> tools;
  std::string client_id{"anonymous"};
  ToolChoice tool_choice{ToolChoice::kAuto};
  std::string forced_tool_name;
  std::map<std::string, std::string> tool_namespaces;
  bool constrained_tools{false};
  bool parallel_tool_calls{true};
  ReasoningOptions reasoning;
  bool add_vision_id{false};
  bool cache_prompt{true};
  bool cache_persist{true};
  bool return_progress{false};
  std::vector<std::string> stop_sequences;
  std::shared_ptr<const sampling::JsonConstraint> response_format;
  std::string response_format_description;
};

class TextModelRunner {
public:
  TextModelRunner() = default;
  virtual ~TextModelRunner() = default;

  TextModelRunner(const TextModelRunner&) = delete;
  TextModelRunner& operator=(const TextModelRunner&) = delete;
  TextModelRunner(TextModelRunner&&) = delete;
  TextModelRunner& operator=(TextModelRunner&&) = delete;

  [[nodiscard]] virtual TextRunnerDescriptor Descriptor() const = 0;
  [[nodiscard]] virtual TextRunnerResourceClaim ResourceClaim() const = 0;
  [[nodiscard]] virtual std::vector<TextExecutionPlan> SupportedPlans()
      const = 0;

  [[nodiscard]] virtual std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const {
    return nullptr;
  }
  [[nodiscard]] virtual sampling::JsonConstraint::ToolFormat ToolFormat()
      const {
    return sampling::JsonConstraint::ToolFormat::kJson;
  }
  /// Identity of the tool dialect that constrains and parses this runner's
  /// calls: a built-in identity (BuiltinToolDialectId) or the `id` of a
  /// registered ToolDialect. The default is the built-in identity of
  /// ToolFormat(), so existing runners keep their format; a runner that
  /// overrides this need not override ToolFormat(). The host rejects a runner
  /// whose identity is neither built in nor registered at load.
  [[nodiscard]] virtual std::string ToolDialectId() const;
  [[nodiscard]] std::shared_ptr<const sampling::TokenConstraint> BindConstraint(
      std::shared_ptr<const sampling::JsonConstraint> grammar) const;
  std::size_t PrepareConstraintVocabulary() const;

  using ReasoningTags = ext::ReasoningTags;
  [[nodiscard]] virtual ReasoningTags ReasoningBudgetTags() const {
    return {"<think>", "</think>", {"</think>"}};
  }
  /// Delimiters the OpenAI-compatible endpoints split reasoning by, buffered
  /// and streamed alike. By default they follow the budget tags: `start` opens
  /// and the first of `ends` closes (Qwen and DeepSeek: `<think>`, `</think>`);
  /// runners that declare no budget tags keep `<think>`, `</think>`.
  /// Override when the literal framing differs from the budget control tokens,
  /// keeping them about the same block: `ReasoningBudgetTags().start` is the
  /// control token that begins `open`, and its `ends` include the one for
  /// `close`. `open` must not begin with whitespace, which parsers skip.
  using ReasoningMarkup = ext::ReasoningMarkup;
  [[nodiscard]] virtual ReasoningMarkup ReasoningDelimiters() const;
  [[nodiscard]] virtual std::uint32_t VocabularySize() const { return 0; }
  [[nodiscard]] std::shared_ptr<const sampling::ReasoningBudget>
  BuildReasoningBudget(std::size_t tokens,
                       std::span<const TextRunnerToken> prompt,
                       ext::InitialOutputState initial =
                           ext::InitialOutputState::kThought) const;

  [[nodiscard]] virtual std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const = 0;
  [[nodiscard]] virtual std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const {
    return Tokenize(text);
  }
  [[nodiscard]] virtual std::optional<std::vector<TextRunnerToken>>
  RenderAndTokenize(const ChatRequest& request) const = 0;
  [[nodiscard]] virtual std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const {
    for (const auto& message : request.messages) {
      if (!message.images.empty()) {
        throw std::invalid_argument("model does not support image input");
      }
    }
    auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    return TextPreparedPrompt{std::move(*tokens), {}};
  }
  [[nodiscard]] virtual std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const {
    const auto prompt = TextModelRunner::PreparePrompt(request);
    if (!prompt)
      return std::nullopt;
    return prompt->tokens.size();
  }
  [[nodiscard]] virtual std::optional<std::string> RenderPrompt(
      const ChatRequest&) const {
    return std::nullopt;
  }
  virtual void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext> context) const {
    if (context != nullptr) {
      throw std::invalid_argument("model does not support this prompt context");
    }
  }
  [[nodiscard]] virtual ext::InitialOutputState InitialOutputState(
      const ChatRequest&) const {
    return ext::InitialOutputState::kAuto;
  }
  [[nodiscard]] virtual std::string Decode(
      std::span<const TextRunnerToken> tokens) const = 0;

  /// The runner's hidden-state encoder, or std::nullopt when it offers none.
  [[nodiscard]] virtual std::optional<TextHiddenStateEncoding>
  HiddenStateEncoding() const {
    return std::nullopt;
  }
  /// Encodes `request.tokens` as one complete prompt (see `TextHiddenStates`
  /// for the returned layout and normalization). Every call starts from an
  /// empty context and is independent of all `TextRunnerState`s: it neither
  /// reads nor changes sessions, prefix caches or snapshots, so repeating a
  /// request returns the same states. Concurrent calls on one runner
  /// serialize. Any working memory belongs to the runner and is released with
  /// it. `states` must not be null. On `kOk`, `*states` holds the result;
  /// otherwise it is empty and `*error` (when not null) explains why. Throws
  /// std::logic_error when
  /// `HiddenStateEncoding()` is std::nullopt, so an unsupported capability is
  /// never reported as an execution failure.
  [[nodiscard]] virtual TextHiddenStateStatus EncodeHiddenStates(
      const TextHiddenStateRequest& request, TextHiddenStates* states,
      std::string* error) const {
    (void)request;
    (void)states;
    (void)error;
    throw std::logic_error("runner does not support hidden-state encoding");
  }

  /// Continuation contract. The cache owns every execution slot: it creates
  /// the resources once, binds a state to each slot, and reconciles a state
  /// after restoring rows into its slot at the given position.
  [[nodiscard]] virtual std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger&, std::size_t) const = 0;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot&) const = 0;
  virtual void ReconcileComponentState(TextRunnerState&, std::size_t) const = 0;
  [[nodiscard]] virtual bool HasDecodeFrontier(const TextRunnerState&) const {
    return true;
  }
  /// Whether the state, live or restored, continues a prompt of
  /// `prompt_tokens`. A model that skips work on rows far from the prompt's
  /// end may keep only what longer prompts need; the cache serves a refused
  /// prompt from a shallower boundary.
  [[nodiscard]] virtual bool CanContinue(const TextRunnerState&,
                                         std::size_t /*prompt_tokens*/) const {
    return true;
  }

  [[nodiscard]] virtual std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState&, sampling::SamplerState&) const {
    return std::nullopt;
  }
  virtual void PreparePrefixReuse(
      TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
    (void)state;
    (void)prefix;
  }
  [[nodiscard]] virtual TextRunnerNgramRows NgramRows(
      const TextRunnerState& state) const {
    (void)state;
    return {};
  }
  virtual void PrepareBatchExecution(TextRunnerState& state) const {
    (void)state;
  }
  [[nodiscard]] virtual TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const = 0;
  /// Returns a claim only when this chunk covers the boundary, ending
  /// beyond it, and can capture it in place. Admission happens before
  /// allocating checkpoint state.
  [[nodiscard]] virtual std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t) const {
    return std::nullopt;
  }
  /// Prefills one chunk like Prefill, capturing each of the ascending
  /// `boundaries` the chunk covers (at most max_in_pass_checkpoints, each
  /// with a PrefillCheckpointBytes claim) without ending a pass there. Each
  /// capture must equal the state a pass ending at its boundary leaves. The
  /// runner invokes `consume` with each boundary's index, synchronously while
  /// that boundary is readable as the lease's component state; it must drain
  /// its captures before returning.
  using InPassConsumer = std::function<void(std::size_t)>;
  [[nodiscard]] virtual TextPrefillStep PrefillThrough(
      TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::span<const std::size_t>, const InPassConsumer&) const {
    throw std::logic_error("runner does not support in-pass checkpoints");
  }
  [[nodiscard]] virtual TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const = 0;
  virtual void Advance(TextRunnerState& state, TextRunnerToken token) const = 0;
  [[nodiscard]] virtual TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const;
  [[nodiscard]] virtual std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const;
  virtual void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const;
  [[nodiscard]] virtual std::size_t CheckpointPosition(
      const TextRunnerState& state) const = 0;
  virtual void PrepareCancellation(TextRunnerState&) const {}
  using DeviceProbeStatus = ext::DeviceProbeStatus;
  [[nodiscard]] virtual DeviceProbeStatus PollDevice() const {
    return DeviceProbeStatus::kUsable;
  }
  [[nodiscard]] virtual bool DeviceUsable() const { return true; }

private:
  mutable std::mutex constraint_mutex_;
  mutable std::shared_ptr<const sampling::ConstraintVocabulary>
      constraint_vocabulary_;
  mutable std::vector<std::shared_ptr<const sampling::TokenConstraint>>
      constraints_;
  mutable std::shared_ptr<const std::vector<bool>> partial_utf8_;
  mutable std::shared_ptr<const std::vector<bool>> blank_tokens_;
};

using TextRunner = TextModelRunner;

class HipTextModelRunner : public TextModelRunner {
public:
  HipTextModelRunner() = default;

  HipTextModelRunner(const HipTextModelRunner&) = delete;
  HipTextModelRunner& operator=(const HipTextModelRunner&) = delete;
  HipTextModelRunner(HipTextModelRunner&&) = delete;
  HipTextModelRunner& operator=(HipTextModelRunner&&) = delete;

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    return probe_.Poll();
  }

  [[nodiscard]] bool DeviceUsable() const override {
    return probe_.WaitUsable(std::chrono::seconds(5), "scheduler");
  }

private:
  ext::DeviceProbe probe_;
};

enum class TextSpeculativeBackend : std::uint8_t {
  kDisabled,
  kDFlash,
  kDSpark,
  kMtp,
};

enum class TextDFlashDraftPolicy : std::uint8_t {
  kAdaptive,
  kFixed,
};

[[nodiscard]] inline TextDFlashDraftPolicy ParseTextDFlashDraftPolicy(
    std::string_view value) {
  if (value == "fixed")
    return TextDFlashDraftPolicy::kFixed;
  if (value.empty() || value == "adaptive")
    return TextDFlashDraftPolicy::kAdaptive;
  throw std::invalid_argument("DFlash2 draft policy must be fixed or adaptive");
}

[[nodiscard]] inline std::string_view TextDFlashDraftPolicyName(
    TextDFlashDraftPolicy policy) {
  return policy == TextDFlashDraftPolicy::kFixed ? "fixed" : "adaptive";
}

struct TextSpeculativeConfig {
  TextSpeculativeBackend backend{TextSpeculativeBackend::kDisabled};
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  TextDFlashDraftPolicy dflash_policy{TextDFlashDraftPolicy::kAdaptive};
  /// The provider's draft policy (see `TextSpeculativeDefaults`); empty when
  /// the backend has no policy.
  std::string draft_policy{};
};

struct TextDiskCacheConfig {
  std::filesystem::path directory;
  std::size_t capacity_bytes{TextRunnerDiskCacheOptions::kDefaultCapacityBytes};
  std::size_t staging_capacity_bytes{0};
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
  std::size_t shared_prefix_min_tokens{128};
  std::size_t shared_prefix_max_boundaries{4};
  std::size_t min_checkpoint_step_tokens{2048};
};

struct TextRunnerOptions {
  std::string model_path;
  std::shared_ptr<const core::GgufReader> reader;
  std::uint32_t max_context{0};
  std::size_t session_count{1};
  TextSpeculativeConfig speculative{};
  std::string vision_model_path{};
  TextDiskCacheConfig disk_cache{};
  TextRunnerRamCacheOptions ram_cache{};
  bool enable_yarn{false};
  /// Explicit command-line values of the options the selected provider
  /// declares for this command (see `ext::FindOption`). An option that was not
  /// given is absent and keeps the provider's default.
  ParsedOptions provider_options{};
};

struct TextBenchOptions {
  std::string model_path;
  std::vector<std::size_t> n_prompts{2048};
  std::vector<std::size_t> n_gens{128};
  std::vector<std::size_t> n_depths{0};
  std::vector<std::size_t> concurrency{1};
  std::size_t repetitions{1};
  std::size_t validate_prefill_tokens{0};
  std::string speculative_backend;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  sampling::SamplingConfig sampling{[] {
    sampling::SamplingConfig c{};
    c.seed = 0;
    return c;
  }()};
  bool enable_yarn{false};
  bool verbose{false};
  /// Explicit command-line values of the options the selected provider
  /// declares for this command (see `ext::FindOption`). An option that was not
  /// given is absent and keeps the provider's default.
  ParsedOptions provider_options{};
};

struct TextPromptOptions {
  std::string model_path;
  std::string vision_model_path;
  std::vector<std::string> image_paths;
  bool add_vision_id{false};
  std::string prompt_text;
  std::string prompt_file;
  std::string system_prompt;
  std::size_t max_tokens{128};
  sampling::SamplingConfig sampling;
  sampling::SamplingOverrides sampling_supplied;
  std::string reasoning_mode{"auto"};
  std::string reasoning_effort{"auto"};
  std::string preserve_thinking{"auto"};
  bool display_prompt{true};
  bool use_chat_template{true};
  bool verbose{false};
  bool force_cpu{false};
  std::string speculative_backend;
  std::string mtp_model_path;
  std::string dflash_model_path;
  std::string draft_policy;
  std::string dspark_model_path;
  std::uint32_t draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  /// Explicit command-line values of the options the selected provider
  /// declares for this command (see `ext::FindOption`). An option that was not
  /// given is absent and keeps the provider's default.
  ParsedOptions provider_options{};

  /// Thinking mode these options select; a reasoning effort implies enabled.
  [[nodiscard]] std::optional<bool> thinking_mode() const {
    if (reasoning_mode == "on" || reasoning_effort != "auto")
      return true;
    if (reasoning_mode == "off")
      return false;
    return std::nullopt;
  }
};

struct ScannedToolArgument {
  std::string name;
  std::string value;
  bool is_string{true};
};

struct ScannedToolCall {
  std::string name;
  std::vector<ScannedToolArgument> arguments;
  std::optional<ScannedToolArgument> unfinished;
  bool valid{false};
  bool complete{false};
};

/// A provider's native tool-call dialect: the identity runners select it by,
/// the grammar that constrains calls and the parser that reads them back.
///
/// Identity: `id` (or, when empty, BuiltinToolDialectId(format)). `json` and
/// `qwen` are parsed by the host and cannot be registered; `deepseek` is the
/// DeepSeek extension's registration. A TextModelRunner selects a dialect with
/// ToolDialectId(), and that one identity drives both constrained generation
/// and the buffered and streamed parsers of every OpenAI-compatible and
/// Messages endpoint. Only the selected dialect's `markers` start calls; other
/// dialects' markup is ordinary text.
///
/// Grammar: `syntax` describes the native call syntax (JsonConstraint::
/// ToolSyntax), compiled from each tool's JSON schema for tool_choice
/// required/auto, strict and best-effort tools; tool_choice none leaves the
/// output unconstrained. Without `syntax`, the built-in grammar of `format`
/// (kQwen or kDeepSeek) is used.
///
/// Parsing: `markers` open a call (each must contain '<' and `syntax->open`
/// must be one of them); `reasoning_markers` are complete native call headers
/// which end a constrained reasoning phase (normally `open + name_prefix`
/// when `syntax->reasoning_header`); `closers` are closing tags the host
/// removes when the model echoes them directly after an accepted call.
/// `scan_tool_calls` receives the text from the first marker to the end,
/// the markers, and must append each call it finds in order and, to
/// `spans`, the [begin, end) byte range of each complete call or call block
/// relative to `text`; text outside spans is content. The host accepts a
/// call that is `complete` and `valid` (false: malformed arguments) and names
/// a declared tool. Argument values are strings (`is_string`) or JSON text,
/// which the host checks against the declared schema: a provider whose native
/// values are not JSON converts them. With `partial`, the text may end inside
/// a call: the scanner then also appends the one unfinished call that later
/// text could still complete, with `complete` false, only the arguments
/// already closed and, in `unfinished`, a string argument still open, and
/// stops. The host streams these as argument deltas, so they must only grow;
/// a call that is already malformed is not appended. The final, non-partial
/// parse is authoritative.
///
/// Lifetime: register during extension registration, before any runner is
/// created; registration is not thread-safe with lookups. The registry owns
/// the dialect; pointers from FindToolDialect stay valid for the process,
/// and registering the same identity again replaces it in place.
/// RegisterToolDialect throws std::invalid_argument for a dialect it cannot
/// use: a built-in host identity, no markers, a marker without '<', no
/// scan_tool_calls, `syntax->open` not among the markers, or neither `syntax`
/// nor a native `format`.
struct ToolDialect {
  sampling::JsonConstraint::ToolFormat format{
      sampling::JsonConstraint::ToolFormat::kJson};
  std::vector<std::string_view> markers;
  std::vector<std::string_view> reasoning_markers;
  std::vector<std::string_view> closers;
  using ScanToolCallsFn = void (*)(
      std::string_view text, std::span<const std::string_view> markers,
      std::vector<ScannedToolCall>* calls,
      std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial);
  ScanToolCallsFn scan_tool_calls{nullptr};
  std::string id;
  std::optional<sampling::JsonConstraint::ToolSyntax> syntax;
};

/// `json`, `qwen` or `deepseek`: the identity of a built-in format.
[[nodiscard]] std::string_view BuiltinToolDialectId(
    sampling::JsonConstraint::ToolFormat format);

using TextProbe = bool (*)(std::string_view model_root);
using TextProbeGguf = bool (*)(const core::GgufReader& reader);
/// Sampling defaults of the model in `reader`, for prompt and chat; `nullopt`
/// leaves controls to the command line.
using TextSamplingDefaultsProbe =
    std::optional<sampling::TextSamplingDefaults> (*)(
        const core::GgufReader& reader);
using TextRunnerFactory = std::shared_ptr<TextModelRunner> (*)(
    const TextRunnerOptions& options, std::string* error);
using TextBenchDriver =
    int (*)(const TextBenchOptions& options,
            const std::shared_ptr<const core::GgufReader>& reader,
            std::chrono::steady_clock::time_point load_start);
using TextPromptDriver =
    int (*)(const TextPromptOptions& options, const core::GgufReader& reader,
            std::chrono::steady_clock::time_point load_start);
using TextChatDriver =
    int (*)(const TextPromptOptions& options, const core::GgufReader& reader,
            std::chrono::steady_clock::time_point load_start);

/// A provider's draft settings for one speculative backend. The commands use
/// them, or the host's (7 draft tokens, minimum 1, no policy) when the provider
/// declares nothing for the backend. The draft flags are ignored.
struct TextSpeculativeDefaults {
  /// The `--speculative` value: `mtp`, `dflash2` or `dspark`.
  std::string backend;
  /// Most tokens proposed per step.
  std::uint32_t max_draft_tokens{7};
  /// Fewest tokens proposed per step.
  std::uint32_t min_draft_tokens{1};
  /// The draft policy; may be empty.
  std::string draft_policy;
  /// No longer read.
  std::vector<std::string> draft_policies;
};

struct TextProvider {
  std::string id;
  TextProbe probe = nullptr;
  TextProbeGguf probe_gguf = nullptr;
  TextRunnerFactory create = nullptr;
  TextBenchDriver bench_driver = nullptr;
  TextPromptDriver prompt_driver = nullptr;
  TextChatDriver chat_driver = nullptr;
  TextSamplingDefaultsProbe sampling_defaults = nullptr;
  /// Whether `probe_gguf` may see a GGUF with tensors core cannot size
  /// (`core::GgufTensorInfo::sized`), such as an extension's own
  /// quantization. A provider that sets it loads those tensors itself from
  /// their `data` and reserved `size_bytes`; the others never probe such a
  /// file.
  bool accepts_unsized_tensors = false;
  /// Draft settings per speculative backend the provider supports.
  std::vector<TextSpeculativeDefaults> speculative_defaults{};
};

void RegisterTextProvider(TextProvider provider);
const TextProvider* FindTextProvider(std::string_view id);
const TextProvider* ProbeTextProvider(std::string_view model_root);
/// The first provider whose `probe_gguf` accepts `reader`, skipping providers
/// without `accepts_unsized_tensors` when the file has unsized tensors.
const TextProvider* ProbeTextProvider(const core::GgufReader& reader);

void RegisterToolDialect(ToolDialect dialect);
/// The registered dialect with this identity, or null.
const ToolDialect* FindToolDialect(std::string_view id);
/// The registered dialect whose identity is BuiltinToolDialectId(format).
const ToolDialect* FindToolDialect(sampling::JsonConstraint::ToolFormat format);
std::vector<const ToolDialect*> AllToolDialects();

}  // namespace gufo::ext
