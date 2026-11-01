#pragma once

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
#include <vector>

#include "gufo/ext/adapter.hpp"
#include "gufo/ext/cache.hpp"
#include "gufo/ext/continuation.hpp"
#include "gufo/ext/device_probe.hpp"
#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/host_budget.hpp"
#include "gufo/ext/ledger.hpp"
#include "gufo/ext/reasoning.hpp"
#include "gufo/ext/sampling.hpp"
#include "gufo/ext/text_contracts.hpp"
#include "src/cache/component_resources.hpp"
#include "src/core/text_sampling_defaults.hpp"

namespace gufo::ext {

using ComponentCacheResources = cache::ComponentCacheResources;
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
};

struct TextPreparedPrompt {
  std::vector<TextRunnerToken> tokens;
  std::shared_ptr<const TextPromptContext> context;
  /// Snapshot before mutable assistant framing. Zero uses the complete prompt.
  std::size_t cache_prefix_tokens{0};
};

struct TextRunnerRamCacheOptions {
  static constexpr std::size_t kAutomaticMaxBytes = std::size_t{32} << 30;
  static constexpr std::size_t kMaxEntries = 128;
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
  bool snapshot{false};
  bool fork{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t batched_multi_token_decode_max_width{0};
  bool prefix_reuse{true};
  bool in_pass_checkpoint{false};
  bool partial_prefill_retention{false};
  bool prefill_rollback{false};
};

struct TextRunnerPersistenceDescriptor {
  std::vector<std::uint8_t> compatibility_identity;
  std::uint32_t payload_version{0};

  bool operator==(const TextRunnerPersistenceDescriptor&) const = default;
};

struct TextRunnerDescriptor {
  std::string model_id;
  std::string state_abi;
  std::uint32_t max_context{0};
  TextRunnerCapabilities capabilities;
  std::optional<TextRunnerPersistenceDescriptor> persistence;
  sampling::TextModelPreset sampling_preset{
      sampling::TextModelPreset::kUnspecified};
};

struct TextRunnerResourceClaim {
  std::optional<std::size_t> resident_weights_bytes;
  std::optional<std::size_t> state_capacity_bytes;
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
  std::optional<std::size_t> retained_snapshot_capacity_bytes;
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;
  bool requires_device_runtime_lock{false};
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

  virtual void SetCancellationCheck(const CancellationCheck&) {}

  [[nodiscard]] virtual TextRunnerMeasuredResources MeasuredResources()
      const noexcept {
    return {};
  }

private:
  bool stop_at_eos_{true};
  std::span<const TextRunnerToken> prefill_lookahead_;
};

class TextRunnerSnapshot : public ContinuationSnapshot {
public:
  TextRunnerSnapshot() = default;
  ~TextRunnerSnapshot() override = default;

  TextRunnerSnapshot(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot& operator=(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot(TextRunnerSnapshot&&) = delete;
  TextRunnerSnapshot& operator=(TextRunnerSnapshot&&) = delete;
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
  [[nodiscard]] std::shared_ptr<const sampling::TokenConstraint> BindConstraint(
      std::shared_ptr<const sampling::JsonConstraint> grammar) const;
  std::size_t PrepareConstraintVocabulary() const;

  using ReasoningTags = ext::ReasoningTags;
  [[nodiscard]] virtual ReasoningTags ReasoningBudgetTags() const {
    return {"<think>", "</think>", {"</think>"}};
  }
  [[nodiscard]] virtual std::uint32_t VocabularySize() const { return 0; }
  [[nodiscard]] std::shared_ptr<const sampling::ReasoningBudget>
  BuildReasoningBudget(std::size_t tokens,
                       std::span<const TextRunnerToken> prompt) const;

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

  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> CreateState()
      const = 0;

  [[nodiscard]] virtual bool UsesComponentCache() const { return false; }
  [[nodiscard]] virtual std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger&, std::size_t) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter&, cache::Slot&) const;
  virtual void ReconcileComponentState(TextRunnerState&, std::size_t) const;
  [[nodiscard]] virtual bool HasDecodeFrontier(const TextRunnerState&) const {
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
  [[nodiscard]] virtual std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t) const {
    return std::nullopt;
  }
  [[nodiscard]] virtual TextPrefillStep PrefillThrough(
      TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t, std::unique_ptr<TextRunnerSnapshot>*,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& = {}) const {
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

  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot>
  SnapshotForPersistence(const TextRunnerState& state) const;

  virtual void RestoreOrFork(TextRunnerState& state,
                             const TextRunnerSnapshot& snapshot) const;

  [[nodiscard]] virtual std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const;

  [[nodiscard]] virtual std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const;

  using SnapshotSink = std::function<void(std::span<const std::uint8_t>)>;
  virtual void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                        const SnapshotSink& sink) const;

  virtual void RestorePersistentSnapshot(
      TextRunnerState& state, std::span<const std::uint8_t> payload) const;

private:
  mutable std::mutex constraint_mutex_;
  mutable std::shared_ptr<const sampling::ConstraintVocabulary>
      constraint_vocabulary_;
  mutable std::vector<std::shared_ptr<const sampling::TokenConstraint>>
      constraints_;
  mutable std::shared_ptr<const std::vector<bool>> partial_utf8_;
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
};

using TextProbe = bool (*)(std::string_view model_root);
using TextProbeGguf = bool (*)(const core::GgufReader& reader);
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

struct TextProvider {
  std::string id;
  TextProbe probe = nullptr;
  TextProbeGguf probe_gguf = nullptr;
  TextRunnerFactory create = nullptr;
  TextBenchDriver bench_driver = nullptr;
  TextPromptDriver prompt_driver = nullptr;
  TextChatDriver chat_driver = nullptr;
};

void RegisterTextProvider(TextProvider provider);
const TextProvider* FindTextProvider(std::string_view id);
const TextProvider* ProbeTextProvider(std::string_view model_root);
const TextProvider* ProbeTextProvider(const core::GgufReader& reader);

void RegisterToolDialect(ToolDialect dialect);
const ToolDialect* FindToolDialect(sampling::JsonConstraint::ToolFormat format);
std::vector<const ToolDialect*> AllToolDialects();

}  // namespace gufo::ext
