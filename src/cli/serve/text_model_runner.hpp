#ifndef GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
#define GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

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
  /// Zero selects min(model snapshot budget, 32 GiB), after session allocation.
  /// Explicit limits may exceed that budget up to the model's ceiling, which
  /// keeps kHostSnapshotHeadroomBytes of host RAM free.
  std::size_t capacity_bytes{0};
};

struct TextRunnerDiskCacheOptions {
  static constexpr std::size_t kDefaultCapacityBytes =
      std::size_t{8} * 1024U * 1024U * 1024U;
  static constexpr std::size_t kAutomaticStagingMaxBytes =
      std::size_t{1} * 1024U * 1024U * 1024U;
  std::filesystem::path directory;
  std::size_t capacity_bytes{kDefaultCapacityBytes};
  /// Zero selects 1/8 of available host RAM, capped at 1 GiB and
  /// capacity_bytes.
  std::size_t staging_capacity_bytes{0};
  /// Shared prefixes shorter than this are cheaper to prefill than to restore.
  std::size_t shared_prefix_min_tokens{128};
  /// Bound on shared-prefix snapshots written while prefilling one request.
  std::size_t shared_prefix_max_boundaries{4};
  /// Minimum token advance before a checkpoint is worth persisting.
  ///
  /// Continuations advance by a few hundred tokens per turn, so persisting
  /// every turn rewrites a largely identical snapshot. The tokens between two
  /// checkpoints are cheaper to re-prefill than that write. Zero disables the
  /// gate.
  std::size_t min_checkpoint_step_tokens{2048};
};

/// Automatic snapshot budget: half the host RAM available after loading,
/// after cgroup limits.
[[nodiscard]] std::size_t HostSnapshotBudgetBytes();
/// RAM always left to the OS and other processes by an explicit cache limit.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;
/// Most an explicit --cache-ram-bytes may claim: available host RAM minus
/// kHostSnapshotHeadroomBytes.
[[nodiscard]] std::size_t HostSnapshotCeilingBytes();

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
  bool incremental_prefill{false};
  bool snapshot{false};
  bool fork{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  /// Zero means no physical-width limit.
  std::size_t batched_multi_token_decode_max_width{0};
  bool prefix_reuse{true};
  bool in_pass_checkpoint{false};
};

/// Model-owned compatibility identity for restart-safe snapshots.
///
/// The bytes are canonical and opaque to serving code. They must cover every
/// model, tokenizer, template, layout, precision, context-policy, and adapter
/// property that can change restored continuation semantics. Executable
/// revisions are intentionally excluded when they retain the same state ABI.
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
};

/// Optional byte claims made before state allocation.
///
/// A missing value is an explicit "not yet measurable" claim. When both the
/// aggregate state capacity and per-request state are known, the common pool
/// validates the requested concurrency before creating model-private state.
struct TextRunnerResourceClaim {
  std::optional<std::size_t> resident_weights_bytes;
  std::optional<std::size_t> state_capacity_bytes;
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
  /// Aggregate bytes currently available for immutable retained snapshots.
  ///
  /// The pool queries this again after creating all mutable request states.
  std::optional<std::size_t> retained_snapshot_capacity_bytes;
  /// Most an explicit RAM-cache limit may claim. Missing means the automatic
  /// retained_snapshot_capacity_bytes is also the ceiling.
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
};

struct TextDecodeSelection {
  bool stop{false};
  TextRunnerToken token{0};
  std::string piece;
};

struct TextDecodeStep {
  std::vector<TextDecodeSelection> selections;
  /// Number of speculative verification rounds actually executed.
  std::size_t draft_rounds{0};
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
  bool stop{false};
  /// Execution actually used for this request, including model-owned subgroup
  /// dispatch. A runner's advertised maximum is not evidence of batching.
  TextExecutionPlan execution_plan{};
  std::exception_ptr failure{};
};

/// Model-private state driven only through TextModelRunner work units.
class TextRunnerState : public ContinuationState {
public:
  using CancellationCheck = std::function<bool()>;

  void SetStopAtEos(bool value) noexcept { stop_at_eos_ = value; }
  [[nodiscard]] bool stop_at_eos() const noexcept { return stop_at_eos_; }

  /// Installs a request-scoped cancellation check for model calls that can
  /// yield internally. Implementations that only yield between work units may
  /// keep the default no-op behavior.
  virtual void SetCancellationCheck(const CancellationCheck&) {}

  /// Actual request-owned allocations when the provider can measure them.
  [[nodiscard]] virtual TextRunnerMeasuredResources MeasuredResources()
      const noexcept {
    return {};
  }

private:
  bool stop_at_eos_{true};
};

/// Immutable model-owned continuation payload.
///
/// Common serving code may account for the payload but must not inspect or
/// reinterpret its bytes.
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

/// Coarse text-model adapter used by cache, scheduling, and HTTP layers.
///
/// Prefill and Advance are the only model execution work units. SelectNext
/// exposes the already-computed frontier token separately so streaming does
/// not wait for the following decode forward pass.
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

  /// Lazily built only for constrained requests; normal text loads pay nothing.
  [[nodiscard]] virtual std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const {
    throw std::invalid_argument("model does not support structured output");
  }
  [[nodiscard]] virtual sampling::JsonConstraint::ToolFormat ToolFormat()
      const {
    return sampling::JsonConstraint::ToolFormat::kJson;
  }
  /// Reuse a compiled grammar bound to this runner's vocabulary.
  [[nodiscard]] std::shared_ptr<const sampling::TokenConstraint> BindConstraint(
      std::shared_ptr<const sampling::JsonConstraint> grammar) const;

  [[nodiscard]] virtual std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const = 0;
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
  /// Called before cache restoration, prefill or selection, including
  /// a complete prefix hit. Must replace, never inherit, the previous request.
  virtual void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext> context) const {
    if (context != nullptr) {
      throw std::invalid_argument("model does not support this prompt context");
    }
  }
  [[nodiscard]] virtual TextGenerationBackend::InitialOutputState
  InitialOutputState(const ChatRequest&) const {
    return TextGenerationBackend::InitialOutputState::kAuto;
  }
  [[nodiscard]] virtual std::string Decode(
      std::span<const TextRunnerToken> tokens) const = 0;

  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> CreateState()
      const = 0;

  /// Optional first-token preview without advancing model state. The caller
  /// supplies a copy of the sampler; normal decoding must reproduce the token.
  /// This lets serving publish a token before capturing its prompt snapshot.
  [[nodiscard]] virtual std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState&, sampling::SamplerState&) const {
    return std::nullopt;
  }
  /// Reconciles model-private metadata after an exact mutable or restored
  /// prefix is leased. The default runner state needs no additional work.
  virtual void PreparePrefixReuse(
      TextRunnerState& state, std::span<const TextRunnerToken> prefix) const {
    (void)state;
    (void)prefix;
  }
  /// Tells a request state that its next execution will use a multi-request
  /// batch. Runners with request-local speculative state may discard work that
  /// the batched path cannot consume.
  virtual void PrepareBatchExecution(TextRunnerState& state) const {
    (void)state;
  }
  /// Processes at most max_input_tokens, yielding after one model-owned
  /// chunk even when the scheduler grants the whole remaining prompt.
  [[nodiscard]] virtual TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const = 0;
  /// Returns a claim only when this chunk can capture the boundary and keep
  /// prefilling beyond it. Admission happens before allocating checkpoint
  /// state.
  [[nodiscard]] virtual std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t) const {
    return std::nullopt;
  }
  [[nodiscard]] virtual TextPrefillStep PrefillThrough(
      TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
      std::size_t, std::size_t, std::unique_ptr<TextRunnerSnapshot>*) const {
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
  /// Retain a safe executed frontier when cancellation interrupts publication
  /// of a completed speculative block. Called with cancellation checks cleared.
  virtual void PrepareCancellation(TextRunnerState&) const {}
  enum class DeviceProbeStatus { kUsable, kPending, kLost };
  /// Submit or poll one private device probe without waiting. Called only by
  /// the scheduler between work units. Pending work is reused, never queued
  /// again.
  [[nodiscard]] virtual DeviceProbeStatus PollDevice() const {
    return DeviceProbeStatus::kUsable;
  }
  /// Bounded check, run only after a failed work unit, that the execution
  /// device still accepts work. False means the context is permanently lost;
  /// a probe that is still pending at its bound reports true.
  [[nodiscard]] virtual bool DeviceUsable() const { return true; }

  /// Captures an immutable exact continuation at CheckpointPosition(state).
  ///
  /// The default implementations fail explicitly for runners that do not
  /// advertise the corresponding capabilities.
  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const;
  /// May run on a capture worker while this state is frozen. Must not mutate
  /// shared execution scratch; other sessions may execute concurrently.
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const;
  /// Complete state before returning so export cannot introduce device copies
  /// while the session resumes prefill. Defaults to the ordinary snapshot.
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot>
  SnapshotForPersistence(const TextRunnerState& state) const;

  virtual void RestoreOrFork(TextRunnerState& state,
                             const TextRunnerSnapshot& snapshot) const;

  /// Returns the exact serialized bytes required by a persistent snapshot.
  ///
  /// Persistent serialization is separate from the in-memory snapshot layout:
  /// a provider may retain GPU-private state in RAM while exposing a compact,
  /// versioned disk representation.
  [[nodiscard]] virtual std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const;

  /// Serializes one immutable snapshot into exactly-sized common-store staging.
  ///
  /// Returns the number of initialized bytes. It must equal
  /// PersistentSnapshotPayloadBytes(snapshot).
  [[nodiscard]] virtual std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const;

  using SnapshotSink = std::function<void(std::span<const std::uint8_t>)>;
  /// Reads only immutable snapshot storage. May run on the persistence worker.
  /// Host-backed providers stream directly without another payload allocation.
  virtual void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                        const SnapshotSink& sink) const;

  /// Restores a version-compatible serialized payload into an existing state.
  virtual void RestorePersistentSnapshot(
      TextRunnerState& state, std::span<const std::uint8_t> payload) const;

private:
  mutable std::mutex constraint_mutex_;
  mutable std::shared_ptr<const sampling::ConstraintVocabulary>
      constraint_vocabulary_;
  mutable std::vector<std::shared_ptr<const sampling::TokenConstraint>>
      constraints_;
};

/// Bounded pool of opaque runner states with exact-prefix continuation reuse.
class TextRunnerPool {
public:
  using CancellationCheck = std::function<bool()>;

  class Request {
  public:
    struct CommitMetrics {
      std::size_t snapshot_bytes{0};
      double snapshot_ms{0.0};
      std::size_t disk_queued_bytes{0};
      double disk_enqueue_ms{0.0};
      /// Shared-prefix snapshots queued for persistence during prefill.
      std::size_t shared_prefix_snapshots{0};
      std::size_t shared_prefix_bytes{0};
      std::size_t shared_prefix_failures{0};
      double shared_prefix_ms{0.0};
    };

    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] ContinuationLookup cache_lookup() const noexcept;
    [[nodiscard]] std::size_t cached_prompt_tokens() const noexcept;
    [[nodiscard]] std::size_t cache_restore_bytes() const noexcept;
    [[nodiscard]] double cache_restore_ms() const noexcept;
    [[nodiscard]] bool cache_disk_hit() const noexcept;
    [[nodiscard]] std::size_t prompt_tokens() const noexcept;
    [[nodiscard]] bool prefill_complete() const noexcept;
    [[nodiscard]] std::span<const TextRunnerToken> prompt() const noexcept;
    /// Prompt tokens already in model state, restored or prefilled.
    [[nodiscard]] std::size_t prefill_position() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> input_identity(
        std::size_t token_count) const;

    /// Lets another request reuse this prompt's first common_tokens.
    ///
    /// Returns the position, at most common_tokens, where this request will
    /// publish a RAM checkpoint before prefilling past it, reusing a planned
    /// checkpoint within kSharedCheckpointSlack tokens. Zero when the runner
    /// cannot snapshot or this request is no longer before that position.
    /// Other requests depend on it, waiting now or arriving later, so it is
    /// retained with continuation priority rather than as an optional copy.
    [[nodiscard]] std::size_t ShareCheckpoint(std::size_t common_tokens);
    /// A planned checkpoint this close to the shared position is cheaper to
    /// use than capturing another one: followers prefill the gap themselves.
    static constexpr std::size_t kSharedCheckpointSlack = 64;
    /// A shared prefix must add at least this many tokens to what a request
    /// can already restore before it waits for a peer or captures an extra
    /// checkpoint for later requests. Shorter gaps cost less to prefill again.
    static constexpr std::size_t kSharedPrefixMinTokens = 512;

    void PrepareBatchExecution();
    [[nodiscard]] TextPrefillStep Prefill(std::size_t max_input_tokens);
    [[nodiscard]] TextDecodeSelection SelectNext();
    void Advance();
    [[nodiscard]] TextDecodeStep DecodeStep(std::size_t max_tokens);

    /// Publishes a reusable continuation boundary.
    ///
    /// Snapshot-capable runners keep the immutable prompt frontier and the
    /// final live state. Mutable-state runners publish their reported
    /// checkpoint.
    CommitMetrics Commit();
    /// Release a cancelled request, retaining only completed model work and
    /// an already captured prompt snapshot. Does not advance or sample tokens.
    CommitMetrics Cancel() noexcept;
    [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken();
    void CapturePromptSnapshot();
    /// Starts capture and polls completion without blocking other sessions.
    [[nodiscard]] bool PreparePromptSnapshot();
    [[nodiscard]] bool SnapshotPending() const;
    void Invalidate() noexcept;

  private:
    friend class TextRunnerPool;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  TextRunnerPool(
      std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
      std::optional<TextRunnerDiskCacheOptions> disk_cache = std::nullopt,
      TextRunnerRamCacheOptions ram_cache = {});
  ~TextRunnerPool();

  TextRunnerPool(const TextRunnerPool&) = delete;
  TextRunnerPool& operator=(const TextRunnerPool&) = delete;
  TextRunnerPool(TextRunnerPool&&) = delete;
  TextRunnerPool& operator=(TextRunnerPool&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] TextExecutionPlan SelectDecodePlan(
      std::size_t ready_requests) const;
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<Request*> requests, std::span<const std::size_t> max_tokens,
      const TextExecutionPlan& plan);
  std::vector<std::exception_ptr> AdvanceBatch(std::span<Request*> requests,
                                               const TextExecutionPlan& plan);
  [[nodiscard]] Request Acquire(
      std::vector<TextRunnerToken> prompt,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      std::shared_ptr<const TextPromptContext> context = {},
      bool reuse_prompt = true, std::size_t cache_prefix_tokens = 0,
      bool stop_at_eos = true);
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});
  /// Longest RAM-retained prefix of prompt that Acquire could reuse, without
  /// leasing a state. An upper bound: Acquire may still prefer a shorter one.
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const TextRunnerToken> prompt,
      const TextPromptContext* context) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Build output constraints and attach server-authored instructions as framing,
/// preserving client message content and the model's native prompt layout.
[[nodiscard]] std::optional<ChatRequest> ConstrainChatRequest(
    const ChatRequest& request, const TextModelRunner& runner,
    sampling::SamplingConfig* sampling,
    std::optional<sampling::JsonConstraint::ToolFormat>* tool_format = nullptr);

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
