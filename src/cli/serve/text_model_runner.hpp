#ifndef GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
#define GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/sampling.hpp"

namespace gufo::server {

using TextRunnerToken = ContinuationToken;

struct TextRunnerDiskCacheOptions {
  std::filesystem::path directory;
  std::size_t capacity_bytes{0};
  std::size_t staging_capacity_bytes{0};
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
  bool requires_device_runtime_lock{false};
};

struct TextRunnerMeasuredResources {
  std::optional<std::size_t> per_request_state_bytes;
  std::optional<std::size_t> temporary_scratch_bytes;
};

struct TextPrefillStep {
  std::size_t consumed_tokens{0};
  bool decode_ready{false};
};

struct TextDecodeSelection {
  bool stop{false};
  TextRunnerToken token{0};
  std::string piece;
};

struct TextDecodeStep {
  std::vector<TextDecodeSelection> selections;
  std::size_t draft_tokens{0};
  std::size_t draft_accepted_tokens{0};
  bool stop{false};
};

/// Model-private state driven only through TextModelRunner work units.
class TextRunnerState : public ContinuationState {
public:
  using CancellationCheck = std::function<bool()>;

  /// Installs a request-scoped cancellation check for model calls that can
  /// yield internally. Implementations that only yield between work units may
  /// keep the default no-op behavior.
  virtual void SetCancellationCheck(const CancellationCheck&) {}

  /// Actual request-owned allocations when the provider can measure them.
  [[nodiscard]] virtual TextRunnerMeasuredResources MeasuredResources()
      const noexcept {
    return {};
  }
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

  [[nodiscard]] virtual std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const = 0;
  [[nodiscard]] virtual std::optional<std::vector<TextRunnerToken>>
  RenderAndTokenize(const ChatRequest& request) const = 0;
  [[nodiscard]] virtual TextGenerationBackend::InitialOutputState
  InitialOutputState(const ChatRequest&) const {
    return TextGenerationBackend::InitialOutputState::kAuto;
  }
  [[nodiscard]] virtual std::string Decode(
      std::span<const TextRunnerToken> tokens) const = 0;

  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> CreateState()
      const = 0;
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
  [[nodiscard]] virtual TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const = 0;
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

  /// Captures an immutable exact continuation at CheckpointPosition(state).
  ///
  /// The default implementations fail explicitly for runners that do not
  /// advertise the corresponding capabilities.
  [[nodiscard]] virtual std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const;
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

  /// Restores a version-compatible serialized payload into an existing state.
  virtual void RestorePersistentSnapshot(
      TextRunnerState& state, std::span<const std::uint8_t> payload) const;
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
      std::size_t disk_write_bytes{0};
      double disk_write_ms{0.0};
    };

    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_prompt_tokens() const noexcept;
    [[nodiscard]] std::size_t cache_restore_bytes() const noexcept;
    [[nodiscard]] double cache_restore_ms() const noexcept;
    [[nodiscard]] bool cache_disk_hit() const noexcept;
    [[nodiscard]] std::size_t prompt_tokens() const noexcept;
    [[nodiscard]] bool prefill_complete() const noexcept;

    void PrepareBatchExecution();
    [[nodiscard]] TextPrefillStep Prefill(std::size_t max_input_tokens);
    [[nodiscard]] TextDecodeSelection SelectNext();
    void Advance();
    [[nodiscard]] TextDecodeStep DecodeStep(std::size_t max_tokens);

    /// Publishes a reusable continuation boundary.
    ///
    /// Snapshot-capable runners publish the immutable prompt frontier captured
    /// before decode. Mutable-state runners publish their reported checkpoint.
    CommitMetrics Commit();
    void Invalidate() noexcept;

  private:
    friend class TextRunnerPool;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  TextRunnerPool(
      std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
      std::optional<TextRunnerDiskCacheOptions> disk_cache = std::nullopt);
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
  void AdvanceBatch(std::span<Request*> requests,
                    const TextExecutionPlan& plan);
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const sampling::SamplingConfig& sampling,
                                const CancellationCheck& is_cancelled = {});
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
