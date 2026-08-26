#ifndef STRIX_SERVER_TEXT_MODEL_RUNNER_HPP_
#define STRIX_SERVER_TEXT_MODEL_RUNNER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/text_generation_backend.hpp"

namespace strix::server {

using TextRunnerToken = ContinuationToken;

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
};

struct TextRunnerDescriptor {
  std::string model_id;
  std::string state_abi;
  std::uint32_t max_context{0};
  TextRunnerCapabilities capabilities;
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
class TextRunnerSnapshot {
public:
  TextRunnerSnapshot() = default;
  virtual ~TextRunnerSnapshot() = default;

  TextRunnerSnapshot(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot& operator=(const TextRunnerSnapshot&) = delete;
  TextRunnerSnapshot(TextRunnerSnapshot&&) = delete;
  TextRunnerSnapshot& operator=(TextRunnerSnapshot&&) = delete;

  [[nodiscard]] virtual std::size_t PayloadBytes() const noexcept = 0;
};

struct TextRunnerAdvance {
  std::reference_wrapper<TextRunnerState> state;
  TextRunnerToken token{0};
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
  [[nodiscard]] virtual std::string Decode(
      std::span<const TextRunnerToken> tokens) const = 0;

  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> CreateState()
      const = 0;
  [[nodiscard]] virtual TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const = 0;
  [[nodiscard]] virtual TextDecodeSelection SelectNext(
      TextRunnerState& state, float temperature,
      std::uint64_t* rng_state) const = 0;
  virtual void Advance(TextRunnerState& state, TextRunnerToken token) const = 0;
  virtual void AdvanceBatch(std::span<const TextRunnerAdvance> advances) const;
  [[nodiscard]] virtual std::size_t CheckpointPosition(
      const TextRunnerState& state) const = 0;

  /// Captures an immutable exact continuation at CheckpointPosition(state).
  ///
  /// The default implementations fail explicitly for runners that do not
  /// advertise the corresponding capabilities.
  [[nodiscard]] virtual std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const;
  [[nodiscard]] virtual std::unique_ptr<TextRunnerState> RestoreOrFork(
      const TextRunnerSnapshot& snapshot) const;
};

/// Bounded pool of opaque runner states with exact-prefix continuation reuse.
class TextRunnerPool {
public:
  using CancellationCheck = std::function<bool()>;

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_prompt_tokens() const noexcept;
    [[nodiscard]] std::size_t prompt_tokens() const noexcept;
    [[nodiscard]] bool prefill_complete() const noexcept;

    [[nodiscard]] TextPrefillStep Prefill(std::size_t max_input_tokens);
    [[nodiscard]] TextDecodeSelection SelectNext(float temperature);
    void Advance();

    /// Publishes the model state at its reported checkpoint boundary.
    void Commit();
    void Invalidate() noexcept;

  private:
    friend class TextRunnerPool;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  TextRunnerPool(std::shared_ptr<TextModelRunner> runner,
                 std::size_t state_count);
  ~TextRunnerPool();

  TextRunnerPool(const TextRunnerPool&) = delete;
  TextRunnerPool& operator=(const TextRunnerPool&) = delete;
  TextRunnerPool(TextRunnerPool&&) = delete;
  TextRunnerPool& operator=(TextRunnerPool&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] TextExecutionPlan SelectDecodePlan(
      std::size_t ready_requests) const;
  void AdvanceBatch(std::span<Request*> requests,
                    const TextExecutionPlan& plan);
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_TEXT_MODEL_RUNNER_HPP_
