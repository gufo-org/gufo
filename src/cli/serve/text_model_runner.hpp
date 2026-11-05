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

#include "gufo/ext/text.hpp"
#include "src/cli/serve/continuation_types.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_publication_wait.hpp"

namespace gufo::server {

using ext::CgroupAvailableBytes;
using ext::ComponentCacheResources;
using ext::HipTextModelRunner;
using ext::HostSnapshotBudgetBytes;
using ext::HostSnapshotCeilingBytes;
using ext::InitialOutputState;
using ext::kHostSnapshotHeadroomBytes;
using ext::MeminfoAvailableBytes;
using ext::ParseRamCacheOption;
using ext::PrefillYield;
using ext::ReasoningMarkup;
using ext::TextDecodeSelection;
using ext::TextDecodeStep;
using ext::TextExecutionPlan;
using ext::TextExecutionPlanKind;
using ext::TextModelRunner;
using ext::TextPrefillStep;
using ext::TextPreparedPrompt;
using ext::TextPromptContext;
using ext::TextRunner;
using ext::TextRunnerAdvance;
using ext::TextRunnerCapabilities;
using ext::TextRunnerDecode;
using ext::TextRunnerDescriptor;
using ext::TextRunnerDiskCacheOptions;
using ext::TextRunnerMeasuredResources;
using ext::TextRunnerNgramRows;
using ext::TextRunnerRamCacheOptions;
using ext::TextRunnerResourceClaim;
using ext::TextRunnerState;
using ext::TextRunnerToken;

/// Bounded pool of opaque runner states with exact-prefix continuation reuse.
class TextRunnerPool {
public:
  using CancellationCheck = std::function<bool()>;

  class Request {
  public:
    struct CommitMetrics {
      std::optional<ComponentCacheMetrics> component_cache;
      std::size_t snapshot_bytes{0};
      double snapshot_ms{0.0};
      std::size_t disk_queued_bytes{0};
      double disk_enqueue_ms{0.0};
      /// Shared-prefix snapshots queued for persistence during prefill.
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
    /// N-gram rows gathered since this request leased its state.
    [[nodiscard]] TextRunnerNgramRows ngram_rows() const;
    [[nodiscard]] std::optional<ComponentCacheMetrics> component_cache_metrics()
        const;
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
    /// With `yield`, a runner with a scheduling profile splits the pass into
    /// at most `layer_groups` layer groups and runs it between them.
    [[nodiscard]] TextPrefillStep Prefill(std::size_t max_input_tokens,
                                          std::size_t layer_groups = 1,
                                          const PrefillYield& yield = {});
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
      bool stop_at_eos = true, bool persist = true);
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});
  /// Longest RAM-retained prefix of prompt that Acquire could reuse, without
  /// leasing a state. An upper bound: Acquire may still prefer a shorter one.
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const TextRunnerToken> prompt,
      const TextPromptContext* context) const;
  /// A disk job is still writing a prefix of prompt much deeper than any
  /// restorable checkpoint; admission may wait for it instead of prefilling.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const TextRunnerToken> prompt, const TextPromptContext* context,
      std::uint64_t checkpoint_id = 0) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Build output constraints and attach server-authored instructions as framing,
/// preserving client message content and the model's native prompt layout.
/// `tool_format` receives the built-in format the request was constrained
/// with (none for a provider syntax) and `tool_dialect` the runner's dialect
/// identity, which the response parsers select by. Throws
/// std::invalid_argument for an identity that is neither built in nor
/// registered.
[[nodiscard]] std::optional<ChatRequest> ConstrainChatRequest(
    const ChatRequest& request, const TextModelRunner& runner,
    sampling::SamplingConfig* sampling,
    std::optional<sampling::JsonConstraint::ToolFormat>* tool_format = nullptr,
    std::string* tool_dialect = nullptr);

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
