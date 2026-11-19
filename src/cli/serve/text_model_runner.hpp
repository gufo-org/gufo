#ifndef GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_
#define GUFO_SERVER_TEXT_MODEL_RUNNER_HPP_

#include <atomic>
#include <chrono>
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
#include "src/ext/host_budget.hpp"
#include "src/mem/account.hpp"

namespace gufo::server {

using ext::AutomaticRetainedBytes;
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
using ext::TextRunnerExpertCacheClaim;
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
    /// can already restore before it waits for a peer's shared checkpoint.
    /// Shorter gaps cost less to prefill again.
    static constexpr std::size_t kSharedPrefixMinTokens = 512;
    /// A prompt that diverges from a retained one after at least this many
    /// common tokens publishes a checkpoint there for later prompts. Lower
    /// than kSharedPrefixMinTokens: nobody waits for it, and a short system
    /// prompt shared by many short requests is worth one RAM copy.
    static constexpr std::size_t kSharedBranchMinTokens = 192;

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
    /// The session's shared-state account (#715), or null.
    [[nodiscard]] std::shared_ptr<mem::StateAccount> state_account() const;

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
  /// The scheduler is idle: nothing runs, is queued or waits. Called on its
  /// thread when it goes idle and about once a second while it stays idle,
  /// to return memory only running work holds (TextModelRunner::Idle()),
  /// and, when other processes leave the host short of its reserve, RAM-cache
  /// memory (ComponentTextCache::RelieveHostPressure). A runner that streams
  /// experts fills the host down to its expert store's floor, which then
  /// replaces the reserve: below it, the expert cache gives memory back
  /// first (Store::Release), and is restored once memory is back.
  void Idle() const;
  /// The host's shortfall below its reserve (ext::HostReserveShortfallBytes)
  /// when it is too short to admit new work: more than
  /// kAdmissionShortfallBytes (#932), so MemAvailable below 2 GiB without
  /// free CMA pages. Running work then finishes, the scheduler goes
  /// idle and the RAM cache gives memory back. A runner that streams experts
  /// refuses below half its expert store's floor instead.
  [[nodiscard]] std::optional<std::uint64_t> AdmissionShortfall() const;
  static constexpr std::uint64_t kAdmissionShortfallBytes =
      ext::kHostSnapshotHeadroomBytes / 2;
  /// Replaces the live host sample (ext::SampleHostMemory), for tests.
  void SetHostSampler(std::function<ext::HostMemorySample()> sampler);
  [[nodiscard]] std::size_t capacity() const noexcept;
  /// Validated resource claim, including host prompt-state admission inputs.
  [[nodiscard]] const TextRunnerResourceClaim& resource_claim() const noexcept;
  [[nodiscard]] TextExecutionPlan SelectDecodePlan(
      std::size_t ready_requests) const;
  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<Request*> requests, std::span<const std::size_t> max_tokens,
      const TextExecutionPlan& plan);
  std::vector<std::exception_ptr> AdvanceBatch(std::span<Request*> requests,
                                               const TextExecutionPlan& plan);
  /// State a request reserved ahead of its session (ReserveState).
  struct StateTicket {
    std::size_t tokens{0};
    mem::StateReservation reservation;
  };
  /// Runs a restore's copies, possibly on another thread, told how long they
  /// would stop the calling thread; see cache::RestoreCopyRunner.
  using RestoreCopyRunner =
      std::function<void(std::chrono::microseconds inline_estimate,
                         const std::function<void()>& copies)>;
  /// With `state`, the session the request lands on claims its tokens from
  /// the ticket's reservation. A restore runs its copies through
  /// `run_copies` when given.
  [[nodiscard]] Request Acquire(
      std::vector<TextRunnerToken> prompt,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      std::shared_ptr<const TextPromptContext> context = {},
      bool reuse_prompt = true, std::size_t cache_prefix_tokens = 0,
      bool stop_at_eos = true, bool persist = true,
      StateTicket* state = nullptr, const RestoreCopyRunner& run_copies = {});

  /// Whether sessions share their demand-mapped state (#715): more sessions
  /// are configured at full context than fit at once. Each request then
  /// claims the tokens it may write, and is admitted only when they fit.
  [[nodiscard]] bool shared_state() const noexcept;
  /// Whether a restore's copies may run beside decoders (`run_copies` in
  /// Acquire): only where they copy on a copy engine. Otherwise they wait
  /// behind decode work, slower than stopping it to restore.
  [[nodiscard]] bool restores_beside_decode() const noexcept;
  /// Generated tokens a request claims ahead, at admission and each time
  /// decoding nears its claim.
  static constexpr std::size_t kStateDecodeTokens = 8192;
  /// Tokens a request of `prompt_tokens` claims at admission: the prompt and
  /// up to kStateDecodeTokens of `max_new_tokens`, within the context and
  /// as far as the shared state could ever hold. Empty when not even the
  /// prompt and one token fit in it: the request can never run.
  [[nodiscard]] std::optional<std::size_t> StateClaimTokens(
      std::size_t prompt_tokens, std::size_t max_new_tokens) const;
  /// The longest prompt the shared state can hold with one generated token,
  /// at most one token short of the context.
  [[nodiscard]] std::size_t MaxPromptTokens() const;
  /// Reserves `tokens` of state for a request about to be admitted, first
  /// reclaiming idle sessions' state when short (never the frontier `prompt`
  /// continues). Empty when they do not fit now. Call where no model work is
  /// in flight.
  [[nodiscard]] std::optional<StateTicket> ReserveState(
      std::size_t tokens, std::span<const TextRunnerToken> prompt,
      const TextPromptContext* context);
  /// Raises a running request's claim to `tokens`, reclaiming idle
  /// sessions' state when short. False when it does not fit now.
  [[nodiscard]] bool ExtendState(Request& request, std::size_t tokens);
  /// Charged state bytes ReclaimState could free.
  [[nodiscard]] std::size_t ReclaimableStateBytes() const;
  [[nodiscard]] Request Acquire(std::vector<TextRunnerToken> prompt,
                                const CancellationCheck& is_cancelled = {});
  /// Longest RAM-retained prefix of prompt that Acquire could reuse, without
  /// leasing a state. An upper bound: Acquire may still prefer a shorter one.
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const TextRunnerToken> prompt,
      const TextPromptContext* context) const;
  /// Starts reading the disk-only checkpoint prompt would restore into the
  /// page cache; see ComponentTextCache::WarmDisk.
  [[nodiscard]] std::shared_ptr<const std::atomic<bool>> WarmDisk(
      std::span<const TextRunnerToken> prompt, const TextPromptContext* context,
      std::size_t stable,
      std::chrono::microseconds beside_after =
          std::chrono::microseconds::max()) const;
  /// A disk job is still writing a prefix of prompt much deeper than any
  /// restorable checkpoint; admission may wait for it instead of prefilling.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const TextRunnerToken> prompt, const TextPromptContext* context,
      std::uint64_t checkpoint_id = 0) const;
  /// When a text-only prompt repeats an idle live frontier's text, uses that
  /// frontier's tokens for it, so a different split of a previous answer does
  /// not lose the frontier (#1154). Moves `cache_prefix_tokens` with the
  /// rewrite. `chat_template` allows whitespace the template trimmed from
  /// the edges of a reply.
  bool AlignToLiveFrontier(std::vector<TextRunnerToken>& prompt,
                           std::size_t& cache_prefix_tokens,
                           bool chat_template) const;

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
