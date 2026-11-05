#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

std::optional<ChatRequest> ConstrainChatRequest(
    const ChatRequest& request, const TextModelRunner& runner,
    sampling::SamplingConfig* sampling,
    std::optional<sampling::JsonConstraint::ToolFormat>* tool_format) {
  // Output parsing needs the model's dialect even without tool constraints
  // (for example Qwen's whitespace boundary after </think>).
  if (tool_format)
    *tool_format = runner.ToolFormat();
  if (!request.response_format &&
      (request.tools.empty() ||
       request.tool_choice == ChatRequest::ToolChoice::kNone))
    return std::nullopt;
  auto constrained = request;
  auto instruction = request.response_format ? request.response_format->prompt()
                                             : std::string();
  auto grammar = request.response_format;
  if (!request.tools.empty() &&
      request.tool_choice != ChatRequest::ToolChoice::kNone) {
    std::vector<sampling::JsonConstraint::Tool> tools;
    const bool required =
        request.tool_choice == ChatRequest::ToolChoice::kRequired;
    auto format = runner.ToolFormat();
    for (const auto& tool : request.tools) {
      const auto definition = tool.definition_json.empty()
                                  ? json::Value()
                                  : json::parse(tool.definition_json);
      const auto* function = definition.find("function");
      const auto* strict = function ? function->find("strict") : nullptr;
      const bool enforce = strict && strict->as_bool();
      auto schema = json::parse(tool.parameters_json);
      auto native =
          sampling::JsonConstraint::ToolParameters(schema, enforce, format);
      tools.emplace_back(tool.name, std::move(native));
    }
    grammar = sampling::JsonConstraint::WithTools(
        grammar, std::move(tools), required,
        !request.response_format && request.parallel_tool_calls, format);
    if (format == sampling::JsonConstraint::ToolFormat::kJson)
      instruction +=
          "\nIf a tool is needed, respond using the JSON tool-call form "
          "<tool_call>{\"name\":\"function_name\",\"arguments\":{...}}</"
          "tool_call>. "
          "Tool arguments must follow the chosen function's schema.";
    if (request.response_format)
      instruction += " The JSON response schema applies to the final answer.";
  }
  if (!request.response_format_description.empty())
    instruction.insert(0, request.response_format_description + "\n\n");
  if (instruction.empty()) {
    // Native constraints follow the model's existing template. In particular
    // they do not change prompt tokens or invalidate continuation checkpoints.
  } else if (!constrained.messages.empty() &&
             (constrained.messages.front().role ==
                  tokenization::ChatRole::kSystem ||
              constrained.messages.front().role ==
                  tokenization::ChatRole::kDeveloper)) {
    constrained.messages.front().framing_suffix += "\n\n" + instruction;
  } else {
    tokenization::ChatMessage message{tokenization::ChatRole::kSystem, ""};
    message.framing_suffix = std::move(instruction);
    constrained.messages.insert(constrained.messages.begin(),
                                std::move(message));
  }
  if (runner.InitialOutputState(request) == ext::InitialOutputState::kReasoning)
    grammar = sampling::JsonConstraint::WithReasoning(grammar);
  sampling->constraint = runner.BindConstraint(grammar);
  return constrained;
}

namespace {

struct ValidatedRunner {
  std::shared_ptr<TextModelRunner> runner;
  TextRunnerDescriptor descriptor;
  TextRunnerResourceClaim resources;
  std::vector<TextExecutionPlan> plans;
};

std::optional<std::size_t> PerStateReservationBytes(
    const TextRunnerResourceClaim& resources) {
  if (!resources.per_request_state_bytes.has_value() ||
      !resources.temporary_scratch_bytes.has_value()) {
    return std::nullopt;
  }
  if (*resources.per_request_state_bytes >
      std::numeric_limits<std::size_t>::max() -
          *resources.temporary_scratch_bytes) {
    throw std::invalid_argument("text runner resource claim overflows");
  }
  return *resources.per_request_state_bytes +
         *resources.temporary_scratch_bytes;
}

ValidatedRunner ValidateRunner(std::shared_ptr<TextModelRunner> runner,
                               std::size_t state_count) {
  if (runner == nullptr) {
    throw std::invalid_argument("text model runner must not be null");
  }
  if (state_count == 0) {
    throw std::invalid_argument("text runner state count must be at least one");
  }

  auto descriptor = runner->Descriptor();
  if (descriptor.model_id.empty()) {
    throw std::invalid_argument("text runner model ID must not be empty");
  }
  if (descriptor.state_abi.empty()) {
    throw std::invalid_argument("text runner state ABI must not be empty");
  }
  if (descriptor.max_context == 0) {
    throw std::invalid_argument(
        "text runner maximum context must be at least one token");
  }
  if (descriptor.capabilities.fork && !descriptor.capabilities.snapshot) {
    throw std::invalid_argument(
        "text runner fork capability requires snapshot support");
  }
  if (descriptor.persistence.has_value()) {
    if (!descriptor.capabilities.snapshot || !descriptor.capabilities.fork) {
      throw std::invalid_argument(
          "text runner persistence requires snapshot and fork support");
    }
    if (descriptor.persistence->compatibility_identity.empty()) {
      throw std::invalid_argument(
          "text runner persistence identity must not be empty");
    }
    if (descriptor.persistence->payload_version == 0) {
      throw std::invalid_argument(
          "text runner persistence payload version must be nonzero");
    }
  }

  auto resources = runner->ResourceClaim();
  const auto per_state_reservation = PerStateReservationBytes(resources);
  if (resources.state_capacity_bytes.has_value() &&
      per_state_reservation.has_value()) {
    const std::size_t per_request = *per_state_reservation;
    const std::size_t capacity = *resources.state_capacity_bytes;
    if (per_request != 0 && state_count > capacity / per_request) {
      const auto describe = [](std::size_t bytes) {
        std::ostringstream text;
        text << bytes << " bytes (" << std::fixed << std::setprecision(1)
             << static_cast<double>(bytes) / static_cast<double>(1ULL << 30)
             << " GiB)";
        return text.str();
      };
      const std::size_t fitting = capacity / per_request;
      std::ostringstream message;
      message << "text runner request-state claim exceeds state capacity: "
              << state_count << " sessions at " << descriptor.max_context
              << "-token context need " << describe(per_request)
              << " each, but only " << describe(capacity)
              << " of device memory is available; ";
      if (fitting == 0) {
        message << "no session fits at this context, reduce --context";
      } else {
        message << "at most " << fitting
                << (fitting == 1 ? " session fits" : " sessions fit")
                << " at this context, reduce --sessions or --context";
      }
      throw std::invalid_argument(message.str());
    }
  }

  auto plans = runner->SupportedPlans();
  if (plans.empty()) {
    throw std::invalid_argument(
        "text runner must expose at least one execution plan");
  }
  bool supports_serial_single = false;
  for (std::size_t index = 0; index < plans.size(); ++index) {
    const auto& plan = plans[index];
    if (plan.physical_width == 0) {
      throw std::invalid_argument(
          "text runner execution plan width must be at least one");
    }
    if ((plan.kind == TextExecutionPlanKind::kSerial &&
         plan.physical_width != 1) ||
        (plan.kind == TextExecutionPlanKind::kBatched &&
         plan.physical_width < 2)) {
      throw std::invalid_argument(
          "text runner execution plan kind and width are inconsistent");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (plans[previous] == plan) {
        throw std::invalid_argument(
            "text runner execution plans must be unique");
      }
    }
    supports_serial_single = supports_serial_single ||
                             (plan.kind == TextExecutionPlanKind::kSerial &&
                              plan.physical_width == 1);
  }
  if (!supports_serial_single) {
    throw std::invalid_argument(
        "text runner must support the serial single-request plan");
  }

  return {
      .runner = std::move(runner),
      .descriptor = std::move(descriptor),
      .resources = resources,
      .plans = std::move(plans),
  };
}

}  // namespace

struct TextRunnerPool::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner, std::size_t state_count,
       std::optional<TextRunnerDiskCacheOptions> disk_cache_options,
       TextRunnerRamCacheOptions ram_cache_options)
      : validated(ValidateRunner(std::move(model_runner), state_count)) {
    if (!validated.runner->UsesComponentCache())
      throw std::invalid_argument(
          "text runner does not provide a component cache adapter");
    {
      const auto resources = validated.runner->ResourceClaim();
      const auto automatic =
          resources.retained_snapshot_capacity_bytes.value_or(0);
      const auto ceiling =
          resources.retained_snapshot_ceiling_bytes.value_or(automatic);
      const auto requested =
          ram_cache_options.capacity_bytes
              ? ram_cache_options.capacity_bytes
              : std::min(TextRunnerRamCacheOptions::kAutomaticMaxBytes,
                         automatic);
      // This is an upper bound. Providers commit backing only after their
      // slots allocate and the cache observes remaining host capacity. Slots
      // allocate under the ceiling; what they pin is charged on top of this.
      // The state claim includes lazy image/rollback storage, not startup
      // usage.
      component_cache = std::make_unique<ComponentTextCache>(
          validated.runner, state_count, std::min(requested, ceiling),
          disk_cache_options, !ram_cache_options.capacity_bytes, ceiling,
          ram_cache_options.enabled);
      if (disk_cache_options) {
        shared_prefix_min_tokens = disk_cache_options->shared_prefix_min_tokens;
        shared_prefix_max_boundaries =
            disk_cache_options->shared_prefix_max_boundaries;
      }
    }
  }

  ValidatedRunner validated;
  std::unique_ptr<ComponentTextCache> component_cache;
  // Publish a live checkpoint before another admission can fall back to the
  // older prompt snapshot. Decode/prefill work never holds this mutex.
  std::timed_mutex admission_mutex;
  std::size_t shared_prefix_min_tokens{0};
  std::size_t shared_prefix_max_boundaries{0};
};

struct TextRunnerPool::Request::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner,
       std::unique_ptr<ComponentTextCache::Lease> state_lease,
       std::vector<TextRunnerToken> prompt_tokens,
       const sampling::SamplingConfig& sampling_config,
       std::shared_ptr<const TextPromptContext> prompt_context,
       std::size_t cache_prefix_tokens, bool overlap_snapshots)
      : runner(std::move(model_runner)),
        lease(std::move(state_lease)),
        prompt(std::move(prompt_tokens)),
        prefill_offset(lease->cached_tokens()),
        decode_ready(prefill_offset == prompt.size()),
        sampler(sampling_config, prompt),
        context(std::move(prompt_context)),
        retain_fallback(cache_prefix_tokens != 0),
        overlap_snapshots(overlap_snapshots) {
    alignment = std::max<std::size_t>(
        1, runner->Descriptor().capabilities.prefill_alignment);
    // On a warm continuation the reused frontier is already a safe fallback.
    // Freeze it before prefill, then retain this turn's own stable boundary
    // before mutable assistant framing.
    auto count = cache_prefix_tokens == 0 ? prompt.size() : cache_prefix_tokens;
    // A newly consumed image needs its own stable checkpoint. Otherwise the
    // next assistant turn can invalidate the complete prompt's framing and
    // force the image through prefill again from the older text frontier.
    const bool new_images =
        context && !std::ranges::equal(context->CacheIdentity(prefill_offset),
                                       context->CacheIdentity(count));
    // A component capture copies the whole private state, so freezing a live
    // frontier just before a nearby stable boundary doubles the synchronous
    // cost of each tool-loop turn. Only an edit inside that gap would need
    // the frontier; it falls back to an earlier boundary instead.
    constexpr std::size_t kNearbyStableTokens = 64;
    const bool nearby_stable =
        prefill_offset < count && count - prefill_offset <= kNearbyStableTokens;
    if (retain_fallback && lease->cache_hit() && prefill_offset <= count &&
        !new_images && !nearby_stable) {
      // Freezing the reused frontier must not cost this turn its own stable
      // boundary. A client that rewrites the assistant turn, as one dropping
      // reasoning does, diverges after that boundary, so retaining only the
      // frontier pins every later turn to the same early position and the
      // re-prefilled tail grows for the rest of the conversation.
      if (prefill_offset < count)
        stable_prefix_position = count;
      count = prefill_offset;
    }
    if (lease->restored_from_disk() && prefill_offset == prompt.size())
      count = prompt.size();
    // Cancellation can leave a component live frontier past the stable
    // checkpoint. That earlier state cannot be captured here; keep its
    // immutable fallback and freeze the complete prompt before decoding.
    if (prefill_offset > count)
      count = prompt.size();
    // Capture the aligned boundary at or before each later target instead:
    // only it continues the prompt exactly (prefill_alignment). A reused
    // frontier is already the state this request continues. The complete
    // prompt stays exact: an unchanged retry continues it without prefill,
    // and a longer prompt never restores it (ComponentTextCache::Acquire).
    if (alignment > 1) {
      stable_prefix_position = Aligned(stable_prefix_position);
      if (stable_prefix_position <= prefill_offset)
        stable_prefix_position = 0;
      // Raw completions have no earlier stable chat boundary. Retain the
      // aligned prompt tail too, or a longer request loses all reuse when
      // Acquire rejects the exact unaligned prompt checkpoint. Advancing this
      // target still captures the complete prompt for unchanged retries.
      if (count > prefill_offset && count < prompt.size()) {
        count = std::max(Aligned(count), prefill_offset);
      } else if (const auto aligned = Aligned(count);
                 aligned > prefill_offset && aligned < count &&
                 lease->TryReserveSnapshot(
                     lease->CaptureBytes(aligned), aligned, retain_fallback,
                     SnapshotPurpose::kContinuation,
                     std::span<const TextRunnerToken>(prompt).first(aligned))) {
        // A disabled cache (or a refused disk-only capture) must not prepare
        // checkpoint scratch or split a pass for a fallback it cannot keep.
        count = aligned;
      }
    }
    snapshot_tokens.reserve(prompt.size());
    snapshot_tokens.assign(prompt.begin(), prompt.begin() + count);
    // A live continuation can already be beyond the desired snapshot point.
    // Keep its existing immutable checkpoint; never snapshot mismatched state.
    prompt_snapshot_attempted = prefill_offset > count;
    auto& state = dynamic_cast<TextRunnerState&>(lease->state());
    if (lease->cache_hit()) {
      runner->PreparePrefixReuse(
          state,
          std::span<const TextRunnerToken>(prompt).first(prefill_offset));
    }
    ngram_baseline = runner->NgramRows(state);
    checkpoints = lease->PlannedCaptures(cache_prefix_tokens);
    if (alignment > 1) {
      for (auto& position : checkpoints)
        position = Aligned(position);
      std::erase_if(checkpoints, [&](std::size_t position) {
        return position <= prefill_offset;
      });
      std::ranges::sort(checkpoints);
      checkpoints.erase(std::ranges::unique(checkpoints).begin(),
                        checkpoints.end());
    }
  }

  [[nodiscard]] std::size_t Aligned(std::size_t position) const noexcept {
    return position - position % alignment;
  }

  ~Impl() {
    // The worker borrows this lease's state; join before any field is
    // destroyed.
    if (snapshot_future.valid())
      snapshot_future.wait();
  }

  void CapturePromptSnapshot(bool wait = true) noexcept {
    if (boundary_capture) {
      if (!wait && snapshot_future.wait_for(std::chrono::seconds(0)) !=
                       std::future_status::ready)
        return;
      FinishSnapshot();
    }
    if (!prompt_snapshot_attempted)
      StartPromptSnapshot(wait);
    if (snapshot_future.valid()) {
      if (!wait && snapshot_future.wait_for(std::chrono::seconds(0)) !=
                       std::future_status::ready)
        return;
      FinishSnapshot();
    }
    if (prompt_snapshot_attempted && prefill_offset == snapshot_tokens.size() &&
        snapshot_tokens.size() < prompt.size()) {
      AdvancePromptTarget();
    }
  }

  void AdvancePromptTarget() noexcept {
    // Keep both the branching fallback and the complete prompt. The latter
    // is captured after first-token publication, as for ordinary requests.
    // Admission may refuse it without discarding the fallback.
    if (prompt_snapshot && retain_snapshot) {
      try {
        snapshot_metrics.snapshot_bytes +=
            lease->PublishSnapshot(snapshot_tokens, std::move(prompt_snapshot),
                                   fallback_position != 0);
      } catch (...) {
        lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure,
                            snapshot_bytes, snapshot_tokens.size());
      }
    }
    prompt_snapshot.reset();
    retain_snapshot = false;
    fallback_position = snapshot_tokens.size();
    // Take this turn's stable boundary next, before the complete prompt.
    // The boundary is a prefix of whatever the client sends next, while the
    // complete prompt ends in assistant framing the client may rewrite.
    if (stable_prefix_position > snapshot_tokens.size() &&
        stable_prefix_position < prompt.size()) {
      snapshot_tokens.assign(prompt.begin(),
                             prompt.begin() + stable_prefix_position);
      stable_prefix_position = 0;
    } else {
      snapshot_tokens = prompt;
    }
    prompt_snapshot_attempted = false;
  }

  void StartPromptSnapshot(bool wait) noexcept {
    if (prompt_snapshot_attempted || prefill_offset != snapshot_tokens.size()) {
      return;
    }
    prompt_snapshot_attempted = true;

    TextRunnerCapabilities capabilities;
    try {
      capabilities = runner->Descriptor().capabilities;
    } catch (...) {
      return;
    }
    if (!capabilities.prefix_reuse || !capabilities.snapshot ||
        !capabilities.fork) {
      return;
    }
    if (lease->cache_hit() && !lease->restored_from_disk() &&
        lease->cached_tokens() == snapshot_tokens.size() &&
        lease->HasSnapshotFor(snapshot_tokens)) {
      return;
    }

    const TextRunnerState* state = nullptr;
    try {
      state = &dynamic_cast<const TextRunnerState&>(lease->state());
    } catch (...) {
      return;
    }
    snapshot_bytes = 0;
    try {
      if (runner->CheckpointPosition(*state) != snapshot_tokens.size()) {
        lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                            snapshot_tokens.size());
        return;
      }
      snapshot_bytes = lease->CaptureBytes();
    } catch (...) {
      lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                          snapshot_tokens.size());
      return;
    }

    retain_snapshot = false;
    try {
      retain_snapshot = lease->TryReserveSnapshot(
          snapshot_bytes, snapshot_tokens.size(),
          retain_fallback && (fallback_position == 0 ||
                              snapshot_tokens.size() == prompt.size()),
          retain_fallback && snapshot_tokens.size() == prompt.size()
              ? SnapshotPurpose::kRetry
              : SnapshotPurpose::kContinuation,
          snapshot_tokens);
    } catch (...) {
      return;
    }
    if (!retain_snapshot)
      return;

    snapshot_start = std::chrono::steady_clock::now();
    try {
      // A warm prefill must await its fallback before mutating recurrent
      // state. Run that copy on this worker instead of starting a thread
      // only to join it. Post-token captures remain asynchronous.
      snapshot_future =
          std::async(wait ? std::launch::deferred : std::launch::async,
                     [this] { return lease->Capture(); });
    } catch (...) {
      lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
                          snapshot_tokens.size());
    }
  }

  void FinishPromptSnapshot() noexcept {
    try {
      prompt_snapshot = snapshot_future.get();
    } catch (...) {
      prompt_snapshot.reset();
    }
    snapshot_metrics.snapshot_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - snapshot_start)
            .count();
    if (prompt_snapshot == nullptr) {
      if (retain_snapshot) {
        lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure,
                            snapshot_bytes, snapshot_tokens.size());
      }
      return;
    }

    if (retain_snapshot && prompt_snapshot->PayloadBytes() != snapshot_bytes) {
      lease->SkipSnapshot(SnapshotEventReason::kReservationMismatch,
                          prompt_snapshot->PayloadBytes(),
                          snapshot_tokens.size());
      retain_snapshot = false;
    }
    if (!retain_snapshot)
      prompt_snapshot.reset();
  }

  void FinishSnapshot() noexcept {
    if (!boundary_capture) {
      FinishPromptSnapshot();
      return;
    }
    try {
      (void)snapshot_future.get();
    } catch (...) {
      // The boundary worker handles failed capture/admission internally.
    }
    boundary_capture = false;
  }

  void CaptureBoundarySnapshot(
      std::size_t position,
      SnapshotPurpose purpose = SnapshotPurpose::kHistory) noexcept {
    if (!overlap_snapshots) {
      // A single execution slot has no peer work to overlap with the copy.
      CaptureBoundarySnapshotNow(position, purpose);
      return;
    }
    // The scheduler already parks leases with a pending snapshot. Reuse that
    // path so allocation, eviction and copying do not stall unrelated users.
    // Direct callers and cancellation join before touching this lease again.
    try {
      snapshot_future =
          std::async(std::launch::async, [this, position, purpose] {
            CaptureBoundarySnapshotNow(position, purpose);
            return std::unique_ptr<TextRunnerSnapshot>{};
          });
      boundary_capture = true;
    } catch (...) {
      // Optional caching must not fail the request if a worker cannot start.
    }
  }

  void CaptureBoundarySnapshotNow(
      std::size_t position,
      SnapshotPurpose purpose = SnapshotPurpose::kHistory) noexcept {
    const auto started = std::chrono::steady_clock::now();
    std::size_t bytes = 0;
    bool reserved = false;
    try {
      const auto capabilities = runner->Descriptor().capabilities;
      if (!capabilities.prefix_reuse || !capabilities.snapshot ||
          !capabilities.fork)
        return;
      const auto& state = dynamic_cast<const TextRunnerState&>(lease->state());
      if (runner->CheckpointPosition(state) != position)
        return;
      const auto prefix =
          std::span<const TextRunnerToken>(prompt).first(position);
      bytes = lease->CaptureBytes();
      // Intermediate copies must not displace the frontier this request
      // branched from, including its stable image/reasoning fallback.
      // Copies other requests depend on compete like continuation
      // boundaries; the rest give way under pressure.
      if (lease->HasSnapshotFor(prefix))
        return;
      reserved = lease->TryReserveSnapshot(bytes, position, true, purpose);
      if (!reserved)
        return;
      std::shared_ptr<const TextRunnerSnapshot> snapshot = lease->Capture();
      if (snapshot) {
        snapshot_metrics.snapshot_bytes += lease->PublishSnapshot(
            {prefix.begin(), prefix.end()}, snapshot, true);
        reserved = false;
      }
    } catch (...) {
    }
    if (reserved)
      lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure, bytes,
                          position);
    snapshot_metrics.snapshot_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
  }

  /// Retains an intermediate checkpoint at `position` while the runner
  /// exposes it inside a prefill pass (component leases only).
  void CaptureInPassCheckpoint(std::size_t position) noexcept {
    std::size_t bytes = 0;
    bool reserved = false;
    try {
      const auto prefix =
          std::span<const TextRunnerToken>(prompt).first(position);
      if (lease->HasSnapshotFor(prefix))
        return;
      const bool shared_with_peers =
          std::ranges::find(shared_checkpoints, position) !=
          shared_checkpoints.end();
      bytes = lease->CaptureBytes();
      reserved = lease->TryReserveSnapshot(bytes, position, true,
                                           shared_with_peers
                                               ? SnapshotPurpose::kBranchPoint
                                               : SnapshotPurpose::kHistory);
      if (!reserved)
        return;
      std::shared_ptr<const TextRunnerSnapshot> snapshot = lease->Capture();
      if (snapshot) {
        snapshot_metrics.snapshot_bytes += lease->PublishSnapshot(
            {prefix.begin(), prefix.end()}, snapshot, true);
        reserved = false;
      }
    } catch (...) {
    }
    if (reserved)
      lease->SkipSnapshot(SnapshotEventReason::kCaptureFailure, bytes,
                          position);
  }

  std::shared_ptr<TextModelRunner> runner;
  std::unique_ptr<ComponentTextCache::Lease> lease;
  std::vector<TextRunnerToken> prompt;
  std::vector<TextRunnerToken> snapshot_tokens;
  /// Bounded intermediate positions retained in RAM before state advances.
  std::vector<std::size_t> checkpoints;
  /// Checkpoints shared with other requests, retained like continuations.
  std::vector<std::size_t> shared_checkpoints;
  bool boundary_capture{false};
  std::vector<TextRunnerToken> generated;
  std::size_t prefill_offset{0};
  bool decode_ready{false};
  /// The leased state's cumulative n-gram rows before this request ran.
  TextRunnerNgramRows ngram_baseline;
  // A throwing model call may have partially mutated device state. Never
  // publish that state as a live continuation, even if its position is stale.
  bool state_reusable{true};
  // The caller's cancellation poll. Checked before each prefill step, and
  // re-armed after a step that runs to completion (see Request::Prefill), so
  // a request cancelled between steps keeps the frontier it reached.
  CancellationCheck cancelled;
  bool stopped{false};
  std::optional<TextDecodeSelection> pending_selection;
  sampling::SamplerState sampler;
  std::shared_ptr<const TextPromptContext> context;
  bool retain_fallback{false};
  bool overlap_snapshots{false};
  std::size_t fallback_position{0};
  /// TextRunnerCapabilities::prefill_alignment.
  std::size_t alignment{1};
  /// This turn's stable boundary when the reused frontier sits before it, so
  /// the boundary is checkpointed after the frozen fallback instead of being
  /// skipped. Zero once taken, or when the two positions coincide.
  std::size_t stable_prefix_position{0};
  [[nodiscard]] std::span<const std::uint8_t> InputIdentity(
      std::size_t token_count) const {
    return context ? context->CacheIdentity(token_count)
                   : std::span<const std::uint8_t>{};
  }
  bool prompt_snapshot_attempted{false};
  std::shared_ptr<const TextRunnerSnapshot> prompt_snapshot;
  TextRunnerPool::Request::CommitMetrics snapshot_metrics;
  std::size_t snapshot_bytes{0};
  bool retain_snapshot{false};
  std::chrono::steady_clock::time_point snapshot_start;
  std::future<std::unique_ptr<TextRunnerSnapshot>> snapshot_future;
};

TextRunnerPool::Request::Request() = default;

TextRunnerPool::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

void TextRunnerPool::Request::CapturePromptSnapshot() {
  if (impl_)
    impl_->CapturePromptSnapshot();
}

bool TextRunnerPool::Request::PreparePromptSnapshot() {
  if (impl_)
    impl_->CapturePromptSnapshot(false);
  return !impl_ || !impl_->snapshot_future.valid();
}

bool TextRunnerPool::Request::SnapshotPending() const {
  return impl_ && impl_->snapshot_future.valid() &&
         impl_->snapshot_future.wait_for(std::chrono::seconds(0)) !=
             std::future_status::ready;
}

std::optional<TextDecodeSelection>
TextRunnerPool::Request::PreviewFirstToken() {
  if (!impl_ || !impl_->decode_ready || !impl_->generated.empty() ||
      impl_->pending_selection || impl_->stopped)
    return std::nullopt;
  auto sampler = impl_->sampler;
  return impl_->runner->PreviewFirstToken(
      dynamic_cast<TextRunnerState&>(impl_->lease->state()), sampler);
}

TextRunnerPool::Request::~Request() = default;

TextRunnerPool::Request::Request(Request&&) noexcept = default;

TextRunnerPool::Request& TextRunnerPool::Request::operator=(
    Request&&) noexcept = default;

TextRunnerPool::Request::operator bool() const noexcept {
  return impl_ != nullptr && static_cast<bool>(impl_->lease);
}

bool TextRunnerPool::Request::cache_hit() const noexcept {
  return impl_ != nullptr && impl_->lease->cache_hit();
}

std::size_t TextRunnerPool::Request::cached_prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->lease->cached_tokens() : 0;
}

ContinuationLookup TextRunnerPool::Request::cache_lookup() const noexcept {
  return impl_ != nullptr ? impl_->lease->lookup() : ContinuationLookup{};
}

std::size_t TextRunnerPool::Request::cache_restore_bytes() const noexcept {
  return impl_ != nullptr ? impl_->lease->restored_snapshot_bytes() : 0;
}

double TextRunnerPool::Request::cache_restore_ms() const noexcept {
  return impl_ != nullptr ? impl_->lease->restore_ms() : 0.0;
}

TextRunnerNgramRows TextRunnerPool::Request::ngram_rows() const {
  if (impl_ == nullptr)
    return {};
  const auto now = impl_->runner->NgramRows(
      dynamic_cast<const TextRunnerState&>(impl_->lease->state()));
  const auto& base = impl_->ngram_baseline;
  return {now.rows - base.rows, now.dedup_rows - base.dedup_rows,
          now.cache_hits - base.cache_hits, now.disk_reads - base.disk_reads};
}

bool TextRunnerPool::Request::cache_disk_hit() const noexcept {
  return impl_ != nullptr && impl_->lease->restored_from_disk();
}

std::optional<ComponentCacheMetrics>
TextRunnerPool::Request::component_cache_metrics() const {
  std::optional<ComponentCacheMetrics> report;
  if (impl_)
    report = impl_->lease->Metrics();
  if (report)
    report->prefilled = impl_->prefill_offset - impl_->lease->cached_tokens();
  return report;
}

std::size_t TextRunnerPool::Request::prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->prompt.size() : 0;
}

bool TextRunnerPool::Request::prefill_complete() const noexcept {
  return impl_ != nullptr && impl_->decode_ready;
}

std::span<const TextRunnerToken> TextRunnerPool::Request::prompt()
    const noexcept {
  return impl_ != nullptr ? std::span<const TextRunnerToken>(impl_->prompt)
                          : std::span<const TextRunnerToken>{};
}

std::size_t TextRunnerPool::Request::prefill_position() const noexcept {
  return impl_ != nullptr ? impl_->prefill_offset : 0;
}

std::span<const std::uint8_t> TextRunnerPool::Request::input_identity(
    std::size_t token_count) const {
  return impl_ != nullptr ? impl_->InputIdentity(token_count)
                          : std::span<const std::uint8_t>{};
}

std::size_t TextRunnerPool::Request::ShareCheckpoint(
    std::size_t common_tokens) {
  if (!*this || impl_->decode_ready || impl_->stopped)
    return 0;
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  if (!capabilities.incremental_prefill || !capabilities.prefix_reuse ||
      !capabilities.snapshot || !capabilities.fork)
    return 0;
  // The complete prompt is retained only after decoding starts; share the
  // last prefill position instead so peers can start sooner.
  const auto target =
      impl_->Aligned(std::min(common_tokens, impl_->prompt.size() - 1));
  const auto offset = impl_->prefill_offset;
  if (target <= offset)
    return 0;
  // A peer can arrive while the stable-boundary copy is pending, or just
  // after it was published. Reuse that nearby frontier instead of inserting
  // another pass a few tokens later (for example, splitting framing 6 + 1).
  if (offset && target - offset <= kSharedCheckpointSlack) {
    const bool capturing =
        impl_->snapshot_future.valid() &&
        (impl_->boundary_capture || impl_->snapshot_tokens.size() == offset);
    const bool published =
        impl_->fallback_position == offset &&
        impl_->lease->HasSnapshotFor(
            std::span<const TextRunnerToken>(impl_->prompt).first(offset));
    if (capturing || published)
      return offset;
  }
  // Prefill already publishes the fallback, this turn's stable boundary and
  // intermediate checkpoints before advancing past them.
  std::size_t planned = 0;
  const auto consider = [&](std::size_t position) {
    if (position > offset && position <= target && position > planned)
      planned = position;
  };
  if (impl_->snapshot_tokens.size() < impl_->prompt.size())
    consider(impl_->snapshot_tokens.size());
  consider(impl_->stable_prefix_position);
  for (const auto position : impl_->checkpoints)
    consider(position);
  auto& checkpoints = impl_->checkpoints;
  const auto position =
      planned != 0 && target - planned <= kSharedCheckpointSlack ? planned
                                                                 : target;
  if (position == target && !std::ranges::binary_search(checkpoints, target))
    checkpoints.insert(std::ranges::upper_bound(checkpoints, target), target);
  // Prompt snapshots are retained as continuation boundaries already.
  if (std::ranges::binary_search(checkpoints, position))
    impl_->shared_checkpoints.push_back(position);
  return position;
}

void TextRunnerPool::Request::PrepareBatchExecution() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  impl_->runner->PrepareBatchExecution(
      dynamic_cast<TextRunnerState&>(impl_->lease->state()));
}

TextPrefillStep TextRunnerPool::Request::Prefill(std::size_t max_input_tokens) {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner cannot prefill with a pending decode token");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }
  if (impl_->decode_ready) {
    return {
        .consumed_tokens = 0,
        .decode_ready = true,
    };
  }
  if (max_input_tokens == 0) {
    throw std::invalid_argument(
        "text runner prefill budget must be at least one token");
  }

  // A step after this request's own steps starts on the alignment: a pass
  // continuing an unaligned prompt pass would not match one pass over the
  // prompt (Flash-Next's sparse tiles group queries from the pass start).
  assert(impl_->prefill_offset == impl_->lease->cached_tokens() ||
         impl_->prefill_offset % impl_->alignment == 0);
  // Direct callers join any capture before mutating its borrowed state.
  impl_->CapturePromptSnapshot();
  const auto snapshot_position = impl_->snapshot_tokens.size();
  while (!impl_->checkpoints.empty() &&
         impl_->checkpoints.front() <= impl_->prefill_offset)
    impl_->checkpoints.erase(impl_->checkpoints.begin());
  // A budget-limited pass ends on an aligned position, so the next pass
  // continues the prompt exactly: shorter, or a budget below the alignment
  // runs to the next one. Every capture boundary is aligned already, so
  // neither crosses one.
  if (const auto end = impl_->prefill_offset + max_input_tokens;
      end < impl_->prompt.size() && end % impl_->alignment != 0) {
    const auto over = end % impl_->alignment;
    max_input_tokens =
        max_input_tokens > over
            ? max_input_tokens - over
            : std::min(max_input_tokens + impl_->alignment - over,
                       impl_->prompt.size() - impl_->prefill_offset);
  }

  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease->state());
  // Captures this chunk takes in place, ascending, without ending a pass
  // there: intermediate checkpoints and the prompt snapshot target. The
  // first boundary the runner cannot capture in place ends the chunk and is
  // captured after it, as are all of them for other runners.
  struct InPassCapture {
    std::size_t position;
    bool prompt;  ///< the prompt snapshot target, else a checkpoint
  };
  std::vector<InPassCapture> in_pass;
  bool in_pass_prompt = false;
  std::size_t stop = impl_->prompt.size();
  const bool in_pass_capable =
      impl_->runner->Descriptor().capabilities.in_pass_checkpoint;
  const bool prompt_pending = !impl_->prompt_snapshot_attempted &&
                              impl_->prefill_offset < snapshot_position &&
                              snapshot_position < impl_->prompt.size();
  if (in_pass_capable) {
    // The prompt targets this pass may reach in turn: the current one, then
    // the stable boundary AdvancePromptTarget moves to after a frozen
    // fallback. The complete prompt follows the pass.
    std::vector<std::size_t> targets;
    if (prompt_pending) {
      targets.push_back(snapshot_position);
      const auto stable = impl_->stable_prefix_position;
      if (stable > snapshot_position && stable < impl_->prompt.size())
        targets.push_back(stable);
    }
    std::vector<InPassCapture> candidates;
    for (const auto position : impl_->checkpoints)
      if (std::ranges::find(targets, position) == targets.end())
        candidates.push_back({position, false});
    for (const auto target : targets)
      candidates.insert(std::ranges::upper_bound(candidates, target, {},
                                                 &InPassCapture::position),
                        {target, true});
    const auto max_in_pass = std::max<std::size_t>(
        1, impl_->runner->Descriptor().capabilities.max_in_pass_checkpoints);
    for (const auto& candidate : candidates) {
      if (candidate.position >= impl_->prefill_offset + max_input_tokens)
        break;
      // The runner captures a bounded number of boundaries per call; the
      // next one ends this chunk and is captured after it.
      if (in_pass.size() == max_in_pass) {
        stop = candidate.position;
        break;
      }
      auto bytes = impl_->runner->PrefillCheckpointBytes(
          state, impl_->prompt, impl_->prefill_offset, max_input_tokens,
          candidate.position);
      if (bytes && candidate.position == snapshot_position) {
        // Admission happens before the pass, as for the prompt snapshot
        // captured after one; later targets are admitted when captured.
        bytes = impl_->lease->CaptureBytes(snapshot_position);
        if (impl_->lease->TryReserveSnapshot(
                *bytes, snapshot_position,
                impl_->retain_fallback && impl_->fallback_position == 0,
                SnapshotPurpose::kContinuation, impl_->snapshot_tokens)) {
          impl_->snapshot_bytes = *bytes;
          impl_->retain_snapshot = true;
          impl_->prompt_snapshot_attempted = true;
          in_pass_prompt = true;
        } else {
          bytes.reset();
        }
      }
      // A later prompt target is taken in place only after the ones
      // before it.
      if (!bytes || (candidate.prompt && !in_pass_prompt)) {
        stop = candidate.position;
        break;
      }
      in_pass.push_back(candidate);
    }
  } else if (!impl_->checkpoints.empty()) {
    stop = impl_->checkpoints.front();
  }
  if (stop < impl_->prefill_offset + max_input_tokens)
    max_input_tokens = stop - impl_->prefill_offset;
  if (!in_pass_prompt && impl_->prefill_offset < snapshot_position)
    max_input_tokens =
        std::min(max_input_tokens, snapshot_position - impl_->prefill_offset);

  const std::size_t remaining = impl_->prompt.size() - impl_->prefill_offset;
  if (impl_->cancelled && impl_->cancelled())
    throw std::runtime_error("generation cancelled");
  impl_->state_reusable = false;
  // Complete the model frontier at the cache boundary, including logits and
  // draft catch-up. Its snapshot can then be restored independently.
  auto frontier = !in_pass_prompt && impl_->prefill_offset < snapshot_position
                      ? snapshot_position
                      : impl_->prompt.size();
  frontier = std::min(frontier, stop);
  const auto model_prompt =
      std::span<const TextRunnerToken>(impl_->prompt).first(frontier);
  // Prompt targets captured in place, in order, with the bytes each
  // reserved. They are published after the step, as AdvancePromptTarget
  // reaches each of them.
  struct PromptCapture {
    std::size_t position;
    std::size_t bytes;
    std::unique_ptr<TextRunnerSnapshot> snapshot;
    /// Its reservation keeps the request's fallback.
    bool preserve{false};
    /// The cache refused the reservation; nothing was captured.
    bool refused{false};
  };
  std::vector<PromptCapture> prompt_captures;
  // In-pass checkpoint `index`, read while the runner exposes it. Optional
  // retention may refuse it; the live prefill stays valid either way.
  const auto consume = [&](std::size_t index,
                           std::unique_ptr<TextRunnerSnapshot> native) {
    const auto& capture = in_pass.at(index);
    if (!capture.prompt) {
      if (!native)
        impl_->CaptureInPassCheckpoint(capture.position);
      return;
    }
    PromptCapture taken{capture.position, impl_->snapshot_bytes,
                        std::move(native)};
    // Only the first target keeps the request's fallback.
    taken.preserve = prompt_captures.empty() && impl_->retain_fallback &&
                     impl_->fallback_position == 0;
    if (!taken.snapshot) {
      try {
        // A checkpoint captured before it in this pass took the lease's
        // reservation and shares rows with it; reserve what this capture
        // adds now.
        taken.bytes = impl_->lease->CaptureBytes();
        const auto tokens = std::span<const TextRunnerToken>(impl_->prompt)
                                .first(capture.position);
        taken.refused = !impl_->lease->TryReserveSnapshot(
            taken.bytes, capture.position, taken.preserve,
            SnapshotPurpose::kContinuation, tokens);
        if (!taken.refused)
          taken.snapshot = impl_->lease->Capture();
      } catch (...) {
        taken.snapshot.reset();
      }
    }
    prompt_captures.push_back(std::move(taken));
  };
  // A runner must either roll back an interrupted step or finish it before
  // cancellation can retain the completed conversation frontier.
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  const bool finish_step = capabilities.partial_prefill_retention &&
                           !capabilities.prefill_rollback && impl_->cancelled;
  struct Rearm {
    TextRunnerState* state;
    const CancellationCheck* check;
    ~Rearm() {
      if (state != nullptr)
        state->SetCancellationCheck(*check);
    }
  } rearm{finish_step ? &state : nullptr, &impl_->cancelled};
  if (finish_step)
    state.SetCancellationCheck({});
  // Fallback captures and admission must finish before pinning starts. The
  // chunk's GPU work can overlap preparation; cached decode never calls it.
  impl_->lease->PreparePrefill(std::min(remaining, max_input_tokens));
  // The following step starts at this frontier and stops at the next one, so
  // a runner may read its inputs during this step.
  // Checkpoints and prompt targets end the following step only when this
  // runner cannot take them in place.
  auto lookahead_end = impl_->prompt.size();
  if (!in_pass_capable) {
    for (const auto point : impl_->checkpoints)
      if (point > frontier)
        lookahead_end = std::min(lookahead_end, point);
    if (snapshot_position > frontier)
      lookahead_end = std::min(lookahead_end, snapshot_position);
  }
  struct ClearLookahead {
    TextRunnerState& state;
    ~ClearLookahead() { state.SetPrefillLookahead({}); }
  } clear_lookahead{state};
  state.SetPrefillLookahead(std::span<const TextRunnerToken>(impl_->prompt)
                                .subspan(frontier, lookahead_end - frontier));
  std::vector<std::size_t> in_pass_positions;
  for (const auto& capture : in_pass)
    in_pass_positions.push_back(capture.position);
  auto step =
      in_pass.empty()
          ? impl_->runner->Prefill(state, model_prompt, impl_->prefill_offset,
                                   max_input_tokens)
          : impl_->runner->PrefillThrough(
                state, model_prompt, impl_->prefill_offset, max_input_tokens,
                in_pass_positions, consume);
  if (step.interrupted) {
    if (!capabilities.prefill_rollback || step.consumed_tokens ||
        step.decode_ready || !impl_->cancelled || !impl_->cancelled() ||
        impl_->runner->CheckpointPosition(state) != impl_->prefill_offset)
      throw std::runtime_error("invalid interrupted prefill frontier");
    impl_->state_reusable = true;
    throw std::runtime_error("generation cancelled");
  }
  const std::size_t maximum_consumed = std::min(remaining, max_input_tokens);
  if (step.consumed_tokens == 0 || step.consumed_tokens > maximum_consumed) {
    throw std::runtime_error(
        "text runner returned an invalid prefill token count");
  }

  impl_->prefill_offset += step.consumed_tokens;
  const bool reached_frontier = impl_->prefill_offset == impl_->prompt.size();
  if (step.decode_ready != (impl_->prefill_offset == model_prompt.size())) {
    throw std::runtime_error(
        "text runner returned an inconsistent prefill boundary");
  }
  impl_->decode_ready = reached_frontier;
  step.decode_ready = reached_frontier;
  impl_->state_reusable = true;
  impl_->snapshot_metrics.snapshot_ms += step.checkpoint_ms;
  if (in_pass_prompt && prompt_captures.empty())
    prompt_captures.push_back({snapshot_position, impl_->snapshot_bytes, {}});
  for (auto& taken : prompt_captures) {
    // AdvancePromptTarget moves through the targets planned above.
    if (impl_->snapshot_tokens.size() != taken.position)
      break;
    impl_->prompt_snapshot_attempted = true;
    impl_->snapshot_bytes = taken.bytes;
    // Checkpoints captured after it in the pass reserved for themselves.
    // Publication admits under the lease's reservation, so restore this
    // target's continuation purpose first.
    const bool reserved =
        taken.snapshot && impl_->lease->TryReserveSnapshot(
                              taken.bytes, taken.position, taken.preserve,
                              SnapshotPurpose::kContinuation,
                              std::span<const TextRunnerToken>(impl_->prompt)
                                  .first(taken.position));
    if (!reserved || impl_->prefill_offset <= taken.position ||
        taken.snapshot->PayloadBytes() != taken.bytes) {
      if (!taken.refused)
        impl_->lease->SkipSnapshot(
            reserved ? SnapshotEventReason::kReservationMismatch
                     : SnapshotEventReason::kByteCapacity,
            taken.snapshot ? taken.snapshot->PayloadBytes() : 0,
            taken.position);
      impl_->retain_snapshot = false;
    } else {
      impl_->retain_snapshot = true;
      impl_->prompt_snapshot = std::move(taken.snapshot);
    }
    impl_->AdvancePromptTarget();
  }
  if (!reached_frontier &&
      impl_->prefill_offset == impl_->snapshot_tokens.size())
    impl_->CapturePromptSnapshot(false);
  // Checkpoints this step captured in place are behind its frontier.
  while (!impl_->checkpoints.empty() &&
         impl_->checkpoints.front() < impl_->prefill_offset)
    impl_->checkpoints.erase(impl_->checkpoints.begin());
  const bool history = !impl_->checkpoints.empty() &&
                       impl_->checkpoints.front() == impl_->prefill_offset;
  if (history)
    impl_->checkpoints.erase(impl_->checkpoints.begin());
  // A prompt/fallback capture also persists this boundary. Do not start a
  // second worker over the same state or replace its pending result.
  const bool shared_with_peers =
      history &&
      std::ranges::find(impl_->shared_checkpoints, impl_->prefill_offset) !=
          impl_->shared_checkpoints.end();
  if (history && impl_->prefill_offset != impl_->snapshot_tokens.size())
    impl_->CaptureBoundarySnapshot(
        impl_->prefill_offset, shared_with_peers ? SnapshotPurpose::kBranchPoint
                                                 : SnapshotPurpose::kHistory);
  return step;
}

TextDecodeSelection TextRunnerPool::Request::SelectNext() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot decode before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner decode token must be advanced before selecting another");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }

  impl_->state_reusable = false;
  auto selection = impl_->runner->SelectNext(
      dynamic_cast<TextRunnerState&>(impl_->lease->state()), impl_->sampler);
  impl_->state_reusable = true;
  if (selection.stop) {
    impl_->stopped = true;
    return selection;
  }
  impl_->sampler.Accept(selection.token);
  impl_->generated.push_back(selection.token);
  impl_->pending_selection = selection;
  return selection;
}

void TextRunnerPool::Request::Advance() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner request has no selected token to advance");
  }
  impl_->CapturePromptSnapshot();
  impl_->state_reusable = false;
  impl_->runner->Advance(dynamic_cast<TextRunnerState&>(impl_->lease->state()),
                         impl_->pending_selection->token);
  impl_->pending_selection.reset();
  impl_->state_reusable = true;
}

TextDecodeStep TextRunnerPool::Request::DecodeStep(std::size_t max_tokens) {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot decode before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    throw std::logic_error(
        "text runner cannot decode a step with a pending token");
  }
  if (impl_->stopped) {
    throw std::logic_error("text runner request already stopped");
  }
  if (max_tokens == 0) {
    throw std::invalid_argument(
        "text runner decode step budget must be at least one token");
  }

  impl_->CapturePromptSnapshot();
  impl_->state_reusable = false;
  auto step = impl_->runner->DecodeStep(
      dynamic_cast<TextRunnerState&>(impl_->lease->state()), max_tokens,
      impl_->sampler);
  if (step.selections.size() > max_tokens ||
      (step.selections.empty() && !step.stop)) {
    throw std::runtime_error("text runner returned an invalid decode step");
  }
  for (const auto& selection : step.selections) {
    if (selection.stop) {
      throw std::runtime_error(
          "text runner decode step contains an embedded stop selection");
    }
    impl_->sampler.Accept(selection.token);
    impl_->generated.push_back(selection.token);
  }
  impl_->stopped = step.stop;
  impl_->state_reusable = true;
  return step;
}

TextRunnerPool::Request::CommitMetrics TextRunnerPool::Request::Commit() {
  if (!*this) {
    throw std::logic_error("text runner request is empty");
  }
  if (!impl_->decode_ready) {
    throw std::logic_error(
        "text runner cannot commit before prefill completes");
  }
  if (impl_->pending_selection.has_value()) {
    if (impl_->runner->Descriptor().capabilities.final_token_advance_required) {
      throw std::logic_error(
          "text runner cannot commit an unadvanced decode token");
    }
    impl_->pending_selection.reset();
  }

  impl_->CapturePromptSnapshot();
  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease->state());
  state.SetCancellationCheck({});
  if (!impl_->runner->Descriptor().capabilities.prefix_reuse) {
    impl_->lease->Invalidate();
    impl_.reset();
    return {};
  }
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  std::vector<ContinuationToken> checkpoint = impl_->prompt;
  checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                    impl_->generated.end());
  const std::size_t position = impl_->runner->CheckpointPosition(state);
  if (position < impl_->lease->cached_tokens() ||
      position > checkpoint.size()) {
    throw std::runtime_error(
        "text runner checkpoint is outside executed token history: position=" +
        std::to_string(position) +
        " cached=" + std::to_string(impl_->lease->cached_tokens()) +
        " prompt=" + std::to_string(impl_->prompt.size()) +
        " generated=" + std::to_string(impl_->generated.size()));
  }
  checkpoint.resize(position);
  // As in Cancel: an unaligned prefill frontier is not kept live. The lease
  // drops a component state after publishing the prompt snapshot, which
  // reads its positions; other states are reset here.
  if (position % impl_->alignment != 0 &&
      position > impl_->lease->cached_tokens() &&
      position <= impl_->prefill_offset) {
    checkpoint.clear();
  }
  if (capabilities.snapshot && capabilities.fork) {
    CommitMetrics metrics = impl_->snapshot_metrics;
    metrics.snapshot_bytes += impl_->lease->Commit(
        std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
        std::move(checkpoint));
    metrics.component_cache = component_cache_metrics();
    if (metrics.component_cache) {
      metrics.disk_queued_bytes = metrics.component_cache->disk_queued_bytes;
      metrics.disk_enqueue_ms = metrics.component_cache->disk_enqueue_ms;
    }
    impl_.reset();
    return metrics;
  }

  std::unique_ptr<TextRunnerSnapshot> snapshot;
  CommitMetrics metrics;
  metrics.snapshot_bytes =
      impl_->lease->Commit(std::move(checkpoint), std::move(snapshot));
  impl_.reset();
  return metrics;
}

TextRunnerPool::Request::CommitMetrics
TextRunnerPool::Request::Cancel() noexcept {
  if (!*this)
    return {};
  try {
    // Join only an existing capture; cancellation must not start a new copy
    // or execute the pending selected token merely to retain the session.
    if (impl_->snapshot_future.valid())
      impl_->FinishSnapshot();
    auto& state = dynamic_cast<TextRunnerState&>(impl_->lease->state());
    state.SetCancellationCheck({});
    const auto capabilities = impl_->runner->Descriptor().capabilities;
    if (!capabilities.prefix_reuse) {
      Invalidate();
      return {};
    }
    std::vector<ContinuationToken> checkpoint;
    if (impl_->state_reusable &&
        (impl_->decode_ready ||
         impl_->prefill_offset == impl_->snapshot_tokens.size() ||
         impl_->prefill_offset == impl_->fallback_position ||
         (capabilities.partial_prefill_retention &&
          impl_->prefill_offset > impl_->lease->cached_tokens()))) {
      if (impl_->decode_ready)
        impl_->runner->PrepareCancellation(state);
      checkpoint.assign(impl_->prompt.begin(),
                        impl_->prompt.begin() + impl_->prefill_offset);
      if (impl_->decode_ready)
        checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                          impl_->generated.end());
      const auto position = impl_->runner->CheckpointPosition(state);
      if (position < impl_->lease->cached_tokens() ||
          position > checkpoint.size())
        throw std::runtime_error(
            "cancelled checkpoint is outside completed work");
      checkpoint.resize(position);
      // This request's prefill frontier off the alignment cannot continue the
      // prompt exactly; a decoded or reused frontier is kept there. As in
      // Commit, the lease drops a component state after publishing.
      if (position % impl_->alignment != 0 &&
          position > impl_->lease->cached_tokens() &&
          position <= impl_->prefill_offset) {
        checkpoint.clear();
      }
    } else {
      state.Invalidate();
    }
    CommitMetrics metrics = impl_->snapshot_metrics;
    if (capabilities.snapshot && capabilities.fork) {
      metrics.snapshot_bytes += impl_->lease->Commit(
          std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
          std::move(checkpoint));
    } else {
      metrics.snapshot_bytes = impl_->lease->Commit(std::move(checkpoint));
    }
    metrics.component_cache = component_cache_metrics();
    if (metrics.component_cache) {
      metrics.disk_queued_bytes = metrics.component_cache->disk_queued_bytes;
      metrics.disk_enqueue_ms = metrics.component_cache->disk_enqueue_ms;
    }
    impl_.reset();
    return metrics;
  } catch (...) {
    Invalidate();
    return {};
  }
}

void TextRunnerPool::Request::Invalidate() noexcept {
  if (impl_ != nullptr) {
    if (impl_->snapshot_future.valid())
      impl_->snapshot_future.wait();
    dynamic_cast<TextRunnerState&>(impl_->lease->state())
        .SetCancellationCheck({});
    impl_->lease->Invalidate();
    impl_.reset();
  }
}

TextRunnerPool::TextRunnerPool(
    std::shared_ptr<TextModelRunner> runner, std::size_t state_count,
    std::optional<TextRunnerDiskCacheOptions> disk_cache,
    TextRunnerRamCacheOptions ram_cache)
    : impl_(std::make_unique<Impl>(std::move(runner), state_count,
                                   std::move(disk_cache), ram_cache)) {}

TextRunnerPool::~TextRunnerPool() = default;

const TextModelRunner& TextRunnerPool::runner() const noexcept {
  return *impl_->validated.runner;
}

std::size_t TextRunnerPool::capacity() const noexcept {
  return impl_->component_cache->capacity();
}

TextExecutionPlan TextRunnerPool::SelectDecodePlan(
    std::size_t ready_requests) const {
  if (ready_requests == 0) {
    throw std::invalid_argument(
        "decode plan requires at least one ready request");
  }

  const auto serial =
      std::find_if(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                   [](const TextExecutionPlan& plan) {
                     return plan.kind == TextExecutionPlanKind::kSerial &&
                            plan.physical_width == 1;
                   });
  if (ready_requests == 1) {
    return *serial;
  }

  const TextExecutionPlan* smallest_covering = nullptr;
  const TextExecutionPlan* widest_available = nullptr;
  for (const auto& plan : impl_->validated.plans) {
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      continue;
    }
    if (widest_available == nullptr ||
        plan.physical_width > widest_available->physical_width) {
      widest_available = &plan;
    }
    if (plan.physical_width >= ready_requests &&
        (smallest_covering == nullptr ||
         plan.physical_width < smallest_covering->physical_width)) {
      smallest_covering = &plan;
    }
  }
  if (smallest_covering != nullptr) {
    return *smallest_covering;
  }
  if (widest_available != nullptr) {
    return *widest_available;
  }
  return *serial;
}

std::vector<std::exception_ptr> TextRunnerPool::AdvanceBatch(
    std::span<Request*> requests, const TextExecutionPlan& plan) {
  if (plan.kind != TextExecutionPlanKind::kBatched || requests.size() < 2 ||
      requests.size() > plan.physical_width) {
    throw std::invalid_argument("invalid batched text execution plan");
  }
  if (std::find(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                plan) == impl_->validated.plans.end()) {
    throw std::invalid_argument(
        "text runner does not support the requested batched plan");
  }

  std::vector<std::exception_ptr> failures(requests.size());
  std::vector<TextRunnerAdvance> advances;
  advances.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* request = requests[index];
    if (request == nullptr || !*request ||
        request->impl_->runner != impl_->validated.runner ||
        !request->impl_->pending_selection.has_value()) {
      throw std::logic_error("invalid request in batched text advance");
    }
    const auto previous_requests = requests.first(index);
    if (std::find(previous_requests.begin(), previous_requests.end(),
                  request) != previous_requests.end()) {
      throw std::invalid_argument(
          "batched text advance contains a duplicate request");
    }
    advances.push_back({
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease->state()),
        .token = request->impl_->pending_selection->token,
        .failure = &failures[index],
    });
  }

  for (auto* request : requests) {
    request->CapturePromptSnapshot();
    request->impl_->state_reusable = false;
  }
  impl_->validated.runner->AdvanceBatch(advances);
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!failures[i]) {
      requests[i]->impl_->pending_selection.reset();
      requests[i]->impl_->state_reusable = true;
    }
  }
  return failures;
}

std::vector<TextDecodeStep> TextRunnerPool::DecodeBatch(
    std::span<Request*> requests, std::span<const std::size_t> max_tokens,
    const TextExecutionPlan& plan) {
  if (plan.kind != TextExecutionPlanKind::kBatched || requests.size() < 2 ||
      requests.size() > plan.physical_width ||
      max_tokens.size() != requests.size()) {
    throw std::invalid_argument("invalid batched text decode plan");
  }
  if (std::find(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                plan) == impl_->validated.plans.end()) {
    throw std::invalid_argument(
        "text runner does not support the requested batched plan");
  }

  std::vector<TextRunnerDecode> decodes;
  decodes.reserve(requests.size());
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto* request = requests[index];
    if (request == nullptr || !*request ||
        request->impl_->runner != impl_->validated.runner ||
        !request->impl_->decode_ready || request->impl_->stopped ||
        request->impl_->pending_selection.has_value() ||
        max_tokens[index] == 0) {
      throw std::logic_error("invalid request in batched text decode");
    }
    const auto previous_requests = requests.first(index);
    if (std::find(previous_requests.begin(), previous_requests.end(),
                  request) != previous_requests.end()) {
      throw std::invalid_argument(
          "batched text decode contains a duplicate request");
    }
    decodes.push_back({
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease->state()),
        .max_tokens = max_tokens[index],
        .sampler = request->impl_->sampler,
    });
  }

  for (auto* request : requests) {
    request->CapturePromptSnapshot();
    request->impl_->state_reusable = false;
  }
  auto steps = impl_->validated.runner->DecodeBatch(decodes);
  if (steps.size() != requests.size()) {
    throw std::runtime_error(
        "text runner returned an invalid decode batch size");
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& step = steps[index];
    if (step.failure)
      continue;
    auto& request = *requests[index]->impl_;
    if (step.execution_plan.physical_width == 0 ||
        step.execution_plan.physical_width > requests.size() ||
        (step.execution_plan.kind == TextExecutionPlanKind::kSerial &&
         step.execution_plan.physical_width != 1)) {
      throw std::runtime_error(
          "text runner returned an invalid execution width");
    }
    if (step.selections.size() > max_tokens[index] ||
        (step.selections.empty() && !step.stop)) {
      throw std::runtime_error(
          "text runner returned an invalid batched decode step");
    }
    for (const auto& selection : step.selections) {
      if (selection.stop) {
        throw std::runtime_error(
            "text runner batched decode contains an embedded stop selection");
      }
      request.sampler.Accept(selection.token);
      request.generated.push_back(selection.token);
    }
    request.stopped = step.stop;
    request.state_reusable = true;
  }
  return steps;
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse_prompt,
    std::size_t cache_prefix_tokens, bool stop_at_eos, bool persist) {
  sampling_config.Validate();
  if (prompt.empty()) {
    throw std::invalid_argument("text runner prompt must not be empty");
  }
  if (prompt.size() > impl_->validated.descriptor.max_context) {
    throw std::length_error("text runner prompt exceeds model context");
  }
  if (cache_prefix_tokens > prompt.size())
    throw std::invalid_argument("cache prefix exceeds prompt length");

  std::unique_lock admission(impl_->admission_mutex, std::defer_lock);
  while (!admission.try_lock_for(std::chrono::milliseconds(10))) {
    if (is_cancelled && is_cancelled())
      return {};
  }

  {
    auto lease = impl_->component_cache->Acquire(
        prompt, is_cancelled, context, reuse_prompt, cache_prefix_tokens,
        stop_at_eos, persist);
    if (!lease)
      return {};
    Request request(std::make_unique<Request::Impl>(
        impl_->validated.runner, std::move(lease), std::move(prompt),
        sampling_config, std::move(context), cache_prefix_tokens,
        capacity() > 1));
    request.impl_->cancelled = is_cancelled;
    if (reuse_prompt) {
      const auto common = impl_->component_cache->CachedPrefixTokens(
          request.prompt(), request.impl_->context.get(), true);
      if (common >=
              request.prefill_position() + Request::kSharedPrefixMinTokens &&
          (common + Request::kSharedCheckpointSlack < request.prompt_tokens() ||
           (impl_->shared_prefix_max_boundaries &&
            common >= impl_->shared_prefix_min_tokens &&
            common < request.prompt_tokens())))
        (void)request.ShareCheckpoint(common);
    }
    return request;
  }
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const CancellationCheck& is_cancelled) {
  return Acquire(std::move(prompt), sampling::SamplingConfig{}, is_cancelled);
}

std::optional<TextPendingPublication> TextRunnerPool::PendingPublication(
    std::span<const TextRunnerToken> prompt, const TextPromptContext* context,
    std::uint64_t checkpoint_id) const {
  return impl_->component_cache->PendingPublication(prompt, context,
                                                    checkpoint_id);
}

std::size_t TextRunnerPool::CachedPrefixTokens(
    std::span<const TextRunnerToken> prompt,
    const TextPromptContext* context) const {
  return impl_->component_cache->CachedPrefixTokens(prompt, context);
}

}  // namespace gufo::server
