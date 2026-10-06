#include "src/cli/serve/text_model_runner.hpp"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/continuation_disk_store.hpp"
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
  if (runner.InitialOutputState(request) ==
      TextGenerationBackend::InitialOutputState::kReasoning)
    grammar = sampling::JsonConstraint::WithReasoning(grammar);
  sampling->constraint = runner.BindConstraint(grammar);
  return constrained;
}

namespace {

/// Host RAM still available, after cgroup limits. Unified-memory device
/// allocations draw from the same RAM, so HIP's device capacity is not used.
std::uint64_t HostAvailableBytes() {
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return 0;
  std::uint64_t available = std::uint64_t(pages) * page_size;
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream fields(line.substr(13));
      std::uint64_t kib = 0;
      if (fields >> kib)
        available = kib * 1024;
      break;
    }
  }
  // A cgroup limit can be much smaller than the host's available memory.
  // Walk parents too: a child may say "max" beneath a limited ancestor.
  std::ifstream membership("/proc/self/cgroup");
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    const std::filesystem::path root("/sys/fs/cgroup");
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used))
        available = std::min(available, limit > used ? limit - used : 0);
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

std::size_t ClampToSize(std::uint64_t bytes) {
  return static_cast<std::size_t>(
      std::min<std::uint64_t>(bytes, std::numeric_limits<std::size_t>::max()));
}

}  // namespace

std::size_t HostSnapshotBudgetBytes() {
  return ClampToSize(HostAvailableBytes() / 2);
}

std::size_t HostSnapshotCeilingBytes() {
  const auto available = HostAvailableBytes();
  return ClampToSize(available > kHostSnapshotHeadroomBytes
                         ? available - kHostSnapshotHeadroomBytes
                         : 0);
}

namespace {

struct ValidatedRunner {
  std::shared_ptr<TextModelRunner> runner;
  TextRunnerDescriptor descriptor;
  TextRunnerResourceClaim resources;
  std::vector<TextExecutionPlan> plans;
  std::size_t snapshot_automatic_bytes{0};
  std::size_t snapshot_max_bytes{0};
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
      throw std::invalid_argument(
          "text runner request-state claim exceeds state capacity");
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

void ReconcileStateBytes(const TextRunnerResourceClaim& resources,
                         const TextRunnerState& state) {
  const auto measured = state.MeasuredResources();
  if (measured.per_request_state_bytes.has_value() &&
      resources.per_request_state_bytes.has_value() &&
      *measured.per_request_state_bytes > *resources.per_request_state_bytes) {
    throw std::runtime_error(
        "text runner measured state exceeds its resource claim");
  }
  if (measured.temporary_scratch_bytes.has_value() &&
      resources.temporary_scratch_bytes.has_value() &&
      *measured.temporary_scratch_bytes > *resources.temporary_scratch_bytes) {
    throw std::runtime_error(
        "text runner measured scratch exceeds its resource claim");
  }
}

std::string_view SnapshotEventActionName(SnapshotEventAction action) noexcept {
  switch (action) {
    case SnapshotEventAction::kRemoved:
      return "removed";
    case SnapshotEventAction::kSkipped:
      return "skipped";
  }
  return "unknown";
}

std::string_view SnapshotEventReasonName(SnapshotEventReason reason) noexcept {
  switch (reason) {
    case SnapshotEventReason::kByteCapacity:
      return "byte_capacity";
    case SnapshotEventReason::kEntryCapacity:
      return "entry_capacity";
    case SnapshotEventReason::kExactReplacement:
      return "exact_replacement";
    case SnapshotEventReason::kCaptureFailure:
      return "capture_failure";
    case SnapshotEventReason::kReservationMismatch:
      return "reservation_mismatch";
  }
  return "unknown";
}

void EmitSnapshotEvent(const SnapshotEvent& event) noexcept {
  // Replacing a retained prefix, or dropping one to fit the byte budget, is
  // routine; request summaries already report whether reuse succeeded.
  // Losing one because every entry is taken is not routine: the server holds
  // fewer conversations than the workload rotates through, so it is doing
  // avoidable full re-prefills. Failed captures also need attention.
  if (event.action == SnapshotEventAction::kRemoved &&
      event.reason != SnapshotEventReason::kEntryCapacity)
    return;
  try {
    std::ostringstream line;
    line << "event=snapshot action=" << SnapshotEventActionName(event.action)
         << " reason=" << SnapshotEventReasonName(event.reason)
         << " bytes=" << event.snapshot_bytes << " tokens=" << event.token_count
         << " retained_bytes=" << event.retained_snapshot_bytes
         << " reserved_bytes=" << event.reserved_snapshot_bytes
         << " capacity_bytes=" << event.capacity_bytes;
    Logger::Warn("cache", line.str());
  } catch (...) {
    // Cache logging must not affect request execution.
  }
}

std::string_view DiskEventActionName(
    ContinuationDiskEventAction action) noexcept {
  switch (action) {
    case ContinuationDiskEventAction::kStored:
      return "stored";
    case ContinuationDiskEventAction::kRestored:
      return "restored";
    case ContinuationDiskEventAction::kMiss:
      return "miss";
    case ContinuationDiskEventAction::kRemoved:
      return "removed";
    case ContinuationDiskEventAction::kSkipped:
      return "skipped";
  }
  return "unknown";
}

std::string_view DiskEventReasonName(
    ContinuationDiskEventReason reason) noexcept {
  switch (reason) {
    case ContinuationDiskEventReason::kSaved:
      return "saved";
    case ContinuationDiskEventReason::kHit:
      return "hit";
    case ContinuationDiskEventReason::kNotFound:
      return "not_found";
    case ContinuationDiskEventReason::kUnsupported:
      return "unsupported";
    case ContinuationDiskEventReason::kByteCapacity:
      return "byte_capacity";
    case ContinuationDiskEventReason::kStagingCapacity:
      return "staging_capacity";
    case ContinuationDiskEventReason::kLru:
      return "lru";
    case ContinuationDiskEventReason::kMinStep:
      return "min_step";
    case ContinuationDiskEventReason::kExactReplacement:
      return "exact_replacement";
    case ContinuationDiskEventReason::kCorrupt:
      return "corrupt";
    case ContinuationDiskEventReason::kChecksumMismatch:
      return "checksum_mismatch";
    case ContinuationDiskEventReason::kUnsafeFile:
      return "unsafe_file";
    case ContinuationDiskEventReason::kIoFailure:
      return "io_failure";
    case ContinuationDiskEventReason::kSerializationFailure:
      return "serialization_failure";
    case ContinuationDiskEventReason::kRestoreFailure:
      return "restore_failure";
    case ContinuationDiskEventReason::kBusy:
      return "busy";
  }
  return "unknown";
}

void EmitDiskEvent(const ContinuationDiskEvent& event) noexcept {
  switch (event.reason) {
    case ContinuationDiskEventReason::kHit:
    case ContinuationDiskEventReason::kNotFound:
    case ContinuationDiskEventReason::kExactReplacement:
    case ContinuationDiskEventReason::kBusy:
      return;
    case ContinuationDiskEventReason::kByteCapacity:
      if (event.action == ContinuationDiskEventAction::kRemoved)
        return;
      break;
    default:
      break;
  }
  try {
    std::ostringstream line;
    line << "event=disk_cache action=" << DiskEventActionName(event.action)
         << " reason=" << DiskEventReasonName(event.reason)
         << " file_bytes=" << event.file_bytes
         << " payload_bytes=" << event.payload_bytes
         << " tokens=" << event.token_count
         << " retained_bytes=" << event.retained_bytes
         << " capacity_bytes=" << event.capacity_bytes
         << " staging_capacity_bytes=" << event.staging_capacity_bytes
         << " staging_used_bytes=" << event.staging_used_bytes;
    if (event.reason == ContinuationDiskEventReason::kSaved) {
      line << " write_ms=" << event.elapsed_ms;
      Logger::Info("cache", line.str());
    } else if (event.reason == ContinuationDiskEventReason::kLru ||
               event.reason == ContinuationDiskEventReason::kMinStep) {
      // Staying inside a configured budget or skipping a checkpoint that
      // barely advances is expected operation, but it has to be visible:
      // otherwise a run that evicts every other conversation looks identical
      // to one that never cached anything.
      Logger::Info("cache", line.str());
    } else {
      Logger::Warn("cache", line.str());
    }
  } catch (...) {
    // Cache logging must not affect request execution.
  }
}

ContinuationCache::SnapshotSupport MakeSnapshotSupport(
    ValidatedRunner* validated, TextRunnerRamCacheOptions options) {
  if (validated == nullptr || !validated->descriptor.capabilities.snapshot ||
      !validated->descriptor.capabilities.fork) {
    return {};
  }
  return {
      .restore =
          [validated](ContinuationState& state,
                      const ContinuationSnapshot& snapshot) {
            auto* text_snapshot =
                dynamic_cast<const TextRunnerSnapshot*>(&snapshot);
            if (text_snapshot == nullptr) {
              throw std::invalid_argument(
                  "continuation snapshot is not a text runner snapshot");
            }
            auto& text_state = dynamic_cast<TextRunnerState&>(state);
            validated->runner->RestoreOrFork(text_state, *text_snapshot);
            ReconcileStateBytes(validated->resources, text_state);
          },
      .capacity_bytes =
          [validated, options] {
            const auto resources = validated->runner->ResourceClaim();
            const auto automatic =
                resources.retained_snapshot_capacity_bytes.value_or(0);
            validated->snapshot_automatic_bytes = std::min(
                TextRunnerRamCacheOptions::kAutomaticMaxBytes, automatic);
            validated->snapshot_max_bytes =
                resources.retained_snapshot_ceiling_bytes.value_or(automatic);
            if (options.capacity_bytes == 0)
              return validated->snapshot_automatic_bytes;
            // An explicit limit is the operator's choice to trade headroom
            // for retention, bounded so the host keeps a fixed reserve.
            return std::min(options.capacity_bytes,
                            validated->snapshot_max_bytes);
          },
      .on_event = EmitSnapshotEvent,
  };
}

}  // namespace

void TextModelRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  for (const auto& advance : advances) {
    try {
      Advance(advance.state.get(), advance.token);
    } catch (...) {
      if (!advance.failure)
        throw;
      *advance.failure = std::current_exception();
    }
  }
}

TextDecodeStep TextModelRunner::DecodeStep(
    TextRunnerState& state, std::size_t max_tokens,
    sampling::SamplerState& sampler) const {
  if (max_tokens == 0) {
    throw std::invalid_argument(
        "text runner decode step budget must be at least one token");
  }
  auto selection = SelectNext(state, sampler);
  if (selection.stop) {
    return {
        .selections = {},
        .draft_tokens = 0,
        .draft_accepted_tokens = 0,
        .stop = true,
    };
  }
  const TextRunnerToken token = selection.token;
  Advance(state, token);
  return {
      .selections = {std::move(selection)},
  };
}

std::vector<TextDecodeStep> TextModelRunner::DecodeBatch(
    std::span<const TextRunnerDecode> decodes) const {
  std::vector<TextDecodeStep> steps;
  steps.reserve(decodes.size());
  for (const auto& decode : decodes) {
    try {
      steps.push_back(DecodeStep(decode.state.get(), decode.max_tokens,
                                 decode.sampler.get()));
    } catch (...) {
      TextDecodeStep failed;
      failed.failure = std::current_exception();
      steps.push_back(std::move(failed));
    }
  }
  return steps;
}

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::Snapshot(
    const TextRunnerState&) const {
  throw std::logic_error("text runner does not support snapshots");
}

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::SnapshotForPersistence(
    const TextRunnerState& state) const {
  return Snapshot(state);
}

std::size_t TextModelRunner::SnapshotPayloadBytes(
    const TextRunnerState&) const {
  throw std::logic_error("text runner does not support snapshot sizing");
}

void TextModelRunner::RestoreOrFork(TextRunnerState&,
                                    const TextRunnerSnapshot&) const {
  throw std::logic_error("text runner does not support snapshot restore/fork");
}

std::size_t TextModelRunner::PersistentSnapshotPayloadBytes(
    const TextRunnerSnapshot&) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

std::size_t TextModelRunner::SerializePersistentSnapshot(
    const TextRunnerSnapshot&, std::span<std::uint8_t>) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

void TextModelRunner::RestorePersistentSnapshot(
    TextRunnerState&, std::span<const std::uint8_t>) const {
  throw std::logic_error("text runner does not support persistent snapshots");
}

struct TextRunnerPool::Impl {
  static constexpr std::size_t kIntermediateCheckpoints = 4;
  Impl(std::shared_ptr<TextModelRunner> model_runner, std::size_t state_count,
       std::optional<TextRunnerDiskCacheOptions> disk_cache_options,
       TextRunnerRamCacheOptions ram_cache_options)
      : validated(ValidateRunner(std::move(model_runner), state_count)),
        cache(
            state_count,
            [this] {
              auto state = validated.runner->CreateState();
              if (state == nullptr) {
                throw std::runtime_error(
                    "text runner state factory returned null");
              }
              ReconcileStateBytes(validated.resources, *state);
              return state;
            },
            MakeSnapshotSupport(&validated, ram_cache_options),
            TextRunnerRamCacheOptions::kMaxEntries) {
    // Entry and byte limits constrain retention independently of session count.
    if (validated.descriptor.capabilities.snapshot) {
      // Report the same post-allocation limits that selected the capacity.
      Logger::Info(
          "cache",
          "event=snapshot_cache_configured sessions=" +
              std::to_string(state_count) + " snapshot_entries=" +
              std::to_string(cache.entry_capacity()) + " capacity_bytes=" +
              std::to_string(cache.snapshot_capacity_bytes()) +
              " automatic_bytes=" +
              std::to_string(validated.snapshot_automatic_bytes) +
              " max_bytes=" + std::to_string(validated.snapshot_max_bytes));
    }
    if (disk_cache_options.has_value()) {
      if (!validated.descriptor.persistence.has_value()) {
        throw std::invalid_argument(
            "text runner does not support persistent snapshots");
      }
      disk_store = std::make_shared<ContinuationDiskStore>(
          ContinuationDiskStoreOptions{
              .directory = std::move(disk_cache_options->directory),
              .capacity_bytes = disk_cache_options->capacity_bytes,
              .staging_capacity_bytes =
                  disk_cache_options->staging_capacity_bytes,
              .min_checkpoint_step_tokens =
                  disk_cache_options->min_checkpoint_step_tokens,
          },
          EmitDiskEvent);
      Logger::Info("cache",
                   "event=disk_cache_configured capacity_bytes=" +
                       std::to_string(disk_store->capacity_bytes()) +
                       " staging_capacity_bytes=" +
                       std::to_string(disk_store->staging_capacity_bytes()));
      shared_prefix_min_tokens = disk_cache_options->shared_prefix_min_tokens;
      shared_prefix_max_boundaries =
          disk_cache_options->shared_prefix_max_boundaries;
    }
  }

  ValidatedRunner validated;
  ContinuationCache cache;
  // Publish a live checkpoint before another admission can fall back to the
  // older prompt snapshot. Decode/prefill work never holds this mutex.
  std::timed_mutex admission_mutex;
  std::shared_ptr<ContinuationDiskStore> disk_store;
  std::size_t shared_prefix_min_tokens{0};
  std::size_t shared_prefix_max_boundaries{0};
};

std::shared_ptr<const sampling::TokenConstraint>
TextModelRunner::BindConstraint(
    std::shared_ptr<const sampling::JsonConstraint> grammar) const {
  const std::lock_guard lock(constraint_mutex_);
  for (const auto& binding : constraints_)
    if (binding->grammar == grammar)
      return binding;
  if (!constraint_vocabulary_)
    constraint_vocabulary_ = BuildConstraintVocabulary();
  auto binding = std::make_shared<sampling::TokenConstraint>();
  binding->grammar = std::move(grammar);
  binding->vocabulary = constraint_vocabulary_;
  if (constraints_.size() >= 4)
    constraints_.erase(constraints_.begin());
  constraints_.push_back(binding);
  return binding;
}

void TextModelRunner::StreamPersistentSnapshot(
    const TextRunnerSnapshot& snapshot, const SnapshotSink& sink) const {
  std::vector<std::uint8_t> payload(PersistentSnapshotPayloadBytes(snapshot));
  if (SerializePersistentSnapshot(snapshot, payload) != payload.size()) {
    throw std::runtime_error("persistent snapshot byte count mismatch");
  }
  sink(payload);
}

struct TextRunnerPool::Request::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner,
       std::shared_ptr<ContinuationDiskStore> persistent_store,
       ContinuationCache::Lease state_lease,
       std::vector<TextRunnerToken> prompt_tokens,
       std::vector<std::size_t> shared_prefix_boundaries,
       const sampling::SamplingConfig& sampling_config,
       std::shared_ptr<const TextPromptContext> prompt_context,
       std::size_t cache_prefix_tokens, bool overlap_snapshots)
      : runner(std::move(model_runner)),
        disk_store(std::move(persistent_store)),
        lease(std::move(state_lease)),
        prompt(std::move(prompt_tokens)),
        boundaries(std::move(shared_prefix_boundaries)),
        prefill_offset(lease.cached_tokens()),
        decode_ready(prefill_offset == prompt.size()),
        sampler(sampling_config, prompt),
        context(std::move(prompt_context)),
        retain_fallback(cache_prefix_tokens != 0),
        overlap_snapshots(overlap_snapshots) {
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
    if (retain_fallback && lease.cache_hit() && prefill_offset <= count &&
        !new_images) {
      // Freezing the reused frontier must not cost this turn its own stable
      // boundary. A client that rewrites the assistant turn, as one dropping
      // reasoning does, diverges after that boundary, so retaining only the
      // frontier pins every later turn to the same early position and the
      // re-prefilled tail grows for the rest of the conversation.
      if (prefill_offset < count)
        stable_prefix_position = count;
      count = prefill_offset;
    }
    if (lease.restored_from_disk() && prefill_offset == prompt.size())
      count = prompt.size();
    snapshot_tokens.reserve(prompt.size());
    snapshot_tokens.assign(prompt.begin(), prompt.begin() + count);
    // A live continuation can already be beyond the desired snapshot point.
    // Keep its existing immutable checkpoint; never snapshot mismatched state.
    prompt_snapshot_attempted = prefill_offset > count;
    auto& state = dynamic_cast<TextRunnerState&>(lease.state());
    if (lease.cache_hit()) {
      runner->PreparePrefixReuse(
          state,
          std::span<const TextRunnerToken>(prompt).first(prefill_offset));
    }
    const auto capabilities = runner->Descriptor().capabilities;
    if (capabilities.incremental_prefill && capabilities.prefix_reuse &&
        capabilities.snapshot && capabilities.fork) {
      // A reused frontier already bounds the work lost on an edit. Do not
      // split a short continuation just to copy a nearby grid checkpoint.
      // Cold long prompts still retain the last grid point for late edits.
      // A grid point just before the prompt end saves little on an edit but
      // would split the final prefill pass and store another snapshot.
      constexpr std::size_t interval = 2048;
      constexpr std::size_t tail = 128;
      const auto grid_points = (prompt.size() - 1) / interval;
      const auto count =
          std::min(grid_points, TextRunnerPool::Impl::kIntermediateCheckpoints);
      for (std::size_t point = 1; point <= count; ++point) {
        const auto position = grid_points * point / count * interval;
        if (position > prefill_offset &&
            position - prefill_offset >= interval &&
            prompt.size() - position > tail &&
            position != snapshot_tokens.size() &&
            position != stable_prefix_position &&
            !lease.HasSnapshotFor(
                std::span<const TextRunnerToken>(prompt).first(position)))
          checkpoints.push_back(position);
      }
    }
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
            lease.PublishSnapshot(snapshot_tokens, std::move(prompt_snapshot),
                                  fallback_position != 0);
      } catch (...) {
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
                           snapshot_tokens.size());
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
    if (lease.cache_hit() && !lease.restored_from_disk() &&
        lease.cached_tokens() == snapshot_tokens.size() &&
        lease.HasSnapshotFor(snapshot_tokens)) {
      return;
    }

    const TextRunnerState* state = nullptr;
    try {
      state = &dynamic_cast<const TextRunnerState&>(lease.state());
    } catch (...) {
      return;
    }
    snapshot_bytes = 0;
    try {
      if (runner->CheckpointPosition(*state) != snapshot_tokens.size()) {
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                           snapshot_tokens.size());
        return;
      }
      snapshot_bytes = runner->SnapshotPayloadBytes(*state);
    } catch (...) {
      lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, 0,
                         snapshot_tokens.size());
      return;
    }

    retain_snapshot = false;
    try {
      retain_snapshot = lease.TryReserveSnapshot(
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
    if (!retain_snapshot) {
      try {
        if (!disk_store)
          return;
        // Copying device state only for disk is costly on the request path.
        // Skip it when the disk store would discard the checkpoint anyway.
        if (disk_store->WithinCheckpointStep(
                *runner, snapshot_tokens,
                InputIdentity(snapshot_tokens.size())))
          return;
        disk_capture = disk_store->ReserveCapture(
            *runner, snapshot_tokens.size(), snapshot_bytes,
            InputIdentity(snapshot_tokens.size()));
        if (!disk_capture)
          return;
      } catch (...) {
        return;
      }
    }

    snapshot_start = std::chrono::steady_clock::now();
    try {
      // A warm prefill must await its fallback before mutating recurrent
      // state. Run that copy on this worker instead of starting a thread
      // only to join it. Post-token captures remain asynchronous.
      snapshot_future =
          std::async(wait ? std::launch::deferred : std::launch::async,
                     [owner = runner, state, persist = bool(disk_store)] {
                       return persist ? owner->SnapshotForPersistence(*state)
                                      : owner->Snapshot(*state);
                     });
    } catch (...) {
      disk_capture.reset();
      if (retain_snapshot)
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
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
      disk_capture.reset();
      if (retain_snapshot) {
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, snapshot_bytes,
                           snapshot_tokens.size());
      }
      return;
    }

    if (retain_snapshot && prompt_snapshot->PayloadBytes() != snapshot_bytes) {
      lease.SkipSnapshot(SnapshotEventReason::kReservationMismatch,
                         prompt_snapshot->PayloadBytes(),
                         snapshot_tokens.size());
      retain_snapshot = false;
    }
    // Charge disk-only snapshots immediately. Holding them until commit would
    // let concurrent requests each pass the same queue-capacity preflight.
    const bool already_persisted =
        lease.restored_from_disk() &&
        lease.cached_tokens() == snapshot_tokens.size();
    if (disk_store && !already_persisted) {
      const auto started = std::chrono::steady_clock::now();
      try {
        const auto identity = InputIdentity(snapshot_tokens.size());
        snapshot_metrics.disk_queued_bytes += disk_store->SaveAsync(
            runner, snapshot_tokens, prompt_snapshot,
            {identity.begin(), identity.end()}, std::move(disk_capture));
      } catch (...) {
        // Earlier checkpoints may already be queued.
      }
      snapshot_metrics.disk_enqueue_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
    }
    disk_capture.reset();
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
      std::size_t position, bool history, bool shared,
      SnapshotPurpose purpose = SnapshotPurpose::kHistory) noexcept {
    if (!overlap_snapshots) {
      // A single execution slot has no peer work to overlap with the copy.
      CaptureBoundarySnapshotNow(position, history, shared, purpose);
      return;
    }
    // The scheduler already parks leases with a pending snapshot. Reuse that
    // path so allocation, eviction and copying do not stall unrelated users.
    // Direct callers and cancellation join before touching this lease again.
    try {
      snapshot_future = std::async(
          std::launch::async, [this, position, history, shared, purpose] {
            CaptureBoundarySnapshotNow(position, history, shared, purpose);
            return std::unique_ptr<TextRunnerSnapshot>{};
          });
      boundary_capture = true;
    } catch (...) {
      // Optional caching must not fail the request if a worker cannot start.
    }
  }

  void CaptureBoundarySnapshotNow(
      std::size_t position, bool history, bool shared,
      SnapshotPurpose purpose = SnapshotPurpose::kHistory) noexcept {
    const auto started = std::chrono::steady_clock::now();
    std::size_t bytes = 0;
    bool reserved = false;
    bool failed = false;
    try {
      const auto capabilities = runner->Descriptor().capabilities;
      if (!capabilities.prefix_reuse || !capabilities.snapshot ||
          !capabilities.fork)
        return;
      const auto& state = dynamic_cast<const TextRunnerState&>(lease.state());
      if (runner->CheckpointPosition(state) != position)
        return;
      const auto prefix =
          std::span<const TextRunnerToken>(prompt).first(position);
      const auto identity = InputIdentity(position);
      bytes = runner->SnapshotPayloadBytes(state);
      // Intermediate copies must not displace the frontier this request
      // branched from, including its stable image/reasoning fallback.
      // Copies other requests depend on compete like continuation
      // boundaries; the rest give way under pressure.
      if (history && !lease.HasSnapshotFor(prefix))
        reserved = lease.TryReserveSnapshot(bytes, position, true, purpose);
      std::unique_ptr<ContinuationDiskStore::CaptureReservation> persistence;
      if (shared && disk_store && !disk_store->Touch(*runner, prefix, identity))
        persistence =
            disk_store->ReserveCapture(*runner, position, bytes, identity);
      if (!reserved && !persistence)
        return;
      std::shared_ptr<const TextRunnerSnapshot> snapshot =
          persistence ? runner->SnapshotForPersistence(state)
                      : runner->Snapshot(state);
      failed = !snapshot;
      if (snapshot && reserved) {
        snapshot_metrics.snapshot_bytes += lease.PublishSnapshot(
            {prefix.begin(), prefix.end()}, snapshot, true);
        reserved = false;
      }
      // A RAM checkpoint and a learned disk boundary can coincide. Share the
      // immutable copy instead of reading the same device state twice.
      if (snapshot && persistence) {
        const auto saved = disk_store->SaveAsync(
            runner, {prefix.begin(), prefix.end()}, std::move(snapshot),
            {identity.begin(), identity.end()}, std::move(persistence),
            /*shared_prefix=*/true);
        if (saved != 0) {
          ++snapshot_metrics.shared_prefix_snapshots;
          snapshot_metrics.shared_prefix_bytes += saved;
        }
      }
    } catch (...) {
      failed = true;
    }
    if (reserved)
      lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, bytes, position);
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    if (history)
      snapshot_metrics.snapshot_ms += elapsed;
    if (shared) {
      snapshot_metrics.shared_prefix_ms += elapsed;
      snapshot_metrics.shared_prefix_failures += failed;
    }
  }

  std::shared_ptr<TextModelRunner> runner;
  std::shared_ptr<ContinuationDiskStore> disk_store;
  ContinuationCache::Lease lease;
  std::vector<TextRunnerToken> prompt;
  std::vector<TextRunnerToken> snapshot_tokens;
  /// Ascending prefill positions to persist, all inside (cached, prompt size).
  std::vector<std::size_t> boundaries;
  /// Bounded intermediate positions retained in RAM before state advances.
  std::vector<std::size_t> checkpoints;
  /// Checkpoints shared with other requests, retained like continuations.
  std::vector<std::size_t> shared_checkpoints;
  bool boundary_capture{false};
  std::vector<TextRunnerToken> generated;
  std::size_t prefill_offset{0};
  bool decode_ready{false};
  // A throwing model call may have partially mutated device state. Never
  // publish that state as a live continuation, even if its position is stale.
  bool state_reusable{true};
  bool stopped{false};
  std::optional<TextDecodeSelection> pending_selection;
  sampling::SamplerState sampler;
  std::shared_ptr<const TextPromptContext> context;
  bool retain_fallback{false};
  bool overlap_snapshots{false};
  std::size_t fallback_position{0};
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
  std::unique_ptr<ContinuationDiskStore::CaptureReservation> disk_capture;
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
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), sampler);
}

TextRunnerPool::Request::~Request() = default;

TextRunnerPool::Request::Request(Request&&) noexcept = default;

TextRunnerPool::Request& TextRunnerPool::Request::operator=(
    Request&&) noexcept = default;

TextRunnerPool::Request::operator bool() const noexcept {
  return impl_ != nullptr && static_cast<bool>(impl_->lease);
}

bool TextRunnerPool::Request::cache_hit() const noexcept {
  return impl_ != nullptr && impl_->lease.cache_hit();
}

std::size_t TextRunnerPool::Request::cached_prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->lease.cached_tokens() : 0;
}

ContinuationLookup TextRunnerPool::Request::cache_lookup() const noexcept {
  return impl_ != nullptr ? impl_->lease.lookup() : ContinuationLookup{};
}

std::size_t TextRunnerPool::Request::cache_restore_bytes() const noexcept {
  return impl_ != nullptr ? impl_->lease.restored_snapshot_bytes() : 0;
}

double TextRunnerPool::Request::cache_restore_ms() const noexcept {
  return impl_ != nullptr ? impl_->lease.restore_ms() : 0.0;
}

bool TextRunnerPool::Request::cache_disk_hit() const noexcept {
  return impl_ != nullptr && impl_->lease.restored_from_disk();
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
  const auto target = std::min(common_tokens, impl_->prompt.size() - 1);
  const auto offset = impl_->prefill_offset;
  if (target <= offset)
    return 0;
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
      dynamic_cast<TextRunnerState&>(impl_->lease.state()));
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

  // Stop exactly on the next shared-prefix boundary so it can be persisted.
  // Direct callers join any capture before mutating its borrowed state.
  impl_->CapturePromptSnapshot();
  const auto snapshot_position = impl_->snapshot_tokens.size();
  while (!impl_->boundaries.empty() &&
         impl_->boundaries.front() <= impl_->prefill_offset) {
    impl_->boundaries.erase(impl_->boundaries.begin());
  }
  if (!impl_->boundaries.empty()) {
    max_input_tokens = std::min(
        max_input_tokens, impl_->boundaries.front() - impl_->prefill_offset);
  }
  while (!impl_->checkpoints.empty() &&
         impl_->checkpoints.front() <= impl_->prefill_offset)
    impl_->checkpoints.erase(impl_->checkpoints.begin());
  if (!impl_->checkpoints.empty())
    max_input_tokens = std::min(
        max_input_tokens, impl_->checkpoints.front() - impl_->prefill_offset);

  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
  bool in_pass = false;
  if (!impl_->disk_store && !impl_->prompt_snapshot_attempted &&
      impl_->prefill_offset < snapshot_position &&
      snapshot_position < impl_->prompt.size() &&
      impl_->runner->Descriptor().capabilities.in_pass_checkpoint) {
    const auto bytes = impl_->runner->PrefillCheckpointBytes(
        state, impl_->prompt, impl_->prefill_offset, max_input_tokens,
        snapshot_position);
    if (bytes) {
      in_pass = impl_->lease.TryReserveSnapshot(
          *bytes, snapshot_position,
          impl_->retain_fallback && impl_->fallback_position == 0,
          SnapshotPurpose::kContinuation, impl_->snapshot_tokens);
      if (in_pass) {
        impl_->snapshot_bytes = *bytes;
        impl_->retain_snapshot = true;
        impl_->prompt_snapshot_attempted = true;
      }
    }
  }
  if (!in_pass && impl_->prefill_offset < snapshot_position)
    max_input_tokens =
        std::min(max_input_tokens, snapshot_position - impl_->prefill_offset);

  const std::size_t remaining = impl_->prompt.size() - impl_->prefill_offset;
  impl_->state_reusable = false;
  // Complete the model frontier at the cache boundary, including logits and
  // draft catch-up. Its snapshot can then be restored independently.
  auto frontier = !in_pass && impl_->prefill_offset < snapshot_position
                      ? snapshot_position
                      : impl_->prompt.size();
  if (!impl_->checkpoints.empty())
    frontier = std::min(frontier, impl_->checkpoints.front());
  if (!impl_->boundaries.empty())
    frontier = std::min(frontier, impl_->boundaries.front());
  const auto model_prompt =
      std::span<const TextRunnerToken>(impl_->prompt).first(frontier);
  std::unique_ptr<TextRunnerSnapshot> captured;
  auto step =
      in_pass ? impl_->runner->PrefillThrough(
                    state, model_prompt, impl_->prefill_offset,
                    max_input_tokens, snapshot_position, &captured)
              : impl_->runner->Prefill(state, model_prompt,
                                       impl_->prefill_offset, max_input_tokens);
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
  if (in_pass) {
    impl_->snapshot_metrics.snapshot_ms += step.checkpoint_ms;
    if (!captured || impl_->prefill_offset <= snapshot_position ||
        captured->PayloadBytes() != impl_->snapshot_bytes) {
      impl_->lease.SkipSnapshot(SnapshotEventReason::kReservationMismatch,
                                captured ? captured->PayloadBytes() : 0,
                                snapshot_position);
      impl_->retain_snapshot = false;
    } else {
      impl_->prompt_snapshot = std::move(captured);
    }
    impl_->AdvancePromptTarget();
  }
  if (!reached_frontier && impl_->prefill_offset == snapshot_position)
    impl_->CapturePromptSnapshot(false);
  const bool history = !impl_->checkpoints.empty() &&
                       impl_->checkpoints.front() == impl_->prefill_offset;
  const bool shared = !impl_->boundaries.empty() &&
                      impl_->boundaries.front() == impl_->prefill_offset;
  if (history)
    impl_->checkpoints.erase(impl_->checkpoints.begin());
  if (shared)
    impl_->boundaries.erase(impl_->boundaries.begin());
  // A prompt/fallback capture also persists this boundary. Do not start a
  // second worker over the same state or replace its pending result.
  const bool shared_with_peers =
      history &&
      std::ranges::find(impl_->shared_checkpoints, impl_->prefill_offset) !=
          impl_->shared_checkpoints.end();
  if ((history || shared) && impl_->prefill_offset != snapshot_position)
    impl_->CaptureBoundarySnapshot(impl_->prefill_offset, history, shared,
                                   shared_with_peers
                                       ? SnapshotPurpose::kContinuation
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
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), impl_->sampler);
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
  impl_->runner->Advance(dynamic_cast<TextRunnerState&>(impl_->lease.state()),
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
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), max_tokens,
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
  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
  state.SetCancellationCheck({});
  if (!impl_->runner->Descriptor().capabilities.prefix_reuse) {
    impl_->lease.Invalidate();
    impl_.reset();
    return {};
  }
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  std::vector<ContinuationToken> checkpoint = impl_->prompt;
  checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                    impl_->generated.end());
  const std::size_t position = impl_->runner->CheckpointPosition(state);
  if (position < impl_->lease.cached_tokens() || position > checkpoint.size()) {
    throw std::runtime_error(
        "text runner checkpoint is outside executed token history: position=" +
        std::to_string(position) +
        " cached=" + std::to_string(impl_->lease.cached_tokens()) +
        " prompt=" + std::to_string(impl_->prompt.size()) +
        " generated=" + std::to_string(impl_->generated.size()));
  }
  checkpoint.resize(position);
  if (capabilities.snapshot && capabilities.fork) {
    CommitMetrics metrics = impl_->snapshot_metrics;
    metrics.snapshot_bytes += impl_->lease.Commit(
        std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
        std::move(checkpoint));
    impl_.reset();
    return metrics;
  }

  std::unique_ptr<TextRunnerSnapshot> snapshot;
  CommitMetrics metrics;
  metrics.snapshot_bytes =
      impl_->lease.Commit(std::move(checkpoint), std::move(snapshot));
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
    auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
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
         impl_->prefill_offset == impl_->fallback_position)) {
      if (impl_->decode_ready)
        impl_->runner->PrepareCancellation(state);
      checkpoint.assign(impl_->prompt.begin(),
                        impl_->prompt.begin() + impl_->prefill_offset);
      if (impl_->decode_ready)
        checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                          impl_->generated.end());
      const auto position = impl_->runner->CheckpointPosition(state);
      if (position < impl_->lease.cached_tokens() ||
          position > checkpoint.size())
        throw std::runtime_error(
            "cancelled checkpoint is outside completed work");
      checkpoint.resize(position);
    } else {
      state.Invalidate();
    }
    CommitMetrics metrics = impl_->snapshot_metrics;
    if (capabilities.snapshot && capabilities.fork) {
      metrics.snapshot_bytes += impl_->lease.Commit(
          std::move(impl_->snapshot_tokens), std::move(impl_->prompt_snapshot),
          std::move(checkpoint));
    } else {
      metrics.snapshot_bytes = impl_->lease.Commit(std::move(checkpoint));
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
    dynamic_cast<TextRunnerState&>(impl_->lease.state())
        .SetCancellationCheck({});
    impl_->lease.Invalidate();
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
  return impl_->cache.capacity();
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
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease.state()),
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
        .state = dynamic_cast<TextRunnerState&>(request->impl_->lease.state()),
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
    std::size_t cache_prefix_tokens, bool stop_at_eos) {
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

  const std::span<const std::uint8_t> identity =
      context ? std::span<const std::uint8_t>(context->cache_identity)
              : std::span<const std::uint8_t>{};
  const std::span<const ContinuationInputPrefix> input_prefixes =
      context
          ? std::span<const ContinuationInputPrefix>(context->cache_prefixes)
          : std::span<const ContinuationInputPrefix>{};
  // Reuse full checkpoints as well as branching fallbacks. Both cache tiers
  // ensure unmarked legacy entries leave room for the stable boundary.
  auto lease = impl_->cache.Acquire(
      prompt, is_cancelled, identity,
      [&](ContinuationState& state) {
        auto& text_state = dynamic_cast<TextRunnerState&>(state);
        text_state.SetCancellationCheck(is_cancelled);
        text_state.SetStopAtEos(stop_at_eos);
        impl_->validated.runner->SetPromptContext(text_state, context);
      },
      reuse_prompt, cache_prefix_tokens, input_prefixes);
  if (!lease) {
    return {};
  }
  // A live frontier can contain generated tokens beyond the prompt snapshot.
  // Freeze it before the first branch mutates it, so peers restore the same
  // decode history instead of feeding generated output through prefill.
  // C1 captures any required fallback immediately before prefill instead.
  if (impl_->cache.capacity() > 1 && lease.cache_hit() &&
      impl_->validated.descriptor.capabilities.snapshot &&
      impl_->validated.descriptor.capabilities.fork &&
      !(is_cancelled && is_cancelled())) {
    const auto prefix =
        std::span<const TextRunnerToken>(prompt).first(lease.cached_tokens());
    if (!lease.HasSnapshotFor(prefix)) {
      auto& runner = *impl_->validated.runner;
      auto& state = dynamic_cast<TextRunnerState&>(lease.state());
      std::size_t bytes = 0;
      std::shared_ptr<const TextRunnerSnapshot> snapshot;
      const auto started = std::chrono::steady_clock::now();
      try {
        if (runner.CheckpointPosition(state) != prefix.size())
          throw std::logic_error("live checkpoint position mismatch");
        bytes = runner.SnapshotPayloadBytes(state);
        if (lease.TryReserveSnapshot(bytes, prefix.size(), true)) {
          snapshot = runner.Snapshot(state);
          if (snapshot == nullptr)
            throw std::runtime_error("live checkpoint capture failed");
          if (lease.PublishSnapshot({prefix.begin(), prefix.end()}, snapshot) ==
              0)
            snapshot.reset();
        }
      } catch (...) {
        snapshot.reset();
        lease.SkipSnapshot(SnapshotEventReason::kCaptureFailure, bytes,
                           prefix.size());
      }
      if (snapshot) {
        if (impl_->disk_store) {
          try {
            const auto prefix_identity =
                PrefixInputIdentity(identity, input_prefixes, prefix.size());
            (void)impl_->disk_store->SaveAsync(
                impl_->validated.runner, {prefix.begin(), prefix.end()},
                snapshot, {prefix_identity.begin(), prefix_identity.end()});
          } catch (...) {
            // RAM branching does not depend on optional disk admission.
          }
        }
        std::ostringstream line;
        line << "event=live_checkpoint tokens=" << prefix.size()
             << " bytes=" << bytes << " capture_ms="
             << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
        Logger::Info("cache", line.str());
      }
    }
  }
  if (reuse_prompt && !lease.cache_hit() && impl_->disk_store != nullptr &&
      !(is_cancelled && is_cancelled())) {
    const auto restore_start = std::chrono::steady_clock::now();
    try {
      const auto restored = impl_->disk_store->RestoreLongestPrefix(
          *impl_->validated.runner,
          dynamic_cast<TextRunnerState&>(lease.state()), prompt, identity,
          cache_prefix_tokens, input_prefixes);
      if (restored.restored) {
        if (context)
          impl_->validated.runner->SetPromptContext(
              dynamic_cast<TextRunnerState&>(lease.state()), context);
        lease.AdoptRestoredPrefix(
            restored.token_count, restored.file_bytes,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - restore_start)
                .count());
      } else {
        auto& state = dynamic_cast<TextRunnerState&>(lease.state());
        state.SetCancellationCheck(is_cancelled);
        impl_->validated.runner->SetPromptContext(state, context);
      }
    } catch (...) {
      lease.state().Invalidate();
      auto& state = dynamic_cast<TextRunnerState&>(lease.state());
      state.SetCancellationCheck(is_cancelled);
      impl_->validated.runner->SetPromptContext(state, context);
    }
  }
  std::vector<std::size_t> boundaries;
  if (reuse_prompt && impl_->disk_store != nullptr &&
      !(is_cancelled && is_cancelled())) {
    try {
      boundaries = impl_->disk_store->SharedPrefixBoundaries(
          *impl_->validated.runner, prompt, impl_->shared_prefix_min_tokens,
          impl_->shared_prefix_max_boundaries, identity, input_prefixes);
    } catch (...) {
      boundaries.clear();
    }
    std::erase_if(boundaries, [&lease](std::size_t boundary) {
      return boundary <= lease.cached_tokens();
    });
  }
  Request request(std::make_unique<Request::Impl>(
      impl_->validated.runner, impl_->disk_store, std::move(lease),
      std::move(prompt), std::move(boundaries), sampling_config,
      std::move(context), cache_prefix_tokens, impl_->cache.capacity() > 1));
  // Prompts that diverge from a retained one after a long common prefix,
  // such as new conversations under one system prompt, publish a checkpoint
  // at the divergence point. Later prompts sharing it restore it exactly
  // instead of the nearest grid checkpoint. A divergence in the last few
  // tokens, such as an edited final message, is already covered by this
  // request's own stable checkpoint; another copy would only take its RAM.
  if (reuse_prompt) {
    const auto shared = impl_->cache.CommonPrefixTokens(
        request.prompt(), identity, input_prefixes);
    if (shared >=
            request.prefill_position() + Request::kSharedPrefixMinTokens &&
        shared + Request::kSharedCheckpointSlack < request.prompt_tokens()) {
      const auto position = request.ShareCheckpoint(shared);
      if (position != 0 && Logger::Enabled(LogLevel::kDebug)) {
        std::ostringstream line;
        line << "event=shared_prefix_learned tokens=" << position
             << " cached=" << request.prefill_position()
             << " prompt_tokens=" << request.prompt_tokens();
        Logger::Debug("cache", line.str());
      }
    }
  }
  return request;
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const CancellationCheck& is_cancelled) {
  return Acquire(std::move(prompt), sampling::SamplingConfig{}, is_cancelled);
}

std::size_t TextRunnerPool::CachedPrefixTokens(
    std::span<const TextRunnerToken> prompt,
    const TextPromptContext* context) const {
  if (context == nullptr)
    return impl_->cache.CachedPrefixTokens(prompt);
  return impl_->cache.CachedPrefixTokens(prompt, context->cache_identity,
                                         context->cache_prefixes);
}

}  // namespace gufo::server
