#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace gufo::server {
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

ContinuationCache::SnapshotSupport MakeSnapshotSupport(
    ValidatedRunner* validated) {
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
  };
}

std::uint64_t MakeRngState() {
  std::random_device random_device;
  return (static_cast<std::uint64_t>(random_device()) << 32U) ^
         static_cast<std::uint64_t>(random_device());
}

}  // namespace

void TextModelRunner::AdvanceBatch(
    std::span<const TextRunnerAdvance> advances) const {
  for (const auto& advance : advances) {
    Advance(advance.state.get(), advance.token);
  }
}

TextDecodeStep TextModelRunner::DecodeStep(TextRunnerState& state,
                                           std::size_t max_tokens,
                                           float temperature,
                                           std::uint64_t* rng_state) const {
  if (max_tokens == 0) {
    throw std::invalid_argument(
        "text runner decode step budget must be at least one token");
  }
  auto selection = SelectNext(state, temperature, rng_state);
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

std::unique_ptr<TextRunnerSnapshot> TextModelRunner::Snapshot(
    const TextRunnerState&) const {
  throw std::logic_error("text runner does not support snapshots");
}

void TextModelRunner::RestoreOrFork(TextRunnerState&,
                                    const TextRunnerSnapshot&) const {
  throw std::logic_error("text runner does not support snapshot restore/fork");
}

struct TextRunnerPool::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner, std::size_t state_count)
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
            MakeSnapshotSupport(&validated)) {}

  ValidatedRunner validated;
  ContinuationCache cache;
};

struct TextRunnerPool::Request::Impl {
  Impl(std::shared_ptr<TextModelRunner> model_runner,
       ContinuationCache::Lease state_lease,
       std::vector<TextRunnerToken> prompt_tokens,
       const CancellationCheck& is_cancelled)
      : runner(std::move(model_runner)),
        lease(std::move(state_lease)),
        prompt(std::move(prompt_tokens)),
        prefill_offset(lease.cached_tokens()),
        decode_ready(prefill_offset == prompt.size()),
        rng_state(MakeRngState()) {
    auto& state = dynamic_cast<TextRunnerState&>(lease.state());
    state.SetCancellationCheck(is_cancelled);
    if (lease.cache_hit()) {
      runner->PreparePrefixReuse(
          state,
          std::span<const TextRunnerToken>(prompt).first(prefill_offset));
    }
  }

  std::shared_ptr<TextModelRunner> runner;
  ContinuationCache::Lease lease;
  std::vector<TextRunnerToken> prompt;
  std::vector<TextRunnerToken> generated;
  std::size_t prefill_offset{0};
  bool decode_ready{false};
  bool stopped{false};
  std::optional<TextDecodeSelection> pending_selection;
  std::uint64_t rng_state{0};
};

TextRunnerPool::Request::Request() = default;

TextRunnerPool::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

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

std::size_t TextRunnerPool::Request::cache_restore_bytes() const noexcept {
  return impl_ != nullptr ? impl_->lease.restored_snapshot_bytes() : 0;
}

double TextRunnerPool::Request::cache_restore_ms() const noexcept {
  return impl_ != nullptr ? impl_->lease.restore_ms() : 0.0;
}

std::size_t TextRunnerPool::Request::prompt_tokens() const noexcept {
  return impl_ != nullptr ? impl_->prompt.size() : 0;
}

bool TextRunnerPool::Request::prefill_complete() const noexcept {
  return impl_ != nullptr && impl_->decode_ready;
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

  const std::size_t remaining = impl_->prompt.size() - impl_->prefill_offset;
  auto step = impl_->runner->Prefill(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), impl_->prompt,
      impl_->prefill_offset, max_input_tokens);
  const std::size_t maximum_consumed = std::min(remaining, max_input_tokens);
  if (step.consumed_tokens == 0 || step.consumed_tokens > maximum_consumed) {
    throw std::runtime_error(
        "text runner returned an invalid prefill token count");
  }

  impl_->prefill_offset += step.consumed_tokens;
  const bool reached_frontier = impl_->prefill_offset == impl_->prompt.size();
  if (step.decode_ready != reached_frontier) {
    throw std::runtime_error(
        "text runner returned an inconsistent prefill boundary");
  }
  impl_->decode_ready = reached_frontier;
  return step;
}

TextDecodeSelection TextRunnerPool::Request::SelectNext(float temperature) {
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

  auto selection = impl_->runner->SelectNext(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), temperature,
      &impl_->rng_state);
  if (selection.stop) {
    impl_->stopped = true;
    return selection;
  }
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
  impl_->runner->Advance(dynamic_cast<TextRunnerState&>(impl_->lease.state()),
                         impl_->pending_selection->token);
  impl_->pending_selection.reset();
}

TextDecodeStep TextRunnerPool::Request::DecodeStep(std::size_t max_tokens,
                                                   float temperature) {
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

  auto step = impl_->runner->DecodeStep(
      dynamic_cast<TextRunnerState&>(impl_->lease.state()), max_tokens,
      temperature, &impl_->rng_state);
  if (step.selections.size() > max_tokens ||
      (step.selections.empty() && !step.stop)) {
    throw std::runtime_error("text runner returned an invalid decode step");
  }
  for (const auto& selection : step.selections) {
    if (selection.stop) {
      throw std::runtime_error(
          "text runner decode step contains an embedded stop selection");
    }
    impl_->generated.push_back(selection.token);
  }
  impl_->stopped = step.stop;
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

  auto& state = dynamic_cast<TextRunnerState&>(impl_->lease.state());
  state.SetCancellationCheck({});
  if (!impl_->runner->Descriptor().capabilities.prefix_reuse) {
    impl_->lease.Invalidate();
    impl_.reset();
    return {};
  }
  std::vector<ContinuationToken> checkpoint = impl_->prompt;
  checkpoint.insert(checkpoint.end(), impl_->generated.begin(),
                    impl_->generated.end());
  const std::size_t position = impl_->runner->CheckpointPosition(state);
  if (position < impl_->lease.cached_tokens() || position > checkpoint.size()) {
    throw std::runtime_error(
        "text runner checkpoint is outside executed token history");
  }
  checkpoint.resize(position);
  std::unique_ptr<ContinuationSnapshot> snapshot;
  CommitMetrics metrics;
  const auto capabilities = impl_->runner->Descriptor().capabilities;
  if (capabilities.snapshot && capabilities.fork) {
    const auto snapshot_start = std::chrono::steady_clock::now();
    snapshot = impl_->runner->Snapshot(state);
    metrics.snapshot_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - snapshot_start)
                              .count();
    if (snapshot == nullptr) {
      throw std::runtime_error("text runner returned an empty snapshot");
    }
    metrics.snapshot_bytes = snapshot->PayloadBytes();
  }
  impl_->lease.Commit(std::move(checkpoint), std::move(snapshot));
  impl_.reset();
  return metrics;
}

void TextRunnerPool::Request::Invalidate() noexcept {
  if (impl_ != nullptr) {
    dynamic_cast<TextRunnerState&>(impl_->lease.state())
        .SetCancellationCheck({});
    impl_->lease.Invalidate();
    impl_.reset();
  }
}

TextRunnerPool::TextRunnerPool(std::shared_ptr<TextModelRunner> runner,
                               std::size_t state_count)
    : impl_(std::make_unique<Impl>(std::move(runner), state_count)) {}

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

void TextRunnerPool::AdvanceBatch(std::span<Request*> requests,
                                  const TextExecutionPlan& plan) {
  if (plan.kind != TextExecutionPlanKind::kBatched || requests.size() < 2 ||
      requests.size() > plan.physical_width) {
    throw std::invalid_argument("invalid batched text execution plan");
  }
  if (std::find(impl_->validated.plans.begin(), impl_->validated.plans.end(),
                plan) == impl_->validated.plans.end()) {
    throw std::invalid_argument(
        "text runner does not support the requested batched plan");
  }

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
    });
  }

  impl_->validated.runner->AdvanceBatch(advances);
  for (auto* request : requests) {
    request->impl_->pending_selection.reset();
  }
}

TextRunnerPool::Request TextRunnerPool::Acquire(
    std::vector<TextRunnerToken> prompt,
    const CancellationCheck& is_cancelled) {
  if (prompt.empty()) {
    throw std::invalid_argument("text runner prompt must not be empty");
  }
  if (prompt.size() > impl_->validated.descriptor.max_context) {
    throw std::length_error("text runner prompt exceeds model context");
  }

  auto lease = impl_->cache.Acquire(prompt, is_cancelled);
  if (!lease) {
    return {};
  }
  return Request(
      std::make_unique<Request::Impl>(impl_->validated.runner, std::move(lease),
                                      std::move(prompt), is_cancelled));
}

}  // namespace gufo::server
