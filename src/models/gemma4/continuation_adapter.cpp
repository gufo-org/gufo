#include "src/models/gemma4/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "src/models/gemma4/continuation_hooks.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"

namespace gufo::models::gemma4 {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
// Global rows share 256-row chunks. Window chunks of 64 rows bound each
// checkpoint's private tail to 63 sliding rows (31B: 52 MB, 26B: 13 MB).
constexpr Rows kGlobalChunkRows = 256;
constexpr Rows kWindowChunkRows = 64;
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{};
  std::uint32_t images{};
  std::uint32_t reserved{};
  std::array<ImageSpan, kContinuationImages> image{};
};
static_assert(std::is_trivially_copyable_v<Metadata>);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
// Loaded byte ranges cover [begin, end) without a gap.
// Copies a gather launch can serve: 16-byte aligned memory the device reads
// and writes in place (device or registered host memory, not pageable).
bool Accessible(const void* pointer) {
  hipPointerAttribute_t attributes{};
  return hipPointerGetAttributes(&attributes, pointer) == hipSuccess &&
         attributes.type != hipMemoryTypeUnregistered;
}
bool Gatherable(const rocm::CopyRun& run) {
  return (reinterpret_cast<std::uintptr_t>(run.from) |
          reinterpret_cast<std::uintptr_t>(run.to) | run.bytes) %
                 16 ==
             0 &&
         Accessible(run.from) && Accessible(run.to);
}
// Device copies one gather launch takes; a fuller batch launches early.
constexpr std::size_t kBatchRuns = 1024;
// Gathers in flight per slot, each reading its own descriptor region.
constexpr std::size_t kGathers = 2;
bool Covered(Coverage ranges, std::size_t begin, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = begin;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    try {
      // Pinned once: allocating pinned memory later waits for queued forwards.
      Check(hipHostMalloc(&descriptors,
                          kGathers * kBatchRuns * sizeof(rocm::CopyRun)));
      Check(hipEventCreateWithFlags(&fence, hipEventDisableTiming));
      this->session->cache_->hooks = this;
      const std::lock_guard lock(adapter.executed_mutex_);
      adapter.states_.push_back(this);
    } catch (...) {
      // A failed construction never reaches ~State; release what succeeded.
      if (fence)
        (void)hipEventDestroy(fence);
      if (descriptors)
        (void)hipHostFree(descriptors);
      throw;
    }
  }
  // Copies queued on the adapter's copy stream; `done` follows them.
  struct Launch {
    hipEvent_t done{};
    Launch() { Check(hipEventCreateWithFlags(&done, hipEventDisableTiming)); }
    ~Launch() { (void)hipEventDestroy(done); }
    Launch(const Launch&) = delete;
    Launch& operator=(const Launch&) = delete;
  };
  struct Job {
    State* owner;
    bool load;
    // Set once the job's device copies are queued; a job left without one
    // fails when waited on.
    std::shared_ptr<const Launch> launch;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    // A batched job is not ready before its batch launches.
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || (launch && hipEventQuery(launch->done) == hipSuccess);
    }
    TransferResult Wait() noexcept {
      // A settled job may outlive its slot: the slot settles every job
      // before it is destroyed. An unsettled one reads the slot below, so
      // its waiter must not run concurrently with the slot's destruction.
      {
        const std::lock_guard lock(mutex);
        if (result)
          return *result;
      }
      owner->Flush();
      const std::lock_guard lock(mutex);
      if (!result) {
        result = launch && hipEventSynchronize(launch->done) == hipSuccess
                     ? TransferResult::kSucceeded
                     : TransferResult::kFailed;
        launch.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  // Launches the batched device copies as one gather on the copy stream.
  void Flush() noexcept {
    const std::lock_guard lock(jobs_mutex);
    FlushLocked();
  }
  // Whether no gather of this slot is still copying.
  bool GathersIdleLocked() const noexcept {
    return std::ranges::all_of(gathers, [](const auto& region) {
      return !region || hipEventQuery(region->done) == hipSuccess;
    });
  }
  void FlushLocked() noexcept {
    if (batch.empty())
      return;
    std::shared_ptr<const Launch> launch;
    try {
      const auto stream = adapter.copy_stream_;
      adapter.AfterExecution(*this, stream);
      // The gather that last used this region may still read it.
      auto& region = gathers[next_gather];
      if (region)
        Check(hipEventSynchronize(region->done));
      std::uint64_t longest = 0;
      for (const auto& run : batch_runs)
        longest = std::max(longest, run.bytes);
      auto* runs = descriptors + next_gather * kBatchRuns;
      std::ranges::copy(batch_runs, runs);
      auto queued = std::make_shared<Launch>();
      rocm::CopyRuns(runs, static_cast<std::uint32_t>(batch_runs.size()),
                     longest, stream);
      Check(hipEventRecord(queued->done, stream));
      region = queued;
      next_gather = (next_gather + 1) % kGathers;
      launch = std::move(queued);
    } catch (...) {
      // The jobs fail when waited on; a launched gather must still finish
      // reading their buffers first.
      (void)hipStreamSynchronize(adapter.copy_stream_);
    }
    for (const auto& job : batch) {
      const std::lock_guard job_lock(job->mutex);
      job->launch = launch;
    }
    batch.clear();
    batch_runs.clear();
  }
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    BeforeRelease();
    session->cache_->hooks = nullptr;
    {
      const std::lock_guard lock(adapter.executed_mutex_);
      std::erase(adapter.states_, this);
    }
    for (const auto& region : gathers)
      if (region)
        (void)hipEventSynchronize(region->done);
    (void)hipEventDestroy(fence);
    (void)hipHostFree(descriptors);
  }
  // An empty session is a valid cold slot; execution clears the session's
  // own flag while it runs, so guards read through `guarding`.
  bool IsValid() const noexcept override {
    return !restoring && !failed &&
           (guarding || session->Position() == 0 || session->IsValid());
  }
  void RequireIdle() const {
    if (!IsValid() || pending)
      throw std::logic_error("Gemma 4 slot is not ready for execution");
  }
  void BeforeWrite(std::uint32_t first, std::uint32_t end) override {
    if (restoring || failed || pending)
      throw std::logic_error("Gemma 4 slot is not ready for mutation");
    // Rows past the frontier (rejected drafts, lookahead) are rewritten
    // routinely, always at or past any checkpoint this branch captured.
    if (first < written_end)
      rewrite_from = std::min(rewrite_from, first);
    written_end = std::max(written_end, end);
    for (std::uint32_t p = first; p < end; ++p)
      occupant[p % occupant.size()] = p;
    const Guarding readable(*this);
    const Rows ring = adapter.ring_;
    for (const auto& component : adapter.components_) {
      const auto& d = component.descriptor;
      if (!cache::HasRows(d.kind) || first >= end)
        continue;
      guard.BeforeOverwrite(d.id, first, end);
      // A ring slot held the position one ring earlier.
      if (d.kind == cache::ComponentKind::kWindowRows && end > ring)
        guard.BeforeOverwrite(d.id, first > ring ? first - ring : 0,
                              end - ring);
    }
  }
  void GuardAll() {
    const Guarding readable(*this);
    for (const auto& component : adapter.components_)
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
  }
  void BeforeRelease() noexcept {
    const Guarding readable(*this);
    for (const auto& component : adapter.components_)
      if (cache::HasRows(component.descriptor.kind))
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
  }
  struct Guarding {
    explicit Guarding(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~Guarding() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  // A fresh branch: no rows written or kept.
  void ForgetRows() {
    std::ranges::fill(occupant, kNoRow);
    written_end = 0;
    rewrite_from = kNoRow;
  }
  static constexpr std::uint32_t kNoRow = UINT32_MAX;
  // Position each ring slot holds; the branch's written end, and the first
  // position it wrote again below that end.
  std::vector<std::uint32_t> occupant =
      std::vector<std::uint32_t>(adapter.ring_, kNoRow);
  std::uint32_t written_end{0}, rewrite_from{kNoRow};
  bool restoring{false}, guarding{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  // Device copies waiting for one gather launch (under jobs_mutex).
  std::vector<std::shared_ptr<Job>> batch;
  std::vector<rocm::CopyRun> batch_runs;
  rocm::CopyRun* descriptors{};
  std::array<std::shared_ptr<const Launch>, kGathers> gathers;
  std::size_t next_gather{0};
  // Under adapter.executed_mutex_: `fence` follows every write of this
  // session queued before it unless the session wrote again (`dirty`).
  hipEvent_t fence{};
  bool dirty{false}, fenced{false};
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  Metadata metadata{};
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      ring_(model_ ? static_cast<std::uint32_t>(model_->SessionRingSlots())
                   : 0),
      identity_(std::move(identity)) {
  if (!model_ || identity_.empty() || !context ||
      context > model_->MaxContext())
    throw std::invalid_argument("invalid Gemma 4 adapter configuration");
  const Config& c = model_->config();
  const Rows window = c.sliding_window - 1;
  // Chunk-aligned window ranges never wrap the ring, and a capture reads a
  // whole window plus its partial leading chunk while up to kSplitRows rows
  // past the frontier (verification) may already be written; prompt
  // lookahead stops short of that chunk (Session::Extend).
  if (ring_ % kWindowChunkRows ||
      ring_ < window + kWindowChunkRows - 1 + rocm::kSplitRows)
    throw std::invalid_argument("Gemma 4 ring cannot hold a window");
  const auto add = [&](ContinuationPart part, std::uint32_t layer,
                       cache::ComponentDescriptor d) {
    d.id = {static_cast<std::uint32_t>(components_.size() + 1)};
    d.layout_version = 1;
    components_.push_back({d, part, layer});
  };
  add(ContinuationPart::kTokens, 0,
      {.kind = cache::ComponentKind::kAppendRows,
       .row_bytes = sizeof(TokenId),
       .rows_per_chunk = kGlobalChunkRows});
  const auto& widths = model_->executor_->key_widths();
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    for (const auto part : {ContinuationPart::kKey, ContinuationPart::kValue}) {
      const std::size_t row =
          std::size_t{part == ContinuationPart::kKey ? widths[l] : c.KvDim(l)} *
          sizeof(std::uint16_t);
      add(part, l,
          c.IsSliding(l)
              ? cache::ComponentDescriptor{.kind = cache::ComponentKind::
                                               kWindowRows,
                                           .row_bytes = row,
                                           .rows_per_chunk = kWindowChunkRows,
                                           .window_rows = window}
              : cache::ComponentDescriptor{
                    .kind = cache::ComponentKind::kAppendRows,
                    .row_bytes = row,
                    .rows_per_chunk = kGlobalChunkRows});
    }
  }
  add(ContinuationPart::kLogits, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = std::size_t{model_->VocabSize()} * sizeof(float)});
  add(ContinuationPart::kHidden, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = std::size_t{c.hidden_size} * sizeof(float)});
  add(ContinuationPart::kMetadata, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = sizeof(Metadata)});
  for (const auto& component : components_)
    descriptors_.push_back(component.descriptor);
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Component layout version (2: 1,024 image records).
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(ring_);
  // Prefill chunk width changes the rows' arithmetic.
  put(model_->options().prefill_chunk);
  auto& executor = *model_->executor_;
  if (executor.fences() != nullptr)
    throw std::logic_error("Gemma 4 model already has a continuation adapter");
  int least = 0, greatest = 0;
  Check(hipDeviceGetStreamPriorityRange(&least, &greatest));
  Check(hipStreamCreateWithPriority(&copy_stream_, hipStreamNonBlocking,
                                    greatest));
  if (const auto status =
          hipEventCreateWithFlags(&executed_, hipEventDisableTiming);
      status != hipSuccess) {
    (void)hipStreamDestroy(copy_stream_);
    Check(status);
  }
  executor.SetFences(this);
}
ContinuationAdapter::~ContinuationAdapter() {
  model_->executor_->SetFences(nullptr);
  (void)hipStreamSynchronize(copy_stream_);
  (void)hipStreamDestroy(copy_stream_);
  (void)hipEventDestroy(executed_);
}
void ContinuationAdapter::BeforeMutation(
    std::span<ContinuationHooks* const> writers) {
  const std::lock_guard lock(executed_mutex_);
  const auto writing = [&](State* state) {
    return std::ranges::find(writers, static_cast<ContinuationHooks*>(state)) !=
           writers.end();
  };
  for (State* state : states_)
    if (writing(state))
      state->dirty = true;
  // The other sessions' writes are all queued: their copies wait for this
  // point rather than for the work queued after it.
  for (State* state : states_)
    if (state->dirty && !writing(state)) {
      Check(hipEventRecord(state->fence, model_->executor_->stream()));
      state->dirty = false;
      state->fenced = true;
    }
}
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Gemma 4 adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > components_.size())
    throw std::invalid_argument("invalid Gemma 4 component ID");
  return components_[id.value - 1];
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
void ContinuationAdapter::AfterExecution(hipStream_t stream) const {
  const std::lock_guard lock(executed_mutex_);
  Check(hipEventRecord(executed_, model_->executor_->stream()));
  Check(hipStreamWaitEvent(stream, executed_, 0));
}
void ContinuationAdapter::AfterExecution(State& state,
                                         hipStream_t stream) const {
  {
    const std::lock_guard lock(executed_mutex_);
    if (state.fenced && !state.dirty) {
      Check(hipStreamWaitEvent(stream, state.fence, 0));
      return;
    }
  }
  AfterExecution(stream);
}
bool ContinuationAdapter::Executed(const State& state) const {
  const std::lock_guard lock(executed_mutex_);
  if (state.fenced && !state.dirty)
    return hipEventQuery(state.fence) == hipSuccess;
  return hipStreamQuery(model_->executor_->stream()) == hipSuccess;
}
std::vector<ContinuationBackingClass> ContinuationAdapter::BackingPlan(
    std::size_t ram) const {
  // Rows, tails and private states share the same committed pages; a payload
  // takes a contiguous page group and charges all of its slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true,
          .persistent_encoding = true,
          .in_place_restore = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Gemma 4 slot");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(descriptors_.size());
  for (const auto& d : descriptors_)
    positions.push_back({d.id, state.session->Position()});
  return positions;
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Gemma 4 prefill plan");
  // Session::Extend's chunks: a remainder of kSplitRows rows or fewer joins
  // the chunk before it. Images can end a pass early.
  const Rows chunk = model_->options().prefill_chunk;
  std::vector<Rows> result;
  while (first < end) {
    const auto remaining = end - first;
    first += remaining <= chunk + rocm::kSplitRows ? remaining : chunk;
    result.push_back(first);
  }
  return result;
}
std::vector<ContinuationAdapter::Piece> ContinuationAdapter::RowPieces(
    State& state, const ContinuationComponent& component, cache::Rows first,
    cache::Rows end) const {
  auto& kv = *state.session->cache_;
  const auto row = component.descriptor.row_bytes;
  auto* base = reinterpret_cast<std::byte*>(
      component.part == ContinuationPart::kKey ? kv.k[component.layer]
                                               : kv.v[component.layer]);
  if (component.descriptor.kind != cache::ComponentKind::kWindowRows)
    return {{base + first * row, (end - first) * row}};
  const auto at = first % kv.ring;
  const auto head = std::min<cache::Rows>(end - first, kv.ring - at);
  std::vector<Piece> pieces{{base + at * row, head * row}};
  if (end - first > head)
    pieces.push_back({base, (end - first - head) * row});
  return pieces;
}
ContinuationAdapter::Piece ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading,
    bool* host) const {
  auto& session = *state.session;
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      if (!loading && session.logits_on_device_) {
        *host = false;
        return {reinterpret_cast<std::byte*>(session.cache_->logits), bytes};
      }
      if (session.logits_.size() * sizeof(float) != bytes)
        throw std::logic_error("Gemma 4 session has no frontier logits");
      *host = true;
      return {reinterpret_cast<std::byte*>(session.logits_.data()), bytes};
    case ContinuationPart::kHidden:
      *host = false;
      return {reinterpret_cast<std::byte*>(session.cache_->hidden), bytes};
    case ContinuationPart::kMetadata:
      *host = true;
      return {reinterpret_cast<std::byte*>(&state.metadata), bytes};
    default:
      throw std::invalid_argument("component is not private state");
  }
}
cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load,
    const std::function<void(std::vector<rocm::CopyRun>&)>& plan) {
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    // Host copies run now; device copies join the gather batch when they
    // can, and otherwise copy at once.
    std::vector<rocm::CopyRun> runs;
    plan(runs);
    if (runs.empty()) {
      job->result = TransferResult::kSucceeded;
      --state.pending;
    } else if (runs.size() <= kBatchRuns &&
               std::ranges::all_of(runs, Gatherable)) {
      const std::lock_guard lock(state.jobs_mutex);
      if (state.batch_runs.size() + runs.size() > kBatchRuns)
        state.FlushLocked();
      state.batch.push_back(job);
      state.batch_runs.insert(state.batch_runs.end(), runs.begin(), runs.end());
      // Launch once the previous gather has finished: the copy stream works
      // while the caller queues its next transfers, which then launch
      // together instead of each paying for a launch. A gather that would
      // wait for queued forwards launches at the first wait instead, so it
      // does not hold up other slots' copies on the shared stream.
      if (state.GathersIdleLocked() && Executed(state))
        state.FlushLocked();
    } else {
      AfterExecution(state, copy_stream_);
      for (const auto& run : runs)
        Check(hipMemcpyAsync(run.to, run.from, run.bytes, hipMemcpyDefault,
                             copy_stream_));
      auto launch = std::make_shared<State::Launch>();
      Check(hipEventRecord(launch->done, copy_stream_));
      const std::lock_guard lock(job->mutex);
      job->launch = std::move(launch);
    }
  } catch (...) {
    // Buffers stay allocated until copies already queued finish.
    (void)hipStreamSynchronize(copy_stream_);
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  BeginRestore(slot, positions, {});
}
std::vector<cache::ResidentRange> ContinuationAdapter::ResidentRows(
    const cache::Slot& slot,
    std::span<const cache::ComponentPosition> positions) const {
  const auto& state = As(slot);
  if (!state.IsValid() || state.pending || positions.empty())
    return {};
  const auto n = positions.front().valid_rows;
  if (!n || n > state.session->Position() ||
      std::ranges::any_of(positions,
                          [&](const auto& p) { return p.valid_rows != n; }))
    return {};
  // Rows below the branch's first rewrite stay; window rows while their ring
  // slots still hold them.
  const cache::Rows end = std::min<cache::Rows>(n, state.rewrite_from);
  std::vector<cache::ResidentRange> kept;
  for (const auto& d : descriptors_) {
    if (!cache::HasRows(d.kind))
      continue;
    if (d.kind != cache::ComponentKind::kWindowRows) {
      kept.push_back({d.id, 0, end});
      continue;
    }
    cache::Rows first = end;
    const auto floor = cache::FirstChunkRow(d, n);
    while (first > floor &&
           state.occupant[(first - 1) % state.occupant.size()] == first - 1)
      --first;
    kept.push_back({d.id, first, end});
  }
  return kept;
}
void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions,
    std::span<const cache::ResidentRange> kept) {
  auto& state = As(slot);
  if (state.restoring || state.failed || state.pending)
    throw std::logic_error(
        "reset failed restores and settle transfers before BeginRestore");
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Gemma 4 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    if (std::ranges::count(positions, d.id, &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Gemma 4 frontier");
    ordered.push_back(
        *std::ranges::find(positions, d.id, &cache::ComponentPosition::id));
  }
  const auto n = ordered.front().valid_rows;
  if (!n || n > context_ || std::ranges::any_of(ordered, [&](const auto& p) {
        return p.valid_rows != n;
      }))
    throw std::invalid_argument("inconsistent Gemma 4 component frontiers");
  // Kept rows must be resident now; the tokens are kept with them.
  std::vector<Coverage> coverage(descriptors_.size());
  bool keep_tokens = false;
  if (!kept.empty()) {
    const auto resident = ResidentRows(slot, ordered);
    for (const auto& k : kept) {
      const auto r =
          std::ranges::find(resident, k.id, &cache::ResidentRange::id);
      if (r == resident.end() || k.first < r->first || k.end > r->end ||
          k.first >= k.end)
        throw std::invalid_argument("Gemma 4 kept rows are not resident");
      const auto& d = Component(k.id).descriptor;
      coverage[k.id.value - 1].emplace_back(k.first * d.row_bytes,
                                            k.end * d.row_bytes);
      keep_tokens |= Component(k.id).part == ContinuationPart::kTokens;
    }
  }
  // Allocate before guarding: a refusal leaves execution valid. Guard every
  // replaced range while the old rows are still readable.
  auto& session = *state.session;
  session.tokens_.reserve(n);
  session.logits_.reserve(model_->VocabSize());
  {
    const State::Guarding readable(state);
    for (const auto& component : components_) {
      const auto& d = component.descriptor;
      if (!cache::HasRows(d.kind))
        continue;
      const auto k = std::ranges::find(kept, d.id, &cache::ResidentRange::id);
      if (k == kept.end()) {
        state.guard.BeforeOverwrite(d.id, 0, context_);
        continue;
      }
      state.guard.BeforeOverwrite(d.id, 0, k->first);
      state.guard.BeforeOverwrite(d.id, k->end, context_);
    }
  }
  Check(hipStreamSynchronize(model_->executor_->stream()));
  auto tokens =
      keep_tokens ? std::move(session.tokens_) : std::vector<TokenId>{};
  session.Reset();
  session.tokens_ = std::move(tokens);
  if (kept.empty()) {
    state.ForgetRows();
  } else {
    // Ring slots past the checkpoint hold the discarded reply.
    for (auto& occupant : state.occupant)
      if (occupant != State::kNoRow && occupant >= n)
        occupant = State::kNoRow;
    state.rewrite_from = State::kNoRow;
  }
  state.written_end = static_cast<std::uint32_t>(n);
  session.held_global_.clear();
  session.held_sliding_.clear();
  session.cache_->written_from = 0;
  session.cache_->written_to = UINT32_MAX;
  session.tokens_.resize(n);
  session.logits_.resize(model_->VocabSize());
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata = {};
  state.restoring = true;
}

cache::Completion ContinuationAdapter::CopyRowsOut(
    const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<std::byte> bytes, cache::Stream&) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  const auto& d = component.descriptor;
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Gemma 4 slot");
  const cache::Rows n = state.session->Position();
  if (!cache::HasRows(d.kind) || first > end || end > n ||
      bytes.size() != (end - first) * d.row_bytes)
    throw std::invalid_argument("invalid Gemma 4 row capture");
  // Rows past the frontier (lookahead, drafts) may already be written, each
  // evicting the ring slot's earlier position.
  if (d.kind == cache::ComponentKind::kWindowRows)
    for (auto p = first; p < end; ++p)
      if (state.occupant[p % state.occupant.size()] != p)
        throw std::invalid_argument("Gemma 4 window rows left the ring");
  return Transfer(state, false, [&](auto& runs) {
    if (bytes.empty())
      return;
    if (component.part == ContinuationPart::kTokens) {
      std::memcpy(bytes.data(), state.session->tokens_.data() + first,
                  bytes.size());
      return;
    }
    std::size_t done = 0;
    for (const auto& piece : RowPieces(state, component, first, end)) {
      runs.push_back({piece.pointer, bytes.data() + done, piece.bytes});
      done += piece.bytes;
    }
  });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<const std::byte> bytes, cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Gemma 4 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto& d = component.descriptor;
    const auto n = state.expected[id.value - 1].valid_rows;
    if (!cache::HasRows(d.kind) || first > end || end > n ||
        first < cache::FirstChunkRow(d, n) ||
        bytes.size() != (end - first) * d.row_bytes)
      throw std::invalid_argument("invalid Gemma 4 row restore");
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * d.row_bytes,
                                                end * d.row_bytes);
    }
    return Transfer(state, true, [&](auto& runs) {
      if (bytes.empty())
        return;
      if (component.part == ContinuationPart::kTokens) {
        std::memcpy(state.session->tokens_.data() + first, bytes.data(),
                    bytes.size());
        return;
      }
      std::size_t done = 0;
      for (const auto& piece : RowPieces(state, component, first, end)) {
        runs.push_back({bytes.data() + done, piece.pointer, piece.bytes});
        done += piece.bytes;
      }
      if (d.kind == cache::ComponentKind::kWindowRows)
        for (auto p = first; p < end; ++p)
          state.occupant[p % state.occupant.size()] =
              static_cast<std::uint32_t>(p);
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Gemma 4 private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream&) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->Position() == 0)
    throw std::logic_error("cannot capture an empty or invalid Gemma 4 slot");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Gemma 4 private piece");
  std::unique_ptr<Metadata> metadata;
  if (component.part == ContinuationPart::kMetadata) {
    const auto& session = *state.session;
    if (session.images_.size() > kContinuationImages)
      throw std::length_error("a Gemma 4 checkpoint holds at most " +
                              std::to_string(kContinuationImages) + " images");
    metadata = std::make_unique<Metadata>();
    metadata->position = session.Position();
    metadata->images = static_cast<std::uint32_t>(session.images_.size());
    std::copy_n(session.images_.begin(), metadata->images,
                metadata->image.begin());
  }
  return Transfer(state, false, [&](auto& runs) {
    if (bytes.empty())
      return;
    if (metadata) {
      std::memcpy(bytes.data(),
                  reinterpret_cast<const std::byte*>(metadata.get()) + offset,
                  bytes.size());
      return;
    }
    bool host = false;
    const auto region = PrivateRegion(state, component, false, &host);
    if (host) {
      std::memcpy(bytes.data(), region.pointer + offset, bytes.size());
      return;
    }
    runs.push_back({region.pointer + offset, bytes.data(), bytes.size()});
  });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Gemma 4 load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Gemma 4 private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Gemma 4 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Gemma 4 private piece");
    bool host = false;
    const auto region = PrivateRegion(state, component, true, &host);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(state, true, [&](auto& runs) {
      if (bytes.empty())
        return;
      if (host)
        std::memcpy(region.pointer + offset, bytes.data(), bytes.size());
      else
        runs.push_back({bytes.data(), region.pointer + offset, bytes.size()});
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Gemma 4 validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& d = descriptors_[i];
    const auto n = expected.valid_rows;
    const bool rows = cache::HasRows(d.kind);
    if (!Covered(state.coverage[i],
                 rows ? cache::FirstChunkRow(d, n) * d.row_bytes : 0,
                 rows ? n * d.row_bytes : d.state_bytes))
      return refuse();
  }
  const auto& metadata = state.metadata;
  const auto n = state.expected.front().valid_rows;
  if (metadata.version != 1 || metadata.position != n ||
      metadata.images > kContinuationImages)
    return refuse();
  std::uint64_t previous_end = 0;
  for (std::uint32_t i = 0; i < metadata.images; ++i) {
    const auto& image = metadata.image[i];
    if (!image.rows || image.offset < previous_end ||
        std::uint64_t{image.offset} + image.rows > n)
      return refuse();
    previous_end = std::uint64_t{image.offset} + image.rows;
  }
  auto& session = *state.session;
  session.images_.assign(metadata.image.begin(),
                         metadata.image.begin() + metadata.images);
  session.logits_on_device_ = false;
  session.stats_ = {};
  session.valid_ = true;
  state.restoring = false;
  state.expected.clear();
  state.coverage.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    resetting = &state;
    state.BeforeRelease();
    auto& session = *state.session;
    session.Reset();
    session.held_global_.clear();
    session.held_sliding_.clear();
    session.cache_->written_from = 0;
    session.cache_->written_to = UINT32_MAX;
    state.ForgetRows();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting)
      resetting->failed = true;
    return false;
  }
}
}  // namespace gufo::models::gemma4
