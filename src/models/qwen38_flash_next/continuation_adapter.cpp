#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/experimental/state_memory.hpp>
#include <gufo/ext/snapshot_copy.hpp>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/continuation_hooks.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
bool Draft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool RotatedKeys(ContinuationPart part) {
  return part == ContinuationPart::kKey ||
         part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftPooledKeys;
}
bool Pool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
struct Metadata {
  std::uint32_t version{2};
  std::uint32_t target{}, draft{}, blocks{}, draft_blocks{}, hidden_base{};
  std::uint32_t residual_valid{}, image_bytes{};
  NgramHistory ngram;
  MtpLengthState policy;
  speculative::PromptLookup<>::State lookup;
  std::array<std::uint8_t, 32> image{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
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
    this->session->continuation_hooks_ = this;
    this->session->session_->continuation_hooks_ = this;
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
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
    session->continuation_hooks_ = nullptr;
    session->session_->continuation_hooks_ = nullptr;
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && (guarding || session->IsValid());
  }
  void BeforeExecution() override {
    if (!IsValid() || pending || at_boundary)
      throw std::logic_error("Flash-Next slot is not ready for execution");
  }
  void BeforeWrite(bool draft, std::uint32_t first, std::uint32_t end,
                   std::uint32_t first_block,
                   std::uint32_t end_block) override {
    Guard(draft, first, end, first_block, end_block, false);
  }
  void BeforeRewrite(bool draft, std::uint32_t first, std::uint32_t end,
                     std::uint32_t first_block,
                     std::uint32_t end_block) override {
    Guard(draft, first, end, first_block, end_block, true);
  }
  void Guard(bool draft, std::uint32_t first, std::uint32_t end,
             std::uint32_t first_block, std::uint32_t end_block,
             bool in_place) {
    // Model execution temporarily disables its public validity flag. The
    // adapter's restore/failure status and outstanding transfers still gate
    // every native write, including graph replay and speculative rollback.
    if (restoring || failed || pending || at_boundary)
      throw std::logic_error("Flash-Next slot is not ready for mutation");
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows ||
          Draft(component.part) != draft)
        continue;
      const auto a = Pool(component.part) ? first_block : first;
      const auto b = Pool(component.part) ? end_block : end;
      if (a < b)
        guard.BeforeOverwrite(component.descriptor.id, a, b);
    }
    // Pages of these rows that a checkpoint or another session shares become
    // the session's own before the write: copies where it still needs their
    // rows (those before an append, or all of them when rewriting in place),
    // cleared pages elsewhere.
    std::string error;
    if (!adapter.model_->executor_->UnshareKv(
            *session->session_, draft, first, end, in_place ? end : first,
            first_block, end_block, in_place ? end_block : first_block, &error))
      throw std::runtime_error("Flash-Next state page copy failed: " + error);
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    if (restoring || pending || at_boundary)
      throw std::logic_error("cannot reset a busy Flash-Next slot");
    GuardAll();
  }
  void AfterReset() override {
    // Native Reset queues zero fills. Settle them before exposing an empty
    // slot to captures on independent streams.
    Check(hipStreamSynchronize(adapter.model_->executor_->stream()));
  }
  void BeforeRestore() override {
    BeforeExecution();
    GuardAll();
  }
  void BeforeRelease() noexcept override {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  void GuardAll() {
    CaptureDuringGuard readable(*this);
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(
            component.descriptor.id, 0,
            Pool(component.part)
                ? adapter.context_ / adapter.model_->config().compress_ratio + 1
                : adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  /// The in-pass checkpoint a boundary capture reads; null when it reads
  /// the live session at the boundary.
  const rocm::PrefillCheckpoint* checkpoint{};
  bool at_boundary{false};
  rocm::Session& CaptureState() const {
    return checkpoint ? *checkpoint->state : *session->session_;
  }
  struct CaptureDuringGuard {
    explicit CaptureDuringGuard(State& state)
        : state(state), previous(std::exchange(state.guarding, true)) {}
    ~CaptureDuringGuard() { state.guarding = previous; }
    State& state;
    bool previous;
  };
  bool restoring{false}, guarding{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         core::SessionMode mode,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      mode_(mode),
      context_(context),
      layout_(model_ ? model_->config() : Config{},
              mode == core::SessionMode::kSpeculative, context,
              model_ ? model_->executor_->max_speculative() : 0),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasMtp()))
    throw std::invalid_argument("invalid Flash-Next adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Private components attest the target checkpoint boundary.
  put(Session::kSnapshotPayloadVersion);
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  put(model_->executor_->max_speculative());
  put(model_->PrefillCapacity());
  put(model_->DecodeConcurrency());
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
  if (int device = 0; hipGetDevice(&device) == hipSuccess)
    park_ = gufo::mem::PagePark::ForDevice(device);
  else
    (void)hipGetLastError();
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another Flash-Next adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.BeforeExecution();
  return *state.session;
}
void ContinuationAdapter::PrefillThrough(
    cache::Slot& slot, std::span<const std::int32_t> prompt,
    std::span<const std::uint32_t> boundaries,
    const std::function<void(std::size_t)>& capture,
    std::span<const std::int32_t> next) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (!capture)
    throw std::invalid_argument("missing component checkpoint consumer");
  std::string error;
  const auto consume = [&](std::size_t index,
                           const rocm::PrefillCheckpoint* checkpoint) {
    struct BoundaryScope {
      State& state;
      bool guarding;
      BoundaryScope(State& state, const rocm::PrefillCheckpoint* checkpoint)
          : state(state), guarding(std::exchange(state.guarding, true)) {
        state.checkpoint = checkpoint;
        state.at_boundary = true;
      }
      ~BoundaryScope() {
        // The checkpoint arena may be reused by the next native pass. Even a
        // rejected consumer must finish reading it before leaving this scope.
        {
          const std::lock_guard lock(state.jobs_mutex);
          for (const auto& job : state.jobs)
            (void)job->Wait();
        }
        state.checkpoint = nullptr;
        state.at_boundary = false;
        state.guarding = guarding;
      }
    } scope(state, checkpoint);
    capture(index);
    if (state.pending)
      throw std::logic_error("in-pass capture returned with pending transfers");
  };
  if (!state.session->SyncThroughComponents(prompt, boundaries, consume, &error,
                                            next))
    throw std::runtime_error("component in-pass prefill failed: " + error);
}
std::optional<cache::CostEstimate> ContinuationAdapter::Cost() const {
  // Fixed least-squares fit of the retained gfx1151 depth sweep; see this
  // model's BENCHMARKS.md. Restore rates are shared measured hardware controls.
  return mode_ == core::SessionMode::kSpeculative
             ? cache::CostEstimate{0.000591766466, 0.000067029640, 96e9, 3.3e9}
             : cache::CostEstimate{0.000573457239, 0.000058015348, 96e9, 3.3e9};
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true,
          .persistent_encoding = true,
          .shared_restore = park_ != nullptr};
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
  auto session = model_->CreateSession(mode_, context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  return layout_.Positions(native.position_, native.mtp_.position,
                           native.blocks_, native.mtp_.blocks);
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid Flash-Next prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    const auto remaining = end - first;
    const auto capacity = remaining <= model_->PlannedThrough()
                              ? model_->PlannedThrough()
                              : model_->PlannedChunk();
    first += std::min<Rows>(remaining, capacity);
    result.push_back(first);
  }
  return result;
}
std::vector<cache::ComponentPosition>
ContinuationAdapter::PrefillCheckpointPositions(const cache::Slot& slot,
                                                Rows boundary) const {
  const auto& state = As(slot);
  if (!state.IsValid() || state.pending || state.at_boundary)
    throw std::logic_error(
        "Flash-Next slot is not ready for checkpoint planning");
  if (boundary <= state.session->Position() || boundary > context_ ||
      !state.session->PrefillCheckpointBytes(
          static_cast<std::uint32_t>(boundary)))
    throw std::invalid_argument("unsupported Flash-Next in-pass boundary");
  const auto& native = *state.session->session_;
  const auto& config = model_->config();
  const auto draft =
      mode_ == core::SessionMode::kSpeculative ? boundary - 1 : 0;
  return layout_.Positions(
      boundary, draft,
      boundary > config.indexer_top_k ? boundary / config.compress_ratio
                                      : native.blocks_,
      draft > config.indexer_top_k ? draft / config.compress_ratio
                                   : native.mtp_.blocks);
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid Flash-Next component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  auto& native = state.CaptureState();
  switch (component.part) {
    case ContinuationPart::kTokens:
      return state.session->tokens_.data();
    case ContinuationPart::kKey:
      return native.attention_[component.layer].k_cache;
    case ContinuationPart::kValue:
      return native.attention_[component.layer].v_cache;
    case ContinuationPart::kPooledKeys:
      return native.attention_[component.layer].block_k;
    case ContinuationPart::kDraftKey:
      return native.mtp_.k_cache;
    case ContinuationPart::kDraftValue:
      return native.mtp_.v_cache;
    case ContinuationPart::kDraftPooledKeys:
      return native.mtp_.block_k;
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

gufo::mem::Region* ContinuationAdapter::KvRegion(
    State& state, const ContinuationComponent& component) const {
  auto& native = *state.session->session_;
  const bool value = component.part == ContinuationPart::kValue ||
                     component.part == ContinuationPart::kDraftValue;
  if (component.part == ContinuationPart::kDraftPooledKeys)
    return native.mtp_pooled_regions_.size() == 1
               ? native.mtp_pooled_regions_[0].get()
               : nullptr;
  if (component.part == ContinuationPart::kDraftKey ||
      component.part == ContinuationPart::kDraftValue)
    return native.mtp_kv_regions_.size() == 2
               ? native.mtp_kv_regions_[value].get()
               : nullptr;
  const bool pooled = component.part == ContinuationPart::kPooledKeys;
  if (component.part != ContinuationPart::kKey && !value && !pooled)
    return nullptr;
  // Per attention layer, in layer order: keys and values, and pooled keys.
  const auto& config = model_->config();
  std::size_t layers = 0;
  for (std::uint32_t layer = 0; layer < component.layer; ++layer)
    layers += !config.IsLinearLayer(layer);
  if (pooled)
    return layers < native.pooled_regions_.size()
               ? native.pooled_regions_[layers].get()
               : nullptr;
  const auto index = 2 * layers + value;
  return index < native.kv_regions_.size() ? native.kv_regions_[index].get()
                                           : nullptr;
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  (void)BeginSharedRestore(slot, positions, {});
}

std::vector<cache::RestoredRows> ContinuationAdapter::BeginSharedRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions,
    std::span<const cache::RestoreSource> sources) {
  auto& state = As(slot);
  state.BeforeExecution();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Flash-Next restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate Flash-Next frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  const auto frontier = [&](ContinuationPart part) -> std::uint32_t {
    for (std::size_t i = 0; i < ordered.size(); ++i)
      if (layout_.Components()[i].part == part) {
        if (ordered[i].valid_rows > context_)
          throw std::invalid_argument("Flash-Next frontier exceeds context");
        return static_cast<std::uint32_t>(ordered[i].valid_rows);
      }
    return 0;
  };
  const auto target = frontier(ContinuationPart::kTokens);
  const auto draft = frontier(ContinuationPart::kDraftKey);
  const auto expected =
      layout_.Positions(target, draft, frontier(ContinuationPart::kPooledKeys),
                        frontier(ContinuationPart::kDraftPooledKeys));
  if (ordered != expected)
    throw std::invalid_argument("inconsistent Flash-Next component frontiers");
  // Allocate host capacity and coverage first; any refusal leaves execution
  // valid. Guard all ranges while the original state is still readable.
  state.session->tokens_.reserve(target);
  state.session->logits_.reserve(model_->VocabSize());
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  Check(hipStreamSynchronize(model_->executor_->stream()));
  state.session->session_->PreserveSnapshots(0, 0);
  // Pages the checkpoint's rows already live in are mapped into the session
  // as they are; pages others share that the restore rewrites are released.
  // Every K/V region goes through this, sources or not: a load must never
  // write a page a checkpoint still reads.
  std::vector<cache::RestoredRows> taken;
  {
    using gufo::mem::kPageBytes;
    struct Plan {
      gufo::mem::Region* region{};
      cache::ComponentId component;
      std::size_t row_bytes{};
      // Rows the restore brings: mapped here or loaded afterwards.
      std::size_t rows{};
      gufo::mem::RestorePlan pages;
    };
    std::vector<Plan> plans;
    for (const auto& component : layout_.Components())
      if (auto* region = KvRegion(state, component);
          region != nullptr && region->demand_mapped())
        plans.push_back({region,
                         component.descriptor.id,
                         component.descriptor.row_bytes,
                         static_cast<std::size_t>(
                             std::ranges::find(ordered, component.descriptor.id,
                                               &cache::ComponentPosition::id)
                                 ->valid_rows),
                         {*region, park_.get()}});
    for (const auto& source : sources)
      if (const auto plan =
              std::ranges::find(plans, source.component, &Plan::component);
          plan != plans.end() && kPageBytes % plan->row_bytes == 0)
        plan->pages.Add(source.first * plan->row_bytes, source.bytes);
    for (auto& plan : plans) {
      const auto adopted = plan.pages.Apply(plan.rows);
      const auto rows = kPageBytes / plan.row_bytes;
      for (std::size_t p = 0; p < adopted.size(); ++p) {
        if (!adopted[p])
          continue;
        if (!taken.empty() && taken.back().component == plan.component &&
            taken.back().end == p * rows)
          taken.back().end += rows;
        else
          taken.push_back({plan.component, p * rows, (p + 1) * rows});
        coverage[plan.component.value - 1].emplace_back(p * kPageBytes,
                                                        (p + 1) * kPageBytes);
      }
    }
    // The pages mapped above reach the device only after the remap barrier:
    // once for all buffers, before any load or read.
    if (std::string error;
        !plans.empty() && !plans.front().region->SettleRemaps(&error))
      throw std::runtime_error("Flash-Next state page map failed: " + error);
  }
  // Restores write K/V rows without a forward pass. A session that gave its
  // pages to a checkpoint needs new ones for every row a load brings, and
  // making a page takes 50-80 us: 1,064 pages for a 76K-token disk restore
  // took 54 ms here. The page pool's thread backs the rows instead while
  // the loads read their files; each load backs what it writes if the pool
  // is not there yet (CopyRowsIn), and Validate backs the rest.
  model_->executor_->MapKvAhead(*state.session->session_, target, draft);
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
  state.session->valid_ = false;
  // The restore rewrites the tokens in place, not through RestoreSnapshot.
  state.session->lookup_.Clear();
  state.session->tokens_.resize(target);
  state.session->ForgetDraftVocab();
  state.session->logits_.resize(model_->VocabSize());
  return taken;
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
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
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component, bool loading) const {
  auto& session = *state.session;
  auto& native = state.CaptureState();
  Region result;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false,
                       std::size_t offset = 0) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
  };
  const auto bytes = component.descriptor.state_bytes;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(state.checkpoint ? const_cast<float*>(state.checkpoint->logits.data())
                           : session.logits_.data(),
          bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(native.linear_[component.layer].conv_state, bytes);
      break;
    case ContinuationPart::kRecurrent:
      add(native.linear_[component.layer].state, bytes);
      break;
    case ContinuationPart::kPleHistory:
      add(native.ple_history_, bytes);
      break;
    case ContinuationPart::kDraftResidual:
      if (loading || native.mtp_.residual_valid)
        add(native.mtp_.h, bytes);
      break;
    case ContinuationPart::kKeptHidden:
      add(native.mtp_.target_hidden,
          loading ? bytes
                  : std::size_t{state.checkpoint ? state.checkpoint->hidden_rows
                                                 : session.KeptHiddenRows()} *
                        model_->config().HcDim() * sizeof(float));
      break;
    case ContinuationPart::kRawKeys:
    case ContinuationPart::kDraftRawKeys: {
      const auto positions = loading ? state.expected : Positions(state);
      const bool draft = Draft(component.part);
      std::uint32_t position = 0, blocks = 0;
      for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto part = layout_.Components()[i].part;
        if (part ==
            (draft ? ContinuationPart::kDraftKey : ContinuationPart::kTokens))
          position = static_cast<std::uint32_t>(positions[i].valid_rows);
        if (part == (draft ? ContinuationPart::kDraftPooledKeys
                           : ContinuationPart::kPooledKeys))
          blocks = static_cast<std::uint32_t>(positions[i].valid_rows);
      }
      const auto& c = model_->config();
      const auto begin = blocks * c.compress_ratio;
      const auto rows = position - begin;
      const auto physical = begin & (native.index_capacity_ - 1);
      const auto tail = std::min(rows, native.index_capacity_ - physical);
      const auto width = std::size_t{c.indexer_head_dim} * sizeof(float);
      auto* pointer = reinterpret_cast<std::byte*>(
          draft ? native.mtp_.index_k
                : native.attention_[component.layer].index_k);
      add(pointer + physical * width, tail * width);
      add(pointer, (rows - tail) * width, false, tail * width);
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect Flash-Next private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.session->session_->spec_tokens_)
    throw std::logic_error(
        "cannot capture an invalid or speculative Flash-Next frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid Flash-Next private piece");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  if (component.part == ContinuationPart::kMetadata) {
    auto& session = *state.session;
    auto& native = state.CaptureState();
    Metadata metadata{};
    metadata.target = native.position_;
    metadata.draft = native.mtp_.position;
    metadata.blocks = native.blocks_;
    metadata.draft_blocks = native.mtp_.blocks;
    // The base a restore rebuilds from the rows a frontier keeps (none
    // without MTP), whichever pass produced it: a split pass, an in-pass
    // checkpoint, a live boundary capture and a native restore record the
    // same base.
    metadata.hidden_base =
        native.position_ - (state.checkpoint ? state.checkpoint->hidden_rows
                                             : session.KeptHiddenRows());
    metadata.residual_valid = native.mtp_.residual_valid;
    metadata.ngram = native.ngram_;
    metadata.policy = session.draft_length_.State();
    metadata.lookup = session.lookup_.Save();
    const auto identity = session.ImageIdentity(native.position_);
    metadata.image_bytes = static_cast<std::uint32_t>(identity.size());
    if (!identity.empty()) {
      if (identity.size() != metadata.image.size())
        throw std::logic_error("invalid Flash-Next image identity");
      std::ranges::copy(identity, metadata.image.begin());
    }
    std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
  }
  auto region = PrivateRegion(state, component, false);
  if (component.part == ContinuationPart::kMetadata)
    region.pieces = {{host_metadata.data(), 0, host_metadata.size(), true}};
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes, which
        // serialize as zeros: clearing the whole recycled copy on the host
        // dominated capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect Flash-Next private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid Flash-Next private piece");
    const auto region = PrivateRegion(state, component, true);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid Flash-Next slot");
  const auto& native = state.CaptureState();
  const auto rows =
      component.part == ContinuationPart::kTokens ? native.position_
      : Pool(component.part)
          ? (Draft(component.part) ? native.mtp_.blocks : native.blocks_)
          : (Draft(component.part) ? native.mtp_.position : native.position_);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > rows || bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid Flash-Next row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : static_cast<std::byte*>(RowData(state, component)) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (component.part == ContinuationPart::kTokens && !bytes.empty())
          std::memcpy(bytes.data(), source, bytes.size());
        else if (!bytes.empty() && RotatedKeys(component.part) &&
                 native.yarn_factor_ != 1.0F)
          CopyNativeKeys(native, component, first, end, bytes, transfer);
        else if (!bytes.empty())
          Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                        transfer.Native()));
      });
}
void ContinuationAdapter::CopyNativeKeys(
    const rocm::Session& native, const ContinuationComponent& component,
    Rows first, Rows end, std::span<std::byte> bytes,
    gufo::hip::TransferStream& transfer) const {
  const auto stream = transfer.Native();
  const bool draft = Draft(component.part);
  const auto* rows = reinterpret_cast<const __half*>(
      component.part == ContinuationPart::kKey
          ? native.attention_[component.layer].k_cache
      : component.part == ContinuationPart::kPooledKeys
          ? native.attention_[component.layer].block_k
      : component.part == ContinuationPart::kDraftKey ? native.mtp_.k_cache
                                                      : native.mtp_.block_k);
  const auto* rope = draft ? native.vision_input_.rope()
                           : native.attention_[component.layer].rope;
  const auto copy = [&](__half* destination) {
    model_->executor_->CopyNativeKeys(
        native, rows, static_cast<std::uint32_t>(first),
        static_cast<std::uint32_t>(end - first), Pool(component.part), rope,
        destination, stream);
  };
  void* mapped{};
  if (hipHostGetDevicePointer(&mapped, bytes.data(), 0) == hipSuccess &&
      mapped) {
    copy(static_cast<__half*>(mapped));
    Check(hipGetLastError());
    return;
  }
  // Ordinary host buffers are not visible to the kernel.
  (void)hipGetLastError();
  void* staging{};
  Check(hipMalloc(&staging, bytes.size()));
  copy(static_cast<__half*>(staging));
  const auto copied = hipMemcpyAsync(bytes.data(), staging, bytes.size(),
                                     hipMemcpyDeviceToHost, stream);
  // Only this transfer must settle before its staging allocation is freed.
  // The native FIFO stream may also carry unrelated cache transfers.
  const auto settled = transfer.Synchronize();
  (void)hipFree(staging);
  Check(copied);
  if (settled != TransferResult::kSucceeded)
    throw std::runtime_error("Flash-Next key capture synchronization failed");
}
std::span<const std::byte> ContinuationAdapter::BorrowRowsView(
    const cache::Slot& slot, cache::ComponentId id, Rows first,
    Rows end) const {
  const auto& state = As(slot);
  const auto& component = Component(id);
  // The host token vector can move when execution appends. Device append
  // allocations stay fixed for the session and are guarded before mutation.
  if (component.part == ContinuationPart::kTokens)
    return {};
  if (!state.IsValid() ||
      component.descriptor.kind != cache::ComponentKind::kAppendRows)
    throw std::logic_error("cannot borrow invalid Flash-Next rows");
  const auto positions = Positions(slot);
  const auto position =
      std::ranges::find(positions, id, &cache::ComponentPosition::id);
  const auto row = component.descriptor.row_bytes;
  if (position == positions.end() || first >= end || end > position->valid_rows)
    throw std::invalid_argument("invalid Flash-Next borrowed range");
  // Cached keys hold the native rotation (CopyRowsOut converts them).
  if (RotatedKeys(component.part) && state.CaptureState().yarn_factor_ != 1.0F)
    return {};
  return {static_cast<const std::byte*>(
              RowData(const_cast<State&>(state), component)) +
              first * row,
          (end - first) * row};
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Flash-Next load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid Flash-Next row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : static_cast<std::byte*>(RowData(state, component)) + first * row;
    // Back the rows this load writes; lock-free once they are.
    if (auto* region = KvRegion(state, component);
        region != nullptr && !bytes.empty())
      if (std::string error; !region->Ensure(end, &error))
        throw std::bad_alloc();
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (component.part == ContinuationPart::kTokens && !bytes.empty())
            std::memcpy(destination, bytes.data(), bytes.size());
          else if (!bytes.empty())
            Check(hip::CopySnapshotToDevice(destination, bytes.data(),
                                            bytes.size(), transfer));
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
    throw std::logic_error("Flash-Next validation requires an active restore");
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
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  auto& session = *state.session;
  auto& native = *session.session_;
  auto policy = session.draft_length_;
  if (metadata.version != 2 || metadata.hidden_base > metadata.target ||
      metadata.target - metadata.hidden_base >
          model_->executor_->max_speculative() ||
      metadata.residual_valid > 1 ||
      (metadata.residual_valid && !metadata.draft) ||
      (metadata.image_bytes != 0 && metadata.image_bytes != 32) ||
      !policy.Restore(metadata.policy) ||
      !speculative::PromptLookup<>::Valid(metadata.lookup, metadata.target))
    return refuse();
  try {
    if (layout_.Positions(metadata.target, metadata.draft, metadata.blocks,
                          metadata.draft_blocks) != state.expected)
      return refuse();
    // Every row and the tile past the frontier that attention reads.
    if (std::string error; !model_->executor_->EnsureKv(native, metadata.target,
                                                        metadata.draft, &error))
      return refuse();
    const auto identity = session.ImageIdentity(metadata.target);
    if (metadata.image_bytes && (identity.size() != 32 ||
                                 !std::ranges::equal(identity, metadata.image)))
      return refuse();
    if (metadata.image_bytes || (session.image_prompt_ && identity.empty())) {
      // Preserve the attached request's future images as well as its matching
      // computed prefix, including checkpoints before the first image.
      native.RestoreVisionLayout(session.image_prompt_->rope,
                                 model_->executor_->stream());
    } else {
      native.ConfigureVision(nullptr, nullptr, model_->executor_->stream());
      session.image_prompt_.reset();
    }
    Check(hipStreamSynchronize(model_->executor_->stream()));
  } catch (...) {
    return refuse();
  }
  native.position_ = metadata.target;
  native.blocks_ = metadata.blocks;
  native.mtp_.position = metadata.draft;
  native.mtp_.blocks = metadata.draft_blocks;
  native.mtp_.residual_valid = metadata.residual_valid;
  native.ngram_ = metadata.ngram;
  native.spec_base_ = metadata.target;
  native.spec_tokens_ = 0;
  native.snapshot_lineage_.reset();
  // Cached keys hold the native rotation whatever factor wrote them, so one
  // history can serve sessions at any factor. The next Feed or decode moves
  // this session to the factor its positions need.
  model_->executor_->AdoptYarnFactor(native, 1.0F);
  session.hidden_base_ = metadata.hidden_base;
  session.draft_length_ = policy;
  session.lookup_.Restore(metadata.lookup);
  session.draft_token_ = 0;
  session.stats_ = {};
  session.valid_ = true;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending || state.at_boundary)
      return false;
    resetting = &state;
    state.BeforeRelease();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    // Reassign a healthy resident slot without rebuilding its MTP graphs.
    // Failed state and explicit Session::Reset still release all scratch.
    state.session->ResetImpl(!state.failed && state.session->IsValid());
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
}  // namespace gufo::models::qwen38_flash_next
