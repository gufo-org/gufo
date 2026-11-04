#include "src/models/deepseek_v4_flash/continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/experimental/snapshot_copy.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/models/deepseek_v4_flash/continuation_hooks.hpp"
#include "src/models/deepseek_v4_flash/runtime/dspark_internal.h"
#include "src/models/deepseek_v4_flash/runtime/model_data_internal.h"

namespace gufo::models::deepseek_v4_flash {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{}, draft_tokens{}, draft_capture{};
  ds4_dspark_request_state request;
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
static_assert(std::is_trivially_copyable_v<Metadata>);
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
ContinuationGeometry ModelGeometry(std::uint32_t draft_stages) {
  ContinuationGeometry geometry{.head_dim = DS4_N_HEAD_DIM,
                                .indexer_head_dim = DS4_N_INDEXER_HEAD_DIM,
                                .window = DS4_N_SWA,
                                .vocabulary = DS4_N_VOCAB,
                                .draft_stages = draft_stages};
  for (std::uint32_t layer = 0; layer < DS4_N_LAYER; ++layer)
    geometry.ratios.push_back(ds4_layer_ratio(layer));
  return geometry;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    ds4_session_set_hooks(native(), this);
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
    Release();
    ds4_session_set_hooks(native(), nullptr);
  }
  [[nodiscard]] ds4_session* native() const { return session->session_; }
  bool IsValid() const noexcept override {
    return !restoring && !failed && ds4_session_coherent(native());
  }
  void RequireIdle() const {
    if (restoring || failed || pending)
      throw std::logic_error("DeepSeek slot is not ready for execution");
  }
  void BeforeWrite(std::uint32_t first, std::uint32_t end) override {
    RequireIdle();
    for (const auto& component : adapter.layout_.Components()) {
      if (component.descriptor.kind != cache::ComponentKind::kAppendRows)
        continue;
      const auto ratio =
          component.part == ContinuationPart::kTokens
              ? 1
              : adapter.layout_.Geometry().ratios[component.index];
      if (first / ratio < end / ratio)
        guard.BeforeOverwrite(component.descriptor.id, first / ratio,
                              end / ratio);
    }
  }
  void BeforeReset() override {
    RequireIdle();
    GuardAll();
  }
  void BeforeRestore() override {
    throw std::logic_error("DeepSeek slots restore through the adapter");
  }
  [[nodiscard]] Rows Capacity(const ContinuationComponent& component) const {
    return component.part == ContinuationPart::kTokens
               ? adapter.context_
               : adapter.context_ /
                     adapter.layout_.Geometry().ratios[component.index];
  }
  void GuardAll() {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, 0, Capacity(component));
  }
  void Release() noexcept {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(component.descriptor.id, 0, Capacity(component));
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false};
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
      layout_(ModelGeometry(mode == core::SessionMode::kSpeculative && model_
                                ? ds4_engine_dspark_stages(model_->engine_)
                                : 0),
              context),
      identity_(std::move(identity)) {
  if ((mode != core::SessionMode::kAutoregressive &&
       mode != core::SessionMode::kSpeculative) ||
      identity_.empty() || !model_ || context > model_->MaxContext() ||
      (mode == core::SessionMode::kSpeculative && !model_->HasDspark()))
    throw std::invalid_argument("invalid DeepSeek adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(1);  // Component layout version.
  put(context);
  put(mode == core::SessionMode::kSpeculative);
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another DeepSeek adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
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
    throw std::logic_error("cannot query an invalid DeepSeek slot");
  return layout_.Positions(
      static_cast<std::uint32_t>(ds4_session_pos(state.native())));
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid DeepSeek prefill plan");
  std::vector<Rows> result;
  const Rows capacity = state.session->PrefillCapacity();
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid DeepSeek component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  switch (component.part) {
    case ContinuationPart::kTokens:
      return ds4_session_tokens(state.native());
    case ContinuationPart::kAttentionRows:
      return ds4_session_continuation_tensor(
          state.native(), ds4_continuation_tensor::kAttentionRows,
          component.index);
    case ContinuationPart::kIndexerRows:
      return ds4_session_continuation_tensor(
          state.native(), ds4_continuation_tensor::kIndexerRows,
          component.index);
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.RequireIdle();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete DeepSeek restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate DeepSeek frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered.front().valid_rows > context_)
    throw std::invalid_argument("DeepSeek frontier exceeds context");
  const auto target = static_cast<std::uint32_t>(ordered.front().valid_rows);
  if (ordered != layout_.Positions(target))
    throw std::invalid_argument("inconsistent DeepSeek component frontiers");
  // Reserve first; any refusal leaves execution valid. Guard all ranges while
  // the original state is still readable.
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  if (!ds4_session_begin_restore(state.native(), target))
    throw std::runtime_error("DeepSeek restore could not reserve its rows");
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
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

// Logical byte layout of one private component at a frontier. A load may
// arrive before the metadata, so rings are placed from the frontier alone. A
// session without DSpark coverage ignores the draft window it receives.
ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component,
    std::uint32_t position) const {
  Region result;
  std::size_t offset = 0;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host});
    offset += bytes;
  };
  const auto& geometry = layout_.Geometry();
  const std::size_t head_row = std::size_t{geometry.head_dim} * sizeof(float);
  const auto ring = [&](void* tensor, ds4_continuation_ring shape) {
    const auto first = (position - shape.live) % shape.capacity;
    const auto tail = std::min(shape.live, shape.capacity - first);
    add(static_cast<std::byte*>(tensor) + first * head_row, tail * head_row);
    add(tensor, (shape.live - tail) * head_row);
  };
  const auto tensor = [&](ds4_continuation_tensor kind) {
    return ds4_session_continuation_tensor(state.native(), kind,
                                           component.index);
  };
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(ds4_session_frontier_logits(state.native()),
          component.descriptor.state_bytes, true);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), state.metadata.size(), true);
      break;
    case ContinuationPart::kDraftWindow:
      ring(tensor(ds4_continuation_tensor::kDraftWindow),
           ds4_session_draft_ring(state.native(), position));
      break;
    case ContinuationPart::kLayerState: {
      ring(tensor(ds4_continuation_tensor::kRaw),
           ds4_session_raw_ring(state.native(), position));
      offset = std::size_t{geometry.window} * head_row;
      const auto ratio = geometry.ratios[component.index];
      if (!ratio)
        break;
      const auto attention = CompressorStateBytes(ratio, geometry.head_dim);
      add(tensor(ds4_continuation_tensor::kAttentionValues), attention);
      add(tensor(ds4_continuation_tensor::kAttentionScores), attention);
      if (ratio != 4)
        break;
      const auto indexer =
          CompressorStateBytes(ratio, geometry.indexer_head_dim);
      add(tensor(ds4_continuation_tensor::kIndexerValues), indexer);
      add(tensor(ds4_continuation_tensor::kIndexerScores), indexer);
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
    throw std::invalid_argument("incorrect DeepSeek private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid DeepSeek frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid DeepSeek private piece");
  auto* native = state.native();
  const auto position = static_cast<std::uint32_t>(ds4_session_pos(native));
  if (!ds4_session_rows_at(native, position))
    throw std::logic_error("DeepSeek rows disagree with the frontier");
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  auto region = [&] {
    if (component.part == ContinuationPart::kMetadata) {
      const bool capture = ds4_session_draft_capture(native);
      const auto draft_tokens = ds4_session_draft_tokens(native);
      if (capture && draft_tokens != position)
        throw std::runtime_error("DSpark prefix lacks complete support state");
      Metadata metadata{.position = position,
                        .draft_tokens = draft_tokens,
                        .draft_capture = capture,
                        .request = *ds4_session_dspark_state(native)};
      std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
      return Region{{{host_metadata.data(), 0, host_metadata.size(), true}}};
    }
    // Rows the session does not cover serialize as zeros.
    if (component.part == ContinuationPart::kDraftWindow &&
        !ds4_session_draft_tokens(native))
      return Region{};
    return PrivateRegion(state, component, position);
  }();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes: clearing
        // the whole recycled copy on the host dominates capture time.
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
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect DeepSeek private buffer size");
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
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid DeepSeek private piece");
    const auto region = PrivateRegion(
        state, component,
        static_cast<std::uint32_t>(state.expected.front().valid_rows));
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
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
// Rows are copied rather than borrowed: compressed caches grow by reallocation,
// so their device address does not outlive the next execution.
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid DeepSeek slot");
  const auto positions = Positions(slot);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > positions[id.value - 1].valid_rows ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid DeepSeek row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : static_cast<std::byte*>(RowData(state, component)) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (component.part == ContinuationPart::kTokens && !bytes.empty())
          std::memcpy(bytes.data(), source, bytes.size());
        else if (!bytes.empty())
          Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                        transfer.Native()));
      });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("DeepSeek load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid DeepSeek row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : static_cast<std::byte*>(RowData(state, component)) + first * row;
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
                                            bytes.size(), transfer.Native()));
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
    throw std::logic_error("DeepSeek validation requires an active restore");
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
  const auto target = state.expected.front().valid_rows;
  const bool speculative = mode_ == core::SessionMode::kSpeculative;
  if (metadata.version != 1 || metadata.position != target ||
      metadata.draft_capture > 1 || (metadata.draft_capture && !speculative) ||
      metadata.draft_tokens != (metadata.draft_capture ? target : 0) ||
      (speculative &&
       metadata.request.force_plain_request == bool(metadata.draft_capture)))
    return refuse();
  if (!ds4_session_finish_restore(state.native(), metadata.position,
                                  metadata.draft_tokens, metadata.draft_capture,
                                  &metadata.request))
    return refuse();
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    state.Release();
    ds4_session_clear(state.native());
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}
}  // namespace gufo::models::deepseek_v4_flash
