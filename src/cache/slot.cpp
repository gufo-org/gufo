#include "src/cache/slot.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gufo::cache {
namespace detail {
struct SlotState {
  std::mutex mutex;
  SlotId id;
  std::uint64_t generation{1};
  bool alive{true};
};
struct RowState {
  ResourceCharge metadata;
  ResourceReservation reservation;
  ResourceCharge charge;
  std::shared_ptr<const void> owner;
  std::span<std::byte> buffer;
  std::shared_ptr<SlotState> slot;
  BorrowedLocation location;
  ComponentId component;
  Rows first{}, end{}, copied{};
  std::size_t row_bytes{};
  std::size_t bytes{};
  ResourceCategory category;
  mutable std::mutex mutex;
  std::condition_variable_any changed;
  std::size_t readers{};
  bool closing{false}, materialized{false}, retired{false};
};
struct LeaseState final : MutationGuard {
  struct Entry {
    ResourceCharge metadata;
    std::weak_ptr<RowState> rows;
  };
  ResourceCharge metadata;
  ResourceLedger& ledger;
  Adapter& adapter;
  Stream& stream;
  std::shared_ptr<SlotState> source;
  std::mutex mutex;
  std::list<Entry> rows;
  bool leased{false}, live{false}, spilling{false}, idle_failed{false};
  bool idle_pending{false};
  std::condition_variable_any changed;
  std::optional<IdleSpillConfig> spill;
  mutable std::mutex metrics_mutex;
  SpillMetrics metrics;
  std::stop_token stop;
  Slot* raw{};
  std::unique_ptr<Slot> slot;
  LeaseState(ResourceLedger& l, Adapter& a, Stream& s, SlotId id,
             std::optional<IdleSpillConfig> config)
      : ledger(l),
        adapter(a),
        stream(s),
        source(std::make_shared<SlotState>()),
        spill(std::move(config)) {
    if (!id.value)
      throw std::invalid_argument("zero slot identifier");
    if (spill) {
      if (!spill->piece_bytes || !spill->acquire_stream || !spill->device_idle)
        throw std::invalid_argument("invalid idle spill configuration");
      for (const auto& c : adapter.Components())
        if (c.kind == ComponentKind::kAppendRows &&
            c.row_bytes > spill->piece_bytes)
          throw std::invalid_argument("spill piece smaller than a row");
    }
    source->id = id;
    auto reservation = ledger.Reserve(ResourceCategory::kMetadata,
                                      sizeof(LeaseState) + sizeof(SlotState));
    metadata = reservation.Convert();
    slot = adapter.CreateSlot(*this);
    raw = slot.get();
  }
  ~LeaseState() override {
    // raw remains readable during the adapter's nonthrowing destruction guard.
    slot.reset();
    std::lock_guard lock(source->mutex);
    source->alive = false;
  }
  void CheckStop() const {
    if (stop.stop_requested())
      throw std::runtime_error("slot lease cancelled");
  }
  using Clock = std::chrono::steady_clock;
  static std::uint64_t Nanoseconds(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                start)
        .count();
  }
  // Execution is exclusive: either a foreground lease or one idle piece.
  // Row metadata remains independently synchronized for readers/persistence.
  bool CopyPiece(const std::shared_ptr<RowState>& r, bool idle, bool release) {
    Rows first, end;
    std::span<std::byte> buffer;
    {
      std::lock_guard lock(r->mutex);
      if (r->materialized || r->retired)
        return true;
      first = r->first + r->copied;
      const auto count =
          spill ? spill->piece_bytes / r->row_bytes : r->end - first;
      end = first + std::min<Rows>(count, r->end - first);
      buffer = r->buffer.subspan(r->copied * r->row_bytes,
                                 (end - first) * r->row_bytes);
      std::lock_guard source_lock(source->mutex);
      if (!source->alive || source->generation != r->location.generation)
        throw std::logic_error("spill source generation mismatch");
      ++r->readers;
    }
    // A source pin also retains backing/reservation and the source generation.
    // Always drop it, including submission, wait and conversion failures.
    struct SourcePin {
      std::shared_ptr<RowState> rows;
      ~SourcePin() {
        std::lock_guard lock(rows->mutex);
        --rows->readers;
        rows->changed.notify_all();
      }
    } pin{r};
    if (first != end) {
      std::unique_ptr<Stream> leased_stream;
      if (spill) {
        const auto stream_start = Clock::now();
        do {
          leased_stream = spill->acquire_stream();
          if (leased_stream)
            break;
          if (idle)
            return false;
          if (!release)
            CheckStop();
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while (true);
        std::lock_guard metrics_lock(metrics_mutex);
        metrics.stream_wait_ns += Nanoseconds(stream_start);
      }
      const auto start = Clock::now();
      auto copy = adapter.CopyRowsOut(*raw, r->component, first, end, buffer,
                                      leased_stream ? *leased_stream : stream);
      if (copy.Wait() != TransferResult::kSucceeded)
        throw std::runtime_error("borrowed row preservation failed");
      {
        std::lock_guard lock(metrics_mutex);
        (idle ? metrics.idle_bytes : metrics.foreground_bytes) += buffer.size();
        (idle ? metrics.idle_copy_ns : metrics.foreground_copy_ns) +=
            Nanoseconds(start);
      }
      std::lock_guard lock(r->mutex);
      r->copied += end - first;
    }
    if (!idle && !release)
      CheckStop();
    std::lock_guard lock(r->mutex);
    if (r->copied == r->end - r->first) {
      // Conversion and publication share the reader/persistence mutex. A
      // partial range remains borrowed even when some backing bytes are ready.
      auto charge = r->reservation.Convert();
      r->charge = std::move(charge);
      r->category = r->charge.Info().category;
      r->materialized = true;
      r->closing = false;
      r->changed.notify_all();
    }
    return true;
  }
  void Preserve(const std::shared_ptr<RowState>& r, bool release) {
    std::unique_lock lock(r->mutex);
    if (!r->materialized && !r->retired) {
      r->closing = true;
      lock.unlock();
      try {
        while (true) {
          if (!release)
            CheckStop();
          (void)CopyPiece(r, false, release);
          lock.lock();
          if (r->materialized)
            break;
          lock.unlock();
        }
      } catch (...) {
        if (!lock.owns_lock())
          lock.lock();
        r->closing = false;
        if (release)
          r->retired = true;
        r->changed.notify_all();
        if (!release)
          throw;
      }
    }
    // Source readers admitted before publication still prevent overwrite.
    if (release)
      r->changed.wait(lock, [&] { return r->readers == 0; });
    else if (!r->changed.wait(lock, stop, [&] { return r->readers == 0; }))
      throw std::runtime_error(
          "slot lease cancelled while waiting for readers");
  }
  void RunIdle(std::stop_token cancelled) {
    std::unique_lock lock(mutex);
    while (!cancelled.stop_requested()) {
      // No polling once retained ranges are complete. Only a released lease
      // can add work; stop-aware waiting also lets facade destruction drain.
      changed.wait(lock, cancelled, [&] {
        return !leased && live && idle_pending && !idle_failed;
      });
      if (cancelled.stop_requested())
        break;
      if (leased || !live || idle_failed)
        continue;
      std::shared_ptr<RowState> candidate;
      for (auto it = rows.begin(); it != rows.end();) {
        auto r = it->rows.lock();
        if (!r) {
          it = rows.erase(it);
          continue;
        }
        ++it;
        std::lock_guard row_lock(r->mutex);
        if (!r->materialized && !r->retired) {
          candidate = std::move(r);
          break;
        }
      }
      if (!candidate) {
        idle_pending = false;
        continue;
      }
      spilling = true;
      lock.unlock();
      bool failed = false;
      bool admitted = false;
      try {
        if (!cancelled.stop_requested() && spill->device_idle()) {
          // Exhausted streams defer idle work, rather than retiring retention.
          admitted = CopyPiece(candidate, true, false);
        }
      } catch (...) {
        failed = true;
        std::lock_guard metrics_lock(metrics_mutex);
        ++metrics.idle_failures;
      }
      lock.lock();
      spilling = false;
      idle_failed = failed;
      changed.notify_all();
      if (!admitted)
        changed.wait_for(lock, cancelled, std::chrono::milliseconds(1),
                         [] { return false; });
    }
  }
  void Guard(ComponentId component, Rows first, Rows end, bool release) {
    if (end < first)
      throw std::invalid_argument("reversed mutation range");
    if (!release)
      CheckStop();
    // Iterate without allocating a request-time snapshot. Leased execution
    // serializes Borrow and mutation; readers never change this registry.
    auto it = rows.begin();
    while (it != rows.end()) {
      auto r = it->rows.lock();
      if (!r) {
        it = rows.erase(it);
        continue;
      }
      ++it;
      if (r->component == component && first < r->end && r->first < end)
        Preserve(r, release);
    }
  }
  void PrepareReassignment() {
    CheckStop();
    // Acquisition can refuse/cancel without mutating the old live frontier.
    // Actual adapter release remains nonthrowing and may retire on failure.
    for (auto it = rows.begin(); it != rows.end();) {
      auto r = it->rows.lock();
      if (!r) {
        it = rows.erase(it);
        continue;
      }
      ++it;
      Preserve(r, false);
    }
    CheckStop();
  }
  void BeforeOverwrite(ComponentId c, Rows first, Rows end) override {
    Guard(c, first, end, false);
    CheckStop();
  }
  void BeforeRelease(ComponentId c, Rows first, Rows end) noexcept override {
    Guard(c, first, end, true);
  }
  void AfterReset() noexcept override {
    std::lock_guard lock(source->mutex);
    if (source->generation == std::numeric_limits<std::uint64_t>::max())
      std::terminate();
    ++source->generation;
  }
  void Invalidate() noexcept {
    live = false;
    if (slot && !adapter.Invalidate(*slot)) {
      slot.reset();
      raw = nullptr;
      AfterReset();
    }
  }
};
}  // namespace detail
RowPin::~RowPin() {
  Release();
}
RowPin::RowPin(RowPin&& other) noexcept
    : state_(std::move(other.state_)),
      location_(std::exchange(other.location_, {})),
      charge_(std::move(other.charge_)),
      owner_(std::move(other.owner_)) {}
RowPin& RowPin::operator=(RowPin&& other) noexcept {
  if (this != &other) {
    Release();
    state_ = std::move(other.state_);
    location_ = std::exchange(other.location_, {});
    charge_ = std::move(other.charge_);
    owner_ = std::move(other.owner_);
  }
  return *this;
}
RowPin RowPin::Clone() const {
  RowPin result;
  result.state_ = state_;
  result.location_ = location_;
  result.charge_ = charge_;
  result.owner_ = owner_;
  if (result.state_ && result.location_) {
    std::lock_guard lock(result.state_->mutex);
    ++result.state_->readers;
  }
  return result;
}
void RowPin::Release() noexcept {
  if (state_ && location_) {
    std::lock_guard lock(state_->mutex);
    --state_->readers;
    state_->changed.notify_all();
  }
  owner_.reset();
  charge_ = {};
  location_.reset();
  state_.reset();
}
BorrowedRows::BorrowedRows(std::shared_ptr<detail::RowState> state)
    : state_(std::move(state)) {}
RowPin BorrowedRows::Pin(std::stop_token stop) const {
  auto& r = *state_;
  std::unique_lock lock(r.mutex);
  if (!r.changed.wait(lock, stop, [&] { return !r.closing; }) ||
      stop.stop_requested())
    throw std::runtime_error("borrowed row read cancelled");
  if (r.retired)
    throw std::runtime_error("borrowed rows retired");
  RowPin pin;
  pin.state_ = state_;
  if (r.materialized) {
    pin.owner_ = r.owner;
  } else {
    std::lock_guard source_lock(r.slot->mutex);
    const bool valid =
        r.slot->alive && r.slot->generation == r.location.generation;
    assert(valid && "adapter overwrote borrowed rows without a guard");
    if (!valid)
      throw std::logic_error("borrowed slot generation mismatch");
    ++r.readers;
    pin.location_ = r.location;
  }
  return pin;
}
std::optional<BorrowedLocation> BorrowedRows::Location() const {
  std::lock_guard lock(state_->mutex);
  return state_->materialized || state_->retired
             ? std::nullopt
             : std::optional(state_->location);
}
std::shared_ptr<const void> BorrowedRows::Owner() const {
  std::lock_guard lock(state_->mutex);
  if (state_->retired)
    return {};
  if (!state_->materialized)
    throw std::logic_error("borrowed rows require a source pin");
  return state_->owner;
}
bool BorrowedRows::IsValid() const {
  std::lock_guard lock(state_->mutex);
  if (state_->retired)
    return false;
  if (state_->materialized)
    return true;
  std::lock_guard source_lock(state_->slot->mutex);
  return state_->slot->alive &&
         state_->slot->generation == state_->location.generation;
}
std::size_t BorrowedRows::Bytes() const {
  return state_->bytes;
}
ResourceCategory BorrowedRows::Category() const {
  std::lock_guard lock(state_->mutex);
  return state_->category;
}
PersistencePin BorrowedRows::PinPersistence() const {
  std::lock_guard lock(state_->mutex);
  if (state_->retired)
    throw std::runtime_error("borrowed rows retired");
  return state_->materialized ? state_->charge.PinPersistence()
                              : state_->reservation.PinPersistence();
}
SlotLease::SlotLease(std::shared_ptr<detail::LeaseState> state)
    : state_(std::move(state)) {}
SlotLease::~SlotLease() {
  Release();
}
SlotLease::SlotLease(SlotLease&& other) noexcept
    : state_(std::move(other.state_)), committed_(other.committed_) {}
SlotLease& SlotLease::operator=(SlotLease&& other) noexcept {
  if (this != &other) {
    Release();
    state_ = std::move(other.state_);
    committed_ = other.committed_;
  }
  return *this;
}
void SlotLease::Release() noexcept {
  if (!state_)
    return;
  if (!committed_ || state_->stop.stop_requested())
    state_->Invalidate();
  {
    std::lock_guard lock(state_->mutex);
    state_->leased = false;
    state_->idle_failed = false;
    state_->idle_pending = true;
    state_->changed.notify_all();
  }
  state_.reset();
}
Slot& SlotLease::Execution() const {
  if (!state_ || !state_->slot)
    throw std::logic_error("empty slot lease");
  state_->CheckStop();
  return *state_->slot;
}
BorrowedLocation SlotLease::Location() const {
  (void)Execution();
  std::lock_guard lock(state_->source->mutex);
  return {state_->source->id, state_->source->generation};
}
std::shared_ptr<BorrowedRows> SlotLease::Borrow(
    ComponentId c, Rows first, Rows end, ResourceReservation reservation,
    std::shared_ptr<const void> owner, std::span<std::byte> backing) {
  // Local ownership fixes teardown order even if validation/admission fails.
  auto held = std::move(reservation);
  auto retained_owner = std::move(owner);
  auto& slot = Execution();
  const auto info = held.Info();
  const auto descriptors = state_->adapter.Components();
  const auto d = std::ranges::find(descriptors, c, &ComponentDescriptor::id);
  const auto positions = state_->adapter.Positions(slot);
  const auto p = std::ranges::find(positions, c, &ComponentPosition::id);
  if (!slot.IsValid() || !retained_owner || !info.assigned_backing ||
      !info.reserved ||
      (info.category != ResourceCategory::kBackingAssigned &&
       info.category != ResourceCategory::kPrivateTail) ||
      d == descriptors.end() || d->kind != ComponentKind::kAppendRows ||
      p == positions.end() || first >= end || end > p->valid_rows ||
      !d->row_bytes ||
      end - first > std::numeric_limits<std::size_t>::max() / d->row_bytes ||
      backing.size() != (end - first) * d->row_bytes ||
      backing.size() > info.bytes)
    throw std::invalid_argument("invalid borrowed row description");
  auto metadata = state_->ledger.Reserve(
      ResourceCategory::kMetadata,
      sizeof(detail::RowState) + sizeof(BorrowedRows) +
          sizeof(detail::LeaseState::Entry) + 2 * sizeof(void*));
  auto rows = std::make_shared<detail::RowState>();
  rows->reservation = std::move(held);
  rows->owner = std::move(retained_owner);
  rows->buffer = backing;
  rows->slot = state_->source;
  rows->location = Location();
  rows->component = c;
  rows->row_bytes = d->row_bytes;
  rows->first = first;
  rows->end = end;
  rows->bytes = info.bytes;
  rows->category = info.category;
  rows->metadata = metadata.Convert();
  auto result = std::shared_ptr<BorrowedRows>(new BorrowedRows(rows));
  state_->rows.push_back({rows->metadata, rows});
  return result;
}
void SlotLease::Reset() {
  (void)Execution();
  if (!state_->adapter.Invalidate(*state_->slot))
    throw std::logic_error("cannot reset a busy slot");
  state_->live = false;
  committed_ = false;
}
void SlotLease::Commit() {
  if (!Execution().IsValid())
    throw std::logic_error("cannot commit an invalid slot");
  committed_ = true;
  state_->live = true;
  Release();
}
LeasedSlot::LeasedSlot(ResourceLedger& ledger, Adapter& adapter, Stream& stream,
                       SlotId id, std::optional<IdleSpillConfig> spill)
    : state_(std::make_shared<detail::LeaseState>(ledger, adapter, stream, id,
                                                  std::move(spill))) {
  if (state_->spill)
    worker_ = std::jthread(
        [state = state_](std::stop_token stop) { state->RunIdle(stop); });
}
LeasedSlot::~LeasedSlot() {
  StopIdleSpill();
}
void LeasedSlot::StopIdleSpill() {
  if (worker_.joinable()) {
    worker_.request_stop();
    worker_.join();
  }
}
SlotLease LeasedSlot::Acquire(std::optional<BorrowedLocation> live,
                              std::stop_token stop) {
  const auto start = detail::LeaseState::Clock::now();
  {
    std::unique_lock lock(state_->mutex);
    if (stop.stop_requested())
      throw std::runtime_error("slot acquisition cancelled");
    if (state_->leased)
      throw SlotBusy();
    if (live) {
      std::lock_guard source_lock(state_->source->mutex);
      if (!state_->live || live->slot != state_->source->id ||
          live->generation != state_->source->generation)
        throw std::logic_error("stale live frontier");
    }
    // Claim first so the worker cannot queue another piece. Drain the current
    // piece before making the slot executable or changing its stop token.
    state_->leased = true;
    if (!state_->changed.wait(lock, stop, [&] { return !state_->spilling; }) ||
        stop.stop_requested()) {
      state_->leased = false;
      state_->changed.notify_all();
      throw std::runtime_error("slot acquisition cancelled during idle spill");
    }
    state_->stop = stop;
  }
  try {
    if (!live) {
      // Finish cancellable admission, copies and reader waits BEFORE entering
      // the adapter's nonthrowing invalidation/release path. On refusal the
      // original generation and execution frontier remain reusable.
      state_->PrepareReassignment();
      state_->Invalidate();
      {
        std::lock_guard lock(state_->metrics_mutex);
        state_->metrics.residual_wait_ns +=
            detail::LeaseState::Nanoseconds(start);
        ++state_->metrics.reassignments;
      }
      if (!state_->slot) {
        state_->slot = state_->adapter.CreateSlot(*state_);
        state_->raw = state_->slot.get();
      }
    }
    return SlotLease(state_);
  } catch (...) {
    std::lock_guard lock(state_->mutex);
    state_->stop = {};
    state_->leased = false;
    state_->changed.notify_all();
    throw;
  }
}

bool LeasedSlot::PreservationComplete() const {
  std::lock_guard lock(state_->mutex);
  if (state_->leased)
    return false;
  for (const auto& entry : state_->rows)
    if (auto r = entry.rows.lock()) {
      std::lock_guard row_lock(r->mutex);
      if (!r->materialized && !r->retired)
        return false;
    }
  return true;
}
SpillMetrics LeasedSlot::Metrics() const {
  std::lock_guard lock(state_->metrics_mutex);
  return state_->metrics;
}
SlotLease AcquireSlot(std::span<LeasedSlot* const> slots,
                      std::stop_token stop) {
  for (auto* slot : slots)
    if (!slot)
      throw std::invalid_argument("null assignment slot");
  for (const bool complete : {true, false})
    for (auto* slot : slots) {
      if (stop.stop_requested())
        throw std::runtime_error("slot assignment cancelled");
      if (complete && !slot->PreservationComplete())
        continue;
      try {
        return slot->Acquire({}, stop);
      } catch (const SlotBusy&) {
        // Busy acquisition is harmless; cancellation must not try another slot.
        if (stop.stop_requested())
          throw;
      }
    }
  throw std::runtime_error("no idle execution slot");
}
}  // namespace gufo::cache
