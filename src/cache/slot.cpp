#include "src/cache/slot.hpp"

#include <algorithm>
#include <cassert>
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
  Rows first{}, end{};
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
  bool leased{false}, live{false};
  std::stop_token stop;
  Slot* raw{};
  std::unique_ptr<Slot> slot;
  LeaseState(ResourceLedger& l, Adapter& a, Stream& s, SlotId id)
      : ledger(l),
        adapter(a),
        stream(s),
        source(std::make_shared<SlotState>()) {
    if (!id.value)
      throw std::invalid_argument("zero slot identifier");
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
  void Preserve(const std::shared_ptr<RowState>& r, bool release) {
    std::unique_lock lock(r->mutex);
    if (r->materialized || r->retired) {
      if (release)
        r->changed.wait(lock, [&] { return r->readers == 0; });
      else if (!r->changed.wait(lock, stop, [&] { return r->readers == 0; }))
        throw std::runtime_error(
            "slot lease cancelled while waiting for readers");
      return;
    }
    r->closing = true;
    lock.unlock();
    try {
      if (!release)
        CheckStop();
      auto copy = adapter.CopyRowsOut(*raw, r->component, r->first, r->end,
                                      r->buffer, stream);
      if (copy.Wait() != TransferResult::kSucceeded)
        throw std::runtime_error("borrowed row preservation failed");
      if (!release)
        CheckStop();
      // Convert may fail; until it succeeds readers still see unchanged rows.
      lock.lock();
      auto charge = r->reservation.Convert();
      r->charge = std::move(charge);
      r->category = r->charge.Info().category;
      r->materialized = true;
      r->closing = false;
      r->changed.notify_all();
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
    // Existing pins still address the source. New pins address backing or are
    // refused after retirement. No slot/ledger lock is held while waiting.
    if (release) {
      r->changed.wait(lock, [&] { return r->readers == 0; });
    } else if (!r->changed.wait(lock, stop, [&] { return r->readers == 0; })) {
      throw std::runtime_error(
          "slot lease cancelled while waiting for readers");
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
                       SlotId id)
    : state_(
          std::make_shared<detail::LeaseState>(ledger, adapter, stream, id)) {}
LeasedSlot::~LeasedSlot() = default;
SlotLease LeasedSlot::Acquire(std::optional<BorrowedLocation> live,
                              std::stop_token stop) {
  {
    std::lock_guard lock(state_->mutex);
    if (state_->leased || stop.stop_requested())
      throw std::runtime_error("slot busy or acquisition cancelled");
    if (live) {
      std::lock_guard source_lock(state_->source->mutex);
      if (!state_->live || live->slot != state_->source->id ||
          live->generation != state_->source->generation)
        throw std::logic_error("stale live frontier");
    }
    state_->leased = true;
    state_->stop = stop;
  }
  SlotLease lease(state_);
  if (!live) {
    state_->Invalidate();
    if (!state_->slot) {
      state_->slot = state_->adapter.CreateSlot(*state_);
      state_->raw = state_->slot.get();
    }
  }
  return lease;
}
}  // namespace gufo::cache
