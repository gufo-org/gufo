#include "src/cache/slot.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace gufo::cache {
namespace detail {
struct SlotState {
  std::mutex mutex;
  SlotId id;
  // Zero denotes a destroyed source. Publish resets after preservation so
  // readers that see a new generation can also see completed row copies.
  std::atomic<std::uint64_t> generation{1};
};
struct RowState {
  ResourceCharge metadata;
  ResourceReservation reservation;
  ResourceCharge charge;
  std::shared_ptr<const void> owner;
  std::span<std::byte> buffer;
  std::span<const std::byte> source_bytes;
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
  bool closing{false};
  enum Completion : unsigned { kMaterialized = 1, kRetired = 2 };
  // Completion bits only accumulate. Publish after the copy and accounting
  // conversion so validity checks of completed rows need no reader mutex.
  std::atomic<unsigned> completion{0};
  bool Materialized() const {
    return completion.load(std::memory_order_relaxed) & kMaterialized;
  }
  bool Retired() const {
    return completion.load(std::memory_order_relaxed) & kRetired;
  }
  bool idle_copy_failed{false};
};
struct LeaseState final : MutationGuard {
  struct Entry {
    ResourceCharge metadata;
    std::weak_ptr<RowState> rows;
    // Immutable copies of the registration, so mutation guards reject
    // unrelated ranges without locking every weak handle per step.
    ComponentId component;
    Rows first{}, end{};
  };
  ResourceCharge metadata;
  ResourceLedger& ledger;
  Adapter& adapter;
  Stream& stream;
  std::shared_ptr<SlotState> source;
  std::mutex mutex;
  std::list<Entry> rows;
  // Foreground mutation can erase registrations. Reset this cursor whenever
  // execution releases the slot; the idle worker never uses it while leased.
  std::list<Entry>::iterator idle_next{rows.end()};
  // Conservative per-component upper bound of registered ends. Appends at
  // or beyond it skip the registry scan on every execution step.
  std::unordered_map<std::uint32_t, Rows> borrowed_end;
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
      if (!spill->piece_bytes || !spill->acquire_stream ||
          !spill->device_idle || !spill->reassignment_batch_size ||
          spill->reassignment_batch_size > 16)
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
    source->generation.store(0, std::memory_order_release);
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
      if (r->Materialized() || r->Retired())
        return true;
      first = r->first + r->copied;
      const auto count =
          spill ? spill->piece_bytes / r->row_bytes : r->end - first;
      end = first + std::min<Rows>(count, r->end - first);
      buffer = r->buffer.subspan(r->copied * r->row_bytes,
                                 (end - first) * r->row_bytes);
      std::lock_guard source_lock(source->mutex);
      if (source->generation.load(std::memory_order_relaxed) !=
          r->location.generation)
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
      r->completion.fetch_or(RowState::kMaterialized,
                             std::memory_order_release);
      r->idle_copy_failed = false;
      r->closing = false;
      r->changed.notify_all();
    }
    return true;
  }
  void Preserve(const std::shared_ptr<RowState>& r, bool release) {
    std::unique_lock lock(r->mutex);
    if (r->buffer.empty()) {
      // A discardable disk source promises no RAM preservation. Closing it
      // rejects future pieces; the one already reading must finish before the
      // adapter can overwrite/free its allocation.
      r->completion.fetch_or(RowState::kRetired, std::memory_order_release);
      r->closing = false;
      r->changed.notify_all();
    }
    if (!r->Materialized() && !r->Retired()) {
      r->closing = true;
      lock.unlock();
      try {
        while (true) {
          if (!release)
            CheckStop();
          (void)CopyPiece(r, false, release);
          lock.lock();
          if (r->Materialized())
            break;
          lock.unlock();
        }
      } catch (...) {
        if (!lock.owns_lock())
          lock.lock();
        r->closing = false;
        if (release)
          r->completion.fetch_or(RowState::kRetired, std::memory_order_release);
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
    // Another slot's prefill keeps the device busy for seconds. Polling every
    // millisecond wakes a core 1000 times a second: on Strix Halo, a core of
    // an otherwise sleeping CCD costs ~4 W of the package limit the GPU
    // shares, ~5% of a long prefill. Back off while deferred; the delay only
    // postpones optional idle work.
    constexpr auto kDeferMin = std::chrono::milliseconds(1);
    constexpr auto kDeferMax = std::chrono::milliseconds(32);
    auto defer = kDeferMin;
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
      for (auto& it = idle_next; it != rows.end();) {
        auto r = it->rows.lock();
        if (!r) {
          it = rows.erase(it);
          continue;
        }
        // Buffer geometry is immutable; completion only accumulates. Keep
        // the cursor on a partial range, but visit completed predecessors
        // once instead of locking and rescanning them for every idle piece.
        if (r->buffer.empty() ||
            r->completion.load(std::memory_order_acquire)) {
          ++it;
          continue;
        }
        candidate = std::move(r);
        break;
      }
      if (!candidate) {
        idle_pending = false;
        defer = kDeferMin;
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
        {
          std::lock_guard row_lock(candidate->mutex);
          candidate->idle_copy_failed = true;
          candidate->changed.notify_all();
        }
        std::lock_guard metrics_lock(metrics_mutex);
        ++metrics.idle_failures;
      }
      lock.lock();
      spilling = false;
      idle_failed = failed;
      changed.notify_all();
      if (admitted) {
        defer = kDeferMin;
      } else {
        changed.wait_for(lock, cancelled, defer, [] { return false; });
        defer = std::min(defer * 2, kDeferMax);
      }
    }
  }
  void Guard(ComponentId component, Rows first, Rows end, bool release) {
    if (end < first)
      throw std::invalid_argument("reversed mutation range");
    if (!release)
      CheckStop();
    if (const auto bound = borrowed_end.find(component.value);
        bound == borrowed_end.end() || bound->second <= first)
      return;
    // Iterate without allocating a request-time snapshot. Leased execution
    // serializes Borrow and mutation; readers never change this registry.
    auto it = rows.begin();
    while (it != rows.end()) {
      if (it->component != component || end <= it->first || it->end <= first) {
        ++it;
        continue;
      }
      auto r = it->rows.lock();
      if (!r) {
        it = rows.erase(it);
        continue;
      }
      ++it;
      Preserve(r, release);
    }
  }
  void PrepareReassignment() {
    CheckStop();
    if (spill && spill->reassignment_batch_size > 1) {
      PrepareBatchedReassignment();
      borrowed_end.clear();
      CheckStop();
      return;
    }
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
    // Every registered range is now preserved; later guards need no scan.
    borrowed_end.clear();
    CheckStop();
  }
  void PrepareBatchedReassignment() {
    struct Copy {
      std::shared_ptr<RowState> rows;
      std::unique_ptr<Stream> stream;
      std::optional<Completion> completion;
      Rows first, end;
      std::span<std::byte> buffer;
      Copy(std::shared_ptr<RowState> r, std::unique_ptr<Stream> s,
           SlotState& source)
          : rows(std::move(r)), stream(std::move(s)) {
        std::lock_guard lock(rows->mutex);
        first = rows->first + rows->copied;
        end = rows->end;
        buffer = rows->buffer.subspan(rows->copied * rows->row_bytes);
        std::lock_guard source_lock(source.mutex);
        if (source.generation.load(std::memory_order_relaxed) !=
            rows->location.generation)
          throw std::logic_error("spill source generation mismatch");
        ++rows->readers;
        rows->closing = true;
      }
      ~Copy() {
        // Also on cancellation/submission failure, drain before releasing the
        // source pin. An unpublished range can be copied again on retry.
        if (completion)
          (void)completion->Wait();
        std::lock_guard lock(rows->mutex);
        --rows->readers;
        rows->closing = false;
        rows->changed.notify_all();
      }
      void Finish() {
        if (completion->Wait() != TransferResult::kSucceeded)
          throw std::runtime_error("borrowed row preservation failed");
        std::lock_guard lock(rows->mutex);
        rows->copied += end - first;
        rows->charge = rows->reservation.Convert();
        rows->category = rows->charge.Info().category;
        rows->completion.fetch_or(RowState::kMaterialized,
                                  std::memory_order_release);
        rows->idle_copy_failed = false;
        rows->closing = false;
        rows->changed.notify_all();
      }
    };
    std::array<std::optional<Copy>, 16> copies;
    std::size_t size = 0;
    Clock::time_point started;
    const auto finish = [&] {
      if (!size)
        return;
      std::array<std::shared_ptr<RowState>, 16> completed;
      std::size_t bytes = 0;
      for (std::size_t i = 0; i < size; ++i) {
        copies[i]->Finish();
        bytes += copies[i]->buffer.size();
        completed[i] = copies[i]->rows;
      }
      // Publish all rows before waiting for external readers: a reader can
      // hold one source pin while waiting for another range in this batch.
      for (std::size_t i = 0; i < size; ++i)
        copies[i].reset();
      {
        std::lock_guard lock(metrics_mutex);
        metrics.foreground_bytes += bytes;
        metrics.foreground_copy_ns += Nanoseconds(started);
      }
      for (std::size_t i = 0; i < size; ++i)
        Preserve(completed[i], false);
      size = 0;
    };
    for (auto it = rows.begin(); it != rows.end();) {
      CheckStop();
      auto r = it->rows.lock();
      if (!r) {
        it = rows.erase(it);
        continue;
      }
      ++it;
      bool eligible;
      {
        std::lock_guard lock(r->mutex);
        eligible = !r->buffer.empty() && !r->Materialized() && !r->Retired() &&
                   (r->end - r->first - r->copied) * r->row_bytes <=
                       spill->piece_bytes;
      }
      if (!eligible) {
        finish();
        Preserve(r, false);
        continue;
      }
      if (size == spill->reassignment_batch_size)
        finish();
      const auto stream_start = Clock::now();
      auto transfer = spill->acquire_stream();
      if (!transfer && size) {
        finish();
        transfer = spill->acquire_stream();
      }
      while (!transfer) {
        CheckStop();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        transfer = spill->acquire_stream();
      }
      {
        std::lock_guard lock(metrics_mutex);
        metrics.stream_wait_ns += Nanoseconds(stream_start);
      }
      if (!size)
        started = Clock::now();
      auto& copy = copies[size++].emplace(r, std::move(transfer), *source);
      copy.completion.emplace(adapter.CopyRowsOut(
          *raw, r->component, copy.first, copy.end, copy.buffer, *copy.stream));
    }
    finish();
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
    const auto generation = source->generation.load(std::memory_order_relaxed);
    if (!generation || generation == std::numeric_limits<std::uint64_t>::max())
      std::terminate();
    source->generation.store(generation + 1, std::memory_order_release);
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
      owner_(std::move(other.owner_)),
      bytes_(std::exchange(other.bytes_, {})) {}
RowPin& RowPin::operator=(RowPin&& other) noexcept {
  if (this != &other) {
    Release();
    state_ = std::move(other.state_);
    location_ = std::exchange(other.location_, {});
    charge_ = std::move(other.charge_);
    owner_ = std::move(other.owner_);
    bytes_ = std::exchange(other.bytes_, {});
  }
  return *this;
}
RowPin RowPin::Clone() const {
  RowPin result;
  result.state_ = state_;
  result.location_ = location_;
  result.charge_ = charge_;
  result.owner_ = owner_;
  result.bytes_ = bytes_;
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
  bytes_ = {};
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
  if (r.Retired())
    throw std::runtime_error("borrowed rows retired");
  RowPin pin;
  pin.state_ = state_;
  if (r.Materialized()) {
    pin.owner_ = r.owner;
    pin.bytes_ = r.buffer;
  } else {
    std::lock_guard source_lock(r.slot->mutex);
    const bool valid = r.slot->generation.load(std::memory_order_relaxed) ==
                       r.location.generation;
    assert(valid && "adapter overwrote borrowed rows without a guard");
    if (!valid)
      throw std::logic_error("borrowed slot generation mismatch");
    ++r.readers;
    pin.location_ = r.location;
    pin.bytes_ = r.source_bytes;
  }
  return pin;
}
std::optional<BorrowedLocation> BorrowedRows::Location() const {
  std::lock_guard lock(state_->mutex);
  return state_->Materialized() || state_->Retired()
             ? std::nullopt
             : std::optional(state_->location);
}
std::shared_ptr<const void> BorrowedRows::Owner() const {
  std::lock_guard lock(state_->mutex);
  if (state_->Retired())
    return {};
  if (!state_->Materialized())
    throw std::logic_error("borrowed rows require a source pin");
  return state_->owner;
}
bool BorrowedRows::IsValid() const {
  // This observes validity; Pin still serializes reads with preservation.
  const auto completion = state_->completion.load(std::memory_order_acquire);
  if (completion)
    return !(completion & detail::RowState::kRetired);
  if (state_->slot->generation.load(std::memory_order_acquire) ==
      state_->location.generation)
    return true;
  // Preservation may have finished between the two reads. A reset/death
  // publishes after completion, so recheck before rejecting the old source.
  const auto completed = state_->completion.load(std::memory_order_acquire);
  return (completed & detail::RowState::kMaterialized) &&
         !(completed & detail::RowState::kRetired);
}
std::size_t BorrowedRows::Bytes() const {
  return state_->bytes;
}
bool BorrowedRows::PreservationFailed() const {
  std::lock_guard lock(state_->mutex);
  return !state_->Materialized() && state_->idle_copy_failed;
}
ResourceCategory BorrowedRows::Category() const {
  std::lock_guard lock(state_->mutex);
  return state_->category;
}
PersistencePin BorrowedRows::PinPersistence() const {
  std::lock_guard lock(state_->mutex);
  if (state_->buffer.empty())
    return {};
  if (state_->Retired())
    throw std::runtime_error("borrowed rows retired");
  return state_->Materialized() ? state_->charge.PinPersistence()
                                : state_->reservation.PinPersistence();
}
bool BorrowedRows::PersistencePinned() const {
  std::lock_guard lock(state_->mutex);
  if (state_->buffer.empty() || state_->Retired())
    return false;
  return state_->Materialized() ? state_->charge.Info().persistence_pinned
                                : state_->reservation.Info().persistence_pinned;
}
bool BorrowedRows::HasBacking() const {
  return !state_->buffer.empty();
}
std::size_t BorrowedRows::SourceViewBytes() const {
  std::lock_guard lock(state_->mutex);
  return state_->source_bytes.size();
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
    state_->idle_next = state_->rows.begin();
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
  return {state_->source->id,
          state_->source->generation.load(std::memory_order_relaxed)};
}
std::shared_ptr<BorrowedRows> SlotLease::Borrow(
    ComponentId c, Rows first, Rows end, ResourceReservation reservation,
    std::shared_ptr<const void> owner, std::span<std::byte> backing,
    std::span<const std::byte> source) {
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
      backing.size() > info.bytes ||
      (!source.empty() && source.size() != backing.size()))
    throw std::invalid_argument("invalid borrowed row description");
  auto metadata = state_->ledger.Reserve(
      ResourceCategory::kMetadata,
      sizeof(detail::RowState) + sizeof(BorrowedRows) +
          sizeof(detail::LeaseState::Entry) + 2 * sizeof(void*));
  auto rows = std::make_shared<detail::RowState>();
  rows->reservation = std::move(held);
  rows->owner = std::move(retained_owner);
  rows->buffer = backing;
  rows->source_bytes = source;
  rows->slot = state_->source;
  rows->location = Location();
  rows->component = c;
  rows->row_bytes = d->row_bytes;
  rows->first = first;
  rows->end = end;
  rows->bytes = info.bytes;
  rows->category = info.category;
  rows->metadata = metadata.Convert();
  // Raise the guard bound before registration: a failure afterwards may only
  // leave it conservatively high, never hide a registered range.
  auto& bound = state_->borrowed_end[c.value];
  bound = std::max(bound, end);
  auto result = std::shared_ptr<BorrowedRows>(new BorrowedRows(rows));
  state_->rows.push_back({rows->metadata, rows, c, first, end});
  return result;
}
std::shared_ptr<BorrowedRows> SlotLease::Observe(
    ComponentId c, Rows first, Rows end, std::span<const std::byte> source) {
  auto& slot = Execution();
  const auto descriptors = state_->adapter.Components();
  const auto d = std::ranges::find(descriptors, c, &ComponentDescriptor::id);
  const auto positions = state_->adapter.Positions(slot);
  const auto p = std::ranges::find(positions, c, &ComponentPosition::id);
  if (!slot.IsValid() || d == descriptors.end() ||
      d->kind != ComponentKind::kAppendRows || p == positions.end() ||
      first >= end || end > p->valid_rows || !d->row_bytes ||
      end - first > std::numeric_limits<std::size_t>::max() / d->row_bytes ||
      source.size() != (end - first) * d->row_bytes)
    throw std::invalid_argument("invalid observed row description");
  auto metadata = state_->ledger.Reserve(
      ResourceCategory::kMetadata,
      sizeof(detail::RowState) + sizeof(BorrowedRows) +
          sizeof(detail::LeaseState::Entry) + 2 * sizeof(void*));
  auto rows = std::make_shared<detail::RowState>();
  rows->source_bytes = source;
  rows->slot = state_->source;
  rows->location = Location();
  rows->component = c;
  rows->row_bytes = d->row_bytes;
  rows->first = first;
  rows->end = end;
  rows->bytes = source.size();
  rows->category = ResourceCategory::kBackingAssigned;
  rows->metadata = metadata.Convert();
  auto& bound = state_->borrowed_end[c.value];
  bound = std::max(bound, end);
  auto result = std::shared_ptr<BorrowedRows>(new BorrowedRows(rows));
  state_->rows.push_back({rows->metadata, rows, c, first, end});
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
          live->generation !=
              state_->source->generation.load(std::memory_order_relaxed))
        throw std::logic_error("stale live frontier");
    }
    // Claim first so the worker cannot queue another piece. Drain the current
    // piece before making the slot executable or changing its stop token.
    state_->leased = true;
    if (!state_->changed.wait(lock, stop, [&] { return !state_->spilling; }) ||
        stop.stop_requested()) {
      state_->leased = false;
      state_->idle_next = state_->rows.begin();
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
    state_->idle_next = state_->rows.begin();
    state_->changed.notify_all();
    throw;
  }
}

bool LeasedSlot::PreservationComplete() const {
  std::lock_guard lock(state_->mutex);
  if (state_->leased)
    return false;
  for (const auto& entry : state_->rows)
    if (auto r = entry.rows.lock())
      if (!r->completion.load(std::memory_order_acquire))
        return false;
  return true;
}
bool LeasedSlot::ExecutionComplete() const {
  const std::lock_guard lock(state_->mutex);
  return !state_->leased &&
         (!state_->slot || state_->slot->ExecutionComplete());
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
