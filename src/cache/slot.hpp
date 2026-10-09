#ifndef GUFO_CACHE_SLOT_HPP_
#define GUFO_CACHE_SLOT_HPP_

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <thread>

#include "src/cache/adapter.hpp"
#include "src/cache/ledger.hpp"

namespace gufo::cache {
namespace detail {
struct SlotState;
struct RowState;
struct LeaseState;
}  // namespace detail
class BorrowedRows;
// Move-only source pin. A borrowed Location uses the leased slot; a committed
// Owner uses backing. Retain the pin through completion on every stream.
class RowPin {
public:
  RowPin() = default;
  ~RowPin();
  RowPin(RowPin&&) noexcept;
  RowPin& operator=(RowPin&&) noexcept;
  RowPin(const RowPin&) = delete;
  RowPin& operator=(const RowPin&) = delete;
  [[nodiscard]] std::optional<BorrowedLocation> Location() const {
    return location_;
  }
  [[nodiscard]] const std::shared_ptr<const void>& Owner() const {
    return owner_;
  }

private:
  friend class BorrowedRows;
  friend class Payload;
  friend class ChunkReference;
  [[nodiscard]] RowPin Clone() const;
  void Release() noexcept;
  std::shared_ptr<detail::RowState> state_;
  std::optional<BorrowedLocation> location_;
  ResourceCharge charge_;
  std::shared_ptr<const void> owner_;
};
// Stable indirection shared by a checkpoint payload and its guard. Preservation
// fills already reserved backing. It never changes checkpoint/chunk topology.
class BorrowedRows {
public:
  [[nodiscard]] RowPin Pin(std::stop_token = {}) const;
  [[nodiscard]] std::optional<BorrowedLocation> Location() const;
  [[nodiscard]] std::shared_ptr<const void> Owner() const;
  [[nodiscard]] bool IsValid() const;
  [[nodiscard]] std::size_t Bytes() const;
  [[nodiscard]] ResourceCategory Category() const;
  [[nodiscard]] PersistencePin PinPersistence() const;

private:
  friend class SlotLease;
  explicit BorrowedRows(std::shared_ptr<detail::RowState>);
  std::shared_ptr<detail::RowState> state_;
};
class SlotLease {
public:
  SlotLease() = default;
  ~SlotLease();
  SlotLease(SlotLease&&) noexcept;
  SlotLease& operator=(SlotLease&&) noexcept;
  SlotLease(const SlotLease&) = delete;
  SlotLease& operator=(const SlotLease&) = delete;
  [[nodiscard]] Slot& Execution() const;
  [[nodiscard]] BorrowedLocation Location() const;
  // Caller supplies initialized committed backing and an assigned reservation.
  // Admission/allocation happens before borrowing; no request-path allocation
  // of backing is performed here. Buffer and owner must describe the same
  // block.
  [[nodiscard]] std::shared_ptr<BorrowedRows> Borrow(
      ComponentId, Rows first, Rows end, ResourceReservation,
      std::shared_ptr<const void> backing_owner, std::span<std::byte> backing);
  void Reset();
  // Validate successful execution/restore before retaining the live frontier.
  // Commit releases this lease; no further execution is allowed through it.
  // Otherwise destruction invalidates (or destroys a busy slot, draining it).
  void Commit();

private:
  friend class LeasedSlot;
  explicit SlotLease(std::shared_ptr<detail::LeaseState>);
  void Release() noexcept;
  std::shared_ptr<detail::LeaseState> state_;
  bool committed_{false};
};
struct IdleSpillConfig {
  // At least one complete row must fit. Each piece leases a fresh stream;
  // null means the pool is busy. Callbacks run outside metadata locks and must
  // not acquire/stop this slot; metric queries are safe. Null defers idle
  // work; foreground waits/retries.
  // Throw for permanent failure (e.g. all pairs quarantined), never return
  // null forever. Captured resources outlive the slot and its leases.
  std::size_t piece_bytes;
  std::function<std::unique_ptr<Stream>()> acquire_stream;
  // Scheduler admission, checked before EVERY idle piece. False yields to
  // device model work. Already submitted pieces drain before yielding.
  std::function<bool()> device_idle;
};
struct SpillMetrics {
  std::size_t idle_bytes{}, foreground_bytes{};
  std::uint64_t idle_copy_ns{}, foreground_copy_ns{}, stream_wait_ns{};
  // Cold acquisition: active idle-piece drain plus required preservation and
  // reader waits. Includes stream acquisition; excludes slot recreation.
  std::uint64_t residual_wait_ns{};
  std::size_t reassignments{}, idle_failures{};
};
// One exclusive lease per model slot; distinct slots/leases may run
// concurrently. Adapter and preservation stream outlive this facade AND all
// outstanding leases. The stream is exclusive to this slot. External transfer
// completions and source pins must settle before Commit, Reset or lease
// destruction. A pin may safely outlive the lease on another thread: release
// waits for it outside metadata locks.
class LeasedSlot {
public:
  LeasedSlot(ResourceLedger&, Adapter&, Stream&, SlotId,
             std::optional<IdleSpillConfig> = {});
  ~LeasedSlot();
  LeasedSlot(const LeasedSlot&) = delete;
  LeasedSlot& operator=(const LeasedSlot&) = delete;
  // No live argument means cold reassignment. Exact live reuse checks
  // generation while acquiring exclusivity. Stale/busy/cancelled acquisition
  // throws.
  [[nodiscard]] SlotLease Acquire(std::optional<BorrowedLocation> live = {},
                                  std::stop_token = {});

  // Snapshot only: another acquirer may win before Acquire. Busy slots return
  // false; successful acquisitions still enforce exclusive generation checks.
  [[nodiscard]] bool PreservationComplete() const;
  [[nodiscard]] SpillMetrics Metrics() const;
  // Stop at a piece boundary, drain its completion and release source pins.
  // Retained partial ranges stay borrowed and can finish in the foreground.
  void StopIdleSpill();

private:
  std::shared_ptr<detail::LeaseState> state_;
  std::jthread worker_;
};
class SlotBusy final : public std::runtime_error {
public:
  SlotBusy() : std::runtime_error("slot busy") {}
};
// Cold assignment tries complete idle slots first, then remaining idle slots.
// Busy races are retried; cancellation/other failures propagate.
[[nodiscard]] SlotLease AcquireSlot(std::span<LeasedSlot* const>,
                                    std::stop_token = {});
}  // namespace gufo::cache
#endif
