#ifndef GUFO_CACHE_SLOT_HPP_
#define GUFO_CACHE_SLOT_HPP_

#include <memory>
#include <optional>
#include <span>
#include <stop_token>

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
// One exclusive lease per model slot; distinct slots/leases may run
// concurrently. Adapter and preservation stream outlive this facade AND all
// outstanding leases. The stream is exclusive to this slot. External transfer
// completions and source pins must settle before Commit, Reset or lease
// destruction. A pin may safely outlive the lease on another thread: release
// waits for it outside metadata locks.
class LeasedSlot {
public:
  LeasedSlot(ResourceLedger&, Adapter&, Stream&, SlotId);
  ~LeasedSlot();
  LeasedSlot(const LeasedSlot&) = delete;
  LeasedSlot& operator=(const LeasedSlot&) = delete;
  // No live argument means cold reassignment. Exact live reuse checks
  // generation while acquiring exclusivity. Stale/busy/cancelled acquisition
  // throws.
  [[nodiscard]] SlotLease Acquire(std::optional<BorrowedLocation> live = {},
                                  std::stop_token = {});

private:
  std::shared_ptr<detail::LeaseState> state_;
};
}  // namespace gufo::cache
#endif
