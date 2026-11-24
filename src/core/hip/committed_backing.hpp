#ifndef GUFO_CORE_HIP_COMMITTED_BACKING_HPP_
#define GUFO_CORE_HIP_COMMITTED_BACKING_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

#include "src/cache/checkpoint.hpp"
#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"
#include "src/core/hip/heap_placement.hpp"
#include "src/mem/lendable.hpp"

namespace gufo::hip {

/// True when a host-page registration that moved device heap free memory
/// from `free_before` to `free_after` still leaves kDispatchReserve, or took
/// nothing from the heap. Whether registered pages are charged to the heap
/// differs between hosts, so the caller measures it rather than assuming.
[[nodiscard]] inline bool RegistrationKeepsReserve(std::uint64_t free_before,
                                                   std::uint64_t free_after) {
  return free_after >= std::min(free_before, kDispatchReserve);
}

/// True when registering `bytes` took at least half of them from the device
/// heap, so this host charges registered pages. Smaller drops are taken as
/// concurrent allocations rather than the registration itself.
[[nodiscard]] inline bool RegistrationCharged(std::uint64_t bytes,
                                              std::uint64_t free_before,
                                              std::uint64_t free_after) {
  return free_before > free_after && free_before - free_after >= bytes / 2;
}

namespace detail {
struct BackingState;
}

// Retains physical storage as well as accounting while a job holds its pin.
class BackingPin {
public:
  BackingPin(const BackingPin&) = default;
  BackingPin(BackingPin&&) noexcept = default;
  BackingPin& operator=(const BackingPin&);
  BackingPin& operator=(BackingPin&&) noexcept;

private:
  friend class BackingBlock;
  BackingPin(std::shared_ptr<detail::BackingState>, cache::PersistencePin);
  std::shared_ptr<detail::BackingState> state_;
  cache::PersistencePin pin_;
};

// A whole block is charged, including any unused payload capacity. Destroying
// an unconverted block cancels capture. Convert only after the copy succeeds.
// Keep this handle (or a pin) alive until all users of Bytes have drained.
class BackingBlock {
public:
  BackingBlock(BackingBlock&&) noexcept = default;
  BackingBlock& operator=(BackingBlock&&) noexcept;
  BackingBlock(const BackingBlock&) = delete;
  BackingBlock& operator=(const BackingBlock&) = delete;
  [[nodiscard]] std::span<std::byte> Bytes() const;
  [[nodiscard]] cache::ResourceAllocationInfo Info() const;
  void Convert();
  [[nodiscard]] cache::Payload TakePayload() &&;
  // Consume an unconverted assigned block as borrowed checkpoint backing.
  // Payload is a prefix of the block; accounting still charges full capacity.
  // The row handle retains the slab after this block/pool facade is destroyed.
  [[nodiscard]] std::shared_ptr<cache::BorrowedRows> Borrow(
      cache::SlotLease&, cache::ComponentId, cache::Rows first, cache::Rows end,
      std::size_t payload_bytes, std::span<const std::byte> source = {}) &&;
  [[nodiscard]] BackingPin PinPersistence() const;

private:
  friend class CommittedBackingPool;
  BackingBlock(std::shared_ptr<detail::BackingState>, std::size_t, std::size_t,
               cache::ResourceReservation);
  std::shared_ptr<detail::BackingState> state_;
  std::size_t index_;
  std::size_t blocks_;
  cache::ResourceReservation reservation_;
  cache::ResourceCharge charge_;
};

struct BackingPoolConfig {
  // A share of D4's RAM budget, including metadata headroom. This pool does
  // not consume the rest of the ledger budget. Block sizes are page multiples.
  std::size_t ram_bytes;
  std::size_t block_bytes;
  std::size_t metadata_headroom_bytes;
  // Zero commits every block at construction. Otherwise TryAcquire commits
  // blocks on demand, one segment of this many bytes (a block multiple) at a
  // time, until ram_bytes is reached. A payload never spans two segments.
  std::size_t segment_bytes{};
  // Asked before each on-demand segment with its bytes; false keeps the pool
  // at its current size, for example to leave host memory to the OS.
  std::function<bool(std::size_t)> admit_segment;
  // Asked after each on-demand segment is registered, with its bytes and the
  // device heap's free bytes before and after; false releases the segment
  // and keeps the pool at its current size. A failed heap query refuses it.
  std::function<bool(std::size_t, std::size_t, std::size_t)> keep_segment;
  // State pages to commit segments from (src/mem/lendable.hpp), so sessions
  // can borrow whole free pages back. Needs on-demand segments and blocks
  // that divide a page; otherwise ignored. A segment takes state pages when
  // `admit_state_pages` allows them, else pinned host pages under
  // admit_segment, as without state pages; only state pages are lent.
  // `shed` evicts one retained checkpoint when too few pages are free,
  // false when none can go; it runs on the state pool's thread without any
  // lock of this pool.
  std::shared_ptr<mem::LendableRange> lendable;
  std::function<bool(std::size_t)> admit_state_pages;
  std::function<bool()> shed;
  // Asked before lending any page; false lends none. Null always lends. A
  // backing that only stages captures on their way to disk has nothing to
  // shed, and every page lent away refuses a capture.
  std::function<bool()> may_lend;
  // Optional: free pages in the state pool now. A checkpoint that shares
  // session pages returns them straight to that pool when it is evicted, so
  // a shed can satisfy a borrower without freeing a page of this pool;
  // lending stops evicting once the state pool gained what was asked.
  std::function<std::size_t()> state_pool_free_pages;
};

class CommittedBackingPool {
public:
  // Reserves metadata for every block first. Without segment_bytes it admits,
  // allocates and touches every payload page here; otherwise none until a
  // TryAcquire finds no free run. Committed pages are never released early.
  // The ledger must outlive this facade.
  CommittedBackingPool(cache::ResourceLedger&, BackingPoolConfig);
  // Thread safe. Races/bad_alloc try other blocks, then commit another
  // segment if the pool may grow, then decline capture. Growth allocates
  // pinned host memory on the calling thread.
  [[nodiscard]] std::optional<BackingBlock> TryAcquire(
      cache::ResourceCategory, std::size_t payload_bytes = 0);
  // Reject a whole capture only when rounded payloads exceed free plus
  // uncommitted capacity. Does not allocate or change the search cursor.
  // Fragmentation, growth limits and races still need TryAcquire's checks.
  // `held_bytes` is budget that payloads outside the pool hold (rows that
  // share session pages): the pool cannot grow into it.
  [[nodiscard]] bool CanFit(std::span<const cache::PayloadRequest>,
                            std::size_t held_bytes = 0) const;
  // Prepares free backing on a background thread so a later TryAcquire finds
  // it ready. Reserve bytes are independent of the physical segment size;
  // zero does nothing and oversized reserves are bounded by capacity.
  void GrowAhead(std::size_t reserve_bytes);
  void WaitForGrowth();
  // Returns once no thread grows ahead: the reserve asked for is free, or
  // growth stopped short of it. Unlike WaitForGrowth it takes no lock, so
  // other threads keep calling GrowAhead and TryAcquire meanwhile. A reserve
  // raised while it waits may hand growth to a new thread, which it does not
  // wait for.
  void WaitWhileGrowingAhead() const;
  // Committed blocks and bytes; both grow with on-demand segments.
  [[nodiscard]] std::size_t BlockCount() const noexcept;
  [[nodiscard]] std::size_t CommittedBytes() const noexcept;
  [[nodiscard]] std::size_t PageCommitAllocations() const noexcept;
  // Linux long-term pins prevent page migration in addition to GPU
  // registration. Zero when io_uring or the caller's memlock limit refuses
  // them; the existing registered-page path remains usable in that case.
  [[nodiscard]] std::size_t LongTermPinnedBytes() const noexcept;
  // Ends loading: with state pages, later segments take free host pages
  // only, since a page made by reclaim may compact beside running work.
  void StartServing() noexcept;
  // True when segments are state pages that sessions can borrow.
  [[nodiscard]] bool Lends() const noexcept;
  // Pages lent to sessions now, and in all.
  [[nodiscard]] std::size_t LentPages() const noexcept;
  [[nodiscard]] std::size_t LentPagesTotal() const noexcept;
  // Gives up to about `bytes` of committed backing whose blocks are all free
  // back to the host (whole state pages, then whole pinned segments) and
  // returns the bytes released. Their blocks stay reserved until a later
  // growth commits them again, before any new segment and under the same
  // admission rules. Nothing is evicted here. Call only while no GPU work is
  // in flight: unmapping a state page waits for the device.
  std::size_t Surrender(std::size_t bytes);
  // Bytes given back and not committed again yet.
  [[nodiscard]] std::size_t SurrenderedBytes() const noexcept;

private:
  std::shared_ptr<detail::BackingState> state_;
  std::mutex ahead_mutex_;
  // Joined before state_ is released; it keeps its own state reference.
  std::jthread ahead_;
};

}  // namespace gufo::hip
#endif
