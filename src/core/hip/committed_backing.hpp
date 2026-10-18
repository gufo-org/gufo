#ifndef GUFO_CORE_HIP_COMMITTED_BACKING_HPP_
#define GUFO_CORE_HIP_COMMITTED_BACKING_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

#include "src/cache/checkpoint.hpp"
#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"

namespace gufo::hip {
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
  [[nodiscard]] bool CanFit(std::span<const cache::PayloadRequest>) const;
  // Starts committing the next on-demand segment on a background thread when
  // less than one segment of free backing remains, so a later TryAcquire
  // finds it ready instead of allocating on its own thread. Returns at once.
  // The count is the desired free-segment reserve, bounded by capacity.
  void GrowAhead(std::size_t segments = 1);
  void WaitForGrowth();
  // Commits every remaining on-demand segment on the calling thread, in the
  // same layout and admission order as growth, after any growth GrowAhead
  // started. False when a segment is declined or cannot be committed; the
  // segments committed so far stay. Once full, the pool never grows again.
  [[nodiscard]] bool CommitAll();
  // Committed blocks and bytes; both grow with on-demand segments.
  [[nodiscard]] std::size_t BlockCount() const noexcept;
  [[nodiscard]] std::size_t CommittedBytes() const noexcept;
  [[nodiscard]] std::size_t PageCommitAllocations() const noexcept;
  // Linux long-term pins prevent page migration in addition to GPU
  // registration. Zero when io_uring or the caller's memlock limit refuses
  // them; the existing registered-page path remains usable in that case.
  [[nodiscard]] std::size_t LongTermPinnedBytes() const noexcept;

private:
  std::shared_ptr<detail::BackingState> state_;
  std::mutex ahead_mutex_;
  // Joined before state_ is released; it keeps its own state reference.
  std::jthread ahead_;
};

}  // namespace gufo::hip
#endif
