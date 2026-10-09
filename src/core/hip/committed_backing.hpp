#ifndef GUFO_CORE_HIP_COMMITTED_BACKING_HPP_
#define GUFO_CORE_HIP_COMMITTED_BACKING_HPP_

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

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
  // Consume an unconverted assigned block as borrowed checkpoint backing.
  // Payload is a prefix of the block; accounting still charges full capacity.
  // The row handle retains the slab after this block/pool facade is destroyed.
  [[nodiscard]] std::shared_ptr<cache::BorrowedRows> Borrow(
      cache::SlotLease&, cache::ComponentId, cache::Rows first, cache::Rows end,
      std::size_t payload_bytes) &&;
  [[nodiscard]] BackingPin PinPersistence() const;

private:
  friend class CommittedBackingPool;
  BackingBlock(std::shared_ptr<detail::BackingState>, std::size_t,
               cache::ResourceReservation);
  std::shared_ptr<detail::BackingState> state_;
  std::size_t index_;
  cache::ResourceReservation reservation_;
  cache::ResourceCharge charge_;
};

struct BackingPoolConfig {
  // A share of D4's RAM budget, including metadata headroom. This pool does
  // not consume the rest of the ledger budget. Block sizes are page multiples.
  std::size_t ram_bytes;
  std::size_t block_bytes;
  std::size_t metadata_headroom_bytes;
};

class CommittedBackingPool {
public:
  // Initialization only: reserves metadata first, admits every block, then
  // allocates and touches all payload pages. No refill or resize API exists.
  CommittedBackingPool(cache::ResourceLedger&, BackingPoolConfig);
  // Thread safe. Races/bad_alloc try other blocks, then decline capture.
  [[nodiscard]] std::optional<BackingBlock> TryAcquire(cache::ResourceCategory);
  [[nodiscard]] std::size_t BlockCount() const noexcept;
  [[nodiscard]] std::size_t CommittedBytes() const noexcept;
  [[nodiscard]] std::size_t PageCommitAllocations() const noexcept;

private:
  std::shared_ptr<detail::BackingState> state_;
};

}  // namespace gufo::hip
#endif
