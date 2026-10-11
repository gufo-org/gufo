#include "src/core/hip/committed_backing.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::hip {
namespace detail {
struct BackingState {
  cache::ResourceCharge metadata;
  std::vector<cache::ResourceCharge> blocks;
  std::atomic<std::size_t> next{};
  std::size_t block_bytes{};
  void* data{};
  ~BackingState() {
    // Physical storage is freed before its original committed charges.
    if (data)
      (void)hipHostFree(data);
  }
};
}  // namespace detail

BackingPin::BackingPin(std::shared_ptr<detail::BackingState> state,
                       cache::PersistencePin pin)
    : state_(std::move(state)), pin_(std::move(pin)) {}
BackingPin& BackingPin::operator=(const BackingPin& other) {
  if (this != &other) {
    pin_ = {};
    state_ = other.state_;
    pin_ = other.pin_;
  }
  return *this;
}
BackingPin& BackingPin::operator=(BackingPin&& other) noexcept {
  if (this != &other) {
    pin_ = {};
    state_ = std::move(other.state_);
    pin_ = std::move(other.pin_);
  }
  return *this;
}
BackingBlock::BackingBlock(std::shared_ptr<detail::BackingState> state,
                           std::size_t index, std::size_t blocks,
                           cache::ResourceReservation reservation)
    : state_(std::move(state)),
      index_(index),
      blocks_(blocks),
      reservation_(std::move(reservation)) {}
BackingBlock& BackingBlock::operator=(BackingBlock&& other) noexcept {
  if (this != &other) {
    reservation_ = {};
    charge_ = {};
    state_ = std::move(other.state_);
    index_ = other.index_;
    blocks_ = other.blocks_;
    reservation_ = std::move(other.reservation_);
    charge_ = std::move(other.charge_);
  }
  return *this;
}
std::span<std::byte> BackingBlock::Bytes() const {
  if (!state_)
    throw std::logic_error("empty backing block");
  return {static_cast<std::byte*>(state_->data) + index_ * state_->block_bytes,
          blocks_ * state_->block_bytes};
}
cache::ResourceAllocationInfo BackingBlock::Info() const {
  return reservation_ ? reservation_.Info() : charge_.Info();
}
void BackingBlock::Convert() {
  charge_ = reservation_.Convert();
}
cache::Payload BackingBlock::TakePayload() && {
  if (!charge_ || reservation_)
    throw std::logic_error("backing capture has not completed");
  const auto bytes = Bytes();
  auto owner = std::shared_ptr<const void>(std::move(state_), bytes.data());
  return cache::Payload::Committed(std::move(charge_), std::move(owner));
}
std::shared_ptr<cache::BorrowedRows> BackingBlock::Borrow(
    cache::SlotLease& lease, cache::ComponentId component, cache::Rows first,
    cache::Rows end, std::size_t payload_bytes,
    std::span<const std::byte> source) && {
  auto buffer = Bytes();
  if (!reservation_ || payload_bytes > buffer.size())
    throw std::invalid_argument("invalid borrowed backing block");
  auto owner = std::shared_ptr<const void>(std::move(state_), buffer.data());
  return lease.Borrow(component, first, end, std::move(reservation_),
                      std::move(owner), buffer.first(payload_bytes), source);
}
BackingPin BackingBlock::PinPersistence() const {
  return BackingPin(state_, reservation_ ? reservation_.PinPersistence()
                                         : charge_.PinPersistence());
}

CommittedBackingPool::CommittedBackingPool(cache::ResourceLedger& ledger,
                                           BackingPoolConfig config) {
  constexpr std::size_t page = 4096;
  if (!config.block_bytes || config.block_bytes % page ||
      config.metadata_headroom_bytes > config.ram_bytes)
    throw std::invalid_argument("invalid committed backing budget/block size");
  const auto count =
      (config.ram_bytes - config.metadata_headroom_bytes) / config.block_bytes;
  // Headroom also covers the ledger's per-block allocations, borrower tokens,
  // pins and allocator overhead. Leave ample space rather than count payload
  // bytes twice. Future store/index metadata needs its own budget share.
  constexpr std::size_t metadata_per_block = 1024;
  if (config.metadata_headroom_bytes < sizeof(detail::BackingState) ||
      count > (config.metadata_headroom_bytes - sizeof(detail::BackingState)) /
                  metadata_per_block)
    throw std::invalid_argument("insufficient backing metadata headroom");
  auto metadata = ledger.Reserve(cache::ResourceCategory::kMetadata,
                                 config.metadata_headroom_bytes);
  auto state = std::make_shared<detail::BackingState>();
  state->metadata = metadata.Convert();
  state->block_bytes = config.block_bytes;
  state->blocks.reserve(count);
  std::vector<cache::ResourceReservation> reservations;
  reservations.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    reservations.push_back(ledger.Reserve(cache::ResourceCategory::kBackingFree,
                                          config.block_bytes));
  try {
    if (count) {
      // Explicit coherence avoids HIP_HOST_COHERENT changing this contract.
      const auto status = hipHostMalloc(
          &state->data, count * config.block_bytes, hipHostMallocCoherent);
      if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
      std::memset(state->data, 0, count * config.block_bytes);
    }
    for (auto& reservation : reservations)
      state->blocks.push_back(reservation.Convert());
  } catch (...) {
    // Convert may fail after pages commit. Free before pending reservations
    // or converted charges unwind, preserving physical-byte accounting.
    if (state->data) {
      (void)hipHostFree(state->data);
      state->data = nullptr;
    }
    throw;
  }
  state_ = std::move(state);
}
std::optional<BackingBlock> CommittedBackingPool::TryAcquire(
    cache::ResourceCategory category, std::size_t payload_bytes) {
  if (category != cache::ResourceCategory::kBackingAssigned &&
      category != cache::ResourceCategory::kPrivateState &&
      category != cache::ResourceCategory::kPrivateTail)
    throw std::invalid_argument("invalid backing assignment category");
  const auto count = state_->blocks.size();
  const auto blocks =
      payload_bytes ? 1 + (payload_bytes - 1) / state_->block_bytes : 1;
  if (blocks > count)
    return std::nullopt;
  auto i = state_->next.load(std::memory_order_relaxed);
  for (std::size_t tried = 0; tried < count;) {
    // An occupied page rules out every candidate run that crosses it. Skip
    // without allocating assignment bookkeeping, including full exhaustion.
    std::size_t free = 0;
    while (free < blocks && i + free < count &&
           state_->blocks[i + free].Info().category ==
               cache::ResourceCategory::kBackingFree)
      ++free;
    if (free == blocks) {
      try {
        auto reservation = cache::ResourceCharge::ReserveBacking(
            std::span(state_->blocks).subspan(i, blocks), category);
        state_->next.store(i + blocks == count ? 0 : i + blocks,
                           std::memory_order_relaxed);
        return BackingBlock(state_, i, blocks, std::move(reservation));
      } catch (const std::bad_alloc&) {
        // A racing assignment or injected bookkeeping failure is atomic.
      }
    }
    const auto step = free < blocks ? std::min(free + 1, count - i) : 1;
    tried += step;
    i = i + step >= count ? 0 : i + step;
  }
  return std::nullopt;
}
std::size_t CommittedBackingPool::BlockCount() const noexcept {
  return state_->blocks.size();
}
std::size_t CommittedBackingPool::CommittedBytes() const noexcept {
  return BlockCount() * state_->block_bytes;
}
std::size_t CommittedBackingPool::PageCommitAllocations() const noexcept {
  return BlockCount() ? 1 : 0;
}

}  // namespace gufo::hip
