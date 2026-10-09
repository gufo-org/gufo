#include "src/core/hip/committed_backing.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::hip {
namespace detail {
struct BackingState {
  cache::ResourceCharge metadata;
  std::vector<cache::ResourceCharge> blocks;
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
                           std::size_t index,
                           cache::ResourceReservation reservation)
    : state_(std::move(state)),
      index_(index),
      reservation_(std::move(reservation)) {}
BackingBlock& BackingBlock::operator=(BackingBlock&& other) noexcept {
  if (this != &other) {
    reservation_ = {};
    charge_ = {};
    state_ = std::move(other.state_);
    index_ = other.index_;
    reservation_ = std::move(other.reservation_);
    charge_ = std::move(other.charge_);
  }
  return *this;
}
std::span<std::byte> BackingBlock::Bytes() const {
  if (!state_)
    throw std::logic_error("empty backing block");
  return {static_cast<std::byte*>(state_->data) + index_ * state_->block_bytes,
          state_->block_bytes};
}
cache::ResourceAllocationInfo BackingBlock::Info() const {
  return reservation_ ? reservation_.Info() : charge_.Info();
}
void BackingBlock::Convert() {
  charge_ = reservation_.Convert();
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
    cache::ResourceCategory category) {
  // Validate even for an empty pool.
  if (category != cache::ResourceCategory::kBackingAssigned &&
      category != cache::ResourceCategory::kPrivateState &&
      category != cache::ResourceCategory::kPrivateTail)
    throw std::invalid_argument("invalid backing assignment category");
  for (std::size_t i = 0; i < state_->blocks.size(); ++i) {
    try {
      return BackingBlock(state_, i,
                          state_->blocks[i].ReserveBacking(category));
    } catch (const std::bad_alloc&) {
      // No payload allocation on exhaustion or an assignment race.
    }
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
