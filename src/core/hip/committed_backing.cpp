#include "src/core/hip/committed_backing.hpp"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::hip {
namespace detail {
namespace {
// Ordinary anonymous pages, then pinned. hipHostMalloc asks the kernel for
// high-order pages: when most host RAM belongs to the GPU, that compacts
// memory and stalls running kernels for up to seconds per GiB. Zero-filled.
void* CommitPages(std::size_t bytes) {
  void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (data == MAP_FAILED)
    return nullptr;
  if (hipHostRegister(data, bytes, hipHostRegisterDefault) != hipSuccess) {
    munmap(data, bytes);
    return nullptr;
  }
  return data;
}
void ReleasePages(void* data, std::size_t bytes) {
  (void)hipHostUnregister(data);
  munmap(data, bytes);
}
}  // namespace
struct BackingState {
  cache::ResourceCharge metadata;
  // Fixed capacity; entries below `committed` are published and never move.
  std::unique_ptr<cache::ResourceCharge[]> blocks;
  std::unique_ptr<void*[]> segments;
  std::size_t capacity{}, segment_blocks{}, block_bytes{};
  std::atomic<std::size_t> committed{}, next{};
  cache::ResourceLedger* ledger{};
  std::function<bool(std::size_t)> admit_segment;
  std::mutex growth;
  std::atomic<bool> growing_ahead{};
  std::byte* Address(std::size_t index) const {
    return static_cast<std::byte*>(segments[index / segment_blocks]) +
           index % segment_blocks * block_bytes;
  }
  std::size_t SegmentBytes(std::size_t segment) const {
    return std::min(segment_blocks, capacity - segment * segment_blocks) *
           block_bytes;
  }
  std::size_t SegmentCount() const {
    const auto blocks = committed.load(std::memory_order_acquire);
    return (blocks + segment_blocks - 1) / segment_blocks;
  }
  // Commits the next segment unless another thread already grew past `seen`.
  // True means the caller should search again.
  bool Grow(std::size_t seen) {
    const std::lock_guard lock(growth);
    const auto first = committed.load(std::memory_order_acquire);
    if (first != seen)
      return true;
    if (first == capacity)
      return false;
    const auto count = std::min(segment_blocks, capacity - first);
    if (admit_segment && !admit_segment(count * block_bytes))
      return false;
    std::vector<cache::ResourceReservation> reservations;
    try {
      reservations.reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        reservations.push_back(ledger->Reserve(
            cache::ResourceCategory::kBackingFree, block_bytes));
    } catch (const std::bad_alloc&) {
      return false;
    }
    auto& data = segments[first / segment_blocks];
    data = CommitPages(count * block_bytes);
    if (!data)
      return false;
    try {
      for (std::size_t i = 0; i < count; ++i)
        blocks[first + i] = reservations[i].Convert();
    } catch (const std::bad_alloc&) {
      // Free before the converted charges unwind, as at initialization.
      ReleasePages(data, count * block_bytes);
      data = nullptr;
      for (std::size_t i = 0; i < count; ++i)
        blocks[first + i] = {};
      return false;
    }
    committed.store(first + count, std::memory_order_release);
    return true;
  }
  ~BackingState() {
    // Physical storage is freed before its original committed charges.
    if (segments)
      for (std::size_t i = 0; i < SegmentCount(); ++i)
        ReleasePages(segments[i], SegmentBytes(i));
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
  return {state_->Address(index_), blocks_ * state_->block_bytes};
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
      config.metadata_headroom_bytes > config.ram_bytes ||
      config.segment_bytes % config.block_bytes ||
      (config.segment_bytes && config.segment_bytes < config.block_bytes))
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
  state->capacity = count;
  state->segment_blocks = config.segment_bytes
                              ? config.segment_bytes / config.block_bytes
                              : std::max<std::size_t>(count, 1);
  state->blocks = std::make_unique<cache::ResourceCharge[]>(count);
  state->segments = std::make_unique<void*[]>(
      (count + state->segment_blocks - 1) / state->segment_blocks + 1);
  state->ledger = &ledger;
  state->admit_segment = std::move(config.admit_segment);
  if (!config.segment_bytes && count) {
    std::vector<cache::ResourceReservation> reservations;
    reservations.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
      reservations.push_back(ledger.Reserve(
          cache::ResourceCategory::kBackingFree, config.block_bytes));
    auto& data = state->segments[0];
    try {
      data = detail::CommitPages(count * config.block_bytes);
      if (!data)
        throw std::runtime_error("cannot commit pinned cache backing");
      for (std::size_t i = 0; i < count; ++i)
        state->blocks[i] = reservations[i].Convert();
    } catch (...) {
      // Convert may fail after pages commit. Free before pending reservations
      // or converted charges unwind, preserving physical-byte accounting.
      if (data) {
        detail::ReleasePages(data, count * config.block_bytes);
        data = nullptr;
      }
      for (std::size_t i = 0; i < count; ++i)
        state->blocks[i] = {};
      throw;
    }
    state->committed.store(count, std::memory_order_release);
  }
  state_ = std::move(state);
}
std::optional<BackingBlock> CommittedBackingPool::TryAcquire(
    cache::ResourceCategory category, std::size_t payload_bytes) {
  if (category != cache::ResourceCategory::kBackingAssigned &&
      category != cache::ResourceCategory::kPrivateState &&
      category != cache::ResourceCategory::kPrivateTail)
    throw std::invalid_argument("invalid backing assignment category");
  const auto blocks =
      payload_bytes ? 1 + (payload_bytes - 1) / state_->block_bytes : 1;
  const auto segment = state_->segment_blocks;
  if (blocks > std::min(segment, state_->capacity))
    return std::nullopt;
  while (true) {
    const auto count = state_->committed.load(std::memory_order_acquire);
    auto i = state_->next.load(std::memory_order_relaxed);
    if (i >= count)
      i = 0;
    // A growing pool is usually full when a capture misses. The ledger's free
    // backing bounds this pool's, so skip a scan that cannot succeed: it
    // costs one ledger lock per committed block.
    const auto free_bytes =
        state_->ledger->Snapshot().bytes[static_cast<std::size_t>(
            cache::ResourceCategory::kBackingFree)];
    const bool may_fit = free_bytes >= blocks * state_->block_bytes;
    for (std::size_t tried = 0; may_fit && blocks <= count && tried < count;) {
      // An occupied page rules out every candidate run that crosses it, and
      // a run ends with its segment. Skip without allocating assignment
      // bookkeeping, including full exhaustion.
      const auto limit = std::min(count, (i / segment + 1) * segment);
      std::size_t free = 0;
      while (free < blocks && i + free < limit &&
             state_->blocks[i + free].Info().category ==
                 cache::ResourceCategory::kBackingFree)
        ++free;
      if (free == blocks) {
        try {
          auto reservation = cache::ResourceCharge::ReserveBacking(
              std::span(state_->blocks.get(), count).subspan(i, blocks),
              category);
          state_->next.store(i + blocks == count ? 0 : i + blocks,
                             std::memory_order_relaxed);
          return BackingBlock(state_, i, blocks, std::move(reservation));
        } catch (const std::bad_alloc&) {
          // A racing assignment or injected bookkeeping failure is atomic.
        }
      }
      const auto step =
          free < blocks
              ? std::min(i + free == limit ? free : free + 1, count - i)
              : 1;
      tried += step;
      i = i + step >= count ? 0 : i + step;
    }
    if (!state_->Grow(count))
      return std::nullopt;
  }
}
void CommittedBackingPool::GrowAhead(std::size_t segments) {
  const auto committed = state_->committed.load(std::memory_order_acquire);
  if (!segments || committed == state_->capacity)
    return;
  const auto free_bytes =
      state_->ledger->Snapshot().bytes[static_cast<std::size_t>(
          cache::ResourceCategory::kBackingFree)];
  const auto target_blocks =
      segments >= 1 + (state_->capacity - 1) / state_->segment_blocks
          ? state_->capacity
          : segments * state_->segment_blocks;
  const auto target_bytes = target_blocks * state_->block_bytes;
  if (free_bytes >= target_bytes)
    return;
  const std::lock_guard lock(ahead_mutex_);
  if (state_->growing_ahead.exchange(true))
    return;
  if (ahead_.joinable())
    ahead_.join();
  try {
    ahead_ = std::jthread([state = state_, committed, target_bytes] {
      try {
        auto seen = committed;
        while (state->ledger->Snapshot().bytes[static_cast<std::size_t>(
                   cache::ResourceCategory::kBackingFree)] < target_bytes) {
          if (!state->Grow(seen))
            break;
          seen = state->committed.load(std::memory_order_acquire);
        }
      } catch (...) {
      }
      state->growing_ahead.store(false);
    });
  } catch (...) {
    state_->growing_ahead.store(false);
  }
}
void CommittedBackingPool::WaitForGrowth() {
  const std::lock_guard lock(ahead_mutex_);
  if (ahead_.joinable())
    ahead_.join();
}
std::size_t CommittedBackingPool::BlockCount() const noexcept {
  return state_->committed.load(std::memory_order_acquire);
}
std::size_t CommittedBackingPool::CommittedBytes() const noexcept {
  return BlockCount() * state_->block_bytes;
}
std::size_t CommittedBackingPool::PageCommitAllocations() const noexcept {
  return state_->SegmentCount();
}

}  // namespace gufo::hip
