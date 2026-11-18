#include "src/core/hip/committed_backing.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/hip/host_allocations.hpp"
#include "src/core/hip/state_memory.hpp"
#include "src/core/platform/page_pin.hpp"
#include "src/core/platform/pages.hpp"

namespace gufo::hip {
namespace detail {
namespace {
struct CommittedPages {
  void* data{};
  int pin{-1};
  explicit operator bool() const { return data != nullptr; }
};

// Ordinary anonymous pages, then registered with the GPU. The platform leaf
// commits, opts out of THP and populates; registration stays here.
CommittedPages CommitPages(std::size_t bytes) {
  void* data = platform::CommitZeroedPages(bytes);
  if (data == nullptr)
    return {};
  const int pin = platform::PinHostPages(data, bytes);
  if (hipHostRegister(data, bytes, hipHostRegisterDefault) != hipSuccess) {
    platform::UnpinHostPages(pin);
    platform::ReleaseCommittedPages(data, bytes);
    return {};
  }
  return {data, pin};
}
bool HeapFree(std::size_t& free_bytes) {
  std::size_t total_bytes = 0;
  return mem::MemGetInfo(&free_bytes, &total_bytes) == hipSuccess;
}
void ReleasePages(CommittedPages pages, std::size_t bytes) {
  (void)hipHostUnregister(pages.data);
  platform::UnpinHostPages(pages.pin);
  platform::ReleaseCommittedPages(pages.data, bytes);
}
}  // namespace
struct BackingState {
  cache::ResourceCharge metadata;
  // Fixed capacity; entries below `committed` are published and never move.
  std::unique_ptr<cache::ResourceCharge[]> blocks;
  std::unique_ptr<CommittedPages[]> segments;
  std::size_t capacity{}, segment_blocks{}, block_bytes{};
  std::atomic<std::size_t> committed{}, next{};
  std::atomic<std::size_t> pinned_bytes{};
  // Keep private states apart from longer-lived KV rows. Rows and tails grow
  // forward so restore can still join adjacent ranges into large transfers.
  // Both cursors can use the whole pool; there is no partition or extra budget.
  std::atomic<std::size_t> next_private{
      std::numeric_limits<std::size_t>::max()};
  cache::ResourceLedger* ledger{};
  std::function<bool(std::size_t)> admit_segment;
  std::function<bool(std::size_t, std::size_t, std::size_t)> keep_segment;
  std::mutex growth;
  std::atomic<bool> growing_ahead{};
  // State pages (see BackingPoolConfig::lendable). A lent page's blocks are
  // held by a reservation until the page comes back.
  std::shared_ptr<mem::LendableRange> lendable;
  std::function<bool(std::size_t)> admit_state_pages;
  std::function<bool()> shed;
  std::function<bool()> may_lend;
  std::size_t page_blocks{};
  std::mutex lend_mutex;
  std::unique_ptr<cache::ResourceReservation[]> lent;
  std::atomic<std::size_t> lent_pages{}, lent_total{};
  // Set once loading ends; from then on new pages come only from free host
  // memory, never from reclaim.
  std::atomic<bool> serving{};
  // Units given back to the host under memory pressure (Surrender): state
  // pages and pinned segments, under `growth`. Their blocks stay reserved,
  // for pages in `lent`, until Grow() commits them again. Retired pages
  // (a decommit that could not unmap) stay reserved for good.
  std::vector<std::size_t> surrendered_pages;
  std::unique_ptr<cache::ResourceReservation[]> surrendered_segments;
  std::atomic<std::size_t> surrendered_bytes{};
  // A segment of pinned host pages has its own address; one of state pages
  // lives in the range.
  std::byte* SegmentData(std::size_t segment) const {
    return segments[segment].data
               ? static_cast<std::byte*>(segments[segment].data)
               : lendable->data() + segment * segment_blocks * block_bytes;
  }
  std::byte* Address(std::size_t index) const {
    return SegmentData(index / segment_blocks) +
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
    // Units given back under pressure come back first, under the same rules
    // as new ones.
    if (const auto again = Recommit())
      return *again;
    if (first == capacity)
      return false;
    const auto count = std::min(segment_blocks, capacity - first);
    // State pages first, when their own rule admits them; otherwise pinned
    // host pages under admit_segment, as without state pages.
    const bool state_pages =
        lendable && admit_state_pages && admit_state_pages(count * block_bytes);
    if (!state_pages && admit_segment && !admit_segment(count * block_bytes))
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
    if (state_pages) {
      // Whole pages; segments start on page boundaries. The pages are the
      // device heap's own, so the rule has already checked its reserve.
      const auto page = lendable->page_bytes();
      const auto begin = first * block_bytes / page;
      const auto end = ((first + count) * block_bytes + page - 1) / page;
      if (lendable->Commit(begin, end - begin,
                           !serving.load(std::memory_order_acquire))) {
        for (std::size_t i = 0; i < count; ++i)
          blocks[first + i] = reservations[i].Convert();
        committed.store(first + count, std::memory_order_release);
        return true;
      }
      // Pages a partial commit made stay in the range for a later segment.
      if (admit_segment && !admit_segment(count * block_bytes))
        return false;
    }
    std::size_t heap_before = 0, heap_after = 0;
    if (keep_segment && !HeapFree(heap_before))
      return false;
    auto& data = segments[first / segment_blocks];
    data = CommitPages(count * block_bytes);
    if (!data)
      return false;
    if (keep_segment &&
        !(HeapFree(heap_after) &&
          keep_segment(count * block_bytes, heap_before, heap_after))) {
      ReleasePages(data, count * block_bytes);
      data = {};
      return false;
    }
    try {
      for (std::size_t i = 0; i < count; ++i)
        blocks[first + i] = reservations[i].Convert();
    } catch (const std::bad_alloc&) {
      // Free before the converted charges unwind, as at initialization.
      ReleasePages(data, count * block_bytes);
      data = {};
      for (std::size_t i = 0; i < count; ++i)
        blocks[first + i] = {};
      return false;
    }
    if (data.pin >= 0)
      pinned_bytes.fetch_add(count * block_bytes, std::memory_order_relaxed);
    committed.store(first + count, std::memory_order_release);
    return true;
  }
  // Commits one surrendered unit again. Nullopt when none can come back
  // under its own rule; false when the host refuses a pinned segment. A
  // refused state page (free pages below the floor, as page cache leaves them
  // while serving) leaves pinned segments and new growth to their own rules,
  // as without state pages. Call with `growth` held.
  std::optional<bool> Recommit() {
    if (!surrendered_pages.empty()) {
      const auto page = lendable->page_bytes();
      const auto p = surrendered_pages.back();
      if (admit_state_pages && admit_state_pages(page) &&
          lendable->Commit(p, 1, !serving.load(std::memory_order_acquire))) {
        surrendered_pages.pop_back();
        surrendered_bytes.fetch_sub(page, std::memory_order_relaxed);
        const std::lock_guard lock(lend_mutex);
        lent[p] = {};
        return true;
      }
    }
    if (!surrendered_segments)
      return std::nullopt;
    const auto count = SegmentCount();
    for (std::size_t i = count; i-- > 0;) {
      if (!surrendered_segments[i])
        continue;
      const auto bytes = SegmentBytes(i);
      if (admit_segment && !admit_segment(bytes))
        return false;
      std::size_t heap_before = 0, heap_after = 0;
      if (keep_segment && !HeapFree(heap_before))
        return false;
      auto data = CommitPages(bytes);
      if (!data)
        return false;
      if (keep_segment && !(HeapFree(heap_after) &&
                            keep_segment(bytes, heap_before, heap_after))) {
        ReleasePages(data, bytes);
        return false;
      }
      segments[i] = data;
      if (data.pin >= 0)
        pinned_bytes.fetch_add(bytes, std::memory_order_relaxed);
      surrendered_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      surrendered_segments[i] = {};
      return true;
    }
    return std::nullopt;
  }
  // Gives up to about `want` bytes of committed backing whose blocks are all
  // free back to the host: whole state pages, then whole pinned segments,
  // from the top. Nothing is evicted here. The caller makes sure no GPU work
  // is in flight: unmapping a state page waits for the device.
  std::size_t Surrender(std::size_t want) {
    const std::lock_guard grow(growth);
    std::size_t released = 0;
    const auto count = committed.load(std::memory_order_acquire);
    if (lendable && page_blocks) {
      const auto page = lendable->page_bytes();
      std::vector<std::size_t> chosen;
      {
        const std::lock_guard lock(lend_mutex);
        for (std::size_t p = count / page_blocks;
             p-- > 0 && chosen.size() * page < want;) {
          if (lent[p] || segments[p * page_blocks / segment_blocks])
            continue;  // Lent, surrendered, or pinned host pages.
          const auto range =
              std::span(blocks.get() + p * page_blocks, page_blocks);
          if (cache::ResourceCharge::FindFreeBacking(range, page_blocks) != 0)
            continue;
          try {
            lent[p] = cache::ResourceCharge::ReserveBacking(
                range, cache::ResourceCategory::kBackingAssigned);
          } catch (const std::bad_alloc&) {
            continue;  // A capture took one of them meanwhile.
          }
          chosen.push_back(p);
        }
      }
      if (!chosen.empty()) {
        const auto result = lendable->Decommit(chosen);
        std::vector<bool> kept(chosen.size(), true);
        const auto mark = [&](std::size_t p) {
          kept[std::ranges::find(chosen, p) - chosen.begin()] = false;
        };
        for (const auto p : result.released) {
          mark(p);
          surrendered_pages.push_back(p);
          released += page;
        }
        for (const auto p : result.retired)
          mark(p);
        surrendered_bytes.fetch_add(result.released.size() * page,
                                    std::memory_order_relaxed);
        const std::lock_guard lock(lend_mutex);
        for (std::size_t i = 0; i < chosen.size(); ++i)
          if (kept[i])
            lent[chosen[i]] = {};
      }
    }
    for (std::size_t i = (count + segment_blocks - 1) / segment_blocks;
         i-- > 0 && released < want;) {
      if (!segments[i])
        continue;
      const auto first = i * segment_blocks;
      const auto range = std::span(blocks.get() + first,
                                   std::min(segment_blocks, count - first));
      if (cache::ResourceCharge::FindFreeBacking(range, range.size()) != 0)
        continue;
      if (!surrendered_segments)
        surrendered_segments = std::make_unique<cache::ResourceReservation[]>(
            (capacity + segment_blocks - 1) / segment_blocks + 1);
      try {
        surrendered_segments[i] = cache::ResourceCharge::ReserveBacking(
            range, cache::ResourceCategory::kBackingAssigned);
      } catch (const std::bad_alloc&) {
        continue;
      }
      const auto bytes = SegmentBytes(i);
      if (segments[i].pin >= 0)
        pinned_bytes.fetch_sub(bytes, std::memory_order_relaxed);
      ReleasePages(segments[i], bytes);
      segments[i] = {};
      released += bytes;
      surrendered_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return released;
  }
  // Up to `want` committed pages whose blocks are all free, reserved for a
  // session until Restore(). Evicts one retained checkpoint at a time while
  // too few are free. Runs on the state pool's thread.
  std::size_t SelectPages(std::size_t want, std::vector<std::size_t>& out) {
    // Bounds one pass: a few evictions free whole pages unless payloads
    // are fragmented, and the pool asks again while it is short.
    constexpr int kMaxSheds = 16;
    const auto start = out.size();
    if (may_lend && !may_lend())
      return 0;
    for (int sheds = 0;; ++sheds) {
      {
        const std::lock_guard lock(lend_mutex);
        const auto pages =
            committed.load(std::memory_order_acquire) / page_blocks;
        // From the top: the forward cursor refills low blocks first.
        for (std::size_t p = pages; p-- > 0 && out.size() - start < want;) {
          if (lent[p] || segments[p * page_blocks / segment_blocks])
            continue;  // Lent already, or pinned host pages.
          const auto range =
              std::span(blocks.get() + p * page_blocks, page_blocks);
          if (cache::ResourceCharge::FindFreeBacking(range, page_blocks) != 0)
            continue;
          try {
            lent[p] = cache::ResourceCharge::ReserveBacking(
                range, cache::ResourceCategory::kBackingAssigned);
          } catch (const std::bad_alloc&) {
            continue;  // A capture took one of them meanwhile.
          }
          out.push_back(p);
          lent_pages.fetch_add(1, std::memory_order_relaxed);
          lent_total.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (out.size() - start >= want || sheds == kMaxSheds || !shed || !shed())
        break;
    }
    return out.size() - start;
  }
  void Restore(std::size_t page) {
    const std::lock_guard lock(lend_mutex);
    if (lent[page]) {
      lent[page] = {};
      lent_pages.fetch_sub(1, std::memory_order_relaxed);
    }
  }
  ~BackingState() {
    if (lendable) {
      lendable->ClearOwner();
      // Lent reservations go before the blocks they hold.
      if (lent)
        for (std::size_t p = 0; p < (capacity + page_blocks - 1) / page_blocks;
             ++p)
          lent[p] = {};
      // Pinned host segments here; the range frees its own pages when its
      // last owner goes.
      for (std::size_t i = 0; i < SegmentCount(); ++i)
        if (segments[i])
          ReleasePages(segments[i], SegmentBytes(i));
      return;
    }
    // Physical storage is freed before its original committed charges.
    if (segments)
      for (std::size_t i = 0; i < SegmentCount(); ++i)
        if (segments[i])  // Surrendered segments have no pages.
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
  state->segments = std::make_unique<detail::CommittedPages[]>(
      (count + state->segment_blocks - 1) / state->segment_blocks + 1);
  state->ledger = &ledger;
  state->admit_segment = std::move(config.admit_segment);
  state->keep_segment = std::move(config.keep_segment);
  if (config.lendable && config.segment_bytes && count &&
      config.lendable->page_bytes() % config.block_bytes == 0 &&
      config.lendable->bytes() >=
          (count * config.block_bytes + config.lendable->page_bytes() - 1) /
              config.lendable->page_bytes() * config.lendable->page_bytes()) {
    const auto page_blocks = config.lendable->page_bytes() / config.block_bytes;
    // Segments in whole pages, so each starts on a page of its own.
    state->segment_blocks =
        (state->segment_blocks + page_blocks - 1) / page_blocks * page_blocks;
    state->page_blocks = page_blocks;
    state->lendable = std::move(config.lendable);
    state->admit_state_pages = std::move(config.admit_state_pages);
    state->shed = std::move(config.shed);
    state->may_lend = std::move(config.may_lend);
    state->lent = std::make_unique<cache::ResourceReservation[]>(
        (count + page_blocks - 1) / page_blocks);
    const std::weak_ptr<detail::BackingState> weak = state;
    state->lendable->SetOwner(
        [weak](std::size_t pages, std::vector<std::size_t>& out) {
          const auto owner = weak.lock();
          return owner ? owner->SelectPages(pages, out) : 0;
        },
        [weak](std::size_t page) {
          if (const auto owner = weak.lock())
            owner->Restore(page);
        });
  }
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
        data = {};
      }
      for (std::size_t i = 0; i < count; ++i)
        state->blocks[i] = {};
      throw;
    }
    if (data.pin >= 0)
      state->pinned_bytes.store(count * config.block_bytes,
                                std::memory_order_relaxed);
    state->committed.store(count, std::memory_order_release);
  }
  state_ = std::move(state);
}
bool CommittedBackingPool::CanFit(
    std::span<const cache::PayloadRequest> payloads) const {
  std::size_t needed = 0;
  for (const auto& payload : payloads) {
    const auto blocks =
        payload.bytes ? 1 + (payload.bytes - 1) / state_->block_bytes : 1;
    if (blocks > std::min(state_->segment_blocks, state_->capacity) ||
        blocks > state_->capacity - needed)
      return false;
    needed += blocks;
  }
  // Read the committed frontier first. Concurrent growth can then only
  // overestimate available space, leaving final admission to TryAcquire.
  const auto uncommitted =
      state_->capacity - state_->committed.load(std::memory_order_acquire);
  if (needed <= uncommitted)
    return true;
  // Other pools on this ledger can also make this bound optimistic. It must
  // never reject a capture whose physical allocations could fit.
  const auto free = state_->ledger->Snapshot().bytes[static_cast<std::size_t>(
                        cache::ResourceCategory::kBackingFree)] /
                    state_->block_bytes;
  return needed - uncommitted <= free;
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
  const bool from_end = category == cache::ResourceCategory::kPrivateState;
  auto& cursor = from_end ? state_->next_private : state_->next;
  while (true) {
    const auto count = state_->committed.load(std::memory_order_acquire);
    auto i = cursor.load(std::memory_order_relaxed);
    if (i >= count)
      i = from_end && count ? count - 1 : 0;
    // A growing pool is usually full when a capture misses. The ledger's free
    // backing bounds this pool's, so skip a scan that cannot succeed: it
    // costs one ledger lock per committed block.
    const auto free_bytes =
        state_->ledger->Snapshot().bytes[static_cast<std::size_t>(
            cache::ResourceCategory::kBackingFree)];
    const bool may_fit = free_bytes >= blocks * state_->block_bytes;
    for (std::size_t tried = 0; may_fit && blocks <= count && tried < count;) {
      // Search one physical segment under one ledger lock instead of locking
      // separately for each block in a large, fragmented cache.
      auto begin = from_end ? i / segment * segment : i;
      auto limit =
          from_end ? i + 1 : std::min(count, (i / segment + 1) * segment);
      // After wrapping, inspect each possible start/end only once. Include
      // the rest of a grouped allocation, but do not retry a reservation
      // that already failed earlier in this scan.
      const auto candidates = std::min(count - tried, limit - begin);
      const auto length =
          candidates + std::min(blocks - 1, limit - begin - candidates);
      if (from_end)
        begin = limit - length;
      else
        limit = begin + length;
      const auto range =
          std::span(state_->blocks.get(), count).subspan(begin, limit - begin);
      const auto offset =
          cache::ResourceCharge::FindFreeBacking(range, blocks, from_end);
      if (offset < range.size()) {
        try {
          auto reservation = cache::ResourceCharge::ReserveBacking(
              range.subspan(offset, blocks), category);
          const auto first = begin + offset;
          cursor.store(from_end
                           ? (first ? first - 1
                                    : std::numeric_limits<std::size_t>::max())
                           : (first + blocks == count ? 0 : first + blocks),
                       std::memory_order_relaxed);
          return BackingBlock(state_, first, blocks, std::move(reservation));
        } catch (const std::bad_alloc&) {
          // A racing assignment or injected bookkeeping failure is atomic.
        }
      }
      const auto step = offset == range.size() ? range.size()
                        : from_end ? range.size() - (offset + blocks) + 1
                                   : offset + 1;
      tried += step;
      i = from_end ? (step > i ? count - 1 : i - step)
                   : (i + step >= count ? 0 : i + step);
    }
    if (!state_->Grow(count))
      return std::nullopt;
  }
}
void CommittedBackingPool::GrowAhead(std::size_t reserve_bytes) {
  const auto committed = state_->committed.load(std::memory_order_acquire);
  if (!reserve_bytes || committed == state_->capacity ||
      state_->growing_ahead.load(std::memory_order_acquire))
    return;
  const auto free_bytes =
      state_->ledger->Snapshot().bytes[static_cast<std::size_t>(
          cache::ResourceCategory::kBackingFree)];
  const auto target_bytes =
      std::min(reserve_bytes, state_->capacity * state_->block_bytes);
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
std::size_t CommittedBackingPool::LongTermPinnedBytes() const noexcept {
  return state_->pinned_bytes.load(std::memory_order_relaxed);
}
void CommittedBackingPool::StartServing() noexcept {
  state_->serving.store(true, std::memory_order_release);
}
bool CommittedBackingPool::Lends() const noexcept {
  return state_->lendable != nullptr;
}
std::size_t CommittedBackingPool::LentPages() const noexcept {
  return state_->lent_pages.load(std::memory_order_relaxed);
}
std::size_t CommittedBackingPool::LentPagesTotal() const noexcept {
  return state_->lent_total.load(std::memory_order_relaxed);
}
std::size_t CommittedBackingPool::Surrender(std::size_t bytes) {
  return bytes ? state_->Surrender(bytes) : 0;
}
std::size_t CommittedBackingPool::SurrenderedBytes() const noexcept {
  return state_->surrendered_bytes.load(std::memory_order_relaxed);
}

}  // namespace gufo::hip
