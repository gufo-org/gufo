#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo;
using Category = cache::ResourceCategory;
using Result = cache::TransferResult;
namespace {
std::atomic<std::size_t> allocations{};
std::atomic<std::size_t> host_allocations{};
std::atomic<std::size_t> max_piece{};
std::atomic<bool> fail_populate{};
bool fail_copy{}, fail_record{}, fail_query{}, fail_wait{}, fail_alloc{};
int copies_before_failure{};
thread_local bool fail_stream_new{}, observe_allocation_cleanup{};
std::atomic<unsigned> allocation_cleanup_drains{};
}  // namespace
void* operator new(std::size_t size) {
  ++host_allocations;
  if (fail_stream_new && size == sizeof(hip::TransferStream)) {
    fail_stream_new = false;
    throw std::bad_alloc();
  }
  if (void* memory = std::malloc(size ? size : 1))
    return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept {
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}
// Observe the actual HIP allocation/creation entry points, not just a counter
// maintained by the implementation. Host bookkeeping is permitted on requests.
extern "C" {
int __real_madvise(void*, std::size_t, int);
int __wrap_madvise(void* data, std::size_t bytes, int advice) {
  if (advice == MADV_POPULATE_WRITE && fail_populate.exchange(false)) {
    errno = ENOMEM;
    return -1;
  }
  return __real_madvise(data, bytes, advice);
}
hipError_t __real_hipHostMalloc(void**, std::size_t, unsigned);
hipError_t __wrap_hipHostMalloc(void** p, std::size_t n, unsigned flags) {
  ++allocations;
  return fail_alloc ? hipErrorOutOfMemory : __real_hipHostMalloc(p, n, flags);
}
hipError_t __real_hipMalloc(void**, std::size_t);
hipError_t __wrap_hipMalloc(void** p, std::size_t n) {
  ++allocations;
  return __real_hipMalloc(p, n);
}
hipError_t __real_hipHostRegister(void*, std::size_t, unsigned);
hipError_t __wrap_hipHostRegister(void* p, std::size_t n, unsigned flags) {
  ++allocations;
  return fail_alloc ? hipErrorOutOfMemory : __real_hipHostRegister(p, n, flags);
}
hipError_t __real_hipMallocManaged(void**, std::size_t, unsigned);
hipError_t __wrap_hipMallocManaged(void** p, std::size_t n, unsigned flags) {
  ++allocations;
  return __real_hipMallocManaged(p, n, flags);
}
hipError_t __real_hipStreamCreateWithFlags(hipStream_t*, unsigned);
hipError_t __wrap_hipStreamCreateWithFlags(hipStream_t* p, unsigned flags) {
  ++allocations;
  return __real_hipStreamCreateWithFlags(p, flags);
}
hipError_t __real_hipEventCreateWithFlags(hipEvent_t*, unsigned);
hipError_t __wrap_hipEventCreateWithFlags(hipEvent_t* p, unsigned flags) {
  ++allocations;
  return __real_hipEventCreateWithFlags(p, flags);
}
hipError_t __real_hipMemcpyAsync(void*, const void*, std::size_t, hipMemcpyKind,
                                 hipStream_t);
hipError_t __wrap_hipMemcpyAsync(void* dst, const void* src, std::size_t n,
                                 hipMemcpyKind kind, hipStream_t stream) {
  max_piece = std::max(max_piece.load(), n);
  if (fail_copy && copies_before_failure-- == 0)
    return hipErrorInvalidValue;
  return __real_hipMemcpyAsync(dst, src, n, kind, stream);
}
hipError_t __real_hipEventRecord(hipEvent_t, hipStream_t);
hipError_t __wrap_hipEventRecord(hipEvent_t event, hipStream_t stream) {
  return fail_record ? hipErrorInvalidValue
                     : __real_hipEventRecord(event, stream);
}
hipError_t __real_hipEventQuery(hipEvent_t);
hipError_t __wrap_hipEventQuery(hipEvent_t event) {
  return fail_query ? hipErrorInvalidValue : __real_hipEventQuery(event);
}
hipError_t __real_hipEventSynchronize(hipEvent_t);
hipError_t __wrap_hipEventSynchronize(hipEvent_t event) {
  return fail_wait ? hipErrorInvalidValue : __real_hipEventSynchronize(event);
}
hipError_t __real_hipStreamSynchronize(hipStream_t);
hipError_t __wrap_hipStreamSynchronize(hipStream_t stream) {
  if (observe_allocation_cleanup)
    ++allocation_cleanup_drains;
  return __real_hipStreamSynchronize(stream);
}
}

namespace {
template<typename Exception, typename F>
void Throws(F action) {
  bool caught = false;
  try {
    action();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}
constexpr std::size_t block_bytes = 64 << 10;
constexpr std::size_t metadata = 16 << 10;
constexpr std::size_t budget = 3 * block_bytes + metadata;
cache::ResourceLimits Limits(std::size_t bytes = budget) {
  return {bytes, bytes, 0, bytes};
}
void Backing() {
  cache::ResourceLedger ledger(Limits());
  {
    hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
    assert(pool.BlockCount() == 3);
    assert(pool.PageCommitAllocations() == 1);
    assert(pool.CommittedBytes() == 3 * block_bytes);
    assert(ledger.Snapshot().total_bytes == budget);
    const auto before = allocations.load();
    auto state = pool.TryAcquire(Category::kPrivateState);
    auto tail = pool.TryAcquire(Category::kPrivateTail);
    auto spill = pool.TryAcquire(Category::kBackingAssigned);
    assert(state && tail && spill);
    assert(state->Info().bytes == block_bytes && state->Info().reserved);
    assert(state->Bytes().data() != tail->Bytes().data());
    assert(std::all_of(state->Bytes().begin(), state->Bytes().end(),
                       [](std::byte b) { return b == std::byte{}; }));
    state->Convert();
    tail->Convert();
    spill->Convert();
    assert(spill->Info().category == Category::kBackingMaterialized);
    assert(ledger.Snapshot().total_bytes == budget);
    assert(!pool.TryAcquire(Category::kPrivateState));
    auto pin = std::make_optional(state->PinPersistence());
    auto* address = state->Bytes().data();
    state.reset();
    assert(!pool.TryAcquire(Category::kPrivateState));
    pin.reset();
    state = pool.TryAcquire(Category::kPrivateState);
    assert(state->Bytes().data() == address);
    state.reset();  // unconverted reservation cancels
    ledger.FailAfter(cache::LedgerStep::kReserve, 0);
    assert(!pool.TryAcquire(Category::kPrivateTail));
    state = pool.TryAcquire(Category::kPrivateTail);
    assert(state);
    ledger.FailAfter(cache::LedgerStep::kConvert, 0);
    Throws<std::bad_alloc>([&] { state->Convert(); });
    assert(state->Info().reserved);
    state->Convert();
    assert(allocations == before);
    Throws<std::invalid_argument>(
        [&] { (void)pool.TryAcquire(Category::kMetadata); });
  }
  assert(ledger.Snapshot().total_bytes == 0);
  // Physical memory survives facade and borrower destruction while pinned.
  std::optional<hip::BackingPin> survivor;
  std::byte* address{};
  {
    hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
    auto block = pool.TryAcquire(Category::kPrivateState);
    block->Convert();
    address = block->Bytes().data();
    survivor.emplace(block->PinPersistence());
  }
  address[0] = std::byte{42};
  assert(ledger.Snapshot().total_bytes == budget);
  survivor.reset();
  assert(ledger.Snapshot().total_bytes == 0);
  // Failed initialization unwinds all reservations and committed metadata.
  fail_alloc = true;
  Throws<std::runtime_error>([&] {
    hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
  });
  fail_alloc = false;
  assert(ledger.Snapshot().total_bytes == 0);
  ledger.FailAfter(cache::LedgerStep::kConvert, 2);  // metadata and first block
  Throws<std::bad_alloc>([&] {
    hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
  });
  assert(ledger.Snapshot().total_bytes == 0);
  Throws<std::invalid_argument>(
      [&] { hip::CommittedBackingPool pool(ledger, {budget, 1, metadata}); });
  {
    hip::CommittedBackingPool empty(ledger, {metadata, block_bytes, metadata});
    assert(empty.CommittedBytes() == 0 &&
           !empty.TryAcquire(Category::kPrivateTail));
  }
  // Filling a large pool must not repeatedly allocate assignment tokens for
  // its occupied prefix. Repeat after release to exercise cursor wraparound.
  {
    constexpr std::size_t count = 128;
    constexpr std::size_t headroom = (count + 4) * 1024;
    constexpr std::size_t capacity = count * block_bytes + headroom;
    cache::ResourceLedger large_ledger(Limits(capacity));
    hip::CommittedBackingPool large(large_ledger,
                                    {capacity, block_bytes, headroom});
    std::vector<hip::BackingBlock> held;
    held.reserve(count);
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
      const auto before_host = host_allocations.load();
      for (std::size_t i = 0; i < count; ++i) {
        auto block = large.TryAcquire(Category::kPrivateTail);
        assert(block);
        held.push_back(std::move(*block));
      }
      assert(host_allocations - before_host <= 2 * count);
      const auto before_full = host_allocations.load();
      assert(!large.TryAcquire(Category::kPrivateTail));
      assert(host_allocations.load() == before_full);
      held.clear();
      assert(large_ledger.Snapshot()
                 .bytes[static_cast<std::size_t>(Category::kBackingFree)] ==
             count * block_bytes);
    }
  }
  // Competing borrowers may never own the same physical block.
  hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
  std::atomic<unsigned> active{};
  const auto before = allocations.load();
  std::vector<std::jthread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < 100; ++j) {
        auto block = pool.TryAcquire(Category::kPrivateTail);
        if (!block)
          continue;
        const auto index = static_cast<unsigned>(
            reinterpret_cast<std::uintptr_t>(block->Bytes().data()) /
            block_bytes % 32);
        const unsigned bit = 1u << index;
        assert(!(active.fetch_or(bit) & bit));
        block->Convert();
        assert(active.fetch_and(~bit) & bit);
      }
    });
  }
  threads.clear();
  assert(allocations == before);
}
void GroupedBacking() {
  constexpr std::size_t pages = 12, headroom = 16 << 10;
  constexpr std::size_t capacity = pages * block_bytes + headroom;
  cache::ResourceLedger ledger(Limits(capacity));
  std::optional<hip::BackingPin> pin;
  std::byte* base{};
  const auto physical = allocations.load();
  {
    hip::CommittedBackingPool pool(ledger, {capacity, block_bytes, headroom});
    auto first = pool.TryAcquire(Category::kPrivateState, 3 * block_bytes - 1);
    assert(first && first->Bytes().size() == 3 * block_bytes);
    const auto* first_data = first->Bytes().data();
    std::fill(first->Bytes().begin(), first->Bytes().end(), std::byte{42});
    first->Convert();
    auto second = pool.TryAcquire(Category::kBackingAssigned, 9 * block_bytes);
    assert(second);
    base = std::min(first->Bytes().data(), second->Bytes().data());
    assert(first_data + 3 * block_bytes == second->Bytes().data() ||
           second->Bytes().data() + 9 * block_bytes == first_data);
    const auto full = host_allocations.load();
    assert(!pool.TryAcquire(Category::kPrivateTail, 2 * block_bytes));
    assert(host_allocations.load() == full);
    first.reset();
    auto reused = pool.TryAcquire(Category::kPrivateTail, 3 * block_bytes);
    assert(reused && reused->Bytes().data() == first_data);
    reused.reset();
    second.reset();
    // Capacity released by a row group immediately serves a larger private
    // state, without another physical allocation or class-specific capacity.
    auto whole = pool.TryAcquire(Category::kPrivateState, pages * block_bytes);
    assert(whole && whole->Bytes().data() == base);
    whole->Convert();
    pin.emplace(whole->PinPersistence());
    assert(ledger.Snapshot().persistence_pinned_bytes == pages * block_bytes);
    whole.reset();
    assert(!pool.TryAcquire(Category::kPrivateTail));
    pin.reset();
    std::atomic<unsigned> active{};
    std::vector<std::jthread> threads;
    for (int thread = 0; thread < 8; ++thread)
      threads.emplace_back([&, thread] {
        const auto n = 2 + thread % 2;
        for (int iteration = 0; iteration < 100; ++iteration) {
          auto group = pool.TryAcquire(
              thread % 2 ? Category::kPrivateState : Category::kBackingAssigned,
              n * block_bytes);
          if (!group)
            continue;
          const auto index = (group->Bytes().data() - base) / block_bytes;
          const unsigned mask = ((1u << n) - 1) << index;
          assert(!(active.fetch_or(mask) & mask));
          group->Convert();
          assert((active.fetch_and(~mask) & mask) == mask);
        }
      });
    threads.clear();
    assert(!active);
    auto last = pool.TryAcquire(Category::kPrivateState, 4 * block_bytes);
    last->Convert();
    pin.emplace(last->PinPersistence());
    assert(pool.PageCommitAllocations() == 1 && allocations == physical + 1);
  }
  // Every page and its physical owner survive destruction of the facade.
  base[4 * block_bytes - 1] = std::byte{17};
  assert(ledger.Snapshot().total_bytes == capacity);
  pin.reset();
  assert(ledger.Snapshot().total_bytes == 0);
}
void CheckpointLifetimes() {
  constexpr std::size_t pages = 12, headroom = 16 << 10;
  constexpr std::size_t capacity = pages * block_bytes + headroom;
  cache::ResourceLedger ledger(Limits(capacity));
  {
    hip::CommittedBackingPool pool(ledger, {capacity, block_bytes, headroom});
    const auto physical = allocations.load();
    std::vector<hip::BackingBlock> private_states, shared_rows;
    // Adjacent checkpoints replace private state but keep sharing older KV
    // rows. Even allocating each larger state first can scatter the holes
    // between those long-lived rows when both use one allocation cursor.
    for (unsigned checkpoint = 0; checkpoint < 3; ++checkpoint) {
      auto state = pool.TryAcquire(Category::kPrivateState, 2 * block_bytes);
      auto rows = pool.TryAcquire(Category::kBackingAssigned, block_bytes);
      assert(state && rows);
      state->Convert();
      rows->Convert();
      std::fill(rows->Bytes().begin(), rows->Bytes().end(),
                std::byte(checkpoint + 1));
      private_states.push_back(std::move(*state));
      shared_rows.push_back(std::move(*rows));
    }
    // A restore joins chunks only when their logical order also advances
    // through adjacent backing addresses. Reverse KV placement would turn
    // one large transfer into one transfer per chunk.
    for (std::size_t i = 1; i < shared_rows.size(); ++i)
      assert(shared_rows[i].Bytes().data() ==
                 shared_rows[i - 1].Bytes().data() + block_bytes &&
             "KV placement prevented contiguous restore transfers");
    private_states.clear();
    assert(ledger.Snapshot()
               .bytes[static_cast<std::size_t>(Category::kBackingFree)] ==
           9 * block_bytes);
    // All free capacity must remain usable by the next private state, with
    // the same physical budget and the shared rows still live and unmodified.
    auto replacement =
        pool.TryAcquire(Category::kPrivateState, 9 * block_bytes);
    assert(replacement && "shared rows fragmented released private states");
    replacement->Convert();
    std::fill(replacement->Bytes().begin(), replacement->Bytes().end(),
              std::byte{0x7f});
    for (unsigned i = 0; i < shared_rows.size(); ++i)
      assert(std::all_of(
          shared_rows[i].Bytes().begin(), shared_rows[i].Bytes().end(),
          [&](std::byte value) { return value == std::byte(i + 1); }));
    assert(allocations.load() == physical);
    assert(!pool.TryAcquire(Category::kPrivateTail));
    replacement.reset();
    shared_rows.clear();
    // Neither allocation category has a fixed share of the pool.
    for (const auto category :
         {Category::kBackingAssigned, Category::kPrivateTail}) {
      auto whole = pool.TryAcquire(category, pages * block_bytes);
      assert(whole);
    }
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void OnDemandBacking() {
  // Five blocks in segments of two: nothing commits until a capture needs it,
  // runs stay inside one segment and the last segment is partial.
  constexpr std::size_t pages = 5, headroom = 16 << 10;
  constexpr std::size_t capacity = pages * block_bytes + headroom;
  cache::ResourceLedger ledger(Limits(capacity));
  std::size_t asked{};
  bool admit = true;
  std::optional<hip::BackingPin> pin;
  std::byte* survivor{};
  {
    const auto physical = allocations.load();
    hip::CommittedBackingPool pool(
        ledger, {capacity, block_bytes, headroom, 2 * block_bytes,
                 [&](std::size_t bytes) {
                   asked += bytes;
                   return admit;
                 }});
    assert(pool.BlockCount() == 0 && pool.PageCommitAllocations() == 0);
    assert(allocations == physical &&
           ledger.Snapshot().total_bytes == headroom);
    assert(!pool.TryAcquire(Category::kPrivateState, 3 * block_bytes));
    assert(allocations == physical && asked == 0);
    auto first = pool.TryAcquire(Category::kPrivateState);
    assert(first && pool.BlockCount() == 2 &&
           pool.PageCommitAllocations() == 1);
    assert(allocations == physical + 1 && asked == 2 * block_bytes);
    assert(std::all_of(first->Bytes().begin(), first->Bytes().end(),
                       [](std::byte b) { return b == std::byte{}; }));
    first->Convert();
    // One free block remains in the first segment; a pair needs the next.
    auto pair = pool.TryAcquire(Category::kBackingAssigned, 2 * block_bytes);
    assert(pair && pool.BlockCount() == 4 && allocations == physical + 2);
    auto single = pool.TryAcquire(Category::kPrivateTail);
    assert(single && pool.BlockCount() == 4);
    assert(single->Bytes().data() == first->Bytes().data() + block_bytes ||
           first->Bytes().data() == single->Bytes().data() + block_bytes);
    // A refused segment leaves the pool as it is; capture is declined.
    admit = false;
    assert(!pool.TryAcquire(Category::kPrivateTail));
    assert(pool.BlockCount() == 4 && allocations == physical + 2);
    admit = true;
    ledger.FailAfter(cache::LedgerStep::kReserve, 0);
    assert(!pool.TryAcquire(Category::kPrivateTail));
    assert(pool.BlockCount() == 4 && allocations == physical + 2);
    fail_alloc = true;
    assert(!pool.TryAcquire(Category::kPrivateTail));
    fail_alloc = false;
    assert(pool.BlockCount() == 4 &&
           ledger.Snapshot().total_bytes == headroom + 4 * block_bytes);
    auto last = pool.TryAcquire(Category::kPrivateTail);
    assert(last && pool.BlockCount() == pages &&
           pool.PageCommitAllocations() == 3);
    assert(pool.CommittedBytes() == pages * block_bytes);
    assert(!pool.TryAcquire(Category::kPrivateTail));
    assert(pool.PageCommitAllocations() == 3);
    last->Convert();
    survivor = last->Bytes().data();
    pin.emplace(last->PinPersistence());
    // Released blocks serve later captures without another segment.
    pair.reset();
    auto again = pool.TryAcquire(Category::kPrivateState, 2 * block_bytes);
    assert(again && pool.PageCommitAllocations() == 3);
  }
  // A grown segment and its owner survive the facade like the first one.
  survivor[block_bytes - 1] = std::byte{9};
  pin.reset();
  assert(ledger.Snapshot().total_bytes == 0);
  // Racing captures grow the pool once per segment and never share a block.
  constexpr std::size_t many = 64, many_headroom = (many + 4) * 1024;
  constexpr std::size_t many_capacity = many * block_bytes + many_headroom;
  cache::ResourceLedger racing_ledger(Limits(many_capacity));
  hip::CommittedBackingPool pool(
      racing_ledger,
      {many_capacity, block_bytes, many_headroom, 8 * block_bytes, {}});
  std::mutex mutex;
  std::vector<const std::byte*> owned;
  const auto release = [&](std::vector<hip::BackingBlock>& held) {
    const std::lock_guard lock(mutex);
    for (const auto& block : held)
      std::erase(owned, block.Bytes().data());
    held.clear();
  };
  std::vector<std::jthread> threads;
  for (int thread = 0; thread < 8; ++thread)
    threads.emplace_back([&] {
      std::vector<hip::BackingBlock> held;
      for (int iteration = 0; iteration < 200; ++iteration) {
        auto block = pool.TryAcquire(Category::kPrivateTail);
        if (!block) {
          release(held);
          continue;
        }
        {
          const std::lock_guard lock(mutex);
          assert(std::ranges::find(owned, block->Bytes().data()) ==
                 owned.end());
          owned.push_back(block->Bytes().data());
        }
        block->Convert();
        held.push_back(std::move(*block));
        if (held.size() == 12)
          release(held);
      }
      release(held);
    });
  threads.clear();
  assert(owned.empty() && pool.BlockCount() % 8 == 0 &&
         pool.PageCommitAllocations() == pool.BlockCount() / 8);
  std::vector<hip::BackingBlock> all;
  while (auto block = pool.TryAcquire(Category::kPrivateTail))
    all.push_back(std::move(*block));
  assert(all.size() == many && pool.PageCommitAllocations() == many / 8);
  // Growing ahead commits the next segment off the capturing thread once
  // less than a segment is free, and stops at capacity.
  constexpr std::size_t ahead_capacity = 4 * block_bytes + headroom;
  cache::ResourceLedger ahead_ledger(Limits(ahead_capacity));
  hip::CommittedBackingPool ahead(
      ahead_ledger,
      {ahead_capacity, block_bytes, headroom, 2 * block_bytes, {}});
  auto held = ahead.TryAcquire(Category::kPrivateTail);
  assert(held && ahead.BlockCount() == 2);
  ahead.GrowAhead();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (ahead.BlockCount() != 4 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  assert(ahead.BlockCount() == 4 && ahead.PageCommitAllocations() == 2);
  ahead.GrowAhead();
  assert(ahead.BlockCount() == 4);
  // A bounded startup reserve finishes before requests acquire its pages.
  // Observe real HIP allocations: neither request below may commit a segment.
  cache::ResourceLedger startup_ledger(Limits(ahead_capacity));
  hip::CommittedBackingPool startup(
      startup_ledger,
      {ahead_capacity, block_bytes, headroom, 2 * block_bytes, {}});
  startup.GrowAhead();
  startup.WaitForGrowth();
  assert(startup.BlockCount() == 2);
  startup.GrowAhead(2);
  startup.WaitForGrowth();
  assert(startup.BlockCount() == 4 && startup.PageCommitAllocations() == 2);
  const auto committed_allocations = allocations.load();
  auto startup_first =
      startup.TryAcquire(Category::kPrivateTail, 2 * block_bytes);
  auto startup_second =
      startup.TryAcquire(Category::kPrivateTail, 2 * block_bytes);
  assert(startup_first && startup_second &&
         allocations.load() == committed_allocations);
}
void CaptureCapacity() {
  constexpr std::size_t pages = 6, headroom = 16 << 10;
  constexpr std::size_t capacity = pages * block_bytes + headroom;
  cache::ResourceLedger ledger(Limits(capacity));
  hip::CommittedBackingPool pool(
      ledger, {capacity, block_bytes, headroom, 3 * block_bytes, {}});
  std::array<cache::PayloadRequest, 2> capture{{
      {{1}, Category::kPrivateState, 0, 0, 2 * block_bytes - 1},
      {{2}, Category::kBackingAssigned, 0, 8, block_bytes + 1},
  }};
  const auto before = ledger.Snapshot();
  const auto physical = allocations.load(), host = host_allocations.load();
  assert(pool.CanFit(capture));
  assert(pool.CanFit({}));
  // Capacity checks must neither commit pages nor allocate metadata.
  assert(pool.BlockCount() == 0 && ledger.Snapshot() == before);
  assert(allocations.load() == physical && host_allocations.load() == host);
  auto first = pool.TryAcquire(Category::kPrivateState, 2 * block_bytes);
  assert(first && pool.BlockCount() == 3 && pool.CanFit(capture));
  auto second = pool.TryAcquire(Category::kPrivateTail, block_bytes);
  assert(second && !pool.CanFit(capture));  // only three uncommitted blocks
  first->Convert();
  auto pin = std::optional(first->PinPersistence());
  first.reset();
  assert(!pool.CanFit(capture));  // persistence retains the assignment
  pin.reset();
  assert(pool.CanFit(capture));
  // Released groups can serve a later capture without another page commit.
  auto third = pool.TryAcquire(Category::kPrivateState, 3 * block_bytes);
  assert(third && pool.BlockCount() == pages);
  third.reset();
  const auto saved = ledger.Snapshot();
  assert(pool.CanFit(capture));
  assert(ledger.Snapshot() == saved);
  capture[0].bytes = 3 * block_bytes + 1;
  assert(!pool.CanFit(capture));  // a payload cannot cross a segment
  capture[0].bytes = std::numeric_limits<std::size_t>::max();
  assert(!pool.CanFit(capture));
  capture[0].bytes = 3 * block_bytes;
  capture[1].bytes = 3 * block_bytes;
  assert(!pool.CanFit(capture));
  second.reset();
  assert(pool.CanFit(capture));
  const auto committed = allocations.load();
  auto a = pool.TryAcquire(capture[0].category, capture[0].bytes);
  auto b = pool.TryAcquire(capture[1].category, capture[1].bytes);
  assert(a && b && !pool.CanFit(capture));
  assert(allocations.load() == committed);
  a.reset();
  b.reset();
  std::array<std::optional<hip::BackingBlock>, pages> scattered;
  for (auto& block : scattered) {
    block = pool.TryAcquire(Category::kPrivateTail);
    assert(block);
  }
  for (const auto i : {1, 3, 5})
    scattered[i].reset();
  capture[0].bytes = 2 * block_bytes;
  const auto fragmented = std::span(capture).first(1);
  assert(pool.CanFit(fragmented));
  assert(!pool.TryAcquire(capture[0].category, capture[0].bytes));
}
void Transfers() {
  cache::ResourceLedger ledger(Limits());
  hip::CommittedBackingPool backing(ledger, {budget, block_bytes, metadata});
  auto source = backing.TryAcquire(Category::kPrivateTail);
  auto destination = backing.TryAcquire(Category::kPrivateTail);
  void* device{};
  assert(hipMalloc(&device, block_bytes) == hipSuccess);
  hip::TransferPool pool(2);
  const auto before = allocations.load();
  auto first = pool.TryAcquire();
  auto second = pool.TryAcquire();
  assert(first && second && first->Native() != second->Native());
  assert(!pool.TryAcquire());
  std::fill(source->Bytes().begin(), source->Bytes().end(), std::byte{0x5a});
  std::fill(destination->Bytes().begin(), destination->Bytes().end(),
            std::byte{0x33});
  // Unaligned endpoints, several pieces, and a partial last piece with guards.
  constexpr std::size_t n = block_bytes - 3;
  constexpr std::size_t piece = 4093;
  max_piece = 0;
  auto up = first->Copy(static_cast<std::byte*>(device) + 1,
                        source->Bytes().data() + 1, n,
                        hip::CopyDirection::kHostToDevice, piece);
  assert(up.Wait() == Result::kSucceeded && up.Ready());
  assert(up.Wait() == Result::kSucceeded);
  Throws<std::logic_error>([&] { (void)first->Complete(); });
  auto down = second->Copy(destination->Bytes().data() + 1,
                           static_cast<std::byte*>(device) + 1, n,
                           hip::CopyDirection::kDeviceToHost, piece);
  assert(down.Wait() == Result::kSucceeded);
  assert(max_piece <= piece);
  assert(destination->Bytes()[0] == std::byte{0x33});
  assert(destination->Bytes()[block_bytes - 1] == std::byte{0x33});
  assert(destination->Bytes()[block_bytes - 2] == std::byte{0x33});
  assert(std::memcmp(source->Bytes().data() + 1,
                     destination->Bytes().data() + 1, n) == 0);
  first.reset();
  second.reset();
  // A waited completion still owns its event until destruction/replacement.
  assert(!pool.TryAcquire());
  {
    hip::TransferPool another(1);
    auto stream = another.TryAcquire();
    auto empty =
        stream->Copy(nullptr, nullptr, 0, hip::CopyDirection::kDeviceToHost, 1);
    assert(empty.Wait() == Result::kSucceeded);
  }
  assert(allocations == before + 2);  // only the explicitly created other pool
  assert(hipFree(device) == hipSuccess);
}
void LifetimeAndFailures() {
  std::optional<cache::Completion> completion;
  {
    hip::TransferPool pool(1);
    auto stream = pool.TryAcquire();
    Throws<std::invalid_argument>([&] {
      (void)stream->Copy(nullptr, nullptr, 0, hip::CopyDirection::kDeviceToHost,
                         0);
    });
    completion.emplace(stream->Complete());
    stream.reset();
    assert(!pool.TryAcquire());
  }
  assert(completion->Wait() == Result::kSucceeded);
  completion.reset();
  for (int failure = 0; failure < 3; ++failure) {
    hip::TransferPool pool(1);
    auto stream = pool.TryAcquire();
    if (failure == 0) {
      fail_record = true;
      Throws<std::runtime_error>([&] { (void)stream->Complete(); });
      fail_record = false;
    } else {
      completion.emplace(stream->Complete());
      if (failure == 1) {
        fail_query = true;
        assert(!completion->Ready());
        fail_query = false;
      } else {
        fail_wait = true;
      }
      assert(completion->Wait() == Result::kFailed);
      fail_wait = false;
      assert(completion->Wait() == Result::kFailed);
      completion.reset();
    }
    assert(stream->Synchronize() == Result::kFailed);
    stream.reset();
    assert(!pool.TryAcquire());
  }
  // A failure after one real queued copy drains before throwing.
  void* device{};
  void* host{};
  assert(hipMalloc(&device, 8192) == hipSuccess);
  assert(hipHostMalloc(&host, 8192, hipHostMallocCoherent) == hipSuccess);
  std::memset(host, 0x71, 8192);
  hip::TransferPool pool(1);
  auto stream = pool.TryAcquire();
  fail_copy = true;
  copies_before_failure = 1;
  Throws<std::runtime_error>([&] {
    (void)stream->Copy(device, host, 8192, hip::CopyDirection::kHostToDevice,
                       4096);
  });
  fail_copy = false;
  std::vector<std::byte> actual(4096);
  assert(hipMemcpy(actual.data(), device, actual.size(),
                   hipMemcpyDeviceToHost) == hipSuccess);
  assert(std::all_of(actual.begin(), actual.end(),
                     [](std::byte b) { return b == std::byte{0x71}; }));
  stream.reset();
  assert(!pool.TryAcquire());
  assert(hipHostFree(host) == hipSuccess);
  assert(hipFree(device) == hipSuccess);
  // Successful completion destruction gives the pair back for reuse.
  hip::TransferPool reusable(1);
  for (int i = 0; i < 10; ++i) {
    auto lease = reusable.TryAcquire();
    assert(lease);
    auto done = lease->Complete();
  }
}
void PendingStreams() {
  hip::TransferPool pool(2);
  auto blocked = pool.TryAcquire();
  auto peer = pool.TryAcquire();
  std::atomic<bool> release{};
  assert(hipLaunchHostFunc(
             blocked->Native(),
             [](void* data) {
               auto& gate = *static_cast<std::atomic<bool>*>(data);
               while (!gate.load())
                 std::this_thread::yield();
             },
             &release) == hipSuccess);
  auto pending = blocked->Complete();
  assert(!pending.Ready());
  assert(!pool.TryAcquire());
  // FIFO on the blocked stream must not hold up the independent peer.
  auto ready = peer->Complete();
  assert(ready.Wait() == Result::kSucceeded);
  assert(!pending.Ready());
  release = true;
  assert(pending.Wait() == Result::kSucceeded);
}
void BorrowedPoolLifetime() {
  using namespace cache::testing;
  cache::ResourceLedger ledger(Limits(2 * budget));
  {
    FakeAdapter adapter;
    FakeStream stream;
    cache::LeasedSlot slot(ledger, adapter, stream, cache::SlotId{1});
    auto lease = slot.Acquire();
    const std::array<cache::Token, 5> tokens{1, 2, 3, 4, 5};
    adapter.Append(lease.Execution(), tokens, tokens);
    std::shared_ptr<cache::BorrowedRows> rows;
    {
      hip::CommittedBackingPool pool(ledger, {budget, block_bytes, metadata});
      auto block = pool.TryAcquire(Category::kBackingAssigned);
      const auto before = allocations.load();
      rows = std::move(*block).Borrow(lease, kTarget, 0, tokens.size(),
                                      sizeof(tokens));
      assert(allocations == before);
      assert(rows->Bytes() == block_bytes);  // Capacity padding stays charged.
      Throws<std::logic_error>([&] { (void)block->Bytes(); });
    }
    lease.Commit();
    lease = slot.Acquire();  // Preserve into the slab after pool facade
                             // destruction.
    assert(!rows->Location());
    auto pin = rows->Pin();
    assert(std::memcmp(pin.Owner().get(), tokens.data(), sizeof(tokens)) == 0);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void AcquireAllocationFailure() {
  hip::TransferPool pool(1);
  fail_stream_new = true;
  observe_allocation_cleanup = true;
  Throws<std::bad_alloc>([&] { (void)pool.TryAcquire(); });
  observe_allocation_cleanup = false;
  assert(!fail_stream_new);
  // An unclaimed lease cannot touch HIP or release a slot: after unlocking,
  // that slot might already belong to another borrower. This failed on the
  // original implementation before a second thread was even required.
  assert(allocation_cleanup_drains == 0);
  auto first = pool.TryAcquire();
  assert(first && !pool.TryAcquire());
  first.reset();
  assert(pool.TryAcquire());
}
void CheckBasePages(std::span<std::byte> data) {
  constexpr std::size_t huge = 2ULL << 20;
  assert(data.size() >= 2 * huge);
  std::vector<unsigned char> resident(data.size() / 4096);
  assert(mincore(data.data(), data.size(), resident.data()) == 0);
  assert(std::all_of(resident.begin(), resident.end(),
                     [](unsigned char page) { return page & 1; }));
  assert(std::all_of(data.begin(), data.end(),
                     [](std::byte b) { return b == std::byte{}; }));
  const auto start = reinterpret_cast<std::uintptr_t>(data.data());
  const auto aligned = (start + huge - 1) & ~(huge - 1);
  assert(aligned + huge <= start + data.size());
  errno = 0;
  const auto collapse =
      madvise(reinterpret_cast<void*>(aligned), huge, MADV_COLLAPSE);
  const auto error = errno;
  if (collapse != -1 || error != EINVAL)
    std::fprintf(stderr, "cache backing collapse returned %d, errno %d\n",
                 collapse, error);
  assert(collapse == -1 && error == EINVAL);
}
void BasePageBacking() {
  constexpr std::size_t bytes = 8ULL << 20;
  cache::ResourceLedger ledger(Limits(bytes + metadata));
  {
    hip::CommittedBackingPool pool(ledger, {bytes + metadata, bytes, metadata});
    auto block = pool.TryAcquire(Category::kPrivateState);
    assert(block);
    auto memory = block->Bytes();
    CheckBasePages(memory);
    void* device = nullptr;
    assert(hipHostGetDevicePointer(&device, memory.data(), 0) == hipSuccess);
    assert(hipMemset(device, 0x5a, memory.size()) == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);
    assert(std::all_of(memory.begin(), memory.end(),
                       [](std::byte b) { return b == std::byte{0x5a}; }));
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void PopulationFailure() {
  for (const std::size_t bytes : {8ULL << 20, 128ULL << 20}) {
    cache::ResourceLedger ledger(Limits(bytes + metadata));
    const auto before = allocations.load();
    // One worker fails while its peers may still populate other ranges.
    // No mapping may be registered or escape, and all charges must unwind.
    fail_populate = true;
    Throws<std::runtime_error>([&] {
      hip::CommittedBackingPool pool(ledger,
                                     {bytes + metadata, bytes, metadata});
    });
    assert(!fail_populate && allocations == before);
    assert(ledger.Snapshot().total_bytes == 0);
  }
}
void LargeBacking() {
  constexpr std::size_t block = 64ULL << 20;
  constexpr std::size_t bytes = 2 * block;
  cache::ResourceLedger ledger(Limits(bytes + metadata));
  {
    hip::CommittedBackingPool pool(ledger, {bytes + metadata, block, metadata});
    auto backing = pool.TryAcquire(Category::kPrivateState, bytes);
    assert(backing && backing->Bytes().size() == bytes);
    auto memory = backing->Bytes();
    CheckBasePages(memory);
    void* device = nullptr;
    assert(hipMalloc(&device, bytes) == hipSuccess);
    assert(hipMemset(device, 0x5b, bytes) == hipSuccess);
    assert(hipMemcpy(memory.data(), device, bytes, hipMemcpyDeviceToHost) ==
           hipSuccess);
    assert(std::all_of(memory.begin(), memory.end(),
                       [](std::byte b) { return b == std::byte{0x5b}; }));
    assert(hipFree(device) == hipSuccess);
  }
  assert(ledger.Snapshot().total_bytes == 0);
  fail_alloc = true;
  Throws<std::runtime_error>([&] {
    hip::CommittedBackingPool pool(ledger, {bytes + metadata, block, metadata});
  });
  fail_alloc = false;
  assert(ledger.Snapshot().total_bytes == 0);
}
}  // namespace
int main() {
  hipDeviceProp_t properties{};
  assert(hipGetDeviceProperties(&properties, 0) == hipSuccess);
  assert(std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0);
  Backing();
  GroupedBacking();
  CheckpointLifetimes();
  OnDemandBacking();
  CaptureCapacity();
  Transfers();
  LifetimeAndFailures();
  PendingStreams();
  AcquireAllocationFailure();
  BorrowedPoolLifetime();
  BasePageBacking();
  PopulationFailure();
  LargeBacking();
  std::puts(
      "PASS: precommitted and on-demand blocks, accounting, racing "
      "borrowers, pinned lifetime, bounded copies, leases and failure "
      "drains");
}
