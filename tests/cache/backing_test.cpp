#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <optional>
#include <thread>
#include <vector>

#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/transfer_pool.hpp"

using namespace gufo;
using Category = cache::ResourceCategory;
using Result = cache::TransferResult;
namespace {
std::atomic<std::size_t> allocations{};
std::atomic<std::size_t> max_piece{};
bool fail_copy{}, fail_record{}, fail_query{}, fail_wait{}, fail_alloc{};
int copies_before_failure{};
thread_local bool fail_stream_new{}, observe_allocation_cleanup{};
std::atomic<unsigned> allocation_cleanup_drains{};
}  // namespace
void* operator new(std::size_t size) {
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
  return __real_hipHostRegister(p, n, flags);
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
}  // namespace
int main() {
  hipDeviceProp_t properties{};
  assert(hipGetDeviceProperties(&properties, 0) == hipSuccess);
  assert(std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0);
  Backing();
  Transfers();
  LifetimeAndFailures();
  PendingStreams();
  AcquireAllocationFailure();
  std::puts(
      "PASS: precommitted blocks, accounting, racing borrowers, pinned "
      "lifetime, bounded copies, leases and failure drains");
}
