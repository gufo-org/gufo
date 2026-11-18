// GPU checks of giving RAM-cache backing back to the host under memory
// pressure (CommittedBackingPool::Surrender, #932): only units whose blocks
// are all free go (whole state pages, whole pinned segments), the payloads
// that stay keep their bytes for the CPU and the GPU, the released blocks stay
// reserved, and growth commits them again before new segments, under the same
// admission rules. Tearing the pool down after a surrender leaves no HIP
// error.
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/state_page_pool.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/lendable.hpp"

namespace {

using gufo::cache::ResourceCategory;
using gufo::mem::kPageBytes;

constexpr std::size_t kBlock = 64 << 10;
constexpr std::size_t kSegment = 8 * kPageBytes;  // 16 MiB
constexpr std::size_t kSegments = 2;

std::size_t FreeBytes(const gufo::cache::ResourceLedger& ledger) {
  return ledger.Snapshot()
      .bytes[static_cast<std::size_t>(ResourceCategory::kBackingFree)];
}

std::size_t HeapFree() {
  std::size_t free = 0, total = 0;
  assert(gufo::mem::MemGetInfo(&free, &total) == hipSuccess);
  return free;
}

// The payload's bytes as the CPU and the GPU read them.
bool Holds(const gufo::hip::BackingBlock& block, std::byte value) {
  const auto bytes = block.Bytes();
  for (const auto b : bytes)
    if (b != value)
      return false;
  std::byte* device = nullptr;
  assert(hipMalloc(&device, bytes.size()) == hipSuccess);
  assert(hipMemcpy(device, bytes.data(), bytes.size(),
                   hipMemcpyDeviceToDevice) == hipSuccess);
  std::vector<std::byte> copy(bytes.size());
  assert(hipMemcpy(copy.data(), device, bytes.size(), hipMemcpyDeviceToHost) ==
         hipSuccess);
  (void)hipFree(device);
  return std::memcmp(copy.data(), bytes.data(), bytes.size()) == 0;
}

}  // namespace

int main() {
  if (hipSetDevice(0) != hipSuccess)
    return 1;
  auto pool = gufo::mem::PagePool::ForDevice(0);
  if (!pool || !pool->exportable()) {
    std::puts("mem_cache_surrender_test: no exportable state pages; skipped");
    return 0;
  }
  // Every page moves only through this test's calls.
  pool->Stop();
  const std::size_t capacity = kSegments * kSegment;
  const std::size_t metadata = (1 << 20) + capacity / kBlock * 1024;
  gufo::cache::ResourceLedger ledger({capacity + metadata + (1 << 20),
                                      capacity + metadata, 1 << 20,
                                      capacity + metadata});
  auto range = gufo::mem::CreateLendableRange(capacity);
  assert(range);
  // The first segment takes state pages, the second pinned host pages; later
  // state pages only while `host_has_room`.
  int state_segments = 1;
  bool filled = false;
  bool host_has_room = true;
  bool state_pages_free = true;  // Free 2 MiB blocks above the free floor.
  std::vector<std::optional<gufo::hip::BackingBlock>> held;
  auto backing = std::make_unique<gufo::hip::CommittedBackingPool>(
      ledger, gufo::hip::BackingPoolConfig{
                  capacity + metadata, kBlock, metadata, kSegment,
                  [&](std::size_t) { return host_has_room; }, nullptr, range,
                  [&](std::size_t) {
                    if (!host_has_room || !state_pages_free)
                      return false;
                    return filled || state_segments-- > 0;
                  },
                  [] { return false; }});
  // One 2 MiB payload per page: page i of the state segment, then the host
  // segment.
  const std::size_t pages = capacity / kPageBytes;
  for (std::size_t i = 0; i < pages; ++i) {
    auto block =
        backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes);
    assert(block);
    std::memset(block->Bytes().data(), 0x5a, kPageBytes);
    held.emplace_back(std::move(block));
  }
  assert(backing->CommittedBytes() == capacity);
  filled = true;

  // Nothing is free: nothing goes, and nothing is evicted here.
  assert(backing->Surrender(capacity) == 0);
  assert(backing->SurrenderedBytes() == 0);

  // Free every other state page and the whole host segment, then give
  // everything free back.
  const std::size_t state_pages = kSegment / kPageBytes;
  for (std::size_t i = 0; i < state_pages; i += 2)
    held[i].reset();
  for (std::size_t i = state_pages; i < pages; ++i)
    held[i].reset();
  const auto heap_before = HeapFree();
  const auto released = backing->Surrender(capacity);
  const auto heap_after = HeapFree();
  assert(released == state_pages / 2 * kPageBytes + kSegment);
  assert(backing->SurrenderedBytes() == released);
  // The state pages went back to the driver (the host segment's pinned host
  // pages are not device heap).
  assert(heap_after >= heap_before + state_pages / 2 * kPageBytes);
  // Released blocks are reserved, not free; the payloads that stayed are
  // intact for the CPU and the GPU.
  assert(FreeBytes(ledger) == 0);
  for (std::size_t i = 1; i < state_pages; i += 2)
    assert(held[i] && Holds(*held[i], std::byte{0x5a}));
  // A second call finds nothing more.
  assert(backing->Surrender(capacity) == 0);

  // While the host refuses new pages, nothing comes back.
  host_has_room = false;
  assert(!backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes));
  assert(backing->SurrenderedBytes() == released);

  // While state pages are refused but pinned host pages are not (free pages
  // below the floor, as page cache leaves them while serving), the released
  // host segment still comes back; the released state pages wait.
  host_has_room = true;
  state_pages_free = false;
  std::vector<std::optional<gufo::hip::BackingBlock>> again;
  for (std::size_t i = 0; i < kSegment / kPageBytes; ++i) {
    auto block =
        backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes);
    assert(block);
    again.emplace_back(std::move(block));
  }
  assert(backing->SurrenderedBytes() == state_pages / 2 * kPageBytes);
  assert(!backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes));
  again.clear();

  // With room again, growth commits the released units first, and their
  // pages work for the CPU and the GPU.
  state_pages_free = true;
  for (std::size_t i = 0; i < state_pages / 2 + kSegment / kPageBytes; ++i) {
    auto block =
        backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes);
    assert(block);
    std::memset(block->Bytes().data(), 0x3c, kPageBytes);
    assert(Holds(*block, std::byte{0x3c}));
    again.emplace_back(std::move(block));
  }
  assert(backing->SurrenderedBytes() == 0);
  assert(backing->CommittedBytes() == capacity);
  assert(!backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes));
  for (std::size_t i = 1; i < state_pages; i += 2)
    assert(Holds(*held[i], std::byte{0x5a}));

  // Teardown after a surrender: released units hold no pages and leave no
  // HIP error behind.
  again.clear();
  held.clear();
  (void)hipGetLastError();
  assert(backing->Surrender(capacity) == capacity);
  backing.reset();
  range.reset();
  assert(hipGetLastError() == hipSuccess);

  // The same with pinned host segments only, as without state pages.
  {
    gufo::cache::ResourceLedger host_ledger({capacity + metadata + (1 << 20),
                                             capacity + metadata, 1 << 20,
                                             capacity + metadata});
    auto host = std::make_unique<gufo::hip::CommittedBackingPool>(
        host_ledger, gufo::hip::BackingPoolConfig{
                         capacity + metadata, kBlock, metadata, kSegment,
                         [](std::size_t) { return true; }, nullptr});
    auto block =
        host->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes);
    assert(block);
    block.reset();
    assert(host->Surrender(capacity) == kSegment);
    host.reset();
    assert(hipGetLastError() == hipSuccess);
  }
  std::printf("mem_cache_surrender_test: ok (released %zu MiB)\n",
              released >> 20);
  return 0;
}
