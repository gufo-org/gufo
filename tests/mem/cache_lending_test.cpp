// GPU checks of the RAM cache on state pages (src/core/hip/committed_backing
// with a mem::LendableRange): segments take state pages or pinned host pages,
// only state pages are lent to sessions, a lent page's blocks stay reserved
// until it comes back, and the cache's checkpoints are evicted (shed) to free
// whole pages. A backing that may not lend gives sessions none.
#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/hip/committed_backing.hpp"
#include "src/core/hip/state_page_pool.hpp"
#include "src/core/hip/state_region.hpp"
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

}  // namespace

int main() {
  if (hipSetDevice(0) != hipSuccess)
    return 1;
  auto pool = gufo::mem::PagePool::ForDevice(0);
  if (!pool || !pool->exportable()) {
    std::puts("mem_cache_lending_test: no exportable state pages; skipped");
    return 0;
  }
  // Without a region the pool thread has nothing to do; stop it so it never
  // borrows on its own (it does below the free floor, as right after a large
  // build) and every page moves only through this test's calls.
  pool->Stop();
  const std::size_t capacity = kSegments * kSegment;
  const std::size_t metadata = (1 << 20) + capacity / kBlock * 1024;
  gufo::cache::ResourceLedger ledger({capacity + metadata + (1 << 20),
                                      capacity + metadata, 1 << 20,
                                      capacity + metadata});
  auto range = gufo::mem::CreateLendableRange(capacity);
  assert(range);
  // The first segment takes state pages, the second pinned host pages.
  int state_segments = 1;
  int sheds = 0;
  bool may_lend = false;
  std::vector<std::optional<gufo::hip::BackingBlock>> held;
  auto backing = std::make_unique<gufo::hip::CommittedBackingPool>(
      ledger, gufo::hip::BackingPoolConfig{
                  capacity + metadata, kBlock, metadata, kSegment,
                  [](std::size_t) { return true; }, nullptr, range,
                  [&](std::size_t) { return state_segments-- > 0; },
                  [&] {
                    // Eviction: drop one held payload, as retention would.
                    if (held.empty())
                      return false;
                    held.pop_back();
                    ++sheds;
                    return true;
                  },
                  [&] { return may_lend; }});
  assert(backing->Lends());
  // Fill both segments with one 2 MiB payload per page, so no page is free
  // and each eviction frees exactly one: the host-page segment's first (they
  // were taken last), which are never lent, then state pages.
  for (std::size_t i = 0; i < capacity / kPageBytes; ++i) {
    auto block =
        backing->TryAcquire(ResourceCategory::kBackingAssigned, kPageBytes);
    assert(block);
    std::memset(block->Bytes().data(), 0x5a, kPageBytes);
    held.emplace_back(std::move(block));
  }
  assert(backing->CommittedBytes() == capacity);
  // The state segment's pages are one pointer for the CPU and the GPU.
  const auto first = held.front()->Bytes();
  std::byte* device = nullptr;
  assert(hipMalloc(&device, kPageBytes) == hipSuccess);
  assert(hipMemcpy(device, first.data(), kPageBytes, hipMemcpyDeviceToDevice) ==
         hipSuccess);
  std::vector<std::byte> copy(kPageBytes);
  assert(hipMemcpy(copy.data(), device, kPageBytes, hipMemcpyDeviceToHost) ==
         hipSuccess);
  assert(std::memcmp(copy.data(), first.data(), kPageBytes) == 0);
  (void)hipFree(device);

  // A backing that may not lend (with the RAM cache off it only stages disk
  // captures) gives sessions no page and sheds nothing.
  const std::size_t state_pages = kSegment / kPageBytes;
  assert(pool->Borrow(state_pages) == 0);
  assert(backing->LentPages() == 0 && sheds == 0);
  may_lend = true;
  // Lending sheds held payloads until state pages are free; host-page
  // segments are never lent.
  const auto lent = pool->Borrow(state_pages + 4);
  assert(lent == state_pages);
  assert(backing->LentPages() == state_pages);
  assert(sheds > 0);
  // The lent pages' blocks are reserved: the cache cannot hand them out.
  const auto free_before = FreeBytes(ledger);
  {
    std::string error;
    auto region = gufo::mem::Region::Create(
        {.row_bytes = 4096, .rows = state_pages * kPageBytes / 4096}, &error);
    assert(region && region->Ensure(region->shape().rows, &error));
    assert(backing->LentPages() == state_pages);
  }
  // The region released the pages: they are back, their blocks free again.
  assert(backing->LentPages() == 0);
  assert(FreeBytes(ledger) >= free_before + kSegment);
  assert(backing->LentPagesTotal() == state_pages);
  held.clear();
  backing.reset();
  range.reset();
  std::printf("mem_cache_lending_test: ok (lent %zu pages, %d sheds)\n", lent,
              sheds);
  return 0;
}
