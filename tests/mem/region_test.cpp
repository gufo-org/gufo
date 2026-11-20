// GPU checks of demand-mapped state regions (src/core/hip/state_*) on the
// current device: segment growth, map-ahead, accounting, recycled-page zeroing
// and graph replay across growth.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/core/hip/state_memory.hpp"
#include "src/core/hip/state_page_pool.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/lendable.hpp"

namespace {

using gufo::mem::kPageBytes;
using gufo::mem::PagePool;
using gufo::mem::Region;
using gufo::mem::StateLedger;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void FillRows(std::uint32_t* data, std::size_t stride_words,
                         std::size_t row_words, std::size_t rows,
                         std::uint32_t salt) {
  const std::size_t segment = blockIdx.y;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < rows * row_words; i += gridDim.x * blockDim.x)
    data[segment * stride_words + i] =
        salt ^ static_cast<std::uint32_t>(segment * 1000003u + i);
}

bool RowsMatch(const Region& region, std::size_t rows, std::uint32_t salt) {
  const auto& shape = region.shape();
  const std::size_t stride = shape.segment_stride
                                 ? shape.segment_stride
                                 : shape.rows * shape.row_bytes;
  std::vector<std::uint32_t> host(rows * shape.row_bytes / 4);
  for (std::size_t s = 0; s < shape.segments; ++s) {
    Check(hipMemcpy(host.data(),
                    static_cast<std::uint8_t*>(region.data()) + s * stride,
                    host.size() * 4, hipMemcpyDeviceToHost),
          "read rows");
    for (std::size_t i = 0; i < host.size(); ++i)
      if (host[i] != (salt ^ static_cast<std::uint32_t>(s * 1000003u + i)))
        return false;
  }
  return true;
}

// Layer-major growth: every segment backs its first rows, odd strides share
// boundary pages, and the address never moves.
void Segments(bool demand) {
  const std::size_t row_bytes = 3 * 1024;  // pages do not divide rows
  const std::size_t rows = 5000;
  const std::size_t segments = 7;
  std::string error;
  auto region = Region::Create(
      {.row_bytes = row_bytes, .rows = rows, .segments = segments}, &error);
  assert(region && error.empty());
  assert(region->demand_mapped() == demand);
  void* const base = region->data();
  assert(region->bytes() == segments * rows * row_bytes);
  const std::size_t words = row_bytes / 4;
  for (const std::size_t backed :
       {std::size_t{1}, std::size_t{700}, std::size_t{2049}, rows}) {
    assert(region->Ensure(backed, &error));
    assert(region->backed_rows() >= backed);
    assert(region->data() == base);
    FillRows<<<dim3(64, segments), 256>>>(static_cast<std::uint32_t*>(base),
                                          rows * words, words, backed, 7);
    Check(hipDeviceSynchronize(), "fill");
    assert(RowsMatch(*region, backed, 7));
  }
  // Rows past the shape clamp instead of failing.
  assert(region->Ensure(rows * 4, &error));
  if (demand)
    assert(region->mapped_bytes() <= gufo::mem::RoundUpToPage(region->bytes()));
}

// The ledger promises the whole region at creation and moves bytes from
// outstanding to mapped as rows are backed; release returns everything.
void Accounting() {
  auto& ledger = StateLedger::Process();
  const auto before = ledger.Totals();
  std::string error;
  {
    auto region =
        Region::Create({.row_bytes = 4096, .rows = 4096}, &error);  // 16 MiB
    assert(region && region->demand_mapped());
    const auto promised = ledger.Totals();
    assert(promised.reserved_bytes == before.reserved_bytes + (16u << 20));
    assert(region->Ensure(512, &error));  // one page
    const auto mapped = ledger.Totals();
    assert(mapped.mapped_bytes == before.mapped_bytes + kPageBytes);
    // MemGetInfo subtracts what is promised but neither mapped nor pooled.
    std::size_t free_less = 0, total = 0, free_raw = 0;
    Check(gufo::mem::MemGetInfo(&free_less, &total), "info");
    Check(hipMemGetInfo(&free_raw, &total), "raw info");
    assert(free_less + mapped.outstanding_bytes >= free_raw - (64u << 20));
  }
  const auto after = ledger.Totals();
  assert(after.reserved_bytes == before.reserved_bytes);
  assert(after.mapped_bytes == before.mapped_bytes);
}

// A session's account charges its regions' pages under a promise cap:
// writes past it fail, map-ahead stops at the need, trimming unmaps the
// pages rows no longer use and keeps the rows that remain readable.
void SharedAccount() {
  using gufo::mem::StateAccount;
  auto& ledger = StateLedger::Process();
  const auto charged_before = ledger.ChargedBytes();
  std::string error;
  {
    auto account = std::make_shared<StateAccount>(/*margin_tokens=*/0);
    // Two segments of 8 MiB, rows of 2 KiB: 1,024 rows per page.
    auto region = Region::Create(
        {.row_bytes = 2048, .rows = 4096, .segments = 2}, &error, account);
    assert(region && region->demand_mapped());
    assert(account->full_bytes() == 8 * kPageBytes);
    ledger.SetPromiseCap(charged_before + 4 * kPageBytes);
    assert(account->Claim(2048));  // two pages per segment
    assert(account->charged_bytes() == 4 * kPageBytes);
    // Map-ahead stops at the need.
    region->MapAhead(4096);
    PagePool::ForDevice(0)->Drain();
    assert(region->backed_rows() == 2048);
    // A write past the cap fails and maps nothing more.
    assert(!region->Ensure(3000, &error));
    assert(region->mapped_bytes() == 4 * kPageBytes);
    const std::size_t words = 2048 / 4;
    FillRows<<<dim3(64, 2), 256>>>(static_cast<std::uint32_t*>(region->data()),
                                   4096 * words, words, 2048, 11);
    Check(hipDeviceSynchronize(), "fill");
    // Trim to 1,024 rows: one page per segment returns, the rows stay.
    assert(account->TrimTo(1024) == 2 * kPageBytes);
    assert(region->mapped_bytes() == 2 * kPageBytes);
    assert(region->backed_rows() == 1024);
    assert(account->charged_bytes() == 2 * kPageBytes);
    assert(RowsMatch(*region, 1024, 11));
    // Growing again maps zeroed pages.
    assert(account->Claim(2048));
    assert(region->Ensure(2048, &error));
    std::vector<std::uint32_t> host(1024 * words);
    Check(hipMemcpy(host.data(),
                    static_cast<std::uint8_t*>(region->data()) + 1024 * 2048,
                    host.size() * 4, hipMemcpyDeviceToHost),
          "read regrown rows");
    assert(std::ranges::all_of(host, [](std::uint32_t v) { return v == 0; }));
    ledger.SetPromiseCap(0);
  }
  assert(ledger.ChargedBytes() == charged_before);
}

// The pool thread maps ahead; a writer then finds its rows backed.
void MapAhead() {
  std::string error;
  auto region = Region::Create(
      {.row_bytes = 2048, .rows = 1 << 16, .segments = 4}, &error);
  assert(region && region->demand_mapped());
  assert(region->Ensure(16, &error));
  region->MapAhead(40000);
  auto pool = PagePool::ForDevice(0);
  pool->Drain();
  assert(region->backed_rows() >= 40000);
  assert(region->Ensure(40000, &error));
}

// Pages recycled from a released region come back zeroed, as new ones do;
// so does memory a whole allocation reuses.
void Recycled() {
  std::string error;
  const std::size_t rows = 8 * kPageBytes / 4096;
  {
    auto dirty = Region::Create({.row_bytes = 4096, .rows = rows}, &error);
    assert(dirty && dirty->Ensure(rows, &error));
    Check(hipMemset(dirty->data(), 0xff, dirty->bytes()), "dirty");
    Check(hipDeviceSynchronize(), "dirty sync");
  }
  auto clean = Region::Create({.row_bytes = 4096, .rows = rows}, &error);
  assert(clean && clean->Ensure(rows, &error));
  std::vector<std::uint8_t> host(clean->bytes());
  Check(
      hipMemcpy(host.data(), clean->data(), host.size(), hipMemcpyDeviceToHost),
      "read clean");
  for (const auto byte : host)
    assert(byte == 0);
}

// A graph captured before growth replays over rows mapped after it.
void GraphAcrossGrowth() {
  std::string error;
  const std::size_t rows = 1 << 15, row_bytes = 1024;
  auto region = Region::Create({.row_bytes = row_bytes, .rows = rows}, &error);
  assert(region && region->Ensure(64, &error));
  hipStream_t stream{};
  Check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
  auto* data = static_cast<std::uint32_t*>(region->data());
  hipGraph_t graph{};
  hipGraphExec_t exec{};
  Check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
        "capture");
  // The pool thread maps while this thread captures.
  region->MapAhead(rows);
  FillRows<<<dim3(64, 1), 256, 0, stream>>>(data, 0, row_bytes / 4, rows, 3);
  Check(hipStreamEndCapture(stream, &graph), "end capture");
  Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
  assert(region->Ensure(rows, &error));
  Check(hipGraphLaunch(exec, stream), "launch");
  Check(hipStreamSynchronize(stream), "replay");
  assert(RowsMatch(*region, rows, 3));
  (void)hipGraphExecDestroy(exec);
  (void)hipGraphDestroy(graph);
  (void)hipStreamDestroy(stream);
}

__global__ void CheckWords(const std::uint32_t* data, std::size_t words,
                           std::uint32_t salt, unsigned* bad) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    if (data[i] != (salt ^ static_cast<std::uint32_t>(i)))
      atomicAdd(bad, 1u);
}

// RAM-cache pages: one pointer for the CPU and the GPU; a lent page maps
// into a session (cleared first) and returns to its slot when the session
// ends; the ledger follows every move.
void Lending() {
  auto pool = PagePool::ForDevice(0);
  if (!pool->exportable()) {
    std::puts("mem_region_test: pages not exportable; lending skipped");
    return;
  }
  // Last case: stop the pool thread so it never borrows on its own (it does
  // below the free floor) and pages move only through this test's calls.
  pool->Stop();
  auto& ledger = StateLedger::Process();
  const auto before = ledger.Totals();
  constexpr std::size_t kPages = 16;
  auto range = gufo::mem::CreateLendableRange(kPages * kPageBytes - 1);
  assert(range && range->bytes() == kPages * kPageBytes);
  assert(range->page_bytes() == kPageBytes);
  assert(range->Commit(0, kPages, true));
  assert(ledger.Totals().lendable_bytes ==
         before.lendable_bytes + kPages * kPageBytes);
  // CPU writes, GPU reads through the same pointer, and back.
  auto* words = reinterpret_cast<std::uint32_t*>(range->data());
  const std::size_t count = range->bytes() / 4;
  for (std::size_t i = 0; i < count; ++i)
    words[i] = 0x5a5a5a5au ^ static_cast<std::uint32_t>(i);
  unsigned* bad = nullptr;
  Check(hipMalloc(&bad, sizeof(unsigned)), "bad");
  Check(hipMemset(bad, 0, sizeof(unsigned)), "bad clear");
  CheckWords<<<256, 256>>>(words, count, 0x5a5a5a5au, bad);
  unsigned mismatches = 1;
  Check(hipMemcpy(&mismatches, bad, sizeof(unsigned), hipMemcpyDeviceToHost),
        "cpu to gpu");
  assert(mismatches == 0);
  FillRows<<<dim3(64, 1), 256>>>(words, 0, count, 1, 0x1234u);
  Check(hipDeviceSynchronize(), "gpu fill");
  for (std::size_t i = 0; i < count; ++i)
    assert(words[i] == (0x1234u ^ static_cast<std::uint32_t>(i)));
  // Lend the top two pages; nothing is lent without an owner.
  std::vector<std::size_t> restored;
  assert(pool->Borrow(2) == 0);
  range->SetOwner(
      [](std::size_t pages, std::vector<std::size_t>& out) {
        for (std::size_t p = kPages; p-- > 0 && out.size() < pages;)
          out.push_back(p);
        return out.size();
      },
      [&](std::size_t page) { restored.push_back(page); });
  const auto lent_before = pool->Stats().lent;
  assert(pool->Borrow(2) == 2);
  assert(pool->Stats().lent == lent_before + 2);
  assert(ledger.Totals().lendable_bytes ==
         before.lendable_bytes + (kPages - 2) * kPageBytes);
  {
    // A region takes the lent pages first; they arrive cleared.
    std::string error;
    auto region = Region::Create(
        {.row_bytes = 4096, .rows = 2 * kPageBytes / 4096}, &error);
    assert(region && region->Ensure(2 * kPageBytes / 4096, &error));
    std::vector<std::uint32_t> host(2 * kPageBytes / 4);
    Check(hipMemcpy(host.data(), region->data(), host.size() * 4,
                    hipMemcpyDeviceToHost),
          "read lent");
    for (const auto word : host)
      assert(word == 0);
    // The cache still maps them: the session's writes show there too.
    FillRows<<<dim3(64, 1), 256>>>(static_cast<std::uint32_t*>(region->data()),
                                   0, host.size(), 1, 0x77u);
    Check(hipDeviceSynchronize(), "session fill");
  }
  // The session ended: both pages are back with the range.
  std::ranges::sort(restored);
  assert((restored == std::vector<std::size_t>{kPages - 2, kPages - 1}));
  assert(ledger.Totals().lendable_bytes ==
         before.lendable_bytes + kPages * kPageBytes);
  range->ClearOwner();
  range.reset();
  const auto after = ledger.Totals();
  assert(after.lendable_bytes == before.lendable_bytes);
  (void)hipFree(bad);
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "device");
  const bool demand = PagePool::ForDevice(0) != nullptr;
  Segments(demand);
  Recycled();
  if (!demand) {
    std::puts("mem_region_test: no demand mapping; whole-allocation path ok");
    return 0;
  }
  Accounting();
  SharedAccount();
  MapAhead();
  GraphAcrossGrowth();
  Lending();
  const auto stats =
      PagePool::ForDevice(0)->Stats();  // after Stop: counters only
  std::printf(
      "mem_region_test: ok (pool free=%zu in_use=%zu ahead=%zu "
      "on_demand=%zu)\n",
      stats.free_pages, stats.pages_in_use, stats.created_ahead,
      stats.created_on_demand);
  return 0;
}
