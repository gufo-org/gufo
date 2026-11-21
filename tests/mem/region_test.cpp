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
#include "src/core/hip/state_shared_page.hpp"
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

// Checks read memory with a kernel, through the device's own translation
// of the address. A runtime copy can disagree with it after an address was
// mapped again (mem::RemapBarrier), and the model's kernels are what counts.
__global__ void CountMismatches(const std::uint32_t* data, std::size_t words,
                                std::uint32_t salt, std::uint32_t base,
                                bool constant, std::uint32_t* bad) {
  std::uint32_t local = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x)
    local += data[i] !=
             (constant ? salt : salt ^ static_cast<std::uint32_t>(base + i));
  if (local)
    atomicAdd(bad, local);
}

std::uint32_t Mismatches(const void* data, std::size_t words,
                         std::uint32_t salt, std::size_t base, bool constant) {
  static std::uint32_t* bad = nullptr;
  if (bad == nullptr)
    Check(hipMalloc(&bad, sizeof(*bad)), "mismatch counter");
  Check(hipMemset(bad, 0, sizeof(*bad)), "clear counter");
  CountMismatches<<<64, 256>>>(static_cast<const std::uint32_t*>(data), words,
                               salt, static_cast<std::uint32_t>(base), constant,
                               bad);
  Check(hipDeviceSynchronize(), "check words");
  std::uint32_t count = 0;
  Check(hipMemcpy(&count, bad, sizeof(count), hipMemcpyDeviceToHost),
        "read counter");
  return count;
}

// Words [0, words) at `data` hold FillRows' pattern from `first_word` on.
bool WordsMatch(const void* data, std::size_t words, std::uint32_t salt,
                std::size_t first_word = 0) {
  return Mismatches(data, words, salt, first_word, false) == 0;
}

bool WordsAre(const void* data, std::size_t words, std::uint32_t value) {
  return Mismatches(data, words, value, 0, true) == 0;
}

bool RowsMatch(const Region& region, std::size_t rows, std::uint32_t salt) {
  const auto& shape = region.shape();
  const std::size_t stride = shape.segment_stride
                                 ? shape.segment_stride
                                 : shape.rows * shape.row_bytes;
  for (std::size_t s = 0; s < shape.segments; ++s)
    if (!WordsMatch(static_cast<std::uint8_t*>(region.data()) + s * stride,
                    rows * shape.row_bytes / 4, salt, s * 1000003u))
      return false;
  return true;
}

// A consumer on a non-blocking stream can write as soon as Create returns.
void EagerClearBeforeFirstWrite() {
  constexpr std::size_t bytes = 4096;
  hipStream_t writer = nullptr;
  Check(hipStreamCreateWithFlags(&writer, hipStreamNonBlocking), "writer");
  std::uint8_t* host = nullptr;
  Check(hipHostMalloc(&host, bytes, hipHostMallocDefault), "pinned sentinel");
  std::fill_n(host, bytes, 0x5a);

  std::string error;
  auto region = Region::Create({.row_bytes = bytes, .rows = 1}, &error);
  assert(region && !region->demand_mapped());
  Check(hipMemcpyAsync(region->data(), host, bytes, hipMemcpyHostToDevice,
                       writer),
        "first write");
  Check(hipStreamSynchronize(writer), "finish first write");
  Check(hipDeviceSynchronize(), "drain outstanding work");
  Check(hipMemcpy(host, region->data(), bytes, hipMemcpyDeviceToHost),
        "read sentinel");
  assert(std::all_of(host, host + bytes,
                     [](std::uint8_t value) { return value == 0x5a; }));

  Check(hipHostFree(host), "free sentinel");
  Check(hipStreamDestroy(writer), "destroy writer");
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
    // The addresses lost their pages a moment ago: the new pages are
    // reachable, and cleared, only after the remap barrier Ensure() ran.
    assert(WordsAre(static_cast<std::uint8_t*>(region->data()) + 1024 * 2048,
                    1024 * words, 0));
    // The rows before them are still the region's own.
    assert(RowsMatch(*region, 1024, 11));
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

// A checkpoint keeps a session's pages instead of copying its rows: the
// park maps them a second time, a second session restores them by mapping,
// the writer copies a page only when it rewrites shared rows, and the ledger
// counts every page once.
void SharedPages() {
  auto park = gufo::mem::PagePark::ForDevice(0);
  if (!park) {
    std::puts("mem_region_test: pages cannot be shared; sharing skipped");
    return;
  }
  auto pool = PagePool::ForDevice(0);
  auto& ledger = StateLedger::Process();
  const auto before = ledger.Totals();
  const auto used_before = pool->Stats().pages_in_use;
  const std::size_t row_bytes = 1024, rows_per_page = kPageBytes / row_bytes;
  const std::size_t pages = 6, rows = pages * rows_per_page;
  const std::size_t page_words = kPageBytes / 4;
  std::string error;
  auto a = Region::Create({.row_bytes = row_bytes, .rows = rows}, &error);
  auto b = Region::Create({.row_bytes = row_bytes, .rows = rows}, &error);
  assert(a && b);
  assert(Region::Find(a->data()) == a);
  assert(Region::Find(static_cast<std::uint8_t*>(a->data()) + a->bytes() - 1) ==
         a);
  assert(Region::Find(static_cast<std::uint8_t*>(b->data()) + 5) == b);
  // The writer fills four pages and half of the fifth.
  const std::size_t frontier = 4 * rows_per_page + rows_per_page / 2;
  assert(a->Ensure(frontier, &error));
  auto* a_words = static_cast<std::uint32_t*>(a->data());
  FillRows<<<dim3(64, 1), 256>>>(a_words, 0, row_bytes / 4, frontier, 0xaau);
  Check(hipDeviceSynchronize(), "fill a");
  const auto mapped = ledger.Totals().mapped_bytes;
  // A checkpoint shares the rows written so far: no copy, no new page.
  std::vector<std::shared_ptr<gufo::mem::ParkedPage>> parked;
  for (std::size_t p = 0; p < 5; ++p) {
    const auto end = std::min((p + 1) * kPageBytes, frontier * row_bytes);
    auto page = a->Share(p, end);
    assert(page);
    parked.push_back(park->Park(page));
    assert(parked.back() && parked.back()->page() == page);
    assert(park->Park(page) == parked.back());  // One address per page.
    assert(park->Find(parked.back()->data() + 4096) == parked.back());
    assert(park->Contains(parked.back()->data()));
  }
  assert(!park->Contains(a->data()));
  assert(a->shared_pages() == 5);
  assert(ledger.Totals().mapped_bytes == mapped);
  assert(ledger.Totals().lendable_bytes == before.lendable_bytes);
  assert(pool->Stats().pages_in_use == used_before + 5);
  for (std::size_t p = 0; p < 4; ++p)
    assert(WordsMatch(parked[p]->data(), page_words, 0xaau, p * page_words));
  // Appending past the shared rows writes in place, even in a shared page.
  assert(a->PrepareWrite(frontier, frontier + 16, frontier, &error));
  assert(a->shared_pages() == 5);
  // A second session restores the four whole pages by mapping them and
  // leaves the fifth, which the checkpoint only half fills, to a load.
  std::vector<std::shared_ptr<gufo::mem::SharedPage>> plan(pages);
  for (std::size_t p = 0; p < 4; ++p)
    plan[p] = parked[p]->page();
  auto adopted = b->Restore(plan, frontier);
  assert(b->SettleRemaps(&error));
  assert((adopted == std::vector<bool>{true, true, true, true, false, false}));
  assert(b->backed_rows() == 4 * rows_per_page && b->shared_pages() == 4);
  assert(ledger.Totals().mapped_bytes == mapped);  // Still counted once.
  assert(b->Ensure(frontier, &error));
  assert(ledger.Totals().mapped_bytes == mapped + kPageBytes);
  assert(WordsMatch(b->data(), 4 * page_words, 0xaau));
  // Restoring the same checkpoint again keeps what is already mapped.
  adopted = b->Restore(plan, frontier);
  assert(b->SettleRemaps(&error));
  assert(adopted[0] && adopted[3] && b->shared_pages() == 4);
  // The second session rewinds into page 2 and rewrites from there: it gets
  // a copy of page 2, whose first rows it keeps, and a cleared page 3; the
  // others stay shared.
  const std::size_t rewind = 2 * rows_per_page + 100;
  assert(b->PrepareWrite(rewind, 4 * rows_per_page, rewind, &error));
  // The new pages are reachable, copied and cleared only after the barrier.
  assert(b->SettleRemaps(&error));
  assert(b->shared_pages() == 2);
  assert(ledger.Totals().mapped_bytes == mapped + 3 * kPageBytes);
  assert(WordsMatch(b->data(), 3 * page_words, 0xaau));
  assert(WordsAre(static_cast<std::uint8_t*>(b->data()) + 3 * kPageBytes,
                  page_words, 0));
  auto* b_words = static_cast<std::uint32_t*>(b->data());
  FillRows<<<dim3(64, 1), 256>>>(b_words + rewind * row_bytes / 4, 0,
                                 row_bytes / 4, 4 * rows_per_page - rewind,
                                 0xbbu);
  Check(hipDeviceSynchronize(), "rewrite b");
  assert(WordsMatch(b->data(), rewind * row_bytes / 4, 0xaau));
  assert(WordsMatch(b_words + rewind * row_bytes / 4, 1024, 0xbbu));
  // Neither the writer nor the checkpoint saw that.
  assert(WordsMatch(a->data(), frontier * row_bytes / 4, 0xaau));
  assert(WordsMatch(parked[3]->data(), page_words, 0xaau, 3 * page_words));
  // The writer is given to another conversation: the checkpoint keeps the
  // pages, now cache memory, and the writer maps new ones as it writes.
  (void)a->Restore({}, rows);
  assert(a->SettleRemaps(&error));
  assert(a->backed_rows() == 0 && a->shared_pages() == 0);
  // Pages 0 and 1 are still mapped by the second session; 2, 3 and 4 only
  // by the checkpoint.
  assert(ledger.Totals().mapped_bytes == mapped);
  assert(ledger.Totals().lendable_bytes ==
         before.lendable_bytes + 3 * kPageBytes);
  assert(a->Ensure(frontier, &error));
  FillRows<<<dim3(64, 1), 256>>>(a_words, 0, row_bytes / 4, frontier, 0xccu);
  Check(hipDeviceSynchronize(), "refill a");
  for (std::size_t p = 0; p < 4; ++p)
    assert(WordsMatch(parked[p]->data(), page_words, 0xaau, p * page_words));
  // The first session restores the checkpoint into its new pages: it takes
  // the shared pages back, and its own pages there return to the pool.
  adopted = a->Restore(plan, frontier);
  assert(a->SettleRemaps(&error));
  assert(adopted[0] && adopted[3] && !adopted[4]);
  assert(WordsMatch(a->data(), 4 * page_words, 0xaau));
  // The checkpoint is evicted: pages no session maps return to the pool,
  // the others stay with their sessions, and the park addresses go at the
  // next sweep.
  const auto stats = park->Stats();
  assert(stats.parked == 5);
  parked.clear();
  plan.clear();
  assert(park->Stats().parked == 0 && park->Stats().stale >= 5);
  assert(ledger.Totals().lendable_bytes == before.lendable_bytes);
  assert(park->Sweep() >= 5 && park->Stats().stale == 0);
  assert(WordsMatch(a->data(), 4 * page_words, 0xaau));
  assert(WordsMatch(b->data(), rewind * row_bytes / 4, 0xaau));
  // A page with a single user left is that user's again: writes go in place.
  const auto mapped_now = ledger.Totals().mapped_bytes;
  assert(a->shared_pages() == 2 && b->shared_pages() == 2);
  assert(a->PrepareWrite(2 * rows_per_page, 3 * rows_per_page,
                         2 * rows_per_page, &error));
  assert(ledger.Totals().mapped_bytes == mapped_now);
  // Pages 0 and 1 are still shared by the two sessions: rewriting page 0 in
  // place copies it, rows included.
  assert(b->PrepareWrite(0, rows_per_page, rows_per_page, &error));
  assert(b->SettleRemaps(&error));
  assert(ledger.Totals().mapped_bytes == mapped_now + kPageBytes);
  assert(a->shared_pages() == 1 && b->shared_pages() == 1);
  assert(WordsMatch(a->data(), 4 * page_words, 0xaau));
  assert(WordsMatch(b->data(), rewind * row_bytes / 4, 0xaau));
  a.reset();
  b.reset();
  const auto after = ledger.Totals();
  assert(after.mapped_bytes == before.mapped_bytes);
  assert(after.lendable_bytes == before.lendable_bytes);
  assert(pool->Stats().pages_in_use == used_before);
}

// A region of several segments shares pages when each segment starts on a
// page: every segment's rows are shared, restored and rewritten as one
// segment's are.
void SharedSegments() {
  auto park = gufo::mem::PagePark::ForDevice(0);
  if (!park)
    return;
  auto pool = PagePool::ForDevice(0);
  auto& ledger = StateLedger::Process();
  const auto before = ledger.Totals();
  const auto used_before = pool->Stats().pages_in_use;
  const std::size_t row_bytes = 1024, rows_per_page = kPageBytes / row_bytes;
  const std::size_t segments = 3, segment_pages = 4;
  const std::size_t rows = segment_pages * rows_per_page;
  std::string error;
  const gufo::mem::RegionShape shape{
      .row_bytes = row_bytes, .rows = rows, .segments = segments};
  auto a = Region::Create(shape, &error);
  auto b = Region::Create(shape, &error);
  assert(a && b);
  const std::size_t stride = rows * row_bytes;
  const std::size_t frontier = 2 * rows_per_page + rows_per_page / 2;
  assert(a->Ensure(frontier, &error));
  FillRows<<<dim3(64, segments), 256>>>(static_cast<std::uint32_t*>(a->data()),
                                        stride / 4, row_bytes / 4, frontier,
                                        0xaau);
  Check(hipDeviceSynchronize(), "fill segments");
  assert(RowsMatch(*a, frontier, 0xaau));
  const auto mapped = ledger.Totals().mapped_bytes;
  std::vector<std::shared_ptr<gufo::mem::ParkedPage>> parked;
  std::vector<std::shared_ptr<gufo::mem::SharedPage>> plan(segments *
                                                           segment_pages);
  for (std::size_t s = 0; s < segments; ++s)
    for (std::size_t p = 0; p < 3; ++p) {
      const std::size_t page = s * segment_pages + p;
      auto shared = a->Share(page, s * stride + std::min((p + 1) * kPageBytes,
                                                         frontier * row_bytes));
      assert(shared);
      parked.push_back(park->Park(shared));
      assert(parked.back());
      if (p < 2)
        plan[page] = shared;
    }
  assert(a->shared_pages() == 9 && ledger.Totals().mapped_bytes == mapped);
  // Appending past the shared rows writes in place in every segment.
  assert(a->PrepareWrite(frontier, frontier + 16, frontier, &error));
  assert(a->shared_pages() == 9);
  // A second region restores the whole pages of every segment by mapping.
  const auto adopted = b->Restore(plan, frontier);
  assert(b->SettleRemaps(&error));
  for (std::size_t page = 0; page < adopted.size(); ++page)
    assert(adopted[page] == (page % segment_pages < 2));
  assert(b->backed_rows() == 2 * rows_per_page && b->shared_pages() == 6);
  assert(ledger.Totals().mapped_bytes == mapped);
  assert(b->Ensure(frontier, &error));
  assert(ledger.Totals().mapped_bytes == mapped + segments * kPageBytes);
  assert(RowsMatch(*b, 2 * rows_per_page, 0xaau));
  // It rewinds into the second page: a copy of that page in every segment,
  // whose first rows it keeps; the first pages stay shared.
  const std::size_t rewind = rows_per_page + 100;
  // One page per segment is copied, back to back: every copy must reach its
  // own new page, which it does only after the barrier.
  assert(b->PrepareWrite(rewind, frontier, rewind, &error));
  assert(b->SettleRemaps(&error));
  assert(b->shared_pages() == 3);
  assert(ledger.Totals().mapped_bytes == mapped + 2 * segments * kPageBytes);
  assert(RowsMatch(*b, rewind, 0xaau));
  // Its new rows reach neither the writer nor the checkpoint.
  FillRows<<<dim3(64, segments), 256>>>(
      static_cast<std::uint32_t*>(b->data()) + rewind * row_bytes / 4,
      stride / 4, row_bytes / 4, frontier - rewind, 0xbbu);
  Check(hipDeviceSynchronize(), "rewrite segments");
  assert(RowsMatch(*b, rewind, 0xaau) && RowsMatch(*a, frontier, 0xaau));
  for (std::size_t s = 0; s < segments; ++s)
    assert(WordsMatch(parked[s * 3 + 1]->data(), kPageBytes / 4, 0xaau,
                      s * 1000003u + kPageBytes / 4));
  // A restore that maps nothing releases what others share in the rows it
  // brings, in every segment.
  (void)a->Restore({}, frontier);
  assert(a->SettleRemaps(&error));
  assert(a->shared_pages() == 0 && a->backed_rows() == 0);
  // Segments that do not start on a page share nothing.
  auto c =
      Region::Create({.row_bytes = 1000, .rows = 5000, .segments = 2}, &error);
  assert(c && c->Ensure(5000, &error) && !c->Share(0, 1000));
  parked.clear();
  plan.clear();
  (void)park->Sweep();
  a.reset();
  b.reset();
  c.reset();
  const auto after = ledger.Totals();
  assert(after.mapped_bytes == before.mapped_bytes);
  assert(after.lendable_bytes == before.lendable_bytes);
  assert(pool->Stats().pages_in_use == used_before);
}

}  // namespace

int main() {
  Check(hipSetDevice(0), "device");
  const bool demand = PagePool::ForDevice(0) != nullptr;
  if (!demand)
    EagerClearBeforeFirstWrite();
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
  SharedPages();
  SharedSegments();
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
