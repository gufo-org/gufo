// CPU checks of the state ledger and host policy (src/mem).
#include "src/mem/ledger.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>

namespace {

using gufo::mem::BackgroundFloorBytes;
using gufo::mem::kPageBytes;
using gufo::mem::ParseMeminfoAvailable;
using gufo::mem::RoundUpToPage;
using gufo::mem::StateLedger;

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

void Totals() {
  StateLedger ledger;
  ledger.AddReserved(10 * kPageBytes);
  ledger.AddMapped(3 * kPageBytes);
  ledger.AddPooled(2 * kPageBytes);
  auto totals = ledger.Totals();
  assert(totals.reserved_bytes == 10 * kPageBytes);
  assert(totals.outstanding_bytes == 5 * kPageBytes);
  assert(ledger.UnmappedBytes() == 7 * kPageBytes);
  // Pages pooled beyond the promise (a region was released) never make the
  // outstanding figure negative.
  ledger.RemoveReserved(10 * kPageBytes);
  ledger.RemoveMapped(3 * kPageBytes);
  totals = ledger.Totals();
  assert(totals.outstanding_bytes == 0);
  assert(ledger.UnmappedBytes() == 0);
}

void Meminfo() {
  assert(!ParseMeminfoAvailable("MemTotal: 100 kB\n"));
  assert(*ParseMeminfoAvailable("MemAvailable: 1000 kB\n") == 1000 * 1024);
  assert(*ParseMeminfoAvailable("MemAvailable: 1000 kB\nCmaFree: 400 kB\n") ==
         600 * 1024);
  assert(*ParseMeminfoAvailable("CmaFree: 4000 kB\nMemAvailable: 1000 kB\n") ==
         0);
  const char* both =
      "MemFree: 300 kB\nMemAvailable: 1000 kB\nCmaFree: 100 kB\n";
  assert(*gufo::mem::ParseMeminfoFree(both) == 200 * 1024);
  assert(*ParseMeminfoAvailable(both) == 900 * 1024);
  assert(!gufo::mem::ParseMeminfoFree("MemAvailable: 1000 kB\n"));
}

void Buddyinfo() {
  const char* text =
      "Node 0, zone      DMA      0      0      0      0      0      0      "
      "0      0      1      1      2 \n"
      "Node 0, zone   Normal   5971   9432   4910   5002   2927   3645   "
      "1580   1719   1871   3283  25372 \n";
  const std::uint64_t order9 = 4096ull << 9, order10 = 4096ull << 10;
  assert(*gufo::mem::ParseBuddyinfoPageBlocks(text) ==
         (1 + 3283) * order9 + (2 + 25372) * order10);
  assert(!gufo::mem::ParseBuddyinfoPageBlocks(""));
}

void Floor() {
  assert(BackgroundFloorBytes(0) == 4 * kGiB);
  assert(BackgroundFloorBytes(32 * kGiB) == 4 * kGiB);
  assert(BackgroundFloorBytes(96 * kGiB) == 6 * kGiB);
  assert(BackgroundFloorBytes(128 * kGiB) == 8 * kGiB);
  assert(BackgroundFloorBytes(512 * kGiB) == 8 * kGiB);
}

void HeapCover() {
  using gufo::mem::HeapCoversHost;
  // 128 GiB host, 125 GiB visible: GTT at 124 GiB lends, at 100 GiB not.
  const std::uint64_t host = 125 * kGiB;
  assert(HeapCoversHost(124 * kGiB, host));
  assert(HeapCoversHost(118 * kGiB, host));
  assert(!HeapCoversHost(100 * kGiB, host));
  assert(!HeapCoversHost(64 * kGiB, host));
  assert(!HeapCoversHost(124 * kGiB, 0));
}

void Rounding() {
  assert(RoundUpToPage(0) == 0);
  assert(RoundUpToPage(1) == kPageBytes);
  assert(RoundUpToPage(kPageBytes) == kPageBytes);
  assert(RoundUpToPage(kPageBytes + 1) == 2 * kPageBytes);
}

void Outstanding() {
  auto& ledger = StateLedger::Process();
  const auto base = ledger.OutstandingBytes();
  ledger.AddReserved(4 * kPageBytes);
  assert(gufo::mem::LessOutstanding(base + 10 * kPageBytes) == 6 * kPageBytes);
  assert(gufo::mem::LessOutstanding(1) == 0);
  ledger.RemoveReserved(4 * kPageBytes);
}

// Cache pages beyond the cache's own grant stand for promised state: they are
// not outstanding a second time, so every other budget sees what it saw
// before the cache borrowed. Within its grant they are ordinary usage.
void Lending() {
  StateLedger ledger;
  ledger.AddReserved(10 * kPageBytes);
  ledger.AddMapped(2 * kPageBytes);
  assert(ledger.BorrowableBytes() == 0);
  ledger.EnableLending();
  assert(ledger.BorrowableBytes() == 8 * kPageBytes);
  ledger.AddLenderGrant(3 * kPageBytes);
  ledger.AddLendable(2 * kPageBytes);
  auto totals = ledger.Totals();
  assert(totals.borrowed_bytes == 0);
  assert(totals.outstanding_bytes == 8 * kPageBytes);
  ledger.AddLendable(4 * kPageBytes);
  totals = ledger.Totals();
  assert(totals.borrowed_bytes == 3 * kPageBytes);
  assert(totals.outstanding_bytes == 5 * kPageBytes);
  // A borrowed page lent to a region: one page leaves the cache and is
  // mapped, so the promise and the borrowing shrink together.
  ledger.RemoveLendable(kPageBytes);
  ledger.AddMapped(kPageBytes);
  totals = ledger.Totals();
  assert(totals.borrowed_bytes == 2 * kPageBytes);
  assert(totals.outstanding_bytes == 5 * kPageBytes);
  // Borrowing never exceeds the promise.
  ledger.AddLendable(20 * kPageBytes);
  totals = ledger.Totals();
  assert(totals.borrowed_bytes == 7 * kPageBytes);
  assert(totals.outstanding_bytes == 0);
}

}  // namespace

int main() {
  Lending();
  Buddyinfo();
  Totals();
  Meminfo();
  Floor();
  HeapCover();
  Rounding();
  Outstanding();
  std::puts("mem_ledger_test: ok");
  return 0;
}
