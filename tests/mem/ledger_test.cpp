// CPU checks of the state ledger and host policy (src/mem).
#include "src/mem/ledger.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/mem/account.hpp"

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

void Fit() {
  using gufo::mem::FitSessions;
  using Kind = gufo::mem::SessionFit::Kind;
  // 4 sessions of 10 (8 demand-mapped) in 25: 2 fit at full context;
  // the 4 share 2 * 10 - 4 * 2 = 12, at least one full session's 8.
  auto fit = FitSessions(4, 10, 8, 25);
  assert(fit.kind == Kind::kShared && fit.full_sessions == 2 &&
         fit.cap_bytes == 12);
  assert(FitSessions(2, 10, 8, 25).kind == Kind::kFits);
  // Nothing demand-mapped, or no session at full context: refused.
  assert(FitSessions(4, 10, 0, 25).kind == Kind::kRefused);
  assert(FitSessions(4, 10, 8, 9).kind == Kind::kRefused);
  // Fixed state of 9 sessions (18) leaves 2 of 20 held: less than one
  // full session's demand-mapped state.
  assert(FitSessions(9, 10, 8, 25).kind == Kind::kRefused);
}

void CapAndAccounts() {
  using gufo::mem::RegionPageBytes;
  using gufo::mem::RegionShape;
  using gufo::mem::StateAccount;
  using gufo::mem::StateReservation;
  auto& ledger = StateLedger::Process();
  // Page union of segments: two 1 MiB segments share one page.
  const RegionShape pair{.row_bytes = 1 << 10, .rows = 1024, .segments = 2};
  assert(RegionPageBytes(pair, 1024) == kPageBytes);
  assert(RegionPageBytes(pair, 0) == 0);
  // One row in each of two 4 MiB segments: the first page of each.
  const RegionShape wide{.row_bytes = 1 << 12, .rows = 1024, .segments = 2};
  assert(RegionPageBytes(wide, 1) == 2 * kPageBytes);
  assert(RegionPageBytes(wide, 1024) == 4 * kPageBytes);

  // Without a cap every charge succeeds, as before accounts existed.
  const RegionShape kv{.row_bytes = 1 << 10, .rows = 64 * 1024};  // 64 MiB
  {
    StateAccount account(/*margin_tokens=*/1024);
    std::vector<std::size_t> trims;
    account.AddRegion(kv, 1, [&](std::size_t rows) { trims.push_back(rows); });
    assert(account.full_bytes() == 32 * kPageBytes);
    assert(account.BytesFor(0) == 0);
    // 1 token + 1024 margin rows of 1 KiB: one page.
    assert(account.BytesFor(1) == kPageBytes);
    assert(account.Claim(64 * 1024));
    assert(account.charged_bytes() == 32 * kPageBytes);
    assert(ledger.ChargedBytes() == 32 * kPageBytes);
    assert(account.SetNeed(0));
    assert(ledger.ChargedBytes() == 0);
    // Mapped pages stay charged past the need until trimmed.
    assert(account.OnMap(3 * kPageBytes));
    assert(account.charged_bytes() == 3 * kPageBytes);
    assert(account.tail_bytes() == 3 * kPageBytes);
    assert(account.Claim(1));
    assert(account.tail_bytes() == 2 * kPageBytes);
    // TrimTo asks each region for its rows: 1 token + margin.
    account.TrimTo(1);
    assert(trims.size() == 1 && trims[0] == 1025);
    account.OnUnmap(2 * kPageBytes);  // what the region would report
    assert(account.charged_bytes() == kPageBytes);
  }
  assert(ledger.ChargedBytes() == 0);

  // With a cap, sessions together never charge more than it.
  ledger.SetPromiseCap(40 * kPageBytes);
  ledger.AddReserved(64 * kPageBytes);
  assert(ledger.Totals().reserved_bytes == 40 * kPageBytes);
  assert(ledger.UnmappedBytes() == 40 * kPageBytes);
  {
    StateAccount a, b;
    a.AddRegion(kv, 1, {});
    b.AddRegion(kv, 1, {});
    assert(a.Claim(64 * 1024));   // 32 pages
    assert(!b.Claim(32 * 1024));  // 16 more pages: over the cap
    assert(b.need_tokens() == 0);
    assert(b.Claim(16 * 1024));  // 8 pages fit
    assert(ledger.ChargeableBytes() == 0);
    assert(!b.OnMap(9 * kPageBytes));  // past its need and the cap
    assert(b.OnMap(8 * kPageBytes));   // within its need: no new charge
    // A reservation taken ahead of the session: the claim draws on it.
    assert(a.SetNeed(32 * 1024));
    assert(ledger.ChargeableBytes() == 16 * kPageBytes);
    StateReservation reservation(16 * kPageBytes);
    assert(reservation.ok());
    assert(ledger.ChargeableBytes() == 0);
    assert(!StateReservation(kPageBytes).ok());
    assert(a.Claim(64 * 1024, &reservation));
    assert(reservation.bytes() == 0);
    // A failed claim returns what it took.
    assert(a.SetNeed(48 * 1024));  // releases 8 pages
    assert(b.Claim(24 * 1024));    // takes 4 of them
    StateReservation some(2 * kPageBytes);
    assert(!a.Claim(64 * 1024, &some));  // needs 8: 2 + 2 left
    assert(some.bytes() == 2 * kPageBytes);
    assert(a.need_tokens() == 48 * 1024);
  }
  assert(ledger.ChargedBytes() == 0);
  ledger.RemoveReserved(64 * kPageBytes);
  ledger.SetPromiseCap(0);
}

}  // namespace

int main() {
  Fit();
  CapAndAccounts();
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
