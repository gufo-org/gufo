#ifndef GUFO_MEM_LEDGER_HPP_
#define GUFO_MEM_LEDGER_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace gufo::mem {

/// One physical page of demand-mapped state. A single size keeps the pool
/// free of external fragmentation, and 2 MiB pages read as fast as hipMalloc
/// memory (64 KiB pages lose a third of random-gather bandwidth on gfx1151).
inline constexpr std::size_t kPageBytes = std::size_t{2} << 20;

[[nodiscard]] constexpr std::size_t RoundUpToPage(std::size_t bytes) noexcept {
  return (bytes + kPageBytes - 1) / kPageBytes * kPageBytes;
}

struct StateMemoryTotals {
  /// Bytes promised to state regions at their full size, at most the
  /// promise cap.
  std::size_t reserved_bytes{0};
  /// Bytes of those regions backed by physical pages.
  std::size_t mapped_bytes{0};
  /// Physical pages created ahead of use and not mapped yet.
  std::size_t pooled_bytes{0};
  /// Promised bytes the host has not committed yet: reserved minus mapped
  /// minus pooled, less what the RAM cache borrowed; never negative.
  std::size_t outstanding_bytes{0};
  /// RAM-cache pages that are state pool pages, which the cache lends back
  /// to regions on demand.
  std::size_t lendable_bytes{0};
  /// The part of them beyond the cache's own grant, which only exists
  /// because it borrowed promised but unwritten state; at most the
  /// uncommitted promise.
  std::size_t borrowed_bytes{0};
};

/// Process-wide account of demand-mapped state, header-only so a consumer
/// (the host budget, admission queries) needs no link to this package. Sessions
/// promise their full context when created, as before, but only the rows they
/// write take physical pages. Every budget sized from free memory (the RAM
/// cache grant, admission, deferred scratch) subtracts the outstanding bytes,
/// so it sees the same memory it would have seen with the state committed up
/// front: the pages a session may still grow into are never handed to another
/// owner, and the unwritten rows stay free for the kernel instead.
///
/// The RAM cache is the one exception. Its payloads live in pool pages, and
/// it may grow past its own grant into the promised but unwritten state:
/// when a region needs a page and the host is at its floor, the pool takes a
/// page from the cache (evicting what it held) instead of creating one. The
/// borrowed bytes already stand for that promise, so they are not
/// outstanding a second time: every other budget sees what it saw before.
class StateLedger {
public:
  [[nodiscard]] static StateLedger& Process() noexcept {
    static StateLedger ledger;
    return ledger;
  }

  void AddReserved(std::size_t bytes) noexcept { Add(reserved_, bytes); }
  void RemoveReserved(std::size_t bytes) noexcept { Sub(reserved_, bytes); }
  void AddMapped(std::size_t bytes) noexcept { Add(mapped_, bytes); }
  void RemoveMapped(std::size_t bytes) noexcept { Sub(mapped_, bytes); }
  void AddPooled(std::size_t bytes) noexcept { Add(pooled_, bytes); }
  void RemovePooled(std::size_t bytes) noexcept { Sub(pooled_, bytes); }
  void AddLendable(std::size_t bytes) noexcept { Add(lendable_, bytes); }
  void RemoveLendable(std::size_t bytes) noexcept { Sub(lendable_, bytes); }
  /// The RAM cache's own grant: lendable bytes up to it are not borrowed.
  void AddLenderGrant(std::size_t bytes) noexcept { Add(lender_grant_, bytes); }
  void RemoveLenderGrant(std::size_t bytes) noexcept {
    Sub(lender_grant_, bytes);
  }
  [[nodiscard]] std::size_t LenderGrantBytes() const noexcept {
    return lender_grant_.load(std::memory_order_relaxed);
  }
  /// Set once a pool can lend RAM-cache pages to regions; only then may
  /// the cache borrow outstanding state.
  void EnableLending() noexcept {
    lending_.store(true, std::memory_order_relaxed);
  }
  [[nodiscard]] bool LendingEnabled() const noexcept {
    return lending_.load(std::memory_order_relaxed);
  }

  /// Caps the bytes sessions may charge (StateAccount), and the promise
  /// every other budget sees, at `bytes`: set when more sessions are
  /// configured at full context than fit (#715). Zero removes the cap.
  void SetPromiseCap(std::size_t bytes) noexcept {
    cap_.store(bytes, std::memory_order_relaxed);
  }
  [[nodiscard]] std::size_t PromiseCap() const noexcept {
    return cap_.load(std::memory_order_relaxed);
  }
  /// Charges `bytes` to the sessions' total unless that passes the cap.
  [[nodiscard]] bool TryCharge(std::size_t bytes) noexcept {
    const std::size_t cap = cap_.load(std::memory_order_relaxed);
    std::size_t current = charged_.load(std::memory_order_relaxed);
    do {
      if (cap != 0 && (current > cap || bytes > cap - current))
        return false;
    } while (!charged_.compare_exchange_weak(current, current + bytes,
                                             std::memory_order_relaxed));
    return true;
  }
  void Uncharge(std::size_t bytes) noexcept { Sub(charged_, bytes); }
  [[nodiscard]] std::size_t ChargedBytes() const noexcept {
    return charged_.load(std::memory_order_relaxed);
  }
  /// Bytes sessions may still charge; the maximum without a cap.
  [[nodiscard]] std::size_t ChargeableBytes() const noexcept {
    const std::size_t cap = cap_.load(std::memory_order_relaxed);
    if (cap == 0)
      return static_cast<std::size_t>(-1);
    const std::size_t charged = charged_.load(std::memory_order_relaxed);
    return cap > charged ? cap - charged : 0;
  }

  [[nodiscard]] StateMemoryTotals Totals() const noexcept {
    const std::size_t cap = cap_.load(std::memory_order_relaxed);
    const std::size_t reserved = reserved_.load(std::memory_order_relaxed);
    StateMemoryTotals totals{
        .reserved_bytes = cap != 0 ? std::min(reserved, cap) : reserved,
        .mapped_bytes = mapped_.load(std::memory_order_relaxed),
        .pooled_bytes = pooled_.load(std::memory_order_relaxed),
        .lendable_bytes = lendable_.load(std::memory_order_relaxed),
    };
    const std::size_t committed = totals.mapped_bytes + totals.pooled_bytes;
    const std::size_t promised = totals.reserved_bytes > committed
                                     ? totals.reserved_bytes - committed
                                     : 0;
    const std::size_t grant = lender_grant_.load(std::memory_order_relaxed);
    const std::size_t beyond =
        totals.lendable_bytes > grant ? totals.lendable_bytes - grant : 0;
    totals.borrowed_bytes = std::min(beyond, promised);
    totals.outstanding_bytes = promised - totals.borrowed_bytes;
    return totals;
  }
  /// Outstanding bytes the RAM cache may borrow: zero unless pages can be
  /// lent back to regions.
  [[nodiscard]] std::size_t BorrowableBytes() const noexcept {
    return LendingEnabled() ? OutstandingBytes() : 0;
  }
  [[nodiscard]] std::size_t OutstandingBytes() const noexcept {
    return Totals().outstanding_bytes;
  }
  /// Reserved bytes not mapped yet, pooled or not: what the pool may still
  /// create pages for without committing more than the regions promised.
  [[nodiscard]] std::size_t UnmappedBytes() const noexcept {
    const std::size_t cap = cap_.load(std::memory_order_relaxed);
    std::size_t reserved = reserved_.load(std::memory_order_relaxed);
    if (cap != 0)
      reserved = std::min(reserved, cap);
    const std::size_t mapped = mapped_.load(std::memory_order_relaxed);
    return reserved > mapped ? reserved - mapped : 0;
  }

private:
  static void Add(std::atomic<std::size_t>& value, std::size_t bytes) noexcept {
    value.fetch_add(bytes, std::memory_order_relaxed);
  }
  static void Sub(std::atomic<std::size_t>& value, std::size_t bytes) noexcept {
    value.fetch_sub(bytes, std::memory_order_relaxed);
  }
  std::atomic<std::size_t> reserved_{0};
  std::atomic<std::size_t> mapped_{0};
  std::atomic<std::size_t> pooled_{0};
  std::atomic<std::size_t> lendable_{0};
  std::atomic<std::size_t> lender_grant_{0};
  std::atomic<std::size_t> cap_{0};
  std::atomic<std::size_t> charged_{0};
  std::atomic<bool> lending_{false};
};

/// `available` less the state the host has promised but not committed.
[[nodiscard]] inline std::uint64_t LessOutstanding(
    std::uint64_t available) noexcept {
  const std::uint64_t outstanding = StateLedger::Process().OutstandingBytes();
  return available > outstanding ? available - outstanding : 0;
}

/// MemAvailable minus CmaFree from /proc/meminfo text, in bytes. Free CMA
/// pages count as available but hold only movable pages, never GPU memory.
[[nodiscard]] std::optional<std::uint64_t> ParseMeminfoAvailable(
    std::string_view meminfo);

/// MemFree minus CmaFree from /proc/meminfo text, in bytes.
[[nodiscard]] std::optional<std::uint64_t> ParseMeminfoFree(
    std::string_view meminfo);

/// Free host pages, without the page cache MemAvailable also counts:
/// MemFree minus CmaFree where /proc/meminfo exists, else
/// HostAvailableBytes(). A 2 MiB page made beyond them comes from reclaim
/// and compaction, and compaction that touches GPU-visible pages evicts the
/// process's GPU queues (measured: 2,335 compaction stalls and 4.6 s of
/// evictions with MemAvailable at 20 GB but 2-4 GB free). Serving creates
/// state pages only above the floor of these.
[[nodiscard]] std::uint64_t HostFreeBytes();

/// Bytes in free blocks of at least `kPageBytes` from /proc/buddyinfo text
/// (order 9 and up with 4 KiB pages), over all zones.
[[nodiscard]] std::optional<std::uint64_t> ParseBuddyinfoPageBlocks(
    std::string_view buddyinfo);

/// Free pages state memory leaves while serving: the background floor plus
/// 4 GiB. Disk-tier writes fill free pages with page cache (7-8 GB in a
/// round of four agents), and with 7-9 GB free the kernel's page migration
/// evicted the GPU queues for about 1 s per round (11-15 GB free: none).
[[nodiscard]] std::uint64_t HostFreeFloorBytes();

/// Whether `bytes` of 2 MiB pages can be made from free blocks as they are:
/// free pages stay above HostFreeFloorBytes() and free 2 MiB blocks cover
/// them with a margin. Free pages alone are not enough: fragmented, they still
/// need compaction (measured with 7-9 GB free: hundreds of compaction stalls
/// and GPU queue evictions). Without /proc/buddyinfo, free pages decide.
[[nodiscard]] bool CanCreatePagesWithoutReclaim(std::uint64_t bytes);

/// Host memory the kernel can hand out without reclaiming anything that
/// matters: MemAvailable minus CmaFree where /proc/meminfo exists, else the
/// platform's physical-available figure. Not reduced by outstanding state.
[[nodiscard]] std::uint64_t HostAvailableBytes();

/// Free host memory below which no 2 MiB page is created in the background.
/// Creating order-9 pages with a few GiB free forces direct compaction, and
/// every compaction pass that touches GPU-visible pages evicts all of the
/// process's GPU queues: prefill ran 10-100x slower for seconds at a time
/// with 3.3 GB free on a 125 GB host. One sixteenth of RAM, 4-8 GiB.
[[nodiscard]] std::uint64_t BackgroundFloorBytes(
    std::uint64_t total_bytes) noexcept;

/// BackgroundFloorBytes() of this host's physical memory.
[[nodiscard]] std::uint64_t HostFloorBytes();

/// Whether the GPU heap (`heap_total_bytes`, from hipMemGetInfo; on Linux
/// the GTT limit) reaches host memory less its background floor. State pages
/// the RAM cache borrows are charged to this heap; the pinned host pages it
/// otherwise uses are not. With a GTT limit below host memory the heap runs
/// out first, and borrowed pages are pages graphs and scratch allocations
/// cannot get (measured: 100 GiB heap on 128 GiB, Flash-Next graph
/// instantiation out of memory with 18 MB of heap left).
[[nodiscard]] bool HeapCoversHost(std::uint64_t heap_total_bytes,
                                  std::uint64_t host_total_bytes) noexcept;

}  // namespace gufo::mem

#endif  // GUFO_MEM_LEDGER_HPP_
