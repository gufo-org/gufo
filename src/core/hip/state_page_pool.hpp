#ifndef GUFO_CORE_HIP_STATE_PAGE_POOL_HPP_
#define GUFO_CORE_HIP_STATE_PAGE_POOL_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/mem/ledger.hpp"

namespace gufo::mem {

class Region;

/// Something that holds pool pages it can give up: the RAM cache, whose
/// payloads live in pool pages. Lend() frees up to `pages` of them (evicting
/// what they held) and appends their handles; they stay mapped where the
/// lender keeps them, unused until TakeBack(). The pool maps them into
/// regions as they are, so lending a page is a map, never an unmap: no
/// device drain and no page creation under memory pressure.
class PageLender {
public:
  virtual ~PageLender() = default;
  virtual std::size_t Lend(
      std::size_t pages, std::vector<hipMemGenericAllocationHandle_t>& out) = 0;
  /// A lent page a region no longer maps. False when the lender cannot take
  /// it (it is shutting down); the pool then keeps the page.
  virtual bool TakeBack(hipMemGenericAllocationHandle_t handle) = 0;
};

struct PagePoolStats {
  std::size_t free_pages{0};
  std::size_t pages_in_use{0};
  /// Pages created on a caller's thread because the pool was empty.
  std::size_t created_on_demand{0};
  /// Pages created by the pool thread.
  std::size_t created_ahead{0};
  /// Background creations skipped because the host was below its floor.
  std::size_t skipped_below_floor{0};
  /// Pages the RAM cache lent to regions, and pages back with it.
  std::size_t lent{0};
  std::size_t returned{0};
  /// Copies and clears queued, or taken by a SettleRemaps() that has not
  /// finished them.
  std::size_t fixups{0};
  /// Scratch aliases mapped as copy sources, until a SettleRemaps() unmaps
  /// them, and the source pages held until then.
  std::size_t aliases{0};
  std::size_t held_sources{0};
};

/// Physical 2 MiB pages of one device, created ahead of use and never freed
/// while serving. hipMemCreate costs 50-80 us idle and 150-310 ms under
/// memory pressure; mapping a pooled page costs about 8 us. Releasing a
/// mapping waits for all in-flight GPU work, so pages go back to the pool,
/// not to the driver.
///
/// One thread per pool creates pages while free host pages stay above the
/// background floor, and maps rows ahead of each region's frontier, so the
/// scheduler normally finds its rows mapped and makes no HIP call. Below the
/// floor it takes pages the RAM cache lends before it creates any: a new
/// page then comes from reclaim and compaction. The pool
/// never holds more free pages than regions have promised but not mapped:
/// with demand mapping, the host never commits more than it did with all
/// state allocated up front.
class PagePool {
public:
  /// The process's pool for `device`, or null when the device has no
  /// virtual memory management (regions then allocate their full size).
  [[nodiscard]] static std::shared_ptr<PagePool> ForDevice(int device);

  /// `may_create(bytes)`: pages can be made from free 2 MiB blocks without
  /// reclaim. `host_available` reads MemAvailable.
  PagePool(int device, std::function<bool(std::uint64_t)> may_create,
           std::function<std::uint64_t()> host_available,
           std::uint64_t floor_bytes);
  ~PagePool();
  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;

  /// A page from the pool, else a new one created here. False when the
  /// driver has no memory left. `*recycled` is set when the page held
  /// another region's rows: new pages are zero, recycled ones are not.
  [[nodiscard]] bool Acquire(hipMemGenericAllocationHandle_t* handle,
                             bool* recycled);
  /// A stream of the pool's own for zeroing recycled pages, created once.
  [[nodiscard]] hipStream_t ZeroStream();
  /// Returns a page that is no longer mapped.
  void Release(hipMemGenericAllocationHandle_t handle);
  /// Maps `rows` of `region` on the pool thread. Requests for the same
  /// region coalesce; the larger target wins.
  void MapAhead(const std::shared_ptr<Region>& region, std::size_t rows);
  /// Creates up to `bytes` more pages ahead of use, while the host stays
  /// above the floor. Called when regions are reserved, so the expected working
  /// set is created at load, before memory fragments.
  void Warm(std::size_t bytes);
  /// Blocks until queued map-ahead work has finished. For tests.
  void Drain();
  /// Stops the pool thread; pages stay where they are. At process exit.
  void Stop();

  /// Whether pages can back a CPU view (exported as device-memory files),
  /// which RAM-cache payloads need, and the GPU heap covers host memory
  /// (mem::HeapCoversHost), so the cache may borrow them.
  [[nodiscard]] bool exportable() const noexcept { return exportable_; }
  /// A new page for a lender, never one of the free pages (the regions'
  /// working set). Made only from free 2 MiB blocks above the free floor,
  /// or, with `may_reclaim` (while loading, when a compaction stall costs
  /// nothing), while MemAvailable stays above the background floor.
  [[nodiscard]] bool AcquireForLender(hipMemGenericAllocationHandle_t* handle,
                                      bool may_reclaim);
  /// Takes back a page that is mapped nowhere (any more) and was not
  /// acquired through Acquire(); it is cleared before a region uses it.
  void Recycle(hipMemGenericAllocationHandle_t handle);
  /// Registers a lender; its pages may be lent to regions. A call already in
  /// progress keeps the lender alive; pages it lent stay with the pool.
  void AddLender(const std::shared_ptr<PageLender>& lender);
  void RemoveLender(const PageLender* lender);
  /// Takes up to `pages` pages from lenders into the free pages now. The
  /// pool thread does this while the host is below its floor; callers must
  /// not hold a lender's locks.
  std::size_t Borrow(std::size_t pages);
  /// Records that `handle` belongs to `lender`: when a region releases it,
  /// it returns there instead of to the free pages.
  void SetHome(hipMemGenericAllocationHandle_t handle,
               const std::shared_ptr<PageLender>& lender);
  /// Forgets the lender of a page about to be released to the driver, so a
  /// later page with the same handle value never goes there.
  void ClearHome(hipMemGenericAllocationHandle_t handle);

  [[nodiscard]] const hipMemAllocationProp& prop() const noexcept {
    return prop_;
  }
  [[nodiscard]] const hipMemAccessDesc& access() const noexcept {
    return access_;
  }
  [[nodiscard]] PagePoolStats Stats() const;

  /// Gives up to `bytes` of free pages back to the driver, and so to the
  /// host, and returns the bytes released; the pool makes pages again as
  /// regions need them. Releasing waits for the device: call it only while
  /// no GPU work is in flight.
  std::size_t ReleaseFree(std::size_t bytes);

  // Remaps and what must follow them. Memory mapped at an address that
  // just lost a mapping is not reachable there until a barrier flushed the
  // translations (RemapBarrier): a page that replaces another is filled or
  // cleared only after the barrier. Regions queue that work here;
  // SettleRemaps() runs the barrier once for everything queued, then the work.

  /// Maps `handle` a second time, at a scratch address, as the source of a
  /// copy queued with QueueCopy(), or given back with ReturnAlias(). Null
  /// when no scratch address is free or the driver refuses. SettleRemaps()
  /// unmaps an alias once its copy ran, was dropped or it came back; one
  /// handed out and neither queued nor given back stays mapped.
  [[nodiscard]] std::byte* AliasForCopy(hipMemGenericAllocationHandle_t handle);
  /// Gives back an alias no copy will read; `keep` holds the page it maps
  /// until SettleRemaps() or UnmapDoneAliases() unmaps it. Ignored for an
  /// address that is not a handed-out alias.
  void ReturnAlias(const std::byte* alias, std::shared_ptr<void> keep);
  /// Unmaps the aliases given back or of dropped copies now, without a
  /// barrier: the device never read them. Call it within this thread's
  /// RemapBarrier::Hold(): an unmap waits for device work in flight.
  void UnmapDoneAliases();
  /// After the barrier: copy one page from `source` (an AliasForCopy()
  /// address; `keep` holds its page until then) to `destination`.
  void QueueCopy(std::byte* destination, const std::byte* source,
                 std::shared_ptr<void> keep);
  /// After the barrier: clear `bytes` at `destination`.
  void QueueClear(std::byte* destination, std::size_t bytes);
  /// Forgets the queued copies and clears into [begin, end), after a
  /// SettleRemaps() that runs some of them: a region calls it for the pages
  /// it unmaps. True when it waited for that call, which may have ended the
  /// caller's RemapBarrier::Hold().
  [[nodiscard]] bool DropFixups(const std::byte* begin, const std::byte* end);
  /// Runs the barrier when a remap waits for one (RemapBarrier::Pending())
  /// or work is queued, then the queued work, and waits for it, as for the
  /// work another thread's call runs now, which may be this thread's. False
  /// when the barrier failed, and then no queued work runs and it waits for
  /// the next call, or when a copy or a clear it ran failed, and then that
  /// work is given up. Call it after a batch of
  /// Region::PrepareWrite() or Region::Restore() calls and before the device
  /// uses those rows, on the thread that runs the model, between its
  /// launches. With no remap pending, no work queued or running in any
  /// thread and no alias left to unmap, it costs two atomic reads.
  [[nodiscard]] bool SettleRemaps();
  static constexpr std::size_t kScratchPages = 64;

  /// Free pages kept for map-ahead and first writes.
  static constexpr std::size_t kLowPages = 256;
  static constexpr std::size_t kHighPages = 512;
  /// The working set created at load: 8 GiB, about 200 ms.
  static constexpr std::size_t kWarmCapBytes = std::size_t{8} << 30;
  /// Pages taken from lenders per pass: 128 MiB, a bounded eviction.
  static constexpr std::size_t kBorrowBatchPages = 64;

private:
  struct Job {
    std::weak_ptr<Region> region;
    std::size_t rows{0};
  };
  struct Fixup {
    std::byte* destination;
    const std::byte* source;  // Null: clear.
    std::size_t bytes;
    std::shared_ptr<void> keep;
  };
  /// A scratch address: handed out by AliasForCopy(), read by a queued copy,
  /// or done (its copy ran or was dropped, or it came back) and waiting for
  /// SettleRemaps() to unmap it, with the page it maps held until then.
  struct Alias {
    enum class State : std::uint8_t { kFree, kHeld, kQueued, kDone };
    State state{State::kFree};
    std::shared_ptr<void> keep;
  };
  void Run();
  [[nodiscard]] bool CreatePage(hipMemGenericAllocationHandle_t* handle);
  [[nodiscard]] std::size_t TargetFreePages();
  /// Borrows when fewer than `pages` are free and the host is near its
  /// floor. Pool thread only, without the pool lock.
  void BorrowBelowFloor(std::size_t pages);
  /// Sends a page with a home back to it. Called without the pool lock.
  [[nodiscard]] bool ReturnHome(hipMemGenericAllocationHandle_t handle);
  /// Stores unfinished_, under fixup_mutex_.
  void CountFixups();
  /// The scratch slot of `address`, kScratchPages for any other address.
  [[nodiscard]] std::size_t SlotOf(const std::byte* address) const;
  /// The alias at `address`, or null; under fixup_mutex_.
  [[nodiscard]] Alias* AliasAt(const std::byte* address);
  /// Marks an alias done, holding `keep`; under fixup_mutex_.
  static void AliasDone(Alias& alias, std::shared_ptr<void> keep);
  /// Unmaps the aliases marked in `unmap` at `scratch`, one call per run of
  /// neighbours.
  static void UnmapAliases(std::byte* scratch,
                           const std::bitset<kScratchPages>& unmap);
  /// The done aliases; under fixup_mutex_.
  [[nodiscard]] std::bitset<kScratchPages> DoneAliases() const;
  /// Frees the slots marked in `unmapped`, moving the pages they held into
  /// `sources`; under fixup_mutex_. A failed unmap is ignored, as for every
  /// other unmap of the pool, the regions and the park.
  void FreeAliases(const std::bitset<kScratchPages>& unmapped,
                   std::vector<std::shared_ptr<void>>& sources);

  int device_;
  std::function<bool(std::uint64_t)> may_create_;
  std::function<std::uint64_t()> host_available_;
  std::uint64_t floor_bytes_;
  hipMemAllocationProp prop_{};
  hipMemAccessDesc access_{};

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable idle_;
  std::vector<hipMemGenericAllocationHandle_t> free_;
  /// Released by a region; handed out before new pages.
  std::vector<hipMemGenericAllocationHandle_t> recycled_;
  hipStream_t zero_stream_{nullptr};
  /// One SettleRemaps() at a time; never held with mutex_.
  std::mutex settle_mutex_;
  mutable std::mutex fixup_mutex_;
  std::vector<Fixup> fixups_;
  /// The fixups a SettleRemaps() took from fixups_ and has neither finished
  /// nor queued again. That call changes it under fixup_mutex_; others read
  /// it under fixup_mutex_.
  std::vector<Fixup> taken_;
  /// fixups_, taken_ and the done aliases, stored under fixup_mutex_
  /// (CountFixups()): zero only when SettleRemaps() has no copy or clear to
  /// finish and no alias to unmap. The fast paths of SettleRemaps() and
  /// DropFixups() read it.
  std::atomic<std::size_t> unfinished_{0};
  std::byte* scratch_{nullptr};
  /// Under fixup_mutex_: the scratch addresses.
  std::array<Alias, kScratchPages> aliases_;
  std::unordered_map<const Region*, Job> jobs_;
  bool exportable_{false};
  /// Lenders, and the lender each lent page belongs to. lender_mutex_ is
  /// never held with mutex_ or during a call into a lender.
  std::mutex lender_mutex_;
  std::vector<std::weak_ptr<PageLender>> lenders_;
  std::unordered_map<hipMemGenericAllocationHandle_t, std::weak_ptr<PageLender>>
      homes_;
  std::size_t lent_{0};
  std::size_t returned_{0};
  std::size_t warm_bytes_{0};
  std::size_t in_use_{0};
  std::size_t created_on_demand_{0};
  std::size_t created_ahead_{0};
  std::size_t skipped_below_floor_{0};
  bool refilling_{false};
  bool busy_{false};
  bool stop_{false};
  std::thread thread_;
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_PAGE_POOL_HPP_
