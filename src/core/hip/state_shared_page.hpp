#ifndef GUFO_CORE_HIP_STATE_SHARED_PAGE_HPP_
#define GUFO_CORE_HIP_STATE_SHARED_PAGE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/mem/ledger.hpp"
#include "src/mem/shared_rows.hpp"

namespace gufo::mem {

class PagePool;

/// One physical page with more than one user: the region that wrote its
/// rows, the RAM cache that retains them for a checkpoint, and regions that
/// restored the checkpoint by mapping the page. Rows are append-only, so the
/// users only read what they share; a region that must write shared rows
/// takes a private copy first (Region::PrepareWrite). The page returns to
/// the pool with its last user.
///
/// The state ledger counts the page once: as mapped while any region maps
/// it, else as cache memory (lendable) while the cache holds it.
class SharedPage {
public:
  /// `handle` is mapped by one region, which the ledger already counts.
  SharedPage(std::shared_ptr<PagePool> pool,
             hipMemGenericAllocationHandle_t handle);
  ~SharedPage();
  SharedPage(const SharedPage&) = delete;
  SharedPage& operator=(const SharedPage&) = delete;

  [[nodiscard]] hipMemGenericAllocationHandle_t handle() const noexcept {
    return handle_;
  }
  /// A region mapped or unmapped the page.
  void AddRegion() noexcept;
  void RemoveRegion() noexcept;
  /// The cache took or dropped its hold on the page.
  void AddHolder() noexcept;
  void RemoveHolder() noexcept;

private:
  std::shared_ptr<PagePool> pool_;
  hipMemGenericAllocationHandle_t handle_;
  std::mutex mutex_;
  std::size_t regions_{1};
  std::size_t holders_{0};
};

class PagePark;

/// A shared page at its address in the park range. Keeps the page and the
/// address until destroyed.
class ParkedPage {
public:
  ~ParkedPage();
  ParkedPage(const ParkedPage&) = delete;
  ParkedPage& operator=(const ParkedPage&) = delete;
  [[nodiscard]] std::byte* data() const noexcept { return data_; }
  [[nodiscard]] const std::shared_ptr<SharedPage>& page() const noexcept {
    return page_;
  }

private:
  friend class PagePark;
  ParkedPage(std::shared_ptr<PagePark> park, std::size_t slot, std::byte* data,
             std::shared_ptr<SharedPage> page)
      : park_(std::move(park)),
        slot_(slot),
        data_(data),
        page_(std::move(page)) {}
  std::shared_ptr<PagePark> park_;
  std::size_t slot_;
  std::byte* data_;
  std::shared_ptr<SharedPage> page_;
};

struct PageParkStats {
  /// Pages parked now, and in all.
  std::size_t parked{0};
  std::size_t parked_total{0};
  /// Addresses whose page left and that Sweep() has not unmapped yet.
  std::size_t stale{0};
  /// Address space reserved for the park, in pages.
  std::size_t reserved{0};
};

/// Where the RAM cache reads the session pages it retains: a device address
/// range of its own in which a shared page is mapped a second time. The
/// address stays valid while the session that wrote the page keeps running,
/// rewinds (it then copies the page it writes) or is given to another
/// conversation, so a checkpoint's rows cost no copy and no second page.
///
/// An address is used for another page only after the device has been idle
/// since its unmap (mem::RemapBarrier): until then the device would still
/// reach the page that left. Mapping never waits for the device; unmapping
/// does. A parked page that
/// is released keeps its mapping until Sweep(), which the owner calls only
/// where no GPU work is in flight (measured on gfx1151: one unmap call
/// beside running kernels waits for them to end, 0.02 s to several seconds;
/// idle it takes about 1 us per page).
class PagePark : public std::enable_shared_from_this<PagePark> {
public:
  /// The process's park for `device`; null where regions are not
  /// demand-mapped or the cache may not hold device pages
  /// (StateLedger::LendingEnabled()).
  [[nodiscard]] static std::shared_ptr<PagePark> ForDevice(int device);

  explicit PagePark(std::shared_ptr<PagePool> pool);
  ~PagePark();
  PagePark(const PagePark&) = delete;
  PagePark& operator=(const PagePark&) = delete;

  /// `page` at its park address, mapped there now if it was not. One page
  /// has one address however often it is parked. Null when the driver
  /// refuses the mapping or the address space is used up.
  [[nodiscard]] std::shared_ptr<ParkedPage> Park(
      const std::shared_ptr<SharedPage>& page);
  /// The parked page whose range holds `address`, or null.
  [[nodiscard]] std::shared_ptr<ParkedPage> Find(const void* address);
  /// Whether `address` lies in the park's address range.
  [[nodiscard]] bool Contains(const void* address) const;
  [[nodiscard]] PagePool& pool() const noexcept { return *pool_; }
  /// Unmaps the addresses released pages left behind and returns how many.
  /// Call only where no GPU work is in flight.
  std::size_t Sweep();
  [[nodiscard]] PageParkStats Stats() const;

  /// Address space taken at a time: 16 GiB.
  static constexpr std::size_t kChunkPages = 8192;
  /// At most 1 TiB of addresses; a park that is full parks nothing more
  /// until Sweep().
  static constexpr std::size_t kMaxChunks = 64;

private:
  friend class ParkedPage;
  enum class Slot : std::uint8_t { kClean, kLive, kStale };
  void Release(std::size_t slot, SharedPage* page) noexcept;
  [[nodiscard]] std::byte* Address(std::size_t slot) const noexcept {
    return chunks_[slot / kChunkPages] + slot % kChunkPages * kPageBytes;
  }

  std::shared_ptr<PagePool> pool_;
  mutable std::mutex mutex_;
  std::vector<std::byte*> chunks_;
  std::vector<Slot> slots_;
  std::vector<std::weak_ptr<ParkedPage>> parked_;
  /// Unmapped addresses, oldest first, with RemapBarrier::Stamp() of their
  /// unmap: one is used again only once it is clean (RemapBarrier::Clean),
  /// so a parked page is readable by the device as soon as it is parked.
  std::deque<std::pair<std::size_t, std::uint64_t>> clean_;
  std::vector<std::size_t> stale_;
  std::unordered_map<SharedPage*, std::size_t> by_page_;
  std::size_t next_{0};
  std::size_t live_{0};
  std::size_t total_{0};
};

/// mem::SharedRows on the current device's park: a row range within one page
/// of a region is shared by parking that page. Null where the device has no
/// park. The page pool installs it as the factory.
[[nodiscard]] std::shared_ptr<SharedRows> CreateParkedRows();

class Region;

/// What a restore takes from a resident checkpoint for one region: told
/// where the checkpoint keeps its rows, it maps every page of the region
/// that those rows fill completely (Region::Restore) and leaves the rest to
/// be loaded.
class RestorePlan {
public:
  /// A null `park` plans nothing: Apply() then only releases what others
  /// share among the rows the restore brings.
  RestorePlan(Region& region, PagePark* park);
  /// The checkpoint keeps the bytes that belong at `offset` of the region at
  /// `bytes`. Rows of one page follow each other.
  void Add(std::size_t offset, std::span<const std::byte> bytes);
  /// Region::Restore() for rows [0, rows) with the pages that are complete.
  /// Per page of the region, whether it now maps the checkpoint's page.
  [[nodiscard]] std::vector<bool> Apply(std::size_t rows);

private:
  Region& region_;
  PagePark* park_;
  std::shared_ptr<ParkedPage> parked_;
  std::vector<std::shared_ptr<SharedPage>> pages_;
  /// Bytes of each page that the checkpoint keeps where the region expects
  /// them, and pages with rows that are elsewhere.
  std::vector<std::size_t> covered_;
  std::vector<bool> mixed_;
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_SHARED_PAGE_HPP_
