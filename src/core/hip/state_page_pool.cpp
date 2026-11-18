#include "src/core/hip/state_page_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <utility>

#include "src/core/hip/state_lendable_range.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/core/platform/pages.hpp"

namespace gufo::mem {
namespace {

bool SupportsDemandMapping(int device, const hipMemAllocationProp& prop) {
  // The ledger holds outstanding state back from host budgets, which is
  // only right where device pages are host RAM.
  if (!platform::DevicePagesUseHostMemory())
    return false;
  int supported = 0;
  if (hipDeviceGetAttribute(&supported,
                            hipDeviceAttributeVirtualMemoryManagementSupported,
                            device) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  if (!supported)
    return false;
  std::size_t granularity = 0;
  if (hipMemGetAllocationGranularity(&granularity, &prop,
                                     hipMemAllocationGranularityMinimum) !=
      hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return granularity != 0 && kPageBytes % granularity == 0;
}

hipMemAllocationProp DeviceProp(int device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  return prop;
}

// Pages that can be exported as device-memory files, so the RAM cache can
// give its payload pages a CPU view. Probed once: the driver may refuse.
bool SupportsExport(hipMemAllocationProp prop) {
  if (!platform::CanMapDeviceFiles())
    return false;
  prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  hipMemGenericAllocationHandle_t handle{};
  if (hipMemCreate(&handle, kPageBytes, &prop, 0) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  int fd = -1;
  const bool exported =
      hipMemExportToShareableHandle(
          &fd, handle, hipMemHandleTypePosixFileDescriptor, 0) == hipSuccess;
  (void)hipGetLastError();
  if (exported)
    platform::CloseDeviceFile(fd);
  (void)hipMemRelease(handle);
  return exported;
}

}  // namespace

namespace {
std::mutex pools_mutex;
// Pools live for the whole process: their pages outlast every region, and
// releasing them during static destruction would race the HIP runtime's own
// teardown.
auto* const pools = new std::map<int, std::shared_ptr<PagePool>>();
}  // namespace

std::shared_ptr<PagePool> PagePool::ForDevice(int device) {
  static auto* unsupported = new std::map<int, bool>();
  std::lock_guard lock(pools_mutex);
  // Registered after the HIP runtime initialized, so it runs before the
  // runtime's teardown: a pool thread still releasing pages then would call
  // into a runtime that is being destroyed.
  static const bool stop_at_exit = [] {
    std::atexit([] {
      std::vector<std::shared_ptr<PagePool>> all;
      {
        std::lock_guard lock(pools_mutex);
        for (const auto& [id, pool] : *pools)
          all.push_back(pool);
      }
      for (const auto& pool : all)
        pool->Stop();
    });
    return true;
  }();
  (void)stop_at_exit;
  if (const auto it = pools->find(device); it != pools->end())
    return it->second;
  if (unsupported->contains(device))
    return nullptr;
  if (!SupportsDemandMapping(device, DeviceProp(device))) {
    (*unsupported)[device] = true;
    return nullptr;
  }
  const auto memory = platform::QueryHostMemory();
  auto pool = std::make_shared<PagePool>(
      device,
      [](std::uint64_t bytes) { return CanCreatePagesWithoutReclaim(bytes); },
      [] { return HostAvailableBytes(); },
      BackgroundFloorBytes(memory ? memory->total_bytes : 0));
  (*pools)[device] = pool;
  return pool;
}

PagePool::PagePool(int device, std::function<bool(std::uint64_t)> may_create,
                   std::function<std::uint64_t()> host_available,
                   std::uint64_t floor_bytes)
    : device_(device),
      may_create_(std::move(may_create)),
      host_available_(std::move(host_available)),
      floor_bytes_(floor_bytes),
      prop_(DeviceProp(device)) {
  // Lend only where borrowed pages do not compete for a GPU heap smaller
  // than host memory; otherwise the cache keeps pinned host pages, as
  // without state memory.
  std::size_t heap_free = 0, heap_total = 0;
  const auto memory = platform::QueryHostMemory();
  if (hipMemGetInfo(&heap_free, &heap_total) != hipSuccess)
    (void)hipGetLastError();
  exportable_ = HeapCoversHost(heap_total, memory ? memory->total_bytes : 0) &&
                SupportsExport(prop_);
  if (exportable_) {
    prop_.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    StateLedger::Process().EnableLending();
    SetLendableRangeFactory(&CreateVmmLendableRange);
  }
  access_.location = prop_.location;
  access_.flags = hipMemAccessFlagsProtReadWrite;
  thread_ = std::thread([this] { Run(); });
}

void PagePool::Stop() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable() && thread_.get_id() != std::this_thread::get_id())
    thread_.join();
}

PagePool::~PagePool() {
  Stop();
  for (auto* list : {&free_, &recycled_})
    for (const auto handle : *list) {
      (void)hipMemRelease(handle);
      StateLedger::Process().RemovePooled(kPageBytes);
    }
  if (zero_stream_)
    (void)hipStreamDestroy(zero_stream_);
}

bool PagePool::CreatePage(hipMemGenericAllocationHandle_t* handle) {
  if (hipMemCreate(handle, kPageBytes, &prop_, 0) == hipSuccess)
    return true;
  (void)hipGetLastError();
  return false;
}

bool PagePool::Acquire(hipMemGenericAllocationHandle_t* handle,
                       bool* recycled) {
  {
    std::lock_guard lock(mutex_);
    auto& list = recycled_.empty() ? free_ : recycled_;
    *recycled = !recycled_.empty();
    if (!list.empty()) {
      *handle = list.back();
      list.pop_back();
      StateLedger::Process().RemovePooled(kPageBytes);
      ++in_use_;
      if (free_.size() + recycled_.size() < kLowPages)
        wake_.notify_one();
      return true;
    }
    ++created_on_demand_;
  }
  wake_.notify_one();
  if (!CreatePage(handle))
    return false;
  std::lock_guard lock(mutex_);
  ++in_use_;
  return true;
}

void PagePool::Release(hipMemGenericAllocationHandle_t handle) {
  if (ReturnHome(handle)) {
    std::lock_guard lock(mutex_);
    --in_use_;
    return;
  }
  {
    std::lock_guard lock(mutex_);
    recycled_.push_back(handle);
    StateLedger::Process().AddPooled(kPageBytes);
    --in_use_;
  }
  wake_.notify_one();
}

bool PagePool::ReturnHome(hipMemGenericAllocationHandle_t handle) {
  std::shared_ptr<PageLender> home;
  {
    std::lock_guard lock(lender_mutex_);
    const auto it = homes_.find(handle);
    if (it == homes_.end())
      return false;
    home = it->second.lock();
    if (!home) {
      homes_.erase(it);
      return false;
    }
  }
  if (!home->TakeBack(handle))
    return false;
  std::lock_guard lock(mutex_);
  ++returned_;
  return true;
}

bool PagePool::AcquireForLender(hipMemGenericAllocationHandle_t* handle,
                                bool may_reclaim) {
  // Always a new page: the free pages are the regions' working set, and a
  // restore right after load that found them gone made its pages on the
  // scheduler thread (+40 ms for 3,143 Qwen 27B rows).
  if (!may_create_(kPageBytes) &&
      !(may_reclaim && host_available_() >= floor_bytes_ + kPageBytes))
    return false;
  return CreatePage(handle);
}

void PagePool::Recycle(hipMemGenericAllocationHandle_t handle) {
  {
    std::lock_guard lock(mutex_);
    recycled_.push_back(handle);
    StateLedger::Process().AddPooled(kPageBytes);
  }
  wake_.notify_one();
}

void PagePool::AddLender(const std::shared_ptr<PageLender>& lender) {
  std::lock_guard lock(lender_mutex_);
  lenders_.push_back(lender);
}

void PagePool::RemoveLender(const PageLender* lender) {
  std::lock_guard lock(lender_mutex_);
  std::erase_if(lenders_, [&](const auto& weak) {
    const auto strong = weak.lock();
    return !strong || strong.get() == lender;
  });
  std::erase_if(homes_, [&](const auto& entry) {
    const auto strong = entry.second.lock();
    return !strong || strong.get() == lender;
  });
}

void PagePool::SetHome(hipMemGenericAllocationHandle_t handle,
                       const std::shared_ptr<PageLender>& lender) {
  std::lock_guard lock(lender_mutex_);
  homes_[handle] = lender;
}

void PagePool::ClearHome(hipMemGenericAllocationHandle_t handle) {
  std::lock_guard lock(lender_mutex_);
  homes_.erase(handle);
}

std::size_t PagePool::Borrow(std::size_t pages) {
  std::vector<std::shared_ptr<PageLender>> lenders;
  {
    std::lock_guard lock(lender_mutex_);
    for (const auto& weak : lenders_)
      if (auto lender = weak.lock())
        lenders.push_back(std::move(lender));
  }
  std::vector<hipMemGenericAllocationHandle_t> handles;
  for (const auto& lender : lenders) {
    if (handles.size() >= pages)
      break;
    (void)lender->Lend(pages - handles.size(), handles);
  }
  if (handles.empty())
    return 0;
  // A lent page holds the cache's bytes: it joins the recycled pages, which
  // a region clears before use.
  std::lock_guard lock(mutex_);
  for (const auto handle : handles)
    recycled_.push_back(handle);
  StateLedger::Process().AddPooled(handles.size() * kPageBytes);
  lent_ += handles.size();
  return handles.size();
}

void PagePool::BorrowBelowFloor(std::size_t pages) {
  {
    std::lock_guard lock(mutex_);
    if (free_.size() + recycled_.size() >= pages)
      return;
  }
  if (!may_create_(pages * kPageBytes))
    (void)Borrow(std::max(pages, kBorrowBatchPages));
}

hipStream_t PagePool::ZeroStream() {
  std::lock_guard lock(mutex_);
  if (zero_stream_ == nullptr &&
      hipStreamCreateWithFlags(&zero_stream_, hipStreamNonBlocking) !=
          hipSuccess) {
    (void)hipGetLastError();
    zero_stream_ = nullptr;
  }
  return zero_stream_;
}

void PagePool::MapAhead(const std::shared_ptr<Region>& region,
                        std::size_t rows) {
  {
    std::lock_guard lock(mutex_);
    auto& job = jobs_[region.get()];
    if (job.region.expired() || job.region.lock() != region) {
      job.region = region;
      job.rows = rows;
    } else {
      job.rows = std::max(job.rows, rows);
    }
  }
  wake_.notify_one();
}

void PagePool::Warm(std::size_t bytes) {
  {
    std::lock_guard lock(mutex_);
    warm_bytes_ = std::min(kWarmCapBytes, warm_bytes_ + bytes);
  }
  wake_.notify_one();
}

void PagePool::Drain() {
  std::unique_lock lock(mutex_);
  idle_.wait(lock, [&] { return jobs_.empty() && !busy_; });
}

std::size_t PagePool::TargetFreePages() {
  // Refill from the low to the high watermark, plus any warm-up still owed,
  // but never beyond the pages regions may still map: demand mapping must
  // not commit more than allocating every region whole did.
  const std::size_t free = free_.size() + recycled_.size();
  if (free < kLowPages)
    refilling_ = true;
  else if (free >= kHighPages)
    refilling_ = false;
  const std::size_t unmapped =
      StateLedger::Process().UnmappedBytes() / kPageBytes;
  const std::size_t wanted = std::max(refilling_ ? kHighPages : kLowPages,
                                      free + warm_bytes_ / kPageBytes);
  return std::min(unmapped, wanted);
}

PagePoolStats PagePool::Stats() const {
  std::lock_guard lock(mutex_);
  return {.free_pages = free_.size() + recycled_.size(),
          .pages_in_use = in_use_,
          .created_on_demand = created_on_demand_,
          .created_ahead = created_ahead_,
          .skipped_below_floor = skipped_below_floor_,
          .lent = lent_,
          .returned = returned_};
}

void PagePool::Run() {
  (void)hipSetDevice(device_);
  std::unique_lock lock(mutex_);
  while (!stop_) {
    if (!jobs_.empty()) {
      auto job = std::move(jobs_.begin()->second);
      jobs_.erase(jobs_.begin());
      busy_ = true;
      lock.unlock();
      if (auto region = job.region.lock()) {
        // A few pages per segment at a time, so a writer that needs the
        // region now never waits for the whole window.
        const std::size_t step = std::max<std::size_t>(
            1, 4 * kPageBytes / region->shape().row_bytes);
        for (std::size_t rows = region->backed_rows();
             rows < region->AheadLimit(job.rows);) {
          // Below the floor, take the cache's borrowed pages before Ensure
          // would create new ones from compaction.
          BorrowBelowFloor(4 * region->shape().segments);
          if (!region->EnsureAhead(std::min(job.rows, rows + step)))
            break;
          rows = region->backed_rows();
        }
      }
      lock.lock();
      busy_ = false;
      continue;
    }
    if (free_.size() + recycled_.size() < TargetFreePages()) {
      // Free 2 MiB blocks first: a page made from them costs no reclaim.
      bool create = may_create_(kPageBytes);
      const std::size_t free = free_.size() + recycled_.size();
      if (!create && free < kLowPages) {
        // Otherwise take pages the RAM cache borrowed from the promised
        // state: a map, no new memory. Only up to the low watermark: pages
        // beyond it would sit idle here while the cache could serve them.
        lock.unlock();
        const std::size_t borrowed =
            Borrow(std::min(kBorrowBatchPages, kLowPages - free));
        lock.lock();
        if (borrowed)
          continue;
      }
      // Pages from reclaim (possibly compaction) only for the warm-up at
      // load, or when writers would otherwise make them on their own
      // threads, and never below the floor of MemAvailable.
      if (!create)
        create = (warm_bytes_ > 0 || free < kLowPages) &&
                 host_available_() >= floor_bytes_ + kPageBytes;
      if (!create) {
        // Writers take pooled or on-demand pages. Look again once memory
        // may have freed.
        ++skipped_below_floor_;
        warm_bytes_ = 0;
        wake_.wait_for(lock, std::chrono::milliseconds(250), [&] {
          return stop_ || !jobs_.empty() ||
                 free_.size() + recycled_.size() < free;
        });
        continue;
      }
      lock.unlock();
      hipMemGenericAllocationHandle_t handle{};
      const bool created = CreatePage(&handle);
      lock.lock();
      if (!created) {
        warm_bytes_ = 0;
        wake_.wait_for(lock, std::chrono::milliseconds(250),
                       [&] { return stop_ || !jobs_.empty(); });
        continue;
      }
      free_.push_back(handle);
      StateLedger::Process().AddPooled(kPageBytes);
      ++created_ahead_;
      warm_bytes_ = warm_bytes_ > kPageBytes ? warm_bytes_ - kPageBytes : 0;
      continue;
    }
    warm_bytes_ = 0;
    // Pages of released regions that no region promises any more go back to
    // the driver, on this thread only: a release may wait for the device.
    const std::size_t keep =
        StateLedger::Process().UnmappedBytes() / kPageBytes;
    if (jobs_.empty() && free_.size() + recycled_.size() > keep) {
      auto& list = recycled_.empty() ? free_ : recycled_;
      const auto handle = list.back();
      list.pop_back();
      StateLedger::Process().RemovePooled(kPageBytes);
      lock.unlock();
      // A page the cache lent goes back to it: the cache still maps it, so
      // releasing it to the driver would free nothing.
      if (!ReturnHome(handle))
        (void)hipMemRelease(handle);
      lock.lock();
      continue;
    }
    idle_.notify_all();
    // Regions notify when they reserve or release; the timeout covers
    // mapped bytes that change the target without a notification.
    wake_.wait_for(lock, std::chrono::seconds(1), [&] {
      return stop_ || !jobs_.empty() ||
             free_.size() + recycled_.size() < TargetFreePages();
    });
  }
  idle_.notify_all();
}

}  // namespace gufo::mem
