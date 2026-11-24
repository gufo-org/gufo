#include "src/core/hip/state_lendable_range.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/hip/state_page_pool.hpp"
#include "src/core/platform/pages.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/remap_barrier.hpp"

namespace gufo::mem {
namespace {

class VmmLendableRange final
    : public LendableRange,
      public PageLender,
      public std::enable_shared_from_this<VmmLendableRange> {
public:
  VmmLendableRange(std::shared_ptr<PagePool> pool, std::byte* base,
                   std::size_t pages)
      : pool_(std::move(pool)),
        base_(base),
        pages_(pages),
        handles_(pages),
        state_(pages, State::kAbsent),
        unmapped_at_(pages, 0) {}

  ~VmmLendableRange() override {
    pool_->RemoveLender(this);
    // Shutdown only: unmapping waits for the device. Lent pages stay with
    // the sessions that map them; only this alias goes.
    auto& ledger = StateLedger::Process();
    for (std::size_t p = 0; p < pages_; ++p) {
      if (state_[p] == State::kAbsent)
        continue;
      std::byte* page = base_ + p * kPageBytes;
      platform::UnmapDeviceFileAt(page, kPageBytes);
      (void)hipMemUnmap(page, kPageBytes);
      if (state_[p] == State::kCommitted) {
        ledger.RemoveLendable(kPageBytes);
        pool_->Recycle(handles_[p]);
      }
    }
    (void)hipGetLastError();
    (void)hipMemAddressFree(base_, pages_ * kPageBytes);
  }

  [[nodiscard]] std::byte* data() const noexcept override { return base_; }
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return pages_ * kPageBytes;
  }
  [[nodiscard]] std::size_t page_bytes() const noexcept override {
    return kPageBytes;
  }

  bool Commit(std::size_t first, std::size_t count, bool may_reclaim) override {
    if (first > pages_ || count > pages_ - first)
      return false;
    const std::lock_guard lock(mutex_);
    // Pages first, in order, then their mappings in parallel: exporting and
    // mapping a page for the CPU costs about 0.15 ms, so 17 GB of cache took
    // 0.9 s at load on one thread.
    std::vector<std::size_t> wanted;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    bool complete = true;
    for (std::size_t p = first; p < first + count; ++p) {
      if (state_[p] != State::kAbsent)
        continue;
      hipMemGenericAllocationHandle_t handle{};
      if (!pool_->AcquireForLender(&handle, may_reclaim)) {
        complete = false;
        break;
      }
      wanted.push_back(p);
      handles.push_back(handle);
    }
    std::vector<std::uint8_t> mapped(wanted.size());
    // Eight lanes while loading; two while serving, beside running work.
    constexpr std::size_t kPagesPerThread = 8;
    const std::size_t lanes = std::clamp<std::size_t>(
        wanted.size() / kPagesPerThread, 1, may_reclaim ? 8 : 2);
    // Lane l maps pages l, l + lanes, ...; this thread takes lane 0 and any
    // lane whose worker could not start.
    const auto map_lane = [&](std::size_t lane) {
      for (std::size_t i = lane; i < wanted.size(); i += lanes)
        mapped[i] = MapPage(wanted[i], handles[i]);
    };
    std::vector<std::thread> workers;
    std::size_t started = 1;
    try {
      for (; started < lanes; ++started)
        workers.emplace_back(map_lane, started);
    } catch (...) {
    }
    for (std::size_t lane = started; lane < lanes; ++lane)
      map_lane(lane);
    map_lane(0);
    for (auto& worker : workers)
      worker.join();
    auto& ledger = StateLedger::Process();
    const auto self = shared_from_this();
    for (std::size_t i = 0; i < wanted.size(); ++i) {
      if (!mapped[i]) {
        pool_->Recycle(handles[i]);
        complete = false;
        continue;
      }
      const std::size_t p = wanted[i];
      handles_[p] = handles[i];
      state_[p] = State::kCommitted;
      index_[handles[i]] = p;
      pool_->SetHome(handles[i], self);
      ledger.AddLendable(kPageBytes);
    }
    return complete;
  }

  void SetOwner(Select select, Restore restore) override {
    auto owner = std::make_shared<const Owner>(
        Owner{std::move(select), std::move(restore)});
    const std::lock_guard lock(owner_mutex_);
    owner_ = std::move(owner);
  }
  void ClearOwner() override {
    const std::lock_guard lock(owner_mutex_);
    owner_.reset();
  }

  Decommitted Decommit(std::span<const std::size_t> pages) override {
    Decommitted result;
    const std::lock_guard lock(mutex_);
    auto& ledger = StateLedger::Process();
    for (const std::size_t p : pages) {
      if (p >= pages_ || state_[p] != State::kCommitted)
        continue;
      std::byte* page = base_ + p * kPageBytes;
      // The CPU view first, then the GPU mapping: the page's memory goes
      // back to the driver only once both are gone.
      platform::UnmapDeviceFileAt(page, kPageBytes);
      if (hipMemUnmap(page, kPageBytes) != hipSuccess) {
        (void)hipGetLastError();
        // Still mapped for the GPU: give the CPU its view back and keep the
        // page. Without one the owner must never use it again, so it is
        // reported released and stays retired here.
        int fd = -1;
        if (hipMemExportToShareableHandle(&fd, handles_[p],
                                          hipMemHandleTypePosixFileDescriptor,
                                          0) == hipSuccess &&
            platform::MapDeviceFileAt(page, kPageBytes, fd))
          continue;
        (void)hipGetLastError();
        state_[p] = State::kRetired;
        ledger.RemoveLendable(kPageBytes);
        result.retired.push_back(p);
        continue;
      }
      const auto handle = handles_[p];
      pool_->ClearHome(handle);
      index_.erase(handle);
      (void)hipMemRelease(handle);
      (void)hipGetLastError();
      handles_[p] = {};
      state_[p] = State::kAbsent;
      // The device may still reach the old page here until it has been idle.
      unmapped_at_[p] = RemapBarrier::Stamp();
      ledger.RemoveLendable(kPageBytes);
      result.released.push_back(p);
    }
    return result;
  }

  std::size_t Lend(std::size_t pages,
                   std::vector<hipMemGenericAllocationHandle_t>& out) override {
    const auto owner = Current();
    if (!owner || !owner->select)
      return 0;
    std::vector<std::size_t> chosen;
    try {
      (void)owner->select(pages, chosen);
    } catch (...) {
      return 0;
    }
    const std::lock_guard lock(mutex_);
    std::size_t lent = 0;
    for (const std::size_t p : chosen) {
      if (p >= pages_ || state_[p] != State::kCommitted)
        continue;
      state_[p] = State::kLent;
      out.push_back(handles_[p]);
      StateLedger::Process().RemoveLendable(kPageBytes);
      ++lent;
    }
    return lent;
  }

  bool TakeBack(hipMemGenericAllocationHandle_t handle) override {
    std::size_t page = 0;
    {
      const std::lock_guard lock(mutex_);
      const auto it = index_.find(handle);
      if (it == index_.end() || state_[it->second] != State::kLent)
        return false;
      page = it->second;
      state_[page] = State::kCommitted;
      StateLedger::Process().AddLendable(kPageBytes);
    }
    if (const auto owner = Current(); owner && owner->restore)
      owner->restore(page);
    return true;
  }

private:
  // kRetired: a decommit that could not unmap the page; never used again.
  enum class State : std::uint8_t { kAbsent, kCommitted, kLent, kRetired };
  struct Owner {
    Select select;
    Restore restore;
  };

  // Calls into the owner hold no lock here: an eviction may wait for the
  // cache's lock while that holder returns a page to this range.
  [[nodiscard]] std::shared_ptr<const Owner> Current() {
    const std::lock_guard lock(owner_mutex_);
    return owner_;
  }

  // GPU mapping, then the CPU view of the same page at the same address.
  // A failure after the GPU map unmaps it again: rare, and Commit runs off
  // the serving path.
  bool MapPage(std::size_t p, hipMemGenericAllocationHandle_t handle) {
    std::byte* page = base_ + p * kPageBytes;
    if (hipMemMap(page, kPageBytes, 0, handle, 0) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    int fd = -1;
    const bool mapped =
        hipMemSetAccess(page, kPageBytes, &pool_->access(), 1) == hipSuccess &&
        hipMemExportToShareableHandle(&fd, handle,
                                      hipMemHandleTypePosixFileDescriptor,
                                      0) == hipSuccess &&
        platform::MapDeviceFileAt(page, kPageBytes, fd);
    if (mapped) {
      // A page committed where another was decommitted: the cache settles
      // before the device writes a capture there (PagePool::SettleRemaps).
      if (!RemapBarrier::Clean(unmapped_at_[p]))
        RemapBarrier::NoteRemap();
      return true;
    }
    (void)hipGetLastError();
    platform::UnmapDeviceFileAt(page, kPageBytes);
    (void)hipMemUnmap(page, kPageBytes);
    (void)hipGetLastError();
    return false;
  }

  std::shared_ptr<PagePool> pool_;
  std::byte* base_;
  std::size_t pages_;
  std::mutex mutex_;
  std::vector<hipMemGenericAllocationHandle_t> handles_;
  std::vector<State> state_;
  /// RemapBarrier::Stamp() of each page's last decommit; 0 without one.
  std::vector<std::uint64_t> unmapped_at_;
  std::unordered_map<hipMemGenericAllocationHandle_t, std::size_t> index_;
  std::mutex owner_mutex_;
  std::shared_ptr<const Owner> owner_;
};

}  // namespace

std::shared_ptr<LendableRange> CreateVmmLendableRange(std::size_t bytes) {
  int device = 0;
  if (bytes == 0 || hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  auto pool = PagePool::ForDevice(device);
  if (!pool || !pool->exportable())
    return nullptr;
  const std::size_t reserved = RoundUpToPage(bytes);
  void* base = nullptr;
  if (hipMemAddressReserve(&base, reserved, kPageBytes, nullptr, 0) !=
      hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  // The CPU view replaces the runtime's own reservation of this range. Never
  // map over an address the runtime did not reserve for the CPU too.
  if (!platform::IsAddressRangeReserved(base, reserved)) {
    (void)hipMemAddressFree(base, reserved);
    return nullptr;
  }
  auto range = std::make_shared<VmmLendableRange>(
      pool, static_cast<std::byte*>(base), reserved / kPageBytes);
  pool->AddLender(range);
  return range;
}

}  // namespace gufo::mem
