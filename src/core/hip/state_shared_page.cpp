#include "src/core/hip/state_shared_page.hpp"

#include <algorithm>
#include <map>
#include <utility>

#include "src/core/hip/state_page_pool.hpp"
#include "src/core/hip/state_region.hpp"
#include "src/mem/remap_barrier.hpp"

namespace gufo::mem {

SharedPage::SharedPage(std::shared_ptr<PagePool> pool,
                       hipMemGenericAllocationHandle_t handle)
    : pool_(std::move(pool)), handle_(handle) {}

SharedPage::~SharedPage() {
  // No region maps it and the cache no longer holds it. A park address may
  // still map it until the next sweep; the pool clears a page before a
  // region uses it again.
  pool_->Release(handle_);
}

void SharedPage::AddRegion() noexcept {
  const std::lock_guard lock(mutex_);
  if (regions_++ != 0)
    return;
  auto& ledger = StateLedger::Process();
  if (holders_ != 0)
    ledger.RemoveLendable(kPageBytes);
  ledger.AddMapped(kPageBytes);
}

void SharedPage::RemoveRegion() noexcept {
  const std::lock_guard lock(mutex_);
  if (--regions_ != 0)
    return;
  auto& ledger = StateLedger::Process();
  ledger.RemoveMapped(kPageBytes);
  if (holders_ != 0)
    ledger.AddLendable(kPageBytes);
}

void SharedPage::AddHolder() noexcept {
  const std::lock_guard lock(mutex_);
  if (holders_++ == 0 && regions_ == 0)
    StateLedger::Process().AddLendable(kPageBytes);
}

void SharedPage::RemoveHolder() noexcept {
  const std::lock_guard lock(mutex_);
  if (--holders_ == 0 && regions_ == 0)
    StateLedger::Process().RemoveLendable(kPageBytes);
}

ParkedPage::~ParkedPage() {
  park_->Release(slot_, page_.get());
  page_->RemoveHolder();
}

namespace {
std::mutex parks_mutex;
// As the pools: parks outlive every checkpoint and are never destroyed
// during static destruction.
auto* const parks = new std::map<int, std::shared_ptr<PagePark>>();
}  // namespace

std::shared_ptr<PagePark> PagePark::ForDevice(int device) {
  auto pool = PagePool::ForDevice(device);
  // Parked pages are device heap pages, as the pages the cache borrows:
  // only where the heap covers host memory.
  if (!pool || !pool->exportable())
    return nullptr;
  const std::lock_guard lock(parks_mutex);
  auto& park = (*parks)[device];
  if (!park)
    park = std::make_shared<PagePark>(std::move(pool));
  return park;
}

PagePark::PagePark(std::shared_ptr<PagePool> pool) : pool_(std::move(pool)) {}

PagePark::~PagePark() {
  for (std::size_t slot = 0; slot < slots_.size(); ++slot)
    if (slots_[slot] != Slot::kClean)
      (void)hipMemUnmap(Address(slot), kPageBytes);
  for (auto* chunk : chunks_)
    (void)hipMemAddressFree(chunk, kChunkPages * kPageBytes);
  (void)hipGetLastError();
}

std::shared_ptr<ParkedPage> PagePark::Park(
    const std::shared_ptr<SharedPage>& page) {
  if (!page)
    return nullptr;
  const std::lock_guard lock(mutex_);
  if (const auto it = by_page_.find(page.get()); it != by_page_.end())
    if (auto parked = parked_[it->second].lock())
      return parked;
  std::size_t slot = 0;
  // The address unmapped longest ago, if the device has been idle since;
  // else a new one, while the range has any.
  const bool room =
      next_ != chunks_.size() * kChunkPages || chunks_.size() != kMaxChunks;
  bool reused = false;
  if (!clean_.empty() &&
      (RemapBarrier::Clean(clean_.front().second) || !room)) {
    slot = clean_.front().first;
    // Out of addresses: the caller's next barrier makes this one reach the
    // page.
    if (!RemapBarrier::Clean(clean_.front().second))
      RemapBarrier::NoteRemap();
    clean_.pop_front();
    reused = true;
  }
  if (!reused) {
    if (next_ == chunks_.size() * kChunkPages) {
      if (chunks_.size() == kMaxChunks)
        return nullptr;
      void* base = nullptr;
      if (hipMemAddressReserve(&base, kChunkPages * kPageBytes, kPageBytes,
                               nullptr, 0) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
      }
      chunks_.push_back(static_cast<std::byte*>(base));
      slots_.resize(chunks_.size() * kChunkPages, Slot::kClean);
      parked_.resize(slots_.size());
    }
    slot = next_++;
  }
  std::byte* const address = Address(slot);
  if (hipMemMap(address, kPageBytes, 0, page->handle(), 0) != hipSuccess) {
    (void)hipGetLastError();
    // First in line again, once the device has been idle.
    clean_.emplace_front(slot, RemapBarrier::Stamp());
    return nullptr;
  }
  if (hipMemSetAccess(address, kPageBytes, &pool_->access(), 1) != hipSuccess) {
    (void)hipGetLastError();
    // Mapped but unusable: unmapped by the next sweep.
    slots_[slot] = Slot::kStale;
    stale_.push_back(slot);
    return nullptr;
  }
  page->AddHolder();
  std::shared_ptr<ParkedPage> parked(
      new ParkedPage(shared_from_this(), slot, address, page));
  slots_[slot] = Slot::kLive;
  parked_[slot] = parked;
  by_page_[page.get()] = slot;
  ++live_;
  ++total_;
  return parked;
}

void PagePark::Release(std::size_t slot, SharedPage* page) noexcept {
  const std::lock_guard lock(mutex_);
  // The page may have been parked again at another address meanwhile.
  if (const auto it = by_page_.find(page);
      it != by_page_.end() && it->second == slot)
    by_page_.erase(it);
  slots_[slot] = Slot::kStale;
  parked_[slot].reset();
  stale_.push_back(slot);
  --live_;
}

std::shared_ptr<ParkedPage> PagePark::Find(const void* address) {
  const auto* const p = static_cast<const std::byte*>(address);
  const std::lock_guard lock(mutex_);
  for (std::size_t c = 0; c < chunks_.size(); ++c) {
    if (p < chunks_[c] || p >= chunks_[c] + kChunkPages * kPageBytes)
      continue;
    const std::size_t slot =
        c * kChunkPages + static_cast<std::size_t>(p - chunks_[c]) / kPageBytes;
    return slots_[slot] == Slot::kLive ? parked_[slot].lock() : nullptr;
  }
  return nullptr;
}

bool PagePark::Contains(const void* address) const {
  const auto* const p = static_cast<const std::byte*>(address);
  const std::lock_guard lock(mutex_);
  return std::ranges::any_of(chunks_, [&](const std::byte* chunk) {
    return p >= chunk && p < chunk + kChunkPages * kPageBytes;
  });
}

std::size_t PagePark::Sweep() {
  const std::lock_guard lock(mutex_);
  if (stale_.empty())
    return 0;
  std::ranges::sort(stale_);
  const std::size_t swept = stale_.size();
  // An unmap waits for device work in flight: hold other threads' back.
  RemapBarrier::Hold();
  const auto stamp = RemapBarrier::Stamp();
  // One call per run of neighbours: every call waits for the device.
  for (std::size_t i = 0; i < stale_.size();) {
    std::size_t j = i + 1;
    while (j < stale_.size() && stale_[j] == stale_[j - 1] + 1 &&
           stale_[j] / kChunkPages == stale_[i] / kChunkPages)
      ++j;
    (void)hipMemUnmap(Address(stale_[i]), (j - i) * kPageBytes);
    for (; i < j; ++i) {
      slots_[stale_[i]] = Slot::kClean;
      clean_.emplace_back(stale_[i], stamp);
    }
  }
  (void)hipGetLastError();
  RemapBarrier::Release();
  stale_.clear();
  return swept;
}

PageParkStats PagePark::Stats() const {
  const std::lock_guard lock(mutex_);
  return {.parked = live_,
          .parked_total = total_,
          .stale = stale_.size(),
          .reserved = chunks_.size() * kChunkPages};
}

namespace {
class ParkedRows final : public SharedRows {
public:
  explicit ParkedRows(std::shared_ptr<PagePark> park)
      : park_(std::move(park)) {}
  std::shared_ptr<const void> Share(std::span<const std::byte> view) override {
    const auto region = Region::Find(view.data());
    if (!region || view.empty())
      return nullptr;
    const auto offset = static_cast<std::size_t>(
        view.data() - static_cast<const std::byte*>(region->data()));
    const auto in_page = offset % kPageBytes;
    // One owner per payload: rows that cross a page are copied.
    if (in_page + view.size() > kPageBytes)
      return nullptr;
    auto parked =
        park_->Park(region->Share(offset / kPageBytes, offset + view.size()));
    if (!parked)
      return nullptr;
    auto* const data = parked->data() + in_page;
    return std::shared_ptr<const void>(std::move(parked), data);
  }
  bool Contains(const void* address) const override {
    return park_->Contains(address);
  }
  void Sweep() override { (void)park_->Sweep(); }
  std::size_t FreePages() const override {
    return park_->pool().Stats().free_pages;
  }
  std::size_t ReleaseFree(std::size_t bytes) override {
    return park_->pool().ReleaseFree(bytes);
  }
  bool SettleRemaps() override { return park_->pool().SettleRemaps(); }

private:
  std::shared_ptr<PagePark> park_;
};
}  // namespace

std::shared_ptr<SharedRows> CreateParkedRows() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  auto park = PagePark::ForDevice(device);
  return park ? std::make_shared<ParkedRows>(std::move(park)) : nullptr;
}

RestorePlan::RestorePlan(Region& region, PagePark* park)
    : region_(region), park_(park) {}

void RestorePlan::Add(std::size_t offset, std::span<const std::byte> bytes) {
  if (park_ == nullptr || bytes.empty())
    return;
  const std::size_t page = offset / kPageBytes;
  const std::size_t in_page = offset % kPageBytes;
  if (in_page + bytes.size() > kPageBytes)
    return;
  if (pages_.empty()) {
    const std::size_t pages = (region_.bytes() + kPageBytes - 1) / kPageBytes;
    pages_.resize(pages);
    covered_.resize(pages);
    mixed_.resize(pages);
  }
  if (page >= pages_.size())
    return;
  // Chunks of one page follow each other: look the page up once.
  if (!parked_ || bytes.data() < parked_->data() ||
      bytes.data() >= parked_->data() + kPageBytes)
    parked_ = park_->Find(bytes.data());
  // The rows must sit in the page where the region expects them.
  if (!parked_ ||
      static_cast<std::size_t>(bytes.data() - parked_->data()) != in_page ||
      (pages_[page] && pages_[page] != parked_->page())) {
    mixed_[page] = true;
    return;
  }
  pages_[page] = parked_->page();
  covered_[page] += bytes.size();
}

std::vector<bool> RestorePlan::Apply(std::size_t rows) {
  // Only pages the checkpoint fills: a page it ends in is loaded, so the
  // region writes its later rows to a page of its own.
  for (std::size_t p = 0; p < pages_.size(); ++p)
    if (mixed_[p] || covered_[p] != kPageBytes)
      pages_[p].reset();
  parked_.reset();
  return region_.Restore(pages_, rows);
}

}  // namespace gufo::mem
