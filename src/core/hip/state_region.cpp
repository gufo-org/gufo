#include "src/core/hip/state_region.hpp"

#include <algorithm>
#include <limits>
#include <map>

#include "src/core/hip/state_page_pool.hpp"
#include "src/core/hip/state_shared_page.hpp"
#include "src/mem/ledger.hpp"
#include "src/mem/remap_barrier.hpp"

namespace gufo::mem {
namespace {

void SetError(std::string* error, const char* message) {
  if (error)
    *error = message;
}

[[nodiscard]] bool CheckedShape(const RegionShape& shape, std::size_t* stride,
                                std::size_t* bytes) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  if (shape.row_bytes == 0 || shape.rows == 0 || shape.segments == 0 ||
      shape.rows > kMax / shape.row_bytes)
    return false;
  const std::size_t run = shape.rows * shape.row_bytes;
  *stride = shape.segment_stride ? shape.segment_stride : run;
  if (*stride < run || shape.segments - 1 > (kMax - run) / *stride)
    return false;
  *bytes = (shape.segments - 1) * *stride + run;
  return *bytes <= kMax - kPageBytes;
}

// RemapBarrier::Stamp() of the last region that gave its address range
// back. The driver may hand the same range to the next region, whose pages
// would then be mapped where others were unmapped: a region created before
// the device has been idle since runs the barrier first.
std::atomic<std::uint64_t> freed_at{0};

// Demand-mapped regions by base address, for Region::Find().
std::mutex registry_mutex;
auto* const registry = new std::map<const std::uint8_t*, Region*>();

}  // namespace

std::shared_ptr<Region> Region::Find(const void* address) {
  const auto* const p = static_cast<const std::uint8_t*>(address);
  const std::lock_guard lock(registry_mutex);
  auto it = registry->upper_bound(p);
  if (it == registry->begin())
    return nullptr;
  --it;
  // A region leaves the registry before its last owner lets go of it.
  return p < it->first + it->second->bytes_
             ? it->second->weak_from_this().lock()
             : nullptr;
}

bool DemandMapped() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return PagePool::ForDevice(device) != nullptr;
}

std::shared_ptr<Region> Region::Create(
    const RegionShape& shape, std::string* error,
    const std::shared_ptr<StateAccount>& account, std::size_t tokens_per_row) {
  std::shared_ptr<Region> region(new Region());
  region->shape_ = shape;
  if (!CheckedShape(shape, &region->stride_, &region->bytes_)) {
    SetError(error, "invalid state region shape");
    return nullptr;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    SetError(error, "no current HIP device for a state region");
    return nullptr;
  }
  if (auto pool = PagePool::ForDevice(device)) {
    const std::size_t reserved = RoundUpToPage(region->bytes_);
    void* base = nullptr;
    if (hipMemAddressReserve(&base, reserved, kPageBytes, nullptr, 0) ==
        hipSuccess) {
      region->pool_ = std::move(pool);
      region->base_ = static_cast<std::uint8_t*>(base);
      region->reserved_bytes_ = RegionPageBytes(shape, shape.rows);
      region->pages_.resize(reserved / kPageBytes);
      region->mapped_.resize(reserved / kPageBytes);
      region->shared_.resize(reserved / kPageBytes);
      region->shared_until_.resize(reserved / kPageBytes);
      region->written_here_.resize(reserved / kPageBytes);
      region->unmapped_at_.resize(reserved / kPageBytes);
      // Sessions are created at load and between requests, never mid-pass.
      if (!RemapBarrier::Clean(freed_at.load(std::memory_order_acquire)))
        RemapBarrier::Settle();
      // Pages belong to one segment each when segments start on a page.
      region->shareable_ =
          shape.segments == 1 || region->stride_ % kPageBytes == 0;
      {
        const std::lock_guard lock(registry_mutex);
        (*registry)[region->base_] = region.get();
      }
      StateLedger::Process().AddReserved(region->reserved_bytes_);
      region->pool_->Warm(region->reserved_bytes_);
      if (account) {
        region->account_ = account;
        region->tokens_per_row_ = std::max<std::size_t>(tokens_per_row, 1);
        account->AddRegion(
            shape, region->tokens_per_row_,
            [weak = std::weak_ptr<Region>(region)](std::size_t rows) {
              if (const auto r = weak.lock())
                r->TrimTo(rows);
            });
      }
      return region;
    }
    (void)hipGetLastError();
  }
  // No virtual memory management, or no address range: the whole buffer,
  // zeroed like the demand-mapped pages and the buffers regions replace.
  void* data = nullptr;
  if (hipMalloc(&data, region->bytes_) != hipSuccess) {
    (void)hipGetLastError();
    SetError(error, "state region allocation failed");
    return nullptr;
  }
  // Finish the default-stream clear before a non-blocking stream can write.
  if (hipMemset(data, 0, region->bytes_) != hipSuccess ||
      hipStreamSynchronize(nullptr) != hipSuccess) {
    (void)hipGetLastError();
    (void)hipFree(data);
    SetError(error, "state region clear failed");
    return nullptr;
  }
  region->base_ = static_cast<std::uint8_t*>(data);
  region->requested_rows_ = shape.rows;
  region->backed_rows_.store(shape.rows, std::memory_order_release);
  region->mapped_bytes_.store(region->bytes_, std::memory_order_relaxed);
  return region;
}

Region::~Region() {
  if (base_ == nullptr)
    return;
  if (!pool_) {
    (void)hipFree(base_);
    return;
  }
  {
    const std::lock_guard lock(registry_mutex);
    registry->erase(base_);
  }
  // Unmapping waits for all in-flight GPU work on the device; regions end
  // with their sessions, at shutdown or between requests, never mid-pass.
  UnmapMarked(mapped_);
  (void)hipGetLastError();
  freed_at.store(RemapBarrier::Stamp(), std::memory_order_release);
  StateLedger::Process().RemoveReserved(reserved_bytes_);
  (void)hipMemAddressFree(base_, pages_.size() * kPageBytes);
}

std::size_t Region::CoveredRows(std::size_t pages_end_bytes,
                                std::size_t segment) const noexcept {
  const std::size_t start = segment * stride_;
  return std::min(shape_.rows, (pages_end_bytes - start) / shape_.row_bytes);
}

bool Region::EnsureAhead(std::size_t rows) {
  std::lock_guard lock(mutex_);
  rows = AheadLimit(rows);
  if (backed_rows() >= rows)
    return true;
  return MapPages(rows, nullptr, true);
}

bool Region::Ensure(std::size_t rows, std::string* error) {
  rows = std::min(rows, shape_.rows);
  if (backed_rows() >= rows)
    return true;
  std::lock_guard lock(mutex_);
  if (backed_rows() >= rows)
    return true;
  return MapPages(rows, error, false);
}

void Region::ReleasePage(std::size_t p) {
  if (shared_[p]) {
    // Other users keep the page; with the last one it returns to the pool.
    shared_[p]->RemoveRegion();
    shared_[p].reset();
  } else {
    pool_->Release(pages_[p]);
    StateLedger::Process().RemoveMapped(kPageBytes);
  }
  mapped_[p] = false;
  mapped_bytes_.fetch_sub(kPageBytes, std::memory_order_relaxed);
  if (account_)
    account_->OnUnmap(kPageBytes);
}

void Region::UnmapRun(std::size_t first, std::size_t end) {
  // One call for the run: each call waits for in-flight GPU work.
  (void)hipMemUnmap(base_ + first * kPageBytes, (end - first) * kPageBytes);
  // The device may go on reaching the old pages through these addresses
  // until it has been idle (RemapBarrier).
  const auto stamp = RemapBarrier::Stamp();
  for (std::size_t p = first; p < end; ++p) {
    unmapped_at_[p] = stamp;
    ReleasePage(p);
  }
}

void Region::UnmapMarked(const std::vector<bool>& drop) {
  for (std::size_t p = 0; p < pages_.size();) {
    if (!drop[p] || !mapped_[p]) {
      ++p;
      continue;
    }
    std::size_t end = p + 1;
    while (end < pages_.size() && drop[end] && mapped_[end])
      ++end;
    UnmapRun(p, end);
    p = end;
  }
}

bool Region::SharedWithOthers(std::size_t p) const noexcept {
  // New users come only through Share() and from existing users, so a sole
  // owner under mutex_ stays one.
  return shared_[p] && shared_[p].use_count() > 1;
}

bool Region::MustReplace(std::size_t p, std::size_t first_byte) {
  if (!mapped_[p] || !shared_[p])
    return false;
  if (!SharedWithOthers(p)) {
    // The last other user left: the page is this region's alone again.
    written_here_[p] = true;
    shared_until_[p] = p * kPageBytes;
    return false;
  }
  return !written_here_[p] ||
         std::max(first_byte, p * kPageBytes) < shared_until_[p];
}

std::size_t Region::MappedPrefixRows() const noexcept {
  std::size_t backed = shape_.rows;
  for (std::size_t s = 0; s < shape_.segments; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t last =
        (start + shape_.rows * shape_.row_bytes + kPageBytes - 1) / kPageBytes;
    std::size_t p = start / kPageBytes;
    while (p < last && mapped_[p])
      ++p;
    backed = std::min(
        backed, p * kPageBytes > start ? CoveredRows(p * kPageBytes, s) : 0);
  }
  return backed;
}

std::size_t Region::SegmentStart(std::size_t p) const noexcept {
  return shape_.segments == 1 ? 0 : p * kPageBytes / stride_ * stride_;
}

std::size_t Region::SharedEnd() const noexcept {
  std::size_t end = 0;
  if (shape_.segments == 1) {
    for (std::size_t p = pages_.size(); p-- > 0;)
      if (mapped_[p] && SharedWithOthers(p))
        return (p + 1) * kPageBytes;
    return 0;
  }
  for (std::size_t p = 0; p < pages_.size(); ++p)
    if (mapped_[p] && SharedWithOthers(p))
      end = std::max(end, (p + 1) * kPageBytes - SegmentStart(p));
  return end;
}

void Region::SettleSharedEnd() noexcept {
  shared_end_.store(
      std::min(shared_end_.load(std::memory_order_relaxed), SharedEnd()),
      std::memory_order_release);
}

std::size_t Region::shared_pages() const {
  const std::lock_guard lock(mutex_);
  std::size_t count = 0;
  for (std::size_t p = 0; p < pages_.size(); ++p)
    count += mapped_[p] && SharedWithOthers(p);
  return count;
}

std::shared_ptr<SharedPage> Region::Share(std::size_t page,
                                          std::size_t end_byte) {
  if (!pool_ || !shareable_ || !StateLedger::Process().LendingEnabled())
    return nullptr;
  const std::lock_guard lock(mutex_);
  if (page >= pages_.size() || !mapped_[page] ||
      end_byte > (page + 1) * kPageBytes)
    return nullptr;
  if (!shared_[page])
    shared_[page] = std::make_shared<SharedPage>(pool_, pages_[page]);
  shared_until_[page] = std::max(shared_until_[page], end_byte);
  const std::size_t end = end_byte - SegmentStart(page);
  if (end > shared_end_.load(std::memory_order_relaxed))
    shared_end_.store(end, std::memory_order_release);
  return shared_[page];
}

bool Region::ReplaceRun(std::size_t first, std::size_t count, bool copy,
                        std::string* error) {
  const auto& access = pool_->access();
  auto& ledger = StateLedger::Process();
  std::vector<hipMemGenericAllocationHandle_t> fresh(count);
  std::vector<bool> recycled(count, false);
  std::size_t acquired = 0;
  const auto give_back = [&] {
    for (std::size_t k = 0; k < acquired; ++k)
      pool_->Release(fresh[k]);
  };
  for (; acquired < count; ++acquired) {
    bool reused = false;
    if (!pool_->Acquire(&fresh[acquired], &reused)) {
      give_back();
      SetError(error, "state memory exhausted");
      return false;
    }
    recycled[acquired] = reused;
  }
  // A copy reads the shared pages through a second mapping, since their
  // addresses here take the new pages now. The copy itself, like a clear,
  // waits for the barrier: until then these addresses still reach the
  // shared pages.
  std::vector<std::byte*> aliases(count, nullptr);
  if (copy) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      std::size_t k = 0;
      for (; k < count; ++k)
        if ((aliases[k] = pool_->AliasForCopy(pages_[first + k])) == nullptr)
          break;
      if (k == count)
        break;
      // No scratch address left: run what is queued, which frees them all
      // (these aliases too; nothing was queued on them yet).
      if (attempt == 1 || !pool_->SettleRemaps()) {
        give_back();
        SetError(error, "state page copy failed");
        return false;
      }
    }
  }
  // Until the caller's SettleRemaps(): an unmap waits for device work in
  // flight, other threads' included.
  RemapBarrier::Hold();
  (void)hipMemUnmap(base_ + first * kPageBytes, count * kPageBytes);
  const auto stamp = RemapBarrier::Stamp();
  bool ok = true;
  for (std::size_t k = 0; k < count; ++k) {
    const std::size_t p = first + k;
    // Kept until the copy ran: the other users may let go of the page.
    std::shared_ptr<SharedPage> old = std::move(shared_[p]);
    old->RemoveRegion();
    unmapped_at_[p] = stamp;
    if (hipMemMap(base_ + p * kPageBytes, kPageBytes, 0, fresh[k], 0) ==
        hipSuccess) {
      pages_[p] = fresh[k];
      written_here_[p] = true;
      shared_until_[p] = p * kPageBytes;
      ledger.AddMapped(kPageBytes);
      auto* const address =
          reinterpret_cast<std::byte*>(base_) + p * kPageBytes;
      if (copy)
        pool_->QueueCopy(address, aliases[k], std::move(old));
      else if (recycled[k])
        // Cleared like any recycled page: tile reads masked past a frontier
        // must never meet NaN patterns.
        pool_->QueueClear(address, kPageBytes);
      continue;
    }
    // The page stays unmapped; the rows from here on are no longer backed.
    ok = false;
    pool_->Release(fresh[k]);
    mapped_[p] = false;
    mapped_bytes_.fetch_sub(kPageBytes, std::memory_order_relaxed);
    if (account_)
      account_->OnUnmap(kPageBytes);
  }
  RemapBarrier::NoteRemap();
  if (!ok) {
    const std::size_t backed = std::min(backed_rows(), MappedPrefixRows());
    requested_rows_ = std::min(requested_rows_, backed);
    backed_rows_.store(backed, std::memory_order_release);
  }
  ok = ok && hipMemSetAccess(base_ + first * kPageBytes, count * kPageBytes,
                             &access, 1) == hipSuccess;
  if (!ok) {
    (void)hipGetLastError();
    SetError(error, "state page map failed");
  }
  return ok;
}

bool Region::SettleRemaps(std::string* error) {
  if (!pool_ || pool_->SettleRemaps())
    return true;
  SetError(error, "state page copy failed");
  return false;
}

bool Region::PrepareWrite(std::size_t first_row, std::size_t end_row,
                          std::size_t kept_rows, std::string* error) {
  if (!pool_ || first_row >= end_row)
    return true;
  const std::size_t first_byte = first_row * shape_.row_bytes;
  if (first_byte >= shared_end_.load(std::memory_order_acquire))
    return true;
  const std::lock_guard lock(mutex_);
  end_row = std::min(end_row, shape_.rows);
  const std::size_t kept_bytes = kept_rows * shape_.row_bytes;
  const std::size_t end_byte = end_row * shape_.row_bytes;
  bool replaced = false;
  // Only regions whose pages belong to one segment each share any.
  for (std::size_t s = 0; shareable_ && s < shape_.segments; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t last = std::min(
        pages_.size(), (start + end_byte + kPageBytes - 1) / kPageBytes);
    for (std::size_t p = (start + first_byte) / kPageBytes; p < last;) {
      if (!MustReplace(p, start + first_byte)) {
        ++p;
        continue;
      }
      // One run of neighbours that are all copied or all cleared.
      const bool copy = p * kPageBytes < start + kept_bytes;
      std::size_t end = p + 1;
      while (end < last && MustReplace(end, start + first_byte) &&
             (end * kPageBytes < start + kept_bytes) == copy &&
             (!copy || end - p < PagePool::kScratchPages))
        ++end;
      if (!ReplaceRun(p, end - p, copy, error))
        return false;
      replaced = true;
      p = end;
    }
  }
  if (replaced || shared_end_.load(std::memory_order_relaxed) <=
                      (end_byte + kPageBytes - 1) / kPageBytes * kPageBytes)
    SettleSharedEnd();
  return true;
}

std::vector<bool> Region::Restore(
    std::span<const std::shared_ptr<SharedPage>> plan, std::size_t rows) {
  std::vector<bool> adopted(plan.size(), false);
  if (!pool_)
    return adopted;
  const std::lock_guard lock(mutex_);
  rows = std::min(rows, shape_.rows);
  // Whether page `p` holds any of rows [0, rows) of its segment. A region
  // that shares no page has none to release or map.
  const std::size_t row_bytes = rows * shape_.row_bytes;
  const auto written = [&](std::size_t p) {
    return shareable_ && p * kPageBytes - SegmentStart(p) < row_bytes;
  };
  std::vector<bool> drop(pages_.size(), false);
  bool dropped = false;
  for (std::size_t p = 0; p < pages_.size(); ++p) {
    if (!mapped_[p] || !written(p))
      continue;
    const SharedPage* const want = p < plan.size() ? plan[p].get() : nullptr;
    if (want != nullptr && shared_[p].get() == want)
      adopted[p] = true;  // Already here: the checkpoint came from this page.
    else if (want != nullptr || SharedWithOthers(p))
      drop[p] = dropped = true;
  }
  if (dropped) {
    // As in ReplaceRun(): held until the caller's SettleRemaps().
    RemapBarrier::Hold();
    UnmapMarked(drop);
  }
  const auto& access = pool_->access();
  // Dropped addresses get pages again as the restore's loads arrive: the
  // caller's barrier makes them clean now, so the pool's thread can back
  // them ahead instead of each load doing it behind the model's work.
  bool remapped = dropped;
  std::size_t run_begin = 0, run_end = 0;
  const auto flush = [&] {
    if (run_end != run_begin)
      (void)hipMemSetAccess(base_ + run_begin * kPageBytes,
                            (run_end - run_begin) * kPageBytes, &access, 1);
    run_begin = run_end = 0;
  };
  for (std::size_t p = 0; p < std::min(plan.size(), pages_.size()); ++p) {
    if (!plan[p] || mapped_[p] || !written(p))
      continue;
    if (account_ && !account_->OnMap(kPageBytes))
      continue;
    if (hipMemMap(base_ + p * kPageBytes, kPageBytes, 0, plan[p]->handle(),
                  0) != hipSuccess) {
      (void)hipGetLastError();
      if (account_)
        account_->OnUnmap(kPageBytes);
      continue;
    }
    if (!RemapBarrier::Clean(unmapped_at_[p]))
      remapped = true;
    pages_[p] = plan[p]->handle();
    shared_[p] = plan[p];
    shared_until_[p] = (p + 1) * kPageBytes;
    written_here_[p] = false;
    plan[p]->AddRegion();
    mapped_[p] = true;
    mapped_bytes_.fetch_add(kPageBytes, std::memory_order_relaxed);
    adopted[p] = true;
    if (run_end != p)
      flush();
    if (run_end == run_begin)
      run_begin = p;
    run_end = p + 1;
  }
  flush();
  (void)hipGetLastError();
  if (remapped)
    RemapBarrier::NoteRemap();
  const std::size_t backed = MappedPrefixRows();
  requested_rows_ = backed;
  backed_rows_.store(backed, std::memory_order_release);
  shared_end_.store(SharedEnd(), std::memory_order_release);
  return adopted;
}

void Region::TrimTo(std::size_t rows) {
  if (!pool_)
    return;
  rows = std::min(rows, shape_.rows);
  std::lock_guard lock(mutex_);
  std::vector<bool> drop(pages_.size(), true);
  for (std::size_t s = 0; s < shape_.segments && rows != 0; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t end =
        (start + rows * shape_.row_bytes + kPageBytes - 1) / kPageBytes;
    for (std::size_t p = start / kPageBytes; p < end; ++p)
      drop[p] = false;
  }
  bool released = false;
  for (std::size_t p = 0; p < pages_.size() && !released; ++p)
    released = mapped_[p] && drop[p];
  if (!released)
    return;
  RemapBarrier::Hold();
  UnmapMarked(drop);
  RemapBarrier::Release();
  (void)hipGetLastError();
  requested_rows_ = std::min(requested_rows_, rows);
  backed_rows_.store(std::min(backed_rows(), rows), std::memory_order_release);
  SettleSharedEnd();
}

bool Region::MapPages(std::size_t rows, std::string* error, bool ahead) {
  auto& ledger = StateLedger::Process();
  const auto& access = pool_->access();
  std::size_t run_begin = 0, run_end = 0;
  const auto flush = [&]() {
    if (run_end == run_begin)
      return true;
    const bool ok = hipMemSetAccess(base_ + run_begin * kPageBytes,
                                    (run_end - run_begin) * kPageBytes, &access,
                                    1) == hipSuccess;
    run_begin = run_end = 0;
    return ok;
  };
  std::vector<std::size_t> recycled;
  std::size_t backed = shape_.rows;
  bool remapped = false;
  for (std::size_t s = 0; s < shape_.segments; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t first =
        (start + requested_rows_ * shape_.row_bytes) / kPageBytes;
    std::size_t end =
        (start + rows * shape_.row_bytes + kPageBytes - 1) / kPageBytes;
    for (std::size_t p = first; p < end; ++p) {
      if (mapped_[p])
        continue;
      // An address that lost a page since the device was last idle reaches
      // its new page only after the barrier. The writer runs it below; the
      // pool's thread, beside running work, stops here and leaves the rest
      // to the writer.
      if (!RemapBarrier::Clean(unmapped_at_[p])) {
        if (ahead) {
          end = p;
          break;
        }
        remapped = true;
      }
      // A session under a capped budget maps only what it may charge.
      if (account_ && !account_->OnMap(kPageBytes)) {
        (void)flush();
        SetError(error, "state memory exhausted");
        return false;
      }
      hipMemGenericAllocationHandle_t handle{};
      bool reused = false;
      if (!pool_->Acquire(&handle, &reused)) {
        if (account_)
          account_->OnUnmap(kPageBytes);
        (void)flush();
        SetError(error, "state memory exhausted");
        return false;
      }
      if (hipMemMap(base_ + p * kPageBytes, kPageBytes, 0, handle, 0) !=
          hipSuccess) {
        (void)hipGetLastError();
        pool_->Release(handle);
        if (account_)
          account_->OnUnmap(kPageBytes);
        (void)flush();
        SetError(error, "state page map failed");
        return false;
      }
      pages_[p] = handle;
      mapped_[p] = true;
      written_here_[p] = true;
      shared_until_[p] = p * kPageBytes;
      if (reused)
        recycled.push_back(p);
      ledger.AddMapped(kPageBytes);
      mapped_bytes_.fetch_add(kPageBytes, std::memory_order_relaxed);
      if (run_end != p && !flush()) {
        (void)hipGetLastError();
        SetError(error, "state page access failed");
        return false;
      }
      if (run_end == run_begin)
        run_begin = p;
      run_end = p + 1;
    }
    backed =
        std::min(backed, CoveredRows(std::max(end * kPageBytes, start), s));
  }
  if (!flush()) {
    (void)hipGetLastError();
    SetError(error, "state page access failed");
    return false;
  }
  if (remapped) {
    RemapBarrier::NoteRemap();
    if (!pool_->SettleRemaps()) {
      SetError(error, "state page copy failed");
      return false;
    }
  }
  // New pages are zero, as the zero-filled buffers they replace were. A
  // recycled page holds another region's rows, possibly of another type:
  // tile reads masked past a frontier must never meet NaN patterns there.
  if (!recycled.empty()) {
    // Device work of this thread's own when it is the pool's.
    const DeviceWorkScope work;
    const hipStream_t stream = pool_->ZeroStream();
    bool zeroed = stream != nullptr;
    for (const std::size_t p : recycled)
      zeroed = zeroed && hipMemsetAsync(base_ + p * kPageBytes, 0, kPageBytes,
                                        stream) == hipSuccess;
    if (!zeroed || hipStreamSynchronize(stream) != hipSuccess) {
      (void)hipGetLastError();
      SetError(error, "state page clear failed");
      return false;
    }
  }
  requested_rows_ = backed;
  backed_rows_.store(backed, std::memory_order_release);
  return true;
}

std::size_t Region::AheadLimit(std::size_t rows) const {
  rows = std::min(rows, shape_.rows);
  // Only sessions that share their state claim a need; the others map
  // ahead freely, as without accounts.
  if (account_ && StateLedger::Process().PromiseCap() != 0)
    rows = std::min(rows, account_->RowsFor(account_->need_tokens(),
                                            tokens_per_row_, shape_.rows));
  return rows;
}

void Region::MapAhead(std::size_t rows) {
  rows = AheadLimit(rows);
  if (pool_ && rows > backed_rows())
    pool_->MapAhead(shared_from_this(), rows);
}

}  // namespace gufo::mem
