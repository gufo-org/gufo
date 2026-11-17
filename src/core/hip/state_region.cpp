#include "src/core/hip/state_region.hpp"

#include <algorithm>
#include <limits>

#include "src/core/hip/state_page_pool.hpp"
#include "src/mem/ledger.hpp"

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

}  // namespace

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
  if (hipMemset(data, 0, region->bytes_) != hipSuccess) {
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
  // Unmapping waits for all in-flight GPU work on the device; regions end
  // with their sessions, at shutdown or between requests, never mid-pass.
  for (std::size_t p = 0; p < pages_.size(); ++p)
    if (mapped_[p])
      UnmapPage(p);
  (void)hipGetLastError();
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
  return MapPages(rows, nullptr);
}

bool Region::Ensure(std::size_t rows, std::string* error) {
  rows = std::min(rows, shape_.rows);
  if (backed_rows() >= rows)
    return true;
  std::lock_guard lock(mutex_);
  if (backed_rows() >= rows)
    return true;
  return MapPages(rows, error);
}

void Region::UnmapPage(std::size_t p) {
  (void)hipMemUnmap(base_ + p * kPageBytes, kPageBytes);
  pool_->Release(pages_[p]);
  mapped_[p] = false;
  StateLedger::Process().RemoveMapped(kPageBytes);
  mapped_bytes_.fetch_sub(kPageBytes, std::memory_order_relaxed);
  if (account_)
    account_->OnUnmap(kPageBytes);
}

void Region::TrimTo(std::size_t rows) {
  if (!pool_)
    return;
  rows = std::min(rows, shape_.rows);
  std::lock_guard lock(mutex_);
  std::vector<bool> keep(pages_.size(), false);
  for (std::size_t s = 0; s < shape_.segments && rows != 0; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t end =
        (start + rows * shape_.row_bytes + kPageBytes - 1) / kPageBytes;
    for (std::size_t p = start / kPageBytes; p < end; ++p)
      keep[p] = true;
  }
  bool released = false;
  for (std::size_t p = 0; p < pages_.size(); ++p)
    if (mapped_[p] && !keep[p]) {
      UnmapPage(p);
      released = true;
    }
  if (!released)
    return;
  (void)hipGetLastError();
  requested_rows_ = std::min(requested_rows_, rows);
  backed_rows_.store(std::min(backed_rows(), rows), std::memory_order_release);
}

bool Region::MapPages(std::size_t rows, std::string* error) {
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
  for (std::size_t s = 0; s < shape_.segments; ++s) {
    const std::size_t start = s * stride_;
    const std::size_t first =
        (start + requested_rows_ * shape_.row_bytes) / kPageBytes;
    const std::size_t end =
        (start + rows * shape_.row_bytes + kPageBytes - 1) / kPageBytes;
    for (std::size_t p = first; p < end; ++p) {
      if (mapped_[p])
        continue;
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
    backed = std::min(backed, CoveredRows(end * kPageBytes, s));
  }
  if (!flush()) {
    (void)hipGetLastError();
    SetError(error, "state page access failed");
    return false;
  }
  // New pages are zero, as the zero-filled buffers they replace were. A
  // recycled page holds another region's rows, possibly of another type:
  // tile reads masked past a frontier must never meet NaN patterns there.
  if (!recycled.empty()) {
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
