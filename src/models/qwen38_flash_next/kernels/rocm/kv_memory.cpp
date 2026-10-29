#include "src/models/qwen38_flash_next/kernels/rocm/kv_memory.hpp"

#include <algorithm>
#include <utility>

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

std::size_t ChunkBytes(std::size_t total) noexcept {
  const std::size_t granule = KvPool::kGranuleBytes;
  return std::min(KvPool::kChunkBytes,
                  (total + granule - 1) / granule * granule);
}

hipMemAllocationProp DeviceProp() {
  int device = 0;
  (void)hipGetDevice(&device);
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  return prop;
}

bool Fail(std::string* error, const char* what, hipError_t status) {
  (void)hipGetLastError();  // HIP errors are sticky
  if (error)
    *error = std::string(what) + ": " + hipGetErrorString(status);
  return false;
}

}  // namespace

bool KvPool::Supported(std::string* error) {
  int device = 0;
  int supported = 0;
  if (hipGetDevice(&device) != hipSuccess ||
      hipDeviceGetAttribute(&supported,
                            hipDeviceAttributeVirtualMemoryManagementSupported,
                            device) != hipSuccess ||
      supported == 0) {
    (void)hipGetLastError();
    if (error)
      *error = "device lacks virtual memory management for growable K/V";
    return false;
  }
  const auto prop = DeviceProp();
  std::size_t granularity = 0;
  const auto status = hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityMinimum);
  if (status != hipSuccess)
    return Fail(error, "K/V allocation granularity", status);
  if (granularity == 0 || kGranuleBytes % granularity != 0) {
    if (error)
      *error = "K/V chunk size is not a multiple of the device granularity";
    return false;
  }
  return true;
}

std::size_t KvPool::mapped() const {
  const std::lock_guard lock(mutex_);
  return mapped_;
}

std::size_t KvPool::Available() const {
  const std::lock_guard lock(mutex_);
  if (budget_ == 0)
    return SIZE_MAX;
  return budget_ > mapped_ ? budget_ - mapped_ : 0;
}

bool KvPool::Charge(std::size_t bytes) {
  const std::lock_guard lock(mutex_);
  if (budget_ != 0 && (mapped_ > budget_ || bytes > budget_ - mapped_))
    return false;
  mapped_ += bytes;
  return true;
}

void KvPool::Refund(std::size_t bytes) {
  const std::lock_guard lock(mutex_);
  mapped_ -= std::min(mapped_, bytes);
}

KvPool::~KvPool() {
  for (const auto& [base, bytes] : ranges_)
    (void)hipMemAddressFree(base, bytes);
}

void* KvPool::ReserveRange(std::size_t bytes, hipError_t* status) {
  const std::lock_guard lock(mutex_);
  *status = hipSuccess;
  const auto reused = std::ranges::find_if(
      free_ranges_, [&](const auto& range) { return range.second == bytes; });
  if (reused != free_ranges_.end()) {
    void* base = reused->first;
    free_ranges_.erase(reused);
    return base;
  }
  void* base = nullptr;
  *status = hipMemAddressReserve(&base, bytes, 0, nullptr, 0);
  if (*status != hipSuccess)
    return nullptr;
  ranges_.emplace_back(base, bytes);
  return base;
}

void KvPool::ReleaseRange(void* base, std::size_t bytes) {
  const std::lock_guard lock(mutex_);
  free_ranges_.emplace_back(base, bytes);
}

GrowableRows::GrowableRows(GrowableRows&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      base_(std::exchange(other.base_, nullptr)),
      reserved_bytes_(std::exchange(other.reserved_bytes_, 0)),
      chunk_(std::exchange(other.chunk_, 0)),
      row_bytes_(std::exchange(other.row_bytes_, 0)),
      max_rows_(std::exchange(other.max_rows_, 0)),
      chunks_(std::move(other.chunks_)) {}

GrowableRows::~GrowableRows() {
  (void)Shrink(0);
  if (base_ != nullptr)
    pool_->ReleaseRange(base_, reserved_bytes_);
}

bool GrowableRows::Reserve(KvPool* pool, std::size_t row_bytes,
                           std::size_t max_rows, std::string* error) {
  if (base_ != nullptr || pool == nullptr || row_bytes == 0 || max_rows == 0) {
    if (error)
      *error = "invalid growable K/V reservation";
    return false;
  }
  const std::size_t chunk = ChunkBytes(row_bytes * max_rows);
  const std::size_t bytes = FullBytes(row_bytes, max_rows);
  hipError_t status = hipSuccess;
  void* base = pool->ReserveRange(bytes, &status);
  if (base == nullptr)
    return Fail(error, "K/V address reservation", status);
  pool_ = pool;
  base_ = static_cast<std::uint8_t*>(base);
  reserved_bytes_ = bytes;
  chunk_ = chunk;
  row_bytes_ = row_bytes;
  max_rows_ = max_rows;
  return true;
}

std::size_t GrowableRows::FullBytes(std::size_t row_bytes,
                                    std::size_t max_rows) noexcept {
  const std::size_t total = row_bytes * max_rows;
  const std::size_t chunk = ChunkBytes(total);
  return chunk == 0 ? 0 : (total + chunk - 1) / chunk * chunk;
}

std::size_t GrowableRows::rows() const noexcept {
  return row_bytes_ == 0 ? 0 : std::min(max_rows_, mapped_bytes() / row_bytes_);
}

std::size_t GrowableRows::ChunksFor(std::size_t rows) const noexcept {
  const std::size_t bytes = std::min(rows, max_rows_) * row_bytes_;
  return chunk_ == 0 ? 0 : (bytes + chunk_ - 1) / chunk_;
}

std::size_t GrowableRows::GrowthBytes(std::size_t rows) const noexcept {
  const std::size_t want = ChunksFor(rows);
  return want > chunks_.size() ? (want - chunks_.size()) * chunk_ : 0;
}

bool GrowableRows::Grow(std::size_t rows, hipStream_t stream,
                        std::string* error) {
  const std::size_t want = ChunksFor(rows);
  if (want <= chunks_.size())
    return true;
  const std::size_t chunk = chunk_;
  const std::size_t first = chunks_.size();
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(stream, &capture) != hipSuccess ||
      capture != hipStreamCaptureStatusNone) {
    (void)hipGetLastError();
    if (error)
      *error = "K/V rows cannot grow during graph capture";
    return false;
  }
  // New pages map beside rows queued work may be using; those stay mapped,
  // so the queue need not drain (draining cost each prefill pass its overlap).
  hipError_t status = hipSuccess;
  if (!pool_->Charge((want - first) * chunk)) {
    if (error)
      *error = "K/V memory budget exhausted";
    return false;
  }
  const auto prop = DeviceProp();
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  const auto undo = [&](hipError_t status, const char* what) {
    pool_->Refund((want - chunks_.size()) * chunk);
    (void)Shrink(first * chunk / row_bytes_);
    return Fail(error, what, status);
  };
  while (chunks_.size() < want) {
    hipMemGenericAllocationHandle_t handle{};
    status = hipMemCreate(&handle, chunk, &prop, 0);
    if (status != hipSuccess)
      return undo(status, "K/V chunk allocation");
    std::uint8_t* at = base_ + chunks_.size() * chunk;
    status = hipMemMap(at, chunk, 0, handle, 0);
    if (status != hipSuccess) {
      (void)hipMemRelease(handle);
      return undo(status, "K/V chunk mapping");
    }
    chunks_.push_back(handle);
    status = hipMemSetAccess(at, chunk, &access, 1);
    if (status != hipSuccess)
      return undo(status, "K/V chunk access");
  }
  status =
      hipMemsetAsync(base_ + first * chunk, 0, (want - first) * chunk, stream);
  if (status != hipSuccess)
    return undo(status, "K/V chunk zero fill");
  return true;
}

std::size_t GrowableRows::Shrink(std::size_t rows) noexcept {
  const std::size_t keep = ChunksFor(rows);
  std::size_t released = 0;
  while (chunks_.size() > keep) {
    std::uint8_t* at = base_ + (chunks_.size() - 1) * chunk_;
    (void)hipMemUnmap(at, chunk_);
    (void)hipMemRelease(chunks_.back());
    chunks_.pop_back();
    released += chunk_;
  }
  if (released != 0)
    pool_->Refund(released);
  return released;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
