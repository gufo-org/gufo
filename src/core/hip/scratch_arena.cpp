#include "src/core/hip/scratch_arena.hpp"

#include <algorithm>

#include "src/core/hip/state_page_pool.hpp"
#include "src/mem/remap_barrier.hpp"

namespace gufo::mem {
namespace {

void SetError(std::string* error, std::string message) {
  if (error)
    *error = std::move(message);
}

[[nodiscard]] std::size_t Pages(std::size_t bytes) {
  return (bytes + ScratchArena::kPage - 1) / ScratchArena::kPage;
}

}  // namespace

std::unique_ptr<ScratchArena> ScratchArena::Reserve(std::size_t capacity,
                                                    std::string* error) {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    return nullptr;
  }
  // Demand mapping is on exactly where the state pool is: the same
  // allocation properties, and device pages that are host RAM.
  const auto pool = PagePool::ForDevice(device);
  if (!pool || capacity == 0)
    return nullptr;
  std::unique_ptr<ScratchArena> arena(new ScratchArena());
  arena->prop_ = pool->prop();
  arena->access_ = pool->access();
  arena->capacity_ = Pages(capacity) * kPage;
  void* base = nullptr;
  if (hipMemAddressReserve(&base, arena->capacity_, kPage, nullptr, 0) !=
      hipSuccess) {
    (void)hipGetLastError();
    SetError(error, "scratch address reservation failed");
    return nullptr;
  }
  arena->base_ = static_cast<std::uint8_t*>(base);
  arena->resident_.assign(arena->capacity_ / kPage, false);
  return arena;
}

ScratchArena::~ScratchArena() {
  if (base_ == nullptr)
    return;
  ReleaseTransient();
  for (const Run& run : resident_runs_)
    UnmapRun(run);
  (void)hipMemAddressFree(base_, capacity_);
  (void)hipGetLastError();
}

bool ScratchArena::MapRun(std::size_t first, std::size_t pages, Run* run,
                          std::string* error) {
  const std::size_t bytes = pages * kPage;
  if (hipMemCreate(&run->handle, bytes, &prop_, 0) != hipSuccess) {
    (void)hipGetLastError();
    SetError(error, "scratch page allocation failed");
    return false;
  }
  void* at = base_ + first * kPage;
  if (hipMemMap(at, bytes, 0, run->handle, 0) != hipSuccess) {
    (void)hipGetLastError();
    (void)hipMemRelease(run->handle);
    SetError(error, "scratch page map failed");
    return false;
  }
  if (hipMemSetAccess(at, bytes, &access_, 1) != hipSuccess) {
    (void)hipGetLastError();
    (void)hipMemUnmap(at, bytes);
    (void)hipMemRelease(run->handle);
    SetError(error, "scratch page access failed");
    return false;
  }
  run->first = first;
  run->pages = pages;
  return true;
}

void ScratchArena::UnmapRun(const Run& run) noexcept {
  (void)hipMemUnmap(base_ + run.first * kPage, run.pages * kPage);
  (void)hipMemRelease(run.handle);
  (void)hipGetLastError();
}

void* ScratchArena::Take(std::size_t bytes, std::size_t resident,
                         std::string* error) {
  const std::size_t pages = Pages(std::max<std::size_t>(bytes, 1));
  if (transient_mapped_ || pages > resident_.size() - used_pages_) {
    SetError(error, "scratch arena exhausted");
    return nullptr;
  }
  void* pointer = base_ + used_pages_ * kPage;
  used_pages_ += pages;
  if (resident != 0 &&
      !MapResident(pointer, std::min(resident, pages * kPage), error))
    return nullptr;
  return pointer;
}

bool ScratchArena::MapResident(const void* pointer, std::size_t bytes,
                               std::string* error) {
  const auto* at = static_cast<const std::uint8_t*>(pointer);
  if (bytes == 0)
    return true;
  if (transient_mapped_ || at < base_ ||
      at + bytes > base_ + used_pages_ * kPage) {
    SetError(error, "scratch resident range is outside the arena");
    return false;
  }
  const std::size_t first = static_cast<std::size_t>(at - base_) / kPage;
  const std::size_t end =
      (static_cast<std::size_t>(at - base_) + bytes + kPage - 1) / kPage;
  for (std::size_t p = first; p < end;) {
    if (resident_[p]) {
      ++p;
      continue;
    }
    std::size_t q = p;
    while (q < end && !resident_[q])
      ++q;
    Run run;
    if (!MapRun(p, q - p, &run, error))
      return false;
    resident_runs_.push_back(run);
    std::fill(resident_.begin() + static_cast<std::ptrdiff_t>(p),
              resident_.begin() + static_cast<std::ptrdiff_t>(q), true);
    resident_pages_ += q - p;
    p = q;
  }
  return true;
}

std::size_t ScratchArena::transient_bytes() const noexcept {
  return (used_pages_ - resident_pages_) * kPage;
}

bool ScratchArena::MapTransient(std::string* error) {
  if (transient_mapped_)
    return true;
  for (std::size_t p = 0; p < used_pages_;) {
    if (resident_[p]) {
      ++p;
      continue;
    }
    std::size_t q = p;
    while (q < used_pages_ && !resident_[q])
      ++q;
    Run run;
    if (!MapRun(p, q - p, &run, error)) {
      ReleaseTransient();
      return false;
    }
    transient_runs_.push_back(run);
    p = q;
  }
  transient_mapped_ = true;
  // The addresses had other pages until the last pass released them, and
  // the device goes on reaching those, which the driver has freed, until it
  // has been idle (RemapBarrier): wait for that before the pass uses the
  // rows. About 0.5 ms, once per wide pass.
  if (!transient_runs_.empty() && !RemapBarrier::Clean(released_at_))
    RemapBarrier::Settle();
  return true;
}

void ScratchArena::ReleaseTransient() noexcept {
  if (!transient_runs_.empty())
    released_at_ = RemapBarrier::Stamp();
  for (const Run& run : transient_runs_)
    UnmapRun(run);
  transient_runs_.clear();
  transient_mapped_ = false;
}

}  // namespace gufo::mem
