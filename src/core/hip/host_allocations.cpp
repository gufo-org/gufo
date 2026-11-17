#include "src/core/hip/host_allocations.hpp"

#include "src/core/platform/page_pin.hpp"
#include "src/core/platform/pages.hpp"

namespace gufo::hip {

HostAllocations::~HostAllocations() {
  Reset();
}

hipError_t HostAllocations::Allocate(void** pointer, std::size_t bytes,
                                     unsigned flags) {
  if (!pointer)
    return hipErrorInvalidValue;
  *pointer = nullptr;
  if (!bytes)
    return hipSuccess;
  // Anonymous, pinned, then registered: see the class comment.
  void* data = platform::CommitZeroedPages(bytes);
  if (!data)
    return hipErrorOutOfMemory;
  const int pin = platform::PinHostPages(data, bytes);
  if (pin < 0) {
    // Without a long-term pin (a low RLIMIT_MEMLOCK, or no such facility),
    // registered anonymous pages evict the GPU queues under compaction for
    // longer than hipHostMalloc pages do, so keep HIP's own allocation.
    platform::ReleaseCommittedPages(data, bytes);
    const auto status = hipHostMalloc(&data, bytes, flags);
    if (status != hipSuccess)
      return status;
    try {
      allocations_.push_back({data, bytes, pin});
    } catch (const std::bad_alloc&) {
      (void)hipHostFree(data);
      return hipErrorOutOfMemory;
    }
    *pointer = data;
    return hipSuccess;
  }
  const auto status = hipHostRegister(data, bytes, hipHostRegisterMapped);
  if (status != hipSuccess) {
    platform::UnpinHostPages(pin);
    platform::ReleaseCommittedPages(data, bytes);
    return status;
  }
  try {
    allocations_.push_back({data, bytes, pin});
  } catch (const std::bad_alloc&) {
    (void)hipHostUnregister(data);
    platform::UnpinHostPages(pin);
    platform::ReleaseCommittedPages(data, bytes);
    return hipErrorOutOfMemory;
  }
  *pointer = data;
  return hipSuccess;
}

void HostAllocations::Reset() noexcept {
  for (const auto& allocation : allocations_) {
    // A pinned buffer is registered anonymous memory; an unpinned one is
    // hipHostMalloc memory.
    if (allocation.pin < 0) {
      (void)hipHostFree(allocation.data);
      continue;
    }
    (void)hipHostUnregister(allocation.data);
    platform::UnpinHostPages(allocation.pin);
    platform::ReleaseCommittedPages(allocation.data, allocation.bytes);
  }
  allocations_.clear();
}

std::size_t HostAllocations::PinnedBytes() const noexcept {
  std::size_t bytes{};
  for (const auto& allocation : allocations_)
    if (allocation.pin >= 0)
      bytes += allocation.bytes;
  return bytes;
}

}  // namespace gufo::hip
