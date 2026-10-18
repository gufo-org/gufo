#include "src/core/hip/host_allocations.hpp"

#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <new>

namespace gufo::hip {

int PinHostPages(void* data, std::size_t bytes) noexcept {
  if (!data || !bytes)
    return -1;
  // Bound each kernel pinning range so request-side CPU page faults can
  // progress between ranges. One ring still owns the complete allocation.
  constexpr std::size_t max_buffer = 8ULL << 20;
  std::vector<iovec> ranges;
  try {
    for (std::size_t offset = 0; offset < bytes;) {
      const auto length = std::min(max_buffer, bytes - offset);
      ranges.push_back({static_cast<std::byte*>(data) + offset, length});
      offset += length;
    }
  } catch (const std::bad_alloc&) {
    return -1;
  }
  io_uring_params parameters{};
  const int ring = syscall(SYS_io_uring_setup, 1, &parameters);
  if (ring < 0)
    return -1;
  if (syscall(SYS_io_uring_register, ring, IORING_REGISTER_BUFFERS,
              ranges.data(), ranges.size()) == 0)
    return ring;
  close(ring);
  return -1;
}

void UnpinHostPages(int ring) noexcept {
  if (ring >= 0) {
    (void)syscall(SYS_io_uring_register, ring, IORING_UNREGISTER_BUFFERS,
                  nullptr, 0);
    close(ring);
  }
}

HostAllocations::~HostAllocations() {
  Reset();
}

hipError_t HostAllocations::Allocate(void** pointer, std::size_t bytes,
                                     unsigned flags) {
  if (!pointer)
    return hipErrorInvalidValue;
  *pointer = nullptr;
  void* data{};
  const auto status = hipHostMalloc(&data, bytes, flags);
  if (status != hipSuccess)
    return status;
  if (!data)
    return hipSuccess;
  const int pin = PinHostPages(data, bytes);
  try {
    allocations_.push_back({data, bytes, pin});
  } catch (const std::bad_alloc&) {
    UnpinHostPages(pin);
    (void)hipHostFree(data);
    return hipErrorOutOfMemory;
  }
  *pointer = data;
  return hipSuccess;
}

void HostAllocations::Reset() noexcept {
  for (const auto& allocation : allocations_) {
    UnpinHostPages(allocation.pin);
    (void)hipHostFree(allocation.data);
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
