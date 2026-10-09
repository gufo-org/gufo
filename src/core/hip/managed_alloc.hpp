#ifndef GUFO_CORE_HIP_MANAGED_ALLOC_HPP_
#define GUFO_CORE_HIP_MANAGED_ALLOC_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

/// True when the HIP device heap is a BIOS carve-out. The carve-out is stolen
/// from the OS before boot, so the heap is at least MemTotal. With no
/// carve-out the runtime heap is a cap below MemTotal.
[[nodiscard]] inline bool DeviceHeapIsCarveOut(
    std::uint64_t device_total_bytes, std::uint64_t host_total_bytes) {
  return host_total_bytes != 0 && device_total_bytes >= host_total_bytes;
}

/// Allocate device memory, preferring the device pool (hipMalloc) and
/// falling back to hipMallocManaged when the pool is exhausted. On the
/// gfx1151 APU the runtime-reported device pool is capped below physical
/// unified memory (a RAM/2 + VRAM heuristic, unaffected by amdgpu.gttsize),
/// while managed allocations draw on the full system pool.
/// hipMalloc also places the allocation in a BIOS carve-out (when one is
/// configured). Both allocations reside in the same physical DRAM, so
/// kernels see identical memory either way.
[[nodiscard]] inline hipError_t AllocateDevice(void** pointer,
                                               std::size_t bytes) {
  hipError_t status = hipMalloc(pointer, bytes);
  if (status == hipErrorOutOfMemory) {
    // Clear the error as it remains sticky
    (void)hipGetLastError();
    status = hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
  }
  return status;
}

[[nodiscard]] inline bool WeightHeapIsCarveOut() {
  static const bool carve_out = [] {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
      return false;
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
      return false;
    const auto host_bytes = static_cast<std::uint64_t>(pages) *
                            static_cast<std::uint64_t>(page_size);
    return DeviceHeapIsCarveOut(total_bytes, host_bytes);
  }();
  return carve_out;
}

/// Weights for the whole process. A carve-out takes hipMalloc, which is the
/// placement used before managed fallback existed. Otherwise hipMalloc would
/// fill the capped device heap and the next kernel launch would fail inside
/// the driver with no memory left for command submission; those weights use
/// hipMallocManaged and leave the heap free.
[[nodiscard]] inline hipError_t AllocateWeights(void** pointer,
                                                std::size_t bytes) {
  if (WeightHeapIsCarveOut())
    return hipMalloc(pointer, bytes);
  return hipMallocManaged(pointer, bytes, hipMemAttachGlobal);
}

/// Typed-pointer overloads; hipMalloc itself accepts T**, so match it.
template<typename T>
[[nodiscard]] inline hipError_t AllocateDevice(T** pointer, std::size_t bytes) {
  return AllocateDevice(reinterpret_cast<void**>(pointer), bytes);
}

template<typename T>
[[nodiscard]] inline hipError_t AllocateWeights(T** pointer,
                                                std::size_t bytes) {
  return AllocateWeights(reinterpret_cast<void**>(pointer), bytes);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_MANAGED_ALLOC_HPP_
