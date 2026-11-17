#ifndef GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_
#define GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

namespace gufo::hip {

// Owns GPU-visible host buffers and their long-term pins together. Each buffer
// is anonymous memory, pinned long-term, then registered with HIP (coherent,
// mapped; the host pointer is the device pointer). hipHostMalloc pages stay
// candidates for compaction even when pinned: every failed migration
// invalidates their GPU mapping and evicts all of the process's GPU queues,
// which froze the GPU for 45-109 s in a probe under memory pressure. Pinned
// anonymous pages are never isolated for migration. `flags` is kept for
// source compatibility; every buffer is coherent and mapped. Drain readers
// and GPU work before destroying the owner. Page pinning lives in
// platform/page_pin.hpp.
class HostAllocations {
public:
  HostAllocations() = default;
  ~HostAllocations();
  HostAllocations(const HostAllocations&) = delete;
  HostAllocations& operator=(const HostAllocations&) = delete;

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault);
  template<typename T>
  [[nodiscard]] hipError_t Allocate(T** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault) {
    return Allocate(reinterpret_cast<void**>(pointer), bytes, flags);
  }
  void Reset() noexcept;
  [[nodiscard]] std::size_t PinnedBytes() const noexcept;

private:
  struct Allocation {
    void* data;
    std::size_t bytes;
    int pin;
  };
  std::vector<Allocation> allocations_;
};

}  // namespace gufo::hip
#endif
