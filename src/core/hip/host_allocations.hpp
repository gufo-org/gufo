#ifndef GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_
#define GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

namespace gufo::hip {

// Best-effort Linux long-term pins prevent CPU compaction from moving GPU
// registrations. A restricted io_uring/memlock limit retains ordinary HIP
// registration. The caller must unpin before freeing the host allocation.
[[nodiscard]] int PinHostPages(void* data, std::size_t bytes) noexcept;
void UnpinHostPages(int ring) noexcept;

// Owns HIP host buffers and their long-term pins together. Allocations retain
// HIP's original coherence and huge-page policy. Drain readers and GPU work
// before destroying the owner.
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
