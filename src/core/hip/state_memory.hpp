#ifndef GUFO_CORE_HIP_STATE_MEMORY_HPP_
#define GUFO_CORE_HIP_STATE_MEMORY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/core/hip/state_page_pool.hpp"
#include "src/mem/ledger.hpp"

namespace gufo::mem {

/// Whether state regions on the current device take pages as rows are
/// written, rather than their whole size at creation (no virtual memory
/// management, or device pages that are not host RAM).
[[nodiscard]] inline bool DemandMapped() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return PagePool::ForDevice(device) != nullptr;
}

/// hipMemGetInfo, less the state sessions were promised but have not
/// mapped yet. Every capacity decision reads free memory through this, so
/// it sees what it saw when sessions allocated their whole context up front.
/// Logging the device's raw usage keeps hipMemGetInfo.
[[nodiscard]] inline hipError_t MemGetInfo(std::size_t* free_bytes,
                                           std::size_t* total_bytes) {
  const hipError_t status = hipMemGetInfo(free_bytes, total_bytes);
  if (status == hipSuccess)
    *free_bytes = static_cast<std::size_t>(LessOutstanding(*free_bytes));
  return status;
}

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_MEMORY_HPP_
