#include "src/core/hip/remap_barrier_device.hpp"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <mutex>

#include "src/mem/remap_barrier.hpp"

namespace gufo::mem {
namespace {

// Large enough that the HIP runtime takes it from the driver.
constexpr std::size_t kFlushBlockBytes = std::size_t{4} << 20;

void DrainDevice() noexcept {
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
}

// The block the previous flush allocated, freed by the next one.
std::mutex flush_mutex;
void* held_block = nullptr;

}  // namespace

bool FlushTranslations() noexcept {
  const std::lock_guard lock(flush_mutex);
  bool flushed = false;
  if (held_block != nullptr) {
    flushed = hipFree(held_block) == hipSuccess;
    (void)hipGetLastError();
    // A block that could not be freed stays held: no second one.
    if (flushed)
      held_block = nullptr;
  }
  if (held_block == nullptr) {
    // Allocating flushes as well, and gives the next flush its block.
    if (hipMalloc(&held_block, kFlushBlockBytes) == hipSuccess)
      flushed = true;
    else
      held_block = nullptr;
    (void)hipGetLastError();
  }
  return flushed;
}

void InstallRemapBarrierDevice() noexcept {
  RemapBarrier::SetDeviceDrain(&DrainDevice);
  RemapBarrier::SetTranslationFlush(&FlushTranslations);
}

}  // namespace gufo::mem
