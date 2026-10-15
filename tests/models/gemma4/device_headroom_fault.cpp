// Process-local device headroom limit for snapshot storage tests. Linked only
// into the model-local target test, not the model or serving library.
#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <limits>

namespace {
std::atomic<std::size_t> free_limit{std::numeric_limits<std::size_t>::max()};
}  // namespace

/// Caps the free device memory HIP reports; the maximum removes the cap.
void LimitReportedDeviceFree(std::size_t bytes) {
  free_limit = bytes;
}

extern "C" hipError_t hipMemGetInfo(std::size_t* free, std::size_t* total) {
  using Function = hipError_t (*)(std::size_t*, std::size_t*);
  static const auto next =
      reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipMemGetInfo"));
  const auto result = next ? next(free, total) : hipErrorUnknown;
  if (result == hipSuccess && free != nullptr && *free > free_limit)
    *free = free_limit;
  return result;
}
