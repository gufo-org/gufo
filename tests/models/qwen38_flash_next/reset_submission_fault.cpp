// Process-local submission failure; never resets or loses the GPU. Linked
// only into the model-local adapter test, not the model or serving library.
#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <atomic>

namespace {
std::atomic<bool> fail_next{false};
}
void FailNextResetSubmission() {
  fail_next = true;
}

extern "C" hipError_t hipMemsetAsync(void* pointer, int value,
                                     std::size_t bytes, hipStream_t stream) {
  if (fail_next.exchange(false))
    return hipErrorInvalidValue;
  using Function = hipError_t (*)(void*, int, std::size_t, hipStream_t);
  static const auto next =
      reinterpret_cast<Function>(::dlsym(RTLD_NEXT, "hipMemsetAsync"));
  return next ? next(pointer, value, bytes, stream) : hipErrorUnknown;
}
