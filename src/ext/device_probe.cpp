#include "src/ext/device_probe.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/logging/logging.hpp"
#endif

namespace gufo::ext {

struct DeviceProbe::Impl {
#if defined(ENGINE_ENABLE_HIP)
  hipStream_t probe_stream_{};
  void* probe_buffer_{};
  mutable bool probe_pending_{false};
#endif
};

DeviceProbe::DeviceProbe() : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_ENABLE_HIP)
  HIP_CHECK(
      hipStreamCreateWithFlags(&impl_->probe_stream_, hipStreamNonBlocking));
  if (const auto error =
          hipMalloc(&impl_->probe_buffer_, sizeof(std::uint32_t));
      error != hipSuccess) {
    hip::LogCleanupError(hipStreamDestroy(impl_->probe_stream_));
    throw std::runtime_error(std::string("device probe allocation: ") +
                             hipGetErrorString(error));
  }
#endif
}

DeviceProbe::~DeviceProbe() {
#if defined(ENGINE_ENABLE_HIP)
  hip::LogCleanupError(hipFree(impl_->probe_buffer_));
  hip::LogCleanupError(hipStreamDestroy(impl_->probe_stream_));
#endif
}

DeviceProbeStatus DeviceProbe::Poll() const {
#if !defined(ENGINE_ENABLE_HIP)
  return DeviceProbeStatus::kUsable;
#else
  if (!impl_->probe_pending_) {
    // Clear the failed work unit's thread-local error before probing.
    (void)hipGetLastError();
    if (hipMemsetAsync(impl_->probe_buffer_, 0, sizeof(std::uint32_t),
                       impl_->probe_stream_) != hipSuccess ||
        hipGetLastError() != hipSuccess)
      return DeviceProbeStatus::kLost;
    impl_->probe_pending_ = true;
  }
  const auto status = hipStreamQuery(impl_->probe_stream_);
  if (status == hipErrorNotReady)
    return DeviceProbeStatus::kPending;
  impl_->probe_pending_ = false;
  return status == hipSuccess ? DeviceProbeStatus::kUsable
                              : DeviceProbeStatus::kLost;
#endif
}

bool DeviceProbe::WaitUsable(std::chrono::seconds bound,
                             std::string_view log_component) const {
  const auto deadline = std::chrono::steady_clock::now() + bound;
  for (;;) {
    const auto status = Poll();
    if (status == DeviceProbeStatus::kUsable)
      return true;
    if (status == DeviceProbeStatus::kLost)
      return false;
    if (std::chrono::steady_clock::now() >= deadline) {
#if defined(ENGINE_ENABLE_HIP)
      server::Logger::Warn(log_component,
                           "event=device_probe_timeout timeout_s=" +
                               std::to_string(bound.count()));
#endif
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace gufo::ext
