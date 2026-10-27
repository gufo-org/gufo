#ifndef GUFO_CORE_HIP_DEVICE_HPP_
#define GUFO_CORE_HIP_DEVICE_HPP_

#include <cstddef>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::hip {

#if defined(ENGINE_ENABLE_HIP)

/// Visible HIP device count, or 0 when the runtime cannot be queried.
[[nodiscard]] int GetDeviceCount() noexcept;

/// Current device for this thread, or -1 when the query fails.
[[nodiscard]] int CurrentDevice() noexcept;

/// Makes `device` current for the enclosing scope and restores the previous
/// device on destruction. A no-op when the requested device is already
/// current, so single-device callers pay only one cheap query.
class DeviceGuard final {
public:
  explicit DeviceGuard(int device);
  ~DeviceGuard();
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
  int previous_{-1};
  bool switched_{false};
};

/// Copies `bytes` from a device allocation on `source_device` into a device
/// allocation on `destination_device`. Uses peer-to-peer access when the
/// device pair supports it and stages through pinned host memory otherwise.
/// Synchronous: returns after the destination bytes are visible on
/// `destination_device`. The source device is drained before reading so the
/// copy observes settled work queued on any source stream.
void CrossDeviceCopy(void* destination, int destination_device,
                     const void* source, int source_device, std::size_t bytes);

/// Two-dimensional CrossDeviceCopy: `height` rows of `width_bytes` each,
/// advancing by `source_pitch` / `destination_pitch` between rows.
void CrossDeviceCopy2D(void* destination, int destination_device,
                       std::size_t destination_pitch, const void* source,
                       int source_device, std::size_t source_pitch,
                       std::size_t width_bytes, std::size_t height);

#endif  // defined(ENGINE_ENABLE_HIP)
}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_DEVICE_HPP_
