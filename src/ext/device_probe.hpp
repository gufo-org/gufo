#pragma once

#include <chrono>
#include <memory>
#include <string_view>

namespace gufo::ext {

enum class DeviceProbeStatus { kUsable, kPending, kLost };

/// Default execution-device probe for a runner: owns a four-byte device buffer
/// and a private stream from construction, so probing a lost context needs no
/// allocation. Device context loss is sticky: after a reset even this memset
/// fails with a hard error. Only such an error marks the device lost; a memset
/// still pending at the deadline may be queued behind long kernels on the
/// shared hardware queues, so it counts as usable.
///
/// The class is HIP-free in the header; builds without HIP report a usable
/// device and allocate nothing.
class DeviceProbe {
public:
  DeviceProbe();
  ~DeviceProbe();

  DeviceProbe(const DeviceProbe&) = delete;
  DeviceProbe& operator=(const DeviceProbe&) = delete;
  DeviceProbe(DeviceProbe&&) = delete;
  DeviceProbe& operator=(DeviceProbe&&) = delete;

  /// Submit or poll the private probe without waiting. Pending work is reused,
  /// never queued again.
  [[nodiscard]] DeviceProbeStatus Poll() const;

  /// Poll until the device is usable or `bound` elapses, reporting the wait
  /// under `log_component`. A probe still pending at the bound is usable.
  [[nodiscard]] bool WaitUsable(std::chrono::seconds bound,
                                std::string_view log_component) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::ext
