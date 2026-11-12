// The NPUs of this machine, as XRT enumerates them: Windows has no sysfs, and
// the NPU driver exposes its devices through XRT.

#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>

#include <exception>

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

std::vector<DeviceInfo> ProbeDevices() {
  std::vector<DeviceInfo> out;
  try {
    const unsigned count = xrt::system::enumerate_devices();
    for (unsigned i = 0; i < count; ++i) {
      DeviceInfo info;
      info.node = "xrt" + std::to_string(i);
      info.driver = xrt::device(i).get_info<xrt::info::device::name>();
      out.push_back(std::move(info));
    }
  } catch (const std::exception&) {
    // No NPU driver: no devices.
  }
  return out;
}

}  // namespace gufo::npu
