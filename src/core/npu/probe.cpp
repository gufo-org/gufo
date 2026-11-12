// The NPUs of this machine, from the kernel's amdxdna accel nodes.

#include "src/core/npu/npu.hpp"

namespace gufo::npu {

std::vector<DeviceInfo> ProbeDevices() {
  return ProbeDevices("/sys");
}

}  // namespace gufo::npu
