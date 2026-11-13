// The NPUs of this machine, from the kernel's amdxdna accel nodes.

#include "src/core/npu/npu.hpp"
#include "src/core/npu/xrt_load.hpp"

namespace gufo::npu {

bool LoadXrt() {
  return true;  // linked normally
}

std::vector<DeviceInfo> ProbeDevices() {
  return ProbeDevices("/sys");
}

}  // namespace gufo::npu
