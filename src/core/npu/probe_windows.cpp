// The NPUs of this machine, as XRT enumerates them: Windows has no sysfs, and
// the NPU driver exposes its devices through XRT.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h first: delayimp.h uses its types.
#include <delayimp.h>
#include <xrt/experimental/xrt_system.h>
#include <xrt/xrt_device.h>

#include <exception>

#include "src/core/npu/npu.hpp"
#include "src/core/npu/xrt_load.hpp"

namespace gufo::npu {
namespace {

int DelayLoadFilter(DWORD code) {
  const DWORD no_dll = VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND);
  const DWORD no_proc =
      VcppException(ERROR_SEVERITY_ERROR, ERROR_PROC_NOT_FOUND);
  return code == no_dll || code == no_proc ? EXCEPTION_EXECUTE_HANDLER
                                           : EXCEPTION_CONTINUE_SEARCH;
}

// Binds every XRT import now; the delay-load helper raises an SEH exception
// when the DLL or one of its functions is missing. No C++ objects here, so
// __try is allowed.
bool BindXrtImports() {
  __try {
    return SUCCEEDED(__HrLoadAllImportsForDll("xrt_coreutil.dll"));
  } __except (DelayLoadFilter(GetExceptionCode())) {
    return false;
  }
}

}  // namespace

bool LoadXrt() {
  static const bool loaded = BindXrtImports();
  return loaded;
}

std::vector<DeviceInfo> ProbeDevices() {
  std::vector<DeviceInfo> out;
  if (!LoadXrt())
    return out;  // no NPU driver
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
