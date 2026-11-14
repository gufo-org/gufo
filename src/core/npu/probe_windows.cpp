// The NPUs of this machine, as XRT enumerates them: Windows has no sysfs, and
// the NPU driver exposes its devices through XRT. The XRT backend is linked
// in, with xrt_coreutil.dll delay-loaded.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h first: delayimp.h uses its types.
#include <delayimp.h>

#include <string>

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

const GufoNpuXrt& Xrt() {
  static const bool loaded = BindXrtImports();
  if (!loaded)
    throw Error(
        "XRT (xrt_coreutil.dll from the AMD NPU driver) is not installed");
  return *gufo_npu_xrt();
}

std::vector<DeviceInfo> ProbeDevices() {
  std::vector<DeviceInfo> out;
  const GufoNpuXrt* xrt = nullptr;
  try {
    xrt = &Xrt();
  } catch (const Error&) {
    return out;  // no NPU driver
  }
  unsigned count = 0;
  if (xrt->device_count(&count))
    return out;  // no NPU driver: no devices
  for (unsigned i = 0; i < count; ++i) {
    void* device = nullptr;
    if (xrt->device_open(i, &device))
      continue;
    const char* name = nullptr;
    DeviceInfo info;
    info.node = "xrt" + std::to_string(i);
    if (!xrt->device_name(device, &name) && name)
      info.driver = name;
    xrt->device_close(device);
    out.push_back(std::move(info));
  }
  return out;
}

}  // namespace gufo::npu
