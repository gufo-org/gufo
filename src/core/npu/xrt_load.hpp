// The XRT backend (xrt_backend.hpp). Linux loads Gufo's XRT module on first
// use; Windows binds the delay-loaded xrt_coreutil.dll, the copy the AMD NPU
// driver installs in System32 next to its device shim. Either way Gufo starts
// on machines without XRT.
#pragma once

#include "src/core/npu/xrt_backend.hpp"

namespace gufo::npu {

/// The backend, loaded once. Throws Error saying why XRT is unavailable.
[[nodiscard]] const GufoNpuXrt& Xrt();

}  // namespace gufo::npu
