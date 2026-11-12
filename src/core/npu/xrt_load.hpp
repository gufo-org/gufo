// Whether XRT can be called. Windows delay-loads xrt_coreutil.dll, the copy
// the AMD NPU driver installs in System32 next to its device shim, so Gufo
// starts on machines without the driver; call this before any XRT function.
#pragma once

namespace gufo::npu {

[[nodiscard]] bool LoadXrt();

}  // namespace gufo::npu
