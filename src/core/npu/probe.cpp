// The NPUs of this machine, from the kernel's amdxdna accel nodes, and the
// XRT module (libgufo_npu_xrt.so), loaded on first NPU use. Gufo itself does
// not link XRT: it starts without XRT, and XRT's library directory, which
// AMD's packages fill with their own copies of common libraries, never
// becomes a search path for Gufo's other libraries.

#include <filesystem>
#include <string>
#include <vector>

#include "src/core/npu/npu.hpp"
#include "src/core/npu/xrt_load.hpp"
#include "src/core/platform/library.hpp"
#include "src/core/platform/module.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::npu {
namespace {

constexpr const char* kModule = "libgufo_npu_xrt.so";

const GufoNpuXrt* Bind(void* module) {
  auto* entry = reinterpret_cast<const GufoNpuXrt* (*)()>(
      platform::ModuleSymbol(module, GUFO_NPU_XRT_SYMBOL));
  const GufoNpuXrt* api = entry ? entry() : nullptr;
  if (!api || api->version != GUFO_NPU_XRT_VERSION)
    throw Error(std::string(kModule) + " is from another Gufo build");
  return api;
}

// <prefix>/lib/gufo beside the executable's bin, then the build tree that
// compiled Gufo, then the loader's search path (the portable artifact runs
// Gufo through a bundled ELF loader whose --library-path holds the module).
// A module that is present but fails to load usually lacks XRT itself.
const GufoNpuXrt* Load() {
  std::vector<std::filesystem::path> candidates;
  try {
    const auto exe = platform::ExecutablePath();
    candidates.push_back(exe.parent_path().parent_path() / "lib" / "gufo" /
                         kModule);
  } catch (const std::exception&) {
  }
#ifdef GUFO_NPU_XRT_BUILD_MODULE
  candidates.emplace_back(GUFO_NPU_XRT_BUILD_MODULE);
#endif
  for (const auto& path : candidates) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
      continue;
    try {
      return Bind(platform::LoadModule(path));
    } catch (const platform::LibraryError& e) {
      throw Error(std::string("XRT is not usable: ") + e.what());
    }
  }
  try {
    return Bind(platform::LoadModule(kModule));
  } catch (const platform::LibraryError& e) {
    throw Error(std::string("XRT is not usable: ") + e.what());
  }
}

}  // namespace

const GufoNpuXrt& Xrt() {
  // Thread-safe; a load that throws is retried on the next call.
  static const GufoNpuXrt* const loaded = Load();
  return *loaded;
}

std::vector<DeviceInfo> ProbeDevices() {
  return ProbeDevices("/sys");
}

}  // namespace gufo::npu
