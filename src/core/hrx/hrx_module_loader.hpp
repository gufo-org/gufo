#ifndef GUFO_CORE_HRX_HRX_MODULE_LOADER_HPP_
#define GUFO_CORE_HRX_HRX_MODULE_LOADER_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hrx/hrx_runtime.h>

namespace gufo::hrx {

class HrxModuleLoader {
public:
  HrxModuleLoader() = default;
  ~HrxModuleLoader();

  HrxModuleLoader(const HrxModuleLoader&) = delete;
  HrxModuleLoader& operator=(const HrxModuleLoader&) = delete;
  HrxModuleLoader(HrxModuleLoader&&) = delete;
  HrxModuleLoader& operator=(HrxModuleLoader&&) = delete;

  hrx_status_t LoadFromFile(hrx_device_t device, const std::string& name,
                            const std::string& filepath,
                            const char* target_family = "amdgpu",
                            const char* target_key = "gfx1151");

  hrx_status_t LoadFromMemory(hrx_device_t device, const std::string& name,
                              std::span<const uint8_t> data,
                              const char* target_family = "amdgpu",
                              const char* target_key = "gfx1151");

  [[nodiscard]] hrx_executable_t GetExecutable(const std::string& name) const;

  void UnloadAll();

private:
  std::unordered_map<std::string, hrx_executable_t> executables_;
};

}  // namespace gufo::hrx

#endif  // GUFO_CORE_HRX_HRX_MODULE_LOADER_HPP_
