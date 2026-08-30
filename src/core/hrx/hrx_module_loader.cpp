#include "src/core/hrx/hrx_module_loader.hpp"

#include "src/core/hrx/hrx_utils.hpp"

namespace gufo::hrx {

HrxModuleLoader::~HrxModuleLoader() {
  UnloadAll();
}

hrx_status_t HrxModuleLoader::LoadFromFile(hrx_device_t device,
                                           const std::string& name,
                                           const std::string& filepath,
                                           const char* target_family,
                                           const char* target_key) {
  hrx_executable_t exec = nullptr;
  hrx_status_t status = hrx_executable_load_file(
      device, filepath.c_str(), target_family, target_key, &exec);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  auto it = executables_.find(name);
  if (it != executables_.end()) {
    hrx_executable_release(it->second);
  }

  executables_[name] = exec;
  return hrx_ok_status();
}

hrx_status_t HrxModuleLoader::LoadFromMemory(hrx_device_t device,
                                             const std::string& name,
                                             std::span<const uint8_t> data,
                                             const char* target_family,
                                             const char* target_key) {
  hrx_executable_t exec = nullptr;
  hrx_status_t status = hrx_executable_load_data(
      device, data.data(), data.size(), target_family, target_key, &exec);
  if (!hrx_status_is_ok(status)) {
    return status;
  }

  auto it = executables_.find(name);
  if (it != executables_.end()) {
    hrx_executable_release(it->second);
  }

  executables_[name] = exec;
  return hrx_ok_status();
}

hrx_executable_t HrxModuleLoader::GetExecutable(const std::string& name) const {
  auto it = executables_.find(name);
  if (it != executables_.end()) {
    return it->second;
  }
  return nullptr;
}

void HrxModuleLoader::UnloadAll() {
  for (auto& [name, exec] : executables_) {
    if (exec != nullptr) {
      hrx_executable_release(exec);
    }
  }
  executables_.clear();
}

}  // namespace gufo::hrx
