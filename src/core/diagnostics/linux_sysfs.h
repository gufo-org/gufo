#ifndef GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_
#define GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::diagnostics {

struct HostUnameInfo {
  std::string sysname;
  std::string nodename;
  std::string release;
  std::string version;
  std::string machine;
};

struct HostCpuInfo {
  std::string model_name;
  std::string vendor_id;
  std::string architecture{"x86_64"};
  std::uint32_t logical_cores{0};
  std::uint32_t physical_cores{0};
  std::uint32_t numa_nodes{1};
};

struct HostMemInfo {
  std::uint64_t total_bytes{0};
  std::uint64_t available_bytes{0};
  std::uint64_t free_bytes{0};
  bool is_unified_memory{true};
};

struct SysfsGpuDevice {
  std::string card_name;
  std::string pci_id;
  std::string pci_slot;
  std::string driver;
  std::string vendor_id;
  std::string device_id;
  std::string power_mode;
  /// Compute units from the matching KFD topology node, as `rocminfo` and
  /// ROCr report them (`simd_count / simd_per_cu`); 0 when no node matches.
  /// HIP's `multiProcessorCount` halves this in WGP mode, the RDNA default.
  std::uint32_t compute_units{0};
  std::map<std::string, std::string> hwmon_sensors;
};

class LinuxSysfs {
public:
  explicit LinuxSysfs(const std::filesystem::path& sys_root = "/sys",
                      const std::filesystem::path& proc_root = "/proc");

  [[nodiscard]] std::optional<HostUnameInfo> QueryUname() const;
  [[nodiscard]] std::optional<HostCpuInfo> QueryCpuInfo() const;
  [[nodiscard]] std::optional<HostMemInfo> QueryMemInfo() const;
  [[nodiscard]] std::vector<SysfsGpuDevice> QueryDrmGpuDevices() const;

  /// Compute units of the KFD topology node at PCI slot `pci_slot`
  /// (`DDDD:BB:DD.F`), or nullopt when KFD exposes no matching GPU node.
  [[nodiscard]] std::optional<std::uint32_t> QueryKfdComputeUnits(
      std::string_view pci_slot) const;

  [[nodiscard]] std::optional<std::string> ReadFile(
      const std::filesystem::path& relative_or_absolute_path) const;

private:
  std::filesystem::path sys_root_;
  std::filesystem::path proc_root_;
};

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_
