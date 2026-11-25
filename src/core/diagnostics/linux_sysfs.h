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

/// The amdgpu GTT pool: the system memory the GPU can map
/// (`mem_info_gtt_total`, `mem_info_gtt_used`).
struct GttInfo {
  std::uint64_t total_bytes{0};
  std::uint64_t used_bytes{0};
};

/// The GTT size docs/install.md recommends: RAM less 4 GiB.
[[nodiscard]] constexpr std::uint64_t RecommendedGttBytes(
    std::uint64_t ram_bytes) noexcept {
  constexpr std::uint64_t kSystemShare = 4ull << 30;
  return ram_bytes > kSystemShare ? ram_bytes - kSystemShare : 0;
}

/// True when `gtt_bytes` is more than 1 GiB below the recommended size; the
/// slack absorbs MiB rounding of `amdgpu.gttsize`.
[[nodiscard]] constexpr bool GttBelowRecommended(
    std::uint64_t gtt_bytes, std::uint64_t ram_bytes) noexcept {
  return gtt_bytes + (1ull << 30) < RecommendedGttBytes(ram_bytes);
}

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
  /// Absent when the driver does not expose the GTT counters.
  std::optional<GttInfo> gtt;
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

  /// GTT of the first DRM card that reports it; nullopt without amdgpu sysfs
  /// (Windows, other drivers). Reads only the counters, so a log line can
  /// afford it.
  [[nodiscard]] std::optional<GttInfo> QueryGtt() const;

  /// Compute units of the KFD topology node at PCI slot `pci_slot`
  /// (`DDDD:BB:DD.F`), or nullopt when KFD exposes no matching GPU node.
  [[nodiscard]] std::optional<std::uint32_t> QueryKfdComputeUnits(
      std::string_view pci_slot) const;

  [[nodiscard]] std::optional<std::string> ReadFile(
      const std::filesystem::path& relative_or_absolute_path) const;

private:
  [[nodiscard]] std::optional<GttInfo> ReadGtt(
      const std::filesystem::path& device_dir) const;

  std::filesystem::path sys_root_;
  std::filesystem::path proc_root_;
};

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_LINUX_SYSFS_H_
