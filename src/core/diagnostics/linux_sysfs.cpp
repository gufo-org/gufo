#include "src/core/diagnostics/linux_sysfs.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string_view>

#include "src/core/platform/os_info.hpp"

namespace gufo::diagnostics {

namespace {

std::string Trim(std::string_view str) {
  const auto start = str.find_first_not_of(" \t\r\n");
  if (start == std::string_view::npos) {
    return {};
  }
  const auto end = str.find_last_not_of(" \t\r\n");
  return std::string(str.substr(start, end - start + 1));
}

std::uint32_t ParseUint(std::string_view str) {
  std::uint32_t val = 0;
  const auto* const begin = str.data();
  const auto* const finish = begin + str.size();
  std::from_chars(begin, finish, val);
  return val;
}

std::optional<std::uint64_t> ParseUint64(std::string_view str) {
  std::uint64_t val = 0;
  const auto* const begin = str.data();
  const auto* const finish = begin + str.size();
  const auto [end, ec] = std::from_chars(begin, finish, val);
  if (ec != std::errc{} || end != finish) {
    return std::nullopt;
  }
  return val;
}

std::vector<std::filesystem::path> DrmCardDevices(
    const std::filesystem::path& sys_root) {
  std::vector<std::filesystem::path> devices;
  const auto drm_path = sys_root / "class" / "drm";
  std::error_code ec;
  if (!std::filesystem::exists(drm_path, ec)) {
    return devices;
  }
  for (const auto& entry : std::filesystem::directory_iterator(drm_path, ec)) {
    if (ec) {
      break;
    }
    const std::string filename = entry.path().filename().string();
    if (!filename.starts_with("card") ||
        filename.find('-') != std::string::npos) {
      continue;
    }
    auto dev_dir = entry.path() / "device";
    if (std::filesystem::exists(dev_dir, ec)) {
      devices.push_back(std::move(dev_dir));
    }
  }
  // directory_iterator order is unspecified; card0 before card1.
  std::sort(devices.begin(), devices.end());
  return devices;
}

}  // namespace

LinuxSysfs::LinuxSysfs(const std::filesystem::path& sys_root,
                       const std::filesystem::path& proc_root)
    : sys_root_(std::filesystem::absolute(sys_root)),
      proc_root_(std::filesystem::absolute(proc_root)) {}

std::optional<std::string> LinuxSysfs::ReadFile(
    const std::filesystem::path& path) const {
  std::filesystem::path target = path;
  if (!std::filesystem::exists(target)) {
    target = sys_root_ / path;
  }
  std::ifstream file(target);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return Trim(buffer.str());
}

std::optional<HostUnameInfo> LinuxSysfs::QueryUname() const {
  const auto version_opt = ReadFile(proc_root_ / "version");
  const auto host = platform::QueryHostIdentity();
  if (!host) {
    return std::nullopt;
  }
  HostUnameInfo info;
  info.sysname = host->sysname;
  info.nodename = host->nodename;
  info.release = host->release;
  info.version = version_opt ? *version_opt : host->version;
  info.machine = host->machine;
  return info;
}

std::optional<HostCpuInfo> LinuxSysfs::QueryCpuInfo() const {
  const auto content_opt = ReadFile(proc_root_ / "cpuinfo");
  if (!content_opt) {
    return std::nullopt;
  }

  HostCpuInfo cpu_info;
  std::istringstream stream(*content_opt);
  std::string line;
  std::uint32_t processor_count = 0;
  std::set<std::string> physical_ids;

  while (std::getline(stream, line)) {
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, colon_pos));
    const std::string value = Trim(line.substr(colon_pos + 1));

    if (key == "model name" && cpu_info.model_name.empty()) {
      cpu_info.model_name = value;
    } else if (key == "vendor_id" && cpu_info.vendor_id.empty()) {
      cpu_info.vendor_id = value;
    } else if (key == "processor") {
      ++processor_count;
    } else if (key == "siblings" && cpu_info.logical_cores == 0) {
      cpu_info.logical_cores = ParseUint(value);
    } else if (key == "cpu cores" && cpu_info.physical_cores == 0) {
      cpu_info.physical_cores = ParseUint(value);
    } else if (key == "physical id") {
      physical_ids.insert(value);
    }
  }

  if (cpu_info.logical_cores == 0) {
    cpu_info.logical_cores = processor_count;
  }
  if (cpu_info.physical_cores == 0) {
    cpu_info.physical_cores = processor_count > 0 ? (processor_count / 2) : 0;
  }
  cpu_info.numa_nodes = physical_ids.empty()
                            ? 1
                            : static_cast<std::uint32_t>(physical_ids.size());
  return cpu_info;
}

std::optional<HostMemInfo> LinuxSysfs::QueryMemInfo() const {
  const auto content_opt = ReadFile(proc_root_ / "meminfo");
  if (!content_opt) {
    return std::nullopt;
  }

  HostMemInfo mem_info;
  std::istringstream stream(*content_opt);
  std::string line;

  while (std::getline(stream, line)) {
    const auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, colon_pos));
    const std::string rest = Trim(line.substr(colon_pos + 1));

    std::uint64_t val_kb = 0;
    std::istringstream val_stream(rest);
    if (val_stream >> val_kb) {
      if (key == "MemTotal") {
        mem_info.total_bytes = val_kb * 1024ULL;
      } else if (key == "MemAvailable") {
        mem_info.available_bytes = val_kb * 1024ULL;
      } else if (key == "MemFree") {
        mem_info.free_bytes = val_kb * 1024ULL;
      }
    }
  }
  return mem_info;
}

std::optional<std::uint32_t> LinuxSysfs::QueryKfdComputeUnits(
    std::string_view pci_slot) const {
  // KFD encodes the PCI address as domain plus location_id = bus << 8 |
  // device << 3 | function, the layout ROCr decodes for its BDF id.
  unsigned domain = 0, bus = 0, device = 0, function = 0;
  const std::string slot(pci_slot);
  if (std::sscanf(slot.c_str(), "%x:%x:%x.%x", &domain, &bus, &device,
                  &function) != 4) {
    return std::nullopt;
  }
  const std::uint64_t location = (bus << 8) | (device << 3) | function;

  const auto nodes = sys_root_ / "class" / "kfd" / "kfd" / "topology" / "nodes";
  std::error_code ec;
  for (const auto& node : std::filesystem::directory_iterator(nodes, ec)) {
    const auto properties = ReadFile(node.path() / "properties");
    if (!properties) {
      continue;
    }
    std::map<std::string, std::uint64_t> values;
    std::istringstream stream(*properties);
    std::string key;
    std::uint64_t value = 0;
    while (stream >> key >> value) {
      values[key] = value;
    }
    const auto field = [&](const char* name) {
      const auto it = values.find(name);
      return it == values.end() ? std::uint64_t{0} : it->second;
    };
    if (field("simd_count") == 0 || field("simd_per_cu") == 0 ||
        field("location_id") != location || field("domain") != domain) {
      continue;
    }
    return static_cast<std::uint32_t>(field("simd_count") /
                                      field("simd_per_cu"));
  }
  return std::nullopt;
}

std::vector<SysfsGpuDevice> LinuxSysfs::QueryDrmGpuDevices() const {
  std::vector<SysfsGpuDevice> devices;
  const auto drm_path = sys_root_ / "class" / "drm";

  std::error_code ec;
  if (!std::filesystem::exists(drm_path, ec)) {
    return devices;
  }

  for (const auto& entry : std::filesystem::directory_iterator(drm_path, ec)) {
    if (ec) {
      break;
    }
    const std::string filename = entry.path().filename().string();
    if (!filename.starts_with("card") ||
        filename.find('-') != std::string::npos) {
      continue;
    }

    const auto dev_dir = entry.path() / "device";
    if (!std::filesystem::exists(dev_dir, ec)) {
      continue;
    }

    SysfsGpuDevice gpu;
    gpu.card_name = filename;
    gpu.vendor_id = ReadFile(dev_dir / "vendor").value_or("");
    gpu.device_id = ReadFile(dev_dir / "device").value_or("");

    const auto uevent_content = ReadFile(dev_dir / "uevent");
    if (uevent_content) {
      std::istringstream stream(*uevent_content);
      std::string line;
      while (std::getline(stream, line)) {
        if (line.starts_with("DRIVER=")) {
          gpu.driver = line.substr(7);
        } else if (line.starts_with("PCI_ID=")) {
          gpu.pci_id = line.substr(7);
        } else if (line.starts_with("PCI_SLOT_NAME=")) {
          gpu.pci_slot = line.substr(14);
        }
      }
    }

    gpu.compute_units = QueryKfdComputeUnits(gpu.pci_slot).value_or(0);
    gpu.gtt = ReadGtt(dev_dir);
    gpu.power_mode = ReadFile(dev_dir / "power_dpm_force_performance_level")
                         .value_or("unavailable (unreadable or not exposed)");

    const auto hwmon_base = dev_dir / "hwmon";
    if (std::filesystem::exists(hwmon_base, ec)) {
      for (const auto& hw_entry :
           std::filesystem::directory_iterator(hwmon_base, ec)) {
        if (ec) {
          break;
        }
        const auto temp1_input = ReadFile(hw_entry.path() / "temp1_input");
        if (temp1_input) {
          const std::uint32_t millicelsius = ParseUint(*temp1_input);
          if (millicelsius > 0) {
            const double temp_c = static_cast<double>(millicelsius) / 1000.0;
            gpu.hwmon_sensors["edge_temperature_c"] =
                std::to_string(temp_c).substr(0, 5);
          }
        }
      }
    }

    devices.push_back(std::move(gpu));
  }

  return devices;
}

std::optional<GttInfo> LinuxSysfs::ReadGtt(
    const std::filesystem::path& device_dir) const {
  const auto total = ReadFile(device_dir / "mem_info_gtt_total");
  const auto used = ReadFile(device_dir / "mem_info_gtt_used");
  if (!total || !used) {
    return std::nullopt;
  }
  const auto total_bytes = ParseUint64(*total);
  const auto used_bytes = ParseUint64(*used);
  if (!total_bytes || !used_bytes || *total_bytes == 0) {
    return std::nullopt;
  }
  return GttInfo{*total_bytes, *used_bytes};
}

std::optional<GttInfo> LinuxSysfs::QueryGtt() const {
  for (const auto& device_dir : DrmCardDevices(sys_root_)) {
    if (auto gtt = ReadGtt(device_dir)) {
      return gtt;
    }
  }
  return std::nullopt;
}

}  // namespace gufo::diagnostics
