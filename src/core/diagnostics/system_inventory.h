#ifndef GUFO_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_
#define GUFO_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include "src/core/diagnostics/linux_sysfs.h"

namespace gufo::diagnostics {

struct CpuInventory {
  std::string model_name;
  std::string vendor;
  std::string architecture{"x86_64"};
  std::uint32_t logical_cores{0};
  std::uint32_t physical_cores{0};
  std::uint32_t numa_nodes{1};
  std::string source{"/proc/cpuinfo"};
};

struct MemoryInventory {
  std::uint64_t total_bytes{0};
  std::uint64_t available_bytes{0};
  std::uint64_t free_bytes{0};
  std::string memory_type{"LPDDR5X (Unified)"};
  bool is_unified{true};
  std::string source{"/proc/meminfo"};
};

struct GpuInventory {
  std::string name{"AMD Radeon 8060S Graphics"};
  std::string architecture{"gfx1151"};
  std::string raw_arch_name;
  std::uint32_t compute_units{40};
  std::uint64_t total_memory_bytes{0};
  std::string driver_name{"amdgpu"};
  std::string pci_id{"1002:1586"};
  std::string pci_slot;
  std::string power_mode{"auto"};
  std::string clock_state{"unavailable (sysfs unreadable or power-managed)"};
  std::map<std::string, std::string> temperatures;
  std::string availability{"available"};
  std::string source{"HIP + /sys/class/drm"};
};

struct DriverToolchainInventory {
  // The OS Gufo runs on ("Linux", "Windows") and the target it matches.
  std::string os_name{"Linux"};
  std::string target_platform{"x86_64-linux"};
  std::string kernel_release;
  std::string amdgpu_status{"loaded"};
  std::string rocm_version{"7.2.3"};
  std::string cxx_compiler{"GCC 15.3.0"};
  std::string cpp_standard{"C++20"};
  std::string source{"Nix Toolchain Pins + Linux UAPI"};
};

struct SystemInventory {
  CpuInventory cpu;
  MemoryInventory memory;
  GpuInventory gpu;
  DriverToolchainInventory toolchain;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] SystemInventory CollectSystemInventory(
    const LinuxSysfs& sysfs = LinuxSysfs());

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_SYSTEM_INVENTORY_H_
