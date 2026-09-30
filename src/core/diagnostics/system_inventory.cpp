#include "src/core/diagnostics/system_inventory.h"

#include <iomanip>
#include <sstream>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::diagnostics {

namespace {

std::string EscapeJsonStr(std::string_view str) {
  std::ostringstream oss;
  for (const char ch : str) {
    if (ch == '"') {
      oss << "\\\"";
    } else if (ch == '\\') {
      oss << "\\\\";
    } else if (ch == '\n') {
      oss << "\\n";
    } else if (ch == '\t') {
      oss << "\\t";
    } else {
      oss << ch;
    }
  }
  return oss.str();
}

}  // namespace

SystemInventory CollectSystemInventory(const LinuxSysfs& sysfs) {
  SystemInventory inv;

  // 1. CPU
  if (const auto cpu_opt = sysfs.QueryCpuInfo()) {
    inv.cpu.model_name = cpu_opt->model_name;
    inv.cpu.vendor = cpu_opt->vendor_id;
    inv.cpu.architecture = cpu_opt->architecture;
    inv.cpu.logical_cores = cpu_opt->logical_cores;
    inv.cpu.physical_cores = cpu_opt->physical_cores;
    inv.cpu.numa_nodes = cpu_opt->numa_nodes;
  } else {
    inv.cpu.model_name = "unavailable (could not read /proc/cpuinfo)";
  }

  // 2. Memory
  if (const auto mem_opt = sysfs.QueryMemInfo()) {
    inv.memory.total_bytes = mem_opt->total_bytes;
    inv.memory.available_bytes = mem_opt->available_bytes;
    inv.memory.free_bytes = mem_opt->free_bytes;
    inv.memory.is_unified = mem_opt->is_unified_memory;
  }

  // 3. Linux Kernel & Driver Inventory
  if (const auto uname_opt = sysfs.QueryUname()) {
    inv.toolchain.kernel_release = uname_opt->release;
  } else {
    inv.toolchain.kernel_release = "unknown";
  }

  // 4. GPU (Sysfs + HIP query)
  const auto gpus = sysfs.QueryDrmGpuDevices();
  if (!gpus.empty()) {
    const auto& primary_gpu = gpus[0];
    inv.gpu.driver_name =
        primary_gpu.driver.empty() ? "amdgpu" : primary_gpu.driver;
    inv.gpu.pci_id = primary_gpu.pci_id;
    inv.gpu.pci_slot = primary_gpu.pci_slot;
    inv.gpu.power_mode = primary_gpu.power_mode;
    inv.gpu.temperatures = primary_gpu.hwmon_sensors;

    const std::string_view pci_view = primary_gpu.pci_id;
    if (pci_view.find("1002:1586") != std::string_view::npos ||
        pci_view.find("1002:1587") != std::string_view::npos ||
        pci_view.find("1002:1588") != std::string_view::npos) {
      inv.gpu.architecture = "gfx1151";
      inv.gpu.availability = "available";
    } else if (pci_view.find("1002:7550") != std::string_view::npos) {
      inv.gpu.architecture = "gfx1201";
      inv.gpu.availability = "available (rdna4-port)";
    } else {
      inv.gpu.architecture = primary_gpu.pci_id;
      inv.gpu.availability = "unsupported GPU (" + primary_gpu.pci_id + ")";
    }
  }

#if defined(ENGINE_ENABLE_HIP)
  int dev_count = 0;
  const hipError_t err = hipGetDeviceCount(&dev_count);
  if (err == hipSuccess && dev_count > 0) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
      inv.gpu.name = prop.name;
      inv.gpu.raw_arch_name = prop.gcnArchName;
      inv.gpu.total_memory_bytes = prop.totalGlobalMem;
      inv.gpu.compute_units =
          static_cast<std::uint32_t>(prop.multiProcessorCount);

      // Normalize architecture to gfx1151
      const std::string_view raw_view(prop.gcnArchName);
      if (raw_view.find("gfx1151") != std::string_view::npos) {
        inv.gpu.architecture = "gfx1151";
        inv.gpu.availability = "available";
      } else if (raw_view.find("gfx1201") != std::string_view::npos ||
                 raw_view.find("gfx12") != std::string_view::npos) {
        inv.gpu.architecture = "gfx1201";
        inv.gpu.availability = "available (rdna4-port)";
      } else {
        inv.gpu.architecture = prop.gcnArchName;
        inv.gpu.availability = "unsupported architecture";
      }
    }
  } else {
    inv.gpu.availability =
        "unavailable (" + std::string(hipGetErrorName(err)) + ")";
  }
#else
  inv.gpu.availability = "uncompiled (ENGINE_ENABLE_HIP=OFF)";
#endif

  return inv;
}

std::string SystemInventory::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";

  // CPU
  oss << "  \"cpu\": {\n";
  oss << "    \"modelName\": \"" << EscapeJsonStr(cpu.model_name) << "\",\n";
  oss << "    \"vendor\": \"" << EscapeJsonStr(cpu.vendor) << "\",\n";
  oss << "    \"architecture\": \"" << EscapeJsonStr(cpu.architecture)
      << "\",\n";
  oss << "    \"logicalCores\": " << cpu.logical_cores << ",\n";
  oss << "    \"physicalCores\": " << cpu.physical_cores << ",\n";
  oss << "    \"numaNodes\": " << cpu.numa_nodes << ",\n";
  oss << "    \"source\": \"" << EscapeJsonStr(cpu.source) << "\"\n";
  oss << "  },\n";

  // Memory
  oss << "  \"memory\": {\n";
  oss << "    \"totalBytes\": " << memory.total_bytes << ",\n";
  oss << "    \"availableBytes\": " << memory.available_bytes << ",\n";
  oss << "    \"freeBytes\": " << memory.free_bytes << ",\n";
  oss << "    \"memoryType\": \"" << EscapeJsonStr(memory.memory_type)
      << "\",\n";
  oss << "    \"isUnified\": " << (memory.is_unified ? "true" : "false")
      << ",\n";
  oss << "    \"source\": \"" << EscapeJsonStr(memory.source) << "\"\n";
  oss << "  },\n";

  // GPU
  oss << "  \"gpu\": {\n";
  oss << "    \"name\": \"" << EscapeJsonStr(gpu.name) << "\",\n";
  oss << "    \"architecture\": \"" << EscapeJsonStr(gpu.architecture)
      << "\",\n";
  oss << "    \"rawArchName\": \"" << EscapeJsonStr(gpu.raw_arch_name)
      << "\",\n";
  oss << "    \"computeUnits\": " << gpu.compute_units << ",\n";
  oss << "    \"totalMemoryBytes\": " << gpu.total_memory_bytes << ",\n";
  oss << "    \"driverName\": \"" << EscapeJsonStr(gpu.driver_name) << "\",\n";
  oss << "    \"pciId\": \"" << EscapeJsonStr(gpu.pci_id) << "\",\n";
  oss << "    \"pciSlot\": \"" << EscapeJsonStr(gpu.pci_slot) << "\",\n";
  oss << "    \"powerMode\": \"" << EscapeJsonStr(gpu.power_mode) << "\",\n";
  oss << "    \"clockState\": \"" << EscapeJsonStr(gpu.clock_state) << "\",\n";
  oss << "    \"availability\": \"" << EscapeJsonStr(gpu.availability)
      << "\",\n";
  oss << "    \"temperatures\": {";
  if (!gpu.temperatures.empty()) {
    oss << "\n";
    std::size_t idx = 0;
    for (const auto& [k, v] : gpu.temperatures) {
      oss << "      \"" << EscapeJsonStr(k) << "\": \"" << EscapeJsonStr(v)
          << "\"";
      if (++idx < gpu.temperatures.size()) {
        oss << ",";
      }
      oss << "\n";
    }
    oss << "    },\n";
  } else {
    oss << "},\n";
  }
  oss << "    \"source\": \"" << EscapeJsonStr(gpu.source) << "\"\n";
  oss << "  },\n";

  // Toolchain & Drivers
  oss << "  \"toolchain\": {\n";
  oss << "    \"kernelRelease\": \"" << EscapeJsonStr(toolchain.kernel_release)
      << "\",\n";
  oss << "    \"amdgpuStatus\": \"" << EscapeJsonStr(toolchain.amdgpu_status)
      << "\",\n";
  oss << "    \"rocmVersion\": \"" << EscapeJsonStr(toolchain.rocm_version)
      << "\",\n";
  oss << "    \"cxxCompiler\": \"" << EscapeJsonStr(toolchain.cxx_compiler)
      << "\",\n";
  oss << "    \"cppStandard\": \"" << EscapeJsonStr(toolchain.cpp_standard)
      << "\",\n";
  oss << "    \"source\": \"" << EscapeJsonStr(toolchain.source) << "\"\n";
  oss << "  }\n";

  oss << "}\n";
  return oss.str();
}

std::string SystemInventory::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Hardware and Software Inventory ===\n\n";

  oss << "CPU Model           : " << cpu.model_name << " (" << cpu.logical_cores
      << " threads, " << cpu.architecture << ")\n";
  oss << "System Memory       : "
      << (memory.total_bytes / (1024ULL * 1024 * 1024)) << " GiB ("
      << memory.memory_type << ")\n";
  oss << "GPU Model           : " << gpu.name << " [" << gpu.architecture
      << ", " << gpu.compute_units << " CUs] (" << gpu.availability << ")\n";
  oss << "Linux Kernel        : " << toolchain.kernel_release << "\n";
  oss << "ROCm Version        : " << toolchain.rocm_version << "\n";

  return oss.str();
}

}  // namespace gufo::diagnostics
