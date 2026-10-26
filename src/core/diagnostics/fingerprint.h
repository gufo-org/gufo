#ifndef GUFO_CORE_DIAGNOSTICS_FINGERPRINT_H_
#define GUFO_CORE_DIAGNOSTICS_FINGERPRINT_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "src/core/diagnostics/system_inventory.h"

namespace gufo::diagnostics {

struct CanonicalFingerprint {
  std::string schema_version{"1.0.0"};
  std::string cpu_model;
  std::string cpu_architecture{"x86_64"};
  std::uint32_t cpu_logical_cores{0};
  std::uint32_t cpu_physical_cores{0};
  std::uint64_t memory_total_bytes{0};
  std::string memory_type{"LPDDR5X (Unified)"};
  std::string gpu_name;
  std::string gpu_architecture{"gfx1151"};
  std::uint32_t gpu_compute_units{0};
  std::string gpu_driver{"amdgpu"};
  std::string gpu_pci_id{"1002:1586"};
  std::string kernel_release;
  std::string rocm_version{"7.2.3"};
  std::string cxx_compiler{"GCC 15.3.0"};
  std::string cpp_standard{"C++20"};

  [[nodiscard]] std::string CanonicalJson() const;
  [[nodiscard]] std::string ComputeFingerprintId() const;
};

struct MachineFingerprint {
  std::string fingerprint_id;
  CanonicalFingerprint canonical;
  // Display only: outside the canonical form, so the id is unchanged.
  std::string os_name{"Linux"};
  std::string target_platform{"x86_64-linux"};

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] MachineFingerprint GenerateMachineFingerprint(
    const SystemInventory& inventory);

[[nodiscard]] std::string ComputeSha256Hex(std::string_view data);

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_FINGERPRINT_H_
