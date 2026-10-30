#ifndef GUFO_CORE_HIP_ARCH_HPP_
#define GUFO_CORE_HIP_ARCH_HPP_

#include <string>
#include <string_view>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::hip {

// RDNA 3.5 integrated APUs.
//
//   gfx1151  Strix Halo   Radeon 8060S / 8050S   40 CU  LPDDR5X 256-bit
//   gfx1150  Strix Point  Radeon 890M            16 CU  DDR5    128-bit
//
// Both expose Wave32 and the same instruction set, so every kernel in this
// repository runs unmodified on either part. What differs is scale: the CU
// count and the memory subsystem. Token generation on a dense 27B model is
// memory-bound, so gfx1150 lands near 5 tok/s against Strix Halo's 13 tok/s
// purely because DDR5-5600 dual-channel delivers 87 GB/s against 240 GB/s.
// That is a hardware ceiling, not a kernel defect.
//
// Keep the architecture string as reported by the driver. Normalizing
// gfx1150 to gfx1151 would make two different machines share a fingerprint
// and a disk-cache identity, which is exactly what the cache keys exist to
// prevent.
inline constexpr std::string_view kGfx1150 = "gfx1150";
inline constexpr std::string_view kGfx1151 = "gfx1151";

// gcnArchName carries feature suffixes ("gfx1150:sramecc+:xnack-"), so match
// on the prefix rather than the whole string.
[[nodiscard]] constexpr bool IsRdna35(std::string_view arch) noexcept {
  return arch.starts_with(kGfx1150) || arch.starts_with(kGfx1151);
}

[[nodiscard]] constexpr bool IsStrixPoint(std::string_view arch) noexcept {
  return arch.starts_with(kGfx1150);
}

[[nodiscard]] constexpr bool IsStrixHalo(std::string_view arch) noexcept {
  return arch.starts_with(kGfx1151);
}

// Human-readable family name for diagnostics and error messages.
[[nodiscard]] constexpr std::string_view Rdna35MarketingName(
    std::string_view arch) noexcept {
  if (IsStrixPoint(arch)) {
    return "AMD Strix Point GPU (gfx1150, Radeon 890M)";
  }
  if (IsStrixHalo(arch)) {
    return "AMD Strix Halo GPU (gfx1151, Radeon 8060S/8050S)";
  }
  return "unsupported AMD GPU";
}

// Unified memory type shipped with each part. Strix Halo pairs its iGPU with
// LPDDR5X on a 256-bit bus; Strix Point uses DDR5 on 128 bits. The fingerprint
// carries this string, so it must follow the part rather than a default.
[[nodiscard]] constexpr std::string_view Rdna35MemoryType(
    std::string_view arch) noexcept {
  if (IsStrixPoint(arch)) {
    return "DDR5 (Unified)";
  }
  if (IsStrixHalo(arch)) {
    return "LPDDR5X (Unified)";
  }
  return "Unified";
}

// PCI device IDs, used when HIP is unavailable or before the driver reports
// gcnArchName. Strix Halo ships as 1586/1587/1588; Strix Point as 150e.
[[nodiscard]] constexpr bool IsRdna35PciId(std::string_view pci_id) noexcept {
  return pci_id.find("1002:1586") != std::string_view::npos ||
         pci_id.find("1002:1587") != std::string_view::npos ||
         pci_id.find("1002:1588") != std::string_view::npos ||
         pci_id.find("1002:150e") != std::string_view::npos;
}

[[nodiscard]] constexpr std::string_view Rdna35PciArch(
    std::string_view pci_id) noexcept {
  if (pci_id.find("1002:150e") != std::string_view::npos) {
    return kGfx1150;
  }
  if (IsRdna35PciId(pci_id)) {
    return kGfx1151;
  }
  return {};
}

#if defined(ENGINE_ENABLE_HIP)
// Architecture of the active HIP device, as the driver reports it. Returns an
// empty string when the query fails. Disk-cache identities include this value
// so that a gfx1150 host never reuses a gfx1151 snapshot: the two parts share
// an instruction set but not a memory subsystem, and nothing guarantees that
// a numerically-sensitive reduction splits identically on both.
[[nodiscard]] inline std::string_view CurrentDeviceArch() {
  static const std::string arch = [] {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) {
      return std::string{};
    }
    return std::string(properties.gcnArchName);
  }();
  return arch;
}

// Human-facing device line for the banner and the startup log: the exact part
// when the driver answers, the family name when the query fails. Reuses the
// marketing names above so the two can never drift. The two parts share an
// instruction set but not a memory subsystem, and the cache identity keys on
// the architecture, so a gfx1150 host and a gfx1151 host should not print the
// same line.
[[nodiscard]] inline std::string DeviceDescription() {
  const std::string_view arch = CurrentDeviceArch();
  if (!arch.empty()) {
    if (IsRdna35(arch)) {
      return std::string(Rdna35MarketingName(arch));
    }
    return std::string(arch);
  }
  return "AMD RDNA 3.5 APU";
}
#else
[[nodiscard]] inline std::string DeviceDescription() {
  return "AMD RDNA 3.5 APU";
}
#endif

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_ARCH_HPP_
