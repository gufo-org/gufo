#ifndef GUFO_MODELS_QWEN_IMAGE_21_HIP_MEMORY_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_HIP_MEMORY_HPP_

#include <cstdint>
#include <string>

namespace gufo::models::qwen_image_21::hip {

/// Device memory as seen by this process. On Windows the GPU's local
/// (dedicated) and non-local (shared, host RAM) segments come from DXGI;
/// hipMemGetInfo reports both together and ignores host memory pressure.
/// Without DXGI (other platforms, or no AMD adapter found) hipMemGetInfo is
/// used alone.
struct DeviceMemory {
  bool hip_valid{false};
  std::uint64_t hip_free{0}, hip_total{0};
  bool segments{false};
  std::uint64_t local_usage{0}, local_budget{0};
  std::uint64_t nonlocal_usage{0}, nonlocal_budget{0};
  /// Dedicated (carve-out) memory of the adapter.
  std::uint64_t dedicated{0};
  std::uint64_t host_available{0};

  /// Bytes this process can still allocate without exhausting host RAM:
  /// the DXGI headroom, bounded by the free dedicated carve-out plus
  /// available host memory minus `host_reserve`. Without DXGI: hipMemGetInfo
  /// free memory, or UINT64_MAX (no limit known) if that query failed.
  std::uint64_t Available(std::uint64_t host_reserve) const;
  std::string Describe() const;
};

DeviceMemory QueryDeviceMemory();

}  // namespace gufo::models::qwen_image_21::hip
#endif
