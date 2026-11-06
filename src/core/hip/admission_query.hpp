#ifndef GUFO_CORE_HIP_ADMISSION_QUERY_HPP_
#define GUFO_CORE_HIP_ADMISSION_QUERY_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

#include "src/core/hip/admission_memory.hpp"
#include "src/core/logging/logging.hpp"
#include "src/core/platform/gpu_budget.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::hip {

/// The readings behind one state-capacity check, kept for LogStateCapacity().
struct AdmissionMemory {
  /// Free bytes of the device heap, as hipMemGetInfo reports them.
  std::uint64_t hip_free_bytes{0};
  /// The OS video-memory budget of this process. Set only on an integrated
  /// GPU whose budget could be read.
  std::optional<platform::GpuLocalBudget> local;
  /// Host RAM available; read only when `local` is set.
  std::optional<std::uint64_t> host_available_bytes;
  /// What the check may count on: AdmissionFreeBytes() of the fields above,
  /// with kAdmissionHostMarginBytes.
  std::uint64_t free_bytes{0};
};

namespace detail {

/// True when `status` is hipSuccess. A failure clears the sticky error so
/// later calls do not see it.
[[nodiscard]] inline bool HipSucceeded(hipError_t status) {
  if (status == hipSuccess)
    return true;
  (void)hipGetLastError();
  return false;
}

/// The OS video-memory budget of the current device, read only on an integrated
/// GPU, whose memory is system RAM. Any failure reads as no budget.
[[nodiscard]] inline std::optional<platform::GpuLocalBudget>
QueryDeviceLocalBudget() {
  int device = 0;
  int integrated = 0;
  hipDeviceProp_t properties{};
  if (!HipSucceeded(hipGetDevice(&device)) ||
      !HipSucceeded(hipDeviceGetAttribute(
          &integrated, hipDeviceAttributeIntegrated, device)) ||
      integrated != 1 ||
      !HipSucceeded(hipGetDeviceProperties(&properties, device))) {
    return std::nullopt;
  }
  const std::string_view gcn_arch_name(
      properties.gcnArchName,
      strnlen(properties.gcnArchName, sizeof(properties.gcnArchName)));
  const auto node =
      DxgiNodeFromMask(properties.luidDeviceNodeMask, gcn_arch_name);
  if (!node.has_value())
    return std::nullopt;
  std::array<std::uint8_t, 8> luid{};
  static_assert(sizeof(luid) == sizeof(properties.luid));
  std::memcpy(luid.data(), properties.luid, luid.size());
  return platform::QueryGpuLocalBudget(luid, *node);
}

}  // namespace detail

/// Reads the memory a state-capacity check admits against, on the current
/// device. Returns nullopt only when hipMemGetInfo fails, as the check did
/// before; a later failure leaves HIP's own figure in `free_bytes`.
[[nodiscard]] inline std::optional<AdmissionMemory> QueryAdmissionMemory() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
    return std::nullopt;
  AdmissionMemory memory;
  memory.hip_free_bytes = free_bytes;
  memory.local = detail::QueryDeviceLocalBudget();
  if (memory.local.has_value()) {
    if (const auto host = platform::QueryHostMemory())
      memory.host_available_bytes = host->available_bytes;
  }
  memory.free_bytes = AdmissionFreeBytes(memory.hip_free_bytes, memory.local,
                                         memory.host_available_bytes,
                                         kAdmissionHostMarginBytes);
  return memory;
}

/// Writes event=state_capacity for one state-capacity check, since a refusal
/// names device memory even when host RAM set the limit. Silent unless DXGI
/// answered, so Linux and discrete GPUs log nothing.
inline void LogStateCapacity(const AdmissionMemory& memory,
                             std::uint64_t capacity, std::uint64_t per_request,
                             std::uint64_t deferred_scratch) {
  if (!memory.local.has_value())
    return;
  server::Logger::Info(
      "loader",
      "event=state_capacity hip_free_bytes=" +
          std::to_string(memory.hip_free_bytes) +
          " dxgi_budget_bytes=" + std::to_string(memory.local->budget_bytes) +
          " dxgi_usage_bytes=" + std::to_string(memory.local->usage_bytes) +
          " host_available_bytes=" +
          (memory.host_available_bytes.has_value()
               ? std::to_string(*memory.host_available_bytes)
               : std::string("unknown")) +
          " host_margin_bytes=" + std::to_string(kAdmissionHostMarginBytes) +
          " admission_free_bytes=" + std::to_string(memory.free_bytes) +
          " capacity_bytes=" + std::to_string(capacity) +
          " per_request_bytes=" + std::to_string(per_request) +
          " deferred_scratch_bytes=" + std::to_string(deferred_scratch));
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_ADMISSION_QUERY_HPP_
