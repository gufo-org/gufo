#include "src/models/qwen_image_21/hip/memory.hpp"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <sstream>

#include "src/core/platform/os_info.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <dxgi1_4.h>
#include <windows.h>
#endif

namespace gufo::models::qwen_image_21::hip {
namespace {

#if defined(_WIN32)
using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);

// Loaded from System32 on first use so the executable does not import
// dxgi.dll; without it the check falls back to hipMemGetInfo.
CreateFactoryFn LoadCreateFactory() {
  static const CreateFactoryFn create = [] {
    HMODULE dxgi =
        LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return dxgi ? reinterpret_cast<CreateFactoryFn>(
                      GetProcAddress(dxgi, "CreateDXGIFactory1"))
                : nullptr;
  }();
  return create;
}

// The first hardware AMD adapter; HIP on Windows allocates through it.
bool QuerySegments(DeviceMemory* memory) {
  const CreateFactoryFn create_factory = LoadCreateFactory();
  IDXGIFactory4* factory = nullptr;
  if (!create_factory ||
      FAILED(create_factory(__uuidof(IDXGIFactory4),
                            reinterpret_cast<void**>(&factory))))
    return false;
  bool found = false;
  IDXGIAdapter1* adapter = nullptr;
  for (UINT i = 0;
       !found && factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc{};
    IDXGIAdapter3* adapter3 = nullptr;
    if (SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002 &&
        !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
        SUCCEEDED(adapter->QueryInterface(
            __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3)))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO local{}, nonlocal{};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
          SUCCEEDED(adapter3->QueryVideoMemoryInfo(
              0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal))) {
        memory->local_usage = local.CurrentUsage;
        memory->local_budget = local.Budget;
        memory->nonlocal_usage = nonlocal.CurrentUsage;
        memory->nonlocal_budget = nonlocal.Budget;
        memory->dedicated = desc.DedicatedVideoMemory;
        found = true;
      }
      adapter3->Release();
    }
    adapter->Release();
  }
  factory->Release();
  return found;
}
#endif

std::uint64_t Headroom(std::uint64_t budget, std::uint64_t usage) {
  return budget > usage ? budget - usage : 0;
}

constexpr std::uint64_t kMiB = 1ULL << 20;

}  // namespace

std::uint64_t DeviceMemory::Available(std::uint64_t host_reserve) const {
  const std::uint64_t host =
      host_available > host_reserve ? host_available - host_reserve : 0;
  // Elsewhere hipMemGetInfo is the only source; without any reading the
  // check is skipped rather than refusing every large request.
  if (!segments)
    return hip_valid ? hip_free : UINT64_MAX;
  const std::uint64_t budget = Headroom(local_budget, local_usage) +
                               Headroom(nonlocal_budget, nonlocal_usage);
  // An APU may report its whole budget as local (Strix Halo: 48 GB local,
  // 0 shared, 32 GB carve-out): beyond the dedicated carve-out every byte
  // comes from host RAM.
  const std::uint64_t used = local_usage + nonlocal_usage;
  return std::min(budget, Headroom(dedicated, used) + host);
}

std::string DeviceMemory::Describe() const {
  std::ostringstream out;
  out << "hip_free_mib=" << hip_free / kMiB
      << " hip_total_mib=" << hip_total / kMiB;
  if (segments)
    out << " dedicated_mib=" << dedicated / kMiB
        << " local_usage_mib=" << local_usage / kMiB
        << " local_budget_mib=" << local_budget / kMiB
        << " shared_usage_mib=" << nonlocal_usage / kMiB
        << " shared_budget_mib=" << nonlocal_budget / kMiB;
  out << " host_available_mib=" << host_available / kMiB;
  return out.str();
}

DeviceMemory QueryDeviceMemory() {
  DeviceMemory memory;
  std::size_t free = 0, total = 0;
  if (hipMemGetInfo(&free, &total) == hipSuccess) {
    memory.hip_valid = true;
    memory.hip_free = free;
    memory.hip_total = total;
  } else {
    (void)hipGetLastError();
  }
#if defined(_WIN32)
  memory.segments = QuerySegments(&memory);
#endif
  memory.host_available = platform::AvailableMemoryBytes();
  return memory;
}

}  // namespace gufo::models::qwen_image_21::hip
