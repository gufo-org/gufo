#include "src/core/platform/gpu_budget.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstring>

namespace gufo::platform {

std::optional<GpuLocalBudget> QueryGpuLocalBudget(
    std::span<const std::uint8_t, 8> luid, unsigned node) noexcept {
  LUID adapter_luid{};
  static_assert(sizeof(adapter_luid) == 8);
  std::memcpy(&adapter_luid, luid.data(), sizeof(adapter_luid));
  // The adapter is resolved afresh and released on return, so nothing
  // outlives the call, whichever device the calling thread has selected.
  Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
  Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()))) ||
      FAILED(factory->EnumAdapterByLuid(
          adapter_luid, IID_PPV_ARGS(adapter.GetAddressOf()))) ||
      FAILED(adapter->QueryVideoMemoryInfo(
          node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
    return std::nullopt;
  }
  return GpuLocalBudget{.budget_bytes = info.Budget,
                        .usage_bytes = info.CurrentUsage};
}

}  // namespace gufo::platform
