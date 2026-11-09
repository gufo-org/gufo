#include "src/core/platform/gpu_budget.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstring>

namespace gufo::platform {
namespace {

using CreateDxgiFactoryFn = HRESULT(WINAPI*)(REFIID, void**);

// CreateDXGIFactory1, from the System32 dxgi.dll loaded on first use and kept
// for the life of the process. Loading it at run time keeps dxgi.dll out of
// the executable's imports, which the Windows bundle check would reject.
// Null when the library or the entry point is missing.
CreateDxgiFactoryFn LoadCreateDxgiFactory() noexcept {
  static const CreateDxgiFactoryFn create = [] {
    const HMODULE dxgi =
        LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return dxgi == nullptr ? nullptr
                           : reinterpret_cast<CreateDxgiFactoryFn>(
                                 GetProcAddress(dxgi, "CreateDXGIFactory1"));
  }();
  return create;
}

}  // namespace

std::optional<GpuLocalBudget> QueryGpuLocalBudget(
    std::span<const std::uint8_t, 8> luid, unsigned node) noexcept {
  const CreateDxgiFactoryFn create_factory = LoadCreateDxgiFactory();
  if (create_factory == nullptr)
    return std::nullopt;
  LUID adapter_luid{};
  static_assert(sizeof(adapter_luid) == 8);
  std::memcpy(&adapter_luid, luid.data(), sizeof(adapter_luid));
  // The adapter is resolved afresh and released on return, so no DXGI object
  // outlives the call, whichever device the calling thread has selected.
  Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
  Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (FAILED(create_factory(IID_PPV_ARGS(factory.GetAddressOf()))) ||
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
