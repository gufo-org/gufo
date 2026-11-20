#include "src/core/hip/copy_engine.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "src/core/platform/library.hpp"

namespace gufo::hip {
namespace {
// ROCr, already loaded by HIP: the same soname resolves to that runtime, so
// no second copy is linked or loaded.
constexpr const char* kRuntime = "libhsa-runtime64.so.1";

class HsaCopyEngine final : public CopyEngine {
public:
  // Null when the runtime, the device's agents or an SDMA engine is missing.
  static std::unique_ptr<HsaCopyEngine> Create() {
    auto engine = std::unique_ptr<HsaCopyEngine>(new HsaCopyEngine);
    return engine->Open() ? std::move(engine) : nullptr;
  }
  Fence CreateFence() override {
    hsa_signal_t signal{};
    if (create_(0, 0, nullptr, &signal) != HSA_STATUS_SUCCESS)
      throw std::runtime_error("cannot create a copy engine signal");
    return signal.handle;
  }
  void DestroyFence(Fence fence) noexcept override {
    (void)destroy_(hsa_signal_t{fence});
  }
  bool Copy(void* destination, const void* source, std::size_t bytes,
            Fence fence) noexcept override {
    const hsa_signal_t signal{fence};
    // The engine decrements the signal once the copy lands.
    add_(signal, 1);
    // Every copy names the CPU as the source agent: ROCr then queues it on
    // the GPU's host-to-device SDMA engine, for device sources too (the
    // engine reads by GPU virtual address). A GPU source agent would take a
    // blit kernel on a compute queue of ROCr's own, or a second SDMA queue.
    if (copy_(destination, gpu_, source, cpu_, bytes, 0, nullptr, signal,
              engine_, true) == HSA_STATUS_SUCCESS)
      return true;
    subtract_(signal, 1);
    return false;
  }
  bool Done(Fence fence) const noexcept override {
    return load_(hsa_signal_t{fence}) == 0;
  }
  void Wait(Fence fence) noexcept override {
    const hsa_signal_t signal{fence};
    // A wait may return early; only the value says the copies landed.
    while (load_(signal) != 0)
      (void)wait_(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                  std::numeric_limits<std::uint64_t>::max(),
                  HSA_WAIT_STATE_BLOCKED);
  }

private:
  HsaCopyEngine() = default;
  template<typename Function>
  void Resolve(Function& function, const char* name) {
    function = reinterpret_cast<Function>(runtime_->Symbol(name));
  }
  bool Open() noexcept {
    try {
      runtime_ = std::make_unique<platform::Library>(kRuntime);
      decltype(&hsa_init) init{};
      decltype(&hsa_iterate_agents) iterate{};
      decltype(&hsa_amd_memory_copy_engine_status) status{};
      Resolve(init, "hsa_init");
      Resolve(iterate, "hsa_iterate_agents");
      Resolve(agent_info_, "hsa_agent_get_info");
      Resolve(status, "hsa_amd_memory_copy_engine_status");
      Resolve(create_, "hsa_signal_create");
      Resolve(destroy_, "hsa_signal_destroy");
      Resolve(add_, "hsa_signal_add_relaxed");
      Resolve(subtract_, "hsa_signal_subtract_relaxed");
      Resolve(load_, "hsa_signal_load_scacquire");
      Resolve(wait_, "hsa_signal_wait_scacquire");
      Resolve(copy_, "hsa_amd_memory_async_copy_on_engine");
      // HIP holds its own reference; this one keeps the runtime up for the
      // process, as the engine does.
      if (init() != HSA_STATUS_SUCCESS)
        return false;
      int device = 0, bus = 0, slot = 0, domain = 0;
      if (hipGetDevice(&device) != hipSuccess ||
          hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, device) !=
              hipSuccess ||
          hipDeviceGetAttribute(&slot, hipDeviceAttributePciDeviceId, device) !=
              hipSuccess ||
          hipDeviceGetAttribute(&domain, hipDeviceAttributePciDomainID,
                                device) != hipSuccess) {
        (void)hipGetLastError();
        return false;
      }
      bus_ = static_cast<std::uint32_t>(bus);
      slot_ = static_cast<std::uint32_t>(slot);
      domain_ = static_cast<std::uint32_t>(domain);
      if (iterate(&HsaCopyEngine::Visit, this) != HSA_STATUS_SUCCESS ||
          !gpu_.handle || !cpu_.handle)
        return false;
      std::uint32_t engines = 0;
      if (status(gpu_, cpu_, &engines) != HSA_STATUS_SUCCESS || !engines)
        return false;
      // One engine, so every restore shares one SDMA queue.
      engine_ = static_cast<hsa_amd_sdma_engine_id_t>(engines & -engines);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  static hsa_status_t Visit(hsa_agent_t agent, void* data) {
    auto& self = *static_cast<HsaCopyEngine*>(data);
    hsa_device_type_t type{};
    if (self.agent_info_(agent, HSA_AGENT_INFO_DEVICE, &type) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_SUCCESS;
    if (type == HSA_DEVICE_TYPE_CPU && !self.cpu_.handle)
      self.cpu_ = agent;
    if (type != HSA_DEVICE_TYPE_GPU || self.gpu_.handle)
      return HSA_STATUS_SUCCESS;
    // The agent at HIP's device's PCI address: bus, device, function bits.
    std::uint32_t bdf = 0, domain = 0;
    if (self.agent_info_(
            agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
            &bdf) == HSA_STATUS_SUCCESS &&
        self.agent_info_(
            agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
            &domain) == HSA_STATUS_SUCCESS &&
        (bdf >> 8) == self.bus_ && ((bdf >> 3) & 0x1f) == self.slot_ &&
        domain == self.domain_)
      self.gpu_ = agent;
    return HSA_STATUS_SUCCESS;
  }

  std::unique_ptr<platform::Library> runtime_;
  decltype(&hsa_agent_get_info) agent_info_{};
  decltype(&hsa_signal_create) create_{};
  decltype(&hsa_signal_destroy) destroy_{};
  decltype(&hsa_signal_add_relaxed) add_{};
  decltype(&hsa_signal_subtract_relaxed) subtract_{};
  decltype(&hsa_signal_load_scacquire) load_{};
  decltype(&hsa_signal_wait_scacquire) wait_{};
  decltype(&hsa_amd_memory_async_copy_on_engine) copy_{};
  hsa_agent_t gpu_{}, cpu_{};
  hsa_amd_sdma_engine_id_t engine_{};
  std::uint32_t bus_{}, slot_{}, domain_{};
};
}  // namespace

CopyEngine* DeviceCopyEngine() noexcept {
  // Intentionally never destroyed: fences may be released during exit.
  static CopyEngine* const engine = HsaCopyEngine::Create().release();
  return engine;
}

}  // namespace gufo::hip
