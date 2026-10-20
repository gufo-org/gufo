#include <hip/hip_runtime.h>

#include <cstring>
#include <memory>
#include <string>

#include "src/core/hip/arch.hpp"
#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::minimax_h3 {

namespace {

void SetHipError(std::string* error, std::string_view operation,
                 hipError_t status) {
  if (error != nullptr) {
    *error = std::string(operation) + ": " + hipGetErrorString(status);
  }
}

class HipResidencyBackend final : public ResidencyBackend {
public:
  bool IsSupportedTarget(std::string* error) override {
    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
      SetHipError(error, "hipGetDevice", status);
      return false;
    }
    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
      SetHipError(error, "hipGetDeviceProperties", status);
      return false;
    }
    if (gufo::hip::IsRdna35(properties.gcnArchName) && properties.integrated != 0) {
      return true;
    }
    if (error != nullptr) {
      *error =
          "MiniMax H3 supports only integrated RDNA 3.5 APUs (gfx1151 Strix "
          "Halo, gfx1150 Strix Point); detected " +
          std::string(properties.gcnArchName);
    }
    return false;
  }

  bool CreateStream(void** stream, std::string* error) override {
    hipStream_t created = nullptr;
    const hipError_t status =
        hipStreamCreateWithFlags(&created, hipStreamNonBlocking);
    if (status != hipSuccess) {
      SetHipError(error, "hipStreamCreateWithFlags", status);
      return false;
    }
    *stream = created;
    return true;
  }

  void DestroyStream(void* stream) noexcept override {
    if (stream != nullptr) {
      (void)hipStreamDestroy(static_cast<hipStream_t>(stream));
    }
  }

  bool Allocate(std::size_t bytes, void** device, std::string* error) override {
    if (bytes == 0) {
      *device = nullptr;
      return true;
    }
    const hipError_t status = hipMalloc(device, bytes);
    if (status != hipSuccess) {
      SetHipError(error, "hipMalloc", status);
      return false;
    }
    return true;
  }

  void Free(void* device) noexcept override {
    if (device != nullptr) {
      (void)hipFree(device);
    }
  }

  bool RegisterReadOnlyMapped(void* host, std::size_t bytes,
                              void** device_alias,
                              std::string* error) override {
    const unsigned int flags = hipHostRegisterMapped | hipHostRegisterReadOnly;
    hipError_t status = hipHostRegister(host, bytes, flags);
    if (status != hipSuccess) {
      SetHipError(error, "hipHostRegister(mapped|readonly)", status);
      return false;
    }
    status = hipHostGetDevicePointer(device_alias, host, 0);
    if (status != hipSuccess) {
      (void)hipHostUnregister(host);
      SetHipError(error, "hipHostGetDevicePointer", status);
      return false;
    }
    return true;
  }

  void Unregister(void* host) noexcept override {
    if (host != nullptr) {
      (void)hipHostUnregister(host);
    }
  }

  bool CopyToDevice(void* device, const void* host, std::size_t bytes,
                    void* stream, std::string* error) override {
    const hipError_t status =
        hipMemcpyAsync(device, host, bytes, hipMemcpyHostToDevice,
                       static_cast<hipStream_t>(stream));
    if (status != hipSuccess) {
      SetHipError(error, "hipMemcpyAsync", status);
      return false;
    }
    return true;
  }

  bool Synchronize(void* stream, std::string* error) override {
    const hipError_t status =
        hipStreamSynchronize(static_cast<hipStream_t>(stream));
    if (status != hipSuccess) {
      SetHipError(error, "hipStreamSynchronize", status);
      return false;
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<ResidencyBackend> CreateHipResidencyBackend(
    std::string* error) {
  auto backend = std::make_unique<HipResidencyBackend>();
  if (!backend->IsSupportedTarget(error)) {
    return nullptr;
  }
  return backend;
}

}  // namespace gufo::minimax_h3
