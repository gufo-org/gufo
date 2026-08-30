#ifndef GUFO_CORE_HRX_HRX_BACKEND_HPP_
#define GUFO_CORE_HRX_HRX_BACKEND_HPP_

#include <hrx/hrx_runtime.h>

namespace gufo::hrx {

class HrxBackend {
public:
  static HrxBackend& Instance() {
    static HrxBackend instance;
    return instance;
  }

  bool Initialize(int device_index = 0);
  void Shutdown();

  [[nodiscard]] hrx_device_t Device() const noexcept { return device_; }
  [[nodiscard]] hrx_stream_t Stream() const noexcept { return stream_; }
  [[nodiscard]] bool IsInitialized() const noexcept { return initialized_; }

private:
  HrxBackend() = default;
  ~HrxBackend() { Shutdown(); }

  HrxBackend(const HrxBackend&) = delete;
  HrxBackend& operator=(const HrxBackend&) = delete;

  hrx_device_t device_{nullptr};
  hrx_stream_t stream_{nullptr};
  bool initialized_{false};
};

}  // namespace gufo::hrx

#endif  // GUFO_CORE_HRX_HRX_BACKEND_HPP_
