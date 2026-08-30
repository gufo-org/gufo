#include "src/core/hrx/hrx_backend.hpp"

#include "src/core/hrx/hrx_utils.hpp"

namespace gufo::hrx {

bool HrxBackend::Initialize(int device_index) {
  if (initialized_) {
    return true;
  }

  hrx_status_t status = hrx_gpu_initialize(0);
  if (!hrx_status_is_ok(status)) {
    if (hrx_status_code(status) != HRX_STATUS_ALREADY_EXISTS) {
      hrx_status_ignore(status);
      return false;
    }
    hrx_status_ignore(status);
  }

  status = hrx_gpu_device_get(device_index, &device_);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return false;
  }

  status = hrx_stream_create(device_, 0, &stream_);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    device_ = nullptr;
    return false;
  }

  initialized_ = true;
  return true;
}

void HrxBackend::Shutdown() {
  if (!initialized_) {
    return;
  }

  if (stream_ != nullptr) {
    (void)hrx_stream_synchronize(stream_);
    hrx_stream_release(stream_);
    stream_ = nullptr;
  }

  device_ = nullptr;
  (void)hrx_gpu_shutdown();

  initialized_ = false;
}

}  // namespace gufo::hrx
