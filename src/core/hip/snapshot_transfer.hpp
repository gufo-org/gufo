#ifndef GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
#define GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace gufo::hip {

/// Copies a frozen session on a separate stream. Callers have completed that
/// session's model operation before capture, and keep its state alive and
/// unchanged until all copies finish. Other sessions may continue executing.
class SnapshotTransfer {
public:
  SnapshotTransfer() {
    Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }
  ~SnapshotTransfer() {
    (void)hipStreamSynchronize(stream_);
    (void)hipStreamDestroy(stream_);
  }
  SnapshotTransfer(const SnapshotTransfer&) = delete;
  SnapshotTransfer& operator=(const SnapshotTransfer&) = delete;

  void Copy(void* destination, const void* source, std::size_t bytes,
            hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    Check(hipMemcpyAsync(destination, source, bytes, kind, stream_));
    Check(hipStreamSynchronize(stream_));
  }

  void Copy2D(void* destination, std::size_t destination_pitch,
              const void* source, std::size_t source_pitch, std::size_t width,
              std::size_t height, hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    Check(hipMemcpy2DAsync(destination, destination_pitch, source, source_pitch,
                           width, height, kind, stream_));
    Check(hipStreamSynchronize(stream_));
  }

private:
  static void Check(hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
  }
  hipStream_t stream_{nullptr};
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
