#ifndef GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
#define GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>

namespace gufo::hip {

/// Copies a frozen session on a separate stream. Callers have completed that
/// session's model operation before capture, and keep its state alive and
/// unchanged until all copies finish. Other sessions may continue executing.
class SnapshotTransfer {
public:
  SnapshotTransfer() {
    Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  }
  /// Borrows a nonblocking stream from a caller's pool and hands it to
  /// `release` once every copy has finished.
  SnapshotTransfer(hipStream_t stream, std::function<void(hipStream_t)> release)
      : stream_(stream), owned_(false), release_(std::move(release)) {}
  ~SnapshotTransfer() {
    (void)hipStreamSynchronize(stream_);
    if (owned_)
      (void)hipStreamDestroy(stream_);
    else if (release_)
      release_(stream_);
  }
  SnapshotTransfer(const SnapshotTransfer&) = delete;
  SnapshotTransfer& operator=(const SnapshotTransfer&) = delete;

  void Copy(void* destination, const void* source, std::size_t bytes,
            hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    Enqueue(destination, source, bytes, kind);
    Finish();
  }

  /// Queue independent frozen regions, then Finish before publishing their
  /// snapshot or mutating/freeing any source or destination storage.
  void Enqueue(void* destination, const void* source, std::size_t bytes,
               hipMemcpyKind kind = hipMemcpyDeviceToHost) {
    Check(hipMemcpyAsync(destination, source, bytes, kind, stream_));
  }
  void Finish() { Check(hipStreamSynchronize(stream_)); }

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
  bool owned_{true};
  std::function<void(hipStream_t)> release_;
};

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_SNAPSHOT_TRANSFER_HPP_
