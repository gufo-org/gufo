#ifndef GUFO_CORE_HIP_TRANSFER_POOL_HPP_
#define GUFO_CORE_HIP_TRANSFER_POOL_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>

#include "src/cache/stream.hpp"

namespace gufo::hip {
namespace detail {
struct TransferPoolState;
struct TransferLease;
}  // namespace detail

enum class CopyDirection { kDeviceToHost, kHostToDevice };

// One event per transfer; a pool may share native FIFO streams between leases.
// Caller orders access to this object; different leases may run concurrently.
// Sources must already be ready and stay immutable; retain all buffers until
// completion settles. A lease accepts one completion only. Complete seals
// Native submissions too. With shared streams, Native submissions must not
// wait for work whose submission depends on another lease completing: the
// native streams impose FIFO ordering between leases.
class TransferStream final : public cache::Stream {
public:
  ~TransferStream() override;
  // Before any Native submission, use an existing execution stream instead of
  // the pool's stream. Retain its owner until both this lease and completion
  // are released; synchronization and error draining use the bound stream.
  void BindTo(hipStream_t, std::shared_ptr<void> owner);
  // Synchronize does not seal this handle; it may be reused until Complete or
  // failure. No further submissions through a saved handle after Complete.
  [[nodiscard]] hipStream_t Native() const;
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override;
  [[nodiscard]] cache::Completion Complete();
  [[nodiscard]] cache::Completion Copy(void* destination, const void* source,
                                       std::size_t bytes, CopyDirection,
                                       std::size_t piece_bytes);

private:
  friend class TransferPool;
  explicit TransferStream(std::shared_ptr<detail::TransferLease>);
  std::shared_ptr<detail::TransferLease> lease_;
};

class TransferPool {
public:
  // Creates all nonblocking streams and timing-disabled events at startup.
  explicit TransferPool(std::size_t capacity);
  // Keep capacity independent event leases over fewer eager native streams.
  // Native FIFO streams are shared when native_streams < capacity. Successful
  // waits stop at this lease's event, with system-scope visibility;
  // a native stream failure quarantines every lease mapped to that stream.
  // Both counts must be nonzero, with native_streams <= capacity. Equal counts
  // retain the one-argument constructor's exclusive stream behavior.
  TransferPool(std::size_t capacity, std::size_t native_streams);
  // Returns null on exhaustion. No HIP allocation/stream/event creation here.
  // An event returns only after BOTH stream and completion release their lease.
  // Failed resources are quarantined rather than reused. A shared pool throws
  // when exhausted after any quarantine: its consumers must not wait for lost
  // capacity to return. Exclusive pools retain null on exhaustion.
  [[nodiscard]] std::unique_ptr<TransferStream> TryAcquire();

private:
  std::shared_ptr<detail::TransferPoolState> state_;
};

}  // namespace gufo::hip
#endif
