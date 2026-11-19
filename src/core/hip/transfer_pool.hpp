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

// Hardware queues of their own that pools may still create in this process
// (TransferPool `own_queues`); one unless set. Each costs one of the device's
// compute queues for the process lifetime, so the serving layer sets it from
// its queue plan before creating any pool.
void SetOwnQueueAllowance(std::size_t queues);
// What is left of that allowance.
[[nodiscard]] std::size_t OwnQueueAllowance();

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
  // A nonzero `priority` creates the native streams with that HIP stream
  // priority (lower is greater; -1 is the greatest on gfx1151).
  // `own_queues` instead gives each native stream a hardware queue of its own
  // while the process allowance lasts (SetOwnQueueAllowance), so its copies
  // never wait behind kernels other streams queued on a shared one; past it,
  // or where the runtime refuses, the stream falls back to `priority`. HIP
  // creates such a stream blocking: it orders against legacy null-stream
  // work, so a model that launches on the null stream must not ask for one.
  TransferPool(std::size_t capacity, std::size_t native_streams,
               int priority = 0, bool own_queues = false);
  // Returns null on exhaustion. No HIP allocation/stream/event creation here.
  // An event returns only after BOTH stream and completion release their lease.
  // Failed resources are quarantined rather than reused. A shared pool throws
  // when exhausted after any quarantine: its consumers must not wait for lost
  // capacity to return. Exclusive pools retain null on exhaustion.
  [[nodiscard]] std::unique_ptr<TransferStream> TryAcquire();
  // Whether every native stream got a hardware queue of its own.
  [[nodiscard]] bool OwnQueues() const noexcept;

private:
  std::shared_ptr<detail::TransferPoolState> state_;
};

}  // namespace gufo::hip
#endif
