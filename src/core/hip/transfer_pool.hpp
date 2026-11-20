#ifndef GUFO_CORE_HIP_TRANSFER_POOL_HPP_
#define GUFO_CORE_HIP_TRANSFER_POOL_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>

#include "src/cache/stream.hpp"

namespace gufo::hip {
class CopyEngine;
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
  // Whether this lease copies on the pool's copy engine (TryAcquire
  // `on_engine`). Native submissions still run on its stream, and Complete
  // covers both.
  [[nodiscard]] bool OnEngine() const noexcept;
  // Queues one copy on the copy engine, which reads and writes the two
  // addresses as they are (see CopyEngine). Unordered against Native work and
  // other engine copies; only for an OnEngine lease that is neither sealed nor
  // failed. A refusal fails the lease and returns false.
  [[nodiscard]] bool EngineCopy(void* destination, const void* source,
                                std::size_t bytes);

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
  // A nonzero `priority` creates the native streams with that HIP stream
  // priority (lower is greater; -1 is the greatest on gfx1151).
  // With an `engine`, each lease gets a fence of its own on it, for leases
  // acquired `on_engine`.
  TransferPool(std::size_t capacity, std::size_t native_streams,
               int priority = 0, CopyEngine* engine = nullptr);
  // Returns null on exhaustion. No HIP allocation/stream/event creation here.
  // An event returns only after BOTH stream and completion release their lease.
  // Failed resources are quarantined rather than reused. A shared pool throws
  // when exhausted after any quarantine: its consumers must not wait for lost
  // capacity to return. Exclusive pools retain null on exhaustion.
  // `on_engine` leases copy on the pool's engine (TransferStream::EngineCopy)
  // where it has one.
  [[nodiscard]] std::unique_ptr<TransferStream> TryAcquire(
      bool on_engine = false);
  // Whether leases can copy on an engine.
  [[nodiscard]] bool HasEngine() const noexcept;

private:
  std::shared_ptr<detail::TransferPoolState> state_;
};

}  // namespace gufo::hip
#endif
