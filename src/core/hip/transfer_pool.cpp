#include "src/core/hip/transfer_pool.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::hip {
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
}  // namespace
namespace detail {
struct TransferPoolState {
  struct Lane {
    hipStream_t stream{};
    std::atomic<bool> failed{false};
  };
  struct Slot {
    hipEvent_t event{};
    bool busy{};
    bool failed{};
  };
  std::mutex mutex;
  std::unique_ptr<Lane[]> lanes;
  std::size_t native_streams{};
  bool shared{};
  std::atomic<bool> degraded{false};
  std::vector<Slot> slots;
  ~TransferPoolState() {
    for (std::size_t i = 0; i < native_streams; ++i) {
      if (lanes[i].stream) {
        (void)hipStreamSynchronize(lanes[i].stream);
        (void)hipStreamDestroy(lanes[i].stream);
      }
    }
    for (auto& slot : slots)
      if (slot.event)
        (void)hipEventDestroy(slot.event);
  }
};
struct TransferLease {
  std::shared_ptr<TransferPoolState> state;
  std::size_t index;
  std::atomic<bool> failed{false};
  bool sealed{};
  bool claimed{};
  bool native_used{};
  bool event_recorded{};
  bool event_settled{};
  bool event_failed{};
  std::mutex drain_mutex;
  hipStream_t bound_stream{};
  std::shared_ptr<void> bound_owner;
  TransferLease(std::shared_ptr<TransferPoolState> state, std::size_t index)
      : state(std::move(state)), index(index) {}
  auto& Slot() const { return state->slots[index]; }
  auto& Lane() const { return state->lanes[index % state->native_streams]; }
  hipStream_t Stream() const {
    return bound_stream ? bound_stream : Lane().stream;
  }
  bool Failed() const noexcept {
    return failed || (!bound_stream && Lane().failed);
  }
  void Fail() noexcept {
    failed = true;
    state->degraded = true;
    // A bound execution stream was supplied and retained by its own owner;
    // its failure does not implicate the unused native transfer stream.
    if (!bound_stream)
      Lane().failed = true;
  }
  cache::TransferResult Drain() noexcept {
    // Stream and completion owners can independently drain the same lease.
    // Serialize its fence state, never holding the pool lock while HIP waits.
    const std::lock_guard lock(drain_mutex);
    if (!state->shared) {
      if (hipStreamSynchronize(Stream()) != hipSuccess)
        Fail();
    } else if (native_used) {
      // Synchronize does not seal the lease: callers may submit again through
      // a saved Native handle without asking Native() for it a second time.
      // Refresh an unsealed fence on every drain, but only after its previous
      // recording settled. Completed leases keep their original fence so
      // destruction cannot wait for unrelated work submitted after Complete.
      if (!sealed && event_settled) {
        event_settled = false;
        if (!event_failed)
          event_recorded = false;
      }
      // Complete recorded the lease's fence. A failed/unfinished Native
      // submission needs the same fence before releasing its operands. Each
      // event belongs to exactly one lease and is never rerecorded while its
      // previous recording is in flight. Native may resume after a successful
      // explicit Synchronize, at which point the old fence is already settled.
      if (!event_recorded && !event_failed) {
        if (hipEventRecord(Slot().event, Stream()) == hipSuccess) {
          event_recorded = true;
          event_settled = false;
        } else {
          event_failed = true;
          Fail();
        }
      }
      if (event_recorded && !event_settled) {
        if (hipEventSynchronize(Slot().event) == hipSuccess)
          event_settled = true;
        else
          Fail();
      }
      // An event error is not evidence that copies stopped. Quarantine first,
      // then drain the affected stream before any source owner can unwind.
      // The successful path never waits for unrelated work queued after its
      // own fence, even when another producer keeps submitting to this lane.
      if (!event_settled) {
        Fail();
        if (hipStreamSynchronize(Stream()) != hipSuccess)
          Fail();
        else
          event_settled = true;
      }
    }
    return Failed() ? cache::TransferResult::kFailed
                    : cache::TransferResult::kSucceeded;
  }
  ~TransferLease() {
    // Metadata allocation may fail before any slot is claimed. Such a lease
    // must neither synchronize nor release another borrower's stream.
    if (!claimed)
      return;
    (void)Drain();
    const std::lock_guard lock(state->mutex);
    Slot().failed = failed;
    Slot().busy = false;
  }
};
}  // namespace detail
namespace {
class TransferSignal final : public cache::CompletionSignal {
public:
  explicit TransferSignal(std::shared_ptr<detail::TransferLease> lease)
      : lease_(std::move(lease)) {}
  bool Ready() const noexcept override {
    const auto status = hipEventQuery(lease_->Slot().event);
    if (status != hipSuccess && status != hipErrorNotReady)
      lease_->Fail();
    // A query error is not proof that buffer access has stopped.
    return status == hipSuccess;
  }
  cache::TransferResult Wait() noexcept override {
    if (!lease_->state->shared &&
        hipEventSynchronize(lease_->Slot().event) != hipSuccess)
      lease_->Fail();
    // Drain also on error; adapters may have submitted work through Native.
    return lease_->Drain();
  }

private:
  std::shared_ptr<detail::TransferLease> lease_;
};
}  // namespace

TransferStream::TransferStream(std::shared_ptr<detail::TransferLease> lease)
    : lease_(std::move(lease)) {}
TransferStream::~TransferStream() {
  (void)Synchronize();
}
void TransferStream::BindTo(hipStream_t stream, std::shared_ptr<void> owner) {
  if (!stream || !owner)
    throw std::invalid_argument("bound transfer requires a stream owner");
  if (lease_->sealed || lease_->Failed() || lease_->native_used ||
      lease_->bound_stream)
    throw std::logic_error("transfer stream was already used");
  lease_->bound_stream = stream;
  lease_->bound_owner = std::move(owner);
}
hipStream_t TransferStream::Native() const {
  if (lease_->sealed || lease_->Failed())
    throw std::logic_error("transfer stream is sealed or failed");
  lease_->native_used = true;
  // A previous manual Synchronize settled its recording before Native could
  // resume. A future synchronization must cover these new submissions too.
  lease_->event_recorded = false;
  lease_->event_settled = false;
  lease_->event_failed = false;
  return lease_->Stream();
}
cache::TransferResult TransferStream::Synchronize() noexcept {
  return lease_->Drain();
}
cache::Completion TransferStream::Complete() {
  const auto stream = Native();
  // Allocate host metadata before recording; on an allocation failure the
  // caller still owns the stream and must drain any Native work.
  auto signal = std::make_unique<TransferSignal>(lease_);
  const auto status = hipEventRecord(lease_->Slot().event, stream);
  lease_->sealed = true;
  if (status != hipSuccess) {
    // Do not rerecord after an error: the runtime may have submitted part of
    // the operation even though no usable fence was returned.
    lease_->event_failed = true;
    lease_->Fail();
    (void)Synchronize();
    Check(status);
  }
  lease_->event_recorded = true;
  return cache::Completion(std::move(signal));
}
cache::Completion TransferStream::Copy(void* destination, const void* source,
                                       std::size_t bytes,
                                       CopyDirection direction,
                                       std::size_t piece_bytes) {
  const auto stream = Native();
  if (!piece_bytes || (bytes && (!destination || !source)) ||
      (direction != CopyDirection::kDeviceToHost &&
       direction != CopyDirection::kHostToDevice))
    throw std::invalid_argument("invalid bounded transfer");
  const auto kind = direction == CopyDirection::kDeviceToHost
                        ? hipMemcpyDeviceToHost
                        : hipMemcpyHostToDevice;
  try {
    for (std::size_t offset = 0; offset < bytes;) {
      const auto piece = std::min(piece_bytes, bytes - offset);
      const auto status = hipMemcpyAsync(
          static_cast<std::byte*>(destination) + offset,
          static_cast<const std::byte*>(source) + offset, piece, kind, stream);
      if (status != hipSuccess)
        lease_->Fail();
      Check(status);
      offset += piece;
    }
    return Complete();
  } catch (...) {
    // Includes host metadata failure after some copies were already queued.
    (void)Synchronize();
    throw;
  }
}
TransferPool::TransferPool(std::size_t capacity)
    : TransferPool(capacity, capacity) {}
TransferPool::TransferPool(std::size_t capacity, std::size_t native_streams) {
  if (!capacity || !native_streams || native_streams > capacity)
    throw std::invalid_argument("invalid transfer pool resource counts");
  auto state = std::make_shared<detail::TransferPoolState>();
  state->lanes =
      std::make_unique<detail::TransferPoolState::Lane[]>(native_streams);
  state->native_streams = native_streams;
  state->shared = native_streams < capacity;
  state->slots.resize(capacity);
  for (std::size_t i = 0; i < capacity; ++i) {
    // Keep the exclusive constructor's creation order; a partial failure is
    // cleaned by state ownership whether it leaves a stream or only events.
    if (i < native_streams)
      Check(hipStreamCreateWithFlags(&state->lanes[i].stream,
                                     hipStreamNonBlocking));
    const unsigned flags =
        hipEventDisableTiming | (state->shared ? hipEventReleaseToSystem : 0);
    Check(hipEventCreateWithFlags(&state->slots[i].event, flags));
  }
  state_ = std::move(state);
}
std::unique_ptr<TransferStream> TransferPool::TryAcquire() {
  const std::lock_guard lock(state_->mutex);
  for (std::size_t i = 0; i < state_->slots.size(); ++i) {
    auto& slot = state_->slots[i];
    if (slot.failed || state_->lanes[i % state_->native_streams].failed)
      continue;
    if (slot.busy)
      continue;
    auto lease = std::make_shared<detail::TransferLease>(state_, i);
    // An unclaimed lease has no return-to-pool action on allocation failure.
    auto stream = std::unique_ptr<TransferStream>(new TransferStream(lease));
    slot.busy = true;
    lease->claimed = true;
    return stream;
  }
  if (state_->shared && state_->degraded)
    throw std::runtime_error("transfer pool capacity lost to quarantine");
  return nullptr;
}

}  // namespace gufo::hip
