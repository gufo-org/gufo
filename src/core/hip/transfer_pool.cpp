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
  struct Slot {
    hipStream_t stream{};
    hipEvent_t event{};
    bool busy{};
    bool failed{};
  };
  std::mutex mutex;
  std::vector<Slot> slots;
  ~TransferPoolState() {
    for (auto& slot : slots) {
      if (slot.stream) {
        (void)hipStreamSynchronize(slot.stream);
        (void)hipStreamDestroy(slot.stream);
      }
      if (slot.event)
        (void)hipEventDestroy(slot.event);
    }
  }
};
struct TransferLease {
  std::shared_ptr<TransferPoolState> state;
  std::size_t index;
  std::atomic<bool> failed{false};
  bool sealed{};
  bool claimed{};
  TransferLease(std::shared_ptr<TransferPoolState> state, std::size_t index)
      : state(std::move(state)), index(index) {}
  auto& Slot() const { return state->slots[index]; }
  cache::TransferResult Drain() noexcept {
    if (hipStreamSynchronize(Slot().stream) != hipSuccess)
      failed = true;
    return failed ? cache::TransferResult::kFailed
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
      lease_->failed = true;
    // A query error is not proof that buffer access has stopped.
    return status == hipSuccess;
  }
  cache::TransferResult Wait() noexcept override {
    if (hipEventSynchronize(lease_->Slot().event) != hipSuccess)
      lease_->failed = true;
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
hipStream_t TransferStream::Native() const {
  if (lease_->sealed || lease_->failed)
    throw std::logic_error("transfer stream is sealed or failed");
  return lease_->Slot().stream;
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
    lease_->failed = true;
    (void)Synchronize();
    Check(status);
  }
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
        lease_->failed = true;
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
TransferPool::TransferPool(std::size_t capacity) {
  if (!capacity)
    throw std::invalid_argument("transfer pool requires a nonzero capacity");
  auto state = std::make_shared<detail::TransferPoolState>();
  state->slots.resize(capacity);
  for (auto& slot : state->slots) {
    Check(hipStreamCreateWithFlags(&slot.stream, hipStreamNonBlocking));
    Check(hipEventCreateWithFlags(&slot.event, hipEventDisableTiming));
  }
  state_ = std::move(state);
}
std::unique_ptr<TransferStream> TransferPool::TryAcquire() {
  const std::lock_guard lock(state_->mutex);
  for (std::size_t i = 0; i < state_->slots.size(); ++i) {
    auto& slot = state_->slots[i];
    if (slot.busy || slot.failed)
      continue;
    auto lease = std::make_shared<detail::TransferLease>(state_, i);
    // An unclaimed lease has no return-to-pool action on allocation failure.
    auto stream = std::unique_ptr<TransferStream>(new TransferStream(lease));
    slot.busy = true;
    lease->claimed = true;
    return stream;
  }
  return nullptr;
}

}  // namespace gufo::hip
