#ifndef GUFO_CORE_CANCELLABLE_GATE_HPP_
#define GUFO_CORE_CANCELLABLE_GATE_HPP_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::core {

// FIFO admission for a runtime with mutable device state. Waiting clients can
// cancel without waiting for the current inference to finish.
class CancellableGate {
public:
  explicit CancellableGate(std::size_t capacity = 1) : capacity_(capacity) {
    if (capacity == 0)
      throw std::invalid_argument("inference capacity must be positive");
  }
  class Lease {
  public:
    explicit Lease(CancellableGate* gate) : gate_(gate) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : gate_(std::exchange(other.gate_, nullptr)) {}
    ~Lease() {
      if (gate_) {
        {
          const std::lock_guard lock(gate_->mutex_);
          --gate_->active_;
        }
        gate_->ready_.notify_all();
      }
    }

  private:
    CancellableGate* gate_;
  };

  std::optional<Lease> Acquire(const std::function<bool()>& cancelled,
                               std::string* error) {
    std::unique_lock lock(mutex_);
    if (queue_.size() >= 16) {
      if (error)
        *error = "audio request queue is full";
      return std::nullopt;
    }
    const auto ticket = ++next_;
    queue_.push_back(ticket);
    while (true) {
      bool stop = false;
      try {
        stop = cancelled && cancelled();
      } catch (...) {
        std::erase(queue_, ticket);
        ready_.notify_all();
        throw;
      }
      if (stop) {
        std::erase(queue_, ticket);
        ready_.notify_all();
        if (error)
          *error = "audio request cancelled while queued";
        return std::nullopt;
      }
      if (active_ < capacity_ && queue_.front() == ticket) {
        queue_.pop_front();
        ++active_;
        ready_.notify_all();
        return Lease(this);
      }
      ready_.wait_for(lock, std::chrono::milliseconds(10));
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::uint64_t> queue_;
  std::uint64_t next_{0};
  std::size_t active_{0};
  std::size_t capacity_;
};

}  // namespace gufo::core
#endif  // GUFO_CORE_CANCELLABLE_GATE_HPP_
