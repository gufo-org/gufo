#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_BUFFER_POOL_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_BUFFER_POOL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next {

/// Recycled snapshot payload buffers.
///
/// Every capture used to fault in a fresh mapping: even with parallel
/// MADV_POPULATE_WRITE workers, populating ~1 GB before the
/// device-to-host copy costs ~170 ms at a 33k-token context, paid on the
/// synchronous frontier-freeze of every warm turn. Released buffers return
/// here instead and are handed to the next capture fully populated. Every
/// capture overwrites its whole buffer, so reuse never exposes stale
/// bytes; the process boundary still separates clients.
class SnapshotBufferPool final {
public:
  struct AcquiredBuffer {
    std::unique_ptr<std::uint8_t[]> data;
    std::uint64_t capacity{0};
    bool populated{false};
  };

  struct Limits {
    std::size_t max_buffers{8};
    std::uint64_t max_bytes{std::uint64_t{4} << 30};
  };

  explicit SnapshotBufferPool(Limits limits) noexcept
      : limits_(limits) {}
  ~SnapshotBufferPool() = default;
  SnapshotBufferPool(const SnapshotBufferPool&) = delete;
  SnapshotBufferPool& operator=(const SnapshotBufferPool&) = delete;
  SnapshotBufferPool(SnapshotBufferPool&&) = delete;
  SnapshotBufferPool& operator=(SnapshotBufferPool&&) = delete;

  /// Hands out the smallest buffer that fits, so mixed snapshot sizes do
  /// not strand large buffers behind small ones. Fresh allocations when
  /// nothing fits.
  [[nodiscard]] AcquiredBuffer Acquire(std::uint64_t size) {
    AcquiredBuffer acquired;
    acquired.capacity = size;
    const std::lock_guard lock(mutex_);
    std::size_t best = buffers_.size();
    for (std::size_t i = 0; i < buffers_.size(); ++i) {
      if (buffers_[i].second < size) {
        continue;
      }
      if (best == buffers_.size() ||
          buffers_[i].second < buffers_[best].second) {
        best = i;
      }
    }
    if (best != buffers_.size()) {
      acquired.capacity = buffers_[best].second;
      acquired.populated = true;
      acquired.data = std::move(buffers_[best].first);
      // The pooled-byte budget tracks retained buffers only; taking one
      // out must release its allowance so repeated reuse cannot exhaust
      // the cap cumulatively.
      pooled_bytes_ -= acquired.capacity;
      buffers_.erase(buffers_.begin() + static_cast<std::ptrdiff_t>(best));
      return acquired;
    }
    acquired.data =
        std::unique_ptr<std::uint8_t[]>(new std::uint8_t[size]);
    return acquired;
  }

  /// Returns a buffer for the next capture. Anything beyond the limits is
  /// freed instead.
  void Release(std::unique_ptr<std::uint8_t[]> buffer,
               std::uint64_t capacity) noexcept {
    try {
      const std::lock_guard lock(mutex_);
      if (pooled_bytes_ + capacity > limits_.max_bytes ||
          buffers_.size() >= limits_.max_buffers) {
        return;
      }
      // Charge only after the insertion succeeded: if vector growth throws,
      // the buffer is dropped by the catch below and must not keep its byte
      // allowance (a leaked charge on the static pool would persist for the
      // process lifetime and eventually disable recycling).
      buffers_.emplace_back(std::move(buffer), capacity);
      pooled_bytes_ += capacity;
    } catch (...) {
      // Dropping a buffer only costs the next capture a fresh mapping.
    }
  }

  [[nodiscard]] std::size_t pooled_buffers() const noexcept {
    const std::lock_guard lock(mutex_);
    return buffers_.size();
  }

  [[nodiscard]] std::uint64_t pooled_bytes() const noexcept {
    const std::lock_guard lock(mutex_);
    return pooled_bytes_;
  }

private:
  Limits limits_;
  mutable std::mutex mutex_;
  std::vector<std::pair<std::unique_ptr<std::uint8_t[]>, std::uint64_t>>
      buffers_;
  std::uint64_t pooled_bytes_{0};
};

inline SnapshotBufferPool& SnapshotBuffers() noexcept {
  // Enough for the concurrent captures of a busy multi-session host plus
  // slack, without holding release-worthy memory against the host budget.
  static SnapshotBufferPool pool({.max_buffers = 8, .max_bytes = std::uint64_t{4} << 30});
  return pool;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SNAPSHOT_BUFFER_POOL_HPP_
