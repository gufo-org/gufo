#ifndef GUFO_SERVER_CONTINUATION_CACHE_HPP_
#define GUFO_SERVER_CONTINUATION_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace gufo::server {

using ContinuationToken = std::uint32_t;

/// Model-private continuation state retained by the common serving cache.
///
/// Implementations own all attention, recurrent, position, and graph-bound
/// state. The cache deliberately cannot inspect or copy those bytes.
class ContinuationState {
public:
  ContinuationState() = default;
  virtual ~ContinuationState() = default;

  ContinuationState(const ContinuationState&) = delete;
  ContinuationState& operator=(const ContinuationState&) = delete;
  ContinuationState(ContinuationState&&) = delete;
  ContinuationState& operator=(ContinuationState&&) = delete;

  /// Discards any partially or fully computed continuation.
  virtual void Invalidate() noexcept = 0;
};

/// Immutable model-private continuation payload.
class ContinuationSnapshot {
public:
  ContinuationSnapshot() = default;
  virtual ~ContinuationSnapshot() = default;

  ContinuationSnapshot(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot& operator=(const ContinuationSnapshot&) = delete;
  ContinuationSnapshot(ContinuationSnapshot&&) = delete;
  ContinuationSnapshot& operator=(ContinuationSnapshot&&) = delete;

  [[nodiscard]] virtual std::size_t PayloadBytes() const noexcept = 0;
};

enum class SnapshotEventAction : std::uint8_t {
  kRemoved,
  kSkipped,
};

enum class SnapshotEventReason : std::uint8_t {
  kByteCapacity,
  kEntryCapacity,
  kExactReplacement,
  kCaptureFailure,
  kReservationMismatch,
};

/// Sanitized snapshot-retention event.
///
/// Token values and prompt contents are deliberately absent.
struct SnapshotEvent {
  SnapshotEventAction action{SnapshotEventAction::kSkipped};
  SnapshotEventReason reason{SnapshotEventReason::kByteCapacity};
  std::size_t snapshot_bytes{0};
  std::size_t token_count{0};
  std::size_t retained_snapshot_bytes{0};
  std::size_t reserved_snapshot_bytes{0};
  std::size_t capacity_bytes{0};
};

/// Bounded exact-prefix cache over opaque model continuation states.
///
/// The first deployment uses one entry. Supporting a bounded entry count here
/// keeps cache policy independent from Qwen, DeepSeek, and future providers.
class ContinuationCache {
public:
  using StateFactory = std::function<std::unique_ptr<ContinuationState>()>;
  using CancellationCheck = std::function<bool()>;
  using SnapshotRestore =
      std::function<void(ContinuationState&, const ContinuationSnapshot&)>;
  using SnapshotCapacity = std::function<std::size_t()>;
  using SnapshotEventSink = std::function<void(const SnapshotEvent&)>;

  struct SnapshotSupport {
    SnapshotRestore restore;
    SnapshotCapacity capacity_bytes;
    SnapshotEventSink on_event;
  };

  class Lease {
  public:
    Lease() = default;
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
      return cache_ != nullptr;
    }
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept { return cache_hit_; }
    [[nodiscard]] std::size_t cached_tokens() const noexcept {
      return cached_tokens_;
    }
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept {
      return restored_snapshot_bytes_;
    }
    [[nodiscard]] double restore_ms() const noexcept { return restore_ms_; }
    [[nodiscard]] bool restored_from_disk() const noexcept {
      return restored_from_disk_;
    }

    /// Records a successful restore performed by an optional lower cache tier.
    void AdoptRestoredPrefix(std::size_t cached_tokens,
                             std::size_t restored_bytes, double restore_ms);

    /// Reserves aggregate retained-snapshot capacity before model allocation.
    ///
    /// Byte-pressure evictions happen synchronously before this returns true.
    [[nodiscard]] bool TryReserveSnapshot(std::size_t snapshot_bytes,
                                          std::size_t token_count);

    /// Releases an admitted reservation and records a sanitized skip reason.
    void SkipSnapshot(SnapshotEventReason reason, std::size_t snapshot_bytes,
                      std::size_t token_count) noexcept;

    /// Atomically publishes the state and the exact tokens it represents.
    ///
    /// Returns the payload bytes actually retained. Snapshot-mode commits may
    /// publish no snapshot when admission or capture failed.
    std::size_t Commit(
        std::vector<ContinuationToken> tokens,
        std::unique_ptr<ContinuationSnapshot> snapshot = nullptr);

    /// Explicitly discards partial state. The destructor does the same if a
    /// lease is not committed.
    void Invalidate() noexcept;

  private:
    friend class ContinuationCache;

    Lease(ContinuationCache* cache, std::size_t index, bool cache_hit,
          std::size_t cached_tokens, std::size_t source_index,
          std::size_t restored_snapshot_bytes, double restore_ms) noexcept;

    ContinuationCache* cache_{nullptr};
    std::size_t index_{0};
    bool cache_hit_{false};
    std::size_t cached_tokens_{0};
    std::size_t source_index_{0};
    std::size_t restored_snapshot_bytes_{0};
    double restore_ms_{0.0};
    bool restored_from_disk_{false};
    std::size_t reserved_snapshot_bytes_{0};
  };

  ContinuationCache(std::size_t capacity, const StateFactory& factory,
                    SnapshotSupport snapshot_support = {});
  ~ContinuationCache();

  ContinuationCache(const ContinuationCache&) = delete;
  ContinuationCache& operator=(const ContinuationCache&) = delete;
  ContinuationCache(ContinuationCache&&) = delete;
  ContinuationCache& operator=(ContinuationCache&&) = delete;

  [[nodiscard]] Lease Acquire(std::span<const ContinuationToken> prompt,
                              const CancellationCheck& is_cancelled = {});

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t snapshot_capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t retained_snapshot_bytes() const noexcept;
  [[nodiscard]] std::size_t reserved_snapshot_bytes() const noexcept;

private:
  struct Entry;

  [[nodiscard]] ContinuationState& StateAt(std::size_t index);
  [[nodiscard]] bool ReserveSnapshot(std::size_t source_index,
                                     std::size_t snapshot_bytes,
                                     std::size_t token_count);
  void SkipSnapshot(std::size_t reservation_bytes, SnapshotEventReason reason,
                    std::size_t snapshot_bytes,
                    std::size_t token_count) noexcept;
  [[nodiscard]] std::size_t Commit(
      std::size_t index, std::size_t source_index,
      std::size_t reservation_bytes, std::vector<ContinuationToken> tokens,
      std::unique_ptr<ContinuationSnapshot> snapshot);
  void Invalidate(std::size_t index, std::size_t reservation_bytes) noexcept;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_CONTINUATION_CACHE_HPP_
