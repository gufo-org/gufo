#ifndef GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_
#define GUFO_FAMILIES_QWEN_VISION_ADMISSION_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/resource_error.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::size_t kEncoderHidden = 1152;
inline constexpr std::size_t kEncoderFfn = 4304;
inline constexpr std::size_t kEncoderHeads = 16;
inline constexpr std::size_t kEncoderPatchDim = 1536;
inline constexpr std::size_t kPinnedUploadBytes = 16ULL << 20;
[[nodiscard]] constexpr std::size_t EncoderTileRows(std::size_t patches) {
  const auto limit = patches == 4096 ? 64U : 128U;
  return patches < limit ? patches : limit;
}

[[nodiscard]] std::size_t CheckedAdd(std::size_t a, std::size_t b);
[[nodiscard]] std::size_t CheckedMultiply(std::size_t a, std::size_t b);

struct ImageDimensions {
  std::uint32_t width;
  std::uint32_t height;
};

/// Geometry has already passed resizing: both sides are positive multiples
/// of 32 and at most 16,777,216 pixels. The encoder uses the same checks.
[[nodiscard]] std::size_t ImageScratchBytes(ImageDimensions image);
[[nodiscard]] std::size_t ImageEmbeddingBytes(ImageDimensions image,
                                              std::uint32_t output_width);
/// Reservations sample remaining host/device capacity at arrival. The observer
/// preserves platform budgets and runtime reserves. Only unmaterialized
/// obligations are deducted: committed allocations already reduced the
/// observation. Snapshot pending counters BEFORE querying memory so concurrent
/// commitment cannot disappear from both accounting and the free-memory
/// observation.
class AdmissionBudget {
public:
  using RuntimeObserver = std::function<std::size_t()>;
  AdmissionBudget(RuntimeObserver observer, std::size_t execution_bytes = 0);
  [[nodiscard]] cache::ResourceReservation Reserve(std::size_t bytes);
  [[nodiscard]] cache::ResourceReservation ReserveShared(std::size_t bytes);
  /// Last observed capacity after execution obligations (not a startup grant).
  [[nodiscard]] std::size_t Capacity() const noexcept {
    return capacity_.load();
  }
  [[nodiscard]] std::size_t ProtectedBytes() const;
  [[nodiscard]] std::size_t AccountedBytes() const;
  [[nodiscard]] std::size_t PendingBytes() const;
  void ClaimExecutionOwner();
  /// Materialization of already admitted text/deferred execution work. Call
  /// only after allocation completes; restore the obligation before freeing
  /// reusable execution storage. These operations never credit free memory.
  void CommitExecution(std::size_t bytes);
  void ReleaseExecution(std::size_t bytes) noexcept;
  /// Total image/copy/cache ownership, including committed allocations.
  [[nodiscard]] std::size_t Outstanding() const;

private:
  friend class AdmissionLease;
  void ReplaceCommitted(std::size_t old_bytes, std::size_t new_bytes) noexcept;
  [[nodiscard]] std::size_t PendingLocked() const;
  [[nodiscard]] cache::ResourceReservation Reserve(
      cache::ResourceCategory category, std::size_t bytes);
  mutable std::mutex mutex_;
  std::atomic<std::size_t> capacity_;
  RuntimeObserver observer_;
  std::size_t execution_pending_{0};
  std::size_t materialized_{0};
  cache::ResourceLedger ledger_;
  std::atomic_flag execution_owner_{};
};

/// Decode/resize processes one image at a time. The working allowance covers
/// the 32 Mi-pixel decoder limit, orientation copy, RGB resize intermediate,
/// filter banks and decoder workspaces. Each prepared image retains at most
/// 16 Mi-pixels of RGB8. These are logical bounds, never eager allocations.
inline constexpr std::size_t kPreparationScratchBytes = std::size_t{1} << 30;
inline constexpr std::size_t kPreparedImageBytes = std::size_t{48} << 20;

/// Reserve before preparing or counting an image prompt. Finish releases only
/// transient work after decode/resize returns. Keep this owner alive until all
/// prepared pixels are freed, including queued/cancelled requests and image
/// history retained by a session. Construction failure releases both claims.
/// A null budget or no images preserves the conservative/text-only path.
class PreparationReservation {
public:
  PreparationReservation(std::shared_ptr<AdmissionBudget> budget,
                         std::size_t image_count);
  PreparationReservation(const PreparationReservation&) = delete;
  PreparationReservation& operator=(const PreparationReservation&) = delete;
  void Finish() noexcept;
  /// On-demand prepared owners keep only actual RGB capacity after resizing.
  void Finish(std::size_t pixel_bytes);

private:
  cache::ResourceReservation scratch_;
  cache::ResourceReservation pixels_;
  cache::ResourceCharge committed_pixels_;
};

/// One session's high-water claim. Grow precedes any image allocation; failed
/// growth leaves it unchanged. A physical allocation failure after Grow keeps
/// the claim, since some buffers may already exist. Clear is legal only after
/// all corresponding device/host work has finished and buffers are freed.
/// Destruction has the same ordering requirement. A null budget is the
/// conservative policy and performs no additional admission.
class AdmissionLease {
public:
  AdmissionLease() = default;
  explicit AdmissionLease(std::shared_ptr<AdmissionBudget> budget);
  ~AdmissionLease();
  AdmissionLease(AdmissionLease&&) noexcept;
  AdmissionLease& operator=(AdmissionLease&&) noexcept;
  AdmissionLease(const AdmissionLease&) = delete;
  AdmissionLease& operator=(const AdmissionLease&) = delete;
  void Grow(std::size_t bytes);
  /// Call after transient allocations/work end; retained and future required
  /// buffers must remain covered. Never allocates and cannot grow the claim.
  void Shrink(std::size_t bytes) noexcept;
  /// Actual retained bytes after synchronous allocation/work completion.
  /// Clear this marker before releasing/replacing covered physical storage.
  void SetCommitted(std::size_t bytes) noexcept;
  void Clear() noexcept;
  [[nodiscard]] std::size_t Bytes() const noexcept { return bytes_; }

private:
  std::shared_ptr<AdmissionBudget> budget_;
  struct Part {
    cache::ResourceReservation reservation;
    std::size_t bytes;
  };
  std::vector<Part> reservations_;
  std::size_t bytes_{0};
  std::size_t committed_{0};
};

struct StartupCapacity {
  std::size_t device_admission_bytes;
  std::size_t host_ceiling_bytes;
  std::size_t available_bytes;
  [[nodiscard]] const char* LimiterName() const noexcept;
};

/// Both inputs describe remaining capacity at the same loading phase. Loaded
/// weights and committed shared buffers are already reflected in these values;
/// their accounting totals must never be added back to either observation.
[[nodiscard]] StartupCapacity SelectStartupCapacity(
    std::size_t device_admission_bytes, std::size_t host_ceiling_bytes);

struct StartupMemoryPlan {
  std::size_t state_capacity_bytes;
  std::size_t per_session_bytes;
  std::size_t shared_reservation_bytes;
  [[nodiscard]] bool Fits(std::size_t sessions) const noexcept;
  [[nodiscard]] std::size_t RequiredBytes(std::size_t sessions) const;
  [[nodiscard]] std::size_t DeficitBytes(std::size_t sessions,
                                         std::size_t available) const;
};

/// Preserve the device/state gate: deduct deferred work and compare the
/// selected per-session estimate. Optional cache growth and runtime image
/// reserves are not startup prerequisites.
[[nodiscard]] StartupMemoryPlan PlanStartupMemory(
    std::size_t available_bytes, std::size_t deferred_scratch,
    std::size_t base_per_session, std::size_t conservative_per_session,
    bool on_demand = false);

}  // namespace gufo::models::qwen::vision

#endif
