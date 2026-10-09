#ifndef GUFO_CACHE_LEDGER_HPP_
#define GUFO_CACHE_LEDGER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace gufo::cache {

enum class ResourceCategory : std::uint8_t {
  kBackingFree,
  kBackingAssigned,
  kBackingMaterialized,
  kPrivateState,
  kPrivateTail,
  kMetadata,
  kTransferStaging,
  kCount,
};
inline constexpr std::size_t kResourceCategoryCount =
    static_cast<std::size_t>(ResourceCategory::kCount);

struct ResourceLimits {
  std::size_t total_bytes;
  std::size_t ram_bytes;  // backing + private state/tails + metadata
  std::size_t staging_bytes;
  std::size_t
      persistence_pinned_bytes;  // overlay, not an additional RAM charge
};
struct ResourceSnapshot {
  // Includes admission reservations. reserved_bytes is a subset of bytes.
  std::array<std::size_t, kResourceCategoryCount> bytes{};
  std::array<std::size_t, kResourceCategoryCount> reserved_bytes{};
  std::size_t total_bytes{};
  std::size_t ram_bytes{};
  std::size_t persistence_pinned_bytes{};
  std::array<std::size_t, kResourceCategoryCount> peak_bytes{};
  std::size_t peak_total_bytes{};
  std::size_t peak_ram_bytes{};
  std::size_t peak_persistence_pinned_bytes{};
  bool operator==(const ResourceSnapshot&) const = default;
};
enum class LedgerStep : std::uint8_t {
  kReserve,
  kConvert,
  kPin,
  kRelease,
  kCount
};
inline constexpr std::size_t kLedgerStepCount =
    static_cast<std::size_t>(LedgerStep::kCount);
struct LedgerLockCost {
  std::uint64_t calls{}, wait_ns{}, hold_ns{}, max_hold_ns{};
};
namespace detail {
struct LedgerState;
struct ChargeToken;
struct PinToken;
}  // namespace detail
// Stable allocation capacity and current ownership, read under the ledger lock.
// An original pool handle cannot stand in for its assigned payload owner.
struct ResourceAllocationInfo {
  ResourceCategory category;
  std::size_t bytes;
  bool reserved;
  bool pool_backing;
  bool assigned_backing;
};
class ResourceCharge;
class PersistencePin;

// Admission precedes allocation/capture; destruction rolls back. Convert only
// after allocation/commit/copy succeeds. It consumes the reservation on success
// and retains it on failure. No physical allocator is called by this package.
class ResourceReservation {
public:
  ResourceReservation() = default;
  ~ResourceReservation() = default;
  ResourceReservation(ResourceReservation&&) noexcept = default;
  ResourceReservation& operator=(ResourceReservation&&) noexcept = default;
  ResourceReservation(const ResourceReservation&) = delete;
  ResourceReservation& operator=(const ResourceReservation&) = delete;
  [[nodiscard]] ResourceCharge Convert();
  [[nodiscard]] ResourceAllocationInfo Info() const;
  // Queued jobs may pin assigned spill or borrowed-tail backing before its
  // rows materialize. Pending new allocations/private captures cannot be
  // pinned. Cancellation returns backing to free only after every queued pin
  // has released it.
  [[nodiscard]] PersistencePin PinPersistence() const;
  explicit operator bool() const noexcept { return bool(token_); }

private:
  friend class ResourceLedger;
  friend class ResourceCharge;
  explicit ResourceReservation(std::shared_ptr<detail::ChargeToken>);
  std::shared_ptr<detail::ChargeToken> token_;
};

// Copying a charge or pin shares its accounting; the last reference releases
// it. These tokens account for bytes, not device objects. Owners must retain
// the tokens until actual allocation/buffer accesses finish, and free physical
// resources before releasing the final charge. Distinct handle objects can be
// used concurrently; concurrent mutation of one handle needs caller ordering.
class PersistencePin {
public:
  PersistencePin() = default;

private:
  friend class ResourceCharge;
  friend class ResourceReservation;
  explicit PersistencePin(std::shared_ptr<detail::PinToken>);
  std::shared_ptr<detail::PinToken> token_;
};
class ResourceCharge {
public:
  ResourceCharge() = default;
  [[nodiscard]] ResourceAllocationInfo Info() const;
  // Reclassify one already committed pool block as kBackingAssigned (spill),
  // kPrivateState or kPrivateTail without another admission/physical charge.
  // Convert after capture/copy completes; spill then becomes materialized.
  // Cancellation or final owner/pin release returns the entire block to free.
  // Callers may race to assign a block; bad_alloc means another borrower holds
  // it and the pool should try another block or decline capture. Invalid pool
  // handles throw logic_error. No request-path page commitment is required.
  [[nodiscard]] ResourceReservation ReserveBacking(ResourceCategory) const;
  // Only retained payload (assigned spill, chunks, private state/tails) counts
  // as persistence-pinned bytes. Staging and metadata use their own admission
  // categories; retain their charges directly for queued-job lifetimes.
  [[nodiscard]] PersistencePin PinPersistence() const;
  explicit operator bool() const noexcept { return bool(token_); }

private:
  friend class ResourceReservation;
  explicit ResourceCharge(std::shared_ptr<detail::ChargeToken>);
  std::shared_ptr<detail::ChargeToken> token_;
};

// One short lock protects only counters and state transitions. No callbacks,
// physical allocations, I/O or device waits occur under it. Shared state lives
// until the final handle, even if this facade is destroyed earlier.
class ResourceLedger {
public:
  // Clock instrumentation is opt-in for the step microbenchmark. Ordinary
  // ledgers never read the clock; LockCosts then returns zeroes.
  explicit ResourceLedger(ResourceLimits, bool measure_lock_cost = false);
  ~ResourceLedger() = default;
  ResourceLedger(const ResourceLedger&) = delete;
  ResourceLedger& operator=(const ResourceLedger&) = delete;
  ResourceLedger(ResourceLedger&&) = delete;
  ResourceLedger& operator=(ResourceLedger&&) = delete;
  // Admit physical payload allocation at initialization/quiescence, or host
  // metadata/staging capacity. Request-time private capture and borrowed tails
  // use ReserveBacking on existing committed blocks instead of allocating.
  [[nodiscard]] ResourceReservation Reserve(ResourceCategory, std::size_t);
  [[nodiscard]] ResourceSnapshot Snapshot() const;
  // Atomic end/start of a global reporting window. The returned snapshot keeps
  // the ending peaks; next-window peaks start at the current charges/pins.
  // Overlapping request-local attribution belongs to card 19, not this reset.
  [[nodiscard]] ResourceSnapshot SnapshotAndResetPeaks();
  [[nodiscard]] std::array<LedgerLockCost, kLedgerStepCount> LockCosts() const;
  // One-shot deterministic failure after n valid attempts at the given step.
  // bad_alloc also indicates budget/pool exhaustion. Invalid arguments throw
  // invalid_argument; invalid/empty handles throw logic_error. Failures leave
  // snapshots (including peaks) unchanged. Release cannot be failed.
  void FailAfter(LedgerStep, std::size_t n);
  void ClearFaults();

private:
  std::shared_ptr<detail::LedgerState> state_;
};

}  // namespace gufo::cache
#endif
