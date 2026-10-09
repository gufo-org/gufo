#ifndef GUFO_CACHE_LEDGER_HPP_
#define GUFO_CACHE_LEDGER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace gufo::cache {

enum class ResourceCategory : std::uint8_t {
  kBackingFree,
  kBackingReserved,
  kBackingMaterialized,
  kPrivateState,
  kPrivateTail,
  kMetadata,
  kTransferStaging,
};
inline constexpr std::size_t kResourceCategoryCount = 7;

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
enum class LedgerStep : std::uint8_t { kReserve, kConvert, kPin, kRelease };
struct LedgerLockCost {
  std::uint64_t calls{}, wait_ns{}, hold_ns{}, max_hold_ns{};
};
namespace detail {
struct LedgerState;
struct ChargeToken;
struct PinToken;
}  // namespace detail
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
  // Queued jobs may pin an assigned, already committed spill block before its
  // rows materialize. Pending new allocations cannot be pinned. Cancellation
  // returns backing to free only after every queued pin has released it.
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
  // Only a converted free backing block can admit a borrower. The whole block
  // moves free -> reserved -> materialized; it returns to free when its last
  // borrower/chunk/pin disappears. The pool retains the original free handle.
  // Copies of that handle cannot assign the same block twice. Card 08 supplies
  // already committed fixed-size blocks; borrowed live rows get no extra
  // charge.
  [[nodiscard]] ResourceReservation ReserveSpill() const;
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
  [[nodiscard]] ResourceReservation Reserve(ResourceCategory, std::size_t);
  [[nodiscard]] ResourceSnapshot Snapshot() const;
  [[nodiscard]] std::array<LedgerLockCost, 4> LockCosts() const;
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
