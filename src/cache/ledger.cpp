#include "src/cache/ledger.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gufo::cache {
namespace {
constexpr std::size_t Index(ResourceCategory category) {
  return static_cast<std::size_t>(category);
}
constexpr std::size_t Index(LedgerStep step) {
  return static_cast<std::size_t>(step);
}
constexpr auto Free = ResourceCategory::kBackingFree;
constexpr auto Spill = ResourceCategory::kBackingAssigned;
constexpr auto Chunk = ResourceCategory::kBackingMaterialized;
constexpr auto Staging = ResourceCategory::kTransferStaging;
}  // namespace
namespace detail {
struct LedgerState {
  explicit LedgerState(ResourceLimits limits, bool timing)
      : limits(limits), timing(timing) {}
  std::mutex mutex;
  const ResourceLimits limits;
  const bool timing;
  ResourceSnapshot snapshot;
  std::array<std::optional<std::size_t>, kLedgerStepCount> faults{};
  std::array<LedgerLockCost, kLedgerStepCount> costs{};
  void Fail(LedgerStep step) {
    auto& count = faults[Index(step)];
    if (!count)
      return;
    if (*count) {
      --*count;
      return;
    }
    count.reset();
    throw std::bad_alloc();
  }
  void Peaks() noexcept {
    auto& s = snapshot;
    for (std::size_t i = 0; i != kResourceCategoryCount; ++i)
      s.peak_bytes[i] = std::max(s.peak_bytes[i], s.bytes[i]);
    s.peak_total_bytes = std::max(s.peak_total_bytes, s.total_bytes);
    s.peak_ram_bytes = std::max(s.peak_ram_bytes, s.ram_bytes);
    s.peak_persistence_pinned_bytes =
        std::max(s.peak_persistence_pinned_bytes, s.persistence_pinned_bytes);
  }
};
class Lock {
public:
  Lock(const Lock&) = delete;
  Lock& operator=(const Lock&) = delete;
  Lock(Lock&&) = delete;
  Lock& operator=(Lock&&) = delete;
  Lock(LedgerState& state, LedgerStep step)
      : state_(state), step_(step), lock_(state.mutex, std::defer_lock) {
    if (state_.timing)
      waiting_ = Clock::now();
    lock_.lock();
    if (state_.timing)
      acquired_ = Clock::now();
  }
  ~Lock() {
    if (state_.timing) {
      const auto end = Clock::now();
      auto& c = state_.costs[Index(step_)];
      const auto hold = Ns(end - acquired_);
      ++c.calls;
      c.wait_ns += Ns(acquired_ - waiting_);
      c.hold_ns += hold;
      c.max_hold_ns = std::max(c.max_hold_ns, hold);
    }
  }

private:
  using Clock = std::chrono::steady_clock;
  static std::uint64_t Ns(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
        .count();
  }
  LedgerState& state_;
  LedgerStep step_;
  std::unique_lock<std::mutex> lock_;
  Clock::time_point waiting_{}, acquired_{};
};
struct Allocation {
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&&) = delete;
  Allocation& operator=(Allocation&&) = delete;
  std::shared_ptr<LedgerState> ledger;
  ResourceCategory category;
  const bool backing;
  std::size_t bytes;
  bool admitted{false};
  bool reserved{true};
  std::size_t pins{0};
  Allocation(std::shared_ptr<LedgerState> ledger, ResourceCategory category,
             std::size_t bytes)
      : ledger(std::move(ledger)),
        category(category),
        backing(category == Free),
        bytes(bytes) {}
  ~Allocation() {
    if (!admitted)
      return;
    const Lock lock(*ledger, LedgerStep::kRelease);
    assert(pins == 0);
    auto& s = ledger->snapshot;
    s.bytes[Index(category)] -= bytes;
    if (reserved)
      s.reserved_bytes[Index(category)] -= bytes;
    s.total_bytes -= bytes;
    if (category != Staging)
      s.ram_bytes -= bytes;
  }
};
struct ChargeToken {
  ChargeToken(const ChargeToken&) = delete;
  ChargeToken& operator=(const ChargeToken&) = delete;
  ChargeToken(ChargeToken&&) = delete;
  ChargeToken& operator=(ChargeToken&&) = delete;
  std::shared_ptr<Allocation> allocation;
  bool pooled{false};
  bool active{false};
  ChargeToken(std::shared_ptr<Allocation> allocation, bool pooled)
      : allocation(std::move(allocation)), pooled(pooled) {}
  ~ChargeToken() {
    if (!pooled || !active)
      return;
    auto& a = *allocation;
    const Lock lock(*a.ledger, LedgerStep::kRelease);
    assert(a.pins == 0 && a.backing && a.category != Free);
    auto& s = a.ledger->snapshot;
    s.bytes[Index(a.category)] -= a.bytes;
    if (a.reserved)
      s.reserved_bytes[Index(a.category)] -= a.bytes;
    s.bytes[Index(Free)] += a.bytes;
    a.category = Free;
    a.reserved = false;
    a.ledger->Peaks();
    // allocation's shared_ptr is destroyed after this lock is released.
  }
};
struct PinToken {
  PinToken(const PinToken&) = delete;
  PinToken& operator=(const PinToken&) = delete;
  PinToken(PinToken&&) = delete;
  PinToken& operator=(PinToken&&) = delete;
  std::shared_ptr<ChargeToken> charge;
  bool active{false};
  explicit PinToken(std::shared_ptr<ChargeToken> charge)
      : charge(std::move(charge)) {}
  ~PinToken() {
    if (!active)
      return;
    auto& a = *charge->allocation;
    const Lock lock(*a.ledger, LedgerStep::kRelease);
    assert(a.pins);
    if (--a.pins == 0)
      a.ledger->snapshot.persistence_pinned_bytes -= a.bytes;
  }
};
}  // namespace detail

ResourceLedger::ResourceLedger(ResourceLimits limits, bool timing)
    : state_(std::make_shared<detail::LedgerState>(limits, timing)) {}
ResourceReservation::ResourceReservation(
    std::shared_ptr<detail::ChargeToken> token)
    : token_(std::move(token)) {}
ResourceCharge::ResourceCharge(std::shared_ptr<detail::ChargeToken> token)
    : token_(std::move(token)) {}
PersistencePin::PersistencePin(std::shared_ptr<detail::PinToken> token)
    : token_(std::move(token)) {}

ResourceReservation ResourceLedger::Reserve(ResourceCategory category,
                                            std::size_t bytes) {
  if (Index(category) >= kResourceCategoryCount || category == Spill ||
      category == Chunk || bytes == 0)
    throw std::invalid_argument(
        "ledger reserve requires a nonzero allocation category");
  // Host bookkeeping allocation is outside the lock and has strong exception
  // safety. Charge payload sizes must include any owner/index metadata
  // separately.
  auto allocation =
      std::make_shared<detail::Allocation>(state_, category, bytes);
  auto token = std::make_shared<detail::ChargeToken>(allocation, false);
  const detail::Lock lock(*state_, LedgerStep::kReserve);
  auto& s = state_->snapshot;
  const auto& limits = state_->limits;
  state_->Fail(LedgerStep::kReserve);
  // Subtraction comparisons avoid overflow even for SIZE_MAX budgets.
  if (bytes > limits.total_bytes - s.total_bytes ||
      (category == Staging
           ? bytes > limits.staging_bytes - s.bytes[Index(Staging)]
           : bytes > limits.ram_bytes - s.ram_bytes))
    throw std::bad_alloc();
  s.bytes[Index(category)] += bytes;
  s.reserved_bytes[Index(category)] += bytes;
  s.total_bytes += bytes;
  if (category != Staging)
    s.ram_bytes += bytes;
  allocation->admitted = true;
  token->active = true;
  state_->Peaks();
  return ResourceReservation(std::move(token));
}
ResourceCharge ResourceReservation::Convert() {
  if (!token_)
    throw std::logic_error("empty ledger reservation");
  auto& a = *token_->allocation;
  const detail::Lock lock(*a.ledger, LedgerStep::kConvert);
  assert(a.reserved);
  a.ledger->Fail(LedgerStep::kConvert);
  auto& s = a.ledger->snapshot;
  s.reserved_bytes[Index(a.category)] -= a.bytes;
  a.reserved = false;
  if (a.category == Spill) {
    s.bytes[Index(Spill)] -= a.bytes;
    s.bytes[Index(Chunk)] += a.bytes;
    a.category = Chunk;
  }
  a.ledger->Peaks();
  return ResourceCharge(std::move(token_));
}
ResourceReservation ResourceCharge::ReserveBacking(
    ResourceCategory category) const {
  if (category != Spill && category != ResourceCategory::kPrivateState &&
      category != ResourceCategory::kPrivateTail)
    throw std::invalid_argument(
        "backing assignment requires spill, private state or tail");
  if (!token_ || token_->pooled || !token_->allocation->backing)
    throw std::logic_error("assignment requires a pool backing handle");
  auto& a = *token_->allocation;
  auto assignment =
      std::make_shared<detail::ChargeToken>(token_->allocation, true);
  const detail::Lock lock(*a.ledger, LedgerStep::kReserve);
  if (a.category != Free) {
    throw std::bad_alloc();
  }
  assert(!a.reserved);
  a.ledger->Fail(LedgerStep::kReserve);
  auto& s = a.ledger->snapshot;
  s.bytes[Index(Free)] -= a.bytes;
  s.bytes[Index(category)] += a.bytes;
  s.reserved_bytes[Index(category)] += a.bytes;
  a.category = category;
  a.reserved = true;
  assignment->active = true;
  a.ledger->Peaks();
  return ResourceReservation(std::move(assignment));
}
namespace {
std::shared_ptr<detail::PinToken> Pin(
    const std::shared_ptr<detail::ChargeToken>& token) {
  if (!token)
    throw std::logic_error("empty ledger charge");
  auto& a = *token->allocation;
  auto pin = std::make_shared<detail::PinToken>(token);
  const detail::Lock lock(*a.ledger, LedgerStep::kPin);
  // The pool handle cannot impersonate the active borrower/chunk owner.
  const bool payload = a.category == Spill || a.category == Chunk ||
                       a.category == ResourceCategory::kPrivateState ||
                       a.category == ResourceCategory::kPrivateTail;
  const bool borrowed =
      token->pooled &&
      (a.category == Spill || a.category == ResourceCategory::kPrivateTail);
  if (!payload || (a.reserved && !borrowed) || (a.backing && !token->pooled))
    throw std::logic_error("persistence pin requires owned payload");
  a.ledger->Fail(LedgerStep::kPin);
  auto& s = a.ledger->snapshot;
  if (a.pins == 0) {
    if (a.bytes >
        a.ledger->limits.persistence_pinned_bytes - s.persistence_pinned_bytes)
      throw std::bad_alloc();
    s.persistence_pinned_bytes += a.bytes;
  }
  ++a.pins;
  pin->active = true;
  a.ledger->Peaks();
  return pin;
}
}  // namespace
PersistencePin ResourceCharge::PinPersistence() const {
  return PersistencePin(Pin(token_));
}
PersistencePin ResourceReservation::PinPersistence() const {
  return PersistencePin(Pin(token_));
}
ResourceSnapshot ResourceLedger::Snapshot() const {
  const std::lock_guard lock(state_->mutex);
  return state_->snapshot;
}
ResourceSnapshot ResourceLedger::SnapshotAndResetPeaks() {
  const std::lock_guard lock(state_->mutex);
  const auto previous = state_->snapshot;
  auto& s = state_->snapshot;
  s.peak_bytes = s.bytes;
  s.peak_total_bytes = s.total_bytes;
  s.peak_ram_bytes = s.ram_bytes;
  s.peak_persistence_pinned_bytes = s.persistence_pinned_bytes;
  return previous;
}
std::array<LedgerLockCost, kLedgerStepCount> ResourceLedger::LockCosts() const {
  const std::lock_guard lock(state_->mutex);
  return state_->costs;
}
void ResourceLedger::FailAfter(LedgerStep step, std::size_t count) {
  if (Index(step) >= Index(LedgerStep::kRelease))
    throw std::invalid_argument("ledger release cannot fail");
  const std::lock_guard lock(state_->mutex);
  state_->faults[Index(step)] = count;
}
void ResourceLedger::ClearFaults() {
  const std::lock_guard lock(state_->mutex);
  state_->faults = {};
}
}  // namespace gufo::cache
