#include "src/cache/ledger.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

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
  ResourceLimits limits;
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
  std::vector<std::shared_ptr<Allocation>> additional;
  template<class F>
  void Each(F&& f) const {
    f(*allocation);
    for (const auto& item : additional)
      f(*item);
  }
  bool pooled{false};
  bool active{false};
  ChargeToken(std::shared_ptr<Allocation> allocation, bool pooled)
      : allocation(std::move(allocation)), pooled(pooled) {}
  ~ChargeToken() {
    if (!pooled || !active)
      return;
    auto& a = *allocation;
    const Lock lock(*a.ledger, LedgerStep::kRelease);
    auto& s = a.ledger->snapshot;
    Each([&](Allocation& block) {
      assert(block.pins == 0 && block.backing && block.category != Free);
      s.bytes[Index(block.category)] -= block.bytes;
      if (block.reserved)
        s.reserved_bytes[Index(block.category)] -= block.bytes;
      s.bytes[Index(Free)] += block.bytes;
      block.category = Free;
      block.reserved = false;
    });
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
    charge->Each([&](Allocation& block) {
      assert(block.pins);
      if (--block.pins == 0)
        a.ledger->snapshot.persistence_pinned_bytes -= block.bytes;
    });
  }
};
}  // namespace detail

ResourceLedger::ResourceLedger(ResourceLimits limits, bool timing)
    : state_(std::make_shared<detail::LedgerState>(limits, timing)) {}
void ResourceLedger::ConstrainLimits(ResourceLimits limits) {
  const detail::Lock lock(*state_, LedgerStep::kReserve);
  const auto& old = state_->limits;
  const auto& s = state_->snapshot;
  if (limits.total_bytes > old.total_bytes ||
      limits.ram_bytes > old.ram_bytes ||
      limits.staging_bytes > old.staging_bytes ||
      limits.persistence_pinned_bytes > old.persistence_pinned_bytes)
    throw std::invalid_argument("ledger limits may only decrease");
  if (limits.total_bytes < s.total_bytes || limits.ram_bytes < s.ram_bytes ||
      limits.staging_bytes < s.bytes[Index(Staging)] ||
      limits.persistence_pinned_bytes < s.persistence_pinned_bytes)
    throw ResourceExhausted();
  state_->limits = limits;
}
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
    throw ResourceExhausted();
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
namespace {
ResourceAllocationInfo Info(const std::shared_ptr<detail::ChargeToken>& token) {
  if (!token)
    throw std::logic_error("empty ledger handle");
  const auto& a = *token->allocation;
  const std::lock_guard lock(a.ledger->mutex);
  std::size_t bytes = 0;
  bool pinned = false;
  token->Each([&](const detail::Allocation& block) {
    assert(block.category == a.category && block.reserved == a.reserved &&
           block.backing == a.backing);
    bytes += block.bytes;
    pinned |= block.pins != 0;
  });
  return {a.category, bytes, a.reserved, a.backing, token->pooled, pinned};
}
}  // namespace
ResourceAllocationInfo ResourceReservation::Info() const {
  return cache::Info(token_);
}
ResourceAllocationInfo ResourceCharge::Info() const {
  return cache::Info(token_);
}
std::size_t ResourceCharge::FindFreeBacking(
    std::span<const ResourceCharge> blocks, std::size_t count, bool from_end) {
  if (!count)
    throw std::invalid_argument("empty backing run");
  if (blocks.empty())
    return 0;
  const auto& first = (from_end ? blocks.back() : blocks.front()).token_;
  if (!first || first->pooled || !first->allocation->backing)
    throw std::logic_error("search requires original pool backing handles");
  const auto& ledger = first->allocation->ledger;
  const std::lock_guard lock(ledger->mutex);
  std::size_t run = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const auto index = from_end ? blocks.size() - 1 - i : i;
    const auto& block = blocks[index].token_;
    if (!block || block->pooled || !block->allocation->backing)
      throw std::logic_error("search requires original pool backing handles");
    if (block->allocation->ledger != ledger)
      throw std::invalid_argument("backing search spans ledgers");
    run = block->allocation->category == Free ? run + 1 : 0;
    if (run == count)
      return from_end ? index : index + 1 - count;
  }
  return blocks.size();
}
ResourceCharge ResourceReservation::Convert() {
  if (!token_)
    throw std::logic_error("empty ledger reservation");
  auto& a = *token_->allocation;
  const detail::Lock lock(*a.ledger, LedgerStep::kConvert);
  a.ledger->Fail(LedgerStep::kConvert);
  auto& s = a.ledger->snapshot;
  token_->Each([&](detail::Allocation& block) {
    assert(block.reserved);
    s.reserved_bytes[Index(block.category)] -= block.bytes;
    block.reserved = false;
    if (block.category == Spill) {
      s.bytes[Index(Spill)] -= block.bytes;
      s.bytes[Index(Chunk)] += block.bytes;
      block.category = Chunk;
    }
  });
  a.ledger->Peaks();
  return ResourceCharge(std::move(token_));
}
ResourceReservation ResourceCharge::ReserveBacking(
    ResourceCategory category) const {
  return ReserveBacking(std::span(this, 1), category);
}
ResourceReservation ResourceCharge::ReserveBacking(
    std::span<const ResourceCharge> blocks, ResourceCategory category) {
  if (category != Spill && category != ResourceCategory::kPrivateState &&
      category != ResourceCategory::kPrivateTail)
    throw std::invalid_argument(
        "backing assignment requires spill, private state or tail");
  if (blocks.empty())
    throw std::invalid_argument("empty backing assignment");
  for (const auto& block : blocks)
    if (!block.token_ || block.token_->pooled ||
        !block.token_->allocation->backing)
      throw std::logic_error(
          "assignment requires original pool backing handles");
  const auto& first = blocks.front().token_->allocation;
  auto assignment = std::make_shared<detail::ChargeToken>(first, true);
  assignment->additional.reserve(blocks.size() - 1);
  for (std::size_t i = 1; i < blocks.size(); ++i) {
    const auto& allocation = blocks[i].token_->allocation;
    if (allocation->ledger != first->ledger)
      throw std::invalid_argument("backing assignment spans ledgers");
    for (std::size_t j = 0; j < i; ++j)
      if (allocation == blocks[j].token_->allocation)
        throw std::invalid_argument("duplicate backing block");
    assignment->additional.push_back(allocation);
  }
  auto& a = *first;
  const detail::Lock lock(*a.ledger, LedgerStep::kReserve);
  assignment->Each([](const detail::Allocation& block) {
    if (block.category != Free)
      throw ResourceExhausted();
    assert(!block.reserved);
  });
  a.ledger->Fail(LedgerStep::kReserve);
  auto& s = a.ledger->snapshot;
  assignment->Each([&](detail::Allocation& block) {
    s.bytes[Index(Free)] -= block.bytes;
    s.bytes[Index(category)] += block.bytes;
    s.reserved_bytes[Index(category)] += block.bytes;
    block.category = category;
    block.reserved = true;
  });
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
  // Validate and admit the complete group before changing any pin counter.
  std::size_t newly_pinned = 0;
  token->Each([&](const detail::Allocation& block) {
    const bool payload = block.category == Spill || block.category == Chunk ||
                         block.category == ResourceCategory::kPrivateState ||
                         block.category == ResourceCategory::kPrivateTail;
    const bool borrowed =
        token->pooled && (block.category == Spill ||
                          block.category == ResourceCategory::kPrivateTail);
    if (!payload || (block.reserved && !borrowed) ||
        (block.backing && !token->pooled))
      throw std::logic_error("persistence pin requires owned payload");
    if (!block.pins)
      newly_pinned += block.bytes;
  });
  a.ledger->Fail(LedgerStep::kPin);
  auto& s = a.ledger->snapshot;
  if (newly_pinned >
      a.ledger->limits.persistence_pinned_bytes - s.persistence_pinned_bytes)
    throw std::bad_alloc();
  s.persistence_pinned_bytes += newly_pinned;
  token->Each([](detail::Allocation& block) { ++block.pins; });
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
ResourceLimits ResourceLedger::Limits() const {
  const std::lock_guard lock(state_->mutex);
  return state_->limits;
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
