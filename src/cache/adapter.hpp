#ifndef GUFO_CACHE_ADAPTER_HPP_
#define GUFO_CACHE_ADAPTER_HPP_

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/identity.hpp"
#include "src/cache/stream.hpp"
#include "src/cache/types.hpp"

namespace gufo::cache {

class MutationGuard {
public:
  MutationGuard() = default;
  virtual ~MutationGuard() = default;
  MutationGuard(const MutationGuard&) = delete;
  MutationGuard& operator=(const MutationGuard&) = delete;
  MutationGuard(MutationGuard&&) = delete;
  MutationGuard& operator=(MutationGuard&&) = delete;
  // Called synchronously on the host, before invalidating execution or queuing
  // any overwrite of [first, end). Preservation copies and reader waits, across
  // every stream, must finish before return; no device-stream ordering is
  // assumed. May throw to refuse. Guard all ranges before changing components.
  virtual void BeforeOverwrite(ComponentId, Rows first, Rows end) = 0;
  // Invalidation/destruction cannot refuse release. Preserve or retire every
  // borrower and finish all preservation transfers/readers before returning.
  virtual void BeforeRelease(ComponentId, Rows first, Rows end) noexcept = 0;
};

// Model-owned execution storage, held by the runner. The guard outlives the
// slot. No execution/capture is permitted between BeginRestore and successful
// Validate.
class Slot {
public:
  Slot() = default;
  virtual ~Slot() = default;
  Slot(const Slot&) = delete;
  Slot& operator=(const Slot&) = delete;
  Slot(Slot&&) = delete;
  Slot& operator=(Slot&&) = delete;
  [[nodiscard]] virtual bool IsValid() const noexcept = 0;
};

class Adapter {
public:
  Adapter() = default;
  virtual ~Adapter() = default;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;
  Adapter(Adapter&&) = delete;
  Adapter& operator=(Adapter&&) = delete;
  [[nodiscard]] virtual Capabilities GetCapabilities() const = 0;
  // Descriptors are stable for the adapter lifetime; each ID is unique.
  [[nodiscard]] virtual std::span<const ComponentDescriptor> Components()
      const = 0;
  [[nodiscard]] virtual Identity CompatibilityIdentity() const = 0;
  [[nodiscard]] virtual std::unique_ptr<Slot> CreateSlot(MutationGuard&) = 0;
  [[nodiscard]] virtual std::vector<ComponentPosition> Positions(
      const Slot&) const = 0;

  // Prepare one complete restore, including an exact shorter frontier. On the
  // host, allocate/check capacity and guard every range that will be replaced
  // while the old slot is still readable. Only then disable execution and
  // allow loads. Refuse if any slot transfer is pending or failure is latched.
  // After success, CopyRowsIn/LoadPrivate need no further preservation waits.
  virtual void BeginRestore(Slot&, std::span<const ComponentPosition>) = 0;

  // Transfers touch exactly the supplied range/buffer. Private state is copied
  // whole; append rows use component-specific positions, never prompt length.
  // Bad arguments/allocation/submission can throw before work is queued. Once
  // queued, transfer errors surface through Completion::Wait. The caller must
  // settle all loads before Validate and invalidate on any restore failure.
  // BeginRestore must precede loads. Non-overlapping row pieces may arrive in
  // any order on independent streams. Overlapping writes require caller order.
  // The adapter MUST latch any failed load (including submission failure), even
  // when a completion's error is discarded by destruction or move assignment.
  [[nodiscard]] virtual Completion CapturePrivate(const Slot&, ComponentId,
                                                  std::span<std::byte>,
                                                  Stream&) = 0;
  [[nodiscard]] virtual Completion CopyRowsOut(const Slot&, ComponentId,
                                               Rows first, Rows end,
                                               std::span<std::byte>,
                                               Stream&) = 0;
  [[nodiscard]] virtual Completion CopyRowsIn(Slot&, ComponentId, Rows first,
                                              Rows end,
                                              std::span<const std::byte>,
                                              Stream&) = 0;
  [[nodiscard]] virtual Completion LoadPrivate(Slot&, ComponentId,
                                               std::span<const std::byte>,
                                               Stream&) = 0;
  // Requires an active BeginRestore. Outside a restore (including after a
  // successful Validate), throw std::logic_error without changing the slot.
  // Check complete loads and exactly one position per component, not
  // row-content integrity (the store/test oracle owns that). During a restore,
  // any failed load or false Validate since the last Invalidate MUST keep this
  // returning false. A false result latches failure, forbidding execution or
  // another restore.
  [[nodiscard]] virtual bool Validate(Slot&,
                                      std::span<const ComponentPosition>) = 0;
  // Return false without changing the slot while any read/load is pending.
  // Otherwise release/retire borrowers, clear failure and restore metadata,
  // and leave a valid empty slot ready for cold prefill. Destruction instead
  // drains all outstanding slot transfers before the nonthrowing release path.
  [[nodiscard]] virtual bool Invalidate(Slot&) noexcept = 0;
};

}  // namespace gufo::cache

#endif
