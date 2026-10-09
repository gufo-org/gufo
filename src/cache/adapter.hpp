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
  // Called before prefill/decode, rollback/rewind, reset, restore and teardown
  // can overwrite or release append rows [first, end). May throw to refuse a
  // mutation; all affected ranges must be guarded before any component changes.
  // During invalidation/teardown it must retire/preserve borrowers without
  // throwing.
  virtual void BeforeOverwrite(ComponentId, Rows first, Rows end) = 0;
};

// Model-owned execution storage, held by the runner. The guard outlives the
// slot. No execution/capture is permitted between a load and successful
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
  [[nodiscard]] virtual Capabilities capabilities() const = 0;
  // Descriptors are stable for the adapter lifetime; each ID is unique.
  [[nodiscard]] virtual std::span<const ComponentDescriptor> Components()
      const = 0;
  [[nodiscard]] virtual Identity CompatibilityIdentity() const = 0;
  [[nodiscard]] virtual std::unique_ptr<Slot> CreateSlot(MutationGuard&) = 0;
  [[nodiscard]] virtual std::vector<ComponentPosition> Positions(
      const Slot&) const = 0;

  // Transfers touch exactly the supplied range/buffer. Private state is copied
  // whole; append rows use component-specific positions, never prompt length.
  // Bad arguments/allocation/submission can throw before work is queued. Once
  // queued, transfer errors surface through Completion::Wait. The caller must
  // settle all loads before Validate and invalidate on any restore failure.
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
  // Requires exactly one position per required component. Failure leaves the
  // destination invalid; it must never execute with partial recurrent/draft
  // state.
  [[nodiscard]] virtual bool Validate(Slot&,
                                      std::span<const ComponentPosition>) = 0;
  // Call only after all slot transfers settle. noexcept failure cleanup
  // retires/preserves borrowers before releasing rows.
  virtual void Invalidate(Slot&) noexcept = 0;
};

}  // namespace gufo::cache

#endif
