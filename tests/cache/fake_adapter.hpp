#ifndef GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_
#define GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"

namespace gufo::cache::testing {

struct FakeTransfer;
struct FakeStreamState;

inline constexpr ComponentId kTarget{1}, kDraft{2}, kRecurrent{3};

// Deterministic single-threaded FIFO stream. A delayed stream advances only
// through Advance, Wait, Synchronize or completion destruction; no wall clock.
// Submit runs on the host. Waiting/synchronizing from a running callback on
// the same stream is a programming error and terminates instead of deadlocking.
class FakeStream final : public Stream {
public:
  explicit FakeStream(bool delayed = false);
  ~FakeStream() override;
  [[nodiscard]] TransferResult Synchronize() noexcept override;
  [[nodiscard]] TransferResult Advance() noexcept;
  [[nodiscard]] Completion Submit(std::function<TransferResult()>);
  void FailNextSubmission() noexcept;

private:
  friend class FakeAdapter;
  class Signal;
  [[nodiscard]] Completion SubmitTracked(
      std::function<TransferResult()>,
      std::vector<std::shared_ptr<FakeTransfer>>*);
  std::shared_ptr<FakeStreamState> state_;
  bool delayed_;
  bool fail_submission_{false};
};

class FakeAdapter final : public Adapter {
public:
  [[nodiscard]] Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const ComponentDescriptor> Components()
      const override;
  [[nodiscard]] Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<Slot> CreateSlot(MutationGuard&) override;
  [[nodiscard]] std::vector<ComponentPosition> Positions(
      const Slot&) const override;
  void BeginRestore(Slot&, std::span<const ComponentPosition>) override;
  [[nodiscard]] Completion CapturePrivate(const Slot&, ComponentId,
                                          std::span<std::byte>,
                                          Stream&) override;
  [[nodiscard]] Completion CopyRowsOut(const Slot&, ComponentId, Rows, Rows,
                                       std::span<std::byte>, Stream&) override;
  [[nodiscard]] Completion CopyRowsIn(Slot&, ComponentId, Rows, Rows,
                                      std::span<const std::byte>,
                                      Stream&) override;
  [[nodiscard]] Completion LoadPrivate(Slot&, ComponentId,
                                       std::span<const std::byte>,
                                       Stream&) override;
  [[nodiscard]] bool Validate(Slot&,
                              std::span<const ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(Slot&) noexcept override;

  // Execution fixture: target and draft advance independently. Recurrent state
  // is a token hash chain. Continued execution detects mismatched checkpoint
  // bytes; Validate checks metadata and load completeness, not row contents.
  void Append(Slot&, std::span<const Token> target,
              std::span<const Token> draft);
  [[nodiscard]] std::uint64_t RecurrentHash(const Slot&) const;
  // Isolate host guard overhead and row-range tests without fixture mutation.
  void GuardRows(Slot&, ComponentId, Rows first, Rows end);
  void FailNextAllocation() noexcept;
  void FailNextTransfer() noexcept;

private:
  [[nodiscard]] Completion SubmitRead(const Slot&, Stream&,
                                      std::function<TransferResult()>);
  [[nodiscard]] Completion SubmitLoad(Slot&, Stream&,
                                      std::function<TransferResult()>);
  bool fail_allocation_{false};
  bool fail_transfer_{false};
};

}  // namespace gufo::cache::testing

#endif
