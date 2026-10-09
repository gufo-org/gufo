#ifndef GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_
#define GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"

namespace gufo::cache::testing {

inline constexpr ComponentId kTarget{1}, kDraft{2}, kRecurrent{3};

// Deterministic single-threaded FIFO stream. A delayed stream advances only
// through Advance, Wait, Synchronize or completion destruction; no wall clock.
class FakeStream final : public Stream {
public:
  explicit FakeStream(bool delayed = false);
  ~FakeStream() override;
  [[nodiscard]] TransferResult Synchronize() noexcept override;
  [[nodiscard]] TransferResult Advance() noexcept;
  [[nodiscard]] Completion Submit(std::function<TransferResult()>);

private:
  struct State;
  class Signal;
  std::shared_ptr<State> state_;
  bool delayed_;
};

class FakeAdapter final : public Adapter {
public:
  [[nodiscard]] Capabilities capabilities() const override;
  [[nodiscard]] std::span<const ComponentDescriptor> Components()
      const override;
  [[nodiscard]] Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<Slot> CreateSlot(MutationGuard&) override;
  [[nodiscard]] std::vector<ComponentPosition> Positions(
      const Slot&) const override;
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
  void Invalidate(Slot&) noexcept override;

  // Execution fixture: target and draft advance independently. Recurrent state
  // is a token hash chain, so mixing rows and private state cannot validate.
  void Append(Slot&, std::span<const Token> target,
              std::span<const Token> draft);
  [[nodiscard]] std::uint64_t RecurrentHash(const Slot&) const;
  void FailNextAllocation() noexcept;
  void FailNextTransfer() noexcept;

private:
  [[nodiscard]] bool TakeTransferFailure() noexcept;
  bool fail_allocation_{false};
  bool fail_transfer_{false};
};

}  // namespace gufo::cache::testing

#endif
