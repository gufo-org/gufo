#ifndef GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_
#define GUFO_CACHE_TESTING_FAKE_ADAPTER_HPP_

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"

namespace gufo::cache::testing {

struct FakeTransfer;
struct FakeStreamState;

inline constexpr ComponentId kTarget{1}, kDraft{2}, kRecurrent{3}, kWindow{4};

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
  // Wider chunks need less capture metadata for the same row bytes.
  explicit FakeAdapter(Rows rows_per_chunk = 4);
  // Opt in to a fourth, windowed component (kWindow) that mirrors the target
  // tokens in a ring of ring_rows slots, like sliding-window attention: a
  // continuation reads the last window_rows, and appending past the ring
  // evicts the oldest rows. Requires ring_rows % 4 == 0 and room for a whole
  // window plus a partial leading chunk.
  FakeAdapter(Rows window_rows, Rows ring_rows);
  [[nodiscard]] Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const ComponentDescriptor> Components()
      const override;
  [[nodiscard]] Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<Slot> CreateSlot(MutationGuard&) override;
  [[nodiscard]] std::vector<ComponentPosition> Positions(
      const Slot&) const override;
  [[nodiscard]] std::vector<Rows> PlanPrefill(const Slot&, Rows first,
                                              Rows end) const override;
  void SetPrefillPassRows(Rows rows);
  void SetExecutionComplete(Slot&, bool);
  [[nodiscard]] std::vector<ComponentPosition> PrefillCheckpointPositions(
      const Slot&, Rows boundary) const override;
  void BeginRestore(Slot&, std::span<const ComponentPosition>) override;
  // Opt-in in-place restore: append rows stay while the slot holds them and
  // window rows while their ring slots still hold them.
  void SetInPlaceRestore(bool enabled) noexcept { in_place_ = enabled; }
  [[nodiscard]] std::vector<ResidentRange> ResidentRows(
      const Slot&, std::span<const ComponentPosition>) const override;
  void BeginRestore(Slot&, std::span<const ComponentPosition>,
                    std::span<const ResidentRange> kept) override;
  [[nodiscard]] Completion CapturePrivate(const Slot&, ComponentId,
                                          std::span<std::byte>,
                                          Stream&) override;
  [[nodiscard]] Completion CopyRowsOut(const Slot&, ComponentId, Rows, Rows,
                                       std::span<std::byte>, Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(const Slot&,
                                                          ComponentId, Rows,
                                                          Rows) const override;
  [[nodiscard]] Completion CopyRowsIn(Slot&, ComponentId, Rows, Rows,
                                      std::span<const std::byte>,
                                      Stream&) override;
  [[nodiscard]] Completion LoadPrivate(Slot&, ComponentId,
                                       std::span<const std::byte>,
                                       Stream&) override;
  [[nodiscard]] Completion CapturePrivatePiece(const Slot&, ComponentId,
                                               std::size_t,
                                               std::span<std::byte>,
                                               Stream&) override;
  [[nodiscard]] Completion LoadPrivatePiece(Slot&, ComponentId, std::size_t,
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
  // Window rows a continuation reads, from the ring; throws if any of them
  // were evicted or never loaded.
  [[nodiscard]] std::vector<Token> WindowTokens(const Slot&) const;
  [[nodiscard]] std::size_t Invalidations() const noexcept {
    return invalidations_.load(std::memory_order_relaxed);
  }

private:
  [[nodiscard]] Completion SubmitRead(const Slot&, Stream&,
                                      std::function<TransferResult()>);
  [[nodiscard]] Completion SubmitLoad(Slot&, Stream&,
                                      std::function<TransferResult()>);
  std::vector<ComponentDescriptor> components_;
  Rows ring_rows_{0};
  bool in_place_{false};
  bool fail_allocation_{false};
  bool fail_transfer_{false};
  std::atomic<std::size_t> invalidations_{};
  Rows prefill_pass_rows_{2048};
};

}  // namespace gufo::cache::testing

#endif
