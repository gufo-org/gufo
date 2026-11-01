#ifndef GUFO_MODELS_QWEN_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_ADAPTER_HPP_

#include <atomic>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/experimental/transfer_pool.hpp>

#include "src/models/qwen/continuation_layout.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace gufo::models::qwen {

struct ContinuationFrontier {
  std::optional<tokenization::TokenId> token;
  std::vector<float> logits;
  bool published{};
  std::vector<tokenization::TokenId> pending;
};

// Slots own the same QwenGpuExecutor used by serving. The caller supplies full
// model/tokenizer/template identity; the adapter adds state ABI, geometry and
// policy.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(
      std::shared_ptr<const hip::QwenGpuModel>, std::uint32_t context,
      cache::Identity,
      hip::QwenExecutionPolicy = hip::QwenExecutionPolicy::Production(),
      std::shared_ptr<const hip::QwenDFlashGpuModel> draft_model = nullptr,
      hip::QwenDFlashGpuDraftConfig draft_config = {},
      cache::ResourceLedger* ledger = nullptr);
  ~ContinuationAdapter() override;
  [[nodiscard]] hip::QwenGpuExecutor& GetExecutor(cache::Slot&);
  [[nodiscard]] hip::QwenDFlashGpuDraftBackend* GetDraftBackend(cache::Slot&);
  [[nodiscard]] ContinuationFrontier& GetFrontier(cache::Slot&);
  // A next-token frontier with logits from the last executed position. A
  // partial-prefill restore carries a placeholder token without logits.
  [[nodiscard]] bool HasDecodeFrontier(const cache::Slot&) const;
  void BeginPrefill(cache::Slot&);
  void FinishPrefill(cache::Slot&) noexcept;
  void RollbackPrefill(cache::Slot&);
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const override {
    return layout_.BackingPlan(ram_bytes);
  }
  // Prompt-capture bytes the live slots pinned and charged to the ledger.
  // Read at startup; a later StopPinning leaves it overstated.
  [[nodiscard]] std::size_t PinnedCaptureBytes() const noexcept {
    return pinned_capture_bytes_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end) const override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);
  [[nodiscard]] cache::Completion TransferRows(State&,
                                               const ContinuationComponent&,
                                               bool load, cache::Rows first,
                                               cache::Rows end, std::byte*,
                                               cache::Stream&);
  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> draft_model_;
  hip::QwenDFlashGpuDraftConfig draft_config_;
  // Charged for each slot's pinned prompt capture; outlives the slots.
  cache::ResourceLedger* ledger_;
  std::atomic<std::size_t> pinned_capture_bytes_{};
  hip::QwenExecutionPolicy policy_;
  std::uint32_t context_;
  ContinuationLayout layout_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};
}  // namespace gufo::models::qwen
#endif
