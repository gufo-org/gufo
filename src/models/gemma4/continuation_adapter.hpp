#ifndef GUFO_MODELS_GEMMA4_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_GEMMA4_CONTINUATION_ADAPTER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/models/gemma4/continuation_hooks.hpp"
#include "src/models/gemma4/engine.hpp"
#include "src/models/gemma4/kernels/rocm/kernels.hpp"

namespace gufo::models::gemma4 {

// One continuation's state, by component (ID = index + 1).
enum class ContinuationPart : std::uint8_t {
  kTokens,
  kKey,
  kValue,
  kLogits,
  kHidden,
  kMetadata,
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};
// Image records a checkpoint holds. A session with more images is not
// captured: a restore that dropped records would prefill again from the first
// image it lacks.
inline constexpr std::size_t kContinuationImages = 1024;
struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

// Serves Gemma 4 sessions to the component cache. Tokens and global-layer
// keys and values are append rows; sliding-layer keys and values are window
// rows held in the session's ring; the frontier logits, its hidden state (the
// MTP drafter's input) and metadata are private. The drafter reads the
// target's cache, so speculative decoding adds no components. The caller
// supplies the artifact and template identity; the adapter adds its layout.
//
// Captures copy their rows instead of lending the slot's: lent rows are
// copied anyway before the slot is reassigned or restored in place, one
// chunk per wait (a 22K-token prompt's global rows took 8-12 ms there),
// while a capture gathers them in one launch at about 100 GB/s. Copies wait
// only for their own session's queued writes (ContinuationFences), not for
// another session's prefill on the shared stream.
class ContinuationAdapter final : public cache::Adapter,
                                  public ContinuationFences {
public:
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context,
                      cache::Identity);
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  // Orders `stream` after the work queued so far on the model's executor, so
  // reads of rows and frontier state see completed writes.
  void AfterExecution(hipStream_t stream) const;
  void BeforeMutation(std::span<ContinuationHooks* const> writers) override;
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;
  [[nodiscard]] std::span<const ContinuationComponent> Layout() const {
    return components_;
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
  [[nodiscard]] std::vector<cache::ResidentRange> ResidentRows(
      const cache::Slot&,
      std::span<const cache::ComponentPosition>) const override;
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>,
                    std::span<const cache::ResidentRange> kept) override;
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
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece {
    std::byte* pointer;
    std::size_t bytes;
  };
  // AfterExecution for one slot: after its own writes only, once another
  // session's write has fenced them.
  void AfterExecution(State&, hipStream_t stream) const;
  // Whether the writes AfterExecution(State&) orders after have finished: a
  // gather launched now cannot hold the copy stream behind queued forwards.
  [[nodiscard]] bool Executed(const State&) const;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  // Device memory of rows [first, end) of a key or value component: one
  // piece, or two where a window range wraps the session's ring.
  [[nodiscard]] std::vector<Piece> RowPieces(State&,
                                             const ContinuationComponent&,
                                             cache::Rows first,
                                             cache::Rows end) const;
  // The private component's whole representation: device memory, or host
  // memory (`host`) the adapter copies on the calling thread.
  [[nodiscard]] Piece PrivateRegion(State&, const ContinuationComponent&,
                                    bool loading, bool* host) const;
  // `plan` runs host copies and lists device copies, which Transfer batches
  // into gathers on copy_stream_: one launches whenever the previous one has
  // finished, the rest at the first wait. The cache's streams carry no work: a
  // pooled stream can share a hardware queue with the model's stream, and
  // its completion then waited behind another session's prefill chunk.
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load,
      const std::function<void(std::vector<rocm::CopyRun>&)>& plan);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::uint32_t ring_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  // Guards executed_ and every slot's fence state.
  mutable std::mutex executed_mutex_;
  hipEvent_t executed_{};
  std::vector<State*> states_;
  // High priority, so on a hardware queue of its own.
  hipStream_t copy_stream_{};
};
}  // namespace gufo::models::gemma4
#endif
