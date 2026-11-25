// Gufo component-cache adapter for Inkling 2 Mini: checkpoint
// capture and restore of a session's state (see state_layout.hpp for the
// component layout).
#pragma once

#include <atomic>
#include <gufo/ext/adapter.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/models/inkling2/engine.hpp"
#include "src/models/inkling2/state_layout.hpp"

namespace gufo::models::inkling2 {

class Inkling2CacheAdapter;

/// One execution slot: owns a Session and the state of an in-progress restore.
/// It installs itself as the session's hooks, so every write of rows is
/// announced to the cache first.
class Inkling2Slot final : public cache::Slot, public SessionHooks {
public:
  Inkling2Slot(Inkling2CacheAdapter& adapter, cache::MutationGuard& guard,
               std::unique_ptr<Session> session);
  ~Inkling2Slot() override;
  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed;
  }
  [[nodiscard]] Session& session() { return *session_; }
  [[nodiscard]] const Session& session() const { return *session_; }
  void BeforeWrite(std::size_t first, std::size_t end) override;

  /// Where the generated text stands inside the model's block structure
  /// (runner.cpp Piece()). Part of the checkpoint metadata.
  Block block = Block::kIdle;

private:
  friend class Inkling2CacheAdapter;
  struct BoundaryView {
    const Engine::PrefillCheckpoint* cp = nullptr;
    Block block = Block::kIdle;
  };
  void GuardAll();  // BeforeOverwrite for every row component, whole context
  void ReleaseAll() noexcept;  // BeforeRelease likewise

  Inkling2CacheAdapter& adapter_;
  cache::MutationGuard& guard_;
  std::unique_ptr<Session> session_;
  const BoundaryView* boundary = nullptr;
  bool restoring = false;
  std::atomic<bool> failed{false};
  std::mutex restore_mutex;
  std::vector<cache::ComponentPosition> expected;
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> coverage;
  StateMeta meta{};
  std::vector<float> logits;
};

class Inkling2CacheAdapter final : public cache::Adapter {
public:
  /// `identity`: canonical bytes covering weights, tokenizer and template,
  /// state ABI, KV dtype and loop hyper-parameters (runner.cpp builds it).
  /// `durable`: the identity is stable across processes (artifact fingerprint
  /// known), so checkpoints may be persisted to disk.
  Inkling2CacheAdapter(std::shared_ptr<Engine> engine, cache::Identity identity,
                       bool durable, std::size_t plan_step);

  [[nodiscard]] Session& GetSession(cache::Slot&) const;
  [[nodiscard]] Inkling2Slot& AsSlot(cache::Slot&) const;
  [[nodiscard]] const StateLayout& layout() const { return layout_; }
  /// For the tests/cache tool.
  [[nodiscard]] Engine* engine_for_test() const { return engine_.get(); }

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram) const override {
    return layout_.BackingPlan(ram);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    return {.continuation = true, .persistent_encoding = durable_};
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    return descriptors_;
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    return identity_;
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  [[nodiscard]] std::vector<cache::ComponentPosition>
  PrefillCheckpointPositions(const cache::Slot&,
                             cache::Rows boundary) const override;
  /// Runs `Engine::Forward` with in-pass checkpoints, exposing each boundary
  /// synchronously through the slot while `consume(index)` reads it.
  void PrefillThrough(cache::Slot&, std::span<const std::int32_t> tokens,
                      std::span<const std::size_t> boundaries,
                      std::span<const Block> blocks,
                      const std::function<void(std::size_t)>& consume,
                      const PrefillYield* yield = nullptr,
                      std::span<const InputRows> rows = {}) const;
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
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&,
                                              cache::ComponentId, cache::Rows,
                                              cache::Rows, std::span<std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows, cache::Rows,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  friend class Inkling2Slot;
  void CapturePrivateInto(const Inkling2Slot&, const StateComponent&,
                          std::size_t offset, std::span<std::byte>) const;
  std::shared_ptr<Engine> engine_;
  StateLayout layout_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  bool durable_;
  std::size_t plan_step_;
};

}  // namespace gufo::models::inkling2
