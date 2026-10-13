#include "src/cli/serve/component_text_cache.hpp"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <thread>

#include "src/cli/serve/text_model_runner.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo;
using namespace gufo::server;
using namespace gufo::cache::testing;
namespace {
struct State final : TextRunnerState {
  FakeAdapter& adapter;
  cache::Slot& slot;
  State(FakeAdapter& adapter, cache::Slot& slot)
      : adapter(adapter), slot(slot) {}
  void Invalidate() noexcept override { (void)adapter.Invalidate(slot); }
};
class Runner : public TextModelRunner {
public:
  mutable FakeAdapter* adapter{};
  mutable std::size_t captures{};
  bool defer_idle{};
  bool in_pass{};
  bool partial{};
  std::size_t private_blocks{128};
  mutable std::size_t through_calls{};
  bool UsesComponentCache() const override { return true; }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "component-fake",
            .state_abi = "component-fake-v1",
            .max_context = 4096,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = false,
                             .prefix_reuse = true,
                             .in_pass_checkpoint = in_pass,
                             .partial_prefill_retention = partial}};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.state_capacity_bytes = 1ULL << 30,
            .per_request_state_bytes = 256,
            .retained_snapshot_capacity_bytes = 1ULL << 22};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{TextExecutionPlanKind::kSerial, 1},
            {TextExecutionPlanKind::kBatched, 2}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {1};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::vector<TextRunnerToken>{1};
  }
  std::string Decode(std::span<const TextRunnerToken>) const override {
    return "";
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error("legacy state must never be constructed");
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    struct Block {
      cache::ResourceCharge pool;
      std::span<std::byte> data;
      std::shared_ptr<const void> owner;
    };
    auto blocks = std::make_shared<std::vector<Block>>();
    // Physical backing commits once, before any request; assigned owner tokens
    // account for the entire block including unused bytes.
    auto data = std::make_shared<std::vector<std::byte>>(2048 * 16 +
                                                         private_blocks * 64);
    for (std::size_t i = 0; i < 2048 + private_blocks; ++i) {
      const auto size = i < 2048 ? 16ULL : 64ULL;
      const auto offset = i < 2048 ? i * 16 : 2048 * 16 + (i - 2048) * 64;
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, size);
      auto bytes = std::span(*data).subspan(offset, size);
      blocks->push_back({reservation.Convert(), bytes,
                         std::shared_ptr<const void>(data, bytes.data())});
    }
    auto resources = std::make_unique<ComponentCacheResources>();
    resources->adapter = std::make_unique<FakeAdapter>();
    adapter = dynamic_cast<FakeAdapter*>(resources->adapter.get());
    resources->owner = blocks;
    resources->stream = [this, foreground = std::this_thread::get_id()]()
        -> std::unique_ptr<cache::Stream> {
      if (defer_idle && std::this_thread::get_id() != foreground)
        return {};
      return std::make_unique<FakeStream>(true);
    };
    resources->transfer_batch_size = 4;
    resources->allocate = [this, blocks,
                           data](const cache::PayloadRequest& request) {
      for (const auto& block : *blocks) {
        if ((request.category == cache::ResourceCategory::kPrivateState) !=
            (block.data.size() == 64))
          continue;
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          ++captures;
          return ComponentCaptureBuffer{
              block.data,
              [reservation, owner = block.owner] {
                return cache::Payload::Committed(reservation->Convert(), owner);
              },
              data.get(),
              [reservation, owner = block.owner, bytes = block.data, request](
                  cache::SlotLease& lease, auto source) {
                return cache::Payload::Borrowed(
                    lease.Borrow(request.component, request.first, request.end,
                                 std::move(*reservation), owner,
                                 bytes.first(request.bytes), source));
              }};
        } catch (const std::bad_alloc&) {
        }
      }
      throw cache::ResourceExhausted();
    };
    resources->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    resources->copy_observed_rows = [](auto source, auto destination,
                                       cache::Stream& stream) {
      return dynamic_cast<FakeStream&>(stream).Submit([source, destination] {
        std::memcpy(destination.data(), source.data(), destination.size());
        return cache::TransferResult::kSucceeded;
      });
    };
    return resources;
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<State>(dynamic_cast<FakeAdapter&>(adapter), slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    const auto& s = dynamic_cast<State&>(state);
    assert(s.adapter.Positions(s.slot).front().valid_rows == position);
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    s.adapter.Append(s.slot, prompt.subspan(offset, count),
                     prompt.subspan(offset, count));
    return {count, offset + count == prompt.size()};
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto end = offset + std::min(budget, prompt.size() - offset);
    return boundary > offset && boundary < end && end - boundary <= 8
               ? std::optional<std::size_t>(1)
               : std::nullopt;
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    ++through_calls;
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    const auto head = prompt.subspan(offset, boundary - offset);
    s.adapter.Append(s.slot, head, head);
    try {
      *checkpoint = capture();
    } catch (...) {
      checkpoint->reset();
    }
    const auto tail = prompt.subspan(boundary, offset + count - boundary);
    s.adapter.Append(s.slot, tail, tail);
    return {count, offset + count == prompt.size()};
  }
  TextDecodeSelection SelectNext(TextRunnerState& state,
                                 sampling::SamplerState&) const override {
    auto& s = dynamic_cast<State&>(state);
    return {.token = static_cast<TextRunnerToken>(
                s.adapter.RecurrentHash(s.slot) % 100000),
            .piece = "x"};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = dynamic_cast<State&>(state);
    s.adapter.Append(s.slot, {&token, 1}, {&token, 1});
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    const auto& s = dynamic_cast<const State&>(state);
    return s.adapter.Positions(s.slot).front().valid_rows;
  }
};
void BudgetUsesAllocatedSlotsInsteadOfLazyStateClaim() {
  class BudgetRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      claim.state_capacity_bytes = 4ULL << 30;
      // Capacity includes image storage allocated only when requested. It is
      // much larger than remaining memory, but these text slots fit.
      claim.per_request_state_bytes = 1ULL << 30;
      claim.temporary_scratch_bytes = 0;
      claim.retained_snapshot_capacity_bytes = adapter ? 1 << 20 : 2 << 20;
      claim.retained_snapshot_ceiling_bytes = adapter ? 2 << 20 : 4 << 20;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this, &ledger](std::size_t budget) {
        assert(ledger.Limits().ram_bytes == budget);
        committed = budget;
      };
      return resources;
    }
  };
  for (const auto requested : {0ULL, 3ULL << 20}) {
    auto runner = std::make_shared<BudgetRunner>();
    TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = requested});
    assert(runner->committed == (requested ? 2ULL << 20 : 1ULL << 20));
    std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
    auto request = pool.Acquire(prompt);
    while (!request.prefill_complete())
      (void)request.Prefill(8);
    assert(request.SelectNext().token != 0);
  }
}
void Prefill(TextRunnerPool::Request& request) {
  while (!request.prefill_complete())
    (void)request.Prefill(8);
}
TextRunnerToken Generate(TextRunnerPool::Request& request) {
  const auto token = request.SelectNext().token;
  request.Advance();
  return token;
}
void InPassCapturePreservesBoundaryAndLiveTail() {
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto cold_runner = std::make_shared<Runner>();
  TextRunnerPool cold_pool(cold_runner, 1);
  auto cold =
      cold_pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
  Prefill(cold);
  const auto expected = Generate(cold);
  cold.Invalidate();
  for (const bool fail : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->in_pass = true;
    TextRunnerPool pool(runner, 2);
    auto seed =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
    if (fail)
      runner->adapter->FailNextTransfer();
    const auto step = seed.Prefill(64);
    assert(step.consumed_tokens == 40 && step.decode_ready);
    assert(runner->through_calls == 1);
    assert(Generate(seed) == expected);
    if (fail) {
      seed.Invalidate();
      continue;
    }
    (void)seed.Commit();
    auto branch = prompt;
    branch.back() = 999;
    auto restored =
        pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, true, 35);
    assert(restored.cached_prompt_tokens() == 35);
    Prefill(restored);
    const auto resumed = Generate(restored);
    restored.Invalidate();
    auto independent =
        cold_pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(independent);
    assert(Generate(independent) == resumed);
    independent.Invalidate();
  }
}
void RestoreFromBusyBorrowedSource() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = source->Capture();
  assert(source->PublishSnapshot(prompt, snapshot, false));
  auto gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
  auto peer = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(peer->cached_tokens() == prompt.size());
  auto& restored = dynamic_cast<State&>(peer->state());
  assert(runner->adapter->RecurrentHash(restored.slot) == hash);
  // Reading the prefix neither takes the source execution lease nor forces
  // it to materialize. Subsequent overwrite is still guarded.
  gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
  const std::array<cache::Token, 1> next{999};
  runner->adapter->Append(state.slot, next, next);
  gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 0);
  assert(runner->adapter->RecurrentHash(restored.slot) == hash);
}
void ColdAssignmentPrefersPreservedSlot() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = source->Capture();
  assert(source->PublishSnapshot(prompt, snapshot, false));
  (void)source->Commit(prompt, {}, prompt);
  // Idle streams are unavailable. Replacing the first slot would require
  // foreground preservation; the untouched second slot needs no copy.
  auto unrelated = prompt;
  unrelated.front() = 999;
  auto replacement = cache.Acquire(unrelated, {}, {}, true, 0, false);
  assert(!replacement->cache_hit());
  const auto metrics = replacement->Metrics();
  assert(metrics.spill.foreground_bytes == 0);
  assert(metrics.ledger.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
}
void CancelledPrefillKeepsCompletedSteps() {
  // A prefill cancelled between steps keeps the frontier it reached as a live
  // prefix (#486, #491), including on the component path Flash-Next uses.
  for (const bool partial : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->partial = partial;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    prompt.push_back(99);
    auto resend = pool.Acquire(prompt, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() == (partial ? 16 : 0));
    resend.Invalidate();
  }
}
void ColdAssignmentKeepsLiveFrontiers() {
  // No checkpoints are borrowed, so every slot has completed preservation.
  // Requests that cannot continue a frontier take the empty slot, then the
  // least recently used one, so the generated reply stays live (#562).
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  const auto turn = [&](const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto live = lease->cache_hit() && !lease->restored_snapshot_bytes();
    const auto cached = lease->cached_tokens();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, tokens);
    return live ? cached : 0;
  };
  std::vector<TextRunnerToken> conversation(12, 7);
  (void)turn(conversation);
  for (const TextRunnerToken other : {100, 101}) {
    (void)turn(std::vector<TextRunnerToken>(12, other));
    const auto frontier = conversation.size();
    conversation.push_back(8);
    assert(turn(conversation) == frontier);
  }
}
void ReuseAndBranches() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 2);
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto initial = pool.Acquire(prompt);
  Prefill(initial);
  const auto token = Generate(initial);
  assert(initial.Commit().snapshot_bytes > 0);
  auto retry = pool.Acquire(prompt);
  assert(retry.cached_prompt_tokens() == prompt.size());
  assert(retry.cache_restore_bytes() > 0);
  assert(Generate(retry) == token);
  auto peer = pool.Acquire(prompt);
  assert(peer.cached_prompt_tokens() == prompt.size());
  assert(Generate(peer) == token);
  (void)peer.Commit();
  (void)retry.Commit();
  runner->adapter->FailNextTransfer();
  auto failed_restore = pool.Acquire(prompt);
  assert(!failed_restore.cache_hit());
  Prefill(failed_restore);
  assert(Generate(failed_restore) == token);
  failed_restore.Invalidate();
  auto history = prompt;
  history.push_back(token);
  auto continued = pool.Acquire(history);
  assert(continued.cached_prompt_tokens() == history.size());
  assert(continued.cache_restore_bytes() == 0);
  (void)continued.Cancel();
  history.back() += 1;
  auto edit = pool.Acquire(history);
  assert(edit.cached_prompt_tokens() == prompt.size());
  Prefill(edit);
  const auto edited = Generate(edit);
  (void)edit.Commit();
  auto cold = pool.Acquire(history, sampling::SamplingConfig{}, {}, {}, false);
  assert(!cold.cache_hit());
  Prefill(cold);
  assert(Generate(cold) == edited);
  cold.Invalidate();
}
void EditBoundariesSurviveActualBackingPressure() {
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 4;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const auto capture = [](auto& lease, const auto& tokens,
                          SnapshotPurpose purpose) {
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), tokens.size(),
                                     false, purpose, {}));
    assert(lease->PublishSnapshot(tokens, lease->Capture(), false));
  };
  for (TextRunnerToken family = 100; family < 104; ++family) {
    std::vector<TextRunnerToken> tokens(12, family);
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, tokens, tokens);
    capture(lease, tokens, SnapshotPurpose::kContinuation);
    (void)lease->Commit({}, {}, tokens);
  }
  std::vector<TextRunnerToken> prompt(12);
  std::iota(prompt.begin(), prompt.end(), 1);
  {
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    for (std::size_t end : {4U, 8U, 12U}) {
      const auto piece = std::span(prompt).subspan(end - 4, 4);
      runner->adapter->Append(state.slot, piece, piece);
      std::vector<TextRunnerToken> prefix(prompt.begin(), prompt.begin() + end);
      capture(lease, prefix,
              end == prompt.size() ? SnapshotPurpose::kContinuation
                                   : SnapshotPurpose::kHistory);
    }
    (void)lease->Commit({}, {}, prompt);
  }
  for (std::size_t edit : {5U, 9U}) {
    auto branch = prompt;
    branch[edit] += 1000;
    auto lease = cache.Acquire(branch, {}, {}, true, 0, true);
    const auto cached = edit < 8 ? 4U : 8U;
    assert(lease->cached_tokens() == cached);
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(branch).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    const auto restored_hash = runner->adapter->RecurrentHash(state.slot);
    auto cold_runner = std::make_shared<Runner>();
    ComponentTextCache cold(cold_runner, 1, 1 << 22, {});
    auto control = cold.Acquire(branch, {}, {}, false, 0, true);
    auto& control_state = dynamic_cast<State&>(control->state());
    cold_runner->adapter->Append(control_state.slot, branch, branch);
    assert(restored_hash ==
           cold_runner->adapter->RecurrentHash(control_state.slot));
  }
}
void ReusedConversationOutlivesOneUseRequests() {
  // Backing holds three private states. A continued conversation keeps its
  // latest boundary while newer one-use requests replace each other.
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 3;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const auto turn = [&](const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto cached = lease->cached_tokens();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), tokens.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     {}));
    assert(lease->PublishSnapshot(tokens, lease->Capture(), false));
    (void)lease->Commit({}, {}, tokens);
    return cached;
  };
  std::vector<TextRunnerToken> conversation(12, 7);
  assert(turn(conversation) == 0);
  conversation.resize(16, 8);
  assert(turn(conversation) == 12);
  for (TextRunnerToken other = 100; other < 103; ++other)
    assert(turn(std::vector<TextRunnerToken>(12, other)) == 0);
  conversation.push_back(9);
  assert(turn(conversation) == 16);
}
void RestartAndCorruption() {
  char pattern[] = "/tmp/gufo-component-serving-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(prompt);
    Prefill(first);
    token = Generate(first);
    (void)first.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto retry = pool.Acquire(prompt);
    assert(retry.cache_disk_hit());
    assert(retry.cached_prompt_tokens() == prompt.size());
    assert(Generate(retry) == token);
    (void)retry.Commit();
  }
  for (const auto& kind : {"chunks", "private"}) {
    for (const auto& file :
         std::filesystem::directory_iterator(directory / "v2" / kind)) {
      std::fstream bytes(file.path(),
                         std::ios::in | std::ios::out | std::ios::binary);
      char byte;
      bytes.read(&byte, 1);
      byte ^= 0x5a;
      bytes.seekp(0);
      bytes.write(&byte, 1);
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto corrupted = pool.Acquire(prompt);
    assert(!corrupted.cache_hit());
    Prefill(corrupted);
    assert(Generate(corrupted) == token);
    corrupted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void FailedCaptureKeepsPublishedBoundary() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(42);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, std::span(prompt).first(40),
                          std::span(prompt).first(40));
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 40, false,
                                   SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = lease->Capture();
  assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 40}, snapshot,
                                false));
  snapshot.reset();
  runner->adapter->Append(state.slot, std::span(prompt).last(2),
                          std::span(prompt).last(2));
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 42, false,
                                   SnapshotPurpose::kContinuation, {}));
  runner->adapter->FailNextTransfer();
  bool failed = false;
  try {
    (void)lease->Capture();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  assert(failed);
  lease->Invalidate();
  auto restored =
      cache.Acquire(std::span(prompt).first(40), {}, {}, true, 0, true);
  assert(restored->cached_tokens() == 40);
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
}
void DiskBudgetEvictsAnUnpinnedConversation() {
  char pattern[] = "/tmp/gufo-component-budget-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1024,
                                        .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
std::uintmax_t ManagedBytes(const std::filesystem::path& directory) {
  std::uintmax_t bytes{};
  for (const auto& file :
       std::filesystem::recursive_directory_iterator(directory / "v2"))
    if (file.is_regular_file())
      bytes += file.file_size();
  return bytes;
}
void LoweredDiskBudgetEvictsOldestAtStartup() {
  char pattern[] = "/tmp/gufo-component-lowered-budget-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // Restarting below the retained bytes keeps the budget's meaning: the
  // oldest conversation leaves at startup instead of failing to start.
  disk.capacity_bytes = ManagedBytes(directory) - 1;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void OrphansLeaveBeforeCheckpoints() {
  char pattern[] = "/tmp/gufo-component-orphan-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // A crash left an unreferenced chunk that pushes the directory over a budget
  // the referenced checkpoints still fit; reclaiming it must suffice.
  disk.capacity_bytes = ManagedBytes(directory);
  const auto orphan = directory / "v2/chunks" / (std::string(32, 'f') + ".bin");
  std::ofstream(orphan, std::ios::binary) << std::string(4096, 'x');
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    assert(!std::filesystem::exists(orphan));
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    for (const auto& prompt : {first, second}) {
      auto restored = pool.Acquire(prompt);
      assert(restored.cache_disk_hit());
      restored.Invalidate();
    }
  }
  std::filesystem::remove_all(directory);
}
void StagingIsInternal() {
  char pattern[] = "/tmp/gufo-component-staging-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  // The legacy option is ignored: even a value below one four-byte row
  // still writes and restores through the internal staging buffer.
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 3};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto restored = pool.Acquire(prompt);
    assert(restored.cache_disk_hit());
    restored.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void StagingCountsAgainstRamCeiling() {
  class CeilingRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      claim.retained_snapshot_capacity_bytes = 2 << 20;
      claim.retained_snapshot_ceiling_bytes = 4 << 20;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t budget) {
        committed = budget;
      };
      return resources;
    }
  };
  char pattern[] = "/tmp/gufo-component-ceiling-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  // Pinned staging and RAM share the ceiling: an explicit 4 MiB beside the
  // internal 1 MiB of staging commits 3 MiB, whatever the ignored option says.
  for (const std::size_t option : {std::size_t{0}, std::size_t{4} << 20}) {
    TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                    .capacity_bytes = 1 << 20,
                                    .staging_capacity_bytes = option};
    auto runner = std::make_shared<CeilingRunner>();
    TextRunnerPool pool(runner, 1, disk, {.capacity_bytes = 4 << 20});
    assert(runner->committed == 3 << 20);
  }
  std::filesystem::remove_all(directory);
}
void DiskDirectoryHasOneOwner() {
  class CommitRunner final : public Runner {
  public:
    mutable bool committed{};
    mutable std::size_t streams{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t) { committed = true; };
      // Every execution slot takes a preservation stream when it allocates.
      resources->stream = [this, stream = std::move(resources->stream)] {
        ++streams;
        return stream();
      };
      return resources;
    }
  };
  char pattern[] = "/tmp/gufo-component-owner-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  {
    TextRunnerPool owner(std::make_shared<CommitRunner>(), 1, disk);
    auto second_runner = std::make_shared<CommitRunner>();
    std::string error;
    try {
      TextRunnerPool second(second_runner, 1, disk);
    } catch (const std::exception& e) {
      error = e.what();
    }
    assert(error.find("ownership") != std::string::npos &&
           error.find(directory.string()) != std::string::npos);
    // The losing server fails before allocating slots or committing its RAM.
    assert(!second_runner->streams && !second_runner->committed);
  }
  // Ownership ends with the owning server.
  {
    auto runner = std::make_shared<CommitRunner>();
    TextRunnerPool reopened(runner, 1, disk);
    assert(runner->committed);
  }
  std::filesystem::remove_all(directory);
}
void GridDoesNotSuppressFirstDiskPrompt() {
  char pattern[] = "/tmp/gufo-component-spacing-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  std::vector<TextRunnerToken> prompt(42);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 22, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, std::span(prompt).first(40),
                            std::span(prompt).first(40));
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 40, true,
                                     SnapshotPurpose::kHistory, {}));
    assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 40},
                                  lease->Capture(), false));
    assert(lease->HasSnapshotFor(std::span(prompt).first(40)));
    assert(!lease->Metrics().disk_queued_bytes);
    runner->adapter->Append(state.slot, std::span(prompt).last(2),
                            std::span(prompt).last(2));
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 42, false,
                                     SnapshotPurpose::kContinuation, {}));
    assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
  }
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 22, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == 42);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
void NearTailDiskBranchSurvivesRestart() {
  char pattern[] = "/tmp/gufo-component-branch-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  std::vector<TextRunnerToken> seed(1100);
  std::iota(seed.begin(), seed.end(), 10);
  auto branch = seed;
  branch.resize(800);
  branch.back() = 9999;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(seed);
    Prefill(first);
    (void)Generate(first);
    (void)first.Commit();
    auto fork = pool.Acquire(branch);
    Prefill(fork);
    (void)Generate(fork);
    (void)fork.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    branch.back() = 8888;
    auto restored = pool.Acquire(branch);
    assert(restored.cache_disk_hit());
    assert(restored.cached_prompt_tokens() == 799);
    Prefill(restored);
    const auto token = Generate(restored);
    restored.Invalidate();
    auto cold = pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold);
    assert(Generate(cold) == token);
  }
  std::filesystem::remove_all(directory);
}
void UnbackedDeepRowsSurviveDiskRestart() {
  char pattern[] = "/tmp/gufo-component-deep-test-XXXXXX";
  const std::filesystem::path directory = mkdtemp(pattern);
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 8,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(5001);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    // Over 40,000 logical row bytes exceed the 32,768-byte row pool.
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    (void)lease->Commit(prompt, {}, prompt);
  }
  {
    const auto seed_size = prompt.size();
    std::vector<TextRunnerToken> appended(63);
    std::iota(appended.begin(), appended.end(), 8000);
    prompt.insert(prompt.end(), appended.begin(), appended.end());
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == seed_size);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
    runner->adapter->Append(state.slot, appended, appended);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    (void)lease->Commit(prompt, {}, prompt);
  }
  {
    cache::ResourceLedger ledger({1 << 24, 1 << 24, 0, 0});
    cache::DiskStore store(ledger, directory, disk.capacity_bytes);
    assert(store.Entries().size() == 2);
    const cache::DiskManifest* parent = nullptr;
    const cache::DiskManifest* child = nullptr;
    for (const auto& entry : store.Entries()) {
      if (entry.Manifest().tokens.size() == 5001)
        parent = &entry.Manifest();
      else if (entry.Manifest().tokens.size() == prompt.size())
        child = &entry.Manifest();
    }
    assert(parent && child);
    for (std::size_t i = 0; i < parent->components.size(); ++i) {
      const auto& before = parent->components[i];
      const auto& after = child->components[i];
      for (std::size_t j = 0; j < before.chunks.size(); ++j)
        assert(before.chunks[j].file == after.chunks[j].file);
      if (before.private_state)
        assert(before.private_state->file != after.private_state->file);
      if (before.tail) {
        assert(after.chunks.size() > before.chunks.size());
        assert(before.tail->file != after.chunks[before.chunks.size()].file);
      }
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() &&
           lease->cached_tokens() == prompt.size());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
}  // namespace
int main() {
  CancelledPrefillKeepsCompletedSteps();
  BudgetUsesAllocatedSlotsInsteadOfLazyStateClaim();
  InPassCapturePreservesBoundaryAndLiveTail();
  RestoreFromBusyBorrowedSource();
  ColdAssignmentPrefersPreservedSlot();
  ColdAssignmentKeepsLiveFrontiers();
  ReuseAndBranches();
  EditBoundariesSurviveActualBackingPressure();
  ReusedConversationOutlivesOneUseRequests();
  RestartAndCorruption();
  FailedCaptureKeepsPublishedBoundary();
  DiskBudgetEvictsAnUnpinnedConversation();
  LoweredDiskBudgetEvictsOldestAtStartup();
  OrphansLeaveBeforeCheckpoints();
  StagingIsInternal();
  StagingCountsAgainstRamCeiling();
  DiskDirectoryHasOneOwner();
  GridDoesNotSuppressFirstDiskPrompt();
  NearTailDiskBranchSurvivesRestart();
  UnbackedDeepRowsSurviveDiskRestart();
}
