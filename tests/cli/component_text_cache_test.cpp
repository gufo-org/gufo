#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cache/retention.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/cli/fake_cache_resources.hpp"
#include "tests/support/temporary_directory.hpp"

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
  std::size_t alignment{1};
  /// Every prefill step's end and in-pass checkpoint, in order.
  mutable std::vector<std::size_t> pass_ends;
  mutable std::vector<std::size_t> in_pass_points;
  std::size_t private_blocks{128};
  // Nonzero: the adapter adds a ring-backed window component, and the next
  // token also depends on the window rows, so a wrong restore changes it.
  cache::Rows window_rows{}, ring_rows{};
  bool private_state{true};
  cache::Rows rows_per_chunk{4};
  std::size_t row_backing_bytes{16};
  mutable std::size_t through_calls{};
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "component-fake",
            .state_abi = "component-fake-v1",
            .max_context = 4096,
            .capabilities = {.incremental_prefill = true,
                             .final_token_advance_required = false,
                             .prefix_reuse = true,
                             .in_pass_checkpoint = in_pass,
                             .partial_prefill_retention = partial,
                             .prefill_alignment = alignment,
                             .max_in_pass_checkpoints = 64U}};
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
  // Image prompts carry their cache identity in the context.
  void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext>) const override {}
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    auto owned = window_rows ? std::make_unique<FakeAdapter>(
                                   window_rows, ring_rows, private_state)
                             : std::make_unique<FakeAdapter>(rows_per_chunk);
    adapter = owned.get();
    return server::testing::MakeFakeCacheResources(
        ledger, std::move(owned),
        {.private_blocks = private_blocks,
         .row_backing_bytes = row_backing_bytes,
         .defer_idle = &defer_idle,
         .captures = &captures});
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
    pass_ends.push_back(offset + count);
    return {count, offset + count == prompt.size()};
  }
  /// Longest prefill tail an in-pass checkpoint may leave after it.
  std::size_t max_in_pass_tail{8};
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto end = offset + std::min(budget, prompt.size() - offset);
    return boundary > offset && boundary < end &&
                   end - boundary <= max_in_pass_tail
               ? std::optional<std::size_t>(1)
               : std::nullopt;
  }
  TextPrefillStep PrefillThrough(TextRunnerState& state,
                                 std::span<const TextRunnerToken> prompt,
                                 std::size_t offset, std::size_t budget,
                                 std::span<const std::size_t> boundaries,
                                 const InPassConsumer& consume) const override {
    ++through_calls;
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    auto at = offset;
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
      const auto head = prompt.subspan(at, boundaries[i] - at);
      s.adapter.Append(s.slot, head, head);
      at = boundaries[i];
      consume(i);
      in_pass_points.push_back(boundaries[i]);
    }
    const auto tail = prompt.subspan(at, offset + count - at);
    s.adapter.Append(s.slot, tail, tail);
    pass_ends.push_back(offset + count);
    return {count, offset + count == prompt.size()};
  }
  TextDecodeSelection SelectNext(TextRunnerState& state,
                                 sampling::SamplerState&) const override {
    auto& s = dynamic_cast<State&>(state);
    auto hash = s.adapter.RecurrentHash(s.slot);
    if (window_rows)
      for (const auto token : s.adapter.WindowTokens(s.slot))
        hash = (hash ^ token) * 0x100000001b3ULL;
    return {.token = static_cast<TextRunnerToken>(hash % 100000), .piece = "x"};
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
// A runner whose prefill continues a prompt exactly only from multiples of
// four (Flash-Next's sparse tiles) ends every step but the last there and
// rounds its stable and intermediate checkpoints down. A restore then starts
// on that boundary and re-prefills the rest. The complete prompt is kept
// exact for an unchanged retry, but a longer prompt never continues it.
void AlignedPrefillKeepsAlignedBoundaries() {
  std::vector<TextRunnerToken> prompt(41);
  std::iota(prompt.begin(), prompt.end(), 10);
  const auto aligned = [](const std::vector<std::size_t>& positions,
                          std::size_t last) {
    return std::ranges::all_of(positions, [&](std::size_t position) {
      return position % 4 == 0 || position == last;
    });
  };
  auto runner = std::make_shared<Runner>();
  runner->alignment = 4;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 2);
  // The stable boundary 35 starts the final user turn.
  auto seed =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
  while (!seed.prefill_complete())
    (void)seed.Prefill(10);
  assert((runner->pass_ends == std::vector<std::size_t>{8, 16, 24, 32, 41}));
  assert(runner->in_pass_points.empty());
  const auto expected = Generate(seed);
  (void)seed.Commit();
  for (const auto [edit, cached] :
       {std::pair{40U, 32U}, std::pair{36U, 32U}, std::pair{33U, 32U}}) {
    auto branch = prompt;
    branch[edit] = 999;
    runner->pass_ends.clear();
    runner->in_pass_points.clear();
    auto restored =
        pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, true, 35);
    assert(restored.cached_prompt_tokens() == cached);
    while (!restored.prefill_complete())
      (void)restored.Prefill(3);
    assert(aligned(runner->pass_ends, branch.size()));
    assert(aligned(runner->in_pass_points, branch.size()));
    (void)Generate(restored);
    restored.Invalidate();
  }
  // A longer prompt restores the aligned boundary, not the complete prompt.
  auto longer = prompt;
  longer.insert(longer.end(), {7, 8, 9});
  runner->pass_ends.clear();
  auto extended =
      pool.Acquire(longer, sampling::SamplingConfig{}, {}, {}, true, 35);
  assert(extended.cached_prompt_tokens() == 32);
  while (!extended.prefill_complete())
    (void)extended.Prefill(3);
  assert(aligned(runner->pass_ends, longer.size()));
  (void)Generate(extended);
  extended.Invalidate();
  // An unchanged retry restores the complete prompt without prefill.
  auto retry =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
  assert(retry.cached_prompt_tokens() == prompt.size());
  assert(retry.prefill_complete());
  assert(Generate(retry) == expected);
  retry.Invalidate();
}
// A request that stops at an unaligned prompt end before advancing a token
// (one-token completions) drops its live frontier there, but still publishes
// the complete prompt, which an unchanged retry restores without prefill.
void UnalignedPromptEndPublishesBeforeDroppingTheFrontier() {
  std::vector<TextRunnerToken> prompt(41);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto runner = std::make_shared<Runner>();
  runner->alignment = 4;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 2);
  auto seed =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
  Prefill(seed);
  (void)seed.Commit();
  auto retry =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
  assert(retry.cached_prompt_tokens() == prompt.size());
  assert(retry.prefill_complete());
  (void)Generate(retry);
  retry.Invalidate();
}
// Raw completions have no separate stable chat boundary. Keep an aligned
// near-end fallback as well as the exact prompt retry across slot reuse.
void UnalignedRawPromptKeepsAnAlignedContinuation() {
  std::vector<TextRunnerToken> prompt(41);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto runner = std::make_shared<Runner>();
  runner->alignment = 4;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 1);
  auto seed = pool.Acquire(prompt, sampling::SamplingConfig{});
  (void)seed.Prefill(64);
  assert((runner->pass_ends == std::vector<std::size_t>{41}));
  assert((runner->in_pass_points == std::vector<std::size_t>{40}));
  const auto expected = Generate(seed);
  (void)seed.Commit();
  auto unrelated =
      pool.Acquire({999, 998, 997, 996}, sampling::SamplingConfig{});
  Prefill(unrelated);
  (void)Generate(unrelated);
  (void)unrelated.Commit();
  auto retry = pool.Acquire(prompt, sampling::SamplingConfig{});
  assert(retry.cached_prompt_tokens() == prompt.size());
  assert(retry.prefill_complete());
  assert(Generate(retry) == expected);
  retry.Invalidate();
  auto longer = prompt;
  longer.insert(longer.end(), {7, 8, 9});
  auto extended = pool.Acquire(longer, sampling::SamplingConfig{});
  assert(extended.cached_prompt_tokens() == 40);
  Prefill(extended);
  (void)Generate(extended);
  extended.Invalidate();
}
void DisabledCacheDoesNotPrepareARawPromptCheckpoint() {
  class NoCheckpointRunner final : public Runner {
  public:
    std::optional<std::size_t> PrefillCheckpointBytes(
        const TextRunnerState&, std::span<const TextRunnerToken>, std::size_t,
        std::size_t, std::size_t) const override {
      throw std::logic_error("disabled cache must not prepare a checkpoint");
    }
  };
  auto runner = std::make_shared<NoCheckpointRunner>();
  runner->alignment = 4;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 1, std::nullopt, {.enabled = false});
  std::vector<TextRunnerToken> prompt(41);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto request = pool.Acquire(prompt, sampling::SamplingConfig{});
  (void)request.Prefill(64);
  assert(request.prefill_complete());
  assert((runner->pass_ends == std::vector<std::size_t>{41}));
  (void)Generate(request);
  (void)request.Commit();
}
// A runner that captures in place anywhere in a pass takes the grid
// checkpoint a long pass covers inside it, without splitting the pass, and
// a branch past it restores that checkpoint like a pass-end capture.
void GridCheckpointsCaptureInPass() {
  std::vector<TextRunnerToken> prompt(4000);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto branch = prompt;
  branch[3000] = 999;
  auto cold_runner = std::make_shared<Runner>();
  TextRunnerPool cold_pool(cold_runner, 1);
  auto cold =
      cold_pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
  Prefill(cold);
  const auto expected = Generate(cold);
  cold.Invalidate();
  auto runner = std::make_shared<Runner>();
  runner->in_pass = true;
  runner->max_in_pass_tail = std::numeric_limits<std::size_t>::max();
  TextRunnerPool pool(runner, 2);
  auto seed =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 3990);
  const auto step = seed.Prefill(8192);
  assert(step.consumed_tokens == prompt.size() && step.decode_ready);
  assert((runner->pass_ends == std::vector<std::size_t>{4000}));
  assert(std::ranges::find(runner->in_pass_points, 2048) !=
         runner->in_pass_points.end());
  assert(std::ranges::find(runner->in_pass_points, 3990) !=
         runner->in_pass_points.end());
  (void)Generate(seed);
  (void)seed.Commit();
  runner->pass_ends.clear();
  auto restored =
      pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, true, 3990);
  assert(restored.cached_prompt_tokens() == 2048);
  Prefill(restored);
  assert(Generate(restored) == expected);
  restored.Invalidate();
}
// A prompt target captured in a pass before a grid checkpoint of the same
// pass is admitted as a continuation boundary, not as the grid checkpoint
// whose reservation the lease held last.
void PromptTargetBeforeGridKeepsItsPurpose() {
  std::vector<TextRunnerToken> prompt(4000);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto runner = std::make_shared<Runner>();
  runner->in_pass = true;
  runner->max_in_pass_tail = std::numeric_limits<std::size_t>::max();
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    TextRunnerPool pool(runner, 1);
    auto seed =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 2000);
    (void)seed.Prefill(16384);
    assert(seed.prefill_complete());
    assert((runner->pass_ends == std::vector<std::size_t>{4000}));
    assert(std::ranges::find(runner->in_pass_points, 2000) !=
               runner->in_pass_points.end() &&
           std::ranges::find(runner->in_pass_points, 2048) !=
               runner->in_pass_points.end());
    (void)Generate(seed);
    (void)seed.Commit();
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto purpose = [&](std::size_t tokens) {
    std::istringstream lines(output.str());
    const std::string admitted =
        "event=retention action=" + std::to_string(static_cast<unsigned>(
                                        cache::RetentionAction::kAdmitted));
    const std::string at = " tokens=" + std::to_string(tokens) + " purpose=";
    for (std::string line; std::getline(lines, line);)
      if (line.find(admitted) != std::string::npos)
        if (const auto i = line.find(at); i != std::string::npos)
          return std::stoi(line.substr(i + at.size()));
    return -1;
  };
  assert(purpose(2000) ==
         static_cast<int>(cache::RetentionPurpose::kContinuation));
  assert(purpose(2048) == static_cast<int>(cache::RetentionPurpose::kHistory));
}
void ColdAcquireResetsOnce() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{1, 2, 3};
  auto first = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(runner->adapter->Invalidations() == 1);
  auto& state = dynamic_cast<State&>(first->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  (void)first->Commit(prompt, {}, prompt);
  auto second = cache.Acquire(prompt, {}, {}, false, 0, true);
  assert(!second->cache_hit());
  assert(runner->adapter->Invalidations() == 2);
}
void ColdAssignmentAvoidsDrainingCancelledSlot(bool deep = false) {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(deep ? 4096 : 3);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto completed = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(completed->state());
  auto* ready_slot = &state.slot;
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  if (deep) {
    assert(completed->TryReserveSnapshot(
        completed->CaptureBytes(), prompt.size(), false,
        SnapshotPurpose::kContinuation, prompt));
    assert(completed->PublishSnapshot(prompt, completed->Capture(), false));
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0,
                               prompt.size());
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0,
                               prompt.size());
  }
  (void)completed->Commit({}, {}, prompt);

  const std::vector<TextRunnerToken> long_prompt{11, 12, 13};
  auto cancelled = cache.Acquire(long_prompt, {}, {}, true, 0, true);
  auto* draining = &dynamic_cast<State&>(cancelled->state()).slot;
  assert(draining != ready_slot);
  runner->adapter->SetExecutionComplete(*draining, false);
  cancelled->Invalidate();

  const std::vector<TextRunnerToken> unrelated{21, 22, 23};
  auto peer = cache.Acquire(unrelated, {}, {}, true, 0, true);
  assert(&dynamic_cast<State&>(peer->state()).slot == ready_slot &&
         "an unrelated peer waited for a draining cancelled slot");
  // Readiness is a preference, not an admission condition. If the ready slot
  // is occupied, the remaining slot is still available for ordered reuse.
  auto fallback = cache.Acquire(unrelated, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(fallback->state()).slot == draining);
  runner->adapter->SetExecutionComplete(*draining, true);
  peer->Invalidate();
  fallback->Invalidate();
  if (deep) {
    auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(restored->cached_tokens() == prompt.size());
    auto& restored_state = dynamic_cast<State&>(restored->state());
    assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
  }
}
void ColdAssignmentKeepsOneDuplicateDeepFrontier() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 3, 1 << 22, {});
  std::vector<TextRunnerToken> conversation(4096);
  std::iota(conversation.begin(), conversation.end(), 10);
  const auto prefix = std::span(conversation).first(4088);
  const auto reply = std::span(conversation).last(8);
  auto first = cache.Acquire(conversation, {}, {}, true, 0, true);
  auto& first_state = dynamic_cast<State&>(first->state());
  runner->adapter->Append(first_state.slot, prefix, prefix);
  assert(first->TryReserveSnapshot(first->CaptureBytes(), prefix.size(), false,
                                   SnapshotPurpose::kContinuation, prefix));
  assert(first->PublishSnapshot({prefix.begin(), prefix.end()},
                                first->Capture(), false));
  runner->adapter->GuardRows(first_state.slot, cache::testing::kTarget, 0,
                             prefix.size());
  runner->adapter->GuardRows(first_state.slot, cache::testing::kDraft, 0,
                             prefix.size());
  // A sibling restores the same ancestor while the leader remains leased.
  auto second = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(second->cached_tokens() == prefix.size());
  auto& second_state = dynamic_cast<State&>(second->state());
  runner->adapter->Append(first_state.slot, reply, reply);
  runner->adapter->Append(second_state.slot, reply, reply);
  const auto hash = runner->adapter->RecurrentHash(first_state.slot);
  assert(runner->adapter->RecurrentHash(second_state.slot) == hash);
  (void)first->Commit({}, {}, conversation);
  (void)second->Commit({}, {}, conversation);

  const std::vector<TextRunnerToken> short_prompt{2, 3, 4};
  auto cancelled = cache.Acquire(short_prompt, {}, {}, true, 0, true);
  auto* draining = &dynamic_cast<State&>(cancelled->state()).slot;
  runner->adapter->SetExecutionComplete(*draining, false);
  cancelled->Invalidate();
  auto peer = cache.Acquire(short_prompt, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(peer->state()).slot != draining);
  // Once one copy is reassigned, the other is unique and must stay live.
  auto fallback = cache.Acquire(short_prompt, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(fallback->state()).slot == draining);
  auto retained = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(retained->cached_tokens() == conversation.size() &&
         !retained->restored_snapshot_bytes());
  assert(runner->adapter->RecurrentHash(
             dynamic_cast<State&>(retained->state()).slot) == hash);
  runner->adapter->SetExecutionComplete(*draining, true);
}
void FailedRestoreResetsOnce() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{1, 2, 3};
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  assert(source->PublishSnapshot(prompt, source->Capture(), false));
  const auto before = runner->adapter->Invalidations();
  runner->adapter->FailNextTransfer();
  auto failed = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(!failed->cache_hit());
  assert(runner->adapter->Invalidations() == before + 2);
  auto& cold = dynamic_cast<State&>(failed->state());
  runner->adapter->Append(cold.slot, prompt, prompt);
  assert(runner->adapter->RecurrentHash(cold.slot) ==
         runner->adapter->RecurrentHash(state.slot));
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
  std::shared_ptr<const ContinuationSnapshot> snapshot = source->Capture();
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
  std::shared_ptr<const ContinuationSnapshot> snapshot = source->Capture();
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
void ImageAfterCachedTextPrefixCaptures() {
  // #584: a request that appends the first image after a cached text-only
  // prefix restores that prefix, whose input identity is empty, and must
  // still capture past the image.
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> text(40);
  std::iota(text.begin(), text.end(), 10);
  {
    auto root = cache.Acquire(text, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(root->state());
    runner->adapter->Append(state.slot, text, text);
    assert(root->TryReserveSnapshot(root->CaptureBytes(), text.size(), false,
                                    SnapshotPurpose::kContinuation, {}));
    std::shared_ptr<const ContinuationSnapshot> snapshot = root->Capture();
    assert(root->PublishSnapshot(text, snapshot, false));
  }
  auto image = text;
  image.insert(image.end(), {50, 51, 52, 53});
  image.insert(image.end(), 16, 7);
  image.insert(image.end(), {60, 61});
  auto context = std::make_shared<TextPromptContext>();
  context->cache_identity = {1, 2, 3};
  context->cache_prefixes.push_back({44, {}});
  auto cold_runner = std::make_shared<Runner>();
  ComponentTextCache cold_cache(cold_runner, 1, 1 << 22, {});
  std::size_t inherited = text.size();
  for (const bool second_image : {false, true}) {
    if (second_image) {
      // A second image must preserve the first image's prefix identity too.
      image.insert(image.end(), {62, 63, 64, 65});
      context = std::make_shared<TextPromptContext>(*context);
      context->cache_prefixes.push_back({image.size(), {1, 2, 3}});
      context->cache_identity = {4, 5, 6};
      image.insert(image.end(), 16, 8);
      image.insert(image.end(), {70, 71});
    }
    auto lease = cache.Acquire(image, {}, context, true, 0, true);
    assert(lease->cached_tokens() == inherited);
    auto& state = dynamic_cast<State&>(lease->state());
    const auto suffix = std::span(image).subspan(inherited);
    runner->adapter->Append(state.slot, suffix, suffix);
    const auto hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), image.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    std::shared_ptr<const ContinuationSnapshot> snapshot = lease->Capture();
    assert(lease->PublishSnapshot(image, snapshot, false));

    // Model the retry after discarding a cancelled assistant: the image prompt
    // is unchanged and only a short text suffix is appended. Keep the source
    // lease active so the retry must restore a checkpoint in the other slot.
    auto retry_prompt = image;
    retry_prompt.push_back(99);
    auto retry = cache.Acquire(retry_prompt, {}, context, true, 0, true);
    assert(retry->cached_tokens() == image.size());
    assert(retry->restored_snapshot_bytes() > 0);
    auto& restored = dynamic_cast<State&>(retry->state());
    assert(runner->adapter->RecurrentHash(restored.slot) == hash);
    // Executing the suffix reads target/draft rows too; private state alone
    // cannot establish that those payloads were restored correctly.
    const auto tail = std::span(retry_prompt).last(1);
    runner->adapter->Append(state.slot, tail, tail);
    runner->adapter->Append(restored.slot, tail, tail);
    auto cold = cold_cache.Acquire(retry_prompt, {}, context, true, 0, true);
    assert(cold->cached_tokens() == 0);
    auto& cold_state = dynamic_cast<State&>(cold->state());
    cold_runner->adapter->Append(cold_state.slot, retry_prompt, retry_prompt);
    const auto expected = cold_runner->adapter->RecurrentHash(cold_state.slot);
    assert(runner->adapter->RecurrentHash(state.slot) == expected);
    assert(runner->adapter->RecurrentHash(restored.slot) == expected);
    inherited = image.size();
  }

  // Image placeholders alone cannot establish a reusable multimodal prefix.
  // Changing the first image permits only the original text checkpoint.
  auto changed = std::make_shared<TextPromptContext>();
  changed->cache_identity = {9, 9, 9};
  changed->cache_prefixes.push_back({44, {}});
  auto replacement = cache.Acquire(image, {}, changed, true, 0, true);
  assert(replacement->cached_tokens() == text.size());
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
void SharedPeerReusesTheCurrentStableCapture() {
  for (const bool settled : {false, true}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 2);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto leader = pool.Acquire(prompt, {}, {}, {}, true, 33);
    const auto first = leader.Prefill(64);
    assert(first.consumed_tokens == 33 && !first.decode_ready);
    if (settled)
      leader.CapturePromptSnapshot();
    assert(leader.ShareCheckpoint(prompt.size()) == 33 &&
           "sharing inserted an extra prefill after the stable boundary");
    const auto last = leader.Prefill(64);
    assert(last.consumed_tokens == 7 && last.decode_ready);
    auto peer = pool.Acquire(prompt, {}, {}, {}, true, 33);
    assert(peer.cached_prompt_tokens() == 33);
    assert(peer.Prefill(64).consumed_tokens == 7);
    assert(Generate(peer) == Generate(leader));
    peer.Invalidate();
    leader.Invalidate();
  }
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> prompt(40, 9);
  auto leader = pool.Acquire(prompt);
  assert(leader.Prefill(16).consumed_tokens == 16);
  assert(leader.ShareCheckpoint(prompt.size()) > 16 &&
         "an uncaptured partial pass cannot be offered to a peer");
  leader.Invalidate();
}
void PrefillLookaheadReachesTheNextFrontier() {
  class LookaheadRunner final : public Runner {
  public:
    struct Step {
      std::size_t frontier, end;
    };
    mutable std::vector<Step> steps;
    mutable const TextRunnerState* state{};
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      const auto ahead = state.prefill_lookahead();
      assert(ahead.empty() || ahead.data() == prompt.data() + prompt.size());
      steps.push_back({prompt.size(), prompt.size() + ahead.size()});
      this->state = &state;
      return Runner::Prefill(state, prompt, offset, budget);
    }
  };
  auto runner = std::make_shared<LookaheadRunner>();
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(4000);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto request = pool.Acquire(prompt);
  while (!request.Prefill(1024).decode_ready) {
  }
  assert(runner->state && runner->state->prefill_lookahead().empty());
  // A grid checkpoint splits the prompt; every step looks ahead exactly to
  // the frontier after its own.
  assert(runner->steps.front().frontier < prompt.size());
  for (std::size_t i = 0; i < runner->steps.size(); ++i) {
    const auto [frontier, end] = runner->steps[i];
    auto next = prompt.size();
    for (std::size_t j = i + 1; j < runner->steps.size(); ++j)
      if (runner->steps[j].frontier > frontier) {
        next = runner->steps[j].frontier;
        break;
      }
    assert(end == (frontier == prompt.size() ? frontier : next));
  }
}
void PrefillStepsSeeTheWholePromptEnd() {
  class PromptEndRunner final : public Runner {
  public:
    mutable std::vector<std::size_t> ends, frontiers;
    mutable const TextRunnerState* state{};
    void Record(const TextRunnerState& state,
                std::span<const TextRunnerToken> prompt) const {
      ends.push_back(state.prefill_prompt_end());
      frontiers.push_back(prompt.size());
      this->state = &state;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      Record(state, prompt);
      return Runner::Prefill(state, prompt, offset, budget);
    }
    TextPrefillStep PrefillThrough(
        TextRunnerState& state, std::span<const TextRunnerToken> prompt,
        std::size_t offset, std::size_t budget,
        std::span<const std::size_t> boundaries,
        const InPassConsumer& consume) const override {
      Record(state, prompt);
      return Runner::PrefillThrough(state, prompt, offset, budget, boundaries,
                                    consume);
    }
  };
  auto runner = std::make_shared<PromptEndRunner>();
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(4000);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto request = pool.Acquire(prompt);
  while (!request.Prefill(1024).decode_ready) {
  }
  assert(runner->state && runner->state->prefill_prompt_end() == 0);
  // A step ending at a cache boundary still sees the whole prompt's end.
  assert(runner->frontiers.front() < prompt.size());
  for (const auto end : runner->ends)
    assert(end == prompt.size());

  auto through_runner = std::make_shared<PromptEndRunner>();
  through_runner->in_pass = true;
  TextRunnerPool through_pool(through_runner, 1);
  std::vector<TextRunnerToken> short_prompt(40);
  std::iota(short_prompt.begin(), short_prompt.end(), 10);
  auto through = through_pool.Acquire(short_prompt, sampling::SamplingConfig{},
                                      {}, {}, true, 35);
  assert(through.Prefill(64).decode_ready);
  assert(through_runner->through_calls == 1);
  assert(through_runner->ends == std::vector<std::size_t>{short_prompt.size()});
  assert(through_runner->state->prefill_prompt_end() == 0);
}
void BackingPreparationFollowsCaptures() {
  class PrepareRunner final : public Runner {
  public:
    mutable std::size_t unfinished{}, preparations{}, next_tokens{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      const auto allocate = resources->allocate;
      resources->allocate = [this, allocate](const auto& request) {
        auto buffer = allocate(request);
        ++unfinished;
        buffer.finish = [this, finish = buffer.finish] {
          auto payload = finish();
          --unfinished;
          return payload;
        };
        buffer.borrow = [this, borrow = buffer.borrow](auto& lease, auto rows) {
          auto payload = borrow(lease, rows);
          --unfinished;
          return payload;
        };
        return buffer;
      };
      resources->prepare_prefill = [this](std::size_t tokens) {
        assert(unfinished == 0 && tokens != 0);
        next_tokens = tokens;
        ++preparations;
      };
      return resources;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      assert(next_tokens == std::min(prompt.size() - offset, budget));
      next_tokens = 0;
      return Runner::Prefill(state, prompt, offset, budget);
    }
  };
  auto runner = std::make_shared<PrepareRunner>();
  TextRunnerPool pool(runner, 1);
  for (const std::size_t size : {256, 600, 601}) {
    const std::vector<TextRunnerToken> prompt(size, 8);
    auto request = pool.Acquire(prompt);
    const auto before = runner->preparations;
    while (!request.Prefill(512).decode_ready) {
    }
    assert(runner->preparations > before);
    (void)request.Commit();
    const auto prepared = runner->preparations;
    auto cached = pool.Acquire(prompt);
    assert(cached.cached_prompt_tokens() == prompt.size());
    assert(cached.Prefill(512).consumed_tokens == 0);
    assert(runner->preparations == prepared);
    cached.Invalidate();
  }
}

void PartialFrontierDoesNotServeAnEqualPrompt() {
  // A prefill cancelled between steps keeps a frontier with no next token. A
  // prompt ending exactly there must recompute its tail instead of decoding
  // from a placeholder, whether the frontier is live or restored.
  class FrontierRunner final : public Runner {
  public:
    mutable std::vector<std::size_t> ready;
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      const auto step = Runner::Prefill(state, prompt, offset, budget);
      if (step.decode_ready)
        ready.push_back(CheckpointPosition(state));
      return step;
    }
    bool HasDecodeFrontier(const TextRunnerState& state) const override {
      return std::ranges::find(ready, CheckpointPosition(state)) != ready.end();
    }
  };
  for (const bool replace_slot : {false, true}) {
    auto runner = std::make_shared<FrontierRunner>();
    runner->partial = true;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    if (replace_slot) {
      auto unrelated = prompt;
      unrelated.front() = 999;
      auto other = pool.Acquire(unrelated, {}, {}, {}, true, 0);
      Prefill(other);
      (void)other.Commit();
    }
    const std::vector<TextRunnerToken> equal(prompt.begin(),
                                             prompt.begin() + 16);
    auto resend = pool.Acquire(equal, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() < equal.size());
    Prefill(resend);
    (void)Generate(resend);
    resend.Invalidate();
  }
}
void RefusedFrontierServesFromShallowerBoundary() {
  // A runner may keep only the state that longer prompts need. A prompt it
  // refuses recomputes from a shallower boundary, live or restored; a longer
  // one continues the frontier.
  class RefusingRunner final : public Runner {
  public:
    bool CanContinue(const TextRunnerState& state,
                     std::size_t prompt_tokens) const override {
      return prompt_tokens >= CheckpointPosition(state) + 8;
    }
  };
  // A live frontier left by a cancelled prefill.
  for (const std::size_t next_size : {20U, 40U}) {
    auto runner = std::make_shared<RefusingRunner>();
    runner->partial = true;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    const std::vector<TextRunnerToken> next_prompt(prompt.begin(),
                                                   prompt.begin() + next_size);
    auto next = pool.Acquire(next_prompt, {}, {}, {}, true, 0);
    assert((next.cached_prompt_tokens() == 16) == (next_size == 40));
    assert(next.cached_prompt_tokens() <= 16);
    Prefill(next);
    (void)Generate(next);
    next.Invalidate();
  }
  // A checkpoint restored after another prompt replaced the slot.
  for (const std::size_t next_size : {37U, 60U}) {
    auto runner = std::make_shared<RefusingRunner>();
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(60);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto seed = pool.Acquire(prompt, {}, {}, {}, true, 35);
    Prefill(seed);
    (void)seed.Commit();
    auto unrelated = prompt;
    unrelated.front() = 999;
    auto other = pool.Acquire(unrelated, {}, {}, {}, true, 0);
    Prefill(other);
    (void)other.Commit();
    auto next_prompt = prompt;
    next_prompt.resize(next_size);
    next_prompt.back() = 998;
    auto next = pool.Acquire(next_prompt, {}, {}, {}, true, 0);
    assert((next.cached_prompt_tokens() == 35) == (next_size == 60));
    Prefill(next);
    (void)Generate(next);
    next.Invalidate();
  }
}
void CompletedCancelledPromptGetsARetrySnapshot() {
  auto runner = std::make_shared<Runner>();
  runner->partial = true;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  {
    auto cancelled = pool.Acquire(prompt, {}, {}, {}, true, 35);
    assert(cancelled.Prefill(64).decode_ready);
    // Cancellation keeps the live prompt at 40 and an immutable fallback at
    // 35, without starting a complete-prompt capture.
    (void)cancelled.Cancel();
  }
  auto resumed = pool.Acquire(prompt, {}, {}, {}, true, 35);
  assert(resumed.cached_prompt_tokens() == prompt.size());
  const auto expected = Generate(resumed);
  assert(resumed.Commit().snapshot_bytes > 0);
  // Replace the execution slot so only immutable checkpoints can serve retry.
  auto unrelated = prompt;
  unrelated.front() = 999;
  auto replacement = pool.Acquire(unrelated, {}, {}, {}, true, 35);
  Prefill(replacement);
  (void)replacement.Commit();
  auto replay = pool.Acquire(prompt, {}, {}, {}, true, 35);
  assert(replay.cached_prompt_tokens() == prompt.size());
  assert(Generate(replay) == expected);
  replay.Invalidate();
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
void ColdSideRequestsKeepPreservedDeepFrontier() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> conversation(4096);
  std::iota(conversation.begin(), conversation.end(), 10);
  {
    auto source = cache.Acquire(conversation, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(source->state());
    const auto prefix = std::span(conversation).first(4088);
    runner->adapter->Append(state.slot, prefix, prefix);
    assert(source->TryReserveSnapshot(source->CaptureBytes(), prefix.size(),
                                      false, SnapshotPurpose::kContinuation,
                                      prefix));
    auto snapshot = source->Capture();
    assert(snapshot);
    assert(source->PublishSnapshot({prefix.begin(), prefix.end()},
                                   std::move(snapshot), false));
    // Deterministically finish preservation. A faster idle worker must not
    // remove the only protection of this live reply against short traffic.
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0,
                               prefix.size());
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0,
                               prefix.size());
    assert(source->Metrics().ledger.reserved_bytes[static_cast<unsigned>(
               cache::ResourceCategory::kBackingAssigned)] == 0);
    const auto reply = std::span(conversation).last(8);
    runner->adapter->Append(state.slot, reply, reply);
    (void)source->Commit({}, {}, conversation);
  }
  for (const TextRunnerToken code : {100, 101, 102}) {
    const std::vector<TextRunnerToken> short_prompt(12, code);
    auto side = cache.Acquire(short_prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(side->state());
    runner->adapter->Append(state.slot, short_prompt, short_prompt);
    // Readiness must not let short cancelled traffic evict the deep reply,
    // even though its older checkpoint has finished preservation.
    runner->adapter->SetExecutionComplete(state.slot, false);
    (void)side->Commit({}, {}, short_prompt);
  }
  auto resumed = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(resumed->cached_tokens() == conversation.size() &&
         !resumed->restored_snapshot_bytes() &&
         "short side requests evicted a preserved deep live frontier");
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
void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}
void Finish(TextRunnerPool::Request& request, std::size_t budget = 64) {
  while (!request.prefill_complete())
    (void)request.Prefill(budget);
}
void PromptReuseCanBeDisabledPerRequest() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt{1, 2};
  {
    auto initial = pool.Acquire(prompt);
    Finish(initial);
    prompt.push_back(Generate(initial));
    (void)initial.Commit();
  }
  prompt.push_back(7);
  auto cold = pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
  Expect(!cold.cache_hit() && cold.cached_prompt_tokens() == 0,
         "cache_prompt=false bypasses live and retained frontiers");
  Expect(cold.Prefill(8).consumed_tokens == prompt.size(),
         "disabled reuse processes every prompt token");
  (void)cold.Commit();
  prompt.push_back(8);
  auto retained = pool.Acquire(prompt);
  Expect(retained.cached_prompt_tokens() == prompt.size() - 1,
         "a no-reuse request can populate the cache for later requests");
  retained.Invalidate();
}
void StableChatPrefixSurvivesInterruptedFraming() {
  const TemporaryDirectory temp("gufo-component-stable-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 1 << 12,
                                        .min_checkpoint_step_tokens = 0};
  auto runner = std::make_shared<Runner>();
  const std::vector<TextRunnerToken> resumed_prompt{1, 2, 3, 50, 51, 60};
  {
    TextRunnerPool pool(runner, 1, disk);
    // {1,2,3} is stable history; {40,41} opens assistant reasoning.
    auto initial = pool.Acquire({1, 2, 3, 40, 41}, sampling::SamplingConfig{},
                                {}, {}, true, 3);
    Expect(initial.Prefill(64).consumed_tokens == 3,
           "prefill stops before mutable assistant framing");
    Finish(initial);
    (void)Generate(initial);
    (void)initial.Cancel();
    // The client omits unfinished reasoning and closes the assistant turn.
    auto resumed = pool.Acquire(resumed_prompt, sampling::SamplingConfig{}, {},
                                {}, true, 5);
    Expect(resumed.cached_prompt_tokens() == 3,
           "changed assistant framing retains all stable prompt tokens");
    resumed.Invalidate();
  }
  TextRunnerPool restarted(runner, 1, disk);
  auto restored = restarted.Acquire(resumed_prompt, sampling::SamplingConfig{},
                                    {}, {}, true, 5);
  Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
         "interrupted stable chat prefix survives server restart");
  restored.Invalidate();
}
void WarmTurnsAdvanceTheStableCheckpoint() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1);
  const auto acquire = [&](std::vector<TextRunnerToken> prompt,
                           std::size_t stable) {
    return pool.Acquire(std::move(prompt), sampling::SamplingConfig{}, {}, {},
                        true, stable);
  };
  // Cold turn: the stable boundary is 3; 40,41 is the assistant framing.
  auto first = acquire({1, 2, 3, 40, 41}, 3);
  Expect(first.Prefill(64).consumed_tokens == 3,
         "cold chat stops at its stable boundary");
  Finish(first);
  const auto reply = Generate(first);
  (void)first.Commit();

  // A warm turn replays the assistant, then a new user turn. Its stable
  // boundary is 7, past the frontier of 6 that it reuses.
  auto second = acquire({1, 2, 3, 40, 41, reply, 7, 40, 41}, 7);
  Expect(
      second.cached_prompt_tokens() == 6 && second.cache_restore_bytes() == 0,
      "retained reasoning reuses generated tokens without a restore");
  Expect(second.Prefill(64).consumed_tokens == 1,
         "warm prefill stops at the stable boundary before framing");
  Finish(second);
  const auto again = Generate(second);
  (void)second.Commit();

  auto repeated = acquire({1, 2, 3, 40, 41, reply, 7, 40, 41}, 7);
  Expect(repeated.cached_prompt_tokens() == 9 && repeated.prefill_complete(),
         "an identical prompt reuses the complete prompt checkpoint");
  Expect(Generate(repeated) == again,
         "the complete checkpoint preserves greedy selection");
  (void)repeated.Commit();

  // Dropped reasoning rewrites 40,41 at token 7, the last stable boundary.
  auto third = acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 40, 41}, 10);
  Expect(third.cached_prompt_tokens() == 7,
         "a warm turn checkpoints its own stable boundary");
  Finish(third);
  (void)Generate(third);
  (void)third.Commit();

  // Dropping reasoning again keeps advancing instead of falling back.
  auto fourth =
      acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 60, 61, 11, 40, 41}, 13);
  Expect(fourth.cached_prompt_tokens() == 10,
         "successive rewritten turns resume from the latest boundary");
  (void)fourth.Cancel();
}
void FullChatCheckpointRestoresWithoutSuffixPrefill() {
  const TemporaryDirectory temp("gufo-component-full-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 1 << 12,
                                        .min_checkpoint_step_tokens = 0};
  const std::vector<TextRunnerToken> prompt{1, 2, 3, 40, 41};
  auto runner = std::make_shared<Runner>();
  TextRunnerToken token{};
  {
    TextRunnerPool writer(runner, 1, disk);
    auto first =
        writer.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 3);
    Finish(first);
    token = Generate(first);
    (void)first.Commit();
  }
  TextRunnerPool reader(runner, 1, disk);
  auto exact =
      reader.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 3);
  Expect(exact.cache_disk_hit() &&
             exact.cached_prompt_tokens() == prompt.size() &&
             exact.prefill_complete(),
         "disk restores the exact prompt instead of recomputing its suffix");
  Expect(Generate(exact) == token, "a disk restore selects the same token");
  (void)exact.Commit();
  auto repeated =
      reader.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 3);
  Expect(!repeated.cache_disk_hit() && repeated.prefill_complete(),
         "a disk-restored prompt is retained for cheap RAM retries");
  (void)repeated.Commit();
  auto changed = reader.Acquire({1, 2, 3, 50, 51, 60},
                                sampling::SamplingConfig{}, {}, {}, true, 5);
  Expect(changed.cached_prompt_tokens() == 3,
         "the earlier fallback survives exact full-prompt restoration");
  changed.Invalidate();
}
void GeneratedFrontierForksBeforeMutation() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 2);
  std::vector<TextRunnerToken> root{1, 2, 3, 4};
  {
    auto first = pool.Acquire(root);
    Finish(first);
    root.push_back(Generate(first));
    (void)first.Commit();
  }
  auto left = root, right = root;
  left.push_back(5);
  right.push_back(6);
  auto first = pool.Acquire(left);
  auto second = pool.Acquire(right);
  // The live frontier serves one fork. No checkpoint covers the generated
  // token, so the peer restores the prompt and prefills that token again.
  Expect(first.cached_prompt_tokens() == root.size() &&
             first.cache_restore_bytes() == 0,
         "the first fork continues the live generated frontier");
  Expect(second.cached_prompt_tokens() == root.size() - 1 &&
             second.cache_restore_bytes() > 0,
         "the peer restores the retained prompt");
  Expect(first.Prefill(64).consumed_tokens == 1 &&
             second.Prefill(64).consumed_tokens == 2,
         "each fork prefills only what its frontier lacks");
  first.Invalidate();
  second.Invalidate();
  auto branch = pool.Acquire({1, 2, 3, 4, 7});
  Expect(branch.cached_prompt_tokens() == 4,
         "freezing a generated frontier preserves the prompt branch");
  branch.Invalidate();
}
void InterruptedPrefillRetainsOnlyItsEntryFrontier() {
  class InterruptedRunner final : public Runner {
  public:
    explicit InterruptedRunner(bool& cancelled) : cancelled_(cancelled) {}
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = Runner::Descriptor();
      descriptor.capabilities.partial_prefill_retention = true;
      descriptor.capabilities.prefill_rollback = rollback;
      return descriptor;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      if (!offset)
        return Runner::Prefill(state, prompt, offset, budget);
      cancelled_ = cancel;
      return {.consumed_tokens = consumed, .interrupted = true};
    }
    bool rollback{true}, cancel{true};
    std::size_t consumed{};

  private:
    bool& cancelled_;
  };
  for (int scenario = 0; scenario < 4; ++scenario) {
    bool cancelled = false;
    auto runner = std::make_shared<InterruptedRunner>(cancelled);
    runner->rollback = scenario != 1;
    runner->cancel = scenario != 2;
    runner->consumed = scenario == 3 ? 1 : 0;
    TextRunnerPool pool(runner, 1);
    auto request = pool.Acquire(
        {1, 2, 3, 4, 5, 6}, sampling::SamplingConfig{},
        [&] { return cancelled; }, {}, true, 5);
    Expect(request.Prefill(4).consumed_tokens == 4,
           "a completed chunk precedes the interrupted pass");
    std::string error;
    try {
      (void)request.Prefill(2);
    } catch (const std::runtime_error& failure) {
      error = failure.what();
    }
    Expect(error == (scenario == 0 ? "generation cancelled"
                                   : "invalid interrupted prefill frontier"),
           "only a cancelled, rolled-back pass may retain its entry frontier");
    (void)request.Cancel();
    cancelled = false;
    auto resumed = pool.Acquire({1, 2, 3, 4, 5, 6, 7});
    Expect(resumed.cached_prompt_tokens() == (scenario == 0 ? 4 : 0),
           "invalid interrupted state must never become a cache hit");
    resumed.Invalidate();
  }
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
  const TemporaryDirectory temp("gufo-component-serving-test");
  const std::filesystem::path directory = temp.path();
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
  std::shared_ptr<const ContinuationSnapshot> snapshot = lease->Capture();
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
void BackingPressureKeepsSharedPrefix(bool capacity_check = false) {
  struct PressureRunner final : Runner {
    bool capacity_check{};
    mutable std::size_t inherited_boundary{}, recopied_prefixes{},
        rejected_plans{}, failed_allocations{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->allocate = [this, allocate = std::move(resources->allocate)](
                                const cache::PayloadRequest& request) {
        if (request.category == cache::ResourceCategory::kBackingAssigned &&
            request.first < inherited_boundary)
          ++recopied_prefixes;
        try {
          return allocate(request);
        } catch (const cache::ResourceExhausted&) {
          ++failed_allocations;
          throw;
        }
      };
      if (capacity_check)
        resources->can_capture = [this, &ledger](auto payloads) {
          const bool needs_private =
              std::ranges::any_of(payloads, [](const auto& payload) {
                return payload.category ==
                       cache::ResourceCategory::kPrivateState;
              });
          const bool fits = !needs_private ||
                            !ledger.Snapshot().bytes[static_cast<std::size_t>(
                                cache::ResourceCategory::kPrivateState)];
          rejected_plans += !fits;
          return fits;
        };
      return resources;
    }
  };
  auto runner = std::make_shared<PressureRunner>();
  runner->capacity_check = capacity_check;
  // The next private frontier cannot fit until the old checkpoint is evicted.
  runner->private_blocks = 1;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(80);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, std::span(prompt).first(64),
                          std::span(prompt).first(64));
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 64, false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 64},
                                lease->Capture(), false));
  runner->inherited_boundary = 64;
  runner->adapter->Append(state.slot, std::span(prompt).subspan(64),
                          std::span(prompt).subspan(64));
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
  assert(!runner->recopied_prefixes);
  if (capacity_check)
    assert(runner->rejected_plans && !runner->failed_allocations);
  else
    assert(runner->failed_allocations);
  assert(!lease->HasSnapshotFor(std::span(prompt).first(64)));
  lease->Invalidate();
  auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(restored->cached_tokens() == prompt.size());
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
}
void FragmentedBackingKeepsCompletePrompt() {
  struct FragmentedRunner final : Runner {
    mutable std::size_t failed_allocations{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      struct Pool {
        std::array<cache::ResourceCharge, 6> pages;
        std::array<cache::ResourceCharge, 2> occupied;
        std::shared_ptr<std::array<std::byte, 96>> data =
            std::make_shared<std::array<std::byte, 96>>();
      };
      auto pool = std::make_shared<Pool>();
      for (auto& page : pool->pages)
        page =
            ledger.Reserve(cache::ResourceCategory::kBackingFree, 16).Convert();
      for (std::size_t i = 0; i < pool->occupied.size(); ++i)
        pool->occupied[i] =
            pool->pages[2 + 2 * i]
                .ReserveBacking(cache::ResourceCategory::kPrivateState)
                .Convert();
      // Free runs have sizes 32, 16, 16. Two 16-byte row captures followed
      // by the 32-byte state cannot fit with first-fit allocation, despite
      // sufficient total space. The existing occupied pages cannot be evicted.
      resources->owner = pool;
      resources->allocate = [this, pool](const cache::PayloadRequest& request) {
        const auto count = (request.bytes + 15) / 16;
        const auto first =
            cache::ResourceCharge::FindFreeBacking(pool->pages, count);
        if (first == pool->pages.size()) {
          ++failed_allocations;
          throw cache::ResourceExhausted();
        }
        auto reservation = std::make_shared<cache::ResourceReservation>(
            cache::ResourceCharge::ReserveBacking(
                std::span(pool->pages).subspan(first, count),
                request.category));
        auto bytes = std::span(*pool->data).subspan(first * 16, count * 16);
        auto owner = std::shared_ptr<const void>(pool->data, bytes.data());
        return ComponentCaptureBuffer{bytes,
                                      [reservation, owner] {
                                        return cache::Payload::Committed(
                                            reservation->Convert(), owner);
                                      },
                                      pool->data.get(),
                                      {}};
      };
      return resources;
    }
  };
  auto runner = std::make_shared<FragmentedRunner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  std::unique_ptr<ContinuationSnapshot> snapshot;
  try {
    snapshot = lease->Capture();
  } catch (const cache::ResourceExhausted&) {
  }
  assert(snapshot && "fragmented backing rejected a complete prompt");
  assert(lease->PublishSnapshot(prompt, std::move(snapshot), false));
  assert(!runner->failed_allocations);
  lease->Invalidate();
  auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(restored->cached_tokens() == prompt.size());
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
  const TextRunnerToken next = 99;
  runner->adapter->Append(restored_state.slot, {&next, 1}, {&next, 1});
}
void DiskBudgetEvictsAnUnpinnedConversation() {
  const TemporaryDirectory temp("gufo-component-budget-test");
  const std::filesystem::path directory = temp.path();
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
void UnpersistedRequestStaysOffDisk() {
  const TemporaryDirectory temp("gufo-component-unpersisted-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto acquire = [&](TextRunnerPool& pool, bool persist) {
    return pool.Acquire(prompt, {}, {}, {}, true, 0, true, persist);
  };
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = acquire(pool, false);
    Prefill(first);
    token = Generate(first);
    assert(first.Commit().disk_queued_bytes == 0);
    // The RAM cache still serves the prompt within this process.
    auto again = acquire(pool, false);
    assert(again.cache_hit() && !again.cache_disk_hit());
    assert(again.cached_prompt_tokens() == prompt.size());
    again.Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto cold = acquire(pool, true);
    assert(!cold.cache_hit());
    Prefill(cold);
    assert(Generate(cold) == token);
    assert(cold.Commit().disk_queued_bytes > 0);
  }
  assert(ManagedBytes(directory) > 0);
  {
    // Reading what other requests stored is allowed.
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto reader = acquire(pool, false);
    assert(reader.cache_disk_hit());
    assert(reader.cached_prompt_tokens() == prompt.size());
    reader.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void UnpersistedOverRamCaptureIsRefused() {
  // A checkpoint larger than the RAM budget would be staged for disk only.
  // A request that stays off disk refuses it instead.
  const TemporaryDirectory temp("gufo-component-unpersisted-over-ram-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 2;
  runner->rows_per_chunk = 4096;
  constexpr std::size_t kRam = 448 << 10;
  {
    ComponentTextCache cache(runner, 1, kRam, disk);
    std::vector<TextRunnerToken> prompt(65536);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true, false);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    const auto bytes = lease->CaptureBytes();
    assert(bytes > kRam);
    assert(lease->TryReserveSnapshot(bytes, prompt.size(), false,
                                     SnapshotPurpose::kContinuation, prompt));
    bool refused = false;
    try {
      (void)lease->Capture();
    } catch (const cache::ResourceExhausted&) {
      refused = true;
    }
    assert(refused);
    lease->Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  std::filesystem::remove_all(directory);
}
void LoweredDiskBudgetEvictsOldestAtStartup() {
  const TemporaryDirectory temp("gufo-component-lowered-budget-test");
  const std::filesystem::path directory = temp.path();
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
  const TemporaryDirectory temp("gufo-component-orphan-test");
  const std::filesystem::path directory = temp.path();
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
  const TemporaryDirectory temp("gufo-component-staging-test");
  const std::filesystem::path directory = temp.path();
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
  const TemporaryDirectory temp("gufo-component-ceiling-test");
  const std::filesystem::path directory = temp.path();
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
// Slots that pin host memory lower both measured budgets and are charged in
// the ledger. They count once, on top of the cache's budget: backing gets the
// whole budget unless the host ceiling binds.
void PinnedSlotBytesCountOnce() {
  constexpr std::size_t kPin = 512 << 10;
  class PinningRunner final : public Runner {
  public:
    mutable std::size_t committed{}, limit{};
    mutable std::vector<cache::ResourceCharge> pins;
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      // Automatic is half the available RAM; the ceiling is all of it.
      claim.retained_snapshot_capacity_bytes =
          (2 << 20) - pins.size() * kPin / 2;
      claim.retained_snapshot_ceiling_bytes = (4 << 20) - pins.size() * kPin;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      // Each slot takes a stream as it allocates; pin one buffer per slot.
      resources->stream = [this, &ledger,
                           stream = std::move(resources->stream)] {
        if (!committed)
          pins.push_back(
              ledger.Reserve(cache::ResourceCategory::kMetadata, kPin)
                  .Convert());
        return stream();
      };
      resources->pinned_bytes = [this] { return pins.size() * kPin; };
      resources->commit_backing = [this, &ledger](std::size_t budget) {
        committed = budget;
        limit = ledger.Limits().ram_bytes;
      };
      return resources;
    }
  };
  // Automatic, an explicit budget smaller than the pins, and one the 4 MiB
  // host ceiling cuts to make room for them.
  for (const auto& [requested, backing] :
       {std::pair{0ULL, 2ULL << 20}, std::pair{1ULL << 20, 1ULL << 20},
        std::pair{8ULL << 20, 3ULL << 20}}) {
    auto runner = std::make_shared<PinningRunner>();
    TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = requested});
    assert(runner->pins.size() == 2);
    assert(runner->committed == backing);
    assert(runner->limit == backing + 2 * kPin);
  }
  // Pins never raise the limit past a budget the other charges exceed: the
  // fake backing alone holds 40 KiB.
  bool refused = false;
  try {
    TextRunnerPool pool(std::make_shared<PinningRunner>(), 2, {},
                        {.capacity_bytes = 16 << 10});
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused);
  // Staging shares the ceiling: the limit is 4 MiB less 1 MiB of staging.
  const TemporaryDirectory temp("gufo-component-pinned-test");
  const std::filesystem::path directory = temp.path();
  {
    const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                          .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<PinningRunner>();
    TextRunnerPool pool(runner, 2, disk, {.capacity_bytes = 8 << 20});
    assert(runner->limit == 3 << 20);
    assert(runner->committed == 2 << 20);
  }
  std::filesystem::remove_all(directory);
}
// Without pins the limit ends at the requested budget, whether or not the
// runner commits backing, and a budget below what is already charged still
// fails at startup.
void UnpinnedRunnersKeepTheirLimit() {
  class LimitRunner final : public Runner {
  public:
    bool commit{};
    mutable cache::ResourceLedger* ledger{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& l, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(l, ram);
      ledger = &l;
      if (commit)
        resources->commit_backing = [](std::size_t) {};
      return resources;
    }
  };
  for (const bool commit : {false, true}) {
    auto runner = std::make_shared<LimitRunner>();
    runner->commit = commit;
    {
      TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = 1 << 20});
      assert(runner->ledger->Limits().ram_bytes == 1 << 20);
    }
    // The fake backing alone holds 40 KiB.
    bool refused = false;
    try {
      TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = 16 << 10});
    } catch (const std::exception&) {
      refused = true;
    }
    assert(refused);
  }
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
  const TemporaryDirectory temp("gufo-component-owner-test");
  const std::filesystem::path directory = temp.path();
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
  const TemporaryDirectory temp("gufo-component-spacing-test");
  const std::filesystem::path directory = temp.path();
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
void PublishedResidentCheckpointKeepsDiskStep() {
  // A published checkpoint that is still resident restores from RAM, so its
  // lookup candidate carries no disk description. The next capture within
  // the step must still be refused.
  const TemporaryDirectory temp("gufo-component-step-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7,
                                        .min_checkpoint_step_tokens = 16};
  std::vector<TextRunnerToken> prompt(48);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> first(prompt.begin(), prompt.begin() + 40);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, disk);
  {
    auto lease = cache.Acquire(first, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, first, first);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), first.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    assert(lease->PublishSnapshot(first, lease->Capture(), false));
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
    // Invalidation releases the execution slot so the idle queue can publish.
    // Wait for completion rather than guessing when a created file is attached.
    for (int i = 0; i < 500 && lease->Metrics().persistence->completed == 0;
         ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(lease->Metrics().persistence->completed == 1);
  }
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->cached_tokens() == first.size() &&
         !lease->restored_from_disk());
  auto& state = dynamic_cast<State&>(lease->state());
  const auto tail = std::span(prompt).subspan(first.size());
  runner->adapter->Append(state.slot, tail, tail);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
  assert(lease->Metrics().disk_queued_bytes == 0);
  lease->Invalidate();
  std::filesystem::remove_all(directory);
}
void NearTailDiskBranchSurvivesRestart() {
  const TemporaryDirectory temp("gufo-component-branch-test");
  const std::filesystem::path directory = temp.path();
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
  struct Gate {
    std::atomic<bool> entered{false};
    std::binary_semaphore reading{0}, release{0};
  };
  struct HeldRunner final : Runner {
    std::shared_ptr<Gate> gate = std::make_shared<Gate>();
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->copy_observed_rows =
          [gate = gate, copy = resources->copy_observed_rows](
              auto source, auto destination, cache::Stream& stream) {
            if (!gate->entered.exchange(true)) {
              gate->reading.release();
              gate->release.acquire();
            }
            return copy(source, destination, stream);
          };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-deep-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 8,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(5001);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<HeldRunner>();
    // Keep the same over-RAM row bytes and a partial tail while avoiding
    // thousands of tiny durable chunk files in this restart contract.
    runner->rows_per_chunk = 64;
    runner->row_backing_bytes = 64 * sizeof(cache::Token);
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
    assert(runner->gate->reading.try_acquire_for(std::chrono::seconds(5)));
    const auto pending = cache.PendingPublication(prompt, {});
    assert(pending && pending->checkpoint_id &&
           pending->pending_tokens == prompt.size() &&
           pending->available_tokens == 0 && pending->Gain() == prompt.size());
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id + 1));
    assert(!cache.PendingPublication(std::span(prompt).first(4096), {}));
    auto different = prompt;
    ++different.front();
    assert(!cache.PendingPublication(different, {}));
    TextPromptContext image;
    image.cache_identity = {99};
    assert(!cache.PendingPublication(prompt, &image));
    (void)lease->Commit(prompt, {}, prompt);
    // A newly usable live frontier makes the pending write unhelpful. The
    // tracked job remains identifiable so its existing deadline can shrink;
    // a fresh admission must not begin waiting for it.
    const auto tracked =
        cache.PendingPublication(prompt, {}, pending->checkpoint_id);
    assert(tracked && tracked->Gain() == 0);
    assert(!cache.PendingPublication(prompt, {}));
    runner->gate->release.release();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (cache.PendingPublication(prompt, {}, pending->checkpoint_id) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id));
  }
  {
    const auto seed_size = prompt.size();
    std::vector<TextRunnerToken> appended(63);
    std::iota(appended.begin(), appended.end(), 8000);
    prompt.insert(prompt.end(), appended.begin(), appended.end());
    auto runner = std::make_shared<Runner>();
    runner->rows_per_chunk = 64;
    runner->row_backing_bytes = 64 * sizeof(cache::Token);
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
    bool checked_tail = false;
    for (std::size_t i = 0; i < parent->components.size(); ++i) {
      const auto& before = parent->components[i];
      const auto& after = child->components[i];
      for (std::size_t j = 0; j < before.chunks.size(); ++j)
        assert(before.chunks[j].file == after.chunks[j].file);
      if (before.private_state)
        assert(before.private_state->file != after.private_state->file);
      if (before.tail) {
        checked_tail = true;
        assert(after.chunks.size() > before.chunks.size());
        assert(before.tail->file != after.chunks[before.chunks.size()].file);
      }
    }
    assert(checked_tail);
  }
  {
    auto runner = std::make_shared<Runner>();
    runner->rows_per_chunk = 64;
    runner->row_backing_bytes = 64 * sizeof(cache::Token);
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() &&
           lease->cached_tokens() == prompt.size());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
// Window components through serving: RAM retries, continuations and edits
// match an uncached control. Window 6 rows, ring 16, chunks of 4.
std::shared_ptr<Runner> WindowRunner(bool private_state) {
  auto runner = std::make_shared<Runner>();
  runner->window_rows = 6;
  runner->ring_rows = 16;
  runner->private_state = private_state;
  return runner;
}
void WindowRowsReuseAndBranches(bool private_state) {
  auto runner = WindowRunner(private_state);
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
  (void)retry.Commit();
  auto history = prompt;
  history.push_back(token);
  for (TextRunnerToken t = 500; t < 530; ++t)
    history.push_back(t);
  auto continued = pool.Acquire(history);
  assert(continued.cached_prompt_tokens() == prompt.size() + 1);
  Prefill(continued);
  const auto next = Generate(continued);
  (void)continued.Commit();
  auto edited_history = history;
  edited_history[45] += 1;
  auto edit = pool.Acquire(edited_history);
  assert(edit.cache_hit() && edit.cached_prompt_tokens() <= 45);
  Prefill(edit);
  const auto edited = Generate(edit);
  (void)edit.Commit();
  for (const auto* tokens : {&history, &edited_history}) {
    auto cold =
        pool.Acquire(*tokens, sampling::SamplingConfig{}, {}, {}, false);
    assert(!cold.cache_hit());
    Prefill(cold);
    assert(Generate(cold) == (tokens == &history ? next : edited));
    cold.Invalidate();
  }
}
// Rows beyond the RAM pool go to disk only. A later capture three tokens
// deeper inherits the parent's second window chunk as its first, and both
// restore with the window a continuation reads.
void WindowedDeepRowsSurviveDiskRestart(bool private_state) {
  const TemporaryDirectory temp("gufo-component-window-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 8,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(5001);
  std::iota(prompt.begin(), prompt.end(), 10);
  const auto window = [&](std::size_t n) {
    return std::vector<cache::Token>(prompt.begin() + n - 6,
                                     prompt.begin() + n);
  };
  const auto capture = [&](ComponentTextCache::Lease& lease) {
    assert(lease.TryReserveSnapshot(lease.CaptureBytes(), prompt.size(), false,
                                    SnapshotPurpose::kContinuation, prompt));
    auto snapshot = lease.Capture();
    assert(snapshot);
    assert(lease.PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease.Metrics().disk_queued_bytes > 40000);
    (void)lease.Commit(prompt, {}, prompt);
  };
  std::uint64_t hash;
  {
    auto runner = WindowRunner(private_state);
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    capture(*lease);
  }
  {
    const auto seed_size = prompt.size();
    const std::vector<TextRunnerToken> appended{8000, 8001, 8002};
    prompt.insert(prompt.end(), appended.begin(), appended.end());
    auto runner = WindowRunner(private_state);
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == seed_size);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
    assert(runner->adapter->WindowTokens(state.slot) == window(seed_size));
    runner->adapter->Append(state.slot, appended, appended);
    hash = runner->adapter->RecurrentHash(state.slot);
    capture(*lease);
  }
  {
    cache::ResourceLedger ledger({1 << 24, 1 << 24, 0, 0});
    cache::DiskStore store(ledger, directory, disk.capacity_bytes);
    assert(store.Entries().size() == 2);
    const cache::DiskManifest* parent = nullptr;
    const cache::DiskManifest* child = nullptr;
    for (const auto& entry : store.Entries())
      (entry.Manifest().tokens.size() == 5001 ? parent : child) =
          &entry.Manifest();
    const auto& before = parent->components.back();
    const auto& after = child->components.back();
    assert(before.descriptor.id == kWindow && after.descriptor.id == kWindow);
    // Parent window chunks [4992, 5000) and tail row 5000; child [4996, 5004).
    assert(before.chunks.size() == 2 && before.tail);
    assert(after.chunks.size() == 2 && !after.tail);
    assert(before.chunks[1].file == after.chunks[0].file);
    assert(before.tail->file != after.chunks[1].file);
  }
  {
    auto runner = WindowRunner(private_state);
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() &&
           lease->cached_tokens() == prompt.size());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
    assert(runner->adapter->WindowTokens(state.slot) == window(prompt.size()));
  }
}
void OverRamCaptureStagesPrivateStateUnderPressure(bool protect) {
  // An over-RAM checkpoint is captured for disk only. Its rows are borrowed
  // from the live slot, but its private state is staged in RAM backing, and
  // two retained checkpoints fill that backing. Wide chunks keep the capture
  // metadata within RAM.
  const TemporaryDirectory temp("gufo-component-over-ram-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 2;
  runner->rows_per_chunk = 4096;
  constexpr std::size_t kRam = 448 << 10;
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    ComponentTextCache cache(runner, 1, kRam, disk);
    const auto history = [](auto& lease, const auto& tokens,
                            std::size_t boundary) {
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), boundary, false,
                                       SnapshotPurpose::kHistory, {}));
      assert(lease->PublishSnapshot({tokens.begin(), tokens.begin() + boundary},
                                    lease->Capture(), false));
    };
    const auto append = [&](auto& lease, std::span<const TextRunnerToken> t) {
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, t, t);
    };
    std::vector<TextRunnerToken> prompt(65536);
    std::iota(prompt.begin(), prompt.end(), 10);
    if (!protect) {
      // The cache may reclaim another conversation's history checkpoints.
      const std::vector<TextRunnerToken> other(4, 7);
      auto lease = cache.Acquire(other, {}, {}, true, 0, true);
      append(lease, std::span(other).first(2));
      history(lease, other, 2);
      append(lease, std::span(other).subspan(2));
      history(lease, other, 4);
      (void)lease->Commit({}, {}, other);
    }
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    if (protect) {
      // The cache keeps this turn's own edit boundaries until it commits.
      append(lease, std::span(prompt).first(2));
      history(lease, prompt, 2);
      append(lease, std::span(prompt).subspan(2, 2));
      history(lease, prompt, 4);
      append(lease, std::span(prompt).subspan(4));
    } else {
      append(lease, prompt);
    }
    const auto bytes = lease->CaptureBytes();
    assert(bytes > kRam);
    assert(lease->TryReserveSnapshot(bytes, prompt.size(), false,
                                     SnapshotPurpose::kContinuation, prompt));
    if (protect) {
      bool refused = false;
      try {
        (void)lease->Capture();
      } catch (const cache::ResourceExhausted&) {
        refused = true;
      }
      assert(refused);
    } else {
      // Persisting would charge the manifest's tokens again, and in this fake
      // they cost as many bytes as the rows. This test covers the capture.
      auto snapshot = lease->Capture();
      assert(snapshot && snapshot->PayloadBytes() == bytes);
    }
    lease->Invalidate();
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  const auto refused =
      log.find("event=capture_refused reason=byte_capacity tokens=65536");
  assert(protect == (refused != std::string::npos));
  // Only the other conversation's history is reclaimed. Edit boundaries stay.
  const auto reclaimed =
      "event=retention action=" +
      std::to_string(static_cast<unsigned>(cache::RetentionAction::kRemoved)) +
      " reason=" +
      std::to_string(
          static_cast<unsigned>(cache::RetentionReason::kByteCapacity));
  assert(protect == (log.find(reclaimed) == std::string::npos));
  assert(log.find("event=capture_failed") == std::string::npos);
  std::filesystem::remove_all(directory);
}
}  // namespace
// An off RAM cache commits no backing, grows none before prefill and takes no
// capture, yet a session keeps its own live frontier and its metrics stay
// whole.
void RamCacheOffCommitsNoBackingAndKeepsLiveFrontiers() {
  class CountingRunner final : public Runner {
  public:
    mutable std::size_t committed{}, prepared{}, claims{};
    TextRunnerResourceClaim ResourceClaim() const override {
      ++claims;
      return Runner::ResourceClaim();
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t) { ++committed; };
      resources->prepare_prefill = [this](std::size_t) { ++prepared; };
      return resources;
    }
  };
  for (const bool enabled : {true, false}) {
    auto runner = std::make_shared<CountingRunner>();
    ComponentTextCache cache(runner, 2, 1 << 22, {}, false, 0, enabled);
    std::vector<TextRunnerToken> conversation(12, 7);
    for (int turn = 0; turn < 2; ++turn) {
      auto lease = cache.Acquire(conversation, {}, {}, true, 0, true);
      if (!enabled) {
        assert(lease->PlannedCaptures(conversation.size()).empty());
        // Off is a policy, independent of a caller's payload estimate.
        for (const auto purpose :
             {SnapshotPurpose::kContinuation, SnapshotPurpose::kRetry,
              SnapshotPurpose::kBranchPoint, SnapshotPurpose::kHistory})
          assert(!lease->TryReserveSnapshot(0, conversation.size(), false,
                                            purpose, conversation));
      }
      const auto cached = lease->cached_tokens();
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(conversation).subspan(cached);
      lease->PreparePrefill(tail.size());
      runner->adapter->Append(state.slot, tail, tail);
      const auto reserved = lease->TryReserveSnapshot(
          lease->CaptureBytes(), conversation.size(), false,
          SnapshotPurpose::kContinuation, conversation);
      assert(reserved == enabled);
      if (reserved)
        assert(lease->PublishSnapshot(conversation, lease->Capture(), false));
      const auto metrics = lease->Metrics();
      assert(metrics.deepest_live == cached);
      const auto json = ComponentCacheJson(metrics);
      assert(json.member_str("schema") == "component-cache-v1");
      assert(json.contains("captures") && json.contains("global_ram_bytes"));
      // The second turn continues the first turn's live frontier.
      assert(cached == (turn ? 12 : 0));
      (void)lease->Commit({}, {}, conversation);
      conversation.push_back(8);
    }
    assert(runner->committed == (enabled ? 1 : 0));
    assert(runner->prepared == (enabled ? 2 : 0));
    // The budget is measured from the runner's claim, which off never reads.
    assert((runner->claims != 0) == enabled);
    assert((runner->captures != 0) == enabled);
  }
}
// The model sampled "sk" "ý" and "Lie" "v"; the next prompt replays that
// answer as text and tokenizes it as "ský" and "L" "iev" (#1154). Only the
// live frontier holds the work, as with the RAM cache off.
void RetokenizedAnswerContinuesTheLiveFrontier() {
  // 50 stands for <|im_end|>; the template trims whitespace before it.
  static const std::map<TextRunnerToken, std::string> pieces{
      {20, "sk"},  {21, "\xC3\xBD"},  {22, "sk\xC3\xBD"}, {30, "L"},
      {31, "iev"}, {32, "Lie"},       {33, "v"},          {34, "iex"},
      {40, "."},   {41, ".\n\n"},     {42, "\n\n"},       {43, "\xE3\x80\x80"},
      {44, " "},   {50, "<|im_end|>"}};
  class TextRunner final : public Runner {
  public:
    std::vector<TextRunnerToken> Tokenize(
        std::string_view text) const override {
      for (const auto& [token, piece] : pieces)
        if (piece == text)
          return {token};
      return {};
    }
    std::string Decode(std::span<const TextRunnerToken> tokens) const override {
      std::string text;
      for (const auto token : tokens) {
        const auto found = pieces.find(token);
        text += found != pieces.end() ? found->second
                                      : "<" + std::to_string(token) + ">";
      }
      return text;
    }
  };
  auto runner = std::make_shared<TextRunner>();
  using Tokens = std::vector<TextRunnerToken>;
  // Commits `frontier`, whose reply the model generated from `reply` on.
  const auto commit = [&](ComponentTextCache& cache, const Tokens& frontier,
                          std::size_t reply) {
    const Tokens prompt(frontier.begin(), frontier.begin() + reply);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    lease->PreparePrefill(prompt.size());
    runner->adapter->Append(state.slot, frontier, frontier);
    (void)lease->Commit({}, {}, frontier);
  };
  // The prompt the cache gives `prompt` after `frontier`, or nothing.
  const auto align = [&](const Tokens& frontier, Tokens prompt,
                         std::size_t reply = 1,
                         bool trimmed = true) -> std::optional<Tokens> {
    ComponentTextCache cache(runner, 1, 0, {}, false, 0, false);
    commit(cache, frontier, reply);
    const auto original = prompt;
    std::size_t stable = 0;
    if (!cache.AlignToLiveFrontier(prompt, stable, 4096, trimmed)) {
      assert(prompt == original && stable == 0);
      return std::nullopt;
    }
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->cache_hit() && lease->cached_tokens() == frontier.size());
    lease->Invalidate();
    return prompt;
  };

  // finish=length on whitespace: the prompt trims it before <|im_end|>.
  assert((align({1, 40, 42}, {1, 40, 50, 6}) == Tokens{1, 40, 42, 50, 6}));
  assert(
      (align({1, 40, 44, 43}, {1, 40, 50, 6}) == Tokens{1, 40, 44, 43, 50, 6}));
  // finish=stop after whitespace, and whitespace merged into punctuation.
  assert((align({1, 40, 42, 50}, {1, 40, 50, 6}) == Tokens{1, 40, 42, 50, 6}));
  assert((align({1, 41}, {1, 40, 50, 6}) == Tokens{1, 41, 50, 6}));
  assert((align({1, 41, 50}, {1, 40, 50, 6}) == Tokens{1, 41, 50, 6}));
  // Leading whitespace of a reply.
  assert((align({1, 42, 32, 33, 50}, {1, 32, 33, 50, 6}) ==
          Tokens{1, 42, 32, 33, 50, 6}));
  // Cut off inside "ský": the rest of that token follows the frontier.
  assert((align({1, 2, 20}, {1, 2, 22, 50, 6}) == Tokens{1, 2, 20, 21, 50, 6}));
  // Whitespace is not dropped before text that differs, nor kept where the
  // prompt continues the frontier differently.
  assert(!align({1, 42, 5}, {1, 6}));
  assert(!align({1, 40, 42, 5}, {1, 40, 50, 6}));
  // Text is never dropped, only whitespace.
  assert(!align({1, 40, 33}, {1, 40, 50, 6}));
  // A fragment that does not tokenize back to itself ("ie" of "Lie").
  assert(!align({1, 30}, {1, 32, 50}));
  // Whitespace a client removed inside a reply, or from a prompt the model
  // did not answer, is an edit, as is any whitespace without a template.
  assert(!align({1, 40, 42, 32, 33, 50}, {1, 40, 32, 33, 50, 6}));
  assert(!align({1, 40, 42, 32, 33, 50}, {1, 40, 32, 33, 50, 6}, 4));
  assert(!align({1, 40, 42}, {1, 40, 50, 6}, 1, false));
  assert(!align({1, 42, 32, 33, 50}, {1, 32, 33, 50, 6}, 1, false));
  assert((align({1, 2, 20, 21, 50}, {1, 2, 22, 50, 6}, 1, false) ==
          Tokens{1, 2, 20, 21, 50, 6}));
  {
    // A later turn: the frontier keeps the whitespace an earlier one
    // restored, so the edges of that reply move with it.
    ComponentTextCache cache(runner, 1, 0, {}, false, 0, false);
    commit(cache, {1, 40, 42, 50}, 1);
    Tokens prompt{1, 40, 50, 6};
    std::size_t stable = 0;
    assert(cache.AlignToLiveFrontier(prompt, stable, 4096, true));
    assert((prompt == Tokens{1, 40, 42, 50, 6}));
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->cache_hit() && lease->cached_tokens() == 4);
    auto& state = dynamic_cast<State&>(lease->state());
    lease->PreparePrefill(prompt.size());
    const Tokens tail{6, 7};
    runner->adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, Tokens{1, 40, 42, 50, 6, 7});
    Tokens next{1, 40, 50, 6, 7, 8};
    assert(cache.AlignToLiveFrontier(next, stable, 4096, true));
    assert((next == Tokens{1, 40, 42, 50, 6, 7, 8}));
  }

  ComponentTextCache cache(runner, 1, 0, {}, false, 0, false);
  const Tokens sampled{1, 2, 3, 20, 21, 4, 32, 33, 5};
  commit(cache, sampled, 3);
  const auto unchanged = [&](Tokens prompt, std::size_t stable = 0,
                             std::size_t limit = 4096) {
    const auto original = prompt;
    const auto original_stable = stable;
    const bool aligned = cache.AlignToLiveFrontier(prompt, stable, limit, true);
    return !aligned && prompt == original && stable == original_stable;
  };
  // Already the frontier's own tokens.
  assert(unchanged({1, 2, 3, 20, 21, 4, 32, 33, 5, 6}));
  // Different text inside the answer.
  assert(unchanged({1, 2, 3, 22, 4, 30, 34, 5, 6}));
  // The prompt edits the frontier's last token.
  assert(unchanged({1, 2, 3, 22, 4, 30, 31, 9, 6}));
  // The prompt ends before the frontier's text does.
  assert(unchanged({1, 2, 3, 22, 4, 32}));
  // The stable boundary lies inside the replaced tokens.
  assert(unchanged({1, 2, 3, 22, 4, 30, 31, 5, 6, 7}, 4));
  // The aligned prompt would reach the context limit.
  assert(unchanged({1, 2, 3, 22, 4, 30, 31, 5, 6, 7}, 0, 11));

  Tokens prompt{1, 2, 3, 22, 4, 30, 31, 5, 6, 7};
  std::size_t stable = 8;
  assert(cache.AlignToLiveFrontier(prompt, stable, 4096, true));
  assert((prompt == Tokens{1, 2, 3, 20, 21, 4, 32, 33, 5, 6, 7}));
  assert(stable == 9);
  auto lease = cache.Acquire(prompt, {}, {}, true, stable, true);
  assert(lease->cache_hit() && lease->cached_tokens() == sampled.size());
  // A busy frontier is not offered.
  Tokens busy{1, 2, 3, 22, 4, 30, 31, 5, 6, 7};
  std::size_t busy_stable = 0;
  assert(!cache.AlignToLiveFrontier(busy, busy_stable, 4096, true));
  lease->Invalidate();
}
// Disk metadata admission is independent of the RAM and disk budgets.
// Account for a fully committed disk-capture pool without allocating its bytes.
void RamCacheOffAllowsLargeDiskIndexMetadata() {
  class AccountingRunner final : public Runner {
  public:
    mutable cache::ResourceLedger* ledger{};
    mutable cache::ResourceCharge backing;
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& resources, std::size_t ram) const override {
      ledger = &resources;
      auto result = Runner::CreateComponentCacheResources(resources, ram);
      result->commit_backing = [this](std::size_t bytes) {
        backing = ledger->Reserve(cache::ResourceCategory::kBackingFree, bytes)
                      .Convert();
      };
      return result;
    }
  };
  for (const std::size_t ram : {std::size_t{1}, std::size_t{2} << 30}) {
    const TemporaryDirectory temp("gufo-component-off-disk-metadata-test");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<AccountingRunner>();
    ComponentTextCache cache(runner, 1, ram, disk, false, 0, false);
    assert(runner->backing.Info().bytes == 512ULL << 20);
    assert(runner->ledger->Limits().staging_bytes ==
           cache::kDefaultDiskStagingBytes);
    const auto before = runner->ledger->Snapshot().ram_bytes;
    auto metadata = runner->ledger->Reserve(cache::ResourceCategory::kMetadata,
                                            128ULL << 20);
    assert(runner->ledger->Snapshot().ram_bytes == before + (128ULL << 20));
    auto payload =
        runner->backing.ReserveBacking(cache::ResourceCategory::kPrivateState)
            .Convert();
    auto pin = payload.PinPersistence();
    assert(runner->ledger->Snapshot().persistence_pinned_bytes == 512ULL << 20);
    auto extra =
        runner->ledger->Reserve(cache::ResourceCategory::kPrivateState, 1)
            .Convert();
    bool rejected = false;
    try {
      auto excess_pin = extra.PinPersistence();
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    assert(rejected);
  }
}
// With a disk cache the off RAM cache keeps no checkpoint in RAM and commits
// only the backing disk captures copy through.
void RamCacheOffWithDiskCommitsOnlyStagingBacking() {
  const TemporaryDirectory temp("gufo-component-off-disk-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  class CountingRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t bytes) {
        committed = bytes;
      };
      return resources;
    }
  };
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::uint64_t hash;
  {
    auto runner = std::make_shared<CountingRunner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk, false, 0, false);
    assert(runner->committed != 0 && runner->committed <= 512ULL << 20);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(!lease->TryReserveSnapshot(0, prompt.size(), false,
                                      SnapshotPurpose::kHistory, prompt));
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    // Nothing is retained in RAM; the checkpoint goes to disk alone.
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 0);
    (void)lease->Commit(prompt, {}, prompt);
  }
  auto runner = std::make_shared<CountingRunner>();
  ComponentTextCache cache(runner, 1, 1, disk, false, 0, false);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->restored_from_disk() &&
         lease->cached_tokens() == prompt.size());
  auto& state = dynamic_cast<State&>(lease->state());
  assert(runner->adapter->RecurrentHash(state.slot) == hash);
}
// A host that keeps less than its reserve after loading measures a ceiling
// below what the slots already pinned. With the RAM cache on, the load fails
// with the remedies instead of a bare bad_alloc. With it off, the server
// still loads, and the disk cache persists and restores.
void HostBelowReserveRefusesUnlessRamCacheOff() {
  constexpr std::size_t kPin = 512 << 10;
  const TemporaryDirectory temp("gufo-component-host-short-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  class ShortHostRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    mutable std::vector<cache::ResourceCharge> pins;
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      // After the slots pinned their buffers, the host has nothing left
      // above its reserve.
      claim.retained_snapshot_capacity_bytes = 0;
      claim.retained_snapshot_ceiling_bytes = 0;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->stream = [this, &ledger,
                           stream = std::move(resources->stream)] {
        if (!committed)
          pins.push_back(
              ledger.Reserve(cache::ResourceCategory::kMetadata, kPin)
                  .Convert());
        return stream();
      };
      resources->pinned_bytes = [this] { return pins.size() * kPin; };
      resources->commit_backing = [this](std::size_t bytes) {
        committed = bytes;
      };
      return resources;
    }
  };
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto refused = [](const auto& load) {
    try {
      load();
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      return message.find("host reserve") != std::string::npos &&
             message.find("--cache-ram-bytes off") != std::string::npos;
    }
    return false;
  };
  for (const bool enabled : {true, false}) {
    // Through the pool, as serving builds it. A host with nothing above its
    // reserve reports a ceiling of 0, also before the slots allocate; the
    // cache's own metadata must still fit to report it, or to serve off.
    const TemporaryDirectory pool_temp("gufo-component-host-zero-test");
    const TextRunnerDiskCacheOptions pool_disk{
        .directory = pool_temp.path().string(),
        .capacity_bytes = 1 << 20,
        .min_checkpoint_step_tokens = 1};
    const TextRunnerRamCacheOptions ram{.capacity_bytes = 1ULL << 30,
                                        .enabled = enabled};
    if (enabled) {
      assert(refused([&] {
        TextRunnerPool pool(std::make_shared<ShortHostRunner>(), 1, pool_disk,
                            ram);
      }));
      continue;
    }
    TextRunnerPool pool(std::make_shared<ShortHostRunner>(), 1, pool_disk, ram);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    assert(Generate(request) != 0);
  }
  // An explicit RAM budget, as in --cache-ram-bytes 1073741824.
  assert(refused([&] {
    ComponentTextCache cache(std::make_shared<ShortHostRunner>(), 1, 1 << 24,
                             disk, false, 0, true);
  }));
  // The same host serves with --cache-ram-bytes off.
  constexpr auto kOff = ComponentTextCache::kOffMetadataBytes;
  std::uint64_t hash;
  {
    auto runner = std::make_shared<ShortHostRunner>();
    ComponentTextCache cache(runner, 1, 0, disk, false, kOff, false);
    assert(!runner->pins.empty());
    // Only the disk-only backing of the off mode, no RAM cache budget.
    assert(runner->committed != 0 && runner->committed <= 512ULL << 20);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 0);
    (void)lease->Commit(prompt, {}, prompt);
  }
  auto runner = std::make_shared<ShortHostRunner>();
  ComponentTextCache cache(runner, 1, 0, disk, false, kOff, false);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->restored_from_disk() &&
         lease->cached_tokens() == prompt.size());
  auto& state = dynamic_cast<State&>(lease->state());
  assert(runner->adapter->RecurrentHash(state.slot) == hash);
}
// An explicit budget the host cannot hold is trimmed, and the cache reports
// what it actually got, which the pool logs as ram_cache_trimmed.
void TrimmedRamBudgetIsReported() {
  class TrimRunner final : public Runner {
  public:
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      claim.retained_snapshot_capacity_bytes = 1 << 20;
      claim.retained_snapshot_ceiling_bytes = 2 << 20;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [](std::size_t) {};
      return resources;
    }
  };
  ComponentTextCache trimmed(std::make_shared<TrimRunner>(), 1, 3 << 20, {},
                             false, 4 << 20, true);
  assert(trimmed.ram_budget() == 2 << 20);
  ComponentTextCache off(std::make_shared<TrimRunner>(), 1, 3 << 20, {}, false,
                         4 << 20, false);
  assert(off.ram_budget() == 0);
}
// With the RAM cache off, a capture within the disk step of a published
// checkpoint has nowhere to go. The cache refuses it at reservation, so no
// caller logs a failed capture. With the RAM cache on it goes to RAM.
void RamCacheOffRefusesCapturesWithinTheDiskStep() {
  for (const bool enabled : {true, false}) {
    const TemporaryDirectory temp("gufo-component-off-step-test");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, 1 << 22, disk, false, 0, enabled);
    std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
    for (int turn = 0; turn < 2; ++turn) {
      auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(prompt).subspan(lease->cached_tokens());
      runner->adapter->Append(state.slot, tail, tail);
      // The first turn has no checkpoint within the step. The second is nine
      // tokens past one, far inside the default 2048-token step.
      const auto reserved =
          lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                    SnapshotPurpose::kContinuation, prompt);
      assert(reserved == (turn == 0 || enabled));
      if (reserved) {
        // Only the RAM cache retains the capture; off sends it to disk alone.
        const auto retained =
            lease->PublishSnapshot(prompt, lease->Capture(), false);
        assert((retained != 0) == enabled);
      }
      (void)lease->Commit(prompt, {}, prompt);
      prompt.push_back(99);
    }
  }
}
// Inputs whose tokens must be evaluated together, such as images. A span is
// [first, end): positions strictly inside it are not legal stops.
struct SpanContext final : TextPromptContext {
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  std::size_t PrefillStop(std::size_t position, bool up) const override {
    for (const auto& [first, end] : spans)
      if (position > first && position < end)
        return up ? end : first;
    return position;
  }
};
std::shared_ptr<SpanContext> Spans(
    std::vector<std::pair<std::size_t, std::size_t>> spans) {
  auto context = std::make_shared<SpanContext>();
  context->spans = std::move(spans);
  return context;
}
std::vector<TextRunnerToken> SpanPrompt(std::size_t tokens) {
  std::vector<TextRunnerToken> prompt(tokens);
  std::iota(prompt.begin(), prompt.end(), 10);
  return prompt;
}
void IndivisibleInputsAreNotSplit() {
  using Stops = std::vector<std::size_t>;
  using Context = std::shared_ptr<const TextPromptContext>;
  // Prefill in steps of `budget` and report where the steps ended.
  const auto steps = [](std::size_t tokens, std::size_t budget, Context context,
                        std::size_t alignment = 1, std::size_t stable = 0) {
    auto runner = std::make_shared<Runner>();
    runner->alignment = alignment;
    TextRunnerPool pool(runner, 1);
    auto request = pool.Acquire(SpanPrompt(tokens), sampling::SamplingConfig{},
                                {}, std::move(context), true, stable);
    while (!request.prefill_complete())
      (void)request.Prefill(budget);
    request.Invalidate();
    return runner->pass_ends;
  };
  // A budget stops before an input, and a step starting at one takes it whole.
  assert(steps(20, 3, Spans({{4, 14}})) == Stops({3, 4, 14, 17, 20}));
  assert(steps(20, 3, Spans({{4, 10}, {10, 16}})) ==
         Stops({3, 4, 10, 16, 19, 20}));
  assert(steps(12, 3, Spans({{0, 6}})) == Stops({6, 9, 12}));
  assert(steps(12, 3, Spans({{8, 12}})) == Stops({3, 6, 8, 12}));
  // A budget ending exactly at an input's edges is not moved.
  assert(steps(20, 3, Spans({{5, 8}})) == Stops({3, 5, 8, 11, 14, 17, 20}));
  // No declared spans change no step.
  assert(steps(20, 3, {}) == Stops({3, 6, 9, 12, 15, 18, 20}));
  assert(steps(20, 3, Spans({})) == steps(20, 3, {}));
  // With an alignment a stop is both aligned and outside every input; only
  // the prompt end may be off the alignment.
  assert(steps(42, 6, Spans({{10, 19}}), 4) ==
         Stops({4, 8, 20, 24, 28, 32, 36, 40, 42}));
  assert(steps(42, 64, Spans({{22, 29}}), 4, 30) == Stops({20, 42}));
  assert(steps(42, 64, Spans({}), 4, 30) == Stops({28, 42}));
  // A planned capture inside an input moves before it.
  const auto planned = steps(3000, 4096, {});
  assert(planned.size() > 1);
  const auto capture = planned.front();
  const auto moved = steps(3000, 4096, Spans({{capture - 8, capture + 8}}));
  assert(moved.front() == capture - 8);
  assert(steps(3000, 4096, Spans({{capture, capture + 8}})).front() == capture);
  // The stable boundary lies inside the input: its checkpoint moves before it.
  for (const bool in_pass : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->in_pass = in_pass;
    runner->max_in_pass_tail = 16;
    TextRunnerPool pool(runner, 1);
    const auto prompt = SpanPrompt(20);
    const auto spans = Spans({{4, 14}});
    auto first =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, spans, true, 10);
    while (!first.prefill_complete())
      (void)first.Prefill(64);
    assert(runner->pass_ends == (in_pass ? Stops({20}) : Stops({4, 20})));
    assert(runner->in_pass_points == (in_pass ? Stops({4}) : Stops{}));
    (void)first.Commit();
    auto edited = prompt;
    edited[12] += 1000;
    auto second =
        pool.Acquire(edited, sampling::SamplingConfig{}, {}, spans, true, 10);
    assert(second.cached_prompt_tokens() == 4);
    second.Invalidate();
  }
  // A peer's shared prefix ending inside an input stops before the input.
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto leader = pool.Acquire(SpanPrompt(1600), sampling::SamplingConfig{}, {},
                               Spans({{900, 1100}}));
    assert(leader.ShareCheckpoint(1000) == 900);
    while (!leader.prefill_complete())
      (void)leader.Prefill(4096);
    assert(runner->pass_ends.front() == 900);
    leader.Invalidate();
  }
  // A checkpoint or live frontier from a prompt without the input cannot
  // resume inside it, even with the same tokens.
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    const auto prompt = SpanPrompt(20);
    auto first =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 10);
    Prefill(first);
    (void)first.Commit();
    auto edited = prompt;
    edited[15] += 1000;
    auto plain =
        pool.Acquire(edited, sampling::SamplingConfig{}, {}, {}, true, 0);
    assert(plain.cached_prompt_tokens() == 10);
    plain.Invalidate();
    auto spanned = pool.Acquire(edited, sampling::SamplingConfig{}, {},
                                Spans({{4, 14}}), true, 0);
    assert(spanned.cached_prompt_tokens() == 0);
    spanned.Invalidate();
  }
  for (const bool spanned : {false, true}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto history = SpanPrompt(10);
    auto first = pool.Acquire(history);
    Prefill(first);
    history.push_back(Generate(first));
    (void)first.Commit();
    history.resize(30, 7);
    Context context;
    if (spanned)
      context = Spans({{4, 14}});
    auto next =
        pool.Acquire(history, sampling::SamplingConfig{}, {}, context, true, 0);
    assert(next.cached_prompt_tokens() == (spanned ? 0 : 11));
    next.Invalidate();
  }
  // A runner stopping inside an input fails the request.
  {
    class ShortStepRunner final : public Runner {
    public:
      TextPrefillStep Prefill(TextRunnerState& state,
                              std::span<const TextRunnerToken> prompt,
                              std::size_t offset,
                              std::size_t budget) const override {
        return Runner::Prefill(state, prompt, offset,
                               std::min<std::size_t>(budget, 6));
      }
    };
    TextRunnerPool pool(std::make_shared<ShortStepRunner>(), 1);
    auto request = pool.Acquire(SpanPrompt(20), sampling::SamplingConfig{}, {},
                                Spans({{4, 14}}));
    bool failed = false;
    try {
      (void)request.Prefill(64);
    } catch (const std::runtime_error&) {
      failed = true;
    }
    assert(failed);
    request.Invalidate();
  }
  // A hook moving a stop the wrong way fails instead of hanging.
  {
    struct WrongWayContext final : TextPromptContext {
      std::size_t PrefillStop(std::size_t position, bool) const override {
        return position > 4 && position < 14 ? 4 : position;
      }
    };
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1);
    auto request = pool.Acquire(SpanPrompt(20), sampling::SamplingConfig{}, {},
                                std::make_shared<WrongWayContext>());
    bool failed = false;
    try {
      while (!request.prefill_complete())
        (void)request.Prefill(3);
    } catch (const std::logic_error&) {
      failed = true;
    }
    assert(failed);
    request.Invalidate();
  }
}
int main() {
  RamCacheOffAllowsLargeDiskIndexMetadata();
  IndivisibleInputsAreNotSplit();
  PromptReuseCanBeDisabledPerRequest();
  StableChatPrefixSurvivesInterruptedFraming();
  WarmTurnsAdvanceTheStableCheckpoint();
  FullChatCheckpointRestoresWithoutSuffixPrefill();
  GeneratedFrontierForksBeforeMutation();
  InterruptedPrefillRetainsOnlyItsEntryFrontier();
  PublishedResidentCheckpointKeepsDiskStep();
  CancelledPrefillKeepsCompletedSteps();
  SharedPeerReusesTheCurrentStableCapture();
  BackingPreparationFollowsCaptures();
  PrefillLookaheadReachesTheNextFrontier();
  PrefillStepsSeeTheWholePromptEnd();
  PartialFrontierDoesNotServeAnEqualPrompt();
  RefusedFrontierServesFromShallowerBoundary();
  CompletedCancelledPromptGetsARetrySnapshot();
  BudgetUsesAllocatedSlotsInsteadOfLazyStateClaim();
  InPassCapturePreservesBoundaryAndLiveTail();
  AlignedPrefillKeepsAlignedBoundaries();
  UnalignedPromptEndPublishesBeforeDroppingTheFrontier();
  UnalignedRawPromptKeepsAnAlignedContinuation();
  DisabledCacheDoesNotPrepareARawPromptCheckpoint();
  GridCheckpointsCaptureInPass();
  PromptTargetBeforeGridKeepsItsPurpose();
  ColdAcquireResetsOnce();
  ColdAssignmentAvoidsDrainingCancelledSlot();
  ColdAssignmentAvoidsDrainingCancelledSlot(true);
  ColdAssignmentKeepsOneDuplicateDeepFrontier();
  FailedRestoreResetsOnce();
  RestoreFromBusyBorrowedSource();
  ImageAfterCachedTextPrefixCaptures();
  ColdAssignmentPrefersPreservedSlot();
  ColdAssignmentKeepsLiveFrontiers();
  ColdSideRequestsKeepPreservedDeepFrontier();
  ReuseAndBranches();
  EditBoundariesSurviveActualBackingPressure();
  ReusedConversationOutlivesOneUseRequests();
  RestartAndCorruption();
  FailedCaptureKeepsPublishedBoundary();
  BackingPressureKeepsSharedPrefix();
  BackingPressureKeepsSharedPrefix(true);
  FragmentedBackingKeepsCompletePrompt();
  DiskBudgetEvictsAnUnpinnedConversation();
  LoweredDiskBudgetEvictsOldestAtStartup();
  UnpersistedRequestStaysOffDisk();
  UnpersistedOverRamCaptureIsRefused();
  OrphansLeaveBeforeCheckpoints();
  StagingIsInternal();
  StagingCountsAgainstRamCeiling();
  PinnedSlotBytesCountOnce();
  UnpinnedRunnersKeepTheirLimit();
  DiskDirectoryHasOneOwner();
  GridDoesNotSuppressFirstDiskPrompt();
  NearTailDiskBranchSurvivesRestart();
  UnbackedDeepRowsSurviveDiskRestart();
  // Without private state is the shape of a model with only attention.
  for (const bool private_state : {true, false}) {
    WindowRowsReuseAndBranches(private_state);
    WindowedDeepRowsSurviveDiskRestart(private_state);
  }
  OverRamCaptureStagesPrivateStateUnderPressure(false);
  OverRamCaptureStagesPrivateStateUnderPressure(true);
  RamCacheOffCommitsNoBackingAndKeepsLiveFrontiers();
  RetokenizedAnswerContinuesTheLiveFrontier();
  RamCacheOffWithDiskCommitsOnlyStagingBacking();
  HostBelowReserveRefusesUnlessRamCacheOff();
  TrimmedRamBudgetIsReported();
  RamCacheOffRefusesCapturesWithinTheDiskStep();
}
