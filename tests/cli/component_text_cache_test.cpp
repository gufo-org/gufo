#include "src/cli/serve/component_text_cache.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <numeric>

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
class Runner final : public TextModelRunner {
public:
  mutable FakeAdapter* adapter{};
  mutable std::size_t captures{};
  bool UsesComponentCache() const override { return true; }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "component-fake",
            .state_abi = "component-fake-v1",
            .max_context = 4096,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = false,
                             .prefix_reuse = true}};
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
      std::shared_ptr<std::array<std::byte, 64>> data;
    };
    auto blocks = std::make_shared<std::vector<Block>>();
    // Physical backing commits once, before any request; assigned owner tokens
    // account for the entire block including unused bytes.
    for (std::size_t i = 0; i < 2048; ++i) {
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, 64);
      auto data = std::make_shared<std::array<std::byte, 64>>();
      blocks->push_back({reservation.Convert(), std::move(data)});
    }
    auto resources = std::make_unique<ComponentCacheResources>();
    resources->adapter = std::make_unique<FakeAdapter>();
    adapter = dynamic_cast<FakeAdapter*>(resources->adapter.get());
    resources->owner = blocks;
    resources->stream = [] { return std::make_unique<FakeStream>(); };
    resources->allocate = [this, blocks](const cache::PayloadRequest& request) {
      for (const auto& block : *blocks) {
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          ++captures;
          auto owner =
              std::shared_ptr<const void>(block.data, block.data->data());
          return ComponentCaptureBuffer{*block.data, [reservation, owner] {
                                          return cache::Payload::Committed(
                                              reservation->Convert(), owner);
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
void Prefill(TextRunnerPool::Request& request) {
  while (!request.prefill_complete())
    (void)request.Prefill(8);
}
TextRunnerToken Generate(TextRunnerPool::Request& request) {
  const auto token = request.SelectNext().token;
  request.Advance();
  return token;
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
}  // namespace
int main() {
  ReuseAndBranches();
  RestartAndCorruption();
  DiskBudgetEvictsAnUnpinnedConversation();
}
