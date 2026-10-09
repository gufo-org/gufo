#include "src/cache/retention.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>

#include "tests/cache/fake_adapter.hpp"
#include "tests/cache/prefix_index_fixture.hpp"
using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {
struct Log final : RetentionEventSink {
  std::vector<RetentionEvent> events;
  void Emit(const RetentionEvent& e) noexcept override { events.push_back(e); }
};
struct Fixture {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  Log log;
  RetentionPolicy policy;
  explicit Fixture(std::size_t limit = 128)
      : policy(ledger, index, &log, limit) {
    index.Register({1}, kIndexComponents);
  }
  CheckpointId Add(std::vector<Token> tokens,
                   RetentionPurpose purpose = RetentionPurpose::kContinuation,
                   Rows stable = 0, CheckpointId source = {},
                   bool preserve = true, Identity input = {}) {
    assert(policy.Admit(
        {tokens, {1}, input, purpose, stable, source, preserve, {}},
        [&] { return IndexCheckpoint(ledger, tokens, {1}, input); }));
    return log.events.back().checkpoint;
  }
  Rows Cached(std::vector<Token> tokens, Identity input = {}) {
    return index.CachedPrefixTokens(
        {{1}, tokens, InputIdentity(tokens.size(), input), 0, true});
  }
};
void Planning() {
  FakeAdapter adapter;
  struct Empty final : MutationGuard {
    void BeforeOverwrite(ComponentId, Rows, Rows) override {}
    void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
  } guard;
  auto slot = adapter.CreateSlot(guard);
  auto plan = PlanCaptures(adapter, *slot, {10000, 0, 1500, 0, true, true, {}});
  std::vector<Rows> expected{1500, 2048, 4096, 6144, 8192, 10000};
  assert(plan.size() == expected.size());
  for (std::size_t i = 0; i < plan.size(); ++i) {
    assert(plan[i].boundary == expected[i]);
    assert(plan[i].splits_pass == (plan[i].boundary == 1500));
  }
  plan = PlanCaptures(adapter, *slot, {2176, 0, 0, 0, true, true, {}});
  assert(plan.size() == 1 && plan[0].boundary == 2176);
  plan = PlanCaptures(adapter, *slot, {2177, 0, 0, 0, true, true, {}});
  assert(plan.size() == 2 && plan[0].boundary == 2048);
  plan = PlanCaptures(adapter, *slot, {5000, 4000, 4500, 4511, true, true, {}});
  assert(plan.size() == 3);  // less than 512 improvement: no learned point/grid
  plan = PlanCaptures(adapter, *slot, {6000, 4000, 4500, 4512, true, true, {}});
  assert(plan.size() == 3 && plan[1].boundary == 4500 && plan[1].required);
  plan = PlanCaptures(adapter, *slot, {6000, 4000, 4500, 4600, true, true, {}});
  assert(plan.size() == 4 && plan[2].purpose == CheckpointPurpose::kLearned);
  adapter.SetPrefillPassRows(2000);
  plan = PlanCaptures(adapter, *slot, {10000, 0, 1501, 0, true, true, {}});
  assert(plan[0].boundary == 1501 && plan[0].required && plan[0].splits_pass);
  assert(plan[1].boundary == 2048 && plan[1].splits_pass);
  assert(plan[2].boundary == 4000 && !plan[2].splits_pass);
  assert(PlanCaptures(adapter, *slot, {}).empty());
  std::array<Rows, 1> retained{2048};
  plan = PlanCaptures(adapter, *slot, {4096, 0, 0, 0, true, false, retained});
  assert(plan.size() == 1 && plan[0].boundary == 4096);
}
void Branches() {
  for (bool branched : {false, true}) {
    Fixture f(4);
    f.Add({9, 9, 9});
    const auto root = f.Add({1, 2, 3});
    f.Add({1, 2, 3, 4, 4});
    if (branched)
      f.Add({1, 2, 3, 5, 5});
    else
      f.Add({7, 7});
    assert(f.policy.Rank(root) == (branched ? 3 : 2));
    f.Add({8, 8, 8, 8});
    assert(f.Cached({1, 2, 3, 6}) == (branched ? 3 : 0));
    assert(f.Cached({9, 9, 9, 1}) == (branched ? 0 : 3));
  }
  {
    Fixture f(8);
    f.Add({9, 9, 9});
    f.Add({7, 7, 7});
    const auto learned = f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3, 4, 4});
    std::vector<Token> incoming{1, 2, 3, 5, 5};
    int attempt = 0;
    assert(f.policy.Admit(
        {incoming, {1}, {}, RetentionPurpose::kContinuation, 0, learned, false},
        [&] {
          if (!attempt++)
            throw ResourceExhausted();
          return IndexCheckpoint(f.ledger, incoming);
        }));
    assert(f.Cached({1, 2, 3, 6}) == 3 && f.Cached({9, 9, 9, 1}) == 0);
  }
  {
    Fixture f(4);
    f.Add({9, 9, 9});
    const auto shallow = f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    const auto deep = f.Add({1, 2, 3, 4, 5}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3, 4, 5, 6});
    assert(f.policy.Rank(shallow) == 2 && f.policy.Rank(deep) == 3);
    f.Add({8, 8, 8});
    assert(f.Cached({1, 2, 3, 7}) == 0 && f.Cached({1, 2, 3, 4, 5, 7}) == 5);
  }
  {
    Fixture f(2);
    f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3});  // exact replacement retains learned status
    const auto id = f.log.events.back().checkpoint;
    f.Add({1, 2, 3, 4});
    assert(f.policy.Rank(id) == 3);
  }
}
void RanksAndSources() {
  Fixture f(4);
  auto root = f.Add({1, 2});
  auto retry = f.Add({1, 2, 3}, RetentionPurpose::kRetry, 2);
  auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
  auto tail = f.Add({1, 2, 3, 4, 5});
  assert(f.policy.Rank(retry) == 0 && f.policy.Rank(history) == 1 &&
         f.policy.Rank(root) == 2 && f.policy.Rank(tail) == 3);
  f.Add({7, 7});
  assert(f.log.events[f.log.events.size() - 2].checkpoint == retry);
  f.Add({8, 8});
  assert(f.log.events[f.log.events.size() - 2].checkpoint == history);
  for (const bool preserve : {true, false}) {
    Fixture g(2);
    const auto other = g.Add({9, 9});
    const auto source = g.Add({1, 2});
    g.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, source, preserve);
    assert(g.Cached({9, 9, 1}) == 2 &&
           g.log.events[g.log.events.size() - 2].checkpoint == source);
    g.policy.Touch(other);
    g.Add({7, 7});
    assert(g.Cached({9, 9, 1}) == 2);  // touch advances LRU
  }
  {
    Fixture g(3);
    const auto root_id = g.Add({1, 2});
    g.Add({9, 9});
    const auto old_tail = g.Add({1, 2, 3, 4});
    g.Add({1, 2, 5}, RetentionPurpose::kContinuation, 0, root_id);
    assert(g.log.events[g.log.events.size() - 2].checkpoint == old_tail);
    assert(g.log.events[g.log.events.size() - 2].reason ==
           RetentionReason::kEditedTail);
  }
  {
    Fixture g(1);
    g.Add({1, 2});
    std::vector<Token> t{9};
    bool called = false;
    assert(!g.policy.Admit({t, {1}, {}, RetentionPurpose::kHistory}, [&] {
      called = true;
      return IndexCheckpoint(g.ledger, t);
    }));
    assert(!called && g.policy.Size() == 1);
  }
  {
    Fixture g(2);
    auto a = g.Add({1, 2}, RetentionPurpose::kContinuation, 0, {}, true, {1});
    g.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, {}, true, {2});
    assert(g.policy.Rank(a) == 3);  // image identities never cover each other
  }
}
void RecordSourceReplacement() {
  for (const bool preserve : {true, false}) {
    Fixture f(1);
    const auto source = f.Add({1, 2});
    f.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, source, preserve);
    assert(f.Cached({1, 2, 3, 4}) == 3);
    assert(f.log.events[f.log.events.size() - 2].checkpoint == source);
  }
  // The byte-pressure flag remains effective while capture is incomplete.
  Fixture f(4);
  const auto other = f.Add({9, 9});
  const auto source = f.Add({1, 2});
  std::vector<Token> incoming{1, 2, 3};
  int attempt = 0;
  assert(f.policy.Admit(
      {incoming, {1}, {}, RetentionPurpose::kContinuation, 0, source, true},
      [&] {
        if (!attempt++)
          throw ResourceExhausted();
        return IndexCheckpoint(f.ledger, incoming);
      }));
  assert(f.log.events[f.log.events.size() - 2].checkpoint == other);
  assert(f.policy.Rank(source) == 2);
}
void FullOptionalExactReplacement() {
  for (const auto purpose :
       {RetentionPurpose::kRetry, RetentionPurpose::kHistory}) {
    for (const std::size_t limit : {1U, 2U}) {
      Fixture f(limit);
      const std::vector<Token> root{1, 2};
      const auto id = f.Add(root);
      if (limit == 2)
        f.Add({9, 9});
      bool called = false;
      assert(!f.policy.Admit({root, {1}, {}, purpose, 0, id}, [&] {
        called = true;
        return IndexCheckpoint(f.ledger, root);
      }));
      assert(!called && f.policy.Size() == limit && f.Cached({1, 2, 5}) == 2);
      assert(f.log.events.back().reason == RetentionReason::kRecordCapacity);
    }
  }
  {
    Fixture f(3);
    const std::vector<Token> root{1, 2};
    const auto id = f.Add(root);
    f.Add({1, 2, 3});
    f.Add({1, 2, 4});
    f.policy.Touch(
        id);  // mirrors Acquire's use before its optional reservation
    assert(f.policy.Rank(id) == 3);
    assert(!f.policy.Admit({root, {1}, {}, RetentionPurpose::kHistory, 0, id},
                           [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.policy.Rank(id) == 3);
    f.Add({9, 9});
    assert(f.Cached({1, 2, 5}) == 2);
  }
  {
    Fixture f(1);
    const std::vector<Token> root{1, 2};
    const auto id = f.Add(root, RetentionPurpose::kBranchPoint);
    assert(!f.policy.Admit({root, {1}, {}, RetentionPurpose::kRetry, 0, id},
                           [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.log.events.back().purpose == RetentionPurpose::kRetry &&
           f.log.events.back().rank == 0);
  }
  {
    // The original guard permits an optional exact replacement if another
    // eligible record exists; an exact learned point must stay learned.
    Fixture f(3);
    const auto id = f.Add({1, 2}, RetentionPurpose::kBranchPoint);
    f.Add({7}, RetentionPurpose::kHistory);
    f.Add({7, 8});
    const std::vector<Token> root{1, 2};
    assert(f.policy.Admit({root, {1}, {}, RetentionPurpose::kHistory, 0, id},
                          [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.policy.Rank(f.log.events.back().checkpoint) == 3);
    assert(f.policy.Size() == 3);
  }
}
void FailureAndReplay() {
  ResourceLedger ledger{kIndexLimits};
  {
    PrefixIndex index{ledger};
    index.Register({1}, kIndexComponents);
    Log log;
    {
      RetentionPolicy policy(ledger, index, &log, 2);
      std::vector<Token> t{1, 2, 3};
      std::stop_source stop;
      stop.request_stop();
      bool called = false;
      assert(!policy.Admit({t,
                            {1},
                            {},
                            RetentionPurpose::kContinuation,
                            0,
                            {},
                            true,
                            stop.get_token()},
                           [&] {
                             called = true;
                             return IndexCheckpoint(ledger, t);
                           }));
      assert(!called);
      assert(!policy.Admit({t, {1}, {}},
                           [&]() -> std::shared_ptr<const Checkpoint> {
                             throw std::runtime_error("transfer");
                           }));
      assert(policy.Size() == 0);
      assert(policy.Admit({t, {1}, {}},
                          [&] { return IndexCheckpoint(ledger, t); }));
      const auto id = log.events.back().checkpoint;
      auto pin = index.Lookup({{1}, t, InputIdentity(t.size())});
      const auto before = ledger.Snapshot().total_bytes;
      policy.Remove(id);
      assert(pin.selected && pin.selected->checkpoint->IsValid());
      assert(log.events.back().unique_bytes_freed ==
             before - ledger.Snapshot().total_bytes);
      assert(log.events.back().unique_bytes_freed <
             pin.selected->checkpoint->MetadataBytes());
      pin = {};
      assert(policy.Admit({t, {1}, {}},
                          [&] { return IndexCheckpoint(ledger, t); }));
      policy.Touch(log.events.back().checkpoint);
    }
    std::map<std::uint64_t, std::uint64_t> retained;
    std::uint64_t sequence = 0;
    for (const auto& e : log.events) {
      assert(e.sequence == ++sequence);
      if (e.action == RetentionAction::kAdmitted)
        assert(retained.emplace(e.checkpoint.value, e.last_used).second);
      if (e.action == RetentionAction::kTouched) {
        assert(retained.contains(e.checkpoint.value));
        retained[e.checkpoint.value] = e.last_used;
      }
      if (e.action == RetentionAction::kRemoved)
        assert(retained.erase(e.checkpoint.value) == 1);
      assert(retained.size() == e.retained_records);
    }
    assert(retained.empty());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void ImpossibleAndAllocationFailure() {
  Fixture f(1);
  const auto id = f.Add({1, 2});
  std::vector<Token> incoming{9};
  RetentionRequest r{incoming, {1}, {}};
  r.new_payload_bytes = kIndexLimits.total_bytes + 1;
  bool called = false;
  assert(!f.policy.Admit(r, [&] {
    called = true;
    return IndexCheckpoint(f.ledger, incoming);
  }));
  assert(!called && f.policy.Size() == 1 && f.Cached({1, 2, 3}) == 2);
  r.new_payload_bytes = 32;
  assert(!f.policy.Admit(r, []() -> std::shared_ptr<const Checkpoint> {
    throw std::bad_alloc();
  }));
  assert(f.policy.Size() == 1 && f.policy.Rank(id) == 3);
  assert(f.log.events.back().reason == RetentionReason::kCaptureFailure);
}
void CapacityAndSharing() {
  // Exact runtime payloads through the fake adapter; shared full chunks remain
  // identical and state/tails are fresh copies. Delayed copies finish before
  // the next Append changes recurrent state.
  ResourceLedger ledger{kIndexLimits};
  {
    FakeAdapter adapter;
    FakeStream stream{true};
    LeasedSlot slot(ledger, adapter, stream, {1});
    auto lease = slot.Acquire();
    PrefixIndex index{ledger};
    index.Register(adapter.CompatibilityIdentity(), adapter.Components());
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    auto history = ExecutionHistory::Cold(ledger, adapter.Components(),
                                          adapter.CompatibilityIdentity());
    std::vector<Token> tokens;
    std::size_t copies = 0;
    auto capture = [&] {
      return history.Capture(
          {tokens, InputIdentity(tokens.size()),
           adapter.Positions(lease.Execution()), CheckpointPurpose::kPrompt, 3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            auto reservation = pool.ReserveBacking(r.category);
            auto bytes = std::make_shared<std::vector<std::byte>>(r.bytes);
            auto copy =
                r.category == ResourceCategory::kPrivateState
                    ? adapter.CapturePrivate(lease.Execution(), r.component,
                                             *bytes, stream)
                    : adapter.CopyRowsOut(lease.Execution(), r.component,
                                          r.first, r.end, *bytes, stream);
            assert(copy.Wait() == TransferResult::kSucceeded);
            ++copies;
            return Payload::Committed(reservation.Convert(), bytes);
          });
    };
    for (int turn = 0; turn < 2; ++turn) {
      std::vector<Token> more{1, 2, 3, 4};
      adapter.Append(lease.Execution(), more, more);
      tokens.insert(tokens.end(), more.begin(), more.end());
      assert(
          policy.Admit({tokens, adapter.CompatibilityIdentity(), {}}, capture));
    }
    assert(copies == 6);  // target+draft chunks and recurrent state each turn
    auto hits = index.Lookup({adapter.CompatibilityIdentity(), tokens,
                              InputIdentity(tokens.size())});
    assert(hits.candidates.size() == 2);
    assert(hits.candidates[0].checkpoint->Components()[0].chunks[0].Id() ==
           hits.candidates[1].checkpoint->Components()[0].chunks[0].Id());
  }
  assert(ledger.Snapshot().total_bytes == 0);
  // A simulated byte-admission failure chooses the same rank/LRU victim and
  // retries only at the same boundary. Pinned storage frees no payload bytes.
  Fixture f(4);
  const auto first = f.Add({1, 2});
  f.Add({9, 9});
  std::vector<Token> t{7, 7};
  int attempts = 0;
  assert(f.policy.Admit({t, {1}, {}}, [&] {
    if (!attempts++)
      throw ResourceExhausted();
    return IndexCheckpoint(f.ledger, t);
  }));
  assert(attempts == 2 &&
         f.log.events[f.log.events.size() - 2].checkpoint == first);
  assert(f.log.events[f.log.events.size() - 2].reason ==
         RetentionReason::kByteCapacity);
}
void InvalidInventoriesPreserveReplacement() {
  for (int fault = 0; fault < 5; ++fault) {
    Fixture f(1);
    const std::vector<Token> tokens{1, 2, 3, 4};
    const auto original = f.Add(tokens);
    auto components = std::vector<ComponentDescriptor>(kIndexComponents.begin(),
                                                       kIndexComponents.end());
    if (fault == 0)
      components.resize(1);
    if (fault == 1)
      ++components[0].layout_version;
    if (fault == 2)
      ++components[0].row_bytes;
    if (fault == 3)
      ++components[0].rows_per_chunk;
    const auto before = f.ledger.Snapshot().total_bytes;
    assert(!f.policy.Admit({tokens, {1}, {}}, [&] {
      return IndexCheckpoint(
          f.ledger, tokens, {1}, {}, components,
          fault == 4 ? std::optional<Rows>{3} : std::nullopt);
    }));
    assert(f.policy.Size() == 1 && f.Cached(tokens) == tokens.size());
    assert(f.policy.Rank(original) == 3);
    assert(f.ledger.Snapshot().total_bytes == before);
    assert(f.log.events.back().action == RetentionAction::kRefused &&
           f.log.events.back().reason == RetentionReason::kInvalid);
    assert(f.log.events.back().ledger_bytes == before);
  }
}
void RealCapacityAndUniqueAdmission() {
  ResourceLedger ledger({32768, 32768, 0, 0});
  {
    constexpr std::array<ComponentDescriptor, 2> components{
        {{{1}, 1, ComponentKind::kAppendRows, 1024, 4, 0},
         {{2}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    PrefixIndex index(ledger);
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    auto history = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<Token> tokens;
    for (std::size_t end = 4; end <= 12; end += 4) {
      tokens.resize(end, 1);
      std::vector<ComponentPosition> positions{{{1}, end}, {{2}, end}};
      auto bytes = history.NewPayloadBytes(positions);
      assert(bytes == 4096 + 32);
      RetentionRequest request{tokens, {1}, {}};
      request.new_payload_bytes = bytes;
      assert(policy.Admit(request, [&] {
        return history.Capture(
            {tokens, InputIdentity(end), positions, CheckpointPurpose::kPrompt,
             3},
            [&](const PayloadRequest& r) {
              auto pool =
                  ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                      .Convert();
              return Payload::Committed(
                  pool.ReserveBacking(r.category).Convert(),
                  std::make_shared<int>(0));
            });
      }));
    }
    assert(policy.Size() == 3);
    assert(policy.RetainedPayloadBytes() == 12 * 1024 + 3 * 32);
    assert(ledger.Snapshot().total_bytes <= 32768);
    // An independent large branch needs real reclamation, with no simulated
    // pressure exception. Full payload exceeds the old available capacity.
    tokens.assign(16, 9);
    auto cold = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<ComponentPosition> positions{{{1}, 16}, {{2}, 16}};
    RetentionRequest request{tokens, {1}, {}};
    request.new_payload_bytes = cold.NewPayloadBytes(positions);
    assert(policy.Admit(request, [&] {
      return cold.Capture(
          {tokens, InputIdentity(tokens.size()), positions,
           CheckpointPurpose::kPrompt, 3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                                      std::make_shared<int>(0));
          });
    }));
    assert(std::ranges::any_of(log.events, [](const auto& e) {
      return e.action == RetentionAction::kRemoved &&
             e.reason == RetentionReason::kByteCapacity;
    }));
    assert(ledger.Snapshot().total_bytes <= 32768);
  }
  assert(ledger.Snapshot().total_bytes == 0);
  {
    Fixture f;
    for (Token token = 1; token <= 129; ++token)
      f.Add({token});
    assert(f.policy.Size() == 128 && f.Cached({1}) == 0 &&
           f.Cached({129}) == 1);
  }
}
}  // namespace
int main() {
  Planning();
  Branches();
  RanksAndSources();
  RecordSourceReplacement();
  FullOptionalExactReplacement();
  FailureAndReplay();
  CapacityAndSharing();
  ImpossibleAndAllocationFailure();
  RealCapacityAndUniqueAdmission();
  InvalidInventoriesPreserveReplacement();
}
