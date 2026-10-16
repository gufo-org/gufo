#include "src/cache/retention.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
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
  // A small budget keeps only the shallowest grid; a retained one still counts.
  adapter.SetPrefillPassRows(2048);
  plan = PlanCaptures(adapter, *slot,
                      {10000, 0, 1500, 0, true, false, {}, std::size_t{1}});
  assert(plan.size() == 3 && plan[1].boundary == 2048 &&
         plan[1].purpose == CheckpointPurpose::kGrid);
  plan = PlanCaptures(adapter, *slot,
                      {10000, 0, 1500, 0, true, false, retained, 1});
  assert(plan.size() == 2 && plan[1].boundary == 10000);
  assert(PlanCaptures(adapter, *slot, {10000, 0, 1500, 0, true, false, {}, 0})
             .size() == 2);
  // Flash-Next keeps about 133 MB of private state per checkpoint.
  assert(GridLimit(2ULL << 30, 133'100'000) == 1);
  assert(GridLimit(8ULL << 30, 133'100'000) == 4);
  assert(GridLimit(1ULL << 30, 133'100'000) == 0);
  assert(GridLimit(1ULL << 30, 0) == 4);
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
void LongPrefixRelationsAfterReplacement() {
  Fixture f(4);
  const std::vector<Token> prefix(131072, 7);
  auto left = prefix, right = prefix;
  left.push_back(8);
  right.push_back(9);
  const auto root = f.Add(prefix);
  const auto a = f.Add(left);
  const auto b = f.Add(right);
  assert(f.policy.Rank(root) == 3);  // both continuations need their fork
  f.policy.Remove(a);
  assert(f.policy.Rank(root) == 2);
  // Reuse the removed record's slot with a shorter, unrelated conversation.
  // Its tokens cannot be indexed at the old parent's boundary.
  const auto unrelated = f.Add({1, 2});
  assert(f.policy.Rank(root) == 2);
  f.policy.Remove(b);
  assert(f.policy.Rank(root) == 3);
  (void)f.Add(right, RetentionPurpose::kContinuation, 0, {}, true, {42});
  assert(f.policy.Rank(root) == 3);  // another input cannot cover the prefix
  f.policy.Remove(unrelated);
  f.Add(right);
  assert(f.policy.Rank(root) == 2);
  f.Add(right, RetentionPurpose::kBranchPoint);  // replace the same boundary
  assert(f.policy.Rank(root) == 2);
  f.Add({10, 11});
  f.Add({12, 13});
  assert(f.Cached(prefix) == 0);  // covered root still loses at record pressure
  assert(f.Cached(right) == right.size());
  assert(f.Cached(right, {42}) == right.size());
}
void ShortPeersDoNotCoverHistory() {
  for (const auto purpose :
       {RetentionPurpose::kContinuation, RetentionPurpose::kBranchPoint}) {
    Fixture f(3);
    const auto root = f.Add({1, 2}, purpose);
    const auto short_tail = f.Add({1, 2, 7});
    const auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
    // A shared system prefix cannot restore the deeper history's rows.
    assert(f.policy.Rank(history) == 3);
    f.Add({1, 2, 8}, RetentionPurpose::kContinuation, 0, root);
    assert(f.log.events[f.log.events.size() - 2].checkpoint == short_tail);
    assert(f.Cached({1, 2, 3, 4, 5}) == 4);
    assert(f.Cached({1, 2, 8, 9}) == 3);
    assert(f.Cached({1, 2, 7, 9}) == 2);
  }
  {
    Fixture f(3);
    const auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
    f.Add({1, 2, 3, 4, 5});
    assert(f.policy.Rank(history) == 1);  // a deeper peer really covers it
  }
  {
    Fixture f(3);
    const auto history =
        f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory, 0, {}, true, {1});
    f.Add({1, 2, 3, 4, 5}, RetentionPurpose::kContinuation, 0, {}, true, {2});
    assert(f.policy.Rank(history) == 3);  // different inputs cannot cover it
  }
}
void HistoryBytePressureCanReplaceOldFamilies() {
  Fixture f(4);
  const auto old = f.Add({9, 9});
  f.Add({8, 8});
  std::vector<Token> history{1, 2};
  unsigned attempts = 0;
  assert(f.policy.Admit({history, {1}, {}, RetentionPurpose::kHistory}, [&] {
    if (!attempts++)
      throw ResourceExhausted();
    return IndexCheckpoint(f.ledger, history);
  }));
  assert(attempts == 2 && f.Cached({1, 2, 3}) == 2);
  assert(f.Cached({9, 9, 1}) == 0 && f.Cached({8, 8, 1}) == 2);
  assert(f.log.events[f.log.events.size() - 2].checkpoint == old);
  assert(f.log.events[f.log.events.size() - 2].reason ==
         RetentionReason::kByteCapacity);
}
void CurrentTurnKeepsItsEditBoundaries() {
  Fixture f(4);
  const auto old = f.Add({9, 9});
  const auto early = f.Add({1, 2}, RetentionPurpose::kHistory);
  const auto later = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
  const auto source = f.Add({1, 2, 3, 4, 5});
  std::array protected_ids{early, later};
  std::vector<Token> incoming{1, 2, 3, 4, 5, 6};
  RetentionRequest request{
      incoming, {1}, {}, RetentionPurpose::kContinuation, 0, source, true};
  request.protected_checkpoints = protected_ids;
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == old);
  assert(f.Cached({1, 2, 7}) == 2 && f.Cached({1, 2, 3, 4, 7}) == 4);
  // Protection lasts for the current turn only; a later independent turn may
  // reclaim these covered histories under the same finite capacity. The
  // deeper one goes first: the shallow boundary may serve other prompts.
  request.protected_checkpoints = {};
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == later);
  assert(f.Cached({1, 2, 7}) == 2 && f.Cached({1, 2, 3, 4, 7}) == 2);
}
// Eight tokens of rows cost as much as one private state: shorter checkpoints
// are cheap to recompute relative to the budget they hold.
constexpr std::array<ComponentDescriptor, 2> kPressureComponents{
    {{{1}, 1, ComponentKind::kAppendRows, 64, 4, 0},
     {{2}, 1, ComponentKind::kPrivateState, 0, 0, 4096}}};
struct PressureFixture {
  ResourceLedger ledger;
  PrefixIndex index{ledger};
  Log log;
  RetentionPolicy policy;
  explicit PressureFixture(std::size_t ram = 1ULL << 30)
      : ledger({ram, ram, 0, ram}), policy(ledger, index, &log) {
    index.Register({1}, kPressureComponents);
  }
  // Each conversation executes separately, so no two share chunks. Admission
  // reclaims through the byte-pressure order when the ledger is full.
  CheckpointId Add(std::vector<Token> tokens,
                   RetentionPurpose purpose = RetentionPurpose::kContinuation,
                   bool reused = false) {
    RetentionRequest request{tokens, {1}, {}, purpose};
    request.reused = reused;
    assert(policy.Admit(request, [&] {
      return IndexCheckpoint(ledger, tokens, {1}, {}, kPressureComponents);
    }));
    return log.events.back().checkpoint;
  }
  CheckpointId Reclaim() {
    std::vector<Token> incoming{99, 99, 99, 99, 99, 99, 99, 99, 99};
    assert(policy.Reclaim({incoming, {1}, {}}));
    return log.events.back().checkpoint;
  }
  Rows Cached(std::vector<Token> tokens) {
    return index.CachedPrefixTokens(
        {{1}, tokens, InputIdentity(tokens.size()), 0, true});
  }
};
std::vector<Token> Conversation(Token first, std::size_t size) {
  std::vector<Token> tokens(size, first);
  for (std::size_t i = 0; i < size; ++i)
    tokens[i] += static_cast<Token>(i * 1000);
  return tokens;
}
void CheapCheckpointsGiveWayFirst() {
  // An idle deep conversation outlives newer short requests (cache-pressure
  // idle resume). Retry copies still go before any cheap checkpoint.
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto first = f.Add(Conversation(2, 7));
  const auto second = f.Add(Conversation(3, 7));
  auto retry_tokens = Conversation(3, 7);
  retry_tokens.push_back(5);
  RetentionRequest retry{retry_tokens, {1}, {}, RetentionPurpose::kRetry, 7};
  assert(f.policy.Admit(retry, [&] {
    return IndexCheckpoint(f.ledger, retry_tokens, {1}, {},
                           kPressureComponents);
  }));
  const auto copy = f.log.events.back().checkpoint;
  assert(f.Reclaim() == copy);
  assert(f.Reclaim() == first && f.Reclaim() == second);
  assert(f.Reclaim() == idle);
}
void ReusedCheapCheckpointKeepsRecency() {
  // A short checkpoint restored one request ago is not cheap any more: its
  // next turn would be a full miss (state-edges). It competes by recency.
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto restored = f.Add(Conversation(2, 7));
  const auto other = f.Add(Conversation(3, 7));
  f.policy.Touch(restored);
  assert(f.Reclaim() == other);
  assert(f.Reclaim() == idle);
  assert(f.Reclaim() == restored);
}
void PersistingCheckpointIsNotAVictim() {
  // While a disk job writes a checkpoint, removing it frees none of its
  // storage and only drops a restorable boundary from the index.
  PressureFixture f;
  const auto older = f.Add(Conversation(1, 16));
  auto tokens = Conversation(2, 16);
  std::vector<ResourceCharge> charges;
  RetentionRequest request{tokens, {1}, {}, RetentionPurpose::kContinuation};
  assert(f.policy.Admit(request, [&] {
    auto history = ExecutionHistory::Cold(f.ledger, kPressureComponents, {1});
    std::vector<ComponentPosition> positions;
    for (const auto& d : kPressureComponents)
      positions.push_back({d.id, tokens.size()});
    return history.Capture(
        {tokens, InputIdentity(tokens.size()), positions,
         CheckpointPurpose::kPrompt, 1},
        [&](const PayloadRequest& payload) {
          auto pool =
              f.ledger.Reserve(ResourceCategory::kBackingFree, payload.bytes)
                  .Convert();
          auto charge = pool.ReserveBacking(payload.category).Convert();
          charges.push_back(charge);
          return Payload::Committed(std::move(charge),
                                    std::make_shared<int>(0));
        });
  }));
  const auto written = f.log.events.back().checkpoint;
  std::vector<PersistencePin> pins;
  for (const auto& charge : charges)
    pins.push_back(charge.PinPersistence());
  // The written checkpoint is newer, so it would otherwise go second.
  f.policy.Touch(older);
  assert(f.Reclaim() == older);
  assert(!f.policy.Reclaim({std::vector<Token>(9, 99), {1}, {}}));
  pins.clear();
  assert(f.Reclaim() == written);
}
void RetryCopyReplacesItsNearbySource() {
  // An unchanged retry restored from its stable boundary may replace that
  // boundary with the full-prompt copy (state-edges), never another
  // conversation's checkpoint, and not when the gap exceeds the framing.
  PressureFixture f;
  const auto other = f.Add(Conversation(1, 16));
  const auto stable = Conversation(2, 7);
  const auto source = f.Add(stable);
  auto retry = stable;
  retry.push_back(5);
  RetentionRequest request{retry, {1}, {}, RetentionPurpose::kRetry, 7};
  request.source = source;
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == source);
  assert(!f.policy.Reclaim(request));
  auto far = Conversation(1, 16);
  far.resize(16 + 65, 3);
  RetentionRequest distant{far, {1}, {}, RetentionPurpose::kRetry, 16};
  distant.source = other;
  assert(!f.policy.Reclaim(distant));
}
void ResumedRetryReclaimsCoveredHistory() {
  PressureFixture f;
  const auto old = Conversation(1, 16);
  const auto ancestor = f.Add({old.begin(), old.begin() + 8});
  f.Add(old);
  const auto unrelated = Conversation(2, 16);
  f.Add(unrelated);
  const auto fork = Conversation(3, 8);
  f.Add(fork);
  auto left = fork, right = fork;
  left.push_back(10);
  right.push_back(20);
  f.Add(left);
  f.Add(right);
  const auto current = Conversation(4, 7);
  const auto source = f.Add({current.begin(), current.begin() + 4});
  const auto stable = f.Add(current);
  const std::array protected_ids{source, stable};
  auto prompt = current;
  prompt.push_back(5);
  RetentionRequest request{prompt, {1}, {}, RetentionPurpose::kRetry, 7};
  request.protected_checkpoints = protected_ids;
  // An optional cold retry cannot evict another conversation's history.
  assert(!f.policy.Reclaim(request));
  auto longer = prompt;
  longer.resize(prompt.size() + 65, 77);
  RetentionRequest distant{longer, {1}, {}, RetentionPurpose::kRetry, 7};
  distant.reused = true;
  distant.reused_tokens = 4;
  distant.protected_checkpoints = protected_ids;
  assert(!f.policy.Reclaim(distant) &&
         "large resumed prompt displaced covered history for a retry");
  request.reused = true;
  request.reused_tokens = current.size();
  assert(f.policy.Reclaim(request) &&
         "resumed retry starved behind covered history");
  assert(f.log.events.back().checkpoint == ancestor);
  assert(f.Cached(old) == old.size());
  assert(f.Cached(unrelated) == unrelated.size());
  assert(f.Cached(fork) == fork.size());
  assert(f.Cached({current.begin(), current.begin() + 4}) == 4);
  assert(f.Cached(current) == current.size());
  // Last copies, branching prefixes and the current turn's edit boundaries
  // remain protected even if a full-prompt retry would still need more space.
  assert(!f.policy.Reclaim(request));
}
void LongResumedRetryReclaimsOnlyOwnAncestors() {
  for (const bool named_source : {false, true}) {
    PressureFixture f;
    const auto unrelated = Conversation(1, 16);
    f.Add({unrelated.begin(), unrelated.begin() + 8});
    f.Add(unrelated);
    const auto current = Conversation(2, 16);
    const auto ancestor = f.Add({current.begin(), current.begin() + 8});
    const auto source = f.Add(current);
    auto prompt = current;
    prompt.resize(216, 77);
    const auto stable = f.Add({prompt.begin(), prompt.end() - 5});
    const std::array protected_ids{source, stable};
    RetentionRequest request{prompt, {1}, {}, RetentionPurpose::kRetry, 211};
    request.reused = true;
    request.reused_tokens = current.size();
    request.source = named_source ? source : CheckpointId{};
    request.protected_checkpoints = protected_ids;
    assert(f.policy.Reclaim(request) &&
           "long resumed retry starved behind its own covered ancestor");
    assert(f.log.events.back().checkpoint == ancestor);
    assert(f.Cached(current) == current.size());
    assert(f.Cached(unrelated) == unrelated.size());
    // The older, unrelated ancestor remains ineligible for this larger turn.
    assert(!f.policy.Reclaim(request));
  }
}
void SharedShortPrefixIsNotCheap() {
  // A new conversation sharing a short system/tools prefix must not evict
  // that prefix to admit itself (tool-agent peers).
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto prefix = Conversation(2, 7);
  const auto shared = f.Add(prefix);
  auto incoming = prefix;
  incoming.insert(incoming.end(), {4, 4, 4});
  assert(f.policy.Reclaim({incoming, {1}, {}}));
  assert(f.log.events.back().checkpoint == idle);
  assert(f.Cached(incoming) == 7);
  (void)shared;
}
void UnreusedConversationKeepsItsStem() {
  // Compaction keeps a stale conversation's system prompt (cache-pressure
  // compacted): its leaf goes before its shallow history boundary. Once a
  // conversation is reused, its covered records go before any leaf.
  PressureFixture f;
  auto old_conversation = Conversation(1, 16);
  const auto old_stem =
      f.Add({old_conversation.begin(), old_conversation.begin() + 8},
            RetentionPurpose::kHistory);
  const auto old_leaf = f.Add(old_conversation);
  auto live = Conversation(2, 16);
  const auto live_stem =
      f.Add({live.begin(), live.begin() + 8}, RetentionPurpose::kHistory);
  const auto live_old = f.Add({live.begin(), live.begin() + 12});
  const auto live_leaf = f.Add(live, RetentionPurpose::kContinuation, true);
  assert(f.Reclaim() == old_leaf);
  auto compacted = old_conversation;
  compacted.resize(9);
  compacted[8] = 7;
  assert(f.Cached(compacted) == 8);
  assert(f.Reclaim() == old_stem);
  // The reused conversation drops its prior turn, then its stem, deepest
  // covered record first, before its own continuation.
  assert(f.Reclaim() == live_old && f.Reclaim() == live_stem);
  assert(f.Reclaim() == live_leaf);
}
void ForkedParentOutlivesItsChildren() {
  // Concurrent subagents share the parent turn but diverge a few tokens later
  // (cache-pressure fan-out), so the parent is no exact branch point. The
  // parent must still outlive the children for its own next turn.
  PressureFixture f;
  auto parent = Conversation(1, 12);
  const auto prior = f.Add({parent.begin(), parent.begin() + 8},
                           RetentionPurpose::kContinuation, true);
  const auto fork = f.Add(parent, RetentionPurpose::kContinuation, true);
  std::array<CheckpointId, 2> children;
  for (Token child = 0; child < 2; ++child) {
    auto tokens = parent;
    tokens.insert(tokens.end(), {500, 501, Token(600 + child), 502});
    children[child] = f.Add(tokens, RetentionPurpose::kContinuation, true);
  }
  assert(f.Reclaim() == prior);
  assert(f.Reclaim() == children[0] && f.Reclaim() == children[1]);
  parent.push_back(7);
  assert(f.Cached(parent) == 12);
  assert(f.Reclaim() == fork);
}
void ReuseProtectionLastsOneTurnover() {
  // A reused conversation outlives newer one-use requests (cache-pressure UI
  // tasks) for one turnover of the RAM budget, then ages out like any other.
  constexpr std::size_t kRam = 64 * 1024;
  PressureFixture f(kRam);
  auto conversation = Conversation(1, 16);
  f.Add(conversation);
  f.policy.Touch(f.log.events.back().checkpoint);
  Token next = 100;
  std::size_t admitted = 0;
  // Unique admitted bytes per one-use request: private state plus rows.
  constexpr std::size_t kRequest = 4096 + 16 * 64;
  while (admitted + kRequest < kRam) {
    f.Add(Conversation(next++, 16));
    admitted += kRequest;
  }
  // The budget filled; least recently used order alone would have removed
  // the conversation before any of these newer requests.
  assert(std::ranges::any_of(f.log.events, [](const auto& e) {
    return e.action == RetentionAction::kRemoved;
  }));
  assert(f.Cached(conversation) == 16);
  while (admitted < 2 * kRam) {
    f.Add(Conversation(next++, 16));
    admitted += kRequest;
    assert(f.ledger.Snapshot().total_bytes <= kRam);
  }
  assert(f.Cached(conversation) == 0);
  assert(std::ranges::any_of(f.log.events, [](const auto& e) {
    return e.action == RetentionAction::kRemoved &&
           e.reason == RetentionReason::kByteCapacity;
  }));
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
      assert(policy.RetainedPayloadBytes() > 0);
      policy.Remove(id);
      assert(policy.RetainedPayloadBytes() == 0);
      assert(log.events.back().retained_payload_bytes == 0);
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
void SharedChunkAccounting() {
  for (const std::size_t chunks : {0, 1, 7, 64, 2048}) {
    constexpr std::size_t capacity = 64ULL << 20;
    ResourceLedger ledger({capacity, capacity, 0, capacity});
    constexpr std::array<ComponentDescriptor, 3> components{
        {{{1}, 1, ComponentKind::kAppendRows, 8, 4, 0},
         {{2}, 1, ComponentKind::kAppendRows, 16, 8, 0},
         {{3}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    PrefixIndex index(ledger);
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    const auto capture = [&](ExecutionHistory& history,
                             const std::vector<Token>& tokens) {
      const auto end = tokens.size();
      return history.Capture(
          {tokens,
           InputIdentity(end),
           {{{1}, end}, {{2}, end}, {{3}, end}},
           CheckpointPurpose::kPrompt,
           3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                                      std::make_shared<int>(0));
          });
    };
    std::vector<std::shared_ptr<const Checkpoint>> retained;
    const auto check = [&] {
      // Independent exhaustive oracle: count each actual chunk identity once,
      // without relying on lineage, prefix length or record insertion order.
      std::set<std::uint64_t> seen;
      std::size_t expected = 0;
      for (const auto& checkpoint : retained)
        for (const auto& c : checkpoint->Components()) {
          if (c.private_state)
            expected += c.private_state->Bytes();
          if (c.tail)
            expected += c.tail->Bytes();
          for (const auto& chunk : c.chunks)
            if (seen.insert(chunk.Id().value).second)
              expected += chunk.Storage().Bytes();
        }
      assert(policy.RetainedPayloadBytes() == expected);
      assert(log.events.back().retained_payload_bytes == expected);
    };
    const auto admit = [&](std::shared_ptr<const Checkpoint> checkpoint) {
      assert(policy.Admit({checkpoint->Tokens(), {1}, {}},
                          [&] { return checkpoint; }));
      retained.push_back(std::move(checkpoint));
      check();
    };
    auto history = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<Token> tokens(chunks * 4 + 1, 1);
    const auto root = capture(history, tokens);
    tokens.resize(tokens.size() + chunks * 4 + 3, 2);
    const auto branch = capture(history, tokens);
    admit(branch);  // Deepest first; shorter shared prefixes arrive later.
    admit(root);
    auto sibling = ExecutionHistory::Restored(ledger, *root);
    tokens.assign(root->Tokens().begin(), root->Tokens().end());
    tokens.resize(tokens.size() + chunks * 8 + 5, 3);
    admit(capture(sibling, tokens));
    auto child = ExecutionHistory::Restored(ledger, *branch);
    tokens.assign(branch->Tokens().begin(), branch->Tokens().end());
    tokens.resize(tokens.size() + chunks * 4 + 7, 4);
    admit(capture(child, tokens));
    auto unrelated = ExecutionHistory::Cold(ledger, components, {1});
    tokens.assign(chunks * 4 + 11, 9);
    admit(capture(unrelated, tokens));
    // Removing records changes which owner accounts for shared bytes. External
    // checkpoint handles keep storage alive, but no longer count as retained.
    for (const std::size_t remove : {1, 0, 1, 0, 0}) {
      const auto id = retained[remove]->Id();
      retained.erase(retained.begin() + remove);
      policy.Remove(id);
      check();
      if (!retained.empty()) {
        policy.Touch(retained.back()->Id());
        check();
      }
    }
    assert(policy.Size() == 0);
    if (chunks == 7) {
      // Exercise repeated incremental additions/removals across many shared
      // checkpoints. Keep external owners alive so ledger release cannot
      // stand in for the logical retained-byte total.
      auto growing = ExecutionHistory::Cold(ledger, components, {1});
      std::vector<std::shared_ptr<const Checkpoint>> saved;
      tokens.clear();
      for (std::size_t n = 0; n < 40; ++n) {
        tokens.resize((n + 1) * 4 + 1, 71);
        saved.push_back(capture(growing, tokens));
        admit(saved.back());
      }
      for (std::size_t n = 0; n < saved.size(); ++n) {
        const auto& checkpoint = saved[n * 17 % saved.size()];
        const auto found =
            std::find(retained.begin(), retained.end(), checkpoint);
        assert(found != retained.end());
        retained.erase(found);
        policy.Remove(checkpoint->Id());
        check();
      }
      for (auto it = saved.rbegin(); it != saved.rend(); ++it)
        admit(*it);
      while (!retained.empty()) {
        const auto id = retained.back()->Id();
        retained.pop_back();
        policy.Remove(id);
        check();
      }
      assert(policy.Size() == 0);
    }
    if (chunks == 0) {
      // A reader may retain only a late chunk after the source checkpoint
      // disappears. Two restored histories then recreate earlier chunks
      // independently while still sharing that late chunk.
      auto source = ExecutionHistory::Cold(ledger, components, {1});
      tokens.assign(8, 41);
      auto original = capture(source, tokens);
      auto left = ExecutionHistory::Restored(ledger, *original);
      auto right = ExecutionHistory::Restored(ledger, *original);
      auto late = original->Components()[0].chunks.back().PinReader();
      original.reset();
      tokens.push_back(42);
      admit(capture(left, tokens));
      tokens.back() = 43;
      admit(capture(right, tokens));
      assert(retained[0]->Components()[0].chunks[0].Id() !=
             retained[1]->Components()[0].chunks[0].Id());
      assert(retained[0]->Components()[0].chunks[1].Id() ==
             retained[1]->Components()[0].chunks[1].Id());
    }
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
      assert(policy.RetainedPayloadBytes() == end * 1024 + end / 4 * 32);
      assert(log.events.back().retained_payload_bytes ==
             policy.RetainedPayloadBytes());
    }
    assert(policy.Size() == 3);
    assert(policy.RetainedPayloadBytes() == 12 * 1024 + 3 * 32);
    const auto latest = log.events.back().checkpoint;
    policy.Touch(latest);
    assert(log.events.back().retained_payload_bytes == 12 * 1024 + 3 * 32);
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
  LongPrefixRelationsAfterReplacement();
  ShortPeersDoNotCoverHistory();
  HistoryBytePressureCanReplaceOldFamilies();
  CurrentTurnKeepsItsEditBoundaries();
  CheapCheckpointsGiveWayFirst();
  ReusedCheapCheckpointKeepsRecency();
  PersistingCheckpointIsNotAVictim();
  RetryCopyReplacesItsNearbySource();
  ResumedRetryReclaimsCoveredHistory();
  LongResumedRetryReclaimsOnlyOwnAncestors();
  SharedShortPrefixIsNotCheap();
  UnreusedConversationKeepsItsStem();
  ForkedParentOutlivesItsChildren();
  ReuseProtectionLastsOneTurnover();
  RecordSourceReplacement();
  FullOptionalExactReplacement();
  FailureAndReplay();
  CapacityAndSharing();
  ImpossibleAndAllocationFailure();
  RealCapacityAndUniqueAdmission();
  SharedChunkAccounting();
  InvalidInventoriesPreserveReplacement();
}
