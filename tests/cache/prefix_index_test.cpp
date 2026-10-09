#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

#include "tests/cache/prefix_index_fixture.hpp"

namespace {
bool watch_allocations = false;
std::size_t largest_allocation = 0;
}  // namespace
// Observe actual edge allocation order when a ledger reservation is rejected.
void* operator new(std::size_t bytes) {
  if (watch_allocations)
    largest_allocation = std::max(largest_allocation, bytes);
  if (void* pointer = std::malloc(bytes ? bytes : 1))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {
using Tokens = std::vector<Token>;
PrefixQuery Query(const Tokens& tokens, Rows stable = 0, Identity input = {},
                  std::vector<InputPrefix> prefixes = {}) {
  return {{1},
          tokens,
          InputIdentity(tokens.size(), std::move(input), std::move(prefixes)),
          stable,
          true};
}
template<class F>
void Invalid(F action) {
  bool threw = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}
void SelectionTable() {
  struct Case {
    Tokens prompt;
    Rows stable;
    bool shorter, marked, live, available;
    Rows expected;
    SelectionReason reason;
  };
  const std::vector<Case> cases{
      {{1, 2, 3, 4},
       0,
       true,
       false,
       true,
       true,
       3,
       SelectionReason::kExactLiveContinuation},
      {{1, 2, 3, 4},
       0,
       true,
       false,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       2,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       2,
       true,
       false,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       2,
       false,
       true,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       1,
       false,
       true,
       false,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       2,
       false,
       false,
       true,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       0,
       true,
       false,
       true,
       false,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2},
       0,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kNoCompatibleBoundary},
      {{1, 9, 3},
       0,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kNoCompatibleBoundary},
  };
  for (const auto& c : cases) {
    ResourceLedger ledger{kIndexLimits};
    {
      PrefixIndex index{ledger};
      index.Register({1}, kIndexComponents);
      const auto deep = IndexCheckpoint(ledger, Tokens{1, 2, 3});
      (void)index.Insert(deep, ResidentComponents(), c.marked ? 2 : 0);
      if (c.shorter)
        (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}),
                           ResidentComponents());
      if (c.live) {
        auto frontier = IndexLive({1, 2, 3});
        frontier.available = c.available;
        (void)index.Insert(std::move(frontier));
      }
      const auto result = index.Lookup(Query(c.prompt, c.stable));
      assert(result.reason == c.reason);
      assert((result.selected ? result.selected->boundary : 0) == c.expected);
      if (result.selected && result.selected->checkpoint)
        assert(result.selected->transfer_bytes == 3 * 4 + 1 * 2 + 32);
      if (c.shorter && c.expected == 3)
        assert(result.candidates.back().boundary == 2);
      assert(index.CachedPrefixTokens(Query(c.prompt, c.stable)) ==
             (c.prompt.size() >= 3 && c.prompt[1] == 2 ? 3 : 0));
    }
    assert(ledger.Snapshot().total_bytes == 0);
  }
}
void ImagesAndLearning() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens text{1, 2, 3}, image{1, 2, 3, 248056, 4},
      appended{1, 2, 3, 248056, 4, 248056, 5};
  auto before =
      index.Insert(IndexCheckpoint(ledger, text), ResidentComponents());
  const auto first = index.Insert(IndexCheckpoint(ledger, image, {1}, {10}),
                                  ResidentComponents());
  auto live = index.Insert(IndexLive(image, {7}, {10}));
  auto query = Query(appended, 0, {30}, {{3, {}}, {5, {10}}});
  assert(index.Lookup(query).selected->live->slot == SlotId{7});
  assert(index.CachedPrefixTokens(query) == 5);
  assert(index.CommonPrefixTokens(query) == 5);
  query = Query(image, 0, {20}, {{3, {}}});
  assert(index.Lookup(query).selected->boundary == 3);
  assert(index.CommonPrefixTokens(query) == 3);
  assert(index.Lookup(Query(image)).selected->boundary == 3);
  index.Erase(before);
  assert(index.Lookup(query).reason == SelectionReason::kInputIdentity);
  assert(index.CommonPrefixTokens(query) == 0);
  index.Erase(first);
  index.SetLiveAvailable(live, false);
  assert(index.CachedPrefixTokens(Query(image, 0, {10})) == 0);
  assert(index.CommonPrefixTokens(Query(image, 0, {10})) == 5);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2}, 0, {10})) == 2);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2})) == 0);
  index.Erase(live);
  // Port the legacy shared-prefix peeks: branches teach their divergence even
  // when no exact checkpoint exists at that boundary; availability is ignored.
  (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2, 3, 4, 5}),
                     ResidentComponents());
  auto unavailable = index.Insert(IndexLive({1, 2, 3, 6, 7}, {8}));
  index.SetLiveAvailable(unavailable, false);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2, 3, 6, 9})) == 4);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2})) == 2);
  assert(index.CommonPrefixTokens(Query(Tokens{9, 9})) == 0);
}
void CoherenceAndFallback() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4};
  auto full = IndexCheckpoint(ledger, tokens);
  auto missing = ResidentComponents();
  missing.pop_back();
  const auto id = index.Insert(full, missing);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.SetAvailability(
      id, {{{1}, true, true}, {{2}, true, false}, {{3}, false, true}});
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.SetAvailability(id, ResidentComponents());
  assert(index.Lookup(Query(tokens)).selected->checkpoint == full);
  // A live frontier cannot stand in for the stable fallback snapshot.
  auto short_live = index.Insert(IndexLive({1, 2}, {4}));
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 2);
  index.Erase(short_live);
  const auto shorter =
      index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}), missing);
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.SetAvailability(shorter, ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 4);
  index.Erase(shorter);
  // A mark on another compatible later checkpoint also attests the fallback.
  const auto marked = index.Insert(IndexCheckpoint(ledger, Tokens{1, 2, 3}),
                                   ResidentComponents(), 2);
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 4);
  index.Erase(marked);
  // Missing draft components and changed layouts must never become hits, even
  // if the producer supplied the same compatibility bytes by mistake.
  auto incomplete = IndexCheckpoint(ledger, tokens, {1}, {},
                                    std::span(kIndexComponents).first(1));
  const auto partial = index.Insert(incomplete, ResidentComponents());
  index.Erase(id);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.Erase(partial);
  auto changed = kIndexComponents;
  changed[0].layout_version++;
  (void)index.Insert(IndexCheckpoint(ledger, tokens, {1}, {}, changed),
                     ResidentComponents());
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  auto frontier = IndexLive(tokens);
  frontier.positions.back().valid_rows--;
  (void)index.Insert(frontier);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  frontier.positions.pop_back();
  (void)index.Insert(frontier);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
}
void CheckpointPrivateBoundary() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4};
  for (const Rows position : {3ULL, 5ULL}) {
    auto malformed =
        IndexCheckpoint(ledger, tokens, {1}, {}, kIndexComponents, position);
    const auto entry = index.Insert(malformed, ResidentComponents());
    assert(index.Lookup(Query(tokens)).reason ==
           SelectionReason::kMissingComponent);
    assert(index.CachedPrefixTokens(Query(tokens)) == 0);
    index.Erase(entry);
  }
  const auto deep =
      index.Insert(IndexCheckpoint(ledger, tokens), ResidentComponents());
  // An incoherent shorter checkpoint cannot supply a stable fallback.
  auto malformed =
      IndexCheckpoint(ledger, Tokens{1, 2}, {1}, {}, kIndexComponents, 1);
  auto entry = index.Insert(malformed, ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.Erase(entry);
  // An incoherent later checkpoint cannot attest an established fallback
  // either.
  malformed =
      IndexCheckpoint(ledger, Tokens{1, 2, 3}, {1}, {}, kIndexComponents, 2);
  entry = index.Insert(malformed, ResidentComponents(), 2);
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.Erase(entry);
  (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}),
                     ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).selected->entry == deep);
}
void TiesAndIsolation() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  index.Register({2}, kIndexComponents);
  const Tokens tokens{1, 2, 3}, prompt{1, 2, 3, 4};
  auto earlier = IndexCheckpoint(ledger, tokens);
  auto later = IndexCheckpoint(ledger, tokens);
  const auto second = index.Insert(later, ResidentComponents());
  (void)index.Insert(earlier, ResidentComponents());
  assert(index.Lookup(Query(prompt)).selected->checkpoint == earlier);
  auto b = index.Insert(IndexLive(tokens, {2}));
  auto a = index.Insert(IndexLive(tokens, {1}));
  assert(index.Lookup(Query(prompt)).selected->entry == a);
  index.Erase(a);
  assert(index.Lookup(Query(prompt)).selected->entry == b);
  index.Erase(b);
  auto deeper =
      index.Insert(IndexCheckpoint(ledger, prompt), ResidentComponents());
  (void)index.Insert(IndexLive(tokens));
  assert(index.Lookup(Query(prompt)).selected->entry == deeper);
  auto query = Query(prompt);
  query.compatibility = {2};
  assert(!index.Lookup(query).selected);
  query.compatibility = {3};
  assert(index.Lookup(query).reason == SelectionReason::kUnknownCompatibility);
  query = Query(prompt);
  query.reuse = false;
  assert(index.Lookup(query).reason == SelectionReason::kDisabled);
  assert(index.CachedPrefixTokens(query) == 4);
  auto pin = index.Lookup(Query(prompt)).selected->checkpoint;
  index.Erase(deeper);
  assert(pin->Boundary() == 4);
  index.Erase(second);
  index.Erase(second);
}
void LongPromptAndMemory() {
  ResourceLedger ledger{kIndexLimits};
  {
    std::vector<Token> prompt(131072, 7);
    std::vector<std::shared_ptr<const Checkpoint>> checkpoints;
    for (std::size_t count = 1024; count <= prompt.size(); count += 1024)
      checkpoints.push_back(
          IndexCheckpoint(ledger, std::span(prompt).first(count)));
    PrefixIndex forward{ledger}, reverse{ledger};
    forward.Register({1}, kIndexComponents);
    reverse.Register({1}, kIndexComponents);
    const auto empty = forward.MetadataBytes();
    std::vector<IndexEntryId> entries;
    for (const auto& cp : checkpoints)
      (void)forward.Insert(cp, ResidentComponents());
    for (auto i = checkpoints.rbegin(); i != checkpoints.rend(); ++i)
      entries.push_back(reverse.Insert(*i, ResidentComponents()));
    assert(forward.MetadataBytes() == reverse.MetadataBytes());
    assert(forward.MetadataBytes() <
           prompt.size() * sizeof(Token) + 128 * 1024);
    const auto result = reverse.Lookup(Query(prompt));
    assert(result.selected->boundary == 131072);
    assert(result.candidates.size() == 128);
    prompt.back() = 9;
    assert(reverse.CachedPrefixTokens(Query(prompt)) == 130048);
    assert(reverse.CommonPrefixTokens(Query(prompt)) == 131071);
    for (const auto id : entries)
      reverse.Erase(id);
    assert(reverse.MetadataBytes() == empty);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void PrefixChurn() {
  ResourceLedger ledger{kIndexLimits};
  {
    PrefixIndex index{ledger};
    index.Register({1}, kIndexComponents);
    std::vector<Token> prompt(131072, 7);
    (void)index.Insert(IndexCheckpoint(ledger, prompt), ResidentComponents());
    const auto retained = index.MetadataBytes();
    for (std::size_t count = 128; count < prompt.size(); count += 128) {
      auto entry = index.Insert(IndexLive(std::vector<Token>(count, 7)));
      index.Erase(entry);
      assert(index.MetadataBytes() == retained);
    }
    const auto result = index.Lookup(Query(prompt));
    assert(result.candidates.size() == 1);
    assert(result.selected->boundary == prompt.size());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void DeferredCompaction() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4, 5};
  (void)index.Insert(IndexCheckpoint(ledger, tokens), ResidentComponents());
  const auto retained = index.MetadataBytes();
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    auto short_live = index.Insert(IndexLive({1, 2}));
    ledger.FailAfter(step, 0);
    index.Erase(short_live);
    assert(index.Lookup(Query(tokens)).candidates.size() == 1);
    assert(index.CachedPrefixTokens(Query(Tokens{1, 2})) == 0);
    assert(index.MetadataBytes() > retained);
    auto retry = index.Insert(IndexLive({1, 2, 3}));
    index.Erase(retry);
    assert(index.MetadataBytes() == retained);
  }
}
void AdmissionBeforeAllocation() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  std::vector<Token> prompt(131072, 7);
  auto checkpoint = IndexCheckpoint(ledger, prompt);
  auto availability = ResidentComponents();
  const auto before = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 1);
  largest_allocation = 0;
  watch_allocations = true;
  bool rejected = false;
  try {
    (void)index.Insert(checkpoint, std::move(availability));
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < prompt.size() * sizeof(Token));
  assert(ledger.Snapshot().total_bytes == before);
  (void)index.Insert(checkpoint, ResidentComponents());
  auto prefix = IndexCheckpoint(ledger, std::span(prompt).first(1024));
  availability = ResidentComponents();
  const auto split_before = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 2);
  largest_allocation = 0;
  watch_allocations = true;
  rejected = false;
  try {
    (void)index.Insert(prefix, std::move(availability));
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < (prompt.size() - 1024) * sizeof(Token));
  assert(ledger.Snapshot().total_bytes == split_before);

  std::vector<ComponentDescriptor> inventory(1024);
  for (std::size_t i = 0; i < inventory.size(); ++i)
    inventory[i] = {{static_cast<std::uint32_t>(i)},
                    1,
                    ComponentKind::kPrivateState,
                    0,
                    0,
                    32};
  ledger.FailAfter(LedgerStep::kReserve, 0);
  largest_allocation = 0;
  watch_allocations = true;
  rejected = false;
  try {
    index.Register({2}, inventory);
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < inventory.size() * sizeof(ComponentDescriptor));
  assert(ledger.Snapshot().total_bytes == split_before);
}
void AdmissionAndValidation() {
  ResourceLedger ledger{kIndexLimits};
  const Tokens original{1, 2, 3, 4}, split{1, 2, 9}, prefix{1, 2};
  auto retained = IndexCheckpoint(ledger, original);
  auto branch = IndexCheckpoint(ledger, split);
  auto parent = IndexCheckpoint(ledger, prefix);
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    for (std::size_t failure = 0; failure < 5; ++failure) {
      PrefixIndex index{ledger};
      index.Register({1}, kIndexComponents);
      (void)index.Insert(retained, ResidentComponents());
      const auto bytes = ledger.Snapshot().total_bytes;
      ledger.FailAfter(step, failure);
      bool failed = false;
      try {
        (void)index.Insert(branch, ResidentComponents());
      } catch (const std::bad_alloc&) {
        failed = true;
      }
      ledger.ClearFaults();
      assert(index.Lookup(Query(original)).selected->checkpoint == retained);
      if (failed)
        assert(ledger.Snapshot().total_bytes == bytes);
      else
        assert(index.Lookup(Query(split)).selected->checkpoint == branch);
    }
  }
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const auto original_id = index.Insert(retained, ResidentComponents());
  const auto parent_id = index.Insert(parent, ResidentComponents());
  const auto branch_id = index.Insert(branch, ResidentComponents());
  index.Erase(parent_id);
  assert(index.Lookup(Query(original)).selected->entry == original_id);
  assert(index.Lookup(Query(split)).selected->entry == branch_id);
  index.Erase(original_id);
  index.Erase(branch_id);
  assert(index.CachedPrefixTokens(Query(original)) == 0);
  Invalid([&] { index.Register({1}, std::span(kIndexComponents).first(1)); });
  Invalid([&] { (void)index.Insert(retained, {{{9}, true, false}}); });
  Invalid([&] {
    (void)index.Insert(retained, {{{1}, true, false}, {{1}, true, false}});
  });
  Invalid([&] { (void)index.Insert(retained, ResidentComponents(), 5); });
  Invalid([&] { (void)index.Lookup(Query(original, 5)); });
  auto query = Query(original);
  query.input = InputIdentity(1);
  Invalid([&] { (void)index.Lookup(query); });
  const auto id = index.Insert(retained, ResidentComponents());
  const auto old_bytes = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 0);
  bool failed = false;
  try {
    index.SetAvailability(id, {});
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  assert(failed);
  assert(ledger.Snapshot().total_bytes == old_bytes);
  assert(index.Lookup(Query(original)).selected->entry == id);
}
}  // namespace
int main() {
  SelectionTable();
  ImagesAndLearning();
  CoherenceAndFallback();
  CheckpointPrivateBoundary();
  TiesAndIsolation();
  LongPromptAndMemory();
  PrefixChurn();
  DeferredCompaction();
  AdmissionBeforeAllocation();
  AdmissionAndValidation();
  std::cout << "prefix index tests passed\n";
}
