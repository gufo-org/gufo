#include <array>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <new>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string_view>

#include "src/cache/retention.hpp"
#include "tests/cache/tiered_fixture.hpp"

namespace {
bool watch_allocations{};
std::size_t watched_allocations{};
}  // namespace
// Observe physical allocation before a rejected ledger admission.
void* operator new(std::size_t bytes) {
  if (watch_allocations)
    ++watched_allocations;
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
namespace fs = std::filesystem;
namespace {
void DeeperDiskAndRAMEviction() {
  TieredFixture f;
  f.Append(3, 0);
  auto short_checkpoint = f.Capture();
  (void)f.index.Insert(short_checkpoint, f.Resident());
  f.Append(37, 36);
  auto deep = f.Capture();
  const auto id = deep->Id();
  auto durable = f.Publish(*deep);
  const auto durable_entry =
      f.index.Insert(f.adapter.CompatibilityIdentity(), durable);
  {
    auto lookup = f.index.Lookup(f.Query());
    assert(lookup.selected->boundary == 40);
    assert(lookup.reason == SelectionReason::kDeepestDurableCheckpoint);
    assert(lookup.selected->disk_pin);
    assert(f.index.HasCheckpoint(f.Query(), 40));
    assert(!f.index.HasCheckpoint(f.Query(), 41));
    assert(f.disk.TryRetire(id, durable->Epoch()) == DiskRetireResult::kPinned);
  }
  f.index.Erase(durable_entry);
  RetentionPolicy retention(f.ledger, f.index);
  RetentionRequest request{f.tokens, f.adapter.CompatibilityIdentity(), {}};
  assert(retention.Admit(request, [&] { return deep; }));
  {
    auto lookup = f.index.Lookup(f.Query());
    f.index.AttachDurable(lookup.selected->entry, durable);
  }
  std::weak_ptr<const Checkpoint> weak = deep;
  deep.reset();
  retention.Remove(id);
  assert(weak.expired());
  {
    auto lookup = f.index.Lookup(f.Query());
    assert(lookup.selected->boundary == 40 && !lookup.selected->checkpoint);
    assert(lookup.selected->durable == durable);
    assert(f.index.HasCheckpoint(f.Query(), 40));
  }
  auto destination = f.adapter.CreateSlot(f.guard);
  TransferTiming timing;
  assert(f.streams.Restore(id, f.digest, f.adapter, *destination, timing));
  assert(f.adapter.Positions(*destination) == f.adapter.Positions(*f.slot));
  assert(f.adapter.RecurrentHash(*destination) ==
         f.adapter.RecurrentHash(*f.slot));
  for (auto descriptor : f.adapter.Components()) {
    const auto position = f.adapter.Positions(*f.slot)[descriptor.id.value - 1];
    const auto bytes = descriptor.kind == ComponentKind::kPrivateState
                           ? descriptor.state_bytes
                           : descriptor.row_bytes * position.valid_rows;
    std::vector<std::byte> expected(bytes), actual(bytes);
    auto capture = [&](const Slot& slot, auto& buffer) {
      return descriptor.kind == ComponentKind::kPrivateState
                 ? f.adapter.CapturePrivate(slot, descriptor.id, buffer,
                                            f.stream)
                 : f.adapter.CopyRowsOut(slot, descriptor.id, 0,
                                         position.valid_rows, buffer, f.stream);
    };
    assert(capture(*f.slot, expected).Wait() == TransferResult::kSucceeded);
    assert(capture(*destination, actual).Wait() == TransferResult::kSucceeded);
    assert(actual == expected);
  }
  const std::array<Token, 2> next{88, 91};
  f.adapter.Append(*destination, next, next);
  f.adapter.Append(*f.slot, next, next);
  assert(f.adapter.RecurrentHash(*destination) ==
         f.adapter.RecurrentHash(*f.slot));
}
void DemotionKeepsIndependentSourcePlans() {
  TieredFixture f;
  f.Append(16, 12);
  auto checkpoint = f.Capture();
  auto durable = f.Publish(*checkpoint);
  const auto entry = f.index.Insert(checkpoint, f.Resident());
  f.index.AttachDurable(entry, durable);
  auto resident = f.index.Lookup(f.Query());
  assert(resident.selected && resident.selected->checkpoint == checkpoint);

  // Reclaim must work even if the ledger refuses another plan allocation.
  f.ledger.FailAfter(LedgerStep::kReserve, 0);
  f.index.DropResident(entry);
  f.ledger.ClearFaults();
  auto disk = f.index.Lookup(f.Query());
  assert(disk.selected && disk.selected->entry == entry);
  assert(!disk.selected->checkpoint && disk.selected->durable == durable);
  assert(disk.selected->disk_pin);
  for (const auto& component : disk.selected->availability->components) {
    assert(!component.resident && component.durable);
    assert(disk.selected->UsesDisk(component.id));
  }
  assert(disk.selected->availability->components.size() ==
         f.adapter.Components().size());
  for (const auto& component : resident.selected->availability->components)
    assert(component.resident && !resident.selected->UsesDisk(component.id));
}
void CorruptionInvalidatesAllDependents() {
  TieredFixture f;
  f.Append(3, 0);
  auto short_checkpoint = f.Capture();
  (void)f.index.Insert(short_checkpoint, f.Resident());
  f.Append(37, 36);
  auto middle = f.Capture();
  auto first = f.Publish(*middle);
  (void)f.index.Insert(f.adapter.CompatibilityIdentity(), first);
  f.Append(8, 8);
  auto deep = f.Capture();
  auto second = f.Publish(*deep);
  const auto second_entry = f.index.Insert(deep, f.Resident());
  f.index.AttachDurable(second_entry, second);
  const auto chunk = first->Manifest().components[0].chunks[0];
  assert(chunk == second->Manifest().components[0].chunks[0]);
  const auto path = f.directory.path / "v2/chunks" / DiskFileName(chunk.file);
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.put('\xff');
  }
  auto destination = f.adapter.CreateSlot(f.guard);
  TransferTiming timing;
  bool fallback{};
  assert(!f.streams.Restore(
      deep->Id(), f.digest, f.adapter, *destination, timing, {},
      [&] {
        fallback = true;
        for (auto position : f.adapter.Positions(*destination))
          assert(position.valid_rows == 0);
        auto lookup = f.index.Lookup(f.Query());
        assert(lookup.selected->checkpoint == short_checkpoint);
      },
      [&](bool private_file, DiskFileId file) {
        f.catalog.Invalidate(private_file, file);
      }));
  assert(fallback);
  assert(!first->Pin() && !second->Pin());
  assert(!f.disk.Open(middle->Id()) && !f.disk.Open(deep->Id()));
  assert(f.catalog.ReclaimInvalid() == 2);
  assert(!fs::exists(path));
  assert(f.catalog.Size() == 0);
}
void GlobalLRUAndSharedFiles() {
  TieredFixture f;
  f.Append(8, 4);
  auto first = f.Capture();
  auto a = f.Publish(*first);
  (void)f.index.Insert(f.adapter.CompatibilityIdentity(), a);
  f.Append(8, 8);
  auto second = f.Capture();
  auto b = f.Publish(*second);
  (void)f.index.Insert(f.adapter.CompatibilityIdentity(), b);
  const auto shared = a->Manifest().components[0].chunks[0];
  const auto path = f.directory.path / "v2/chunks" / DiskFileName(shared.file);
  assert(shared == b->Manifest().components[0].chunks[0]);
  auto pin = a->Pin();
  assert(f.catalog.EvictOne());  // Pinned oldest is skipped.
  assert(f.disk.Open(first->Id()) && !f.disk.Open(second->Id()));
  assert(fs::exists(path));
  pin = {};
  // A retired ID can be republished, but its former description cannot pin it.
  auto replacement = f.Publish(*second);
  (void)f.index.Insert(f.adapter.CompatibilityIdentity(), replacement);
  assert(replacement->Epoch() != b->Epoch() && !b->Pin());
  f.catalog.Touch(first->Id());
  assert(f.catalog.EvictOne());
  assert(f.disk.Open(first->Id()) && !f.disk.Open(second->Id()));
  assert(fs::exists(path));
  assert(f.catalog.EvictOne());
  assert(!fs::exists(path) && f.disk.Stats().managed_bytes == 0);
}
void SupersededDiskTurnsLeaveBeforeAnotherConversation() {
  TieredFixture f;
  // An unrelated conversation is older than every active-run checkpoint.
  auto idle_history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                             f.adapter.CompatibilityIdentity());
  const std::vector<Token> idle_tokens(8, 99);
  f.adapter.Append(*f.slot, idle_tokens, idle_tokens);
  auto idle = idle_history.Capture(
      {idle_tokens, InputIdentity(8), f.adapter.Positions(*f.slot),
       CheckpointPurpose::kGenerated, 1},
      [&](const PayloadRequest& request) {
        auto pool =
            f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                .Convert();
        auto bytes = std::make_shared<std::vector<std::byte>>(request.bytes);
        auto assigned = pool.ReserveBacking(request.category);
        auto completion =
            request.category == ResourceCategory::kPrivateState
                ? f.adapter.CapturePrivate(*f.slot, request.component, *bytes,
                                           f.stream)
                : f.adapter.CopyRowsOut(*f.slot, request.component,
                                        request.first, request.end, *bytes,
                                        f.stream);
        assert(completion.Wait() == TransferResult::kSucceeded);
        return Payload::Committed(assigned.Convert(), bytes);
      });
  auto idle_description = f.Publish(*idle);
  assert(f.adapter.Invalidate(*f.slot));
  std::vector<CheckpointId> active;
  for (int turn = 0; turn < 4; ++turn) {
    f.Append(4, 4);
    const auto checkpoint = f.Capture(CheckpointPurpose::kGenerated);
    active.push_back(checkpoint->Id());
    (void)f.Publish(*checkpoint);
  }
  // The result names the retired entry and why it was chosen.
  const auto evicted = f.catalog.EvictOne();
  assert(evicted && evicted.superseded && !evicted.reused);
  assert(evicted.description->Manifest().checkpoint == active.front());
  assert(!f.disk.Open(active.front()));
  assert(f.disk.Open(idle->Id()));
  for (std::size_t i = 1; i < active.size(); ++i)
    assert(f.disk.Open(active[i]));
}
void PromptAnchorOutlivesCoveredAndLeafGeneratedFrontiers() {
  TieredFixture f;
  f.Append(8, 4);
  const auto prompt_cp = f.Capture(CheckpointPurpose::kPrompt);
  (void)f.Publish(*prompt_cp);
  std::vector<CheckpointId> active;
  for (int turn = 0; turn < 4; ++turn) {
    f.Append(4, 4);
    const auto checkpoint = f.Capture(CheckpointPurpose::kGenerated);
    active.push_back(checkpoint->Id());
    (void)f.Publish(*checkpoint);
  }
  // Turn 0 has 3 newer generated descendants: superseded (Field 0 = 0).
  assert(f.catalog.EvictOne());
  assert(!f.disk.Open(active[0]));
  assert(f.disk.Open(prompt_cp->Id()));
  // Turns 1 and 2 are covered intermediates (rank 0). Turn 2 has a private
  // tail while Turn 1 aligns to a chunk boundary, so Turn 2 releases more
  // unique bytes and leaves first.
  const auto covered = f.catalog.EvictOne();
  assert(covered && !covered.superseded && !covered.reused);
  assert(covered.description->Manifest().checkpoint == active[2]);
  assert(!f.disk.Open(active[2]));
  assert(f.disk.Open(active[1]));
  assert(f.disk.Open(active[3]));
  assert(f.disk.Open(prompt_cp->Id()));
  assert(f.catalog.EvictOne());
  assert(!f.disk.Open(active[1]));
  assert(f.disk.Open(active[3]));
  assert(f.disk.Open(prompt_cp->Id()));
  // Turn 3 is leaf (rank 1).
  assert(f.catalog.EvictOne());
  assert(!f.disk.Open(active[3]));
  // Prompt anchor is root (shared, rank 2): outlives all of them!
  assert(f.disk.Open(prompt_cp->Id()));
  assert(f.catalog.EvictOne());
  assert(!f.disk.Open(prompt_cp->Id()));
  assert(f.catalog.Size() == 0);
}
// Captures `tokens` on a fresh execution of the fixture slot.
std::shared_ptr<const Checkpoint> CaptureCold(
    TieredFixture& f, ExecutionHistory& history,
    const std::vector<Token>& tokens) {
  f.adapter.Append(*f.slot, tokens, tokens);
  auto checkpoint = history.Capture(
      {tokens, InputIdentity(tokens.size()), f.adapter.Positions(*f.slot),
       CheckpointPurpose::kPrompt, 1},
      [&](const PayloadRequest& request) {
        auto pool =
            f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                .Convert();
        auto bytes = std::make_shared<std::vector<std::byte>>(request.bytes);
        auto assigned = pool.ReserveBacking(request.category);
        auto completion =
            request.category == ResourceCategory::kPrivateState
                ? f.adapter.CapturePrivate(*f.slot, request.component, *bytes,
                                           f.stream)
                : f.adapter.CopyRowsOut(*f.slot, request.component,
                                        request.first, request.end, *bytes,
                                        f.stream);
        assert(completion.Wait() == TransferResult::kSucceeded);
        return Payload::Committed(assigned.Convert(), bytes);
      });
  assert(f.adapter.Invalidate(*f.slot));
  return checkpoint;
}
CheckpointId Evicted(TieredFixture& f) {
  const auto evicted = f.catalog.EvictOne();
  assert(evicted);
  return evicted.description->Manifest().checkpoint;
}
// #1254: a prefix published after its leaf must not outlive it. The
// conversation's covered prefix leaves first, then its leaf, then the root
// anchor. Restoring the prefix instead means the client did not match the
// leaf, so the restore keeps the prefix and lends the leaf nothing.
void LeafOutlivesItsLaterPrefix() {
  for (const bool restored : {false, true}) {
    TieredFixture f;
    f.Append(8, 4);
    const auto root = f.Capture();
    f.Append(4, 4);
    const auto prefix = f.Capture();
    f.Append(4, 4);
    const auto leaf = f.Capture();
    (void)f.Publish(*root);
    if (restored) {
      (void)f.Publish(*prefix);
      (void)f.Publish(*leaf);
      f.catalog.Touch(prefix->Id());
      assert(Evicted(f) == leaf->Id());
      assert(Evicted(f) == prefix->Id());
    } else {
      (void)f.Publish(*leaf);
      (void)f.Publish(*prefix);
      assert(Evicted(f) == prefix->Id());
      assert(f.disk.Open(leaf->Id()) && f.disk.Open(root->Id()));
      assert(Evicted(f) == leaf->Id());
    }
    assert(Evicted(f) == root->Id());
  }
}
// A prefix two branches share lends its admission to neither leaf: publishing
// it after both keeps no older leaf ahead of a newer conversation.
void SharedPrefixLendsNothingToItsBranches() {
  TieredFixture f;
  auto other = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                      f.adapter.CompatibilityIdentity());
  std::vector<Token> sibling_tokens(12);
  std::iota(sibling_tokens.begin(), sibling_tokens.begin() + 8, 7);
  std::iota(sibling_tokens.begin() + 8, sibling_tokens.end(), 15);
  sibling_tokens.insert(sibling_tokens.end(), {500, 501, 502, 503});
  const auto sibling = CaptureCold(f, other, sibling_tokens);
  auto unrelated_history = ExecutionHistory::Cold(
      f.ledger, f.adapter.Components(), f.adapter.CompatibilityIdentity());
  const auto unrelated =
      CaptureCold(f, unrelated_history, std::vector<Token>(8, 99));
  f.Append(8, 4);
  const auto root = f.Capture();
  f.Append(4, 4);
  const auto shared = f.Capture();
  assert(std::equal(f.tokens.begin(), f.tokens.end(), sibling_tokens.begin()));
  f.Append(4, 4);
  const auto leaf = f.Capture();
  for (const auto& checkpoint : {root, leaf, sibling, unrelated, shared})
    (void)f.Publish(*checkpoint);
  assert(Evicted(f) == leaf->Id());
  // Without the leaf, the prefix belongs to the sibling's conversation alone.
  assert(Evicted(f) == unrelated->Id());
  assert(Evicted(f) == shared->Id());
  assert(Evicted(f) == sibling->Id());
  assert(Evicted(f) == root->Id());
}
void DiskUseHistorySurvivesCatalogRestart() {
  TieredFixture f;
  f.Append(8, 4);
  auto first = f.Capture();
  (void)f.Publish(*first);
  f.Append(4, 4);
  auto second = f.Capture();
  (void)f.Publish(*second);
  f.catalog.FlushUses();
  const auto before = f.disk.LastUse(first->Id());
  f.catalog.Touch(first->Id());
  assert(f.disk.LastUse(first->Id()) == before);
  f.catalog.FlushUses();
  assert(f.disk.LastUse(first->Id()) > f.disk.LastUse(second->Id()));
  DiskCatalog recovered(f.ledger, f.disk, f.index);
  assert(recovered.EvictOne());
  assert(f.disk.Open(first->Id()) && !f.disk.Open(second->Id()));
  assert(recovered.EvictOne());
  assert(f.disk.Stats().managed_bytes == 0);
}
void DiskReuseExpiresAfterAByteTurnoverAndSurvivesRestart() {
  for (const std::uint64_t turnover : {1U, 1U << 20}) {
    TieredFixture f;
    CheckpointId first, second;
    {
      DiskCatalog catalog(f.ledger, f.disk, f.index, turnover);
      f.Append(8, 4);
      auto a = f.Capture();
      first = a->Id();
      (void)f.Publish(*a, false);
      catalog.Track(first);
      catalog.Touch(first);
      catalog.FlushUses();
      assert(f.adapter.Invalidate(*f.slot));
      auto other = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
      f.tokens.clear();
      f.drafts.clear();
      f.Append(8, 4);
      auto b = other.Capture(
          {f.tokens, InputIdentity(f.tokens.size()),
           f.adapter.Positions(*f.slot), CheckpointPurpose::kPrompt, 1},
          [&](const PayloadRequest& request) {
            auto pool =
                f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                    .Convert();
            auto bytes =
                std::make_shared<std::vector<std::byte>>(request.bytes);
            auto assigned = pool.ReserveBacking(request.category);
            auto completion =
                request.category == ResourceCategory::kPrivateState
                    ? f.adapter.CapturePrivate(*f.slot, request.component,
                                               *bytes, f.stream)
                    : f.adapter.CopyRowsOut(*f.slot, request.component,
                                            request.first, request.end, *bytes,
                                            f.stream);
            assert(completion.Wait() == TransferResult::kSucceeded);
            return Payload::Committed(assigned.Convert(), bytes);
          });
      second = b->Id();
      (void)f.Publish(*b, false);
      catalog.Track(second);
      catalog.FlushUses();
      const auto history = f.disk.UseHistory(first);
      assert(history.reused_at && history.admitted_bytes > *history.reused_at);
    }
    DiskCatalog recovered(f.ledger, f.disk, f.index, turnover);
    assert(recovered.EvictOne());
    // A recent reused conversation outlives newer untouched admissions, but
    // loses that protection after one budget of newly admitted unique bytes.
    assert(bool(f.disk.Open(first)) == (turnover > 1));
    assert(bool(f.disk.Open(second)) == (turnover == 1));
  }
}
void AUsedFrontierIsNotAnOptionalVictim() {
  TieredFixture f;
  RetentionPolicy policy(f.ledger, f.index);
  f.Append(8, 8);
  auto first = f.Capture(CheckpointPurpose::kGenerated);
  RetentionRequest request{f.tokens, f.adapter.CompatibilityIdentity(), {}};
  assert(policy.Admit(request, [&] { return first; }));
  assert(policy.Rank(first->Id()) == 0);
  policy.Touch(first->Id());
  assert(policy.Rank(first->Id()) == 3);
  request.optional_frontier = true;
  assert(!policy.Reclaim(request));
  f.Append(4, 4);
  auto second = f.Capture(CheckpointPurpose::kGenerated);
  request.tokens = f.tokens;
  assert(policy.Admit(request, [&] { return second; }));
  assert(policy.Rank(second->Id()) == 0);
  assert(policy.Reclaim(request));
  assert(policy.Size() == 1 && policy.Rank(first->Id()) == 3);
}
void DiskEvictionNeedsNoLedgerHeadroom() {
  TieredFixture f;
  f.Append(8, 4);
  const auto checkpoint = f.Capture();
  (void)f.Publish(*checkpoint);
  const auto spare =
      f.ledger.Limits().ram_bytes - f.ledger.Snapshot().ram_bytes;
  auto occupied =
      f.ledger.Reserve(ResourceCategory::kMetadata, spare).Convert();
  assert(f.ledger.Snapshot().ram_bytes == f.ledger.Limits().ram_bytes);
  assert(f.catalog.EvictOne());
  assert(f.catalog.Size() == 0 && f.disk.Stats().managed_bytes == 0);
}
void RecordPressureDropsUnusedFrontierBeforeRetry() {
  TieredFixture f;
  RetentionPolicy policy(f.ledger, f.index, nullptr, 3);
  f.Append(8, 4);
  auto stable = f.Capture();
  assert(policy.Admit({f.tokens, f.adapter.CompatibilityIdentity(), {}},
                      [&] { return stable; }));
  f.Append(4, 4);
  auto retry = f.Capture();
  RetentionRequest request{f.tokens,
                           f.adapter.CompatibilityIdentity(),
                           {},
                           RetentionPurpose::kRetry,
                           8};
  assert(policy.Admit(request, [&] { return retry; }));
  f.Append(4, 4);
  auto frontier = f.Capture(CheckpointPurpose::kGenerated);
  request.tokens = f.tokens;
  request.purpose = RetentionPurpose::kContinuation;
  assert(policy.Admit(request, [&] { return frontier; }));
  f.Append(4, 4);
  auto next = f.Capture();
  request.tokens = f.tokens;
  request.purpose = RetentionPurpose::kRetry;
  assert(policy.Admit(request, [&] { return next; }));
  assert(f.index.CachedPrefixTokens(
             {f.adapter.CompatibilityIdentity(), retry->Tokens(),
              InputIdentity(retry->Boundary())}) == retry->Boundary());
}
void MixedSourcesAndCompatibility() {
  TieredFixture f;
  f.Append(40, 36);
  auto checkpoint = f.Capture();
  auto description = f.Publish(*checkpoint);
  const auto entry = f.index.Insert(checkpoint, f.Resident());
  f.index.AttachDurable(entry, description);
  f.index.SetAvailability(
      entry, {{{1}, true, false}, {{2}, false, true}, {{3}, true, false}});
  {
    auto lookup = f.index.Lookup(f.Query());
    assert(lookup.reason == SelectionReason::kDeepestMixedCheckpoint);
    assert(!lookup.selected->UsesDisk({1}) && lookup.selected->UsesDisk({2}));
    assert(!lookup.selected->UsesDisk({3}));
    assert(lookup.selected->checkpoint == checkpoint &&
           lookup.selected->durable == description);
    assert(f.index.HasCheckpoint(f.Query(), checkpoint->Boundary()));
    f.index.SetAvailability(entry, f.Resident());
    // The retained source plan must survive a later availability replacement.
    assert(lookup.selected->UsesDisk({2}));
  }
  auto digest = f.digest;
  digest[1] = 1;
  f.index.Register({99}, f.adapter.Components(), digest);
  bool rejected{};
  try {
    (void)f.index.Insert(Identity{99}, description);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  auto components = std::vector<ComponentDescriptor>(
      f.adapter.Components().begin(), f.adapter.Components().end());
  components[1].layout_version++;
  f.index.Register({98}, components, f.digest);
  rejected = false;
  try {
    (void)f.index.Insert(Identity{98}, description);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
void PublicationCorruptionQuarantinesImmediately() {
  for (bool queued : {false, true}) {
    TieredFixture f;
    f.Append(16, 12);
    auto checkpoint = f.Capture();
    auto description = f.Publish(*checkpoint);
    auto entry = f.index.Insert(checkpoint, f.Resident());
    f.index.AttachDurable(entry, description);
    f.Append(8, 8);
    auto second = f.Capture();
    auto other = f.Publish(*second);
    (void)f.index.Insert(f.adapter.CompatibilityIdentity(), other);
    auto manifest = description->Manifest();
    manifest.checkpoint = {999};
    manifest.components.resize(1);
    const auto chunk = manifest.components[0].chunks[0];
    auto conflicting = manifest;
    conflicting.components[0].chunks[0].checksum ^= 1;
    TransferTiming conflict_timing;
    bool rejected{};
    try {
      f.streams.Write(TieredFile(4, 999), conflicting, {}, conflict_timing);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected && !description->Quarantined());
    assert(description->Pin() && f.index.Lookup(f.Query()).selected);

    {
      std::fstream file(
          f.directory.path / "v2/chunks" / DiskFileName(chunk.file),
          std::ios::binary | std::ios::in | std::ios::out);
      file.put('\xff');
    }
    if (queued) {
      PersistenceQueue queue(f.ledger, f.streams, 1, 1, [] { return true; });
      assert(queue.TrySubmit(
          PersistenceJob(f.ledger, TieredFile(4, 999), manifest, {})));
      queue.Drain();
      assert(queue.Stats().failed == 1);
    } else {
      TransferTiming timing;
      bool failed{};
      try {
        f.streams.Write(TieredFile(4, 999), manifest, {}, timing);
      } catch (const DiskDependencyError& error) {
        failed = !error.private_file && error.file == chunk.file;
      }
      assert(failed);
    }
    assert(description->Quarantined() && other->Quarantined());
    assert(!description->Pin() && !other->Pin());
    assert(!f.disk.Open(checkpoint->Id()) && !f.disk.Open(second->Id()));
    assert(!f.catalog.Find(checkpoint->Id()));
    assert(!f.index.Lookup(f.Query()).selected);
    assert(!f.index.HasCheckpoint(f.Query(), 0));
    // No worker mutates the caller-serialized prefix index/catalog. Cleanup
    // reconciles retained descriptions even after disk-layer invalidation.
    f.catalog.Reconcile();
    assert(f.catalog.ReclaimInvalid() == 2);
    assert(f.disk.Stats().managed_bytes == 0);
    assert(description->Quarantined());
  }
}
void RepublishedQuarantineCannotRehabilitateOldRAM() {
  for (bool direct_removal : {false, true}) {
    TieredFixture f;
    f.Append(16, 12);
    auto checkpoint = f.Capture();
    auto old = f.Publish(*checkpoint);
    auto entry = f.index.Insert(checkpoint, f.Resident());
    f.index.AttachDurable(entry, old);
    auto manifest = old->Manifest();
    manifest.checkpoint = {999};
    manifest.components.resize(1);
    const auto chunk = manifest.components[0].chunks[0];
    {
      std::fstream file(
          f.directory.path / "v2/chunks" / DiskFileName(chunk.file),
          std::ios::binary | std::ios::in | std::ios::out);
      file.put('\xff');
    }
    TransferTiming timing;
    bool failed{};
    try {
      f.streams.Write(TieredFile(4, 999), manifest, {}, timing);
    } catch (const DiskDependencyError&) {
      failed = true;
    }
    assert(failed && old->Quarantined());
    assert(!f.index.Lookup(f.Query()).selected);
    // Direct retirement/republication may precede caller reconciliation. Track
    // must erase the poisoned RAM record before dropping its old marker.
    if (direct_removal) {
      f.index.RemoveDurable(*old);
      assert(!f.index.Lookup(f.Query()).selected);
    }
    assert(f.disk.Retire(checkpoint->Id()));
    auto replacement = f.Publish(*checkpoint);
    assert(replacement->Epoch() != old->Epoch());
    assert(!replacement->Quarantined() && old->Quarantined());
    assert(!f.index.Lookup(f.Query()).selected);
    f.catalog.Reconcile();
    assert(!f.index.Lookup(f.Query()).selected);
    (void)f.index.Insert(f.adapter.CompatibilityIdentity(), replacement);
    assert(f.index.Lookup(f.Query()).selected->durable == replacement);
  }
}
void InvalidUnusedRAMComponentStillSelectsMixed() {
  TieredFixture f;
  LeasedSlot local(f.ledger, f.adapter, f.stream, SlotId{55});
  auto lease = local.Acquire();
  f.tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  f.drafts = {1, 2, 3, 4};
  f.adapter.Append(lease.Execution(), f.tokens, f.drafts);
  auto checkpoint = f.history.Capture(
      {f.tokens, InputIdentity(f.tokens.size()),
       f.adapter.Positions(lease.Execution()), CheckpointPurpose::kPrompt, 1},
      [&](const PayloadRequest& request) {
        auto admission =
            f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes);
        auto bytes = std::make_shared<std::vector<std::byte>>(request.bytes);
        auto pool = admission.Convert();
        auto assigned = pool.ReserveBacking(request.category);
        if (request.component == kTarget)
          return Payload::Borrowed(
              lease.Borrow(request.component, request.first, request.end,
                           std::move(assigned), bytes, *bytes));
        auto copy =
            request.category == ResourceCategory::kPrivateState
                ? f.adapter.CapturePrivate(lease.Execution(), request.component,
                                           *bytes, f.stream)
                : f.adapter.CopyRowsOut(lease.Execution(), request.component,
                                        request.first, request.end, *bytes,
                                        f.stream);
        assert(copy.Wait() == TransferResult::kSucceeded);
        return Payload::Committed(assigned.Convert(), bytes);
      });
  auto description = f.Publish(*checkpoint, true, &lease.Execution());
  auto entry = f.index.Insert(checkpoint, f.Resident());
  f.index.AttachDurable(entry, description);
  f.index.SetAvailability(
      entry, {{{1}, false, true}, {{2}, true, false}, {{3}, true, false}});
  assert(f.index.Lookup(f.Query()).selected);
  f.adapter.FailNextTransfer();
  lease.Reset();
  assert(!checkpoint->IsValid());
  assert(checkpoint->Components()[1].IsValid());
  assert(checkpoint->Components()[2].IsValid());
  auto lookup = f.index.Lookup(f.Query());
  assert(lookup.reason == SelectionReason::kDeepestMixedCheckpoint);
  assert(lookup.selected->UsesDisk(kTarget));
  assert(!lookup.selected->UsesDisk(kDraft));
  assert(!lookup.selected->UsesDisk({3}));
  // Also resolve a resident-preferred component to disk when its RAM payload
  // alone becomes invalid; the healthy resident components retain preference.
  f.index.SetAvailability(
      entry, {{{1}, true, true}, {{2}, true, false}, {{3}, true, false}});
  auto fallback = f.index.Lookup(f.Query());
  assert(fallback.selected && fallback.selected->UsesDisk(kTarget));
  assert(!fallback.selected->UsesDisk(kDraft));
}
void DurableAvailabilityAdmissionOrder() {
  // Ledger bookkeeping is explicitly outside payload admission. Measure its
  // allocations so the observer rejects any additional plan allocation.
  ResourceLedger bookkeeping{{4096, 4096, 4096, 4096}};
  bookkeeping.FailAfter(LedgerStep::kReserve, 0);
  watched_allocations = 0;
  watch_allocations = true;
  bool bookkeeping_failed{};
  try {
    (void)bookkeeping.Reserve(ResourceCategory::kMetadata, 1);
  } catch (const std::bad_alloc&) {
    bookkeeping_failed = true;
  }
  watch_allocations = false;
  assert(bookkeeping_failed);
  const auto bookkeeping_allocations = watched_allocations;
  for (bool attach : {false, true}) {
    for (auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
      TieredFixture f;
      f.Append(16, 12);
      auto checkpoint = f.Capture();
      auto description = f.Publish(*checkpoint);
      auto compatibility = f.adapter.CompatibilityIdentity();
      const auto entry =
          attach ? f.index.Insert(checkpoint, f.Resident()) : IndexEntryId{};
      const auto baseline = f.ledger.Snapshot();
      f.ledger.FailAfter(step, 0);
      watched_allocations = 0;
      watch_allocations = step == LedgerStep::kReserve;
      bool failed{};
      try {
        if (attach)
          f.index.AttachDurable(entry, description);
        else
          (void)f.index.Insert(std::move(compatibility), description);
      } catch (const std::bad_alloc&) {
        failed = true;
      }
      watch_allocations = false;
      f.ledger.ClearFaults();
      assert(failed);
      if (step == LedgerStep::kReserve) {
        assert(watched_allocations == bookkeeping_allocations);
        assert(f.ledger.Snapshot() == baseline);
      } else {
        assert(f.ledger.Snapshot().total_bytes == baseline.total_bytes);
      }
      auto lookup = f.index.Lookup(f.Query());
      if (attach) {
        assert(lookup.selected && lookup.selected->entry == entry);
        assert(!lookup.selected->durable);
      } else {
        assert(!lookup.selected);
      }
    }
  }
}
void CatalogAdmissionFailures() {
  for (auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    TieredFixture f;
    f.Append(16, 12);
    auto checkpoint = f.Capture();
    auto description = f.Publish(*checkpoint, false);
    const auto baseline = f.ledger.Snapshot().total_bytes;
    bool completed{};
    for (unsigned failure = 0; failure < 80 && !completed; ++failure) {
      f.ledger.FailAfter(step, failure);
      try {
        f.catalog.Track(checkpoint->Id());
        completed = true;
      } catch (const std::bad_alloc&) {
      }
      f.ledger.ClearFaults();
      if (!completed) {
        assert(f.catalog.Size() == 0 && f.catalog.MetadataBytes() == 0);
        assert(f.ledger.Snapshot().total_bytes == baseline);
      }
    }
    assert(completed && f.catalog.Size() == 1);
    assert(f.catalog.MetadataBytes() > 0);
    assert(f.catalog.EvictOne());
    assert(f.catalog.Size() == 0 && f.catalog.MetadataBytes() == 0);
  }
}
// Publishes a checkpoint that holds only its token list and one private file
// of its own, TieredFile(3, id), so metadata dominates the way a long
// conversation's does (#1329).
constexpr ComponentDescriptor kMetadataState{
    {1}, 1, ComponentKind::kPrivateState, 0, 0, 32};
void PublishMetadataCheckpoint(
    DiskStore& disk, std::uint64_t id, std::span<const Token> tokens,
    std::uint64_t lineage, std::uint8_t digest,
    CheckpointPurpose purpose = CheckpointPurpose::kPrompt,
    Identity input = {}) {
  static constexpr std::array<std::uint8_t, 32> kBytes{};
  DiskManifest manifest;
  manifest.checkpoint = {id};
  manifest.lineage = {lineage};
  manifest.compatibility[0] = digest;
  manifest.input = std::move(input);
  manifest.purpose = purpose;
  manifest.tokens.assign(tokens.begin(), tokens.end());
  const auto file = TieredFile(3, id);
  manifest.components.push_back(
      {kMetadataState,
       {{1}, tokens.size()},
       {},
       {},
       DiskPayload{file, kBytes.size(), DiskChecksum(kBytes)}});
  const std::array<DiskWriteBuffer, 1> buffers{{{true, file, kBytes}}};
  disk.Publish(TieredFile(4, id), manifest, buffers);
}
// Serving keeps a checkpoint's tokens only in the catalog's and the index's
// tries: kSeparate, released once tracked. kManifest keeps every token list
// in the store, which grows quadratically with a conversation.
struct MetadataStore {
  ResourceLedger ledger;
  TieredDirectory directory;
  DiskStore disk;
  PrefixIndex index{ledger};
  explicit MetadataStore(std::size_t ram_bytes,
                         DiskTokenStorage storage = DiskTokenStorage::kSeparate)
      : ledger({ram_bytes * 2, ram_bytes, 16, 1 << 20}),
        disk(ledger, directory.path, 1 << 24, storage) {}
  void Publish(std::uint64_t id, std::span<const Token> tokens,
               std::uint64_t lineage = 1) {
    PublishMetadataCheckpoint(disk, id, tokens, lineage, 42);
  }
  // Tracks a publication and releases the store's tokens, as serving does.
  void Track(DiskCatalog& catalog, std::uint64_t id,
             const DiskEvictionVisitor& retired = {}) {
    catalog.Track({id}, retired);
    disk.ReleaseTokens({id});
  }
  std::size_t Metadata() const {
    return ledger.Snapshot()
        .bytes[static_cast<std::size_t>(ResourceCategory::kMetadata)];
  }
};
void DiskMetadataAllowanceRule() {
  constexpr std::size_t kMiB = std::size_t{1} << 20;
  assert(DiskMetadataAllowance(0) == SIZE_MAX);
  assert(DiskMetadataAllowance(448 << 10) == 16 * kMiB);
  assert(DiskMetadataAllowance(256 * kMiB) == 16 * kMiB);
  assert(DiskMetadataAllowance(1024 * kMiB) == 32 * kMiB);
  assert(DiskMetadataAllowance(2048 * kMiB) == 64 * kMiB);
  assert(DiskMetadataAllowance(32768 * kMiB) == 1024 * kMiB);
}
void CatalogMetadataTracksAdmissionAndRetirement() {
  MetadataStore s(1 << 24);
  std::vector<Token> tokens(4096);
  std::iota(tokens.begin(), tokens.end(), 7);
  for (std::uint64_t id = 1; id <= 4; ++id)
    s.Publish(id, std::span(tokens).first(id * 512));
  DiskCatalog catalog(s.ledger, s.disk, s.index);
  s.disk.ReleaseTokens();
  // The four nested token lists share one trie path: their unique tokens
  // count once, not 1 + 2 + 3 + 4 times.
  const std::size_t unique = 4 * 512 * sizeof(Token);
  assert(catalog.MetadataBytes() > unique);
  assert(catalog.MetadataBytes() < unique + 4 * 2048 + 1024);
  // Retirement releases what the catalog counted for the entry, apart from
  // the store's entry slot, which stays as vector capacity, within the bound
  // the catalog assumes for the store's private availability record.
  while (catalog.Size()) {
    const auto counted = catalog.MetadataBytes();
    const auto charged = s.Metadata();
    assert(catalog.EvictOne());
    const auto released = charged - s.Metadata();
    const auto uncounted = counted - catalog.MetadataBytes();
    assert(released <= uncounted && uncounted - released <= 128);
  }
  assert(catalog.MetadataBytes() == 0);
}
void MetadataAllowanceEvictsInsteadOfFillingTheLedger() {
  // Checkpoint n holds n turns, each a prefix of the next; without the
  // allowance a store that keeps every token list (quadratic in the
  // conversation, as chunk references still are) outgrows the ledger's
  // metadata room.
  constexpr std::size_t kTurn = 2048, kTurns = 8, kRam = std::size_t{384} << 10,
                        kAllowance = 64 << 10, kCapture = 192 << 10;
  std::vector<Token> conversation(kTurn * kTurns);
  std::iota(conversation.begin(), conversation.end(), 7);
  for (const bool bounded : {false, true}) {
    MetadataStore s(kRam, DiskTokenStorage::kManifest);
    DiskCatalog catalog(s.ledger, s.disk, s.index, UINT64_MAX,
                        bounded ? kAllowance : SIZE_MAX);
    std::vector<std::uint64_t> evicted;
    bool refused{};
    for (std::uint64_t n = 1; n <= kTurns && !refused; ++n) {
      const auto tokens = std::span(conversation).first(n * kTurn);
      try {
        s.Publish(n, tokens);
        catalog.Track({n}, [&](const DiskEviction& eviction) {
          evicted.push_back(eviction.description->Manifest().checkpoint.value);
        });
        // A RAM capture of fixed size must still be admitted.
        (void)s.ledger.Reserve(ResourceCategory::kMetadata, kCapture);
      } catch (const ResourceExhausted&) {
        refused = true;
        continue;
      }
      assert(catalog.Find({n}));
      if (bounded)
        assert(catalog.MetadataBytes() <=
               kAllowance + tokens.size() * sizeof(Token) + 4096);
    }
    if (!bounded) {
      assert(refused && evicted.empty());
      continue;
    }
    assert(!refused && !evicted.empty());
    // One conversation: covered intermediates retire oldest first. Its first
    // checkpoint has no prefix, so it ranks as a shared anchor and outlives
    // them; the leaf stays.
    auto intermediates = evicted;
    std::erase(intermediates, 1);
    assert(!intermediates.empty() && intermediates.front() == 2);
    assert(std::ranges::is_sorted(intermediates));
    assert(std::ranges::adjacent_find(evicted) == evicted.end());
    assert(std::ranges::max(evicted) < kTurns && catalog.Find({kTurns}));
    assert(catalog.Size() + evicted.size() == kTurns);
  }
}
// Publishes `count` single-checkpoint conversations of 1024 tokens and
// returns an allowance that holds them with 256 bytes to spare.
std::size_t PublishConversations(MetadataStore& s, std::uint64_t count) {
  std::vector<Token> tokens(1024);
  std::iota(tokens.begin(), tokens.end(), 7);
  for (std::uint64_t id = 1; id <= count; ++id) {
    tokens[0] = static_cast<Token>(id);
    s.Publish(id, tokens, id);
  }
  return DiskCatalog(s.ledger, s.disk, s.index).MetadataBytes() + 256;
}
void PublishConversation(MetadataStore& s, std::uint64_t id) {
  std::vector<Token> tokens(1024);
  std::iota(tokens.begin(), tokens.end(), 7);
  tokens[0] = static_cast<Token>(id);
  s.Publish(id, tokens, id);
}
void MetadataAllowanceKeepsTheNewPublication() {
  // Restored conversations rank above a fresh publication, so retiring after
  // tracking would delete the checkpoint just written.
  MetadataStore s(1 << 24);
  const auto allowance = PublishConversations(s, 4);
  DiskCatalog catalog(s.ledger, s.disk, s.index, UINT64_MAX, allowance);
  s.disk.ReleaseTokens();
  assert(catalog.Size() == 4 && catalog.MetadataBytes() <= allowance);
  for (std::uint64_t id = 1; id <= 4; ++id)
    catalog.Touch({id});
  std::vector<std::uint64_t> evicted;
  const auto record = [&](const DiskEviction& eviction) {
    evicted.push_back(eviction.description->Manifest().checkpoint.value);
  };
  // The pass retires to an eighth of the allowance below it: two entries.
  PublishConversation(s, 5);
  s.Track(catalog, 5, [&](const DiskEviction& eviction) {
    assert(eviction.reused);
    record(eviction);
  });
  assert((evicted == std::vector<std::uint64_t>{1, 2}));
  assert(catalog.Find({5}) && s.disk.Open({5}));
  assert(catalog.MetadataBytes() <= allowance);
  // That slack admits the next publication without a pass.
  PublishConversation(s, 6);
  s.Track(catalog, 6, record);
  assert(evicted.size() == 2 && catalog.Size() == 4);
  // Known limitation: conversations restored within one disk turnover rank
  // above every unrestored one, so the next pass retires the two fresh
  // conversations before the restored ones.
  PublishConversation(s, 7);
  s.Track(catalog, 7, record);
  assert((evicted == std::vector<std::uint64_t>{1, 2, 5, 6}));
  assert(catalog.Find({3}) && catalog.Find({4}) && catalog.Find({7}));
}
void MetadataAllowancePassFailures() {
  MetadataStore s(1 << 24);
  const auto allowance = PublishConversations(s, 4);
  DiskCatalog catalog(s.ledger, s.disk, s.index, UINT64_MAX, allowance);
  s.disk.ReleaseTokens();
  std::vector<std::uint64_t> evicted;
  const auto record = [&](const DiskEviction& eviction) {
    evicted.push_back(eviction.description->Manifest().checkpoint.value);
  };
  // A busy store ends the pass; the publication is still tracked.
  PublishConversation(s, 5);
  {
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    s.disk.SetCrashHook([&](DiskPublicationStep step) {
      if (step == DiskPublicationStep::kPayloadWritten) {
        entered.set_value();
        released.wait();
      }
    });
    auto writer =
        std::async(std::launch::async, [&] { PublishConversation(s, 6); });
    entered.get_future().wait();
    s.Track(catalog, 5, record);
    release.set_value();
    writer.get();
    s.disk.SetCrashHook({});
  }
  assert(evicted.empty() && catalog.Find({5}) && catalog.Size() == 5);
  assert(catalog.MetadataBytes() > allowance);
  // A failure during the pass ends it; the publication is still tracked.
  bool thrown{};
  s.Track(catalog, 6, [&](const DiskEviction& eviction) {
    record(eviction);
    thrown = true;
    throw std::runtime_error("visitor failed");
  });
  assert(thrown && (evicted == std::vector<std::uint64_t>{1}));
  assert(catalog.Find({6}) && catalog.Size() == 5);
  // A reader-pinned candidate is skipped and the next one retires.
  {
    const auto pin = catalog.Find({2})->Pin();
    assert(pin);
    assert(catalog.FitMetadata(record) == 2);
  }
  assert((evicted == std::vector<std::uint64_t>{1, 3, 4}));
  assert(catalog.Find({2}) && catalog.MetadataBytes() <= allowance);
  // Accounting stayed consistent through every failure.
  while (catalog.EvictOne()) {
  }
  assert(catalog.Size() == 0 && catalog.MetadataBytes() == 0);
}
void StartupFitsRecoveredMetadata() {
  MetadataStore s(1 << 24);
  const auto allowance = (PublishConversations(s, 4) - 256) / 2;
  DiskCatalog catalog(s.ledger, s.disk, s.index, UINT64_MAX, allowance);
  s.disk.ReleaseTokens();
  assert(catalog.Size() == 4 && catalog.MetadataBytes() > allowance);
  std::size_t seen{};
  const auto retired =
      catalog.FitMetadata([&](const DiskEviction&) { ++seen; });
  // Two entries fit the allowance; the slack below it retires a third.
  assert(retired == 3 && seen == retired && catalog.Size() == 1);
  assert(catalog.MetadataBytes() <= allowance);
  assert(catalog.FitMetadata() == 0);
}
// Tracking released tokens fails loudly, before the allowance pass retires
// anything, and leaves the catalog as it was.
void CatalogTrackRequiresTokens() {
  MetadataStore s(1 << 24);
  std::vector<Token> tokens(64);
  std::iota(tokens.begin(), tokens.end(), 7);
  s.Publish(1, tokens);
  const auto allowance = DiskCatalog(s.ledger, s.disk, s.index).MetadataBytes();
  DiskCatalog catalog(s.ledger, s.disk, s.index, UINT64_MAX, allowance + 16);
  s.disk.ReleaseTokens();
  tokens[0] = 99;
  s.Publish(2, tokens, 2);
  s.disk.ReleaseTokens({2});
  const auto before = catalog.MetadataBytes();
  bool thrown{};
  try {
    catalog.Track({2});
  } catch (const std::logic_error&) {
    thrown = true;
  }
  assert(thrown && catalog.Size() == 1 && catalog.Find({1}));
  assert(catalog.MetadataBytes() == before);
  // A catalog over released tokens cannot be built either.
  thrown = false;
  try {
    DiskCatalog reopened(s.ledger, s.disk, s.index);
  } catch (const std::logic_error&) {
    thrown = true;
  }
  assert(thrown);
}
// The trie answers prefix questions about tracked publications without the
// store's tokens: equal and strict prefixes, never longer, diverged,
// untracked, retired or re-published ones. Quarantine is Pin's business.
void CatalogIsPrefixOf() {
  MetadataStore s(1 << 24);
  std::vector<Token> b(150);
  std::iota(b.begin(), b.end(), 7);
  const auto a = std::span<const Token>(b).first(100);
  auto c = std::vector<Token>(b.begin(), b.begin() + 120);
  c[60] = 1;  // diverges inside a's range
  auto d = std::vector<Token>(b.begin(), b.begin() + 100);
  d.push_back(2);  // diverges right after a
  s.Publish(1, a);
  s.Publish(2, b);
  s.Publish(3, c, 2);
  s.Publish(4, d, 3);
  DiskCatalog catalog(s.ledger, s.disk, s.index);
  s.disk.ReleaseTokens();
  const auto epoch = [&](std::uint64_t id) {
    return catalog.Find({id})->Epoch();
  };
  assert(catalog.IsPrefixOf({1}, epoch(1), a));
  assert(catalog.IsPrefixOf({1}, epoch(1), b));
  assert(catalog.IsPrefixOf({1}, epoch(1), d));
  assert(!catalog.IsPrefixOf({1}, epoch(1), c));
  assert(!catalog.IsPrefixOf({2}, epoch(2), a));
  assert(!catalog.IsPrefixOf({2}, epoch(2), c));
  assert(!catalog.IsPrefixOf({4}, epoch(4), b));
  assert(catalog.IsPrefixOf({3}, epoch(3), c));
  assert(!catalog.IsPrefixOf({1}, epoch(1) + 100, b));
  assert(!catalog.IsPrefixOf({9}, 1, b));
  assert(catalog.IsPrefixOf({1}, epoch(1), std::span(b).first(100)));
  assert(!catalog.IsPrefixOf({1}, epoch(1), std::span(b).first(99)));
  // A quarantined record still answers; its Pin fails instead.
  const auto quarantined = epoch(3);
  catalog.Invalidate(true, TieredFile(3, 3));
  assert(catalog.IsPrefixOf({3}, quarantined, c) && !catalog.Find({3}));
  // Re-published under the same ID: only the new epoch and tokens answer.
  const auto old = epoch(1);
  assert(s.disk.Retire({1}));
  std::vector<Token> other(80, 5);
  s.Publish(1, other);
  s.Track(catalog, 1);
  assert(epoch(1) != old && !catalog.IsPrefixOf({1}, old, b));
  assert(catalog.IsPrefixOf({1}, epoch(1), other));
  assert(!catalog.IsPrefixOf({1}, epoch(1), b));
  // Retired: no longer tracked. Each record answers its own epoch and tokens
  // until it leaves, so only retirement can turn the answer.
  const std::array<std::span<const Token>, 4> own{other, b, c, d};
  const std::array<std::uint64_t, 4> epochs{epoch(1), epoch(2), quarantined,
                                            epoch(4)};
  for (std::uint64_t id = 1; id <= 4; ++id)
    assert(catalog.IsPrefixOf({id}, epochs[id - 1], own[id - 1]));
  while (catalog.Size() > 1)
    (void)catalog.EvictOne();
  const auto survivor = catalog.RankForTesting().front().id;
  for (std::uint64_t id = 1; id <= 4; ++id)
    assert(catalog.IsPrefixOf({id}, epochs[id - 1], own[id - 1]) ==
           (id == survivor));
}
// Token lists live once per digest, so a conversation's disk metadata grows
// linearly with its checkpoints. A store that keeps every list grows
// quadratically, as MetadataAllowanceEvictsInsteadOfFillingTheLedger shows.
void DiskMetadataGrowsLinearlyWithCheckpoints() {
  constexpr std::size_t kTurn = 512;
  static constexpr ComponentDescriptor kState{
      {1}, 1, ComponentKind::kPrivateState, 0, 0, 32};
  CompatibilityDigest digest{};
  digest[0] = 42;
  const auto measure = [&](std::size_t turns) {
    MetadataStore s(1 << 24, DiskTokenStorage::kSeparate);
    s.index.Register({9}, std::span(&kState, 1), digest);
    DiskCatalog catalog(s.ledger, s.disk, s.index);
    std::vector<Token> conversation(turns * kTurn);
    std::iota(conversation.begin(), conversation.end(), 7);
    const auto before = s.Metadata();
    std::vector<IndexEntryId> entries;
    for (std::uint64_t n = 1; n <= turns; ++n) {
      s.Publish(n, std::span(conversation).first(n * kTurn));
      catalog.Track({n});
      entries.push_back(s.index.Insert({9}, catalog.Find({n})));
      s.disk.ReleaseTokens({n});
    }
    const auto grown = s.Metadata() - before;
    assert(catalog.MetadataBytes() <=
           turns * 2048 + conversation.size() * sizeof(Token));
    for (const auto entry : entries)
      s.index.Erase(entry);
    return grown;
  };
  const auto grown = measure(4), grown_twice = measure(8);
  assert(grown_twice <= grown * 22 / 10 + 4096);
}
// A restart where the allowance bound at shutdown: recovery hands each
// manifest's tokens to the catalog's tries as it decodes, so its peak stays
// near the allowance instead of every retained checkpoint's token list.
void RecoveryHandsTokensToTheCatalog() {
  constexpr std::size_t kTurn = 1024, kTurns = 20, kAllowance = 92 << 10;
  static constexpr ComponentDescriptor kState{
      {1}, 1, ComponentKind::kPrivateState, 0, 0, 32};
  CompatibilityDigest digest{};
  digest[0] = 42;
  std::vector<Token> conversation(kTurn * kTurns);
  std::iota(conversation.begin(), conversation.end(), 7);
  TieredDirectory directory;
  std::size_t evicted{};
  {
    ResourceLedger ledger({1 << 25, 1 << 24, 16, 1 << 20});
    DiskStore disk(ledger, directory.path, 1 << 24,
                   DiskTokenStorage::kSeparate);
    PrefixIndex index(ledger);
    DiskCatalog catalog(ledger, disk, index, UINT64_MAX, kAllowance);
    for (std::uint64_t n = 1; n <= kTurns; ++n) {
      PublishMetadataCheckpoint(
          disk, n, std::span(conversation).first(n * kTurn), 1, 42);
      catalog.Track({n}, [&](const DiskEviction&) { ++evicted; });
      disk.ReleaseTokens({n});
      assert(catalog.MetadataBytes() <=
             kAllowance + kTurn * sizeof(Token) + 8192);
    }
  }
  assert(evicted > 0);
  const auto metadata_peak = [](const ResourceLedger& ledger) {
    return ledger.Snapshot()
        .peak_bytes[static_cast<std::size_t>(ResourceCategory::kMetadata)];
  };
  std::size_t retained_tokens{}, manifest_peak{};
  {
    ResourceLedger ledger({1 << 25, 1 << 24, 16, 1 << 20});
    DiskStore disk(ledger, directory.path, 1 << 24);
    for (const auto& entry : disk.Entries())
      retained_tokens += entry.TokenCount() * sizeof(Token);
    PrefixIndex index(ledger);
    DiskCatalog catalog(ledger, disk, index, UINT64_MAX, kAllowance);
    manifest_peak = metadata_peak(ledger);
  }
  // The directory holds several times the allowance in token lists.
  assert(retained_tokens > 3 * kAllowance && manifest_peak > retained_tokens);
  ResourceLedger ledger({1 << 25, 1 << 24, 16, 1 << 20});
  DiskCatalogTokens recovered(ledger);
  DiskStore disk(ledger, directory.path, 1 << 24, DiskTokenStorage::kSeparate,
                 recovered.Sink());
  assert(recovered.TokenBytes() == kTurns * kTurn * sizeof(Token));
  PrefixIndex index(ledger);
  index.Register({9}, std::span(&kState, 1), digest);
  DiskCatalog catalog(ledger, disk, index, UINT64_MAX, kAllowance, &recovered);
  assert(catalog.Size() == disk.Entries().size());
  assert(catalog.MetadataBytes() <= kAllowance);
  // Peak: what the catalog holds, the store's entry slots and one manifest
  // being decoded.
  assert(metadata_peak(ledger) <=
         kAllowance + 64 * 1024 + 2 * kTurns * kTurn * sizeof(Token));
  assert(metadata_peak(ledger) * 2 < manifest_peak);
  // The index rebuilds every recovered path from the trie.
  for (const auto& entry : disk.Entries()) {
    const auto id = entry.Manifest().checkpoint;
    assert(disk.Tokens(id).empty());
    assert(catalog.VisitTokens(id, [&](std::span<const Token> tokens) {
      assert(std::ranges::equal(
          tokens, std::span(conversation).first(entry.TokenCount())));
      (void)index.Insert({9}, catalog.Find(id), tokens);
    }));
  }
  const auto lookup =
      index.Lookup({{9}, conversation, InputIdentity(conversation.size()), 0});
  assert(lookup.selected && lookup.selected->boundary == conversation.size());
}
// Recovered tokens attach by store epoch, which another store reuses: a
// catalog over another store refuses them rather than track a wrong path.
void CatalogRefusesAnotherStoresTokens() {
  std::vector<Token> tokens(100);
  std::iota(tokens.begin(), tokens.end(), 7);
  TieredDirectory recovered_from;
  {
    ResourceLedger ledger({1 << 25, 1 << 24, 16, 1 << 20});
    DiskStore disk(ledger, recovered_from.path, 1 << 24,
                   DiskTokenStorage::kSeparate);
    PublishMetadataCheckpoint(disk, 1, tokens, 1, 42);
  }
  // The same epoch and digest under another checkpoint, then the same
  // checkpoint with other tokens.
  for (const auto& [id, count] :
       {std::pair<std::uint64_t, std::size_t>{2, 100}, {1, 60}}) {
    ResourceLedger ledger({1 << 25, 1 << 24, 16, 1 << 20});
    DiskCatalogTokens recovered(ledger);
    DiskStore source(ledger, recovered_from.path, 1 << 24,
                     DiskTokenStorage::kSeparate, recovered.Sink());
    const auto recovered_epoch = source.Entries().front().Epoch();
    TieredDirectory other;
    DiskStore disk(ledger, other.path, 1 << 24, DiskTokenStorage::kSeparate);
    PublishMetadataCheckpoint(disk, id, std::span(tokens).first(count), 1, 42);
    assert(disk.Entries().front().Epoch() == recovered_epoch);
    PrefixIndex index(ledger);
    bool thrown{};
    try {
      DiskCatalog catalog(ledger, disk, index, UINT64_MAX, SIZE_MAX,
                          &recovered);
    } catch (const std::logic_error&) {
      thrown = true;
    }
    assert(thrown);
  }
}
// Durable-only generated frontiers keep their tokens only in the index tree;
// the visit rebuilds them, in entry order, across edge splits and merges.
void VisitGeneratedRebuildsDurableOnlyPaths() {
  TieredFixture f(DiskTokenStorage::kSeparate);
  const auto identity = f.adapter.CompatibilityIdentity();
  std::vector<std::shared_ptr<const Checkpoint>> checkpoints;
  std::vector<IndexEntryId> entries;
  for (unsigned turn = 0; turn < 2; ++turn) {
    f.Append(16, 12);
    checkpoints.push_back(f.Capture(CheckpointPurpose::kGenerated));
    const auto durable = f.Publish(*checkpoints.back());
    entries.push_back(f.index.Insert(checkpoints.back(), f.Resident()));
    f.index.AttachDurable(entries.back(), durable);
  }
  f.disk.ReleaseTokens();
  for (const auto entry : entries)
    f.index.DropResident(entry);
  const auto visit = [&] {
    std::vector<std::vector<Token>> seen;
    f.index.VisitGenerated(identity, [&](std::span<const Token> tokens) {
      // Reentrant const lookups stay valid.
      (void)f.index.CachedPrefixTokens(
          {identity, tokens, InputIdentity(tokens.size()), 0});
      seen.emplace_back(tokens.begin(), tokens.end());
    });
    return seen;
  };
  const std::vector<std::vector<Token>> expected{
      {checkpoints[0]->Tokens().begin(), checkpoints[0]->Tokens().end()},
      {checkpoints[1]->Tokens().begin(), checkpoints[1]->Tokens().end()}};
  assert(visit() == expected);
  // A frontier diverging inside the first path splits its edge.
  std::vector<Token> fork(checkpoints[0]->Tokens().begin(),
                          checkpoints[0]->Tokens().begin() + 10);
  fork.push_back(1);
  const auto live = f.index.Insert(LiveFrontier{{}, fork, identity, {}, {}});
  assert(visit() == expected);
  // Erasing it merges the edge back.
  f.index.Erase(live);
  assert(visit() == expected);
  // Alignment is best effort: a full ledger skips rebuilt paths.
  f.ledger.FailAfter(LedgerStep::kReserve, 0);
  assert(visit().empty());
  f.ledger.ClearFaults();
  assert(visit() == expected);
}
// A RAM checkpoint attaches its durable copy by comparing tokens, so it
// fails loudly once the store released them; a durable record needs none.
void AttachDurableRejectsReleasedTokens() {
  TieredFixture f(DiskTokenStorage::kSeparate);
  const auto identity = f.adapter.CompatibilityIdentity();
  f.Append(16, 12);
  const auto checkpoint = f.Capture();
  const auto durable = f.Publish(*checkpoint);
  const auto entry = f.index.Insert(checkpoint, f.Resident());
  f.disk.ReleaseTokens();
  bool thrown{};
  try {
    f.index.AttachDurable(entry, durable);
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown && !f.index.IsDurable(entry));
  thrown = false;
  try {
    (void)f.index.Insert(identity, durable);
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown);
  // With the tokens attested, the durable record indexes and re-attaches.
  const auto durable_entry = f.index.Insert(identity, durable, f.tokens);
  f.index.AttachDurable(durable_entry, durable);
  assert(f.index.IsDurable(durable_entry));
  // Attested tokens must match the publication's count.
  thrown = false;
  try {
    (void)f.index.Insert(identity, durable,
                         std::span<const Token>(f.tokens).first(3));
  } catch (const std::invalid_argument&) {
    thrown = true;
  }
  assert(thrown);
}
// What the token-list ranking reads from one checkpoint's manifest, and
// whether the test quarantined it since the store last opened.
struct OracleCheckpoint {
  std::vector<Token> tokens;
  std::uint64_t lineage{};
  Identity input;
  std::uint8_t digest{};
  CheckpointPurpose purpose{};
  bool invalid{};
};
using OracleModel = std::map<std::uint64_t, OracleCheckpoint>;
// Verbatim copy of DiskCatalog's token-list ranking (Below, branch counting
// and Order) as #1471 left it, over the model. Clocks and unique payload
// bytes come from the catalog's records; they read no tokens. Returns the
// records with the model's validity and the oracle's branch and order.
bool OracleBelow(const OracleCheckpoint& a, const OracleCheckpoint& b) {
  return a.digest == b.digest &&
         (a.input == b.input || a.lineage == b.lineage) &&
         a.tokens.size() < b.tokens.size() &&
         std::equal(a.tokens.begin(), a.tokens.end(), b.tokens.begin());
}
std::vector<DiskCatalogRankForTesting> OracleRank(
    std::vector<DiskCatalogRankForTesting> ranks, const OracleModel& model,
    std::uint64_t admitted, std::uint64_t turnover) {
  assert(ranks.size() == model.size());
  std::vector<const OracleCheckpoint*> checkpoints;
  for (auto& rank : ranks) {
    checkpoints.push_back(&model.at(rank.id));
    rank.invalid = checkpoints.back()->invalid;
    rank.branch = 0;
  }
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    if (ranks[i].invalid)
      continue;
    const auto& manifest = *checkpoints[i];
    for (std::size_t j = 0; j < i; ++j) {
      if (ranks[j].invalid)
        continue;
      const auto& peer_manifest = *checkpoints[j];
      const auto size = manifest.tokens.size(),
                 peer_size = peer_manifest.tokens.size();
      if (size == peer_size ? peer_manifest.digest == manifest.digest &&
                                  peer_manifest.tokens == manifest.tokens
          : size < peer_size ? OracleBelow(manifest, peer_manifest)
                             : OracleBelow(peer_manifest, manifest)) {
        ++ranks[i].branch;
        ++ranks[j].branch;
      }
    }
  }
  for (std::size_t i = 0; i < ranks.size(); ++i) {
    auto& record = ranks[i];
    const auto& manifest = *checkpoints[i];
    auto recency = record.used;
    bool reused = record.reused_at && admitted - *record.reused_at < turnover;
    bool covered = false, rooted = false,
         fork = manifest.purpose == CheckpointPurpose::kLearned;
    std::optional<Token> next;
    std::size_t newer = 0;
    for (std::size_t j = 0; j < ranks.size(); ++j) {
      const auto& peer = ranks[j];
      if (peer.invalid)
        continue;
      const auto& peer_manifest = *checkpoints[j];
      if (OracleBelow(peer_manifest, manifest)) {
        rooted = true;
        if (peer_manifest.purpose != CheckpointPurpose::kLearned &&
            peer.branch == record.branch)
          recency = std::max(recency, peer.published);
      }
      if (!OracleBelow(manifest, peer_manifest))
        continue;
      covered = true;
      recency = std::max(recency, peer.used);
      reused |= peer.reused_at && admitted - *peer.reused_at < turnover;
      const auto token = peer_manifest.tokens[manifest.tokens.size()];
      fork |= next && *next != token;
      next = token;
      newer += peer_manifest.lineage == manifest.lineage &&
               peer_manifest.purpose == CheckpointPurpose::kGenerated;
    }
    const bool superseded = manifest.purpose == CheckpointPurpose::kGenerated &&
                            !fork && newer >= 3;
    const bool shared = fork || (covered && !rooted);
    record.order = {
        superseded ? 0U
        : reused   ? 2U
                   : 1U,
        superseded ? 0 : recency,
        shared    ? 2U
        : covered ? 0U
                  : 1U,
        std::numeric_limits<std::uint64_t>::max() - record.unique_bytes,
        record.id};
  }
  return ranks;
}
// Ranked records, best candidate first.
std::vector<const DiskCatalogRankForTesting*> Sorted(
    const std::vector<DiskCatalogRankForTesting>& ranks) {
  std::vector<const DiskCatalogRankForTesting*> sorted;
  for (const auto& rank : ranks)
    sorted.push_back(&rank);
  std::ranges::sort(sorted, {}, &DiskCatalogRankForTesting::order);
  return sorted;
}
// A restartable store of metadata checkpoints, each with a private file of
// its own, and the token lists the catalog ranks them by. Its prefix index
// holds what the server's would: every durable checkpoint, inserted on
// publication and on startup, and index-only live frontiers that split and
// extend the catalog's token paths.
struct OracleStore {
  struct Live {
    std::uint8_t digest{};
    std::vector<Token> tokens;
    IndexEntryId entry;
  };
  ResourceLedger ledger{{1 << 26, 1 << 25, 16, 1 << 20}};
  TieredDirectory directory;
  std::uint64_t turnover;
  std::size_t allowance{SIZE_MAX};
  std::unique_ptr<DiskStore> disk;
  std::unique_ptr<PrefixIndex> index;
  std::unique_ptr<DiskCatalog> catalog;
  OracleModel model;
  std::vector<Live> live;
  std::size_t index_baseline{};
  std::uint64_t next_id{1};
  // Full-ledger retirements whose index record leaves a record-less unary
  // node, which the index cannot merge without headroom.
  std::size_t unmerged{};
  // kSeparate runs as serving does: recovery hands its tokens to the
  // catalog, and each publication releases the store's copy once indexed.
  DiskTokenStorage storage;
  explicit OracleStore(std::uint64_t turnover_bytes,
                       DiskTokenStorage token_storage)
      : turnover(turnover_bytes), storage(token_storage) {
    Open();
  }
  void Open() {
    std::optional<DiskCatalogTokens> recovered;
    if (storage == DiskTokenStorage::kSeparate)
      recovered.emplace(ledger);
    disk = std::make_unique<DiskStore>(
        ledger, directory.path, 1 << 24, storage,
        recovered ? recovered->Sink() : DiskRecoveredTokens{});
    index = std::make_unique<PrefixIndex>(ledger);
    for (const std::uint8_t digest : {42, 43}) {
      CompatibilityDigest compatibility{};
      compatibility[0] = digest;
      index->Register(Identity{digest}, std::span(&kMetadataState, 1),
                      compatibility);
    }
    index_baseline = index->MetadataBytes();
    catalog = std::make_unique<DiskCatalog>(ledger, *disk, *index, turnover,
                                            allowance,
                                            recovered ? &*recovered : nullptr);
    // As the server starts: fit the allowance with one ranking, then index
    // what remains, from the catalog's trie once the store holds no tokens.
    const auto ranks = Check();
    std::vector<std::uint64_t> retired;
    (void)catalog->FitMetadata([&](const DiskEviction& eviction) {
      retired.push_back(eviction.description->Manifest().checkpoint.value);
    });
    Retired(ranks, retired);
    for (const auto& [id, checkpoint] : model) {
      const Identity identity{checkpoint.digest};
      if (storage == DiskTokenStorage::kManifest)
        (void)index->Insert(identity, catalog->Find({id}));
      else if (!catalog->VisitTokens({id}, [&](std::span<const Token> tokens) {
                 (void)index->Insert(identity, catalog->Find({id}), tokens);
               }))
        std::abort();
    }
    Check();
  }
  // Flushes use history and reopens every layer over the same directory,
  // under a new metadata allowance.
  void Restart(std::size_t metadata_allowance = SIZE_MAX) {
    catalog.reset();
    live.clear();
    index.reset();
    disk.reset();
    for (auto& [id, checkpoint] : model) {
      (void)id;
      checkpoint.invalid = false;
    }
    allowance = metadata_allowance;
    Open();
  }
  // Ranks and reconciles, and checks every record against the oracle.
  std::vector<DiskCatalogRankForTesting> Check() {
    auto ranks = catalog->RankForTesting();
    const auto expected =
        OracleRank(ranks, model, catalog->AdmittedBytesForTesting(), turnover);
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      assert(ranks[i].invalid == expected[i].invalid);
      assert(ranks[i].branch == expected[i].branch);
      assert(ranks[i].order == expected[i].order);
    }
    return ranks;
  }
  // The oracle's ranking of the records as they stand, without ranking or
  // reconciling them in the catalog first.
  std::vector<DiskCatalogRankForTesting> Predict() const {
    return OracleRank(catalog->RecordsForTesting(), model,
                      catalog->AdmittedBytesForTesting(), turnover);
  }
  // A metadata pass ranks once, then retires along that ranking.
  void Retired(const std::vector<DiskCatalogRankForTesting>& ranks,
               const std::vector<std::uint64_t>& retired) {
    const auto sorted = Sorted(ranks);
    assert(retired.size() <= sorted.size());
    for (std::size_t i = 0; i < retired.size(); ++i) {
      assert(retired[i] == sorted[i]->id);
      model.erase(retired[i]);
    }
  }
  std::uint64_t Publish(OracleCheckpoint checkpoint) {
    const auto id = next_id++;
    PublishMetadataCheckpoint(*disk, id, checkpoint.tokens, checkpoint.lineage,
                              checkpoint.digest, checkpoint.purpose,
                              checkpoint.input);
    const auto ranks = Check();
    std::vector<std::uint64_t> retired;
    catalog->Track({id}, [&](const DiskEviction& eviction) {
      retired.push_back(eviction.description->Manifest().checkpoint.value);
    });
    Retired(ranks, retired);
    // As the server does once a publication with no RAM copy completes.
    (void)index->Insert(Identity{checkpoint.digest}, catalog->Find({id}));
    disk->ReleaseTokens({id});
    model.emplace(id, std::move(checkpoint));
    Check();
    return id;
  }
  void AddLive(std::uint8_t digest, std::vector<Token> tokens) {
    const auto count = tokens.size();
    const auto entry = index->Insert(LiveFrontier{{SlotId{live.size() + 1}, 1},
                                                  tokens,
                                                  Identity{digest},
                                                  {},
                                                  {{{1}, count}},
                                                  true});
    live.push_back({digest, std::move(tokens), entry});
  }
  void Touch(std::uint64_t id) {
    catalog->Touch({id});
    Check();
  }
  // Quarantines through the catalog, or in the store alone, which the
  // catalog learns of only when it next ranks.
  void Quarantine(std::uint64_t id, bool store_only = false) {
    if (store_only)
      disk->Invalidate({id}, disk->Describe({id})->Epoch());
    else
      catalog->Invalidate(true, TieredFile(3, id));
    model.at(id).invalid = true;
  }
  // Whether erasing `id`'s index record leaves a record-less node with a
  // single child: the index's other records, at `id`'s tokens, continue
  // with one next token and none ends there.
  bool LeavesUnaryNode(std::uint64_t id) const {
    const auto& retired = model.at(id);
    if (retired.invalid || retired.tokens.empty())
      return false;
    std::set<Token> next;
    const auto ends = [&](std::uint8_t digest,
                          const std::vector<Token>& tokens) {
      if (digest != retired.digest || tokens.size() < retired.tokens.size() ||
          !std::equal(retired.tokens.begin(), retired.tokens.end(),
                      tokens.begin()))
        return false;
      if (tokens.size() == retired.tokens.size())
        return true;
      next.insert(tokens[retired.tokens.size()]);
      return false;
    };
    for (const auto& [peer_id, peer] : model)
      if (peer_id != id && !peer.invalid && ends(peer.digest, peer.tokens))
        return false;
    for (const auto& frontier : live)
      if (ends(frontier.digest, frontier.tokens))
        return false;
    return next.size() == 1;
  }
  // Retires the oracle's best candidate, or the next one while a reader pins
  // it. A full ledger must not change what retires. Unless `rank_first` checks
  // the ranking first, the catalog's own retirement is the first to rank and
  // reconcile.
  void Evict(bool pin_first = false, bool full_ledger = false,
             bool rank_first = true) {
    const auto ranks = rank_first ? Check() : Predict();
    const auto sorted = Sorted(ranks);
    assert(!sorted.empty());
    DiskReadPin pin;
    auto expected = sorted.front();
    if (pin_first) {
      auto pinned = std::ranges::find_if(
          sorted, [](const auto* rank) { return !rank->invalid; });
      assert(pinned != sorted.end() && sorted.size() > 1);
      pin = catalog->Find({(*pinned)->id})->Pin();
      assert(pin);
      expected = pinned == sorted.begin() ? sorted[1] : sorted.front();
    }
    ResourceCharge occupied;
    if (full_ledger) {
      unmerged += LeavesUnaryNode(expected->id);
      const auto spare =
          ledger.Limits().ram_bytes - ledger.Snapshot().ram_bytes;
      occupied = ledger.Reserve(ResourceCategory::kMetadata, spare).Convert();
    }
    const auto eviction = catalog->EvictOne();
    occupied = {};
    assert(eviction);
    assert(eviction.description->Manifest().checkpoint.value == expected->id);
    assert(eviction.superseded == (expected->order[0] == 0));
    assert(eviction.reused == (expected->order[0] == 2));
    model.erase(expected->id);
    Check();
  }
};
// Disk eviction ranks by token prefixes. Compare every record's branch count
// and full order, and every retirement, with a token-list model across forks,
// equal tokens under other ids, inputs, lineages and digests, every purpose,
// quarantine in the catalog and in the store alone, reuse within and past a
// turnover, pins, a full ledger, metadata passes along one ranking and a
// restart over a reopened store, while the prefix index holds the same
// checkpoints and live frontiers. Each store mode runs it: kSeparate holds
// tokens only in the tries, as serving does.
void EvictionOrderMatchesTokenOracle(DiskTokenStorage storage) {
  // Each checkpoint adds 32 unique payload bytes: reuse lasts six of them.
  OracleStore s(6 * 32, storage);
  std::mt19937_64 random(1329);
  const auto turn = [&](std::vector<Token> tokens) {
    const auto count = 8 + random() % 24;
    for (std::uint64_t i = 0; i < count; ++i)
      tokens.push_back(static_cast<Token>(random() % 50000));
    return tokens;
  };
  using P = CheckpointPurpose;
  const auto purpose = [&] {
    return std::array{P::kPrompt, P::kGenerated, P::kLearned,
                      P::kGrid}[random() % 4];
  };
  const auto system = turn({});
  // Conversation 1: a learned system prompt, then four generated turns, so
  // early turns are superseded, and a prompt leaf.
  std::vector<std::vector<Token>> one{system};
  std::vector<std::uint64_t> one_ids{
      s.Publish({system, 1, {1}, 42, P::kLearned})};
  for (int i = 0; i < 5; ++i) {
    one.push_back(turn(one.back()));
    one_ids.push_back(s.Publish(
        {one.back(), 1, {1}, 42, i < 4 ? P::kGenerated : P::kPrompt}));
  }
  s.Touch(one_ids[2]);
  // Conversation 2 forks conversation 1 after its second turn.
  auto two = turn(one[2]);
  const auto two_first = s.Publish({two, 2, {2}, 42, P::kGenerated});
  two = turn(two);
  (void)s.Publish({two, 2, {2}, 42, P::kPrompt});
  // The same tokens under another digest are a separate conversation.
  (void)s.Publish({one[1], 3, {1}, 43, P::kPrompt});
  (void)s.Publish({one[2], 3, {1}, 43, P::kGenerated});
  (void)s.Publish({system, 3, {1}, 43, P::kPrompt});
  // Equal tokens under other ids: same input, same lineage, neither.
  (void)s.Publish({one[3], 4, {1}, 42, P::kGenerated});
  (void)s.Publish({one[3], 1, {9}, 42, P::kPrompt});
  const auto unrelated = s.Publish({one[3], 5, {7}, 42, P::kGenerated});
  // An empty checkpoint sits below every same-input or same-lineage one.
  (void)s.Publish({{}, 6, {1}, 42, P::kPrompt});
  s.Touch(two_first);
  // Conversation 4 publishes its prefix after its leaf.
  const auto four_prefix = turn(system);
  const auto four_leaf = turn(four_prefix);
  (void)s.Publish({four_leaf, 7, {3}, 42, P::kPrompt});
  (void)s.Publish({four_prefix, 7, {3}, 42, P::kPrompt});
  // A learned prefix published after its leaf lends the leaf nothing.
  const auto five_prefix = turn(system);
  (void)s.Publish({turn(five_prefix), 13, {5}, 42, P::kPrompt});
  (void)s.Publish({five_prefix, 13, {5}, 42, P::kLearned});
  // A quarantined leaf, used last, neither covers nor renews its prefix.
  const auto six_prefix = turn(system);
  (void)s.Publish({six_prefix, 14, {6}, 42, P::kPrompt});
  const auto six_leaf = s.Publish({turn(six_prefix), 14, {6}, 42, P::kPrompt});
  // Conversation 7 sits below a prefix quarantined later, published last. A
  // grid leaf on a branch of its own takes nothing from it; a prompt whose
  // only prefix it is stays a shared anchor, and outlasts three generated
  // turns without a fork, as a prompt; the first of those turns has two
  // valid generated successors and a quarantined one that forks elsewhere,
  // so it is neither superseded nor a fork.
  const auto seven_prefix = turn({});
  const auto seven_prompt = turn(seven_prefix);
  const auto seven_turn = turn(seven_prompt);
  const auto seven_next = turn(seven_turn), seven_fork = turn(seven_turn);
  assert(seven_next[seven_turn.size()] != seven_fork[seven_turn.size()]);
  (void)s.Publish({turn(seven_prefix), 15, {8}, 42, P::kGrid});
  (void)s.Publish({seven_prompt, 15, {8}, 42, P::kPrompt});
  (void)s.Publish({seven_turn, 15, {8}, 42, P::kGenerated});
  (void)s.Publish({seven_next, 15, {8}, 42, P::kGenerated});
  (void)s.Publish({turn(seven_next), 15, {8}, 42, P::kGenerated});
  const auto seven_invalid =
      s.Publish({seven_fork, 15, {8}, 42, P::kGenerated});
  const auto seven_root = s.Publish({seven_prefix, 15, {8}, 42, P::kPrompt});
  // Live frontiers inside a catalog edge, past a leaf and on the other
  // digest exist only in the index.
  s.AddLive(42, {seven_next.begin(),
                 seven_next.begin() +
                     static_cast<std::ptrdiff_t>(seven_turn.size() + 3)});
  s.AddLive(42, turn(four_leaf));
  s.AddLive(43, {one[2].begin(), one[2].end() - 2});
  // Quarantine conversation 2's first checkpoint, conversation 6's leaf, in
  // reuse, and conversation 7's prefix and forked turn.
  s.Touch(six_leaf);
  for (const auto id : {two_first, six_leaf, seven_root, seven_invalid})
    s.Quarantine(id);
  s.Check();
  // Random conversations over both digests, with forks of earlier ones.
  std::vector<std::vector<Token>> heads{system, one[1], two, four_prefix};
  for (std::uint64_t lineage = 8; lineage < 10; ++lineage) {
    auto tokens = heads[random() % heads.size()];
    const std::uint8_t digest = random() % 4 ? 42 : 43;
    const Identity input{static_cast<std::uint8_t>(random() % 3)};
    for (int i = 0; i < 3; ++i) {
      tokens = turn(tokens);
      (void)s.Publish({tokens, lineage, input, digest, purpose()});
    }
    heads.push_back(tokens);
    s.Touch(one_ids[random() % one_ids.size()]);
  }
  s.Touch(unrelated);
  s.Evict();
  s.Evict(true);
  s.Evict(false, true);
  // A store-side quarantine that changes the best candidate reaches the
  // catalog only through its own retirement's ranking.
  {
    const auto ranks = s.Predict();
    const auto best = Sorted(ranks).front()->id;
    const auto changes = std::ranges::find_if(ranks, [&](const auto& rank) {
      if (rank.invalid)
        return false;
      auto model = s.model;
      model.at(rank.id).invalid = true;
      const auto other =
          OracleRank(s.catalog->RecordsForTesting(), model,
                     s.catalog->AdmittedBytesForTesting(), s.turnover);
      return Sorted(other).front()->id != best;
    });
    assert(changes != ranks.end());
    s.Quarantine(changes->id, true);
  }
  s.Evict(false, false, false);
  // A reopened store recovers the use history the catalog flushed; the
  // quarantine was in memory only. A smaller allowance retires several
  // checkpoints along one ranking at startup.
  assert(std::ranges::any_of(s.Check(), &DiskCatalogRankForTesting::invalid));
  const auto before = s.model.size();
  s.Restart(s.catalog->MetadataBytes() / 2);
  assert(before - s.model.size() >= 2);
  // The pass stops an eighth of the allowance below it.
  assert(s.catalog->MetadataBytes() <= s.allowance - s.allowance / 8);
  assert(std::ranges::none_of(s.Check(), &DiskCatalogRankForTesting::invalid));
  for (int i = 0; i < 2; ++i)
    (void)s.Publish(
        {turn(heads[random() % heads.size()]), 12, {0}, 42, purpose()});
  s.AddLive(42, turn(heads[random() % heads.size()]));
  s.Touch(s.model.begin()->first);
  for (std::size_t evictions = 0; !s.model.empty(); ++evictions)
    s.Evict(evictions == 3 && s.model.size() > 1, evictions % 2 == 1);
  assert(s.catalog->Size() == 0 && s.catalog->MetadataBytes() == 0);
  // Some full-ledger retirement left an unmerged node; later erasures and
  // the live frontiers' own leave the index as registration left it.
  assert(s.unmerged > 0);
  for (const auto& frontier : s.live)
    s.index->Erase(frontier.entry);
  assert(s.index->MetadataBytes() == s.index_baseline);
}
// Retiring a covered intermediate when the ledger admits nothing cannot fund
// the catalog trie's merge, which leaves a record-less unary node on the
// path. The ranking must read through it exactly as the token lists do.
void RankingReadsThroughAnUnmergedTrieNode() {
  const std::vector<Token> prefix(40, 3);
  auto middle = prefix;
  middle.resize(80, 4);
  auto leaf = middle;
  leaf.resize(120, 5);
  OracleStore s(UINT64_MAX, DiskTokenStorage::kSeparate);
  // The learned prefix ranks as a shared fork and the leaf as the frontier,
  // so the covered middle retires first.
  (void)s.Publish({prefix, 1, {1}, 42, CheckpointPurpose::kLearned});
  (void)s.Publish({middle, 1, {1}, 42, CheckpointPurpose::kPrompt});
  (void)s.Publish({leaf, 1, {1}, 42, CheckpointPurpose::kPrompt});
  // A retirement with headroom merges the middle node away.
  s.Evict();
  assert(!s.model.contains(2));
  const auto merged = s.catalog->MetadataBytes();
  // The middle again, retired when the ledger admits nothing. Drop the index
  // record first, so the refused admission is the catalog trie's merge
  // rather than the index's.
  const auto again =
      s.Publish({middle, 1, {1}, 42, CheckpointPurpose::kPrompt});
  s.index->RemoveDurable(*s.catalog->Find({again}));
  s.ledger.FailAfter(LedgerStep::kReserve, 0);
  const auto eviction = s.catalog->EvictOne();
  s.ledger.ClearFaults();
  assert(eviction &&
         eviction.description->Manifest().checkpoint.value == again);
  s.model.erase(again);
  s.Check();
  // The refused merge left the middle node and its header apart.
  assert(s.catalog->MetadataBytes() > merged);
  // A sibling forks the prefix through the unary node.
  auto sibling = prefix;
  sibling.resize(90, 6);
  (void)s.Publish({sibling, 1, {1}, 42, CheckpointPurpose::kPrompt});
  while (!s.model.empty())
    s.Evict();
  assert(s.catalog->MetadataBytes() == 0);
}
void EvictionDoesNotWaitForYieldingWriter() {
  TieredFixture f;
  f.Append(8, 4);
  auto checkpoint = f.Capture();
  auto description = f.Publish(*checkpoint);
  auto manifest = description->Manifest();
  manifest.checkpoint = {999};
  // Reuse the existing full chunks; private state requires a fresh source, so
  // this narrow optional writer fixture contains only its target component.
  manifest.components.resize(1);
  std::promise<void> yielding;
  std::atomic<bool> observed{};
  PersistenceQueue queue(
      f.ledger, f.streams, 1, 1,
      [&] {
        if (!observed.exchange(true))
          yielding.set_value();
        return false;
      },
      {}, {}, std::chrono::hours(1));
  assert(queue.TrySubmit(
      PersistenceJob(f.ledger, TieredFile(4, 999), manifest, {})));
  yielding.get_future().wait();
  auto eviction =
      std::async(std::launch::async, [&] { return f.catalog.EvictOne(); });
  assert(eviction.wait_for(std::chrono::seconds(2)) ==
         std::future_status::ready);
  assert(!eviction.get());
  queue.Stop();
  assert(f.catalog.EvictOne());
}
void QuarantineRacingPublication() {
  for (unsigned phase : {0, 1, 2}) {
    TieredFixture f;
    f.Append(8, 4);
    auto checkpoint = f.Capture();
    auto description = f.Publish(*checkpoint);
    auto manifest = description->Manifest();
    manifest.checkpoint = {999};
    manifest.components.resize(1);
    const auto file = manifest.components[0].chunks[0].file;
    bool invalidated{}, failed{};
    if (phase) {
      f.disk.SetCrashHook([&](DiskPublicationStep step) {
        if (step ==
            (phase == 1
                 ? DiskPublicationStep::kDependenciesSynced
                 : DiskPublicationStep::kFinalTemporaryDirectorySynced)) {
          f.catalog.Invalidate(false, file);
          invalidated = true;
        }
      });
    }
    TransferTiming timing;
    try {
      f.streams.Write(TieredFile(4, 999), manifest, {}, timing, {}, [&] {
        if (!phase && !invalidated) {
          f.catalog.Invalidate(false, file);
          invalidated = true;
        }
        return true;
      });
    } catch (const std::exception&) {
      failed = true;
    }
    assert(failed && invalidated);
    assert(!f.disk.Open({999}) && !description->Pin());
    f.disk.SetCrashHook({});
    if (phase == 2) {
      f.disk.ReclaimOrphans();
      f.disk.WaitForReclamation();
    }
    assert(f.catalog.ReclaimInvalid() == 1);
    assert(f.disk.Stats().managed_bytes == 0);
  }
}
struct RetentionLog final : RetentionEventSink {
  std::vector<RetentionEvent> events;
  void Emit(const RetentionEvent& e) noexcept override { events.push_back(e); }
};
void DurableCheckpointEvictedAheadOfNonDurableUnderPressure() {
  RetentionLog log;
  TieredFixture f;
  RetentionPolicy retention(f.ledger, f.index, &log);

  // Non-durable first conversation created earlier (older recency).
  f.Append(16, 0);
  const auto non_durable_tokens = f.tokens;
  auto non_durable = f.Capture();
  const auto non_durable_id = non_durable->Id();
  RetentionRequest req1{
      non_durable_tokens, f.adapter.CompatibilityIdentity(), {}};
  assert(retention.Admit(req1, [&] { return non_durable; }));
  auto lookup1 = f.index.Lookup(f.Query());
  assert(lookup1.selected);
  const auto non_durable_entry = lookup1.selected->entry;
  assert(!f.index.IsDurable(non_durable_entry));

  // Reset slot to create an independent second conversation with distinct
  // tokens.
  assert(f.adapter.Invalidate(*f.slot));
  std::vector<Token> durable_tokens(16);
  std::iota(durable_tokens.begin(), durable_tokens.end(), 200);
  f.adapter.Append(*f.slot, durable_tokens, {});
  auto durable_history = ExecutionHistory::Cold(
      f.ledger, f.adapter.Components(), f.adapter.CompatibilityIdentity());
  auto durable_checkpoint = durable_history.Capture(
      {durable_tokens, InputIdentity(durable_tokens.size()),
       f.adapter.Positions(*f.slot), CheckpointPurpose::kPrompt, 1},
      [&](const PayloadRequest& request) {
        auto admission =
            f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes);
        auto bytes = std::make_shared<std::vector<std::byte>>(request.bytes);
        auto pool = admission.Convert();
        auto assigned = pool.ReserveBacking(request.category);
        auto completion =
            request.category == ResourceCategory::kPrivateState
                ? f.adapter.CapturePrivate(*f.slot, request.component, *bytes,
                                           f.stream)
                : f.adapter.CopyRowsOut(*f.slot, request.component,
                                        request.first, request.end, *bytes,
                                        f.stream);
        assert(completion.Wait() == TransferResult::kSucceeded);
        return Payload::Committed(assigned.Convert(), bytes);
      });
  const auto durable_id = durable_checkpoint->Id();

  // Admit second conversation (newer recency).
  RetentionRequest req2{durable_tokens, f.adapter.CompatibilityIdentity(), {}};
  assert(retention.Admit(req2, [&] { return durable_checkpoint; }));
  auto lookup2 = f.index.Lookup({{f.adapter.CompatibilityIdentity()},
                                 durable_tokens,
                                 InputIdentity(durable_tokens.size()),
                                 0,
                                 true});
  assert(lookup2.selected);
  const auto durable_entry = lookup2.selected->entry;

  // Publish second checkpoint to disk and attach it, making it durable.
  auto durable_desc = f.Publish(*durable_checkpoint);
  f.index.AttachDurable(durable_entry, durable_desc);
  assert(f.index.IsDurable(durable_entry));

  // Under pure LRU, the older non-durable checkpoint would be evicted first.
  // Under PressureOrder, durable checkpoints are evicted first from RAM (rank 0
  // vs 1) within the same tier so non-durable state without disk backing is
  // preserved.
  RetentionRequest reclaim_req{
      std::vector<Token>{999}, f.adapter.CompatibilityIdentity(), {}};
  assert(retention.Reclaim(reclaim_req));
  assert(log.events.back().checkpoint == durable_id);

  // The durable checkpoint dropped its RAM residency but remains searchable
  // from disk via PrefixIndex.
  auto lookup_durable_after =
      f.index.Lookup({{f.adapter.CompatibilityIdentity()},
                      durable_tokens,
                      InputIdentity(durable_tokens.size()),
                      0,
                      true});
  assert(lookup_durable_after.selected &&
         lookup_durable_after.selected->durable == durable_desc &&
         !lookup_durable_after.selected->checkpoint);

  // The older non-durable checkpoint is still resident in RAM.
  auto lookup_non_durable_after =
      f.index.Lookup({{f.adapter.CompatibilityIdentity()},
                      non_durable_tokens,
                      InputIdentity(non_durable_tokens.size()),
                      0,
                      true});
  assert(lookup_non_durable_after.selected &&
         lookup_non_durable_after.selected->entry == non_durable_entry &&
         lookup_non_durable_after.selected->checkpoint);

  // Subsequent pressure now reclaims the non-durable checkpoint.
  assert(retention.Reclaim(reclaim_req));
  assert(log.events.back().checkpoint == non_durable_id);
}
}  // namespace
// `oracle` runs only the token oracle, which CTest runs as a test of its own
// beside the rest.
int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "oracle") {
    EvictionOrderMatchesTokenOracle(DiskTokenStorage::kManifest);
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "oracle-separate") {
    EvictionOrderMatchesTokenOracle(DiskTokenStorage::kSeparate);
    RankingReadsThroughAnUnmergedTrieNode();
    return 0;
  }
  assert(argc == 1);
  DeeperDiskAndRAMEviction();
  DemotionKeepsIndependentSourcePlans();
  CorruptionInvalidatesAllDependents();
  GlobalLRUAndSharedFiles();
  SupersededDiskTurnsLeaveBeforeAnotherConversation();
  PromptAnchorOutlivesCoveredAndLeafGeneratedFrontiers();
  LeafOutlivesItsLaterPrefix();
  SharedPrefixLendsNothingToItsBranches();
  DiskUseHistorySurvivesCatalogRestart();
  DiskReuseExpiresAfterAByteTurnoverAndSurvivesRestart();
  AUsedFrontierIsNotAnOptionalVictim();
  DiskEvictionNeedsNoLedgerHeadroom();
  RecordPressureDropsUnusedFrontierBeforeRetry();
  MixedSourcesAndCompatibility();
  PublicationCorruptionQuarantinesImmediately();
  RepublishedQuarantineCannotRehabilitateOldRAM();
  InvalidUnusedRAMComponentStillSelectsMixed();
  DurableAvailabilityAdmissionOrder();
  CatalogAdmissionFailures();
  DiskMetadataAllowanceRule();
  CatalogMetadataTracksAdmissionAndRetirement();
  MetadataAllowanceEvictsInsteadOfFillingTheLedger();
  MetadataAllowanceKeepsTheNewPublication();
  MetadataAllowancePassFailures();
  StartupFitsRecoveredMetadata();
  CatalogTrackRequiresTokens();
  CatalogIsPrefixOf();
  DiskMetadataGrowsLinearlyWithCheckpoints();
  RecoveryHandsTokensToTheCatalog();
  CatalogRefusesAnotherStoresTokens();
  VisitGeneratedRebuildsDurableOnlyPaths();
  AttachDurableRejectsReleasedTokens();
  EvictionDoesNotWaitForYieldingWriter();
  QuarantineRacingPublication();
  DurableCheckpointEvictedAheadOfNonDurableUnderPressure();
}
