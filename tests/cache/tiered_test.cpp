#include <cstdlib>
#include <fstream>
#include <future>
#include <new>

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
  assert(f.catalog.EvictOne());
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
  assert(f.catalog.EvictOne());
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
        assert(f.catalog.Size() == 0);
        assert(f.ledger.Snapshot().total_bytes == baseline);
      }
    }
    assert(completed && f.catalog.Size() == 1);
    assert(f.catalog.EvictOne());
    assert(f.catalog.Size() == 0);
  }
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
int main() {
  DeeperDiskAndRAMEviction();
  CorruptionInvalidatesAllDependents();
  GlobalLRUAndSharedFiles();
  SupersededDiskTurnsLeaveBeforeAnotherConversation();
  PromptAnchorOutlivesCoveredAndLeafGeneratedFrontiers();
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
  EvictionDoesNotWaitForYieldingWriter();
  QuarantineRacingPublication();
  DurableCheckpointEvictedAheadOfNonDurableUnderPressure();
}
