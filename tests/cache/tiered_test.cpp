#include <fstream>
#include <future>

#include "src/cache/retention.hpp"
#include "tests/cache/tiered_fixture.hpp"

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
  PersistenceQueue queue(f.ledger, f.streams, 1, 1, [&] {
    if (!observed.exchange(true))
      yielding.set_value();
    return false;
  });
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
}  // namespace
int main() {
  DeeperDiskAndRAMEviction();
  CorruptionInvalidatesAllDependents();
  GlobalLRUAndSharedFiles();
  MixedSourcesAndCompatibility();
  CatalogAdmissionFailures();
  EvictionDoesNotWaitForYieldingWriter();
  QuarantineRacingPublication();
}
