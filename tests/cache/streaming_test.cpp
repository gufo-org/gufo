#include "src/cache/streaming.hpp"

#include <algorithm>
#include <array>
#include <barrier>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <numeric>
#include <thread>

#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace fs = std::filesystem;
namespace {
struct Directory {
  fs::path path;
  Directory() {
    char pattern[] = "/tmp/gufo-streaming-test-XXXXXX";
    path = mkdtemp(pattern);
  }
  ~Directory() { fs::remove_all(path); }
};
struct Guard final : MutationGuard {
  void BeforeOverwrite(ComponentId, Rows, Rows) override {}
  void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
};
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n;
  id[1] = n >> 8;
  return id;
}
StagingAllocation Allocate(std::size_t bytes) {
  auto buffer = std::make_shared<std::vector<std::uint8_t>>(bytes);
  return {buffer, *buffer};
}
std::unique_ptr<Stream> Lease() {
  return std::make_unique<FakeStream>(true);
}
struct Captured {
  ResourceCharge charge;
  std::shared_ptr<std::vector<std::byte>> owner;
  bool private_file;
  DiskFileId file;
  std::size_t alignment;
};
struct Fixture {
  FakeAdapter adapter;
  Guard guard;
  std::unique_ptr<Slot> source = adapter.CreateSlot(guard);
  DiskManifest manifest;
  std::vector<Captured> buffers;
  std::uint64_t hash;
  explicit Fixture(ResourceLedger& ledger, unsigned checkpoint = 1,
                   unsigned file_base = 0, unsigned target = 40) {
    std::vector<Token> tokens(target);
    std::iota(tokens.begin(), tokens.end(), 7);
    adapter.Append(*source, tokens, std::span(tokens).first(target - 4));
    hash = adapter.RecurrentHash(*source);
    manifest.checkpoint = {checkpoint};
    manifest.lineage = {1};
    manifest.compatibility[0] = 42;
    manifest.tokens = tokens;
    auto positions = adapter.Positions(*source);
    FakeStream stream;
    unsigned next = file_base + 1;
    for (std::size_t i = 0; i < adapter.Components().size(); ++i) {
      const auto descriptor = adapter.Components()[i];
      DiskComponent component{descriptor, positions[i], {}, {}, {}};
      auto capture = [&](Rows first, Rows end, bool private_state,
                         bool private_file = false) {
        const auto bytes = private_state ? descriptor.state_bytes
                                         : (end - first) * descriptor.row_bytes;
        const auto category = private_state ? ResourceCategory::kPrivateState
                              : private_file
                                  ? ResourceCategory::kPrivateTail
                                  : ResourceCategory::kBackingAssigned;
        auto pool_admission =
            ledger.Reserve(ResourceCategory::kBackingFree, bytes);
        auto owner = std::make_shared<std::vector<std::byte>>(bytes);
        auto pool = pool_admission.Convert();
        auto admission = pool.ReserveBacking(category);
        auto completion =
            private_state
                ? adapter.CapturePrivate(*source, descriptor.id, *owner, stream)
                : adapter.CopyRowsOut(*source, descriptor.id, first, end,
                                      *owner, stream);
        assert(completion.Wait() == TransferResult::kSucceeded);
        const auto file = Id(next++);
        buffers.push_back({admission.Convert(), owner,
                           private_state || private_file, file,
                           private_state ? 1 : descriptor.row_bytes});
        // New streamed files compute CRC; no pre-serialization checksum pass.
        return DiskPayload{file, bytes, 0};
      };
      if (descriptor.kind == ComponentKind::kPrivateState)
        component.private_state = capture(0, 0, true);
      else {
        const auto complete = positions[i].valid_rows /
                              descriptor.rows_per_chunk *
                              descriptor.rows_per_chunk;
        for (Rows first = 0; first < complete;
             first += descriptor.rows_per_chunk)
          component.chunks.push_back(
              capture(first, first + descriptor.rows_per_chunk, false));
        if (complete != positions[i].valid_rows)
          component.tail =
              capture(complete, positions[i].valid_rows, false, true);
      }
      manifest.components.push_back(std::move(component));
    }
  }
  std::vector<PersistenceSource> Sources(std::function<void()> on_copy = {},
                                         std::function<void()> on_settle = {}) {
    std::vector<PersistenceSource> result;
    for (const auto& buffer : buffers) {
      auto* data = buffer.owner->data();
      result.emplace_back(
          buffer.private_file, buffer.file, buffer.owner->size(),
          buffer.alignment, buffer.charge, buffer.owner,
          [data, on_copy, on_settle](auto offset, auto bytes, Stream& stream) {
            if (on_copy)
              on_copy();
            return dynamic_cast<FakeStream&>(stream).Submit(
                [data, offset, bytes, on_settle] {
                  if (on_settle)
                    on_settle();
                  std::memcpy(bytes.data(), data + offset, bytes.size());
                  return TransferResult::kSucceeded;
                });
          });
    }
    return result;
  }
};
void PrefetchDoesNotValidateOrAdmitPayload() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 35, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 35, Allocate, Lease);
  Fixture fixture(ledger);
  auto sources = fixture.Sources();
  TransferTiming written;
  streams.Write(Id(100), fixture.manifest, sources, written);
  auto snapshot = disk.Open({1});
  assert(snapshot);
  const auto payload = snapshot->Manifest().components[0].chunks[0];
  const auto before = ledger.Snapshot().total_bytes;
  disk.PrefetchPayload(*snapshot, false, payload, 1);
  disk.PrefetchPayload(*snapshot, false, payload, 0);
  assert(ledger.Snapshot().total_bytes == before);
  auto foreign = payload;
  foreign.file = Id(999);
  bool refused = false;
  try {
    disk.PrefetchPayload(*snapshot, false, foreign, foreign.bytes);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  assert(refused);
  std::array<std::uint8_t, 16> staging{};
  auto read =
      disk.ReadPayload(*snapshot, false, payload, staging, [](auto, auto) {});
  assert(!read.verification_cached && read.bytes == payload.bytes);
  disk.PrefetchPayload(*snapshot, false, payload, payload.bytes);
  {
    std::fstream corrupt(dir.path / "v2/chunks" / DiskFileName(payload.file),
                         std::ios::in | std::ios::out | std::ios::binary);
    corrupt.put('\xff');
  }
  bool invalid = false;
  try {
    (void)disk.ReadPayload(*snapshot, false, payload, staging,
                           [](auto, auto) {});
  } catch (const DiskDependencyError& error) {
    invalid = !error.private_file && error.file == payload.file;
  }
  assert(invalid);
}
void RestoreGroupsAdjacentFilesWithinStaging() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 35, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 35, Allocate, Lease);
  Fixture fixture(ledger, 1, 0, 43);
  auto sources = fixture.Sources();
  TransferTiming written;
  streams.Write(Id(100), fixture.manifest, sources, written);
  auto destination = fixture.adapter.CreateSlot(fixture.guard);
  TransferTiming restored;
  assert(streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                         *destination, restored));
  // 19 row chunks and two partial tails become eleven row-aligned transfers,
  // plus one private-state transfer, without expanding the 35-byte buffer.
  assert(restored.pieces == 12 && restored.bytes == 360);
  assert(fixture.adapter.RecurrentHash(*destination) == fixture.hash);
  const std::array<Token, 2> next{88, 91};
  fixture.adapter.Append(*destination, next, next);
  fixture.adapter.Append(*fixture.source, next, next);
  assert(fixture.adapter.RecurrentHash(*destination) ==
         fixture.adapter.RecurrentHash(*fixture.source));
  assert(ledger.Snapshot().peak_bytes[static_cast<std::size_t>(
             ResourceCategory::kTransferStaging)] == 35);
  // The second file of a group must still validate independently. No partial
  // group, nor an earlier successful group, may leave partially restored state.
  const auto file = fixture.manifest.components[0].chunks[3].file;
  {
    std::fstream corrupt(dir.path / "v2/chunks" / DiskFileName(file),
                         std::ios::in | std::ios::out | std::ios::binary);
    corrupt.put('\xff');
  }
  TransferTiming failed;
  assert(!streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                          *destination, failed));
  const auto positions = fixture.adapter.Positions(*destination);
  assert(
      std::ranges::all_of(positions, [](auto p) { return p.valid_rows == 0; }));
}
void RoundTripAndFailures() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 11, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  unsigned allocations{};
  StreamedStore streams(
      ledger, disk, 10,
      [&](auto bytes) {
        ++allocations;
        return Allocate(bytes);
      },
      Lease);
  Fixture fixture(ledger);
  {
    auto sources = fixture.Sources();
    TransferTiming timing;
    streams.Write(Id(100), fixture.manifest, sources, timing);
    assert(timing.bytes == 336 && timing.bytes >= 10 * streams.StagingBytes());
    assert(timing.pieces > fixture.buffers.size());
    assert(timing.filesystem_ns && timing.fsync_ns && timing.stream_wait_ns);
  }
  assert(allocations == 1);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  {
    auto snapshot = disk.Open({1});
    auto reused = snapshot->Manifest();
    reused.checkpoint = {2};
    reused.tokens.resize(4);
    reused.components.resize(1);
    reused.components[0].position.valid_rows = 4;
    reused.components[0].chunks.resize(1);
    TransferTiming reuse_timing;
    streams.Write(Id(101), reused, {}, reuse_timing);
    assert(reuse_timing.bytes == 0);
  }
  assert(disk.Retire({2}));
  {
    // Serving queues later checkpoints with the same immutable chunks before
    // their first publication has supplied checksums to any durable manifest.
    auto reused = fixture.manifest;
    reused.checkpoint = {3};
    reused.tokens.resize(4);
    reused.components.resize(1);
    reused.components[0].position.valid_rows = 4;
    reused.components[0].chunks.resize(1);
    auto sources = fixture.Sources();
    TransferTiming timing;
    streams.Write(Id(102), reused, std::span(sources).first(1), timing);
    assert(timing.bytes == 0);
    const auto original = disk.Open({1});
    const auto published = disk.Open({3});
    assert(published);
    for (std::size_t i = 0; i < reused.components.size(); ++i) {
      assert(published->Manifest().components[i].chunks.front() ==
             original->Manifest().components[i].chunks.front());
    }
  }
  assert(disk.Retire({3}));
  {
    auto snapshot = disk.Open({1});
    const auto payload = snapshot->Manifest().components[0].chunks[0];
    const auto baseline = ledger.Snapshot().total_bytes;
    std::array<std::uint8_t, 5> piece{};
    for (auto fault : {LedgerStep::kReserve, LedgerStep::kConvert}) {
      ledger.FailAfter(fault, 0);
      auto read =
          disk.ReadPayload(*snapshot, false, payload, piece, [](auto, auto) {});
      assert(!read.verification_cached && read.bytes == payload.bytes);
      ledger.ClearFaults();
      assert(ledger.Snapshot().total_bytes == baseline);
    }
  }
  auto destination = fixture.adapter.CreateSlot(fixture.guard);
  TransferTiming timing;
  assert(streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                         *destination, timing));
  assert(fixture.adapter.Positions(*destination) ==
         fixture.adapter.Positions(*fixture.source));
  assert(fixture.adapter.RecurrentHash(*destination) == fixture.hash);
  {
    FakeStream oracle;
    for (const auto& descriptor : fixture.adapter.Components()) {
      const auto rows =
          fixture.adapter.Positions(*fixture.source)[descriptor.id.value - 1]
              .valid_rows;
      const auto bytes = descriptor.kind == ComponentKind::kPrivateState
                             ? descriptor.state_bytes
                             : rows * descriptor.row_bytes;
      std::vector<std::byte> expected(bytes), actual(bytes);
      auto capture = [&](const Slot& slot, auto& buffer) {
        return descriptor.kind == ComponentKind::kPrivateState
                   ? fixture.adapter.CapturePrivate(slot, descriptor.id, buffer,
                                                    oracle)
                   : fixture.adapter.CopyRowsOut(slot, descriptor.id, 0, rows,
                                                 buffer, oracle);
      };
      assert(capture(*fixture.source, expected).Wait() ==
             TransferResult::kSucceeded);
      assert(capture(*destination, actual).Wait() ==
             TransferResult::kSucceeded);
      assert(actual == expected);
    }
  }
  const std::array<Token, 2> next{88, 91};
  fixture.adapter.Append(*destination, next, next);
  fixture.adapter.Append(*fixture.source, next, next);
  assert(fixture.adapter.RecurrentHash(*destination) ==
         fixture.adapter.RecurrentHash(*fixture.source));
  assert(ledger.Snapshot().peak_bytes[static_cast<std::size_t>(
             ResourceCategory::kTransferStaging)] == 10);
  // Verification survives identical identity and is discarded on a changed
  // inode, size, mtime or ctime. Read all bytes even on a verification hit.
  {
    auto snapshot = disk.Open({1});
    const auto& payload = snapshot->Manifest().components[0].chunks[0];
    std::array<std::uint8_t, 5> staging{};
    auto stats =
        disk.ReadPayload(*snapshot, false, payload, staging, [](auto, auto) {});
    assert(stats.verification_cached && stats.bytes == payload.bytes);
    assert(!disk.Retire({1}));
  }
  const auto path = dir.path / "v2/chunks" / DiskFileName(Id(1));
  {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    char corrupt = 99;
    file.write(&corrupt, 1);
  }
  {
    // Resolving an unspecified checksum must still attest the existing file;
    // a new durable checkpoint cannot hide corruption in a shared chunk.
    auto reused = fixture.manifest;
    reused.checkpoint = {4};
    reused.tokens.resize(4);
    reused.components.resize(1);
    reused.components[0].position.valid_rows = 4;
    reused.components[0].chunks.resize(1);
    auto sources = fixture.Sources();
    bool rejected{};
    try {
      streams.Write(Id(103), reused, std::span(sources).first(1), timing);
    } catch (const DiskDependencyError&) {
      rejected = true;
    }
    assert(rejected && !disk.Open({4}));
  }
  bool fallback{};
  assert(!streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                          *destination, timing, {}, [&] {
                            assert(destination->IsValid());
                            for (auto position :
                                 fixture.adapter.Positions(*destination))
                              assert(position.valid_rows == 0);
                            fallback = true;
                            fixture.adapter.Append(*destination, next, next);
                          }));
  assert(fallback &&
         fixture.adapter.Positions(*destination)[0].valid_rows == 2);
  // No leaked read pin after failure.
  assert(disk.Retire({1}));
  {
    auto sources = fixture.Sources();
    streams.Write(Id(100), fixture.manifest, sources, timing);
  }
  std::stop_source cancellation;
  unsigned leases{};
  StreamedStore cancellable(ledger, disk, 1, Allocate,
                            [&]() -> std::unique_ptr<Stream> {
                              if (++leases == 3)
                                cancellation.request_stop();
                              return Lease();
                            });
  // A row larger than staging must fail before BeginRestore and still cold
  // invalidate; private pieces are independently covered with the main pool.
  assert(!cancellable.Restore({1}, fixture.manifest.compatibility,
                              fixture.adapter, *destination, timing));
}
void LiveRowsStreamWithoutWholeHistoryBacking() {
  for (const bool retire : {false, true}) {
    Directory dir;
    ResourceLedger ledger{{1 << 22, 1 << 21, 8, 4096}};
    DiskStore disk(ledger, dir.path, 1 << 20);
    StreamedStore streams(ledger, disk, 8, Allocate, Lease);
    Fixture fixture(ledger);
    std::erase_if(fixture.buffers,
                  [](const auto& b) { return !b.private_file; });
    FakeStream execution;
    LeasedSlot slot(ledger, fixture.adapter, execution, SlotId{1});
    auto lease = slot.Acquire();
    fixture.adapter.Append(lease.Execution(), fixture.manifest.tokens,
                           std::span(fixture.manifest.tokens).first(36));
    auto sources = fixture.Sources();
    const auto before = ledger.Snapshot();
    for (const auto& component : fixture.manifest.components) {
      const auto& d = component.descriptor;
      if (d.kind != ComponentKind::kAppendRows)
        continue;
      auto rows = lease.Observe(
          d.id, 0, component.position.valid_rows,
          fixture.adapter.BorrowRowsView(lease.Execution(), d.id, 0,
                                         component.position.valid_rows));
      std::size_t first = 0;
      for (const auto& chunk : component.chunks) {
        sources.emplace_back(
            false, chunk.file, chunk.bytes, d.row_bytes, rows, first,
            [](auto source, auto destination, Stream& stream) {
              return dynamic_cast<FakeStream&>(stream).Submit(
                  [source, destination] {
                    std::memcpy(destination.data(), source.data(),
                                destination.size());
                    return TransferResult::kSucceeded;
                  });
            });
        first += chunk.bytes;
      }
    }
    const auto after = ledger.Snapshot();
    // Observations cost metadata only, with no assignment/spill payload.
    for (const auto category :
         {ResourceCategory::kBackingFree, ResourceCategory::kBackingAssigned,
          ResourceCategory::kBackingMaterialized})
      assert(before.bytes[static_cast<std::size_t>(category)] ==
             after.bytes[static_cast<std::size_t>(category)]);
    if (retire)
      fixture.adapter.GuardRows(lease.Execution(), kTarget, 0, 40);
    TransferTiming timing;
    if (retire) {
      bool failed = false;
      try {
        streams.Write(Id(100), fixture.manifest, sources, timing);
      } catch (const std::exception&) {
        failed = true;
      }
      assert(failed && !disk.Open({1}));
    } else {
      streams.Write(Id(100), fixture.manifest, sources, timing);
      auto restored = fixture.adapter.CreateSlot(fixture.guard);
      assert(streams.Restore({1}, fixture.manifest.compatibility,
                             fixture.adapter, *restored, timing));
      assert(fixture.adapter.RecurrentHash(*restored) == fixture.hash);
    }
    assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  }
}
void BorrowedSourcesWaitWithoutBlockingForeground() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 11, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 10, Allocate, Lease);
  Fixture fixture(ledger);
  FakeStream preservation;
  LeasedSlot slot(ledger, fixture.adapter, preservation, {1});
  auto lease = slot.Acquire();
  fixture.adapter.Append(lease.Execution(), fixture.manifest.tokens,
                         std::span(fixture.manifest.tokens).first(36));
  auto allocation = std::make_shared<std::array<std::byte, 16>>();
  auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
  auto rows = lease.Borrow(
      kTarget, 0, 4, pool.ReserveBacking(ResourceCategory::kBackingAssigned),
      allocation, *allocation);
  auto committed = fixture.Sources();
  std::vector<PersistenceSource> sources;
  sources.emplace_back(false, fixture.buffers.front().file, 16, 4, rows);
  for (std::size_t i = 1; i < committed.size(); ++i)
    sources.push_back(std::move(committed[i]));
  {
    PersistenceQueue queue(ledger, streams, 2, 4096, [] { return true; });
    assert(queue.TrySubmit(
        PersistenceJob(ledger, Id(100), fixture.manifest, std::move(sources))));
    // An assigned reservation is pinned before materialization, without a
    // second physical charge or a reader pin blocking the source's mutation.
    assert(ledger.Snapshot().persistence_pinned_bytes == 336);
    assert(rows->Location());
    assert(!disk.Open({1}));
    Fixture foreground(ledger, 2, 100);
    auto foreground_sources = foreground.Sources();
    TransferTiming foreground_timing;
    streams.Write(Id(200), foreground.manifest, foreground_sources,
                  foreground_timing);
    assert(disk.Open({2}));
    assert(!disk.Open({1}));
    // Cold reassignment preserves before invalidation. The worker then reads
    // that immutable backing and publishes the original complete frontier.
    lease.Commit();
    auto replacement = slot.Acquire();
    assert(!rows->Location());
    queue.Drain();
    assert(queue.Stats().completed == 1);
    assert(ledger.Snapshot().persistence_pinned_bytes == 0);
    Guard guard;
    auto restored = fixture.adapter.CreateSlot(guard);
    TransferTiming timing;
    assert(streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                           *restored, timing));
    assert(fixture.adapter.RecurrentHash(*restored) == fixture.hash);
    std::array<std::byte, 16> restored_rows;
    auto capture = fixture.adapter.CopyRowsOut(*restored, kTarget, 0, 4,
                                               restored_rows, preservation);
    assert(capture.Wait() == TransferResult::kSucceeded);
    assert(restored_rows == *allocation);
  }
}
void LiveBorrowedSourcesPublishFromSlot() {
  // With live reads, rows still in a leased slot are copied from there under
  // a reader pin: the job neither waits for preservation nor materializes.
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 11, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 10, Allocate, Lease);
  Fixture fixture(ledger);
  FakeStream preservation;
  LeasedSlot slot(ledger, fixture.adapter, preservation, {1});
  auto lease = slot.Acquire();
  fixture.adapter.Append(lease.Execution(), fixture.manifest.tokens,
                         std::span(fixture.manifest.tokens).first(36));
  std::array<std::byte, 16> native;
  auto read = fixture.adapter.CopyRowsOut(lease.Execution(), kTarget, 0, 4,
                                          native, preservation);
  assert(read.Wait() == TransferResult::kSucceeded);
  auto allocation = std::make_shared<std::array<std::byte, 16>>();
  auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
  auto rows = lease.Borrow(
      kTarget, 0, 4, pool.ReserveBacking(ResourceCategory::kBackingAssigned),
      allocation, *allocation, native);
  struct Done final : CompletionSignal {
    bool Ready() const noexcept override { return true; }
    TransferResult Wait() noexcept override {
      return TransferResult::kSucceeded;
    }
  };
  unsigned live_reads{};
  auto committed = fixture.Sources();
  std::vector<PersistenceSource> sources;
  sources.emplace_back(false, fixture.buffers.front().file, 16, 4, rows,
                       [&](std::span<const std::byte> source,
                           std::span<std::byte> destination, Stream&) {
                         ++live_reads;
                         std::memcpy(destination.data(), source.data(),
                                     destination.size());
                         return Completion(std::make_unique<Done>());
                       });
  for (std::size_t i = 1; i < committed.size(); ++i)
    sources.push_back(std::move(committed[i]));
  PersistenceQueue queue(ledger, streams, 2, 4096, [] { return true; });
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(100), fixture.manifest, std::move(sources))));
  queue.Drain();
  assert(queue.Stats().completed == 1 && live_reads != 0);
  assert(rows->Location());
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  Guard guard;
  auto restored = fixture.adapter.CreateSlot(guard);
  TransferTiming timing;
  assert(streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                         *restored, timing));
  assert(fixture.adapter.RecurrentHash(*restored) == fixture.hash);
  std::array<std::byte, 16> restored_rows;
  auto capture = fixture.adapter.CopyRowsOut(*restored, kTarget, 0, 4,
                                             restored_rows, preservation);
  assert(capture.Wait() == TransferResult::kSucceeded);
  assert(restored_rows == native);
  lease.Commit();
}
void StopWaitingBorrowedPublication(bool fail_idle) {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 11, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 10, Allocate, Lease);
  Fixture fixture(ledger);
  FakeStream preservation;
  LeasedSlot slot(
      ledger, fixture.adapter, preservation, {1},
      IdleSpillConfig{16, Lease, [fail_idle] { return fail_idle; }});
  auto lease = slot.Acquire();
  fixture.adapter.Append(lease.Execution(), fixture.manifest.tokens,
                         std::span(fixture.manifest.tokens).first(36));
  auto allocation = std::make_shared<std::array<std::byte, 16>>();
  auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
  auto rows = lease.Borrow(
      kTarget, 0, 4, pool.ReserveBacking(ResourceCategory::kBackingAssigned),
      allocation, *allocation);
  auto committed = fixture.Sources();
  std::vector<PersistenceSource> sources;
  sources.emplace_back(false, fixture.buffers.front().file, 16, 4, rows);
  for (std::size_t i = 1; i < committed.size(); ++i)
    sources.push_back(std::move(committed[i]));
  PersistenceQueue queue(ledger, streams, 2, 4096, [] { return true; });
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(100), fixture.manifest, std::move(sources))));
  if (fail_idle) {
    fixture.adapter.FailNextTransfer();
    lease.Commit();
    queue.Drain();
    assert(queue.Stats().failed == 1);
    assert(rows->PreservationFailed());
  } else {
    queue.Stop();
  }
  assert(!disk.Open({1}));
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  // Cancellation does not retire retained source rows or consume their
  // reservation. The original live execution can still preserve them later.
  assert(rows->Location());
  assert(rows->IsValid());
  if (!fail_idle)
    lease.Commit();
  auto replacement = slot.Acquire();
  assert(!rows->Location());
  assert(!rows->PreservationFailed());
}
void CancelRestore() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  Fixture fixture(ledger);
  std::stop_source stop;
  unsigned leases{};
  bool cancel{};
  StreamedStore streams(ledger, disk, 16, Allocate, [&] {
    if (cancel && ++leases == 3)
      stop.request_stop();
    return Lease();
  });
  auto sources = fixture.Sources();
  TransferTiming timing;
  streams.Write(Id(100), fixture.manifest, sources, timing);
  auto destination = fixture.adapter.CreateSlot(fixture.guard);
  cancel = true;
  bool fallback{};
  assert(!streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                          *destination, timing, stop.get_token(),
                          [&] { fallback = true; }));
  assert(leases == 3 && !fallback && destination->IsValid());
  for (auto position : fixture.adapter.Positions(*destination))
    assert(position.valid_rows == 0);
  assert(disk.Retire({1}));
}
void QueueBoundsAndYield() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 8192}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture first(ledger), second(ledger, 2, 30), third(ledger, 3, 60);
  std::atomic<bool> idle{}, copied{};
  PersistenceQueue queue(
      ledger, streams, 2, 672, [&] { return idle.load(); }, {}, {},
      std::chrono::hours(1));
  assert(queue.TrySubmit(PersistenceJob(
      ledger, Id(100), first.manifest, first.Sources([&] { copied = true; }))));
  assert(!queue.TrySubmit(
      PersistenceJob(ledger, Id(101), first.manifest, first.Sources())));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(102), second.manifest, second.Sources())));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(103), third.manifest, third.Sources())));
  auto stats = queue.Stats();
  assert(stats.pending == 2 && stats.pinned_bytes == 672);
  assert(stats.coalesced == 1 && stats.superseded == 1 && !stats.skipped &&
         !copied);
  assert(stats.accepted == 3);
  // While the writer yields, committed source ownership doesn't hold live
  // slot readers or the disk metadata lock.
  auto lookup = std::async(std::launch::async, [&] { return disk.Open({99}); });
  assert(lookup.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
  assert(!lookup.get());
  auto counters = std::async(std::launch::async, [&] { return disk.Stats(); });
  assert(counters.wait_for(std::chrono::seconds(2)) ==
         std::future_status::ready);
  (void)counters.get();
  idle = true;
  queue.Drain();
  stats = queue.Stats();
  assert(stats.completed == 2 && !stats.failed && !stats.pending);
  assert(stats.peak_pending == 2 && stats.peak_pinned_bytes == 672);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  assert(disk.Entries().size() == 2);
  assert(disk.Open({3}));  // The recent job survives a full optional queue.
  assert(ledger.Snapshot().peak_persistence_pinned_bytes == 672);
  idle = false;
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(103), third.manifest, third.Sources())));
  queue.Stop();
  assert(queue.Stats().pending == 0 && queue.Stats().pinned_bytes == 0);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
}
void BusyModelYieldIsBounded() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 8192}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture fixture(ledger);
  // A model that is never idle must not keep a job's pins forever.
  PersistenceQueue queue(
      ledger, streams, 1, 336, [] { return false; }, {}, {},
      std::chrono::milliseconds(20));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(100), fixture.manifest, fixture.Sources())));
  queue.Drain();
  assert(queue.Stats().completed == 1);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  assert(disk.Open({1}));
}
void NewestPendingFrontierSurvives() {
  for (const bool learned : {false, true}) {
    for (const bool pin_failure : {false, true}) {
      Directory dir;
      ResourceLedger ledger{{1 << 22, 1 << 21, 16, pin_failure ? 780U : 8192U}};
      DiskStore disk(ledger, dir.path, 1 << 20);
      StreamedStore streams(ledger, disk, 16, Allocate, Lease);
      Fixture active(ledger), prior(ledger, 2, 30, 48), next(ledger, 3, 60, 56);
      if (learned)
        prior.manifest.purpose = CheckpointPurpose::kLearned;
      std::atomic<bool> idle{}, started{}, discarded{};
      PersistenceQueue* query{};
      PersistenceQueue queue(
          ledger, streams, 2, 800,
          [&] {
            started = true;
            return idle.load();
          },
          [&](CheckpointId id, bool succeeded) {
            if (id == CheckpointId{2} && !succeeded) {
              // Replacement callbacks may query the queue and release serving
              // ownership; neither operation may run under the queue lock.
              assert(query->Stats().pending <= 2);
              discarded = true;
            }
          },
          {}, std::chrono::hours(1));
      query = &queue;
      assert(queue.TrySubmit(
          PersistenceJob(ledger, Id(100), active.manifest, active.Sources())));
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!started && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      assert(started);
      assert(queue.TrySubmit(
          PersistenceJob(ledger, Id(101), prior.manifest, prior.Sources())));
      assert(queue.TrySubmit(PersistenceJob(ledger, Id(102), next.manifest,
                                            next.Sources())) ==
             (!learned && !pin_failure));
      assert(discarded == !learned);
      assert(queue.Stats().pinned_bytes == (learned       ? 736
                                            : pin_failure ? 336
                                                          : 800));
      assert(ledger.Snapshot().peak_persistence_pinned_bytes <=
             (pin_failure ? 780 : 800));
      idle = true;
      queue.Drain();
      assert(disk.Open({1}));
      assert(bool(disk.Open({2})) == learned);
      assert(bool(disk.Open({3})) == (!learned && !pin_failure));
      assert(ledger.Snapshot().persistence_pinned_bytes == 0);
      if (!learned && !pin_failure) {
        auto destination = next.adapter.CreateSlot(next.guard);
        TransferTiming timing;
        assert(streams.Restore({3}, next.manifest.compatibility, next.adapter,
                               *destination, timing));
        assert(next.adapter.RecurrentHash(*destination) == next.hash);
      }
    }
  }
}
void FirstPendingAnchorSurvivesCoalescing() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 8192}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture active(ledger, 1, 120), anchor(ledger, 2, 30),
      middle(ledger, 3, 60, 48), latest(ledger, 4, 90, 56);
  active.manifest.compatibility[1] = 1;
  std::atomic<bool> idle{}, started{};
  PersistenceQueue queue(
      ledger, streams, 3, 1136,
      [&] {
        started = true;
        return idle.load();
      },
      {}, {}, std::chrono::hours(1));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(1000), active.manifest, active.Sources())));
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!started && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  assert(started);
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(1001), anchor.manifest, anchor.Sources())));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(1002), middle.manifest, middle.Sources())));
  assert(!queue.Stats().superseded);
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(1003), latest.manifest, latest.Sources())));
  assert(queue.Stats().superseded == 1 && queue.Stats().coalesced == 1);
  assert(queue.Stats().pending == 3 && queue.Stats().pinned_bytes == 1136);
  assert(ledger.Snapshot().peak_persistence_pinned_bytes <= 1136);
  idle = true;
  queue.Drain();
  assert(disk.Open({1}) && disk.Open({2}) && disk.Open({4}));
  assert(!disk.Open({3}));
  auto destination = latest.adapter.CreateSlot(latest.guard);
  TransferTiming timing;
  assert(streams.Restore({4}, latest.manifest.compatibility, latest.adapter,
                         *destination, timing));
  assert(latest.adapter.RecurrentHash(*destination) == latest.hash);
}
void AdmissionFailures() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  const auto baseline = ledger.Snapshot().total_bytes;
  for (auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    bool succeeded{};
    for (unsigned failure = 0; failure < 8 && !succeeded; ++failure) {
      ledger.FailAfter(step, failure);
      try {
        StreamedStore streams(
            ledger, disk, 16,
            [&](auto bytes) {
              auto owner = std::shared_ptr<std::uint8_t>(
                  new std::uint8_t[bytes], [&](auto* data) {
                    assert(ledger.Snapshot().bytes[static_cast<std::size_t>(
                               ResourceCategory::kTransferStaging)] == 16);
                    delete[] data;
                  });
              return StagingAllocation{owner, {owner.get(), bytes}};
            },
            Lease);
        succeeded = true;
      } catch (const std::bad_alloc&) {
      }
      ledger.ClearFaults();
      assert(ledger.Snapshot().total_bytes == baseline);
    }
    assert(succeeded);
  }
  Fixture fixture(ledger);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  PersistenceQueue queue(ledger, streams, 1, 336, [] { return false; });
  for (unsigned failure = 0; failure < fixture.buffers.size(); ++failure) {
    ledger.FailAfter(LedgerStep::kPin, failure);
    assert(!queue.TrySubmit(
        PersistenceJob(ledger, Id(100), fixture.manifest, fixture.Sources())));
    ledger.ClearFaults();
    assert(queue.Stats().pending == 0);
    assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  }
}
void FailedSubmission() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  unsigned leases{};
  StreamedStore streams(ledger, disk, 16, Allocate, [&] {
    auto stream = std::make_unique<FakeStream>(true);
    if (++leases == 3)
      stream->FailNextSubmission();
    return stream;
  });
  Fixture fixture(ledger);
  auto sources = fixture.Sources();
  TransferTiming costs;
  bool failed{};
  try {
    streams.Write(Id(100), fixture.manifest, sources, costs);
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed && disk.Entries().empty());
  assert(costs.filesystem_ns && costs.fsync_ns && costs.bytes == 32);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  disk.ReclaimOrphans();
  disk.WaitForReclamation();
  assert(disk.Stats().managed_bytes == 0);
  streams.Write(Id(100), fixture.manifest, sources, costs);
  auto destination = fixture.adapter.CreateSlot(fixture.guard);
  fixture.adapter.FailNextTransfer();
  bool fallback{};
  assert(!streams.Restore({1}, fixture.manifest.compatibility, fixture.adapter,
                          *destination, costs, {}, [&] {
                            for (auto position :
                                 fixture.adapter.Positions(*destination))
                              assert(position.valid_rows == 0);
                            fallback = true;
                          }));
  assert(fallback && disk.Retire({1}));
}
void RestorePreemptsYield() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture first(ledger), second(ledger, 2, 30);
  auto sources = first.Sources();
  TransferTiming write;
  streams.Write(Id(100), first.manifest, sources, write);
  std::promise<void> yielding;
  std::atomic<bool> observed{};
  PersistenceQueue queue(
      ledger, streams, 1, 336,
      [&] {
        if (!observed.exchange(true))
          yielding.set_value();
        return false;
      },
      {}, {}, std::chrono::hours(1));
  assert(queue.TrySubmit(
      PersistenceJob(ledger, Id(101), second.manifest, second.Sources())));
  yielding.get_future().wait();
  auto destination = first.adapter.CreateSlot(first.guard);
  auto restore = std::async(std::launch::async, [&] {
    TransferTiming timing;
    return streams.Restore({1}, first.manifest.compatibility, first.adapter,
                           *destination, timing);
  });
  assert(restore.wait_for(std::chrono::seconds(2)) ==
         std::future_status::ready);
  assert(restore.get());
  assert(first.adapter.RecurrentHash(*destination) == first.hash);
  queue.Stop();
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
  assert(disk.Retire({1}));
}
void CancelledStagingWait() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture first(ledger), second(ledger, 2, 30);
  auto first_sources = first.Sources();
  TransferTiming write;
  streams.Write(Id(100), first.manifest, first_sources, write);
  std::promise<void> holding, release;
  auto released = release.get_future().share();
  std::atomic<bool> observed{};
  auto second_sources = second.Sources([&] {
    if (!observed.exchange(true))
      holding.set_value();
    released.wait();
  });
  auto writer = std::async(std::launch::async, [&] {
    TransferTiming timing;
    streams.Write(Id(101), second.manifest, second_sources, timing);
  });
  holding.get_future().wait();
  std::stop_source stop;
  TransferTiming timing;
  auto destination = first.adapter.CreateSlot(first.guard);
  std::promise<void> started;
  auto restore = std::async(std::launch::async, [&] {
    started.set_value();
    return streams.Restore({1}, first.manifest.compatibility, first.adapter,
                           *destination, timing, stop.get_token());
  });
  started.get_future().wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  stop.request_stop();
  assert(restore.wait_for(std::chrono::seconds(2)) ==
         std::future_status::ready);
  assert(!restore.get());
  assert(timing.staging_wait_ns > 0);
  release.set_value();
  writer.get();
  assert(disk.Retire({1}) && disk.Retire({2}));
}
void CancelledPublicationWait() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 8192}, true};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture first(ledger), second(ledger, 2, 30);
  std::promise<void> yielding;
  std::atomic<bool> observed{};
  PersistenceQueue blocked(ledger, streams, 1, 336, [&] {
    if (!observed.exchange(true))
      yielding.set_value();
    return false;
  });
  PersistenceQueue cancelled(ledger, streams, 1, 336, [] { return true; });
  assert(blocked.TrySubmit(
      PersistenceJob(ledger, Id(100), first.manifest, first.Sources())));
  yielding.get_future().wait();
  PersistenceJob second_job(ledger, Id(101), second.manifest, second.Sources());
  const auto reserves =
      ledger.LockCosts()[static_cast<std::size_t>(LedgerStep::kReserve)].calls;
  assert(cancelled.TrySubmit(std::move(second_job)));
  // Admission in Write proves the second worker has taken its job. Stopping
  // now must succeed without resuming the first writer's indefinite yield.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (ledger.LockCosts()[static_cast<std::size_t>(LedgerStep::kReserve)]
             .calls == reserves) {
    assert(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  auto stopped = std::async(std::launch::async, [&] { cancelled.Stop(); });
  assert(stopped.wait_for(std::chrono::seconds(2)) ==
         std::future_status::ready);
  stopped.get();
  assert(cancelled.Stats().failed == 1);
  blocked.Stop();
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
}
void ConcurrentStop() {
  Directory dir;
  ResourceLedger ledger{{1 << 22, 1 << 21, 16, 4096}};
  DiskStore disk(ledger, dir.path, 1 << 20);
  StreamedStore streams(ledger, disk, 16, Allocate, Lease);
  Fixture fixture(ledger);
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  std::atomic<bool> observed{};
  PersistenceQueue queue(ledger, streams, 1, 336, [] { return true; });
  assert(queue.TrySubmit(PersistenceJob(ledger, Id(100), fixture.manifest,
                                        fixture.Sources({}, [&] {
                                          if (!observed.exchange(true)) {
                                            entered.set_value();
                                            released.wait();
                                          }
                                        }))));
  entered.get_future().wait();
  std::barrier start(3);
  auto stop = [&] {
    start.arrive_and_wait();
    queue.Stop();
  };
  auto one = std::async(std::launch::async, stop);
  auto two = std::async(std::launch::async, stop);
  start.arrive_and_wait();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  release.set_value();
  assert(one.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
  assert(two.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
  one.get();
  two.get();
  assert(queue.Stats().pending == 0);
  assert(ledger.Snapshot().persistence_pinned_bytes == 0);
}
void MinimumStagingHoldsWidestRow() {
  // Private state streams in pieces of any size; rows never split.
  const std::array<ComponentDescriptor, 3> mixed{{
      {ComponentId{1}, 1, ComponentKind::kAppendRows, 4, 256, 0},
      {ComponentId{2}, 1, ComponentKind::kAppendRows, 1024, 256, 0},
      {ComponentId{3}, 1, ComponentKind::kPrivateState, 0, 0, 1 << 20},
  }};
  assert(MinimumStagingBytes(mixed) == 1024);
  assert(MinimumStagingBytes(std::span(mixed).last(1)) == 1);
}
}  // namespace
int main() {
  MinimumStagingHoldsWidestRow();
  PrefetchDoesNotValidateOrAdmitPayload();
  RestoreGroupsAdjacentFilesWithinStaging();
  LiveRowsStreamWithoutWholeHistoryBacking();
  BorrowedSourcesWaitWithoutBlockingForeground();
  LiveBorrowedSourcesPublishFromSlot();
  StopWaitingBorrowedPublication(false);
  StopWaitingBorrowedPublication(true);
  RoundTripAndFailures();
  CancelRestore();
  QueueBoundsAndYield();
  BusyModelYieldIsBounded();
  NewestPendingFrontierSurvives();
  FirstPendingAnchorSurvivesCoalescing();
  AdmissionFailures();
  FailedSubmission();
  RestorePreemptsYield();
  CancelledStagingWait();
  CancelledPublicationWait();
  ConcurrentStop();
}
