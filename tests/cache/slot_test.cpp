#include "src/cache/slot.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/cache/prefix_index.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;
using namespace std::chrono_literals;
namespace {
constexpr ResourceLimits kLimits{1 << 20, 1 << 20, 1 << 20, 1 << 20};
template<class F>
void Throws(F&& f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception&) {
    threw = true;
  }
  assert(threw);
}
struct Fixture {
  ResourceLedger ledger{kLimits};
  FakeAdapter adapter;
  FakeStream stream{true};
  std::unique_ptr<LeasedSlot> slots{
      std::make_unique<LeasedSlot>(ledger, adapter, stream, SlotId{1})};
  SlotLease lease{slots->Acquire()};
  std::vector<ResourceCharge> pools;
  std::vector<Token> tokens{1, 2, 3, 4, 5};
  Fixture() { adapter.Append(lease.Execution(), tokens, tokens); }
  ~Fixture() {
    lease = {};
    slots.reset();
    pools.clear();
    assert(ledger.Snapshot().total_bytes == 0);
  }
  Payload Save(const PayloadRequest& r) {
    auto admission = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes);
    auto buffer = std::make_shared<std::vector<std::byte>>(r.bytes);
    auto pool = admission.Convert();
    auto reservation = pool.ReserveBacking(r.category);
    pools.push_back(pool);
    if (r.category == ResourceCategory::kPrivateState) {
      auto copy = adapter.CapturePrivate(lease.Execution(), r.component,
                                         *buffer, stream);
      assert(copy.Wait() == TransferResult::kSucceeded);
      return Payload::Committed(reservation.Convert(), buffer);
    }
    return Payload::Borrowed(lease.Borrow(
        r.component, r.first, r.end, std::move(reservation), buffer, *buffer));
  }
  std::shared_ptr<const Checkpoint> Capture(ExecutionHistory& history) {
    return history.Capture(
        {tokens, InputIdentity(tokens.size()),
         adapter.Positions(lease.Execution()), CheckpointPurpose::kPrompt, 1},
        [this](const auto& r) { return Save(r); });
  }
  std::vector<std::byte> Read(const Payload& p, ComponentId c, Rows first,
                              Rows end) {
    auto pin = p.PinRows();
    if (pin.Owner())
      return *std::static_pointer_cast<const std::vector<std::byte>>(
          pin.Owner());
    std::vector<std::byte> bytes(p.Bytes());
    auto copy =
        adapter.CopyRowsOut(lease.Execution(), c, first, end, bytes, stream);
    assert(copy.Wait() == TransferResult::kSucceeded);
    return bytes;
  }
};
void Paths() {
  // Prefill/decode share Append; rewind and speculative rollback share the
  // exact-shorter BeginRestore protocol. Reset/reassignment/destruction use
  // the nonthrowing release guard. Every path compares actual retained bytes.
  for (int path = 0; path < 7; ++path) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    const auto& chunk = checkpoint->Components()[0].chunks[0];
    const auto expected = f.Read(chunk.Storage(), kTarget, 0, 4);
    const auto expected_tail =
        f.Read(*checkpoint->Components()[0].tail, kTarget, 4, 5);
    const auto old = f.lease.Location();
    if (path < 2) {
      std::array<Token, 1> suffix{9};
      f.adapter.Append(f.lease.Execution(), suffix, suffix);
    } else if (path < 4) {
      auto positions = f.adapter.Positions(f.lease.Execution());
      for (auto& p : positions)
        p.valid_rows = 2;
      f.adapter.BeginRestore(f.lease.Execution(), positions);
      // Abandon this deliberately incomplete shorter restore; lease RAII
      // resets.
      f.lease = {};
    } else if (path == 4) {
      f.lease.Reset();
      assert(f.lease.Location().generation > old.generation);
    } else if (path == 5) {
      f.lease.Commit();
      f.lease = f.slots->Acquire();
      assert(f.lease.Location().generation > old.generation);
    } else {
      f.lease = {};
    }
    assert(checkpoint->IsValid());
    assert(!chunk.Storage().BorrowedFrom());
    assert(f.Read(chunk.Storage(), kTarget, 0, 4) == expected);
    assert(f.Read(*checkpoint->Components()[0].tail, kTarget, 4, 5) ==
           expected_tail);
  }
}
void Failures() {
  for (int fault = 0; fault < 3; ++fault) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    const auto positions = f.adapter.Positions(f.lease.Execution());
    const auto& chunk = checkpoint->Components()[0].chunks[0];
    const auto bytes = f.Read(chunk.Storage(), kTarget, 0, 4);
    if (fault == 0)
      f.adapter.FailNextAllocation();
    if (fault == 1)
      f.adapter.FailNextTransfer();
    if (fault == 2)
      f.ledger.FailAfter(LedgerStep::kConvert, 0);
    std::array<Token, 1> suffix{9};
    Throws([&] {
      if (fault == 0)
        f.adapter.BeginRestore(f.lease.Execution(), positions);
      else
        f.adapter.Append(f.lease.Execution(), suffix, suffix);
    });
    assert(checkpoint->IsValid());
    assert(f.adapter.Positions(f.lease.Execution()) == positions);
    assert(f.Read(chunk.Storage(), kTarget, 0, 4) == bytes);
    f.ledger.ClearFaults();
  }
}
void EveryPreservationConversion() {
  for (std::size_t failed = 0; failed < 4; ++failed) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    auto positions = f.adapter.Positions(f.lease.Execution());
    auto target =
        f.Read(checkpoint->Components()[0].chunks[0].Storage(), kTarget, 0, 4);
    auto draft =
        f.Read(checkpoint->Components()[1].chunks[0].Storage(), kDraft, 0, 4);
    f.ledger.FailAfter(LedgerStep::kConvert, failed);
    std::array<Token, 1> suffix{9};
    Throws([&] { f.adapter.Append(f.lease.Execution(), suffix, suffix); });
    assert(checkpoint->IsValid());
    assert(f.adapter.Positions(f.lease.Execution()) == positions);
    assert(f.Read(checkpoint->Components()[0].chunks[0].Storage(), kTarget, 0,
                  4) == target);
    assert(f.Read(checkpoint->Components()[1].chunks[0].Storage(), kDraft, 0,
                  4) == draft);
  }
}
void ChunkPinsBlockMutation() {
  Fixture f;
  auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                        f.adapter.CompatibilityIdentity());
  auto checkpoint = f.Capture(history);
  auto reader = checkpoint->Components()[0].chunks[0].PinReader();
  auto persistence = checkpoint->Components()[0].chunks[0].PinPersistence();
  auto worker = std::async(std::launch::async, [&] {
    f.adapter.GuardRows(f.lease.Execution(), kTarget, 0, 1);
  });
  while (checkpoint->Components()[0].chunks[0].Storage().BorrowedFrom())
    std::this_thread::yield();
  // Copying existing pins during mutation must not wait on its own readers.
  auto reader_copy = reader;
  auto persistence_copy = persistence;
  assert(worker.wait_for(20ms) == std::future_status::timeout);
  {
    auto moved = std::move(reader);
    auto moved_persistence = std::move(persistence);
  }
  assert(worker.wait_for(20ms) == std::future_status::timeout);
  reader_copy = checkpoint->Components()[1].chunks[0];
  persistence_copy = checkpoint->Components()[1].chunks[0];
  worker.get();
  assert(checkpoint->IsValid());
}
void ConcurrentPersistenceAdmission() {
  for (int round = 0; round < 20; ++round) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    const auto& chunk = checkpoint->Components()[0].chunks[0];
    std::barrier ready(5);
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i)
      workers.emplace_back([&] {
        ready.arrive_and_wait();
        for (int j = 0; j < 100; ++j) {
          auto pin = chunk.PinPersistence();
          auto copy = pin;
        }
      });
    ready.arrive_and_wait();
    f.adapter.GuardRows(f.lease.Execution(), kTarget, 0, 4);
    for (auto& worker : workers)
      worker.join();
    assert(checkpoint->IsValid());
    assert(!chunk.Storage().BorrowedFrom());
    assert(f.ledger.Snapshot().persistence_pinned_bytes == 0);
  }
}
void Retirement() {
  Fixture f;
  auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                        f.adapter.CompatibilityIdentity());
  auto checkpoint = f.Capture(history);
  PrefixIndex index(f.ledger);
  index.Register(f.adapter.CompatibilityIdentity(), f.adapter.Components());
  (void)index.Insert(checkpoint, {{kTarget, true, false},
                                  {kDraft, true, false},
                                  {kRecurrent, true, false}});
  f.adapter.FailNextTransfer();
  f.lease.Reset();
  assert(!checkpoint->IsValid());
  Throws([&] { (void)ExecutionHistory::Restored(f.ledger, *checkpoint); });
  const PrefixQuery query{f.adapter.CompatibilityIdentity(), f.tokens,
                          InputIdentity(f.tokens.size()), 0, true};
  assert(!index.Lookup(query).selected);
  assert(index.CachedPrefixTokens(query) == 0);
  Throws(
      [&] { (void)checkpoint->Components()[0].chunks[0].Storage().PinRows(); });
}
void PinsAndCancellation() {
  for (bool cancel : {false, true}) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    const auto live = f.lease.Location();
    f.lease.Commit();
    std::stop_source stop;
    f.lease = f.slots->Acquire(live, stop.get_token());
    auto pin = checkpoint->Components()[0].chunks[0].Storage().PinRows();
    Slot& source_slot = f.lease.Execution();
    std::promise<void> started;
    auto worker = std::async(std::launch::async, [&] {
      started.set_value();
      std::array<Token, 1> suffix{9};
      if (cancel)
        Throws([&] { f.adapter.Append(f.lease.Execution(), suffix, suffix); });
      else
        f.adapter.Append(f.lease.Execution(), suffix, suffix);
    });
    started.get_future().wait();
    assert(worker.wait_for(20ms) == std::future_status::timeout);
    // New pins no longer keep a source reader alive after preservation.
    while (checkpoint->Components()[0].chunks[0].Storage().BorrowedFrom())
      std::this_thread::yield();
    auto backing_pin =
        checkpoint->Components()[0].chunks[0].Storage().PinRows();
    assert(backing_pin.Owner());
    // A slow restore can still read its pinned source on a different stream
    // after the guard has copied backing. Overwrite waits through completion.
    FakeStream restore_stream(true);
    std::vector<std::byte> source_bytes(4 * sizeof(Token));
    auto restore_read = f.adapter.CopyRowsOut(source_slot, kTarget, 0, 4,
                                              source_bytes, restore_stream);
    assert(!restore_read.Ready());
    assert(worker.wait_for(20ms) == std::future_status::timeout);
    assert(restore_read.Wait() == TransferResult::kSucceeded);
    assert(source_bytes ==
           *std::static_pointer_cast<const std::vector<std::byte>>(
               backing_pin.Owner()));
    if (cancel) {
      stop.request_stop();
      assert(worker.wait_for(1s) == std::future_status::ready);
      worker.get();
      auto release = std::async(std::launch::async, [&] { f.lease = {}; });
      assert(release.wait_for(20ms) == std::future_status::timeout);
      pin = {};
      release.get();
    } else {
      pin = {};
      worker.get();
    }
    assert(checkpoint->IsValid());
  }
}
void RestoreAndContinue() {
  Fixture f;
  auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                        f.adapter.CompatibilityIdentity());
  auto checkpoint = f.Capture(history);
  FakeStream stream;
  LeasedSlot destination(f.ledger, f.adapter, stream, SlotId{2});
  auto restored = destination.Acquire();
  std::vector<ComponentPosition> positions;
  for (const auto& c : checkpoint->Components())
    positions.push_back(c.position);
  f.adapter.BeginRestore(restored.Execution(), positions);
  for (const auto& c : checkpoint->Components()) {
    auto load = [&](const Payload& p, Rows first, Rows end) {
      auto bytes = f.Read(p, c.descriptor.id, first, end);
      auto copy = f.adapter.CopyRowsIn(restored.Execution(), c.descriptor.id,
                                       first, end, bytes, stream);
      assert(copy.Wait() == TransferResult::kSucceeded);
    };
    for (const auto& chunk : c.chunks)
      load(chunk.Storage(), chunk.First(), chunk.End());
    if (c.tail)
      load(*c.tail,
           c.position.valid_rows -
               c.position.valid_rows % c.descriptor.rows_per_chunk,
           c.position.valid_rows);
    if (c.private_state) {
      auto pin = c.private_state->PinRows();
      auto buffer =
          std::static_pointer_cast<const std::vector<std::byte>>(pin.Owner());
      auto copy = f.adapter.LoadPrivate(restored.Execution(), c.descriptor.id,
                                        *buffer, stream);
      assert(copy.Wait() == TransferResult::kSucceeded);
    }
  }
  assert(f.adapter.Validate(restored.Execution(), positions));
  assert(f.adapter.RecurrentHash(restored.Execution()) ==
         f.adapter.RecurrentHash(f.lease.Execution()));
  auto continuation = ExecutionHistory::Restored(f.ledger, *checkpoint);
  assert(continuation.Lineage() == history.Lineage());
  std::array<Token, 2> suffix{7, 8};
  f.adapter.Append(restored.Execution(), suffix, suffix);
  f.adapter.Append(f.lease.Execution(), suffix, suffix);
  assert(f.adapter.RecurrentHash(restored.Execution()) ==
         f.adapter.RecurrentHash(f.lease.Execution()));
}
void CancellationSteps() {
  for (int phase = 0; phase < 3; ++phase) {
    Fixture f;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    auto checkpoint = f.Capture(history);
    auto live = f.lease.Location();
    f.lease.Commit();
    std::stop_source stop;
    f.lease = f.slots->Acquire(live, stop.get_token());
    const auto original = f.adapter.Positions(f.lease.Execution());
    Slot& source = f.lease.Execution();
    if (phase == 0)
      stop.request_stop();
    std::optional<Completion> cancellation;
    if (phase == 1)
      cancellation.emplace(f.stream.Submit([&] {
        stop.request_stop();
        return TransferResult::kSucceeded;
      }));
    if (phase == 2) {
      auto charge =
          f.ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
      auto buffer = std::make_shared<std::vector<std::byte>>(16);
      stop.request_stop();
      Throws([&] {
        (void)f.lease.Borrow(
            kTarget, 0, 4,
            charge.ReserveBacking(ResourceCategory::kBackingAssigned), buffer,
            *buffer);
      });
      assert(charge.Info().category == ResourceCategory::kBackingFree);
    }
    std::array<Token, 1> suffix{9};
    Throws([&] { f.adapter.Append(source, suffix, suffix); });
    assert(f.adapter.Positions(source) == original);
    assert(checkpoint->IsValid());
    f.lease = {};
    assert(checkpoint->IsValid());
  }
}
void Leases() {
  ResourceLedger ledger{kLimits};
  {
    FakeAdapter adapter;
    FakeStream stream;
    LeasedSlot slot(ledger, adapter, stream, SlotId{1});
    auto lease = slot.Acquire();
    const auto live = lease.Location();
    Throws([&] { (void)slot.Acquire(); });
    lease.Commit();
    lease = slot.Acquire(live);
    lease.Reset();
    const auto reset = lease.Location();
    assert(reset.generation > live.generation);
    lease.Commit();
    Throws([&] { (void)slot.Acquire(live); });
    auto next = slot.Acquire(reset);
    next = {};  // Uncommitted lease invalidates live reuse.
    Throws([&] { (void)slot.Acquire(reset); });
    std::stop_source stop;
    stop.request_stop();
    Throws([&] { (void)slot.Acquire({}, stop.get_token()); });
    std::atomic<int> acquired{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
      threads.emplace_back([&] {
        for (int j = 0; j < 200; ++j) {
          try {
            auto l = slot.Acquire();
            ++acquired;
            l.Commit();
          } catch (const std::runtime_error&) {
          }
        }
      });
    for (auto& t : threads)
      t.join();
    assert(acquired > 0);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void TeardownAndCancelledAdmission() {
  ResourceLedger ledger{kLimits};
  {
    FakeAdapter adapter;
    FakeStream stream;
    auto slot =
        std::make_unique<LeasedSlot>(ledger, adapter, stream, SlotId{1});
    auto lease = slot->Acquire();
    std::array<Token, 4> tokens{1, 2, 3, 4};
    adapter.Append(lease.Execution(), tokens, tokens);
    auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
    auto buffer = std::make_shared<std::vector<std::byte>>(16);
    ledger.FailAfter(LedgerStep::kReserve, 0);
    Throws([&] {
      (void)lease.Borrow(
          kTarget, 0, 4,
          pool.ReserveBacking(ResourceCategory::kBackingAssigned), buffer,
          *buffer);
    });
    assert(pool.Info().category == ResourceCategory::kBackingFree);
    auto rows = lease.Borrow(
        kTarget, 0, 4, pool.ReserveBacking(ResourceCategory::kBackingAssigned),
        buffer, *buffer);
    slot.reset();  // The exclusive lease keeps slot/guard alive.
    lease = {};    // Final destruction preserves retained rows.
    assert(rows->IsValid());
    assert(!rows->Location());
    assert(rows->Pin().Owner());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
// Simulate an adapter that resets its generation while omitting preservation.
class BadAdapter final : public Adapter {
public:
  FakeAdapter fake;
  MutationGuard* guard{};
  Capabilities GetCapabilities() const override {
    return fake.GetCapabilities();
  }
  std::span<const ComponentDescriptor> Components() const override {
    return fake.Components();
  }
  Identity CompatibilityIdentity() const override {
    return fake.CompatibilityIdentity();
  }
  std::unique_ptr<Slot> CreateSlot(MutationGuard& g) override {
    guard = &g;
    return fake.CreateSlot(g);
  }
  std::vector<ComponentPosition> Positions(const Slot& s) const override {
    return fake.Positions(s);
  }
  void BeginRestore(Slot& s, std::span<const ComponentPosition> p) override {
    fake.BeginRestore(s, p);
  }
  Completion CapturePrivate(const Slot& s, ComponentId c,
                            std::span<std::byte> b, Stream& t) override {
    return fake.CapturePrivate(s, c, b, t);
  }
  Completion CopyRowsOut(const Slot& s, ComponentId c, Rows a, Rows b,
                         std::span<std::byte> v, Stream& t) override {
    return fake.CopyRowsOut(s, c, a, b, v, t);
  }
  Completion CopyRowsIn(Slot& s, ComponentId c, Rows a, Rows b,
                        std::span<const std::byte> v, Stream& t) override {
    return fake.CopyRowsIn(s, c, a, b, v, t);
  }
  Completion LoadPrivate(Slot& s, ComponentId c, std::span<const std::byte> b,
                         Stream& t) override {
    return fake.LoadPrivate(s, c, b, t);
  }
  bool Validate(Slot& s, std::span<const ComponentPosition> p) override {
    return fake.Validate(s, p);
  }
  bool Invalidate(Slot& s) noexcept override { return fake.Invalidate(s); }
};
void MissedGuard() {
#ifndef NDEBUG
  const auto child = fork();
  assert(child >= 0);
  if (!child) {
    ResourceLedger ledger{kLimits};
    BadAdapter adapter;
    FakeStream stream;
    LeasedSlot slot(ledger, adapter, stream, SlotId{1});
    auto lease = slot.Acquire();
    std::array<Token, 4> tokens{1, 2, 3, 4};
    adapter.fake.Append(lease.Execution(), tokens, tokens);
    auto pool = ledger.Reserve(ResourceCategory::kBackingFree, 16).Convert();
    auto buffer = std::make_shared<std::vector<std::byte>>(16);
    auto rows = lease.Borrow(
        kTarget, 0, 4, pool.ReserveBacking(ResourceCategory::kBackingAssigned),
        buffer, *buffer);
    adapter.guard->AfterReset();
    (void)rows->Pin();
    _exit(0);
  }
  int status{};
  assert(waitpid(child, &status, 0) == child);
  assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
#endif
}
}  // namespace
int main() {
  Paths();
  Failures();
  EveryPreservationConversion();
  ChunkPinsBlockMutation();
  ConcurrentPersistenceAdmission();
  Retirement();
  PinsAndCancellation();
  Leases();
  RestoreAndContinue();
  CancellationSteps();
  TeardownAndCancelledAdmission();
  MissedGuard();
  std::cout << "slot leases and mutation checks passed\n";
}
