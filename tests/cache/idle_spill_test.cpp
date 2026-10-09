#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/cache/checkpoint.hpp"
#include "src/cache/slot.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;
using namespace std::chrono_literals;
namespace {
// A submitted read completes only after the test grants one permit. The fake
// adapter owns actual row copying; this wrapper controls completion, not bytes.
struct Gate {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::pair<Rows, Rows>> ranges;
  int permits{};
  void WaitFor(std::size_t n) {
    std::unique_lock lock(mutex);
    assert(changed.wait_for(lock, 5s, [&] { return ranges.size() >= n; }));
  }
  void Allow(int n = 1) {
    std::lock_guard lock(mutex);
    permits += n;
    changed.notify_all();
  }
};
class Signal final : public CompletionSignal {
public:
  Signal(Gate& g, Completion c) : gate(g), copy(std::move(c)) {}
  bool Ready() const noexcept override { return false; }
  TransferResult Wait() noexcept override {
    {
      std::unique_lock lock(gate.mutex);
      gate.changed.wait(lock, [&] { return gate.permits > 0; });
      --gate.permits;
    }
    return copy.Wait();
  }
  Gate& gate;
  Completion copy;
};
class AdapterFixture final : public Adapter {
public:
  FakeAdapter fake;
  Gate gate;
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
    return fake.CreateSlot(g);
  }
  std::vector<ComponentPosition> Positions(const Slot& s) const override {
    return fake.Positions(s);
  }
  std::vector<Rows> PlanPrefill(const Slot& s, Rows a, Rows b) const override {
    return fake.PlanPrefill(s, a, b);
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
    auto copy = fake.CopyRowsOut(s, c, a, b, v, t);
    {
      std::lock_guard lock(gate.mutex);
      gate.ranges.emplace_back(a, b);
      gate.changed.notify_all();
    }
    return Completion(std::make_unique<Signal>(gate, std::move(copy)));
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
struct LedgerAudit {
  ResourceLedger& ledger;
  ~LedgerAudit() { assert(ledger.Snapshot().total_bytes == 0); }
};
struct Fixture {
  ResourceLedger ledger{{1 << 20, 1 << 20, 0, 1 << 20}};
  LedgerAudit audit{ledger};
  AdapterFixture adapter;
  FakeStream foreground;
  std::atomic<bool> device_idle{true}, streams_available{true};
  std::atomic<int> acquired{}, stream_attempts{};
  LeasedSlot slot{ledger, adapter, foreground, SlotId{1},
                  IdleSpillConfig{8,
                                  [this]() -> std::unique_ptr<Stream> {
                                    ++stream_attempts;
                                    if (!streams_available)
                                      return {};
                                    ++acquired;
                                    return std::make_unique<FakeStream>(true);
                                  },
                                  [this] { return device_idle.load(); }}};
  SlotLease lease{slot.Acquire()};
  std::array<Token, 5> tokens{1, 2, 3, 4, 5};
  std::shared_ptr<std::vector<std::byte>> buffer{
      std::make_shared<std::vector<std::byte>>(sizeof(tokens))};
  ResourceCharge pool{
      ledger.Reserve(ResourceCategory::kBackingFree, sizeof(tokens)).Convert()};
  std::shared_ptr<BorrowedRows> rows;
  Fixture() {
    adapter.fake.Append(lease.Execution(), tokens, tokens);
    rows = lease.Borrow(kTarget, 0, tokens.size(),
                        pool.ReserveBacking(ResourceCategory::kBackingAssigned),
                        buffer, *buffer);
  }
  ~Fixture() {
    adapter.gate.Allow(100);
    slot.StopIdleSpill();
    lease = {};
  }
  void Verify() {
    assert(rows->IsValid());
    assert(!rows->Location());
    auto pin = rows->Pin();
    assert(pin.Owner());
    assert(std::memcmp(buffer->data(), tokens.data(), sizeof(tokens)) == 0);
  }
};
void IdleAndZeroIdle() {
  for (bool idle : {false, true}) {
    Fixture f;
    f.device_idle = idle;
    const auto before = f.slot.Metrics();
    f.lease.Commit();
    if (idle) {
      f.adapter.gate.WaitFor(1);
      assert(f.rows->Location());
      assert(f.rows->Pin()
                 .Location());  // A partial chunk remains source-readable.
      f.adapter.gate.Allow(3);
      f.adapter.gate.WaitFor(3);
      while (!f.slot.PreservationComplete())
        std::this_thread::yield();
      f.Verify();
    } else {
      f.adapter.gate.Allow(3);
    }
    f.lease = f.slot.Acquire();
    f.Verify();
    const auto metrics = f.slot.Metrics();
    assert(metrics.reassignments == before.reassignments + 1);
    assert(metrics.residual_wait_ns > before.residual_wait_ns);
    assert(metrics.idle_bytes == (idle ? sizeof(f.tokens) : 0));
    assert(metrics.foreground_bytes == (idle ? 0 : sizeof(f.tokens)));
    assert(f.acquired == 3);  // Fresh stream lease for every bounded piece.
    std::lock_guard lock(f.adapter.gate.mutex);
    assert((f.adapter.gate.ranges ==
            std::vector<std::pair<Rows, Rows>>{{0, 2}, {2, 4}, {4, 5}}));
  }
}
void MidSpill() {
  Fixture f;
  f.lease.Commit();
  f.adapter.gate.WaitFor(1);
  f.adapter.gate.Allow();
  f.adapter.gate.WaitFor(2);
  assert(f.rows->Location());  // Even a completed first piece is unpublished.
  auto request =
      std::async(std::launch::async, [&] { return f.slot.Acquire(); });
  assert(request.wait_for(20ms) == std::future_status::timeout);
  f.device_idle = false;
  f.adapter.gate.Allow(2);
  f.lease = request.get();
  f.Verify();
  assert(f.slot.Metrics().idle_bytes == 16);
  assert(f.slot.Metrics().foreground_bytes == 4);
}
void CancellationAndPins() {
  Fixture f;
  f.lease.Commit();
  f.adapter.gate.WaitFor(1);
  std::stop_source cancel;
  auto request = std::async(std::launch::async, [&] {
    try {
      (void)f.slot.Acquire({}, cancel.get_token());
      assert(false);
    } catch (const std::runtime_error&) {
    }
  });
  assert(request.wait_for(20ms) == std::future_status::timeout);
  cancel.request_stop();
  assert(request.wait_for(1s) == std::future_status::ready);
  request.get();
  auto shutdown =
      std::async(std::launch::async, [&] { f.slot.StopIdleSpill(); });
  assert(shutdown.wait_for(20ms) == std::future_status::timeout);
  f.adapter.gate.Allow();
  shutdown.get();
  assert(f.rows->Location());
  // Source pin from the cancelled worker is gone: reassignment completes once
  // the explicitly held reader releases, with no orphan preservation reader.
  auto pin = f.rows->Pin();
  f.adapter.gate.Allow(2);
  auto replacement =
      std::async(std::launch::async, [&] { return f.slot.Acquire(); });
  f.adapter.gate.WaitFor(3);
  assert(replacement.wait_for(20ms) == std::future_status::timeout);
  pin = {};
  f.lease = replacement.get();
  f.Verify();
  assert(f.ledger.Snapshot().persistence_pinned_bytes == 0);
}
void PriorityAndPoolExhaustion() {
  Fixture f;
  f.device_idle = false;
  f.lease.Commit();
  std::this_thread::sleep_for(10ms);
  assert(f.acquired == 0);
  f.streams_available = false;
  f.device_idle = true;
  std::this_thread::sleep_for(10ms);
  assert(f.acquired == 0);
  assert(f.slot.Metrics().idle_failures == 0);
  f.streams_available = true;
  f.adapter.gate.WaitFor(1);
  f.device_idle = false;
  f.adapter.gate.Allow();
  while (f.slot.Metrics().idle_bytes != 8)
    std::this_thread::yield();
  std::this_thread::sleep_for(10ms);
  assert(f.acquired == 1);  // Scheduler is consulted between pieces.
  f.adapter.gate.Allow(2);
  f.lease = f.slot.Acquire();
  f.Verify();
}
void ForegroundStreamWait() {
  Fixture f;
  f.device_idle = false;
  f.streams_available = false;
  f.lease.Commit();
  auto request =
      std::async(std::launch::async, [&] { return f.slot.Acquire(); });
  assert(request.wait_for(20ms) == std::future_status::timeout);
  f.streams_available = true;
  f.adapter.gate.Allow(3);
  f.lease = request.get();
  f.Verify();
  assert(f.slot.Metrics().stream_wait_ns >= 20000000);
}
void CancelForegroundStreamWait() {
  Fixture f;
  f.device_idle = false;
  auto live = f.lease.Location();
  f.lease.Commit();
  std::stop_source stop;
  f.lease = f.slot.Acquire(live, stop.get_token());
  f.streams_available = false;
  auto mutation = std::async(std::launch::async, [&] {
    try {
      f.adapter.fake.GuardRows(f.lease.Execution(), kTarget, 0, 5);
      assert(false);
    } catch (const std::runtime_error&) {
    }
  });
  assert(mutation.wait_for(20ms) == std::future_status::timeout);
  stop.request_stop();
  mutation.get();
  assert(f.rows->Location());
  assert(f.rows->Pin().Location());
  f.streams_available = true;
  f.adapter.gate.Allow(3);
  f.lease = {};
  f.Verify();
}
void CancelColdReassignment() {
  for (int phase = 0; phase < 3; ++phase) {
    Fixture f;
    f.device_idle = false;
    const auto live = f.lease.Location();
    f.lease.Commit();
    std::stop_source stop;
    RowPin pin;
    if (phase == 0)
      f.streams_available = false;
    if (phase == 2)
      pin = f.rows->Pin();
    auto request = std::async(std::launch::async, [&] {
      try {
        auto lease = f.slot.Acquire({}, stop.get_token());
        return false;
      } catch (const std::runtime_error&) {
        return true;
      }
    });
    if (phase == 0) {
      while (f.stream_attempts < 3)
        std::this_thread::yield();
    } else {
      f.adapter.gate.WaitFor(1);
      if (phase == 2) {
        f.adapter.gate.Allow(3);
        f.adapter.gate.WaitFor(3);
        while (f.rows->Location())
          std::this_thread::yield();
      }
    }
    assert(request.wait_for(20ms) == std::future_status::timeout);
    stop.request_stop();
    if (phase == 1) {
      // Submitted work must settle even after cancellation. Only this piece
      // finishes; the cancelled acquirer cannot enqueue remaining pieces.
      assert(request.wait_for(20ms) == std::future_status::timeout);
      f.adapter.gate.Allow();
    }
    assert(request.wait_for(1s) == std::future_status::ready);
    assert(request.get());
    assert(f.rows->IsValid());
    if (phase != 2)
      assert(f.rows->Location());
    f.lease = f.slot.Acquire(live);
    assert(f.lease.Location() == live);
    assert(f.adapter.Positions(f.lease.Execution())[0].valid_rows == 5);
    pin = {};
    f.streams_available = true;
    f.adapter.gate.Allow(3);
    f.lease = {};
    f.Verify();
  }
}
void Failure() {
  for (bool conversion : {false, true}) {
    Fixture f;
    if (conversion)
      f.ledger.FailAfter(LedgerStep::kConvert, 0);
    else
      f.adapter.fake.FailNextTransfer();
    f.adapter.gate.Allow(3);
    f.lease.Commit();
    while (f.slot.Metrics().idle_failures == 0)
      std::this_thread::yield();
    assert(f.rows->IsValid());
    assert(f.rows->Location());
    f.adapter.gate.Allow(3);
    f.lease = f.slot.Acquire();
    f.Verify();
  }
}
void RetainedCheckpoints() {
  for (int mode = 0; mode < 3; ++mode) {
    Fixture f;
    f.device_idle = mode != 0;
    auto history = ExecutionHistory::Cold(f.ledger, f.adapter.Components(),
                                          f.adapter.CompatibilityIdentity());
    std::vector<ResourceCharge> pools;
    auto checkpoint = history.Capture(
        {f.tokens, InputIdentity(f.tokens.size()),
         f.adapter.Positions(f.lease.Execution()), CheckpointPurpose::kPrompt,
         1},
        [&](const PayloadRequest& request) {
          auto buffer = std::make_shared<std::vector<std::byte>>(request.bytes);
          auto pool =
              f.ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                  .Convert();
          auto reservation = pool.ReserveBacking(request.category);
          pools.push_back(pool);
          if (request.category == ResourceCategory::kPrivateState) {
            auto copy = f.adapter.CapturePrivate(
                f.lease.Execution(), request.component, *buffer, f.foreground);
            assert(copy.Wait() == TransferResult::kSucceeded);
            return Payload::Committed(reservation.Convert(), buffer);
          }
          return Payload::Borrowed(
              f.lease.Borrow(request.component, request.first, request.end,
                             std::move(reservation), buffer, *buffer));
        });
    f.lease.Commit();
    if (mode == 2) {
      f.adapter.gate.WaitFor(1);
      f.adapter.gate.Allow();
      f.adapter.gate.WaitFor(2);
      f.device_idle = false;
    }
    f.adapter.gate.Allow(100);
    if (mode == 1)
      while (!f.slot.PreservationComplete())
        std::this_thread::yield();
    f.lease = f.slot.Acquire();
    assert(checkpoint->IsValid());
    for (const auto& component : checkpoint->Components()) {
      for (const auto& chunk : component.chunks) {
        const auto pin = chunk.Storage().PinRows();
        const auto bytes =
            std::static_pointer_cast<const std::vector<std::byte>>(pin.Owner());
        assert(std::memcmp(bytes->data(), f.tokens.data() + chunk.First(),
                           bytes->size()) == 0);
      }
      if (component.tail) {
        const auto pin = component.tail->PinRows();
        const auto bytes =
            std::static_pointer_cast<const std::vector<std::byte>>(pin.Owner());
        assert(std::memcmp(bytes->data(), f.tokens.data() + 4, bytes->size()) ==
               0);
      }
    }
    auto restored = ExecutionHistory::Restored(f.ledger, *checkpoint);
    assert(restored.Lineage() == history.Lineage());
  }
}
void PersistencePublicationRace() {
  Fixture f;
  f.lease.Commit();
  f.adapter.gate.WaitFor(1);
  std::atomic<bool> finished{};
  auto readers = std::async(std::launch::async, [&] {
    while (!finished) {
      auto pin = f.rows->PinPersistence();
      auto copy = pin;
      auto source = f.rows->Pin();
      assert(source.Location() || source.Owner());
    }
  });
  f.adapter.gate.Allow(3);
  while (!f.slot.PreservationComplete())
    std::this_thread::yield();
  finished = true;
  readers.get();
  f.lease = f.slot.Acquire();
  f.Verify();
}
void SlotPreference() {
  Fixture f;
  f.device_idle = false;
  f.lease.Commit();
  FakeStream stream;
  LeasedSlot complete(f.ledger, f.adapter, stream, SlotId{2});
  std::array<LeasedSlot*, 2> slots{&f.slot, &complete};
  auto chosen = AcquireSlot(slots);
  assert(chosen.Location().slot == SlotId{2});
  f.adapter.gate.Allow(3);
  auto fallback = AcquireSlot(slots);
  assert(fallback.Location().slot == SlotId{1});
  f.Verify();
}
}  // namespace
int main() {
  IdleAndZeroIdle();
  MidSpill();
  CancellationAndPins();
  PriorityAndPoolExhaustion();
  Failure();
  ForegroundStreamWait();
  CancelForegroundStreamWait();
  CancelColdReassignment();
  SlotPreference();
  RetainedCheckpoints();
  PersistencePublicationRace();
}
