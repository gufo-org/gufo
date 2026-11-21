// CPU checks of the remap barrier's bookkeeping (src/mem/remap_barrier.hpp):
// which addresses are clean, which remaps wait for a barrier, and how other
// threads' device work is held back while one runs. Whether the device then
// reaches the new memory is mem_region_test's part.
#include "src/mem/remap_barrier.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

using gufo::mem::DeviceWorkScope;
using gufo::mem::RemapBarrier;
using Clock = std::chrono::steady_clock;

std::atomic<int> drains{0};
std::atomic<bool> in_drain{false};
std::atomic<int> entered_during_drain{0};

void Stamps() {
  // An address that never had a mapping is clean.
  assert(RemapBarrier::Clean(0));
  const auto first = RemapBarrier::Stamp();
  assert(first != 0 && !RemapBarrier::Clean(first) && !RemapBarrier::Pending());
  RemapBarrier::NoteRemap();
  assert(RemapBarrier::Pending());
  const auto before = RemapBarrier::Stats();
  const auto start = Clock::now();
  RemapBarrier::Settle();
  // The barrier kept the device idle for its whole idle time.
  assert(Clock::now() - start >=
         std::chrono::microseconds(RemapBarrier::kIdleMicros));
  assert(RemapBarrier::Clean(first) && !RemapBarrier::Pending());
  // What is unmapped afterwards waits for the next barrier.
  const auto second = RemapBarrier::Stamp();
  assert(second > first && !RemapBarrier::Clean(second));
  const auto after = RemapBarrier::Stats();
  assert(after.settles == before.settles + 1 &&
         after.micros >= before.micros + RemapBarrier::kIdleMicros);
  RemapBarrier::Settle();
  assert(RemapBarrier::Clean(second));
}

void Drain() {
  RemapBarrier::SetDeviceDrain(+[]() noexcept {
    // Returns at once, as a wait on an idle device does: a slow one would
    // read as work still running.
    in_drain.store(true);
    ++drains;
    in_drain.store(false);
  });
  const int before = drains.load();
  RemapBarrier::Settle();
  // Once before the idle time and once after it, to find work still running.
  assert(drains.load() >= before + 2);
}

// Device work another thread brackets never starts inside a barrier, and a
// barrier waits for the work that is open when it begins.
void HoldsOthersBack() {
  std::atomic<bool> stop{false};
  std::atomic<long> scopes{0};
  std::thread worker([&] {
    while (!stop.load()) {
      {
        const DeviceWorkScope work;
        if (in_drain.load())
          ++entered_during_drain;
        ++scopes;
      }
      std::this_thread::yield();
    }
  });
  while (scopes.load() < 100)
    std::this_thread::yield();
  const auto before = RemapBarrier::Stats();
  for (int i = 0; i < 20; ++i) {
    const auto stamp = RemapBarrier::Stamp();
    RemapBarrier::NoteRemap();
    RemapBarrier::Settle();
    assert(RemapBarrier::Clean(stamp) && !RemapBarrier::Pending());
    // The held worker resumes once the barrier ends. Barriers back to back
    // close the gate again before a waiter wakes, so give it the chance.
    const long resumed = scopes.load();
    const auto deadline = Clock::now() + std::chrono::seconds(1);
    while (scopes.load() == resumed && Clock::now() < deadline)
      std::this_thread::yield();
    assert(scopes.load() > resumed);
  }
  stop.store(true);
  worker.join();
  // The worker ran between barriers and was never inside one.
  assert(entered_during_drain.load() == 0);
  // Whether each barrier found the device quiet depends on this machine's
  // scheduling; that all of them returned is the point.
  assert(RemapBarrier::Stats().settles == before.settles + 20);
}

// A hold keeps other threads' device work back until it is released, and
// one that nobody releases ends by itself.
void HoldAndRelease() {
  std::atomic<bool> entered{false};
  RemapBarrier::Hold();
  std::thread worker([&] {
    const DeviceWorkScope work;
    entered.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  assert(!entered.load());
  RemapBarrier::Release();
  worker.join();
  assert(entered.load());
  // Released by Settle() as well.
  entered.store(false);
  RemapBarrier::Hold();
  std::thread second([&] {
    const DeviceWorkScope work;
    entered.store(true);
  });
  RemapBarrier::Settle();
  second.join();
  assert(entered.load());
  // Never released: the gate opens after kHoldLimit.
  RemapBarrier::Hold();
  const auto start = Clock::now();
  std::thread third([] { const DeviceWorkScope work; });
  third.join();
  assert(Clock::now() - start >= RemapBarrier::kHoldLimit / 2);
  RemapBarrier::Release();
}

// A scope that never closes cannot hang the model: the barrier gives up
// after a bounded wait and says so.
void BoundedWait() {
  RemapBarrier::Enter();
  const auto before = RemapBarrier::Stats();
  std::thread settler([] { RemapBarrier::Settle(); });
  settler.join();
  assert(RemapBarrier::Stats().unquiet > before.unquiet);
  RemapBarrier::Leave();
}

}  // namespace

int main() {
  Stamps();
  Drain();
  HoldsOthersBack();
  HoldAndRelease();
  BoundedWait();
  std::puts("mem_remap_barrier_test: ok");
  return 0;
}
