#include <chrono>
#include <iostream>

#include "src/cache/slot.hpp"
#include "tests/cache/fake_adapter.hpp"
using namespace gufo::cache;
using namespace gufo::cache::testing;
int main() {
  ResourceLedger ledger({1 << 20, 1 << 20, 0, 0});
  FakeAdapter adapter;
  FakeStream stream;
  LeasedSlot slot(ledger, adapter, stream, SlotId{1});
  auto lease = slot.Acquire();
  const auto live = lease.Location();
  lease.Commit();
  constexpr int iterations = 100000;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto l = slot.Acquire(live);
    l.Commit();
  }
  auto elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "live acquire/commit/release ns/op "
            << std::chrono::duration<double, std::nano>(elapsed).count() /
                   iterations
            << '\n';
  class Empty final : public MutationGuard {
    void BeforeOverwrite(ComponentId, Rows, Rows) override {}
    void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
  } empty;
  auto control = adapter.CreateSlot(empty);
  auto active = slot.Acquire(live);
  auto measure = [&](Slot& s) {
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      adapter.GuardRows(s, kTarget, 0, 1);
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - begin)
               .count() /
           iterations;
  };
  auto guarded = measure(active.Execution());
  auto noop = measure(*control);
  std::cout << "empty guard ns/op " << guarded << " noop ns/op " << noop
            << " incremental guard ns/op " << guarded - noop << '\n';
}
