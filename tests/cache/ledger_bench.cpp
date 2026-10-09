#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#include "src/cache/ledger.hpp"

using namespace gufo::cache;
namespace {
using Clock = std::chrono::steady_clock;
using Samples = std::array<std::vector<std::uint64_t>, 3>;
std::uint64_t Ns(Clock::duration elapsed) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}
void Run(unsigned count, ResourceCategory category, bool lock_timing) {
  const bool pooled = category != ResourceCategory::kMetadata;
  constexpr unsigned iterations = 20000;
  ResourceLedger ledger({1 << 20, 1 << 20, 0, 0}, lock_timing);
  std::vector<ResourceCharge> pools;
  if (pooled)
    for (unsigned i = 0; i != count; ++i)
      pools.push_back(
          ledger.Reserve(ResourceCategory::kBackingFree, 64).Convert());
  const auto initial_costs = ledger.LockCosts();
  std::vector<Samples> samples(count);
  for (auto& thread_samples : samples)
    for (auto& operation : thread_samples)
      operation.resize(iterations);
  std::barrier start(count + 1);
  std::vector<std::thread> threads;
  for (unsigned t = 0; t != count; ++t)
    threads.emplace_back([&, t] {
      start.arrive_and_wait();
      for (unsigned i = 0; i != iterations; ++i) {
        const auto begin = Clock::now();
        auto reservation =
            pooled ? pools[t].ReserveBacking(category)
                   : ledger.Reserve(ResourceCategory::kMetadata, 64);
        const auto reserved = Clock::now();
        auto charge = reservation.Convert();
        const auto converted = Clock::now();
        charge = {};
        const auto released = Clock::now();
        samples[t][0][i] = Ns(reserved - begin);
        samples[t][1][i] = Ns(converted - reserved);
        samples[t][2][i] = Ns(released - converted);
      }
    });
  const auto begin = Clock::now();
  start.arrive_and_wait();
  for (auto& thread : threads)
    thread.join();
  const auto elapsed = Ns(Clock::now() - begin);
  const auto costs = ledger.LockCosts();
  const char* kind = "metadata_allocation";
  if (category == ResourceCategory::kBackingAssigned)
    kind = "spill_block";
  else if (category == ResourceCategory::kPrivateState)
    kind = "private_state_block";
  else if (category == ResourceCategory::kPrivateTail)
    kind = "private_tail_block";
  std::cout << "{\"threads\":" << count
            << ",\"iterations_per_thread\":" << iterations << ",\"kind\":\""
            << kind << "\",\"lock_timing\":" << (lock_timing ? "true" : "false")
            << ",\"wall_ns\":" << elapsed << ",\"operations\":{";
  constexpr std::array names{"reserve", "convert", "release"};
  constexpr std::array steps{LedgerStep::kReserve, LedgerStep::kConvert,
                             LedgerStep::kRelease};
  for (unsigned op = 0; op != 3; ++op) {
    std::vector<std::uint64_t> merged;
    for (auto& thread_samples : samples)
      merged.insert(merged.end(), thread_samples[op].begin(),
                    thread_samples[op].end());
    std::sort(merged.begin(), merged.end());
    std::uint64_t sum = 0;
    for (auto ns : merged)
      sum += ns;
    if (op)
      std::cout << ',';
    const auto index = static_cast<std::size_t>(steps[op]);
    const auto calls = costs[index].calls - initial_costs[index].calls;
    std::cout << '"' << names[op] << "\":{\"mean_ns\":" << sum / merged.size()
              << ",\"p50_ns\":" << merged[merged.size() / 2]
              << ",\"p95_ns\":" << merged[merged.size() * 95 / 100]
              << ",\"p99_ns\":" << merged[merged.size() * 99 / 100];
    if (calls)
      std::cout << ",\"lock_calls\":" << calls << ",\"mean_lock_wait_ns\":"
                << (costs[index].wait_ns - initial_costs[index].wait_ns) / calls
                << ",\"mean_lock_hold_ns\":"
                << (costs[index].hold_ns - initial_costs[index].hold_ns) / calls
                << ",\"max_lock_hold_ns\":" << costs[index].max_hold_ns;
    std::cout << '}';
  }
  std::cout << "}}\n";
}
}  // namespace
int main() {
  for (bool lock_timing : {false, true})
    for (auto category :
         {ResourceCategory::kMetadata, ResourceCategory::kBackingAssigned,
          ResourceCategory::kPrivateState, ResourceCategory::kPrivateTail})
      for (unsigned threads : {1U, 8U})
        Run(threads, category, lock_timing);
}
