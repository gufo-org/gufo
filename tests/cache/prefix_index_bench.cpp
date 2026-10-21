#include <chrono>
#include <iostream>

#include "tests/cache/prefix_index_fixture.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;
int main() {
  ResourceLedger ledger{kIndexLimits};
  {
    PrefixIndex index{ledger};
    index.Register({1}, kIndexComponents);
    std::vector<Token> prompt(131072);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = static_cast<Token>(i % 30000);
    const auto empty = index.MetadataBytes();
    for (std::size_t i = 1; i <= 128; ++i)
      (void)index.Insert(
          IndexCheckpoint(ledger, std::span(prompt).first(i * 1024)),
          ResidentComponents());
    const auto retained = index.MetadataBytes();
    PrefixQuery query{{1}, prompt, InputIdentity(prompt.size()), 0, true};
    std::uint64_t checksum = 0;
    constexpr unsigned iterations = 1000;
    for (unsigned i = 0; i < 20; ++i)
      checksum += index.Lookup(query).selected->boundary;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) {
      const auto result = index.Lookup(query);
      if (!result.selected || result.selected->boundary != prompt.size() ||
          result.candidates.size() != 128)
        return 1;
      checksum += result.selected->boundary;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto presence_start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) {
      if (!index.HasCheckpoint(query, prompt.size() - 4096))
        return 1;
      ++checksum;
    }
    const auto presence_elapsed =
        std::chrono::steady_clock::now() - presence_start;
    std::cout
        << "tokens=131072 checkpoints=128 iterations=" << iterations
        << " lookup_us="
        << std::chrono::duration<double, std::micro>(elapsed).count() /
               iterations
        << " presence_us="
        << std::chrono::duration<double, std::micro>(presence_elapsed).count() /
               iterations
        << " empty_index_bytes=" << empty
        << " retained_index_bytes=" << retained
        << " incremental_bytes_per_checkpoint="
        << static_cast<double>(retained - empty) / 128
        << " checksum=" << checksum << '\n';
  }
  if (ledger.Snapshot().total_bytes != 0)
    return 1;
}
