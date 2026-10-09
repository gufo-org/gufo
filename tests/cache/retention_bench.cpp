#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <optional>

#include "tests/cache/fake_adapter.hpp"
using namespace gufo::cache;
using namespace gufo::cache::testing;
int main() {
  class Empty final : public MutationGuard {
    void BeforeOverwrite(ComponentId, Rows, Rows) override {}
    void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
  } guard;
  FakeAdapter adapter;
  auto slot = adapter.CreateSlot(guard);
  FakeStream capture_stream(true), other_stream(true);
  std::array<std::byte, 32> bytes;
  std::uint64_t checksum = 0;
  for (int mode = 0; mode < 3; ++mode) {
    double elapsed = 0;
    for (int iteration = 0; iteration < 1100; ++iteration) {
      std::optional<Completion> unrelated;
      if (mode)
        unrelated.emplace(
            (mode == 1 ? capture_stream : other_stream).Submit([&] {
              for (int i = 0; i < 10000; ++i)
                checksum = checksum * 6364136223846793005ULL + 1;
              return TransferResult::kSucceeded;
            }));
      const auto start = std::chrono::steady_clock::now();
      auto transfer =
          adapter.CapturePrivate(*slot, kRecurrent, bytes, capture_stream);
      assert(transfer.Wait() == TransferResult::kSucceeded);
      if (iteration >= 100)
        elapsed += std::chrono::duration<double, std::nano>(
                       std::chrono::steady_clock::now() - start)
                       .count();
      if (mode)
        assert(unrelated->Wait() == TransferResult::kSucceeded);
    }
    std::cout << "host capture submission+wait ns/checkpoint mode=" << mode
              << " " << elapsed / 1000 << '\n';
  }
  std::cout << "checksum " << checksum << '\n';
}
