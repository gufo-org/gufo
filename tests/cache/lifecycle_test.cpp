#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/cache/identity.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;

namespace {
class Guard final : public MutationGuard {
public:
  void BeforeOverwrite(ComponentId, Rows, Rows) override {
    if (preserve)
      preserve();
    if (reject)
      throw std::runtime_error("refused");
  }
  void BeforeRelease(ComponentId, Rows, Rows) noexcept override { ++releases; }
  std::function<void()> preserve;
  bool reject{false};
  unsigned releases{0};
};

void ReleaseAfterRefusal() {
  FakeAdapter adapter;
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const std::array<Token, 2> tokens{1, 2};
  adapter.Append(*slot, tokens, tokens);
  guard.reject = true;
  try {
    adapter.Append(*slot, tokens, tokens);
    assert(false);
  } catch (const std::runtime_error&) {
    assert(slot->IsValid());
  }
  assert(adapter.Invalidate(*slot));
  assert(slot->IsValid());
  guard.reject = false;
  adapter.Append(*slot, tokens, tokens);  // Cold execution after reset.
  guard.reject = true;
  slot.reset();  // Must use the nonthrowing release path.
  assert(guard.releases == 4);
}

void CapturePinsAndDestruction() {
  FakeAdapter adapter;
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const std::array<Token, 2> tokens{1, 2};
  adapter.Append(*slot, tokens, tokens);
  const auto before = adapter.Positions(*slot);
  FakeStream stream(true);
  std::vector<std::byte> state(adapter.Components().back().state_bytes);
  auto capture = adapter.CapturePrivate(*slot, kRecurrent, state, stream);
  assert(!capture.Ready());
  try {
    adapter.Append(*slot, tokens, tokens);
    assert(false);
  } catch (const std::logic_error&) {
    assert(adapter.Positions(*slot) == before);
  }
  assert(!adapter.Invalidate(*slot));
  assert(adapter.Positions(*slot) == before);
  try {
    adapter.BeginRestore(*slot, before);
    assert(false);
  } catch (const std::logic_error&) {
    assert(adapter.Positions(*slot) == before);
  }
  assert(capture.Wait() == TransferResult::kSucceeded);
  std::array<std::byte, 8> rows{};
  auto read = adapter.CopyRowsOut(*slot, kTarget, 0, 2, rows, stream);
  assert(!adapter.Invalidate(*slot));
  slot.reset();  // Destruction must drain reads before freeing the rows.
  assert(read.Ready());
  assert(read.Wait() == TransferResult::kSucceeded);
  assert(rows.front() == std::byte{1});
}

void HostPreservation(bool delayed) {
  FakeAdapter adapter;
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const std::array<Token, 2> tokens{7, 8};
  adapter.Append(*slot, tokens, tokens);
  const auto positions = adapter.Positions(*slot);
  FakeStream stream(delayed), preserve_stream(true);
  std::array<std::byte, 8> rows{};
  std::vector<std::byte> state(adapter.Components().back().state_bytes);
  assert(adapter.CopyRowsOut(*slot, kTarget, 0, 2, rows, stream).Wait() ==
         TransferResult::kSucceeded);
  assert(adapter.CapturePrivate(*slot, kRecurrent, state, stream).Wait() ==
         TransferResult::kSucceeded);
  unsigned preserved = 0;
  guard.preserve = [&] {
    assert(slot->IsValid());
    std::array<std::byte, 8> saved{};
    assert(adapter.CopyRowsOut(*slot, kTarget, 0, 2, saved, preserve_stream)
               .Wait() == TransferResult::kSucceeded);
    assert(saved == rows);
    ++preserved;
  };
  adapter.BeginRestore(*slot, positions);
  assert(preserved == 2);
  guard.preserve = {};
  auto a = adapter.CopyRowsIn(*slot, kTarget, 0, 2, rows, stream);
  auto b = adapter.CopyRowsIn(*slot, kDraft, 0, 2, rows, stream);
  auto c = adapter.LoadPrivate(*slot, kRecurrent, state, stream);
  assert(c.Wait() == TransferResult::kSucceeded);
  assert(a.Wait() == TransferResult::kSucceeded);
  assert(b.Wait() == TransferResult::kSucceeded);
  assert(adapter.Validate(*slot, positions));
}

void IdentityBounds() {
  static_assert(!std::is_convertible_v<Identity, InputIdentity>);
  const InputIdentity input(8, {9}, {{3, {}}, {3, {1}}});
  assert(input.At(3).empty());  // Equal boundaries retain the first match.
  bool failed = false;
  try {
    (void)InputIdentity(8, {9}, {{10, {}}});
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  assert(failed);
}

void GuardMustSettlePreservation() {
  FakeAdapter adapter;
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const std::array<Token, 2> tokens{1, 2};
  adapter.Append(*slot, tokens, tokens);
  const auto positions = adapter.Positions(*slot);
  FakeStream stream(true);
  std::array<std::byte, 8> saved{};
  std::vector<Completion> copies;
  guard.preserve = [&] {
    copies.push_back(adapter.CopyRowsOut(*slot, kTarget, 0, 2, saved, stream));
  };
  try {
    adapter.BeginRestore(*slot, positions);
    assert(false);
  } catch (const std::logic_error&) {
    assert(slot->IsValid());
    assert(adapter.Positions(*slot) == positions);
  }
  guard.preserve = {};
  copies.clear();
  adapter.Append(*slot, tokens, tokens);
}
}  // namespace

int main() {
  ReleaseAfterRefusal();
  CapturePinsAndDestruction();
  HostPreservation(false);
  HostPreservation(true);
  GuardMustSettlePreservation();
  IdentityBounds();
  std::cout << "cache adapter lifecycle passed\n";
}
