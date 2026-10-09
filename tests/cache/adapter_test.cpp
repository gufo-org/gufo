#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cache/events.hpp"
#include "src/cache/identity.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;

namespace {
class Guard final : public MutationGuard {
public:
  void BeforeOverwrite(ComponentId id, Rows first, Rows end) override {
    assert(first <= end);
    calls.push_back({id, first, end});
    if (reject)
      throw std::runtime_error("preservation refused");
  }
  struct Call {
    ComponentId id;
    Rows first;
    Rows end;
  };
  std::vector<Call> calls;
  bool reject{false};
};

void RoundTrip() {
  FakeAdapter adapter;
  Guard guard;
  auto source = adapter.CreateSlot(guard);
  const std::array<Token, 5> target{11, 22, 33, 44, 55};
  const std::array<Token, 3> draft{11, 22, 33};
  adapter.Append(*source, target, draft);
  const auto positions = adapter.Positions(*source);
  assert((positions == std::vector<ComponentPosition>{
                           {kTarget, 5}, {kDraft, 3}, {kRecurrent, 5}}));
  assert(adapter.capabilities().continuation);
  assert(adapter.Components().size() == 3);
  assert(adapter.CompatibilityIdentity() ==
         FakeAdapter{}.CompatibilityIdentity());

  FakeStream stream(true);
  std::array<std::byte, 24> state{};
  std::array<std::byte, 20> target_bytes{};
  std::array<std::byte, 12> draft_bytes{};
  auto private_copy =
      adapter.CapturePrivate(*source, kRecurrent, state, stream);
  auto target_copy =
      adapter.CopyRowsOut(*source, kTarget, 0, 5, target_bytes, stream);
  auto draft_copy =
      adapter.CopyRowsOut(*source, kDraft, 0, 3, draft_bytes, stream);
  assert(!private_copy.Ready());
  assert(stream.Advance() == TransferResult::kSucceeded);
  assert(private_copy.Ready());
  assert(!target_copy.Ready());
  assert(draft_copy.Wait() == TransferResult::kSucceeded);
  assert(target_copy.Ready());
  assert(private_copy.Wait() == TransferResult::kSucceeded);

  auto restored = adapter.CreateSlot(guard);
  auto target_load =
      adapter.CopyRowsIn(*restored, kTarget, 0, 5, target_bytes, stream);
  auto draft_load =
      adapter.CopyRowsIn(*restored, kDraft, 0, 3, draft_bytes, stream);
  auto private_load = adapter.LoadPrivate(*restored, kRecurrent, state, stream);
  assert(!restored->IsValid());
  assert(!adapter.Validate(*restored, positions));
  assert(private_load.Wait() == TransferResult::kSucceeded);
  assert(target_load.Wait() == TransferResult::kSucceeded);
  assert(draft_load.Wait() == TransferResult::kSucceeded);
  assert(adapter.Validate(*restored, positions));
  assert(restored->IsValid());
  assert(adapter.RecurrentHash(*restored) == adapter.RecurrentHash(*source));

  const std::array<Token, 1> suffix{66};
  adapter.Append(*source, suffix, suffix);
  adapter.Append(*restored, suffix, suffix);
  assert(adapter.RecurrentHash(*restored) == adapter.RecurrentHash(*source));
  // Coherent private state must reject rows from another computation.
  restored = adapter.CreateSlot(guard);
  assert(
      adapter.CopyRowsIn(*restored, kDraft, 0, 3, draft_bytes, stream).Wait() ==
      TransferResult::kSucceeded);
  target_bytes.front() ^= std::byte{1};
  assert(adapter.CopyRowsIn(*restored, kTarget, 0, 5, target_bytes, stream)
             .Wait() == TransferResult::kSucceeded);
  assert(adapter.LoadPrivate(*restored, kRecurrent, state, stream).Wait() ==
         TransferResult::kSucceeded);
  assert(!adapter.Validate(*restored, positions));
  assert(!restored->IsValid());
}

void PartialAndBoundedRestore() {
  FakeAdapter adapter;
  Guard guard;
  FakeStream stream;
  auto source = adapter.CreateSlot(guard);
  const std::array<Token, 5> target{1, 2, 3, 4, 5};
  const std::array<Token, 2> draft{1, 2};
  adapter.Append(*source, target, draft);
  const auto positions = adapter.Positions(*source);
  std::array<std::byte, 24> state{};
  std::array<std::byte, 20> rows{};
  std::array<std::byte, 8> draft_rows{};
  assert(adapter.CapturePrivate(*source, kRecurrent, state, stream).Wait() ==
         TransferResult::kSucceeded);
  assert(adapter.CopyRowsOut(*source, kTarget, 0, 5, rows, stream).Wait() ==
         TransferResult::kSucceeded);
  assert(
      adapter.CopyRowsOut(*source, kDraft, 0, 2, draft_rows, stream).Wait() ==
      TransferResult::kSucceeded);
  auto restored = adapter.CreateSlot(guard);
  // Private state may load before bounded row pieces, but cannot execute yet.
  assert(adapter.LoadPrivate(*restored, kRecurrent, state, stream).Wait() ==
         TransferResult::kSucceeded);
  assert(adapter
             .CopyRowsIn(*restored, kTarget, 0, 2,
                         std::span<const std::byte>(rows).first(8), stream)
             .Wait() == TransferResult::kSucceeded);
  assert(!adapter.Validate(*restored, positions));
  assert(adapter
             .CopyRowsIn(*restored, kTarget, 2, 5,
                         std::span<const std::byte>(rows).subspan(8), stream)
             .Wait() == TransferResult::kSucceeded);
  assert(!adapter.Validate(*restored, positions));  // Draft is still missing.
  assert(
      adapter.CopyRowsIn(*restored, kDraft, 0, 2, draft_rows, stream).Wait() ==
      TransferResult::kSucceeded);
  assert(adapter.Validate(*restored, positions));
  auto duplicate = positions;
  duplicate.back() = duplicate.front();
  assert(!adapter.Validate(*restored, duplicate));
  auto reordered = positions;
  std::reverse(reordered.begin(), reordered.end());
  assert(adapter.Validate(*restored, reordered));
  // A bounded overwrite must leave rows after end intact.
  assert(adapter
             .CopyRowsIn(*restored, kTarget, 0, 2,
                         std::span<const std::byte>(rows).first(8), stream)
             .Wait() == TransferResult::kSucceeded);
  assert(adapter.Validate(*restored, positions));
  draft_rows.front() ^= std::byte{1};
  assert(
      adapter.CopyRowsIn(*restored, kDraft, 0, 2, draft_rows, stream).Wait() ==
      TransferResult::kSucceeded);
  assert(!adapter.Validate(*restored, positions));
}

void CompletionOwnership() {
  FakeStream stream(true);
  int first = 0, second = 0;
  auto one = stream.Submit([&] {
    ++first;
    return TransferResult::kFailed;
  });
  auto two = stream.Submit([&] {
    ++second;
    return TransferResult::kSucceeded;
  });
  one = std::move(two);  // Old work drains before its handle is replaced.
  assert(first == 1 && second == 0);
  // Wait has an explicit moved-from contract.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assert(two.Wait() == TransferResult::kFailed);
  auto moved = std::move(one);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assert(one.Wait() == TransferResult::kFailed);
  assert(moved.Wait() == TransferResult::kSucceeded);
  assert(second == 1);
  auto completed = std::move(moved);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assert(moved.Wait() == TransferResult::kFailed);
  assert(completed.Wait() == TransferResult::kSucceeded);
  auto failed = stream.Submit([] { return TransferResult::kFailed; });
  auto passed = stream.Submit([] { return TransferResult::kSucceeded; });
  assert(stream.Synchronize() == TransferResult::kFailed);
  assert(failed.Wait() == TransferResult::kFailed);
  assert(passed.Wait() == TransferResult::kSucceeded);
}

void FailuresAndLifetime() {
  FakeAdapter adapter;
  Guard guard;
  adapter.FailNextAllocation();
  bool failed = false;
  try {
    (void)adapter.CreateSlot(guard);
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  assert(failed);
  auto slot = adapter.CreateSlot(guard);
  const std::array<Token, 2> tokens{7, 8};
  adapter.Append(*slot, tokens, tokens);
  const auto before = adapter.RecurrentHash(*slot);
  guard.reject = true;
  try {
    adapter.Append(*slot, tokens, tokens);
    assert(false);
  } catch (const std::runtime_error&) {
    assert(adapter.RecurrentHash(*slot) == before);
    assert(adapter.Positions(*slot).front().valid_rows == 2);
  }
  guard.reject = false;

  FakeStream stream(true);
  std::array<std::byte, 8> bytes{};
  adapter.FailNextTransfer();
  auto failure = adapter.CopyRowsOut(*slot, kTarget, 0, 2, bytes, stream);
  assert(!failure.Ready());
  assert(failure.Wait() == TransferResult::kFailed);
  assert(failure.Wait() == TransferResult::kFailed);
  assert(slot->IsValid());
  {
    auto copy = adapter.CopyRowsOut(*slot, kTarget, 0, 2, bytes, stream);
    assert(!copy.Ready());
  }  // Destroying a completion drains its transfer before releasing buffers.
  assert(bytes.front() == std::byte{7});
  adapter.FailNextTransfer();
  auto load = adapter.CopyRowsIn(*slot, kTarget, 0, 2, bytes, stream);
  assert(load.Wait() == TransferResult::kFailed);
  assert(!slot->IsValid());
  assert(!adapter.Validate(*slot, adapter.Positions(*slot)));
  try {
    adapter.Append(*slot, tokens, tokens);
    assert(false);
  } catch (const std::invalid_argument&) {
    assert(!slot->IsValid());
  }
  adapter.Invalidate(*slot);
  assert(adapter.Positions(*slot).front().valid_rows == 0);

  bool invalid_range = false;
  try {
    (void)adapter.CopyRowsIn(*slot, kTarget, 2, 1, bytes, stream);
  } catch (const std::invalid_argument&) {
    invalid_range = true;
  }
  assert(invalid_range);
  slot = adapter.CreateSlot(guard);
  guard.calls.clear();
  adapter.Append(*slot, tokens, tokens);
  slot.reset();
  assert(guard.calls.size() ==
         4);  // Append and destruction protect both arrays.
}

void InputBoundaries() {
  const Identity image_a{1, 2, 3}, image_b{4, 5, 6};
  const std::vector<InputPrefix> before_image{{3, {}}};
  assert(PrefixInputIdentity(image_a, before_image, 3).empty());
  assert(PrefixInputIdentity(image_a, before_image, 4).size() == 3);
  const InputIdentity first{image_a, {{3, {}}}};
  const InputIdentity changed{image_b, {{3, {}}}};
  assert(first.Matches(changed, 3));
  assert(!first.Matches(changed, 4));
  const InputIdentity appended{{9}, {{3, {}}, {5, image_a}}};
  assert(first.Matches(appended, 4));
  assert(first.Matches(appended, 5));
  assert(!first.Matches(appended, 6));
  bool unordered = false;
  try {
    (void)InputIdentity{image_a, {{5, {}}, {3, image_a}}};
  } catch (const std::invalid_argument&) {
    unordered = true;
  }
  assert(unordered);
}
}  // namespace

int main() {
  RoundTrip();
  PartialAndBoundedRestore();
  CompletionOwnership();
  FailuresAndLifetime();
  InputBoundaries();
  std::cout << "cache adapter contracts passed\n";
}
