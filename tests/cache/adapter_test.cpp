#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/cache/identity.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;

namespace {
template<class Exception, class Function>
void Throws(Function&& function) {
  bool caught = false;
  try {
    std::forward<Function>(function)();
  } catch (const Exception&) {
    caught = true;
  }
  assert(caught);
}
class Guard final : public MutationGuard {
public:
  void BeforeOverwrite(ComponentId, Rows first, Rows end) override {
    assert(first <= end);
    if (reject)
      throw std::runtime_error("preservation refused");
  }
  void BeforeRelease(ComponentId, Rows first, Rows end) noexcept override {
    assert(first <= end);
  }
  bool reject{false};
};
struct Checkpoint {
  std::vector<ComponentPosition> positions;
  std::vector<std::byte> target, draft, state;
};
Checkpoint Save(FakeAdapter& adapter, const Slot& slot, FakeStream& stream) {
  Checkpoint saved{adapter.Positions(slot), {}, {}, {}};
  saved.target.resize(static_cast<std::size_t>(saved.positions[0].valid_rows) *
                      sizeof(Token));
  saved.draft.resize(static_cast<std::size_t>(saved.positions[1].valid_rows) *
                     sizeof(Token));
  saved.state.resize(adapter.Components().back().state_bytes);
  auto a = adapter.CopyRowsOut(slot, kTarget, 0, saved.positions[0].valid_rows,
                               saved.target, stream);
  auto b = adapter.CopyRowsOut(slot, kDraft, 0, saved.positions[1].valid_rows,
                               saved.draft, stream);
  auto c = adapter.CapturePrivate(slot, kRecurrent, saved.state, stream);
  assert(c.Wait() == TransferResult::kSucceeded);
  assert(a.Wait() == TransferResult::kSucceeded);
  assert(b.Wait() == TransferResult::kSucceeded);
  return saved;
}
void Load(FakeAdapter& adapter, Slot& slot, const Checkpoint& saved,
          FakeStream& stream) {
  adapter.BeginRestore(slot, saved.positions);
  auto a = adapter.CopyRowsIn(slot, kTarget, 0, saved.positions[0].valid_rows,
                              saved.target, stream);
  auto b = adapter.CopyRowsIn(slot, kDraft, 0, saved.positions[1].valid_rows,
                              saved.draft, stream);
  auto c = adapter.LoadPrivate(slot, kRecurrent, saved.state, stream);
  assert(c.Wait() == TransferResult::kSucceeded);
  assert(a.Wait() == TransferResult::kSucceeded);
  assert(b.Wait() == TransferResult::kSucceeded);
}
const std::array<Token, 5> kTargetTokens{11, 22, 33, 44, 55};
const std::array<Token, 3> kDraftTokens{11, 22, 33};
const std::array<Token, 1> kSuffix{66};

void RoundTripAndContentOracle() {
  FakeAdapter adapter;
  Guard guard;
  auto source = adapter.CreateSlot(guard);
  adapter.Append(*source, kTargetTokens, kDraftTokens);
  assert((adapter.Positions(*source) ==
          std::vector<ComponentPosition>{
              {kTarget, 5}, {kDraft, 3}, {kRecurrent, 5}}));
  assert(adapter.GetCapabilities().continuation);
  FakeStream stream(true);
  const auto saved = Save(adapter, *source, stream);
  const auto before = adapter.RecurrentHash(*source);
  adapter.Append(*source, kSuffix, kSuffix);
  const auto expected = adapter.RecurrentHash(*source);
  auto restored = adapter.CreateSlot(guard);
  Load(adapter, *restored, saved, stream);
  auto reordered = saved.positions;
  std::reverse(reordered.begin(), reordered.end());
  assert(adapter.Validate(*restored, reordered));
  assert(adapter.RecurrentHash(*restored) == before);
  adapter.Append(*restored, kSuffix, kSuffix);
  assert(adapter.RecurrentHash(*restored) == expected);

  // Metadata validation cannot verify KV/private contents. The independent
  // execution comparison detects corrupted target, draft and private bytes.
  for (unsigned component = 0; component != 3; ++component) {
    auto corrupted = saved;
    if (component == 0)
      corrupted.target.front() ^= std::byte{1};
    else if (component == 1)
      corrupted.draft.front() ^= std::byte{1};
    else
      corrupted.state[2 * sizeof(std::uint64_t)] ^= std::byte{1};
    assert(adapter.Invalidate(*restored));
    Load(adapter, *restored, corrupted, stream);
    assert(adapter.Validate(*restored, saved.positions));
    adapter.Append(*restored, kSuffix, kSuffix);
    assert(adapter.RecurrentHash(*restored) != expected);
  }
  // A shorter checkpoint restores directly into the longer frontier.
  assert(adapter.Invalidate(*restored));
  adapter.Append(*restored, kTargetTokens, kDraftTokens);
  adapter.Append(*restored, kSuffix, kSuffix);
  Load(adapter, *restored, saved, stream);
  assert(adapter.Validate(*restored, saved.positions));
  assert(adapter.Positions(*restored) == saved.positions);
  adapter.Append(*restored, kSuffix, kSuffix);
  assert(adapter.RecurrentHash(*restored) == expected);
}

void ValidationMisusePreservesLiveState() {
  FakeAdapter adapter;
  Guard guard;
  FakeStream stream(true);
  auto source = adapter.CreateSlot(guard);
  adapter.Append(*source, kTargetTokens, kDraftTokens);
  const auto saved = Save(adapter, *source, stream);
  const auto before = adapter.RecurrentHash(*source);
  adapter.Append(*source, kSuffix, kSuffix);
  const auto expected = adapter.RecurrentHash(*source);

  for (const bool restored : {false, true}) {
    auto slot = adapter.CreateSlot(guard);
    if (restored) {
      Load(adapter, *slot, saved, stream);
      assert(adapter.Validate(*slot, saved.positions));
    } else {
      adapter.Append(*slot, kTargetTokens, kDraftTokens);
    }
    Throws<std::logic_error>(
        [&] { (void)adapter.Validate(*slot, saved.positions); });
    assert(slot->IsValid());
    assert(adapter.RecurrentHash(*slot) == before);
    const auto after = Save(adapter, *slot, stream);
    assert(after.positions == saved.positions);
    assert(after.target == saved.target);
    assert(after.draft == saved.draft);
    assert(after.state == saved.state);
    adapter.Append(*slot, kSuffix, kSuffix);
    assert(adapter.RecurrentHash(*slot) == expected);
  }
}

void OutOfOrderPieces() {
  FakeAdapter adapter;
  Guard guard;
  auto source = adapter.CreateSlot(guard);
  adapter.Append(*source, kTargetTokens, kDraftTokens);
  FakeStream first(true), second(true);
  const auto saved = Save(adapter, *source, first);
  auto slot = adapter.CreateSlot(guard);
  adapter.BeginRestore(*slot, saved.positions);
  auto suffix = adapter.CopyRowsIn(
      *slot, kTarget, 2, 5, std::span<const std::byte>(saved.target).subspan(8),
      first);
  auto prefix = adapter.CopyRowsIn(
      *slot, kTarget, 0, 2, std::span<const std::byte>(saved.target).first(8),
      second);
  assert(suffix.Wait() == TransferResult::kSucceeded);
  assert(adapter.Positions(*slot)[0].valid_rows == 0);
  assert(!adapter.Invalidate(*slot));
  auto draft = adapter.CopyRowsIn(*slot, kDraft, 0, 3, saved.draft, first);
  auto state = adapter.LoadPrivate(*slot, kRecurrent, saved.state, first);
  assert(state.Wait() == TransferResult::kSucceeded);
  assert(draft.Wait() == TransferResult::kSucceeded);
  assert(prefix.Wait() == TransferResult::kSucceeded);
  assert(adapter.Validate(*slot, saved.positions));
  adapter.Append(*slot, kSuffix, kSuffix);
  adapter.Append(*source, kSuffix, kSuffix);
  assert(adapter.RecurrentHash(*slot) == adapter.RecurrentHash(*source));
}

void FailedValidationNeedsReset() {
  FakeAdapter adapter;
  Guard guard;
  auto source = adapter.CreateSlot(guard);
  adapter.Append(*source, kTargetTokens, kDraftTokens);
  FakeStream stream(true);
  const auto saved = Save(adapter, *source, stream);
  auto slot = adapter.CreateSlot(guard);
  // A validation attempted with pending loads poisons this restore permanently.
  adapter.BeginRestore(*slot, saved.positions);
  auto a = adapter.CopyRowsIn(*slot, kTarget, 0, 5, saved.target, stream);
  auto b = adapter.CopyRowsIn(*slot, kDraft, 0, 3, saved.draft, stream);
  auto c = adapter.LoadPrivate(*slot, kRecurrent, saved.state, stream);
  assert(!adapter.Validate(*slot, saved.positions));
  assert(!adapter.Invalidate(*slot));
  assert(c.Wait() == TransferResult::kSucceeded);
  assert(a.Wait() == TransferResult::kSucceeded);
  assert(b.Wait() == TransferResult::kSucceeded);
  assert(!adapter.Validate(*slot, saved.positions));
  assert(adapter.Invalidate(*slot));
  assert(slot->IsValid());

  for (unsigned missing = 0; missing != 3; ++missing) {
    adapter.BeginRestore(*slot, saved.positions);
    if (missing != 0)
      assert(adapter.CopyRowsIn(*slot, kTarget, 0, 5, saved.target, stream)
                 .Wait() == TransferResult::kSucceeded);
    if (missing != 1)
      assert(
          adapter.CopyRowsIn(*slot, kDraft, 0, 3, saved.draft, stream).Wait() ==
          TransferResult::kSucceeded);
    if (missing != 2)
      assert(
          adapter.LoadPrivate(*slot, kRecurrent, saved.state, stream).Wait() ==
          TransferResult::kSucceeded);
    assert(!adapter.Validate(*slot, saved.positions));
    assert(!adapter.Validate(*slot, adapter.Positions(*slot)));
    assert(adapter.Invalidate(*slot));
  }
  Load(adapter, *slot, saved, stream);
  auto duplicate = saved.positions;
  duplicate.back() = duplicate.front();
  assert(!adapter.Validate(*slot, duplicate));
  assert(!adapter.Validate(*slot, saved.positions));
  assert(adapter.Invalidate(*slot));
  Load(adapter, *slot, saved, stream);
  auto wrong = saved.positions;
  --wrong.front().valid_rows;
  assert(!adapter.Validate(*slot, wrong));
  assert(!adapter.Validate(*slot, saved.positions));
  assert(adapter.Invalidate(*slot));
  Load(adapter, *slot, saved, stream);
  assert(adapter.Validate(*slot, saved.positions));
}

void DiscardedLoadFailures() {
  for (const bool replace : {false, true}) {
    FakeAdapter adapter;
    Guard guard;
    auto source = adapter.CreateSlot(guard);
    adapter.Append(*source, kTargetTokens, kDraftTokens);
    FakeStream stream(true);
    const auto saved = Save(adapter, *source, stream);
    auto slot = adapter.CreateSlot(guard);
    Load(adapter, *slot, saved, stream);
    // All components already exist: a failed rewrite must still reject them.
    adapter.FailNextTransfer();
    {
      auto failed =
          adapter.CopyRowsIn(*slot, kTarget, 0, 5, saved.target, stream);
      if (replace)
        failed = adapter.LoadPrivate(*slot, kRecurrent, saved.state, stream);
      // Destruction/replacement discards the failed result, not the slot latch.
    }
    assert(!slot->IsValid());
    assert(!adapter.Validate(*slot, saved.positions));
    assert(adapter.Invalidate(*slot));
    adapter.Append(*slot, kTargetTokens, kDraftTokens);
  }
}

class ForeignStream final : public Stream {
public:
  TransferResult Synchronize() noexcept override {
    return TransferResult::kSucceeded;
  }
};
void InjectionSurvivesSubmissionFailure() {
  FakeAdapter adapter;
  Guard guard;
  adapter.FailNextAllocation();
  Throws<std::bad_alloc>([&] { (void)adapter.CreateSlot(guard); });
  auto slot = adapter.CreateSlot(guard);
  adapter.Append(*slot, kTargetTokens, kDraftTokens);
  FakeStream stream(true);
  ForeignStream foreign;
  const auto saved = Save(adapter, *slot, stream);
  const auto before = adapter.RecurrentHash(*slot);
  adapter.FailNextAllocation();
  try {
    adapter.BeginRestore(*slot, saved.positions);
    assert(false);
  } catch (const std::bad_alloc&) {
    assert(slot->IsValid());
  }
  guard.reject = true;
  try {
    adapter.BeginRestore(*slot, saved.positions);
    assert(false);
  } catch (const std::runtime_error&) {
    assert(adapter.RecurrentHash(*slot) == before);
  }
  guard.reject = false;
  std::vector<std::byte> bytes(saved.state.size());
  adapter.FailNextTransfer();
  Throws<std::bad_cast>(
      [&] { (void)adapter.CapturePrivate(*slot, kRecurrent, bytes, foreign); });
  stream.FailNextSubmission();
  Throws<std::bad_alloc>(
      [&] { (void)adapter.CapturePrivate(*slot, kRecurrent, bytes, stream); });
  assert(adapter.CapturePrivate(*slot, kRecurrent, bytes, stream).Wait() ==
         TransferResult::kFailed);
  adapter.BeginRestore(*slot, saved.positions);
  adapter.FailNextTransfer();
  stream.FailNextSubmission();
  Throws<std::bad_alloc>([&] {
    (void)adapter.LoadPrivate(*slot, kRecurrent, saved.state, stream);
  });
  assert(!adapter.Validate(*slot, saved.positions));
  assert(adapter.Invalidate(*slot));
  assert(adapter.CapturePrivate(*slot, kRecurrent, bytes, stream).Wait() ==
         TransferResult::kFailed);
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
  one = std::move(two);
  assert(first == 1 && second == 0);
  // The public contract explicitly supports Wait on moved-from handles.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  assert(two.Wait() == TransferResult::kFailed);
  auto moved = std::move(one);
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

void ReentrantWaitFailsImmediately() {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    std::set_terminate([] { _exit(86); });
    FakeStream stream(true);
    std::optional<Completion> completion;
    completion.emplace(stream.Submit([&] {
      (void)completion->Wait();
      return TransferResult::kSucceeded;
    }));
    (void)completion->Wait();
    _exit(1);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
void InputBoundaries() {
  const Identity image_a{1, 2, 3}, image_b{4, 5, 6};
  const InputIdentity first(8, image_a, {{3, {}}});
  const InputIdentity changed(8, image_b, {{3, {}}});
  assert(first.At(3).empty());
  assert(first.At(4).size() == 3);
  assert(first.Matches(changed, 3));
  assert(!first.Matches(changed, 4));
  const InputIdentity appended(8, {9}, {{3, {}}, {5, image_a}});
  assert(first.Matches(appended, 4));
  assert(first.Matches(appended, 5));
  assert(!first.Matches(appended, 6));
  Throws<std::invalid_argument>(
      [&] { (void)InputIdentity(8, image_a, {{5, {}}, {3, image_a}}); });
  Throws<std::invalid_argument>([&] { (void)first.At(9); });
}
}  // namespace
int main() {
  RoundTripAndContentOracle();
  ValidationMisusePreservesLiveState();
  OutOfOrderPieces();
  FailedValidationNeedsReset();
  DiscardedLoadFailures();
  InjectionSurvivesSubmissionFailure();
  CompletionOwnership();
  ReentrantWaitFailsImmediately();
  InputBoundaries();
  std::cout << "cache adapter contracts passed\n";
}
