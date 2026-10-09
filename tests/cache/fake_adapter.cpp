#include "tests/cache/fake_adapter.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gufo::cache::testing {
namespace {
struct Job {
  std::function<TransferResult()> run;
  std::optional<TransferResult> result;
};

std::uint64_t Hash(std::span<const Token> tokens) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (Token token : tokens) {
    hash ^= token;
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

std::uint64_t HashState(std::span<const Token> target,
                        std::span<const Token> draft) {
  return Hash(target) * 0x100000001b3ULL + Hash(draft);
}

class FakeSlot final : public Slot {
public:
  explicit FakeSlot(MutationGuard& guard) : guard_(guard) {}
  ~FakeSlot() override { ProtectRows(); }
  bool IsValid() const noexcept override { return valid; }
  void ProtectRows() {
    if (!target.empty())
      guard_.BeforeOverwrite(kTarget, 0, target.size());
    if (!draft.empty())
      guard_.BeforeOverwrite(kDraft, 0, draft.size());
  }
  MutationGuard& guard_;
  std::vector<Token> target, draft;
  // Target boundary, draft boundary, recurrent hash; whole checkpoint-private.
  std::array<std::uint64_t, 3> recurrent{0, 0, HashState({}, {})};
  std::size_t pending_loads{0};
  bool valid{true};
  bool load_failed{false};
};

FakeSlot& AsSlot(Slot& slot) {
  return dynamic_cast<FakeSlot&>(slot);
}
const FakeSlot& AsSlot(const Slot& slot) {
  return dynamic_cast<const FakeSlot&>(slot);
}
FakeStream& AsStream(Stream& stream) {
  return dynamic_cast<FakeStream&>(stream);
}
std::vector<Token>& RowArray(FakeSlot& slot, ComponentId id) {
  if (id == kTarget)
    return slot.target;
  if (id == kDraft)
    return slot.draft;
  throw std::invalid_argument("not an append component");
}
const std::vector<Token>& RowArray(const FakeSlot& slot, ComponentId id) {
  if (id == kTarget)
    return slot.target;
  if (id == kDraft)
    return slot.draft;
  throw std::invalid_argument("not an append component");
}
void CheckRange(Rows first, Rows end, std::size_t bytes) {
  if (first > end ||
      end > std::numeric_limits<std::size_t>::max() / sizeof(Token) ||
      (end - first) * sizeof(Token) != bytes)
    throw std::invalid_argument("invalid row range or buffer size");
}
void CheckPrivate(ComponentId id, std::size_t bytes) {
  if (id != kRecurrent || bytes != 3 * sizeof(std::uint64_t))
    throw std::invalid_argument("invalid private component or buffer size");
}

Completion SubmitLoad(FakeSlot& slot, FakeStream& stream,
                      std::function<TransferResult()> operation) {
  ++slot.pending_loads;
  try {
    auto completion = stream.Submit([&slot, operation = std::move(operation)] {
      TransferResult result = TransferResult::kFailed;
      try {
        result = operation();
      } catch (...) {
        // The completion reports preservation/allocation failures too.
      }
      --slot.pending_loads;
      slot.load_failed |= result == TransferResult::kFailed;
      return result;
    });
    slot.valid = false;
    return completion;
  } catch (...) {
    --slot.pending_loads;
    throw;
  }
}
}  // namespace

struct FakeStream::State {
  std::deque<std::shared_ptr<Job>> jobs;
  TransferResult Advance() noexcept {
    if (jobs.empty())
      return TransferResult::kSucceeded;
    auto job = jobs.front();
    jobs.pop_front();
    try {
      job->result = job->run();
    } catch (...) {
      job->result = TransferResult::kFailed;
    }
    job->run = {};
    return *job->result;
  }
};

class FakeStream::Signal final : public CompletionSignal {
public:
  Signal(std::shared_ptr<State> state, std::shared_ptr<Job> job)
      : state_(std::move(state)), job_(std::move(job)) {}
  bool Ready() const noexcept override { return job_->result.has_value(); }
  TransferResult Wait() noexcept override {
    while (!job_->result)
      (void)state_->Advance();
    return *job_->result;
  }

private:
  std::shared_ptr<State> state_;
  std::shared_ptr<Job> job_;
};

FakeStream::FakeStream(bool delayed)
    : state_(std::make_shared<State>()), delayed_(delayed) {}
FakeStream::~FakeStream() {
  (void)Synchronize();
}
TransferResult FakeStream::Advance() noexcept {
  return state_->Advance();
}
TransferResult FakeStream::Synchronize() noexcept {
  TransferResult result = TransferResult::kSucceeded;
  while (!state_->jobs.empty()) {
    if (Advance() == TransferResult::kFailed)
      result = TransferResult::kFailed;
  }
  return result;
}
Completion FakeStream::Submit(std::function<TransferResult()> run) {
  auto job = std::make_shared<Job>();
  job->run = std::move(run);
  auto signal = std::make_unique<Signal>(state_, job);
  // All throwing allocations precede queue publication.
  state_->jobs.push_back(job);
  if (!delayed_)
    (void)Synchronize();
  return Completion(std::move(signal));
}

Capabilities FakeAdapter::capabilities() const {
  return {.continuation = true};
}
std::span<const ComponentDescriptor> FakeAdapter::Components() const {
  static constexpr std::array<ComponentDescriptor, 3> components{{
      {kTarget, 1, ComponentKind::kAppendRows, sizeof(Token), 4, 0},
      {kDraft, 1, ComponentKind::kAppendRows, sizeof(Token), 4, 0},
      {kRecurrent, 1, ComponentKind::kPrivateState, 0, 0, 24},
  }};
  return components;
}
Identity FakeAdapter::CompatibilityIdentity() const {
  return {'f', 'a', 'k', 'e', 1};
}
std::unique_ptr<Slot> FakeAdapter::CreateSlot(MutationGuard& guard) {
  if (std::exchange(fail_allocation_, false))
    throw std::bad_alloc();
  return std::make_unique<FakeSlot>(guard);
}
std::vector<ComponentPosition> FakeAdapter::Positions(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  return {{kTarget, state.target.size()},
          {kDraft, state.draft.size()},
          {kRecurrent, state.recurrent[0]}};
}
Completion FakeAdapter::CapturePrivate(const Slot& slot, ComponentId id,
                                       std::span<std::byte> dst,
                                       Stream& stream) {
  CheckPrivate(id, dst.size());
  const auto& state = AsSlot(slot);
  if (!state.valid)
    throw std::invalid_argument("cannot capture an invalid slot");
  const bool fail = TakeTransferFailure();
  return AsStream(stream).Submit([&state, dst, fail] {
    if (fail)
      return TransferResult::kFailed;
    std::memcpy(dst.data(), state.recurrent.data(), dst.size());
    return TransferResult::kSucceeded;
  });
}
Completion FakeAdapter::CopyRowsOut(const Slot& slot, ComponentId id,
                                    Rows first, Rows end,
                                    std::span<std::byte> dst, Stream& stream) {
  CheckRange(first, end, dst.size());
  const auto& state = AsSlot(slot);
  const auto& rows = RowArray(state, id);
  if (!state.valid || end > rows.size())
    throw std::invalid_argument("cannot capture unavailable rows");
  const bool fail = TakeTransferFailure();
  return AsStream(stream).Submit([&rows, first, dst, fail] {
    if (fail)
      return TransferResult::kFailed;
    if (!dst.empty())
      std::memcpy(dst.data(), rows.data() + first, dst.size());
    return TransferResult::kSucceeded;
  });
}
Completion FakeAdapter::CopyRowsIn(Slot& slot, ComponentId id, Rows first,
                                   Rows end, std::span<const std::byte> src,
                                   Stream& stream) {
  CheckRange(first, end, src.size());
  auto& state = AsSlot(slot);
  (void)RowArray(state, id);
  const bool fail = TakeTransferFailure();
  return SubmitLoad(
      state, AsStream(stream), [&state, id, first, end, src, fail] {
        if (fail)
          return TransferResult::kFailed;
        auto& rows = RowArray(state, id);
        if (first > rows.size())
          return TransferResult::kFailed;
        // Growing a vector can release borrowed prefix storage too.
        const Rows protected_first = end > rows.capacity() ? 0 : first;
        state.guard_.BeforeOverwrite(id, protected_first, end);
        if (end > rows.size())
          rows.resize(static_cast<std::size_t>(end));
        if (!src.empty())
          std::memcpy(rows.data() + first, src.data(), src.size());
        return TransferResult::kSucceeded;
      });
}
Completion FakeAdapter::LoadPrivate(Slot& slot, ComponentId id,
                                    std::span<const std::byte> src,
                                    Stream& stream) {
  CheckPrivate(id, src.size());
  auto& state = AsSlot(slot);
  const bool fail = TakeTransferFailure();
  return SubmitLoad(state, AsStream(stream), [&state, src, fail] {
    if (fail)
      return TransferResult::kFailed;
    std::memcpy(state.recurrent.data(), src.data(), src.size());
    return TransferResult::kSucceeded;
  });
}
bool FakeAdapter::Validate(Slot& slot,
                           std::span<const ComponentPosition> positions) {
  auto& state = AsSlot(slot);
  const auto actual = Positions(slot);
  state.valid =
      state.pending_loads == 0 && !state.load_failed && positions.size() == 3 &&
      std::ranges::all_of(actual,
                          [&](const auto& position) {
                            return std::ranges::count(positions, position) == 1;
                          }) &&
      state.recurrent[0] == state.target.size() &&
      state.recurrent[1] == state.draft.size() &&
      state.recurrent[2] == HashState(state.target, state.draft);
  return state.valid;
}
void FakeAdapter::Invalidate(Slot& slot) noexcept {
  auto& state = AsSlot(slot);
  state.ProtectRows();
  state.target.clear();
  state.draft.clear();
  state.recurrent = {0, 0, HashState({}, {})};
  state.valid = false;
  state.load_failed = false;
}
void FakeAdapter::Append(Slot& slot, std::span<const Token> target,
                         std::span<const Token> draft) {
  auto& state = AsSlot(slot);
  if (!state.valid || state.pending_loads != 0)
    throw std::invalid_argument("cannot execute a partially restored slot");
  // Stage allocations before any mutation, and guard both components first.
  auto new_target = state.target;
  auto new_draft = state.draft;
  new_target.insert(new_target.end(), target.begin(), target.end());
  new_draft.insert(new_draft.end(), draft.begin(), draft.end());
  // Replacing the vectors releases their old backing, including prefix rows.
  state.guard_.BeforeOverwrite(kTarget, 0, new_target.size());
  state.guard_.BeforeOverwrite(kDraft, 0, new_draft.size());
  state.target = std::move(new_target);
  state.draft = std::move(new_draft);
  state.recurrent = {state.target.size(), state.draft.size(),
                     HashState(state.target, state.draft)};
}
std::uint64_t FakeAdapter::RecurrentHash(const Slot& slot) const {
  return AsSlot(slot).recurrent[2];
}
void FakeAdapter::FailNextAllocation() noexcept {
  fail_allocation_ = true;
}
void FakeAdapter::FailNextTransfer() noexcept {
  fail_transfer_ = true;
}
bool FakeAdapter::TakeTransferFailure() noexcept {
  return std::exchange(fail_transfer_, false);
}

}  // namespace gufo::cache::testing
