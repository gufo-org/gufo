#include "tests/cache/fake_adapter.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gufo::cache::testing {
struct FakeTransfer {
  std::function<TransferResult()> run;
  std::optional<TransferResult> result;
  std::weak_ptr<FakeStreamState> stream;
};

struct FakeStreamState {
  std::deque<std::shared_ptr<FakeTransfer>> jobs;
  bool running{false};
  TransferResult Advance() noexcept {
    // A callback cannot synchronously wait for itself or later FIFO work.
    // This is a programming error; fail immediately instead of spinning.
    if (running)
      std::terminate();
    if (jobs.empty())
      return TransferResult::kSucceeded;
    auto job = jobs.front();
    jobs.pop_front();
    running = true;
    try {
      job->result = job->run();
    } catch (...) {
      job->result = TransferResult::kFailed;
    }
    running = false;
    job->run = {};
    return *job->result;
  }
};

namespace {
TransferResult WaitFor(const std::shared_ptr<FakeTransfer>& job) noexcept {
  if (!job->result) {
    const auto stream = job->stream.lock();
    if (!stream || stream->running)
      std::terminate();
    while (!job->result)
      (void)stream->Advance();
  }
  return *job->result;
}

constexpr std::uint64_t kSeed = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kMultiplier = 0x100000001b3ULL;
std::uint64_t RowHash(std::span<const Token> tokens) {
  std::uint64_t hash = kSeed;
  for (const Token token : tokens)
    hash = (hash ^ token) * kMultiplier;
  return hash;
}
// Execution depends on both restored private state and preceding row bytes.
// It is deliberately separate from structural validation of a checkpoint.
std::uint64_t AdvanceHash(std::uint64_t hash, std::span<const Token> rows,
                          std::size_t first) {
  auto context = RowHash(rows.first(first));
  for (const Token token : rows.subspan(first)) {
    hash = (hash ^ context ^ token) * kMultiplier;
    context = (context ^ token) * kMultiplier;
  }
  return hash;
}

class FakeSlot final : public Slot {
public:
  explicit FakeSlot(MutationGuard& guard) : guard_(guard) {}
  FakeSlot(const FakeSlot&) = delete;
  FakeSlot& operator=(const FakeSlot&) = delete;
  FakeSlot(FakeSlot&&) = delete;
  FakeSlot& operator=(FakeSlot&&) = delete;
  ~FakeSlot() override {
    for (const auto& transfer : transfers)
      (void)WaitFor(transfer);
    ReleaseRows();
  }
  bool IsValid() const noexcept override { return valid; }
  void ReleaseRows() noexcept {
    if (!target.empty())
      guard_.BeforeRelease(kTarget, 0, target.size());
    if (!draft.empty())
      guard_.BeforeRelease(kDraft, 0, draft.size());
  }
  bool Busy() const noexcept {
    return pending_reads != 0 || pending_loads != 0;
  }
  MutationGuard& guard_;
  std::vector<Token> target, draft;
  // Target/draft boundaries and their private recurrent hash chains.
  std::array<std::uint64_t, 4> recurrent{0, 0, kSeed, kSeed};
  std::array<ComponentPosition, 3> expected{};
  std::vector<std::uint8_t> target_loaded, draft_loaded;
  mutable std::vector<std::shared_ptr<FakeTransfer>> transfers;
  mutable std::size_t pending_reads{0};
  std::size_t pending_loads{0};
  bool valid{true};
  bool restoring{false};
  bool private_loaded{false};
  bool failed{false};
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
  if (id != kRecurrent || bytes != 4 * sizeof(std::uint64_t))
    throw std::invalid_argument("invalid private component or buffer size");
}
Rows Position(std::span<const ComponentPosition> positions, ComponentId id) {
  if (std::ranges::count(positions, id, &ComponentPosition::id) != 1)
    throw std::invalid_argument("missing or duplicate component position");
  return std::ranges::find(positions, id, &ComponentPosition::id)->valid_rows;
}
Rows ContiguousRows(const std::vector<std::uint8_t>& loaded) {
  return static_cast<Rows>(std::ranges::find(loaded, 0) - loaded.begin());
}
void RequireCapture(const FakeSlot& slot) {
  if (!slot.valid || slot.restoring || slot.failed)
    throw std::logic_error("cannot capture an invalid slot");
}
void RequireLoad(const FakeSlot& slot) {
  if (!slot.restoring || slot.failed)
    throw std::logic_error("load requires an unfailed BeginRestore");
}
}  // namespace

class FakeStream::Signal final : public CompletionSignal {
public:
  Signal(std::shared_ptr<FakeStreamState> state,
         std::shared_ptr<FakeTransfer> job)
      : state_(std::move(state)), job_(std::move(job)) {}
  bool Ready() const noexcept override { return job_->result.has_value(); }
  TransferResult Wait() noexcept override { return WaitFor(job_); }

private:
  std::shared_ptr<FakeStreamState> state_;
  std::shared_ptr<FakeTransfer> job_;
};

FakeStream::FakeStream(bool delayed)
    : state_(std::make_shared<FakeStreamState>()), delayed_(delayed) {}
FakeStream::~FakeStream() {
  (void)Synchronize();
}
TransferResult FakeStream::Advance() noexcept {
  return state_->Advance();
}
TransferResult FakeStream::Synchronize() noexcept {
  if (state_->running)
    std::terminate();
  TransferResult result = TransferResult::kSucceeded;
  while (!state_->jobs.empty()) {
    if (Advance() == TransferResult::kFailed)
      result = TransferResult::kFailed;
  }
  return result;
}
Completion FakeStream::Submit(std::function<TransferResult()> run) {
  return SubmitTracked(std::move(run), nullptr);
}
void FakeStream::FailNextSubmission() noexcept {
  fail_submission_ = true;
}
Completion FakeStream::SubmitTracked(
    std::function<TransferResult()> run,
    std::vector<std::shared_ptr<FakeTransfer>>* transfers) {
  if (state_->running)
    throw std::logic_error(
        "submission must run on the host, outside stream jobs");
  if (std::exchange(fail_submission_, false))
    throw std::bad_alloc();
  auto job = std::make_shared<FakeTransfer>();
  job->run = std::move(run);
  job->stream = state_;
  auto signal = std::make_unique<Signal>(state_, job);
  if (transfers) {
    std::erase_if(*transfers,
                  [](const auto& old) { return old->result.has_value(); });
    transfers->push_back(job);
  }
  try {
    state_->jobs.push_back(job);
  } catch (...) {
    if (transfers)
      transfers->pop_back();
    throw;
  }
  if (!delayed_)
    (void)Synchronize();
  return Completion(std::move(signal));
}

Capabilities FakeAdapter::GetCapabilities() const {
  return {.continuation = true};
}
std::span<const ComponentDescriptor> FakeAdapter::Components() const {
  static constexpr std::array<ComponentDescriptor, 3> components{{
      {kTarget, 1, ComponentKind::kAppendRows, sizeof(Token), 4, 0},
      {kDraft, 1, ComponentKind::kAppendRows, sizeof(Token), 4, 0},
      {kRecurrent, 1, ComponentKind::kPrivateState, 0, 0, 32},
  }};
  return components;
}
Identity FakeAdapter::CompatibilityIdentity() const {
  return {'f', 'a', 'k', 'e', 2};
}
std::unique_ptr<Slot> FakeAdapter::CreateSlot(MutationGuard& guard) {
  if (std::exchange(fail_allocation_, false))
    throw std::bad_alloc();
  return std::make_unique<FakeSlot>(guard);
}
std::vector<ComponentPosition> FakeAdapter::Positions(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  return {{kTarget, state.restoring ? ContiguousRows(state.target_loaded)
                                    : state.target.size()},
          {kDraft, state.restoring ? ContiguousRows(state.draft_loaded)
                                   : state.draft.size()},
          {kRecurrent, state.recurrent[0]}};
}
void FakeAdapter::BeginRestore(Slot& slot,
                               std::span<const ComponentPosition> positions) {
  auto& state = AsSlot(slot);
  if (state.Busy() || state.failed || state.restoring)
    throw std::logic_error(
        "reset failed restores and settle transfers before BeginRestore");
  if (positions.size() != 3)
    throw std::invalid_argument("restore requires every component position");
  const Rows target_end = Position(positions, kTarget);
  const Rows draft_end = Position(positions, kDraft);
  if (Position(positions, kRecurrent) != target_end)
    throw std::invalid_argument(
        "private boundary differs from target boundary");
  CheckRange(0, target_end,
             static_cast<std::size_t>(target_end) * sizeof(Token));
  CheckRange(0, draft_end, static_cast<std::size_t>(draft_end) * sizeof(Token));
  if (std::exchange(fail_allocation_, false))
    throw std::bad_alloc();
  std::vector<Token> target(static_cast<std::size_t>(target_end));
  std::vector<Token> draft(static_cast<std::size_t>(draft_end));
  std::vector<std::uint8_t> target_loaded(target.size()),
      draft_loaded(draft.size());
  // All preservation runs while the old rows/private state remain readable.
  state.guard_.BeforeOverwrite(kTarget, 0,
                               std::max<Rows>(state.target.size(), target_end));
  state.guard_.BeforeOverwrite(kDraft, 0,
                               std::max<Rows>(state.draft.size(), draft_end));
  if (state.Busy())
    throw std::logic_error(
        "guard returned with preservation transfers pending");
  state.target = std::move(target);
  state.draft = std::move(draft);
  state.target_loaded = std::move(target_loaded);
  state.draft_loaded = std::move(draft_loaded);
  state.expected = {
      {{kTarget, target_end}, {kDraft, draft_end}, {kRecurrent, target_end}}};
  state.recurrent = {0, 0, kSeed, kSeed};
  state.private_loaded = false;
  state.restoring = true;
  state.valid = false;
}

Completion FakeAdapter::SubmitRead(const Slot& slot, Stream& stream,
                                   std::function<TransferResult()> operation) {
  const auto& state = AsSlot(slot);
  auto& fifo = AsStream(stream);
  const bool fail = fail_transfer_;
  ++state.pending_reads;
  try {
    auto completion = fifo.SubmitTracked(
        [&state, operation = std::move(operation), fail] {
          TransferResult result = TransferResult::kFailed;
          try {
            if (!fail)
              result = operation();
          } catch (...) {
            result = TransferResult::kFailed;
          }
          --state.pending_reads;
          return result;
        },
        &state.transfers);
    fail_transfer_ =
        false;  // Only consume injection after successful submission.
    return completion;
  } catch (...) {
    --state.pending_reads;
    throw;
  }
}
Completion FakeAdapter::SubmitLoad(Slot& slot, Stream& stream,
                                   std::function<TransferResult()> operation) {
  auto& state = AsSlot(slot);
  ++state.pending_loads;
  try {
    auto& fifo = AsStream(stream);
    const bool fail = fail_transfer_;
    auto completion = fifo.SubmitTracked(
        [&state, operation = std::move(operation), fail] {
          TransferResult result = TransferResult::kFailed;
          try {
            if (!fail)
              result = operation();
          } catch (...) {
            result = TransferResult::kFailed;
          }
          --state.pending_loads;
          state.failed |= result == TransferResult::kFailed;
          return result;
        },
        &state.transfers);
    fail_transfer_ = false;
    return completion;
  } catch (...) {
    --state.pending_loads;
    state.failed = true;
    throw;
  }
}
Completion FakeAdapter::CapturePrivate(const Slot& slot, ComponentId id,
                                       std::span<std::byte> dst,
                                       Stream& stream) {
  CheckPrivate(id, dst.size());
  const auto& state = AsSlot(slot);
  RequireCapture(state);
  return SubmitRead(slot, stream, [&state, dst] {
    std::memcpy(dst.data(), state.recurrent.data(), dst.size());
    return TransferResult::kSucceeded;
  });
}
Completion FakeAdapter::CopyRowsOut(const Slot& slot, ComponentId id,
                                    Rows first, Rows end,
                                    std::span<std::byte> dst, Stream& stream) {
  CheckRange(first, end, dst.size());
  const auto& state = AsSlot(slot);
  RequireCapture(state);
  const auto& rows = RowArray(state, id);
  if (end > rows.size())
    throw std::invalid_argument("cannot capture unavailable rows");
  return SubmitRead(slot, stream, [&rows, first, dst] {
    if (!dst.empty())
      std::memcpy(dst.data(), rows.data() + first, dst.size());
    return TransferResult::kSucceeded;
  });
}
Completion FakeAdapter::CopyRowsIn(Slot& slot, ComponentId id, Rows first,
                                   Rows end, std::span<const std::byte> src,
                                   Stream& stream) {
  auto& state = AsSlot(slot);
  RequireLoad(state);
  try {
    CheckRange(first, end, src.size());
    auto& rows = RowArray(state, id);
    if (end > rows.size())
      throw std::invalid_argument("restore range exceeds prepared capacity");
    return SubmitLoad(slot, stream, [&state, id, first, end, src] {
      auto& destination = RowArray(state, id);
      auto& loaded = id == kTarget ? state.target_loaded : state.draft_loaded;
      if (!src.empty())
        std::memcpy(destination.data() + first, src.data(), src.size());
      std::ranges::fill(
          std::span(loaded).subspan(static_cast<std::size_t>(first),
                                    static_cast<std::size_t>(end - first)),
          std::uint8_t{1});
      return TransferResult::kSucceeded;
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
Completion FakeAdapter::LoadPrivate(Slot& slot, ComponentId id,
                                    std::span<const std::byte> src,
                                    Stream& stream) {
  auto& state = AsSlot(slot);
  RequireLoad(state);
  try {
    CheckPrivate(id, src.size());
    return SubmitLoad(slot, stream, [&state, src] {
      std::memcpy(state.recurrent.data(), src.data(), src.size());
      state.private_loaded = true;
      return TransferResult::kSucceeded;
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
bool FakeAdapter::Validate(Slot& slot,
                           std::span<const ComponentPosition> positions) {
  auto& state = AsSlot(slot);
  if (!state.restoring)
    throw std::logic_error("validation requires an active restore");
  const auto actual = Positions(slot);
  const bool complete =
      !state.Busy() && !state.failed && state.private_loaded &&
      positions.size() == 3 &&
      ContiguousRows(state.target_loaded) == state.target.size() &&
      ContiguousRows(state.draft_loaded) == state.draft.size() &&
      std::ranges::all_of(state.expected,
                          [&](const auto& position) {
                            return std::ranges::count(positions, position) ==
                                       1 &&
                                   std::ranges::count(actual, position) == 1;
                          }) &&
      state.recurrent[1] == state.draft.size();
  state.valid = complete;
  state.failed |= !complete;
  if (complete)
    state.restoring = false;
  return complete;
}
bool FakeAdapter::Invalidate(Slot& slot) noexcept {
  auto& state = AsSlot(slot);
  if (state.Busy())
    return false;
  state.ReleaseRows();
  state.target.clear();
  state.draft.clear();
  state.target_loaded.clear();
  state.draft_loaded.clear();
  state.expected = {};
  state.recurrent = {0, 0, kSeed, kSeed};
  state.transfers.clear();
  state.restoring = false;
  state.failed = false;
  state.private_loaded = false;
  state.valid = true;
  return true;
}
void FakeAdapter::Append(Slot& slot, std::span<const Token> target,
                         std::span<const Token> draft) {
  auto& state = AsSlot(slot);
  if (!state.valid || state.failed || state.restoring || state.Busy())
    throw std::logic_error(
        "cannot execute an invalid slot or mutate pending transfers");
  auto new_target = state.target;
  auto new_draft = state.draft;
  new_target.insert(new_target.end(), target.begin(), target.end());
  new_draft.insert(new_draft.end(), draft.begin(), draft.end());
  const auto target_hash =
      AdvanceHash(state.recurrent[2], new_target, state.target.size());
  const auto draft_hash =
      AdvanceHash(state.recurrent[3], new_draft, state.draft.size());
  state.guard_.BeforeOverwrite(kTarget, 0, new_target.size());
  state.guard_.BeforeOverwrite(kDraft, 0, new_draft.size());
  if (state.Busy())
    throw std::logic_error(
        "guard returned with preservation transfers pending");
  state.target = std::move(new_target);
  state.draft = std::move(new_draft);
  state.recurrent = {state.target.size(), state.draft.size(), target_hash,
                     draft_hash};
}
std::uint64_t FakeAdapter::RecurrentHash(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  return state.recurrent[2] * kMultiplier + state.recurrent[3];
}
void FakeAdapter::FailNextAllocation() noexcept {
  fail_allocation_ = true;
}
void FakeAdapter::FailNextTransfer() noexcept {
  fail_transfer_ = true;
}
}  // namespace gufo::cache::testing
