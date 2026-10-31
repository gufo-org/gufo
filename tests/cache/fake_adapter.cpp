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

constexpr Rows kEmptyRingSlot = std::numeric_limits<Rows>::max();

class FakeSlot final : public Slot {
public:
  FakeSlot(MutationGuard& guard, Rows ring_rows)
      : guard_(guard), ring(ring_rows), occupant(ring_rows, kEmptyRingSlot) {}
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
  bool ExecutionComplete() const noexcept override {
    return execution_complete;
  }
  void ReleaseRows() noexcept {
    if (!target.empty())
      guard_.BeforeRelease(kTarget, 0, target.size());
    if (!draft.empty())
      guard_.BeforeRelease(kDraft, 0, draft.size());
    if (window_rows)
      guard_.BeforeRelease(kWindow, 0, window_rows);
  }
  [[nodiscard]] bool Resident(Rows first, Rows end) const {
    for (Rows row = first; row < end; ++row)
      if (occupant[row % ring.size()] != row)
        return false;
    return true;
  }
  bool Busy() const noexcept {
    return pending_reads != 0 || pending_loads != 0;
  }
  MutationGuard& guard_;
  std::vector<Token> target, draft;
  // Target/draft boundaries and their private recurrent hash chains.
  std::array<std::uint64_t, 4> recurrent{0, 0, kSeed, kSeed};
  std::vector<ComponentPosition> expected;
  std::vector<std::uint8_t> target_loaded, draft_loaded;
  // Window rows mirror the target tokens. The ring never reallocates, so
  // borrowed views stay fixed; occupant names the row each slot holds.
  std::vector<Token> ring;
  std::vector<Rows> occupant;
  Rows window_rows{0};
  std::vector<std::uint8_t> window_loaded;
  mutable std::vector<std::shared_ptr<FakeTransfer>> transfers;
  mutable std::size_t pending_reads{0};
  std::size_t pending_loads{0};
  bool valid{true};
  bool execution_complete{true};
  bool restoring{false};
  bool private_loaded{false};
  std::array<bool, 32> private_pieces{};
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

namespace {
const ComponentDescriptor* WindowDescriptor(
    std::span<const ComponentDescriptor> components) {
  const auto it =
      std::ranges::find(components, kWindow, &ComponentDescriptor::id);
  return it == components.end() ? nullptr : &*it;
}
// First row of the loaded window prefix, then its contiguous loaded end.
Rows WindowLoadedEnd(const FakeSlot& slot, const ComponentDescriptor& d) {
  auto row = FirstChunkRow(d, slot.window_rows);
  while (row < slot.window_rows && slot.window_loaded[row])
    ++row;
  return row;
}
}  // namespace
FakeAdapter::FakeAdapter(Rows window_rows, Rows ring_rows) : FakeAdapter() {
  if (!window_rows || ring_rows % 4 || ring_rows < window_rows + 3)
    throw std::invalid_argument("fake window ring cannot hold a window");
  components_.push_back({kWindow, 1, ComponentKind::kWindowRows, sizeof(Token),
                         4, 0, window_rows});
  ring_rows_ = ring_rows;
}
Capabilities FakeAdapter::GetCapabilities() const {
  return {.continuation = true,
          .persistent_encoding = true,
          .in_place_restore = in_place_};
}
FakeAdapter::FakeAdapter(Rows rows_per_chunk)
    : components_{
          {kTarget, 1, ComponentKind::kAppendRows, sizeof(Token),
           rows_per_chunk, 0},
          {kDraft, 1, ComponentKind::kAppendRows, sizeof(Token), rows_per_chunk,
           0},
          {kRecurrent, 1, ComponentKind::kPrivateState, 0, 0, 32},
      } {}
std::span<const ComponentDescriptor> FakeAdapter::Components() const {
  return components_;
}
Identity FakeAdapter::CompatibilityIdentity() const {
  if (!ring_rows_)
    return {'f', 'a', 'k', 'e', 2};
  return {'f',
          'a',
          'k',
          'e',
          2,
          'w',
          static_cast<std::uint8_t>(components_.back().window_rows),
          static_cast<std::uint8_t>(ring_rows_)};
}
std::unique_ptr<Slot> FakeAdapter::CreateSlot(MutationGuard& guard) {
  if (std::exchange(fail_allocation_, false))
    throw std::bad_alloc();
  return std::make_unique<FakeSlot>(guard, ring_rows_);
}
void FakeAdapter::SetPrefillPassRows(Rows rows) {
  if (!rows)
    throw std::invalid_argument("empty prefill pass");
  prefill_pass_rows_ = rows;
}
void FakeAdapter::SetExecutionComplete(Slot& slot, bool complete) {
  AsSlot(slot).execution_complete = complete;
}
std::vector<Rows> FakeAdapter::PlanPrefill(const Slot& slot, Rows first,
                                           Rows end) const {
  if (!slot.IsValid() || first > end)
    throw std::invalid_argument("invalid prefill plan");
  std::vector<Rows> result;
  while (first < end) {
    first += std::min(prefill_pass_rows_, end - first);
    result.push_back(first);
  }
  return result;
}
std::vector<ComponentPosition> FakeAdapter::Positions(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  std::vector<ComponentPosition> positions{
      {kTarget, state.restoring ? ContiguousRows(state.target_loaded)
                                : state.target.size()},
      {kDraft, state.restoring ? ContiguousRows(state.draft_loaded)
                               : state.draft.size()},
      {kRecurrent, state.recurrent[0]}};
  if (const auto* window = WindowDescriptor(components_))
    positions.push_back({kWindow, state.restoring
                                      ? WindowLoadedEnd(state, *window)
                                      : state.window_rows});
  return positions;
}
std::vector<ComponentPosition> FakeAdapter::PrefillCheckpointPositions(
    const Slot& slot, Rows boundary) const {
  const auto& state = AsSlot(slot);
  if (!slot.IsValid() || state.Busy() || boundary <= state.target.size())
    throw std::invalid_argument("invalid fake in-pass boundary");
  // The serving fixture advances target and draft by the same prefill tail.
  std::vector<ComponentPosition> positions{
      {kTarget, boundary},
      {kDraft, state.draft.size() + boundary - state.target.size()},
      {kRecurrent, boundary}};
  if (ring_rows_)
    positions.push_back({kWindow, boundary});
  return positions;
}
void FakeAdapter::BeginRestore(Slot& slot,
                               std::span<const ComponentPosition> positions) {
  BeginRestore(slot, positions, {});
}
std::vector<ResidentRange> FakeAdapter::ResidentRows(
    const Slot& slot, std::span<const ComponentPosition> positions) const {
  const auto& state = AsSlot(slot);
  if (!in_place_ || !state.valid || state.restoring || state.Busy())
    return {};
  const Rows target_end = Position(positions, kTarget);
  const Rows draft_end = Position(positions, kDraft);
  std::vector<ResidentRange> kept{
      {kTarget, 0, std::min<Rows>(target_end, state.target.size())},
      {kDraft, 0, std::min<Rows>(draft_end, state.draft.size())}};
  if (const auto* window = WindowDescriptor(components_)) {
    const Rows end = std::min(target_end, state.window_rows);
    Rows first = end;
    while (first > FirstChunkRow(*window, target_end) &&
           state.Resident(first - 1, first))
      --first;
    kept.push_back({kWindow, first, end});
  }
  return kept;
}
void FakeAdapter::BeginRestore(Slot& slot,
                               std::span<const ComponentPosition> positions,
                               std::span<const ResidentRange> kept) {
  auto& state = AsSlot(slot);
  if (state.Busy() || state.failed || state.restoring)
    throw std::logic_error(
        "reset failed restores and settle transfers before BeginRestore");
  if (positions.size() != components_.size())
    throw std::invalid_argument("restore requires every component position");
  const Rows target_end = Position(positions, kTarget);
  const Rows draft_end = Position(positions, kDraft);
  if (Position(positions, kRecurrent) != target_end ||
      (ring_rows_ && Position(positions, kWindow) != target_end))
    throw std::invalid_argument(
        "private boundary differs from target boundary");
  CheckRange(0, target_end,
             static_cast<std::size_t>(target_end) * sizeof(Token));
  CheckRange(0, draft_end, static_cast<std::size_t>(draft_end) * sizeof(Token));
  if (std::exchange(fail_allocation_, false))
    throw std::bad_alloc();
  const auto kept_range = [&](ComponentId id) {
    const auto k = std::ranges::find(kept, id, &ResidentRange::id);
    return k == kept.end() ? std::pair<Rows, Rows>{0, 0}
                           : std::pair{k->first, k->end};
  };
  const auto [target_keep_first, target_keep] = kept_range(kTarget);
  const auto [draft_keep_first, draft_keep] = kept_range(kDraft);
  const auto [window_keep_first, window_keep] = kept_range(kWindow);
  if (!kept.empty() &&
      (!in_place_ || target_keep_first || draft_keep_first ||
       target_keep > std::min<Rows>(target_end, state.target.size()) ||
       draft_keep > std::min<Rows>(draft_end, state.draft.size()) ||
       (window_keep_first < window_keep &&
        (window_keep > target_end ||
         !state.Resident(window_keep_first, window_keep)))))
    throw std::invalid_argument("kept rows are not resident");
  std::vector<Token> target(static_cast<std::size_t>(target_end));
  std::vector<Token> draft(static_cast<std::size_t>(draft_end));
  std::copy_n(state.target.begin(), target_keep, target.begin());
  std::copy_n(state.draft.begin(), draft_keep, draft.begin());
  std::vector<std::uint8_t> target_loaded(target.size()),
      draft_loaded(draft.size());
  std::fill_n(target_loaded.begin(), target_keep, 1);
  std::fill_n(draft_loaded.begin(), draft_keep, 1);
  // All preservation runs while the old rows/private state remain readable.
  state.guard_.BeforeOverwrite(kTarget, target_keep,
                               std::max<Rows>(state.target.size(), target_end));
  state.guard_.BeforeOverwrite(kDraft, draft_keep,
                               std::max<Rows>(state.draft.size(), draft_end));
  if (ring_rows_) {
    state.guard_.BeforeOverwrite(kWindow, 0, window_keep_first);
    state.guard_.BeforeOverwrite(kWindow,
                                 std::max(window_keep, window_keep_first),
                                 std::max(state.window_rows, target_end));
  }
  if (state.Busy())
    throw std::logic_error(
        "guard returned with preservation transfers pending");
  state.target = std::move(target);
  state.draft = std::move(draft);
  state.target_loaded = std::move(target_loaded);
  state.draft_loaded = std::move(draft_loaded);
  state.expected = {
      {kTarget, target_end}, {kDraft, draft_end}, {kRecurrent, target_end}};
  if (ring_rows_) {
    state.expected.push_back({kWindow, target_end});
    for (auto& occupant : state.occupant)
      if (occupant < window_keep_first || occupant >= window_keep)
        occupant = kEmptyRingSlot;
    state.window_rows = target_end;
    state.window_loaded.assign(static_cast<std::size_t>(target_end), 0);
    std::fill(
        state.window_loaded.begin() + window_keep_first,
        state.window_loaded.begin() + std::max(window_keep_first, window_keep),
        1);
  }
  state.recurrent = {0, 0, kSeed, kSeed};
  state.private_loaded = false;
  state.private_pieces.fill(false);
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
  return CapturePrivatePiece(slot, id, 0, dst, stream);
}
Completion FakeAdapter::CapturePrivatePiece(const Slot& slot, ComponentId id,
                                            std::size_t offset,
                                            std::span<std::byte> dst,
                                            Stream& stream) {
  if (id != kRecurrent || offset > 32 || dst.size() > 32 - offset)
    throw std::invalid_argument("invalid private capture piece");
  const auto& state = AsSlot(slot);
  RequireCapture(state);
  return SubmitRead(slot, stream, [&state, offset, dst] {
    std::memcpy(
        dst.data(),
        reinterpret_cast<const std::byte*>(state.recurrent.data()) + offset,
        dst.size());
    return TransferResult::kSucceeded;
  });
}
Completion FakeAdapter::CopyRowsOut(const Slot& slot, ComponentId id,
                                    Rows first, Rows end,
                                    std::span<std::byte> dst, Stream& stream) {
  CheckRange(first, end, dst.size());
  const auto& state = AsSlot(slot);
  RequireCapture(state);
  if (ring_rows_ && id == kWindow) {
    if (end > state.window_rows || !state.Resident(first, end))
      throw std::invalid_argument("cannot capture evicted window rows");
    return SubmitRead(slot, stream, [&state, first, end, dst] {
      for (Rows row = first; row < end; ++row)
        std::memcpy(dst.data() + (row - first) * sizeof(Token),
                    &state.ring[row % state.ring.size()], sizeof(Token));
      return TransferResult::kSucceeded;
    });
  }
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
    if (ring_rows_ && id == kWindow) {
      // Only rows a continuation reads, from the first retained chunk.
      if (first < FirstChunkRow(components_.back(), state.window_rows) ||
          end > state.window_rows)
        throw std::invalid_argument("restore range outside the window");
      return SubmitLoad(slot, stream, [&state, first, end, src] {
        for (Rows row = first; row < end; ++row) {
          const auto at = row % state.ring.size();
          std::memcpy(&state.ring[at],
                      src.data() + (row - first) * sizeof(Token),
                      sizeof(Token));
          state.occupant[at] = row;
          state.window_loaded[row] = 1;
        }
        return TransferResult::kSucceeded;
      });
    }
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
  } catch (...) {
    state.failed = true;
    throw;
  }
  return LoadPrivatePiece(slot, id, 0, src, stream);
}
Completion FakeAdapter::LoadPrivatePiece(Slot& slot, ComponentId id,
                                         std::size_t offset,
                                         std::span<const std::byte> src,
                                         Stream& stream) {
  auto& state = AsSlot(slot);
  RequireLoad(state);
  try {
    if (id != kRecurrent || offset > 32 || src.size() > 32 - offset)
      throw std::invalid_argument("invalid private load piece");
    return SubmitLoad(slot, stream, [&state, offset, src] {
      std::memcpy(reinterpret_cast<std::byte*>(state.recurrent.data()) + offset,
                  src.data(), src.size());
      std::fill_n(state.private_pieces.begin() + offset, src.size(), true);
      state.private_loaded = std::ranges::all_of(
          state.private_pieces, [](bool loaded) { return loaded; });
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
      positions.size() == components_.size() &&
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
  invalidations_.fetch_add(1, std::memory_order_relaxed);
  auto& state = AsSlot(slot);
  if (state.Busy())
    return false;
  state.ReleaseRows();
  state.target.clear();
  state.draft.clear();
  state.target_loaded.clear();
  state.draft_loaded.clear();
  std::ranges::fill(state.occupant, kEmptyRingSlot);
  state.window_rows = 0;
  state.window_loaded.clear();
  state.expected = {};
  state.recurrent = {0, 0, kSeed, kSeed};
  state.transfers.clear();
  state.restoring = false;
  state.failed = false;
  state.private_loaded = false;
  state.private_pieces.fill(false);
  state.valid = true;
  state.guard_.AfterReset();
  return true;
}
std::span<const std::byte> FakeAdapter::BorrowRowsView(const Slot& slot,
                                                       ComponentId component,
                                                       Rows first,
                                                       Rows end) const {
  const auto& state = AsSlot(slot);
  RequireCapture(state);
  if (ring_rows_ && component == kWindow) {
    if (first >= end || end > state.window_rows || !state.Resident(first, end))
      throw std::invalid_argument("invalid borrowed fake window range");
    // A range that wraps the ring is not one span and cannot be borrowed.
    const auto at = first % state.ring.size();
    if (at + (end - first) > state.ring.size())
      return {};
    return std::as_bytes(std::span(state.ring).subspan(at, end - first));
  }
  const auto& rows = RowArray(state, component);
  if (first >= end || end > rows.size())
    throw std::invalid_argument("invalid borrowed fake range");
  // Append, restore and release guard the old allocation before replacing
  // the vector, so views stay fixed even though this fixture moves storage.
  return std::as_bytes(std::span(rows).subspan(first, end - first));
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
  const Rows first = state.window_rows, end = new_target.size();
  if (ring_rows_) {
    // Each new row evicts the one a ring length earlier, then is written.
    const auto evict = [&](Rows row) {
      return row > ring_rows_ ? row - ring_rows_ : 0;
    };
    if (evict(first) < evict(end))
      state.guard_.BeforeOverwrite(kWindow, evict(first), evict(end));
    if (first < end)
      state.guard_.BeforeOverwrite(kWindow, first, end);
  }
  if (state.Busy())
    throw std::logic_error(
        "guard returned with preservation transfers pending");
  if (ring_rows_) {
    for (Rows row = first; row < end; ++row) {
      state.ring[row % ring_rows_] = new_target[row];
      state.occupant[row % ring_rows_] = row;
    }
    state.window_rows = end;
  }
  state.target = std::move(new_target);
  state.draft = std::move(new_draft);
  state.recurrent = {state.target.size(), state.draft.size(), target_hash,
                     draft_hash};
}
std::uint64_t FakeAdapter::RecurrentHash(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  return state.recurrent[2] * kMultiplier + state.recurrent[3];
}
void FakeAdapter::GuardRows(Slot& slot, ComponentId c, Rows first, Rows end) {
  AsSlot(slot).guard_.BeforeOverwrite(c, first, end);
}
std::vector<Token> FakeAdapter::WindowTokens(const Slot& slot) const {
  const auto& state = AsSlot(slot);
  if (!ring_rows_)
    throw std::logic_error("fake adapter has no window component");
  const auto first = WindowFirstRow(components_.back(), state.window_rows);
  if (!state.Resident(first, state.window_rows))
    throw std::logic_error("window rows evicted or not loaded");
  std::vector<Token> tokens;
  for (Rows row = first; row < state.window_rows; ++row)
    tokens.push_back(state.ring[row % ring_rows_]);
  return tokens;
}
void FakeAdapter::FailNextAllocation() noexcept {
  fail_allocation_ = true;
}
void FakeAdapter::FailNextTransfer() noexcept {
  fail_transfer_ = true;
}
}  // namespace gufo::cache::testing
