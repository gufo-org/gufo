// Checks how BatchScheduler packs concurrent calls into shared batches: FIFO
// order, whole callers, round-robin turns for callers larger than one batch,
// image callers that run in their own batch, cancellation, and shared
// failures. A fake device stands in for the GPU.

#include "src/models/embedding_gemma2/batch_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using gufo::models::embedding_gemma2::BatchScheduler;
using gufo::models::embedding_gemma2::EncoderSequence;
using Outcome = BatchScheduler::Outcome;
using Sequence = EncoderSequence;
using Batches = std::vector<std::vector<std::uint32_t>>;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL embedding_gemma2_batch_scheduler_test: " << message
              << '\n';
    std::exit(1);
  }
}

constexpr std::size_t kMaxTokens = 100;
// Sequence IDs with a meaning to the fake device.
constexpr std::uint32_t kHold = 1;   // The batch waits for Release().
constexpr std::uint32_t kFail = 2;   // The batch returns false.
constexpr std::uint32_t kThrow = 3;  // The batch throws.

/// `length` copies of `id`, so the device can name each sequence.
Sequence Tokens(std::size_t length, std::uint32_t id) {
  return {.tokens = std::vector<std::uint32_t>(length, id), .images = {}};
}

std::vector<float> EmbeddingOf(const Sequence& sequence) {
  return {static_cast<float>(sequence.tokens.front()),
          static_cast<float>(sequence.tokens.size()),
          static_cast<float>(sequence.images.size())};
}

/// Records each batch as the IDs of its sequences.
class FakeDevice {
public:
  bool operator()(const std::vector<const Sequence*>& sequences,
                  std::vector<std::vector<float>>* embeddings,
                  std::string* error) {
    std::vector<std::uint32_t> ids;
    std::size_t tokens = 0;
    for (const Sequence* sequence : sequences) {
      ids.push_back(sequence->tokens.front());
      tokens += sequence->tokens.size();
    }
    Check(!ids.empty() && tokens <= kMaxTokens,
          "a batch is non-empty and fits the token budget");
    if (!Use(ids, "device failed", error)) {
      return false;
    }
    for (const Sequence* sequence : sequences) {
      embeddings->push_back(EmbeddingOf(*sequence));
    }
    return true;
  }

  void WaitHeld() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return held_; });
  }

  void Release() {
    const std::lock_guard lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

  Batches batches() {
    const std::lock_guard lock(mutex_);
    return batches_;
  }

  /// The thread that ran each batch.
  std::vector<std::thread::id> threads() {
    const std::lock_guard lock(mutex_);
    return threads_;
  }

private:
  /// Holds the device for one batch. Fails it as `kFail` and `kThrow` ask.
  bool Use(const std::vector<std::uint32_t>& ids, const char* failure,
           std::string* error) {
    Check(!running_.exchange(true), "batches never overlap");
    {
      std::unique_lock lock(mutex_);
      batches_.push_back(ids);
      threads_.push_back(std::this_thread::get_id());
      if (std::ranges::find(ids, kHold) != ids.end()) {
        held_ = true;
        changed_.notify_all();
        changed_.wait(lock, [&] { return released_; });
      }
    }
    // Long enough for concurrent callers to queue behind the batch.
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    running_ = false;
    if (std::ranges::find(ids, kFail) != ids.end()) {
      *error = failure;
      return false;
    }
    if (std::ranges::find(ids, kThrow) != ids.end()) {
      throw std::runtime_error("device threw");
    }
    return true;
  }

  std::atomic<bool> running_{false};
  std::mutex mutex_;
  std::condition_variable changed_;
  Batches batches_;
  std::vector<std::thread::id> threads_;
  bool held_{false};
  bool released_{false};
};

/// One `Run` call on its own thread.
struct Call {
  Call(BatchScheduler& scheduler, std::vector<Sequence> input,
       std::function<bool()> cancelled)
      : sequences(std::move(input)),
        is_cancelled(std::move(cancelled)),
        thread([this, &scheduler] {
          try {
            outcome =
                scheduler.Run(sequences, is_cancelled, &embeddings, &error);
          } catch (const std::exception&) {
            threw = true;
          }
          finished = true;
        }) {}

  std::vector<Sequence> sequences;
  std::function<bool()> is_cancelled;
  std::vector<std::vector<float>> embeddings;
  std::string error;
  Outcome outcome{Outcome::kFailed};
  bool threw{false};
  std::atomic<bool> finished{false};
  std::jthread thread;  // Last, so it starts after the other members exist.
};

void ExpectEmbeddings(const std::vector<Sequence>& sequences, Outcome outcome,
                      const std::vector<std::vector<float>>& embeddings,
                      std::string_view what) {
  Check(outcome == Outcome::kOk, what);
  Check(embeddings.size() == sequences.size(), what);
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    Check(embeddings[i] == EmbeddingOf(sequences[i]), what);
  }
}

void ExpectEmbeddings(const Call& call, std::string_view what) {
  ExpectEmbeddings(call.sequences, call.outcome, call.embeddings, what);
}

/// A fake device, a scheduler on it, and the calls started against it.
struct Rig {
  FakeDevice device;
  BatchScheduler scheduler{kMaxTokens, std::ref(device)};
  std::deque<Call> calls;
  std::vector<std::vector<float>> embeddings;
  std::string error;

  /// Runs a call on this thread.
  Outcome Run(const std::vector<Sequence>& sequences,
              const std::function<bool()>& is_cancelled = {}) {
    return scheduler.Run(sequences, is_cancelled, &embeddings, &error);
  }

  /// Starts a call whose first batch holds the device, and returns once it
  /// does.
  Call& Hold(std::vector<Sequence> sequences,
             std::function<bool()> cancelled = {}) {
    Call& call = calls.emplace_back(scheduler, std::move(sequences),
                                    std::move(cancelled));
    device.WaitHeld();
    return call;
  }

  /// Starts a call behind the held one, and returns once it is queued.
  Call& Queue(std::vector<Sequence> sequences,
              std::function<bool()> cancelled = {}) {
    return WaitQueued(calls.emplace_back(scheduler, std::move(sequences),
                                         std::move(cancelled)));
  }

  /// Releases the held batch and waits for every call to finish.
  void Drain() {
    device.Release();
    for (Call& call : calls) {
      call.thread.join();
    }
  }

  void ExpectAllEmbedded() const {
    for (const Call& call : calls) {
      ExpectEmbeddings(call, "each caller gets its own vectors in input order");
    }
  }

private:
  Call& WaitQueued(Call& call) {
    while (scheduler.queued() != calls.size() - 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return call;
  }
};

void TestLoneCaller() {
  Rig rig;
  const std::vector<Sequence> fits = {Tokens(30, 10), Tokens(40, 11)};
  ExpectEmbeddings(fits, rig.Run(fits), rig.embeddings,
                   "a lone caller runs at once");
  const std::vector<Sequence> spans = {Tokens(50, 12), Tokens(40, 13),
                                       Tokens(30, 14)};
  ExpectEmbeddings(spans, rig.Run(spans), rig.embeddings,
                   "a lone caller larger than one batch");
  Check(rig.Run({}) == Outcome::kOk && rig.embeddings.empty(),
        "no sequences, no batch");
  Check(rig.device.batches() == Batches{{10, 11}, {12, 13}, {14}},
        "a lone caller packs its own sequences greedily");
}

void TestSharedBatch() {
  Rig rig;
  rig.Hold({Tokens(10, kHold)});
  std::vector<std::uint32_t> expected;
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t first = 100 + 2 * i;
    rig.Queue({Tokens(5, first), Tokens(5, first + 1)});
    expected.insert(expected.end(), {first, first + 1});
  }
  rig.Drain();
  rig.ExpectAllEmbedded();
  Check(rig.device.batches() == Batches{{kHold}, expected},
        "callers queued behind a busy device share the next batch in arrival "
        "order");
}

void TestCallerThatFitsIsNotSplit() {
  Rig rig;
  rig.Hold({Tokens(10, kHold)});
  rig.Queue({Tokens(30, 10), Tokens(30, 11)});
  // Fits in an empty batch but not in the 40 tokens the first caller leaves.
  rig.Queue({Tokens(25, 12), Tokens(25, 13)});
  rig.Queue({Tokens(5, 14)});
  rig.Drain();
  rig.ExpectAllEmbedded();
  Check(rig.device.batches() == Batches{{kHold}, {10, 11}, {12, 13, 14}},
        "a caller that fits in one batch is never split, and later callers do "
        "not pass it");
}

/// A short caller arrives during the first batch of a long one.
void CheckShortArrivesDuringLongCaller(std::vector<Sequence> long_call,
                                       const Batches& expected,
                                       std::string_view what) {
  Rig rig;
  rig.Hold(std::move(long_call));
  rig.Queue({Tokens(5, 20)});
  rig.Drain();
  rig.ExpectAllEmbedded();
  Check(rig.device.batches() == expected, what);
}

void TestLargeCallerTakesTurns() {
  CheckShortArrivesDuringLongCaller(
      {Tokens(80, kHold), Tokens(80, 10), Tokens(80, 11)},
      {{kHold}, {20}, {10}, {11}},
      "a long caller goes behind the callers that arrived during its batch");
  CheckShortArrivesDuringLongCaller(
      {Tokens(60, kHold), Tokens(60, 10), Tokens(30, 11)},
      {{kHold}, {20, 10, 11}},
      "the rest of a long caller shares the next batch when it fits");
}

void TestCancellation() {
  {
    Rig rig;
    Check(
        rig.Run({Tokens(5, 10)}, [] { return true; }) == Outcome::kCancelled &&
            rig.embeddings.empty(),
        "a cancelled caller returns kCancelled");
    Check(rig.device.batches().empty(), "a cancelled caller never runs");
  }
  // Cancelled while queued. The cancelled caller is at the front, and the
  // caller behind it still runs.
  {
    Rig rig;
    std::atomic<bool> gone{false};
    rig.Hold({Tokens(10, kHold)});
    Call& cancelled =
        rig.Queue({Tokens(5, 10)}, [&gone] { return gone.load(); });
    Call& kept = rig.Queue({Tokens(5, 11)});
    const std::thread::id kept_thread = kept.thread.get_id();
    gone = true;
    rig.Drain();
    Check(cancelled.outcome == Outcome::kCancelled &&
              cancelled.embeddings.empty(),
          "a caller cancelled while queued returns kCancelled");
    ExpectEmbeddings(kept, "the caller behind a cancelled one runs");
    Check(rig.device.batches() == Batches{{kHold}, {11}},
          "a caller cancelled while queued never reaches the device");
    const auto threads = rig.device.threads();
    Check(threads.size() == 2 && threads[1] == kept_thread,
          "the caller behind a cancelled head runs its own batch");
  }
  // Cancelled in the middle of a forming batch. The callers on both sides
  // share the batch.
  {
    Rig rig;
    rig.Hold({Tokens(10, kHold)});
    rig.Queue({Tokens(5, 10)});
    Call& middle = rig.Queue({Tokens(5, 11)}, [] { return true; });
    rig.Queue({Tokens(5, 12)});
    rig.Drain();
    Check(middle.outcome == Outcome::kCancelled && middle.embeddings.empty(),
          "a caller cancelled in the middle of a batch returns kCancelled");
    ExpectEmbeddings(rig.calls[1], "the caller before a cancelled one runs");
    ExpectEmbeddings(rig.calls[3], "the caller after a cancelled one runs");
    Check(rig.device.batches() == Batches{{kHold}, {10, 12}},
          "the callers around a cancelled one share a batch without it");
  }
  // Cancelled during the first of two batches. The second never runs.
  {
    Rig rig;
    std::atomic<bool> gone{false};
    Call& cancelled = rig.Hold({Tokens(60, kHold), Tokens(60, 10)},
                               [&gone] { return gone.load(); });
    Call& next = rig.Queue({Tokens(5, 11)});
    gone = true;
    rig.Drain();
    Check(cancelled.outcome == Outcome::kCancelled &&
              cancelled.embeddings.empty(),
          "a caller cancelled between its batches returns kCancelled");
    ExpectEmbeddings(next, "the caller behind a cancelled one runs");
    Check(rig.device.batches() == Batches{{kHold}, {11}},
          "a caller cancelled mid-batch runs no further batches");
  }
  // Cancelled during its last batch. The batch finishes, and the caller's
  // owner decides whether to drop the result.
  {
    Rig rig;
    std::atomic<bool> gone{false};
    Call& call = rig.Hold({Tokens(10, kHold)}, [&gone] { return gone.load(); });
    gone = true;
    rig.Drain();
    ExpectEmbeddings(call, "a running batch finishes");
  }
}

void TestFailure() {
  {
    Rig rig;
    Call& hold = rig.Hold({Tokens(10, kHold)});
    rig.Queue({Tokens(5, 10)});
    rig.Queue({Tokens(5, kFail)});
    rig.Queue({Tokens(5, 11)});
    rig.Drain();
    ExpectEmbeddings(hold, "the batch before the failure");
    for (std::size_t i = 1; i < rig.calls.size(); ++i) {
      Check(rig.calls[i].outcome == Outcome::kFailed &&
                rig.calls[i].error == "device failed" &&
                rig.calls[i].embeddings.empty(),
            "every caller in a failed batch gets its error");
    }
    Check(rig.device.batches() == Batches{{kHold}, {10, kFail, 11}},
          "the failed callers shared one batch");
  }
  {
    Rig rig;
    Check(rig.Run({Tokens(60, kFail), Tokens(60, 10)}) == Outcome::kFailed &&
              rig.embeddings.empty() && rig.error == "device failed",
          "a caller whose first batch fails returns kFailed");
    Check(rig.Run({Tokens(5, kThrow)}) == Outcome::kFailed &&
              rig.error == "device threw",
          "a thrown exception fails the batch");
    const std::vector<Sequence> after = {Tokens(5, 11)};
    ExpectEmbeddings(after, rig.Run(after), rig.embeddings,
                     "a caller after a failure and an exception runs");
    Check(rig.device.batches() == Batches{{kFail}, {kThrow}, {11}},
          "a failed caller runs no further batches");
  }
  {
    Rig rig;
    Check(rig.Run({Tokens(60, 10), Tokens(60, kFail)}) == Outcome::kFailed &&
              rig.embeddings.empty() && rig.error == "device failed",
          "a caller whose second batch fails drops its first batch's vectors");
    Check(rig.device.batches() == Batches{{10}, {kFail}},
          "the caller ran both of its batches");
  }
}

void TestSequenceLongerThanBatch() {
  Rig rig;
  Check(rig.Run({Tokens(5, 10), Tokens(kMaxTokens + 1, 11)}) ==
                Outcome::kFailed &&
            rig.embeddings.empty() && !rig.error.empty(),
        "a sequence longer than a batch fails the call");
  Check(rig.device.batches().empty(),
        "a call with a sequence longer than a batch sends nothing to the "
        "device");
  const std::vector<Sequence> after = {Tokens(kMaxTokens, 12)};
  ExpectEmbeddings(after, rig.Run(after), rig.embeddings,
                   "a sequence of exactly one batch runs");
}

/// A hook that throws breaks its contract, but must not leave the scheduler
/// busy. The call that builds the batch sees the exception, the call whose hook
/// threw fails, and the call behind them still runs.
void TestExceptionWhileBuildingBatch() {
  Rig rig;
  rig.Hold({Tokens(10, kHold)});
  Call& builder = rig.Queue({Tokens(5, 10)});
  Call& thrower = rig.Queue({Tokens(5, 11)}, []() -> bool {
    throw std::runtime_error("hook threw");
  });
  Call& behind = rig.Queue({Tokens(5, 12)});
  rig.device.Release();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!behind.finished && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Check(behind.finished, "a call behind a failed batch build still runs");
  rig.Drain();
  Check(builder.threw, "the exception reaches the call that built the batch");
  Check(thrower.outcome == Outcome::kFailed && !thrower.threw &&
            thrower.error == "embedding batch failed",
        "the call whose hook threw fails");
  ExpectEmbeddings(behind, "the call behind runs");
  Check(rig.device.batches() == Batches{{kHold}, {12}},
        "the failed calls never reach the device");
  Check(rig.scheduler.queued() == 0, "no failed call stays queued");
}

/// A caller whose sequences hold images runs in a batch of its own, in
/// arrival order, and the runner receives the images. Text callers around it
/// still pack with each other.
void TestImageCallerGetsOwnBatch() {
  Sequence image = Tokens(10, 30);
  image.images.resize(2);
  {
    Rig rig;
    rig.Hold({Tokens(10, kHold)});
    rig.Queue({Tokens(5, 31)});
    rig.Queue({image});
    rig.Drain();
    rig.ExpectAllEmbedded();
    Check(rig.device.batches() == Batches{{kHold}, {31}, {30}},
          "an image caller behind a text caller does not join its batch");
  }
  {
    Rig rig;
    rig.Hold({Tokens(10, kHold)});
    rig.Queue({image});
    rig.Queue({Tokens(5, 31)});
    rig.Drain();
    rig.ExpectAllEmbedded();
    Check(rig.device.batches() == Batches{{kHold}, {30}, {31}},
          "a text caller behind an image caller does not join its batch");
  }
  {
    Rig rig;
    rig.Hold({Tokens(10, kHold)});
    rig.Queue({Tokens(5, 31)});
    rig.Queue({Tokens(5, 32)});
    rig.Queue({image});
    rig.Queue({Tokens(5, 33)});
    rig.Queue({Tokens(5, 34)});
    rig.Drain();
    rig.ExpectAllEmbedded();
    Check(rig.device.batches() == Batches{{kHold}, {31, 32}, {30}, {33, 34}},
          "text callers around an image caller keep arrival order");
  }
}

/// Many threads with random calls, some cancelled from the start. Every call
/// gets its own vectors in order, the fake device checks that no batch
/// overlaps another, and no cancelled sequence reaches the device.
void TestConcurrentCallers() {
  Rig rig;
  constexpr std::uint32_t kThreads = 16;
  constexpr std::uint32_t kCalls = 40;
  std::mutex mutex;
  std::set<std::uint32_t> cancelled_ids;
  {
    std::vector<std::jthread> threads;
    for (std::uint32_t t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        std::mt19937 random(t);
        for (std::uint32_t c = 0; c < kCalls; ++c) {
          std::vector<Sequence> sequences(1 + random() % 6);
          for (std::uint32_t i = 0; i < sequences.size(); ++i) {
            sequences[i] =
                Tokens(1 + random() % 60, 1000 + (t * kCalls + c) * 6 + i);
          }
          const bool cancel = random() % 8 == 0;
          std::vector<std::vector<float>> embeddings;
          std::string error;
          const Outcome outcome = rig.scheduler.Run(
              sequences, [cancel] { return cancel; }, &embeddings, &error);
          if (cancel) {
            Check(outcome == Outcome::kCancelled && embeddings.empty(),
                  "a cancelled call returns kCancelled");
            const std::lock_guard lock(mutex);
            for (const Sequence& sequence : sequences) {
              cancelled_ids.insert(sequence.tokens.front());
            }
          } else {
            ExpectEmbeddings(sequences, outcome, embeddings,
                             "concurrent calls get their own vectors");
          }
        }
      });
    }
  }
  const Batches batches = rig.device.batches();
  Check(std::ranges::none_of(batches,
                             [&](const std::vector<std::uint32_t>& batch) {
                               return std::ranges::any_of(
                                   batch, [&](std::uint32_t id) {
                                     return cancelled_ids.contains(id);
                                   });
                             }),
        "no cancelled sequence reaches the device");
  Check(rig.scheduler.queued() == 0, "the queue drains");
  Check(std::ranges::any_of(batches,
                            [](const std::vector<std::uint32_t>& batch) {
                              return (batch.front() - 1000) / 6 !=
                                     (batch.back() - 1000) / 6;
                            }),
        "some batches hold sequences of more than one call");
}

}  // namespace

int main() {
  TestLoneCaller();
  TestSharedBatch();
  TestCallerThatFitsIsNotSplit();
  TestLargeCallerTakesTurns();
  TestCancellation();
  TestFailure();
  TestSequenceLongerThanBatch();
  TestExceptionWhileBuildingBatch();
  TestImageCallerGetsOwnBatch();
  TestConcurrentCallers();
  std::cout << "PASS embedding_gemma2_batch_scheduler_test\n";
  return 0;
}
