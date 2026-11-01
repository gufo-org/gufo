#include "batch_scheduler.hpp"

#include <cassert>
#include <exception>
#include <optional>
#include <utility>

namespace gufo::models::embedding_gemma2 {
namespace {

/// Runs `work`, and turns an exception into a false return with its message.
template<typename Work>
bool Guarded(const Work& work, std::string* error) {
  try {
    return work();
  } catch (const std::exception& failure) {
    *error = failure.what();
  } catch (...) {
    *error = "embedding device work failed with an unknown exception";
  }
  return false;
}

}  // namespace

/// One `Run` call. It lives on the caller's stack and is touched only under
/// the scheduler mutex until `outcome` is set.
struct BatchScheduler::Caller {
  const std::vector<EncoderSequence>& sequences;
  const std::function<bool()>& is_cancelled;
  std::vector<std::vector<float>>& embeddings;
  /// Some sequence holds an image or audio, so the caller gets a batch of its
  /// own.
  bool has_media{false};
  /// First sequence not yet taken into a batch.
  std::size_t next{0};
  std::size_t remaining_tokens{0};
  std::optional<Outcome> outcome;
  std::string error;
};

/// The sequences `[first, last)` of one caller within a batch.
struct BatchScheduler::Slice {
  Caller* caller;
  std::size_t first;
  std::size_t last;
};

/// Held by `RunNextBatch`. If it exits by exception, it fails every caller it
/// popped that has no outcome, frees the device and wakes the waiters, so an
/// allocation failure cannot wedge the scheduler.
struct BatchScheduler::Unwind {
  ~Unwind() {
    if (std::uncaught_exceptions() == exceptions) {
      return;
    }
    if (!lock.owns_lock()) {
      lock.lock();
    }
    const auto fail = [&](Caller* caller) {
      if (caller == nullptr || caller->outcome.has_value()) {
        return;
      }
      caller->outcome = Outcome::kFailed;
      std::erase(scheduler.queue_, caller);
      try {
        caller->error = "embedding batch failed";
      } catch (...) {  // The caller still fails, with no message.
      }
    };
    if (slices != nullptr) {
      for (const Slice& slice : *slices) {
        fail(slice.caller);
      }
    }
    fail(popped);
    scheduler.running_ = false;
    scheduler.changed_.notify_all();
  }

  BatchScheduler& scheduler;
  std::unique_lock<std::mutex>& lock;
  const std::vector<Slice>* slices{nullptr};
  /// The caller taken off the queue that is not yet in `slices`.
  Caller* popped{nullptr};
  const int exceptions{std::uncaught_exceptions()};
};

BatchScheduler::BatchScheduler(std::size_t max_batch_tokens, Runner runner)
    : max_batch_tokens_(max_batch_tokens), runner_(std::move(runner)) {}

BatchScheduler::Outcome BatchScheduler::Run(
    const std::vector<EncoderSequence>& sequences,
    const std::function<bool()>& is_cancelled,
    std::vector<std::vector<float>>* embeddings, std::string* error) {
  embeddings->clear();
  if (sequences.empty()) {
    return Outcome::kOk;
  }
  std::size_t tokens = 0;
  bool has_media = false;
  for (const EncoderSequence& sequence : sequences) {
    has_media =
        has_media || !sequence.images.empty() || !sequence.audio.empty();
    if (sequence.tokens.size() > max_batch_tokens_) {
      *error = "a sequence is longer than the batch token limit";
      return Outcome::kFailed;
    }
    tokens += sequence.tokens.size();
  }
  embeddings->reserve(sequences.size());
  Caller caller{.sequences = sequences,
                .is_cancelled = is_cancelled,
                .embeddings = *embeddings,
                .has_media = has_media,
                .remaining_tokens = tokens};
  const Outcome outcome = Wait(caller, error);
  if (outcome != Outcome::kOk) {
    embeddings->clear();
  }
  return outcome;
}

BatchScheduler::Outcome BatchScheduler::Wait(Caller& caller,
                                             std::string* error) {
  std::unique_lock lock(mutex_);
  queue_.push_back(&caller);
  while (true) {
    changed_.wait(lock, [&] {
      return caller.outcome.has_value() ||
             (!running_ && !queue_.empty() && queue_.front() == &caller);
    });
    if (caller.outcome.has_value()) {
      break;
    }
    RunNextBatch(lock);
  }
  if (*caller.outcome == Outcome::kFailed) {
    *error = std::move(caller.error);
  }
  return *caller.outcome;
}

std::size_t BatchScheduler::queued() const {
  const std::lock_guard lock(mutex_);
  return queue_.size();
}

void BatchScheduler::RunNextBatch(std::unique_lock<std::mutex>& lock) {
  running_ = true;
  // The head is the calling thread's own caller.
  const Caller* const own = queue_.front();
  std::vector<Slice> slices;
  Unwind unwind{.scheduler = *this, .lock = lock, .slices = &slices};
  std::vector<const EncoderSequence*> sequences;
  std::size_t tokens = 0;
  while (!queue_.empty()) {
    Caller& caller = *queue_.front();
    if (caller.has_media && !sequences.empty()) {
      break;
    }
    std::size_t last = caller.next;
    if (caller.remaining_tokens <= max_batch_tokens_ - tokens) {
      last = caller.sequences.size();
    } else if (tokens == 0) {
      for (std::size_t taken = 0;
           taken + caller.sequences[last].tokens.size() <= max_batch_tokens_;
           ++last) {
        taken += caller.sequences[last].tokens.size();
      }
    } else {
      break;
    }
    queue_.pop_front();
    unwind.popped = &caller;
    if (caller.is_cancelled && caller.is_cancelled()) {
      caller.outcome = Outcome::kCancelled;
      if (&caller == own) {
        // Leave the batch to the next caller's thread, not this one.
        running_ = false;
        changed_.notify_all();
        return;
      }
      continue;
    }
    slices.push_back({&caller, caller.next, last});
    for (std::size_t index = caller.next; index < last; ++index) {
      sequences.push_back(&caller.sequences[index]);
      tokens += caller.sequences[index].tokens.size();
      caller.remaining_tokens -= caller.sequences[index].tokens.size();
    }
    caller.next = last;
    if (last < caller.sequences.size() || caller.has_media) {
      break;
    }
  }
  // Wakes callers cancelled above.
  changed_.notify_all();

  lock.unlock();
  std::vector<std::vector<float>> embeddings;
  std::string error;
  const bool ok =
      Guarded([&] { return runner_(sequences, &embeddings, &error); }, &error);
  lock.lock();

  assert(!ok || embeddings.size() == sequences.size());
  std::size_t offset = 0;
  for (const Slice& slice : slices) {
    Caller& caller = *slice.caller;
    if (!ok) {
      caller.error = error;
      caller.outcome = Outcome::kFailed;
      continue;
    }
    for (std::size_t index = slice.first; index < slice.last; ++index) {
      caller.embeddings.push_back(std::move(embeddings[offset++]));
    }
    if (caller.next == caller.sequences.size()) {
      caller.outcome = Outcome::kOk;
    } else {
      queue_.push_back(&caller);
    }
  }
  running_ = false;
  changed_.notify_all();
}

}  // namespace gufo::models::embedding_gemma2
