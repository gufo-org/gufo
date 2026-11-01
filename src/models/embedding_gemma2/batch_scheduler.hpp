#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_BATCH_SCHEDULER_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_BATCH_SCHEDULER_HPP_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "audio_processor.hpp"
#include "config.hpp"
#include "image_processor.hpp"

namespace gufo::models::embedding_gemma2 {

struct SequenceImage {
  std::size_t offset{0};
  ImagePatches patches;
  /// Video frames share the vision tower but use their own placeholders.
  std::uint32_t token_id{kImageTokenId};
};
struct EncoderSequence {
  std::vector<std::uint32_t> tokens;
  std::vector<SequenceImage> images;
  std::vector<SequenceAudio> audio;
  std::function<bool()> is_cancelled;
};

/// Packs the token sequences of concurrent text callers into shared device
/// batches. Callers queue in arrival order. When the device is free, the
/// caller at the front runs one batch on its own thread for itself and the
/// callers behind it, then hands the device on. A lone caller runs at once.
///
/// A batch takes whole callers, oldest first, while their remaining sequences
/// fit in `max_batch_tokens`. The first caller that does not fit ends the
/// batch, so no caller passes another, and a caller that fits in one batch
/// never spans two. Only a caller too large for one batch is split, and only
/// at the start of a batch. It then rejoins the queue behind the callers that
/// arrived during that batch, so they wait for at most one of its batches.
/// A caller with an image or audio runs in a batch of its own, so a text caller
/// never waits for another caller's modality encoder pass. At the front, it
/// takes the batch alone, and it may still be split by tokens. Behind other
/// callers, it ends the batch like a caller that does not fit.
class BatchScheduler {
public:
  /// Embeds one batch into one vector per sequence, in order, or returns
  /// false with `error`. Calls never overlap. A thrown exception fails the
  /// batch like a false return.
  using Runner = std::function<bool(
      const std::vector<const EncoderSequence*>& sequences,
      std::vector<std::vector<float>>* embeddings, std::string* error)>;

  enum class Outcome { kOk, kCancelled, kFailed };

  BatchScheduler(std::size_t max_batch_tokens, Runner runner);

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;
  BatchScheduler(BatchScheduler&&) = delete;
  BatchScheduler& operator=(BatchScheduler&&) = delete;

  /// Blocks until every sequence is embedded and returns the vectors in input
  /// order. A sequence longer than `max_batch_tokens` fails the call.
  /// `is_cancelled` is polled before each batch that would take this call's
  /// sequences. The poll runs on the thread that builds the batch, which may
  /// belong to another call, with the scheduler's lock held, so it must not
  /// block or throw. A cancelled call returns `kCancelled` and sends no
  /// further sequences to the runner. A failed batch returns `kFailed` with
  /// the runner's error to every call with sequences in it.
  [[nodiscard]] Outcome Run(const std::vector<EncoderSequence>& sequences,
                            const std::function<bool()>& is_cancelled,
                            std::vector<std::vector<float>>* embeddings,
                            std::string* error);

  /// Calls waiting for their next batch. Tests use it to wait for
  /// callers to queue.
  [[nodiscard]] std::size_t queued() const;

private:
  struct Caller;
  struct Slice;
  struct Unwind;

  /// Queues `caller` and serves the device while it is at the front, until
  /// it has an outcome.
  Outcome Wait(Caller& caller, std::string* error);
  /// Runs the next batch. Called with `lock` held and the device free.
  void RunNextBatch(std::unique_lock<std::mutex>& lock);

  const std::size_t max_batch_tokens_;
  const Runner runner_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Caller*> queue_;
  bool running_{false};
};

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_BATCH_SCHEDULER_HPP_
