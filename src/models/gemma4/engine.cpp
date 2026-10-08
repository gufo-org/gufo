#include "src/models/gemma4/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

namespace gufo::models::gemma4 {

static_assert(kDefaultDraftTokens == rocm::kMaxDraftTokens,
              "the default draft chain is the longest one");
namespace {

constexpr std::array<char, 8> kSnapshotMagic = {'G', '4', 'S', 'N',
                                                'A', 'P', '0', '1'};

struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t position;
  std::uint32_t hidden;
  std::uint32_t vocab;
  std::uint32_t layers;
  std::uint32_t window;
};
static_assert(sizeof(SnapshotHeader) == 32);
/// Snapshot image record: u32 offset, u32 rows, 32-byte identity.
constexpr std::size_t kImageRecordBytes = 8 + 32;

/// First position whose KV a later token may attend in `layer`.
std::uint32_t FirstLiveRow(const Config& c, std::uint32_t layer,
                           std::uint32_t position) {
  if (!c.IsSliding(layer) || position < c.sliding_window) {
    return 0;
  }
  return position - (c.sliding_window - 1);
}

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
  return false;
}

}  // namespace

/// Snapshot memory of one model. hipFree waits for every forward queued on
/// the device (200 ms behind a prefill), so a worker thread frees device
/// blocks rather than the thread dropping the last reference while other
/// sessions run. SessionSnapshot::Stream gathers into host buffers
/// registered on first use and kept, on a private high-priority stream:
/// registering or freeing pinned memory also waits for the device, and
/// registered ordinary memory stays cached for the CPU, which reads every
/// byte (pinned allocations are uncached here). A stream created at load
/// would shift the hardware queues the executor's streams share (server
/// decode 3.4% slower).
struct SnapshotMemory : std::enable_shared_from_this<SnapshotMemory> {
  static constexpr std::uint64_t kStagingBytes = std::uint64_t{64} << 20;
  static constexpr std::uint32_t kStagingRuns = 8192;

  ~SnapshotMemory() {
    {
      std::lock_guard lock(free_mutex);
      stopping = true;
    }
    free_ready.notify_one();
    if (freer.joinable()) {
      freer.join();
    }
    for (void* p : frees) {
      (void)hipFree(p);
    }
    if (registered) {
      (void)hipHostUnregister(runs.data());
      (void)hipHostUnregister(out.data());
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }
  /// Device bytes, freed by the worker; null on failure.
  std::shared_ptr<std::uint8_t> Allocate(std::uint64_t bytes) {
    void* at = nullptr;
    if (hipMalloc(&at, bytes) != hipSuccess) {
      return nullptr;
    }
    return std::shared_ptr<std::uint8_t>(
        static_cast<std::uint8_t*>(at),
        [memory = shared_from_this()](std::uint8_t* p) { memory->Free(p); });
  }
  void Free(void* p) {
    {
      std::lock_guard lock(free_mutex);
      frees.push_back(p);
      if (!freer.joinable()) {
        freer = std::thread([this] { FreeLoop(); });
      }
    }
    free_ready.notify_one();
  }
  void FreeLoop() {
    std::unique_lock lock(free_mutex);
    while (true) {
      free_ready.wait(lock, [&] { return stopping || !frees.empty(); });
      if (frees.empty()) {
        return;
      }
      std::vector<void*> batch;
      batch.swap(frees);
      lock.unlock();
      for (void* p : batch) {
        (void)hipFree(p);
      }
      lock.lock();
    }
  }
  /// Stream's staging, under `mutex`.
  bool StagingReady() {
    if (registered) {
      return true;
    }
    // A high-priority stream runs on its own hardware queue; on a shared one
    // the gather waited behind a whole queued prefill.
    int least = 0;
    int greatest = 0;
    if (stream == nullptr &&
        (hipDeviceGetStreamPriorityRange(&least, &greatest) != hipSuccess ||
         hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, greatest) !=
             hipSuccess)) {
      return false;
    }
    out.resize(kStagingBytes);
    runs.resize(kStagingRuns);
    if (hipHostRegister(out.data(), kStagingBytes, hipHostRegisterDefault) !=
        hipSuccess) {
      return false;
    }
    if (hipHostRegister(runs.data(), kStagingRuns * sizeof(rocm::CopyRun),
                        hipHostRegisterDefault) != hipSuccess) {
      (void)hipHostUnregister(out.data());
      return false;
    }
    registered = true;
    return true;
  }

  std::mutex free_mutex;
  std::condition_variable free_ready;
  std::vector<void*> frees;
  bool stopping{false};
  std::thread freer;
  std::mutex mutex;  ///< Stream's staging
  hipStream_t stream{nullptr};
  bool registered{false};
  std::vector<rocm::CopyRun> runs;
  std::vector<std::uint8_t> out;
};

Model::Model() = default;

void Model::SetTapSink(LayerTapSink sink) {
  std::lock_guard lock(mutex_);
  if (!sink) {
    executor_->SetTapSink({});
    return;
  }
  executor_->SetTapSink(
      [sink = std::move(sink)](std::uint32_t layer, const float* rows,
                               std::uint32_t count, hipStream_t stream) {
        if (hipStreamSynchronize(stream) != hipSuccess) {
          throw std::runtime_error("gemma4 layer tap synchronization failed");
        }
        sink(layer, rows, count);
      });
}
Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  m->options_ = options;
  if (options.max_context == 0 || options.prefill_chunk == 0 ||
      options.max_logit_rows == 0) {
    Fail(error_msg, "gemma4 model options must be positive");
    return nullptr;
  }
  m->snapshot_memory_ = std::make_shared<SnapshotMemory>();
  std::string error;
  std::unique_ptr<core::GgufReader> reader =
      core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    Fail(error_msg, error);
    return nullptr;
  }
  m->reader_ = std::move(reader);
  if (!ChatTemplate::ValidateGgufTemplate(*m->reader_, &error)) {
    Fail(error_msg, error);
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, &error);
  if (!weights) {
    Fail(error_msg, error);
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  if (options.max_context > m->weights_->config.context_length) {
    Fail(error_msg, "requested context exceeds the model's native " +
                        std::to_string(m->weights_->config.context_length));
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::CreateFromGguf(*m->reader_, &error);
  if (!m->tokenizer_) {
    Fail(error_msg, error);
    return nullptr;
  }
  if (!options.mtp_model_path.empty()) {
    if (options.draft_tokens == 0 ||
        options.draft_tokens > rocm::kMaxDraftTokens) {
      Fail(error_msg, "gemma4 draft tokens must be 1.." +
                          std::to_string(rocm::kMaxDraftTokens));
      return nullptr;
    }
    if (options.min_draft_tokens == 0 ||
        options.min_draft_tokens > options.draft_tokens) {
      Fail(error_msg, "gemma4 minimum draft tokens must be 1..draft tokens");
      return nullptr;
    }
    std::unique_ptr<core::GgufReader> draft_reader =
        core::GgufReader::OpenFile(options.mtp_model_path, &error);
    if (!draft_reader) {
      Fail(error_msg, error);
      return nullptr;
    }
    m->draft_reader_ = std::move(draft_reader);
    auto draft = DraftWeights::Bind(*m->draft_reader_, *m->weights_, &error);
    if (!draft) {
      Fail(error_msg, "MTP drafter: " + error);
      return nullptr;
    }
    m->draft_weights_ = std::make_unique<DraftWeights>(std::move(*draft));
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->draft_weights_.get(),
                                         m->draft_reader_.get(), &error);
  if (!m->device_) {
    Fail(error_msg, error);
    return nullptr;
  }
  try {
    // A final remainder of up to kSplitRows rows joins the last prefill
    // chunk (Session::Extend).
    m->executor_ = std::make_unique<rocm::Executor>(
        *m->device_, options.prefill_chunk + rocm::kSplitRows,
        options.max_logit_rows, options.max_context);
  } catch (const std::exception& e) {
    Fail(error_msg, e.what());
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0) {
    max_context = options_.max_context;
  }
  auto cache = executor_->CreateCache(max_context, error_msg);
  if (!cache) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(cache)));
}

std::vector<TokenId> Model::Tokenize(std::string_view text) const {
  return tokenizer_->Encode(text, false, true);
}

std::string Model::Decode(std::span<const TokenId> tokens) const {
  return tokenizer_->Decode(tokens, false);
}

std::string Model::TokenText(TokenId token) const {
  return tokenizer_->TokenText(token, false);
}

bool Model::IsStopToken(TokenId token) const noexcept {
  return tokenizer_->IsEndOfGeneration(token);
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->vocab_size;
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() +
         rocm::Executor::ScratchBytes(
             config(), HasMtp() ? &draft_weights_->config : nullptr,
             VocabSize(), device_->max_cols(), device_->max_half_cols(),
             options_.prefill_chunk + rocm::kSplitRows, options_.max_logit_rows,
             options_.max_context);
}

std::size_t Model::SessionBytes(std::uint32_t context) const noexcept {
  return rocm::Executor::CacheBytes(config(), context, executor_->ring(),
                                    executor_->key_widths());
}

std::size_t Model::SessionRingSlots() const noexcept {
  return executor_->ring();
}

std::string Model::ModelName() const {
  const auto name = reader_->GetMetadataString("general.name");
  return name ? std::string(*name) : std::string("gemma4");
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::KvCache> cache)
    : model_(std::move(model)), cache_(std::move(cache)) {}

Session::~Session() = default;

std::uint32_t Session::ContextSize() const noexcept {
  return cache_->max_context;
}

std::size_t Session::AllocatedBytes() const noexcept {
  return cache_->bytes;
}

void Session::Reset() {
  DropLookahead();
  pending_.reset();
  tokens_.clear();
  images_.clear();
  lookup_.Clear();
  logits_.clear();
  valid_ = false;
  // A restored snapshot starts a request without Sync.
  if (model_->options_.draft_calibration == DraftCalibrationScope::kRequest) {
    for (auto& calibration : calibration_) {
      calibration.Reset();
    }
  }
}

void Session::DropLookahead() noexcept {
  ahead_.clear();
  ahead_logits_.clear();
}

bool Session::Extend(std::size_t begin, std::string* error_msg,
                     const ImageEmbeddings& embed,
                     std::span<const TokenId> lookahead) {
  auto& executor = *model_->executor_;
  const std::size_t chunk = model_->options_.prefill_chunk;
  // The lookahead rows ride in the last forward; tokens_ holds them only
  // while it runs.
  const std::size_t frontier = tokens_.size();
  tokens_.insert(tokens_.end(), lookahead.begin(), lookahead.end());
  DropLookahead();
  valid_ = false;
  try {
    std::lock_guard lock(model_->mutex_);
    std::vector<rocm::ImageRows> rows;
    std::vector<float> logits;
    std::size_t count = 0;
    for (std::size_t start = begin; start < tokens_.size(); start += count) {
      std::size_t end = std::min(start + chunk, tokens_.size());
      // A remainder of at most kSplitRows rows joins this chunk: alone it
      // would read every weight again for a few rows.
      if (tokens_.size() - end <= rocm::kSplitRows &&
          tokens_.size() - start <= executor.max_rows()) {
        end = tokens_.size();
      }
      rows.clear();
      for (std::size_t i = 0; i < images_.size(); ++i) {
        const ImageSpan& image = images_[i];
        const std::size_t image_end = std::size_t{image.offset} + image.rows;
        if (image_end <= start || image.offset >= end) {
          continue;
        }
        if (image.offset < start) {
          throw std::logic_error("gemma4 prefill resumed inside an image");
        }
        if (image_end > end) {
          // Stop before an image that does not fit; one that starts the
          // chunk fits because images are at most a chunk long.
          end = image.offset > start ? image.offset : image_end;
          if (image.offset > start) {
            break;
          }
        }
        const float* embedding = embed ? embed(i) : nullptr;
        if (embedding == nullptr) {
          throw std::invalid_argument("gemma4 image has no embeddings");
        }
        rows.push_back({static_cast<std::uint32_t>(image.offset - start),
                        image.rows, embedding});
      }
      count = end - start;
      const bool last = end == tokens_.size();
      // Logit rows: the frontier's last token when this chunk holds it, and
      // the last lookahead row.
      std::vector<std::uint32_t> logit_rows;
      const bool holds_frontier = start < frontier && frontier <= end;
      if (holds_frontier) {
        logit_rows.push_back(static_cast<std::uint32_t>(frontier - 1 - start));
      }
      if (last && !lookahead.empty()) {
        logit_rows.push_back(static_cast<std::uint32_t>(count - 1));
      }
      executor.Forward(*cache_, std::span(tokens_).subspan(start, count),
                       static_cast<std::uint32_t>(start), logit_rows, rows);
      if (logit_rows.empty()) {
        continue;
      }
      executor.CopyLogits(logit_rows.size(), &logits);
      const std::size_t vocab = model_->VocabSize();
      if (holds_frontier) {
        executor.CommitHidden(*cache_, logit_rows.front());
        logits_.assign(logits.begin(),
                       logits.begin() + static_cast<std::ptrdiff_t>(vocab));
      }
      if (last && !lookahead.empty()) {
        executor.StashHidden(*cache_, logit_rows.back());
        ahead_logits_.assign(logits.end() - static_cast<std::ptrdiff_t>(vocab),
                             logits.end());
      }
    }
  } catch (const std::exception& e) {
    tokens_.resize(begin);
    std::erase_if(images_, [&](const ImageSpan& image) {
      return std::size_t{image.offset} + image.rows > begin;
    });
    lookup_.Clear();
    DropLookahead();
    return Fail(error_msg, e.what());
  }
  tokens_.resize(frontier);
  ahead_.assign(lookahead.begin(), lookahead.end());
  valid_ = true;
  return true;
}

bool Session::Sync(std::span<const TokenId> prompt, std::string* error_msg) {
  return Sync(prompt, {}, {}, error_msg);
}

bool Session::Sync(std::span<const TokenId> prompt,
                   std::span<const ImageSpan> images,
                   const ImageEmbeddings& embed, std::string* error_msg,
                   std::span<const TokenId> lookahead) {
  if (prompt.empty()) {
    Reset();
    return Fail(error_msg, "prompt is empty");
  }
  if (model_->options_.draft_calibration == DraftCalibrationScope::kRequest) {
    for (auto& calibration : calibration_) {
      calibration.Reset();
    }
  }
  if (prompt.size() > cache_->max_context) {
    return Fail(error_msg, "prompt exceeds the session context");
  }
  // Rows the previous Sync evaluated past its frontier: adopt them if this
  // prompt asks for exactly those tokens next.
  if (!ahead_.empty()) {
    const bool adopt =
        valid_ && !pending_ &&
        prompt.size() == tokens_.size() + ahead_.size() &&
        std::ranges::equal(prompt.first(tokens_.size()), tokens_) &&
        std::ranges::equal(prompt.subspan(tokens_.size()), ahead_) &&
        std::ranges::equal(images, images_);
    if (adopt) {
      try {
        std::lock_guard lock(model_->mutex_);
        model_->executor_->AdoptStashedHidden(*cache_);
      } catch (const std::exception& e) {
        Reset();
        return Fail(error_msg, e.what());
      }
      tokens_.insert(tokens_.end(), ahead_.begin(), ahead_.end());
      logits_.swap(ahead_logits_);
      DropLookahead();
      return true;
    }
    DropLookahead();
  }
  std::size_t previous_end = 0;
  for (const ImageSpan& image : images) {
    // A text token follows every image, so the frontier is never inside one.
    if (image.rows == 0 || image.offset < previous_end ||
        std::size_t{image.offset} + image.rows >= prompt.size() ||
        image.rows > model_->executor_->max_rows()) {
      return Fail(error_msg, "invalid image placement");
    }
    previous_end = std::size_t{image.offset} + image.rows;
  }
  pending_.reset();
  std::size_t common = 0;
  const std::size_t limit = std::min(prompt.size(), tokens_.size());
  while (common < limit && prompt[common] == tokens_[common]) {
    ++common;
  }
  // Equal tokens do not mean equal images: stop at the first image that
  // differs, and never resume inside an image of the new prompt.
  for (std::size_t i = 0; i < std::max(images.size(), images_.size()); ++i) {
    if (i < images.size() && i < images_.size() && images[i] == images_[i]) {
      continue;
    }
    std::size_t differs = common;
    if (i < images.size()) {
      differs = std::min<std::size_t>(differs, images[i].offset);
    }
    if (i < images_.size()) {
      differs = std::min<std::size_t>(differs, images_[i].offset);
    }
    common = differs;
    break;
  }
  if (common == prompt.size() && common == tokens_.size() && valid_) {
    return true;
  }
  // The last prompt token is re-evaluated to produce its logits.
  common = std::min(common, prompt.size() - 1);
  for (const ImageSpan& image : images) {
    if (common > image.offset && common < image.offset + image.rows) {
      common = image.offset;
    }
  }
  // Rewinding needs the sliding window before `common` still in the ring.
  const Config& c = model_->config();
  if (tokens_.size() - common + c.sliding_window > cache_->ring) {
    common = 0;
  }
  tokens_.assign(prompt.begin(), prompt.end());
  images_.assign(images.begin(), images.end());
  lookup_.Clear();
  const bool ahead = lookahead.size() <= rocm::kSplitRows &&
                     prompt.size() + lookahead.size() <= cache_->max_context;
  return Extend(common, error_msg, embed,
                ahead ? lookahead : std::span<const TokenId>{});
}

bool Session::Evaluate(TokenId token, std::string* error_msg) {
  pending_.reset();
  if (tokens_.size() >= cache_->max_context) {
    return Fail(error_msg, "session context is full");
  }
  tokens_.push_back(token);
  return Extend(tokens_.size() - 1, error_msg);
}

bool Session::EvaluateAll(std::span<const TokenId> tokens,
                          std::vector<float>* logits, std::string* error_msg) {
  auto& executor = *model_->executor_;
  DropLookahead();
  if (tokens_.size() + tokens.size() > cache_->max_context) {
    return Fail(error_msg, "tokens exceed the session context");
  }
  const std::size_t vocab = model_->VocabSize();
  const std::size_t chunk = std::min<std::size_t>(
      executor.max_rows(), model_->options_.max_logit_rows);
  logits->clear();
  valid_ = false;
  pending_.reset();
  try {
    std::lock_guard lock(model_->mutex_);
    std::vector<std::uint32_t> rows;
    std::vector<float> part;
    for (std::size_t start = 0; start < tokens.size(); start += chunk) {
      const std::size_t count = std::min(chunk, tokens.size() - start);
      rows.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        rows[i] = static_cast<std::uint32_t>(i);
      }
      const auto first = static_cast<std::uint32_t>(tokens_.size());
      tokens_.insert(tokens_.end(), tokens.begin() + start,
                     tokens.begin() + start + count);
      executor.Forward(*cache_, tokens.subspan(start, count), first, rows);
      executor.CommitHidden(*cache_, static_cast<std::uint32_t>(count - 1));
      executor.CopyLogits(count, &part);
      logits->insert(logits->end(), part.begin(), part.end());
    }
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  logits_.assign(logits->end() - static_cast<std::ptrdiff_t>(vocab),
                 logits->end());
  valid_ = true;
  return true;
}

namespace {

// The confidence policy: a chain continues while the drafter's estimate that
// every draft so far is accepted (the product of its confidences) stays at
// or above a floor; the draft that falls below it is not verified. Greedy
// chains use the drafter's top-1 share of its top-64 candidates, sampled
// chains the probability of the sampled proposal, capped at four drafts.
// Fitted on the UD-Q4_K_XL drafter (docs/models/gemma-4-31b/EXPERIMENTS.md);
// the calibrated policy (draft_policy.hpp) replaces these constants.
constexpr float kGreedyChainFloor = 0.5F;
constexpr float kSampledChainFloor = 0.3F;
constexpr std::uint32_t kSampledDraftCap = 4;

/// The top candidate's softmax share among the top-64 logits.
float TopShare(const qwen38_flash_next::MtpCandidateLogits& candidates) {
  float top = candidates.logits[0];
  for (std::size_t i = 1; i < candidates.size; ++i) {
    top = std::max(top, candidates.logits[i]);
  }
  double total = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    total += std::exp(static_cast<double>(candidates.logits[i] - top));
  }
  return static_cast<float>(1.0 / total);
}

/// The argmax candidate, lowest token id on ties.
std::int32_t TopToken(const qwen38_flash_next::MtpCandidateLogits& candidates) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < candidates.size; ++i) {
    if (candidates.logits[i] > candidates.logits[best] ||
        (candidates.logits[i] == candidates.logits[best] &&
         candidates.ids[i] < candidates.ids[best])) {
      best = i;
    }
  }
  return static_cast<std::int32_t>(candidates.ids[best]);
}

/// The drafter's choice after `chosen`: a sibling draft at the same depth.
struct Alternative {
  std::int32_t token{-1};
  /// Calibration signal: the drafter's probability of the sibling once the
  /// draft beside it is ruled out.
  float signal{0.0F};
};

/// Greedy: the runner-up candidate (TopToken's order).
Alternative GreedyAlternative(
    const qwen38_flash_next::MtpCandidateLogits& candidates,
    std::int32_t chosen) {
  std::size_t best = candidates.size;
  std::size_t top = candidates.size;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    if (static_cast<std::int32_t>(candidates.ids[i]) == chosen) {
      top = i;
      continue;
    }
    if (best == candidates.size ||
        candidates.logits[i] > candidates.logits[best] ||
        (candidates.logits[i] == candidates.logits[best] &&
         candidates.ids[i] < candidates.ids[best])) {
      best = i;
    }
  }
  if (best == candidates.size || top == candidates.size) {
    return {};
  }
  double total = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    total += std::exp(
        static_cast<double>(candidates.logits[i] - candidates.logits[top]));
  }
  const double second = std::exp(static_cast<double>(candidates.logits[best] -
                                                     candidates.logits[top])) /
                        total;
  const double rest = 1.0 - 1.0 / total;
  return {static_cast<std::int32_t>(candidates.ids[best]),
          rest > 0.0 ? static_cast<float>(second / rest) : 0.0F};
}

/// Sampled: a draw from the proposal without its token.
Alternative SampledAlternative(const qwen38_flash_next::MtpProposal& proposal,
                               std::uint64_t* rng) {
  const double rest = 1.0 - static_cast<double>(proposal.probability);
  if (!(rest > 0.0)) {
    return {};
  }
  const double draw = sampling::Uniform(rng) * rest;
  double cumulative = 0.0;
  std::size_t last = proposal.size;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    if (proposal.ids[i] == proposal.token || proposal.probabilities[i] <= 0) {
      continue;
    }
    last = i;
    cumulative += proposal.probabilities[i];
    if (draw < cumulative) {
      break;
    }
  }
  if (last == proposal.size) {
    return {};
  }
  return {static_cast<std::int32_t>(proposal.ids[last]),
          static_cast<float>(proposal.probabilities[last] / rest)};
}

struct SiblingVerification {
  sampling::TokenId token;
  bool accepted;  ///< the chain's draft
  bool sibling;   ///< the sibling, after the draft's rejection
};

/// p/q verification of a draft with a sibling drawn from q without it:
/// recursive rejection sampling. The draft is accepted with min(1, p/q);
/// else the sibling, against the residual r = norm(max(p - q, 0)) and its
/// own proposal q' = q without the draft, with min(1, r/q'); else the token
/// comes from norm(max(r - q', 0)). The emitted token follows p exactly.
SiblingVerification VerifyWithSibling(std::span<const float> logits,
                                      const qwen38_flash_next::MtpProposal& q,
                                      sampling::TokenId sibling,
                                      sampling::SamplerState& sampler) {
  const auto target = sampler.Distribution(logits);
  if (sampler.Uniform() * q.probability < target.probability(q.token)) {
    return {q.token, true, false};
  }
  const auto draft = [&q](sampling::TokenId token) {
    for (std::size_t i = 0; i < q.size; ++i) {
      if (q.ids[i] == token) {
        return static_cast<double>(q.probabilities[i]);
      }
    }
    return 0.0;
  };
  std::vector<sampling::Probability> residual;
  double total = 0.0;
  for (const auto& entry : target.entries()) {
    const double value = std::max(entry.value - draft(entry.token), 0.0);
    if (value > 0.0) {
      residual.push_back({.token = entry.token, .value = value});
      total += value;
    }
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    return {target.Sample(sampler.mutable_rng_state()), false, false};
  }
  const double rest = 1.0 - static_cast<double>(q.probability);
  const double own = draft(sibling) / rest;
  const double r =
      std::max(target.probability(sibling) - draft(sibling), 0.0) / total;
  if (sampler.Uniform() * own < r) {
    return {sibling, false, true};
  }
  std::vector<sampling::Probability> second;
  for (const auto& entry : residual) {
    const double other =
        entry.token == q.token ? 0.0 : draft(entry.token) / rest;
    const double value = std::max(entry.value / total - other, 0.0);
    if (value > 0.0) {
      second.push_back({.token = entry.token, .value = value});
    }
  }
  if (second.empty()) {
    return {sampling::SamplingDistribution(std::move(residual))
                .Sample(sampler.mutable_rng_state()),
            false, false};
  }
  return {sampling::SamplingDistribution(std::move(second))
              .Sample(sampler.mutable_rng_state()),
          false, false};
}

}  // namespace

/// One cycle's drafting: the length policy, sampled proposals and
/// prompt-lookup copies, asked for one proposal per drafter step.
struct Session::CycleDraft {
  CycleDraft(Session& owner, Cycle& owner_cycle, std::uint32_t chain_steps,
             std::uint32_t slots, const DraftBatch& others,
             DraftShare* batch_share)
      : session(owner),
        cycle(owner_cycle),
        policy(owner.model_->options_.draft_policy),
        min_drafts(owner.model_->options_.min_draft_tokens),
        steps(chain_steps),
        max_drafts(slots),
        alone(Alone(owner, owner_cycle)),
        costs(SharedCosts(owner, owner_cycle, alone ? nullptr : batch_share)),
        calibrated(owner.Calibration(owner_cycle.sampled), costs, min_drafts,
                   chain_steps, alone ? DraftBatch{} : others),
        share(batch_share),
        context{*owner.pending_} {
    session.lookup_.Extend(session.tokens_);
    if (cycle.sampled) {
      // A cycle-local proposal stream; target draws keep the sampler's.
      draft_rng = sampling::NextRandom(cycle.sampler->mutable_rng_state());
      // Siblings draw from their own stream: the chain's drafts stay those
      // of a cycle without them.
      sibling_rng = draft_rng ^ 0x9E3779B97F4A7C15ULL;
      // Drafts are proposals: only the target's verification is
      // constrained (a grammar may exclude every drafter candidate).
      draft_sampler = cycle.sampler->WithoutConstraint();
    }
  }

  /// Whether the cycle prices its drafts as if alone. The tokens a seed
  /// draws depend on the drafts proposed, so with per-request calibration
  /// (exact seeded replay) a sampled cycle ignores the sessions batched
  /// beside it. Greedy output is exact either way.
  static bool Alone(const Session& owner, const Cycle& c) {
    return c.sampled && owner.model_->options_.draft_calibration ==
                            DraftCalibrationScope::kRequest;
  }

  /// Drafter steps shared by the sessions of a batch cost each a share.
  static DraftCosts SharedCosts(const Session& owner, const Cycle& c,
                                const DraftShare* batch_share) {
    DraftCosts costs =
        DraftCostsAt(c.position, owner.model_->config().HasExperts());
    if (batch_share != nullptr && batch_share->sessions > 1) {
      for (float& ms : costs.draft) {
        ms /= static_cast<float>(batch_share->sessions);
      }
    }
    return costs;
  }

  [[nodiscard]] bool Worthwhile() const {
    return policy != DraftPolicy::kCalibrated || calibrated.FirstDraftCanPay();
  }

  /// Fills the remaining slots with the tokens that followed an earlier
  /// occurrence of the context (at least 12 tokens); they end the chain.
  bool Copy() {
    const auto match = session.lookup_.Find(session.tokens_, context);
    std::size_t room = max_drafts - (context.size() - 1);
    if (SharesSpare()) {
      room = std::min<std::size_t>(room, FreeRows());
    }
    if (match.length == 0 || room == 0) {
      return false;
    }
    const std::size_t count =
        std::min(room, session.tokens_.size() - match.start);
    copies.assign(
        session.tokens_.begin() + static_cast<std::ptrdiff_t>(match.start),
        session.tokens_.begin() +
            static_cast<std::ptrdiff_t>(match.start + count));
    for (std::size_t i = 0; i < count; ++i) {
      TakeRow();
    }
    return count != 0;
  }

  /// Whether this cycle may go past its even share into the batch's spare
  /// rows (a calibrated cycle priced against the batch).
  [[nodiscard]] bool SharesSpare() const {
    return share != nullptr && share->spare + share->fair != 0 && !alone &&
           policy == DraftPolicy::kCalibrated;
  }
  /// Rows this cycle may still add: what remains of its share, then spare.
  [[nodiscard]] std::uint32_t FreeRows() const {
    return (used < share->fair ? share->fair - used : 0) + share->spare;
  }
  /// Counts one more row, past the share from the spare rows.
  void TakeRow() {
    if (!SharesSpare()) {
      return;
    }
    if (used >= share->fair && share->spare > 0) {
      --share->spare;
    }
    ++used;
  }

  /// Whether the draft just proposed ends the chain unverified. `kept`
  /// drafts precede it; `confidence` feeds the confidence policy's chain.
  bool Stop(std::size_t kept, float confidence,
            const qwen38_flash_next::MtpCandidateLogits& c) {
    switch (policy) {
      case DraftPolicy::kCalibrated: {
        if (SharesSpare() && FreeRows() == 0) {
          return true;
        }
        const float signal = DraftSignal(c);
        const float before = calibrated.Expected();
        if (share != nullptr && !alone) {
          // The other sessions as they stand now.
          const auto own_rows = static_cast<std::uint32_t>(kept + 1);
          calibrated.SetOthers({.rows = share->rows - own_rows,
                                .expected = share->expected - before});
        }
        if (!calibrated.Include(signal)) {
          return true;
        }
        if (share != nullptr) {
          share->rows += 1;
          share->expected += calibrated.Expected() - before;
        }
        TakeRow();
        cycle.signals.push_back(signal);
        return false;
      }
      case DraftPolicy::kConfidence:
        chain *= confidence;
        return kept >= min_drafts &&
               chain < (cycle.sampled ? kSampledChainFloor : kGreedyChainFloor);
      case DraftPolicy::kFixed:
        break;
    }
    return false;
  }

  rocm::DraftProposal Propose(const qwen38_flash_next::MtpCandidateLogits& c) {
    if (cycle.sampled) {
      auto proposal =
          qwen38_flash_next::SampleMtpProposal(c, draft_sampler, &draft_rng);
      if (Stop(cycle.proposals.size(), proposal.probability, c)) {
        (void)Copy();
        return rocm::DraftProposal{};
      }
      draft_sampler.Accept(proposal.token);
      alternatives.push_back(SampledAlternative(proposal, &sibling_rng));
      cycle.proposals.push_back(proposal);
      context.push_back(static_cast<std::int32_t>(proposal.token));
      return rocm::DraftProposal{
          .token = static_cast<std::int32_t>(proposal.token), .last = Copy()};
    }
    if (Stop(context.size() - 1, TopShare(c), c)) {
      (void)Copy();
      return rocm::DraftProposal{};
    }
    const std::int32_t token = TopToken(c);
    alternatives.push_back(GreedyAlternative(c, token));
    context.push_back(token);
    return rocm::DraftProposal{.token = token, .last = Copy()};
  }

  Session& session;
  Cycle& cycle;
  DraftPolicy policy;
  std::uint32_t min_drafts;
  std::uint32_t steps;  ///< drafter steps the chain may run
  std::uint32_t max_drafts;
  bool alone;
  DraftCosts costs;
  CalibratedChain calibrated;
  DraftShare* share;
  float chain{1.0F};
  /// Rows past the pending one taken from the batch (SharesSpare).
  std::uint32_t used{0};
  /// The context the next token continues: pending plus kept drafts.
  std::vector<std::int32_t> context;
  std::vector<std::int32_t> copies;
  std::vector<std::int32_t> drafts;
  /// Per MTP draft, the sibling it could have beside it.
  std::vector<Alternative> alternatives;
  sampling::SamplerState draft_sampler{sampling::SamplingConfig{}, {}};
  std::uint64_t draft_rng{0};
  std::uint64_t sibling_rng{0};
};

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  DropLookahead();
  Cycle cycle{.max_tokens = max_tokens,
              .sampler = &sampler,
              .result = result,
              .stop_at_eos = stop_at_eos};
  std::lock_guard lock(model_->mutex_);
  if (!BeginCycle(cycle, model_->DraftTokens(), error_msg)) {
    return false;
  }
  if (cycle.done) {
    return true;
  }
  auto& executor = *model_->executor_;
  std::vector<float> logits;
  try {
    std::vector<std::uint32_t> logit_rows(cycle.rows.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(*cache_, cycle.rows, cycle.position, logit_rows, {},
                     static_cast<std::uint32_t>(cycle.siblings.size()));
    executor.CopyLogits(cycle.rows.size(), &logits);
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  FinishCycle(cycle, logits, 0);
  return true;
}

DraftCalibration& Session::Calibration(bool sampled, bool sibling) {
  auto& tables =
      model_->options_.draft_calibration == DraftCalibrationScope::kRequest
          ? calibration_
          : model_->calibration_;
  return tables[(sampled ? 1 : 0) + (sibling ? 2 : 0)];
}

bool Session::BeginCycle(Cycle& cycle, std::uint32_t draft_limit,
                         std::string* error_msg, const DraftBatch& others,
                         DraftShare* share) {
  auto& sampler = *cycle.sampler;
  auto* result = cycle.result;
  result->tokens.clear();
  result->stop = false;
  cycle.done = true;
  if (cycle.max_tokens == 0) {
    return true;
  }
  if (!pending_) {
    if (!valid_) {
      return Fail(error_msg, "session has no logits to decode from");
    }
    const auto token = static_cast<TokenId>(sampler.Sample(logits_));
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    result->stop = cycle.stop_at_eos && model_->IsStopToken(token);
    pending_ = token;
    if (result->stop || result->tokens.size() >= cycle.max_tokens) {
      return true;
    }
  }
  const TokenId pending = *pending_;
  const auto position = static_cast<std::uint32_t>(tokens_.size());
  if (position + 1 >= cache_->max_context) {
    return Fail(error_msg, "session context is full");
  }
  cycle.done = false;
  cycle.position = position;
  cycle.rows.assign(1, pending);
  const std::size_t budget = cycle.max_tokens - result->tokens.size();
  std::uint32_t steps = 0;
  if (model_->HasMtp() && budget > 1) {
    steps = static_cast<std::uint32_t>(std::min<std::size_t>(
        {draft_limit, budget - 1, cache_->max_context - position - 1}));
  }
  // With random sampling the drafter samples its proposals too, and each is
  // verified by p/q rejection with residual correction; greedy decoding
  // keeps argmax drafts accepted when the target's own choice agrees.
  cycle.sampled = steps > 0 && sampler.config().uses_random_sampling();
  // Prompt lookup may fill every slot; the policy may stop MTP drafts
  // earlier.
  const ModelOptions& options = model_->options_;
  const DraftPolicy policy = options.draft_policy;
  // Past the even split only calibrated cycles priced against the batch
  // (they may take its spare rows).
  if (share != nullptr &&
      (policy != DraftPolicy::kCalibrated ||
       (cycle.sampled &&
        options.draft_calibration == DraftCalibrationScope::kRequest))) {
    steps = std::min(steps, share->fair);
  }
  const std::uint32_t max_drafts = steps;
  if (cycle.sampled && policy == DraftPolicy::kConfidence) {
    steps = std::min(steps, kSampledDraftCap);
  }
  cycle.signals.clear();
  if (steps == 0) {
    return true;
  }
  try {
    cycle.draft = std::make_unique<CycleDraft>(*this, cycle, steps, max_drafts,
                                               others, share);
    if (!cycle.draft->Worthwhile()) {
      // Beside other sessions not even a certain draft would pay: verify
      // the pending token alone without running the drafter.
      cycle.draft.reset();
      return true;
    }
    if (cycle.draft->Copy()) {
      // The context continues an earlier passage: verify the copies
      // without running the drafter.
      FinishDraft(cycle);
      return true;
    }
    if (share == nullptr) {
      CycleDraft& draft = *cycle.draft;
      model_->executor_->DraftChain(
          *cache_, pending, position, steps, &draft.drafts,
          [&draft](const qwen38_flash_next::MtpCandidateLogits& c) {
            return draft.Propose(c);
          });
      FinishDraft(cycle);
    }
  } catch (const std::exception& e) {
    cycle.draft.reset();
    Reset();
    return Fail(error_msg, e.what());
  }
  return true;
}

void Session::FinishDraft(Cycle& cycle) {
  CycleDraft& draft = *cycle.draft;
  if (cycle.sampled) {
    // A copied token is a point-mass proposal: accepted with the target's
    // probability, a rejection resampling without it.
    for (const std::int32_t token : draft.copies) {
      qwen38_flash_next::MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      cycle.proposals.push_back(proposal);
    }
  }
  cycle.rows.insert(cycle.rows.end(), draft.drafts.begin(), draft.drafts.end());
  cycle.rows.insert(cycle.rows.end(), draft.copies.begin(), draft.copies.end());
  cycle.copied = draft.copies.size();
  if (draft.policy == DraftPolicy::kCalibrated) {
    // Siblings, shallowest first, while each pays for its row: the
    // drafter's next choice where the chain's draft is rejected. How many
    // fit depends on the rows a batch leaves, and they change the target's
    // draws, so an exact-replay sampled cycle (alone) proposes none.
    auto rows = static_cast<std::uint32_t>(cycle.rows.size());
    const DraftCalibration& hits = Calibration(cycle.sampled, true);
    if (draft.share != nullptr && !draft.alone) {
      const auto own = static_cast<std::uint32_t>(draft.drafts.size() + 1);
      draft.calibrated.SetOthers(
          {.rows = draft.share->rows - own,
           .expected = draft.share->expected - draft.calibrated.Expected()});
    }
    for (std::size_t i = 0;
         !draft.alone && i < draft.drafts.size() &&
         i < draft.alternatives.size() && rows < draft.max_drafts + 1 &&
         (!draft.SharesSpare() || draft.FreeRows() > 0);
         ++i) {
      const Alternative& alt = draft.alternatives[i];
      const float before = draft.calibrated.Expected();
      if (alt.token < 0 ||
          !draft.calibrated.IncludeSibling(static_cast<std::uint32_t>(i), rows,
                                           hits.Estimate(alt.signal))) {
        break;
      }
      if (draft.share != nullptr) {
        draft.share->rows += 1;
        draft.share->expected += draft.calibrated.Expected() - before;
      }
      draft.TakeRow();
      cycle.siblings.push_back(alt.token);
      cycle.sibling_signals.push_back(alt.signal);
      ++rows;
    }
    cycle.rows.insert(cycle.rows.end(), cycle.siblings.begin(),
                      cycle.siblings.end());
    cycle.expected = draft.calibrated.Expected();
    cycle.draft_ms = draft.calibrated.DraftMs();
  }
  cycle.draft.reset();
}

void Session::FinishCycle(Cycle& cycle, std::span<const float> logits,
                          std::uint32_t first_hidden_row) {
  auto& sampler = *cycle.sampler;
  auto* result = cycle.result;
  const auto& rows = cycle.rows;
  const auto emit = [&](TokenId token) {
    sampler.Accept(static_cast<sampling::TokenId>(token));
    result->tokens.push_back(token);
    if (cycle.stop_at_eos && model_->IsStopToken(token)) {
      result->stop = true;
    }
  };
  // Row i predicts the token after rows[i]; drafts stay while the target's
  // own sample agrees with them. Sibling rows trail the chain.
  const std::size_t siblings = cycle.siblings.size();
  const std::size_t chain = rows.size() - siblings;
  const std::size_t vocab = model_->VocabSize();
  std::size_t row = 0;
  std::uint64_t accepted = 0;
  bool agrees = false;
  bool sibling = false;
  for (;; ++row) {
    const auto row_logits = logits.subspan(row * vocab, vocab);
    agrees = false;
    const bool beside = row < siblings && row + 1 < chain;
    if (cycle.sampled && row + 1 < chain) {
      if (beside) {
        const auto verified = VerifyWithSibling(
            row_logits, cycle.proposals[row], cycle.siblings[row], sampler);
        emit(static_cast<TokenId>(verified.token));
        agrees = verified.accepted;
        sibling = verified.sibling;
      } else {
        const auto verified = qwen38_flash_next::VerifyMtpProposal(
            row_logits, cycle.proposals[row], sampler);
        emit(static_cast<TokenId>(verified.token));
        agrees = verified.accepted;
      }
    } else {
      emit(static_cast<TokenId>(sampler.Sample(row_logits)));
      agrees = row + 1 < chain && result->tokens.back() == rows[row + 1];
      sibling =
          beside && !agrees && result->tokens.back() == cycle.siblings[row];
    }
    if (!agrees || result->stop || result->tokens.size() >= cycle.max_tokens) {
      break;
    }
    ++accepted;
  }
  if (row < siblings && row + 1 < chain && !agrees) {
    Calibration(cycle.sampled, true)
        .Observe(cycle.sibling_signals[row], sibling);
  }
  // An accepted sibling (the target's own token) commits in place of the
  // rejected draft, and its row's token follows.
  std::size_t last = row;
  if (sibling && !result->stop && result->tokens.size() < cycle.max_tokens) {
    last = chain + row;
    emit(static_cast<TokenId>(
        sampler.Sample(logits.subspan(last * vocab, vocab))));
    tokens_.insert(tokens_.end(), rows.begin(), rows.begin() + row + 1);
    tokens_.push_back(cycle.siblings[row]);
    model_->executor_->MoveKey(
        *cache_, cycle.position + static_cast<std::uint32_t>(chain + row),
        cycle.position + static_cast<std::uint32_t>(row + 1));
    stats_.siblings_accepted += 1;
  } else {
    // Rows [0, row] are committed; the last emitted token becomes pending.
    tokens_.insert(tokens_.end(), rows.begin(), rows.begin() + row + 1);
  }
  model_->executor_->CommitHidden(
      *cache_, first_hidden_row + static_cast<std::uint32_t>(last));
  const auto kept = logits.subspan(last * vocab, vocab);
  logits_.assign(kept.begin(), kept.end());
  valid_ = true;
  pending_ = result->tokens.back();
  stats_.cycles += 1;
  stats_.verified += chain > 1 ? 1 : 0;
  stats_.drafted += chain - 1;
  stats_.accepted += accepted;
  stats_.siblings += siblings;
  // Copies trail the MTP drafts.
  const std::size_t mtp = chain - 1 - cycle.copied;
  stats_.copied += cycle.copied;
  // The calibrated policy learns from every draft verification judged: the
  // accepted ones and the one that ended the chain (copies trail the MTP
  // drafts and carry no signal).
  if (!cycle.signals.empty()) {
    DraftCalibration& calibration = Calibration(cycle.sampled);
    const std::size_t judged =
        std::min(row + (row + 1 < chain ? 1 : 0), cycle.signals.size());
    for (std::size_t j = 0; j < judged; ++j) {
      calibration.Observe(cycle.signals[j], j < row || agrees);
    }
  }
  stats_.copied_accepted += accepted > mtp ? accepted - mtp : 0;
}

bool Session::DecodeBatch(std::span<BatchDecode> decodes) {
  if (decodes.empty()) {
    return true;
  }
  for (BatchDecode& decode : decodes) {
    decode.session->DropLookahead();
  }
  Model& model = *decodes.front().session->model_;
  std::lock_guard lock(model.mutex_);
  // One verification forward keeps its single-session arithmetic up to
  // kSplitRows rows, shared evenly.
  // Every session takes its pending row and an even share of the rest;
  // calibrated sessions may also take the rows that split leaves over.
  const auto sessions_count = static_cast<std::uint32_t>(decodes.size());
  const std::uint32_t fair =
      std::max<std::uint32_t>(1, rocm::kSplitRows / sessions_count) - 1;
  const std::uint32_t spare =
      sessions_count > 1 ? rocm::kSplitRows - sessions_count * (fair + 1) : 0;
  const auto draft_limit = static_cast<std::uint32_t>(
      std::min<std::size_t>(model.DraftTokens(), fair + spare));
  std::vector<Cycle> cycles(decodes.size());
  std::vector<rocm::Executor::Segment> segments;
  std::vector<std::int32_t> tokens;
  std::vector<std::size_t> active;
  bool ok = true;
  // The sessions draft together, one drafter forward per step; the
  // calibrated policy prices each session's drafts against the whole
  // forward as it stands (every session starts with its pending row) and
  // charges each a share of the drafter steps.
  const auto sessions = static_cast<std::uint32_t>(decodes.size());
  DraftShare share{.sessions = sessions,
                   .rows = sessions,
                   .expected = static_cast<float>(sessions),
                   .fair = fair,
                   .spare = spare};
  const DraftBatch others{.rows = sessions - 1,
                          .expected = static_cast<float>(sessions - 1)};
  std::vector<rocm::Executor::DraftJob> jobs;
  std::vector<std::size_t> drafting;
  for (std::size_t i = 0; i < decodes.size(); ++i) {
    auto& d = decodes[i];
    if (d.session->model_.get() != &model) {
      throw std::invalid_argument("gemma4 batch mixes models");
    }
    cycles[i] = Cycle{.max_tokens = d.max_tokens,
                      .sampler = d.sampler,
                      .result = d.result,
                      .stop_at_eos = d.stop_at_eos};
    if (!d.session->BeginCycle(cycles[i], draft_limit, &d.error, others,
                               sessions > 1 ? &share : nullptr)) {
      ok = false;
      continue;
    }
    if (cycles[i].draft) {
      CycleDraft& draft = *cycles[i].draft;
      jobs.push_back({d.session->cache_.get(), cycles[i].rows.front(),
                      cycles[i].position, draft.steps,
                      [&draft](const qwen38_flash_next::MtpCandidateLogits& c) {
                        return draft.Propose(c);
                      },
                      &draft.drafts});
      drafting.push_back(i);
    }
  }
  if (!jobs.empty()) {
    try {
      model.executor_->DraftChains(jobs);
    } catch (const std::exception& e) {
      for (const std::size_t i : drafting) {
        cycles[i].draft.reset();
        decodes[i].session->Reset();
        decodes[i].error = e.what();
        cycles[i].done = true;
      }
      ok = false;
    }
    for (const std::size_t i : drafting) {
      if (cycles[i].draft) {
        decodes[i].session->FinishDraft(cycles[i]);
      }
    }
  }
  for (std::size_t i = 0; i < decodes.size(); ++i) {
    if (cycles[i].done || !decodes[i].error.empty()) {
      continue;
    }
    segments.push_back({decodes[i].session->cache_.get(), cycles[i].position,
                        static_cast<std::uint32_t>(cycles[i].rows.size()),
                        static_cast<std::uint32_t>(cycles[i].siblings.size())});
    tokens.insert(tokens.end(), cycles[i].rows.begin(), cycles[i].rows.end());
    active.push_back(i);
  }
  if (active.empty()) {
    return ok;
  }
  auto& executor = *model.executor_;
  std::vector<float> logits;
  try {
    std::vector<std::uint32_t> logit_rows(tokens.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(segments, tokens, logit_rows);
    executor.CopyLogits(tokens.size(), &logits);
  } catch (const std::exception& e) {
    for (const std::size_t i : active) {
      decodes[i].session->Reset();
      decodes[i].error = e.what();
    }
    return false;
  }
  const std::size_t vocab = model.VocabSize();
  std::uint32_t row0 = 0;
  for (const std::size_t i : active) {
    const std::size_t count = cycles[i].rows.size();
    decodes[i].session->FinishCycle(
        cycles[i],
        std::span<const float>(logits).subspan(row0 * vocab, count * vocab),
        row0);
    row0 += static_cast<std::uint32_t>(count);
  }
  return ok;
}

bool Session::EvaluateBatch(std::span<Session* const> sessions,
                            std::span<const TokenId> tokens,
                            std::string* error_msg) {
  for (Session* session : sessions) {
    session->DropLookahead();
  }
  if (sessions.size() != tokens.size()) {
    return Fail(error_msg, "gemma4 batch needs one token per session");
  }
  if (sessions.empty()) {
    return true;
  }
  if (sessions.size() > rocm::kSplitRows) {
    return Fail(error_msg, "gemma4 batch is wider than one decode forward");
  }
  Model& model = *sessions.front()->model_;
  std::lock_guard lock(model.mutex_);
  std::vector<rocm::Executor::Segment> segments;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    if (s.model_.get() != &model) {
      throw std::invalid_argument("gemma4 batch mixes models");
    }
    if (s.tokens_.size() >= s.cache_->max_context) {
      return Fail(error_msg, "session context is full");
    }
    segments.push_back(
        {s.cache_.get(), static_cast<std::uint32_t>(s.tokens_.size()), 1});
  }
  std::vector<std::int32_t> rows(tokens.begin(), tokens.end());
  std::vector<std::uint32_t> logit_rows(rows.size());
  for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
    logit_rows[i] = i;
  }
  auto& executor = *model.executor_;
  std::vector<float> logits;
  try {
    executor.Forward(segments, rows, logit_rows);
    for (std::uint32_t i = 0; i < sessions.size(); ++i) {
      executor.CommitHidden(*sessions[i]->cache_, i);
    }
    executor.CopyLogits(rows.size(), &logits);
  } catch (const std::exception& e) {
    for (Session* s : sessions) {
      s->Reset();
    }
    return Fail(error_msg, e.what());
  }
  const std::size_t vocab = model.VocabSize();
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    s.pending_.reset();
    s.tokens_.push_back(tokens[i]);
    s.logits_.assign(
        logits.begin() + static_cast<std::ptrdiff_t>(i * vocab),
        logits.begin() + static_cast<std::ptrdiff_t>((i + 1) * vocab));
    s.valid_ = true;
  }
  return true;
}

namespace {

/// Positions per snapshot block of the global and the sliding layers. Ring
/// sizes are multiples of both, so a block never wraps in a sliding layer's
/// ring. Rows around the blocks are copied into each snapshot: short sliding
/// blocks keep those few where rows are large.
constexpr std::uint32_t kGlobalBlockRows = 256;
constexpr std::uint32_t kSlidingBlockRows = 64;

constexpr std::uint32_t BlockRows(bool sliding) {
  return sliding ? kSlidingBlockRows : kGlobalBlockRows;
}

/// The blocks of a snapshot at `n`: global layers' blocks [0, global),
/// sliding layers' blocks [sliding_first, sliding_end) — whole blocks inside
/// the live window, the only sliding rows the ring is sure to hold.
struct BlockShape {
  std::uint32_t n;
  std::uint32_t live;  ///< first live sliding row
  std::uint32_t global;
  std::uint32_t sliding_first;
  std::uint32_t sliding_end;
};

BlockShape ShapeOf(const Config& c, std::uint32_t n) {
  BlockShape s{};
  s.n = n;
  s.live = n < c.sliding_window ? 0 : n - (c.sliding_window - 1);
  s.global = n / kGlobalBlockRows;
  s.sliding_first = (s.live + kSlidingBlockRows - 1) / kSlidingBlockRows;
  s.sliding_end = std::max(n / kSlidingBlockRows, s.sliding_first);
  return s;
}

/// Byte offsets of each layer's K and V rows inside a block of its kind.
struct BlockLayout {
  std::vector<std::uint64_t> k_at;
  std::vector<std::uint64_t> v_at;
  std::uint64_t global_bytes{0};
  std::uint64_t sliding_bytes{0};
};

BlockLayout LayoutOf(const Config& c,
                     const std::vector<std::uint32_t>& key_widths) {
  BlockLayout layout;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const bool sliding = c.IsSliding(l);
    std::uint64_t& bytes = sliding ? layout.sliding_bytes : layout.global_bytes;
    layout.k_at.push_back(bytes);
    bytes += std::uint64_t{BlockRows(sliding)} * key_widths[l] * 2;
    layout.v_at.push_back(bytes);
    bytes += std::uint64_t{BlockRows(sliding)} * c.KvDim(l) * 2;
  }
  return layout;
}

constexpr std::uint32_t kRest = UINT32_MAX;

/// Positions [first, first + rows) of layer `layer`'s K or V rows (`values`),
/// contiguous in the session's cache and in the snapshot: in block `block`
/// of the layer's kind, or in the rest (kRest), at byte `offset`.
struct BodyRun {
  std::uint32_t layer;
  bool values;
  std::uint32_t first;
  std::uint32_t rows;
  std::uint64_t row_bytes;
  std::uint32_t block;
  std::uint64_t offset;
};

/// Visits a snapshot body in payload order: layer by layer, K rows then V
/// rows by position. Rest rows follow one another in that order; returns
/// their bytes.
template<typename Visit>
std::uint64_t VisitBody(const Config& c,
                        const std::vector<std::uint32_t>& key_widths,
                        const BlockLayout& layout, const BlockShape& s,
                        std::uint32_t ring, const Visit& visit) {
  std::uint64_t rest = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const bool sliding = c.IsSliding(l);
    const std::uint32_t block_rows = BlockRows(sliding);
    const std::uint32_t blocked_begin =
        sliding ? s.sliding_first * block_rows : 0;
    const std::uint32_t blocked_end =
        (sliding ? s.sliding_end : s.global) * block_rows;
    for (const bool values : {false, true}) {
      const std::uint64_t row =
          std::uint64_t{values ? c.KvDim(l) : key_widths[l]} * 2;
      for (std::uint32_t p = sliding ? s.live : 0; p < s.n;) {
        BodyRun run{l, values, p, 0, row, kRest, rest};
        std::uint32_t end = s.n;
        if (p >= blocked_begin && p < blocked_end) {
          run.block = p / block_rows;
          end = (run.block + 1) * block_rows;
          run.offset = (values ? layout.v_at : layout.k_at)[l] +
                       std::uint64_t{p - run.block * block_rows} * row;
        } else if (p < blocked_begin) {
          end = std::min(blocked_begin, s.n);
        }
        if (sliding) {
          end = std::min(end, (p / ring + 1) * ring);
        }
        run.rows = end - p;
        if (run.block == kRest) {
          rest += run.rows * row;
        }
        visit(run);
        p = end;
      }
    }
  }
  return rest;
}

/// Appends a copy, extending the last one where both sides continue it.
void AddRun(std::vector<rocm::CopyRun>& runs, const void* from, void* to,
            std::uint64_t bytes) {
  if (!runs.empty()) {
    rocm::CopyRun& last = runs.back();
    if (static_cast<const std::uint8_t*>(last.from) + last.bytes == from &&
        static_cast<std::uint8_t*>(last.to) + last.bytes == to) {
      last.bytes += bytes;
      return;
    }
  }
  runs.push_back({from, to, bytes});
}

/// Runs `runs` on the executor's stream, holding `mutex`, and waits for them.
bool CopyOnStream(rocm::Executor& executor, std::mutex& mutex,
                  std::span<const rocm::CopyRun> runs) {
  std::lock_guard lock(mutex);
  bool queued = true;
  try {
    executor.CopyRuns(runs);
  } catch (const std::exception&) {
    queued = false;
  }
  return hipStreamSynchronize(executor.stream()) == hipSuccess && queued;
}

}  // namespace

bool SessionSnapshot::CopyTo(std::span<std::uint8_t> destination) const {
  if (destination.size() != SizeBytes()) {
    return false;
  }
  std::size_t at = 0;
  return Stream([&](std::span<const std::uint8_t> piece) {
    std::memcpy(destination.data() + at, piece.data(), piece.size());
    at += piece.size();
  });
}

bool SessionSnapshot::Stream(const Sink& sink) const {
  // Each piece of the body gathers into host memory in one launch on a
  // private stream: a persistence worker neither waits behind nor holds up
  // the executor's queue, and pays no per-row copy overhead.
  constexpr std::uint64_t kLongestRun = std::uint64_t{4} << 20;
  SnapshotMemory& staging = *memory_;
  std::lock_guard lock(staging.mutex);
  bool ok = staging.StagingReady();
  std::uint32_t count = 0;
  std::uint64_t filled = 0;
  std::uint64_t longest = 0;
  const auto flush = [&] {
    try {
      rocm::CopyRuns(staging.runs.data(), count, longest, staging.stream);
    } catch (const std::exception&) {
      ok = false;
    }
    ok = hipStreamSynchronize(staging.stream) == hipSuccess && ok;
    if (ok) {
      sink(std::span<const std::uint8_t>(staging.out.data(), filled));
    }
    count = 0;
    filled = 0;
    longest = 0;
  };
  if (ok) {
    sink(head_);
  }
  for (const Piece& piece : body_) {
    for (std::uint64_t done = 0; ok && done < piece.bytes;) {
      const std::uint64_t bytes =
          std::min({piece.bytes - done, SnapshotMemory::kStagingBytes - filled,
                    kLongestRun});
      staging.runs[count++] = {piece.at + done, staging.out.data() + filled,
                               bytes};
      longest = std::max(longest, bytes);
      filled += bytes;
      done += bytes;
      if (filled == SnapshotMemory::kStagingBytes ||
          count == SnapshotMemory::kStagingRuns) {
        flush();
      }
    }
  }
  if (ok && count != 0) {
    flush();
  }
  if (ok) {
    sink(tail_);
  }
  return ok;
}

void Session::SettleHeld() const {
  const std::uint64_t from = cache_->written_from;
  const std::uint64_t to = cache_->written_to;
  const auto settle = [&](auto& held, bool sliding) {
    const std::uint64_t rows = BlockRows(sliding);
    for (std::size_t i = 0; i < held.size(); ++i) {
      // Rewritten, or (sliding) a write a ring further on took a slot.
      if ((i + 1) * rows > from || (sliding && i * rows + cache_->ring < to)) {
        held[i].reset();
      }
    }
    while (!held.empty() && held.back().expired()) {
      held.pop_back();
    }
  };
  settle(held_global_, false);
  settle(held_sliding_, true);
  cache_->written_from = UINT32_MAX;
  cache_->written_to = 0;
}

std::uint64_t Session::SnapshotBytes() const {
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  std::uint64_t bytes = sizeof(SnapshotHeader) + std::uint64_t{n} * 4 + 4 +
                        images_.size() * kImageRecordBytes;
  const auto& key_widths = model_->executor_->key_widths();
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    bytes += std::uint64_t{n - FirstLiveRow(c, l, n)} *
             (key_widths[l] + c.KvDim(l)) * 2;
  }
  return bytes + std::uint64_t{c.hidden_size} * 4 +
         std::uint64_t{model_->VocabSize()} * 4;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty()) {
    Fail(error_msg, "only an evaluated session can be saved");
    return nullptr;
  }
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  auto snapshot = std::unique_ptr<SessionSnapshot>(new SessionSnapshot());
  auto& head = snapshot->head_;
  head.resize(sizeof(SnapshotHeader) + std::size_t{n} * 4 + 4 +
              images_.size() * kImageRecordBytes);
  const SnapshotHeader header{
      kSnapshotMagic,  kSnapshotPayloadVersion, n,
      c.hidden_size,   model_->VocabSize(),     c.num_layers,
      c.sliding_window};
  std::uint8_t* at = head.data();
  std::memcpy(at, &header, sizeof(header));
  at += sizeof(header);
  std::memcpy(at, tokens_.data(), tokens_.size() * 4);
  at += tokens_.size() * 4;
  const auto image_count = static_cast<std::uint32_t>(images_.size());
  std::memcpy(at, &image_count, 4);
  at += 4;
  for (const ImageSpan& image : images_) {
    std::memcpy(at, &image.offset, 4);
    std::memcpy(at + 4, &image.rows, 4);
    std::memcpy(at + 8, image.identity.data(), image.identity.size());
    at += kImageRecordBytes;
  }
  auto& tail = snapshot->tail_;
  tail.resize(std::size_t{model_->VocabSize()} * 4);
  std::memcpy(tail.data(), logits_.data(), tail.size());
  snapshot->memory_ = model_->snapshot_memory_;

  auto& executor = *model_->executor_;
  const auto& key_widths = executor.key_widths();
  const BlockLayout layout = LayoutOf(c, key_widths);
  const BlockShape shape = ShapeOf(c, n);
  const std::uint32_t ring = cache_->ring;
  // Planning and allocation touch only this session's state, so other
  // sessions' forwards proceed until the copy.
  SettleHeld();
  SnapshotMemory& memory = *model_->snapshot_memory_;
  // Blocks the cache still holds are shared; the others are copied, each
  // kind's new blocks into one allocation.
  snapshot->sliding_first_ = shape.sliding_first;
  std::vector<bool> fresh_global(shape.global);
  std::vector<bool> fresh_sliding(shape.sliding_end - shape.sliding_first);
  const auto gather =
      [&](std::vector<std::shared_ptr<const std::uint8_t>>& blocks,
          std::uint32_t first, std::vector<bool>& fresh,
          const std::vector<std::weak_ptr<const std::uint8_t>>& held,
          std::uint64_t block_bytes) {
        blocks.resize(fresh.size());
        std::uint64_t missing = 0;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
          if (first + i < held.size()) {
            blocks[i] = held[first + i].lock();
          }
          fresh[i] = blocks[i] == nullptr;
          missing += fresh[i] ? 1 : 0;
        }
        if (missing == 0) {
          return true;
        }
        const auto region = memory.Allocate(missing * block_bytes);
        if (region == nullptr) {
          return false;
        }
        std::uint64_t next = 0;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
          if (fresh[i]) {
            blocks[i] = std::shared_ptr<const std::uint8_t>(
                region, region.get() + next * block_bytes);
            ++next;
          }
        }
        return true;
      };
  const std::uint64_t hidden_bytes = std::uint64_t{c.hidden_size} * 4;
  const std::uint64_t rest_bytes =
      VisitBody(c, key_widths, layout, shape, ring, [](const BodyRun&) {});
  const auto rest = memory.Allocate(rest_bytes + hidden_bytes);
  if (rest == nullptr ||
      !gather(snapshot->global_, 0, fresh_global, held_global_,
              layout.global_bytes) ||
      !gather(snapshot->sliding_, shape.sliding_first, fresh_sliding,
              held_sliding_, layout.sliding_bytes)) {
    Fail(error_msg, "snapshot allocation failed");
    return nullptr;
  }
  snapshot->rest_ = rest;
  std::vector<rocm::CopyRun> runs;
  auto& body = snapshot->body_;
  const auto add_piece = [&](const std::uint8_t* from, std::uint64_t bytes) {
    if (!body.empty() && body.back().at + body.back().bytes == from) {
      body.back().bytes += bytes;
    } else {
      body.push_back({from, bytes});
    }
    snapshot->body_bytes_ += bytes;
  };
  VisitBody(c, key_widths, layout, shape, ring, [&](const BodyRun& r) {
    const bool sliding = c.IsSliding(r.layer);
    std::uint8_t* to = rest.get() + r.offset;
    bool copy = true;
    if (r.block != kRest) {
      const std::size_t i = sliding ? r.block - shape.sliding_first : r.block;
      const auto& blocks = sliding ? snapshot->sliding_ : snapshot->global_;
      // Fresh blocks are written here, before any other snapshot shares them.
      to = const_cast<std::uint8_t*>(blocks[i].get()) + r.offset;
      copy = sliding ? fresh_sliding[i] : fresh_global[i];
    }
    const std::uint64_t bytes = std::uint64_t{r.rows} * r.row_bytes;
    if (copy) {
      const std::uint16_t* cache =
          r.values ? cache_->v[r.layer] : cache_->k[r.layer];
      const std::uint32_t slot = sliding ? r.first % ring : r.first;
      AddRun(runs,
             reinterpret_cast<const std::uint8_t*>(cache) +
                 std::uint64_t{slot} * r.row_bytes,
             to, bytes);
    }
    add_piece(to, bytes);
  });
  std::uint8_t* hidden = rest.get() + rest_bytes;
  AddRun(runs, cache_->hidden, hidden, hidden_bytes);
  add_piece(hidden, hidden_bytes);
  if (!CopyOnStream(executor, model_->mutex_, runs)) {
    Fail(error_msg, "snapshot copy failed");
    return nullptr;
  }
  held_global_.resize(std::max<std::size_t>(held_global_.size(), shape.global));
  held_sliding_.resize(
      std::max<std::size_t>(held_sliding_.size(), shape.sliding_end));
  std::ranges::copy(snapshot->global_, held_global_.begin());
  std::ranges::copy(snapshot->sliding_,
                    held_sliding_.begin() + shape.sliding_first);
  return snapshot;
}

std::size_t Session::ReadHead(std::span<const std::uint8_t> head,
                              std::uint64_t total, std::string* error_msg) {
  Reset();
  const Config& c = model_->config();
  SnapshotHeader header{};
  if (head.size() < sizeof(header)) {
    Fail(error_msg, "snapshot is truncated");
    return 0;
  }
  std::memcpy(&header, head.data(), sizeof(header));
  if (header.magic != kSnapshotMagic ||
      header.version != kSnapshotPayloadVersion ||
      header.hidden != c.hidden_size || header.vocab != model_->VocabSize() ||
      header.layers != c.num_layers || header.window != c.sliding_window) {
    Fail(error_msg, "snapshot belongs to another model or version");
    return 0;
  }
  const std::uint32_t n = header.position;
  if (n == 0 || n > cache_->max_context) {
    Fail(error_msg, "snapshot position exceeds the session context");
    return 0;
  }
  tokens_.resize(n);
  lookup_.Clear();
  const std::size_t images_at = sizeof(header) + std::size_t{n} * 4;
  std::uint32_t image_count = 0;
  if (head.size() >= images_at + 4) {
    std::memcpy(&image_count, head.data() + images_at, 4);
  }
  if (head.size() < images_at + 4 ||
      image_count > (head.size() - images_at - 4) / kImageRecordBytes) {
    Reset();
    Fail(error_msg, "snapshot size does not match its header");
    return 0;
  }
  images_.resize(image_count);
  if (total != SnapshotBytes()) {
    Reset();
    Fail(error_msg, "snapshot size does not match its header");
    return 0;
  }
  const std::uint8_t* at = head.data() + sizeof(header);
  std::memcpy(tokens_.data(), at, std::size_t{n} * 4);
  at += std::size_t{n} * 4 + 4;
  std::size_t previous_end = 0;
  for (ImageSpan& image : images_) {
    std::memcpy(&image.offset, at, 4);
    std::memcpy(&image.rows, at + 4, 4);
    std::memcpy(image.identity.data(), at + 8, image.identity.size());
    at += kImageRecordBytes;
    if (image.rows == 0 || image.offset < previous_end ||
        std::size_t{image.offset} + image.rows >= n) {
      Reset();
      Fail(error_msg, "snapshot image placement is invalid");
      return 0;
    }
    previous_end = std::size_t{image.offset} + image.rows;
  }
  return images_at + 4 + std::size_t{image_count} * kImageRecordBytes;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  const std::size_t head_bytes =
      ReadHead(snapshot.head_, snapshot.SizeBytes(), error_msg);
  if (head_bytes == 0) {
    return false;
  }
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  const BlockShape shape = ShapeOf(c, n);
  if (head_bytes != snapshot.head_.size() ||
      snapshot.global_.size() != shape.global ||
      snapshot.sliding_first_ != shape.sliding_first ||
      snapshot.sliding_.size() != shape.sliding_end - shape.sliding_first) {
    Reset();
    return Fail(error_msg, "snapshot size does not match its header");
  }
  auto& executor = *model_->executor_;
  const auto& key_widths = executor.key_widths();
  const BlockLayout layout = LayoutOf(c, key_widths);
  const std::uint32_t ring = cache_->ring;
  SettleHeld();
  // Blocks the cache still holds keep their rows.
  const auto held = [&](std::uint32_t i, bool sliding) {
    if (sliding) {
      return i < held_sliding_.size() &&
             held_sliding_[i].lock() ==
                 snapshot.sliding_[i - shape.sliding_first];
    }
    return i < held_global_.size() &&
           held_global_[i].lock() == snapshot.global_[i];
  };
  std::vector<rocm::CopyRun> runs;
  const std::uint64_t rest_bytes =
      VisitBody(c, key_widths, layout, shape, ring, [&](const BodyRun& r) {
        const bool sliding = c.IsSliding(r.layer);
        const std::uint8_t* from = snapshot.rest_.get() + r.offset;
        if (r.block != kRest) {
          if (held(r.block, sliding)) {
            return;
          }
          from = (sliding ? snapshot.sliding_[r.block - shape.sliding_first]
                          : snapshot.global_[r.block])
                     .get() +
                 r.offset;
        }
        std::uint16_t* cache =
            r.values ? cache_->v[r.layer] : cache_->k[r.layer];
        const std::uint32_t slot = sliding ? r.first % ring : r.first;
        AddRun(runs, from,
               reinterpret_cast<std::uint8_t*>(cache) +
                   std::uint64_t{slot} * r.row_bytes,
               std::uint64_t{r.rows} * r.row_bytes);
      });
  AddRun(runs, snapshot.rest_.get() + rest_bytes, cache_->hidden,
         std::uint64_t{c.hidden_size} * 4);
  held_global_.clear();
  held_sliding_.clear();
  if (!CopyOnStream(executor, model_->mutex_, runs)) {
    Reset();
    return Fail(error_msg, "snapshot restore failed");
  }
  held_global_.assign(snapshot.global_.begin(), snapshot.global_.end());
  held_sliding_.resize(shape.sliding_end);
  std::ranges::copy(snapshot.sliding_,
                    held_sliding_.begin() + shape.sliding_first);
  logits_.assign(reinterpret_cast<const float*>(snapshot.tail_.data()),
                 reinterpret_cast<const float*>(snapshot.tail_.data()) +
                     model_->VocabSize());
  valid_ = true;
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  const std::size_t head_bytes = ReadHead(payload, payload.size(), error_msg);
  if (head_bytes == 0) {
    return false;
  }
  const Config& c = model_->config();
  const auto n = static_cast<std::uint32_t>(tokens_.size());
  const std::uint8_t* body = payload.data() + head_bytes;
  const std::size_t tail_bytes = std::size_t{model_->VocabSize()} * 4;
  const hipStream_t stream = model_->executor_->stream();
  std::lock_guard lock(model_->mutex_);
  held_global_.clear();
  held_sliding_.clear();
  const auto copy_rows = [&](std::uint16_t* cache, std::uint32_t layer,
                             std::uint32_t first, std::uint32_t width) {
    const std::size_t row = std::size_t{width} * 2;
    const bool ring = c.IsSliding(layer);
    for (std::uint32_t p = first; p < n;) {
      const std::uint32_t slot = ring ? p % cache_->ring : p;
      const std::uint32_t run =
          ring ? std::min(n - p, cache_->ring - slot) : n - p;
      (void)hipMemcpyAsync(
          reinterpret_cast<std::uint8_t*>(cache) + std::size_t{slot} * row,
          body, std::size_t{run} * row, hipMemcpyHostToDevice, stream);
      body += std::size_t{run} * row;
      p += run;
    }
  };
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::uint32_t first = FirstLiveRow(c, l, n);
    copy_rows(cache_->k[l], l, first, model_->executor_->key_widths()[l]);
    copy_rows(cache_->v[l], l, first, c.KvDim(l));
  }
  (void)hipMemcpyAsync(cache_->hidden, body, std::size_t{c.hidden_size} * 4,
                       hipMemcpyHostToDevice, stream);
  if (hipStreamSynchronize(stream) != hipSuccess) {
    Reset();
    return Fail(error_msg, "snapshot restore failed");
  }
  logits_.resize(model_->VocabSize());
  std::memcpy(logits_.data(), payload.data() + (payload.size() - tail_bytes),
              tail_bytes);
  valid_ = true;
  return true;
}
}  // namespace gufo::models::gemma4
