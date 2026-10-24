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

#include "src/cli/serve/logging.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "src/models/gemma4/draft_vocab.hpp"
#include "src/models/gemma4/kernels/rocm/device_model.hpp"
#include "src/models/gemma4/kernels/rocm/executor.hpp"
#include "src/models/gemma4/sibling_verification.hpp"
#include "src/models/gemma4/weights.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
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
  /// Device memory every snapshot allocation leaves free: the runtime's, plus
  /// buffers allocated per request (image encoding needs ~0.6 GB at 1120
  /// tokens). The snapshot budget counts host memory, which on an APU need
  /// not bound the device's.
  static constexpr std::uint64_t kDeviceReserveBytes = std::uint64_t{1} << 30;

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
  /// Device bytes, freed by the worker; null on failure or when they would
  /// leave less than the reserve free.
  std::shared_ptr<std::uint8_t> Allocate(std::uint64_t bytes) {
    std::size_t free = 0;
    std::size_t total = 0;
    if (hipMemGetInfo(&free, &total) != hipSuccess || free < bytes ||
        free - bytes < kDeviceReserveBytes) {
      return nullptr;
    }
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
  // Prompts render with the built-in template whatever the GGUF carries, so
  // fine-tunes shipping another revision of the Gemma 4 template still load.
  if (!ChatTemplate::ValidateGgufTemplate(*m->reader_, &error)) {
    server::Logger::Warn(
        "gemma4", error + "; rendering prompts with the built-in " +
                      std::string(ChatTemplate::TemplateId()) + " template");
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
  if (!options.draft_trace_path.empty()) {
    if (!m->draft_weights_) {
      Fail(error_msg, "gemma4 draft trace requires an MTP drafter");
      return nullptr;
    }
    try {
      m->trace_ = std::make_unique<CycleTrace>(options.draft_trace_path);
    } catch (const std::exception& e) {
      Fail(error_msg, e.what());
      return nullptr;
    }
  }
  // Only the mixture-of-experts cycle is short enough for the smaller draft
  // head to pay (draft_vocab.hpp).
  const std::vector<TokenId> draft_order =
      m->draft_weights_ && m->weights_->config.HasExperts()
          ? DraftHeadOrder(*m->tokenizer_)
          : std::vector<TokenId>{};
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, m->draft_weights_.get(),
      m->draft_reader_.get(), draft_order, options.prefill_weights, &error);
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
                                    executor_->key_widths(), VocabSize(),
                                    executor_->draft_table_bytes());
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
  draft_vocab_.Reset();
  logits_.clear();
  logits_on_device_ = false;
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
  ahead_stops_.clear();
}

namespace {
// Most prompt lookahead rows one prefill forward evaluates past its frontier.
constexpr std::size_t kRiddenRows = 64;
}  // namespace

bool Session::Extend(std::size_t begin, std::string* error_msg,
                     const ImageEmbeddings& embed,
                     std::span<const TokenId> lookahead,
                     std::span<const std::size_t> stops) {
  auto& executor = *model_->executor_;
  const std::size_t chunk = model_->options_.prefill_chunk;
  const std::size_t frontier = tokens_.size();
  DropLookahead();
  std::vector<AheadStop> ahead;
  // Lookahead tokens evaluated in the frontier's forward; tokens_ holds them
  // only while it runs.
  std::size_t ridden = 0;
  valid_ = false;
  try {
    std::lock_guard lock(model_->mutex_);
    std::vector<rocm::ImageRows> rows;
    std::size_t count = 0;
    for (std::size_t start = begin; start < frontier; start += count) {
      std::size_t end = std::min(start + chunk, frontier);
      // A remainder of at most kSplitRows rows joins this chunk: alone it
      // would read every weight again for a few rows.
      if (frontier - end <= rocm::kSplitRows &&
          frontier - start <= executor.max_rows()) {
        end = frontier;
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
      const bool last = end == frontier;
      if (last && !lookahead.empty()) {
        // The lookahead rides in the frontier's forward where rows are left,
        // whole or up to a later stop. It keeps the frontier's window and the
        // rest of that window's first 64-row chunk in the ring, as a capture
        // at the frontier reads them. Each ridden row also delays that
        // capture (and requests waiting for it); past kRiddenRows the delay
        // exceeds the separate forward it saves (26B: one of a few rows costs
        // about as much as 120 rows of a long one).
        const std::size_t room = std::min<std::size_t>(
            {executor.max_rows() - count,
             cache_->ring - model_->config().sliding_window - 64, kRiddenRows});
        if (lookahead.size() <= room) {
          ridden = lookahead.size();
        } else {
          for (const std::size_t stop : stops) {
            if (stop <= room) {
              ridden = stop;
            }
          }
        }
        for (const std::size_t stop : stops) {
          if (stop != 0 && stop < ridden &&
              ahead.size() + 1 < rocm::kSplitRows &&
              ahead.size() + 2 < model_->options_.max_logit_rows &&
              (ahead.empty() || stop > ahead.back().count)) {
            ahead.push_back(
                {stop, static_cast<std::uint32_t>(ahead.size()), {}});
          }
        }
        if (ridden != 0 && model_->options_.max_logit_rows >= 2) {
          ahead.push_back(
              {ridden, static_cast<std::uint32_t>(ahead.size()), {}});
          tokens_.insert(
              tokens_.end(), lookahead.begin(),
              lookahead.begin() + static_cast<std::ptrdiff_t>(ridden));
          count += ridden;
        } else {
          ahead.clear();
          ridden = 0;
        }
      }
      // Logit rows: the frontier's last token, and the lookahead rows ending
      // each stop.
      std::vector<std::uint32_t> logit_rows;
      if (last) {
        logit_rows.push_back(static_cast<std::uint32_t>(frontier - 1 - start));
        for (const AheadStop& stop : ahead) {
          logit_rows.push_back(
              static_cast<std::uint32_t>(frontier + stop.count - 1 - start));
        }
      }
      executor.Forward(*cache_, std::span(tokens_).subspan(start, count),
                       static_cast<std::uint32_t>(start), logit_rows, rows);
      if (!last) {
        continue;
      }
      executor.CommitHidden(*cache_, logit_rows.front());
      for (std::size_t i = 0; i < ahead.size(); ++i) {
        executor.StashHidden(*cache_, logit_rows[1 + i], ahead[i].slot);
      }
      const auto logits = executor.ReadLogits(logit_rows.size());
      const std::size_t vocab = model_->VocabSize();
      const auto row = [&](std::size_t i) {
        const auto at = logits.begin() + static_cast<std::ptrdiff_t>(i * vocab);
        return std::vector<float>(at, at + static_cast<std::ptrdiff_t>(vocab));
      };
      logits_ = row(0);
      logits_on_device_ = false;
      for (std::size_t i = 0; i < ahead.size(); ++i) {
        ahead[i].logits = row(1 + i);
      }
    }
  } catch (const std::exception& e) {
    tokens_.resize(begin);
    std::erase_if(images_, [&](const ImageSpan& image) {
      return std::size_t{image.offset} + image.rows > begin;
    });
    lookup_.Clear();
    draft_vocab_.Reset();
    DropLookahead();
    return Fail(error_msg, e.what());
  }
  tokens_.resize(frontier);
  ahead_.assign(lookahead.begin(),
                lookahead.begin() + static_cast<std::ptrdiff_t>(ridden));
  ahead_stops_ = std::move(ahead);
  valid_ = true;
  return true;
}

bool Session::Sync(std::span<const TokenId> prompt, std::string* error_msg) {
  return Sync(prompt, {}, {}, error_msg);
}

void Session::BeginRequest() {
  lookup_.BeginRequest();
}

bool Session::Sync(std::span<const TokenId> prompt,
                   std::span<const ImageSpan> images,
                   const ImageEmbeddings& embed, std::string* error_msg,
                   std::span<const TokenId> lookahead,
                   std::span<const std::size_t> lookahead_stops) {
  if (prompt.empty()) {
    Reset();
    return Fail(error_msg, "prompt is empty");
  }
  if (!std::ranges::is_sorted(lookahead_stops, std::ranges::less_equal{}) ||
      (!lookahead_stops.empty() &&
       lookahead_stops.back() >= lookahead.size())) {
    return Fail(error_msg, "lookahead stops must ascend below the lookahead");
  }
  BeginRequest();
  if (model_->options_.draft_calibration == DraftCalibrationScope::kRequest) {
    for (auto& calibration : calibration_) {
      calibration.Reset();
    }
  }
  if (prompt.size() > cache_->max_context) {
    return Fail(error_msg, "prompt exceeds the session context");
  }
  // Rows an earlier Sync evaluated past its frontier: adopt them up to a
  // stop if this prompt asks for exactly those tokens next, and keep the
  // rest for the next Sync.
  if (!ahead_.empty()) {
    const auto stop = std::ranges::find_if(ahead_stops_, [&](const auto& s) {
      return prompt.size() == tokens_.size() + s.count;
    });
    const bool adopt =
        valid_ && !pending_ && stop != ahead_stops_.end() &&
        std::ranges::equal(prompt.first(tokens_.size()), tokens_) &&
        std::ranges::equal(prompt.subspan(tokens_.size()),
                           std::span(ahead_).first(stop->count)) &&
        std::ranges::equal(images, images_);
    if (adopt) {
      try {
        std::lock_guard lock(model_->mutex_);
        model_->executor_->AdoptStashedHidden(*cache_, stop->slot);
      } catch (const std::exception& e) {
        Reset();
        return Fail(error_msg, e.what());
      }
      const std::size_t count = stop->count;
      const auto adopted = static_cast<std::ptrdiff_t>(count);
      tokens_.insert(tokens_.end(), ahead_.begin(), ahead_.begin() + adopted);
      logits_.swap(stop->logits);
      logits_on_device_ = false;
      ahead_.erase(ahead_.begin(), ahead_.begin() + adopted);
      ahead_stops_.erase(ahead_stops_.begin(), stop + 1);
      for (auto& later : ahead_stops_) {
        later.count -= count;
      }
      if (ahead_.empty()) {
        DropLookahead();
      }
      return true;
    }
    // Between stops, the rows the prompt shares with the lookahead stay:
    // only the last one is evaluated again, for its logits.
    if (valid_ && !pending_ && prompt.size() > tokens_.size() + 1 &&
        std::ranges::equal(prompt.first(tokens_.size()), tokens_) &&
        std::ranges::equal(images, images_)) {
      std::size_t shared = 0;
      while (shared < ahead_.size() &&
             tokens_.size() + shared < prompt.size() &&
             prompt[tokens_.size() + shared] == ahead_[shared]) {
        ++shared;
      }
      if (shared != 0) {
        tokens_.insert(tokens_.end(), ahead_.begin(),
                       ahead_.begin() + static_cast<std::ptrdiff_t>(shared));
        valid_ = false;
      }
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
  // The lookup index stays valid while the prompt extends the tokens.
  const bool extends = common == tokens_.size();
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
  if (!extends) {
    lookup_.Clear();
  }
  draft_vocab_.Reset();
  const bool ahead = prompt.size() + lookahead.size() <= cache_->max_context;
  return Extend(common, error_msg, embed,
                ahead ? lookahead : std::span<const TokenId>{},
                ahead ? lookahead_stops : std::span<const std::size_t>{});
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
  logits_on_device_ = false;
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

/// The two largest softmax shares among the top-64 logits.
std::pair<float, float> TopTwoShares(
    const qwen38_flash_next::MtpCandidateLogits& candidates) {
  float top = -INFINITY;
  float second = -INFINITY;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    const float l = candidates.logits[i];
    if (l > top) {
      second = top;
      top = l;
    } else if (l > second) {
      second = l;
    }
  }
  double total = 0.0;
  for (std::size_t i = 0; i < candidates.size; ++i) {
    total += std::exp(static_cast<double>(candidates.logits[i] - top));
  }
  const double runner =
      candidates.size > 1 ? std::exp(static_cast<double>(second - top)) : 0.0;
  return {static_cast<float>(1.0 / total), static_cast<float>(runner / total)};
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

/// Whether a sampler's target distributions can come from the verification
/// rows' LogitCandidates: random top-k sampling within kCandidateTop with no
/// constraint mask now (FinishCycle checks each row again).
bool CandidateSampling(const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();
  return config.uses_random_sampling() && config.top_k > 0 &&
         static_cast<std::uint32_t>(config.top_k) <= rocm::kCandidateTop &&
         config.min_keep <= rocm::kCandidateTop &&
         !sampler.NeedsConstraintMask();
}

/// Adds to `extra` the tokens a cycle's sampler may penalize in any of its
/// rows: those it penalizes now and the cycle's own rows (accepting them
/// adds no others). Sorted and unique once the caller sorts it.
void AddPenalizable(const sampling::SamplerState& sampler,
                    std::span<const TokenId> rows,
                    std::span<const TokenId> siblings,
                    std::vector<std::uint32_t>* extra) {
  if (!sampler.config().penalties_enabled()) {
    return;
  }
  for (const auto& penalty : sampler.penalties()) {
    extra->push_back(penalty.token);
  }
  extra->insert(extra->end(), rows.begin(), rows.end());
  extra->insert(extra->end(), siblings.begin(), siblings.end());
}

void SortUnique(std::vector<std::uint32_t>* tokens) {
  std::ranges::sort(*tokens);
  tokens->erase(std::ranges::unique(*tokens).begin(), tokens->end());
}

}  // namespace

/// A cycle's view of its verification forward's LogitCandidates. Every row
/// carries the logits of `extra`, which holds each token the sampler could
/// penalize, so the candidates hold every row's top-k after penalties; rows
/// a constraint masks, or whose candidates overflowed, read `full`.
struct Session::VerifyCandidates {
  const rocm::Executor::Candidates* rows{nullptr};
  std::uint32_t first{0};  ///< the cycle's first row in `rows`
  std::span<const std::uint32_t> extra;
  /// The cycle's full logit rows, copied on first use.
  std::function<std::span<const float>()> full;
};

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
    if (cycle.trace) {
      cycle.trace->step_cap = chain_steps;
    }
    // A request's first cycle copies nothing (PromptLookup::Step).
    lookup = session.lookup_.Step(cycle.position);
    if (lookup) {
      session.lookup_.Extend(session.tokens_);
    }
    if (cycle.sampled) {
      // A cycle-local proposal stream; target draws keep the sampler's.
      draft_rng = sampling::NextRandom(cycle.sampler->mutable_rng_state());
      // Siblings draw from their own stream: the chain's drafts stay those
      // of a cycle without them.
      sibling_rng = draft_rng ^ 0x9E3779B97F4A7C15ULL;
      // Drafts are proposals: only the target's verification is
      // constrained (a grammar may exclude every drafter candidate). Dense
      // targets' drafts skip top-p too, which keeps a sibling candidate
      // beside each draft: at top-p 0.95, 59% of 31B UD-Q8_K_XL story and
      // code-sampled proposals held one token, while chain acceptance
      // sum(min(p, q)) is 0.541 without it against 0.544 (rejection sampling
      // is exact for any q). The 26B-A4B's verification rows cost too much
      // for siblings to pay, so its drafts keep top-p.
      draft_sampler = cycle.sampler->WithoutConstraint();
      if (!session.model_->config().HasExperts()) {
        draft_sampler = draft_sampler.WithoutTopP();
      }
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
  /// occurrence of the context (PromptLookup's gates), as far as the other
  /// occurrences matching about as well agree; they end the chain, and a
  /// stop token ends them. Copies nothing unless at least `minimum` tokens
  /// qualify.
  bool Copy(std::size_t minimum = 1) {
    if (!lookup) {
      return false;
    }
    if (cycle.stop_at_eos && session.model_->IsStopToken(context.back())) {
      return false;
    }
    std::size_t room = max_drafts - (context.size() - 1);
    if (SharesSpare()) {
      room = std::min<std::size_t>(room, FreeRows());
    }
    if (room < minimum) {
      return false;
    }
    std::size_t agreed = 0;
    const auto match = session.lookup_.Find(
        session.tokens_, context, speculative::PromptLookup<>::kMinMatch,
        /*require_resume=*/false, &agreed);
    if (match.length == 0) {
      return false;
    }
    const std::size_t available = std::min(room, agreed);
    if (available < minimum) {
      return false;
    }
    copies.clear();
    for (std::size_t i = 0; i < available; ++i) {
      const std::int32_t token = session.tokens_[match.start + i];
      copies.push_back(token);
      TakeRow();
      if (cycle.stop_at_eos && session.model_->IsStopToken(token)) {
        break;
      }
    }
    cycle.copy_start = match.start;
    return true;
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
        CycleRecord* trace = cycle.trace ? &*cycle.trace : nullptr;
        const float estimate =
            trace != nullptr
                ? session.Calibration(cycle.sampled).Estimate(signal)
                : 0.0F;
        if (!calibrated.Include(signal)) {
          const CalibratedChain::Decision d = calibrated.LastDecision();
          if (trace != nullptr && d.compared) {
            trace->stop = CycleRecord::Stop{.signal = signal,
                                            .estimate = estimate,
                                            .gain = d.gain,
                                            .cost = d.cost};
          }
          return true;
        }
        if (trace != nullptr) {
          const CalibratedChain::Decision d = calibrated.LastDecision();
          trace->estimates.push_back(estimate);
          trace->margins.push_back(d.compared ? d.gain - d.cost : 0.0F);
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
    if (cycle.trace) {
      ++cycle.trace->drafter_steps;
      cycle.trace->proposal_signals.push_back(DraftSignal(c));
      const auto [top, second] = TopTwoShares(c);
      cycle.trace->proposal_top.push_back(top);
      cycle.trace->proposal_margin.push_back(top - second);
    }
    if (cycle.sampled) {
      auto proposal =
          qwen38_flash_next::SampleMtpProposal(c, draft_sampler, &draft_rng);
      if (cycle.trace) {
        cycle.trace->proposal_q.push_back(proposal.probability);
        CycleRecord::Sparse& raw = cycle.trace->draft_candidates.emplace_back();
        raw.ids.assign(c.ids.begin(), c.ids.begin() + c.size);
        raw.values.assign(c.logits.begin(), c.logits.begin() + c.size);
      }
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
    if (cycle.trace) {
      cycle.trace->proposal_q.push_back(cycle.trace->proposal_top.back());
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
  /// Whether this cycle may copy (not a request's first).
  bool lookup{false};
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
  bool small_head{false};
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
  if (cycle.trace) {
    cycle.trace->drafted_ns = CycleTrace::Now();
    cycle.trace->forward = model_->forwards_++;
  }
  auto& executor = *model_->executor_;
  const bool use_argmax =
      !cycle.sampled && sampler.config().can_use_unmodified_argmax();
  bool use_candidates = !use_argmax && !cycle.trace &&
                        executor.CanReadCandidates() &&
                        CandidateSampling(sampler);
  std::vector<std::uint32_t> extra;
  if (use_candidates) {
    AddPenalizable(sampler, cycle.rows, cycle.siblings, &extra);
    SortUnique(&extra);
    use_candidates = extra.size() <= rocm::kMaxCandidateExtra;
  }
  std::span<const float> logits;
  std::span<const std::uint32_t> argmax;
  try {
    std::vector<std::uint32_t> logit_rows(cycle.rows.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(*cache_, cycle.rows, cycle.position, logit_rows, {},
                     static_cast<std::uint32_t>(cycle.siblings.size()));
    rocm::Executor::Candidates candidates;
    VerifyCandidates view;
    if (use_argmax) {
      argmax = executor.ReadArgmax(cycle.rows.size());
    } else if (use_candidates) {
      candidates = executor.ReadCandidates(cycle.rows.size(), extra);
      view = {.rows = &candidates,
              .extra = extra,
              .full = [&executor, n = cycle.rows.size()] {
                return executor.ReadLogits(n);
              }};
    } else {
      logits = executor.ReadLogits(cycle.rows.size());
    }
    if (cycle.trace) {
      cycle.trace->verified_ns = CycleTrace::Now();
    }
    FinishCycle(cycle, logits, argmax, 0, use_candidates ? &view : nullptr);
  } catch (const std::exception& e) {
    Reset();
    return Fail(error_msg, e.what());
  }
  return true;
}

DraftCalibration& Session::Calibration(bool sampled, bool sibling) {
  auto& tables =
      model_->options_.draft_calibration == DraftCalibrationScope::kRequest
          ? calibration_
          : model_->calibration_;
  return tables[(sampled ? 1 : 0) + (sibling ? 2 : 0)];
}

std::span<const float> Session::Logits() const {
  if (!valid_) {
    return {};
  }
  if (logits_on_device_) {
    std::lock_guard lock(model_->mutex_);
    EnsureLogits();
  }
  return logits_;
}

void Session::EnsureLogits() const {
  if (logits_on_device_) {
    model_->executor_->CopyCacheLogits(*cache_, &logits_);
    logits_on_device_ = false;
  }
}

bool Session::BeginCycle(Cycle& cycle, std::uint32_t draft_limit,
                         std::string* error_msg, const DraftBatch& others,
                         DraftShare* share) {
  auto& sampler = *cycle.sampler;
  auto* result = cycle.result;
  result->tokens.clear();
  result->stop = false;
  cycle.done = true;
  cycle.trace.reset();
  if (model_->trace_) {
    cycle.trace.emplace();
    cycle.trace->begin_ns = CycleTrace::Now();
  }
  if (cycle.max_tokens == 0) {
    return true;
  }
  if (!pending_) {
    if (!valid_) {
      return Fail(error_msg, "session has no logits to decode from");
    }
    EnsureLogits();
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
    // A context that continues an earlier passage verifies at least three
    // copies without running the drafter (the drafter reads the target's
    // cache, so a skipped step costs nothing later).
    if (cycle.draft->Copy(3)) {
      FinishDraft(cycle);
      return true;
    }
    if (share == nullptr) {
      CycleDraft& draft = *cycle.draft;
      const rocm::DraftHead head = PrepareDraftHead(pending);
      draft.small_head = head.small;
      model_->executor_->DraftChain(
          *cache_, pending, position, steps, &draft.drafts,
          [&draft](const qwen38_flash_next::MtpCandidateLogits& c) {
            return draft.Propose(c);
          },
          head);
      FinishDraft(cycle);
    }
  } catch (const std::exception& e) {
    cycle.draft.reset();
    Reset();
    return Fail(error_msg, e.what());
  }
  return true;
}

rocm::DraftHead Session::PrepareDraftHead(TokenId anchor) {
  const std::span<const std::uint32_t> rows =
      model_->device_->draft_head_rows();
  if (rows.empty()) {
    return {};
  }
  const auto common = [rows](std::int32_t token) {
    return token >= 0 && static_cast<std::size_t>(token) < rows.size() &&
           rows[static_cast<std::size_t>(token)] < kDraftHeadRows;
  };
  draft_added_.clear();
  draft_vocab_.CatchUp(tokens_, common, &draft_added_);
  model_->executor_->LoadDraftTable(
      *cache_,
      static_cast<std::uint32_t>(draft_vocab_.rare().size() -
                                 draft_added_.size()),
      draft_added_);
  return {.small = draft_vocab_.Fits(anchor, common),
          .table = draft_vocab_.rare()};
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
      if (cycle.trace) {
        cycle.trace->sibling_estimates.push_back(hits.Estimate(alt.signal));
      }
      ++rows;
    }
    cycle.rows.insert(cycle.rows.end(), cycle.siblings.begin(),
                      cycle.siblings.end());
    cycle.expected = draft.calibrated.Expected();
    cycle.draft_ms = draft.calibrated.DraftMs();
    if (cycle.trace) {
      cycle.trace->priced_draft_ms = cycle.draft_ms;
      cycle.trace->priced_verify_ms = draft.calibrated.Verify(rows);
      cycle.trace->expected = cycle.expected;
    }
  }
  if (cycle.trace) {
    for (std::size_t i = 0; i < draft.drafts.size(); ++i) {
      const Alternative alt =
          i < draft.alternatives.size() ? draft.alternatives[i] : Alternative{};
      cycle.trace->runner_ups.push_back(alt.token);
      cycle.trace->runner_up_signals.push_back(alt.signal);
    }
  }
  cycle.small_head = draft.small_head;
  cycle.draft.reset();
}

void Session::TraceRows(Cycle& cycle, std::span<const float> logits,
                        std::span<const std::uint32_t> argmax,
                        std::size_t chain) const {
  CycleRecord& trace = *cycle.trace;
  const std::size_t vocab = model_->VocabSize();
  const auto target = [&](std::size_t r) {
    if (!argmax.empty()) {
      return static_cast<std::int32_t>(argmax[r]);
    }
    const auto row = logits.subspan(r * vocab, vocab);
    return static_cast<std::int32_t>(std::ranges::max_element(row) -
                                     row.begin());
  };
  for (std::size_t r = 0; r + 1 < chain; ++r) {
    const bool beside = r < trace.runner_ups.size();
    const std::int32_t up = beside ? trace.runner_ups[r] : -1;
    if (!cycle.sampled) {
      const std::int32_t t = target(r);
      trace.row_acceptance.push_back(t == cycle.rows[r + 1] ? 1.0F : 0.0F);
      if (beside) {
        trace.runner_up_acceptance.push_back(up >= 0 && t == up ? 1.0F : 0.0F);
      }
      continue;
    }
    const auto& q = cycle.proposals[r];
    const auto p =
        cycle.sampler->Distribution(logits.subspan(r * vocab, vocab));
    trace.row_acceptance.push_back(static_cast<float>(std::min(
        1.0, p.probability(q.token) / static_cast<double>(q.probability))));
    CycleRecord::Sparse& qd = trace.proposal_dist.emplace_back();
    qd.ids.assign(q.ids.begin(), q.ids.begin() + q.size);
    qd.values.assign(q.probabilities.begin(), q.probabilities.begin() + q.size);
    CycleRecord::Sparse& pd = trace.target_dist.emplace_back();
    for (const auto& entry : p.entries()) {
      pd.ids.push_back(static_cast<std::int32_t>(entry.token));
      pd.values.push_back(static_cast<float>(entry.value));
    }
    if (!beside) {
      continue;
    }
    // VerifyWithSibling's second stage.
    const auto mass = [&q](sampling::TokenId token) {
      for (std::size_t i = 0; i < q.size; ++i) {
        if (q.ids[i] == token) {
          return static_cast<double>(q.probabilities[i]);
        }
      }
      return 0.0;
    };
    double total = 0.0;
    for (const auto& entry : p.entries()) {
      total += std::max(entry.value - mass(entry.token), 0.0);
    }
    const double rest = 1.0 - static_cast<double>(q.probability);
    double hit = 0.0;
    if (up >= 0 && total > 0.0 && rest > 0.0) {
      const auto sibling = static_cast<sampling::TokenId>(up);
      const double own = mass(sibling) / rest;
      const double residual =
          std::max(p.probability(sibling) - mass(sibling), 0.0) / total;
      hit = own > 0.0 ? std::min(1.0, residual / own) : 0.0;
    }
    trace.runner_up_acceptance.push_back(static_cast<float>(hit));
  }
}

void Session::FinishCycle(Cycle& cycle, std::span<const float> logits,
                          std::span<const std::uint32_t> argmax,
                          std::uint32_t first_hidden_row,
                          const VerifyCandidates* candidates) {
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
  // Row r's candidates give its exact target distribution while the sampler
  // needs no constraint mask (nor forces a token) and holds no deferred draw.
  const auto compact = [&](std::size_t r) {
    return candidates != nullptr &&
           candidates->rows->Complete(candidates->first + r) &&
           !sampler.NeedsConstraintMask() && !sampler.SaveDrawState().pending;
  };
  const auto full_row = [&](std::size_t r) {
    if (logits.empty()) {
      logits = candidates->full();
    }
    return logits.subspan(r * vocab, vocab);
  };
  const auto candidate_row = [&](std::size_t r) {
    const std::size_t row = candidates->first + r;
    candidate_row_.clear();
    for (const auto& candidate : candidates->rows->Top(row)) {
      if (!std::ranges::binary_search(candidates->extra, candidate.token)) {
        candidate_row_.push_back(candidate);
      }
    }
    const auto extra = candidates->rows->Extra(row);
    candidate_row_.insert(candidate_row_.end(), extra.begin(), extra.end());
    return std::span<const sampling::LogitCandidate>(candidate_row_);
  };
  const auto target = [&](std::size_t r) {
    return compact(r) ? sampler.DistributionFromCandidates(candidate_row(r))
                      : sampler.Distribution(full_row(r));
  };
  const auto sample_row = [&](std::size_t r) {
    if (!argmax.empty()) {
      return static_cast<TokenId>(hip::CheckedSampleToken(argmax[r], vocab));
    }
    if (compact(r)) {
      return static_cast<TokenId>(
          target(r).Sample(sampler.mutable_rng_state()));
    }
    return static_cast<TokenId>(sampler.Sample(full_row(r)));
  };
  if (cycle.trace) {
    TraceRows(cycle, logits, argmax, chain);
  }
  std::size_t row = 0;
  std::uint64_t accepted = 0;
  bool agrees = false;
  bool sibling = false;
  for (;; ++row) {
    agrees = false;
    const bool beside = row < siblings && row + 1 < chain;
    if (cycle.sampled && row + 1 < chain) {
      if (beside) {
        const auto verified = VerifyWithSibling(
            target(row), cycle.proposals[row], cycle.siblings[row], sampler);
        emit(static_cast<TokenId>(verified.token));
        agrees = verified.accepted;
        sibling = verified.sibling;
      } else {
        const auto verified = qwen38_flash_next::VerifyMtpTarget(
            vocab, cycle.proposals[row], sampler, [&] { return target(row); });
        emit(static_cast<TokenId>(verified.token));
        agrees = verified.accepted;
      }
    } else {
      emit(sample_row(row));
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
    if (cycle.trace) {
      cycle.trace->sibling_observed = sibling;
    }
  }
  // An accepted sibling (the target's own token) commits in place of the
  // rejected draft, and its row's token follows.
  std::size_t last = row;
  if (sibling && !result->stop && result->tokens.size() < cycle.max_tokens) {
    last = chain + row;
    emit(sample_row(last));
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
  if (cycle.trace && cycle.sampled && !logits.empty()) {
    // The distribution the pending token was drawn from.
    const auto p = sampler.Distribution(logits.subspan(last * vocab, vocab));
    double top = 0.0;
    double entropy = 0.0;
    for (const auto& entry : p.entries()) {
      top = std::max(top, entry.value);
      if (entry.value > 0.0) {
        entropy -= entry.value * std::log(entry.value);
      }
    }
    cycle.trace->pending_top = static_cast<float>(top);
    cycle.trace->pending_entropy = static_cast<float>(entropy);
  }
  const auto committed_row =
      first_hidden_row + static_cast<std::uint32_t>(last);
  model_->executor_->CommitHidden(*cache_, committed_row);
  // Greedy and candidate-sampled decode keep the row on the device: nothing
  // reads it on the host before the next cycle (EnsureLogits).
  if (!argmax.empty() || candidates != nullptr) {
    model_->executor_->CommitLogits(*cache_, committed_row);
    logits_on_device_ = true;
  } else {
    const auto kept = logits.subspan(last * vocab, vocab);
    logits_.assign(kept.begin(), kept.end());
    logits_on_device_ = false;
  }
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
      if (cycle.trace) {
        cycle.trace->observed.push_back(j < row || agrees);
      }
    }
  }
  const std::size_t copies_kept = accepted > mtp ? accepted - mtp : 0;
  stats_.copied_accepted += copies_kept;
  if (cycle.copied != 0) {
    lookup_.Observe(cycle.copy_start, cycle.copied, copies_kept);
  }
  if (cycle.small_head) {
    stats_.small_head_cycles += 1;
    stats_.small_head_drafted += mtp;
    stats_.small_head_accepted += std::min(accepted, mtp);
  }
  if (cycle.trace) {
    CycleRecord& trace = *cycle.trace;
    trace.position = cycle.position;
    trace.sampled = cycle.sampled;
    if (cycle.sampled) {
      const auto& config = cycle.sampler->config();
      trace.temperature = config.temperature;
      trace.top_k = config.top_k;
      trace.top_p = config.top_p;
    }
    trace.rows = static_cast<std::uint32_t>(rows.size());
    trace.copied = static_cast<std::uint32_t>(cycle.copied);
    trace.siblings = static_cast<std::uint32_t>(siblings);
    trace.accepted = static_cast<std::uint32_t>(accepted);
    trace.sibling_accepted = sibling;
    trace.small_head = cycle.small_head;
    trace.emitted = static_cast<std::uint32_t>(result->tokens.size());
    trace.signals = cycle.signals;
    trace.end_ns = CycleTrace::Now();
    model_->trace_->Write(trace);
  }
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
                      &draft.drafts,
                      d.session->PrepareDraftHead(cycles[i].rows.front())});
      drafting.push_back(i);
    }
  }
  if (!jobs.empty()) {
    // A lone chain drafts on its small head (Executor::DraftChains).
    if (jobs.size() == 1) {
      cycles[drafting.front()].draft->small_head = jobs.front().head.small;
    }
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
  if (model.trace_) {
    const std::int64_t drafted = CycleTrace::Now();
    for (const std::size_t i : active) {
      cycles[i].trace->drafted_ns = drafted;
      cycles[i].trace->forward = model.forwards_;
      cycles[i].trace->sessions = static_cast<std::uint32_t>(active.size());
    }
    ++model.forwards_;
  }
  auto& executor = *model.executor_;
  // Each cycle reads argmax ids (greedy) or sampling candidates; if any
  // needs full logits, all read them.
  std::vector<std::uint8_t> sample(cycles.size(), 0);
  bool any_argmax = false;
  bool any_candidates = false;
  bool full = model.trace_ != nullptr || !executor.CanReadCandidates();
  std::vector<std::uint32_t> extra;
  for (const std::size_t i : active) {
    const auto& sampler = *cycles[i].sampler;
    if (!cycles[i].sampled && sampler.config().can_use_unmodified_argmax()) {
      any_argmax = true;
    } else if (CandidateSampling(sampler)) {
      sample[i] = 1;
      any_candidates = true;
      AddPenalizable(sampler, cycles[i].rows, cycles[i].siblings, &extra);
    } else {
      full = true;
    }
  }
  SortUnique(&extra);
  full = full || extra.size() > rocm::kMaxCandidateExtra;
  if (full) {
    any_argmax = false;
    any_candidates = false;
  }
  std::span<const float> logits;
  std::span<const std::uint32_t> argmax;
  try {
    std::vector<std::uint32_t> logit_rows(tokens.size());
    for (std::uint32_t i = 0; i < logit_rows.size(); ++i) {
      logit_rows[i] = i;
    }
    executor.Forward(segments, tokens, logit_rows);
    rocm::Executor::Candidates candidates;
    if (full) {
      logits = executor.ReadLogits(tokens.size());
    }
    if (any_argmax) {
      argmax = executor.ReadArgmax(tokens.size());
    }
    if (any_candidates) {
      candidates = executor.ReadCandidates(tokens.size(), extra);
    }
    if (model.trace_) {
      const std::int64_t verified = CycleTrace::Now();
      for (const std::size_t i : active) {
        cycles[i].trace->verified_ns = verified;
      }
    }
    const std::size_t vocab = model.VocabSize();
    // Rows a candidate cycle needs in full are copied once for the batch.
    std::span<const float> fetched;
    const auto fetch = [&] {
      if (fetched.empty()) {
        fetched = executor.ReadLogits(tokens.size());
      }
      return fetched;
    };
    std::uint32_t row0 = 0;
    for (const std::size_t i : active) {
      const std::size_t count = cycles[i].rows.size();
      VerifyCandidates view{.rows = &candidates,
                            .first = row0,
                            .extra = extra,
                            .full = [&fetch, row0, count, vocab] {
                              return fetch().subspan(row0 * vocab,
                                                     count * vocab);
                            }};
      decodes[i].session->FinishCycle(
          cycles[i],
          full ? logits.subspan(row0 * vocab, count * vocab)
               : std::span<const float>{},
          !full && sample[i] == 0 ? argmax.subspan(row0, count)
                                  : std::span<const std::uint32_t>{},
          row0, !full && sample[i] != 0 ? &view : nullptr);
      row0 += static_cast<std::uint32_t>(count);
    }
  } catch (const std::exception& e) {
    for (const std::size_t i : active) {
      decodes[i].session->Reset();
      decodes[i].error = e.what();
    }
    return false;
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
  std::span<const float> logits;
  try {
    executor.Forward(segments, rows, logit_rows);
    for (std::uint32_t i = 0; i < sessions.size(); ++i) {
      executor.CommitHidden(*sessions[i]->cache_, i);
    }
    logits = executor.ReadLogits(rows.size());
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
    s.logits_on_device_ = false;
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
bool CopyOnStream(rocm::Executor& executor, FifoMutex& mutex,
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

/// Queues `runs` on the executor's stream holding `mutex` only to enqueue
/// them, then waits outside it: a decoding scheduler takes the model between
/// steps, and holding it through the copy kept those steps waiting on this
/// thread. Later work on the stream runs after the copy.
bool CopyBehindStream(rocm::Executor& executor, FifoMutex& mutex,
                      std::span<const rocm::CopyRun> runs) {
  hipEvent_t done = nullptr;
  if (hipEventCreateWithFlags(&done, hipEventDisableTiming) != hipSuccess) {
    return false;
  }
  bool queued = true;
  {
    std::lock_guard lock(mutex);
    try {
      executor.CopyRuns(runs);
    } catch (const std::exception&) {
      queued = false;
    }
    queued = hipEventRecord(done, executor.stream()) == hipSuccess && queued;
  }
  const bool copied = queued && hipEventSynchronize(done) == hipSuccess;
  (void)hipEventDestroy(done);
  return copied;
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
  std::memcpy(tail.data(), Logits().data(), tail.size());
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
  if (!CopyBehindStream(executor, model_->mutex_, runs)) {
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
  draft_vocab_.Reset();
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
  logits_on_device_ = false;
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
  logits_on_device_ = false;
  valid_ = true;
  return true;
}
}  // namespace gufo::models::gemma4
