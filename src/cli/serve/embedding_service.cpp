#include "src/cli/serve/embedding_service.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/models/embedding_gemma2/loader.hpp"
#include "src/models/embedding_gemma2/tokenizer.hpp"

namespace gufo::server {

struct EmbeddingService::Impl {
  explicit Impl(EmbeddingServiceOptions supplied)
      : options(std::move(supplied)) {
    if (options.model_id.empty()) {
      options.model_id = "embeddinggemma-2";
    }
    if (options.runner) {
      // An injected runner needs no checkpoint, so the model is not validated
      // unless the caller asks for it.
      if (options.validate_model &&
          !models::embedding_gemma2::LooksLikeEmbeddingGemma2(
              options.model_root.string())) {
        initialization_error = kInvalidModel;
        return;
      }
      ready = true;
      return;
    }
#if defined(ENGINE_ENABLE_HIP)
    if (!models::embedding_gemma2::LooksLikeEmbeddingGemma2(
            options.model_root.string())) {
      initialization_error = kInvalidModel;
      return;
    }
    std::string error;
    std::optional<models::embedding_gemma2::Tokenizer> loaded =
        models::embedding_gemma2::Tokenizer::Load(options.model_root.string(),
                                                  &error);
    if (!loaded.has_value()) {
      initialization_error = "embedding tokenizer failed to load: " + error;
      return;
    }
    tokenizer.emplace(std::move(*loaded));
    encoder = models::embedding_gemma2::hip::EncoderRuntime::Create(
        options.model_root.string(), &error);
    if (encoder == nullptr) {
      initialization_error = "embedding encoder failed to load: " + error;
      return;
    }
    ready = true;
#else
    initialization_error = "embedding service requires the HIP backend";
#endif
  }

#if defined(ENGINE_ENABLE_HIP)
  // Tokenizes before taking any device turn, so a large request does not
  // block other requests that are ready to run. The encoder hands out device
  // turns one batch at a time.
  EmbeddingStatus RunNative(const std::vector<std::string>& texts,
                            const std::function<bool()>& is_cancelled,
                            EmbeddingBatch* batch, std::string* error) {
    std::vector<std::vector<std::uint32_t>> sequences;
    sequences.reserve(texts.size());
    batch->token_counts.clear();
    // Normalization never shortens the text and no token covers more than
    // max_token_bytes() of it, so a longer input cannot fit. Rejecting it here
    // bounds the tokenizing work an oversized input can cause.
    const std::size_t byte_limit =
        EmbeddingService::kMaxInputTokens * tokenizer->max_token_bytes();
    for (std::size_t index = 0; index < texts.size(); ++index) {
      if (texts[index].size() > byte_limit) {
        batch->failed_index = index;
        return EmbeddingStatus::kInputTooLong;
      }
      sequences.push_back(tokenizer->Encode(texts[index]));
      if (sequences.back().size() > EmbeddingService::kMaxInputTokens) {
        batch->failed_index = index;
        return EmbeddingStatus::kInputTooLong;
      }
      batch->token_counts.push_back(sequences.back().size());
    }
    if (!encoder->Embed(sequences, &batch->embeddings, error, is_cancelled)) {
      if (encoder->device_lost()) {
        return EmbeddingStatus::kDeviceLost;
      }
      if (is_cancelled && is_cancelled()) {
        return EmbeddingStatus::kCancelled;
      }
      return EmbeddingStatus::kFailed;
    }
    return EmbeddingStatus::kOk;
  }
#endif

  static constexpr const char* kInvalidModel =
      "embedding service requires an EmbeddingGemma 2 model directory";

  EmbeddingServiceOptions options;
#if defined(ENGINE_ENABLE_HIP)
  std::optional<models::embedding_gemma2::Tokenizer> tokenizer;
  std::unique_ptr<models::embedding_gemma2::hip::EncoderRuntime> encoder;
#endif
  std::string initialization_error;
  // Serializes the injected runner, which has no device turns of its own.
  std::mutex runner_mutex;
  std::atomic<bool> device_lost{false};
  bool ready{false};
};

EmbeddingService::EmbeddingService(EmbeddingServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

EmbeddingService::~EmbeddingService() = default;

bool EmbeddingService::ready() const noexcept {
  return impl_->ready;
}

bool EmbeddingService::device_lost() const noexcept {
  return impl_->device_lost.load(std::memory_order_acquire);
}

std::string EmbeddingService::initialization_error() const {
  return impl_->initialization_error;
}

std::string EmbeddingService::model_id() const {
  return impl_->options.model_id;
}

std::string EmbeddingService::backend_name() const {
  return "native-hip";
}

EmbeddingStatus EmbeddingService::Embed(
    const std::vector<std::string>& texts,
    const std::function<bool()>& is_cancelled, EmbeddingBatch* batch,
    std::string* error) {
  *batch = {};
  if (!impl_->ready) {
    if (error != nullptr) {
      *error = impl_->initialization_error;
    }
    return EmbeddingStatus::kFailed;
  }
  EmbeddingStatus status = EmbeddingStatus::kFailed;
  if (impl_->options.runner) {
    const std::lock_guard lock(impl_->runner_mutex);
    if (is_cancelled && is_cancelled()) {
      if (error != nullptr) {
        *error = "embedding request cancelled";
      }
      return EmbeddingStatus::kCancelled;
    }
    status = impl_->options.runner(texts, batch, error);
  } else {
#if defined(ENGINE_ENABLE_HIP)
    status = impl_->RunNative(texts, is_cancelled, batch, error);
#endif
  }
  if (status == EmbeddingStatus::kDeviceLost &&
      !impl_->device_lost.exchange(true, std::memory_order_acq_rel)) {
    Logger::Error("embedding",
                  "event=device_lost remedy=restart reason=" +
                      (error != nullptr ? *error : std::string("unknown")));
  }
  return status;
}

}  // namespace gufo::server
