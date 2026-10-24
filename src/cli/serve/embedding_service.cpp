#include "src/cli/serve/embedding_service.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/core/image.hpp"
#include "src/models/embedding_gemma2/image_processor.hpp"
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
    (void)models::embedding_gemma2::ValidateImageProcessor(
        options.model_root.string(), &image_processor_error);
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
  EmbeddingStatus RunNative(const std::vector<EmbeddingInput>& inputs,
                            const std::function<bool()>& is_cancelled,
                            EmbeddingBatch* batch, std::string* error) {
    using namespace models::embedding_gemma2;
    std::vector<hip::EncoderSequence> sequences;
    sequences.reserve(inputs.size());
    core::ImageReadBudget budget;
    const std::size_t byte_limit =
        EmbeddingService::kMaxInputTokens * tokenizer->max_token_bytes();
    std::size_t image_count = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (is_cancelled && is_cancelled())
        return EmbeddingStatus::kCancelled;
      hip::EncoderSequence sequence;
      std::string text;
      // Use the tokenizer once for the complete input, preserving BPE merges
      // across adjacent text parts and the reference BOS/EOS framing.
      for (const auto& part : inputs[index].content) {
        if (is_cancelled && is_cancelled())
          return EmbeddingStatus::kCancelled;
        if (part.type == EmbeddingPart::Type::kText) {
          text += part.value;
        } else {
          if (!encoder->config().supported_vision) {
            if (error)
              *error = "unsupported embedding vision configuration";
            return EmbeddingStatus::kInvalidInput;
          }
          if (!image_processor_error.empty()) {
            if (error)
              *error = image_processor_error;
            return EmbeddingStatus::kInvalidInput;
          }
          if (++image_count > EmbeddingService::kMaxImages) {
            if (error)
              *error = "embedding requests accept at most 32 images";
            return EmbeddingStatus::kInvalidInput;
          }
          try {
            const auto bytes = core::ReadImageUrl(part.value, budget);
            auto patches = ProcessImage(core::DecodeImage(bytes));
            text += kBeginImageToken;
            for (std::size_t j = 0; j < patches.soft_tokens(); ++j)
              text += kImageToken;
            text += kEndImageToken;
            sequence.images.push_back({0, std::move(patches)});
          } catch (const std::exception& failure) {
            if (error)
              *error = failure.what();
            return EmbeddingStatus::kInvalidInput;
          }
        }
        if (text.size() > byte_limit) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
      }
      sequence.tokens = tokenizer->Encode(text);
      if (!sequence.images.empty()) {
        std::size_t expected = 0;
        for (const auto& image : sequence.images)
          expected += image.patches.soft_tokens();
        // Validate after joining and tokenizing: a literal placeholder may
        // straddle adjacent text parts. It must never capture another image's
        // features or leave an unfilled image token in the text backbone.
        if (static_cast<std::size_t>(std::ranges::count(
                sequence.tokens, kImageTokenId)) != expected) {
          if (error)
            *error =
                "text must not contain image placeholders when images are "
                "supplied";
          return EmbeddingStatus::kInvalidInput;
        }
      }
      if (sequence.tokens.size() > EmbeddingService::kMaxInputTokens) {
        batch->failed_index = index;
        return EmbeddingStatus::kInputTooLong;
      }
      std::size_t cursor = 0;
      for (auto& image : sequence.images) {
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != kImageTokenId)
          ++cursor;
        image.offset = cursor;
        cursor += image.patches.soft_tokens();
      }
      batch->token_counts.push_back(sequence.tokens.size());
      sequences.push_back(std::move(sequence));
    }
    if (!encoder->EmbedMultimodal(sequences, &batch->embeddings, error,
                                  is_cancelled)) {
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
  std::string image_processor_error;
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
  std::vector<EmbeddingInput> inputs;
  for (const auto& text : texts)
    inputs.push_back({{{EmbeddingPart::Type::kText, text}}});
  return EmbedInputs(inputs, is_cancelled, batch, error);
}

EmbeddingStatus EmbeddingService::EmbedInputs(
    const std::vector<EmbeddingInput>& inputs,
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
    status = impl_->options.runner(inputs, batch, error);
  } else {
#if defined(ENGINE_ENABLE_HIP)
    status = impl_->RunNative(inputs, is_cancelled, batch, error);
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
