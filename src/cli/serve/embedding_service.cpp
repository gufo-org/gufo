#include "src/cli/serve/embedding_service.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
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
  // Tokenizes before joining the encoder's queue, so a large request does not
  // block other requests that are ready to run.
  EmbeddingStatus RunNative(const std::vector<EmbeddingInput>& inputs,
                            const EmbeddingRequestOptions& request,
                            const std::function<bool()>& is_cancelled,
                            EmbeddingBatch* batch, std::string* error) {
    using namespace models::embedding_gemma2;
    std::vector<EncoderSequence> sequences;
    sequences.reserve(inputs.size());
    core::ImageReadBudget budget;
    const std::size_t byte_limit =
        EmbeddingService::kMaxInputTokens * tokenizer->max_token_bytes();
    std::size_t image_count = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (is_cancelled && is_cancelled())
        return EmbeddingStatus::kCancelled;
      EncoderSequence sequence;
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
        if (!request.truncate && text.size() > byte_limit) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
      }
      // A text-only input reads just the bytes its kept tokens can span. An
      // input with images keeps every placeholder: one ending past the bytes
      // the kept tokens can span cannot fit, and text after the last image
      // is bounded like a text-only input.
      if (request.truncate && !sequence.images.empty() &&
          text.size() / 2 > byte_limit) {
        const std::size_t end = text.rfind(kEndImageToken);
        if (end == std::string::npos ||
            end + std::string_view(kEndImageToken).size() > byte_limit) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
        text.resize(2 * byte_limit);
      }
      sequence.tokens = request.truncate && sequence.images.empty()
                            ? tokenizer->EncodeTruncated(
                                  text, EmbeddingService::kMaxInputTokens)
                            : tokenizer->Encode(text);
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
      if (request.truncate &&
          sequence.tokens.size() > EmbeddingService::kMaxInputTokens) {
        (void)TruncateKeepingImages(
            &sequence.tokens, EmbeddingService::kMaxInputTokens, kImageTokenId);
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

bool TruncateKeepingImages(std::vector<std::uint32_t>* tokens,
                           std::size_t max_tokens, std::uint32_t image_token) {
  if (tokens->size() <= max_tokens) {
    return true;
  }
  if (max_tokens < 2) {
    return false;
  }
  // Keep the first tokens and the final <eos>. The last image placeholder and
  // the end-of-image marker after it must both stay.
  const std::size_t keep = max_tokens - 1;
  const auto last_image =
      std::ranges::find(tokens->rbegin(), tokens->rend(), image_token);
  const std::size_t image_end =
      last_image == tokens->rend()
          ? 0
          : static_cast<std::size_t>(tokens->rend() - last_image) + 1;
  if (image_end > keep) {
    return false;
  }
  tokens->erase(tokens->begin() + static_cast<std::ptrdiff_t>(keep),
                tokens->end() - 1);
  return true;
}

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
    std::string* error, const EmbeddingRequestOptions& request) {
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
    status = impl_->options.runner(inputs, request, batch, error);
  } else {
#if defined(ENGINE_ENABLE_HIP)
    status = impl_->RunNative(inputs, request, is_cancelled, batch, error);
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
