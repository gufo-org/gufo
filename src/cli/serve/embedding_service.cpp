#include "src/cli/serve/embedding_service.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/library.hpp"
#include "src/models/embedding_gemma2/audio_decoder.hpp"
#include "src/models/embedding_gemma2/audio_processor.hpp"
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
    (void)models::embedding_gemma2::ValidateAudioProcessor(
        options.model_root.string(), &audio_processor_error);
    (void)models::embedding_gemma2::ValidateVideoProcessor(
        options.model_root.string(), &video_processor_error);
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
    std::size_t audio_count = 0;
    std::size_t video_count = 0;
    std::size_t video_frames = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (is_cancelled && is_cancelled())
        return EmbeddingStatus::kCancelled;
      EncoderSequence sequence;
      sequence.is_cancelled = is_cancelled;
      std::string text;
      // Use the tokenizer once for the complete input, preserving BPE merges
      // across adjacent text parts and the reference BOS/EOS framing.
      for (const auto& part : inputs[index].content) {
        if (is_cancelled && is_cancelled())
          return EmbeddingStatus::kCancelled;
        if (part.type == EmbeddingPart::Type::kText) {
          text += part.value;
        } else if (part.type == EmbeddingPart::Type::kVideo) {
          if (!encoder->config().supported_video ||
              !video_processor_error.empty()) {
            if (error)
              *error = video_processor_error.empty()
                           ? "unsupported embedding vision configuration"
                           : video_processor_error;
            return EmbeddingStatus::kInvalidVideoInput;
          }
          try {
            if (++video_count > EmbeddingService::kMaxVideos)
              throw std::invalid_argument(
                  "embedding requests accept at most 32 videos");
            const auto bytes = core::ReadVideoUrl(part.value, budget);
            auto frames =
                ProcessVideo(bytes, request.video, is_cancelled,
                             EmbeddingService::kMaxVideoFrames - video_frames);
            video_frames += frames.size();
            if (video_frames > EmbeddingService::kMaxVideoFrames)
              throw std::invalid_argument(
                  "embedding requests accept at most 64 sampled video frames");
            text += FrameVideo(frames, request.video);
            for (auto& frame : frames)
              sequence.images.push_back(
                  {0, std::move(frame.patches), kVideoTokenId});
          } catch (const platform::LibraryError& failure) {
            if (error)
              *error = failure.what();
            return EmbeddingStatus::kFailed;
          } catch (const std::exception& failure) {
            if (is_cancelled && is_cancelled())
              return EmbeddingStatus::kCancelled;
            if (error)
              *error = failure.what();
            return EmbeddingStatus::kInvalidVideoInput;
          }
        } else if (part.type == EmbeddingPart::Type::kAudio) {
          if (!encoder->config().supported_audio ||
              !audio_processor_error.empty()) {
            if (error)
              *error = audio_processor_error.empty()
                           ? "unsupported embedding audio configuration"
                           : audio_processor_error;
            return EmbeddingStatus::kInvalidAudioInput;
          }
          if (++audio_count > EmbeddingService::kMaxAudio) {
            if (error)
              *error = "embedding requests accept at most 32 audio clips";
            return EmbeddingStatus::kInvalidAudioInput;
          }
          try {
            const auto bytes = core::ReadAudioUrl(part.value, budget);
            const auto waveform = DecodeAudio(bytes, is_cancelled);
            auto features = ProcessAudio(waveform, 0, is_cancelled);
            if (!features.soft_tokens()) {
              if (error)
                *error = "embedding audio needs at least 161 samples at 16 kHz";
              return EmbeddingStatus::kInvalidAudioInput;
            }
            text += kBeginAudioToken;
            for (std::size_t j = 0; j < features.soft_tokens(); ++j)
              text += kAudioToken;
            text += kEndAudioToken;
            sequence.audio.push_back({0, std::move(features)});
          } catch (const std::exception& failure) {
            if (is_cancelled && is_cancelled())
              return EmbeddingStatus::kCancelled;
            if (error)
              *error = failure.what();
            return EmbeddingStatus::kInvalidAudioInput;
          }
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
            auto patches = ProcessImage(core::DecodeImage(bytes),
                                        request.image_soft_tokens);
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
      if (request.truncate &&
          (!sequence.images.empty() || !sequence.audio.empty()) &&
          text.size() / 2 > byte_limit) {
        const auto image_end = text.rfind(kEndImageToken);
        const auto audio_end = text.rfind(kEndAudioToken);
        const std::size_t end = image_end == std::string::npos ? audio_end
                                : audio_end == std::string::npos
                                    ? image_end
                                    : std::max(image_end, audio_end);
        if (end == std::string::npos ||
            end + std::string_view(kEndImageToken).size() > byte_limit) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
        text.resize(2 * byte_limit);
      }
      sequence.tokens =
          request.truncate && sequence.images.empty() && sequence.audio.empty()
              ? tokenizer->EncodeTruncated(text,
                                           EmbeddingService::kMaxInputTokens)
              : tokenizer->Encode(text);
      if (!sequence.images.empty()) {
        for (const auto token : {kImageTokenId, kVideoTokenId}) {
          std::size_t expected = 0;
          for (const auto& image : sequence.images)
            if (image.token_id == token)
              expected += image.patches.soft_tokens();
          if (static_cast<std::size_t>(
                  std::ranges::count(sequence.tokens, token)) != expected) {
            if (error)
              *error =
                  "text must not contain image/video placeholders when vision "
                  "inputs are supplied";
            return token == kVideoTokenId ? EmbeddingStatus::kInvalidVideoInput
                                          : EmbeddingStatus::kInvalidInput;
          }
        }
      }
      if (!sequence.audio.empty()) {
        std::size_t expected = 0;
        for (const auto& clip : sequence.audio)
          expected += clip.features.soft_tokens();
        if (static_cast<std::size_t>(std::ranges::count(
                sequence.tokens, kAudioTokenId)) != expected) {
          if (error)
            *error =
                "text must not contain audio placeholders when audio is "
                "supplied";
          return EmbeddingStatus::kInvalidAudioInput;
        }
      }
      if (request.truncate &&
          sequence.tokens.size() > EmbeddingService::kMaxInputTokens) {
        const auto last_video = std::ranges::find(
            sequence.tokens.rbegin(), sequence.tokens.rend(), kVideoTokenId);
        if (last_video != sequence.tokens.rend() &&
            static_cast<std::size_t>(sequence.tokens.rend() - last_video) + 1 >
                EmbeddingService::kMaxInputTokens - 1) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
        const auto last = std::ranges::find(
            sequence.tokens.rbegin(), sequence.tokens.rend(), kAudioTokenId);
        if (!sequence.audio.empty() && last != sequence.tokens.rend() &&
            static_cast<std::size_t>(sequence.tokens.rend() - last) + 1 >
                EmbeddingService::kMaxInputTokens - 1) {
          batch->failed_index = index;
          return EmbeddingStatus::kInputTooLong;
        }
        (void)TruncateKeepingImages(
            &sequence.tokens, EmbeddingService::kMaxInputTokens,
            sequence.images.empty() ? kAudioTokenId : kImageTokenId);
      }
      if (sequence.tokens.size() > EmbeddingService::kMaxInputTokens) {
        batch->failed_index = index;
        return EmbeddingStatus::kInputTooLong;
      }
      std::size_t cursor = 0;
      for (auto& image : sequence.images) {
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != image.token_id)
          ++cursor;
        image.offset = cursor;
        cursor += image.patches.soft_tokens();
      }
      cursor = 0;
      for (auto& clip : sequence.audio) {
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != kAudioTokenId)
          ++cursor;
        clip.offset = cursor;
        cursor += clip.features.soft_tokens();
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
  std::string audio_processor_error;
  std::string video_processor_error;
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

bool LoadSentenceTransformerPrompts(const std::filesystem::path& model_root,
                                    std::map<std::string, std::string>* prompts,
                                    std::string* error) {
  prompts->clear();
  const auto path = model_root / "config_sentence_transformers.json";
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return true;
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  try {
    const json::Value root = json::parse(text);
    const json::Value* table =
        root.is_object() ? root.find("prompts") : nullptr;
    if (table == nullptr || table->is_null()) {
      return true;
    }
    if (!table->is_object()) {
      *error = path.string() + ": 'prompts' must be an object";
      return false;
    }
    for (const auto& [name, value] : table->members()) {
      if (!value.is_string()) {
        *error = path.string() + ": prompt '" + name + "' must be a string";
        return false;
      }
      prompts->emplace(name, value.str());
    }
  } catch (const std::exception& parse_error) {
    *error = path.string() + ": " + parse_error.what();
    return false;
  }
  return true;
}

EmbeddingService::EmbeddingService(EmbeddingServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& name = impl_->options.default_prompt_name;
  if (impl_->ready && !name.empty() && !impl_->options.prompts.contains(name)) {
    impl_->ready = false;
    impl_->initialization_error =
        "--default-prompt-name '" + name + "' is not a prompt of this model";
  }
}

const std::string* EmbeddingService::prompt(const std::string& name) const {
  const auto it = impl_->options.prompts.find(name);
  return it == impl_->options.prompts.end() ? nullptr : &it->second;
}

const std::string& EmbeddingService::default_prompt_name() const {
  return impl_->options.default_prompt_name;
}

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
