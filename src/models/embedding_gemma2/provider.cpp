#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <gufo/ext/embeddings.hpp>
#include <gufo/ext/generated/embedding_gemma2_constants.hpp>
#include <gufo/ext/image.hpp>
#include <gufo/ext/platform/library.hpp>
#include <gufo/ext/registry.hpp>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "audio_decoder.hpp"
#include "audio_processor.hpp"
#include "config.hpp"
#include "hip/encoder_runtime.hpp"
#include "image_processor.hpp"
#include "loader.hpp"
#include "probe.hpp"
#include "tokenizer.hpp"
#include "video_processor.hpp"

namespace gufo::models::embedding_gemma2 {
namespace {

bool TruncateKeepingImages(std::vector<std::uint32_t>* tokens,
                           std::size_t max_tokens, std::uint32_t image_token) {
  if (tokens->size() <= max_tokens) {
    return true;
  }
  if (max_tokens < 2) {
    return false;
  }
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

#if defined(ENGINE_ENABLE_HIP)
class EmbeddingGemma2Runner : public ext::EmbeddingRunner {
public:
  EmbeddingGemma2Runner(Tokenizer tokenizer_in,
                        std::unique_ptr<hip::EncoderRuntime> encoder_in,
                        std::string image_err, std::string audio_err,
                        std::string video_err)
      : tokenizer_(std::move(tokenizer_in)),
        encoder_(std::move(encoder_in)),
        image_processor_error_(std::move(image_err)),
        audio_processor_error_(std::move(audio_err)),
        video_processor_error_(std::move(video_err)) {}

  ext::EmbeddingStatus EmbedInputs(
      const std::vector<ext::EmbeddingInput>& inputs,
      const std::function<bool()>& is_cancelled, ext::EmbeddingBatch* batch,
      std::string* error = nullptr,
      const ext::EmbeddingRequestOptions& request = {}) override {
    std::vector<EncoderSequence> sequences;
    sequences.reserve(inputs.size());
    core::ImageReadBudget budget;
    const std::size_t byte_limit =
        ext::kMaxInputTokens * tokenizer_.max_token_bytes();
    std::size_t image_count = 0;
    std::size_t audio_count = 0;
    std::size_t video_count = 0;
    std::size_t video_frames = 0;

    for (std::size_t index = 0; index < inputs.size(); ++index) {
      if (is_cancelled && is_cancelled()) {
        return ext::EmbeddingStatus::kCancelled;
      }
      EncoderSequence sequence;
      sequence.is_cancelled = is_cancelled;
      std::string text;
      for (const auto& part : inputs[index].content) {
        if (is_cancelled && is_cancelled()) {
          return ext::EmbeddingStatus::kCancelled;
        }
        if (part.type == ext::EmbeddingPart::Type::kText) {
          text += part.value;
        } else if (part.type == ext::EmbeddingPart::Type::kVideo) {
          if (!encoder_->config().supported_video ||
              !video_processor_error_.empty()) {
            if (error) {
              *error = video_processor_error_.empty()
                           ? "unsupported embedding vision configuration"
                           : video_processor_error_;
            }
            return ext::EmbeddingStatus::kInvalidVideoInput;
          }
          try {
            if (++video_count > ext::kMaxVideos) {
              throw std::invalid_argument(
                  "embedding requests accept at most 32 videos");
            }
            const auto bytes = core::ReadVideoUrl(part.value, budget);
            VideoOptions video_opts{
                .fps = request.video.fps,
                .max_frames = request.video.max_frames,
                .timestamps = request.video.timestamps,
                .truncate_overflow = request.video.truncate_overflow,
            };
            auto frames = ProcessVideo(bytes, video_opts, is_cancelled,
                                       ext::kMaxVideoFrames - video_frames);
            video_frames += frames.size();
            if (video_frames > ext::kMaxVideoFrames) {
              throw std::invalid_argument(
                  "embedding requests accept at most 64 sampled video frames");
            }
            text += FrameVideo(frames, video_opts);
            for (auto& frame : frames) {
              sequence.images.push_back(
                  {0, std::move(frame.patches), kVideoTokenId});
            }
          } catch (const platform::LibraryError& failure) {
            if (error) {
              *error = failure.what();
            }
            return ext::EmbeddingStatus::kFailed;
          } catch (const std::exception& failure) {
            if (is_cancelled && is_cancelled()) {
              return ext::EmbeddingStatus::kCancelled;
            }
            if (error) {
              *error = failure.what();
            }
            return ext::EmbeddingStatus::kInvalidVideoInput;
          }
        } else if (part.type == ext::EmbeddingPart::Type::kAudio) {
          if (!encoder_->config().supported_audio ||
              !audio_processor_error_.empty()) {
            if (error) {
              *error = audio_processor_error_.empty()
                           ? "unsupported embedding audio configuration"
                           : audio_processor_error_;
            }
            return ext::EmbeddingStatus::kInvalidAudioInput;
          }
          if (++audio_count > ext::kMaxAudio) {
            if (error) {
              *error = "embedding requests accept at most 32 audio clips";
            }
            return ext::EmbeddingStatus::kInvalidAudioInput;
          }
          try {
            const auto bytes = core::ReadAudioUrl(part.value, budget);
            const auto waveform = DecodeAudio(bytes, is_cancelled);
            auto features = ProcessAudio(waveform, 0, is_cancelled);
            if (!features.soft_tokens()) {
              if (error) {
                *error = "embedding audio needs at least 161 samples at 16 kHz";
              }
              return ext::EmbeddingStatus::kInvalidAudioInput;
            }
            text += kBeginAudioToken;
            for (std::size_t j = 0; j < features.soft_tokens(); ++j) {
              text += kAudioToken;
            }
            text += kEndAudioToken;
            sequence.audio.push_back({0, std::move(features)});
          } catch (const std::exception& failure) {
            if (is_cancelled && is_cancelled()) {
              return ext::EmbeddingStatus::kCancelled;
            }
            if (error) {
              *error = failure.what();
            }
            return ext::EmbeddingStatus::kInvalidAudioInput;
          }
        } else {
          if (!encoder_->config().supported_vision) {
            if (error) {
              *error = "unsupported embedding vision configuration";
            }
            return ext::EmbeddingStatus::kInvalidInput;
          }
          if (!image_processor_error_.empty()) {
            if (error) {
              *error = image_processor_error_;
            }
            return ext::EmbeddingStatus::kInvalidInput;
          }
          if (++image_count > ext::kMaxImages) {
            if (error) {
              *error = "embedding requests accept at most 32 images";
            }
            return ext::EmbeddingStatus::kInvalidInput;
          }
          try {
            const auto bytes = core::ReadImageUrl(part.value, budget);
            auto patches = ProcessImage(core::DecodeImage(bytes),
                                        request.image_soft_tokens);
            text += kBeginImageToken;
            for (std::size_t j = 0; j < patches.soft_tokens(); ++j) {
              text += kImageToken;
            }
            text += kEndImageToken;
            sequence.images.push_back({0, std::move(patches)});
          } catch (const std::exception& failure) {
            if (error) {
              *error = failure.what();
            }
            return ext::EmbeddingStatus::kInvalidInput;
          }
        }
        if (!request.truncate && text.size() > byte_limit) {
          batch->failed_index = index;
          return ext::EmbeddingStatus::kInputTooLong;
        }
      }
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
          return ext::EmbeddingStatus::kInputTooLong;
        }
        text.resize(2 * byte_limit);
      }
      sequence.tokens =
          request.truncate && sequence.images.empty() && sequence.audio.empty()
              ? tokenizer_.EncodeTruncated(text, ext::kMaxInputTokens)
              : tokenizer_.Encode(text);
      if (!sequence.images.empty()) {
        for (const auto token : {kImageTokenId, kVideoTokenId}) {
          std::size_t expected = 0;
          for (const auto& image : sequence.images) {
            if (image.token_id == token) {
              expected += image.patches.soft_tokens();
            }
          }
          if (static_cast<std::size_t>(
                  std::ranges::count(sequence.tokens, token)) != expected) {
            if (error) {
              *error =
                  "text must not contain image/video placeholders when vision "
                  "inputs are supplied";
            }
            return token == kVideoTokenId
                       ? ext::EmbeddingStatus::kInvalidVideoInput
                       : ext::EmbeddingStatus::kInvalidInput;
          }
        }
      }
      if (!sequence.audio.empty()) {
        std::size_t expected = 0;
        for (const auto& clip : sequence.audio) {
          expected += clip.features.soft_tokens();
        }
        if (static_cast<std::size_t>(std::ranges::count(
                sequence.tokens, kAudioTokenId)) != expected) {
          if (error) {
            *error =
                "text must not contain audio placeholders when audio is "
                "supplied";
          }
          return ext::EmbeddingStatus::kInvalidAudioInput;
        }
      }
      if (request.truncate && sequence.tokens.size() > ext::kMaxInputTokens) {
        const auto last_video = std::ranges::find(
            sequence.tokens.rbegin(), sequence.tokens.rend(), kVideoTokenId);
        if (last_video != sequence.tokens.rend() &&
            static_cast<std::size_t>(sequence.tokens.rend() - last_video) + 1 >
                ext::kMaxInputTokens - 1) {
          batch->failed_index = index;
          return ext::EmbeddingStatus::kInputTooLong;
        }
        const auto last = std::ranges::find(
            sequence.tokens.rbegin(), sequence.tokens.rend(), kAudioTokenId);
        if (!sequence.audio.empty() && last != sequence.tokens.rend() &&
            static_cast<std::size_t>(sequence.tokens.rend() - last) + 1 >
                ext::kMaxInputTokens - 1) {
          batch->failed_index = index;
          return ext::EmbeddingStatus::kInputTooLong;
        }
        (void)TruncateKeepingImages(
            &sequence.tokens, ext::kMaxInputTokens,
            sequence.images.empty() ? kAudioTokenId : kImageTokenId);
      }
      if (sequence.tokens.size() > ext::kMaxInputTokens) {
        batch->failed_index = index;
        return ext::EmbeddingStatus::kInputTooLong;
      }
      std::size_t cursor = 0;
      for (auto& image : sequence.images) {
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != image.token_id) {
          ++cursor;
        }
        image.offset = cursor;
        cursor += image.patches.soft_tokens();
      }
      cursor = 0;
      for (auto& clip : sequence.audio) {
        while (cursor < sequence.tokens.size() &&
               sequence.tokens[cursor] != kAudioTokenId) {
          ++cursor;
        }
        clip.offset = cursor;
        cursor += clip.features.soft_tokens();
      }
      batch->token_counts.push_back(sequence.tokens.size());
      sequences.push_back(std::move(sequence));
    }
    if (!encoder_->EmbedMultimodal(sequences, &batch->embeddings, error,
                                   is_cancelled)) {
      if (encoder_->device_lost()) {
        device_lost_.store(true);
        return ext::EmbeddingStatus::kDeviceLost;
      }
      if (is_cancelled && is_cancelled()) {
        return ext::EmbeddingStatus::kCancelled;
      }
      return ext::EmbeddingStatus::kFailed;
    }
    return ext::EmbeddingStatus::kOk;
  }

  bool device_lost() const noexcept override {
    return device_lost_.load() || (encoder_ && encoder_->device_lost());
  }

private:
  Tokenizer tokenizer_;
  std::unique_ptr<hip::EncoderRuntime> encoder_;
  std::string image_processor_error_;
  std::string audio_processor_error_;
  std::string video_processor_error_;
  std::atomic<bool> device_lost_{false};
};
#endif

}  // namespace

#if defined(ENGINE_ENABLE_HIP)
std::unique_ptr<ext::EmbeddingRunner> CreateEmbeddingGemma2Runner(
    const std::string& model_root, std::string* error) {
  if (!ProbeEmbeddingGemma2(model_root)) {
    if (error) {
      *error = "embedding service requires an EmbeddingGemma 2 model directory";
    }
    return nullptr;
  }
  std::string load_err;
  std::optional<Tokenizer> loaded = Tokenizer::Load(model_root, &load_err);
  if (!loaded.has_value()) {
    if (error) {
      *error = "embedding tokenizer failed to load: " + load_err;
    }
    return nullptr;
  }
  std::string image_processor_error;
  std::string audio_processor_error;
  std::string video_processor_error;
  (void)ValidateImageProcessor(model_root, &image_processor_error);
  (void)ValidateAudioProcessor(model_root, &audio_processor_error);
  (void)ValidateVideoProcessor(model_root, &video_processor_error);

  auto encoder = hip::EncoderRuntime::Create(model_root, &load_err);
  if (!encoder) {
    if (error) {
      *error = "embedding encoder failed to load: " + load_err;
    }
    return nullptr;
  }
  return std::make_unique<EmbeddingGemma2Runner>(
      std::move(*loaded), std::move(encoder), std::move(image_processor_error),
      std::move(audio_processor_error), std::move(video_processor_error));
}
#endif

}  // namespace gufo::models::embedding_gemma2

void embedding_gemma2_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_embedding_gemma2::
                           kProvider_embeddinggemma_2);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::embedding_gemma2;
  registry.AddEmbeddingProvider({
      .id = "embeddinggemma-2",
      .probe = &ProbeEmbeddingGemma2,
      .resolve = &ResolveEmbeddingGemma2Artifacts,
      .create = &CreateEmbeddingGemma2Runner,
  });
#endif
}
