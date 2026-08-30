#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/models/qwen3_asr/audio.hpp"
#include "src/models/qwen3_asr/config.hpp"
#include "src/models/qwen3_asr/hip/audio_encoder_runtime.hpp"
#include "src/models/qwen3_asr/hip/text_decoder_runtime.hpp"
#include "src/models/qwen3_asr/prompt.hpp"
#include "src/models/qwen3_asr/tokenizer.hpp"

namespace gufo::models::qwen3_asr::hip {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string Lower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

std::optional<std::string> NormalizeLanguage(
    const std::optional<std::string>& requested,
    const std::vector<std::string>& supported) {
  if (!requested.has_value() || requested->empty() ||
      Lower(*requested) == "auto") {
    return std::nullopt;
  }
  const std::string lower = Lower(*requested);
  static constexpr std::pair<std::string_view, std::string_view> kIsoNames[] = {
      {"zh", "Chinese"},   {"en", "English"},    {"yue", "Cantonese"},
      {"ar", "Arabic"},    {"de", "German"},     {"fr", "French"},
      {"es", "Spanish"},   {"pt", "Portuguese"}, {"id", "Indonesian"},
      {"it", "Italian"},   {"ko", "Korean"},     {"ru", "Russian"},
      {"th", "Thai"},      {"vi", "Vietnamese"}, {"ja", "Japanese"},
      {"tr", "Turkish"},   {"hi", "Hindi"},      {"ms", "Malay"},
      {"nl", "Dutch"},     {"sv", "Swedish"},    {"da", "Danish"},
      {"fi", "Finnish"},   {"pl", "Polish"},     {"cs", "Czech"},
      {"fil", "Filipino"}, {"fa", "Persian"},    {"el", "Greek"},
      {"ro", "Romanian"},  {"hu", "Hungarian"},  {"mk", "Macedonian"},
  };
  for (const auto& [code, name] : kIsoNames) {
    if (lower == code) {
      return std::string(name);
    }
  }
  for (const std::string& language : supported) {
    if (lower == Lower(language)) {
      return language;
    }
  }
  throw std::invalid_argument("Qwen3-ASR language is unsupported: " +
                              *requested);
}

void CheckCancellation(const CancellationCheck& is_cancelled) {
  if (is_cancelled && is_cancelled()) {
    throw std::runtime_error("Qwen3-ASR transcription cancelled");
  }
}

}  // namespace

struct TranscriptionHipRuntime::Impl {
  Impl(std::string model_root, std::size_t token_capacity) {
    if (token_capacity < 32U) {
      throw std::invalid_argument(
          "Qwen3-ASR context capacity must be at least 32 tokens");
    }
    const std::optional<ModelConfig> loaded_config =
        LoadModelConfigFromPath(model_root);
    if (!loaded_config.has_value() || !IsSupportedModelConfig(*loaded_config)) {
      throw std::invalid_argument(
          "Qwen3-ASR runtime requires the supported 1.7B checkpoint");
    }
    config = *loaded_config;
    root = std::move(model_root);
    maximum_context_tokens = token_capacity;

    std::string load_error;
    if (!Tokenizer::Load(root, &tokenizer, &load_error)) {
      throw std::runtime_error(load_error);
    }
    audio_encoder = AudioEncoderHipRuntime::Create(root, &load_error);
    if (audio_encoder == nullptr) {
      throw std::runtime_error(load_error);
    }
    text_decoder =
        TextDecoderHipRuntime::Create(root, token_capacity, &load_error);
    if (text_decoder == nullptr) {
      throw std::runtime_error(load_error);
    }
  }

  bool Transcribe(const TranscriptionRequest& request,
                  const CancellationCheck& is_cancelled,
                  TranscriptionResult* result, std::string* error) {
    if (result == nullptr) {
      SetError(error, "Qwen3-ASR transcription result must not be null");
      return false;
    }
    *result = {};
    const Clock::time_point total_begin = Clock::now();
    try {
      if (request.wav.empty()) {
        throw std::invalid_argument("Qwen3-ASR WAV input must not be empty");
      }
      if (request.max_new_tokens == 0U) {
        throw std::invalid_argument(
            "Qwen3-ASR max_new_tokens must be positive");
      }
      const std::optional<std::string> language =
          NormalizeLanguage(request.language, config.supported_languages);
      CheckCancellation(is_cancelled);

      AudioBuffer decoded_audio;
      const Clock::time_point decode_begin = Clock::now();
      std::string phase_error;
      if (!DecodeWav(request.wav, &decoded_audio, &phase_error)) {
        throw std::invalid_argument(phase_error);
      }
      std::vector<float> waveform = ResampleMono16k(decoded_audio);
      const Clock::time_point decode_end = Clock::now();
      result->timings.decode_audio_ms = Milliseconds(decode_begin, decode_end);
      result->audio_samples = waveform.size();
      CheckCancellation(is_cancelled);

      const Clock::time_point features_begin = Clock::now();
      LogMelFeatures features = ComputeLogMelFeatures(waveform);
      const Clock::time_point features_end = Clock::now();
      result->timings.feature_extraction_ms =
          Milliseconds(features_begin, features_end);
      result->mel_frames = features.frames;
      CheckCancellation(is_cancelled);

      const std::size_t expected_audio_tokens =
          AudioEmbeddingTokenCount(features.frames);
      std::vector<std::uint32_t> prompt = BuildPrompt(
          tokenizer, PromptOptions{
                         .context = request.context,
                         .language = language,
                         .audio_embedding_tokens = expected_audio_tokens,
                     });
      result->prompt_tokens = prompt.size();
      if (prompt.size() + request.max_new_tokens > maximum_context_tokens) {
        throw std::length_error(
            "Qwen3-ASR audio, context, and requested output exceed the "
            "configured context capacity");
      }

      const Clock::time_point encoder_begin = Clock::now();
      AudioEncoderTrace encoded;
      if (!audio_encoder->Encode(features.values, features.frames, &encoded,
                                 &phase_error)) {
        throw std::runtime_error(phase_error);
      }
      const Clock::time_point encoder_end = Clock::now();
      result->timings.audio_encoder_ms =
          Milliseconds(encoder_begin, encoder_end);
      result->audio_tokens = encoded.final.tokens;
      if (encoded.final.tokens != expected_audio_tokens) {
        throw std::runtime_error(
            "Qwen3-ASR audio frontend token count is inconsistent");
      }
      const auto finite = [](const AudioEncoderOutput& output) {
        return std::ranges::all_of(
            output.values, [](float value) { return std::isfinite(value); });
      };
      if (!finite(encoded.frontend)) {
        throw std::runtime_error(
            "Qwen3-ASR audio frontend produced non-finite embeddings");
      }
      if (!finite(encoded.layer0)) {
        throw std::runtime_error(
            "Qwen3-ASR audio encoder layer 0 produced non-finite embeddings");
      }
      if (!finite(encoded.final)) {
        throw std::runtime_error(
            "Qwen3-ASR audio encoder produced non-finite embeddings");
      }
      CheckCancellation(is_cancelled);

      const Clock::time_point decoder_begin = Clock::now();
      if (!text_decoder->Generate(prompt, encoded.final.values,
                                  encoded.final.tokens, request.max_new_tokens,
                                  &result->generated_ids, &phase_error)) {
        throw std::runtime_error(phase_error);
      }
      const Clock::time_point decoder_end = Clock::now();
      result->timings.text_decoder_ms =
          Milliseconds(decoder_begin, decoder_end);
      CheckCancellation(is_cancelled);

      result->decoded = tokenizer.Decode(result->generated_ids, true);
      Transcription parsed = ParseTranscription(result->decoded, language);
      result->language = std::move(parsed.language);
      result->text = std::move(parsed.text);
      result->sample_rate = kAudioSampleRate;
      result->timings.total_ms = Milliseconds(total_begin, Clock::now());
      return true;
    } catch (const std::exception& exception) {
      *result = {};
      SetError(error, exception.what());
      return false;
    }
  }

  ModelConfig config;
  std::string root;
  std::size_t maximum_context_tokens{0};
  Tokenizer tokenizer;
  std::unique_ptr<AudioEncoderHipRuntime> audio_encoder;
  std::unique_ptr<TextDecoderHipRuntime> text_decoder;
};

TranscriptionHipRuntime::TranscriptionHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TranscriptionHipRuntime::~TranscriptionHipRuntime() = default;

std::unique_ptr<TranscriptionHipRuntime> TranscriptionHipRuntime::Create(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  try {
    return std::unique_ptr<TranscriptionHipRuntime>(new TranscriptionHipRuntime(
        std::make_unique<Impl>(model_root, maximum_context_tokens)));
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return nullptr;
  }
}

bool TranscriptionHipRuntime::Transcribe(const TranscriptionRequest& request,
                                         const CancellationCheck& is_cancelled,
                                         TranscriptionResult* result,
                                         std::string* error) {
  return impl_->Transcribe(request, is_cancelled, result, error);
}

}  // namespace gufo::models::qwen3_asr::hip

#else

namespace gufo::models::qwen3_asr::hip {

struct TranscriptionHipRuntime::Impl {};

TranscriptionHipRuntime::TranscriptionHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TranscriptionHipRuntime::~TranscriptionHipRuntime() = default;

std::unique_ptr<TranscriptionHipRuntime> TranscriptionHipRuntime::Create(
    const std::string&, std::size_t, std::string* error) {
  if (error != nullptr) {
    *error = "Qwen3-ASR native transcription requires a HIP build";
  }
  return nullptr;
}

bool TranscriptionHipRuntime::Transcribe(const TranscriptionRequest&,
                                         const CancellationCheck&,
                                         TranscriptionResult* result,
                                         std::string* error) {
  if (result != nullptr) {
    *result = {};
  }
  if (error != nullptr) {
    *error = "Qwen3-ASR native transcription requires a HIP build";
  }
  return false;
}

}  // namespace gufo::models::qwen3_asr::hip

#endif
