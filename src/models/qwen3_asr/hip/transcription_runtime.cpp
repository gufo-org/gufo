#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"

#include <utility>

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/core/cancellable_gate.hpp"
#include "src/core/utf8.hpp"
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

    auto audio_future = std::async(std::launch::async, [model_root = root] {
      std::string error;
      auto runtime = AudioEncoderHipRuntime::Create(model_root, &error);
      return std::pair(std::move(runtime), std::move(error));
    });
    auto text_future =
        std::async(std::launch::async, [model_root = root, token_capacity] {
          std::string error;
          auto runtime =
              TextDecoderHipRuntime::Create(model_root, token_capacity, &error);
          return std::pair(std::move(runtime), std::move(error));
        });

    std::string load_error;
    if (!Tokenizer::Load(root, &tokenizer, &load_error)) {
      throw std::runtime_error(load_error);
    }
    auto [loaded_audio, audio_error] = audio_future.get();
    if (loaded_audio == nullptr) {
      throw std::runtime_error(audio_error);
    }
    auto [loaded_text, text_error] = text_future.get();
    if (loaded_text == nullptr) {
      throw std::runtime_error(text_error);
    }
    audio_encoder = std::move(loaded_audio);
    text_decoder = std::move(loaded_text);
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
      auto admission = frontend_gate.Acquire(is_cancelled, error);
      if (!admission)
        return false;
      if (request.wav.empty() == request.pcm16k.empty()) {
        throw std::invalid_argument(
            "Qwen3-ASR requires exactly one audio source");
      }
      if (request.max_new_tokens == 0U) {
        throw std::invalid_argument(
            "Qwen3-ASR max_new_tokens must be positive");
      }
      const std::optional<std::string> language =
          NormalizeLanguage(request.language, config.supported_languages);
      CheckCancellation(is_cancelled);

      const Clock::time_point decode_begin = Clock::now();
      std::string phase_error;
      std::vector<float> decoded_waveform;
      std::span<const float> waveform = request.pcm16k;
      if (!request.wav.empty()) {
        AudioBuffer decoded_audio;
        if (!DecodeWav(request.wav, &decoded_audio, &phase_error)) {
          throw std::invalid_argument(phase_error);
        }
        decoded_waveform = ResampleMono16k(decoded_audio);
        waveform = decoded_waveform;
      }
      for (const float sample : waveform) {
        if (!std::isfinite(sample)) {
          throw std::invalid_argument("Qwen3-ASR PCM must be finite");
        }
      }
      result->timings.decode_audio_ms =
          Milliseconds(decode_begin, Clock::now());
      result->audio_samples = waveform.size();
      CheckCancellation(is_cancelled);

      const auto make_prompt = [&](std::size_t audio_tokens) {
        const std::lock_guard lock(tokenizer_mutex);
        return BuildPrompt(
            tokenizer, PromptOptions{.context = request.context,
                                     .language = language,
                                     .audio_embedding_tokens = audio_tokens});
      };
      const auto decode_text = [&](std::span<const std::uint32_t> ids) {
        // Context/language encoding can lazily replace the tokenizer while
        // another request is generating its transcript.
        const std::lock_guard lock(tokenizer_mutex);
        return tokenizer.Decode(ids, true);
      };
      const auto base = make_prompt(1);
      const std::size_t overhead = base.size() - 1;
      if (overhead >= maximum_context_tokens ||
          request.max_new_tokens >= maximum_context_tokens - overhead) {
        throw std::length_error("Qwen3-ASR context and output exceed capacity");
      }
      const std::size_t budget =
          maximum_context_tokens - overhead - request.max_new_tokens;
      const auto chunks =
          SplitAudio(waveform, AudioSamplesForTokenBudget(budget));
      result->chunks = chunks.size();
      std::string previous_language;
      std::string published;
      for (const auto& chunk : chunks) {
        CheckCancellation(is_cancelled);
        std::span<const float> chunk_waveform =
            waveform.subspan(chunk.offset, chunk.samples);
        std::vector<float> padded;
        // Match upstream's padding of short tails after splitting. Preserve
        // the existing frontend exactly for an unsplit recording.
        if (chunks.size() > 1 && chunk.samples < kAudioSampleRate / 2) {
          padded.assign(chunk_waveform.begin(), chunk_waveform.end());
          padded.resize(kAudioSampleRate / 2, 0.0F);
          chunk_waveform = padded;
        }
        const auto features_begin = Clock::now();
        LogMelFeatures features = ComputeLogMelFeatures(chunk_waveform);
        result->timings.feature_extraction_ms +=
            Milliseconds(features_begin, Clock::now());
        result->mel_frames += features.frames;
        const auto expected = AudioEmbeddingTokenCount(features.frames);
        auto prompt = make_prompt(expected);
        if (prompt.size() > maximum_context_tokens ||
            request.max_new_tokens > maximum_context_tokens - prompt.size()) {
          throw std::length_error(
              "Qwen3-ASR audio chunk exceeds context capacity");
        }
        result->prompt_tokens += prompt.size();
        CheckCancellation(is_cancelled);
        auto gpu_lease = gpu_gate.Acquire(is_cancelled, &phase_error);
        if (!gpu_lease)
          throw std::runtime_error(phase_error);
        const auto encoder_begin = Clock::now();
        AudioEncoderDeviceOutput encoded;
        if (!audio_encoder->EncodeDevice(features.values, features.frames,
                                         &encoded, &phase_error,
                                         is_cancelled)) {
          throw std::runtime_error(phase_error);
        }
        result->timings.audio_encoder_ms +=
            Milliseconds(encoder_begin, Clock::now());
        result->audio_tokens += encoded.tokens;
        if (encoded.tokens != expected) {
          throw std::runtime_error(
              "Qwen3-ASR audio frontend token count is inconsistent");
        }
        CheckCancellation(is_cancelled);
        std::vector<std::uint32_t> generated;
        const auto progress = [&](std::span<const std::uint32_t> ids) {
          CheckCancellation(is_cancelled);
          if (!request.on_text)
            return true;
          const std::string raw = decode_text(ids);
          // Automatic language metadata is not user-visible transcript text.
          if (!language && raw.find("<asr_text>") == std::string::npos)
            return true;
          auto partial = ParseTranscription(raw, language);
          const std::string current =
              core::Utf8Decoder{}.Push(result->text + partial.text);
          if (current == published)
            return true;
          published = current;
          return request.on_text(published);
        };
        const auto decoder_begin = Clock::now();
        if (!text_decoder->GenerateDevice(
                prompt, encoded.values, encoded.tokens, request.max_new_tokens,
                &generated, &phase_error, progress, is_cancelled)) {
          throw std::runtime_error(phase_error);
        }
        result->timings.text_decoder_ms +=
            Milliseconds(decoder_begin, Clock::now());
        const std::string raw = decode_text(generated);
        const auto parsed = ParseTranscription(raw, language);
        result->text += parsed.text;
        result->decoded += raw;
        result->generated_ids.insert(result->generated_ids.end(),
                                     generated.begin(), generated.end());
        if (!parsed.language.empty() && parsed.language != previous_language) {
          if (!result->language.empty())
            result->language += ",";
          result->language += parsed.language;
          previous_language = parsed.language;
        }
      }
      result->text = core::Utf8Decoder{}.Push(result->text, true);
      if (request.on_text && result->text != published &&
          !request.on_text(result->text)) {
        throw std::runtime_error("Qwen3-ASR transcription cancelled");
      }
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
  std::mutex tokenizer_mutex;
  // Bound CPU frontend memory, and yield the shared device state between
  // chunks so a long upload cannot monopolize every shorter request.
  core::CancellableGate frontend_gate{2};
  core::CancellableGate gpu_gate;
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
