#include "src/models/qwen3_tts/hip/synthesis_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/qwen3_tts/audio.hpp"
#include "src/models/qwen3_tts/hip/speaker_encoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/speech_decoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/speech_encoder_runtime.hpp"
#include "src/models/qwen3_tts/hip/talker_runtime.hpp"
#include "src/models/qwen3_tts/tokenizer.hpp"

namespace gufo::models::qwen3_tts::hip {
namespace {

void SetError(std::string* error, std::string_view message) {
  if (error != nullptr) {
    error->assign(message);
  }
}

bool SameAudio(const AudioBuffer& left, const AudioBuffer& right) {
  return left.sample_rate == right.sample_rate &&
         left.channels == right.channels && left.samples == right.samples;
}

struct PromptTokens {
  std::vector<std::uint32_t> input;
  std::vector<std::uint32_t> instruction;
};

struct NativePrompt {
  TalkerPromptOutput talker;
  SpeechEncoderOutput reference_codes;
  bool prepend_reference_audio{false};
};

struct BaseAudioFeatures {
  AudioBuffer audio;
  SpeakerEncoderOutput speaker;
  std::unique_ptr<SpeechEncoderOutput> speech;
};

}  // namespace

struct SynthesisHipRuntime::Impl {
  using PromptBuilder = bool (Impl::*)(const SynthesisRequest&,
                                       const PromptTokens&, NativePrompt*,
                                       std::string*);

  [[nodiscard]] static std::unique_ptr<Impl> Create(
      const std::string& model_root, std::size_t maximum_context_tokens,
      ModelVariant variant, std::string* error) {
    auto impl =
        std::unique_ptr<Impl>(new Impl(maximum_context_tokens, variant));
    if (!impl->Initialize(model_root, error)) {
      return nullptr;
    }
    return impl;
  }

  [[nodiscard]] bool Generate(const SynthesisRequest& request,
                              const CancellationCheck& is_cancelled,
                              SynthesisResult* result, std::string* error) {
    if (!PrepareResult(result, error) || IsCancelled(is_cancelled, error)) {
      return false;
    }

    PromptTokens tokens;
    if (!Tokenize(request, &tokens, error)) {
      return false;
    }

    NativePrompt prompt;
    if (!(this->*prompt_builder_)(request, tokens, &prompt, error) ||
        !FitsContext(prompt.talker, request.max_new_tokens, error)) {
      return false;
    }

    if (request.on_audio) {
      return GenerateStreaming(request, prompt, is_cancelled, result, error);
    }
    TalkerGenerationOutput generated;
    if (!GenerateCodecFrames(request, prompt.talker, is_cancelled, &generated,
                             error) ||
        IsCancelled(is_cancelled, error)) {
      return false;
    }
    return Decode(prompt, generated, result, error);
  }

private:
  [[nodiscard]] bool GenerateStreaming(const SynthesisRequest& request,
                                       const NativePrompt& prompt,
                                       const CancellationCheck& is_cancelled,
                                       SynthesisResult* result,
                                       std::string* error) {
    std::vector<std::uint32_t> codes;
    const std::size_t reference_frames =
        prompt.prepend_reference_audio ? prompt.reference_codes.frames : 0;
    if (reference_frames != 0)
      codes = prompt.reference_codes.codes;
    std::size_t decoded_frames = 0;
    std::size_t emitted_frames = 0;
    std::string stream_error;
    const auto emit = [&](const TalkerGenerationOutput& generated, bool flush) {
      const std::size_t pending = generated.frames - emitted_frames;
      if (pending == 0 || (!flush && pending < (emitted_frames == 0 ? 4 : 16)))
        return true;
      if (IsCancelled(is_cancelled, &stream_error))
        return false;
      codes.resize(reference_frames * 16);
      codes.insert(codes.end(), generated.codes.begin(), generated.codes.end());
      SpeechDecoderOutput audio;
      const bool first_reference = decoded_frames == 0 && reference_frames != 0;
      if (!(first_reference ? decoder_->DecodeAfterReference(
                                  codes, reference_frames + generated.frames,
                                  reference_frames, &audio, &stream_error)
                            : decoder_->DecodeIncremental(
                                  codes, reference_frames + generated.frames,
                                  decoded_frames, &audio, &stream_error)))
        return false;
      decoded_frames = reference_frames + generated.frames;
      emitted_frames = generated.frames;
      const auto samples = std::span<const float>(audio.samples);
      if (!request.on_audio(samples)) {
        stream_error = "Qwen3-TTS audio stream cancelled";
        return false;
      }
      result->sample_count += samples.size();
      return true;
    };
    TalkerGenerationOutput generated;
    if (!talker_->Generate(
            prompt.talker, request.max_new_tokens, request.sampling,
            is_cancelled, &generated, error,
            [&](const auto& progress) { return emit(progress, false); }) ||
        !emit(generated, true) || generated.frames == 0) {
      if (!stream_error.empty())
        SetError(error, stream_error);
      *result = {};
      return false;
    }
    result->sample_rate = 24000;
    result->code_groups = static_cast<std::uint32_t>(generated.code_groups);
    result->codes.assign(generated.codes.begin(), generated.codes.end());
    return true;
  }

  Impl(std::size_t maximum_context_tokens, ModelVariant variant)
      : maximum_context_tokens_(maximum_context_tokens), variant_(variant) {}

  [[nodiscard]] bool Initialize(const std::string& model_root,
                                std::string* error) {
    if (maximum_context_tokens_ == 0) {
      SetError(error, "Qwen3-TTS native context capacity must be positive");
      return false;
    }
    if (!SelectPromptBuilder(error) ||
        !Tokenizer::Load(model_root, &tokenizer_, error)) {
      return false;
    }
    talker_ =
        TalkerHipRuntime::Create(model_root, maximum_context_tokens_, error);
    if (talker_ == nullptr) {
      return false;
    }
    decoder_ = SpeechDecoderHipRuntime::Create(model_root, error);
    if (decoder_ == nullptr) {
      return false;
    }
    return variant_ != ModelVariant::kBase ||
           InitializeBaseEncoders(model_root, error);
  }

  [[nodiscard]] bool SelectPromptBuilder(std::string* error) {
    switch (variant_) {
      case ModelVariant::kCustomVoice:
        prompt_builder_ = &Impl::BuildCustomVoicePrompt;
        return true;
      case ModelVariant::kVoiceDesign:
        prompt_builder_ = &Impl::BuildVoiceDesignPrompt;
        return true;
      case ModelVariant::kBase:
        prompt_builder_ = &Impl::BuildBasePrompt;
        return true;
      case ModelVariant::kUnsupported:
        SetError(error, "unsupported Qwen3-TTS variant");
        return false;
    }
    SetError(error, "unsupported Qwen3-TTS variant");
    return false;
  }

  [[nodiscard]] bool InitializeBaseEncoders(const std::string& model_root,
                                            std::string* error) {
    speech_encoder_ = SpeechEncoderHipRuntime::Create(model_root, error);
    if (speech_encoder_ == nullptr) {
      return false;
    }
    speaker_encoder_ = SpeakerEncoderHipRuntime::Create(model_root, error);
    return speaker_encoder_ != nullptr;
  }

  [[nodiscard]] static bool PrepareResult(SynthesisResult* result,
                                          std::string* error) {
    if (result == nullptr) {
      SetError(error, "Qwen3-TTS native result must not be null");
      return false;
    }
    *result = {};
    return true;
  }

  [[nodiscard]] static bool IsCancelled(const CancellationCheck& is_cancelled,
                                        std::string* error) {
    if (!is_cancelled || !is_cancelled()) {
      return false;
    }
    SetError(error, "Qwen3-TTS native generation cancelled");
    return true;
  }

  [[nodiscard]] bool Tokenize(const SynthesisRequest& request,
                              PromptTokens* tokens, std::string* error) const {
    if (!tokenizer_.EncodeAssistantPrompt(request.text, &tokens->input,
                                          error)) {
      return false;
    }
    return request.instruct.empty() ||
           tokenizer_.EncodeInstructionPrompt(request.instruct,
                                              &tokens->instruction, error);
  }

  [[nodiscard]] bool BuildCustomVoicePrompt(const SynthesisRequest& request,
                                            const PromptTokens& tokens,
                                            NativePrompt* prompt,
                                            std::string* error) {
    return talker_->BuildCustomVoicePrompt(tokens.input, tokens.instruction,
                                           request.speaker, request.language,
                                           &prompt->talker, error);
  }

  [[nodiscard]] bool BuildVoiceDesignPrompt(const SynthesisRequest& request,
                                            const PromptTokens& tokens,
                                            NativePrompt* prompt,
                                            std::string* error) {
    if (request.instruct.empty()) {
      SetError(error, "Qwen3-TTS VoiceDesign requires a non-empty instruction");
      return false;
    }
    return talker_->BuildVoiceDesignPrompt(tokens.input, tokens.instruction,
                                           request.language, &prompt->talker,
                                           error);
  }

  [[nodiscard]] bool BuildBasePrompt(const SynthesisRequest& request,
                                     const PromptTokens& tokens,
                                     NativePrompt* prompt, std::string* error) {
    BaseAudioFeatures* features =
        PrepareBaseAudio(request.reference_audio, error);
    if (features == nullptr) {
      return false;
    }

    std::vector<std::uint32_t> reference_ids;
    if (!request.speaker_embedding_only &&
        !PrepareBaseReference(request, *features, &reference_ids,
                              &prompt->reference_codes, error)) {
      return false;
    }
    prompt->prepend_reference_audio = !request.speaker_embedding_only;
    return talker_->BuildVoiceClonePrompt(
        {
            .input_ids = tokens.input,
            .reference_ids = reference_ids,
            .reference_codes = prompt->reference_codes.codes,
            .speaker_embedding = features->speaker.embedding,
            .reference_frames = prompt->reference_codes.frames,
            .language = request.language,
            .icl_mode = !request.speaker_embedding_only,
        },
        &prompt->talker, error);
  }

  [[nodiscard]] BaseAudioFeatures* PrepareBaseAudio(const AudioBuffer& audio,
                                                    std::string* error) {
    if (audio.samples.empty()) {
      SetError(error, "Qwen3-TTS Base requires reference audio");
      return nullptr;
    }
    if (speaker_encoder_ == nullptr || speech_encoder_ == nullptr) {
      SetError(error, "Qwen3-TTS Base encoders are not initialized");
      return nullptr;
    }
    if (base_audio_features_ != nullptr &&
        SameAudio(base_audio_features_->audio, audio)) {
      return base_audio_features_.get();
    }

    SpeakerEncoderOutput speaker;
    if (!speaker_encoder_->Encode(audio, &speaker, error)) {
      return nullptr;
    }
    base_audio_features_ = std::make_unique<BaseAudioFeatures>(
        BaseAudioFeatures{audio, std::move(speaker), nullptr});
    return base_audio_features_.get();
  }

  [[nodiscard]] bool PrepareBaseReference(
      const SynthesisRequest& request, BaseAudioFeatures& features,
      std::vector<std::uint32_t>* reference_ids,
      SpeechEncoderOutput* reference_codes, std::string* error) {
    if (request.reference_text.empty()) {
      SetError(error, "Qwen3-TTS Base ICL requires reference text");
      return false;
    }
    if (!tokenizer_.EncodeReferencePrompt(request.reference_text, reference_ids,
                                          error)) {
      return false;
    }
    if (features.speech != nullptr) {
      *reference_codes = *features.speech;
      return true;
    }
    auto encoded = std::make_unique<SpeechEncoderOutput>();
    if (!speech_encoder_->Encode(request.reference_audio, encoded.get(),
                                 error)) {
      return false;
    }
    *reference_codes = *encoded;
    features.speech = std::move(encoded);
    return true;
  }

  [[nodiscard]] bool FitsContext(const TalkerPromptOutput& prompt,
                                 std::size_t maximum_new_tokens,
                                 std::string* error) const {
    const std::size_t available =
        maximum_context_tokens_ -
        std::min(prompt.tokens, maximum_context_tokens_);
    if (maximum_new_tokens <= available) {
      return true;
    }
    SetError(error,
             "Qwen3-TTS prompt plus generated tokens exceeds native context "
             "capacity");
    return false;
  }

  [[nodiscard]] bool GenerateCodecFrames(const SynthesisRequest& request,
                                         const TalkerPromptOutput& prompt,
                                         const CancellationCheck& is_cancelled,
                                         TalkerGenerationOutput* generated,
                                         std::string* error) {
    const SamplingOptions& sampling = request.sampling;
    if (!talker_->Generate(prompt, request.max_new_tokens, sampling,
                           is_cancelled, generated, error)) {
      return false;
    }
    if (generated->frames != 0 && generated->code_groups == 16 &&
        generated->codes.size() == generated->frames * generated->code_groups) {
      return true;
    }
    SetError(error, "Qwen3-TTS native talker produced invalid codec frames");
    return false;
  }

  [[nodiscard]] bool Decode(const NativePrompt& prompt,
                            const TalkerGenerationOutput& generated,
                            SynthesisResult* result, std::string* error) {
    std::vector<std::uint32_t> decoder_codes;
    std::span<const std::uint32_t> codes_to_decode = generated.codes;
    std::size_t decoder_frames = generated.frames;
    if (prompt.prepend_reference_audio) {
      decoder_codes.reserve(prompt.reference_codes.codes.size() +
                            generated.codes.size());
      decoder_codes.insert(decoder_codes.end(),
                           prompt.reference_codes.codes.begin(),
                           prompt.reference_codes.codes.end());
      decoder_codes.insert(decoder_codes.end(), generated.codes.begin(),
                           generated.codes.end());
      codes_to_decode = decoder_codes;
      decoder_frames += prompt.reference_codes.frames;
    }

    SpeechDecoderOutput audio;
    if (!(prompt.prepend_reference_audio
              ? decoder_->DecodeAfterReference(codes_to_decode, decoder_frames,
                                               prompt.reference_codes.frames,
                                               &audio, error)
              : decoder_->Decode(codes_to_decode, decoder_frames, &audio,
                                 nullptr, error))) {
      return false;
    }
    PopulateResult(generated, std::move(audio), result);
    return true;
  }

  static void PopulateResult(const TalkerGenerationOutput& generated,
                             SpeechDecoderOutput audio,
                             SynthesisResult* result) {
    result->sample_rate = audio.sample_rate;
    result->code_groups = static_cast<std::uint32_t>(generated.code_groups);
    result->samples = std::move(audio.samples);
    result->sample_count = result->samples.size();
    result->codes.reserve(generated.codes.size());
    for (const std::uint32_t code : generated.codes) {
      result->codes.push_back(static_cast<std::int32_t>(code));
    }
  }

  std::size_t maximum_context_tokens_;
  ModelVariant variant_;
  PromptBuilder prompt_builder_{nullptr};
  Tokenizer tokenizer_;
  std::unique_ptr<TalkerHipRuntime> talker_;
  std::unique_ptr<SpeechDecoderHipRuntime> decoder_;
  std::unique_ptr<SpeechEncoderHipRuntime> speech_encoder_;
  std::unique_ptr<SpeakerEncoderHipRuntime> speaker_encoder_;
  std::unique_ptr<BaseAudioFeatures> base_audio_features_;
};

std::unique_ptr<SynthesisHipRuntime> SynthesisHipRuntime::Create(
    const std::string& model_root, std::size_t maximum_context_tokens,
    ModelVariant variant, std::string* error) {
  auto impl = Impl::Create(model_root, maximum_context_tokens, variant, error);
  if (impl == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<SynthesisHipRuntime>(
      new SynthesisHipRuntime(std::move(impl)));
}

SynthesisHipRuntime::SynthesisHipRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SynthesisHipRuntime::~SynthesisHipRuntime() = default;

bool SynthesisHipRuntime::Generate(const SynthesisRequest& request,
                                   const CancellationCheck& is_cancelled,
                                   SynthesisResult* result,
                                   std::string* error) {
  return impl_->Generate(request, is_cancelled, result, error);
}

}  // namespace gufo::models::qwen3_tts::hip
