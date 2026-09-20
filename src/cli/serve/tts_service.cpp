#include "src/cli/serve/tts_service.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "src/core/cancellable_gate.hpp"
#include "src/models/qwen3_tts/config.hpp"
#include "src/models/qwen3_tts/hip/synthesis_runtime.hpp"

namespace gufo::server {
namespace {

std::string ModelId(models::qwen3_tts::ModelVariant variant) {
  switch (variant) {
    case models::qwen3_tts::ModelVariant::kBase:
      return "qwen3-tts-12hz-1.7b-base";
    case models::qwen3_tts::ModelVariant::kVoiceDesign:
      return "qwen3-tts-12hz-1.7b-voice-design";
    case models::qwen3_tts::ModelVariant::kCustomVoice:
      return "qwen3-tts-12hz-1.7b-customvoice";
    case models::qwen3_tts::ModelVariant::kUnsupported:
      break;
  }
  return {};
}

}  // namespace

struct TtsService::Impl {
  explicit Impl(TtsServiceOptions supplied) : options(std::move(supplied)) {
    std::optional<models::qwen3_tts::ModelConfig> config;
    if (options.validate_model) {
      config = models::qwen3_tts::LoadModelConfigFromPath(
          options.model_root.string());
      if (!config.has_value() ||
          !models::qwen3_tts::IsSupportedModelConfig(*config)) {
        initialization_error =
            "Qwen3-TTS service requires a supported 1.7B model";
        return;
      }
      options.variant = config->variant;
      if (options.voices.empty() &&
          options.variant == models::qwen3_tts::ModelVariant::kCustomVoice) {
        for (const auto& [name, unused] : config->talker.spk_id) {
          (void)unused;
          options.voices.push_back(name);
        }
      }
    }
    if (options.model_id.empty()) {
      options.model_id = ModelId(options.variant);
    }
    if (options.voices.empty()) {
      if (options.variant == models::qwen3_tts::ModelVariant::kVoiceDesign) {
        options.voices.push_back("voice-design");
      } else if (options.variant == models::qwen3_tts::ModelVariant::kBase) {
        options.voices.push_back("voice-clone");
      }
    }
    // Presets carry reference audio, which only the Base checkpoint consumes:
    // CustomVoice selects a trained speaker embedding and VoiceDesign is
    // driven by `instruct`, so neither has anything to apply them to.
    if (!options.voice_presets.empty() &&
        options.variant != models::qwen3_tts::ModelVariant::kBase) {
      initialization_error =
          "voice presets require a Qwen3-TTS Base checkpoint";
      return;
    }
    for (const auto& [name, unused] : options.voice_presets) {
      (void)unused;
      if (std::ranges::find(options.voices, name) == options.voices.end()) {
        options.voices.push_back(name);
      }
    }
    if (!options.runner) {
      native_runtime = models::qwen3_tts::hip::SynthesisHipRuntime::Create(
          options.model_root.string(), options.native_context_tokens,
          options.variant, &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.runner =
          [this](const models::qwen3_tts::SynthesisRequest& request,
                 const models::qwen3_tts::CancellationCheck& is_cancelled,
                 models::qwen3_tts::SynthesisResult* result,
                 std::string* error) {
            return native_runtime->Generate(request, is_cancelled, result,
                                            error);
          };
    }
    std::ranges::sort(options.voices);
    ready = true;
  }

  TtsServiceOptions options;
  std::unique_ptr<models::qwen3_tts::hip::SynthesisHipRuntime> native_runtime;
  std::string initialization_error;
  core::CancellableGate generation_gate;
  bool ready{false};
};

TtsService::TtsService(TtsServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

TtsService::~TtsService() = default;

bool TtsService::ready() const noexcept {
  return impl_->ready;
}

std::string TtsService::initialization_error() const {
  return impl_->initialization_error;
}

std::string TtsService::model_id() const {
  return impl_->options.model_id;
}

std::string TtsService::backend_name() const {
  return "native-hip";
}

std::vector<std::string> TtsService::voices() const {
  return impl_->options.voices;
}

const TtsVoicePreset* TtsService::voice_preset(const std::string& name) const {
  const auto found = impl_->options.voice_presets.find(name);
  return found == impl_->options.voice_presets.end() ? nullptr : &found->second;
}

models::qwen3_tts::ModelVariant TtsService::variant() const noexcept {
  return impl_->options.variant;
}

bool TtsService::Synthesize(
    const models::qwen3_tts::SynthesisRequest& request,
    const models::qwen3_tts::CancellationCheck& is_cancelled,
    models::qwen3_tts::SynthesisResult* result, std::string* error) {
  if (!impl_->ready) {
    if (error != nullptr) {
      *error = impl_->initialization_error;
    }
    return false;
  }
  auto lease = impl_->generation_gate.Acquire(is_cancelled, error);
  if (!lease) {
    if (result)
      *result = {};
    return false;
  }
  return impl_->options.runner(request, is_cancelled, result, error);
}

}  // namespace gufo::server
