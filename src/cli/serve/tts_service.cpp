#include "src/cli/serve/tts_service.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "src/core/cancellable_gate.hpp"

namespace gufo::server {
namespace {

std::string ModelId(ext::TtsVariant variant) {
  switch (variant) {
    case ext::TtsVariant::kBase:
      return "qwen3-tts-12hz-1.7b-base";
    case ext::TtsVariant::kVoiceDesign:
      return "qwen3-tts-12hz-1.7b-voice-design";
    case ext::TtsVariant::kCustomVoice:
      return "qwen3-tts-12hz-1.7b-customvoice";
    case ext::TtsVariant::kUnsupported:
      break;
  }
  return {};
}

}  // namespace

struct TtsService::Impl {
  explicit Impl(TtsServiceOptions supplied) : options(std::move(supplied)) {
    const ext::TtsProvider* provider = nullptr;
    if (!options.model_root.empty()) {
      provider = ext::ProbeTtsProvider(options.model_root.string());
    }
    if (!provider && !options.model_id.empty()) {
      provider = ext::FindTtsProvider(options.model_id);
    }

    if (options.validate_model) {
      if (!provider || !provider->probe ||
          !provider->probe(options.model_root.string())) {
        initialization_error =
            "Qwen3-TTS service requires a supported 1.7B model";
        return;
      }
    }

    if (!options.runner) {
      if (!provider || !provider->create) {
        initialization_error = "TTS provider not available for model: " +
                               options.model_root.string();
        return;
      }
      native_runtime = provider->create(options.model_root.string(),
                                        options.native_context_tokens,
                                        &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.variant = native_runtime->Variant();
      if (options.voices.empty()) {
        options.voices = native_runtime->Voices();
      }
      options.runner = [this](const ext::SynthesisRequest& request,
                              const ext::CancellationCheck& is_cancelled,
                              ext::SynthesisResult* result,
                              std::string* error) {
        return native_runtime->Synthesize(request, is_cancelled, result, error);
      };
    }

    if (options.model_id.empty()) {
      options.model_id = ModelId(options.variant);
    }
    if (options.voices.empty()) {
      if (options.variant == ext::TtsVariant::kVoiceDesign) {
        options.voices.push_back("voice-design");
      } else if (options.variant == ext::TtsVariant::kBase) {
        options.voices.push_back("voice-clone");
      }
    }
    // Presets carry reference audio, which only the Base checkpoint consumes:
    // CustomVoice selects a trained speaker embedding and VoiceDesign is
    // driven by `instruct`, so neither has anything to apply them to.
    if (!options.voice_presets.empty() &&
        options.variant != ext::TtsVariant::kBase) {
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
    std::ranges::sort(options.voices);
    ready = true;
  }

  TtsServiceOptions options;
  std::unique_ptr<ext::TtsRunner> native_runtime;
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

ext::TtsVariant TtsService::variant() const noexcept {
  return impl_->options.variant;
}

bool TtsService::Synthesize(const ext::SynthesisRequest& request,
                            const ext::CancellationCheck& is_cancelled,
                            ext::SynthesisResult* result, std::string* error) {
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
