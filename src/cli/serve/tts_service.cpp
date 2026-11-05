#include "src/cli/serve/tts_service.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "src/core/cancellable_gate.hpp"

namespace gufo::server {
namespace {

constexpr std::uint32_t kMaximumSampleRate = 192000;

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
      if (!provider || (!provider->create && !provider->create_with_options)) {
        initialization_error = "TTS provider not available for model: " +
                               options.model_root.string();
        return;
      }
      if (provider->create_with_options)
        native_runtime = provider->create_with_options(
            {.model_root = options.model_root.string(),
             .maximum_context_tokens = options.native_context_tokens,
             .projector_path = options.projector_path.string()},
            &initialization_error);
      else if (!options.projector_path.empty())
        initialization_error =
            "this TTS model does not take a projector sidecar";
      else
        native_runtime = provider->create(options.model_root.string(),
                                          options.native_context_tokens,
                                          &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.sample_rate = native_runtime->SampleRate();
      options.variant = native_runtime->Variant();
      if (!options.conditioning.has_value()) {
        options.conditioning = native_runtime->Conditioning();
      }
      options.concurrent = options.concurrent || native_runtime->Concurrent();
      if (options.model_id.empty())
        options.model_id = native_runtime->ModelId();
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

    if (options.sample_rate == 0 || options.sample_rate > kMaximumSampleRate) {
      initialization_error = "invalid TTS output sample rate";
      return;
    }
    if (!options.conditioning.has_value()) {
      options.conditioning =
          ext::DefaultTtsConditioningCapabilities(options.variant);
    }
    const auto& cond = *options.conditioning;
    if (!cond.unconditioned &&
        cond.reference_audio == ext::TtsConditioningMode::kUnsupported &&
        !cond.reference_presets) {
      initialization_error =
          "TTS provider does not support any synthesis conditioning mode";
      return;
    }
    if (cond.unconditioned &&
        cond.reference_audio == ext::TtsConditioningMode::kRequired) {
      initialization_error =
          "TTS provider cannot require reference audio when unconditioned "
          "synthesis is enabled";
      return;
    }

    if (options.model_id.empty()) {
      options.model_id = ModelId(options.variant);
    }
    if (options.voices.empty()) {
      if (!cond.default_voice.empty()) {
        options.voices.push_back(cond.default_voice);
      } else if (cond.instructions == ext::TtsConditioningMode::kRequired) {
        options.voices.push_back("voice-design");
      } else if (cond.reference_audio !=
                 ext::TtsConditioningMode::kUnsupported) {
        options.voices.push_back("voice-clone");
      }
    }
    if (!options.voice_presets.empty() && !cond.reference_presets) {
      initialization_error =
          "voice presets require a TTS model that supports reference "
          "presets";
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

bool TtsService::serves_model(std::string_view name) const {
  return name == model_id() || name == "qwen3-tts" ||
         std::find(impl_->options.model_aliases.begin(),
                   impl_->options.model_aliases.end(),
                   name) != impl_->options.model_aliases.end();
}

const std::vector<std::string>& TtsService::model_aliases() const {
  return impl_->options.model_aliases;
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

std::uint32_t TtsService::sample_rate() const noexcept {
  return impl_->options.sample_rate;
}

ext::TtsVariant TtsService::variant() const noexcept {
  return impl_->options.variant;
}

const ext::TtsConditioningCapabilities& TtsService::conditioning()
    const noexcept {
  // Construction always fills `conditioning` before the service can become
  // ready; the fallback only guards use on a service that failed to start.
  static const ext::TtsConditioningCapabilities kNoneSupported{
      /*unconditioned=*/false,
      /*require_voice=*/false,
      /*default_voice=*/{},
      /*reference_audio=*/ext::TtsConditioningMode::kUnsupported,
      /*reference_presets=*/false,
      /*instructions=*/ext::TtsConditioningMode::kUnsupported};
  return impl_->options.conditioning.has_value() ? *impl_->options.conditioning
                                                 : kNoneSupported;
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
  const auto run = [&] {
    if (!impl_->options.runner(request, is_cancelled, result, error)) {
      return false;
    }
    if (result == nullptr ||
        result->sample_rate != impl_->options.sample_rate) {
      if (error != nullptr) {
        *error = kInvalidSynthesisResultError;
      }
      if (result)
        *result = {};
      return false;
    }
    return true;
  };
  if (impl_->options.concurrent)
    return run();
  auto lease = impl_->generation_gate.Acquire(is_cancelled, error);
  if (!lease) {
    if (result)
      *result = {};
    return false;
  }
  return run();
}

}  // namespace gufo::server
