#include "src/cli/serve/asr_service.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/asr_metrics.hpp"
#include "src/core/cancellable_gate.hpp"

namespace gufo::server {

namespace {

class MeteredLiveTranscription final : public LiveTranscription {
public:
  MeteredLiveTranscription(std::unique_ptr<LiveTranscription> inner,
                           const std::shared_ptr<AsrMetrics>& metrics,
                           std::uint32_t rate)
      : scope_(metrics, metrics->live_sessions),
        audio_seconds_(metrics->live_audio_seconds),
        rate_(rate),
        inner_(std::move(inner)) {}

  bool Push(std::span<const float> samples, const Text& on_text,
            std::string* error) override {
    const bool ok = inner_->Push(samples, on_text, error);
    if (ok && rate_ != 0)
      audio_seconds_.Add(static_cast<double>(samples.size()) /
                         static_cast<double>(rate_));
    return ok;
  }
  bool Finish(const Text& on_text, std::string* error) override {
    return inner_->Finish(on_text, error);
  }
  const std::string& text() const override { return inner_->text(); }

private:
  AsrSessionScope scope_;
  core::metrics::FloatCounter audio_seconds_;
  std::uint32_t rate_;
  std::unique_ptr<LiveTranscription> inner_;
};

class MeteredVoiceConversation final : public VoiceConversation {
public:
  MeteredVoiceConversation(std::unique_ptr<VoiceConversation> inner,
                           const std::shared_ptr<AsrMetrics>& metrics)
      : scope_(metrics, metrics->voice_sessions), inner_(std::move(inner)) {}

  bool Configure(const Settings& settings, std::string* error) override {
    return inner_->Configure(settings, error);
  }
  bool AppendAudio(std::span<const float> samples,
                   std::string* error) override {
    return inner_->AppendAudio(samples, error);
  }
  bool CommitAudio(std::size_t* samples, std::string* error) override {
    return inner_->CommitAudio(samples, error);
  }
  bool ClearAudio(std::string* error) override {
    return inner_->ClearAudio(error);
  }
  bool audio_open() const override { return inner_->audio_open(); }
  bool AddUserText(std::string_view text, std::string* error) override {
    return inner_->AddUserText(text, error);
  }
  bool AddToolOutputs(std::span<const std::string> outputs,
                      std::string* error) override {
    return inner_->AddToolOutputs(outputs, error);
  }
  bool Respond(const Settings& settings, const Callbacks& callbacks,
               Reply* reply, std::string* error) override {
    return inner_->Respond(settings, callbacks, reply, error);
  }
  bool TruncateReply(std::size_t samples, std::string* error) override {
    return inner_->TruncateReply(samples, error);
  }
  std::vector<std::string> voices() const override { return inner_->voices(); }
  bool supports_response_options() const override {
    return inner_->supports_response_options();
  }
  bool DeleteUserTurn(std::string* error) override {
    return inner_->DeleteUserTurn(error);
  }

private:
  AsrSessionScope scope_;
  std::unique_ptr<VoiceConversation> inner_;
};

}  // namespace

struct AsrService::Impl {
  explicit Impl(AsrServiceOptions supplied) : options(std::move(supplied)) {
    const std::string requested_id =
        options.model_id.empty() ? "qwen3-asr-1.7b" : options.model_id;

    const ext::AsrProvider* provider = nullptr;
    if (!options.model_root.empty()) {
      provider = ext::ProbeAsrProvider(options.model_root.string());
    }
    if (!provider) {
      provider = ext::FindAsrProvider(requested_id);
      if (!provider && requested_id == "qwen3-asr-1.7b") {
        provider = ext::FindAsrProvider("qwen3-asr");
      } else if (!provider && requested_id == "qwen3-asr") {
        provider = ext::FindAsrProvider("qwen3-asr-1.7b");
      }
    }

    if (options.validate_model) {
      if (!provider || !provider->probe ||
          !provider->probe(options.model_root.string())) {
        initialization_error =
            "Qwen3-ASR service requires the supported 1.7B model";
        return;
      }
    }

    if (!options.runner) {
      if (!provider || (!provider->create && !provider->create_with_options)) {
        initialization_error = "ASR provider not available for model: " +
                               options.model_root.string();
        return;
      }
      if (provider->create_with_options)
        native_runtime = provider->create_with_options(
            {.model_root = options.model_root.string(),
             .maximum_context_tokens = options.native_context_tokens,
             .projector_path = options.projector_path.string(),
             .provider_options = options.provider_options},
            &initialization_error);
      else if (!options.projector_path.empty())
        initialization_error =
            "this ASR model does not take a projector sidecar";
      else
        native_runtime = provider->create(options.model_root.string(),
                                          options.native_context_tokens,
                                          &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      if (options.model_id.empty())
        options.model_id = native_runtime->ModelId();
      if (!options.live && native_runtime->SupportsLiveTranscription())
        options.live = [this](std::uint32_t rate, std::string* error) {
          return native_runtime->CreateLiveTranscription(rate, error);
        };
      if (!options.voice && native_runtime->SupportsVoiceConversation())
        options.voice = [this](std::uint32_t rate, std::string* error) {
          return native_runtime->CreateVoiceConversation(rate, error);
        };
      options.runner = [this](const ext::TranscriptionRequest& request,
                              const ext::CancellationCheck& is_cancelled,
                              ext::TranscriptionResult* result,
                              std::string* error) {
        return native_runtime->Transcribe(request, is_cancelled, result, error);
      };
    }
    if (options.model_id.empty())
      options.model_id = requested_id;
    ready = true;
  }

  AsrServiceOptions options;
  std::unique_ptr<ext::AsrRunner> native_runtime;
  std::string initialization_error;
  core::CancellableGate transcription_gate;
  std::shared_ptr<AsrMetrics> metrics{std::make_shared<AsrMetrics>()};
  bool ready{false};
};

AsrService::AsrService(AsrServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

AsrService::~AsrService() = default;

bool AsrService::ready() const noexcept {
  return impl_->ready;
}

std::string AsrService::initialization_error() const {
  return impl_->initialization_error;
}

std::string AsrService::model_id() const {
  return impl_->options.model_id;
}

bool AsrService::serves_model(std::string_view name) const {
  return name == model_id() || name == "qwen3-asr" ||
         name == "qwen3-asr-1.7b" ||
         std::find(impl_->options.model_aliases.begin(),
                   impl_->options.model_aliases.end(),
                   name) != impl_->options.model_aliases.end();
}

const std::vector<std::string>& AsrService::model_aliases() const {
  return impl_->options.model_aliases;
}

std::string AsrService::backend_name() const {
  return "native-hip";
}

std::string AsrService::RenderMetrics() const {
  return impl_->metrics->registry.Render();
}

bool AsrService::Transcribe(const ext::TranscriptionRequest& request,
                            const ext::CancellationCheck& is_cancelled,
                            ext::TranscriptionResult* result,
                            std::string* error) {
  AsrRequestScope scope(*impl_->metrics);
  const bool ok = TranscribeAdmitted(request, is_cancelled, result, error);
  if (ok)
    scope.Succeeded(result);
  else if (is_cancelled && is_cancelled())
    scope.Cancelled();
  else if (result != nullptr && result->invalid_audio)
    scope.InvalidAudio();
  return ok;
}

bool AsrService::TranscribeAdmitted(const ext::TranscriptionRequest& request,
                                    const ext::CancellationCheck& is_cancelled,
                                    ext::TranscriptionResult* result,
                                    std::string* error) {
  if (!impl_->ready) {
    if (error != nullptr) {
      *error = impl_->initialization_error;
    }
    return false;
  }
  // Native inference manages admission per chunk, allowing other requests'
  // frontend work to proceed while the GPU is occupied.
  if (impl_->native_runtime)
    return impl_->options.runner(request, is_cancelled, result, error);
  auto lease = impl_->transcription_gate.Acquire(is_cancelled, error);
  if (!lease) {
    if (result)
      *result = {};
    return false;
  }
  return impl_->options.runner(request, is_cancelled, result, error);
}

bool AsrService::supports_live() const noexcept {
  return impl_->ready && bool(impl_->options.live);
}

std::unique_ptr<LiveTranscription> AsrService::CreateLiveTranscription(
    std::uint32_t rate, std::string* error) {
  if (!supports_live()) {
    *error = "live transcription is not supported by this model";
    return nullptr;
  }
  auto live = impl_->options.live(rate, error);
  if (live == nullptr)
    return nullptr;
  return std::make_unique<MeteredLiveTranscription>(std::move(live),
                                                    impl_->metrics, rate);
}

bool AsrService::supports_voice() const noexcept {
  return impl_->ready && bool(impl_->options.voice);
}

std::unique_ptr<VoiceConversation> AsrService::CreateVoiceConversation(
    std::uint32_t rate, std::string* error) {
  if (!supports_voice()) {
    *error = "spoken conversations are not supported by this model";
    return nullptr;
  }
  auto voice = impl_->options.voice(rate, error);
  if (voice == nullptr)
    return nullptr;
  return std::make_unique<MeteredVoiceConversation>(std::move(voice),
                                                    impl_->metrics);
}

}  // namespace gufo::server
