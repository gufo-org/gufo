#include "src/cli/serve/asr_service.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/cancellable_gate.hpp"

namespace gufo::server {

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
             .projector_path = options.projector_path.string()},
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

bool AsrService::Transcribe(const ext::TranscriptionRequest& request,
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
  return impl_->options.live(rate, error);
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
  return impl_->options.voice(rate, error);
}

}  // namespace gufo::server
