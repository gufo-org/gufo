#include "src/cli/serve/asr_service.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "src/core/cancellable_gate.hpp"

namespace gufo::server {

struct AsrService::Impl {
  explicit Impl(AsrServiceOptions supplied) : options(std::move(supplied)) {
    if (options.model_id.empty()) {
      options.model_id = "qwen3-asr-1.7b";
    }

    const ext::AsrProvider* provider = nullptr;
    if (!options.model_root.empty()) {
      provider = ext::ProbeAsrProvider(options.model_root.string());
    }
    if (!provider && !options.model_id.empty()) {
      provider = ext::FindAsrProvider(options.model_id);
      if (!provider && options.model_id == "qwen3-asr-1.7b") {
        provider = ext::FindAsrProvider("qwen3-asr");
      } else if (!provider && options.model_id == "qwen3-asr") {
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
      if (!provider || !provider->create) {
        initialization_error = "ASR provider not available for model: " +
                               options.model_root.string();
        return;
      }
      native_runtime = provider->create(options.model_root.string(),
                                        options.native_context_tokens,
                                        &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.runner = [this](const ext::TranscriptionRequest& request,
                              const ext::CancellationCheck& is_cancelled,
                              ext::TranscriptionResult* result,
                              std::string* error) {
        return native_runtime->Transcribe(request, is_cancelled, result, error);
      };
    }
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

}  // namespace gufo::server
