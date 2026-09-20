#include "src/cli/serve/asr_service.hpp"

#include <optional>
#include <utility>

#include "src/core/cancellable_gate.hpp"
#include "src/models/qwen3_asr/config.hpp"
#include "src/models/qwen3_asr/hip/transcription_runtime.hpp"

namespace gufo::server {

struct AsrService::Impl {
  explicit Impl(AsrServiceOptions supplied) : options(std::move(supplied)) {
    if (options.validate_model) {
      const std::optional<models::qwen3_asr::ModelConfig> config =
          models::qwen3_asr::LoadModelConfigFromPath(
              options.model_root.string());
      if (!config.has_value() ||
          !models::qwen3_asr::IsSupportedModelConfig(*config)) {
        initialization_error =
            "Qwen3-ASR service requires the supported 1.7B model";
        return;
      }
    }
    if (options.model_id.empty()) {
      options.model_id = "qwen3-asr-1.7b";
    }
    if (!options.runner) {
      native_runtime = models::qwen3_asr::hip::TranscriptionHipRuntime::Create(
          options.model_root.string(), options.native_context_tokens,
          &initialization_error);
      if (native_runtime == nullptr) {
        return;
      }
      options.runner =
          [this](const models::qwen3_asr::TranscriptionRequest& request,
                 const models::qwen3_asr::CancellationCheck& is_cancelled,
                 models::qwen3_asr::TranscriptionResult* result,
                 std::string* error) {
            return native_runtime->Transcribe(request, is_cancelled, result,
                                              error);
          };
    }
    ready = true;
  }

  AsrServiceOptions options;
  std::unique_ptr<models::qwen3_asr::hip::TranscriptionHipRuntime>
      native_runtime;
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

bool AsrService::Transcribe(
    const models::qwen3_asr::TranscriptionRequest& request,
    const models::qwen3_asr::CancellationCheck& is_cancelled,
    models::qwen3_asr::TranscriptionResult* result, std::string* error) {
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
