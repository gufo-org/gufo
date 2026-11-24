#include <gufo/ext/asr.hpp>
#include <gufo/ext/generated/qwen3_asr_constants.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/registry.hpp>

#include "hip/transcription_runtime.hpp"
#include "probe.hpp"

namespace gufo::models::qwen3_asr {

#if defined(ENGINE_ENABLE_HIP)
class Qwen3AsrRunnerImpl : public ext::AsrRunner {
public:
  explicit Qwen3AsrRunnerImpl(
      std::unique_ptr<hip::TranscriptionHipRuntime> runtime)
      : runtime_(std::move(runtime)) {}

  bool Transcribe(const ext::TranscriptionRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::TranscriptionResult* result,
                  std::string* error) override {
    return runtime_->Transcribe(request, is_cancelled, result, error);
  }

private:
  std::unique_ptr<hip::TranscriptionHipRuntime> runtime_;
};

std::unique_ptr<ext::AsrRunner> CreateQwen3AsrRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  auto runtime = hip::TranscriptionHipRuntime::Create(
      model_root, maximum_context_tokens, error);
  if (!runtime) {
    return nullptr;
  }
  return std::make_unique<Qwen3AsrRunnerImpl>(std::move(runtime));
}

std::unique_ptr<ext::AsrRunner> CreateQwen3AsrRunnerWithOptions(
    const ext::AsrRunnerOptions& options, std::string* error) {
  if (!options.projector_path.empty()) {
    if (error)
      *error = "Qwen3-ASR does not take --mmproj";
    return nullptr;
  }
  const bool* npu = ext::FindOption<bool>(options.provider_options, "--npu");
  auto runtime = hip::TranscriptionHipRuntime::Create(
      options.model_root, options.maximum_context_tokens, error,
      npu != nullptr && *npu);
  if (!runtime) {
    return nullptr;
  }
  return std::make_unique<Qwen3AsrRunnerImpl>(std::move(runtime));
}
#endif

}  // namespace gufo::models::qwen3_asr

void qwen3_asr_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen3_asr::kProvider_qwen3_asr);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::qwen3_asr;
  registry.AddAsrProvider({
      .id = "qwen3-asr",
      .probe = &ProbeQwen3Asr,
      .resolve = &ResolveQwen3AsrArtifacts,
      .create = &CreateQwen3AsrRunner,
      .create_with_options = &CreateQwen3AsrRunnerWithOptions,
  });
  registry.AddAsrProvider({
      .id = "qwen3-asr-1.7b",
      .probe = &ProbeQwen3Asr,
      .resolve = &ResolveQwen3AsrArtifacts,
      .create = &CreateQwen3AsrRunner,
      .create_with_options = &CreateQwen3AsrRunnerWithOptions,
  });
#endif
}
