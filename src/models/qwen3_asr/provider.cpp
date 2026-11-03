#include <gufo/ext/asr.hpp>
#include <gufo/ext/generated/qwen3_asr_constants.hpp>
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
  });
  registry.AddAsrProvider({
      .id = "qwen3-asr-1.7b",
      .probe = &ProbeQwen3Asr,
      .resolve = &ResolveQwen3AsrArtifacts,
      .create = &CreateQwen3AsrRunner,
  });
#endif
}
