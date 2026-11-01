#include <gufo/ext/generated/qwen3_tts_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "config.hpp"
#include "hip/synthesis_runtime.hpp"
#include "probe.hpp"

namespace gufo::models::qwen3_tts {

#if defined(ENGINE_ENABLE_HIP)
class Qwen3TtsRunnerImpl : public ext::TtsRunner {
public:
  Qwen3TtsRunnerImpl(std::unique_ptr<hip::SynthesisHipRuntime> runtime,
                     ext::TtsVariant variant, std::vector<std::string> voices)
      : runtime_(std::move(runtime)),
        variant_(variant),
        voices_(std::move(voices)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    return runtime_->Generate(request, is_cancelled, result, error);
  }

  std::vector<std::string> Voices() const override {
    if (!voices_.empty()) {
      return voices_;
    }
    if (variant_ == ext::TtsVariant::kVoiceDesign) {
      return {"voice-design"};
    }
    if (variant_ == ext::TtsVariant::kBase) {
      return {"voice-clone"};
    }
    return {};
  }

  ext::TtsVariant Variant() const override { return variant_; }

private:
  std::unique_ptr<hip::SynthesisHipRuntime> runtime_;
  ext::TtsVariant variant_;
  std::vector<std::string> voices_;
};

std::unique_ptr<ext::TtsRunner> CreateQwen3TtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error) {
  const auto config = LoadModelConfigFromPath(model_root);
  if (!config.has_value()) {
    if (error != nullptr) {
      *error = "cannot load config from " + model_root;
    }
    return nullptr;
  }
  auto runtime = hip::SynthesisHipRuntime::Create(
      model_root, maximum_context_tokens, config->variant, error);
  if (!runtime) {
    return nullptr;
  }
  std::vector<std::string> voices;
  for (const auto& [spk, id] : config->talker.spk_id) {
    voices.push_back(spk);
  }
  ext::TtsVariant ext_variant = config->variant == ModelVariant::kBase
                                    ? ext::TtsVariant::kBase
                                : config->variant == ModelVariant::kVoiceDesign
                                    ? ext::TtsVariant::kVoiceDesign
                                : config->variant == ModelVariant::kCustomVoice
                                    ? ext::TtsVariant::kCustomVoice
                                    : ext::TtsVariant::kUnsupported;
  return std::make_unique<Qwen3TtsRunnerImpl>(std::move(runtime), ext_variant,
                                              std::move(voices));
}
#endif

}  // namespace gufo::models::qwen3_tts

void qwen3_tts_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_qwen3_tts::kProvider_qwen3_tts);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::qwen3_tts;
  registry.AddTtsProvider({
      .id = "qwen3-tts",
      .probe = &ProbeQwen3Tts,
      .resolve = &ResolveQwen3TtsArtifacts,
      .create = &CreateQwen3TtsRunner,
  });
#endif
}
