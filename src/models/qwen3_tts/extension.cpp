#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>
#include <string>
#include <string_view>

#include "config.hpp"
#include "loader.hpp"
#include "probe.hpp"

namespace gufo::models::qwen3_tts {

bool ProbeQwen3Tts(std::string_view model_root) {
  return LooksLikeQwen3Tts(std::string(model_root));
}

bool ResolveQwen3TtsArtifacts(std::string_view model_root, std::string* error) {
  const auto config = LoadModelConfigFromPath(std::string(model_root));
  if (!config.has_value()) {
    if (error != nullptr) {
      *error = "cannot load config.json from " + std::string(model_root);
    }
    return false;
  }
  if (!IsSupportedModelConfig(*config)) {
    if (error != nullptr) {
      *error = "unsupported Qwen3-TTS model in " + std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen3_tts

void qwen3_tts_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::qwen3_tts;
  registry.AddTtsProvider({
      .id = "qwen3-tts",
      .probe = &ProbeQwen3Tts,
      .resolve = &ResolveQwen3TtsArtifacts,
      .create = nullptr,
  });
}
