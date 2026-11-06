#include <gufo/ext/asr.hpp>
#include <gufo/ext/registry.hpp>
#include <string>
#include <string_view>

#include "config.hpp"
#include "loader.hpp"
#include "probe.hpp"

namespace gufo::models::qwen3_asr {

bool ProbeQwen3Asr(std::string_view model_root) {
  return LooksLikeQwen3Asr(std::string(model_root));
}

bool ResolveQwen3AsrArtifacts(std::string_view model_root, std::string* error) {
  const auto config = LoadModelConfigFromPath(std::string(model_root));
  if (!config.has_value()) {
    if (error != nullptr) {
      *error = "cannot load config.json from " + std::string(model_root);
    }
    return false;
  }
  if (!IsSupportedModelConfig(*config)) {
    if (error != nullptr) {
      *error = "unsupported Qwen3-ASR model in " + std::string(model_root);
    }
    return false;
  }
  return true;
}

}  // namespace gufo::models::qwen3_asr

void qwen3_asr_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::qwen3_asr;
  registry.AddAsrProvider({
      .id = "qwen3-asr",
      .probe = &ProbeQwen3Asr,
      .resolve = &ResolveQwen3AsrArtifacts,
      .create = nullptr,
  });
  registry.AddAsrProvider({
      .id = "qwen3-asr-1.7b",
      .probe = &ProbeQwen3Asr,
      .resolve = &ResolveQwen3AsrArtifacts,
      .create = nullptr,
  });
}
