#include <filesystem>
#include <gufo/ext/music.hpp>
#include <gufo/ext/registry.hpp>
#include <string>
#include <string_view>

#include "config.hpp"
#include "probe.hpp"

namespace gufo::models::yue2 {

bool ProbeYue2(std::string_view model_root) {
  return LooksLikeYue2(std::filesystem::path(model_root));
}

bool ResolveYue2Artifacts(std::string_view model_root, std::string* error) {
  BackboneConfig backbone;
  GenerationDefaults defaults;
  // The decoder may come from --vae-model, so only the backbone is checked.
  const auto model = FindModelDirectory(std::filesystem::path(model_root));
  if (!model) {
    if (error != nullptr)
      *error = "not a YuE2 model directory: " + std::string(model_root);
    return false;
  }
  return LoadBackboneConfig(*model, &backbone, &defaults, error);
}

}  // namespace gufo::models::yue2

void yue2_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::yue2;
  registry.AddMusicProvider({
      .id = "yue2",
      .probe = &ProbeYue2,
      .resolve = &ResolveYue2Artifacts,
      .create_with_options = nullptr,
  });
}
