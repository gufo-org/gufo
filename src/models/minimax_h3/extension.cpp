#include <cstdlib>
#include <filesystem>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/video.hpp>
#include <string>
#include <string_view>

#include "src/models/minimax_h3/generation.hpp"
#include "src/models/minimax_h3/probe.hpp"
#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::models::minimax_h3 {

bool ProbeMinimaxH3(std::string_view model_root) {
  try {
    const std::filesystem::path root(model_root);
    if (!std::filesystem::is_directory(root)) {
      return false;
    }
    if (std::filesystem::exists(root / "FL2VA" / "transformer" /
                                "config.json") ||
        std::filesystem::exists(root / "FL2VA" / "model_index.json") ||
        std::filesystem::exists(root / "model_index.json")) {
      return true;
    }
    return false;
  } catch (...) {
    return false;
  }
}

bool ResolveMinimaxH3Artifacts(std::string_view model_root,
                               std::string_view source_manifest,
                               std::string* error) {
  if (!ProbeMinimaxH3(model_root)) {
    if (error != nullptr) {
      *error =
          "not a valid MiniMax H3 model directory: " + std::string(model_root);
    }
    return false;
  }
  std::filesystem::path manifest_path(source_manifest);
  if (manifest_path.empty()) {
    const char* env_manifest = std::getenv("GUFO_H3_SOURCE_MANIFEST");
    if (env_manifest != nullptr && *env_manifest != '\0') {
      manifest_path = env_manifest;
    }
#if defined(GUFO_H3_SOURCE_MANIFEST)
    if (manifest_path.empty()) {
      manifest_path = GUFO_H3_SOURCE_MANIFEST;
    }
#endif
  }
  std::error_code ec;
  if (!manifest_path.empty() &&
      std::filesystem::is_regular_file(manifest_path, ec)) {
    return gufo::minimax_h3::ModelInventory::Inspect(model_root, manifest_path,
                                                     error)
        .has_value();
  }
  return true;
}

std::optional<ext::VideoParameters> ResolvePreset(std::string_view name,
                                                  std::string* error) {
  auto params = gufo::minimax_h3::ResolveGenerationPreset(name, error);
  if (!params) {
    return std::nullopt;
  }
  ext::VideoParameters result;
  result.preset = std::move(params->preset);
  result.internal_width = params->internal_width;
  result.internal_height = params->internal_height;
  result.output_width = params->output_width;
  result.output_height = params->output_height;
  result.frames = params->frames;
  result.evaluations = params->evaluations;
  result.active_blocks = params->active_blocks;
  result.reuse_interval = params->reuse_interval;
  result.row_parallel_attention = params->row_parallel_attention;
  result.decode_audio = params->decode_audio;
  result.mux = params->mux;
  result.selected_frames = std::move(params->selected_frames);
  return result;
}

std::optional<int> ResolveDuration(std::string_view seconds,
                                   std::string* error) {
  return gufo::minimax_h3::ResolveDurationFrames(seconds, error);
}

bool ValidateParameters(const ext::VideoParameters& p, std::string* error) {
  gufo::minimax_h3::GenerationParameters internal_p{
      .preset = p.preset,
      .internal_width = p.internal_width,
      .internal_height = p.internal_height,
      .output_width = p.output_width,
      .output_height = p.output_height,
      .frames = p.frames,
      .evaluations = p.evaluations,
      .active_blocks = p.active_blocks,
      .reuse_interval = p.reuse_interval,
      .row_parallel_attention = p.row_parallel_attention,
      .decode_audio = p.decode_audio,
      .mux = p.mux,
      .selected_frames = p.selected_frames,
  };
  return gufo::minimax_h3::ValidateGenerationParameters(internal_p, error);
}

std::string ParametersJson(const ext::VideoParameters& p, std::uint64_t seed,
                           std::string_view prompt_sha256) {
  gufo::minimax_h3::GenerationParameters internal_p{
      .preset = p.preset,
      .internal_width = p.internal_width,
      .internal_height = p.internal_height,
      .output_width = p.output_width,
      .output_height = p.output_height,
      .frames = p.frames,
      .evaluations = p.evaluations,
      .active_blocks = p.active_blocks,
      .reuse_interval = p.reuse_interval,
      .row_parallel_attention = p.row_parallel_attention,
      .decode_audio = p.decode_audio,
      .mux = p.mux,
      .selected_frames = p.selected_frames,
  };
  return gufo::minimax_h3::GenerationParametersJson(internal_p, seed,
                                                    prompt_sha256);
}

std::string TelemetryJson(const ext::VideoTelemetry& telemetry) {
  if (!telemetry.extra_json.empty()) {
    return telemetry.extra_json;
  }
  return "{}";
}

}  // namespace gufo::models::minimax_h3

void minimax_h3_register(gufo::ext::Registry& registry) {
  using namespace gufo::models::minimax_h3;
  registry.AddVideoProvider({
      .id = "minimax-h3",
      .probe = &ProbeMinimaxH3,
      .resolve = &ResolveMinimaxH3Artifacts,
      .create = nullptr,
      .resolve_preset = &ResolvePreset,
      .resolve_duration_frames = &ResolveDuration,
      .validate_parameters = &ValidateParameters,
      .parameters_json = &ParametersJson,
      .telemetry_json = &TelemetryJson,
  });
}
