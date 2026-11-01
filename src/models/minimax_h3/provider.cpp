#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <gufo/ext/generated/minimax_h3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/video.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "src/models/minimax_h3/generation.hpp"
#include "src/models/minimax_h3/probe.hpp"
#include "src/models/minimax_h3/runtime.hpp"

namespace gufo::models::minimax_h3 {

class MinimaxH3RunnerImpl : public gufo::ext::VideoRunner {
public:
  explicit MinimaxH3RunnerImpl(std::filesystem::path model_root)
      : model_root_(std::move(model_root)) {}

  bool Generate(const gufo::ext::VideoRequest& request,
                const gufo::ext::CancellationCheck& is_cancelled,
                gufo::ext::VideoProgress progress, void* progress_opaque,
                gufo::ext::VideoTelemetry* telemetry,
                std::string* error) override {
#if defined(ENGINE_ENABLE_HIP)
    gufo::minimax_h3::GenerationRequest req{
        .model_root =
            request.model_root.empty() ? model_root_ : request.model_root,
        .source_manifest = request.source_manifest,
        .output_path = request.output_path,
        .frames_directory = request.frames_directory,
        .latents_directory = request.latents_directory,
        .prompt = request.prompt,
        .seed = request.seed,
        .parameters =
            {
                .preset = request.parameters.preset,
                .internal_width = request.parameters.internal_width,
                .internal_height = request.parameters.internal_height,
                .output_width = request.parameters.output_width,
                .output_height = request.parameters.output_height,
                .frames = request.parameters.frames,
                .evaluations = request.parameters.evaluations,
                .active_blocks = request.parameters.active_blocks,
                .reuse_interval = request.parameters.reuse_interval,
                .row_parallel_attention =
                    request.parameters.row_parallel_attention,
                .decode_audio = request.parameters.decode_audio,
                .mux = request.parameters.mux,
                .selected_frames = request.parameters.selected_frames,
            },
    };
    gufo::minimax_h3::CancellationToken token;
    std::jthread watcher;
    if (is_cancelled) {
      watcher =
          std::jthread([&token, &is_cancelled](const std::stop_token& stop) {
            while (!stop.stop_requested()) {
              if (is_cancelled()) {
                token.Cancel();
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
          });
    }
    gufo::minimax_h3::GenerationTelemetry internal_telem;
    const bool ok = gufo::minimax_h3::GenerateTextVideo(
        req, &token,
        reinterpret_cast<gufo::minimax_h3::GenerationProgress>(progress),
        progress_opaque, &internal_telem, error);
    if (is_cancelled) {
      watcher.request_stop();
      if (watcher.joinable()) {
        watcher.join();
      }
    }
    if (telemetry != nullptr) {
      telemetry->inventory_ms = internal_telem.inventory_ms;
      telemetry->tokenizer_ms = internal_telem.tokenizer_ms;
      telemetry->prompt_ms = internal_telem.prompt_ms;
      telemetry->denoise_ms = internal_telem.denoiser_ms;
      telemetry->video_vae_ms = internal_telem.video_vae_ms;
      telemetry->first_preview_ms = internal_telem.first_preview_ms;
      telemetry->audio_vae_ms = internal_telem.audio_vae_ms;
      telemetry->media_ms = internal_telem.media_ms;
      telemetry->total_ms = internal_telem.total_ms;
      telemetry->prompt_tokens = internal_telem.prompt_tokens;
      telemetry->extra_json =
          gufo::minimax_h3::GenerationTelemetryJson(internal_telem);
    }
    return ok;
#else
    (void)request;
    (void)is_cancelled;
    (void)progress;
    (void)progress_opaque;
    (void)telemetry;
    if (error != nullptr) {
      *error = "MiniMax H3 generation requires HIP support";
    }
    return false;
#endif
  }

private:
  std::filesystem::path model_root_;
};

#if defined(ENGINE_ENABLE_HIP)
std::unique_ptr<ext::VideoRunner> CreateMinimaxH3Runner(
    const std::string& model_root, std::string*) {
  return std::make_unique<MinimaxH3RunnerImpl>(model_root);
}
#endif

}  // namespace gufo::models::minimax_h3

void minimax_h3_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_minimax_h3::kProvider_minimax_h3);
#if defined(ENGINE_ENABLE_HIP)
  using namespace gufo::models::minimax_h3;
  registry.AddVideoProvider({
      .id = "minimax-h3",
      .probe = &ProbeMinimaxH3,
      .resolve = &ResolveMinimaxH3Artifacts,
      .create = &CreateMinimaxH3Runner,
      .resolve_preset = &ResolvePreset,
      .resolve_duration_frames = &ResolveDuration,
      .validate_parameters = &ValidateParameters,
      .parameters_json = &ParametersJson,
      .telemetry_json = &TelemetryJson,
  });
#endif
}
