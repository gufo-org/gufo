#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::ext {

struct VideoParameters {
  std::string preset;
  int internal_width{0};
  int internal_height{0};
  int output_width{0};
  int output_height{0};
  int frames{0};
  int evaluations{0};
  int active_blocks{0};
  int reuse_interval{0};
  bool row_parallel_attention{true};
  bool decode_audio{true};
  bool mux{true};
  std::vector<int> selected_frames;
};

struct VideoRequest {
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path output_path;
  std::filesystem::path frames_directory;
  std::filesystem::path latents_directory;
  std::string prompt;
  std::uint64_t seed{0};
  VideoParameters parameters;
  /// Optional keyframe conditioning anchors (issue #177). Empty paths select
  /// the text-only path, which must stay bit-identical. When set, the file is
  /// a PNG/JPEG/WebP image in display orientation, staged by the caller into a
  /// location owned for the request's lifetime (server jobs stage inside the
  /// job directory so expiry sweeps reclaim it). `first_frame_path` anchors the
  /// first generated frame, `last_frame_path` the final frame; both may be
  /// supplied together. With an anchor, `parameters` geometry is a size
  /// budget: a runner may generate at the anchor's aspect ratio within it
  /// and reports the generated canvas in its telemetry. Runners that do not
  /// implement keyframe conditioning MUST report an error rather than
  /// silently ignore these anchors.
  std::filesystem::path first_frame_path;
  std::filesystem::path last_frame_path;
};

struct VideoTelemetry {
  double inventory_ms{0.0};
  double tokenizer_ms{0.0};
  double prompt_ms{0.0};
  double denoise_ms{0.0};
  double video_vae_ms{0.0};
  double first_preview_ms{0.0};
  double audio_vae_ms{0.0};
  double media_ms{0.0};
  double total_ms{0.0};
  std::size_t prompt_tokens{0};
  std::string extra_json;
};

struct VideoResult {
  bool success{false};
  VideoTelemetry telemetry;
  std::string error;
};

using VideoProgress = void (*)(std::string_view phase, int completed, int total,
                               void* opaque);
using CancellationCheck = std::function<bool()>;

/// Abstract capability runner for video generation models.
class VideoRunner {
public:
  virtual ~VideoRunner() = default;

  [[nodiscard]] virtual bool Generate(const VideoRequest& request,
                                      const CancellationCheck& is_cancelled,
                                      VideoProgress progress,
                                      void* progress_opaque,
                                      VideoTelemetry* telemetry,
                                      std::string* error) = 0;
};

using VideoProbe = bool (*)(std::string_view model_root);
using VideoArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string_view source_manifest,
                                       std::string* error);
using VideoRunnerFactory = std::unique_ptr<VideoRunner> (*)(
    const std::string& model_root, std::string* error);

struct VideoProvider {
  std::string id;
  VideoProbe probe = nullptr;
  VideoArtifactResolver resolve = nullptr;
  VideoRunnerFactory create = nullptr;
  std::optional<VideoParameters> (*resolve_preset)(
      std::string_view name, std::string* error) = nullptr;
  std::optional<int> (*resolve_duration_frames)(std::string_view seconds,
                                                std::string* error) = nullptr;
  bool (*validate_parameters)(const VideoParameters& parameters,
                              std::string* error) = nullptr;
  std::string (*parameters_json)(const VideoParameters& parameters,
                                 std::uint64_t seed,
                                 std::string_view prompt_sha256) = nullptr;
  std::string (*telemetry_json)(const VideoTelemetry& telemetry) = nullptr;
};

void RegisterVideoProvider(VideoProvider provider);
const VideoProvider* FindVideoProvider(std::string_view id);
const VideoProvider* ProbeVideoProvider(std::string_view model_root);

}  // namespace gufo::ext
