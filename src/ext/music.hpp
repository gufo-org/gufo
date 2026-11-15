#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/options.hpp"
#include "src/ext/tts.hpp"

namespace gufo::ext {

/// Command whose manifest options reach `MusicRunnerOptions::options`.
inline constexpr std::string_view kMusicServeCommand = "serve music";
inline constexpr std::string_view kMusicCommand = "music";

enum class MusicSupport : std::uint8_t {
  kUnsupported,
  kOptional,
  kRequired,
};

/// What `MusicRequest::duration_seconds` means to a runner.
enum class MusicDurationControl : std::uint8_t {
  /// The output has the requested length; a request must give one.
  kExact,
  /// The output ends where the model stops, at most the requested length.
  kMaximum,
  /// The model alone chooses the length; a request must not give one.
  kNone,
};

/// Request fields a runner accepts. The service rejects every field a runner
/// declares unsupported before inference, so `Generate` never sees one.
struct MusicCapabilities {
  MusicSupport lyrics{MusicSupport::kOptional};
  /// `bpm`, `key` and `beats_per_bar`, all or none.
  MusicSupport tempo_key_meter{MusicSupport::kUnsupported};
  MusicSupport language{MusicSupport::kUnsupported};
  MusicSupport reference_audio{MusicSupport::kUnsupported};
  /// A symbolic score the runner plays instead of planning one.
  MusicSupport score{MusicSupport::kUnsupported};
  /// Accepted `MusicRequest::score_format` values, for example "abc".
  std::vector<std::string> score_formats;
  /// The runner returns the score it planned or played.
  bool score_output{false};
  /// The runner can stop after planning (`MusicRequest::plan_only`).
  bool plan_only{false};
  bool guidance{false};
  bool steps{false};
  MusicDurationControl duration{MusicDurationControl::kExact};
  /// Longest output in seconds; 0 means no declared bound.
  double max_seconds{0.0};
};

/// Frame-interleaved `[frames][channels]` float PCM at `sample_rate`.
using MusicAudio = TtsAudioBuffer;

struct MusicRequest {
  /// Style text: genre, instruments, voice, mood.
  std::string prompt;
  /// Empty means instrumental.
  std::string lyrics;
  std::optional<std::string> language;
  std::optional<double> bpm;
  std::optional<std::string> key;
  std::optional<int> beats_per_bar;
  /// Meaning set by `MusicCapabilities::duration`.
  std::optional<double> duration_seconds;
  /// Stereo or mono reference waveform; empty when none was given.
  MusicAudio reference_audio;
  /// Empty means the runner plans its own score.
  std::string score;
  std::string score_format;
  /// Stop after the score; `MusicResult::audio` stays empty.
  bool plan_only{false};
  std::optional<float> guidance;
  std::optional<int> steps;
  /// Empty means the runner's own default seed.
  std::optional<std::uint64_t> seed;
  /// JSON object of provider options, validated by the runner.
  std::string provider_json;
};

enum class MusicFinish : std::uint8_t {
  kComplete,
  /// A generation budget ended a stage early; the audio still plays.
  kTruncated,
};

struct MusicResult {
  /// Empty for a plan-only request.
  MusicAudio audio;
  std::string score;
  std::string score_format;
  MusicFinish finish{MusicFinish::kComplete};
  double total_ms{0.0};
  /// JSON object of provider timings and counters.
  std::string telemetry_json;
};

/// Reports `completed` of `total` units of `phase`; `total == 0` means the
/// total is unknown, for example while a model writes until its end token.
/// Returning false cancels generation.
using MusicProgress = std::function<bool(
    std::string_view phase, std::uint64_t completed, std::uint64_t total)>;

/// Abstract capability runner for music generation models.
class MusicRunner {
public:
  virtual ~MusicRunner() = default;

  /// A cancelled generation fails with an error and no partial result.
  [[nodiscard]] virtual bool Generate(const MusicRequest& request,
                                      const CancellationCheck& is_cancelled,
                                      const MusicProgress& progress,
                                      MusicResult* result,
                                      std::string* error) = 0;

  [[nodiscard]] virtual MusicCapabilities Capabilities() const = 0;
  /// Checks provider-specific constraints, including `provider_json`, after
  /// the service has applied `Capabilities()`. Runs before admission.
  [[nodiscard]] virtual bool ValidateRequest(const MusicRequest& request,
                                             std::string* error) const {
    (void)request;
    (void)error;
    return true;
  }
  /// Fixed output sample rate in Hz, in [1, 192000].
  [[nodiscard]] virtual std::uint32_t SampleRate() const = 0;
  [[nodiscard]] virtual std::uint32_t Channels() const { return 2; }
  /// Public model ID when the service names none; empty keeps the core
  /// default.
  [[nodiscard]] virtual std::string ModelId() const { return {}; }
  /// The runner admits concurrent Generate calls itself, so the service does
  /// not serialize jobs.
  [[nodiscard]] virtual bool Concurrent() const { return false; }
};

/// Model artifacts and validated manifest options for `serve music` and
/// `music`. Absent options keep the provider's own defaults.
struct MusicRunnerOptions {
  std::string model_root;
  ParsedOptions options;
};

using MusicProbe = bool (*)(std::string_view model_root);
using MusicArtifactResolver = bool (*)(std::string_view model_root,
                                       std::string* error);
using MusicRunnerOptionsFactory = std::unique_ptr<MusicRunner> (*)(
    const MusicRunnerOptions& options, std::string* error);

struct MusicProvider {
  std::string id;
  MusicProbe probe = nullptr;
  MusicArtifactResolver resolve = nullptr;
  MusicRunnerOptionsFactory create_with_options = nullptr;
};

void RegisterMusicProvider(MusicProvider provider);
const MusicProvider* FindMusicProvider(std::string_view id);
const MusicProvider* ProbeMusicProvider(std::string_view model_root);

/// Checks a request against declared capabilities: the service's 400 before
/// inference. Returns the offending request field in `field` (for example
/// "bpm") and a message in `error`.
[[nodiscard]] bool ValidateMusicRequest(const MusicCapabilities& capabilities,
                                        const MusicRequest& request,
                                        std::string* field, std::string* error);

}  // namespace gufo::ext
