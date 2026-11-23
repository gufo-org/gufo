#include "src/ext/music.hpp"

#include <algorithm>
#include <cmath>

namespace gufo::ext {
namespace {

bool Fail(std::string_view name, std::string message, std::string* field,
          std::string* error) {
  if (field != nullptr)
    *field = name;
  if (error != nullptr)
    *error = std::move(message);
  return false;
}

// A field the runner does not take is rejected, a required one must be set.
bool CheckSupport(MusicSupport support, bool present, std::string_view name,
                  std::string* field, std::string* error) {
  if (support == MusicSupport::kUnsupported && present)
    return Fail(name, "this model does not accept '" + std::string(name) + "'",
                field, error);
  if (support == MusicSupport::kRequired && !present)
    return Fail(name, "this model requires '" + std::string(name) + "'", field,
                error);
  return true;
}

}  // namespace

bool ValidateMusicRequest(const MusicCapabilities& capabilities,
                          const MusicRequest& request, std::string* field,
                          std::string* error) {
  if (request.prompt.find('\0') != std::string::npos)
    return Fail("prompt", "prompt contains a NUL byte", field, error);
  if (request.lyrics.find('\0') != std::string::npos)
    return Fail("lyrics", "lyrics contain a NUL byte", field, error);
  if (!CheckSupport(capabilities.lyrics, !request.lyrics.empty(), "lyrics",
                    field, error) ||
      !CheckSupport(capabilities.language, request.language.has_value(),
                    "language", field, error) ||
      !CheckSupport(capabilities.reference_audio,
                    !request.reference_audio.samples.empty(), "reference_audio",
                    field, error) ||
      !CheckSupport(capabilities.score, !request.score.empty(), "score", field,
                    error))
    return false;
  const bool any_meter = request.bpm.has_value() || request.key.has_value() ||
                         request.beats_per_bar.has_value();
  if (capabilities.tempo_key_meter == MusicSupport::kUnsupported && any_meter)
    return Fail(request.bpm   ? "bpm"
                : request.key ? "key"
                              : "time_signature",
                "this model takes tempo, key and meter in the prompt or the "
                "score, not as fields",
                field, error);
  if (capabilities.tempo_key_meter == MusicSupport::kRequired &&
      !(request.bpm && request.key && request.beats_per_bar))
    return Fail(!request.bpm   ? "bpm"
                : !request.key ? "key"
                               : "time_signature",
                "this model requires bpm, key and time_signature", field,
                error);
  if (request.bpm && (!std::isfinite(*request.bpm) || *request.bpm <= 0.0 ||
                      *request.bpm > 1000.0))
    return Fail("bpm", "bpm must be in (0, 1000]", field, error);
  if (request.beats_per_bar &&
      (*request.beats_per_bar < 1 || *request.beats_per_bar > 64))
    return Fail("time_signature", "beats per bar must be in [1, 64]", field,
                error);
  if (!request.score.empty() || !request.score_format.empty()) {
    if (request.score.empty())
      return Fail("score", "a score format needs a score", field, error);
    if (std::ranges::find(capabilities.score_formats, request.score_format) ==
        capabilities.score_formats.end())
      return Fail("score",
                  "unsupported score format '" + request.score_format + "'",
                  field, error);
    if (request.score.find('\0') != std::string::npos)
      return Fail("score", "score contains a NUL byte", field, error);
  }
  if (request.plan_only && !capabilities.plan_only)
    return Fail("plan_only", "this model does not support plan-only requests",
                field, error);
  if (request.plan_only && !request.score.empty())
    return Fail("plan_only", "plan_only and a supplied score are exclusive",
                field, error);
  if (request.guidance && !capabilities.guidance)
    return Fail("guidance_scale", "this model does not accept guidance", field,
                error);
  if (request.guidance && !std::isfinite(*request.guidance))
    return Fail("guidance_scale", "guidance must be finite", field, error);
  if (request.steps && !capabilities.steps)
    return Fail("steps", "this model does not accept steps", field, error);
  if (request.steps && (*request.steps < 1 || *request.steps > 1000))
    return Fail("steps", "steps must be in [1, 1000]", field, error);
  const auto& duration = request.duration_seconds;
  if (duration && capabilities.duration == MusicDurationControl::kNone)
    return Fail("seconds", "this model chooses the length itself", field,
                error);
  if (!duration && capabilities.duration == MusicDurationControl::kExact)
    return Fail("seconds", "this model requires 'seconds'", field, error);
  if (duration && (!std::isfinite(*duration) || *duration <= 0.0 ||
                   (capabilities.max_seconds > 0.0 &&
                    *duration > capabilities.max_seconds)))
    return Fail("seconds",
                capabilities.max_seconds > 0.0
                    ? "seconds must be in (0, " +
                          std::to_string(static_cast<long long>(
                              capabilities.max_seconds)) +
                          "]"
                    : std::string("seconds must be positive"),
                field, error);
  return true;
}

}  // namespace gufo::ext
