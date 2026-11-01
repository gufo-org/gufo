#include "src/models/minimax_h3/media.hpp"

#include <charconv>
#include <cmath>
#include <gufo/ext/platform/process.hpp>
#include <string_view>
#include <vector>

namespace gufo::minimax_h3 {
namespace {

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

bool SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
  return false;
}

using KeyValues = std::vector<std::pair<std::string, std::string>>;

std::string_view Trim(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t' ||
                            value.front() == '\r' || value.front() == '\n')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
                            value.back() == '\r' || value.back() == '\n')) {
    value.remove_suffix(1);
  }
  return value;
}

KeyValues ParseKeyValues(std::string_view output) {
  KeyValues entries;
  while (!output.empty()) {
    const std::size_t end = output.find('\n');
    std::string_view line = output.substr(0, end);
    output = end == std::string_view::npos ? std::string_view{}
                                           : output.substr(end + 1);
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    const std::size_t equal = line.find('=');
    if (equal == std::string_view::npos) {
      continue;
    }
    entries.emplace_back(std::string(Trim(line.substr(0, equal))),
                         std::string(Trim(line.substr(equal + 1))));
  }
  return entries;
}

const std::string* FindValue(const KeyValues& entries, std::string_view key) {
  for (const auto& [name, value] : entries) {
    if (name == key) {
      return &value;
    }
  }
  return nullptr;
}

bool ParseInteger(std::string_view text, int* output) {
  if (text.empty()) {
    return false;
  }
  int value = 0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseDouble(std::string_view text, double* output) {
  if (text.empty() || text == "N/A") {
    *output = 0.0;
    return true;
  }
  double value = 0.0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(value)) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseRate(std::string_view text, double* output) {
  const std::size_t slash = text.find('/');
  if (slash == std::string_view::npos) {
    return ParseDouble(text, output);
  }
  double numerator = 0.0;
  double denominator = 0.0;
  if (!ParseDouble(text.substr(0, slash), &numerator) ||
      !ParseDouble(text.substr(slash + 1), &denominator) ||
      denominator == 0.0) {
    return false;
  }
  *output = numerator / denominator;
  return std::isfinite(*output);
}

}  // namespace

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          const MediaVideoOptions& video,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  if (path.empty() || frame_count <= 0 || video.input_width <= 0 ||
      video.input_height <= 0 || video.output_width <= 0 ||
      video.output_height <= 0 || video.fps <= 0 || samples <= 0 ||
      channels <= 0 || sample_rate <= 0) {
    return SetError(error, "invalid MiniMax H3 media multiplexing arguments");
  }

  std::error_code filesystem_error;
  const std::filesystem::path partial_path =
      gufo::platform::PartialMediaPath(path);
  std::filesystem::remove(partial_path, filesystem_error);

  gufo::platform::MuxProcessTelemetry mux_telemetry;
  const auto is_cancelled = [&] { return IsCancelled(cancellation); };
  const bool ok = gufo::platform::RunMediaMux(
      partial_path, rgb24, frame_count, video.input_width, video.input_height,
      video.output_width, video.output_height, video.fps, channel_major_pcm,
      samples, channels, sample_rate, is_cancelled, &mux_telemetry, error);

  if (telemetry != nullptr) {
    telemetry->video_bytes = mux_telemetry.video_bytes;
    telemetry->audio_bytes = mux_telemetry.audio_bytes;
    telemetry->maximum_audio_staging_bytes =
        mux_telemetry.maximum_audio_staging_bytes;
    telemetry->child_status = mux_telemetry.child_status;
    telemetry->cancelled = mux_telemetry.cancelled;
  }
  if (!ok) {
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }

  MediaProbe probe;
  const double expected_video_duration =
      static_cast<double>(frame_count) / video.fps;
  const double expected_audio_duration =
      static_cast<double>(samples) / sample_rate;
  if (!ProbeMediaFile(partial_path, &probe, error) ||
      probe.video_codec != "h264" || probe.audio_codec != "aac" ||
      probe.width != video.output_width ||
      probe.height != video.output_height ||
      std::abs(probe.video_frame_rate - video.fps) > 1.0e-6 ||
      probe.audio_channels != channels ||
      probe.audio_sample_rate != sample_rate ||
      std::abs(probe.video_start_seconds - probe.audio_start_seconds) >
          1.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - expected_video_duration) >
          1.0 / video.fps ||
      std::abs(probe.audio_duration_seconds - expected_audio_duration) >
          2048.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - probe.audio_duration_seconds) >
          0.075) {
    std::filesystem::remove(partial_path, filesystem_error);
    if (error != nullptr && error->empty()) {
      *error = "generated MiniMax H3 media violates the delivery contract";
    }
    return false;
  }

  std::filesystem::rename(partial_path, path, filesystem_error);
  if (filesystem_error) {
    const std::string message = filesystem_error.message();
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot publish generated media: " + message);
    return false;
  }
  return true;
}

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          int width, int height, int fps,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  return WriteSynchronizedMp4(path, rgb24, frame_count,
                              {.input_width = width,
                               .input_height = height,
                               .output_width = width,
                               .output_height = height,
                               .fps = fps},
                              channel_major_pcm, samples, channels, sample_rate,
                              cancellation, telemetry, error);
}

bool ProbeMediaFile(const std::filesystem::path& path, MediaProbe* probe,
                    std::string* error) {
  if (path.empty() || probe == nullptr) {
    return SetError(error, "invalid MiniMax H3 media probe arguments");
  }
  *probe = {};
  const std::string path_text = path.string();
  std::string video_output;
  std::vector<std::string> video_arguments = {
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=codec_name,width,height,r_frame_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path_text,
  };
  if (!gufo::platform::RunCapture(gufo::platform::FfprobeProgram(),
                                  video_arguments, &video_output, error)) {
    return false;
  }
  const KeyValues video = ParseKeyValues(video_output);
  const std::string* video_codec = FindValue(video, "codec_name");
  const std::string* width = FindValue(video, "width");
  const std::string* height = FindValue(video, "height");
  const std::string* frame_rate = FindValue(video, "r_frame_rate");
  const std::string* video_start = FindValue(video, "start_time");
  const std::string* video_duration = FindValue(video, "duration");
  if (video_codec == nullptr || width == nullptr || height == nullptr ||
      frame_rate == nullptr || video_start == nullptr ||
      video_duration == nullptr || !ParseInteger(*width, &probe->width) ||
      !ParseInteger(*height, &probe->height) ||
      !ParseRate(*frame_rate, &probe->video_frame_rate) ||
      !ParseDouble(*video_start, &probe->video_start_seconds) ||
      !ParseDouble(*video_duration, &probe->video_duration_seconds)) {
    return SetError(error,
                    "FFprobe returned malformed MiniMax H3 video metadata");
  }
  probe->video_codec = *video_codec;

  std::string audio_output;
  std::vector<std::string> audio_arguments = {
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "a:0",
      "-show_entries",
      "stream=codec_name,channels,sample_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path_text,
  };
  if (!gufo::platform::RunCapture(gufo::platform::FfprobeProgram(),
                                  audio_arguments, &audio_output, error)) {
    return false;
  }
  const KeyValues audio = ParseKeyValues(audio_output);
  const std::string* audio_codec = FindValue(audio, "codec_name");
  const std::string* audio_channels = FindValue(audio, "channels");
  const std::string* audio_rate = FindValue(audio, "sample_rate");
  const std::string* audio_start = FindValue(audio, "start_time");
  const std::string* audio_duration = FindValue(audio, "duration");
  if (audio_codec == nullptr || audio_channels == nullptr ||
      audio_rate == nullptr || audio_start == nullptr ||
      audio_duration == nullptr ||
      !ParseInteger(*audio_channels, &probe->audio_channels) ||
      !ParseInteger(*audio_rate, &probe->audio_sample_rate) ||
      !ParseDouble(*audio_start, &probe->audio_start_seconds) ||
      !ParseDouble(*audio_duration, &probe->audio_duration_seconds)) {
    return SetError(error,
                    "FFprobe returned malformed MiniMax H3 audio metadata");
  }
  probe->audio_codec = *audio_codec;
  return true;
}

}  // namespace gufo::minimax_h3
