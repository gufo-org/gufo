#ifndef GUFO_CORE_PLATFORM_PROCESS_HPP_
#define GUFO_CORE_PLATFORM_PROCESS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::platform {

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

[[nodiscard]] const char* FfmpegProgram();
[[nodiscard]] const char* FfprobeProgram();

[[nodiscard]] std::filesystem::path PartialMediaPath(
    const std::filesystem::path& destination);

[[nodiscard]] bool RunCapture(const char* program,
                              std::span<const std::string> arguments,
                              std::string* output,
                              std::string* error = nullptr);

struct MuxProcessTelemetry {
  std::uint64_t video_bytes{0};
  std::uint64_t audio_bytes{0};
  std::size_t maximum_audio_staging_bytes{0};
  int child_status{-1};
  bool cancelled{false};
};

[[nodiscard]] bool RunMediaMux(
    const std::filesystem::path& destination,
    std::span<const std::uint8_t> rgb24, int frame_count, int input_width,
    int input_height, int output_width, int output_height, int fps,
    std::span<const float> channel_major_pcm, int samples, int channels,
    int sample_rate, const std::function<bool()>& is_cancelled,
    MuxProcessTelemetry* telemetry, std::string* error = nullptr);

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PROCESS_HPP_
