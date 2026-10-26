#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "src/core/platform/env.hpp"
#include "src/models/minimax_h3/media.hpp"

namespace {

using gufo::minimax_h3::CancellationToken;
using gufo::minimax_h3::MediaMuxTelemetry;
using gufo::minimax_h3::MediaProbe;
using gufo::minimax_h3::MediaVideoOptions;
using gufo::minimax_h3::ProbeMediaFile;
using gufo::minimax_h3::WriteSynchronizedMp4;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL minimax_h3_media_test: " << message << '\n';
  std::exit(1);
}

std::vector<std::uint8_t> MakeFrames(int frames, int width, int height) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(frames) * width *
                                height * 3U);
  for (int frame = 0; frame < frames; ++frame) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const std::size_t pixel =
            ((static_cast<std::size_t>(frame) * height + y) * width + x) * 3U;
        rgb[pixel] = static_cast<std::uint8_t>((x * 255) / (width - 1));
        rgb[pixel + 1] = static_cast<std::uint8_t>((y * 255) / (height - 1));
        rgb[pixel + 2] =
            static_cast<std::uint8_t>((frame * 255) / (frames - 1));
      }
    }
  }
  return rgb;
}

std::vector<float> MakePcm(int samples) {
  std::vector<float> pcm(static_cast<std::size_t>(samples) * 2U);
  constexpr double kTau = 6.28318530717958647692;
  for (int sample = 0; sample < samples; ++sample) {
    pcm[sample] =
        0.08F * static_cast<float>(std::sin(
                    kTau * 220.0 * static_cast<double>(sample) / 32000.0));
    pcm[static_cast<std::size_t>(samples) + sample] =
        0.08F * static_cast<float>(std::sin(
                    kTau * 330.0 * static_cast<double>(sample) / 32000.0));
  }
  return pcm;
}

void WriteSentinel(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
  output << "existing-user-media";
  if (!output) {
    Fail("cannot create media lifecycle sentinel");
  }
}

bool HasSentinel(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>()) == "existing-user-media";
}

bool HasPartialFor(const std::filesystem::path& path) {
  const std::string prefix = path.filename().string() + ".gufo-partial-";
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(path.parent_path())) {
    if (entry.path().filename().string().starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  constexpr int kFrames = 22;
  constexpr int kWidth = 32;
  constexpr int kHeight = 32;
  constexpr int kOutputWidth = 64;
  constexpr int kOutputHeight = 48;
  constexpr int kSamples = 29600;
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "gufo-h3-media-test.mp4";
  const std::filesystem::path cancelled_output =
      std::filesystem::temp_directory_path() /
      "gufo-h3-media-cancelled-test.mp4";
  const std::filesystem::path error_output =
      std::filesystem::temp_directory_path() / "gufo-h3-media-error-test.mp4";
  const std::filesystem::path probe_error_output =
      std::filesystem::temp_directory_path() /
      "gufo-h3-media-probe-error-test.mp4";
  std::error_code ignored;
  std::filesystem::remove(output, ignored);
  std::filesystem::remove(cancelled_output, ignored);
  std::filesystem::remove(error_output, ignored);
  std::filesystem::remove(probe_error_output, ignored);
  const std::vector<std::uint8_t> frames = MakeFrames(kFrames, kWidth, kHeight);
  const std::vector<float> pcm = MakePcm(kSamples);
  WriteSentinel(output);
  MediaMuxTelemetry telemetry;
  std::string error;
  if (!WriteSynchronizedMp4(output, frames, kFrames,
                            MediaVideoOptions{.input_width = kWidth,
                                              .input_height = kHeight,
                                              .output_width = kOutputWidth,
                                              .output_height = kOutputHeight,
                                              .fps = 24},
                            pcm, kSamples, 2, 32000, nullptr, &telemetry,
                            &error)) {
    if (error.find("cannot start FFmpeg") != std::string::npos) {
      std::cout << "SKIP: " << error << '\n';
      return 77;
    }
    Fail(error);
  }
  if (telemetry.video_bytes != frames.size() ||
      telemetry.audio_bytes != pcm.size() * sizeof(float) ||
      telemetry.maximum_audio_staging_bytes > 4096U * 2U * sizeof(float) ||
      telemetry.child_status != 0 || telemetry.cancelled) {
    Fail("media mux telemetry violates bounded streaming contract");
  }
  MediaProbe probe;
  if (!ProbeMediaFile(output, &probe, &error)) {
    Fail(error);
  }
  if (HasPartialFor(output)) {
    Fail("successful media publish retained a partial file");
  }
  if (probe.video_codec != "h264" || probe.audio_codec != "aac" ||
      probe.width != kOutputWidth || probe.height != kOutputHeight ||
      std::abs(probe.video_frame_rate - 24.0) > 1.0e-6 ||
      probe.audio_channels != 2 || probe.audio_sample_rate != 32000 ||
      std::abs(probe.video_start_seconds - probe.audio_start_seconds) >
          1.0 / 32000.0 ||
      std::abs(probe.video_duration_seconds -
               static_cast<double>(kFrames) / 24.0) > 1.0 / 24.0 ||
      std::abs(probe.audio_duration_seconds -
               static_cast<double>(kSamples) / 32000.0) > 2048.0 / 32000.0 ||
      std::abs(probe.video_duration_seconds - probe.audio_duration_seconds) >
          0.075) {
    Fail("FFprobe A/V geometry or synchronization gate failed");
  }

  CancellationToken pre_cancelled;
  pre_cancelled.Cancel();
  if (WriteSynchronizedMp4(cancelled_output, frames, kFrames, kWidth, kHeight,
                           24, pcm, kSamples, 2, 32000, &pre_cancelled,
                           &telemetry, &error) ||
      !telemetry.cancelled || std::filesystem::exists(cancelled_output) ||
      HasPartialFor(cancelled_output)) {
    Fail("pre-spawn media cancellation was not clean");
  }

  const std::vector<std::uint8_t> long_frames = MakeFrames(60, 512, 512);
  const std::vector<float> long_pcm = MakePcm(80000);
  WriteSentinel(cancelled_output);
  CancellationToken active_cancel;
  bool active_result = true;
  std::thread mux([&] {
    active_result = WriteSynchronizedMp4(cancelled_output, long_frames, 60, 512,
                                         512, 24, long_pcm, 80000, 2, 32000,
                                         &active_cancel, &telemetry, &error);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  active_cancel.Cancel();
  mux.join();
  if (active_result || !telemetry.cancelled || !HasSentinel(cancelled_output) ||
      HasPartialFor(cancelled_output)) {
    Fail("active FFmpeg cancellation did not preserve the prior output");
  }
  std::filesystem::remove(cancelled_output, ignored);

  const char* saved = std::getenv("GUFO_FFMPEG");
  const std::string saved_value = saved == nullptr ? "" : saved;
  WriteSentinel(error_output);
  gufo::platform::SetEnv("GUFO_FFMPEG", "/bin/false");
  const bool error_result = WriteSynchronizedMp4(
      error_output, frames, kFrames, kWidth, kHeight, 24, pcm, kSamples, 2,
      32000, nullptr, &telemetry, &error);
  if (saved == nullptr) {
    gufo::platform::UnsetEnv("GUFO_FFMPEG");
  } else {
    gufo::platform::SetEnv("GUFO_FFMPEG", saved_value);
  }
  if (error_result || !HasSentinel(error_output) ||
      HasPartialFor(error_output)) {
    Fail("FFmpeg child error did not preserve the prior output");
  }

  const char* saved_probe = std::getenv("GUFO_FFPROBE");
  const std::string saved_probe_value =
      saved_probe == nullptr ? "" : saved_probe;
  WriteSentinel(probe_error_output);
  gufo::platform::SetEnv("GUFO_FFPROBE", "/bin/false");
  const bool probe_error_result = WriteSynchronizedMp4(
      probe_error_output, frames, kFrames, kWidth, kHeight, 24, pcm, kSamples,
      2, 32000, nullptr, &telemetry, &error);
  if (saved_probe == nullptr) {
    gufo::platform::UnsetEnv("GUFO_FFPROBE");
  } else {
    gufo::platform::SetEnv("GUFO_FFPROBE", saved_probe_value);
  }
  if (probe_error_result || !HasSentinel(probe_error_output) ||
      HasPartialFor(probe_error_output)) {
    Fail("FFprobe validation error did not preserve the prior output");
  }
  std::filesystem::remove(error_output, ignored);
  std::filesystem::remove(probe_error_output, ignored);
  std::filesystem::remove(output, ignored);
  std::cout
      << "ok: bounded concurrent RGB24/PCM mux produced synchronized H.264/AAC"
      << '\n';
  return 0;
}
