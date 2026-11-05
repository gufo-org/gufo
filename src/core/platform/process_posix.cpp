#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/platform/process.hpp"

extern char**
    environ;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

namespace gufo::platform {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;
constexpr int kAudioChunkSamples = 4096;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool EnsureSigpipeIgnored(std::string* error) {
  static std::once_flag once;
  static int failure = 0;
  std::call_once(once, [] {
    struct sigaction ignore{};
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    if (sigaction(SIGPIPE, &ignore, nullptr) != 0) {
      failure = errno;
    }
  });
  if (failure == 0) {
    return true;
  }
  if (error != nullptr) {
    *error = "cannot ignore SIGPIPE for FFmpeg writers: " +
             std::string(std::strerror(failure));
  }
  return false;
}

void CloseFd(int* descriptor) {
  if (*descriptor >= 0) {
    while (close(*descriptor) != 0 && errno == EINTR) {
    }
    *descriptor = -1;
  }
}

bool MakeNonBlocking(int descriptor, std::string* error) {
  const int flags = fcntl(descriptor, F_GETFL);
  if (flags < 0 || fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0) {
    SetError(error, "cannot configure FFmpeg media pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

bool WaitForChild(pid_t child, int* status, std::string* error) {
  while (waitpid(child, status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    SetError(error, "cannot wait for media process: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

ssize_t CancellableWrite(int descriptor, const void* data, std::size_t bytes,
                         const std::function<bool()>& is_cancelled,
                         const std::atomic<bool>& stop) {
  while (!stop.load(std::memory_order_acquire) &&
         (!is_cancelled || !is_cancelled())) {
    const ssize_t amount = write(descriptor, data, bytes);
    if (amount >= 0) {
      return amount;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return -1;
    }
    struct pollfd readiness{descriptor, POLLOUT, 0};
    const int result = poll(&readiness, 1, 50);
    if (result < 0 && errno != EINTR) {
      return -1;
    }
    if (result > 0 && (readiness.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      errno = EPIPE;
      return -1;
    }
  }
  errno = ECANCELED;
  return -1;
}

int AddCloseUnlessKept(posix_spawn_file_actions_t* actions, int descriptor,
                       int keep_a, int keep_b) {
  if (descriptor == keep_a || descriptor == keep_b) {
    return 0;
  }
  return posix_spawn_file_actions_addclose(actions, descriptor);
}

std::vector<char*> MutableArgv(std::span<std::string> storage) {
  std::vector<char*> arguments;
  arguments.reserve(storage.size() + 1);
  for (std::string& argument : storage) {
    arguments.push_back(argument.data());
  }
  arguments.push_back(nullptr);
  return arguments;
}

}  // namespace

const char* FfmpegProgram() {
  const char* override_val = std::getenv("GUFO_FFMPEG");
  return override_val != nullptr && *override_val != '\0'
             ? override_val
             : GUFO_FFMPEG_EXECUTABLE;
}

const char* FfprobeProgram() {
  const char* override_val = std::getenv("GUFO_FFPROBE");
  return override_val != nullptr && *override_val != '\0'
             ? override_val
             : GUFO_FFPROBE_EXECUTABLE;
}

std::filesystem::path PartialMediaPath(
    const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> sequence{0};
  const std::string name =
      destination.filename().string() + ".gufo-partial-" +
      std::to_string(static_cast<std::uint64_t>(getpid())) + "-" +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".mp4";
  return destination.parent_path() / name;
}

bool RunCapture(const char* program, std::span<const std::string> arguments_in,
                std::string* output, std::string* error) {
  std::vector<std::string> storage(arguments_in.begin(), arguments_in.end());
  std::vector<char*> arguments = MutableArgv(storage);
  int stream[2] = {-1, -1};
  if (pipe(stream) != 0) {
    SetError(error, "cannot create media inspection pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code = posix_spawn_file_actions_adddup2(&actions, stream[1], STDOUT_FILENO);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[0]);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[1]);
  }
  pid_t child = -1;
  if (code == 0) {
    code = posix_spawnp(&child, program, &actions, nullptr, arguments.data(),
                        environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&stream[1]);
  if (code != 0) {
    CloseFd(&stream[0]);
    SetError(error, "cannot start " + std::string(program) + ": " +
                        std::string(std::strerror(code)));
    return false;
  }
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    const ssize_t amount = read(stream[0], buffer.data(), buffer.size());
    if (amount < 0 && errno == EINTR) {
      continue;
    }
    if (amount <= 0) {
      break;
    }
    if (output->size() + static_cast<std::size_t>(amount) > (1U << 20U)) {
      CloseFd(&stream[0]);
      (void)kill(child, SIGTERM);
      int ignored = 0;
      (void)WaitForChild(child, &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), static_cast<std::size_t>(amount));
  }
  CloseFd(&stream[0]);
  int status = 0;
  if (!WaitForChild(child, &status, error)) {
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    SetError(error,
             std::string(program) + " exited with status " +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    return false;
  }
  return true;
}

bool RunMediaMux(const std::filesystem::path& destination,
                 std::span<const std::uint8_t> rgb24, int frame_count,
                 int input_width, int input_height, int output_width,
                 int output_height, int fps,
                 std::span<const float> channel_major_pcm, int samples,
                 int channels, int sample_rate,
                 const std::function<bool()>& is_cancelled,
                 MuxProcessTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  const std::size_t frame_bytes =
      static_cast<std::size_t>(input_width) * input_height * 3U;
  const std::uint64_t video_bytes =
      static_cast<std::uint64_t>(frame_count) * frame_bytes;
  const std::uint64_t audio_elements =
      static_cast<std::uint64_t>(samples) * channels;

  std::error_code filesystem_error;
  int video_pipe[2] = {-1, -1};
  int audio_pipe[2] = {-1, -1};
  if (pipe(video_pipe) != 0 || pipe(audio_pipe) != 0) {
    const int saved = errno;
    CloseFd(&video_pipe[0]);
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[0]);
    CloseFd(&audio_pipe[1]);
    SetError(error, "cannot create FFmpeg media pipes: " +
                        std::string(std::strerror(saved)));
    return false;
  }
  const int audio_target =
      std::max({video_pipe[0], video_pipe[1], audio_pipe[0], audio_pipe[1]}) +
      1;
  const std::string size =
      std::to_string(input_width) + "x" + std::to_string(input_height);
  const std::string output_size =
      std::to_string(output_width) + ":" + std::to_string(output_height);
  const std::string rate = std::to_string(fps);
  const std::string audio_rate = std::to_string(sample_rate);
  const std::string audio_channels = std::to_string(channels);
  const std::string audio_input = "pipe:" + std::to_string(audio_target);
  const std::string output_path = destination.string();
  std::vector<std::string> argument_storage = {
      "ffmpeg",
      "-y",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pixel_format",
      "rgb24",
      "-video_size",
      size,
      "-framerate",
      rate,
      "-i",
      "pipe:0",
      "-f",
      "f32le",
      "-ar",
      audio_rate,
      "-ac",
      audio_channels,
      "-i",
      audio_input,
      "-map",
      "0:v:0",
      "-map",
      "1:a:0",
      "-vf",
      "scale=" + output_size + ":flags=lanczos,setsar=1",
      "-c:v",
      "libx264",
      "-preset",
      "fast",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-c:a",
      "aac",
      "-b:a",
      "192k",
      "-movflags",
      "+faststart",
      output_path,
  };
  std::vector<char*> arguments = MutableArgv(argument_storage);
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, video_pipe[0], STDIN_FILENO);
  }
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, audio_pipe[0], audio_target);
  }
  const std::array<int, 4> descriptors = {video_pipe[0], video_pipe[1],
                                          audio_pipe[0], audio_pipe[1]};
  for (const int descriptor : descriptors) {
    if (code == 0) {
      code =
          AddCloseUnlessKept(&actions, descriptor, STDIN_FILENO, audio_target);
    }
  }
  pid_t child = -1;
  if (code == 0) {
    code = posix_spawnp(&child, FfmpegProgram(), &actions, nullptr,
                        arguments.data(), environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&video_pipe[0]);
  CloseFd(&audio_pipe[0]);
  if (code != 0) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "cannot start FFmpeg: " + std::string(std::strerror(code)));
    return false;
  }
  if (!MakeNonBlocking(video_pipe[1], error) ||
      !MakeNonBlocking(audio_pipe[1], error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    std::filesystem::remove(destination, filesystem_error);
    return false;
  }

  std::atomic<bool> stop{false};
  std::atomic<int> video_error{0};
  std::atomic<int> audio_error{0};
  std::uint64_t written_video = 0;
  std::uint64_t written_audio = 0;
  std::size_t maximum_audio_staging = 0;
  if (!EnsureSigpipeIgnored(error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    std::filesystem::remove(destination, filesystem_error);
    return false;
  }

  std::thread video_thread([&] {
    const std::uint8_t* cursor = rgb24.data();
    std::size_t remaining = rgb24.size();
    while (remaining != 0 && !stop.load(std::memory_order_acquire)) {
      if (is_cancelled && is_cancelled()) {
        stop.store(true, std::memory_order_release);
        break;
      }
      const std::size_t req = std::min(remaining, kIoChunkBytes);
      const ssize_t amount =
          CancellableWrite(video_pipe[1], cursor, req, is_cancelled, stop);
      if (amount <= 0) {
        if ((!is_cancelled || !is_cancelled()) &&
            !stop.load(std::memory_order_acquire)) {
          video_error.store(amount < 0 ? errno : EIO,
                            std::memory_order_release);
        }
        stop.store(true, std::memory_order_release);
        break;
      }
      cursor += amount;
      remaining -= static_cast<std::size_t>(amount);
      written_video += static_cast<std::uint64_t>(amount);
    }
    CloseFd(&video_pipe[1]);
  });
  std::thread audio_thread([&] {
    std::vector<float> interleaved(
        static_cast<std::size_t>(kAudioChunkSamples) * channels);
    maximum_audio_staging = interleaved.size() * sizeof(float);
    int offset = 0;
    while (offset < samples && !stop.load(std::memory_order_acquire)) {
      if (is_cancelled && is_cancelled()) {
        stop.store(true, std::memory_order_release);
        break;
      }
      const int chunk = std::min(kAudioChunkSamples, samples - offset);
      for (int sample = 0; sample < chunk; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
          interleaved[static_cast<std::size_t>(sample) * channels + channel] =
              channel_major_pcm[static_cast<std::size_t>(channel) * samples +
                                offset + sample];
        }
      }
      const std::uint8_t* cursor =
          reinterpret_cast<const std::uint8_t*>(interleaved.data());
      std::size_t remaining =
          static_cast<std::size_t>(chunk) * channels * sizeof(float);
      while (remaining != 0 && !stop.load(std::memory_order_acquire)) {
        const ssize_t amount = CancellableWrite(audio_pipe[1], cursor,
                                                remaining, is_cancelled, stop);
        if (amount <= 0) {
          if ((!is_cancelled || !is_cancelled()) &&
              !stop.load(std::memory_order_acquire)) {
            audio_error.store(amount < 0 ? errno : EIO,
                              std::memory_order_release);
          }
          stop.store(true, std::memory_order_release);
          break;
        }
        cursor += amount;
        remaining -= static_cast<std::size_t>(amount);
        written_audio += static_cast<std::uint64_t>(amount);
      }
      offset += chunk;
    }
    CloseFd(&audio_pipe[1]);
  });
  video_thread.join();
  audio_thread.join();

  const bool cancelled = is_cancelled && is_cancelled();
  if (stop.load(std::memory_order_acquire) || cancelled) {
    (void)kill(child, SIGTERM);
  }
  int status = 0;
  const bool waited = WaitForChild(child, &status, error);
  if (telemetry != nullptr) {
    telemetry->video_bytes = written_video;
    telemetry->audio_bytes = written_audio;
    telemetry->maximum_audio_staging_bytes = maximum_audio_staging;
    telemetry->child_status =
        waited && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    telemetry->cancelled = cancelled;
  }
  if (!waited) {
    std::filesystem::remove(destination, filesystem_error);
    return false;
  }
  if (cancelled) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "MiniMax H3 media mux cancelled");
    return false;
  }
  const int writer_error = video_error.load(std::memory_order_acquire) != 0
                               ? video_error.load(std::memory_order_relaxed)
                               : audio_error.load(std::memory_order_acquire);
  if (writer_error != 0) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "cannot stream generated media to FFmpeg: " +
                        std::string(std::strerror(writer_error)));
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error,
             "FFmpeg exited with status " +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    return false;
  }
  if (written_video != video_bytes ||
      written_audio != audio_elements * sizeof(float)) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "FFmpeg media streams ended before all bytes were sent");
    return false;
  }
  return true;
}

}  // namespace gufo::platform
