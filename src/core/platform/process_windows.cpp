#include "src/core/platform/process.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace gufo::platform {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;
constexpr int kAudioChunkSamples = 4096;
constexpr DWORD kPollMilliseconds = 50;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string WindowsMessage(DWORD code) {
  return std::system_category().message(static_cast<int>(code));
}

std::wstring Widen(std::string_view text) {
  if (text.empty()) {
    return {};
  }
  const int length = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(std::max(length, 0)), L'\0');
  if (length > 0) {
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        wide.data(), length);
  }
  return wide;
}

class Handle {
public:
  Handle() = default;
  explicit Handle(HANDLE handle)
      : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
  ~Handle() { Reset(); }
  Handle(Handle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  [[nodiscard]] HANDLE get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }
  void Reset() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
      handle_ = nullptr;
    }
  }

private:
  HANDLE handle_{nullptr};
};

void AppendArgument(std::wstring* line, const std::wstring& argument) {
  if (!line->empty()) {
    line->push_back(L' ');
  }
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    line->append(argument);
    return;
  }
  line->push_back(L'"');
  for (auto it = argument.begin();; ++it) {
    std::size_t backslashes = 0;
    while (it != argument.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == argument.end()) {
      line->append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      line->append(backslashes * 2 + 1, L'\\');
    } else {
      line->append(backslashes, L'\\');
    }
    line->push_back(*it);
  }
  line->push_back(L'"');
}

Handle InheritableNul(DWORD access) {
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  return Handle(CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &attributes, OPEN_EXISTING, 0, nullptr));
}

Handle InheritableStandard(DWORD which) {
  const HANDLE source = GetStdHandle(which);
  HANDLE copy = nullptr;
  if (source != nullptr && source != INVALID_HANDLE_VALUE &&
      DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &copy,
                      0, TRUE, DUPLICATE_SAME_ACCESS)) {
    return Handle(copy);
  }
  return InheritableNul(GENERIC_WRITE);
}

bool Spawn(const char* program, std::string_view label,
           const std::vector<std::wstring>& arguments, HANDLE output,
           Handle* process, std::string* error) {
  std::wstring line;
  AppendArgument(&line, Widen(program));
  for (const std::wstring& argument : arguments) {
    AppendArgument(&line, argument);
  }
  Handle input = InheritableNul(GENERIC_READ);
  Handle parent_output;
  if (output == nullptr) {
    parent_output = InheritableStandard(STD_OUTPUT_HANDLE);
    output = parent_output.get();
  }
  Handle errors = InheritableStandard(STD_ERROR_HANDLE);
  if (!input || output == nullptr || !errors) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::array<HANDLE, 3> inherited = {input.get(), output, errors.get()};
  SIZE_T attribute_bytes = 0;
  (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  std::vector<unsigned char> attribute_storage(attribute_bytes);
  auto* attributes =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = output;
  startup.StartupInfo.hStdError = errors.get();
  startup.lpAttributeList = attributes;
  PROCESS_INFORMATION info{};
  const bool created =
      UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
          sizeof(inherited), nullptr, nullptr) &&
      CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                     EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                     nullptr, &startup.StartupInfo, &info);
  const DWORD failure = created ? ERROR_SUCCESS : GetLastError();
  DeleteProcThreadAttributeList(attributes);
  if (!created) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(failure));
    return false;
  }
  CloseHandle(info.hThread);
  *process = Handle(info.hProcess);
  return true;
}

bool WaitForChild(HANDLE child, int* status, std::string* error) {
  DWORD code = 0;
  if (WaitForSingleObject(child, INFINITE) != WAIT_OBJECT_0 ||
      !GetExitCodeProcess(child, &code)) {
    SetError(error, "cannot wait for media process: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  *status = static_cast<int>(code);
  return true;
}

Handle CreateOutboundPipe(const std::wstring& name) {
  return Handle(
      CreateNamedPipeW(name.c_str(),
                       PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED |
                           FILE_FLAG_FIRST_PIPE_INSTANCE,
                       PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                       1, static_cast<DWORD>(kIoChunkBytes), 0, 0, nullptr));
}

bool AwaitOverlapped(HANDLE pipe, OVERLAPPED* overlapped, HANDLE child,
                     const std::function<bool()>& is_cancelled,
                     const std::atomic<bool>& stop, DWORD* transferred,
                     DWORD* failure) {
  const std::array<HANDLE, 2> waits = {overlapped->hEvent, child};
  while (true) {
    const DWORD result =
        WaitForMultipleObjects(static_cast<DWORD>(waits.size()), waits.data(),
                               FALSE, kPollMilliseconds);
    if (result == WAIT_OBJECT_0) {
      if (GetOverlappedResult(pipe, overlapped, transferred, FALSE)) {
        return true;
      }
      *failure = GetLastError();
      return false;
    }
    const DWORD wait_failure =
        result == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
    const bool child_exited = result == WAIT_OBJECT_0 + 1;
    if (child_exited || result == WAIT_FAILED ||
        stop.load(std::memory_order_acquire) ||
        (is_cancelled && is_cancelled())) {
      CancelIoEx(pipe, overlapped);
      if (GetOverlappedResult(pipe, overlapped, transferred, TRUE)) {
        return true;
      }
      *failure = child_exited            ? ERROR_BROKEN_PIPE
                 : result == WAIT_FAILED ? wait_failure
                                         : ERROR_OPERATION_ABORTED;
      return false;
    }
  }
}

bool ConnectPipe(HANDLE pipe, HANDLE child,
                 const std::function<bool()>& is_cancelled,
                 const std::atomic<bool>& stop, DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  OVERLAPPED overlapped{};
  overlapped.hEvent = event.get();
  if (ConnectNamedPipe(pipe, &overlapped)) {
    return true;
  }
  const DWORD code = GetLastError();
  if (code == ERROR_PIPE_CONNECTED) {
    return true;
  }
  if (code != ERROR_IO_PENDING) {
    *failure = code;
    return false;
  }
  DWORD ignored = 0;
  return AwaitOverlapped(pipe, &overlapped, child, is_cancelled, stop, &ignored,
                         failure);
}

bool WriteAll(HANDLE pipe, const std::uint8_t* data, std::size_t bytes,
              HANDLE child, const std::function<bool()>& is_cancelled,
              const std::atomic<bool>& stop, std::uint64_t* written,
              DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  while (bytes != 0) {
    if (stop.load(std::memory_order_acquire) ||
        (is_cancelled && is_cancelled())) {
      *failure = ERROR_OPERATION_ABORTED;
      return false;
    }
    OVERLAPPED overlapped{};
    overlapped.hEvent = event.get();
    ResetEvent(event.get());
    const auto request =
        static_cast<DWORD>(std::min<std::size_t>(bytes, kIoChunkBytes));
    if (!WriteFile(pipe, data, request, nullptr, &overlapped)) {
      const DWORD code = GetLastError();
      if (code != ERROR_IO_PENDING) {
        *failure = code;
        return false;
      }
    }
    DWORD amount = 0;
    if (!AwaitOverlapped(pipe, &overlapped, child, is_cancelled, stop, &amount,
                         failure)) {
      return false;
    }
    if (amount == 0) {
      *failure = ERROR_BROKEN_PIPE;
      return false;
    }
    data += amount;
    bytes -= amount;
    *written += amount;
  }
  return true;
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
      std::to_string(static_cast<std::uint64_t>(GetCurrentProcessId())) + "-" +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".mp4";
  return destination.parent_path() / name;
}

bool RunCapture(const char* program, std::span<const std::string> arguments_in,
                std::string* output, std::string* error) {
  SECURITY_ATTRIBUTES inheritable{sizeof(inheritable), nullptr, TRUE};
  HANDLE read_end = nullptr;
  HANDLE write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, &inheritable, 0)) {
    SetError(error, "cannot create media inspection pipe: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  Handle read_pipe(read_end);
  Handle write_pipe(write_end);
  if (!SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0)) {
    SetError(error, "cannot configure media inspection pipe: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::vector<std::wstring> arguments;
  arguments.reserve(arguments_in.size());
  for (const std::string& arg : arguments_in) {
    arguments.push_back(Widen(arg));
  }
  Handle child;
  if (!Spawn(program, program, arguments, write_pipe.get(), &child, error)) {
    return false;
  }
  write_pipe.Reset();
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    DWORD amount = 0;
    if (!ReadFile(read_pipe.get(), buffer.data(),
                  static_cast<DWORD>(buffer.size()), &amount, nullptr)) {
      const DWORD failure = GetLastError();
      if (failure == ERROR_BROKEN_PIPE) {
        break;
      }
      SetError(error, "cannot read media inspection output: " +
                          WindowsMessage(failure));
      return false;
    }
    if (amount == 0) {
      break;
    }
    if (output->size() + static_cast<std::size_t>(amount) > (1U << 20U)) {
      (void)TerminateProcess(child.get(), 1);
      int ignored = 0;
      (void)WaitForChild(child.get(), &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), static_cast<std::size_t>(amount));
  }
  int status = 0;
  if (!WaitForChild(child.get(), &status, error)) {
    return false;
  }
  if (status != 0) {
    SetError(error, std::string(program) + " exited with status " +
                        std::to_string(status));
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

  static std::atomic<std::uint64_t> pipe_sequence{0};
  const std::wstring base_pipe_name =
      L"\\\\.\\pipe\\gufo-minimax-h3-" +
      std::to_wstring(static_cast<std::uint64_t>(GetCurrentProcessId())) +
      L"-" +
      std::to_wstring(pipe_sequence.fetch_add(1, std::memory_order_relaxed));
  const std::wstring video_pipe_name = base_pipe_name + L"-video";
  const std::wstring audio_pipe_name = base_pipe_name + L"-audio";
  Handle video_pipe = CreateOutboundPipe(video_pipe_name);
  Handle audio_pipe = CreateOutboundPipe(audio_pipe_name);
  if (!video_pipe || !audio_pipe) {
    SetError(error, "cannot create named pipes for FFmpeg: " +
                        WindowsMessage(GetLastError()));
    return false;
  }

  const std::wstring size =
      std::to_wstring(input_width) + L"x" + std::to_wstring(input_height);
  const std::wstring output_size =
      std::to_wstring(output_width) + L":" + std::to_wstring(output_height);
  const std::wstring rate = std::to_wstring(fps);
  const std::wstring audio_rate = std::to_wstring(sample_rate);
  const std::wstring audio_channels = std::to_wstring(channels);
  std::vector<std::wstring> arguments = {
      L"-y",
      L"-loglevel",
      L"error",
      L"-f",
      L"rawvideo",
      L"-pixel_format",
      L"rgb24",
      L"-video_size",
      size,
      L"-framerate",
      rate,
      L"-i",
      video_pipe_name,
      L"-f",
      L"f32le",
      L"-ar",
      audio_rate,
      L"-ac",
      audio_channels,
      L"-i",
      audio_pipe_name,
      L"-map",
      L"0:v:0",
      L"-map",
      L"1:a:0",
      L"-vf",
      L"scale=" + output_size + L":flags=lanczos,setsar=1",
      L"-c:v",
      L"libx264",
      L"-preset",
      L"fast",
      L"-crf",
      L"18",
      L"-pix_fmt",
      L"yuv420p",
      L"-c:a",
      L"aac",
      L"-b:a",
      L"192k",
      L"-movflags",
      L"+faststart",
      Widen(destination.string()),
  };

  Handle child;
  if (!Spawn(FfmpegProgram(), "FFmpeg", arguments, nullptr, &child, error)) {
    std::error_code ec;
    std::filesystem::remove(destination, ec);
    return false;
  }

  std::atomic<bool> stop{false};
  std::atomic<DWORD> video_error{ERROR_SUCCESS};
  std::atomic<DWORD> audio_error{ERROR_SUCCESS};
  std::uint64_t written_video = 0;
  std::uint64_t written_audio = 0;
  std::size_t maximum_audio_staging = 0;

  std::thread video_thread([&] {
    DWORD failure = ERROR_SUCCESS;
    if (!ConnectPipe(video_pipe.get(), child.get(), is_cancelled, stop,
                     &failure)) {
      video_error.store(failure, std::memory_order_release);
      stop.store(true, std::memory_order_release);
      video_pipe.Reset();
      return;
    }
    if (!WriteAll(video_pipe.get(), rgb24.data(), rgb24.size(), child.get(),
                  is_cancelled, stop, &written_video, &failure)) {
      video_error.store(failure, std::memory_order_release);
      stop.store(true, std::memory_order_release);
    }
    video_pipe.Reset();
  });

  std::thread audio_thread([&] {
    DWORD failure = ERROR_SUCCESS;
    if (!ConnectPipe(audio_pipe.get(), child.get(), is_cancelled, stop,
                     &failure)) {
      audio_error.store(failure, std::memory_order_release);
      stop.store(true, std::memory_order_release);
      audio_pipe.Reset();
      return;
    }
    std::vector<float> interleaved(
        static_cast<std::size_t>(kAudioChunkSamples) * channels);
    maximum_audio_staging = interleaved.size() * sizeof(float);
    int offset = 0;
    while (offset < samples && !stop.load(std::memory_order_acquire)) {
      const int chunk = std::min(kAudioChunkSamples, samples - offset);
      for (int sample = 0; sample < chunk; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
          interleaved[static_cast<std::size_t>(sample) * channels + channel] =
              channel_major_pcm[static_cast<std::size_t>(channel) * samples +
                                offset + sample];
        }
      }
      const auto* cursor =
          reinterpret_cast<const std::uint8_t*>(interleaved.data());
      const std::size_t bytes =
          static_cast<std::size_t>(chunk) * channels * sizeof(float);
      if (!WriteAll(audio_pipe.get(), cursor, bytes, child.get(), is_cancelled,
                    stop, &written_audio, &failure)) {
        audio_error.store(failure, std::memory_order_release);
        stop.store(true, std::memory_order_release);
        break;
      }
      offset += chunk;
    }
    audio_pipe.Reset();
  });

  video_thread.join();
  audio_thread.join();

  const bool cancelled = is_cancelled && is_cancelled();
  if (stop.load(std::memory_order_acquire) || cancelled) {
    (void)TerminateProcess(child.get(), 1);
  }
  int status = 0;
  const bool waited = WaitForChild(child.get(), &status, error);
  if (telemetry != nullptr) {
    telemetry->video_bytes = written_video;
    telemetry->audio_bytes = written_audio;
    telemetry->maximum_audio_staging_bytes = maximum_audio_staging;
    telemetry->child_status = waited ? status : -1;
    telemetry->cancelled = cancelled;
  }
  std::error_code filesystem_error;
  if (!waited) {
    std::filesystem::remove(destination, filesystem_error);
    return false;
  }
  if (cancelled) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "MiniMax H3 media mux cancelled");
    return false;
  }
  const DWORD writer_error =
      video_error.load(std::memory_order_acquire) != ERROR_SUCCESS
          ? video_error.load(std::memory_order_relaxed)
          : audio_error.load(std::memory_order_acquire);
  if (writer_error != ERROR_SUCCESS) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "cannot stream generated media to FFmpeg: " +
                        WindowsMessage(writer_error));
    return false;
  }
  if (status != 0) {
    std::filesystem::remove(destination, filesystem_error);
    SetError(error, "FFmpeg exited with status " + std::to_string(status));
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
