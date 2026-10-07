#include "src/cli/serve/serve.hpp"

#include <arpa/inet.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/sampling_request.hpp"
#include "src/core/diagnostics/gpu_queues.h"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif
#include "src/cli/serve/trace.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::cli {
namespace {

class ModelLoadLog {
public:
  ModelLoadLog(std::string kind, const std::filesystem::path& artifact)
      : kind_(std::move(kind)), start_(std::chrono::steady_clock::now()) {
    server::Logger::Info("loader", "event=load_started kind=" + kind_ +
                                       " artifact=" + artifact.string() + " " +
                                       server::Logger::MemoryStatus());
  }
  ~ModelLoadLog() {
    if (!completed_) {
      try {
        server::Logger::Error("loader", "event=load_failed kind=" + kind_ +
                                            " " +
                                            server::Logger::MemoryStatus());
      } catch (...) {
      }
    }
  }
  void Complete(std::string_view details, bool gpu_loaded = true) {
    completed_ = true;
    std::ostringstream out;
    out << "event=load_completed kind=" << kind_ << " elapsed_ms="
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start_)
               .count()
        << ' ' << details << ' ' << server::Logger::MemoryStatus();
#if defined(ENGINE_ENABLE_HIP)
    std::size_t free = 0, total = 0;
    if (gpu_loaded && hipMemGetInfo(&free, &total) == hipSuccess) {
      out << " gpu_device_used_mib=" << (total - free) / (1024 * 1024)
          << " gpu_device_total_mib=" << total / (1024 * 1024);
    }
#else
    (void)gpu_loaded;
#endif
    server::Logger::Info("loader", out.str());
  }

private:
  std::string kind_;
  std::chrono::steady_clock::time_point start_;
  bool completed_{false};
};

// Reports a fatal signal on the way out. A driver or runtime failure during a
// load can abort the process outright, which runs no destructor and no
// terminate handler, leaving a log that stops at event=load_started. Only
// async-signal-safe calls are allowed here, so the line is assembled by hand
// and written straight to the descriptor.
extern "C" void ReportFatalSignal(int number) {
  static constexpr char kPrefix[] =
      "[ERROR] [server] event=fatal_signal signal=";
  char digits[8];
  std::size_t length = 0;
  int value = number;
  if (value <= 0) {
    digits[length++] = '0';
  } else {
    char reversed[8];
    std::size_t count = 0;
    while (value > 0 && count < sizeof(reversed)) {
      reversed[count++] = static_cast<char>('0' + (value % 10));
      value /= 10;
    }
    while (count > 0) {
      digits[length++] = reversed[--count];
    }
  }
  (void)::write(STDERR_FILENO, kPrefix, sizeof(kPrefix) - 1);
  (void)::write(STDERR_FILENO, digits, length);
  (void)::write(STDERR_FILENO, "\n", 1);
  // Restore the default action and re-raise, so the exit status and any core
  // dump still describe the original fault.
  struct sigaction restore{};
  restore.sa_handler = SIG_DFL;
  (void)::sigemptyset(&restore.sa_mask);
  (void)::sigaction(number, &restore, nullptr);
  (void)::raise(number);
}

void InstallFatalSignalReporter() {
  struct sigaction action{};
  action.sa_handler = ReportFatalSignal;
  (void)::sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  for (const int number : {SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
    (void)::sigaction(number, &action, nullptr);
  }
}

// Installs a terminate handler that names the reason before the process dies.
// A HIP or driver failure during a load throws, and an exception that reaches
// the top of main terminates without unwinding the stack, so ModelLoadLog's
// destructor never runs and the log ends at event=load_started.
void ReportTerminationReason() {
  static std::terminate_handler previous = nullptr;
  previous = std::set_terminate([] {
    std::string reason = "unknown";
    if (std::current_exception() != nullptr) {
      try {
        std::rethrow_exception(std::current_exception());
      } catch (const std::exception& error) {
        reason = error.what();
      } catch (...) {
        reason = "non-standard exception";
      }
    }
    try {
      server::Logger::Error("server", "event=terminated reason=" + reason);
    } catch (...) {
      // The handler must not throw on its way out.
    }
    if (previous != nullptr) {
      previous();
    }
    std::abort();
  });
}

// Exit status after the GPU context is lost (EX_TEMPFAIL). Only a new process
// recovers, so a supervisor must restart it.
constexpr int kDeviceLostExitStatus = 75;

// A lost GPU context cannot recover in-process. Leave through the SIGTERM
// shutdown path so a supervisor restarts the server, and bound that teardown:
// joining requests or freeing device memory may block on the dead device.
void ShutdownAfterDeviceLoss() {
  constexpr auto kShutdownTimeout = std::chrono::seconds(10);
  // Arm the watchdog before logging or shutdown: neither a blocked log sink
  // nor a dead-device join may prevent the forced exit.
  try {
    std::thread([kShutdownTimeout] {
      std::this_thread::sleep_for(kShutdownTimeout);
      ::_exit(kDeviceLostExitStatus);
    }).detach();
  } catch (...) {
    ::_exit(kDeviceLostExitStatus);
  }
  server::Logger::Error(
      "server", "event=device_lost_shutdown exit_status=" +
                    std::to_string(kDeviceLostExitStatus) +
                    " timeout_s=" + std::to_string(kShutdownTimeout.count()));
  (void)std::raise(SIGTERM);
}

std::optional<ReasoningEffort> ParseReasoningEffort(std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

// Server-level options are accepted on either side of the modality
// subcommand. Registering them from one place keeps `gufo serve --help`, every
// `gufo serve <modality> --help`, and the parser that actually consumes them
// from drifting apart.

/// `--log-level` and its `-v/--verbose` shorthand share one sink. `level`
/// holds the canonical spelling when given and stays empty otherwise; the two
/// spellings together are a conflict, reported where the level is armed.
struct ServerLogOptions {
  bool verbose = false;
  std::optional<server::LogLevel> level;
};

// Verbosity help text. `gufo serve --help` lists these options by hand while
// every modality help registers them through the parser, so both readings share
// one source for the wording.
constexpr std::string_view kLogLevelHelp =
    "Log verbosity: error, warn, info or debug (default: info)";
constexpr std::string_view kVerboseHelp = "Shorthand for --log-level=debug";
constexpr std::string_view kTraceHelp =
    "Append text request bodies, rendered prompts, generated text and replies "
    "to PATH as JSON Lines; the file holds client content";

void AddServerOptions(gufo::cli::ArgParser& parser, std::string* host,
                      int* port, std::size_t* session_count,
                      std::size_t* max_connections,
                      std::size_t* max_request_body_bytes, std::string* api_key,
                      ServerLogOptions* log) {
  parser.AddOption("-i", "--host", "IP", "Bind address", "Server", host);
  parser.AddOption("-p", "--port", "N", "Port to listen on", "Server", port);
  if (session_count != nullptr)
    parser.AddOption("-j", "--sessions", "N",
                     "Preallocated GPU request sessions", "Server",
                     session_count);
  parser.AddOption("", "--max-connections", "N",
                   "Maximum simultaneous HTTP connections", "Server",
                   max_connections);
  parser.AddOption("", "--max-request-bytes", "N",
                   "Maximum HTTP request body bytes", "Server",
                   max_request_body_bytes);
  parser.AddOption("", "--api-key", "KEY", "API key", "Server", api_key);
  parser.AddCustomOption(
      "", "--log-level", "LEVEL", kLogLevelHelp, "Logging",
      [log](std::string_view flag, std::string_view value, std::string* error) {
        const auto level = server::LogLevelFromName(value);
        if (!level.has_value()) {
          *error = std::string(flag) + " must be error, warn, info or debug";
          return false;
        }
        log->level = *level;
        return true;
      });
  parser.AddFlag("-v", "--verbose", kVerboseHelp, "Logging", &log->verbose);
}

// Help-only sinks for AddServerOptions, so subcommand help can list the
// server options it shares with `gufo serve` without parsing into them.
struct ServerOptionHelpTargets {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  ServerLogOptions log;
};

void AddServerOptionsForHelp(gufo::cli::ArgParser& parser,
                             ServerOptionHelpTargets* targets,
                             bool include_sessions = true) {
  AddServerOptions(parser, &targets->host, &targets->port,
                   include_sessions ? &targets->session_count : nullptr,
                   &targets->max_connections, &targets->max_request_body_bytes,
                   &targets->api_key, &targets->log);
}

// Split a `NAME=VALUE` CLI spec. Returns false when either side is empty.
bool SplitNameValue(std::string_view spec, std::string_view flag,
                    std::string* name, std::string* value, std::string* error) {
  const std::size_t separator = spec.find('=');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 >= spec.size()) {
    *error = std::string(flag) + " expects NAME=VALUE";
    return false;
  }
  *name = std::string(spec.substr(0, separator));
  *value = std::string(spec.substr(separator + 1));
  return true;
}

std::optional<std::string> ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::nullopt;
  }
  std::string text{std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>()};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

// Resolve a `--voice-text` value: an existing file is read for its contents,
// anything else is taken as the transcript itself.
std::string ResolveReferenceText(const std::string& value) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(std::filesystem::path(value), ec)) {
    if (const auto text = ReadTextFile(std::filesystem::path(value))) {
      return *text;
    }
  }
  return value;
}

// Build the registered voices from `--voice NAME=WAV` and the optional
// `--voice-text NAME=<text|path>` overrides. Resolution happens after parsing
// so the two flags may appear in any order.
bool BuildVoicePresets(
    const std::vector<std::pair<std::string, std::string>>& voice_specs,
    const std::map<std::string, std::string>& text_specs,
    const std::map<std::string, std::string>& language_specs,
    std::map<std::string, server::TtsVoicePreset>* presets,
    std::string* error) {
  for (const auto& [name, wav] : voice_specs) {
    if (presets->contains(name)) {
      *error = "duplicate --voice name '" + name + "'";
      return false;
    }
    const std::filesystem::path wav_path(wav);
    std::ifstream wav_file(wav_path, std::ios::binary);
    if (!wav_file) {
      *error = "cannot open voice reference " + wav_path.string();
      return false;
    }
    const std::vector<char> wav_bytes{std::istreambuf_iterator<char>(wav_file),
                                      std::istreambuf_iterator<char>()};

    server::TtsVoicePreset preset;
    std::string decode_error;
    if (!models::qwen3_tts::DecodeWav(
            std::as_bytes(std::span<const char>(wav_bytes)),
            &preset.reference_audio, &decode_error)) {
      *error = wav_path.string() + ": " + decode_error;
      return false;
    }

    // An explicit --voice-text wins; otherwise fall back to a `.txt` sidecar
    // beside the WAV. With neither, the voice clones from the speaker
    // embedding alone, which needs no transcript.
    if (const auto override_text = text_specs.find(name);
        override_text != text_specs.end()) {
      preset.reference_text = ResolveReferenceText(override_text->second);
    } else {
      std::filesystem::path sidecar = wav_path;
      sidecar.replace_extension(".txt");
      if (const auto text = ReadTextFile(sidecar)) {
        preset.reference_text = *text;
      }
    }
    preset.speaker_embedding_only = preset.reference_text.empty();
    if (const auto language = language_specs.find(name);
        language != language_specs.end()) {
      preset.language = language->second;
    }
    presets->emplace(name, std::move(preset));
  }

  for (const auto& [name, unused] : text_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-text names unknown voice '" + name + "'";
      return false;
    }
  }
  for (const auto& [name, unused] : language_specs) {
    (void)unused;
    if (!presets->contains(name)) {
      *error = "--voice-lang names unknown voice '" + name + "'";
      return false;
    }
  }
  return true;
}

std::optional<ReasoningOptions> ResolveReasoningDefaults(
    std::string_view mode, std::string_view effort, std::string_view preserve,
    std::string* error) {
  ReasoningOptions options;
  if (mode == "on") {
    options.enabled = true;
  } else if (mode == "off") {
    options.enabled = false;
  } else if (mode != "auto") {
    *error = "--think must be on, off, or auto";
    return std::nullopt;
  }

  if (effort != "auto") {
    const auto parsed = ParseReasoningEffort(effort);
    if (!parsed.has_value()) {
      *error =
          "--reasoning-effort must be auto, minimal, low, medium, high, "
          "xhigh, or max";
      return std::nullopt;
    }
    if (options.enabled == false) {
      *error = "--reasoning-effort cannot be set while --think is off";
      return std::nullopt;
    }
    options.enabled = true;
    options.effort = parsed;
  }

  if (preserve == "on") {
    options.preserve_thinking = true;
  } else if (preserve == "off") {
    options.preserve_thinking = false;
  } else if (preserve != "auto") {
    *error = "--preserve-thinking must be on, off, or auto";
    return std::nullopt;
  }
  return options;
}

struct SpeechOptions {
  std::filesystem::path model;
  std::size_t context;
  std::string served_model_name;
  std::vector<std::pair<std::string, std::string>> voice_specs;
  std::map<std::string, std::string> voice_text_specs;
  std::map<std::string, std::string> voice_lang_specs;
};

void AddSpeechOptions(ArgParser& parser, bool tts, SpeechOptions* options) {
  parser.AddOption("-m", "--model", "DIR",
                   tts ? "Qwen3-TTS 12Hz 1.7B model directory"
                       : "Qwen3-ASR 1.7B model directory",
                   "Model", &options->model);
  parser.AddOption("-c", "--context", "N",
                   tts ? "Context capacity (default: 4096)"
                       : "Context capacity per audio chunk (default: 1024)",
                   "Model", &options->context);
  parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                   "Model", &options->served_model_name);
  if (!tts)
    return;
  parser.AddCustomOption(
      "", "--voice", "NAME=PATH",
      "Register a named Qwen3-TTS Base voice from a reference WAV "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string path;
        if (!SplitNameValue(value, "--voice", &name, &path, err)) {
          return false;
        }
        options->voice_specs.emplace_back(std::move(name), std::move(path));
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-lang", "NAME=LANGUAGE",
      "Language a --voice speaks; used when a request omits 'language' "
      "(repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string language;
        if (!SplitNameValue(value, "--voice-lang", &name, &language, err)) {
          return false;
        }
        if (!options->voice_lang_specs
                 .emplace(std::move(name), std::move(language))
                 .second) {
          *err = "duplicate --voice-lang name";
          return false;
        }
        return true;
      });
  parser.AddCustomOption(
      "", "--voice-text", "NAME=TEXT|PATH",
      "Reference transcript for a --voice, given inline or as a file path; "
      "defaults to a .txt sidecar beside the WAV (repeatable)",
      "Model",
      [options](std::string_view, std::string_view value, std::string* err) {
        std::string name;
        std::string text;
        if (!SplitNameValue(value, "--voice-text", &name, &text, err)) {
          return false;
        }
        if (!options->voice_text_specs.emplace(std::move(name), std::move(text))
                 .second) {
          *err = "duplicate --voice-text name";
          return false;
        }
        return true;
      });
}

}  // namespace

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand) {
  if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser(std::string(program_name) + " serve image",
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }
  if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve video",
        "Start the MiniMax H3 text-to-video HTTP generation server.");
    parser.AddOption("-m", "--model", "DIR",
                     "Operator-supplied MiniMax H3 directory", "Model",
                     &video_model);
    parser.AddOption(
        "", "--root", "DIR",
        "Storage root for persistent video jobs (default: video-jobs)",
        "Storage", &video_root);
    parser.AddOption("", "--manifest", "PATH", "Pinned H3 manifest override",
                     "Storage", &video_manifest);
    parser.AddOption("", "--ttl", "SEC",
                     "Completed-artifact TTL in seconds (default: 3600)",
                     "Storage", &video_ttl_seconds);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "tts" || subcommand == "asr") {
    const bool tts = subcommand == "tts";
    SpeechOptions options{.context = tts ? 4096U : 1024U};
    ArgParser parser(
        std::string(program_name) + " serve " + std::string(subcommand),
        tts ? "Serve Qwen3-TTS speech synthesis."
            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, tts, &options);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help, false);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "llm") {
    std::string model;
    std::string served_model_name;
    std::uint32_t max_context = 0;
    std::int64_t max_tokens = -1;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string draft_policy;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::string vision_model_path;
    std::size_t draft_tokens = 7;
    std::size_t min_draft_tokens = 1;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    std::size_t max_pending_requests_per_client = 0;
    std::uint64_t request_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::size_t cache_ram_bytes = 0;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        server::TextRunnerDiskCacheOptions::kDefaultCapacityBytes;
    std::size_t cache_disk_staging_bytes = 0;
    bool log_progress = false;
    std::string trace_path;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");

    // Model & Context
    parser.AddOption("-m", "--model", "PATH",
                     "Path to GGUF model file (required)", "Model", &model);
    parser.AddOption("", "--mmproj", "PATH",
                     "Qwen BF16 vision sidecar (auto-discovered beside model)",
                     "Model", &vision_model_path);
    parser.AddOption("", "--served-model-name", "ID",
                     "Model identifier exposed by the OpenAI API", "Model",
                     &served_model_name);
    parser.AddOption(
        "-c", "--context", "N",
        "Context tokens per session (default: 0 = model native context)",
        "Model", &max_context);

    // Sampling Defaults
    parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default new-token limit (default: -1 = until EOS or context full)",
        "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(parser, &sampling_config, "Sampling Defaults", true,
                            true);

    // Reasoning Defaults
    parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode: on, off, or auto (default: model)",
        "Reasoning Defaults", &reasoning_mode);
    parser.AddOption(
        "", "--reasoning-effort", "LEVEL",
        "Default effort: auto, minimal, low, medium, high, xhigh, or max",
        "Reasoning Defaults", &reasoning_effort);
    parser.AddOption("", "--preserve-thinking", "MODE",
                     "Replay prior reasoning: on, off, or auto",
                     "Reasoning Defaults", &preserve_thinking);

    // Speculative & Hardware
    parser.AddOption("", "--speculative", "MODE",
                     "HTTP draft backend: dspark, dflash2, mtp, or off",
                     "Speculative", &speculative_backend);
    parser.AddOption("", "--dflash-model", "PATH",
                     "Path to Qwen DFlash2 GGUF file", "Speculative",
                     &dflash_model_path);
    parser.AddOption(
        "", "--draft-policy", "POLICY",
        "DFlash2 block length: fixed or adaptive (default: adaptive)",
        "Speculative", &draft_policy);
    parser.AddOption("", "--dspark-model", "PATH",
                     "Path to DeepSeek V4 Flash DSpark support GGUF file",
                     "Speculative", &dspark_model_path);
    parser.AddOption("", "--mtp-model", "PATH",
                     "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the "
                     "mtp-...-shared-*.gguf sidecar)",
                     "Speculative", &mtp_model_path);
    parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7)",
        "Speculative", &draft_tokens);

    parser.AddOption("", "--min-draft-tokens", "N",
                     "Adaptive draft floor (default: 1)", "Speculative",
                     &min_draft_tokens);
    parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512)",
        "Scheduling", &prefill_chunk_tokens);
    parser.AddOption("", "--max-pending", "N",
                     "Maximum queued generation requests (default: 16)",
                     "Scheduling", &max_pending_requests);
    parser.AddOption(
        "", "--max-pending-per-client", "N",
        "Maximum queued requests per client IP (default: --max-pending)",
        "Scheduling", &max_pending_requests_per_client);
    parser.AddOption(
        "", "--request-timeout-ms", "MS",
        "Queue plus generation timeout; 0 disables it (default: 0)",
        "Scheduling", &request_timeout_ms);
    parser.AddOption("", "--max-output-bytes", "N",
                     "Maximum generated bytes per request (default: 1048576)",
                     "Scheduling", &max_output_bytes);
    parser.AddOption("", "--max-buffered-output-bytes", "N",
                     "Maximum queued stream bytes per request (default: 65536)",
                     "Scheduling", &max_buffered_output_bytes);
    parser.AddOption(
        "", "--max-buffered-output-total", "N",
        "Maximum queued stream bytes across requests (default: 262144)",
        "Scheduling", &max_buffered_output_bytes_total);
    parser.AddOption(
        "", "--cache-ram-bytes", "N",
        "Retained RAM-cache byte budget (default: 0 = auto: half of free "
        "RAM, at most 32 GiB; explicit values may use free RAM minus 4 GiB)",
        "Cache", &cache_ram_bytes);
    parser.AddOption("", "--cache-disk", "DIR",
                     "Opt-in restart-safe continuation cache directory",
                     "Cache", &cache_disk_directory);
    parser.AddOption("", "--cache-disk-bytes", "N",
                     "Retained disk-cache byte budget (default: " +
                         std::to_string(cache_disk_bytes) + ")",
                     "Cache", &cache_disk_bytes);
    parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "RAM limit for queued snapshots and each disk read "
        "(default: 0 = auto, at most the disk budget and 1/8 available RAM)",
        "Cache", &cache_disk_staging_bytes);
    parser.AddFlag("", "--log-progress",
                   "Log live prefill and decode progress (needs "
                   "--log-level=info or debug)",
                   "Logging", &log_progress);
    parser.AddOption("", "--trace", "PATH", kTraceHelp, "Logging", &trace_path);
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  std::cout
      << "Usage: " << program_name
      << " serve [SERVER_OPTIONS] [COMMAND] [OPTIONS]\n\n"
      << "Start the OpenAI-compatible HTTP server for a specific model "
         "modality.\n\n"
      << "Commands:\n"
      << "  llm       Serve text LLM endpoints (/v1/chat/completions, "
         "/v1/completions) [default]\n"
      << "  video     Serve MiniMax H3 video generation endpoint "
         "(/v1/video/generations)\n"
      << "  image     Serve Qwen-Image-2.1 (/v1/images/generations, "
         "/v1/images/edits)\n"
      << "  tts       Serve Qwen3-TTS speech synthesis (/v1/audio/speech)\n"
      << "  asr       Serve Qwen3-ASR transcription "
         "(/v1/audio/transcriptions)\n\n"
      << "Server:\n"
      << "  -i, --host <IP>        Bind address (default: 127.0.0.1)\n"
      << "  -p, --port <N>         Port to listen on (default: 8080)\n"
      << "  -j, --sessions <N>     Preallocated GPU request sessions (default: "
         "1)\n"
      << "      --max-connections <N>\n"
      << "                         Maximum simultaneous HTTP connections "
         "(default: 16)\n"
      << "      --max-request-bytes <N>\n"
      << "                         Maximum HTTP request body bytes (default: "
         "8388608)\n"
      << "      --api-key <KEY>    Require Bearer authorization for requests\n"
      << "\nLogging:\n"
      << "      --log-level <LEVEL>\n"
      << "                         " << kLogLevelHelp << "\n"
      << "  -v, --verbose          " << kVerboseHelp << "\n"
      << "\nGeneral:\n"
      << "  -h, --help             Print help\n";
}

int RunServe(std::span<const char* const> args) {
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  // An exception that escapes a load path terminates without unwinding, so no
  // destructor reports it and the process exits with an empty log. Naming the
  // reason here is the difference between a diagnosable failure and a server
  // that simply vanished after event=load_started.
  ReportTerminationReason();
  InstallFatalSignalReporter();

  std::string host = "127.0.0.1";
  if (const char* env_host = std::getenv("HOST");
      env_host != nullptr && *env_host != '\0') {
    host = env_host;
  } else if (const char* env_strix_host = std::getenv("GUFO_HOST");
             env_strix_host != nullptr && *env_strix_host != '\0') {
    host = env_strix_host;
  }

  int port = 8080;
  if (const char* env_port = std::getenv("PORT");
      env_port != nullptr && *env_port != '\0') {
    const std::string_view sv(env_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  } else if (const char* env_strix_port = std::getenv("GUFO_PORT");
             env_strix_port != nullptr && *env_strix_port != '\0') {
    const std::string_view sv(env_strix_port);
    int parsed_port = 0;
    auto [ptr, ec] =
        std::from_chars(sv.data(), sv.data() + sv.size(), parsed_port);
    if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
      port = parsed_port;
    }
  }

  std::size_t session_count = 1;
  std::size_t max_connections = 16;
  std::size_t max_request_body_bytes =
      static_cast<std::size_t>(8) * 1024 * 1024;
  std::string api_key;
  ServerLogOptions log_options;

  // Identify the modality only after leading server options. The selected
  // parser consumes every option together, so --served-model-name -v and
  // --model audio cannot be mistaken for server flags or subcommands.
  ArgParser server_parser("gufo serve");
  const auto add_server_options = [&](ArgParser& parser,
                                      bool include_sessions = true) {
    AddServerOptions(
        parser, &host, &port, include_sessions ? &session_count : nullptr,
        &max_connections, &max_request_body_bytes, &api_key, &log_options);
  };
  add_server_options(server_parser);
  std::string subcommand = "llm";
  std::vector<const char*> sub_args(args.begin(), args.end());
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    const auto equals = arg.find('=');
    const auto* option = server_parser.FindOption(arg.substr(0, equals));
    if (option != nullptr) {
      if (!option->is_flag && equals == std::string_view::npos) {
        ++i;
      }
      continue;
    }
    if (arg == "llm" || arg == "video" || (arg == "tts" || arg == "asr") ||
        arg == "image") {
      subcommand = arg;
      sub_args.erase(sub_args.begin() + static_cast<std::ptrdiff_t>(i));
    } else if (arg == "--help" || arg == "-h" || arg == "help") {
      const std::string_view topic =
          arg == "help" && i + 1 < args.size() ? args[i + 1] : "";
      if (!topic.empty() && topic != "llm" && topic != "video" &&
          topic != "tts" && topic != "asr" && topic != "image") {
        std::cerr << "Error: unknown serve command '" << topic << "'\n";
        return 2;
      }
      PrintServeHelp("gufo", topic);
      return 0;
    }
    break;
  }
  // Hardware queues are claimed on the first HIP dispatch and held for the
  // lifetime of the process, so the budget is decided once per invocation,
  // after the modality parser has armed the log threshold and before any
  // model loads. It lives in `prepare_server_options` so the emitted
  // `queue_budget` line already obeys `--log-level`. Help output loads nothing
  // and returns before that point.

  const auto prepare_server_options = [&] {
    in_addr address{};
    if (::inet_pton(AF_INET, host.c_str(), &address) != 1) {
      std::cerr << "Error: --host must be an IPv4 address\n";
      return false;
    }
    if (port < 0 || port > 65535) {
      std::cerr << "Error: --port must be between 0 and 65535\n";
      return false;
    }
    if (session_count == 0 || max_connections == 0 ||
        max_request_body_bytes == 0) {
      std::cerr << "Error: server limits must be positive\n";
      return false;
    }
    // Arm verbosity here: this runs after every server option is parsed and
    // before the first loader, cache or HTTP line is written, so one call sets
    // the level for all modalities. `-v/--verbose` is shorthand for
    // `--log-level=debug`, so the two spellings together are a conflict to
    // report, not a precedence to resolve silently.
    if (log_options.verbose && log_options.level.has_value()) {
      std::cerr << "Error: --verbose is shorthand for --log-level=debug, so "
                   "it cannot combine with --log-level\n";
      return false;
    }
    const server::LogLevel resolved =
        log_options.verbose
            ? server::LogLevel::kDebug
            : log_options.level.value_or(server::LogLevel::kInfo);
    server::Logger::SetLevel(resolved);
    // The queue plan is applied only after a valid invocation is established,
    // and the `queue_budget` line is emitted now so an absolute threshold
    // (`--log-level=error`) suppresses this INFO startup diagnostic with the
    // rest of the boot sequence instead of leaking it before SetLevel ran.
    const auto profile = subcommand == "llm" ? diagnostics::QueueProfile::kText
                         : (subcommand == "tts" || subcommand == "asr")
                             ? diagnostics::QueueProfile::kAudio
                             : diagnostics::QueueProfile::kUnmeasured;
    const auto plan =
        diagnostics::PlanQueues(profile, diagnostics::QueryQueueCensus(),
                                std::getenv("GPU_MAX_HW_QUEUES"));
    diagnostics::ApplyQueuePlan(plan);
    server::Logger::Info("gpu",
                         diagnostics::DescribeQueuePlan(subcommand, plan));
    if (plan.may_exceed_budget) {
      server::Logger::Warn("gpu", diagnostics::DescribeQueuePressure(plan));
    }
    // Deliberately emitted before the model opens: a load that fails or hangs
    // never reaches the INFO `event=listening` banner in HttpServer::run, so
    // this is the only record of the resolved options and the armed threshold.
    // The overlap on the listener limits is intentional: the two lines mark
    // different points in the lifecycle at different tiers, so neither can
    // replace the other.
    const std::string options_line =
        "event=options host=" + host + " port=" + std::to_string(port) +
        " max_connections=" + std::to_string(max_connections) +
        " max_request_bytes=" + std::to_string(max_request_body_bytes) +
        " api_key=" + (api_key.empty() ? "unset" : "set") +
        " log_level=" + std::string(server::LogLevelName(resolved));
    server::Logger::Debug("server", options_line);
    return true;
  };
  std::string parse_err;

  std::shared_ptr<server::InferenceBackend> backend;
  std::shared_ptr<server::VideoJobService> video_jobs;
  std::shared_ptr<server::TtsService> tts;
  std::shared_ptr<server::AsrService> asr;
  std::shared_ptr<server::ImageService> images;

  if (subcommand == "image") {
    std::filesystem::path model;
    std::string name = "Qwen-Image-2.1";
    ArgParser parser("gufo serve image",
                     "Serve Qwen-Image-2.1 generation and editing.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen-Image-2.1 safetensors directory", "Model", &model);
    parser.AddOption("", "--served-model-name", "NAME", "Public API model ID",
                     "Model", &name);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "image");
      return 0;
    }
    if (!prepare_server_options())
      return 2;
    if (model.empty()) {
      std::cerr << "Error: --model <DIR> is required for image server\n";
      return 2;
    }
    ModelLoadLog load_log("qwen_image_21", model);
    try {
      images = std::make_shared<server::ImageService>(model, name);
    } catch (const std::exception& error) {
      std::cerr << "Error loading Qwen-Image-2.1: " << error.what() << '\n';
      return 1;
    }
    load_log.Complete("model=" + name + " weights=mapped upload=on_demand");
  } else if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    gufo::cli::ArgParser video_parser(
        "gufo serve video", "Start the MiniMax H3 video generation server.");
    video_parser.AddOption("-m", "--model", "DIR",
                           "Operator-supplied MiniMax H3 directory", "Model",
                           &video_model);
    video_parser.AddOption("", "--root", "DIR",
                           "Storage root for persistent video jobs", "Storage",
                           &video_root);
    video_parser.AddOption("", "--manifest", "PATH",
                           "Pinned H3 manifest override", "Storage",
                           &video_manifest);
    video_parser.AddOption("", "--ttl", "SEC",
                           "Completed-artifact TTL in seconds", "Storage",
                           &video_ttl_seconds);

    add_server_options(video_parser);
    if (!video_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "video");
      return 2;
    }
    if (video_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "video");
      return 0;
    }
    if (!prepare_server_options()) {
      return 2;
    }

    if (video_ttl_seconds == 0 ||
        video_ttl_seconds >
            static_cast<std::uint64_t>(std::chrono::seconds::max().count())) {
      std::cerr << "Error: --ttl must be a positive duration\n";
      return 2;
    }

    if (video_model.empty()) {
      std::cerr << "Error: --model <DIR> is required for video server\n";
      PrintServeHelp("gufo", "video");
      return 2;
    }

    ModelLoadLog load_log("video_inventory", video_model);
    video_jobs = std::make_shared<server::VideoJobService>(
        server::VideoJobServiceOptions{
            .model_root = video_model,
            .source_manifest = video_manifest,
            .storage_root = video_root,
            .queue_capacity = 1,
            .artifact_ttl = std::chrono::seconds(
                static_cast<std::chrono::seconds::rep>(video_ttl_seconds)),
            .validate_model_inventory = true,
            .id_factory = {},
            .now = {},
            .runner = {},
        });
    if (!video_jobs->ready()) {
      std::cerr << "Error enabling MiniMax H3 video service: "
                << video_jobs->initialization_error() << '\n';
      return 1;
    }
    load_log.Complete(
        "model=minimax-h3 sessions=1 queue_capacity=1 weights=lazy", false);
  } else if (subcommand == "tts" || subcommand == "asr") {
    const bool is_tts = subcommand == "tts";
    SpeechOptions options{.context = is_tts ? 4096U : 1024U};
    ArgParser parser("gufo serve " + subcommand,
                     is_tts ? "Serve Qwen3-TTS speech synthesis."
                            : "Serve Qwen3-ASR transcription.");
    AddSpeechOptions(parser, is_tts, &options);
    add_server_options(parser, false);
    if (!parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << '\n';
      return 2;
    }
    if (parser.IsHelpRequested()) {
      PrintServeHelp("gufo", subcommand);
      return 0;
    }
    if (!prepare_server_options())
      return 2;
    if (options.model.empty()) {
      std::cerr << "Error: --model <DIR> is required for " << subcommand
                << " server\n";
      return 2;
    }
    if (options.context < (is_tts ? 1U : 32U)) {
      std::cerr << "Error: --context must be at least " << (is_tts ? 1 : 32)
                << '\n';
      return 2;
    }
    ModelLoadLog load_log(subcommand, options.model);
    if (is_tts) {
      std::map<std::string, server::TtsVoicePreset> voice_presets;
      if (!BuildVoicePresets(options.voice_specs, options.voice_text_specs,
                             options.voice_lang_specs, &voice_presets,
                             &parse_err)) {
        std::cerr << "Error: " << parse_err << '\n';
        return 2;
      }
      try {
        tts = std::make_shared<server::TtsService>(server::TtsServiceOptions{
            .model_root = options.model,
            .native_context_tokens = options.context,
            .validate_model = true,
            .model_id = options.served_model_name,
            .voices = {},
            .voice_presets = std::move(voice_presets),
            .runner = {},
        });
      } catch (const std::exception& error) {
        std::cerr << "Error loading Qwen3-TTS: " << error.what() << '\n';
        return 1;
      }
      if (!tts->ready()) {
        std::cerr << "Error enabling Qwen3-TTS service: "
                  << tts->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + tts->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    } else {
      try {
        asr = std::make_shared<server::AsrService>(server::AsrServiceOptions{
            .model_root = options.model,
            .native_context_tokens = options.context,
            .validate_model = true,
            .model_id = options.served_model_name,
            .runner = {},
        });
      } catch (const std::exception& error) {
        std::cerr << "Error loading Qwen3-ASR: " << error.what() << '\n';
        return 1;
      }
      if (!asr->ready()) {
        std::cerr << "Error enabling Qwen3-ASR service: "
                  << asr->initialization_error() << '\n';
        return 1;
      }
      load_log.Complete(
          "model=" + asr->model_id() +
          " sessions=1 context_tokens=" + std::to_string(options.context));
    }
  } else {
    // Default to LLM server
    std::string model;
    std::string served_model_name;
    std::uint32_t max_context = 0;
    std::int64_t max_tokens = -1;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string draft_policy;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::string vision_model_path;
    std::size_t draft_tokens = 7;
    std::size_t min_draft_tokens = 1;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    // Defaults to --max-pending after parsing unless supplied.
    std::size_t max_pending_requests_per_client = 0;
    std::uint64_t request_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::size_t cache_ram_bytes = 0;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        server::TextRunnerDiskCacheOptions::kDefaultCapacityBytes;
    std::size_t cache_disk_staging_bytes = 0;
    bool log_progress = false;
    std::string trace_path;

    gufo::cli::ArgParser llm_parser(
        "gufo serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    llm_parser.AddOption("-m", "--model", "PATH",
                         "Path to GGUF model file (required)", "Model", &model);
    llm_parser.AddOption(
        "", "--mmproj", "PATH",
        "Qwen BF16 vision sidecar (auto-discovered beside model)", "Model",
        &vision_model_path);
    llm_parser.AddOption("", "--served-model-name", "ID",
                         "Model identifier exposed by the OpenAI API", "Model",
                         &served_model_name);
    llm_parser.AddOption(
        "-c", "--context", "N",
        "Context tokens per session (default: 0 = model native context)",
        "Model", &max_context);
    llm_parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default new-token limit (default: -1 = until EOS or context full)",
        "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(llm_parser, &sampling_config, "Sampling Defaults",
                            true, true);
    llm_parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode: on, off, or auto (default: model)",
        "Reasoning Defaults", &reasoning_mode);
    llm_parser.AddOption(
        "", "--reasoning-effort", "LEVEL",
        "Default effort: auto, minimal, low, medium, high, xhigh, or max",
        "Reasoning Defaults", &reasoning_effort);
    llm_parser.AddOption("", "--preserve-thinking", "MODE",
                         "Replay prior reasoning: on, off, or auto",
                         "Reasoning Defaults", &preserve_thinking);
    llm_parser.AddOption("", "--speculative", "MODE",
                         "HTTP draft backend: dspark, dflash2, mtp, or off",
                         "Speculative", &speculative_backend);
    llm_parser.AddOption("", "--dflash-model", "PATH",
                         "Path to Qwen DFlash2 GGUF file", "Speculative",
                         &dflash_model_path);
    llm_parser.AddOption(
        "", "--draft-policy", "POLICY",
        "DFlash2 block length: fixed or adaptive (default: adaptive)",
        "Speculative", &draft_policy);
    llm_parser.AddOption("", "--dspark-model", "PATH",
                         "Path to DeepSeek V4 Flash DSpark support GGUF file",
                         "Speculative", &dspark_model_path);
    llm_parser.AddOption(
        "", "--mtp-model", "PATH",
        "Path to the Qwen MTP draft GGUF (Qwen3.8-Flash-Next: the "
        "mtp-...-shared-*.gguf sidecar)",
        "Speculative", &mtp_model_path);
    llm_parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7)",
        "Speculative", &draft_tokens);

    llm_parser.AddOption("", "--min-draft-tokens", "N",
                         "Adaptive draft floor (default: 1)", "Speculative",
                         &min_draft_tokens);
    llm_parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512)",
        "Scheduling", &prefill_chunk_tokens);
    llm_parser.AddOption("", "--max-pending", "N",
                         "Maximum queued generation requests (default: 16)",
                         "Scheduling", &max_pending_requests);
    llm_parser.AddOption(
        "", "--max-pending-per-client", "N",
        "Maximum queued requests per client IP (default: --max-pending)",
        "Scheduling", &max_pending_requests_per_client);
    llm_parser.AddOption(
        "", "--request-timeout-ms", "MS",
        "Queue plus generation timeout; 0 disables it (default: 0)",
        "Scheduling", &request_timeout_ms);
    llm_parser.AddOption(
        "", "--max-output-bytes", "N",
        "Maximum generated bytes per request (default: 1048576)", "Scheduling",
        &max_output_bytes);
    llm_parser.AddOption(
        "", "--max-buffered-output-bytes", "N",
        "Maximum queued stream bytes per request (default: 65536)",
        "Scheduling", &max_buffered_output_bytes);
    llm_parser.AddOption(
        "", "--max-buffered-output-total", "N",
        "Maximum queued stream bytes across requests (default: 262144)",
        "Scheduling", &max_buffered_output_bytes_total);
    llm_parser.AddOption(
        "", "--cache-ram-bytes", "N",
        "Retained RAM-cache byte budget (default: 0 = auto: half of free "
        "RAM, at most 32 GiB; explicit values may use free RAM minus 4 GiB)",
        "Cache", &cache_ram_bytes);
    llm_parser.AddOption("", "--cache-disk", "DIR",
                         "Opt-in restart-safe continuation cache directory",
                         "Cache", &cache_disk_directory);
    llm_parser.AddOption("", "--cache-disk-bytes", "N",
                         "Retained disk-cache byte budget (default: " +
                             std::to_string(cache_disk_bytes) + ")",
                         "Cache", &cache_disk_bytes);
    llm_parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "RAM limit for queued snapshots and each disk read "
        "(default: 0 = auto, at most the disk budget and 1/8 available RAM)",
        "Cache", &cache_disk_staging_bytes);
    llm_parser.AddFlag("", "--log-progress",
                       "Log live prefill and decode progress (needs "
                       "--log-level=info or debug)",
                       "Logging", &log_progress);
    llm_parser.AddOption("", "--trace", "PATH", kTraceHelp, "Logging",
                         &trace_path);

    add_server_options(llm_parser);
    if (!llm_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "llm");
      return 2;
    }
    if (llm_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "llm");
      return 0;
    }
    if (!llm_parser.WasSupplied("--max-pending-per-client")) {
      max_pending_requests_per_client = max_pending_requests;
    }
    if (!prepare_server_options()) {
      return 2;
    }
    // Progress lines are INFO-tier. Under a quieter threshold `--log-progress`
    // would be accepted and then silently discarded, so reject the combination
    // instead of ignoring an option the caller asked for.
    if (log_progress && !server::Logger::Enabled(server::LogLevel::kInfo)) {
      std::cerr << "Error: --log-progress needs --log-level=info or "
                   "--log-level=debug\n";
      return 2;
    }
    // Opened before the model loads, so an unwritable path fails at once
    // instead of after the weights are resident.
    if (!trace_path.empty()) {
      if (const auto error = server::Trace::Open(trace_path)) {
        std::cerr << "Error: cannot open --trace file '" << trace_path
                  << "': " << *error << "\n";
        return 2;
      }
      server::Logger::Warn("trace",
                           "event=trace_enabled path=" + trace_path +
                               " content=prompts,generated_text,replies");
    }
    sampling::SamplingConfig validated_sampling;
    const bool sampling_valid = !server::ParseSamplingConfig(
        json::Value::object(), sampling_config, &validated_sampling);
    if (max_tokens < -1 || max_tokens == 0 ||
        max_tokens > std::numeric_limits<std::uint32_t>::max() ||
        prefill_chunk_tokens == 0 || max_pending_requests == 0 ||
        max_pending_requests_per_client == 0 ||
        max_pending_requests_per_client > max_pending_requests ||
        max_output_bytes == 0 || max_buffered_output_bytes == 0 ||
        max_buffered_output_bytes_total == 0 ||
        (!cache_disk_directory.empty() && cache_disk_bytes == 0) ||
        request_timeout_ms > static_cast<std::uint64_t>(
                                 std::chrono::milliseconds::max().count()) ||
        !sampling_valid) {
      std::cerr << "Error: sampling and scheduling limits are invalid\n";
      return 2;
    }
    if (draft_tokens == 0 || min_draft_tokens == 0 ||
        min_draft_tokens > draft_tokens ||
        draft_tokens > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Error: speculative draft limits are invalid\n";
      return 2;
    }
    const auto reasoning_defaults = ResolveReasoningDefaults(
        reasoning_mode, reasoning_effort, preserve_thinking, &parse_err);
    if (!reasoning_defaults.has_value()) {
      std::cerr << "Error: " << parse_err << "\n";
      return 2;
    }

    server::TextSpeculativeConfig speculative_config;
    if (speculative_backend.empty() && !dspark_model_path.empty()) {
      speculative_backend = "dspark";
    }
    if (speculative_backend.empty() || speculative_backend == "off") {
      speculative_config.backend = server::TextSpeculativeBackend::kDisabled;
    } else if (speculative_backend == "dflash2") {
      speculative_config.backend = server::TextSpeculativeBackend::kDFlash;
    } else if (speculative_backend == "dspark") {
      speculative_config.backend = server::TextSpeculativeBackend::kDSpark;
    } else if (speculative_backend == "mtp") {
      speculative_config.backend = server::TextSpeculativeBackend::kMtp;
    } else {
      std::cerr << "Error: speculative backend '" << speculative_backend
                << "' is not supported by the HTTP server\n";
      return 2;
    }
    try {
      if (!draft_policy.empty() &&
          speculative_config.backend != server::TextSpeculativeBackend::kDFlash)
        throw std::invalid_argument("--draft-policy requires DFlash2");
      speculative_config.dflash_policy =
          speculative::ParseDFlashDraftPolicy(draft_policy);
    } catch (const std::invalid_argument& exception) {
      std::cerr << "Error: " << exception.what() << '\n';
      return 2;
    }
    if (speculative_config.backend == server::TextSpeculativeBackend::kDFlash &&
        (dflash_model_path.empty() || min_draft_tokens != 1)) {
      std::cerr << "Error: DFlash2 requires --dflash-model and "
                   "--min-draft-tokens 1; bound blocks with --draft-tokens\n";
      return 2;
    }
    speculative_config.draft_model_path =
        speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? dspark_model_path
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? mtp_model_path
            : dflash_model_path;
    speculative_config.max_draft_tokens =
        static_cast<std::uint32_t>(draft_tokens);
    speculative_config.min_draft_tokens =
        static_cast<std::uint32_t>(min_draft_tokens);
    if (model.empty()) {
      std::cerr << "Error: --model <PATH> is required\n";
      return 2;
    }
    std::string err;
    ModelLoadLog load_log("text", model);
    backend = std::make_shared<server::InferenceBackend>();
    if (!backend->load(model, &err, max_context, session_count,
                       server::TextPrefillPolicy{
                           .decode_active_tokens = prefill_chunk_tokens,
                       },
                       server::TextSchedulerPolicy{
                           .max_pending_requests = max_pending_requests,
                           .max_pending_requests_per_client =
                               max_pending_requests_per_client,
                           .max_output_bytes_per_request = max_output_bytes,
                           .max_buffered_output_bytes_per_request =
                               max_buffered_output_bytes,
                           .max_buffered_output_bytes_total =
                               max_buffered_output_bytes_total,
                           .request_timeout =
                               std::chrono::milliseconds{
                                   static_cast<std::chrono::milliseconds::rep>(
                                       request_timeout_ms)},
                           .log_progress = log_progress,
                       },
                       speculative_config,
                       server::TextDiskCacheConfig{
                           .directory = cache_disk_directory,
                           .capacity_bytes = cache_disk_bytes,
                           .staging_capacity_bytes = cache_disk_staging_bytes,
                           .model_artifact_fingerprint = {},
                       },
                       vision_model_path,
                       server::TextRunnerRamCacheOptions{
                           .capacity_bytes = cache_ram_bytes})) {
      std::cerr << "Error loading model '" << model << "': " << err << "\n";
      return 1;
    }
    backend->set_model_id(served_model_name);
    backend->set_sampling_defaults(
        max_tokens < 0 ? 0 : static_cast<std::size_t>(max_tokens),
        sampling_config, SamplingOptionsSupplied(llm_parser));
    backend->set_reasoning_defaults(*reasoning_defaults);
    const auto effective_sampling =
        backend->sampling_defaults().Resolve(reasoning_defaults->enabled);
    server::ChatRequest default_request;
    default_request.reasoning = *reasoning_defaults;
    const auto initial_output = backend->initial_output_state(default_request);
    std::ostringstream sampling_log;
    sampling_log
        << "event=defaults thinking="
        << (initial_output == server::TextGenerationBackend::
                                  InitialOutputState::kReasoning
                ? "on"
            : initial_output ==
                    server::TextGenerationBackend::InitialOutputState::kContent
                ? "off"
                : "auto")
        << " temperature=" << effective_sampling.temperature
        << " top_k=" << effective_sampling.top_k
        << " top_p=" << effective_sampling.top_p
        << " min_p=" << effective_sampling.min_p
        << " min_keep=" << effective_sampling.min_keep
        << " seed=" << effective_sampling.seed
        << " repeat_penalty=" << effective_sampling.repeat_penalty
        << " repeat_last_n=" << effective_sampling.repeat_last_n
        << " frequency_penalty=" << effective_sampling.frequency_penalty
        << " presence_penalty=" << effective_sampling.presence_penalty;
    server::Logger::Info("sampling", sampling_log.str());
    const char* speculation =
        speculative_config.backend == server::TextSpeculativeBackend::kDFlash
            ? "dflash2"
        : speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? "dspark"
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? "mtp"
            : "off";
    load_log.Complete(
        "model=" + backend->model_id() +
        " sessions=" + std::to_string(session_count) + " context_tokens=" +
        std::to_string(backend->max_context()) + " speculative=" + speculation +
        " draft_limit=" + std::to_string(speculative_config.max_draft_tokens) +
        " disk_cache=" + (cache_disk_directory.empty() ? "off" : "enabled"));
  }

  // run() joins every request thread, so no hook call outlives it.
  std::atomic<bool> device_lost{false};
  server::HttpServer server(
      host, port, backend, video_jobs, tts, asr,
      server::HttpServerOptions{
          .max_request_body_bytes = max_request_body_bytes,
          .max_connections = max_connections,
          .api_key = std::move(api_key),
          .on_device_lost =
              [&device_lost] {
                device_lost.store(true);
                ShutdownAfterDeviceLoss();
              },
      },
      images);
  std::string err;
  if (!server.start(&err)) {
    std::cerr << "Error starting HTTP server: " << err << "\n";
    return 1;
  }
  // Releasing model state on a lost device crashes; the kernel reclaims it.
  // The backend is checked as well: an external SIGTERM can stop the server
  // after the scheduler records the loss but before a request fires the hook.
  const auto exit_if_device_lost = [&device_lost, &backend] {
    if (device_lost.load() || (backend != nullptr && backend->device_lost())) {
      ::_exit(kDeviceLostExitStatus);
    }
  };
  try {
    server.run(/*handle_signals=*/true);
  } catch (...) {
    exit_if_device_lost();
    throw;
  }
  exit_if_device_lost();
  return 0;
}

}  // namespace gufo::cli
