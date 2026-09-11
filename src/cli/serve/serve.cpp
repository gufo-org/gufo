#include "src/cli/serve/serve.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
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
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"
#include "src/models/qwen3_tts/audio.hpp"

namespace gufo::cli {
namespace {

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
void AddServerOptions(gufo::cli::ArgParser& parser, std::string* host,
                      int* port, std::size_t* session_count,
                      std::size_t* max_connections,
                      std::size_t* max_request_body_bytes, std::string* api_key,
                      bool* verbose) {
  parser.AddOption("-i", "--host", "IP", "Bind address", "Server", host);
  parser.AddOption("-p", "--port", "N", "Port to listen on", "Server", port);
  parser.AddOption("-j", "--sessions", "N", "Preallocated GPU request sessions",
                   "Server", session_count);
  parser.AddOption("", "--max-connections", "N",
                   "Maximum simultaneous HTTP connections", "Server",
                   max_connections);
  parser.AddOption("", "--max-request-bytes", "N",
                   "Maximum HTTP request body bytes", "Server",
                   max_request_body_bytes);
  parser.AddOption("", "--api-key", "KEY", "API key", "Server", api_key);
  parser.AddFlag("-v", "--verbose", "Verbose logging", "General", verbose);
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
  bool verbose = false;
};

void AddServerOptionsForHelp(gufo::cli::ArgParser& parser,
                             ServerOptionHelpTargets* targets) {
  AddServerOptions(parser, &targets->host, &targets->port,
                   &targets->session_count, &targets->max_connections,
                   &targets->max_request_body_bytes, &targets->api_key,
                   &targets->verbose);
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

}  // namespace

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand) {
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

  if (subcommand == "audio" || subcommand == "tts") {
    std::filesystem::path default_model;
    std::size_t default_context = 4096;
    std::filesystem::path tts_model;
    std::filesystem::path asr_model;
    std::size_t tts_context_tokens = 4096;
    std::size_t asr_context_tokens = 1024;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve audio",
        "Start the Qwen3 audio HTTP server (Qwen3-TTS synthesis, Qwen3-ASR "
        "transcription, or both).");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen3-TTS 12Hz 1.7B model directory (alias for "
                     "--tts-model)",
                     "Model", &default_model);
    parser.AddOption(
        "-c", "--context", "N",
        "Qwen3-TTS context capacity (alias for --tts-context, default: 4096)",
        "Model", &default_context);
    parser.AddOption("", "--tts-model", "DIR",
                     "Qwen3-TTS 12Hz 1.7B model directory", "Model",
                     &tts_model);
    parser.AddOption("", "--asr-model", "DIR", "Qwen3-ASR-1.7B model directory",
                     "Model", &asr_model);
    parser.AddOption("", "--tts-context", "N",
                     "Qwen3-TTS context capacity (default: 4096)", "Model",
                     &tts_context_tokens);
    parser.AddOption("", "--asr-context", "N",
                     "Qwen3-ASR context capacity (default: 1024)", "Model",
                     &asr_context_tokens);
    parser.AddCustomOption(
        "", "--voice", "NAME=PATH",
        "Register a named Qwen3-TTS Base voice from a reference WAV "
        "(repeatable)",
        "Model",
        [](std::string_view, std::string_view, std::string*) { return true; });
    parser.AddCustomOption(
        "", "--voice-lang", "NAME=LANGUAGE",
        "Language a --voice speaks; used when a request omits 'language' "
        "(repeatable)",
        "Model",
        [](std::string_view, std::string_view, std::string*) { return true; });
    parser.AddCustomOption(
        "", "--voice-text", "NAME=TEXT|PATH",
        "Reference transcript for a --voice, given inline or as a file path; "
        "defaults to a .txt sidecar beside the WAV (repeatable)",
        "Model",
        [](std::string_view, std::string_view, std::string*) { return true; });
    ServerOptionHelpTargets server_help;
    AddServerOptionsForHelp(parser, &server_help);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "llm") {
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::string served_model_name;
    std::uint32_t max_context = 4096;
    std::size_t max_tokens = 128;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "off";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::size_t draft_tokens = 7;
    std::size_t min_draft_tokens = 1;
    std::size_t draft_vocab = 0;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    std::size_t max_pending_requests_per_client = 4;
    std::uint64_t request_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        static_cast<std::size_t>(4) * 1024U * 1024U * 1024U;
    std::size_t cache_disk_staging_bytes =
        static_cast<std::size_t>(512) * 1024U * 1024U;
    bool force_cpu = false;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");

    // Model & Context
    parser.AddOption(
        "-m", "--model", "PATH",
        "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)",
        "Model", &model);
    parser.AddOption("", "--served-model-name", "ID",
                     "Model identifier exposed by the OpenAI API", "Model",
                     &served_model_name);
    parser.AddOption("-c", "--context", "N",
                     "Maximum context tokens (default: 4096)", "Model",
                     &max_context);

    // Sampling Defaults
    parser.AddOption("-n", "--max-tokens", "N",
                     "Default maximum new tokens per response (default: 128)",
                     "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(parser, &sampling_config, "Sampling Defaults");

    // Reasoning Defaults
    parser.AddOption("", "--think", "MODE",
                     "Default reasoning mode: on, off, or auto (default: off)",
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
                     "HTTP draft backend: dspark, dflash, dflash2, mtp, or off",
                     "Speculative", &speculative_backend);
    parser.AddOption("", "--dflash-model", "PATH",
                     "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                     "Speculative", &dflash_model_path);
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
    parser.AddOption("", "--spec-draft-n-max", "N",
                     "llama.cpp-compatible alias for --draft-tokens",
                     "Speculative", &draft_tokens);

    parser.AddOption("", "--min-draft-tokens", "N",
                     "Adaptive draft floor (default: 1)", "Speculative",
                     &min_draft_tokens);
    parser.AddOption("", "--spec-draft-n-min", "N",
                     "llama.cpp-compatible alias for --min-draft-tokens",
                     "Speculative", &min_draft_tokens);
    parser.AddOption("", "--draft-vocab", "N",
                     "Qwen3.8-Flash-Next MTP: score drafts over the first N "
                     "token ids only (default: 0 = full vocabulary)",
                     "Speculative", &draft_vocab);
    parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512)",
        "Scheduling", &prefill_chunk_tokens);
    parser.AddOption("", "--max-pending", "N",
                     "Maximum queued generation requests (default: 16)",
                     "Scheduling", &max_pending_requests);
    parser.AddOption("", "--max-pending-per-client", "N",
                     "Maximum queued requests per X-Client-ID (default: 4)",
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
    parser.AddOption("", "--cache-disk", "DIR",
                     "Opt-in restart-safe continuation cache directory",
                     "Cache", &cache_disk_directory);
    parser.AddOption(
        "", "--cache-disk-bytes", "N",
        "Total retained disk-cache byte budget (default: 4294967296)", "Cache",
        &cache_disk_bytes);
    parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "Single-operation RAM staging byte limit (default: 536870912)", "Cache",
        &cache_disk_staging_bytes);
    parser.AddFlag("", "--cpu",
                   "Force CPU OpenMP execution fallback instead of GPU ROCm",
                   "Hardware", &force_cpu);
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
      << "  audio     Serve Qwen3-TTS and/or Qwen3-ASR endpoints "
         "(/v1/audio/speech,\n"
      << "            /v1/audio/transcriptions) via --tts-model and "
         "--asr-model\n\n"
      << "Server Options:\n"
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
      << "      --api-key <KEY>    Optional Bearer authorization API key "
         "(TODO: "
         "auth middleware)\n"
      << "  -v, --verbose          Print detailed server metrics and request "
         "traces\n"
      << "  -h, --help             Print help\n";
}

int RunServe(std::span<const char* const> args) {
  (void)std::setvbuf(stdout, nullptr, _IONBF, 0);
  (void)std::setvbuf(stderr, nullptr, _IONBF, 0);
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

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
  bool verbose = false;

  // Split into server-level options and modality subcommand
  std::vector<const char*> server_args;
  std::string subcommand;
  std::vector<const char*> sub_args;

  // Server options bind to the server no matter which side of the subcommand
  // they appear on: `gufo serve --port 8099 audio -m DIR` and
  // `gufo serve audio -m DIR --port 8099` are equivalent. No modality
  // subcommand defines a colliding short flag (-i/-p/-j/-v are server-only).
  //
  // Classification is positional and does not know which subcommand options
  // take a value, so a subcommand option whose *value* is spelled exactly like
  // a server flag (`--system -v`) would bind that value to the server. Use
  // `--system=-v` for such values.
  const auto is_server_value_option = [](std::string_view arg) {
    return arg == "-i" || arg == "--host" || arg == "-p" || arg == "--port" ||
           arg == "-j" || arg == "--sessions" || arg == "--api-key" ||
           arg == "--max-connections" || arg == "--max-request-bytes";
  };
  const auto is_server_inline_option = [](std::string_view arg) {
    return arg.starts_with("-i=") || arg.starts_with("--host=") ||
           arg.starts_with("-p=") || arg.starts_with("--port=") ||
           arg.starts_with("-j=") || arg.starts_with("--sessions=") ||
           arg.starts_with("--api-key=") ||
           arg.starts_with("--max-connections=") ||
           arg.starts_with("--max-request-bytes=");
  };
  const auto is_server_flag = [](std::string_view arg) {
    return arg == "-v" || arg == "--verbose";
  };

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (subcommand.empty()) {
      if (arg == "llm" || arg == "video" || arg == "audio" || arg == "tts") {
        subcommand = arg;
        continue;
      }
      if (arg == "asr" || arg == "stt") {
        // Removed in favor of the single audio server. Without this the token
        // falls through as an LLM positional and fails as a bad GGUF path.
        std::cerr << "Error: 'gufo serve asr' has been replaced by the audio "
                     "server.\n"
                  << "Use: gufo serve audio --asr-model <DIR>\n";
        return 2;
      }
      if (arg == "help") {
        if (i + 1 < args.size()) {
          PrintServeHelp("gufo", args[i + 1]);
        } else {
          PrintServeHelp("gufo");
        }
        return 0;
      }
      if (arg == "-h" || arg == "--help") {
        PrintServeHelp("gufo");
        return 0;
      }
    }
    if (is_server_value_option(arg)) {
      server_args.push_back(args[i]);
      if (i + 1 < args.size()) {
        server_args.push_back(args[++i]);
      }
      continue;
    }
    if (is_server_inline_option(arg) || is_server_flag(arg)) {
      server_args.push_back(args[i]);
      continue;
    }
    // Anything else belongs to the modality subcommand, which for the bare
    // `gufo serve -m model.gguf` form defaults to llm.
    sub_args.push_back(args[i]);
  }

  // Parse server options
  gufo::cli::ArgParser server_parser(
      "gufo serve", "Start the OpenAI-compatible HTTP server.");
  AddServerOptions(server_parser, &host, &port, &session_count,
                   &max_connections, &max_request_body_bytes, &api_key,
                   &verbose);

  std::string parse_err;
  if (!server_parser.Parse(server_args, &parse_err)) {
    std::cerr << "Error: " << parse_err << "\n";
    PrintServeHelp("gufo");
    return 2;
  }
  if (session_count == 0 || max_connections == 0 ||
      max_request_body_bytes == 0) {
    std::cerr << "Error: server limits must be positive\n";
    return 2;
  }

  std::shared_ptr<server::InferenceBackend> backend;
  std::shared_ptr<server::VideoJobService> video_jobs;
  std::shared_ptr<server::TtsService> tts;
  std::shared_ptr<server::AsrService> asr;

  if (subcommand == "video") {
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

    if (!video_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "video");
      return 2;
    }
    if (video_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "video");
      return 0;
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
  } else if (subcommand == "audio" || subcommand == "tts") {
    // One audio server hosts Qwen3-TTS synthesis, Qwen3-ASR transcription, or
    // both: HttpServer already dispatches /v1/audio/speech and
    // /v1/audio/transcriptions from independent services. Each service is
    // selected by naming its checkpoint; a bare --model/--context is a
    // backward-compatible alias for the TTS pair.
    const std::string_view help_topic = "audio";

    std::filesystem::path default_model;
    std::size_t default_context = 4096;
    std::filesystem::path tts_model;
    std::filesystem::path asr_model;
    std::size_t tts_context_tokens = 4096;
    std::size_t asr_context_tokens = 1024;

    gufo::cli::ArgParser audio_parser(
        "gufo serve audio",
        "Start the Qwen3 audio HTTP server (Qwen3-TTS synthesis, Qwen3-ASR "
        "transcription, or both).");
    audio_parser.AddOption(
        "-m", "--model", "DIR",
        "Qwen3-TTS 12Hz 1.7B model directory (alias for --tts-model)", "Model",
        &default_model);
    audio_parser.AddOption(
        "-c", "--context", "N",
        "Qwen3-TTS context capacity (alias for --tts-context, default: 4096)",
        "Model", &default_context);
    audio_parser.AddOption("", "--tts-model", "DIR",
                           "Qwen3-TTS 12Hz 1.7B model directory", "Model",
                           &tts_model);
    audio_parser.AddOption("", "--asr-model", "DIR",
                           "Qwen3-ASR-1.7B model directory", "Model",
                           &asr_model);
    audio_parser.AddOption("", "--tts-context", "N",
                           "Qwen3-TTS context capacity (default: 4096)",
                           "Model", &tts_context_tokens);
    audio_parser.AddOption("", "--asr-context", "N",
                           "Qwen3-ASR context capacity (default: 1024)",
                           "Model", &asr_context_tokens);

    std::map<std::string, server::TtsVoicePreset> voice_presets;
    std::vector<std::pair<std::string, std::string>> voice_specs;
    std::map<std::string, std::string> voice_text_specs;
    std::map<std::string, std::string> voice_lang_specs;
    audio_parser.AddCustomOption(
        "", "--voice", "NAME=PATH",
        "Register a named Qwen3-TTS Base voice from a reference WAV "
        "(repeatable)",
        "Model",
        [&voice_specs](std::string_view, std::string_view value,
                       std::string* err) {
          std::string name;
          std::string path;
          if (!SplitNameValue(value, "--voice", &name, &path, err)) {
            return false;
          }
          voice_specs.emplace_back(std::move(name), std::move(path));
          return true;
        });
    audio_parser.AddCustomOption(
        "", "--voice-lang", "NAME=LANGUAGE",
        "Language a --voice speaks; used when a request omits 'language' "
        "(repeatable)",
        "Model",
        [&voice_lang_specs](std::string_view, std::string_view value,
                            std::string* err) {
          std::string name;
          std::string language;
          if (!SplitNameValue(value, "--voice-lang", &name, &language, err)) {
            return false;
          }
          if (!voice_lang_specs.emplace(std::move(name), std::move(language))
                   .second) {
            *err = "duplicate --voice-lang name";
            return false;
          }
          return true;
        });
    audio_parser.AddCustomOption(
        "", "--voice-text", "NAME=TEXT|PATH",
        "Reference transcript for a --voice, given inline or as a file path; "
        "defaults to a .txt sidecar beside the WAV (repeatable)",
        "Model",
        [&voice_text_specs](std::string_view, std::string_view value,
                            std::string* err) {
          std::string name;
          std::string text;
          if (!SplitNameValue(value, "--voice-text", &name, &text, err)) {
            return false;
          }
          if (!voice_text_specs.emplace(std::move(name), std::move(text))
                   .second) {
            *err = "duplicate --voice-text name";
            return false;
          }
          return true;
        });

    if (!audio_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", help_topic);
      return 2;
    }
    if (audio_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", help_topic);
      return 0;
    }

    // --model/--context are aliases for the TTS pair. Supplying both
    // spellings for the same service is ambiguous.
    if (!default_model.empty()) {
      if (!tts_model.empty()) {
        std::cerr << "Error: --model conflicts with --tts-model\n";
        return 2;
      }
      tts_model = default_model;
    }
    if (default_context != 4096U) {
      tts_context_tokens = default_context;
    }

    if (!BuildVoicePresets(voice_specs, voice_text_specs, voice_lang_specs,
                           &voice_presets, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", help_topic);
      return 2;
    }

    if (!voice_presets.empty() && tts_model.empty()) {
      std::cerr << "Error: --voice requires a Qwen3-TTS checkpoint\n";
      PrintServeHelp("gufo", help_topic);
      return 2;
    }
    if (tts_model.empty() && asr_model.empty()) {
      std::cerr << "Error: at least one of --tts-model <DIR> or --asr-model "
                   "<DIR> is required for the audio server\n";
      PrintServeHelp("gufo", help_topic);
      return 2;
    }
    if (!asr_model.empty() && asr_context_tokens < 32U) {
      std::cerr << "Error: the ASR context must be at least 32\n";
      PrintServeHelp("gufo", help_topic);
      return 2;
    }

    if (!tts_model.empty()) {
      tts = std::make_shared<server::TtsService>(server::TtsServiceOptions{
          .model_root = tts_model,
          .native_context_tokens = tts_context_tokens,
          .validate_model = true,
          .model_id = {},
          .voices = {},
          .voice_presets = std::move(voice_presets),
          .runner = {},
      });
      if (!tts->ready()) {
        std::cerr << "Error enabling Qwen3-TTS service: "
                  << tts->initialization_error() << '\n';
        return 1;
      }
    }

    if (!asr_model.empty()) {
      asr = std::make_shared<server::AsrService>(server::AsrServiceOptions{
          .model_root = asr_model,
          .native_context_tokens = asr_context_tokens,
          .validate_model = true,
          .model_id = {},
          .runner = {},
      });
      if (!asr->ready()) {
        std::cerr << "Error enabling Qwen3-ASR service: "
                  << asr->initialization_error() << '\n';
        return 1;
      }
    }
  } else {
    // Default to LLM server
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::string served_model_name;
    std::uint32_t max_context = 4096;
    std::size_t max_tokens = 128;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "off";
    std::string reasoning_effort = "auto";
    std::string preserve_thinking = "auto";
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string dspark_model_path;
    std::string mtp_model_path;
    std::size_t draft_tokens = 7;
    std::size_t min_draft_tokens = 1;
    std::size_t draft_vocab = 0;
    std::size_t prefill_chunk_tokens =
        server::kDefaultDecodeActivePrefillTokens;
    std::size_t max_pending_requests = 16;
    std::size_t max_pending_requests_per_client = 4;
    std::uint64_t request_timeout_ms = 0;
    std::size_t max_output_bytes = server::kDefaultMaxOutputBytes;
    std::size_t max_buffered_output_bytes =
        server::kDefaultMaxBufferedOutputBytes;
    std::size_t max_buffered_output_bytes_total =
        server::kDefaultMaxBufferedOutputBytesTotal;
    std::filesystem::path cache_disk_directory;
    std::size_t cache_disk_bytes =
        static_cast<std::size_t>(4) * 1024U * 1024U * 1024U;
    std::size_t cache_disk_staging_bytes =
        static_cast<std::size_t>(512) * 1024U * 1024U;
    bool force_cpu = false;

    gufo::cli::ArgParser llm_parser(
        "gufo serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    llm_parser.AddOption(
        "-m", "--model", "PATH",
        "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)",
        "Model", &model);
    llm_parser.AddOption("", "--served-model-name", "ID",
                         "Model identifier exposed by the OpenAI API", "Model",
                         &served_model_name);
    llm_parser.AddOption("-c", "--context", "N",
                         "Maximum context tokens (default: 4096)", "Model",
                         &max_context);
    llm_parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default maximum new tokens per response (default: 128)",
        "Sampling Defaults", &max_tokens);
    RegisterSamplingOptions(llm_parser, &sampling_config, "Sampling Defaults");
    llm_parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode: on, off, or auto (default: off)",
        "Reasoning Defaults", &reasoning_mode);
    llm_parser.AddOption(
        "", "--reasoning-effort", "LEVEL",
        "Default effort: auto, minimal, low, medium, high, xhigh, or max",
        "Reasoning Defaults", &reasoning_effort);
    llm_parser.AddOption("", "--preserve-thinking", "MODE",
                         "Replay prior reasoning: on, off, or auto",
                         "Reasoning Defaults", &preserve_thinking);
    llm_parser.AddOption(
        "", "--speculative", "MODE",
        "HTTP draft backend: dspark, dflash, dflash2, mtp, or off",
        "Speculative", &speculative_backend);
    llm_parser.AddOption("", "--dflash-model", "PATH",
                         "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                         "Speculative", &dflash_model_path);
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
    llm_parser.AddOption("", "--spec-draft-n-max", "N",
                         "llama.cpp-compatible alias for --draft-tokens",
                         "Speculative", &draft_tokens);
    llm_parser.AddOption("", "--min-draft-tokens", "N",
                         "Adaptive draft floor (default: 1)", "Speculative",
                         &min_draft_tokens);
    llm_parser.AddOption("", "--spec-draft-n-min", "N",
                         "llama.cpp-compatible alias for --min-draft-tokens",
                         "Speculative", &min_draft_tokens);
    llm_parser.AddOption("", "--draft-vocab", "N",
                         "Qwen3.8-Flash-Next MTP: score drafts over the first "
                         "N token ids only (default: 0 = full vocabulary)",
                         "Speculative", &draft_vocab);
    llm_parser.AddOption(
        "", "--prefill-chunk", "N",
        "Maximum prompt tokens between active decode rounds (default: 512)",
        "Scheduling", &prefill_chunk_tokens);
    llm_parser.AddOption("", "--max-pending", "N",
                         "Maximum queued generation requests (default: 16)",
                         "Scheduling", &max_pending_requests);
    llm_parser.AddOption("", "--max-pending-per-client", "N",
                         "Maximum queued requests per X-Client-ID (default: 4)",
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
    llm_parser.AddOption("", "--cache-disk", "DIR",
                         "Opt-in restart-safe continuation cache directory",
                         "Cache", &cache_disk_directory);
    llm_parser.AddOption(
        "", "--cache-disk-bytes", "N",
        "Total retained disk-cache byte budget (default: 4294967296)", "Cache",
        &cache_disk_bytes);
    llm_parser.AddOption(
        "", "--cache-disk-staging-bytes", "N",
        "Single-operation RAM staging byte limit (default: 536870912)", "Cache",
        &cache_disk_staging_bytes);
    llm_parser.AddFlag(
        "", "--cpu", "Force CPU OpenMP execution fallback instead of GPU ROCm",
        "Hardware", &force_cpu);

    if (!llm_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "llm");
      return 2;
    }
    if (llm_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "llm");
      return 0;
    }
    bool sampling_valid = true;
    try {
      sampling_config.Validate();
    } catch (const std::invalid_argument&) {
      sampling_valid = false;
    }
    if (max_tokens == 0 || prefill_chunk_tokens == 0 ||
        max_pending_requests == 0 || max_pending_requests_per_client == 0 ||
        max_pending_requests_per_client > max_pending_requests ||
        max_output_bytes == 0 || max_buffered_output_bytes == 0 ||
        max_buffered_output_bytes_total == 0 ||
        (!cache_disk_directory.empty() &&
         (cache_disk_bytes == 0 || cache_disk_staging_bytes == 0)) ||
        request_timeout_ms > static_cast<std::uint64_t>(
                                 std::chrono::milliseconds::max().count()) ||
        !sampling_valid || sampling_config.temperature > 2.0F) {
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
    } else if (speculative_backend == "dflash" ||
               speculative_backend == "dflash2" ||
               speculative_backend == "dflash-2") {
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
    speculative_config.draft_model_path =
        speculative_config.backend == server::TextSpeculativeBackend::kDSpark
            ? dspark_model_path
        : speculative_config.backend == server::TextSpeculativeBackend::kMtp
            ? mtp_model_path
            : dflash_model_path;
    if (draft_vocab > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Error: --draft-vocab is out of range\n";
      return 2;
    }
    speculative_config.draft_vocab = static_cast<std::uint32_t>(draft_vocab);
    speculative_config.max_draft_tokens =
        static_cast<std::uint32_t>(draft_tokens);
    speculative_config.min_draft_tokens =
        static_cast<std::uint32_t>(min_draft_tokens);
    std::string err;
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
                       },
                       speculative_config,
                       server::TextDiskCacheConfig{
                           .directory = cache_disk_directory,
                           .capacity_bytes = cache_disk_bytes,
                           .staging_capacity_bytes = cache_disk_staging_bytes,
                           .model_artifact_fingerprint = {},
                       })) {
      std::cerr << "Error loading model '" << model << "': " << err << "\n";
      return 1;
    }
    if (speculative_config.backend == server::TextSpeculativeBackend::kDFlash) {
      std::cout << "[Speculative]: DFlash enabled (max_draft_tokens="
                << speculative_config.max_draft_tokens
                << ", min_draft_tokens=" << speculative_config.min_draft_tokens
                << ")\n";
    } else if (speculative_config.backend ==
               server::TextSpeculativeBackend::kDSpark) {
      std::cout << "[Speculative]: DSpark enabled\n";
    } else if (speculative_config.backend ==
               server::TextSpeculativeBackend::kMtp) {
      std::cout << "[Speculative]: MTP enabled (max_draft_tokens="
                << speculative_config.max_draft_tokens
                << ", draft_vocab=" << speculative_config.draft_vocab << ")\n";
    }
    backend->set_model_id(served_model_name);
    backend->set_sampling_defaults(max_tokens, sampling_config);
    backend->set_reasoning_defaults(*reasoning_defaults);
  }

  server::HttpServer server(
      host, port, backend, video_jobs, tts, asr,
      server::HttpServerLimits{
          .max_request_body_bytes = max_request_body_bytes,
          .max_connections = max_connections,
      });
  std::string err;
  if (!server.start(&err)) {
    std::cerr << "Error starting HTTP server: " << err << "\n";
    return 1;
  }
  server.run();
  return 0;
}

}  // namespace gufo::cli
