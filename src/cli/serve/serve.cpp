#include "src/cli/serve/serve.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/sampling_options.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"
#include "src/core/speculative/draft_policy.hpp"

namespace gufo::cli {

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
    parser.PrintHelp();
    return;
  }

  if (subcommand == "audio" || subcommand == "tts") {
    std::filesystem::path tts_model;
    std::size_t tts_context_tokens = 4096;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve audio",
        "Start the Qwen3-TTS text-to-speech HTTP synthesis server.");
    parser.AddOption("-m", "--model", "DIR",
                     "Qwen3-TTS 12Hz 1.7B model directory", "Model",
                     &tts_model);
    parser.AddOption(
        "-c", "--context", "N",
        "Native prompt+generation context capacity (default: 4096)", "Model",
        &tts_context_tokens);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "asr" || subcommand == "stt") {
    std::filesystem::path asr_model;
    std::size_t asr_context_tokens = 1024;

    gufo::cli::ArgParser parser(
        std::string(program_name) + " serve asr",
        "Start the Qwen3-ASR speech-to-text HTTP transcription server.");
    parser.AddOption("-m", "--model", "DIR", "Qwen3-ASR-1.7B model directory",
                     "Model", &asr_model);
    parser.AddOption(
        "-c", "--context", "N",
        "Native prompt+generation context capacity (default: 1024)", "Model",
        &asr_context_tokens);
    parser.PrintHelp();
    return;
  }

  if (subcommand == "llm") {
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::string served_model_name;
    std::uint32_t max_context = 4096;
    std::string system_prompt =
        "You are a helpful, respectful, and honest assistant.";
    std::string chat_template;
    bool use_chat_template = true;
    std::size_t max_tokens = 128;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::int64_t reasoning_budget = -1;
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string mtp_model_path;
    std::size_t draft_tokens = 7;
    std::string draft_policy = "auto";
    std::size_t min_draft_tokens = 1;
    float draft_p_min = 0.0F;
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

    // Prompt Defaults
    parser.AddOption("", "--system", "PROMPT",
                     "Default system instructions (default: helpful assistant)",
                     "Prompt Defaults", &system_prompt);
    parser.AddInverseFlag("", "--raw",
                          "Disable chat template framing by default",
                          "Prompt Defaults", &use_chat_template);
    parser.AddOption("", "--chat-template", "NAME",
                     "Default Jinja chat template override (e.g. qwen, chatml, "
                     "deepseek)",
                     "Prompt Defaults", &chat_template);

    // Reasoning Defaults
    parser.AddOption("", "--think", "MODE",
                     "Default reasoning mode for thinking models: on, off, or "
                     "auto (default: auto) (TODO: qwen, deepseek)",
                     "Reasoning Defaults", &reasoning_mode);
    parser.AddOption("", "--reasoning-budget", "N",
                     "Default token cap for thinking traces (default: -1 = "
                     "unlimited) (TODO: qwen, deepseek)",
                     "Reasoning Defaults", &reasoning_budget);

    // Speculative & Hardware
    parser.AddOption("", "--speculative", "MODE",
                     "HTTP draft backend: dflash, dflash2, or off",
                     "Speculative", &speculative_backend);
    parser.AddOption("", "--dflash-model", "PATH",
                     "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                     "Speculative", &dflash_model_path);
    parser.AddOption("", "--mtp-model", "PATH",
                     "Path to quantized Qwen MTP draft head GGUF file",
                     "Speculative", &mtp_model_path);
    parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7)",
        "Speculative", &draft_tokens);
    parser.AddOption("", "--spec-draft-n-max", "N",
                     "llama.cpp-compatible alias for --draft-tokens",
                     "Speculative", &draft_tokens);
    parser.AddOption(
        "", "--draft-policy", "MODE",
        "Draft sizing: auto, fixed, rolling, or accepted-ema (default: auto, "
        "which is fixed for DFlash-2 and rolling otherwise)",
        "Speculative", &draft_policy);
    parser.AddOption("", "--min-draft-tokens", "N",
                     "Adaptive draft floor (default: 1)", "Speculative",
                     &min_draft_tokens);
    parser.AddOption("", "--spec-draft-n-min", "N",
                     "llama.cpp-compatible alias for --min-draft-tokens",
                     "Speculative", &min_draft_tokens);
    parser.AddOption(
        "", "--spec-draft-p-min", "P",
        "Stop at the first draft token below confidence P; 0 disables "
        "(default: 0)",
        "Speculative", &draft_p_min);
    parser.AddOption("", "--draft-p-min", "P", "Alias for --spec-draft-p-min",
                     "Speculative", &draft_p_min);
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
      << "  audio     Serve Qwen3-TTS text-to-speech endpoint "
         "(/v1/audio/speech)\n\n"
      << "  asr       Serve Qwen3-ASR speech-to-text endpoint "
         "(/v1/audio/transcriptions)\n\n"
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

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (subcommand.empty()) {
      if (arg == "llm" || arg == "video" || arg == "audio" || arg == "tts" ||
          arg == "asr" || arg == "stt") {
        subcommand = arg;
        continue;
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
      // Check for server-specific flags
      if (arg == "-i" || arg == "--host" || arg == "-p" || arg == "--port" ||
          arg == "-j" || arg == "--sessions" || arg == "--api-key") {
        server_args.push_back(args[i]);
        if (i + 1 < args.size()) {
          server_args.push_back(args[++i]);
        }
        continue;
      }
      if (arg.starts_with("-i=") || arg.starts_with("--host=") ||
          arg.starts_with("-p=") || arg.starts_with("--port=") ||
          arg.starts_with("-j=") || arg.starts_with("--sessions=") ||
          arg.starts_with("--api-key=") ||
          arg.starts_with("--max-connections=") ||
          arg.starts_with("--max-request-bytes=")) {
        server_args.push_back(args[i]);
        continue;
      }
      if (arg == "--max-connections" || arg == "--max-request-bytes") {
        server_args.push_back(args[i]);
        if (i + 1 < args.size()) {
          server_args.push_back(args[++i]);
        }
        continue;
      }
      if (arg == "-v" || arg == "--verbose") {
        server_args.push_back(args[i]);
        continue;
      }
      // If none of the above, it's a direct LLM option (e.g. -m model.gguf)
      sub_args.push_back(args[i]);
    } else {
      sub_args.push_back(args[i]);
    }
  }

  // Parse server options
  gufo::cli::ArgParser server_parser(
      "gufo serve", "Start the OpenAI-compatible HTTP server.");
  server_parser.AddOption("-i", "--host", "IP", "Bind address", "Server",
                          &host);
  server_parser.AddOption("-p", "--port", "N", "Port to listen on", "Server",
                          &port);
  server_parser.AddOption("-j", "--sessions", "N",
                          "Preallocated GPU request sessions", "Server",
                          &session_count);
  server_parser.AddOption("", "--max-connections", "N",
                          "Maximum simultaneous HTTP connections", "Server",
                          &max_connections);
  server_parser.AddOption("", "--max-request-bytes", "N",
                          "Maximum HTTP request body bytes", "Server",
                          &max_request_body_bytes);
  server_parser.AddOption("", "--api-key", "KEY", "API key", "Server",
                          &api_key);
  server_parser.AddFlag("-v", "--verbose", "Verbose logging", "General",
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
    std::filesystem::path tts_model;
    std::size_t tts_context_tokens = 4096;

    gufo::cli::ArgParser audio_parser(
        "gufo serve audio", "Start the Qwen3-TTS audio synthesis server.");
    audio_parser.AddOption("-m", "--model", "DIR",
                           "Qwen3-TTS 12Hz 1.7B model directory", "Model",
                           &tts_model);
    audio_parser.AddOption(
        "-c", "--context", "N",
        "Native prompt+generation context capacity (default: 4096)", "Model",
        &tts_context_tokens);

    if (!audio_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "audio");
      return 2;
    }
    if (audio_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "audio");
      return 0;
    }

    if (tts_model.empty()) {
      std::cerr << "Error: --model <DIR> is required for audio server\n";
      PrintServeHelp("gufo", "audio");
      return 2;
    }

    tts = std::make_shared<server::TtsService>(server::TtsServiceOptions{
        .model_root = tts_model,
        .native_context_tokens = tts_context_tokens,
        .validate_model = true,
        .model_id = {},
        .voices = {},
        .runner = {},
    });
    if (!tts->ready()) {
      std::cerr << "Error enabling Qwen3-TTS service: "
                << tts->initialization_error() << '\n';
      return 1;
    }
  } else if (subcommand == "asr" || subcommand == "stt") {
    std::filesystem::path asr_model;
    std::size_t asr_context_tokens = 1024;

    gufo::cli::ArgParser asr_parser(
        "gufo serve asr",
        "Start the Qwen3-ASR speech-to-text transcription server.");
    asr_parser.AddOption("-m", "--model", "DIR",
                         "Qwen3-ASR-1.7B model directory", "Model", &asr_model);
    asr_parser.AddOption(
        "-c", "--context", "N",
        "Native prompt+generation context capacity (default: 1024)", "Model",
        &asr_context_tokens);

    if (!asr_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("gufo", "asr");
      return 2;
    }
    if (asr_parser.IsHelpRequested()) {
      PrintServeHelp("gufo", "asr");
      return 0;
    }
    if (asr_model.empty() || asr_context_tokens < 32U) {
      std::cerr << "Error: --model <DIR> and a context of at least 32 are "
                   "required for ASR server\n";
      PrintServeHelp("gufo", "asr");
      return 2;
    }

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
  } else {
    // Default to LLM server
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::string served_model_name;
    std::uint32_t max_context = 4096;
    std::string system_prompt =
        "You are a helpful, respectful, and honest assistant.";
    std::string chat_template;
    bool use_chat_template = true;
    std::size_t max_tokens = 128;
    sampling::SamplingConfig sampling_config;
    std::string reasoning_mode = "auto";
    std::int64_t reasoning_budget = -1;
    std::string speculative_backend;
    std::string dflash_model_path;
    std::string mtp_model_path;
    std::size_t draft_tokens = 7;
    std::string draft_policy = "auto";
    std::size_t min_draft_tokens = 1;
    float draft_p_min = 0.0F;
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
        "", "--system", "PROMPT",
        "Default system instructions (default: helpful assistant)",
        "Prompt Defaults", &system_prompt);
    llm_parser.AddInverseFlag("", "--raw",
                              "Disable chat template framing by default",
                              "Prompt Defaults", &use_chat_template);
    llm_parser.AddOption(
        "", "--chat-template", "NAME",
        "Default Jinja chat template override (e.g. qwen, chatml, deepseek)",
        "Prompt Defaults", &chat_template);
    llm_parser.AddOption(
        "", "--think", "MODE",
        "Default reasoning mode for thinking models: on, off, or auto "
        "(default: auto) (TODO: qwen, deepseek)",
        "Reasoning Defaults", &reasoning_mode);
    llm_parser.AddOption(
        "", "--reasoning-budget", "N",
        "Default token cap for thinking traces (default: -1 = unlimited) "
        "(TODO: qwen, deepseek)",
        "Reasoning Defaults", &reasoning_budget);
    llm_parser.AddOption("", "--speculative", "MODE",
                         "HTTP draft backend: dflash, dflash2, or off",
                         "Speculative", &speculative_backend);
    llm_parser.AddOption("", "--dflash-model", "PATH",
                         "Path to quantized Qwen DFlash/DFlash-2 GGUF file",
                         "Speculative", &dflash_model_path);
    llm_parser.AddOption("", "--mtp-model", "PATH",
                         "Path to quantized Qwen MTP draft head GGUF file",
                         "Speculative", &mtp_model_path);
    llm_parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 7)",
        "Speculative", &draft_tokens);
    llm_parser.AddOption("", "--spec-draft-n-max", "N",
                         "llama.cpp-compatible alias for --draft-tokens",
                         "Speculative", &draft_tokens);
    llm_parser.AddOption(
        "", "--draft-policy", "MODE",
        "Draft sizing: auto, fixed, rolling, or accepted-ema (default: auto, "
        "which is fixed for DFlash-2 and rolling otherwise)",
        "Speculative", &draft_policy);
    llm_parser.AddOption("", "--min-draft-tokens", "N",
                         "Adaptive draft floor (default: 1)", "Speculative",
                         &min_draft_tokens);
    llm_parser.AddOption("", "--spec-draft-n-min", "N",
                         "llama.cpp-compatible alias for --min-draft-tokens",
                         "Speculative", &min_draft_tokens);
    llm_parser.AddOption(
        "", "--spec-draft-p-min", "P",
        "Stop at the first draft token below confidence P; 0 disables "
        "(default: 0)",
        "Speculative", &draft_p_min);
    llm_parser.AddOption("", "--draft-p-min", "P",
                         "Alias for --spec-draft-p-min", "Speculative",
                         &draft_p_min);
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
        draft_tokens > std::numeric_limits<std::uint32_t>::max() ||
        !std::isfinite(draft_p_min) || draft_p_min < 0.0F ||
        draft_p_min > 1.0F) {
      std::cerr << "Error: speculative draft limits are invalid\n";
      return 2;
    }

    server::TextSpeculativeConfig speculative_config;
    if (speculative_backend.empty() || speculative_backend == "off") {
      speculative_config.backend = server::TextSpeculativeBackend::kDisabled;
    } else if (speculative_backend == "dflash" ||
               speculative_backend == "dflash2" ||
               speculative_backend == "dflash-2") {
      speculative_config.backend = server::TextSpeculativeBackend::kDFlash;
    } else {
      std::cerr << "Error: speculative backend '" << speculative_backend
                << "' is not supported by the HTTP server\n";
      return 2;
    }
    speculative_config.draft_model_path = dflash_model_path;
    speculative_config.max_draft_tokens =
        static_cast<std::uint32_t>(draft_tokens);
    speculative_config.min_draft_tokens =
        static_cast<std::uint32_t>(min_draft_tokens);
    speculative_config.draft_p_min = draft_p_min;
    // `auto` follows the measured best per backend; see ResolveDraftPolicy.
    const std::string_view resolved_draft_policy =
        speculative::ResolveDraftPolicy(draft_policy, true);
    if (resolved_draft_policy == "fixed") {
      speculative_config.draft_policy = server::TextDraftPolicy::kFixed;
    } else if (resolved_draft_policy == "rolling") {
      speculative_config.draft_policy =
          server::TextDraftPolicy::kRollingAcceptance;
    } else if (resolved_draft_policy == "accepted-ema") {
      speculative_config.draft_policy =
          server::TextDraftPolicy::kAcceptedTokenEma;
    } else {
      std::cerr << "Error: unknown speculative draft policy '" << draft_policy
                << "'\n";
      return 2;
    }

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
                << ", draft_p_min=" << speculative_config.draft_p_min << ")\n";
    }
    backend->set_model_id(served_model_name);
    backend->set_sampling_defaults(max_tokens, sampling_config);
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
