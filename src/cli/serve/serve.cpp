#include "src/cli/serve/serve.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_jobs.hpp"
#include "src/cli/video/video.hpp"

namespace strix::cli {

void PrintServeHelp(std::string_view program_name,
                    std::string_view subcommand) {
  if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    strix::cli::ArgParser parser(
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

    strix::cli::ArgParser parser(
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

  if (subcommand == "llm") {
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::uint32_t max_context = 4096;
    std::string system_prompt =
        "You are a helpful, respectful, and honest assistant.";
    std::string chat_template;
    bool use_chat_template = true;
    std::size_t max_tokens = 128;
    float temperature = 0.0F;
    float top_p = 1.0F;
    std::int32_t top_k = 0;
    float min_p = 0.0F;
    std::int64_t seed = -1;
    float repeat_penalty = 1.0F;
    std::size_t repeat_last_n = 64;
    std::string reasoning_mode = "auto";
    std::int64_t reasoning_budget = -1;
    std::string speculative_backend;
    std::string mtp_model_path;
    std::size_t draft_tokens = 3;
    bool force_cpu = false;

    strix::cli::ArgParser parser(
        std::string(program_name) + " serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");

    // Model & Context
    parser.AddOption(
        "-m", "--model", "PATH",
        "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)",
        "Model", &model);
    parser.AddOption("-c", "--context", "N",
                     "Maximum context tokens (default: 4096)", "Model",
                     &max_context);

    // Sampling Defaults
    parser.AddOption("-n", "--max-tokens", "N",
                     "Default maximum new tokens per response (default: 128)",
                     "Sampling Defaults", &max_tokens);
    parser.AddOption("-t", "--temperature", "T",
                     "Default sampling temperature (default: 0.0 = greedy)",
                     "Sampling Defaults", &temperature);
    parser.AddOption(
        "", "--top-p", "P",
        "Default nucleus sampling probability cutoff (default: 1.0) "
        "(TODO: qwen)",
        "Sampling Defaults", &top_p);
    parser.AddOption(
        "", "--top-k", "K",
        "Default Top-K token cutoff (default: 0 = disabled) (TODO: "
        "qwen)",
        "Sampling Defaults", &top_k);
    parser.AddOption("", "--min-p", "P",
                     "Default Min-P threshold relative to top token (default: "
                     "0.0) (TODO: qwen)",
                     "Sampling Defaults", &min_p);
    parser.AddOption("-s", "--seed", "N",
                     "Default RNG seed for generation (default: -1 = random)",
                     "Sampling Defaults", &seed);
    parser.AddOption(
        "", "--repeat-penalty", "N",
        "Default repetition penalty multiplier (default: 1.0 = off) "
        "(TODO: qwen, deepseek)",
        "Sampling Defaults", &repeat_penalty);
    parser.AddOption("", "--repeat-last-n", "N",
                     "Default lookback window for repetition penalty (default: "
                     "64) (TODO: qwen, deepseek)",
                     "Sampling Defaults", &repeat_last_n);

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
                     "Draft backend: mtp, mtp-npu, npu, pld, or self",
                     "Speculative", &speculative_backend);
    parser.AddOption("", "--mtp-model", "PATH",
                     "Path to quantized Qwen MTP draft head GGUF file",
                     "Speculative", &mtp_model_path);
    parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 3)",
        "Speculative", &draft_tokens);
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
      << "Server Options:\n"
      << "  -i, --host <IP>        Bind address (default: 127.0.0.1)\n"
      << "  -p, --port <N>         Port to listen on (default: 8080)\n"
      << "  -j, --sessions <N>     Preallocated GPU request sessions (default: "
         "1)\n"
      << "      --api-key <KEY>    Optional Bearer authorization API key "
         "(TODO: "
         "auth middleware)\n"
      << "  -v, --verbose          Print detailed server metrics and request "
         "traces\n"
      << "  -h, --help             Print help\n";
}

int RunServe(std::span<const char* const> args) {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::size_t session_count = 1;
  std::string api_key;
  bool verbose = false;

  // Split into server-level options and modality subcommand
  std::vector<const char*> server_args;
  std::string subcommand;
  std::vector<const char*> sub_args;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (subcommand.empty()) {
      if (arg == "llm" || arg == "video" || arg == "audio" || arg == "tts") {
        subcommand = arg;
        continue;
      }
      if (arg == "help") {
        if (i + 1 < args.size()) {
          PrintServeHelp("strix", args[i + 1]);
        } else {
          PrintServeHelp("strix");
        }
        return 0;
      }
      if (arg == "-h" || arg == "--help") {
        PrintServeHelp("strix");
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
          arg.starts_with("--api-key=")) {
        server_args.push_back(args[i]);
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
  strix::cli::ArgParser server_parser(
      "strix serve", "Start the OpenAI-compatible HTTP server.");
  server_parser.AddOption("-i", "--host", "IP", "Bind address", "Server",
                          &host);
  server_parser.AddOption("-p", "--port", "N", "Port to listen on", "Server",
                          &port);
  server_parser.AddOption("-j", "--sessions", "N",
                          "Preallocated GPU request sessions", "Server",
                          &session_count);
  server_parser.AddOption("", "--api-key", "KEY", "API key", "Server",
                          &api_key);
  server_parser.AddFlag("-v", "--verbose", "Verbose logging", "General",
                        &verbose);

  std::string parse_err;
  if (!server_parser.Parse(server_args, &parse_err)) {
    std::cerr << "Error: " << parse_err << "\n";
    PrintServeHelp("strix");
    return 2;
  }

  std::shared_ptr<server::InferenceBackend> backend;
  std::shared_ptr<server::VideoJobService> video_jobs;
  std::shared_ptr<server::TtsService> tts;

  if (subcommand == "video") {
    std::filesystem::path video_model;
    std::filesystem::path video_root = "video-jobs";
    std::filesystem::path video_manifest = DefaultH3SourceManifest();
    std::uint64_t video_ttl_seconds = 3600;

    strix::cli::ArgParser video_parser(
        "strix serve video", "Start the MiniMax H3 video generation server.");
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
      PrintServeHelp("strix", "video");
      return 2;
    }
    if (video_parser.IsHelpRequested()) {
      PrintServeHelp("strix", "video");
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
      PrintServeHelp("strix", "video");
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

    strix::cli::ArgParser audio_parser(
        "strix serve audio", "Start the Qwen3-TTS audio synthesis server.");
    audio_parser.AddOption("-m", "--model", "DIR",
                           "Qwen3-TTS 12Hz 1.7B model directory", "Model",
                           &tts_model);
    audio_parser.AddOption(
        "-c", "--context", "N",
        "Native prompt+generation context capacity (default: 4096)", "Model",
        &tts_context_tokens);

    if (!audio_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("strix", "audio");
      return 2;
    }
    if (audio_parser.IsHelpRequested()) {
      PrintServeHelp("strix", "audio");
      return 0;
    }

    if (tts_model.empty()) {
      std::cerr << "Error: --model <DIR> is required for audio server\n";
      PrintServeHelp("strix", "audio");
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
  } else {
    // Default to LLM server
    std::string model = "models/Qwen3.5-4B-BF16.gguf";
    std::uint32_t max_context = 4096;
    std::string system_prompt =
        "You are a helpful, respectful, and honest assistant.";
    std::string chat_template;
    bool use_chat_template = true;
    std::size_t max_tokens = 128;
    float temperature = 0.0F;
    float top_p = 1.0F;
    std::int32_t top_k = 0;
    float min_p = 0.0F;
    std::int64_t seed = -1;
    float repeat_penalty = 1.0F;
    std::size_t repeat_last_n = 64;
    std::string reasoning_mode = "auto";
    std::int64_t reasoning_budget = -1;
    std::string speculative_backend;
    std::string mtp_model_path;
    std::size_t draft_tokens = 3;
    bool force_cpu = false;

    strix::cli::ArgParser llm_parser(
        "strix serve llm",
        "Start the OpenAI/Anthropic-compatible text LLM HTTP server.");
    llm_parser.AddOption(
        "-m", "--model", "PATH",
        "Path to GGUF model file (default: models/Qwen3.5-4B-BF16.gguf)",
        "Model", &model);
    llm_parser.AddOption("-c", "--context", "N",
                         "Maximum context tokens (default: 4096)", "Model",
                         &max_context);
    llm_parser.AddOption(
        "-n", "--max-tokens", "N",
        "Default maximum new tokens per response (default: 128)",
        "Sampling Defaults", &max_tokens);
    llm_parser.AddOption("-t", "--temperature", "T",
                         "Default sampling temperature (default: 0.0 = greedy)",
                         "Sampling Defaults", &temperature);
    llm_parser.AddOption(
        "", "--top-p", "P",
        "Default nucleus sampling probability cutoff (default: 1.0) "
        "(TODO: qwen)",
        "Sampling Defaults", &top_p);
    llm_parser.AddOption(
        "", "--top-k", "K",
        "Default Top-K token cutoff (default: 0 = disabled) (TODO: qwen)",
        "Sampling Defaults", &top_k);
    llm_parser.AddOption(
        "", "--min-p", "P",
        "Default Min-P threshold relative to top token (default: 0.0) (TODO: "
        "qwen)",
        "Sampling Defaults", &min_p);
    llm_parser.AddOption(
        "-s", "--seed", "N",
        "Default RNG seed for generation (default: -1 = random)",
        "Sampling Defaults", &seed);
    llm_parser.AddOption(
        "", "--repeat-penalty", "N",
        "Default repetition penalty multiplier (default: 1.0 = off) (TODO: "
        "qwen, deepseek)",
        "Sampling Defaults", &repeat_penalty);
    llm_parser.AddOption(
        "", "--repeat-last-n", "N",
        "Default lookback window for repetition penalty (default: 64) (TODO: "
        "qwen, deepseek)",
        "Sampling Defaults", &repeat_last_n);
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
                         "Draft backend: mtp, mtp-npu, npu, pld, or self",
                         "Speculative", &speculative_backend);
    llm_parser.AddOption("", "--mtp-model", "PATH",
                         "Path to quantized Qwen MTP draft head GGUF file",
                         "Speculative", &mtp_model_path);
    llm_parser.AddOption(
        "-d", "--draft-tokens", "N",
        "Maximum speculative draft tokens evaluated per step (default: 3)",
        "Speculative", &draft_tokens);
    llm_parser.AddFlag(
        "", "--cpu", "Force CPU OpenMP execution fallback instead of GPU ROCm",
        "Hardware", &force_cpu);

    if (!llm_parser.Parse(sub_args, &parse_err)) {
      std::cerr << "Error: " << parse_err << "\n";
      PrintServeHelp("strix", "llm");
      return 2;
    }
    if (llm_parser.IsHelpRequested()) {
      PrintServeHelp("strix", "llm");
      return 0;
    }

    std::string err;
    backend = std::make_shared<server::InferenceBackend>();
    if (!backend->load(model, &err, max_context, session_count)) {
      std::cerr << "Error loading model '" << model << "': " << err << "\n";
      return 1;
    }
  }

  server::HttpServer server(host, port, backend, video_jobs, tts);
  std::string err;
  if (!server.start(&err)) {
    std::cerr << "Error starting HTTP server: " << err << "\n";
    return 1;
  }
  server.run();
  return 0;
}

}  // namespace strix::cli
