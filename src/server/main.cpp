#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "src/cli/diagnose.h"
#include "src/server/bench_cli.hpp"
#include "src/server/http_server.hpp"
#include "src/server/inference_backend.hpp"
#include "src/server/prompt_cli.hpp"
#include "src/server/tts_service.hpp"
#include "src/server/video_cli.hpp"
#include "src/server/video_jobs.hpp"

namespace strix::server {
namespace {

constexpr std::string_view version_string = "strix-server 0.1.0";

void print_version() {
  std::cout << version_string << '\n';
}

void print_help(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS] [COMMAND]\n\n"
      << "Commands:\n"
      << "  serve     Start the OpenAI-compatible server\n"
      << "  chat      Maintain an interactive conversation\n"
      << "  prompt    Execute one request and exit\n"
      << "  video     Generate MiniMax H3 text-to-video on Strix Halo\n"
      << "  bench     Run inference throughput benchmarks\n"
      << "  diagnose  Run non-interactive system and hardware diagnostics\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n";
}

void PrintServeHelp() {
  std::cout
      << "Usage: strix-server serve [OPTIONS]\n\n"
      << "Start the OpenAI/Anthropic/llama/sdapi-compatible HTTP server.\n\n"
      << "Options:\n"
      << "  -i, --host <IP>    Bind address (default: 127.0.0.1)\n"
      << "  -p, --port <N>     Port to listen on (default: 8080)\n"
      << "  -m, --model <PATH> Path to GGUF model file "
         "(default: models/Qwen3.5-4B-BF16.gguf)\n"
      << "  -c, --context <N>  Maximum context tokens (default: 4096)\n"
      << "  -j, --sessions <N> Preallocated GPU request sessions (default: 1)\n"
      << "  --video-model <DIR> Operator-supplied MiniMax H3 directory\n"
      << "  --video-root <DIR>  Persistent video jobs (default: video-jobs)\n"
      << "  --video-manifest <PATH> Pinned H3 manifest override\n"
      << "  --video-ttl <SEC>   Completed-artifact TTL (default: 3600)\n"
      << "  --tts-model <DIR>   Qwen3-TTS 12Hz 1.7B model directory\n"
      << "  --tts-context <N>   Native prompt+generation capacity "
         "(default: 4096)\n"
      << "  --speculative, --speculative-decoding <MODE>\n"
      << "                      Draft backend: dflash, dflash2, mtp, "
         "mtp-npu, npu, pld, self, or off\n"
      << "  --dflash-model <PATH> Quantized Qwen DFlash/DFlash-2 GGUF\n"
      << "  --mtp-model <PATH>    Quantized Qwen MTP GGUF for mtp modes\n"
      << "  -h, --help         Print this help\n";
}

int RunServe(std::span<const char* const> args) {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string model = "models/Qwen3.5-4B-BF16.gguf";
  bool text_model_explicit = false;
  std::uint32_t max_context = 4096;
  std::size_t session_count = 1;
  std::string speculative_backend;
  std::string dflash_model;
  std::string mtp_model;
  std::filesystem::path video_model;
  std::filesystem::path video_root = "video-jobs";
  std::filesystem::path video_manifest = DefaultH3SourceManifest();
  std::uint64_t video_ttl_seconds = 3600;
  std::filesystem::path tts_model;
  std::size_t tts_context_tokens = 4096;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "-h" || a == "--help") {
      PrintServeHelp();
      return 0;
    }
    if ((a == "-i" || a == "--host") && i + 1 < args.size()) {
      host = args[i + 1];
      ++i;
    } else if ((a == "-p" || a == "--port") && i + 1 < args.size()) {
      port = std::stoi(args[i + 1]);
      ++i;
    } else if ((a == "-m" || a == "--model") && i + 1 < args.size()) {
      model = args[i + 1];
      text_model_explicit = true;
      ++i;
    } else if ((a == "-c" || a == "--context") && i + 1 < args.size()) {
      max_context =
          static_cast<std::uint32_t>(std::stoul(std::string(args[i + 1])));
      ++i;
    } else if ((a == "-j" || a == "--sessions") && i + 1 < args.size()) {
      session_count = std::stoul(std::string(args[i + 1]));
      ++i;
    } else if ((a == "--speculative" || a == "--speculative-decoding") &&
               i + 1 < args.size()) {
      speculative_backend = args[++i];
    } else if (a == "--dflash-model" && i + 1 < args.size()) {
      dflash_model = args[++i];
    } else if (a == "--mtp-model" && i + 1 < args.size()) {
      mtp_model = args[++i];
    } else if (a == "--video-model" && i + 1 < args.size()) {
      video_model = args[++i];
    } else if (a == "--video-root" && i + 1 < args.size()) {
      video_root = args[++i];
    } else if (a == "--video-manifest" && i + 1 < args.size()) {
      video_manifest = args[++i];
    } else if (a == "--video-ttl" && i + 1 < args.size()) {
      video_ttl_seconds = std::stoull(std::string(args[++i]));
    } else if (a == "--tts-model" && i + 1 < args.size()) {
      tts_model = args[++i];
    } else if (a == "--tts-context" && i + 1 < args.size()) {
      tts_context_tokens = std::stoul(std::string(args[++i]));
    } else {
      std::cerr << "Error: unknown or incomplete serve option '" << a << "'\n";
      PrintServeHelp();
      return 2;
    }
  }

  if (video_ttl_seconds == 0 ||
      video_ttl_seconds >
          static_cast<std::uint64_t>(std::chrono::seconds::max().count())) {
    std::cerr << "Error: --video-ttl must be a positive duration\n";
    return 2;
  }

  std::shared_ptr<InferenceBackend> backend;
  std::string err;
  if ((video_model.empty() && tts_model.empty()) || text_model_explicit) {
    backend = std::make_shared<InferenceBackend>();
    if (!backend->load(model, &err, max_context, session_count)) {
      std::cerr << "Error loading model '" << model << "': " << err << "\n";
      return 1;
    }
  }

  std::shared_ptr<VideoJobService> video_jobs;
  if (!video_model.empty()) {
    video_jobs = std::make_shared<VideoJobService>(VideoJobServiceOptions{
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
  }

  std::shared_ptr<TtsService> tts;
  if (!tts_model.empty()) {
    tts = std::make_shared<TtsService>(TtsServiceOptions{
        .model_root = tts_model,
        .native_context_tokens = tts_context_tokens,
        .validate_model = true,
        .model_id = {},
        .voices = {},
        .runner = {},
    });
    if (!tts->ready()) {
      std::cerr << "Error enabling Qwen3-TTS audio service: "
                << tts->initialization_error() << '\n';
      return 1;
    }
  }

  HttpServer server(host, port, backend, video_jobs, tts);
  if (!server.start(&err)) {
    std::cerr << "Error starting HTTP server: " << err << "\n";
    return 1;
  }
  server.run();
  return 0;
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    print_help("strix-server");
    return 2;
  }

  const std::string_view program_name = args[0];
  const auto options = args.subspan(1);

  if (options.empty()) {
    std::cerr << "Error: no command or option provided\n";
    print_help(program_name);
    return 2;
  }

  const std::string_view first_arg = options[0];
  if (first_arg == "--version" || first_arg == "-v") {
    print_version();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    print_help(program_name);
    return 0;
  }

  if (first_arg == "diagnose") {
    return strix::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return RunBench(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return RunVideo(options.subspan(1));
  }

  if (first_arg == "chat") {
    return RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return RunServe(options.subspan(1));
  }

  std::cerr << "Error: unknown command or option '" << first_arg << "'\n";
  print_help(program_name);
  return 2;
}

}  // namespace
}  // namespace strix::server

int main(int argc, char* argv[]) {
  return strix::server::run(
      std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
}
