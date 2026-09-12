#include <array>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/bench/bench.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/eval/eval.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/transcribe/transcribe.hpp"
#include "src/cli/video/video.hpp"

#ifndef GUFO_VERSION
#define GUFO_VERSION "development"
#endif

constexpr std::string_view kGufoVersion = GUFO_VERSION;

namespace {

void print_version() {
  std::cout << "gufo version " << kGufoVersion << "\n";
}

void print_help(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " <COMMAND> [OPTIONS]\n\n"
      << "Fast LLM & DiT video inference engine for AMD Strix Halo "
         "(gfx1151).\n\n"
      << "Commands:\n"
      << "  serve          Start the OpenAI-compatible HTTP server\n"
      << "  prompt         Execute one prompt request and exit\n"
      << "  chat           Start an interactive terminal conversation\n"
      << "  bench          Benchmark prompt processing and token generation\n"
      << "  eval           Evaluate an OpenAI-compatible text server\n"
      << "  video          Generate MiniMax H3 text-to-video\n"
      << "  transcribe     Transcribe WAV audio with Qwen3-ASR-1.7B\n"
      << "  diagnose       Inspect system hardware, memory, and drivers\n"
      << "  help           Print help for a specific command\n\n"
      << "Run '" << program_name << " help <COMMAND>' or '" << program_name
      << " <COMMAND> --help' for details on a command\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n";
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    print_help("gufo");
    return 2;
  }

  const std::string_view program_name = args[0];
  const auto options = args.subspan(1);

  if (options.empty()) {
    print_help(program_name);
    return 0;
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

  if (first_arg == "help") {
    if (options.size() <= 1) {
      print_help(program_name);
      return 0;
    }
    const std::string_view sub = options[1];
    const std::array<const char*, 1> help_flag = {"--help"};
    if (sub == "serve") {
      if (options.size() > 2) {
        const std::array<const char*, 2> serve_help_flags = {options[2],
                                                             "--help"};
        return gufo::cli::RunServe(serve_help_flags);
      }
      return gufo::cli::RunServe(help_flag);
    }
    if (sub == "prompt") {
      return gufo::cli::RunPrompt(help_flag);
    }
    if (sub == "bench") {
      return gufo::cli::RunBench(help_flag);
    }
    if (sub == "eval") {
      return gufo::cli::RunEval(help_flag);
    }
    if (sub == "video") {
      return gufo::cli::RunVideo(help_flag);
    }
    if (sub == "transcribe" || sub == "asr") {
      return gufo::cli::RunTranscribe(help_flag);
    }
    if (sub == "chat") {
      return gufo::cli::RunChat(help_flag);
    }
    if (sub == "probe") {
      return gufo::cli::RunProbe(help_flag);
    }
    if (sub == "diagnose" || sub == "info") {
      return gufo::cli::RunDiagnose(help_flag);
    }
    std::cerr << "Error: unknown help topic '" << sub << "'\n";
    print_help(program_name);
    return 2;
  }

  if (first_arg == "probe") {
    return gufo::cli::RunProbe(options.subspan(1));
  }

  if (first_arg == "diagnose" || first_arg == "info") {
    return gufo::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return gufo::cli::RunBench(options.subspan(1));
  }

  if (first_arg == "eval") {
    return gufo::cli::RunEval(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return gufo::cli::RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return gufo::cli::RunVideo(options.subspan(1));
  }

  if (first_arg == "transcribe" || first_arg == "asr") {
    return gufo::cli::RunTranscribe(options.subspan(1));
  }

  if (first_arg == "chat") {
    return gufo::cli::RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return gufo::cli::RunServe(options.subspan(1));
  }

  // If invoked as legacy binary name or flag passed directly, route to serve
  if (program_name.ends_with("gufo-server") ||
      program_name.ends_with("/gufo-server")) {
    return gufo::cli::RunServe(options);
  }

  std::cerr << "Error: unknown command '" << first_arg << "'\n";
  print_help(program_name);
  return 2;
}

}  // namespace

int main(int argc, char* argv[]) {
  const std::span<const char* const> args(argv, static_cast<std::size_t>(argc));
  return run(args);
}
