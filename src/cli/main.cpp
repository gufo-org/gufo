#include <array>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/bench/bench.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/video/video.hpp"

constexpr std::string_view kStrixVersion = "0.1.0";

namespace {

void print_version() {
  std::cout << "strix version " << kStrixVersion << "\n";
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
      << "  video          Generate MiniMax H3 text-to-video\n"
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
    print_help("strix");
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
        return strix::cli::RunServe(serve_help_flags);
      }
      return strix::cli::RunServe(help_flag);
    }
    if (sub == "prompt") {
      return strix::cli::RunPrompt(help_flag);
    }
    if (sub == "bench") {
      return strix::cli::RunBench(help_flag);
    }
    if (sub == "video") {
      return strix::cli::RunVideo(help_flag);
    }
    if (sub == "chat") {
      return strix::cli::RunChat(help_flag);
    }
    if (sub == "diagnose" || sub == "probe" || sub == "info") {
      return strix::cli::RunDiagnose(help_flag);
    }
    std::cerr << "Error: unknown help topic '" << sub << "'\n";
    print_help(program_name);
    return 2;
  }

  if (first_arg == "probe") {
    return strix::cli::RunProbe(options.subspan(1));
  }

  if (first_arg == "diagnose" || first_arg == "info") {
    return strix::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return strix::cli::RunBench(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return strix::cli::RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return strix::cli::RunVideo(options.subspan(1));
  }

  if (first_arg == "chat") {
    return strix::cli::RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return strix::cli::RunServe(options.subspan(1));
  }

  // If invoked as legacy binary name or flag passed directly, route to serve
  if (program_name.ends_with("strix-server") ||
      program_name.ends_with("/strix-server")) {
    return strix::cli::RunServe(options);
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
