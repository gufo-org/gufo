#include "src/cli/music/music.hpp"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "gufo/ext/music.hpp"
#include "gufo/ext/registry.hpp"
#include "src/cli/arg_parser.hpp"
#include "src/cli/provider_options.hpp"
#include "src/cli/serve/music_wav.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
namespace {

volatile std::sig_atomic_t g_music_cancel = 0;  // NOLINT
void HandleMusicInterrupt(int) {
  g_music_cancel = 1;
}

struct MusicOptions {
  std::string model;
  std::string prompt;
  std::string lyrics;
  std::string lyrics_file;
  std::string score_file;
  std::string planning;
  std::string output;
  std::string score_output;
  std::string report;
  double seconds{0.0};
  double guidance{-1.0};
  int steps{0};
  std::uint64_t seed{831001};
  bool plan_only{false};
  bool float32{false};
};

bool ReadText(const std::string& path, std::string* text) {
  std::ifstream input(std::filesystem::path(path), std::ios::binary);
  if (!input)
    return false;
  text->assign(std::istreambuf_iterator<char>(input),
               std::istreambuf_iterator<char>());
  return true;
}

bool WriteBytes(const std::string& path, const std::string& bytes) {
  std::ofstream output(std::filesystem::path(path),
                       std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(output);
}

ArgParser MakeParser(MusicOptions* o) {
  ArgParser parser("gufo music [OPTIONS]",
                   "Generate a song with a music model (YuE2).");
  parser.AddOption("-m", "--model", "DIR", "Music model directory", "Model",
                   &o->model);
  parser.AddOption("-p", "--prompt", "TEXT",
                   "Style: genre, instruments, voice, language, tempo", "Input",
                   &o->prompt);
  parser.AddOption("", "--lyrics", "TEXT",
                   "Lyrics with [Verse]/[Chorus] tags; empty is instrumental",
                   "Input", &o->lyrics);
  parser.AddOption("", "--lyrics-file", "PATH", "Read the lyrics from a file",
                   "Input", &o->lyrics_file);
  parser.AddOption("", "--score", "PATH",
                   "ABC score to play instead of planning one", "Input",
                   &o->score_file);
  parser.AddOption("", "--planning", "MODE",
                   "Score planning: full, melody or off (default: full)",
                   "Generation", &o->planning);
  parser.AddOption("", "--seconds", "S",
                   "Longest song in seconds (default: the model's choice)",
                   "Generation", &o->seconds);
  parser.AddOption("", "--guidance", "G",
                   "Classifier-free guidance scale (default: the model's)",
                   "Generation", &o->guidance);
  parser.AddOption("", "--steps", "N",
                   "Acoustic solver steps (default: the model's)", "Generation",
                   &o->steps);
  parser.AddOption("", "--seed", "N", "Seed (default: 831001)", "Generation",
                   &o->seed);
  parser.AddFlag("", "--plan-only", "Write the score and stop", "Generation",
                 &o->plan_only);
  parser.AddOption("-o", "--output", "WAV", "Output WAV path", "Output",
                   &o->output);
  parser.AddFlag("", "--float32", "Write IEEE float WAV instead of PCM16",
                 "Output", &o->float32);
  parser.AddOption("", "--score-output", "PATH",
                   "Write the score the song played", "Output",
                   &o->score_output);
  parser.AddOption("", "--report", "PATH", "Write timings as JSON", "Output",
                   &o->report);
  return parser;
}

}  // namespace

int RunMusic(std::span<const char* const> args) {
  MusicOptions o;
  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  auto parser = MakeParser(&o);
  ProviderOptions declared(registry, std::string(ext::kMusicCommand),
                           ext::Capability::kMusicGeneration);
  declared.Register(parser);
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  if (o.model.empty() || (o.output.empty() && !o.plan_only) ||
      (o.plan_only && o.score_output.empty())) {
    std::cerr << "Error: music needs --model and --output (--score-output "
                 "with --plan-only)\n";
    return 2;
  }
  ext::MusicRequest request;
  request.prompt = o.prompt;
  request.lyrics = o.lyrics;
  if (!o.lyrics_file.empty() && !ReadText(o.lyrics_file, &request.lyrics)) {
    std::cerr << "Error: cannot read " << o.lyrics_file << '\n';
    return 2;
  }
  if (!o.score_file.empty()) {
    if (!ReadText(o.score_file, &request.score)) {
      std::cerr << "Error: cannot read " << o.score_file << '\n';
      return 2;
    }
    request.score_format = "abc";
  }
  if (o.seconds > 0.0)
    request.duration_seconds = o.seconds;
  if (o.guidance >= 0.0)
    request.guidance = static_cast<float>(o.guidance);
  if (o.steps > 0)
    request.steps = o.steps;
  request.seed = o.seed;
  request.plan_only = o.plan_only;
  if (!o.planning.empty()) {
    json::Value options = json::Value::object();
    options["planning"] = o.planning;
    request.provider_json = options.dump();
  }

  const auto* provider = ext::ProbeMusicProvider(o.model);
  if (provider == nullptr) {
    std::cerr << "Error: no music model claims " << o.model << '\n';
    return 2;
  }
  if (provider->create_with_options == nullptr) {
    std::cerr << "Error: " << provider->id
              << " is not available in this build\n";
    return 1;
  }
  ext::MusicRunnerOptions runner_options;
  runner_options.model_root = o.model;
  try {
    declared.Parse();
    runner_options.options = declared.Select(provider->id);
  } catch (const std::invalid_argument& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 2;
  }
  auto runner = provider->create_with_options(runner_options, &error);
  if (!runner) {
    std::cerr << "Error loading " << provider->id << ": " << error << '\n';
    return 1;
  }
  std::string field;
  if (!ext::ValidateMusicRequest(runner->Capabilities(), request, &field,
                                 &error) ||
      !runner->ValidateRequest(request, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  g_music_cancel = 0;
  const auto previous = std::signal(SIGINT, HandleMusicInterrupt);
  std::string last_phase;
  ext::MusicResult result;
  const bool ok = runner->Generate(
      request, [] { return g_music_cancel != 0; },
      [&](std::string_view phase, std::uint64_t completed,
          std::uint64_t total) {
        if (phase != last_phase || completed % 100 == 0 || completed == total) {
          std::cerr << "\r" << phase << ' ' << completed;
          if (total != 0)
            std::cerr << '/' << total;
          std::cerr << "        " << std::flush;
          if (phase != last_phase && !last_phase.empty())
            std::cerr << '\n';
          last_phase = std::string(phase);
        }
        return g_music_cancel == 0;
      },
      &result, &error);
  (void)std::signal(SIGINT, previous);
  std::cerr << '\n';
  if (!ok) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  if (!o.score_output.empty() && !WriteBytes(o.score_output, result.score)) {
    std::cerr << "Error: cannot write " << o.score_output << '\n';
    return 1;
  }
  if (!o.plan_only) {
    const std::string wav = server::EncodeMusicWav(result.audio, o.float32);
    if (wav.empty() || !WriteBytes(o.output, wav)) {
      std::cerr << "Error: cannot write " << o.output << '\n';
      return 1;
    }
  }
  if (!o.report.empty() && !WriteBytes(o.report, result.telemetry_json)) {
    std::cerr << "Error: cannot write " << o.report << '\n';
    return 1;
  }
  const double seconds =
      result.audio.channels == 0
          ? 0.0
          : static_cast<double>(result.audio.samples.size()) /
                result.audio.channels / result.audio.sample_rate;
  std::cout << "music: " << seconds << " s of audio in "
            << result.total_ms / 1000.0 << " s"
            << (result.finish == ext::MusicFinish::kTruncated ? " (truncated)"
                                                              : "")
            << '\n'
            << result.telemetry_json << '\n';
  return 0;
}

}  // namespace gufo::cli
