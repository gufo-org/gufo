#include "src/cli/video/video.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/shutdown_signal.hpp"
#include "src/ext/video.hpp"

#ifndef GUFO_H3_SOURCE_MANIFEST
#define GUFO_H3_SOURCE_MANIFEST \
  "share/gufo/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
#endif
#ifndef GUFO_H3_INSTALLED_SOURCE_MANIFEST
#define GUFO_H3_INSTALLED_SOURCE_MANIFEST GUFO_H3_SOURCE_MANIFEST
#endif

namespace gufo::cli {

namespace {

volatile std::sig_atomic_t g_cancel_requested = 0;  // NOLINT

std::uint64_t NextReportNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

void HandleInterrupt(int) {
  g_cancel_requested = 1;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

template<typename Integer>
bool ParseInteger(std::string_view text, Integer* output) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *output);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

struct VideoOverrides {
  std::string preset = "exact";
  std::optional<std::string> selected_frames;
  std::string kernel = "row-parallel";
  std::optional<int> width, height, output_width, output_height;
  std::optional<int> frames, steps, blocks, reuse;
  bool no_audio = false;
  bool no_mux = false;
};

void RegisterVideoOptions(ArgParser& parser, VideoCliOptions& options,
                          VideoOverrides& overrides) {
  auto& request = options.request;
  parser.AddOption("-m", "--model", "DIR", "MiniMax H3 model directory",
                   "Model & Output", &request.model_root);
  parser.AddOption("-o", "--output", "MP4", "Destination MP4 path",
                   "Model & Output", &request.output_path);
  parser.AddOption(
      "", "--preset", "NAME",
      "exact, exact-1344x768, fast, aggressive, or dev (default: exact)",
      "Model & Output", &overrides.preset);
  parser.AddOption("", "--seed", "N", "Noise RNG seed (default: 42)",
                   "Denoising", &request.seed);
  const auto integer_override = [&](std::string_view name,
                                    std::string_view description,
                                    std::optional<int>* target) {
    parser.AddCustomOption(
        "", name, "N", description, "Denoising",
        [target](std::string_view flag, std::string_view value,
                 std::string* error) {
          int parsed = 0;
          if (!ParseInteger(value, &parsed)) {
            SetError(error, "Invalid integer for " + std::string(flag));
            return false;
          }
          *target = parsed;
          return true;
        });
  };
  integer_override("--width", "Override preset internal canvas width",
                   &overrides.width);
  integer_override("--height", "Override preset internal canvas height",
                   &overrides.height);
  integer_override("--output-width", "Override preset encoded width",
                   &overrides.output_width);
  integer_override("--output-height", "Override preset encoded height",
                   &overrides.output_height);
  integer_override("--frames", "Override preset frame count (22+17n)",
                   &overrides.frames);
  integer_override("--steps", "Override preset denoising evaluations",
                   &overrides.steps);
  integer_override("--blocks", "Override preset active DiT prefix blocks",
                   &overrides.blocks);
  integer_override("--reuse",
                   "Override preset denoiser reuse interval (1 = off)",
                   &overrides.reuse);
  parser.AddOption("", "--attention-kernel", "NAME",
                   "row-parallel or scalar (default: row-parallel)",
                   "Diagnostics", &overrides.kernel);
  parser.AddCustomOption(
      "", "--selected-frames", "LIST",
      "Frames to extract (first,middle,last or frame indices)", "Diagnostics",
      [&overrides](std::string_view, std::string_view value, std::string*) {
        overrides.selected_frames = value;
        return true;
      });
  parser.AddOption("", "--frames-dir", "DIR", "PPM frame dump directory",
                   "Diagnostics", &request.frames_directory);
  parser.AddOption("", "--latents-dir", "DIR",
                   "Final F32 latent dump directory", "Diagnostics",
                   &request.latents_directory);
  parser.AddFlag("", "--no-audio", "Skip AudioVAE (requires --no-mux)",
                 "Diagnostics", &overrides.no_audio);
  parser.AddFlag("", "--no-mux", "Skip MP4 composition (requires --frames-dir)",
                 "Diagnostics", &overrides.no_mux);
  parser.AddOption("", "--manifest", "PATH", "Pinned source manifest override",
                   "Diagnostics", &request.source_manifest);
  parser.AddOption("", "--parameters", "PATH", "Parameter report JSON path",
                   "Diagnostics", &options.parameters_path);
  parser.AddOption("", "--report", "PATH", "Timing and memory report JSON path",
                   "Diagnostics", &options.telemetry_path);
  parser.AddFlag("", "--profile", "Print phase progress and hardware metrics",
                 "General", &options.profile);
  parser.JoinPositionals(&request.prompt);
}

std::optional<std::vector<int>> ParseSelectedFrames(std::string_view text,
                                                    int frames,
                                                    std::string* error) {
  std::vector<int> result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    const std::string_view item = text.substr(begin, end - begin);
    int frame = -1;
    if (item == "first") {
      frame = 0;
    } else if (item == "middle") {
      frame = frames / 2;
    } else if (item == "last") {
      frame = frames - 1;
    } else if (!ParseInteger(item, &frame)) {
      SetError(error,
               "invalid MiniMax H3 selected frame: " + std::string(item));
      return std::nullopt;
    }
    result.push_back(frame);
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create H3 report directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial = path.string() + ".gufo-partial-" +
                                        std::to_string(getpid()) + "-" +
                                        std::to_string(NextReportNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output << contents;
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      SetError(error, "cannot write H3 report");
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    SetError(error, "cannot publish H3 report: " + filesystem_error.message());
    return false;
  }
  return true;
}

std::string PromptSha256(std::string_view prompt) {
  return crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(prompt.data()), prompt.size()));
}

std::optional<ext::VideoParameters> ResolveGenerationPreset(
    std::string_view preset, std::string* error = nullptr) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->resolve_preset) {
    return provider->resolve_preset(preset, error);
  }
  if (error)
    *error = "video provider 'minimax-h3' not available";
  return std::nullopt;
}

bool ValidateGenerationParameters(const ext::VideoParameters& parameters,
                                  std::string* error = nullptr) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->validate_parameters) {
    return provider->validate_parameters(parameters, error);
  }
  if (error)
    *error = "video provider 'minimax-h3' not available";
  return false;
}

std::string GenerationParametersJson(const ext::VideoParameters& parameters,
                                     std::uint64_t seed,
                                     std::string_view prompt_sha256) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->parameters_json) {
    return provider->parameters_json(parameters, seed, prompt_sha256);
  }
  return "{}";
}

std::string GenerationTelemetryJson(const ext::VideoTelemetry& telemetry) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->telemetry_json) {
    return provider->telemetry_json(telemetry);
  }
  return "{}";
}

struct CliProgress {
  std::atomic<bool>* cancellation{nullptr};
  bool profile{false};
};

void PrintProgress(std::string_view phase, int completed, int total,
                   void* opaque) {
  auto* state = static_cast<CliProgress*>(opaque);
  if (state == nullptr) {
    return;
  }
  if (g_cancel_requested != 0 && state->cancellation != nullptr) {
    state->cancellation->store(true, std::memory_order_relaxed);
  }
  if (state->profile) {
    std::cerr << "h3: " << phase << ' ' << completed << '/' << total << '\n';
  }
}

}  // namespace

std::filesystem::path DefaultH3SourceManifest() {
  std::filesystem::path build_path = GUFO_H3_SOURCE_MANIFEST;
  std::error_code error;
  if (std::filesystem::is_regular_file(build_path, error)) {
    return build_path;
  }
  return GUFO_H3_INSTALLED_SOURCE_MANIFEST;
}

void PrintVideoHelp(std::string_view program_name) {
  VideoCliOptions options;
  VideoOverrides overrides;
  ArgParser parser(
      std::string(program_name) + " video",
      "Generate MiniMax H3 text-to-video on a Strix Halo gfx1151 GPU.\n"
      "Supply a model directory and a positional text prompt. Canvas, frame,\n"
      "and denoising defaults come from the selected preset.");
  RegisterVideoOptions(parser, options, overrides);
  parser.PrintHelp();
}

std::optional<VideoCliOptions> ParseVideoOptions(
    std::span<const char* const> args, std::string* error) {
  VideoCliOptions options;
  options.request.seed = 42;
  options.request.source_manifest = DefaultH3SourceManifest();
  VideoOverrides overrides;
  ArgParser parser("gufo video");
  RegisterVideoOptions(parser, options, overrides);
  if (!parser.Parse(args, error) || parser.IsHelpRequested()) {
    return std::nullopt;
  }
  auto parameters = ResolveGenerationPreset(overrides.preset, error);
  if (!parameters)
    return std::nullopt;
  auto& p = *parameters;
  p.internal_width = overrides.width.value_or(p.internal_width);
  p.internal_height = overrides.height.value_or(p.internal_height);
  p.output_width = overrides.output_width.value_or(p.output_width);
  p.output_height = overrides.output_height.value_or(p.output_height);
  p.frames = overrides.frames.value_or(p.frames);
  p.evaluations = overrides.steps.value_or(p.evaluations);
  p.active_blocks = overrides.blocks.value_or(p.active_blocks);
  p.reuse_interval = overrides.reuse.value_or(p.reuse_interval);
  if (overrides.kernel != "row-parallel" && overrides.kernel != "scalar") {
    SetError(error, "--attention-kernel must be row-parallel or scalar");
    return std::nullopt;
  }
  p.row_parallel_attention = overrides.kernel == "row-parallel";
  if (overrides.no_audio)
    p.decode_audio = false;
  if (overrides.no_mux)
    p.mux = false;
  if (overrides.selected_frames) {
    auto frames =
        ParseSelectedFrames(*overrides.selected_frames, p.frames, error);
    if (!frames)
      return std::nullopt;
    p.selected_frames = std::move(*frames);
  }
  options.request.parameters = std::move(p);
  if (options.request.model_root.empty() || options.request.prompt.empty()) {
    SetError(error, "MiniMax H3 video requires --model and a prompt");
    return std::nullopt;
  }
  if (!options.request.parameters.mux &&
      options.request.frames_directory.empty()) {
    SetError(error, "--no-mux requires --frames-dir");
    return std::nullopt;
  }
  if (!options.request.parameters.decode_audio &&
      options.request.parameters.mux) {
    SetError(error, "--no-audio requires --no-mux");
    return std::nullopt;
  }
  if (options.request.parameters.mux && options.request.output_path.empty()) {
    SetError(error, "the selected H3 preset requires --output");
    return std::nullopt;
  }
  if (!ValidateGenerationParameters(options.request.parameters, error)) {
    return std::nullopt;
  }
  const std::filesystem::path report_base =
      options.request.parameters.mux ? options.request.output_path
                                     : options.request.frames_directory;
  if (options.parameters_path.empty()) {
    options.parameters_path = report_base.string() + ".parameters.json";
  }
  if (options.telemetry_path.empty()) {
    options.telemetry_path = report_base.string() + ".telemetry.json";
  }
  return options;
}

int RunVideo(std::span<const char* const> args) {
  std::string error;
  auto options = ParseVideoOptions(args, &error);
  if (!options.has_value()) {
    if (error.empty()) {
      PrintVideoHelp("gufo");
      return 0;
    }
    std::cerr << "Error: " << error << '\n';
    PrintVideoHelp("gufo");
    return 2;
  }
  if (!WriteAtomic(options->parameters_path,
                   GenerationParametersJson(
                       options->request.parameters, options->request.seed,
                       PromptSha256(options->request.prompt)),
                   &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }

  g_cancel_requested = 0;
  std::optional<platform::ShutdownSignalGuard> interrupt_guard;
  try {
    interrupt_guard.emplace(HandleInterrupt);
  } catch (const std::system_error&) {
  }
  std::atomic<bool> cancellation{false};
  std::jthread interrupt_watcher([&cancellation](const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      if (g_cancel_requested != 0) {
        cancellation.store(true, std::memory_order_relaxed);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });
  CliProgress progress{&cancellation, options->profile};
  ext::VideoTelemetry telemetry;
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (!provider || !provider->create) {
    std::cerr << "Error: MiniMax H3 video provider not registered\n";
    return 1;
  }
  auto runner = provider->create(options->request.model_root.string(), &error);
  if (!runner) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  const bool generated = runner->Generate(
      options->request,
      [&] { return cancellation.load(std::memory_order_relaxed); },
      PrintProgress, &progress, &telemetry, &error);
  interrupt_watcher.request_stop();
  interrupt_watcher.join();
  interrupt_guard.reset();
  if (!generated) {
    std::cerr << "Error: " << error << '\n';
    return cancellation.load(std::memory_order_relaxed) ? 130 : 1;
  }
  const std::string report = GenerationTelemetryJson(telemetry);
  if (!WriteAtomic(options->telemetry_path, report, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 1;
  }
  if (options->profile) {
    std::cerr << report;
  }
  return 0;
}

}  // namespace gufo::cli
