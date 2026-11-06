#include "src/cli/video/video.hpp"

#include <array>
#include <cstdlib>
#include <gufo/ext/registry.hpp>
#include <iostream>
#include <string>

#include "src/ext/video.hpp"

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

void TestPresets() {
  const std::array<const char*, 7> fast = {"--model", "/models/h3", "--preset",
                                           "fast",    "--output",   "out.mp4",
                                           "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(fast, &error);
  Expect(options.has_value(), "fast preset parses");
  Expect(options->request.parameters.internal_width == 384,
         "fast internal width");
  Expect(options->request.parameters.output_width == 512, "fast output width");
  Expect(options->request.parameters.evaluations == 19,
         "fast denoiser evaluations");
  Expect(options->request.parameters.active_blocks == 45, "fast blocks");
  Expect(options->request.parameters.reuse_interval == 2, "fast reuse");
  const auto* provider = gufo::ext::FindVideoProvider("minimax-h3");
  Expect(provider != nullptr && provider->parameters_json != nullptr,
         "minimax-h3 provider available");
  const std::string parameters = provider->parameters_json(
      options->request.parameters, 42,
      "df0ff96bcdb3a350f115cc39daaf0d7523814258890aa90c06fa784e3ad06169");
  Expect(parameters.find("gufo.minimax-h3-text-generation.v1") !=
             std::string::npos,
         "versioned parameter schema");
  Expect(parameters.find("\"token_reduction\": false") != std::string::npos,
         "token reduction remains disabled");

  const std::array<const char*, 7> aggressive = {
      "--model",  "/models/h3", "--preset", "aggressive",
      "--output", "out.mp4",    "A fox"};
  options = gufo::cli::ParseVideoOptions(aggressive, &error);
  Expect(options.has_value(), "aggressive preset parses");
  Expect(options->request.parameters.internal_width == 320,
         "aggressive internal width");
  Expect(options->request.parameters.active_blocks == 40, "aggressive blocks");
  Expect(options->request.parameters.reuse_interval == 3, "aggressive reuse");

  const std::array<const char*, 7> fullres = {
      "--model",  "/models/h3", "--preset",    "exact-1344x768",
      "--output", "out.mp4",    "A goalkeeper"};
  options = gufo::cli::ParseVideoOptions(fullres, &error);
  Expect(options.has_value(), "full-resolution exact preset parses");
  Expect(options->request.parameters.internal_width == 1344 &&
             options->request.parameters.internal_height == 768 &&
             options->request.parameters.frames == 124 &&
             options->request.parameters.evaluations == 49,
         "full-resolution CLI contract");
}

void TestDevelopmentAndOverrides() {
  const std::array<const char*, 9> args = {
      "--model", "/models/h3", "--preset", "dev",  "--frames-dir",
      "frames",  "--seed",     "123",      "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(args, &error);
  Expect(options.has_value(), "development preset parses");
  Expect(!options->request.parameters.mux, "development skips mux");
  Expect(!options->request.parameters.decode_audio, "development skips audio");
  Expect(options->request.parameters.selected_frames ==
             std::vector<int>({0, 11, 21}),
         "development selected frames");
  Expect(options->request.seed == 123, "seed override");

  const std::array<const char*, 19> custom = {"--model",
                                              "/models/h3",
                                              "--preset",
                                              "dev",
                                              "--frames-dir",
                                              "frames",
                                              "--latents-dir",
                                              "latents",
                                              "--width",
                                              "320",
                                              "--height",
                                              "320",
                                              "--steps",
                                              "7",
                                              "--blocks",
                                              "48",
                                              "--attention-kernel",
                                              "scalar",
                                              "A fox"};
  options = gufo::cli::ParseVideoOptions(custom, &error);
  Expect(options.has_value(), "custom controls parse");
  Expect(options->request.parameters.internal_width == 320, "width override");
  Expect(options->request.parameters.evaluations == 7, "step override");
  Expect(options->request.parameters.active_blocks == 48, "block override");
  Expect(options->request.latents_directory == "latents",
         "latent diagnostic output override");
  Expect(!options->request.parameters.row_parallel_attention,
         "scalar attention fallback override");
}

void TestSelectedFramesAndFailures() {
  const std::array<const char*, 9> selected = {
      "--model",      "/models/h3", "--preset",          "dev",
      "--frames-dir", "frames",     "--selected-frames", "last,first,middle",
      "A fox"};
  std::string error;
  auto options = gufo::cli::ParseVideoOptions(selected, &error);
  Expect(options.has_value(), "symbolic selected frames parse");
  Expect(options->request.parameters.selected_frames ==
             std::vector<int>({0, 11, 21}),
         "selected frames normalize");

  const std::array<const char*, 5> unsupported = {
      "--model", "/models/h3", "--first-frame", "input.png", "A fox"};
  Expect(!gufo::cli::ParseVideoOptions(unsupported, &error).has_value(),
         "first-frame input rejected");
  Expect(error.find("Unknown option") != std::string::npos,
         "unsupported input is an unknown option");

  const std::array<const char*, 4> no_audio = {"--model", "/models/h3",
                                               "--no-audio", "A fox"};
  Expect(!gufo::cli::ParseVideoOptions(no_audio, &error).has_value(),
         "audio cannot be disabled while muxing");
}

void TestSharedOptionParsing() {
  std::string error;
  const auto options = gufo::cli::ParseVideoOptions(
      std::array{"--model", "--preset", "--steps=7", "--preset=dev",
                 "--frames-dir=frames", "--profile", "A fox"},
      &error);
  Expect(options.has_value(), "inline options and flag-like values parse");
  Expect(options->request.model_root == "--preset",
         "option values are not scanned as flags");
  Expect(options->request.parameters.evaluations == 7,
         "explicit overrides win regardless of preset argument order");
  Expect(options->profile, "documented profile option is wired");

  const auto help_value = gufo::cli::ParseVideoOptions(
      std::array{"--model", "--help", "--preset=dev", "--frames-dir=frames",
                 "A fox"},
      &error);
  Expect(help_value && help_value->request.model_root == "--help",
         "help used as an option value does not trigger help");
  Expect(
      !gufo::cli::ParseVideoOptions(std::array{"--selected-frames="}, &error) &&
          !error.empty(),
      "explicit empty frame selection is rejected");
  Expect(!gufo::cli::ParseVideoOptions(std::array{"--steps=7junk"}, &error) &&
             !error.empty(),
         "partial integers are rejected");
  Expect(!gufo::cli::ParseVideoOptions(std::array{"--help"}, &error) &&
             error.empty(),
         "actual help request succeeds without requiring a model");
}

}  // namespace

void minimax_h3_register(gufo::ext::Registry&);

int main() {
  gufo::ext::Registry registry;
  minimax_h3_register(registry);
  TestPresets();
  TestDevelopmentAndOverrides();
  TestSelectedFramesAndFailures();
  TestSharedOptionParsing();
  std::cout << "All MiniMax H3 video CLI tests passed.\n";
  return 0;
}
