#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/core/platform/os_info.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace {

std::string dispatched;
gufo::sampling::SamplingConfig dispatched_sampling;
bool invalid_defaults = false;

std::optional<gufo::sampling::TextSamplingDefaults> SamplingDefaults(
    const gufo::core::GgufReader&) {
  gufo::sampling::TextSamplingDefaults defaults;
  defaults.thinking_unspecified = {.temperature = 0.75F, .top_k = 7};
  defaults.thinking_enabled = {.temperature = 0.25F, .top_k = 5};
  defaults.thinking_disabled = {.temperature = 0.5F, .top_k = 6};
  if (invalid_defaults) {
    defaults.thinking_unspecified.temperature = -1.0F;
  }
  return defaults;
}

int Dispatch(std::string_view command, const gufo::core::GgufReader& reader) {
  if (reader.GetMetadataString("general.architecture") != "test-text") {
    return 1;
  }
  dispatched = command;
  return 42;
}

}  // namespace

// Stand in for the generated registry. No provider exists until the CLI calls
// this function, so each command is tested in a separate process.
void gufo::ext::RegisterBuiltExtensions(gufo::ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "test-text",
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return reader.GetMetadataString("general.architecture") ==
                   "test-text";
          },
      .bench_driver =
          +[](const TextBenchOptions&,
              const std::shared_ptr<const core::GgufReader>& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("bench", *reader);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("prompt", reader);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("chat", reader);
          },
      .sampling_defaults = &SamplingDefaults,
  });
}

namespace {

int Run(std::string_view command, std::span<const char* const> args) {
  dispatched.clear();
  return command == "prompt" ? gufo::cli::RunPrompt(args)
                             : gufo::cli::RunChat(args);
}

// The real command must select the provider's defaults for its thinking mode
// and keep explicit values, as #935 once dropped this step.
bool CheckSamplingDefaults(std::string_view command, const char* model) {
  struct Case {
    std::vector<const char*> flags;
    float temperature;
    int top_k;
  };
  const Case cases[] = {
      {{}, 0.75F, 7},
      {{"--think", "auto"}, 0.75F, 7},
      {{"--think", "on"}, 0.25F, 5},
      {{"--think", "off"}, 0.5F, 6},
      {{"--reasoning-effort", "high"}, 0.25F, 5},
      {{"--think", "off", "--temperature", "0"}, 0.0F, 6},
      {{"--think", "on", "--top-k", "0"}, 0.25F, 0},
  };
  for (const auto& entry : cases) {
    std::vector<const char*> args = {"--model", model};
    args.insert(args.end(), entry.flags.begin(), entry.flags.end());
    if (Run(command, args) != 42 ||
        dispatched_sampling.temperature != entry.temperature ||
        dispatched_sampling.top_k != entry.top_k) {
      return false;
    }
  }
  invalid_defaults = true;
  const std::array<const char*, 2> args = {"--model", model};
  const int status = Run(command, args);
  const bool rejected = status == 1 && dispatched.empty();
  invalid_defaults = false;
  return rejected;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return 2;
  }
  const std::string_view command = argv[1];
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-text-dispatch-" +
                     std::to_string(gufo::platform::ProcessId()) + ".gguf");
  gufo::test_fixture::GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "test-text");
  const auto bytes = builder.Build(0);
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!file) {
      return 1;
    }
  }
  const auto model = path.string();
  const std::array<const char*, 2> args = {"--model", model.c_str()};
  int status = 1;
  if (command == "prompt") {
    status = gufo::cli::RunPrompt(args);
  } else if (command == "chat") {
    status = gufo::cli::RunChat(args);
  } else if (command == "bench") {
    status = gufo::cli::RunBench(args);
  }
  const bool dispatched_command = status == 42 && dispatched == command;
  const bool sampling_ok =
      command == "bench" || CheckSamplingDefaults(command, model.c_str());
  std::filesystem::remove(path);
  if (!dispatched_command) {
    std::cerr << "FAIL: " << command
              << " must register and dispatch to its built text provider\n";
    return 1;
  }
  if (!sampling_ok) {
    std::cerr << "FAIL: " << command
              << " must resolve its provider's sampling defaults\n";
    return 1;
  }
  return 0;
}
