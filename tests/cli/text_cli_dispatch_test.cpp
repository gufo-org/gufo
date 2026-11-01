#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/core/platform/os_info.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace {

std::string dispatched;

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
          +[](const TextPromptOptions&, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("prompt", reader);
          },
      .chat_driver =
          +[](const TextPromptOptions&, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("chat", reader);
          },
  });
}

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
  std::filesystem::remove(path);
  if (status != 42 || dispatched != command) {
    std::cerr << "FAIL: " << command
              << " must register and dispatch to its built text provider\n";
    return 1;
  }
  return 0;
}
