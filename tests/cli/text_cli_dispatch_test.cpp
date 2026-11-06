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
#include <sstream>
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
gufo::ext::ParsedOptions dispatched_options;
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

int Dispatch(std::string_view command, const gufo::core::GgufReader& reader,
             const gufo::ext::ParsedOptions& options) {
  const std::string_view architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "test-text" && architecture != "other-text") {
    return 1;
  }
  dispatched = std::string(command) + ":" + std::string(architecture);
  dispatched_options = options;
  return 42;
}

bool ProbeArchitecture(const gufo::core::GgufReader& reader,
                       std::string_view architecture) {
  return reader.GetMetadataString("general.architecture") == architecture;
}

}  // namespace

// Stand in for the generated registry. No provider exists until the CLI calls
// this function, so each command is tested in a separate process.
void gufo::ext::RegisterBuiltExtensions(gufo::ext::Registry& registry) {
  const std::vector<std::string> all = {"prompt", "chat", "bench"};
  registry.BeginExtension({.name = "fixture", .version = "1", .source = ""});
  // Provider options of each type, one restricted to prompt and chat, and a
  // core-owned spelling (--draft-tokens) that stays with core.
  registry.AddProvider({
      .id = "test-text",
      .keys = {"gguf:test-text"},
      .options =
          {
              {"--fixture-flag", OptionType::kBoolean, all, "A switch"},
              {"--fixture-count", OptionType::kInteger, all, ""},
              {"--fixture-scale", OptionType::kNumber, {"prompt", "chat"}, ""},
              {"--fixture-file", OptionType::kPath, all, ""},
              {"--draft-tokens", OptionType::kInteger, all, "Not delivered"},
          },
  });
  registry.AddProvider({
      .id = "other-text",
      .keys = {"gguf:other-text"},
      .options =
          {
              {"--other-only", OptionType::kString, all, ""},
              {"--fixture-count", OptionType::kInteger, all, ""},
          },
  });
  registry.AddTextProvider({
      .id = "test-text",
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return ProbeArchitecture(reader, "test-text");
          },
      .bench_driver =
          +[](const TextBenchOptions& options,
              const std::shared_ptr<const core::GgufReader>& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("bench", *reader, options.provider_options);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("prompt", reader, options.provider_options);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("chat", reader, options.provider_options);
          },
      .sampling_defaults = &SamplingDefaults,
  });
  registry.AddTextProvider({
      .id = "other-text",
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return ProbeArchitecture(reader, "other-text");
          },
      .bench_driver =
          +[](const TextBenchOptions& options,
              const std::shared_ptr<const core::GgufReader>& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("bench", *reader, options.provider_options);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("prompt", reader, options.provider_options);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("chat", reader, options.provider_options);
          },
  });
}

namespace {

int Run(std::string_view command, std::span<const char* const> args) {
  dispatched.clear();
  dispatched_options.clear();
  if (command == "bench") {
    return gufo::cli::RunBench(args);
  }
  return command == "prompt" ? gufo::cli::RunPrompt(args)
                             : gufo::cli::RunChat(args);
}

int Run(std::string_view command, const std::string& model,
        std::vector<const char*> flags) {
  std::vector<const char*> args = {"--model", model.c_str()};
  args.insert(args.end(), flags.begin(), flags.end());
  return Run(command, args);
}

bool Fail(std::string_view command, std::string_view what) {
  std::cerr << "FAIL: " << command << ": " << what << "\n";
  return false;
}

// Declared options reach the selected provider's driver with explicit values
// only; other providers' options, undeclared commands, type errors and
// repeated options fail before dispatch.
bool CheckProviderOptions(std::string_view command, const std::string& model,
                          const std::string& other) {
  using gufo::ext::FindOption;
  const bool bench = command == "bench";
  const std::string selected = std::string(command) + ":test-text";
  if (Run(command, model, {}) != 42 || !dispatched_options.empty()) {
    return Fail(command, "no options must deliver an empty payload");
  }
  if (Run(command, model,
          {"--fixture-flag", "--fixture-count", "0", "--fixture-file=a b",
           "--draft-tokens", "9"}) != 42 ||
      dispatched != selected || dispatched_options.size() != 3 ||
      *FindOption<bool>(dispatched_options, "--fixture-flag") != true ||
      *FindOption<std::int64_t>(dispatched_options, "--fixture-count") != 0 ||
      *FindOption<std::string>(dispatched_options, "--fixture-file") != "a b") {
    return Fail(command,
                "explicit values, zero included, must be delivered "
                "and core-owned spellings stay with core");
  }
  if (Run(command, model, {"--fixture-flag=false"}) != 42 ||
      *FindOption<bool>(dispatched_options, "--fixture-flag") != false) {
    return Fail(command, "an explicit false must be delivered");
  }
  const int scale = Run(command, model, {"--fixture-scale", "0.5"});
  if (bench ? (scale != 2 || !dispatched.empty())
            : (scale != 42 || *FindOption<double>(dispatched_options,
                                                  "--fixture-scale") != 0.5)) {
    return Fail(command, "an option applies only to its declared commands");
  }
  for (const auto& flags : std::vector<std::vector<const char*>>{
           {"--fixture-count", "x"},
           {"--fixture-count", "1", "--fixture-count", "2"},
           {"--fixture-flag=yes"},
           {"--other-only", "x"},
           {"--not-declared"}}) {
    if (Run(command, model, flags) != 2 || !dispatched.empty()) {
      return Fail(command, std::string("must reject ") + flags.front());
    }
  }
  // The other provider receives only its own options.
  if (Run(command, other, {"--other-only", "x", "--draft-tokens", "9"}) != 42 ||
      dispatched_options.size() != 1 ||
      *FindOption<std::string>(dispatched_options, "--other-only") != "x") {
    return Fail(command, "the other provider must receive only its options");
  }
  return true;
}

bool CheckHelp(std::string_view command) {
  std::ostringstream captured;
  auto* const previous = std::cout.rdbuf(captured.rdbuf());
  const std::array<const char*, 1> args = {"--help"};
  Run(command, args);
  std::cout.rdbuf(previous);
  const auto help = captured.str();
  const bool listed =
      help.find("Provider options:") != std::string::npos &&
      help.find("--fixture-flag") != std::string::npos &&
      help.find("A switch (test-text)") != std::string::npos &&
      help.find("--fixture-count <N>") != std::string::npos &&
      help.find("(test-text, other-text)") != std::string::npos &&
      help.find("Not delivered") == std::string::npos;
  const bool scale = help.find("--fixture-scale") != std::string::npos;
  return listed && scale == (command != "bench")
             ? true
             : Fail(command, "help must list the declared provider options");
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
  const auto write_model = [](std::string_view architecture) {
    const auto path =
        std::filesystem::temp_directory_path() /
        ("gufo-text-dispatch-" + std::to_string(gufo::platform::ProcessId()) +
         "-" + std::string(architecture) + ".gguf");
    gufo::test_fixture::GgufBuilder builder;
    builder.AddMetadataString("general.architecture",
                              std::string(architecture));
    const auto bytes = builder.Build(0);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return file ? path : std::filesystem::path{};
  };
  const auto path = write_model("test-text");
  const auto other_path = write_model("other-text");
  if (path.empty() || other_path.empty()) {
    return 1;
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
  const bool dispatched_command =
      status == 42 && dispatched == std::string(command) + ":test-text";
  const bool sampling_ok =
      command == "bench" || CheckSamplingDefaults(command, model.c_str());
  const bool options_ok =
      CheckProviderOptions(command, model, other_path.string()) &&
      CheckHelp(command);
  std::filesystem::remove(path);
  std::filesystem::remove(other_path);
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
  return options_ok ? 0 : 1;
}
