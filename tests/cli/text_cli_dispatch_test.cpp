#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/bench/bench.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/runner_cli.hpp"
#include "src/cli/speculative_settings.hpp"
#include "src/core/platform/os_info.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace {

std::string dispatched;
gufo::sampling::SamplingConfig dispatched_sampling;
gufo::ext::ParsedOptions dispatched_options;
std::size_t dispatched_draft_tokens = 0;
std::size_t dispatched_min_draft_tokens = 0;
std::string dispatched_draft_policy;
bool invalid_defaults = false;
// What the runner-only providers' factories last received and answered.
int runner_created = 0;
gufo::ext::TextSpeculativeConfig runner_speculative;
std::string runner_error;

// The runner of `runner-text` drafts from a separate file; `self-draft-text`
// checkpoints carry their own draft, as EXL3 Flash-Next packs do. Neither
// returns a runner: the error tells which way the factory decided.
std::shared_ptr<gufo::ext::TextModelRunner> CreateRunner(
    const gufo::ext::TextRunnerOptions& options, bool self_draft,
    std::string* error) {
  ++runner_created;
  runner_speculative = options.speculative;
  const bool missing = options.speculative.backend !=
                           gufo::ext::TextSpeculativeBackend::kDisabled &&
                       options.speculative.draft_model_path.empty();
  *error = missing && !self_draft ? "fixture runner requires a draft model"
                                  : "fixture runner";
  runner_error = *error;
  return nullptr;
}

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

template<typename Options>
int Dispatch(std::string_view command, const gufo::core::GgufReader& reader,
             const Options& driver_options) {
  const auto& options = driver_options.provider_options;
  dispatched_draft_tokens = driver_options.draft_tokens;
  dispatched_min_draft_tokens = driver_options.min_draft_tokens;
  dispatched_draft_policy = driver_options.draft_policy;
  const std::string_view architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "test-text" && architecture != "other-text" &&
      architecture != "test-unsized") {
    return 1;
  }
  dispatched = std::string(command) + ":" + std::string(architecture);
  dispatched_options = options;
  return 42;
}

// A model directory a runner-only provider claims holds a file named after it.
bool ProbeDirectory(std::string_view root, std::string_view provider) {
  std::error_code error;
  return std::filesystem::is_regular_file(
      std::filesystem::path(root) / std::string(provider), error);
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
  // Another capability's options never reach the text commands.
  registry.AddProvider({
      .id = "test-embedding",
      .capability = gufo::ext::Capability::kEmbeddings,
      .keys = {"safetensors:test-embedding"},
      .options = {{"--fixture-embedding", OptionType::kString, all, ""}},
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
            return Dispatch("bench", *reader, options);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("prompt", reader, options);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("chat", reader, options);
          },
      .sampling_defaults = &SamplingDefaults,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 15,
                                .min_draft_tokens = 2,
                                .draft_policy = "calibrated",
                                .draft_policies = {"calibrated", "fixed"}}},
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
            return Dispatch("bench", *reader, options);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("prompt", reader, options);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("chat", reader, options);
          },
  });
  // Has no drivers: the commands load its runner through the server backend.
  registry.AddTextProvider({
      .id = "runner-text",
      .probe =
          +[](std::string_view root) {
            return ProbeDirectory(root, "runner-text");
          },
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return ProbeArchitecture(reader, "runner-text");
          },
      .create =
          +[](const TextRunnerOptions& options, std::string* error) {
            return CreateRunner(options, false, error);
          },
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 5,
                                .min_draft_tokens = 3,
                                .draft_policy = "calibrated",
                                .draft_policies = {"calibrated"}}},
  });
  registry.AddTextProvider({
      .id = "self-draft-text",
      .probe =
          +[](std::string_view root) {
            return ProbeDirectory(root, "self-draft-text");
          },
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return ProbeArchitecture(reader, "self-draft-text");
          },
      .create =
          +[](const TextRunnerOptions& options, std::string* error) {
            return CreateRunner(options, true, error);
          },
  });
  // Owns a tensor layout core cannot size; the others never see such a file.
  registry.AddTextProvider({
      .id = "test-unsized",
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return ProbeArchitecture(reader, "test-unsized");
          },
      .bench_driver =
          +[](const TextBenchOptions& options,
              const std::shared_ptr<const core::GgufReader>& reader,
              std::chrono::steady_clock::time_point) {
            return Dispatch("bench", *reader, options);
          },
      .prompt_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("prompt", reader, options);
          },
      .chat_driver =
          +[](const TextPromptOptions& options, const core::GgufReader& reader,
              std::chrono::steady_clock::time_point) {
            dispatched_sampling = options.sampling;
            return Dispatch("chat", reader, options);
          },
      .sampling_defaults = &SamplingDefaults,
      .accepts_unsized_tensors = true,
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
           {"--fixture-embedding", "x"},
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

// Draft settings: the provider's default for the backend, else the host's 7,
// 1 and no policy. The deprecated draft flags are ignored.
bool CheckSpeculativeDefaults(std::string_view command,
                              const std::string& model,
                              const std::string& other) {
  struct Case {
    const std::string* model;
    std::vector<const char*> flags;
    std::size_t max;
    std::size_t min;
    std::string_view policy;
    const char* what;
  };
  const std::vector<Case> cases = {
      {&model,
       {"--speculative", "mtp", "--mtp-model", "x"},
       15,
       2,
       "calibrated",
       "the defaults"},
      {&model,
       {"--speculative", "mtp", "--mtp-model", "x", "--draft-tokens", "9"},
       15,
       2,
       "calibrated",
       "an ignored limit"},
      {&model,
       {"--speculative", "mtp", "--mtp-model", "x", "--min-draft-tokens", "1",
        "--draft-policy", "fixed"},
       15,
       2,
       "calibrated",
       "an ignored minimum and policy"},
      {&model, {}, 7, 1, "", "no backend"},
      {&model,
       {"--speculative", "dspark", "--dspark-model", "x"},
       7,
       1,
       "",
       "a backend the provider does not declare"},
      {&other,
       {"--speculative", "mtp", "--mtp-model", "x"},
       7,
       1,
       "",
       "a provider declaring none"},
      {&other,
       {"--speculative", "dflash2", "--dflash-model", "x", "--draft-policy",
        "fixed"},
       7,
       1,
       "",
       "an ignored DFlash2 policy"},
  };
  for (const auto& entry : cases) {
    if (Run(command, *entry.model, entry.flags) != 42 ||
        dispatched_draft_tokens != entry.max ||
        dispatched_min_draft_tokens != entry.min ||
        dispatched_draft_policy != entry.policy) {
      return Fail(command,
                  std::string("wrong draft settings for ") + entry.what);
    }
  }
  const std::vector<std::pair<const std::string*, std::vector<const char*>>>
      ignored = {
          {&model, {"--draft-policy", "fixed"}},
          {&model,
           {"--speculative", "mtp", "--mtp-model", "x", "--draft-policy",
            "unknown"}},
          {&model,
           {"--speculative", "mtp", "--mtp-model", "x", "--draft-tokens", "2",
            "--min-draft-tokens", "3"}},
          {&other,
           {"--speculative", "mtp", "--mtp-model", "x", "--draft-policy",
            "fixed"}},
          {&model,
           {"--speculative", "mtp", "--mtp-model", "x", "--min-draft-tokens",
            "16"}},
          {&model,
           {"--speculative", "mtp", "--mtp-model", "x", "--draft-tokens", "1"}},
      };
  for (const auto& [selected, flags] : ignored) {
    if (Run(command, *selected, flags) != 42) {
      return Fail(command, "must ignore deprecated draft flags");
    }
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
      help.find("Not delivered") == std::string::npos &&
      help.find("--fixture-embedding") == std::string::npos;
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

// The runner options `flags` give a provider without a driver, as the command
// parses them and prepares its runner; false when either rejects them.
bool PrepareRunner(std::string_view command, std::string_view provider_id,
                   std::vector<const char*> flags,
                   gufo::ext::TextRunnerOptions* options) {
  const auto* provider = gufo::ext::FindTextProvider(provider_id);
  std::string error;
  if (command == "bench") {
    const auto parsed = gufo::cli::ParseBenchOptions(flags, &error);
    return parsed &&
           gufo::cli::PrepareRunnerOptions(*parsed, *provider, options, &error);
  }
  const auto parsed = gufo::cli::ParsePromptOptions(flags, &error);
  return parsed &&
         gufo::cli::PrepareRunnerOptions(*parsed, *provider, options, &error);
}

// A provider without a driver loads its runner with the speculative
// configuration `gufo serve` would give it: the backend, the draft model of
// that backend's option and the provider's draft settings. The factory then
// accepts or rejects the backend, so speculation is never dropped silently.
// The preparation is checked in every build; the load needs the GPU engine.
bool CheckRunnerSpeculative(std::string_view command,
                            const std::string& model) {
  runner_error.clear();
  using Backend = gufo::ext::TextSpeculativeBackend;
  struct Case {
    std::vector<const char*> flags;
    Backend backend;
    std::string_view draft_model;
    std::uint32_t max;
    std::uint32_t min;
    std::string_view policy;
  };
  const std::vector<Case> cases = {
      {{"--speculative", "mtp", "--mtp-model", "mtp.gguf", "--dflash-model",
        "dflash.gguf"},
       Backend::kMtp,
       "mtp.gguf",
       5,
       3,
       "calibrated"},
      {{"--speculative", "dflash2", "--dflash-model", "dflash.gguf"},
       Backend::kDFlash,
       "dflash.gguf",
       7,
       1,
       ""},
      {{"--dspark-model", "dspark.gguf"},
       Backend::kDSpark,
       "dspark.gguf",
       7,
       1,
       ""},
      {{"--dflash-model", "dflash.gguf"}, Backend::kDisabled, "", 7, 1, ""},
      {{}, Backend::kDisabled, "", 7, 1, ""},
  };
  const auto matches = [](const gufo::ext::TextSpeculativeConfig& received,
                          const Case& entry) {
    return received.backend == entry.backend &&
           received.draft_model_path == entry.draft_model &&
           received.max_draft_tokens == entry.max &&
           received.min_draft_tokens == entry.min &&
           received.draft_policy == entry.policy;
  };
  for (const auto& entry : cases) {
    gufo::ext::TextRunnerOptions prepared;
    if (!PrepareRunner(command, "runner-text", entry.flags, &prepared) ||
        !matches(prepared.speculative, entry)) {
      return Fail(command,
                  "the runner options must carry the speculative "
                  "settings");
    }
    runner_created = 0;
    runner_speculative = {};
    const int status = Run(command, model, entry.flags);
#if defined(ENGINE_ENABLE_HIP)
    const bool loaded = status == 1 && runner_created == 1 &&
                        matches(runner_speculative, entry);
#else
    // Without the GPU engine no runner loads.
    const bool loaded = status == 1 && runner_created == 0;
#endif
    if (!loaded) {
      return Fail(command, "the runner must receive the speculative settings");
    }
  }
  return true;
}

// Without its draft model a backend still reaches a runner, from a GGUF or a
// model directory, which decides as under `gufo serve`: a checkpoint may carry
// its own draft. A provider's own drivers read the draft model from the
// command line, so the command rejects a missing one before dispatching.
struct RunnerModel {
  std::string path;
  bool self_draft;
};

bool CheckMissingDraftModel(std::string_view command,
                            const std::string& driver_model,
                            const std::vector<RunnerModel>& runner_models) {
  using Backend = gufo::ext::TextSpeculativeBackend;
  const struct {
    const char* backend;
    Backend expected;
  } backends[] = {{"mtp", Backend::kMtp},
                  {"dflash2", Backend::kDFlash},
                  {"dspark", Backend::kDSpark}};
  for (const auto& [backend, expected] : backends) {
    const std::vector<const char*> flags = {"--speculative", backend};
    for (const auto* provider : {"runner-text", "self-draft-text"}) {
      gufo::ext::TextRunnerOptions prepared;
      if (!PrepareRunner(command, provider, flags, &prepared) ||
          prepared.speculative.backend != expected ||
          !prepared.speculative.draft_model_path.empty()) {
        return Fail(command,
                    "a runner must receive a backend without its "
                    "draft model");
      }
    }
    for (const auto& [model, self_draft] : runner_models) {
      runner_created = 0;
      runner_speculative = {};
      runner_error.clear();
      const int status = Run(command, model, flags);
#if defined(ENGINE_ENABLE_HIP)
      const bool decided =
          status == 1 && runner_created == 1 &&
          runner_speculative.backend == expected &&
          runner_error == (self_draft ? "fixture runner"
                                      : "fixture runner requires a draft "
                                        "model");
#else
      // Without the GPU engine no runner loads.
      (void)self_draft;
      const bool decided = status == 1 && runner_created == 0;
#endif
      if (!decided) {
        return Fail(command, "the runner must decide on a missing draft model");
      }
    }
    if (Run(command, driver_model, flags) != 2 || !dispatched.empty()) {
      return Fail(command, "a driver must not run without its draft model");
    }
  }
  return true;
}

// Each backend drafts from its own option's model, which may be empty; unknown
// backends are refused. RequireDraftModel refuses an empty draft model.
bool CheckSpeculativeConfig() {
  using Backend = gufo::ext::TextSpeculativeBackend;
  const auto is = [](std::string_view backend, const std::string& draft,
                     Backend expected, std::string_view draft_model) {
    const auto config =
        gufo::cli::SpeculativeConfigFor(backend, draft, draft, draft);
    return config.backend == expected && config.draft_model_path == draft_model;
  };
  const auto rejects = [](std::string_view backend, std::string_view message) {
    try {
      gufo::cli::RequireDraftModel(backend, {}, {}, {});
    } catch (const std::invalid_argument& exception) {
      return std::string_view(exception.what()) == message;
    }
    return false;
  };
  const bool mapped =
      is("", "d", Backend::kDisabled, "") &&
      is("off", "d", Backend::kDisabled, "") &&
      gufo::cli::SpeculativeConfigFor("mtp", "m", "f", "s").draft_model_path ==
          "m" &&
      gufo::cli::SpeculativeConfigFor("dflash2", "m", "f", "s")
              .draft_model_path == "f" &&
      gufo::cli::SpeculativeConfigFor("dspark", "m", "f", "s")
              .draft_model_path == "s" &&
      is("mtp", "d", Backend::kMtp, "d") &&
      is("dflash2", "d", Backend::kDFlash, "d") &&
      is("dspark", "d", Backend::kDSpark, "d") &&
      is("mtp", "", Backend::kMtp, "");
  bool required = true;
  try {
    gufo::cli::RequireDraftModel("", {}, {}, {});
    gufo::cli::RequireDraftModel("off", {}, {}, {});
    gufo::cli::RequireDraftModel("mtp", "m", {}, {});
  } catch (const std::invalid_argument&) {
    required = false;
  }
  const bool rejected =
      required && rejects("mtp", "MTP requires --mtp-model") &&
      rejects("dflash2", "DFlash2 requires --dflash-model") &&
      rejects("dspark", "DSpark requires --dspark-model") &&
      rejects("eagle", "speculative backend 'eagle' is not supported");
  return mapped && rejected
             ? true
             : Fail("speculative", "wrong configuration for a backend");
}

}  // namespace

// Only a provider that accepts unsized tensors receives such a file, through
// the usual reader, probe and sampling defaults; for the others the command
// fails as before, without dispatching.
bool CheckUnsizedTensors(std::string_view command) {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo-text-dispatch-" + std::to_string(gufo::platform::ProcessId()) +
       "-unsized.gguf");
  const auto write = [&](std::string_view architecture) {
    gufo::test_fixture::GgufBuilder builder;
    builder.AddMetadataString("general.architecture",
                              std::string(architecture));
    // Q4_K rows are whole 256-element blocks; core cannot size 100 columns.
    builder.AddTensor("custom.weight", {100, 2}, gufo::core::GgmlType::kQ4_K,
                      0);
    const auto bytes = builder.Build(256);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return static_cast<bool>(file);
  };
  const auto model = path.string();
  const std::array<const char*, 2> args = {"--model", model.c_str()};
  bool ok = write("test-text") && Run(command, args) == 1 && dispatched.empty();
  dispatched_sampling = {};
  ok = ok && write("test-unsized") && Run(command, args) == 42 &&
       dispatched == std::string(command) + ":test-unsized" &&
       (command == "bench" || dispatched_sampling.temperature == 0.75F);
  std::filesystem::remove(path);
  return ok ? true
            : Fail(command,
                   "unsized tensors must reach only a provider accepting "
                   "them");
}

namespace {

// A server backend that, like the scheduler, hands Wait the answer's pieces
// only for a streamed request; every request's history is kept.
class ChatBackend final : public gufo::server::TextGenerationBackend {
public:
  std::vector<gufo::server::ChatRequest> requests;

  std::string model_id() const override { return "chat-fixture"; }
  bool ready() const override { return true; }
  std::size_t count_tokens(std::string_view) const override { return 0; }
  Result complete(std::string_view, std::size_t,
                  const gufo::sampling::SamplingConfig&,
                  const CancellationCheck&, const TokenCallback&,
                  std::string_view, const std::vector<std::string>&, bool,
                  bool) override {
    return {};
  }
  Result chat(const gufo::server::ChatRequest&, std::size_t,
              const gufo::sampling::SamplingConfig&, const CancellationCheck&,
              const TokenCallback&) override {
    return {};
  }
  std::shared_ptr<GenerationRequest> start_chat(
      const gufo::server::ChatRequest& request, std::size_t,
      const gufo::sampling::SamplingConfig&, const CancellationCheck&,
      bool stream_output) override {
    requests.push_back(request);
    return std::make_shared<Pending>(requests.size(), stream_output);
  }

private:
  class Pending final : public GenerationRequest {
  public:
    Pending(std::size_t turn, bool stream) : turn_(turn), stream_(stream) {}
    Result Wait(const TokenCallback& on_token, const ProgressCallback&,
                const StartCallback&) override {
      const std::vector<std::string> pieces = {
          "<think>why ", std::to_string(turn_), "</think>", "Answer ",
          std::to_string(turn_)};
      Result result;
      for (const auto& piece : pieces) {
        result.text += piece;
        if (stream_ && on_token)
          on_token(gufo::server::TokenPiece(piece));
      }
      result.completion_tokens = pieces.size();
      return result;
    }
    void Cancel() noexcept override {}

  private:
    std::size_t turn_;
    bool stream_;
  };
};

}  // namespace

// `gufo chat` on a runner-only provider prints each streamed answer and keeps
// the conversation, reasoning apart, for the next turn.
bool CheckChatSession() {
  ChatBackend backend;
  gufo::ext::TextPromptOptions options;
  options.system_prompt = "Be brief.";
  std::istringstream in("first\n\nsecond\n/exit\nnever\n");
  std::ostringstream out;
  std::size_t answers = 0;
  const int status = gufo::cli::RunChatSession(
      backend, options, in, out,
      [&](const gufo::server::TextGenerationBackend::Result&) { ++answers; });
  using Role = gufo::ext::ChatRole;
  const auto& requests = backend.requests;
  const auto is = [](const gufo::ext::ChatMessage& message, Role role,
                     std::string_view content, std::string_view thought = {}) {
    return message.role == role && message.content == content &&
           message.thought == thought;
  };
  const bool ok =
      status == 0 && answers == 2 &&
      out.str() ==
          "\n> <think>why 1</think>Answer 1\n\n> \n> <think>why "
          "2</think>Answer 2\n\n> " &&
      requests.size() == 2 && requests[0].messages.size() == 2 &&
      is(requests[0].messages[0], Role::kSystem, "Be brief.") &&
      is(requests[0].messages[1], Role::kUser, "first") &&
      requests[1].messages.size() == 4 &&
      is(requests[1].messages[2], Role::kAssistant, "Answer 1", "why 1") &&
      is(requests[1].messages[3], Role::kUser, "second");
  if (!ok)
    std::cerr << "chat output: " << out.str() << "\n";
  return ok ? true
            : Fail("chat", "each streamed answer must be printed and kept");
}

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
  const auto runner_path = write_model("runner-text");
  const auto self_draft_path = write_model("self-draft-text");
  // Model directories the runner-only providers claim.
  const auto write_directory = [](std::string_view provider) {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("gufo-text-dispatch-" + std::to_string(gufo::platform::ProcessId()) +
         "-" + std::string(provider));
    std::filesystem::create_directories(root);
    std::ofstream(root / std::string(provider)) << "";
    return root;
  };
  const auto runner_dir = write_directory("runner-text");
  const auto self_draft_dir = write_directory("self-draft-text");
  if (path.empty() || other_path.empty() || runner_path.empty() ||
      self_draft_path.empty()) {
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
      CheckSpeculativeDefaults(command, model, other_path.string()) &&
      CheckHelp(command) && CheckSpeculativeConfig() &&
      CheckRunnerSpeculative(command, runner_path.string()) &&
      CheckMissingDraftModel(command, model,
                             {{runner_path.string(), false},
                              {self_draft_path.string(), true},
                              {runner_dir.string(), false},
                              {self_draft_dir.string(), true}});
  const bool unsized_ok =
      CheckUnsizedTensors(command) && (command != "chat" || CheckChatSession());
  std::filesystem::remove(path);
  std::filesystem::remove(other_path);
  std::filesystem::remove(runner_path);
  std::filesystem::remove(self_draft_path);
  std::filesystem::remove_all(runner_dir);
  std::filesystem::remove_all(self_draft_dir);
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
  return options_ok && unsized_ok ? 0 : 1;
}
