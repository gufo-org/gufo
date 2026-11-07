// `gufo serve` delivers the selected text provider's declared options to its
// runner factory, after the provider is probed and before the runner exists.

#include <array>
#include <filesystem>
#include <fstream>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/cli/provider_options.hpp"
#include "src/cli/serve/inference_backend.hpp"
#include "src/core/platform/os_info.hpp"
#include "tests/fixtures/gguf_builder.hpp"

namespace {

int created = 0;
gufo::ext::ParsedOptions received;

std::shared_ptr<gufo::ext::TextModelRunner> Create(
    const gufo::ext::TextRunnerOptions& options, std::string* error) {
  ++created;
  received = options.provider_options;
  *error = "fixture runner";
  return nullptr;
}

}  // namespace

void gufo::ext::RegisterBuiltExtensions(gufo::ext::Registry& registry) {
  registry.BeginExtension({.name = "fixture", .version = "1", .source = ""});
  registry.AddProvider({
      .id = "serve-text",
      .keys = {"gguf:serve-text"},
      .options = {{"--fixture-tokens", OptionType::kInteger, {"serve"}, ""},
                  {"--draft-tokens", OptionType::kInteger, {"serve"}, ""}},
  });
  registry.AddProvider({
      .id = "other-text",
      .keys = {"gguf:other-text"},
      .options = {{"--other-only", OptionType::kString, {"serve"}, ""}},
  });
  registry.AddTextProvider({
      .id = "serve-text",
      .probe_gguf =
          +[](const core::GgufReader& reader) {
            return reader.GetMetadataString("general.architecture") ==
                   "serve-text";
          },
      .create = &Create,
  });
}

namespace {

bool Expect(bool condition, std::string_view what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << "\n";
  }
  return condition;
}

// Parses `flags` as `gufo serve` would and loads `model` with the resolver.
bool Load(const gufo::ext::Registry& registry, const std::string& model,
          std::vector<const char*> flags, std::string* error) {
  gufo::cli::ArgParser parser("gufo serve");
  std::size_t draft_tokens = 7;
  parser.AddOption("-d", "--draft-tokens", "N", "", "Speculative",
                   &draft_tokens);
  gufo::cli::ProviderOptions provider_options(registry, "serve");
  provider_options.Register(parser);
  if (!parser.Parse(flags, error)) {
    return false;
  }
  provider_options.Parse();
  created = 0;
  received.clear();
  gufo::server::InferenceBackend backend;
  return backend.load(
      model, error, 0, 1, {}, {}, {}, {}, {}, {}, false,
      [&](const gufo::ext::TextProvider& provider,
          const gufo::core::GgufReader&, gufo::ext::TextRunnerOptions* options,
          std::string* message) {
        try {
          options->provider_options = provider_options.Select(provider.id);
          return true;
        } catch (const std::invalid_argument& exception) {
          *message = exception.what();
          return false;
        }
      });
}

}  // namespace

int main() {
  gufo::ext::Registry registry;
  gufo::ext::RegisterBuiltExtensions(registry);
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-serve-options-" +
                     std::to_string(gufo::platform::ProcessId()) + ".gguf");
  gufo::test_fixture::GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "serve-text");
  const auto bytes = builder.Build(0);
  {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  const auto model = path.string();
  using gufo::ext::FindOption;
  bool ok = true;
  std::string error;

  ok &= Expect(
      !Load(registry, model, {"--fixture-tokens", "0", "-d", "9"}, &error) &&
          created == 1 && error == "fixture runner" && received.size() == 1 &&
          *FindOption<std::int64_t>(received, "--fixture-tokens") == 0,
      "the factory must receive the explicit provider values, and core-owned "
      "spellings stay with core");
  ok &= Expect(
      !Load(registry, model, {}, &error) && created == 1 && received.empty(),
      "absent options must stay absent");
  ok &= Expect(!Load(registry, model, {"--other-only", "x"}, &error) &&
                   created == 0 &&
                   error ==
                       "provider serve-text does not support --other-only for "
                       "serve",
               "another provider's option must fail before the runner");
  std::filesystem::remove(path);
  return ok ? 0 : 1;
}
