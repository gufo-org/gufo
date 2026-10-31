#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gufo/ext/generated/alpha_constants.hpp"
#include "gufo/ext/host.hpp"
#include "gufo/ext/registry.hpp"
#include "gufo/ext/version.hpp"
#include "src/cli/serve/logging.hpp"

namespace {
using namespace gufo::ext;
void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}
template<class Function>
void Reject(Function function, std::string_view text) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    Require(std::string_view(error.what()).find(text) != std::string_view::npos,
            error.what());
    return;
  }
  throw std::runtime_error("expected rejection: " + std::string(text));
}
ParsedOptions Parse(const Registry& registry,
                    std::initializer_list<std::string_view> arguments,
                    std::string_view command = "prompt") {
  return ParseOptions(registry, command, {arguments.begin(), arguments.size()});
}
void TestRegistryAndParser() {
  Registry registry;
  Require(RuntimeApiVersion() == "0.1.2", "runtime/header API version");
  RegisterBuiltExtensions(registry);
  Require(registry.Extensions().size() == 3,
          "built family provenance retained");
  Require(registry.Extensions()[0].name == "base" &&
              registry.Extensions()[1].name == "alpha" &&
              registry.Extensions()[2].name == "beta",
          "stable dependency/discovery ordering");
  Require(registry.Providers().size() == 3, "all metadata retained");
  const auto& alpha = registry.Provider("alpha");
  const auto& beta = registry.Provider("beta");
  const auto& hip = registry.Provider("alpha-hip");
  Require(alpha.extension == "alpha" && beta.extension == "beta",
          "provider ownership");
  Require(alpha.model_ids == std::vector<std::string>{"alpha-model"},
          "model IDs");
  Require(alpha.keys == std::vector<std::string>{"gguf:alpha"},
          "generated key constant");
  Require(alpha.options[1].help ==
              "Path to a \"draft\" file: @PROJECT_SOURCE_DIR@ ${NO_EXPANSION} "
              "$<CONFIG>",
          "metadata is inert data during generation");
  Require(hip.available == static_cast<bool>(TEST_PROVIDER_HIP),
          "target availability");
  auto parsed = Parse(
      registry, {"--draft-tokens", "-3", "--alpha-path=weights.gguf", "--scale",
                 "1.25", "--label=hello", "--speculative=draft"});
  Require(std::get<std::int64_t>(parsed.at("--draft-tokens")) == -3,
          "integer type");
  Require(std::get<double>(parsed.at("--scale")) == 1.25, "number type");
  Require(std::get<std::string>(parsed.at("--alpha-path")) == "weights.gguf",
          "path type");
  Require(std::get<std::string>(parsed.at("--label")) == "hello",
          "string type");
  ValidateSelectedOptions(registry, "prompt", alpha, parsed);
  Reject([&] { ValidateSelectedOptions(registry, "prompt", beta, parsed); },
         "provider beta");
  Reject([&] { Parse(registry, {"--alpha-path=x"}, "bench"); },
         "unknown option");
  Reject([&] { Parse(registry, {"--draft-tokens=3x"}); }, "invalid value");
  Reject([&] { Parse(registry, {"--draft-tokens=9223372036854775808"}); },
         "invalid value");
  Reject([&] { Parse(registry, {"--scale=nan"}); }, "invalid value");
  Reject([&] { Parse(registry, {"--scale=1e999"}); }, "invalid value");
  Reject([&] { Parse(registry, {"--label"}); }, "missing value");
  Reject([&] { Parse(registry, {"--label", "--scale=2"}); }, "missing value");
  Reject([&] { Parse(registry, {"--label=a", "--label=b"}); },
         "duplicate option");
  Reject([&] { Parse(registry, {"--speculative=unknown"}); },
         "unknown --speculative value");
  auto other = Parse(registry, {"--speculative=other"});
  Reject([&] { ValidateSelectedOptions(registry, "prompt", alpha, other); },
         "provider alpha");
  ValidateSelectedOptions(registry, "prompt", beta, other);
  ValidateSelectedOptions(registry, "prompt", alpha,
                          Parse(registry, {"--speculative=off"}));
  Reject([&] { Parse(registry, {"--provider=unknown"}); }, "unknown provider");
  Reject(
      [&] {
        ValidateSelectedOptions(registry, "prompt", alpha,
                                Parse(registry, {"--provider=beta"}));
      },
      "does not match");
  if (TEST_PROVIDER_HIP) {
    auto flag = Parse(registry, {"--hip-only"});
    Require(std::get<bool>(flag.at("--hip-only")), "implicit true");
    ValidateSelectedOptions(registry, "prompt", hip, flag);
    auto off = Parse(registry, {"--hip-only=false"});
    Require(!std::get<bool>(off.at("--hip-only")), "explicit false");
    Reject([&] { Parse(registry, {"--hip-only=yes"}); }, "invalid value");
    Reject([&] { ValidateSelectedOptions(registry, "prompt", alpha, flag); },
           "provider alpha");
  } else {
    Require(hip.unavailable_reason == "requires hip", "unavailable reason");
    Reject([&] { Parse(registry, {"--hip-only"}); }, "unknown option");
    Reject([&] { Parse(registry, {"--provider=alpha-hip"}); }, "requires hip");
    Reject([&] { Parse(registry, {"--speculative=hip-draft"}); },
           "unknown --speculative value");
  }
  const std::array<OptionSpec, 1> core{
      {{"--context", OptionType::kInteger, {"prompt"}, ""}}};
  const std::array<std::string_view, 1> args{"--context=2048"};
  auto settings = ParseOptions(registry, "prompt", args, core);
  ValidateSelectedOptions(registry, "prompt", alpha, settings);
  Require(std::get<std::int64_t>(settings.at("--context")) == 2048,
          "core option parser");
  Reject([&] { registry.AddProvider(alpha); }, "duplicate");
  Reject([&] { registry.BeginExtension({"alpha", "1.0.0", "test"}); },
         "duplicate");
  auto invalid = alpha;
  invalid.id = "invalid";
  invalid.options = {{"--port", OptionType::kInteger, {"prompt"}, ""}};
  Reject([&] { registry.AddProvider(invalid); }, "core-owned");
  invalid.options = {{"--draft-tokens", OptionType::kString, {"prompt"}, ""}};
  Reject([&] { registry.AddProvider(invalid); }, "conflicting");
}
void TestHost() {
  std::string observed;
  Host host{&observed,
            [](void* context, LogLevel level, std::string_view component,
               std::string_view message) {
              Require(level == LogLevel::kWarn && component == "test",
                      "injected host logging");
              *static_cast<std::string*>(context) = message;
            }};
  host.Log(LogLevel::kWarn, "test", "captured");
  Require(observed == "captured", "callback context");
  const auto previous = gufo::server::Logger::Level();
  gufo::server::Logger::SetLevel(LogLevel::kError);
  std::ostringstream output;
  auto* old = std::clog.rdbuf(output.rdbuf());
  Host{}.Log(LogLevel::kInfo, "host", "hidden");
  Host{}.Log(LogLevel::kError, "host", "escaped\nline");
  std::clog.rdbuf(old);
  gufo::server::Logger::SetLevel(previous);
  Require(output.str().find("hidden") == std::string::npos,
          "shared log threshold");
  Require(output.str().find("escaped\\x0aline") != std::string::npos,
          "shared log escaping");
}
}  // namespace
int main() {
  try {
    TestRegistryAndParser();
    TestHost();
    std::cout << "extension runtime contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
