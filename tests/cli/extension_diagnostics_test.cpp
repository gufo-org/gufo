#include "src/cli/extension_diagnostics.hpp"

#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

// Stand in for the generated registry: three text providers and two embedding
// providers declare shared keys, across extensions and capabilities.
void gufo::ext::RegisterBuiltExtensions(gufo::ext::Registry& registry) {
  registry.BeginExtension({.name = "first", .version = "1", .source = ""});
  registry.AddProvider({
      .id = "first-text",
      .keys = {"gguf:shared", "gguf:first-only"},
  });
  registry.AddTextProvider({.id = "first-text"});
  registry.AddProvider({
      .id = "first-embedding",
      .capability = Capability::kEmbeddings,
      .keys = {"gguf:shared", "safetensors:pair"},
  });
  registry.BeginExtension({.name = "second", .version = "1", .source = ""});
  registry.AddProvider({
      .id = "second-text",
      .keys = {"gguf:shared"},
  });
  registry.AddTextProvider(
      {.id = "second-text", .accepts_unsized_tensors = true});
  registry.AddProvider({
      .id = "second-gpu-text",
      .keys = {"gguf:shared"},
      .available = false,
      .unavailable_reason = "requires hip",
  });
  registry.AddProvider({
      .id = "second-embedding",
      .capability = Capability::kEmbeddings,
      .keys = {"safetensors:pair"},
  });
}
std::string_view gufo::ext::ExtensionSelectionMode() {
  return "explicit";
}

namespace {

int failures = 0;

void Require(bool condition, std::string_view what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

std::vector<std::string> Ids(const gufo::json::Value& shared) {
  std::vector<std::string> ids;
  for (const auto& provider : shared.find("probeOrder")->items()) {
    ids.push_back(provider.member_str("id"));
  }
  return ids;
}

}  // namespace

int main() {
  const auto report = gufo::json::parse(gufo::cli::ExtensionDiagnostics(true));
  const auto& shared = report.find("sharedKeys")->items();
  Require(shared.size() == 2, "one entry per shared key and capability");
  if (shared.size() == 2) {
    // A key two capabilities declare once each is not shared, and a key one
    // provider declares is not listed.
    const auto& text = shared[0];
    Require(text.member_str("key") == "gguf:shared" &&
                text.member_str("capability") == "text_generation",
            "text providers share gguf:shared");
    Require(Ids(text) == std::vector<std::string>{"first-text", "second-text",
                                                  "second-gpu-text"},
            "providers are listed in registration order");
    const auto& order = text.find("probeOrder")->items();
    Require(order.size() == 3 && order[0].member_str("extension") == "first" &&
                order[0].find("available")->as_bool() &&
                !order[0].find("acceptsUnsizedTensors")->as_bool() &&
                order[1].find("acceptsUnsizedTensors")->as_bool() &&
                !order[2].find("available")->as_bool(),
            "each provider reports its extension, availability and whether "
            "it probes unsized files");
    const auto& embeddings = shared[1];
    Require(embeddings.member_str("key") == "safetensors:pair" &&
                embeddings.member_str("capability") == "embeddings",
            "embedding providers share safetensors:pair");
    Require(Ids(embeddings) ==
                std::vector<std::string>{"first-embedding", "second-embedding"},
            "embedding providers are listed in registration order");
    Require(embeddings.find("probeOrder")
                    ->items()[0]
                    .find("acceptsUnsizedTensors") == nullptr,
            "only text providers report unsized-tensor probing");
  }

  const std::string human = gufo::cli::ExtensionDiagnostics(false);
  const std::string expected =
      "  shared key gguf:shared (text_generation): the first probe to accept "
      "a file serves it, asked in this order\n"
      "    1. first-text (first), skipped for files with unsized tensors\n"
      "    2. second-text (second), also asked for files with unsized "
      "tensors\n"
      "    - second-gpu-text (second): unavailable, requires hip\n"
      "  shared key safetensors:pair (embeddings): the first probe to accept "
      "a file serves it, asked in this order\n"
      "    1. first-embedding (first)\n"
      "    2. second-embedding (second)\n";
  Require(human.ends_with(expected), "the text report names the probe order");
  Require(human.find("gguf:first-only") == std::string::npos,
          "an unshared key is not reported");
  if (failures != 0) {
    std::cerr << human;
    return 1;
  }
  std::cout << "PASS: extension diagnostics name shared keys in probe order\n";
  return 0;
}
