#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/embeddings.hpp"
#include "gufo/ext/options.hpp"

namespace gufo::ext {
// Adding a capability requires a core contract. Extensions cannot invent kinds.
enum class Capability {
  kTextGeneration,
  kEmbeddings,
  kSpeechRecognition,
  kSpeechSynthesis,
  kImageGeneration,
  kVideoGeneration,
};
struct ExtensionInfo {
  std::string name;
  std::string version;
  std::string source;
};
struct ExcludedExtension {
  ExtensionInfo extension;
  std::vector<std::string> keys;
  std::vector<std::string> model_ids;
  std::vector<std::string> options;
  std::string reason;
};
struct ProviderInfo {
  std::string id;
  Capability capability = Capability::kTextGeneration;
  std::vector<std::string> keys;
  std::vector<std::string> model_ids;
  std::vector<std::string> speculative;
  std::vector<OptionSpec> options;
  bool available = true;
  std::string unavailable_reason;
  std::string extension;
};
class Registry {
public:
  void BeginExtension(ExtensionInfo extension);
  void AddProvider(ProviderInfo provider);
  void AddExcluded(ExcludedExtension extension);
  void AddEmbeddingProvider(EmbeddingProvider provider);
  std::span<const ExcludedExtension> Excluded() const { return excluded_; }
  // Empty means the key was not found among excluded extensions.
  std::string ExcludedKeyDiagnostic(std::string_view key) const;
  std::span<const ExtensionInfo> Extensions() const { return extensions_; }
  std::span<const ProviderInfo> Providers() const { return providers_; }
  const ProviderInfo& Provider(std::string_view id) const;

private:
  std::vector<ExtensionInfo> extensions_;
  std::vector<ProviderInfo> providers_;
  std::vector<ExcludedExtension> excluded_;
  std::string current_extension_;
};
// Defined in generated per-executable source, never a static initializer.
void RegisterBuiltExtensions(Registry& registry);
// Defined beside the generated registry: discovery, explicit or lock.
std::string_view ExtensionSelectionMode();
}  // namespace gufo::ext
