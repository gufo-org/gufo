#include "src/cli/extension_diagnostics.hpp"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/text.hpp"
#include "gufo/ext/version.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
namespace {
// Empty for an extension that records no revision.
std::string RevisionText(const ext::ExtensionInfo& extension) {
  if (extension.revision.empty())
    return "";
  return " revision " + extension.revision +
         (extension.dirty ? " with uncommitted changes" : "");
}

void AddRevision(json::Value& entry, const ext::ExtensionInfo& extension) {
  if (extension.revision.empty())
    return;
  entry["revision"] = extension.revision;
  entry["dirty"] = extension.dirty;
}

std::string_view CapabilityName(ext::Capability capability) {
  switch (capability) {
    case ext::Capability::kTextGeneration:
      return "text_generation";
    case ext::Capability::kEmbeddings:
      return "embeddings";
    case ext::Capability::kSpeechRecognition:
      return "speech_recognition";
    case ext::Capability::kSpeechSynthesis:
      return "speech_synthesis";
    case ext::Capability::kImageGeneration:
      return "image_generation";
    case ext::Capability::kVideoGeneration:
      return "video_generation";
    case ext::Capability::kDecision:
      return "decision";
  }
  return "unknown";
}

struct SharedKey {
  std::string key;
  ext::Capability capability;
  // Registration order, which is the order their probes are asked in.
  std::vector<const ext::ProviderInfo*> providers;
};

// An artifact key that several providers of one capability declare. A key
// never selects a provider: every probe of the capability is asked in
// registration order and the first to accept a file serves it.
std::vector<SharedKey> SharedKeys(const ext::Registry& registry) {
  std::vector<SharedKey> keys;
  for (const auto& provider : registry.Providers()) {
    for (const auto& key : provider.keys) {
      auto found = std::ranges::find_if(keys, [&](const SharedKey& shared) {
        return shared.key == key && shared.capability == provider.capability;
      });
      if (found == keys.end()) {
        keys.push_back({key, provider.capability, {}});
        found = keys.end() - 1;
      }
      if (std::ranges::find(found->providers, &provider) ==
          found->providers.end()) {
        found->providers.push_back(&provider);
      }
    }
  }
  std::erase_if(keys, [](const SharedKey& shared) {
    return shared.providers.size() < 2;
  });
  return keys;
}

bool AcceptsUnsizedTensors(const ext::ProviderInfo& provider) {
  const auto* text = ext::FindTextProvider(provider.id);
  return text != nullptr && text->accepts_unsized_tensors;
}
}  // namespace
std::string ExtensionDiagnostics(bool json_mode) {
  ext::Registry registry;
  ext::RegisterBuiltExtensions(registry);
  if (!json_mode) {
    std::ostringstream out;
    out << "Extension API: " << ext::RuntimeApiVersion() << "\n"
        << "Extension selection: " << ext::ExtensionSelectionMode() << "\n"
        << "Built extensions:";
    if (registry.Extensions().empty())
      out << " none";
    out << "\n";
    for (const auto& extension : registry.Extensions()) {
      out << "  " << extension.name << " " << extension.version << " ("
          << extension.source << ")" << RevisionText(extension) << "\n";
    }
    for (const auto& provider : registry.Providers()) {
      out << "  provider " << provider.id << ": "
          << (provider.available ? "available" : provider.unavailable_reason)
          << "\n";
    }
    for (const auto& excluded : registry.Excluded()) {
      out << "  excluded " << excluded.extension.name << " "
          << excluded.extension.version << " (" << excluded.extension.source
          << ")" << RevisionText(excluded.extension) << ": " << excluded.reason
          << "\n";
    }
    for (const auto& shared : SharedKeys(registry)) {
      out << "  shared key " << shared.key << " ("
          << CapabilityName(shared.capability)
          << "): the first probe to accept a file serves it, asked in this "
             "order\n";
      std::size_t order = 0;
      for (const auto* provider : shared.providers) {
        if (!provider->available) {
          out << "    - " << provider->id << " (" << provider->extension
              << "): unavailable, " << provider->unavailable_reason << "\n";
          continue;
        }
        out << "    " << ++order << ". " << provider->id << " ("
            << provider->extension << ")";
        if (shared.capability == ext::Capability::kTextGeneration) {
          out << (AcceptsUnsizedTensors(*provider)
                      ? ", also asked for files with unsized tensors"
                      : ", skipped for files with unsized tensors");
        }
        out << "\n";
      }
    }
    return out.str();
  }
  auto report = json::Value::object();
  report["extensionApiVersion"] = std::string(ext::RuntimeApiVersion());
  report["selectionMode"] = std::string(ext::ExtensionSelectionMode());
  auto built = json::Value::array();
  for (const auto& extension : registry.Extensions()) {
    auto entry = json::Value::object();
    entry["name"] = extension.name;
    entry["version"] = extension.version;
    entry["source"] = extension.source;
    AddRevision(entry, extension);
    built.push_back(std::move(entry));
  }
  report["builtExtensions"] = std::move(built);
  auto excluded = json::Value::array();
  for (const auto& extension : registry.Excluded()) {
    auto entry = json::Value::object();
    entry["name"] = extension.extension.name;
    entry["version"] = extension.extension.version;
    entry["source"] = extension.extension.source;
    AddRevision(entry, extension.extension);
    entry["reason"] = extension.reason;
    auto keys = json::Value::array();
    for (const auto& key : extension.keys)
      keys.push_back(key);
    entry["keys"] = std::move(keys);
    excluded.push_back(std::move(entry));
  }
  report["excludedExtensions"] = std::move(excluded);
  auto providers = json::Value::array();
  for (const auto& provider : registry.Providers()) {
    auto entry = json::Value::object();
    entry["id"] = provider.id;
    entry["extension"] = provider.extension;
    entry["available"] = provider.available;
    entry["reason"] = provider.unavailable_reason;
    providers.push_back(std::move(entry));
  }
  report["providers"] = std::move(providers);
  auto shared_keys = json::Value::array();
  for (const auto& shared : SharedKeys(registry)) {
    auto entry = json::Value::object();
    entry["key"] = shared.key;
    entry["capability"] = std::string(CapabilityName(shared.capability));
    auto probe_order = json::Value::array();
    for (const auto* provider : shared.providers) {
      auto candidate = json::Value::object();
      candidate["id"] = provider->id;
      candidate["extension"] = provider->extension;
      candidate["available"] = provider->available;
      if (shared.capability == ext::Capability::kTextGeneration) {
        candidate["acceptsUnsizedTensors"] = AcceptsUnsizedTensors(*provider);
      }
      probe_order.push_back(std::move(candidate));
    }
    entry["probeOrder"] = std::move(probe_order);
    shared_keys.push_back(std::move(entry));
  }
  report["sharedKeys"] = std::move(shared_keys);
  return report.dump() + "\n";
}
}  // namespace gufo::cli
