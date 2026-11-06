#include "src/cli/extension_diagnostics.hpp"

#include <sstream>

#include "gufo/ext/registry.hpp"
#include "gufo/ext/version.hpp"
#include "src/core/json.hpp"

namespace gufo::cli {
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
          << extension.source << ")\n";
    }
    for (const auto& provider : registry.Providers()) {
      out << "  provider " << provider.id << ": "
          << (provider.available ? "available" : provider.unavailable_reason)
          << "\n";
    }
    for (const auto& excluded : registry.Excluded()) {
      out << "  excluded " << excluded.extension.name << " "
          << excluded.extension.version << " (" << excluded.extension.source
          << "): " << excluded.reason << "\n";
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
    built.push_back(std::move(entry));
  }
  report["builtExtensions"] = std::move(built);
  auto excluded = json::Value::array();
  for (const auto& extension : registry.Excluded()) {
    auto entry = json::Value::object();
    entry["name"] = extension.extension.name;
    entry["version"] = extension.extension.version;
    entry["source"] = extension.extension.source;
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
  return report.dump() + "\n";
}
}  // namespace gufo::cli
