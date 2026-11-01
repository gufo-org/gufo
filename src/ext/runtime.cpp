#include <algorithm>
#include <stdexcept>
#include <utility>

#include "gufo/ext/host.hpp"
#include "gufo/ext/registry.hpp"
#include "gufo/ext/version.hpp"

namespace gufo::ext {
std::string_view RuntimeApiVersion() {
  return kApiVersion;
}
void Host::Log(LogLevel level, std::string_view component,
               std::string_view message) const {
  if (log) {
    log(context, level, component, message);
  } else {
    server::Logger::Log(level, component, message);
  }
}
void Registry::BeginExtension(ExtensionInfo extension) {
  if (extension.name.empty() ||
      std::ranges::any_of(extensions_, [&](const auto& e) {
        return e.name == extension.name;
      })) {
    throw std::invalid_argument("duplicate or empty extension name: " +
                                extension.name);
  }
  current_extension_ = extension.name;
  extensions_.push_back(std::move(extension));
}
void Registry::AddProvider(ProviderInfo provider) {
  if (current_extension_.empty()) {
    throw std::logic_error("BeginExtension must precede provider registration");
  }
  if (provider.id.empty() ||
      std::ranges::any_of(providers_,
                          [&](const auto& p) { return p.id == provider.id; })) {
    throw std::invalid_argument("duplicate or empty provider ID: " +
                                provider.id);
  }
  for (const auto& option : provider.options) {
    if (IsCoreOption(option.name)) {
      throw std::invalid_argument("core-owned option: " + option.name);
    }
    for (const auto& existing : providers_) {
      for (const auto& other : existing.options) {
        for (const auto& command : option.commands) {
          if (option.name == other.name && option.type != other.type &&
              std::ranges::find(other.commands, command) !=
                  other.commands.end()) {
            throw std::invalid_argument("conflicting option types: " +
                                        option.name);
          }
        }
      }
    }
  }
  provider.extension = current_extension_;
  providers_.push_back(std::move(provider));
}
const ProviderInfo& Registry::Provider(std::string_view id) const {
  const auto found = std::ranges::find(providers_, id, &ProviderInfo::id);
  if (found == providers_.end()) {
    throw std::invalid_argument("unknown provider: " + std::string(id));
  }
  return *found;
}
void Registry::AddExcluded(ExcludedExtension extension) {
  if (extension.extension.name.empty() ||
      std::ranges::any_of(excluded_, [&](const auto& e) {
        return e.extension.name == extension.extension.name;
      })) {
    throw std::invalid_argument("duplicate or empty excluded extension name");
  }
  excluded_.push_back(std::move(extension));
}
std::string Registry::ExcludedKeyDiagnostic(std::string_view key) const {
  std::string message;
  for (const auto& extension : excluded_) {
    if (std::ranges::find(extension.keys, key) != extension.keys.end()) {
      if (!message.empty())
        message += "; ";
      message += "built without extension " + extension.extension.name + " (" +
                 extension.reason + "); rebuild with -DGUFO_EXTENSIONS=" +
                 extension.extension.name + " and its declared dependencies";
    }
  }
  return message;
}

static std::vector<EmbeddingProvider> g_embedding_providers;

void RegisterEmbeddingProvider(EmbeddingProvider provider) {
  for (auto& existing : g_embedding_providers) {
    if (existing.id == provider.id) {
      existing = std::move(provider);
      return;
    }
  }
  g_embedding_providers.push_back(std::move(provider));
}

const EmbeddingProvider* FindEmbeddingProvider(std::string_view id) {
  for (const auto& provider : g_embedding_providers) {
    if (provider.id == id) {
      return &provider;
    }
  }
  return nullptr;
}

const EmbeddingProvider* ProbeEmbeddingProvider(std::string_view model_root) {
  for (const auto& provider : g_embedding_providers) {
    if (provider.probe && provider.probe(model_root)) {
      return &provider;
    }
  }
  return nullptr;
}

void Registry::AddEmbeddingProvider(EmbeddingProvider provider) {
  RegisterEmbeddingProvider(std::move(provider));
}
}  // namespace gufo::ext
