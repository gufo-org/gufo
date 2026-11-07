#include "src/cli/extension/modules.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "src/cli/extension/store.hpp"
#include "src/core/platform/module.hpp"

namespace gufo::cli::extension_modules {
namespace {
using BuildIdFn = const char* (*)();
using ManifestFn = const char* (*)();
using RegisterFn = void (*)(ext::Registry*, const char*);

std::atomic<bool> g_disabled{false};

struct Module {
  std::string name;
  std::string source;
  json::Value manifest;
  RegisterFn register_extension = nullptr;
  bool replace_builtin = false;
};

ext::Capability ParseCapability(const std::string& name) {
  static const std::map<std::string, ext::Capability, std::less<>>
      kCapabilities = {
          {"text_generation", ext::Capability::kTextGeneration},
          {"embeddings", ext::Capability::kEmbeddings},
          {"speech_recognition", ext::Capability::kSpeechRecognition},
          {"speech_synthesis", ext::Capability::kSpeechSynthesis},
          {"image_generation", ext::Capability::kImageGeneration},
          {"video_generation", ext::Capability::kVideoGeneration},
          {"decision", ext::Capability::kDecision},
      };
  const auto found = kCapabilities.find(name);
  if (found == kCapabilities.end())
    throw std::runtime_error("unknown capability " + name);
  return found->second;
}

std::vector<std::string> Strings(const json::Value& object,
                                 const std::string& key) {
  std::vector<std::string> values;
  if (const auto* array = object.find(key); array && array->is_array()) {
    for (const auto& value : array->items())
      values.push_back(value.get_str());
  }
  return values;
}

// The providers a manifest declares, in the registry's terms.
std::vector<ext::ProviderInfo> Providers(const json::Value& manifest) {
  std::vector<ext::ProviderInfo> providers;
  const auto* declared = manifest.find("providers");
  if (!declared || !declared->is_array())
    throw std::runtime_error("manifest has no providers");
  for (const auto& provider : declared->items()) {
    ext::ProviderInfo info;
    info.id = provider.member_str("id");
    info.capability = ParseCapability(provider.member_str("capability"));
    info.keys = Strings(provider, "keys");
    info.model_ids = Strings(provider, "model_ids");
    info.extension = manifest.member_str("name");
    providers.push_back(std::move(info));
  }
  return providers;
}

bool Shares(const std::vector<std::string>& left,
            const std::vector<std::string>& right) {
  return std::ranges::any_of(left, [&](const auto& value) {
    return std::ranges::find(right, value) != right.end();
  });
}

void CheckProviders(const std::vector<ext::ProviderInfo>& existing,
                    const json::Value& manifest, std::string_view ignore) {
  for (const auto& provider : Providers(manifest)) {
    for (const auto& other : existing) {
      if (!ignore.empty() && other.extension == ignore)
        continue;
      if (other.id == provider.id)
        throw std::runtime_error("provider " + provider.id +
                                 " is already registered by " +
                                 other.extension);
      if (other.capability != provider.capability)
        continue;
      if (Shares(other.keys, provider.keys))
        throw std::runtime_error("provider " + provider.id +
                                 " shares an artifact key with provider " +
                                 other.id + " of " + other.extension);
      if (Shares(other.model_ids, provider.model_ids))
        throw std::runtime_error("provider " + provider.id +
                                 " shares a model ID with provider " +
                                 other.id + " of " + other.extension);
    }
  }
}

Module Load(const std::filesystem::path& path) {
  Module module;
  module.source = "module:" + std::filesystem::absolute(path).string();
  void* handle = platform::LoadModule(path);
  const auto symbol = [&](const char* name) {
    void* address = platform::ModuleSymbol(handle, name);
    if (!address)
      throw std::runtime_error(path.string() + " does not export " + name);
    return address;
  };
  const std::string_view build_id =
      reinterpret_cast<BuildIdFn>(symbol("gufo_ext_module_build_id_v1"))();
  if (build_id != extension_store::HostBuildId())
    throw std::runtime_error(path.string() + " was built for build identity " +
                             std::string(build_id) + ", this gufo is " +
                             std::string(extension_store::HostBuildId()));
  module.manifest = json::parse(
      reinterpret_cast<ManifestFn>(symbol("gufo_ext_module_manifest_v1"))());
  module.name = module.manifest.member_str("name");
  module.register_extension =
      reinterpret_cast<RegisterFn>(symbol("gufo_ext_module_register_v1"));
  return module;
}

void Warn(const std::string& message) {
  std::cerr << "Warning: " << message << "\n";
}

const std::vector<Module>& Modules() {
  // Loaded once per process; every Registry instance registers the same set.
  static std::once_flag once;
  static std::vector<Module> modules;
  std::call_once(once, [] {
    if (g_disabled)
      return;
    if (!platform::ModulesBindToExecutable(GUFO_EXECUTABLE_FILE_NAME)) {
      Warn(std::string("extension modules load only into ") +
           GUFO_EXECUTABLE_FILE_NAME);
      return;
    }
    try {
      const auto store = extension_store::StoreDirectory();
      for (const auto& installed : extension_store::Read(store)) {
        const auto path = extension_store::ModulePath(
            store, installed.name, extension_store::HostBuildId());
        if (!std::filesystem::exists(path)) {
          Warn("extension " + installed.name +
               " is not built for this gufo; run 'gufo extension update " +
               installed.name + "'");
          continue;
        }
        try {
          auto module = Load(path);
          if (module.name != installed.name)
            throw std::runtime_error("module declares extension " +
                                     module.name);
          module.replace_builtin = installed.replace_builtin;
          modules.push_back(std::move(module));
        } catch (const std::exception& error) {
          Warn("skipping extension " + installed.name + ": " + error.what());
        }
      }
    } catch (const std::exception& error) {
      Warn(std::string("skipping installed extensions: ") + error.what());
    }
  });
  return modules;
}
}  // namespace

void Disable() {
  g_disabled = true;
}

void CheckConflicts(const ext::Registry& registry, const json::Value& manifest,
                    std::string_view ignore) {
  const auto name = manifest.member_str("name");
  if (name != ignore &&
      std::ranges::any_of(registry.Extensions(),
                          [&](const auto& e) { return e.name == name; }))
    throw std::runtime_error("extension " + name + " is already registered");
  const auto providers = registry.Providers();
  CheckProviders({providers.begin(), providers.end()}, manifest, ignore);
}

void CheckConflicts(const json::Value& manifest, const json::Value& other) {
  if (manifest.member_str("name") == other.member_str("name"))
    throw std::runtime_error("extension " + other.member_str("name") +
                             " is already installed");
  CheckProviders(Providers(other), manifest, {});
}

void Verify(const std::filesystem::path& module_path) {
  const auto module = Load(module_path);
  ext::Registry registry;
  module.register_extension(&registry, module.source.c_str());
}
}  // namespace gufo::cli::extension_modules

namespace gufo::ext {
bool ExtensionModuleShadows(std::string_view name) {
  return std::ranges::any_of(
      cli::extension_modules::Modules(), [&](const auto& module) {
        return module.replace_builtin && module.name == name;
      });
}

void RegisterExtensionModules(Registry& registry) {
  for (const auto& module : cli::extension_modules::Modules()) {
    try {
      cli::extension_modules::CheckConflicts(registry, module.manifest);
      module.register_extension(&registry, module.source.c_str());
    } catch (const std::exception& error) {
      cli::extension_modules::Warn("skipping extension " + module.name + ": " +
                                   error.what());
    }
  }
}
}  // namespace gufo::ext
