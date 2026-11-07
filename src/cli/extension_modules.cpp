// Experimental loading of extensions built as modules (gufo-org/gufo#1075).
// GUFO_LOAD_EXTENSIONS lists module files separated by ':'. A module must have
// been built against this executable's exact build identity, and it may not
// claim an extension, provider or model already registered.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/module.hpp"

namespace gufo::ext {
namespace {
using BuildIdFn = const char* (*)();
using ManifestFn = const char* (*)();
using RegisterFn = void (*)(Registry*, const char*);

struct Module {
  std::string source;
  json::Value manifest;
  RegisterFn register_extension = nullptr;
};

Capability ParseCapability(const std::string& name) {
  static const std::map<std::string, Capability, std::less<>> kCapabilities = {
      {"text_generation", Capability::kTextGeneration},
      {"embeddings", Capability::kEmbeddings},
      {"speech_recognition", Capability::kSpeechRecognition},
      {"speech_synthesis", Capability::kSpeechSynthesis},
      {"image_generation", Capability::kImageGeneration},
      {"video_generation", Capability::kVideoGeneration},
      {"decision", Capability::kDecision},
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

bool Shares(const std::vector<std::string>& left,
            const std::vector<std::string>& right) {
  return std::ranges::any_of(left, [&](const auto& value) {
    return std::ranges::find(right, value) != right.end();
  });
}

// Checks the manifest before registration: Registry rejects duplicates only
// after the module's entry ran, and capability tables replace providers by ID.
void CheckConflicts(const Registry& registry, const Module& module) {
  const auto fail = [&](const std::string& reason) {
    throw std::runtime_error("extension module " + module.source + ": " +
                             reason);
  };
  const auto name = module.manifest.member_str("name");
  if (std::ranges::any_of(registry.Extensions(),
                          [&](const auto& e) { return e.name == name; }))
    fail("extension " + name + " is already registered");
  const auto* providers = module.manifest.find("providers");
  if (!providers || !providers->is_array())
    fail("manifest has no providers");
  for (const auto& provider : providers->items()) {
    const auto id = provider.member_str("id");
    const auto capability = ParseCapability(provider.member_str("capability"));
    const auto keys = Strings(provider, "keys");
    const auto model_ids = Strings(provider, "model_ids");
    for (const auto& existing : registry.Providers()) {
      if (existing.id == id)
        fail("provider " + id + " is already registered");
      if (existing.capability != capability)
        continue;
      if (Shares(existing.keys, keys))
        fail("provider " + id + " shares an artifact key with provider " +
             existing.id);
      if (Shares(existing.model_ids, model_ids))
        fail("provider " + id + " shares a model ID with provider " +
             existing.id);
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
      throw std::runtime_error("extension module " + path.string() +
                               " does not export " + name);
    return address;
  };
  const auto build_id =
      reinterpret_cast<BuildIdFn>(symbol("gufo_ext_module_build_id_v1"))();
  if (std::string_view(build_id) != GUFO_EXTENSION_BUILD_ID)
    throw std::runtime_error("extension module " + path.string() +
                             " was built for build identity " + build_id +
                             ", this gufo is " + GUFO_EXTENSION_BUILD_ID +
                             "; rebuild the module against this gufo");
  module.manifest = json::parse(
      reinterpret_cast<ManifestFn>(symbol("gufo_ext_module_manifest_v1"))());
  module.register_extension =
      reinterpret_cast<RegisterFn>(symbol("gufo_ext_module_register_v1"));
  return module;
}

const std::vector<Module>& Modules() {
  // Loaded once per process; every Registry instance registers the same set.
  static std::once_flag once;
  static std::vector<Module> modules;
  std::call_once(once, [] {
    const char* list = std::getenv("GUFO_LOAD_EXTENSIONS");
    if (!list)
      return;
    std::vector<Module> loaded;
    std::string_view rest(list);
    while (!rest.empty()) {
      const auto end = rest.find(':');
      const auto item = rest.substr(0, end);
      if (!item.empty())
        loaded.push_back(Load(std::filesystem::path(item)));
      rest = end == std::string_view::npos ? std::string_view{}
                                           : rest.substr(end + 1);
    }
    modules = std::move(loaded);
  });
  return modules;
}
}  // namespace

void RegisterExtensionModules(Registry& registry) {
  // A requested module that cannot load is a configuration error; continuing
  // without it would serve a different set of models than asked for.
  try {
    for (const auto& module : Modules()) {
      CheckConflicts(registry, module);
      module.register_extension(&registry, module.source.c_str());
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    std::exit(2);
  }
}
}  // namespace gufo::ext
