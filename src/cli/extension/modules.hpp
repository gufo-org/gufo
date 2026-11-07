#pragma once

// Loading of installed extension modules (gufo-org/gufo#1075).

#include <filesystem>
#include <string_view>

#include "src/core/json.hpp"

namespace gufo::ext {
class Registry;
// Called by the generated registry of the gufo executable: whether an
// installed module replaces the built-in extension `name`, and registering
// every loadable module after the built-in extensions. A module that is
// stale, unloadable or conflicting is skipped with a warning, so a broken
// module never keeps `gufo extension` from repairing the store.
bool ExtensionModuleShadows(std::string_view name);
void RegisterExtensionModules(Registry& registry);
}  // namespace gufo::ext

namespace gufo::cli::extension_modules {
// `gufo extension` manages the store without loading it.
void Disable();
// Throws when `manifest` claims an extension, provider, artifact key or model
// ID registered in `registry` by an extension other than `ignore`.
void CheckConflicts(const ext::Registry& registry, const json::Value& manifest,
                    std::string_view ignore = {});
// Throws when `other` (another installed manifest) conflicts with `manifest`.
void CheckConflicts(const json::Value& manifest, const json::Value& other);
// Loads `module` like gufo would, checks its build identity and exports, and
// runs its registration entry in an empty registry.
void Verify(const std::filesystem::path& module);
}  // namespace gufo::cli::extension_modules
