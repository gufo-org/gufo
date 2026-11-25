#pragma once

// Installed extension modules (gufo-org/gufo#1075). The store lives in
// $GUFO_EXTENSIONS_DIR, else <user data>/gufo/extensions:
//
//   installed.json                 state, written atomically
//   <name>/source/                 pinned extension source
//   <name>/modules/<build-id><ext> module built for one gufo build identity
//
// A gufo executable loads the module matching its own build identity; any
// other build needs `gufo extension update`.

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::cli::extension_store {
struct Source {
  std::string kind;  // "path" or "git"
  std::string location;
  std::string revision;  // git commit; empty for a path
};

struct Installed {
  std::string name;
  std::string version;
  Source source;
  std::string digest;  // SHA-256 over the pinned source tree
  bool replace_builtin = false;
};

// This executable's extension build identity.
std::string_view HostBuildId();
// The in-tree extensions built into this executable that a module may list in
// `depends`, each with the identity its dependents are built against ("" for
// one that exports nothing). Empty where modules cannot have dependencies.
const std::vector<std::pair<std::string, std::string>>& HostExtensions();
// The build identity of the module `manifest` describes: HostBuildId(), plus
// the identities of the host extensions it depends on that export anything,
// as cmake/GufoExtension.cmake's _gufo_module_build_id computes it.
std::string ModuleBuildId(const json::Value& manifest);

std::filesystem::path StoreDirectory();
std::vector<Installed> Read(const std::filesystem::path& store);
void Write(const std::filesystem::path& store,
           const std::vector<Installed>& installed);
std::filesystem::path SourceDirectory(const std::filesystem::path& store,
                                      std::string_view name);
std::filesystem::path ModulePath(const std::filesystem::path& store,
                                 std::string_view name,
                                 std::string_view build_id);
// The pinned manifest; throws when the source is missing or malformed.
json::Value Manifest(const std::filesystem::path& source);
// The module path of installed extension `name` for this executable, from its
// pinned manifest; throws when the source is missing or malformed.
std::filesystem::path InstalledModulePath(const std::filesystem::path& store,
                                          std::string_view name);

// The SDK installed with this executable, or staged beside it in a build tree.
struct Sdk {
  std::filesystem::path root;
  json::Value description;  // sdk.json
};
// Throws when no SDK belongs to this executable.
Sdk LocateSdk();
}  // namespace gufo::cli::extension_store
