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

// The SDK installed with this executable, or staged beside it in a build tree.
struct Sdk {
  std::filesystem::path root;
  json::Value description;  // sdk.json
};
// Throws when no SDK belongs to this executable.
Sdk LocateSdk();
}  // namespace gufo::cli::extension_store
