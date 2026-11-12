#pragma once

// The reviewed extension registry (https://github.com/gufo-org/gufo-extension):
// one index.json listing extensions and their reviewed releases.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::cli::extension_registry {
struct Release {
  std::string name;
  std::string version;
  std::string source;    // HTTPS Git URL
  std::string revision;  // reviewed commit
  std::string digest;    // tree SHA-256, as the store records it
};

// The registry index: $GUFO_EXTENSION_REGISTRY (a URL or a local file), else
// the official one. Throws when it cannot be read or parsed.
json::Value FetchIndex();
std::string IndexLocation();

// The newest release of `name` (or exactly `version`) that this gufo can
// build: listed for this platform, and its api_version within the host's
// compatibility series and revision. Empty when none qualifies; `why` says
// what ruled the releases out.
std::optional<Release> Resolve(const json::Value& index, std::string_view name,
                               std::string_view version, std::string* why);

// One registry extension as `gufo extension list --available` shows it.
struct Listing {
  std::string name;
  std::string description;
  std::string newest;  // newest release this gufo supports; empty if none
  std::string reason;  // why none qualifies
};
std::vector<Listing> List(const json::Value& index);

// MAJOR.MINOR.PATCH ordering; malformed versions order first.
bool Newer(std::string_view candidate, std::string_view installed);
}  // namespace gufo::cli::extension_registry
