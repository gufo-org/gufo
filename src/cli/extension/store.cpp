#include "src/cli/extension/store.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/platform/module.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::cli::extension_store {
namespace {
std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}
}  // namespace

std::string_view HostBuildId() {
  return GUFO_EXTENSION_BUILD_ID;
}

std::filesystem::path StoreDirectory() {
  if (const char* override_dir = std::getenv("GUFO_EXTENSIONS_DIR");
      override_dir != nullptr && *override_dir != '\0')
    return override_dir;
  const auto data = platform::UserDataDirectory();
  if (data.empty())
    throw std::runtime_error(
        "cannot determine the user data directory; set GUFO_EXTENSIONS_DIR");
  return data / "gufo" / "extensions";
}

std::vector<Installed> Read(const std::filesystem::path& store) {
  const auto path = store / "installed.json";
  std::vector<Installed> installed;
  if (!std::filesystem::exists(path))
    return installed;
  const auto state = json::parse(ReadText(path));
  if (state.member_size("schema") != 1)
    throw std::runtime_error(path.string() + ": unsupported schema");
  if (const auto* entries = state.find("extensions");
      entries && entries->is_array()) {
    for (const auto& entry : entries->items()) {
      Installed item;
      item.name = entry.member_str("name");
      item.version = entry.member_str("version");
      item.digest = entry.member_str("digest");
      if (const auto* replace = entry.find("replace_builtin");
          replace && replace->is_bool())
        item.replace_builtin = replace->as_bool();
      if (const auto* source = entry.find("source"); source) {
        item.source.kind = source->member_str("kind");
        item.source.location = source->member_str("location");
        item.source.revision = source->member_str("revision");
      }
      installed.push_back(std::move(item));
    }
  }
  return installed;
}

void Write(const std::filesystem::path& store,
           const std::vector<Installed>& installed) {
  auto entries = json::Value::array();
  for (const auto& item : installed) {
    auto entry = json::Value::object();
    entry["name"] = item.name;
    entry["version"] = item.version;
    auto source = json::Value::object();
    source["kind"] = item.source.kind;
    source["location"] = item.source.location;
    source["revision"] = item.source.revision;
    entry["source"] = std::move(source);
    entry["digest"] = item.digest;
    entry["replace_builtin"] = item.replace_builtin;
    entries.push_back(std::move(entry));
  }
  auto state = json::Value::object();
  state["schema"] = 1;
  state["extensions"] = std::move(entries);
  std::filesystem::create_directories(store);
  const auto temporary = store / "installed.json.tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << state.dump() << "\n";
    if (!output)
      throw std::runtime_error("cannot write " + temporary.string());
  }
  std::filesystem::rename(temporary, store / "installed.json");
}

std::filesystem::path SourceDirectory(const std::filesystem::path& store,
                                      std::string_view name) {
  return store / std::string(name) / "source";
}

std::filesystem::path ModulePath(const std::filesystem::path& store,
                                 std::string_view name,
                                 std::string_view build_id) {
  return store / std::string(name) / "modules" /
         (std::string(build_id) + std::string(platform::ModuleFileSuffix()));
}

json::Value Manifest(const std::filesystem::path& source) {
  return json::parse(ReadText(source / "extension.json"));
}

Sdk LocateSdk() {
  // GUFO_SDK_DIR names an SDK packaged separately. Installed: <prefix>/bin/gufo
  // and <prefix>/lib/gufo/sdk. Build tree: the SDK is staged beside gufo.
  std::vector<std::filesystem::path> candidates;
  if (const char* override_dir = std::getenv("GUFO_SDK_DIR");
      override_dir != nullptr && *override_dir != '\0') {
    candidates.emplace_back(override_dir);
  } else {
    const auto directory = platform::ExecutablePath().parent_path();
    candidates = {directory.parent_path() / "lib" / "gufo" / "sdk",
                  directory / "sdk"};
  }
  for (const auto& candidate : candidates) {
    if (!std::filesystem::exists(candidate / "sdk.json"))
      continue;
    Sdk sdk{candidate, json::parse(ReadText(candidate / "sdk.json"))};
    if (sdk.description.member_str("build_id") != HostBuildId())
      throw std::runtime_error("the SDK in " + candidate.string() +
                               " belongs to another gufo build");
    return sdk;
  }
  throw std::runtime_error(
      "this gufo has no extension SDK: install it with gufo (lib/gufo/sdk) "
      "or set GUFO_SDK_DIR");
}
}  // namespace gufo::cli::extension_store
