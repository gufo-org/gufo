#include "src/cli/extension/store.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "src/core/crypto/sha256.hpp"
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

const std::vector<std::pair<std::string, std::string>>& HostExtensions() {
  // `name=identity` pairs joined by commas (gufo_extension_host_identities).
  static const auto extensions = [] {
    std::vector<std::pair<std::string, std::string>> parsed;
    std::string_view text = GUFO_EXTENSION_HOST_EXTENSIONS;
    while (!text.empty()) {
      const auto comma = text.find(',');
      const auto pair = text.substr(0, comma);
      const auto equals = pair.find('=');
      parsed.emplace_back(std::string(pair.substr(0, equals)),
                          std::string(pair.substr(equals + 1)));
      text = comma == std::string_view::npos ? std::string_view{}
                                             : text.substr(comma + 1);
    }
    return parsed;
  }();
  return extensions;
}

std::string ModuleBuildId(const json::Value& manifest) {
  std::vector<std::string> depends;
  if (const auto* array = manifest.find("depends");
      array && array->is_array()) {
    for (const auto& value : array->items())
      depends.push_back(value.get_str());
  }
  std::ranges::sort(depends);
  std::string lines;
  for (const auto& dependency : depends) {
    for (const auto& [name, identity] : HostExtensions()) {
      if (name == dependency && !identity.empty())
        lines += "host_extension=" + name + ":" + identity + "\n";
    }
  }
  const std::string base(HostBuildId());
  if (lines.empty())
    return base;
  const auto material = base + "\n" + lines;
  return crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(material.data()), material.size()));
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

std::filesystem::path InstalledModulePath(const std::filesystem::path& store,
                                          std::string_view name) {
  return ModulePath(store, name,
                    ModuleBuildId(Manifest(SourceDirectory(store, name))));
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
