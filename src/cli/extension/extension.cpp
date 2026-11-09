#include "src/cli/extension/extension.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "gufo/ext/registry.hpp"
#include "src/cli/extension/modules.hpp"
#include "src/cli/extension/registry.hpp"
#include "src/cli/extension/store.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/env.hpp"
#include "src/core/platform/module.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/core/platform/process.hpp"
#include "src/core/platform/publish.hpp"

namespace gufo::cli {
namespace {
namespace fs = std::filesystem;
namespace store = extension_store;
namespace registry = extension_registry;

constexpr std::string_view kUsage =
    "Usage: gufo extension <COMMAND> [OPTIONS]\n\n"
    "Build extensions as modules for this gufo and manage them. Installed\n"
    "extensions load in every later gufo command; gufo itself is not "
    "rebuilt.\n\n"
    "Commands:\n"
    "  install <SOURCE>     Build and install an extension. SOURCE is a\n"
    "                       reviewed registry release, <name>[@<version>], a\n"
    "                       local directory or <git-url>@<revision>\n"
    "      --replace-builtin  Replace the built-in extension of the same name\n"
    "      --yes            Do not ask before building unvalidated code\n"
    "  list [--json]        List built-in, installed and registry extensions\n"
    "  update [NAME...]     Move registry installs to their newest reviewed\n"
    "                       release and build what this gufo lacks\n"
    "  remove <NAME>        Remove an installed extension\n\n"
    "Installed extensions live in $GUFO_EXTENSIONS_DIR, else\n"
    "$XDG_DATA_HOME/gufo/extensions. The registry index is\n"
    "$GUFO_EXTENSION_REGISTRY, else "
    "https://github.com/gufo-org/gufo-extension.\n";

struct Tools {
  std::string cmake;
  std::string generator;
  std::string make_program;
};

const std::regex& NamePattern() {
  static const std::regex pattern("^[a-z][a-z0-9_]*$");
  return pattern;
}

// Every tool the module build needs, reported together before any work.
Tools Preflight(const store::Sdk& sdk, bool needs_git) {
  std::vector<std::string> missing;
  const auto& description = sdk.description;
  const auto resolve = [&](const std::string& recorded,
                           std::string_view program) -> std::string {
    if (!recorded.empty() && fs::exists(recorded))
      return recorded;
    const auto found = platform::FindExecutable(program);
    return found.empty() ? std::string{} : found.string();
  };
  Tools tools;
  tools.cmake = resolve(description.member_str("cmake"), "cmake");
  if (tools.cmake.empty())
    missing.push_back("CMake (cmake), version 3.27 or newer");
  tools.generator = description.member_str("generator");
  if (tools.generator == "Ninja") {
    tools.make_program =
        resolve(description.member_str("make_program"), "ninja");
    if (tools.make_program.empty())
      missing.push_back("Ninja (ninja)");
  }
  if (const auto* compilers = description.find("compilers");
      compilers && compilers->is_object()) {
    for (const auto& [language, label] :
         {std::pair{"c", "C compiler"}, std::pair{"cxx", "C++ compiler"},
          std::pair{"hip", "HIP compiler"}}) {
      const auto path = compilers->member_str(language);
      if (!path.empty() && !fs::exists(path))
        missing.push_back(std::string(label) + " " + path);
    }
  }
  if (const auto path = description.member_str("hip_device_lib_path");
      !path.empty() && !fs::exists(path))
    missing.push_back("ROCm device libraries " + path);
  if (const auto* dependencies = description.find("dependencies");
      dependencies && dependencies->is_array()) {
    for (const auto& dependency : dependencies->items()) {
      const auto path = dependency.get_str();
      if (!path.empty() && !fs::exists(path))
        missing.push_back("dependency " + path);
    }
  }
  // The build environment of the host, for toolchains that read it.
  if (const auto* environment = description.find("environment");
      environment && environment->is_object()) {
    for (const auto& [name, value] : environment->members()) {
      if (std::getenv(name.c_str()) == nullptr)
        (void)platform::SetEnv(name, value.get_str());
    }
  }
  if (needs_git && platform::FindExecutable("git").empty())
    missing.push_back("Git (git)");
  if (!missing.empty()) {
    std::string message =
        "missing build tooling for extension modules of this gufo:";
    for (const auto& item : missing)
      message += "\n  - " + item;
    message +=
        "\nModules must be built with the exact toolchain and libraries gufo "
        "was built with (" +
        (sdk.root / "profile.cmake").string() +
        "). Use the environment that built gufo, its `nix develop` shell or "
        "the gufo-runtime image.";
    throw std::runtime_error(message);
  }
  return tools;
}

void Run(const Tools& tools, const std::vector<std::string>& arguments,
         const std::string& what) {
  std::string error;
  const int status =
      platform::RunInherited(tools.cmake.c_str(), arguments, &error);
  if (status != 0)
    throw std::runtime_error(what + " failed" +
                             (error.empty() ? "" : ": " + error));
}

// Builds `name` from `source` into `module`, leaving no build tree behind.
void Build(const store::Sdk& sdk, const Tools& tools, const fs::path& source,
           const std::string& name, const fs::path& build,
           const fs::path& module) {
  fs::remove_all(build);
  std::vector<std::string> configure = {
      "-S",
      sdk.root.string(),
      "-B",
      build.string(),
      "-G",
      tools.generator,
      "-C",
      (sdk.root / "profile.cmake").string(),
      "-DGUFO_EXTENSION_DIRS=" + source.string(),
      "-DGUFO_EXTENSION_MODULES=" + name,
  };
  if (!tools.make_program.empty())
    configure.push_back("-DCMAKE_MAKE_PROGRAM=" + tools.make_program);
  Run(tools, configure, "configuring extension " + name);
  Run(tools,
      {"--build", build.string(), "--target", "gufo_ext_module_" + name,
       "--parallel"},
      "building extension " + name);
  const auto built =
      build / "extension-modules" /
      ("gufo-ext-" + name + std::string(platform::ModuleFileSuffix()));
  fs::create_directories(module.parent_path());
  fs::copy_file(built, module, fs::copy_options::overwrite_existing);
  fs::remove_all(build);
}

std::string TreeDigest(const fs::path& root) {
  std::vector<fs::path> files;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file())
      files.push_back(fs::relative(entry.path(), root));
  }
  std::ranges::sort(files);
  crypto::Sha256Hasher hasher;
  std::vector<char> buffer(1 << 16);
  for (const auto& file : files) {
    const auto name = file.generic_string();
    hasher.Update(
        {reinterpret_cast<const std::uint8_t*>(name.data()), name.size() + 1});
    std::ifstream input(root / file, std::ios::binary);
    while (input.read(buffer.data(),
                      static_cast<std::streamsize>(buffer.size())) ||
           input.gcount() > 0) {
      hasher.Update({reinterpret_cast<const std::uint8_t*>(buffer.data()),
                     static_cast<std::size_t>(input.gcount())});
    }
  }
  return hasher.FinishHex();
}

void CopySource(const fs::path& from, const fs::path& to) {
  fs::create_directories(to);
  for (auto it = fs::recursive_directory_iterator(from);
       it != fs::recursive_directory_iterator(); ++it) {
    const auto relative = fs::relative(it->path(), from);
    if (relative.begin()->string() == ".git") {
      it.disable_recursion_pending();
      continue;
    }
    if (it->is_directory())
      fs::create_directories(to / relative);
    else
      fs::copy(it->path(), to / relative, fs::copy_options::copy_symlinks);
  }
}

struct Request {
  store::Source source;
  bool replace_builtin = false;
  bool yes = false;
  std::string registry_name;     // install <name>[@<version>]
  std::string registry_version;  // empty: the newest compatible release
};

const std::regex& RegistryPattern() {
  static const std::regex pattern(
      "^([a-z][a-z0-9_]*)(@([0-9]+\\.[0-9]+\\.[0-9]+))?$");
  return pattern;
}

Request ParseSource(const std::string& text) {
  Request request;
  std::error_code ec;
  if (fs::is_directory(text, ec)) {
    request.source = {"path", fs::canonical(text).string(), ""};
    return request;
  }
  const bool url = text.starts_with("https://") ||
                   text.starts_with("http://") || text.starts_with("ssh://") ||
                   text.starts_with("file://") || text.starts_with("git@");
  if (url) {
    const auto at = text.rfind('@');
    if (at == std::string::npos || at + 1 == text.size() ||
        (text.starts_with("git@") && at < 4))
      throw std::runtime_error("pin a revision: " + text + "@<commit-or-tag>");
    request.source = {"git", text.substr(0, at), text.substr(at + 1)};
    return request;
  }
  if (text == "list" || text == "update" || text == "remove" ||
      text == "install")
    throw std::runtime_error("'" + text +
                             "' is a command: did you mean 'gufo extension " +
                             text + "'?");
  std::smatch match;
  if (std::regex_match(text, match, RegistryPattern())) {
    request.registry_name = match[1];
    request.registry_version = match[3];
    return request;
  }
  throw std::runtime_error("no such directory, git source or registry name: " +
                           text);
}

bool Confirm(const Request& request) {
  const auto& source = request.source;
  std::cerr
      << "WARNING: " << source.location
      << (source.revision.empty() ? "" : "@" + source.revision)
      << " has not been validated by the Gufo maintainers.\n"
         "It may contain malicious code, which runs while it builds and in "
         "every\ngufo command that loads it. Review the source before "
         "continuing.\n";
  if (request.yes)
    return true;
  if (!platform::StdinIsTerminal())
    throw std::runtime_error(
        "pass --yes to install unvalidated code without a terminal");
  std::cerr << "Continue? [y/N] " << std::flush;
  std::string answer;
  std::getline(std::cin, answer);
  return answer == "y" || answer == "Y" || answer == "yes";
}

// Fetches the pinned source into `destination`; returns the exact revision.
// A Git source is cloned into `scratch` and only its tree is copied: on
// Windows, files under a fresh .git can stay locked by Git helpers or
// scanners for a while, so the clone is removed best-effort.
std::string Fetch(const store::Source& source, const fs::path& destination,
                  const fs::path& scratch) {
  if (source.kind == "path") {
    CopySource(source.location, destination);
    return {};
  }
  std::error_code ignored;
  fs::remove_all(scratch, ignored);
  const auto clone_dir = scratch / "clone";
  std::string error;
  // Exact bytes: line-ending conversion would change the reviewed digest.
  const std::vector<std::string> clone = {
      "-c",       "core.autocrlf=false", "clone",         "--quiet",
      "--config", "core.autocrlf=false", source.location, clone_dir.string()};
  const std::vector<std::string> checkout = {"-C",       clone_dir.string(),
                                             "checkout", "--quiet",
                                             "--detach", source.revision};
  if (platform::RunInherited("git", clone, &error) != 0 ||
      platform::RunInherited("git", checkout, &error) != 0)
    throw std::runtime_error("cannot fetch " + source.location + "@" +
                             source.revision +
                             (error.empty() ? "" : ": " + error));
  // Detached: HEAD holds the commit itself.
  std::string commit;
  {
    std::ifstream head(clone_dir / ".git" / "HEAD");
    std::getline(head, commit);
  }
  CopySource(clone_dir, destination);
  fs::remove_all(scratch, ignored);
  return commit;
}

std::vector<std::string> BuiltinNames(const ext::Registry& registry) {
  std::vector<std::string> names;
  for (const auto& extension : registry.Extensions())
    names.push_back(extension.name);
  return names;
}

platform::FileLock LockStore(const fs::path& directory) {
  fs::create_directories(directory);
  return platform::FileLock::Acquire((directory / ".lock").string());
}

std::string Describe(const store::Installed& installed) {
  if (installed.source.kind == "registry")
    return "reviewed release from the extension registry, " +
           installed.source.location + "@" + installed.source.revision;
  return installed.source.location +
         (installed.source.revision.empty()
              ? ", sha256 " + installed.digest.substr(0, 12)
              : "@" + installed.source.revision);
}

// Fetches, checks, builds and records one extension; the store is locked.
// A registry release must match its reviewed digest before anything builds.
store::Installed Perform(const fs::path& directory, const store::Sdk& sdk,
                         const Tools& tools, const store::Source& source,
                         bool replace_builtin,
                         const std::string& expected_digest) {
  const auto staging = directory / ".staging";
  fs::remove_all(staging);
  store::Installed installed;
  installed.source = source;
  installed.source.revision =
      Fetch(source, staging / "source", directory / ".fetch");
  const auto manifest = store::Manifest(staging / "source");
  installed.name = manifest.member_str("name");
  installed.version = manifest.member_str("version");
  installed.replace_builtin = replace_builtin;
  if (!std::regex_match(installed.name, NamePattern()))
    throw std::runtime_error("invalid extension name '" + installed.name + "'");
  installed.digest = TreeDigest(staging / "source");
  if (!expected_digest.empty() && installed.digest != expected_digest)
    throw std::runtime_error(
        installed.name + " at " + installed.source.revision +
        " does not match its reviewed release: sha256 " + installed.digest +
        ", the registry lists " + expected_digest);

  ext::Registry builtins;
  ext::RegisterBuiltExtensions(builtins);
  const auto names = BuiltinNames(builtins);
  const bool builtin = std::ranges::find(names, installed.name) != names.end();
  if (builtin && !replace_builtin)
    throw std::runtime_error(installed.name +
                             " is built into this gufo; pass "
                             "--replace-builtin to replace it");
  if (!builtin && replace_builtin)
    throw std::runtime_error("--replace-builtin: this gufo has no built-in " +
                             installed.name);
  extension_modules::CheckConflicts(builtins, manifest,
                                    replace_builtin ? installed.name : "");
  auto all = store::Read(directory);
  for (const auto& other : all) {
    if (other.name != installed.name)
      extension_modules::CheckConflicts(
          manifest,
          store::Manifest(store::SourceDirectory(directory, other.name)));
  }

  const auto module = staging / "modules" /
                      (std::string(store::HostBuildId()) +
                       std::string(platform::ModuleFileSuffix()));
  Build(sdk, tools, staging / "source", installed.name, staging / "build",
        module);
  extension_modules::Verify(module);

  fs::remove_all(directory / installed.name);
  fs::rename(staging, directory / installed.name);
  std::erase_if(
      all, [&](const auto& other) { return other.name == installed.name; });
  all.push_back(installed);
  store::Write(directory, all);
  return installed;
}

int Install(Request request) {
  std::string expected_digest;
  if (!request.registry_name.empty()) {
    std::string why;
    const auto release =
        registry::Resolve(registry::FetchIndex(), request.registry_name,
                          request.registry_version, &why);
    if (!release)
      throw std::runtime_error(why);
    request.source = {"registry", release->source, release->revision};
    expected_digest = release->digest;
    std::cerr << "Installing " << release->name << " " << release->version
              << ", reviewed at " << release->revision << ", from "
              << registry::IndexLocation() << "\n";
  }
  const auto sdk = store::LocateSdk();
  const auto tools = Preflight(sdk, request.source.kind != "path");
  if (request.source.kind != "registry" && !Confirm(request)) {
    std::cerr << "Not installed.\n";
    return 1;
  }
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  const auto installed = Perform(directory, sdk, tools, request.source,
                                 request.replace_builtin, expected_digest);
  std::cout << "Installed " << installed.name << " " << installed.version
            << " (" << Describe(installed) << ")\n";
  return 0;
}

// Brings installed extensions up to date: a registry install moves to its
// newest reviewed release this gufo supports, and every extension without a
// module for this gufo's build identity is built from its pinned source.
int Update(std::span<const char* const> names) {
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  const auto all = store::Read(directory);
  std::vector<std::string> selected(names.begin(), names.end());
  for (const auto& name : selected) {
    if (std::ranges::none_of(all,
                             [&](const auto& e) { return e.name == name; }))
      throw std::runtime_error("extension " + name + " is not installed");
  }
  std::optional<json::Value> index;
  std::vector<std::pair<store::Installed, registry::Release>> upgrades;
  std::vector<store::Installed> stale;
  for (const auto& installed : all) {
    if (!selected.empty() &&
        std::ranges::find(selected, installed.name) == selected.end())
      continue;
    if (installed.source.kind == "registry") {
      if (!index)
        index = registry::FetchIndex();
      std::string why;
      const auto release = registry::Resolve(*index, installed.name, {}, &why);
      if (release && registry::Newer(release->version, installed.version)) {
        upgrades.emplace_back(installed, *release);
        continue;
      }
    }
    if (fs::exists(
            store::ModulePath(directory, installed.name, store::HostBuildId())))
      std::cout << installed.name << " " << installed.version
                << " is up to date\n";
    else
      stale.push_back(installed);
  }
  if (upgrades.empty() && stale.empty())
    return 0;
  const auto sdk = store::LocateSdk();
  const auto tools = Preflight(sdk, !upgrades.empty());
  for (const auto& [installed, release] : upgrades) {
    const auto updated = Perform(directory, sdk, tools,
                                 {"registry", release.source, release.revision},
                                 installed.replace_builtin, release.digest);
    std::cout << "Updated " << installed.name << " " << installed.version
              << " to " << updated.version << " (" << Describe(updated)
              << ")\n";
  }
  for (const auto& installed : stale) {
    const auto module =
        store::ModulePath(directory, installed.name, store::HostBuildId());
    auto pending = module;
    pending += ".new";
    Build(sdk, tools, store::SourceDirectory(directory, installed.name),
          installed.name, directory / installed.name / "build", pending);
    extension_modules::Verify(pending);
    fs::rename(pending, module);
    std::cout << "Updated " << installed.name << " " << installed.version
              << " for this gufo\n";
  }
  return 0;
}

int Remove(const std::string& name) {
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  auto all = store::Read(directory);
  if (std::erase_if(all, [&](const auto& e) { return e.name == name; }) == 0)
    throw std::runtime_error("extension " + name + " is not installed");
  store::Write(directory, all);
  fs::remove_all(directory / name);
  std::cout << "Removed " << name
            << ". Running gufo processes keep it until they exit.\n";
  return 0;
}

// Every extension this gufo knows about, one row per name: built in,
// installed as a module, or offered by the registry. The registry is optional
// here: without it the local extensions are still listed.
int List(bool json_mode) {
  struct Row {
    std::string name;
    std::string version;  // built-in or installed version
    std::string description;
    bool builtin = false;
    const store::Installed* installed = nullptr;
    std::string status;  // installed module: ready or needs update
    std::string newest;  // newest registry release this gufo supports
    std::string reason;  // why no registry release qualifies
  };
  std::vector<Row> rows;
  const auto row = [&](const std::string& name) -> Row& {
    const auto found = std::ranges::find(rows, name, &Row::name);
    if (found != rows.end())
      return *found;
    rows.push_back(Row{name});
    return rows.back();
  };
  ext::Registry builtins;
  ext::RegisterBuiltExtensions(builtins);
  for (const auto& extension : builtins.Extensions()) {
    auto& item = row(extension.name);
    item.builtin = true;
    item.version = extension.version;
  }
  const auto directory = store::StoreDirectory();
  const auto all = store::Read(directory);
  for (const auto& installed : all) {
    auto& item = row(installed.name);
    item.installed = &installed;
    item.version = installed.version;
    item.status = fs::exists(store::ModulePath(directory, installed.name,
                                               store::HostBuildId()))
                      ? "ready"
                      : "needs update";
  }
  std::string registry_error;
  try {
    for (const auto& listing : registry::List(registry::FetchIndex())) {
      auto& item = row(listing.name);
      item.description = listing.description;
      item.newest = listing.newest;
      item.reason = listing.reason;
    }
  } catch (const std::exception& error) {
    registry_error = error.what();
  }
  std::ranges::sort(rows, {}, &Row::name);

  if (json_mode) {
    auto report = json::Value::object();
    report["buildId"] = std::string(store::HostBuildId());
    report["store"] = directory.string();
    report["registry"] = registry::IndexLocation();
    report["registryError"] = registry_error;
    auto extensions = json::Value::array();
    for (const auto& item : rows) {
      auto entry = json::Value::object();
      entry["name"] = item.name;
      entry["version"] = item.version;
      entry["description"] = item.description;
      entry["builtin"] = item.builtin;
      entry["installed"] = item.installed != nullptr;
      entry["status"] = item.status;
      entry["newest"] = item.newest;
      entry["reason"] = item.reason;
      if (item.installed) {
        entry["sourceKind"] = item.installed->source.kind;
        entry["source"] = item.installed->source.location;
        entry["revision"] = item.installed->source.revision;
        entry["digest"] = item.installed->digest;
        entry["replaceBuiltin"] = item.installed->replace_builtin;
      }
      extensions.push_back(std::move(entry));
    }
    report["extensions"] = std::move(extensions);
    std::cout << report.dump() << "\n";
    return 0;
  }
  if (!registry_error.empty())
    std::cerr << "Warning: " << registry_error << "\n";
  std::cout << "Extensions (installed in " << directory.string() << "):\n";
  for (const auto& item : rows) {
    std::cout << "  " << item.name << " "
              << (item.version.empty() ? item.newest : item.version);
    std::vector<std::string> marks;
    if (item.installed) {
      marks.push_back("installed, " + item.status);
      if (item.installed->replace_builtin)
        marks.push_back("replaces built-in");
    } else if (item.builtin) {
      marks.push_back("built-in");
    }
    if (!item.version.empty() && !item.newest.empty() &&
        registry::Newer(item.newest, item.version))
      marks.push_back(item.newest + " available");
    if (item.version.empty() && item.newest.empty() && !item.reason.empty())
      marks.push_back(item.reason);
    if (!marks.empty()) {
      std::string joined;
      for (const auto& mark : marks)
        joined += (joined.empty() ? "" : "; ") + mark;
      std::cout << "  [" << joined << "]";
    }
    if (!item.description.empty())
      std::cout << "  " << item.description;
    if (item.installed && item.installed->source.kind != "registry")
      std::cout << "  " << item.installed->source.location
                << (item.installed->source.revision.empty()
                        ? ""
                        : "@" + item.installed->source.revision);
    std::cout << "\n";
  }
  return 0;
}
}  // namespace

int RunExtension(std::span<const char* const> args) {
  extension_modules::Disable();
  if (args.empty() || std::string_view(args[0]) == "--help" ||
      std::string_view(args[0]) == "-h") {
    std::cout << kUsage;
    return args.empty() ? 2 : 0;
  }
  const std::string_view command = args[0];
  const auto rest = args.subspan(1);
  try {
    if (command == "install") {
      Request request;
      std::string source;
      for (const std::string_view arg : rest) {
        if (arg == "--replace-builtin")
          request.replace_builtin = true;
        else if (arg == "--yes")
          request.yes = true;
        else if (arg.starts_with("-") || !source.empty())
          throw std::runtime_error("unexpected argument " + std::string(arg));
        else
          source = arg;
      }
      if (source.empty())
        throw std::runtime_error("install needs a SOURCE");
      auto parsed = ParseSource(source);
      parsed.replace_builtin = request.replace_builtin;
      parsed.yes = request.yes;
      return Install(parsed);
    }
    if (command == "list") {
      if (rest.size() > 1 ||
          (rest.size() == 1 && std::string_view(rest[0]) != "--json"))
        throw std::runtime_error("usage: gufo extension list [--json]");
      return List(rest.size() == 1);
    }
    if (command == "update")
      return Update(rest);
    if (command == "remove") {
      if (rest.size() != 1)
        throw std::runtime_error("usage: gufo extension remove <NAME>");
      return Remove(rest[0]);
    }
    throw std::runtime_error("unknown extension command " +
                             std::string(command));
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
}  // namespace gufo::cli
