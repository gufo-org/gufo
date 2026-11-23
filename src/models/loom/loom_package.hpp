// Serving a Loom package as it is published: the card's preset for this GPU,
// and parameter files found by their origin when they are not at their card
// path (the layout `gufo model install` writes: the loom-fun repository with
// each origin's files at their repository paths beside it).
#ifndef GUFO_LOOM_PACKAGE_HPP_
#define GUFO_LOOM_PACKAGE_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::loom {

namespace package {

namespace fs = std::filesystem;

inline std::string ReadText(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("loom: cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), {}};
}

// The GPU Loom opens by default, as its driver reports it
// (loom_text_embed_query_gpu): its target ("gfx1151"; empty when unknown) and
// device-local memory (on an APU, system RAM on Windows and the VRAM carve-out
// on Linux; zero when unknown).
struct Gpu {
  std::string target;
  std::uint64_t memory_bytes = 0;
};

// The preset to serve: the card's only one, or the one loom-fun cards publish
// for this GPU: `chat` for Strix Halo (gfx1151), `chat-gfx1150` or
// `chat-gfx1152` (else `chat`) for those APUs, `chat-r9700` for a gfx1201 with
// 24 GB or more (else `chat-16gb`). Any other GPU, or a card without the GPU's
// preset, serves gfx1151's `chat`, and `warning` says so. Empty lets Loom
// decide (one preset).
inline std::string ChoosePreset(const json::Value& card, const Gpu& gpu,
                                std::string* note, std::string* warning) {
  const auto* presets = card.find("presets");
  if (presets == nullptr || !presets->is_object() ||
      presets->members().size() < 2)
    return {};
  auto has = [&](std::string_view name) {
    return presets->find(std::string(name)) != nullptr;
  };
  std::string names;
  for (const auto& [name, value] : presets->members())
    names += (names.empty() ? "" : ", ") + name;
  const std::string gpu_name = gpu.target.empty() ? "unknown GPU" : gpu.target;
  std::vector<std::string> wanted;
  if (gpu.target == "gfx1151")
    wanted = {"chat"};
  else if (gpu.target == "gfx1150" || gpu.target == "gfx1152")
    wanted = {"chat-" + gpu.target, "chat"};
  else if (gpu.target == "gfx1201")
    wanted = gpu.memory_bytes >= (std::uint64_t{24} << 30)
                 ? std::vector<std::string>{"chat-r9700", "chat-16gb"}
                 : std::vector<std::string>{"chat-16gb"};
  for (const auto& name : wanted) {
    if (has(name)) {
      if (note)
        *note = "preset " + name + " for " + gpu_name;
      return name;
    }
  }
  if (!has("chat"))
    throw std::runtime_error("loom: this package has no preset for " +
                             gpu_name +
                             " and no gfx1151 preset (presets: " + names + ")");
  if (warning)
    *warning = "this package has no preset for " + gpu_name +
               "; serving the gfx1151 preset chat (presets: " + names + ")";
  return "chat";
}

// The Loom SDK release a package was built for: its llm_config.json's
// "sdk": {"hrx_loom": "X.Y.Z"}, "0.1.0" for a package from before the field,
// or empty when the package has no llm_config.json (nothing to check).
inline std::string PackageSdk(const fs::path& root) {
  std::error_code error;
  const auto path = root / "llm_config.json";
  if (!fs::is_regular_file(path, error))
    return {};
  try {
    const auto config = json::parse(ReadText(path));
    if (const auto* sdk = config.find("sdk"); sdk && sdk->is_object()) {
      const auto version = sdk->member_str("hrx_loom");
      if (!version.empty())
        return version;
    }
  } catch (const std::exception&) {
    return {};
  }
  return "0.1.0";
}

// "0.2" of "0.2.0": releases with the same major.minor share package formats.
inline std::string_view SdkSeries(std::string_view version) {
  const auto first = version.find('.');
  const auto second =
      first == std::string_view::npos ? first : version.find('.', first + 1);
  return version.substr(0, second);
}

// A package built for another Loom SDK release fails inside Loom in ways that
// do not say why: refuse it here with what to fetch instead.
inline void CheckSdk(const fs::path& root, std::string_view runtime) {
  const auto built = PackageSdk(root);
  if (built.empty() || runtime.empty() ||
      SdkSeries(built) == SdkSeries(runtime))
    return;
  throw std::runtime_error(
      "loom: this package is built for Loom SDK " + built +
      ", and this extension runs Loom SDK " + std::string(runtime) +
      ". Install the model with `gufo model install` (it pins packages built "
      "for " +
      std::string(runtime) +
      "), or fetch the package revision for it: loom-fun tags the last "
      "revision for each "
      "earlier SDK (v" +
      std::string(SdkSeries(built)) +
      ".0, ...), and the default branch serves the newest.");
}

// A normalized relative path (no "..", ".", empty or absolute parts), as Loom
// requires of card paths: links are made only inside the private directory.
inline bool SafeRelative(std::string_view path) {
  if (path.empty() || path.front() == '/' ||
      path.find('\\') != std::string_view::npos ||
      path.find(':') != std::string_view::npos)
    return false;
  std::size_t start = 0;
  while (start <= path.size()) {
    const auto end = std::min(path.find('/', start), path.size());
    const auto part = path.substr(start, end - start);
    if (part.empty() || part == "." || part == "..")
      return false;
    start = end + 1;
  }
  return true;
}

// A parameter file the card names: its group ("model", "draft", "vision"),
// its path in the package and, when the card says where it comes from, its
// origin ("checkpoint") and repository path.
struct ParameterFile {
  std::string group;
  std::string path;
  std::string origin;
  std::string origin_path;
};

inline void CollectFiles(const json::Value& files, const std::string& group,
                         std::vector<ParameterFile>& out) {
  if (!files.is_array())
    return;
  for (const auto& item : files.items()) {
    if (!item.is_object())
      continue;
    ParameterFile file;
    file.group = group;
    file.path = item.member_str("path");
    const auto origin = item.member_str("origin");
    if (origin.size() > 1 && origin[0] == '@') {
      const auto slash = origin.find('/');
      if (slash != std::string::npos) {
        file.origin = origin.substr(1, slash - 1);
        file.origin_path = origin.substr(slash + 1);
      }
    }
    if (!file.path.empty())
      out.push_back(std::move(file));
  }
}

// Every parameter file of the preset's variant (all variants when the preset
// does not name one).
inline std::vector<ParameterFile> ParameterFiles(const json::Value& card,
                                                 const std::string& preset) {
  std::string variant_name;
  if (const auto* presets = card.find("presets");
      presets && presets->is_object()) {
    const json::Value* chosen = nullptr;
    if (!preset.empty())
      chosen = presets->find(preset);
    else if (presets->members().size() == 1)
      chosen = &presets->members().front().second;
    if (chosen && chosen->is_object())
      variant_name = chosen->member_str("variant");
  }
  std::vector<ParameterFile> files;
  const auto* variants = card.find("variants");
  if (variants == nullptr || !variants->is_object())
    return files;
  for (const auto& [name, variant] : variants->members()) {
    if (!variant_name.empty() && name != variant_name)
      continue;
    const auto* services = variant.find("services");
    if (services == nullptr || !services->is_object())
      continue;
    for (const auto& [service_name, service] : services->members()) {
      (void)service_name;
      const auto* parameters = service.find("parameters");
      if (parameters == nullptr || !parameters->is_object())
        continue;
      for (const auto& [group, parameter] : parameters->members()) {
        if (const auto* list = parameter.find("files"))
          CollectFiles(*list, group, files);
      }
    }
  }
  return files;
}

// Where a parameter file is found when it is not at its card path: at its
// origin's repository path in the package directory, where `gufo model
// install` puts it.
inline std::optional<fs::path> Locate(const ParameterFile& file,
                                      const fs::path& root) {
  std::error_code error;
  if (!file.origin_path.empty() &&
      fs::is_regular_file(root / file.origin_path, error))
    return root / file.origin_path;
  return std::nullopt;
}

// A package as Loom should open it. `load_path` is the package directory, or
// a private directory of links (removed by the owner) when parameter files
// had to be found by origin.
struct Resolved {
  std::string load_path;
  std::string preset;
  fs::path overlay;  // removed with this value (after Loom closes the package)
  // Held locked while the overlay is in use (see RemoveStale).
  platform::PlatformFile owner = platform::kInvalidFile;
  std::string note;
  // Said once at load, as a warning.
  std::string warning;

  Resolved() = default;
  Resolved(const Resolved&) = delete;
  Resolved& operator=(const Resolved&) = delete;
  Resolved(Resolved&& other) noexcept { *this = std::move(other); }
  Resolved& operator=(Resolved&& other) noexcept {
    if (this != &other) {
      Remove();
      load_path = std::move(other.load_path);
      preset = std::move(other.preset);
      overlay = std::exchange(other.overlay, {});
      owner = std::exchange(other.owner, platform::kInvalidFile);
      note = std::move(other.note);
      warning = std::move(other.warning);
    }
    return *this;
  }
  ~Resolved() { Remove(); }

private:
  void Remove() {
    if (platform::FileValid(owner)) {
      platform::CloseFile(owner);
      owner = platform::kInvalidFile;
    }
    if (!overlay.empty()) {
      std::error_code error;
      fs::remove_all(overlay, error);
      overlay.clear();
    }
  }
};

inline constexpr std::string_view kOwnerLock = ".owner";

// Opens and locks a private package directory's owner file; an invalid handle
// when another process holds it (or it cannot be opened).
inline platform::PlatformFile LockOwner(const fs::path& directory) {
  std::error_code error;
  platform::Directory dir;
  if (!dir.Open(directory, error))
    return platform::kInvalidFile;
  auto file = platform::OpenAt(dir, kOwnerLock,
                               platform::FileMode::kOpenOrCreate, error);
  if (!platform::FileValid(file))
    return platform::kInvalidFile;
  if (!platform::TryLockExclusive(file, error)) {
    platform::CloseFile(file);
    return platform::kInvalidFile;
  }
  return file;
}

// Removes private package directories (gufo_loom_*) whose process ended
// without closing its model (a crash, a kill): their owner lock is free. A
// directory of another user cannot be opened, so it is left alone.
inline void RemoveStale(const fs::path& temp) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(temp, error)) {
    const auto name = entry.path().filename().string();
    if (!name.starts_with("gufo_loom_") || entry.is_symlink(error) ||
        !entry.is_directory(error) ||
        !fs::exists(entry.path() / kOwnerLock, error))
      continue;
    const auto owner = LockOwner(entry.path());
    if (!platform::FileValid(owner))
      continue;
    platform::CloseFile(owner);
    fs::remove_all(entry.path(), error);
  }
}

// Links `at` to `target`. Windows refuses symbolic links without Developer
// Mode or administrator rights: files then get hard links (same volume) and
// directories become real ones holding links.
inline void Link(const fs::path& target, const fs::path& at) {
  fs::create_directories(at.parent_path());
  std::error_code error;
  const bool directory = fs::is_directory(target);
  if (directory)
    fs::create_directory_symlink(target, at, error);
  else
    fs::create_symlink(target, at, error);
  if (!error)
    return;
  if (directory) {
    fs::create_directory(at);
    for (const auto& entry : fs::directory_iterator(target))
      Link(entry.path(), at / entry.path().filename());
    return;
  }
  std::error_code hard;
  fs::create_hard_link(target, at, hard);
  if (!hard)
    return;
  throw std::runtime_error(
      "loom: cannot link " + target.string() + " at " + at.string() + ": " +
      error.message() +
      ". On Windows, enable Developer Mode (symbolic links), or set TMP to "
      "a folder on the model's drive (hard links)");
}

// Files given for a card group with Gufo's own options (--mmproj for
// "vision", the draft model for "draft"), each replacing that group's only
// file. The option names say which one, for errors.
struct Override {
  fs::path file;
  std::string option;
};

inline Resolved Resolve(const std::string& path,
                        const std::map<std::string, Override>& overrides,
                        const Gpu& gpu, std::string_view runtime_sdk = {}) {
  Resolved resolved;
  resolved.load_path = path;
  const fs::path root =
      fs::is_directory(path) ? fs::path(path) : fs::path(path).parent_path();
  CheckSdk(root, runtime_sdk);
  const auto card = json::parse(ReadText(root / "model.json"));
  resolved.preset = ChoosePreset(card, gpu, &resolved.note, &resolved.warning);
  std::vector<std::pair<std::string, fs::path>> links;
  std::vector<std::string> missing;
  std::error_code error;
  const auto files = ParameterFiles(card, resolved.preset);
  for (const auto& [group, given] : overrides) {
    const auto count = std::ranges::count(files, group, &ParameterFile::group);
    if (count != 1)
      throw std::runtime_error("loom: " + given.option +
                               " replaces the package's " + group +
                               " file, but this preset has " +
                               (count == 0 ? std::string("none")
                                           : std::to_string(count) + " files"));
    if (!fs::is_regular_file(given.file, error))
      throw std::runtime_error("loom: " + given.option + " " +
                               given.file.string() + " is not a file");
  }
  for (const auto& file : files) {
    if (!SafeRelative(file.path) ||
        (!file.origin_path.empty() && !SafeRelative(file.origin_path)))
      throw std::runtime_error(
          "loom: the card names an invalid parameter path '" + file.path + "'");
    if (const auto given = overrides.find(file.group);
        given != overrides.end()) {
      links.emplace_back(file.path, fs::absolute(given->second.file, error));
      continue;
    }
    if (fs::exists(root / file.path, error))
      continue;
    if (const auto found = Locate(file, root))
      links.emplace_back(file.path, fs::absolute(*found, error));
    else
      missing.push_back(file.path + (file.origin.empty()
                                         ? ""
                                         : " (@" + file.origin + "/" +
                                               file.origin_path + ")"));
  }
  if (!missing.empty()) {
    std::string list;
    for (const auto& item : missing)
      list += "\n  " + item;
    throw std::runtime_error(
        "loom: the package's parameter files are missing:" + list +
        "\nInstall the model with `gufo model install`, fetch the package with "
        "loom-fetch, or download each origin's files into the package "
        "directory at their repository paths.");
  }
  if (links.empty())
    return resolved;
  // A directory of links: the package's own entries, plus each found file at
  // its card path. Loom resolves every path the card names inside it.
  const fs::path temp = fs::temp_directory_path();
  RemoveStale(temp);
  for (int attempt = 0; attempt < 100 && resolved.overlay.empty(); ++attempt) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::array<std::uint8_t, 6> random{};
    if (!platform::FillRandomBytes(random))
      break;
    std::string name =
        "gufo_loom_" + std::to_string(platform::ProcessId()) + "_";
    for (const auto byte : random) {
      name += kHex[byte >> 4];
      name += kHex[byte & 15];
    }
    std::error_code created;
    if (fs::create_directory(temp / name, created))
      resolved.overlay =
          temp / name;  // from here on, `resolved` removes it on a throw
  }
  if (resolved.overlay.empty())
    throw std::runtime_error(
        "loom: cannot create a private package directory in " + temp.string());
  resolved.owner = LockOwner(resolved.overlay);
  if (!platform::FileValid(resolved.owner))
    throw std::runtime_error("loom: cannot lock " +
                             (resolved.overlay / kOwnerLock).string());
  const auto& link = Link;
  const auto absolute_root = fs::absolute(root);
  std::set<std::string> parents;  // first components of the card paths to place
  for (const auto& [card_path, target] : links)
    if (fs::path(card_path).has_parent_path())
      parents.insert(fs::path(card_path).begin()->string());
  for (const auto& entry : fs::directory_iterator(absolute_root)) {
    const auto name = entry.path().filename().string();
    if (!parents.contains(name)) {
      link(entry.path(), resolved.overlay / name);
      continue;
    }
    // A directory the card places files in: a real one here, holding links
    // to the package's own files in it.
    if (!fs::is_directory(entry.path()))
      throw std::runtime_error("loom: the package has a file '" + name +
                               "' where its card expects a directory");
    for (const auto& inner : fs::recursive_directory_iterator(
             entry.path(), fs::directory_options::follow_directory_symlink)) {
      if (!inner.is_directory())
        link(inner.path(), resolved.overlay / name /
                               fs::relative(inner.path(), entry.path()));
    }
  }
  for (const auto& [card_path, target] : links) {
    // An override replaces the package's own file there.
    const auto at = resolved.overlay / card_path;
    fs::remove(at, error);
    link(target, at);
  }
  resolved.load_path = resolved.overlay.string();
  return resolved;
}

}  // namespace package

}  // namespace gufo::loom

#endif  // GUFO_LOOM_PACKAGE_HPP_
