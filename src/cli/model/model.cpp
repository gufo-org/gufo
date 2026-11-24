#include "src/cli/model/model.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "src/cli/model/catalog.hpp"
#include "src/cli/model/download.hpp"
#include "src/cli/model/source.hpp"
#include "src/cli/model/store.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/publish.hpp"

namespace gufo::cli {
namespace {
namespace fs = std::filesystem;
namespace catalog = model_catalog;
namespace store = model_store;
namespace download = model_download;

constexpr std::string_view kUsage =
    "Usage: gufo model <COMMAND> [OPTIONS]\n\n"
    "Download and manage the model weights that extensions offer. Each\n"
    "extension pins its files by revision, size and SHA-256.\n\n"
    "Commands:\n"
    "  list [--json]        List every model, the extension that offers it\n"
    "                       and what is installed\n"
    "      --registry         Also list the models of reviewed registry\n"
    "                         extensions that are not installed\n"
    "  install <ID>         Download a model\n"
    "  install <OWNER>/<NAME>\n"
    "                       Download a model by its Hugging Face repository,\n"
    "                         e.g. loom-fun/<NAME> for any Loom package\n"
    "      --component ROLE   Install this optional component as well;\n"
    "                         repeat for several. Without it, the default\n"
    "                         components are installed\n"
    "  update [ID...]       Move installed models to the files their\n"
    "                       extension pins now\n"
    "  remove <ID>          Remove a model and the files no other model uses\n"
    "  verify [ID...]       Check installed files against their SHA-256\n\n"
    "Options for install and update:\n"
    "      --connections N  Parallel connections (default 16)\n\n"
    "A model installed by repository goes by that repository everywhere.\n"
    "Pass an installed model's ID or repository to --model to use it.\n"
    "Models live in\n"
    "$GUFO_MODELS_DIR, else $XDG_DATA_HOME/gufo/models. HF_TOKEN, or the\n"
    "token `hf auth login` saved, is sent to Hugging Face for gated models\n"
    "and higher rate limits; HF_ENDPOINT selects a mirror. Files already in\n"
    "the Hugging Face cache are reused.\n";

std::string Bytes(std::uint64_t bytes) {
  const double value = static_cast<double>(bytes);
  char text[32];
  if (value >= 1e9)
    std::snprintf(text, sizeof text, "%.1f GB", value / 1e9);
  else
    std::snprintf(text, sizeof text, "%.0f MB", value / 1e6);
  return text;
}

std::uint64_t Size(const std::vector<const catalog::Component*>& components) {
  std::uint64_t total = 0;
  for (const auto* component : components) {
    for (const auto& file : component->files)
      total += file.size;
  }
  return total;
}

std::vector<const catalog::Component*> Defaults(const catalog::Model& model) {
  std::vector<const catalog::Component*> selected;
  for (const auto& component : model.components) {
    if (component.install != catalog::Install::kOptional)
      selected.push_back(&component);
  }
  return selected;
}

// Hugging Face repository names ignore case.
bool SameRepository(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::ranges::equal(a, b, [](unsigned char x, unsigned char y) {
           return std::tolower(x) == std::tolower(y);
         });
}

// What users type for an installed model: the repository it was resolved
// from (`loom-fun/<name>`), else its ID.
const std::string& Name(const store::Installed& model) {
  return model.source.empty() ? model.id : model.source;
}

// The installed model `name` (an ID, or the repository it came from) names.
template<class Models>
auto* FindInstalled(Models& installed, std::string_view name) {
  const auto found = std::ranges::find_if(installed, [&](const auto& model) {
    return model.id == name ||
           (!model.source.empty() && SameRepository(model.source, name));
  });
  return found == installed.end() ? nullptr : &*found;
}

// The store IDs of `names`; throws for one that is not installed.
std::vector<std::string> InstalledIds(
    const std::vector<store::Installed>& installed,
    const std::vector<std::string>& names) {
  std::vector<std::string> ids;
  for (const auto& name : names) {
    const auto* model = FindInstalled(installed, name);
    if (model == nullptr)
      throw std::runtime_error("model " + name + " is not installed");
    ids.push_back(model->id);
  }
  return ids;
}

std::string Roles(const catalog::Model& model) {
  std::string roles;
  for (const auto& component : model.components)
    roles += (roles.empty() ? "" : ", ") + component.role + " (" +
             std::string(catalog::InstallName(component.install)) + ")";
  return roles;
}

const catalog::Model& Require(const catalog::Catalog& models,
                              std::string_view id) {
  if (const auto* model = models.Find(id))
    return *model;
  std::string message = "no extension offers a model '" + std::string(id) +
                        "'; see gufo model list";
  for (const auto& problem : models.problems)
    message += "\n  " + problem;
  throw std::runtime_error(message);
}

// True when the installed files are exactly the ones the extension pins.
bool Current(const store::Installed& installed, const catalog::Model& model) {
  for (const auto& component : installed.components) {
    const auto* pinned = model.Find(component.role);
    if (pinned == nullptr || pinned->entry != component.entry ||
        pinned->option != component.option ||
        pinned->files.size() != component.files.size())
      return false;
    for (std::size_t i = 0; i < pinned->files.size(); ++i) {
      if (pinned->files[i].path != component.files[i].path ||
          pinned->files[i].sha256 != component.files[i].sha256)
        return false;
    }
  }
  const auto primary = model.Primary().role;
  return std::ranges::any_of(installed.components,
                             [&](const auto& c) { return c.role == primary; });
}

// Downloads `components` of `model` and records them; the store is locked.
void Apply(const fs::path& directory, const catalog::Model& model,
           const std::vector<const catalog::Component*>& components,
           std::vector<store::Installed>& installed,
           const download::Options& options, const std::string& source = {}) {
  std::vector<download::Item> items;
  for (const auto* component : components) {
    for (const auto& file : component->files) {
      if (std::ranges::none_of(items, [&](const auto& item) {
            return item.sha256 == file.sha256;
          }))
        items.push_back({component->repo, component->revision, file.path,
                         file.size, file.sha256});
    }
  }
  download::Fetch(items, directory, options);
  store::Installed record;
  record.id = model.id;
  record.extension = model.extension;
  record.extension_version = model.extension_version;
  record.source = source;
  for (const auto* component : components)
    record.components.push_back({component->role, component->option,
                                 component->repo, component->revision,
                                 component->entry, component->files});
  store::Materialize(directory, record);
  std::erase_if(installed, [&](const auto& m) { return m.id == model.id; });
  installed.push_back(std::move(record));
  std::ranges::sort(installed, {}, &store::Installed::id);
  store::Write(directory, installed);
  store::CollectGarbage(directory, installed);
}

void PrintUsage(const fs::path& directory, const store::Installed& model) {
  std::cout << "Use it with --model " << Name(model) << "\n";
  // A resolved model's runner finds its other files in the model directory.
  if (!model.source.empty())
    return;
  for (const auto& component : model.components) {
    if (component.option != "--model")
      std::cout << "  " << component.option << " "
                << store::EntryPath(directory, model, component).string()
                << "\n";
  }
}

platform::FileLock LockStore(const fs::path& directory) {
  fs::create_directories(directory);
  return platform::FileLock::Acquire((directory / ".lock").string());
}

// Installs the model a source resolves `reference` to.
int InstallResolved(const std::string& reference,
                    const std::vector<std::string>& roles,
                    const download::Options& options) {
  if (!roles.empty())
    throw std::runtime_error(
        "--component applies to declared models; a resolved model installs "
        "every component");
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  auto installed = store::Read(directory);
  std::cout << "Resolving " << reference << "\n" << std::flush;
  const auto model = model_source::Resolve(reference, directory);
  if (const auto* existing = FindInstalled(installed, model.id);
      existing != nullptr && !SameRepository(existing->source, reference))
    throw std::runtime_error("model " + model.id +
                             " is already installed from another source; "
                             "remove it first");
  std::vector<const catalog::Component*> selected;
  for (const auto& component : model.components)
    selected.push_back(&component);
  std::cout << "Installing " << reference << " (" << model.extension << " "
            << model.extension_version
            << (model.license.empty() ? "" : ", license " + model.license)
            << "): ";
  for (std::size_t i = 0; i < selected.size(); ++i)
    std::cout << (i == 0 ? "" : ", ") << selected[i]->role << " "
              << selected[i]->repo << "@" << selected[i]->revision.substr(0, 8);
  std::cout << " (" << Bytes(Size(selected)) << ")\n" << std::flush;
  Apply(directory, model, selected, installed, options, reference);
  std::cout << "Installed " << reference << " in "
            << store::ModelDirectory(directory, model.id).string() << "\n";
  PrintUsage(directory, *FindInstalled(installed, model.id));
  return 0;
}

int Install(const std::string& id, const std::vector<std::string>& roles,
            const download::Options& options) {
  const auto models = catalog::Load();
  if (models.Find(id) == nullptr && model_source::HasSource(id))
    return InstallResolved(id, roles, options);
  const auto& model = Require(models, id);
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  auto installed = store::Read(directory);
  std::vector<const catalog::Component*> selected;
  for (const auto& component : model.components) {
    const bool asked = std::ranges::find(roles, component.role) != roles.end();
    const auto* existing = FindInstalled(installed, id);
    const bool kept = existing != nullptr && existing->Find(component.role);
    if (component.install == catalog::Install::kRequired || asked || kept ||
        (roles.empty() && component.install == catalog::Install::kDefault))
      selected.push_back(&component);
  }
  for (const auto& role : roles) {
    if (model.Find(role) == nullptr)
      throw std::runtime_error(id + " has no component " + role + "; it has " +
                               Roles(model));
  }
  std::cout << "Installing " << model.id << " from " << model.extension << " "
            << model.extension_version
            << (model.license.empty() ? "" : ", license " + model.license)
            << ": ";
  for (std::size_t i = 0; i < selected.size(); ++i)
    std::cout << (i == 0 ? "" : ", ") << selected[i]->role;
  std::cout << " (" << Bytes(Size(selected)) << ")\n" << std::flush;
  Apply(directory, model, selected, installed, options);
  std::cout << "Installed " << model.id << " in "
            << store::ModelDirectory(directory, model.id).string() << "\n";
  PrintUsage(directory, *FindInstalled(installed, id));
  return 0;
}

int Update(const std::vector<std::string>& names,
           const download::Options& options) {
  const auto models = catalog::Load();
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  auto installed = store::Read(directory);
  const auto ids = InstalledIds(installed, names);
  int status = 0;
  const auto snapshot = installed;
  for (const auto& current : snapshot) {
    if (!ids.empty() && std::ranges::find(ids, current.id) == ids.end())
      continue;
    if (!current.source.empty()) {
      // Like a declared model no extension offers any more, a repository that
      // cannot be resolved keeps its files while the others update.
      catalog::Model resolved;
      try {
        resolved = model_source::Resolve(current.source, directory);
      } catch (const std::exception& error) {
        std::cerr << current.source << ": " << error.what()
                  << "; keeping the installed files\n";
        status = 1;
        continue;
      }
      if (Current(current, resolved)) {
        std::cout << current.source << " is up to date\n";
        continue;
      }
      // The repository's card may now name another ID; it must not take over
      // a model installed from elsewhere.
      if (const auto* existing = FindInstalled(installed, resolved.id);
          resolved.id != current.id && existing != nullptr) {
        std::cerr << current.source << " now resolves to " << resolved.id
                  << ", which is already installed; keeping the installed "
                     "files\n";
        status = 1;
        continue;
      }
      std::vector<const catalog::Component*> selected;
      for (const auto& component : resolved.components)
        selected.push_back(&component);
      std::cout << "Updating " << current.source << "\n" << std::flush;
      Apply(directory, resolved, selected, installed, options, current.source);
      if (resolved.id != current.id) {
        // Apply recorded the new ID; the old record would shadow it.
        std::erase_if(installed,
                      [&](const auto& m) { return m.id == current.id; });
        store::Write(directory, installed);
        fs::remove_all(store::ModelDirectory(directory, current.id));
        store::CollectGarbage(directory, installed);
      }
      std::cout << "Updated " << current.source << "\n";
      continue;
    }
    const auto* model = models.Find(current.id);
    if (model == nullptr) {
      std::cerr << current.id << ": no loaded extension offers it any more ("
                << current.extension << " " << current.extension_version
                << "); keeping the installed files\n";
      status = 1;
      continue;
    }
    if (Current(current, *model)) {
      std::cout << current.id << " is up to date\n";
      continue;
    }
    std::vector<const catalog::Component*> selected;
    for (const auto& component : model->components) {
      if (component.install == catalog::Install::kRequired ||
          current.Find(component.role) != nullptr)
        selected.push_back(&component);
    }
    std::cout << "Updating " << current.id << " to " << model->extension << " "
              << model->extension_version << "\n"
              << std::flush;
    Apply(directory, *model, selected, installed, options);
    std::cout << "Updated " << current.id << "\n";
  }
  return status;
}

int Remove(const std::string& name) {
  const auto directory = store::StoreDirectory();
  const auto lock = LockStore(directory);
  auto installed = store::Read(directory);
  const auto* named = FindInstalled(installed, name);
  const std::string id = named != nullptr ? named->id : name;
  // Partial downloads of this model go too.
  std::size_t partials = 0;
  const auto models = catalog::Load();
  if (const auto* model = models.Find(id)) {
    std::error_code ec;
    for (const auto& component : model->components) {
      for (const auto& file : component.files) {
        for (const auto* suffix : {".part", ".progress"})
          partials +=
              fs::remove(directory / "downloads" / (file.sha256 + suffix), ec)
                  ? 1
                  : 0;
      }
    }
  }
  if (std::erase_if(installed, [&](const auto& m) { return m.id == id; }) ==
      0) {
    if (partials == 0)
      throw std::runtime_error("model " + id + " is not installed");
    std::cout << "Removed the partial download of " << id << "\n";
    return 0;
  }
  store::Write(directory, installed);
  fs::remove_all(store::ModelDirectory(directory, id));
  store::CollectGarbage(directory, installed);
  std::cout << "Removed " << name
            << ". Running gufo processes keep its files until they exit.\n";
  return 0;
}

int Verify(const std::vector<std::string>& names) {
  const auto directory = store::StoreDirectory();
  auto installed = store::Read(directory);
  const auto ids = InstalledIds(installed, names);
  int status = 0;
  for (const auto& model : installed) {
    if (!ids.empty() && std::ranges::find(ids, model.id) == ids.end())
      continue;
    std::size_t bad = 0;
    for (const auto& component : model.components) {
      for (const auto& file : component.files) {
        const auto path =
            store::ModelDirectory(directory, model.id) / fs::path(file.path);
        std::error_code ec;
        std::string problem;
        if (!fs::is_regular_file(path, ec))
          problem = "missing";
        else if (fs::file_size(path, ec) != file.size)
          problem = "wrong size";
        else if (crypto::Sha256FileHex(path) != file.sha256)
          problem = "SHA-256 mismatch";
        if (!problem.empty()) {
          std::cout << Name(model) << ": " << file.path << ": " << problem
                    << "\n";
          // Install trusts the blobs it finds: drop the damaged one.
          fs::remove(store::BlobPath(directory, file.sha256), ec);
          ++bad;
        }
      }
    }
    if (bad == 0)
      std::cout << Name(model) << ": ok\n";
    else
      status = 1;
  }
  if (status != 0)
    std::cout << "Run gufo model install <ID> again to repair a model.\n";
  return status;
}

int List(bool json_mode, bool registry) {
  const auto models = catalog::Load(registry);
  const auto directory = store::StoreDirectory();
  const auto installed = store::Read(directory);
  struct Row {
    std::string id;
    std::string extension;
    std::string description;
    std::uint64_t size = 0;
    std::string status;
    const catalog::Model* model = nullptr;
    const store::Installed* installed = nullptr;
  };
  std::vector<Row> rows;
  for (const auto& model : models.models) {
    Row row{model.id,          model.extension,
            model.description, Size(Defaults(model)),
            "available",       &model};
    if (model.extension_status == "not installed")
      row.status = "extension not installed";
    const auto found =
        std::ranges::find(installed, model.id, &store::Installed::id);
    if (found != installed.end()) {
      row.installed = &*found;
      row.status = model.components.empty() || Current(*found, model)
                       ? "installed"
                       : "update available";
    }
    rows.push_back(std::move(row));
  }
  for (const auto& model : installed) {
    if (models.Find(model.id) != nullptr)
      continue;
    if (!model.source.empty())
      rows.push_back(
          {model.source, model.extension, "", 0, "installed", nullptr, &model});
    else
      rows.push_back({model.id, model.extension, "", 0,
                      "installed, no loaded extension offers it", nullptr,
                      &model});
  }
  std::ranges::sort(rows, {}, &Row::id);

  if (json_mode) {
    auto report = json::Value::object();
    report["store"] = directory.string();
    auto problems = json::Value::array();
    for (const auto& problem : models.problems)
      problems.push_back(problem);
    report["problems"] = std::move(problems);
    auto entries = json::Value::array();
    for (const auto& row : rows) {
      auto entry = json::Value::object();
      entry["id"] = row.id;
      entry["extension"] = row.extension;
      entry["description"] = row.description;
      entry["status"] = row.status;
      entry["downloadable"] =
          row.model != nullptr && !row.model->components.empty();
      entry["extensionStatus"] =
          row.model != nullptr ? row.model->extension_status : "not installed";
      if (row.model != nullptr) {
        entry["extensionVersion"] = row.model->extension_version;
        entry["provider"] = row.model->provider;
        entry["license"] = row.model->license;
        entry["defaultBytes"] = row.size;
        auto components = json::Value::array();
        for (const auto& component : row.model->components) {
          auto item = json::Value::object();
          item["role"] = component.role;
          item["option"] = component.option;
          item["install"] =
              std::string(catalog::InstallName(component.install));
          item["source"] = component.repo + "@" + component.revision;
          item["bytes"] = Size({&component});
          item["installed"] = row.installed != nullptr &&
                              row.installed->Find(component.role) != nullptr;
          components.push_back(std::move(item));
        }
        entry["components"] = std::move(components);
      }
      if (row.installed != nullptr) {
        for (const auto& component : row.installed->components) {
          if (component.option == "--model")
            entry["path"] =
                store::EntryPath(directory, *row.installed, component).string();
        }
      }
      entries.push_back(std::move(entry));
    }
    report["models"] = std::move(entries);
    auto sources = json::Value::array();
    for (const auto& source : model_source::Sources()) {
      auto entry = json::Value::object();
      entry["owner"] = source.owner;
      entry["extension"] = source.extension;
      sources.push_back(std::move(entry));
    }
    report["sources"] = std::move(sources);
    std::cout << report.dump() << "\n";
    return 0;
  }
  for (const auto& problem : models.problems)
    std::cerr << "Warning: " << problem << "\n";
  std::cout << "Models (installed in " << directory.string() << "):\n";
  for (auto& row : rows) {
    if (row.installed != nullptr && row.model != nullptr) {
      std::string roles;
      for (const auto& component : row.installed->components)
        roles += (roles.empty() ? "" : ", ") + component.role;
      row.status += " (" + roles + ")";
    }
  }
  std::size_t id_width = 2;
  std::size_t extension_width = 9;
  std::size_t status_width = 6;
  for (const auto& row : rows) {
    id_width = std::max(id_width, row.id.size());
    extension_width = std::max(extension_width, row.extension.size());
    status_width = std::max(status_width, row.status.size());
  }
  const auto pad = [](const std::string& text, std::size_t width) {
    return text + std::string(width - std::min(width, text.size()) + 2, ' ');
  };
  std::cout << "  " << pad("ID", id_width) << pad("EXTENSION", extension_width)
            << pad("SIZE", 8) << pad("STATUS", status_width) << "DESCRIPTION\n";
  for (const auto& row : rows) {
    std::cout << "  " << pad(row.id, id_width)
              << pad(row.extension, extension_width)
              << pad(row.size > 0 ? Bytes(row.size) : "", 8)
              << pad(row.status, status_width) << row.description << "\n";
  }
  if (std::ranges::any_of(rows, [](const Row& row) {
        return row.model != nullptr &&
               row.model->extension_status == "not installed";
      }))
    std::cout << "\nInstall an extension with gufo extension install "
                 "<extension>. Models\nwithout a size pin no download; pass "
                 "their weights to --model.\n";
  for (const auto& source : model_source::Sources())
    std::cout << "\nAny " << source.extension << " model installs by its "
              << "repository: gufo model install " << source.owner
              << "/<NAME>\n(https://huggingface.co/" << source.owner << ")\n";
  return 0;
}

std::size_t ParseConnections(std::string_view text) {
  std::size_t value = 0;
  for (const char c : text) {
    if (c < '0' || c > '9' || value > 1000)
      throw std::runtime_error("--connections takes a number from 1 to 64");
    value = value * 10 + static_cast<std::size_t>(c - '0');
  }
  if (value < 1 || value > 64)
    throw std::runtime_error("--connections takes a number from 1 to 64");
  return value;
}
}  // namespace

int RunModel(std::span<const char* const> args) {
  if (args.empty() || std::string_view(args[0]) == "--help" ||
      std::string_view(args[0]) == "-h") {
    std::cout << kUsage;
    return args.empty() ? 2 : 0;
  }
  const std::string_view command = args[0];
  const auto rest = args.subspan(1);
  try {
    download::Options options;
    std::vector<std::string> positional;
    std::vector<std::string> roles;
    bool json_mode = false;
    bool registry = false;
    for (std::size_t i = 0; i < rest.size(); ++i) {
      const std::string_view arg = rest[i];
      const bool has_value = i + 1 < rest.size();
      if (arg == "--json" && command == "list")
        json_mode = true;
      else if (arg == "--registry" && command == "list")
        registry = true;
      else if (arg == "--component" && command == "install" && has_value)
        roles.emplace_back(rest[++i]);
      else if (arg == "--connections" &&
               (command == "install" || command == "update") && has_value)
        options.connections = ParseConnections(rest[++i]);
      else if (arg.starts_with("-"))
        throw std::runtime_error("unexpected argument " + std::string(arg) +
                                 "; see gufo model --help");
      else
        positional.emplace_back(arg);
    }
    if (command == "list") {
      if (!positional.empty())
        throw std::runtime_error(
            "usage: gufo model list [--json] [--registry]");
      return List(json_mode, registry);
    }
    if (command == "install") {
      if (positional.size() != 1)
        throw std::runtime_error("usage: gufo model install <ID>");
      return Install(positional[0], roles, options);
    }
    if (command == "update")
      return Update(positional, options);
    if (command == "remove") {
      if (positional.size() != 1)
        throw std::runtime_error("usage: gufo model remove <ID>");
      return Remove(positional[0]);
    }
    if (command == "verify")
      return Verify(positional);
    throw std::runtime_error("unknown model command " + std::string(command));
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}

std::vector<std::string> ResolveModelArguments(
    std::span<const char* const> args) {
  std::vector<std::string> resolved(args.begin(), args.end());
  if (resolved.size() < 3 || resolved[1] == "model" ||
      resolved[1] == "extension" || resolved[1] == "help")
    return resolved;
  static const std::regex id_pattern("^[a-z0-9][a-z0-9._-]*$");
  std::optional<std::vector<store::Installed>> installed;
  std::optional<fs::path> directory;
  for (std::size_t i = 2; i < resolved.size(); ++i) {
    std::string* value = nullptr;
    std::string prefix;
    if ((resolved[i] == "--model" || resolved[i] == "-m") &&
        i + 1 < resolved.size()) {
      value = &resolved[++i];
    } else if (resolved[i].starts_with("--model=")) {
      prefix = "--model=";
      value = &resolved[i];
    } else {
      continue;
    }
    auto id = value->substr(prefix.size());
    std::error_code ec;
    if (fs::exists(id, ec))
      continue;
    const bool reference =
        id.find('/') != std::string::npos && model_source::HasSource(id);
    if (!reference && !std::regex_match(id, id_pattern))
      continue;
    if (!installed) {
      try {
        directory = store::StoreDirectory();
        installed = store::Read(*directory);
      } catch (const std::exception&) {
        installed.emplace();
      }
    }
    const auto* found = FindInstalled(*installed, id);
    if (found != nullptr) {
      for (const auto& component : found->components) {
        if (component.option == "--model")
          *value =
              prefix + store::EntryPath(*directory, *found, component).string();
      }
      continue;
    }
    if (reference || catalog::Load().Find(id) != nullptr)
      throw std::runtime_error(
          "model " + id + " is not installed; run gufo model install " + id);
  }
  return resolved;
}

}  // namespace gufo::cli
