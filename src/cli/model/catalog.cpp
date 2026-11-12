#include "src/cli/model/catalog.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "src/cli/extension/store.hpp"

namespace gufo::cli {
std::string_view
BuiltinModelCatalog();  // generated with the extension registry
}

namespace gufo::cli::model_catalog {
namespace {
namespace fs = std::filesystem;

const std::regex& IdPattern() {
  static const std::regex pattern("^[a-z0-9][a-z0-9._-]*$");
  return pattern;
}
const std::regex& RolePattern() {
  static const std::regex pattern("^[a-z][a-z0-9_-]*$");
  return pattern;
}
const std::regex& OptionPattern() {
  static const std::regex pattern("^--[a-z][a-z0-9-]*$");
  return pattern;
}
const std::regex& RepoPattern() {
  static const std::regex pattern(
      "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$");
  return pattern;
}
const std::regex& RevisionPattern() {
  static const std::regex pattern("^[0-9a-f]{40}$");
  return pattern;
}
const std::regex& Sha256Pattern() {
  static const std::regex pattern("^[0-9a-f]{64}$");
  return pattern;
}

std::vector<std::string> Strings(const json::Value& object,
                                 const std::string& key) {
  std::vector<std::string> values;
  const auto* array = object.find(key);
  if (array == nullptr)
    return values;
  if (!array->is_array())
    throw std::runtime_error(key + " must be an array of strings");
  for (const auto& item : array->items()) {
    if (!item.is_string())
      throw std::runtime_error(key + " must be an array of strings");
    values.push_back(item.str());
  }
  return values;
}

Install ParseInstall(const std::string& text) {
  if (text == "required")
    return Install::kRequired;
  if (text == "default")
    return Install::kDefault;
  if (text == "optional")
    return Install::kOptional;
  throw std::runtime_error("install must be required, default or optional");
}

std::string ReadText(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

Component ParseComponent(const json::Value& declared, const json::Value* lock) {
  Component component;
  component.role = declared.member_str("role");
  if (!std::regex_match(component.role, RolePattern()))
    throw std::runtime_error("invalid component role '" + component.role + "'");
  const auto where = "component " + component.role + ": ";
  component.option = declared.member_str("option");
  if (!std::regex_match(component.option, OptionPattern()))
    throw std::runtime_error(where + "invalid option '" + component.option +
                             "'");
  component.install = ParseInstall(declared.member_str("install"));
  const auto* source = declared.find("source");
  if (source == nullptr || !source->is_object())
    throw std::runtime_error(where + "missing source");
  component.repo = source->member_str("hf");
  component.revision = source->member_str("revision");
  if (!std::regex_match(component.repo, RepoPattern()))
    throw std::runtime_error(where + "source.hf must be <owner>/<name>");
  if (!std::regex_match(component.revision, RevisionPattern()))
    throw std::runtime_error(where + "source.revision must be a 40-hex commit");
  if (lock == nullptr || !lock->is_object())
    throw std::runtime_error(where +
                             "not in models.lock.json; run "
                             "tools/models/lock.py");
  // The lock must describe exactly this source and selection.
  const auto* locked_source = lock->find("source");
  if (locked_source == nullptr ||
      locked_source->member_str("hf") != component.repo ||
      locked_source->member_str("revision") != component.revision ||
      Strings(*lock, "include") != Strings(declared, "include") ||
      Strings(*lock, "exclude") != Strings(declared, "exclude"))
    throw std::runtime_error(where +
                             "models.lock.json is stale; run "
                             "tools/models/lock.py");
  const auto* files = lock->find("files");
  if (files == nullptr || !files->is_array() || files->items().empty())
    throw std::runtime_error(where + "the lock lists no files");
  for (const auto& item : files->items()) {
    File file;
    file.path = item.member_str("path");
    file.sha256 = item.member_str("sha256");
    const auto* size = item.find("size");
    if (!SafeRelativePath(file.path))
      throw std::runtime_error(where + "unsafe file path '" + file.path + "'");
    if (!std::regex_match(file.sha256, Sha256Pattern()))
      throw std::runtime_error(where + file.path + " has no SHA-256");
    if (size == nullptr || !size->is_number() || size->as_double(-1) < 0)
      throw std::runtime_error(where + file.path + " has no size");
    file.size = static_cast<std::uint64_t>(size->as_double());
    component.files.push_back(std::move(file));
  }
  component.entry = declared.member_str("entry");
  if (component.entry.empty()) {
    if (component.files.size() != 1)
      throw std::runtime_error(where +
                               "several files: name the entry, or \".\" for "
                               "the model directory");
    component.entry = component.files.front().path;
  } else if (component.entry != "." &&
             std::ranges::none_of(component.files, [&](const File& file) {
               return file.path == component.entry;
             })) {
    throw std::runtime_error(where + "entry " + component.entry +
                             " is not one of its files");
  }
  return component;
}

Model ParseModel(const json::Value& declared, const json::Value& lock) {
  Model model;
  model.id = declared.member_str("id");
  if (!std::regex_match(model.id, IdPattern()) ||
      model.id.find("..") != std::string::npos)
    throw std::runtime_error("invalid model id '" + model.id + "'");
  model.provider = declared.member_str("provider");
  model.license = declared.member_str("license");
  model.description = declared.member_str("description");
  if (model.provider.empty())
    throw std::runtime_error("model " + model.id + ": missing provider");
  const auto* components = declared.find("components");
  if (components == nullptr || !components->is_array() ||
      components->items().empty())
    throw std::runtime_error("model " + model.id + ": no components");
  const auto* locked = lock.is_object() ? lock.find(model.id) : nullptr;
  int primary = 0;
  for (const auto& item : components->items()) {
    const auto* role_lock = locked != nullptr && locked->is_object()
                                ? locked->find(item.member_str("role"))
                                : nullptr;
    try {
      auto component = ParseComponent(item, role_lock);
      if (model.Find(component.role) != nullptr)
        throw std::runtime_error("duplicate component " + component.role);
      if (component.option == "--model") {
        if (component.install != Install::kRequired)
          throw std::runtime_error("the --model component must be required");
        ++primary;
      }
      model.components.push_back(std::move(component));
    } catch (const std::exception& error) {
      throw std::runtime_error("model " + model.id + ": " + error.what());
    }
  }
  if (primary != 1)
    throw std::runtime_error("model " + model.id +
                             ": exactly one component must fill --model");
  // Components share the model directory: one path, one file.
  std::vector<const File*> seen;
  for (const auto& component : model.components) {
    for (const auto& file : component.files) {
      for (const auto* other : seen) {
        if (other->path == file.path && other->sha256 != file.sha256)
          throw std::runtime_error("model " + model.id + ": two files at " +
                                   file.path);
      }
      seen.push_back(&file);
    }
  }
  return model;
}

// Adds one extension's models; a broken extension becomes a problem.
void Add(Catalog& catalog, const json::Value& models, const json::Value& lock,
         const std::string& extension, const std::string& version) {
  try {
    for (auto& model : Parse(models, lock, extension, version)) {
      if (const auto* other = catalog.Find(model.id)) {
        catalog.problems.push_back(extension + ": model " + model.id +
                                   " is already provided by " +
                                   other->extension);
        continue;
      }
      catalog.models.push_back(std::move(model));
    }
  } catch (const std::exception& error) {
    catalog.problems.push_back(extension + ": " + error.what());
  }
}
}  // namespace

const Component* Model::Find(std::string_view role) const {
  for (const auto& component : components) {
    if (component.role == role)
      return &component;
  }
  return nullptr;
}

const Component& Model::Primary() const {
  for (const auto& component : components) {
    if (component.option == "--model")
      return component;
  }
  throw std::logic_error("model without a --model component");
}

const Model* Catalog::Find(std::string_view id) const {
  for (const auto& model : models) {
    if (model.id == id)
      return &model;
  }
  return nullptr;
}

std::vector<Model> Parse(const json::Value& models, const json::Value& lock,
                         const std::string& extension,
                         const std::string& version) {
  std::vector<Model> parsed;
  if (!models.is_array())
    throw std::runtime_error("models must be an array");
  for (const auto& item : models.items()) {
    auto model = ParseModel(item, lock);
    if (std::ranges::any_of(
            parsed, [&](const Model& other) { return other.id == model.id; }))
      throw std::runtime_error("duplicate model " + model.id);
    model.extension = extension;
    model.extension_version = version;
    parsed.push_back(std::move(model));
  }
  return parsed;
}

Catalog Load() {
  Catalog catalog;
  std::vector<extension_store::Installed> installed;
  fs::path store;
  try {
    store = extension_store::StoreDirectory();
    installed = extension_store::Read(store);
  } catch (const std::exception& error) {
    catalog.problems.push_back(std::string("installed extensions: ") +
                               error.what());
  }
  // Installed extensions first: they replace built-ins of the same name.
  for (const auto& extension : installed) {
    try {
      const auto source =
          extension_store::SourceDirectory(store, extension.name);
      const auto manifest = extension_store::Manifest(source);
      const auto* models = manifest.find("models");
      if (models == nullptr)
        continue;
      const auto lock_path = source / "models.lock.json";
      const auto lock = fs::exists(lock_path) ? json::parse(ReadText(lock_path))
                                              : json::Value::object();
      const auto* locked = lock.find("models");
      Add(catalog, *models, locked ? *locked : json::Value::object(),
          extension.name, extension.version);
    } catch (const std::exception& error) {
      catalog.problems.push_back(extension.name + ": " + error.what());
    }
  }
  const auto builtin = json::parse(BuiltinModelCatalog());
  for (const auto& entry : builtin.items()) {
    const auto name = entry.member_str("name");
    if (std::ranges::any_of(installed, [&](const auto& extension) {
          return extension.name == name;
        }))
      continue;
    const auto* models = entry.find("models");
    const auto* lock = entry.find("lock");
    const auto* locked =
        lock != nullptr && lock->is_object() ? lock->find("models") : nullptr;
    Add(catalog, models ? *models : json::Value::array(),
        locked ? *locked : json::Value::object(), name,
        entry.member_str("version"));
  }
  std::ranges::sort(catalog.models, {}, &Model::id);
  return catalog;
}

bool SafeRelativePath(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.back() == '/' ||
      path.find('\\') != std::string_view::npos ||
      path.find(':') != std::string_view::npos ||
      path.find('\0') != std::string_view::npos)
    return false;
  std::size_t start = 0;
  while (start <= path.size()) {
    const auto end = std::min(path.find('/', start), path.size());
    const auto segment = path.substr(start, end - start);
    if (segment.empty() || segment == "." || segment == "..")
      return false;
    start = end + 1;
  }
  return true;
}

std::string_view InstallName(Install install) {
  switch (install) {
    case Install::kRequired:
      return "required";
    case Install::kDefault:
      return "default";
    case Install::kOptional:
      return "optional";
  }
  return "optional";
}

}  // namespace gufo::cli::model_catalog
