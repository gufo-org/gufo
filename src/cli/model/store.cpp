#include "src/cli/model/store.hpp"

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "src/core/json.hpp"
#include "src/core/platform/fs.hpp"

namespace gufo::cli::model_store {
namespace {
namespace fs = std::filesystem;

std::string ReadText(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.string());
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

// Links `target` to `blob`, copying where the file system has no hard links.
void Link(const fs::path& blob, const fs::path& target) {
  fs::create_directories(target.parent_path());
  std::error_code ec;
  fs::create_hard_link(blob, target, ec);
  if (ec)
    fs::copy_file(blob, target, fs::copy_options::overwrite_existing);
}
}  // namespace

const Component* Installed::Find(std::string_view role) const {
  for (const auto& component : components) {
    if (component.role == role)
      return &component;
  }
  return nullptr;
}

fs::path StoreDirectory() {
  if (const char* override_dir = std::getenv("GUFO_MODELS_DIR");
      override_dir != nullptr && *override_dir != '\0')
    return override_dir;
  const auto data = platform::UserDataDirectory();
  if (data.empty())
    throw std::runtime_error(
        "cannot determine the user data directory; set GUFO_MODELS_DIR");
  return data / "gufo" / "models";
}

std::vector<Installed> Read(const fs::path& store) {
  const auto path = store / "installed.json";
  std::vector<Installed> installed;
  if (!fs::exists(path))
    return installed;
  const auto state = json::parse(ReadText(path));
  if (state.member_size("schema") != 1)
    throw std::runtime_error(path.string() + ": unsupported schema");
  const auto* models = state.find("models");
  if (models == nullptr || !models->is_array())
    return installed;
  for (const auto& entry : models->items()) {
    Installed model;
    model.id = entry.member_str("id");
    model.extension = entry.member_str("extension");
    model.extension_version = entry.member_str("extension_version");
    if (!model_catalog::SafeRelativePath(model.id))
      throw std::runtime_error(path.string() + ": invalid model id");
    if (const auto* components = entry.find("components");
        components && components->is_array()) {
      for (const auto& item : components->items()) {
        Component component;
        component.role = item.member_str("role");
        component.option = item.member_str("option");
        component.repo = item.member_str("repo");
        component.revision = item.member_str("revision");
        component.entry = item.member_str("entry");
        if (const auto* files = item.find("files");
            files && files->is_array()) {
          for (const auto& file : files->items()) {
            component.files.push_back(
                {file.member_str("path"),
                 static_cast<std::uint64_t>(file.member_double("size")),
                 file.member_str("sha256")});
          }
        }
        model.components.push_back(std::move(component));
      }
    }
    installed.push_back(std::move(model));
  }
  return installed;
}

void Write(const fs::path& store, const std::vector<Installed>& installed) {
  auto models = json::Value::array();
  for (const auto& model : installed) {
    auto entry = json::Value::object();
    entry["id"] = model.id;
    entry["extension"] = model.extension;
    entry["extension_version"] = model.extension_version;
    auto components = json::Value::array();
    for (const auto& component : model.components) {
      auto item = json::Value::object();
      item["role"] = component.role;
      item["option"] = component.option;
      item["repo"] = component.repo;
      item["revision"] = component.revision;
      item["entry"] = component.entry;
      auto files = json::Value::array();
      for (const auto& file : component.files) {
        auto record = json::Value::object();
        record["path"] = file.path;
        record["size"] = file.size;
        record["sha256"] = file.sha256;
        files.push_back(std::move(record));
      }
      item["files"] = std::move(files);
      components.push_back(std::move(item));
    }
    entry["components"] = std::move(components);
    models.push_back(std::move(entry));
  }
  auto state = json::Value::object();
  state["schema"] = 1;
  state["models"] = std::move(models);
  fs::create_directories(store);
  const auto temporary = store / "installed.json.tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << state.dump() << "\n";
    if (!output)
      throw std::runtime_error("cannot write " + temporary.string());
  }
  fs::rename(temporary, store / "installed.json");
}

fs::path BlobPath(const fs::path& store, std::string_view sha256) {
  return store / "blobs" / std::string(sha256);
}

fs::path ModelDirectory(const fs::path& store, std::string_view id) {
  return store / std::string(id);
}

fs::path EntryPath(const fs::path& store, const Installed& model,
                   const Component& component) {
  const auto directory = ModelDirectory(store, model.id);
  if (component.entry == ".")
    return directory;
  // Lock paths use '/'; give commands the platform's separator.
  return (directory / fs::path(component.entry)).make_preferred();
}

void Materialize(const fs::path& store, const Installed& model) {
  const auto directory = ModelDirectory(store, model.id);
  auto staging = directory;
  staging += ".staging";
  fs::remove_all(staging);
  for (const auto& component : model.components) {
    for (const auto& file : component.files) {
      const auto target = staging / fs::path(file.path);
      if (!fs::exists(target))
        Link(BlobPath(store, file.sha256), target);
    }
  }
  // Running gufo processes keep the files they opened: replacing the
  // directory never truncates a blob.
  fs::remove_all(directory);
  fs::rename(staging, directory);
}

void CollectGarbage(const fs::path& store,
                    const std::vector<Installed>& installed) {
  std::set<std::string> referenced;
  for (const auto& model : installed) {
    for (const auto& component : model.components) {
      for (const auto& file : component.files)
        referenced.insert(file.sha256);
    }
  }
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(store / "blobs", ec)) {
    if (!referenced.contains(entry.path().filename().string()))
      fs::remove(entry.path(), ec);
  }
}

}  // namespace gufo::cli::model_store
