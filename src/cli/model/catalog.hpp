#pragma once
// The models that extensions offer for download. An extension declares each
// model's components in extension.json ("models") and pins every file's size
// and SHA-256 in models.lock.json beside it (tools/models/lock.py). Built-in
// extensions are embedded in the executable; installed extensions are read
// from their pinned source in the extension store.
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::cli::model_catalog {

struct File {
  std::string path;  // relative to the repository root, '/'-separated
  std::uint64_t size = 0;
  std::string sha256;
};

enum class Install { kRequired, kDefault, kOptional };

struct Component {
  std::string role;
  std::string option;  // the command option this component's path fills
  Install install = Install::kOptional;
  std::string repo;      // Hugging Face repository, <owner>/<name>
  std::string revision;  // 40-hex commit
  std::string entry;     // path passed to `option`; "." is the model directory
  std::vector<File> files;
};

struct Model {
  std::string id;
  std::string provider;
  std::string license;
  std::string description;
  std::string extension;
  std::string extension_version;
  std::vector<Component> components;

  const Component* Find(std::string_view role) const;
  // The component that `--model` takes.
  const Component& Primary() const;
};

// Validates one extension's models against its lock. Throws
// std::runtime_error naming the first problem.
std::vector<Model> Parse(const json::Value& models, const json::Value& lock,
                         const std::string& extension,
                         const std::string& version);

struct Catalog {
  std::vector<Model> models;
  // Extensions whose models could not be read, with the reason.
  std::vector<std::string> problems;

  const Model* Find(std::string_view id) const;
};

// The models of the built-in extensions and of the installed ones; an
// installed extension replaces a built-in of the same name.
Catalog Load();

// True when `path` is a safe relative path inside a model directory.
bool SafeRelativePath(std::string_view path);

std::string_view InstallName(Install install);

}  // namespace gufo::cli::model_catalog
