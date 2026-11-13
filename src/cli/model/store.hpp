#pragma once
// Installed model weights. The store lives in $GUFO_MODELS_DIR, else
// <user data>/gufo/models:
//
//   installed.json        state, written atomically
//   blobs/<sha256>        each file once, named by its content
//   <id>/<path>           the model directory: hard links to blobs, laid out
//                         as in the source repository
//   downloads/            partial downloads, resumed by the next install
//
// Models that share a file share its blob; a blob no installed model lists
// is deleted.
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/model/catalog.hpp"

namespace gufo::cli::model_store {

struct Component {
  std::string role;
  std::string option;
  std::string repo;
  std::string revision;
  std::string entry;
  std::vector<model_catalog::File> files;
};

struct Installed {
  std::string id;
  std::string extension;
  std::string extension_version;
  std::vector<Component> components;

  const Component* Find(std::string_view role) const;
};

std::filesystem::path StoreDirectory();
std::vector<Installed> Read(const std::filesystem::path& store);
void Write(const std::filesystem::path& store,
           const std::vector<Installed>& installed);

std::filesystem::path BlobPath(const std::filesystem::path& store,
                               std::string_view sha256);
std::filesystem::path ModelDirectory(const std::filesystem::path& store,
                                     std::string_view id);
// The path a component's option takes.
std::filesystem::path EntryPath(const std::filesystem::path& store,
                                const Installed& model,
                                const Component& component);

// Lays out the model directory from blobs, replacing the previous one.
void Materialize(const std::filesystem::path& store, const Installed& model);

// Deletes the blobs no installed model lists.
void CollectGarbage(const std::filesystem::path& store,
                    const std::vector<Installed>& installed);

}  // namespace gufo::cli::model_store
