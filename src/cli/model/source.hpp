#pragma once
// Models resolved when they are installed. An extension registers a model
// source for a Hugging Face owner (ext::ModelSource); `gufo model install
// <owner>/<name>` then asks it for the model's components and pins each
// file's size and SHA-256 from the Hub: LFS files by their recorded SHA-256,
// other files by downloading them and checking their Git object ID.
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/model/catalog.hpp"

namespace gufo::cli::model_source {

// True when `reference` is `<owner>/<name>` and a source is registered for
// the owner.
bool HasSource(std::string_view reference);

struct Source {
  std::string owner;      // Hugging Face owner, e.g. loom-fun
  std::string extension;  // the extension that registered it
};
// Every registered source, by owner.
std::vector<Source> Sources();

// The model `reference` resolves to, with every component required. Files
// that are not in LFS are stored as blobs of `store` while their SHA-256 is
// computed. Throws std::runtime_error.
model_catalog::Model Resolve(std::string_view reference,
                             const std::filesystem::path& store);

// True when `path` (a repository path) or its file name matches `pattern`,
// in which `*` matches any run of characters within one path part.
bool Matches(std::string_view pattern, std::string_view path);

}  // namespace gufo::cli::model_source
