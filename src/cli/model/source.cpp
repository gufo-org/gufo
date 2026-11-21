#include "src/cli/model/source.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "gufo/ext/model_source.hpp"
#include "gufo/ext/registry.hpp"
#include "src/cli/model/download.hpp"
#include "src/cli/model/store.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/json.hpp"

namespace gufo::cli::model_source {
namespace {
namespace fs = std::filesystem;
namespace catalog = model_catalog;
namespace download = model_download;

// The built-in extensions' sources, registered once per process.
const ext::Registry& Registry() {
  static const ext::Registry registry = [] {
    ext::Registry built;
    ext::RegisterBuiltExtensions(built);
    return built;
  }();
  return registry;
}

const ext::ModelSource* Find(std::string_view reference) {
  static const std::regex pattern(
      "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$");
  if (!std::regex_match(reference.begin(), reference.end(), pattern))
    return nullptr;
  (void)Registry();
  return ext::FindModelSource(reference.substr(0, reference.find('/')));
}

std::string Hex(std::span<const unsigned char> bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  for (const auto byte : bytes) {
    text += kHex[byte >> 4];
    text += kHex[byte & 15];
  }
  return text;
}

// The Git object ID of a file's content, as the Hub reports a non-LFS file.
std::string GitBlobId(std::string_view content) {
  const auto header =
      "blob " + std::to_string(content.size()) + std::string(1, '\0');
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha1(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), header.data(), header.size()) != 1 ||
      EVP_DigestUpdate(context.get(), content.data(), content.size()) != 1 ||
      EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1)
    throw std::runtime_error("cannot hash a downloaded file");
  return Hex(std::span(digest.data(), size));
}

bool Match(std::string_view pattern, std::string_view text) {
  if (pattern.empty())
    return text.empty();
  if (pattern.front() == '*') {
    for (std::size_t skip = 0; skip <= text.size(); ++skip) {
      if (skip > 0 && text[skip - 1] == '/')
        return false;
      if (Match(pattern.substr(1), text.substr(skip)))
        return true;
    }
    return false;
  }
  return !text.empty() && pattern.front() == text.front() &&
         Match(pattern.substr(1), text.substr(1));
}

// The component's files at its revision, with their sizes and SHA-256. Small
// files that are not in LFS are downloaded into the store's blobs here.
std::vector<catalog::File> Files(const ext::ModelSourceComponent& component,
                                 const fs::path& store) {
  const auto endpoint = download::Endpoint();
  const auto tree = json::parse(
      download::Get(endpoint + "/api/models/" + component.repo + "/tree/" +
                    component.revision + "?recursive=true"));
  if (!tree.is_array())
    throw std::runtime_error("cannot list " + component.repo);
  std::vector<catalog::File> files;
  for (const auto& entry : tree.items()) {
    if (entry.member_str("type") != "file")
      continue;
    const auto path = entry.member_str("path");
    const bool wanted =
        component.include.empty()
            ? std::ranges::none_of(
                  component.exclude,
                  [&](const auto& pattern) { return Matches(pattern, path); })
            : std::ranges::find(component.include, path) !=
                  component.include.end();
    if (!wanted)
      continue;
    if (!catalog::SafeRelativePath(path))
      throw std::runtime_error(component.repo + " has an invalid path " + path);
    const auto size = static_cast<std::uint64_t>(entry.member_double("size"));
    if (const auto* lfs = entry.find("lfs"); lfs && lfs->is_object()) {
      const auto sha256 = lfs->member_str("oid");
      if (sha256.size() != 64)
        throw std::runtime_error(component.repo + ": no SHA-256 for " + path);
      files.push_back({path, size, sha256});
      continue;
    }
    const auto content =
        download::Get(endpoint + "/" + component.repo + "/resolve/" +
                          component.revision + "/" + download::EscapePath(path),
                      size);
    if (content.size() != size || GitBlobId(content) != entry.member_str("oid"))
      throw std::runtime_error(component.repo + ": " + path +
                               " does not match its Git object ID");
    const auto sha256 = crypto::Sha256Hex(std::span(
        reinterpret_cast<const std::uint8_t*>(content.data()), content.size()));
    const auto blob = model_store::BlobPath(store, sha256);
    std::error_code ec;
    if (!fs::is_regular_file(blob, ec)) {
      fs::create_directories(blob.parent_path());
      auto temporary = blob;
      temporary += ".tmp";
      {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(content.data(),
                     static_cast<std::streamsize>(content.size()));
        if (!output)
          throw std::runtime_error("cannot write " + temporary.string());
      }
      fs::rename(temporary, blob);
    }
    files.push_back({path, size, sha256});
  }
  if (!component.include.empty() && files.size() != component.include.size())
    throw std::runtime_error(component.repo + "@" +
                             component.revision.substr(0, 8) +
                             " lacks files the package names");
  if (files.empty())
    throw std::runtime_error(component.repo + " has no files to install");
  return files;
}

}  // namespace

bool HasSource(std::string_view reference) {
  return Find(reference) != nullptr;
}

std::vector<Source> Sources() {
  const auto& registry = Registry();
  std::vector<Source> sources;
  for (const auto& source : ext::ModelSources()) {
    Source entry{source.owner, {}};
    for (const auto& provider : registry.Providers()) {
      if (provider.id == source.provider)
        entry.extension = provider.extension;
    }
    sources.push_back(std::move(entry));
  }
  std::ranges::sort(sources, {}, &Source::owner);
  return sources;
}

bool Matches(std::string_view pattern, std::string_view path) {
  const auto slash = path.rfind('/');
  return Match(pattern, path) || (slash != std::string_view::npos &&
                                  Match(pattern, path.substr(slash + 1)));
}

catalog::Model Resolve(std::string_view reference, const fs::path& store) {
  const auto* source = Find(reference);
  if (source == nullptr || !source->resolve)
    throw std::runtime_error("no extension resolves " + std::string(reference));
  const auto endpoint = download::Endpoint();
  const ext::ModelSourceHub hub{
      .api =
          [&](std::string_view path) {
            return download::Get(endpoint + "/api/" + std::string(path));
          },
      .file =
          [&](std::string_view repo, std::string_view revision,
              std::string_view path) {
            return download::Get(endpoint + "/" + std::string(repo) +
                                 "/resolve/" + std::string(revision) + "/" +
                                 download::EscapePath(path));
          },
  };
  const auto resolved = source->resolve(reference, hub);
  static const std::regex id_pattern("^[a-z0-9][a-z0-9._-]*$");
  static const std::regex revision_pattern("^[0-9a-f]{40}$");
  if (!std::regex_match(resolved.id, id_pattern))
    throw std::runtime_error(std::string(reference) +
                             " resolved to an invalid ID '" + resolved.id +
                             "'");
  catalog::Model model;
  model.id = resolved.id;
  model.provider = source->provider;
  model.license = resolved.license;
  model.description = resolved.description;
  for (const auto& provider : Registry().Providers()) {
    if (provider.id == source->provider)
      model.extension = provider.extension;
  }
  for (const auto& extension : Registry().Extensions()) {
    if (extension.name == model.extension)
      model.extension_version = extension.version;
  }
  model.extension_status = "built-in";
  std::size_t primary = 0;
  for (const auto& component : resolved.components) {
    if (!std::regex_match(component.revision, revision_pattern) ||
        component.repo.find('/') == std::string::npos ||
        (component.entry != "." && !catalog::SafeRelativePath(component.entry)))
      throw std::runtime_error(std::string(reference) + ": invalid component " +
                               component.role);
    primary += component.option == "--model" ? 1 : 0;
    model.components.push_back({.role = component.role,
                                .option = component.option,
                                .install = catalog::Install::kRequired,
                                .repo = component.repo,
                                .revision = component.revision,
                                .entry = component.entry,
                                .files = Files(component, store)});
  }
  if (primary != 1)
    throw std::runtime_error(std::string(reference) +
                             " must resolve to exactly one --model component");
  return model;
}

}  // namespace gufo::cli::model_source
