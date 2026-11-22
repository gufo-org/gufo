// Every loom-fun package as a Gufo model, resolved when it is installed:
// `gufo model install loom-fun/<repo>`. The package revision built for this
// runtime's Loom SDK, and one component per origin its card's parameter files
// come from, at the revision the card pins.
#ifndef GUFO_LOOM_SOURCE_HPP_
#define GUFO_LOOM_SOURCE_HPP_

#include <algorithm>
#include <cctype>
#include <gufo/ext/json.hpp>
#include <gufo/ext/model_source.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "loom_package.hpp"

namespace gufo::loom::source {

inline constexpr std::string_view kOwner = "loom-fun";

// Package files the runtime does not read: the card's prose, reports, images.
inline const std::vector<std::string> kPackageExclude = {
    "README.md", "REPORT.md", "info.json", ".gitattributes",
    "*.png",     "*.jpg",     "*.svg"};

// The package revision for `sdk`: its v<sdk> tag, else the default branch.
inline std::string SdkRevision(std::string_view repo, std::string_view sdk,
                               const ext::ModelSourceHub& hub) {
  const auto refs =
      json::parse(hub.api("models/" + std::string(repo) + "/refs"));
  if (const auto* tags = refs.find("tags"); tags && tags->is_array()) {
    for (const auto& tag : tags->items()) {
      if (tag.member_str("name") == "v" + std::string(sdk))
        return tag.member_str("targetCommit");
    }
  }
  return json::parse(hub.api("models/" + std::string(repo))).member_str("sha");
}

// The license a repository's card states, else the one its tags name.
inline std::string License(std::string_view repo,
                           const ext::ModelSourceHub& hub) {
  const auto info = json::parse(hub.api("models/" + std::string(repo)));
  std::string value;
  if (const auto* card = info.find("cardData"); card && card->is_object()) {
    value = card->member_str("license");
    if (value == "other" && !card->member_str("license_name").empty())
      value = card->member_str("license_name");
  }
  if (value.empty()) {
    if (const auto* tags = info.find("tags"); tags && tags->is_array()) {
      for (const auto& tag : tags->items()) {
        if (tag.is_string() && tag.str().starts_with("license:")) {
          value = tag.str().substr(8);
          break;
        }
      }
    }
  }
  return value;
}

// The model ID: the repository name, lowercased ("unsloth-qwen3.5-2b-loom").
inline std::string ModelId(std::string_view repo) {
  std::string id;
  for (const char c : repo.substr(repo.find('/') + 1)) {
    const auto lower =
        static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool keep = std::isalnum(static_cast<unsigned char>(lower)) ||
                      lower == '.' || lower == '_' || lower == '-';
    id += keep ? lower : '-';
  }
  while (!id.empty() && !std::isalnum(static_cast<unsigned char>(id.front())))
    id.erase(id.begin());
  return id;
}

inline ext::ModelSourceModel Resolve(std::string_view repo,
                                     const ext::ModelSourceHub& hub,
                                     std::string_view sdk) {
  if (!repo.starts_with(std::string(kOwner) + "/"))
    throw std::runtime_error("loom: " + std::string(repo) + " is not a " +
                             std::string(kOwner) + " repository");
  const auto revision = SdkRevision(repo, sdk, hub);
  if (revision.size() != 40)
    throw std::runtime_error("loom: no revision of " + std::string(repo));
  const auto card = json::parse(hub.file(repo, revision, "model.json"));
  // The loom-text provider serves text packages only (image packages, such
  // as upscalers, have no text service).
  bool text = false;
  if (const auto* services = card.find("services");
      services && services->is_object()) {
    for (const auto& [name, service] : services->members())
      text =
          text || (service.is_object() && service.member_str("kind") == "text");
  }
  if (!text)
    throw std::runtime_error("loom: " + std::string(repo) +
                             " is not a text model package; Gufo serves "
                             "Loom text packages only");
  // A package for another SDK series fails inside Loom: refuse it here.
  const auto config = json::parse(hub.file(repo, revision, "llm_config.json"));
  std::string built = "0.1.0";
  if (const auto* field = config.find("sdk");
      field && field->is_object() && !field->member_str("hrx_loom").empty())
    built = field->member_str("hrx_loom");
  if (package::SdkSeries(built) != package::SdkSeries(sdk))
    throw std::runtime_error("loom: " + std::string(repo) + "@" +
                             revision.substr(0, 8) + " is built for Loom SDK " +
                             built + ", and this Gufo runs Loom SDK " +
                             std::string(sdk) + "; the package has no v" +
                             std::string(sdk) + " tag");

  ext::ModelSourceModel model;
  model.id = ModelId(repo);
  model.components.push_back({.role = "package",
                              .option = "--model",
                              .repo = std::string(repo),
                              .revision = revision,
                              .entry = ".",
                              .include = {},
                              .exclude = kPackageExclude});
  // Each origin's files, in the order the card names them.
  std::vector<std::pair<std::string, std::vector<std::string>>> origins;
  for (const auto& file : package::ParameterFiles(card, "")) {
    if (file.origin.empty())
      continue;
    if (!package::SafeRelative(file.origin_path))
      throw std::runtime_error("loom: the card names an invalid path '" +
                               file.origin_path + "'");
    auto found = origins.begin();
    while (found != origins.end() && found->first != file.origin)
      ++found;
    if (found == origins.end()) {
      origins.emplace_back(file.origin, std::vector<std::string>{});
      found = origins.end() - 1;
    }
    if (std::find(found->second.begin(), found->second.end(),
                  file.origin_path) == found->second.end())
      found->second.push_back(file.origin_path);
  }
  const auto* declared = card.find("origins");
  std::string primary = origins.empty() ? "" : origins.front().first;
  for (const auto& [origin, paths] : origins)
    primary = origin == "checkpoint" ? origin : primary;
  std::string licenses;
  for (const auto& [origin, paths] : origins) {
    // Any Hugging Face origin installs: the runner finds its files in the
    // model directory, so none fills an option.
    const auto* entry = declared ? declared->find(origin) : nullptr;
    if (entry == nullptr || entry->member_str("kind") != "hf")
      throw std::runtime_error("loom: " + std::string(repo) + ": origin " +
                               origin + " is not on Hugging Face");
    const auto source = entry->member_str("repository");
    const auto pinned = entry->member_str("revision");
    if (source.find('/') == std::string::npos || pinned.size() != 40)
      throw std::runtime_error("loom: " + std::string(repo) + ": origin " +
                               origin + " is not pinned to a revision");
    model.components.push_back(
        {.role = origin,
         .option = {},
         .repo = source,
         .revision = pinned,
         .entry = paths.size() == 1 ? paths.front() : ".",
         .include = paths,
         .exclude = {}});
    auto license = License(source, hub);
    if (license.empty())
      license = "no license stated";
    // The checkpoint's license comes first, else the first origin's (the
    // weights of a card without a checkpoint origin, such as "quantized").
    if (origin == primary)
      licenses = license + (licenses.empty() ? "" : "; " + licenses);
    else if (licenses.find(license) == std::string::npos)
      licenses += (licenses.empty() ? "" : "; ") + origin + " (" + source +
                  "): " + license;
  }
  model.license = licenses.empty() ? "no license stated" : licenses;
  const auto name =
      card.find("model") ? card.find("model")->member_str("id") : std::string();
  model.description = (name.empty() ? std::string(repo) : name) +
                      " on AMD Loom (" + std::string(repo) + ")";
  return model;
}

}  // namespace gufo::loom::source

#endif  // GUFO_LOOM_SOURCE_HPP_
