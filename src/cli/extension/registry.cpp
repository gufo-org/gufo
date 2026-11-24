#include "src/cli/extension/registry.hpp"

#include <curl/curl.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "gufo/ext/version.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/core/platform/process.hpp"

namespace gufo::cli::extension_registry {
namespace {
constexpr std::string_view kOfficialIndex =
    "https://raw.githubusercontent.com/gufo-org/gufo-extension/main/"
    "index.json";
constexpr std::size_t kMaximumIndexBytes = 16U << 20U;

std::size_t Append(char* data, std::size_t size, std::size_t count,
                   void* output) {
  auto* text = static_cast<std::string*>(output);
  if (text->size() + size * count > kMaximumIndexBytes)
    return 0;
  text->append(data, size * count);
  return size * count;
}

std::string Download(const std::string& url) {
  if (!url.starts_with("https://"))
    throw std::runtime_error(
        "the extension registry must be served over "
        "HTTPS: " +
        url);
  const std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
      curl_easy_init(), curl_easy_cleanup);
  if (!curl)
    throw std::runtime_error("cannot initialize libcurl");
  std::string body;
  std::array<char, CURL_ERROR_SIZE> error{};
  curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https");
  curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
  curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 15L);
  curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, Append);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, error.data());
  if (curl_easy_perform(curl.get()) != CURLE_OK)
    throw std::runtime_error("cannot fetch the extension registry " + url +
                             ": " + error.data());
  return body;
}

struct ApiVersion {
  int major = -1;
  int minor = -1;
  int revision = -1;
};

ApiVersion ParseVersion(std::string_view text) {
  ApiVersion version;
  int* parts[] = {&version.major, &version.minor, &version.revision};
  std::size_t index = 0;
  int value = -1;
  for (const char c : text) {
    if (std::isdigit(static_cast<unsigned char>(c))) {
      value = (value < 0 ? 0 : value * 10) + (c - '0');
    } else if (c == '.' && value >= 0 && index < 2) {
      *parts[index++] = value;
      value = -1;
    } else {
      return {};
    }
  }
  if (index != 2 || value < 0)
    return {};
  *parts[2] = value;
  return version;
}

std::string HostPlatform() {
  std::string name = platform::HostOsName();
  for (auto& c : name)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return name;
}

// ">= MAJOR.MINOR.REVISION" against this gufo: same series, revision at most
// the host's.
bool Compatible(std::string_view requirement) {
  if (!requirement.starts_with(">= "))
    return false;
  const auto requested = ParseVersion(requirement.substr(3));
  const auto host = ParseVersion(ext::RuntimeApiVersion());
  return requested.major >= 0 && requested.major == host.major &&
         requested.minor == host.minor && requested.revision <= host.revision;
}
}  // namespace

std::string IndexLocation() {
  if (const char* location = std::getenv("GUFO_EXTENSION_REGISTRY");
      location != nullptr && *location != '\0')
    return location;
  return std::string(kOfficialIndex);
}

json::Value FetchIndex() {
  const auto location = IndexLocation();
  std::string text;
  if (std::filesystem::is_regular_file(location)) {
    std::ifstream input(location, std::ios::binary);
    std::ostringstream content;
    content << input.rdbuf();
    text = content.str();
  } else {
    text = Download(location);
  }
  auto index = json::parse(text);
  if (index.member_size("schema") != 1 || !index.find("extensions") ||
      !index.find("extensions")->is_object())
    throw std::runtime_error("unsupported extension registry index at " +
                             location);
  return index;
}

bool Newer(std::string_view candidate, std::string_view installed) {
  const auto a = ParseVersion(candidate);
  const auto b = ParseVersion(installed);
  return std::tie(a.major, a.minor, a.revision) >
         std::tie(b.major, b.minor, b.revision);
}

std::optional<Release> Resolve(const json::Value& index, std::string_view name,
                               std::string_view version, std::string* why) {
  const auto* entry = index.find("extensions")->find(std::string(name));
  if (!entry) {
    *why = "the extension registry has no extension " + std::string(name);
    std::string names;
    for (const auto& [available, _] : index.find("extensions")->members())
      names += (names.empty() ? "" : ", ") + available;
    if (!names.empty())
      *why += "; it lists " + names;
    return std::nullopt;
  }
  const auto platform = HostPlatform();
  std::optional<Release> best;
  bool listed = false;
  if (const auto* releases = entry->find("releases");
      releases && releases->is_array()) {
    for (const auto& release : releases->items()) {
      const auto candidate = release.member_str("version");
      if (!version.empty() && candidate != version)
        continue;
      listed = true;
      bool on_platform = false;
      if (const auto* platforms = release.find("platforms");
          platforms && platforms->is_array()) {
        for (const auto& item : platforms->items())
          on_platform = on_platform || item.get_str() == platform;
      }
      if (!on_platform || !Compatible(release.member_str("api_version")))
        continue;
      if (best && !Newer(candidate, best->version))
        continue;
      auto digest = release.member_str("digest");
      if (digest.starts_with("sha256:"))
        digest = digest.substr(7);
      best = Release{std::string(name), candidate, entry->member_str("source"),
                     release.member_str("revision"), digest};
    }
  }
  if (!best) {
    *why = !listed ? "the extension registry has no " + std::string(name) +
                         " release " + std::string(version)
                   : "no reviewed " + std::string(name) + " release supports " +
                         platform + " and extension API " +
                         std::string(ext::RuntimeApiVersion());
  }
  return best;
}

namespace {
std::string Git(const std::filesystem::path& repository,
                std::vector<std::string> arguments) {
  arguments.insert(arguments.begin(), {"-C", repository.string()});
  std::string output;
  std::string error;
  if (!platform::RunCapture("git", arguments, &output, &error))
    throw std::runtime_error(error);
  return output;
}
}  // namespace

Metadata FetchMetadata(const Release& release,
                       const std::filesystem::path& cache) {
  namespace fs = std::filesystem;
  static const std::regex commit("^[0-9a-f]{40}$");
  if (!std::regex_match(release.revision, commit))
    throw std::runtime_error("reviewed revision is not a commit: " +
                             release.revision);
  const auto cached =
      cache.empty() ? fs::path() : cache / (release.revision + ".json");
  std::error_code ec;
  if (!cached.empty() && fs::is_regular_file(cached, ec)) {
    std::ifstream input(cached, std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    try {
      auto value = json::parse(text.str());
      if (value.find("manifest") && value.find("lock"))
        return {*value.find("manifest"), *value.find("lock")};
    } catch (const std::exception&) {
      // A damaged cache entry is fetched again.
    }
  }
  std::random_device random;
  const auto scratch = fs::temp_directory_path() /
                       ("gufo-registry-" + release.revision.substr(0, 12) +
                        "-" + std::to_string(random()));
  struct Remove {
    fs::path path;
    ~Remove() {
      // Git marks objects read-only, which Windows refuses to delete.
      std::error_code ignored;
      for (fs::recursive_directory_iterator it(path, ignored), end; it != end;
           it.increment(ignored))
        fs::permissions(it->path(), fs::perms::owner_write,
                        fs::perm_options::add, ignored);
      fs::remove_all(path, ignored);
    }
  } cleanup{scratch};
  Metadata metadata;
  try {
    fs::create_directories(scratch);
    Git(scratch, {"init", "--quiet"});
    Git(scratch, {"remote", "add", "origin", release.source});
    Git(scratch, {"fetch", "--quiet", "--depth=1", "--filter=blob:none",
                  "origin", release.revision});
    const auto files =
        Git(scratch, {"ls-tree", "--name-only", release.revision});
    metadata.manifest = json::parse(
        Git(scratch, {"show", release.revision + ":extension.json"}));
    metadata.lock =
        ("\n" + files).find("\nmodels.lock.json\n") != std::string::npos
            ? json::parse(Git(scratch,
                              {"show", release.revision + ":models.lock.json"}))
            : json::Value::object();
  } catch (const std::exception& error) {
    throw std::runtime_error("cannot read " + release.source + "@" +
                             release.revision + ": " + error.what());
  }
  if (metadata.manifest.member_str("name") != release.name)
    throw std::runtime_error(release.source + "@" + release.revision +
                             " is not extension " + release.name);
  if (cached.empty())
    return metadata;
  auto value = json::Value::object();
  value["manifest"] = metadata.manifest;
  value["lock"] = metadata.lock;
  const auto partial = cached.string() + ".partial-" + std::to_string(random());
  fs::create_directories(cache, ec);
  if (std::ofstream output(partial, std::ios::binary); output) {
    output << value.dump();
    output.close();
    if (output)
      fs::rename(partial, cached, ec);
  }
  fs::remove(partial, ec);
  return metadata;
}

std::vector<Listing> List(const json::Value& index) {
  std::vector<Listing> listings;
  for (const auto& [name, entry] : index.find("extensions")->members()) {
    Listing listing{name, entry.member_str("description"), {}, {}};
    if (const auto release = Resolve(index, name, {}, &listing.reason))
      listing.newest = release->version;
    listings.push_back(std::move(listing));
  }
  return listings;
}
}  // namespace gufo::cli::extension_registry
