#include "src/cli/serve/api_keys.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <ranges>
#include <span>
#include <sstream>
#include <unordered_set>

#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

constexpr std::size_t kMaxNameLength = 64;
constexpr std::size_t kMaxFileBytes = static_cast<std::size_t>(1) << 20;

bool ValidName(std::string_view name) {
  return !name.empty() && name.size() <= kMaxNameLength &&
         std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
         });
}

std::optional<std::string> ParseDigest(std::string_view value) {
  constexpr std::string_view kPrefix = "sha256:";
  if (!value.starts_with(kPrefix))
    return std::nullopt;
  value.remove_prefix(kPrefix.size());
  if (value.size() != 64)
    return std::nullopt;
  std::string digest;
  digest.reserve(value.size());
  for (const char c : value) {
    if (c >= '0' && c <= '9') {
      digest.push_back(c);
    } else if (c >= 'a' && c <= 'f') {
      digest.push_back(c);
    } else if (c >= 'A' && c <= 'F') {
      digest.push_back(static_cast<char>(c - 'A' + 'a'));
    } else {
      return std::nullopt;
    }
  }
  return digest;
}

std::vector<std::string_view> SplitFields(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t position = 0;
  while (position < line.size()) {
    while (position < line.size() &&
           (line[position] == ' ' || line[position] == '\t'))
      ++position;
    const std::size_t start = position;
    while (position < line.size() && line[position] != ' ' &&
           line[position] != '\t')
      ++position;
    if (position > start)
      fields.push_back(line.substr(start, position - start));
  }
  return fields;
}

}  // namespace

bool ApiKey::allows_model(std::string_view model) const {
  return models.empty() || std::ranges::find(models, model) != models.end();
}

std::string ApiKeyDigest(std::string_view credential) {
  return crypto::Sha256Hex(
      std::span(reinterpret_cast<const std::uint8_t*>(credential.data()),
                credential.size()));
}

ApiKeyTable ApiKeyTable::FromKey(std::string_view key) {
  ApiKeyTable table;
  table.keys_.push_back({.name = {}, .digest = ApiKeyDigest(key)});
  return table;
}

std::optional<ApiKeyTable> ApiKeyTable::Parse(std::string_view text,
                                              std::string* error) {
  ApiKeyTable table;
  std::unordered_set<std::string> names;
  std::unordered_set<std::string> digests;
  std::size_t line_number = 0;
  const auto fail = [&](const std::string& message) {
    if (error != nullptr)
      *error = "line " + std::to_string(line_number) + ": " + message;
    return std::nullopt;
  };
  for (const auto range : std::views::split(text, '\n')) {
    ++line_number;
    std::string_view line(range.begin(), range.end());
    if (line.ends_with('\r'))
      line.remove_suffix(1);
    const auto fields = SplitFields(line);
    if (fields.empty() || fields.front().starts_with('#'))
      continue;
    if (fields.size() < 2)
      return fail("expected NAME sha256:HEX [access=...] [models=...]");
    ApiKey key;
    if (!ValidName(fields[0]))
      return fail("key name must be 1-64 characters of A-Z a-z 0-9 . _ -");
    key.name = std::string(fields[0]);
    auto digest = ParseDigest(fields[1]);
    if (!digest)
      return fail("credential must be sha256: followed by 64 hex digits");
    key.digest = std::move(*digest);
    bool access_seen = false;
    bool models_seen = false;
    for (const auto option : fields | std::views::drop(2)) {
      if (option.starts_with("access=") && !access_seen) {
        access_seen = true;
        const auto value = option.substr(7);
        if (value == "inference") {
          key.access = ApiKeyAccess::kInference;
        } else if (value == "monitor") {
          key.access = ApiKeyAccess::kMonitor;
        } else {
          return fail("access must be inference or monitor");
        }
      } else if (option.starts_with("models=") && !models_seen) {
        models_seen = true;
        // An empty list must not read as "no restriction".
        if (option.size() == 7)
          return fail("models must list served model names");
        for (const auto part : std::views::split(option.substr(7), ',')) {
          const std::string_view model(part.begin(), part.end());
          if (model.empty())
            return fail("models must list served model names");
          key.models.emplace_back(model);
        }
      } else {
        return fail("unknown or repeated option '" + std::string(option) + "'");
      }
    }
    if (key.access == ApiKeyAccess::kMonitor && !key.models.empty())
      return fail("a monitor key cannot list models");
    if (!names.insert(key.name).second)
      return fail("duplicate key name '" + key.name + "'");
    if (!digests.insert(key.digest).second)
      return fail("credential of '" + key.name + "' is already listed");
    table.keys_.push_back(std::move(key));
  }
  if (table.keys_.empty()) {
    if (error != nullptr)
      *error = "no keys";
    return std::nullopt;
  }
  return table;
}

std::optional<ApiKeyTable> ApiKeyTable::Load(const std::filesystem::path& path,
                                             std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr)
      *error = path.string() + ": cannot open";
    return std::nullopt;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  const std::string text = std::move(contents).str();
  if (input.bad() || text.size() > kMaxFileBytes) {
    if (error != nullptr)
      *error = path.string() + ": cannot read (limit 1 MiB)";
    return std::nullopt;
  }
  std::string reason;
  auto table = Parse(text, &reason);
  if (!table && error != nullptr)
    *error = path.string() + ": " + reason;
  return table;
}

const ApiKey* ApiKeyTable::match(std::string_view credential) const {
  // Comparing digests does not disclose matching prefixes of a key.
  const std::string digest = ApiKeyDigest(credential);
  for (const auto& key : keys_) {
    if (key.digest == digest)
      return &key;
  }
  return nullptr;
}

}  // namespace gufo::server
