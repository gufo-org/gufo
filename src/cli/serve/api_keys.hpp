#ifndef GUFO_SERVER_API_KEYS_HPP_
#define GUFO_SERVER_API_KEYS_HPP_

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::server {

/// What a key may do. Monitoring keys read health, readiness, models, props,
/// slots and metrics; they cannot submit work.
enum class ApiKeyAccess { kInference, kMonitor };

struct ApiKey {
  /// Empty for the single `--api-key`; otherwise the file's NAME.
  std::string name;
  /// Lowercase hex SHA-256 of the bearer credential.
  std::string digest;
  ApiKeyAccess access{ApiKeyAccess::kInference};
  /// Served model names this key may request; empty allows every model.
  std::vector<std::string> models;

  [[nodiscard]] bool allows_model(std::string_view model) const;
};

/// Lowercase hex SHA-256 of a bearer credential, as stored in key files.
[[nodiscard]] std::string ApiKeyDigest(std::string_view credential);

/// The credentials a server accepts. Immutable once built; reloads replace
/// the whole table.
class ApiKeyTable {
public:
  ApiKeyTable() = default;

  /// One unnamed key with full access (`--api-key`).
  static ApiKeyTable FromKey(std::string_view key);

  /// Parses an `--api-key-file`. One key per line:
  ///   NAME sha256:HEX [access=inference|monitor] [models=NAME,...]
  /// Blank lines and lines starting with `#` are ignored. Returns nullopt and
  /// sets `*error` (with the line number) for any invalid line, duplicate
  /// name or duplicate credential, or a file without keys.
  static std::optional<ApiKeyTable> Parse(std::string_view text,
                                          std::string* error);

  /// Reads and parses `path`; errors name the file.
  static std::optional<ApiKeyTable> Load(const std::filesystem::path& path,
                                         std::string* error);

  /// The key whose digest matches the credential, or nullptr.
  [[nodiscard]] const ApiKey* match(std::string_view credential) const;

  [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return keys_.size(); }

private:
  std::vector<ApiKey> keys_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_API_KEYS_HPP_
