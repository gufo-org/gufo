#ifndef GUFO_SERVER_MEDIA_JOBS_HPP_
#define GUFO_SERVER_MEDIA_JOBS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "src/cli/serve/http_server.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

enum class MediaJobStatus : std::uint8_t {
  kQueued,
  kInProgress,
  kCompleted,
  kFailed,
  kCancelled,
};

[[nodiscard]] std::string_view ToString(MediaJobStatus status) noexcept;

enum class MediaJobResult : std::uint8_t {
  kOk,
  kInvalid,
  kQueueFull,
  kNotFound,
  kNotReady,
  kIoError,
};

/// A job as clients and the persisted `job.json` see it. `metadata` is the
/// kind's own JSON object (no prompt text: hashes only).
struct MediaJobSnapshot {
  std::string id;
  MediaJobStatus status{MediaJobStatus::kQueued};
  std::string model;
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  std::int64_t expires_at{0};
  std::string phase;
  std::uint64_t completed{0};
  /// 0 means unknown.
  std::uint64_t total{0};
  std::string error_code;
  std::string error_message;
  json::Value metadata = json::Value::object();
  /// Published file name to media type.
  std::map<std::string, std::string, std::less<>> files;
};

/// What a running job sees: its directory, cancellation and progress.
class MediaJobContext {
public:
  virtual ~MediaJobContext() = default;
  [[nodiscard]] virtual const std::filesystem::path& directory() const = 0;
  [[nodiscard]] virtual bool cancelled() const = 0;
  /// Returns false once the job is cancelled.
  virtual bool Progress(std::string_view phase, std::uint64_t completed,
                        std::uint64_t total) = 0;
  /// Merges members into the job's metadata.
  virtual void SetMetadata(const json::Value& members) = 0;
  /// Atomically writes `name` in the job directory and publishes it when the
  /// job completes.
  [[nodiscard]] virtual bool WriteFile(std::string_view name,
                                       std::string_view media_type,
                                       std::string_view contents) = 0;
};

/// Runs the job. Returns false with `error` set on failure; a cancelled job
/// returns false and the store records the cancellation.
using MediaWork =
    std::function<bool(MediaJobContext& context, std::string* error)>;

struct MediaJobStoreOptions {
  std::filesystem::path root;
  /// Job IDs are `<prefix>_<24 hex digits>`; also the log component.
  std::string id_prefix;
  /// Persisted `job.json` schema, for example "gufo.music-job.v1".
  std::string schema;
  std::size_t queue_capacity{4};
  std::chrono::seconds ttl{std::chrono::hours(1)};
  std::function<std::string()> id_factory;
  std::function<std::int64_t()> now;
};

struct MediaJobFile {
  std::filesystem::path path;
  std::string media_type;
  std::uint64_t bytes{0};
};

/// A persistent, single-worker job store for generated media: create,
/// status, files, delete, restart recovery of completed jobs and TTL expiry,
/// one directory per job. Kinds supply the work and the response shape.
class MediaJobStore {
public:
  explicit MediaJobStore(MediaJobStoreOptions options);
  ~MediaJobStore();
  MediaJobStore(const MediaJobStore&) = delete;
  MediaJobStore& operator=(const MediaJobStore&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;

  struct Created {
    MediaJobResult result{MediaJobResult::kInvalid};
    std::optional<MediaJobSnapshot> job;
    std::string error;
  };
  [[nodiscard]] Created Create(std::string model, json::Value metadata,
                               MediaWork work);
  [[nodiscard]] std::optional<MediaJobSnapshot> Get(std::string_view id);
  /// A completed job's file.
  [[nodiscard]] MediaJobResult File(std::string_view id, std::string_view name,
                                    MediaJobFile* file);
  /// Cancels a running job and removes it with its files.
  [[nodiscard]] std::optional<MediaJobSnapshot> Delete(std::string_view id);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Serves `file` with `Range: bytes=` support.
[[nodiscard]] HttpResponse ServeMediaFile(const HttpRequest& request,
                                          const MediaJobFile& file);

}  // namespace gufo::server

#endif  // GUFO_SERVER_MEDIA_JOBS_HPP_
