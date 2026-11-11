#ifndef GUFO_SERVER_VIDEO_JOBS_HPP_
#define GUFO_SERVER_VIDEO_JOBS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/ext/video.hpp"

namespace gufo::server {

enum class VideoJobStatus : std::uint8_t {
  kQueued,
  kInProgress,
  kCompleted,
  kFailed,
  kCancelled,
};

[[nodiscard]] std::string_view ToString(VideoJobStatus status) noexcept;

enum class VideoJobResult : std::uint8_t {
  kOk,
  kInvalid,
  kQueueFull,
  kNotFound,
  kNotReady,
  kExpired,
  kIoError,
};

struct VideoJobRequest {
  std::string model{"minimax-h3"};
  std::string prompt;
  std::string size{"512x512"};
  std::string seconds{"1"};
  std::string output_format{"mp4"};
  ext::VideoParameters parameters;
  std::uint64_t seed{42};
  /// Uploaded keyframe bytes for conditioning (issue #177), already
  /// decoded and validated at admission. Empty selects no anchor. Create()
  /// stages each non-empty vector as `first_frame.img`/`last_frame.img`
  /// INSIDE the job directory so cancellation, DELETE, and TTL expiry sweeps
  /// reclaim the uploaded imagery together with the generated output.
  std::vector<std::uint8_t> first_frame_bytes;
  std::vector<std::uint8_t> last_frame_bytes;
};

struct VideoJobSnapshot {
  std::string id;
  VideoJobStatus status{VideoJobStatus::kQueued};
  int progress{0};
  std::int64_t created_at{0};
  std::int64_t completed_at{0};
  std::int64_t expires_at{0};
  std::string model;
  std::string size;
  std::string seconds;
  std::string output_format;
  std::string error_code;
  std::string error_message;
  /// SHA-256 of the staged keyframe bytes, verbatim as uploaded in any
  /// accepted format ("" when absent). Kept in
  /// the persisted snapshot so job provenance survives restart recovery and
  /// is merged into the job's parameters.json report.
  std::string first_frame_sha256;
  std::string last_frame_sha256;
};

struct VideoJobContent {
  std::filesystem::path path;
  std::string media_type;
  std::uint64_t bytes{0};
};

struct VideoJobServiceOptions {
  std::filesystem::path model_root;
  std::filesystem::path source_manifest;
  std::filesystem::path storage_root;
  std::size_t queue_capacity{1};
  std::chrono::seconds artifact_ttl{std::chrono::hours(1)};
  /// Additional IDs accepted from requests. They behave as the root
  /// `minimax-h3` name; the preset-selecting variants stay fixed (#957).
  std::vector<std::string> model_aliases;
  bool validate_model_inventory{true};
  std::function<std::string()> id_factory;
  std::function<std::int64_t()> now;
  std::function<bool(const ext::VideoRequest&, const ext::CancellationCheck&,
                     ext::VideoProgress, void*, ext::VideoTelemetry*,
                     std::string*)>
      runner;
};

struct VideoJobCreateResult {
  VideoJobResult result{VideoJobResult::kInvalid};
  std::optional<VideoJobSnapshot> job;
  std::string error;
};

struct VideoJobLookupResult {
  VideoJobResult result{VideoJobResult::kNotFound};
  std::optional<VideoJobSnapshot> job;
  std::string error;
};

struct VideoJobContentResult {
  VideoJobResult result{VideoJobResult::kNotFound};
  std::optional<VideoJobContent> content;
  std::string error;
};

class VideoJobService {
public:
  explicit VideoJobService(VideoJobServiceOptions options);
  ~VideoJobService();

  VideoJobService(const VideoJobService&) = delete;
  VideoJobService& operator=(const VideoJobService&) = delete;
  VideoJobService(VideoJobService&&) = delete;
  VideoJobService& operator=(VideoJobService&&) = delete;

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] std::string initialization_error() const;
  /// True for a canonical `minimax-h3` name or a configured alias (#957).
  [[nodiscard]] bool serves_model(std::string_view name) const;
  /// The request name as a canonical model: aliases map to the root
  /// `minimax-h3`, unknown names are returned unchanged for rejection.
  [[nodiscard]] std::string normalize_model(std::string_view name) const;
  [[nodiscard]] const std::vector<std::string>& model_aliases() const;
  [[nodiscard]] VideoJobCreateResult Create(const VideoJobRequest& request);
  [[nodiscard]] VideoJobLookupResult Get(std::string_view id);
  [[nodiscard]] VideoJobContentResult Content(std::string_view id);
  [[nodiscard]] VideoJobLookupResult Delete(std::string_view id);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_VIDEO_JOBS_HPP_
