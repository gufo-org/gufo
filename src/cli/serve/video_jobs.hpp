#ifndef STRIX_SERVER_VIDEO_JOBS_HPP_
#define STRIX_SERVER_VIDEO_JOBS_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "src/models/minimax_h3/generation.hpp"

namespace strix::server {

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
  minimax_h3::GenerationParameters parameters;
  std::uint64_t seed{42};
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
  bool validate_model_inventory{true};
  std::function<std::string()> id_factory;
  std::function<std::int64_t()> now;
  std::function<bool(const minimax_h3::GenerationRequest&,
                     const minimax_h3::CancellationToken*,
                     minimax_h3::GenerationProgress, void*,
                     minimax_h3::GenerationTelemetry*, std::string*)>
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
  [[nodiscard]] VideoJobCreateResult Create(const VideoJobRequest& request);
  [[nodiscard]] VideoJobLookupResult Get(std::string_view id);
  [[nodiscard]] VideoJobContentResult Content(std::string_view id);
  [[nodiscard]] VideoJobLookupResult Delete(std::string_view id);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace strix::server

#endif  // STRIX_SERVER_VIDEO_JOBS_HPP_
