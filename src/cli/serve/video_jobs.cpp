#include "src/cli/serve/video_jobs.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <ranges>
#include <sstream>
#include <stop_token>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/json.hpp"
#include "src/models/minimax_h3/sha256.hpp"

namespace strix::server {
namespace {

constexpr std::size_t kMaximumPromptBytes = 4096;

bool IsH3Model(std::string_view model) {
  return model == "minimax-h3" || model == "minimax-h3-exact" ||
         model == "minimax-h3-fast" || model == "minimax-h3-aggressive" ||
         model == "minimax-h3-dev" || model == "minimax-h3-fullres";
}

bool ModelMatchesPreset(std::string_view model, std::string_view preset) {
  if (model == "minimax-h3") {
    return true;
  }
  return (model == "minimax-h3-exact" && preset == "exact-512") ||
         (model == "minimax-h3-fast" && preset == "fast-384") ||
         (model == "minimax-h3-aggressive" && preset == "aggressive-320") ||
         (model == "minimax-h3-dev" && preset == "development-256") ||
         (model == "minimax-h3-fullres" && preset == "exact-1344x768");
}

bool MatchesFrozenOutputContract(const VideoJobRequest& request) {
  const auto& parameters = request.parameters;
  if (!ModelMatchesPreset(request.model, parameters.preset)) {
    return false;
  }
  if (request.output_format == "mp4") {
    const bool one_second =
        request.seconds == "1" && request.size == "512x512" &&
        parameters.output_width == 512 && parameters.output_height == 512 &&
        parameters.frames == 22 &&
        (parameters.preset == "exact-512" || parameters.preset == "fast-384" ||
         parameters.preset == "aggressive-320");
    const bool released_full_resolution =
        request.seconds == "5" && request.size == "1344x768" &&
        parameters.preset == "exact-1344x768" &&
        parameters.internal_width == 1344 &&
        parameters.internal_height == 768 && parameters.output_width == 1344 &&
        parameters.output_height == 768 && parameters.frames == 124;
    return (one_second || released_full_resolution) && parameters.mux &&
           parameters.decode_audio && parameters.selected_frames.empty();
  }
  return request.output_format == "ppm" && request.size == "256x256" &&
         parameters.preset == "development-256" &&
         parameters.internal_width == 256 &&
         parameters.internal_height == 256 && parameters.output_width == 256 &&
         parameters.output_height == 256 && parameters.frames == 22 &&
         !parameters.mux && !parameters.decode_audio &&
         parameters.selected_frames.size() == 1;
}

void WipeString(std::string* value) noexcept {
  if (value == nullptr) {
    return;
  }
  volatile char* const data = value->data();
  for (std::size_t index = 0; index < value->size(); ++index) {
    data[index] = '\0';
  }
  value->clear();
}

std::string RandomVideoId() {
  std::array<unsigned char, 12> bytes{};
  std::random_device random;
  for (unsigned char& byte : bytes) {
    byte = static_cast<unsigned char>(random());
  }
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "video_";
  id.reserve(6 + bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    id.push_back(kHex[byte >> 4U]);
    id.push_back(kHex[byte & 0xFU]);
  }
  return id;
}

std::int64_t UnixNow() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool IsVideoId(std::string_view id) {
  if (!id.starts_with("video_") || id.size() != 30) {
    return false;
  }
  return std::ranges::all_of(id.substr(6), [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

bool IsStrictDescendant(const std::filesystem::path& root,
                        const std::filesystem::path& candidate) {
  auto root_iterator = root.begin();
  auto candidate_iterator = candidate.begin();
  for (; root_iterator != root.end() && candidate_iterator != candidate.end();
       ++root_iterator, ++candidate_iterator) {
    if (*root_iterator != *candidate_iterator) {
      return false;
    }
  }
  return root_iterator == root.end() && candidate_iterator != candidate.end();
}

bool IsSafeRegularContent(const std::filesystem::path& storage_root,
                          const std::filesystem::path& candidate,
                          std::uint64_t* bytes = nullptr) {
  std::error_code filesystem_error;
  const std::filesystem::file_status content_status =
      std::filesystem::symlink_status(candidate, filesystem_error);
  if (filesystem_error || std::filesystem::is_symlink(content_status) ||
      !std::filesystem::is_regular_file(content_status)) {
    return false;
  }
  const std::filesystem::path canonical_content =
      std::filesystem::canonical(candidate, filesystem_error);
  if (filesystem_error ||
      !IsStrictDescendant(storage_root, canonical_content)) {
    return false;
  }
  if (bytes != nullptr) {
    *bytes = std::filesystem::file_size(candidate, filesystem_error);
    if (filesystem_error) {
      return false;
    }
  }
  return true;
}

std::uint64_t NextFileNonce() {
  static std::atomic_uint64_t nonce{0};
  return nonce.fetch_add(1, std::memory_order_relaxed);
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents,
                 std::string* error) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(path.parent_path(), filesystem_error);
  if (filesystem_error) {
    if (error != nullptr) {
      *error =
          "cannot create video job directory: " + filesystem_error.message();
    }
    return false;
  }
  const std::filesystem::path partial = path.string() + ".strix-partial-" +
                                        std::to_string(getpid()) + "-" +
                                        std::to_string(NextFileNonce());
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::filesystem::remove(partial, filesystem_error);
      if (error != nullptr) {
        *error = "cannot write video job metadata";
      }
      return false;
    }
  }
  std::filesystem::rename(partial, path, filesystem_error);
  if (filesystem_error) {
    std::filesystem::remove(partial, filesystem_error);
    if (error != nullptr) {
      *error =
          "cannot publish video job metadata: " + filesystem_error.message();
    }
    return false;
  }
  return true;
}

std::string SnapshotJson(const VideoJobSnapshot& snapshot) {
  json::Value root = json::Value::object();
  root["schema"] = "strix.video-job.v1";
  root["id"] = snapshot.id;
  root["status"] = std::string(ToString(snapshot.status));
  root["progress"] = snapshot.progress;
  root["created_at"] = static_cast<long long>(snapshot.created_at);
  root["completed_at"] = static_cast<long long>(snapshot.completed_at);
  root["expires_at"] = static_cast<long long>(snapshot.expires_at);
  root["model"] = snapshot.model;
  root["size"] = snapshot.size;
  root["seconds"] = snapshot.seconds;
  root["output_format"] = snapshot.output_format;
  if (!snapshot.error_code.empty()) {
    json::Value error = json::Value::object();
    error["code"] = snapshot.error_code;
    error["message"] = snapshot.error_message;
    root["error"] = std::move(error);
  } else {
    root["error"] = json::Value();
  }
  return root.dump() + "\n";
}

int ProgressPercent(std::string_view phase, int completed, int total) {
  const auto scaled = [completed, total](int begin, int end) {
    if (total <= 0) {
      return begin;
    }
    const int bounded = std::clamp(completed, 0, total);
    return begin + ((end - begin) * bounded) / total;
  };
  if (phase == "inventory") {
    return scaled(1, 3);
  }
  if (phase == "prompt") {
    return scaled(3, 10);
  }
  if (phase == "denoiser") {
    return scaled(10, 80);
  }
  if (phase == "video_vae") {
    return scaled(80, 90);
  }
  if (phase == "audio_vae") {
    return scaled(90, 97);
  }
  if (phase == "media") {
    return scaled(97, 99);
  }
  return 1;
}

}  // namespace

std::string_view ToString(VideoJobStatus status) noexcept {
  switch (status) {
    case VideoJobStatus::kQueued:
      return "queued";
    case VideoJobStatus::kInProgress:
      return "in_progress";
    case VideoJobStatus::kCompleted:
      return "completed";
    case VideoJobStatus::kFailed:
      return "failed";
    case VideoJobStatus::kCancelled:
      return "cancelled";
  }
  return "failed";
}

struct VideoJobService::Impl {
  struct Job {
    VideoJobSnapshot snapshot;
    VideoJobRequest request;
    std::filesystem::path directory;
    std::filesystem::path content_path;
    std::string media_type;
    minimax_h3::CancellationToken cancellation;
    bool deleted{false};
  };

  explicit Impl(VideoJobServiceOptions supplied)
      : options(std::move(supplied)) {
    if (!options.id_factory) {
      options.id_factory = RandomVideoId;
    }
    if (!options.now) {
      options.now = UnixNow;
    }
    if (!options.runner) {
      options.runner = minimax_h3::GenerateTextVideo;
    }
    if (options.model_root.empty() || options.source_manifest.empty() ||
        options.storage_root.empty() || options.queue_capacity == 0 ||
        options.artifact_ttl <= std::chrono::seconds::zero()) {
      initialization_error = "invalid MiniMax H3 video job configuration";
      return;
    }
#if !defined(ENGINE_ENABLE_HIP)
    if (options.validate_model_inventory) {
      initialization_error =
          "MiniMax H3 video serving requires a ROCm HIP-enabled build; no CPU "
          "inference fallback is provided";
      return;
    }
#endif
    if (options.validate_model_inventory) {
      std::string inspection_error;
      if (!minimax_h3::ModelInventory::Inspect(
               options.model_root, options.source_manifest, &inspection_error)
               .has_value()) {
        initialization_error = std::move(inspection_error);
        return;
      }
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(options.storage_root, filesystem_error);
    if (filesystem_error) {
      initialization_error =
          "cannot create video job storage: " + filesystem_error.message();
      return;
    }
    canonical_storage_root =
        std::filesystem::canonical(options.storage_root, filesystem_error);
    if (filesystem_error) {
      initialization_error =
          "cannot resolve video job storage: " + filesystem_error.message();
      return;
    }
    options.storage_root = canonical_storage_root;
    RecoverJobs();
    if (!initialization_error.empty()) {
      return;
    }
    worker = std::jthread([this](const std::stop_token& stop) { Run(stop); });
    ready = true;
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (worker.joinable()) {
      worker.request_stop();
      {
        const std::lock_guard<std::mutex> lock(mutex);
        if (active != nullptr) {
          active->cancellation.Cancel();
        }
        for (const auto& job : queue) {
          job->cancellation.Cancel();
          WipeString(&job->request.prompt);
        }
      }
      condition.notify_all();
      worker.join();
    }
  }

  VideoJobSnapshot Snapshot(const Job& job) const { return job.snapshot; }

  bool Persist(const VideoJobSnapshot& snapshot, std::string* error = nullptr) {
    return WriteAtomic(options.storage_root / snapshot.id / "job.json",
                       SnapshotJson(snapshot), error);
  }

  void RecoverJobs() {
    std::error_code filesystem_error;
    const std::int64_t now_value = options.now();
    for (const auto& entry : std::filesystem::directory_iterator(
             options.storage_root, filesystem_error)) {
      if (filesystem_error) {
        initialization_error =
            "cannot inspect video job storage: " + filesystem_error.message();
        return;
      }
      const std::filesystem::file_status entry_status =
          entry.symlink_status(filesystem_error);
      if (filesystem_error) {
        initialization_error =
            "cannot inspect video job entry: " + filesystem_error.message();
        return;
      }
      const std::string directory_id = entry.path().filename().string();
      if (std::filesystem::is_symlink(entry_status) ||
          !std::filesystem::is_directory(entry_status) ||
          !IsVideoId(directory_id)) {
        continue;
      }
      for (const auto& child : std::filesystem::directory_iterator(
               entry.path(), filesystem_error)) {
        if (filesystem_error) {
          break;
        }
        if (child.path().filename().string().find(".strix-partial-") !=
            std::string::npos) {
          std::filesystem::remove_all(child.path(), filesystem_error);
          filesystem_error.clear();
        }
      }
      const std::filesystem::path metadata = entry.path() / "job.json";
      std::ifstream input(metadata, std::ios::binary);
      const std::string text((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
      if (text.empty()) {
        std::filesystem::remove_all(entry.path(), filesystem_error);
        filesystem_error.clear();
        continue;
      }
      try {
        const json::Value value = json::parse(text);
        auto job = std::make_shared<Job>();
        job->snapshot.id = value.member_str("id");
        job->snapshot.model = value.member_str("model");
        job->snapshot.size = value.member_str("size");
        job->snapshot.seconds = value.member_str("seconds");
        job->snapshot.output_format = value.member_str("output_format");
        const double progress = value.member_double("progress", -1);
        const double created_at = value.member_double("created_at", -1);
        const double completed_at = value.member_double("completed_at", -1);
        const double expires_at = value.member_double("expires_at", -1);
        job->directory = entry.path();
        const bool valid_output = (job->snapshot.output_format == "mp4" &&
                                   (job->snapshot.size == "512x512" ||
                                    job->snapshot.size == "1344x768")) ||
                                  (job->snapshot.output_format == "ppm" &&
                                   job->snapshot.size == "256x256");
        const bool valid_numbers =
            std::isfinite(progress) && std::floor(progress) == progress &&
            progress == 100.0 && std::isfinite(created_at) &&
            std::floor(created_at) == created_at && created_at > 0.0 &&
            created_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max()) &&
            std::isfinite(completed_at) &&
            std::floor(completed_at) == completed_at &&
            completed_at >= created_at &&
            completed_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max()) &&
            std::isfinite(expires_at) && std::floor(expires_at) == expires_at &&
            expires_at > completed_at &&
            expires_at <=
                static_cast<double>(std::numeric_limits<std::int64_t>::max());
        if (value.member_str("schema") != "strix.video-job.v1" ||
            !IsVideoId(job->snapshot.id) || job->snapshot.id != directory_id ||
            !IsH3Model(job->snapshot.model) ||
            (job->snapshot.seconds != "1" && job->snapshot.seconds != "5") ||
            !valid_output || !valid_numbers ||
            expires_at <= static_cast<double>(now_value) ||
            value.member_str("status") != "completed") {
          std::filesystem::remove_all(entry.path(), filesystem_error);
          filesystem_error.clear();
          continue;
        }
        job->snapshot.progress = static_cast<int>(progress);
        job->snapshot.created_at = static_cast<std::int64_t>(created_at);
        job->snapshot.completed_at = static_cast<std::int64_t>(completed_at);
        job->snapshot.expires_at = static_cast<std::int64_t>(expires_at);
        if (job->snapshot.output_format == "mp4") {
          job->content_path = entry.path() / "output.mp4";
          job->media_type = "video/mp4";
        } else {
          job->content_path = entry.path() / "output.ppm";
          job->media_type = "image/x-portable-pixmap";
        }
        if (!IsSafeRegularContent(canonical_storage_root, job->content_path)) {
          std::filesystem::remove_all(entry.path(), filesystem_error);
          filesystem_error.clear();
          continue;
        }
        job->snapshot.status = VideoJobStatus::kCompleted;
        jobs.emplace(job->snapshot.id, std::move(job));
      } catch (...) {
        std::filesystem::remove_all(entry.path(), filesystem_error);
        filesystem_error.clear();
      }
    }
  }

  void SweepExpiredLocked() {
    const std::int64_t now_value = options.now();
    for (auto iterator = jobs.begin(); iterator != jobs.end();) {
      const auto& job = iterator->second;
      if (job->snapshot.expires_at > 0 &&
          job->snapshot.expires_at <= now_value &&
          job->snapshot.status != VideoJobStatus::kInProgress &&
          job->snapshot.status != VideoJobStatus::kQueued) {
        std::error_code ignored;
        std::filesystem::remove_all(job->directory, ignored);
        iterator = jobs.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  void UpdateProgress(const std::shared_ptr<Job>& job, std::string_view phase,
                      int completed, int total) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (job->deleted || job->snapshot.status != VideoJobStatus::kInProgress) {
      return;
    }
    job->snapshot.progress =
        std::max(job->snapshot.progress,
                 std::clamp(ProgressPercent(phase, completed, total), 1, 99));
  }

  struct ProgressContext {
    Impl* service{nullptr};
    std::shared_ptr<Job> job;
  };

  static void Progress(std::string_view phase, int completed, int total,
                       void* opaque) {
    auto* context = static_cast<ProgressContext*>(opaque);
    if (context != nullptr && context->service != nullptr) {
      context->service->UpdateProgress(context->job, phase, completed, total);
    }
  }

  void Run(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      std::shared_ptr<Job> job;
      bool started = true;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock,
                       [&] { return stop.stop_requested() || !queue.empty(); });
        if (stop.stop_requested()) {
          break;
        }
        job = queue.front();
        queue.pop_front();
        if (job->deleted) {
          continue;
        }
        active = job;
        job->snapshot.status = VideoJobStatus::kInProgress;
        job->snapshot.progress = 1;
        std::string persist_error;
        if (!Persist(job->snapshot, &persist_error)) {
          active.reset();
          WipeString(&job->request.prompt);
          job->snapshot.status = VideoJobStatus::kFailed;
          job->snapshot.completed_at = options.now();
          job->snapshot.expires_at =
              job->snapshot.completed_at + options.artifact_ttl.count();
          job->snapshot.error_code = "storage_error";
          job->snapshot.error_message = "cannot persist video job transition";
          (void)Persist(job->snapshot);
          started = false;
        }
      }
      if (!started) {
        condition.notify_all();
        continue;
      }

      minimax_h3::GenerationRequest generation{
          .model_root = options.model_root,
          .source_manifest = options.source_manifest,
          .output_path = job->directory / "output.mp4",
          .frames_directory = job->request.output_format == "ppm"
                                  ? job->directory / "frames"
                                  : std::filesystem::path(),
          .latents_directory = {},
          .prompt = std::move(job->request.prompt),
          .seed = job->request.seed,
          .parameters = job->request.parameters,
      };
      WipeString(&job->request.prompt);
      const std::string prompt_hash = minimax_h3::Sha256(std::span(
          reinterpret_cast<const unsigned char*>(generation.prompt.data()),
          generation.prompt.size()));
      std::string metadata_error;
      if (!WriteAtomic(
              job->directory / "parameters.json",
              minimax_h3::GenerationParametersJson(
                  job->request.parameters, job->request.seed, prompt_hash),
              &metadata_error)) {
        WipeString(&generation.prompt);
        const std::lock_guard<std::mutex> lock(mutex);
        active.reset();
        job->snapshot.status = VideoJobStatus::kFailed;
        job->snapshot.completed_at = options.now();
        job->snapshot.expires_at =
            job->snapshot.completed_at + options.artifact_ttl.count();
        job->snapshot.error_code = "storage_error";
        job->snapshot.error_message = "cannot persist video job parameters";
        (void)Persist(job->snapshot);
        condition.notify_all();
        continue;
      }
      ProgressContext progress{this, job};
      minimax_h3::GenerationTelemetry telemetry;
      std::string error;
      bool success = false;
      try {
        success = options.runner(generation, &job->cancellation, Progress,
                                 &progress, &telemetry, &error);
      } catch (const std::exception&) {
        error = "video generation backend threw an exception";
      } catch (...) {
        error = "video generation backend failed unexpectedly";
      }
      WipeString(&generation.prompt);

      std::filesystem::path content_path;
      std::string media_type;
      bool output_ready = success && !job->cancellation.IsCancelled();
      if (output_ready && job->request.output_format == "mp4") {
        content_path = job->directory / "output.mp4";
        media_type = "video/mp4";
      } else if (output_ready) {
        std::ostringstream filename;
        filename << "frame-" << std::setw(4) << std::setfill('0')
                 << job->request.parameters.selected_frames.front() << ".ppm";
        const std::filesystem::path generated =
            job->directory / "frames" / filename.str();
        content_path = job->directory / "output.ppm";
        std::error_code rename_error;
        std::filesystem::rename(generated, content_path, rename_error);
        if (!rename_error) {
          std::filesystem::remove_all(job->directory / "frames", rename_error);
        } else {
          output_ready = false;
        }
        media_type = "image/x-portable-pixmap";
      }
      if (output_ready) {
        output_ready =
            IsSafeRegularContent(canonical_storage_root, content_path);
      }
      bool storage_failure = false;
      if (output_ready) {
        std::string telemetry_error;
        if (!WriteAtomic(job->directory / "telemetry.json",
                         minimax_h3::GenerationTelemetryJson(telemetry),
                         &telemetry_error)) {
          output_ready = false;
          storage_failure = true;
        }
      }

      VideoJobSnapshot snapshot;
      bool deleted = false;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        active.reset();
        deleted = job->deleted;
        if (!deleted) {
          const std::int64_t completed_at = options.now();
          snapshot = job->snapshot;
          snapshot.completed_at = completed_at;
          snapshot.expires_at = completed_at + options.artifact_ttl.count();
          if (output_ready) {
            snapshot.status = VideoJobStatus::kCompleted;
            snapshot.progress = 100;
          } else if (job->cancellation.IsCancelled()) {
            snapshot.status = VideoJobStatus::kCancelled;
            snapshot.error_code = "cancelled";
            snapshot.error_message = "video generation was cancelled";
          } else if (storage_failure) {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.error_code = "storage_error";
            snapshot.error_message = "cannot persist video job telemetry";
          } else {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.error_code = "generation_failed";
            snapshot.error_message = "video generation failed";
          }
          std::string persist_error;
          if (!Persist(snapshot, &persist_error)) {
            snapshot.status = VideoJobStatus::kFailed;
            snapshot.progress = std::min(snapshot.progress, 99);
            snapshot.error_code = "storage_error";
            snapshot.error_message = "cannot persist video job completion";
            (void)Persist(snapshot);
          }
          job->snapshot = snapshot;
          if (snapshot.status == VideoJobStatus::kCompleted) {
            job->content_path = std::move(content_path);
            job->media_type = std::move(media_type);
          }
        }
      }
      if (deleted) {
        std::error_code ignored;
        std::filesystem::remove_all(job->directory, ignored);
      } else {
        if (snapshot.status != VideoJobStatus::kCompleted) {
          std::error_code ignored;
          std::filesystem::remove(job->directory / "output.mp4", ignored);
          std::filesystem::remove(job->directory / "output.ppm", ignored);
          std::filesystem::remove_all(job->directory / "frames", ignored);
          std::filesystem::remove(job->directory / "telemetry.json", ignored);
        }
      }
      condition.notify_all();
    }
  }

  VideoJobServiceOptions options;
  bool ready{false};
  std::string initialization_error;
  std::filesystem::path canonical_storage_root;
  std::mutex mutex;
  std::condition_variable condition;
  std::map<std::string, std::shared_ptr<Job>, std::less<>> jobs;
  std::deque<std::shared_ptr<Job>> queue;
  std::shared_ptr<Job> active;
  std::jthread worker;
};

VideoJobService::VideoJobService(VideoJobServiceOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

VideoJobService::~VideoJobService() = default;

bool VideoJobService::ready() const noexcept {
  return impl_ != nullptr && impl_->ready;
}

std::string VideoJobService::initialization_error() const {
  return impl_ == nullptr ? "video job service is unavailable"
                          : impl_->initialization_error;
}

VideoJobCreateResult VideoJobService::Create(const VideoJobRequest& request) {
  if (!ready()) {
    return {.result = VideoJobResult::kIoError,
            .job = std::nullopt,
            .error = initialization_error()};
  }
  if (!IsH3Model(request.model) || request.prompt.empty() ||
      request.prompt.size() > kMaximumPromptBytes ||
      request.prompt.find('\0') != std::string::npos ||
      (request.seconds != "1" && request.seconds != "5") ||
      (request.output_format != "mp4" && request.output_format != "ppm") ||
      !minimax_h3::ValidateGenerationParameters(request.parameters, nullptr) ||
      !MatchesFrozenOutputContract(request)) {
    return {.result = VideoJobResult::kInvalid,
            .job = std::nullopt,
            .error = "invalid MiniMax H3 video job request"};
  }

  auto job = std::make_shared<Impl::Job>();
  job->request = request;
  job->snapshot.id = impl_->options.id_factory();
  job->snapshot.status = VideoJobStatus::kQueued;
  job->snapshot.created_at = impl_->options.now();
  job->snapshot.model = request.model;
  job->snapshot.size = request.size;
  job->snapshot.seconds = request.seconds;
  job->snapshot.output_format = request.output_format;
  if (!IsVideoId(job->snapshot.id)) {
    WipeString(&job->request.prompt);
    return {.result = VideoJobResult::kInvalid,
            .job = std::nullopt,
            .error = "video job ID factory returned an invalid ID"};
  }
  job->directory = impl_->options.storage_root / job->snapshot.id;

  VideoJobSnapshot admitted;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    if (impl_->queue.size() >= impl_->options.queue_capacity) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kQueueFull,
              .job = std::nullopt,
              .error = "MiniMax H3 video queue is full"};
    }
    if (impl_->jobs.contains(job->snapshot.id)) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kInvalid,
              .job = std::nullopt,
              .error = "duplicate video job ID"};
    }
    std::error_code filesystem_error;
    const bool directory_exists =
        std::filesystem::exists(job->directory, filesystem_error);
    if (filesystem_error) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kIoError,
              .job = std::nullopt,
              .error = "cannot inspect video job destination"};
    }
    if (directory_exists) {
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kInvalid,
              .job = std::nullopt,
              .error = "video job destination already exists"};
    }
    impl_->jobs.emplace(job->snapshot.id, job);
    impl_->queue.push_back(job);
    std::string persist_error;
    if (!impl_->Persist(job->snapshot, &persist_error)) {
      impl_->queue.pop_back();
      impl_->jobs.erase(job->snapshot.id);
      std::error_code ignored;
      std::filesystem::remove_all(job->directory, ignored);
      WipeString(&job->request.prompt);
      return {.result = VideoJobResult::kIoError,
              .job = std::nullopt,
              .error = std::move(persist_error)};
    }
    admitted = impl_->Snapshot(*job);
  }
  impl_->condition.notify_one();
  return {
      .result = VideoJobResult::kOk, .job = std::move(admitted), .error = {}};
}

VideoJobLookupResult VideoJobService::Get(std::string_view id) {
  if (!IsVideoId(id)) {
    return {.result = VideoJobResult::kNotFound,
            .job = std::nullopt,
            .error = "video job not found"};
  }
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->SweepExpiredLocked();
  const auto iterator = impl_->jobs.find(id);
  if (iterator == impl_->jobs.end()) {
    return {.result = VideoJobResult::kNotFound,
            .job = std::nullopt,
            .error = "video job not found"};
  }
  return {.result = VideoJobResult::kOk,
          .job = impl_->Snapshot(*iterator->second),
          .error = {}};
}

VideoJobContentResult VideoJobService::Content(std::string_view id) {
  std::shared_ptr<Impl::Job> job;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    const auto iterator = impl_->jobs.find(id);
    if (iterator == impl_->jobs.end()) {
      return {.result = VideoJobResult::kNotFound,
              .content = std::nullopt,
              .error = "video job not found"};
    }
    job = iterator->second;
    if (job->snapshot.status != VideoJobStatus::kCompleted) {
      return {.result = VideoJobResult::kNotReady,
              .content = std::nullopt,
              .error = "video content is not ready"};
    }
  }
  std::uint64_t bytes = 0;
  if (!IsSafeRegularContent(impl_->canonical_storage_root, job->content_path,
                            &bytes)) {
    return {.result = VideoJobResult::kIoError,
            .content = std::nullopt,
            .error = "video content is unavailable"};
  }
  return {.result = VideoJobResult::kOk,
          .content =
              VideoJobContent{
                  .path = job->content_path,
                  .media_type = job->media_type,
                  .bytes = bytes,
              },
          .error = {}};
}

VideoJobLookupResult VideoJobService::Delete(std::string_view id) {
  std::shared_ptr<Impl::Job> job;
  VideoJobSnapshot snapshot;
  bool was_active = false;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepExpiredLocked();
    const auto iterator = impl_->jobs.find(id);
    if (iterator == impl_->jobs.end()) {
      return {.result = VideoJobResult::kNotFound,
              .job = std::nullopt,
              .error = "video job not found"};
    }
    job = iterator->second;
    snapshot = job->snapshot;
    job->deleted = true;
    job->cancellation.Cancel();
    WipeString(&job->request.prompt);
    was_active = impl_->active == job;
    std::erase(impl_->queue, job);
    impl_->jobs.erase(iterator);
  }
  if (!was_active) {
    std::error_code ignored;
    std::filesystem::remove_all(job->directory, ignored);
  }
  impl_->condition.notify_all();
  return {
      .result = VideoJobResult::kOk, .job = std::move(snapshot), .error = {}};
}

}  // namespace strix::server
