#include "src/cli/serve/media_jobs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <random>
#include <stop_token>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/logging.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::server {
namespace {

constexpr std::size_t kIdDigits = 24;

std::int64_t UnixNow() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string RandomId(const std::string& prefix) {
  std::random_device random;
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = prefix + "_";
  for (std::size_t i = 0; i < kIdDigits; ++i)
    id.push_back(kHex[random() & 0xFU]);
  return id;
}

bool IsJobId(std::string_view id, std::string_view prefix) {
  if (id.size() != prefix.size() + 1 + kIdDigits || !id.starts_with(prefix) ||
      id[prefix.size()] != '_')
    return false;
  return std::ranges::all_of(id.substr(prefix.size() + 1), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

// A file name the job may publish: no separators, no hidden or partial files.
bool IsPublishedName(std::string_view name) {
  return !name.empty() && name.size() <= 64 && name.front() != '.' &&
         name != "job.json" && std::ranges::all_of(name, [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                  c == '.' || c == '_' || c == '-';
         });
}

bool WriteAtomic(const std::filesystem::path& path, std::string_view contents) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return false;
  static std::atomic_uint64_t nonce{0};
  const std::filesystem::path partial =
      path.string() + ".gufo-partial-" + std::to_string(platform::ProcessId()) +
      "-" + std::to_string(nonce.fetch_add(1, std::memory_order_relaxed));
  {
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::filesystem::remove(partial, error);
      return false;
    }
  }
  std::filesystem::rename(partial, path, error);
  if (error) {
    std::filesystem::remove(partial, error);
    return false;
  }
  return true;
}

json::Value SnapshotJson(const MediaJobSnapshot& job,
                         const std::string& schema) {
  json::Value root = json::Value::object();
  root["schema"] = schema;
  root["id"] = job.id;
  root["status"] = std::string(ToString(job.status));
  root["model"] = job.model;
  root["created_at"] = static_cast<long long>(job.created_at);
  root["completed_at"] = static_cast<long long>(job.completed_at);
  root["expires_at"] = static_cast<long long>(job.expires_at);
  root["phase"] = job.phase;
  root["completed"] = static_cast<long long>(job.completed);
  root["total"] = static_cast<long long>(job.total);
  root["metadata"] = job.metadata;
  json::Value files = json::Value::object();
  for (const auto& [name, type] : job.files)
    files[name] = type;
  root["files"] = std::move(files);
  if (!job.error_code.empty()) {
    json::Value error = json::Value::object();
    error["code"] = job.error_code;
    error["message"] = job.error_message;
    root["error"] = std::move(error);
  }
  return root;
}

}  // namespace

std::string_view ToString(MediaJobStatus status) noexcept {
  switch (status) {
    case MediaJobStatus::kQueued:
      return "queued";
    case MediaJobStatus::kInProgress:
      return "in_progress";
    case MediaJobStatus::kCompleted:
      return "completed";
    case MediaJobStatus::kFailed:
      return "failed";
    case MediaJobStatus::kCancelled:
      return "cancelled";
  }
  return "failed";
}

struct MediaJobStore::Impl {
  struct Job {
    MediaJobSnapshot snapshot;
    MediaWork work;
    std::filesystem::path directory;
    std::map<std::string, std::string, std::less<>> pending_files;
    std::atomic<bool> cancelled{false};
    bool deleted{false};
  };

  class Context final : public MediaJobContext {
  public:
    Context(Impl* store, std::shared_ptr<Job> job)
        : store_(store), job_(std::move(job)) {}
    const std::filesystem::path& directory() const override {
      return job_->directory;
    }
    bool cancelled() const override {
      return job_->cancelled.load(std::memory_order_relaxed);
    }
    bool Progress(std::string_view phase, std::uint64_t completed,
                  std::uint64_t total) override {
      const std::lock_guard<std::mutex> lock(store_->mutex);
      auto& snapshot = job_->snapshot;
      if (snapshot.phase != phase)
        Logger::Info(store_->options.id_prefix,
                     "job=" + snapshot.id + " event=phase phase=" +
                         std::string(phase) + " " + Logger::MemoryStatus());
      snapshot.phase = phase;
      snapshot.completed = completed;
      snapshot.total = total;
      return !cancelled();
    }
    void SetMetadata(const json::Value& members) override {
      const std::lock_guard<std::mutex> lock(store_->mutex);
      for (const auto& [key, value] : members.members())
        job_->snapshot.metadata[key] = value;
    }
    bool WriteFile(std::string_view name, std::string_view media_type,
                   std::string_view contents) override {
      if (!IsPublishedName(name) ||
          !WriteAtomic(job_->directory / std::string(name), contents))
        return false;
      const std::lock_guard<std::mutex> lock(store_->mutex);
      job_->pending_files[std::string(name)] = std::string(media_type);
      return true;
    }

  private:
    Impl* store_;
    std::shared_ptr<Job> job_;
  };

  explicit Impl(MediaJobStoreOptions supplied) : options(std::move(supplied)) {
    if (!options.now)
      options.now = UnixNow;
    if (!options.id_factory) {
      const std::string prefix = options.id_prefix;
      options.id_factory = [prefix] { return RandomId(prefix); };
    }
    if (options.root.empty() || options.id_prefix.empty() ||
        options.schema.empty() || options.queue_capacity == 0 ||
        options.ttl <= std::chrono::seconds::zero()) {
      initialization_error = "invalid media job store configuration";
      return;
    }
    std::error_code error;
    std::filesystem::create_directories(options.root, error);
    if (!error)
      options.root = std::filesystem::canonical(options.root, error);
    if (error) {
      initialization_error = "cannot create job storage " +
                             options.root.string() + ": " + error.message();
      return;
    }
    Recover();
    worker = std::jthread([this](const std::stop_token& stop) { Run(stop); });
    ready = true;
  }

  ~Impl() {
    if (!worker.joinable())
      return;
    worker.request_stop();
    {
      const std::lock_guard<std::mutex> lock(mutex);
      for (auto& [id, job] : jobs)
        job->cancelled.store(true, std::memory_order_relaxed);
    }
    condition.notify_all();
    worker.join();
  }

  bool Persist(const MediaJobSnapshot& snapshot) {
    return WriteAtomic(options.root / snapshot.id / "job.json",
                       SnapshotJson(snapshot, options.schema).dump() + "\n");
  }

  // Restores completed, unexpired jobs whose files are all present; removes
  // everything else, including partial writes of an interrupted job.
  void Recover() {
    std::error_code error;
    const std::int64_t now = options.now();
    for (const auto& entry :
         std::filesystem::directory_iterator(options.root, error)) {
      const std::string id = entry.path().filename().string();
      if (!IsJobId(id, options.id_prefix) || entry.is_symlink() ||
          !entry.is_directory())
        continue;
      std::string text;
      {
        // Closed before any removal below: Windows cannot delete open files.
        std::ifstream input(entry.path() / "job.json", std::ios::binary);
        text.assign(std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>());
      }
      bool keep = false;
      auto job = std::make_shared<Job>();
      try {
        const json::Value value = json::parse(text);
        auto& s = job->snapshot;
        s.id = value.member_str("id");
        s.model = value.member_str("model");
        s.created_at =
            static_cast<std::int64_t>(value.member_double("created_at"));
        s.completed_at =
            static_cast<std::int64_t>(value.member_double("completed_at"));
        s.expires_at =
            static_cast<std::int64_t>(value.member_double("expires_at"));
        s.phase = value.member_str("phase");
        s.completed = value.member_size("completed");
        s.total = value.member_size("total");
        if (const json::Value* metadata = value.find("metadata");
            metadata != nullptr && metadata->is_object())
          s.metadata = *metadata;
        s.status = MediaJobStatus::kCompleted;
        keep = value.member_str("schema") == options.schema && s.id == id &&
               value.member_str("status") == "completed" && s.expires_at > now;
        if (const json::Value* files = value.find("files");
            keep && files != nullptr && files->is_object()) {
          for (const auto& [name, type] : files->members()) {
            std::error_code status;
            keep =
                keep && IsPublishedName(name) && type.is_string() &&
                std::filesystem::is_regular_file(entry.path() / name, status);
            s.files[name] = type.is_string() ? type.str() : std::string();
          }
        }
      } catch (...) {
        keep = false;
      }
      if (!keep) {
        std::filesystem::remove_all(entry.path(), error);
        error.clear();
        continue;
      }
      // A crash between a write and its rename leaves a partial file behind.
      std::vector<std::filesystem::path> partials;
      for (const auto& file :
           std::filesystem::directory_iterator(entry.path(), error)) {
        if (file.path().filename().string().find(".gufo-partial-") !=
            std::string::npos)
          partials.push_back(file.path());
      }
      for (const auto& partial : partials)
        std::filesystem::remove(partial, error);
      error.clear();
      job->directory = entry.path();
      jobs.emplace(id, std::move(job));
    }
  }

  void SweepLocked() {
    const std::int64_t now = options.now();
    for (auto it = jobs.begin(); it != jobs.end();) {
      const auto& s = it->second->snapshot;
      if (s.expires_at > 0 && s.expires_at <= now &&
          s.status != MediaJobStatus::kQueued &&
          s.status != MediaJobStatus::kInProgress) {
        std::error_code ignored;
        std::filesystem::remove_all(it->second->directory, ignored);
        it = jobs.erase(it);
      } else {
        ++it;
      }
    }
  }

  void Finish(const std::shared_ptr<Job>& job, bool success,
              const std::string& error, bool deleted) {
    if (deleted) {
      std::error_code ignored;
      std::filesystem::remove_all(job->directory, ignored);
      return;
    }
    auto& s = job->snapshot;
    s.completed_at = options.now();
    s.expires_at = s.completed_at + options.ttl.count();
    if (success) {
      s.status = MediaJobStatus::kCompleted;
      s.files = job->pending_files;
    } else if (job->cancelled.load(std::memory_order_relaxed)) {
      s.status = MediaJobStatus::kCancelled;
      s.error_code = "cancelled";
      s.error_message = "generation was cancelled";
    } else {
      s.status = MediaJobStatus::kFailed;
      s.error_code = "generation_failed";
      s.error_message = error.empty() ? "generation failed" : error;
    }
    if (!Persist(s)) {
      s.status = MediaJobStatus::kFailed;
      s.error_code = "storage_error";
      s.error_message = "cannot persist the job";
      s.files.clear();
    }
    if (s.status != MediaJobStatus::kCompleted) {
      for (const auto& [name, type] : job->pending_files) {
        std::error_code ignored;
        std::filesystem::remove(job->directory / name, ignored);
      }
    }
  }

  void Run(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock,
                       [&] { return stop.stop_requested() || !queue.empty(); });
        if (stop.stop_requested())
          return;
        job = queue.front();
        queue.pop_front();
        if (job->deleted)
          continue;
        job->snapshot.status = MediaJobStatus::kInProgress;
        (void)Persist(job->snapshot);
      }
      const auto start = std::chrono::steady_clock::now();
      Context context(this, job);
      std::string error;
      bool success = false;
      try {
        success = job->work(context, &error);
      } catch (const std::exception& exception) {
        error = exception.what();
      } catch (...) {
        error = "generation failed unexpectedly";
      }
      job->work = nullptr;
      bool deleted = false;
      std::string state;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        deleted = job->deleted;
        Finish(job, success, error, deleted);
        state =
            deleted ? "deleted" : std::string(ToString(job->snapshot.status));
      }
      Logger::Log(state == "failed" ? LogLevel::kError : LogLevel::kInfo,
                  options.id_prefix,
                  "job=" + job->snapshot.id + " event=finished state=" + state +
                      " elapsed_ms=" +
                      std::to_string(
                          std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count()) +
                      (error.empty() ? std::string() : " error=" + error));
    }
  }

  MediaJobStoreOptions options;
  bool ready{false};
  std::string initialization_error;
  std::mutex mutex;
  std::condition_variable condition;
  std::map<std::string, std::shared_ptr<Job>, std::less<>> jobs;
  std::deque<std::shared_ptr<Job>> queue;
  std::jthread worker;
};

MediaJobStore::MediaJobStore(MediaJobStoreOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
MediaJobStore::~MediaJobStore() = default;

bool MediaJobStore::ready() const noexcept {
  return impl_->ready;
}

std::string MediaJobStore::initialization_error() const {
  return impl_->initialization_error;
}

MediaJobStore::Created MediaJobStore::Create(std::string model,
                                             json::Value metadata,
                                             MediaWork work) {
  if (!ready())
    return {MediaJobResult::kIoError, std::nullopt, initialization_error()};
  auto job = std::make_shared<Impl::Job>();
  auto& s = job->snapshot;
  s.id = impl_->options.id_factory();
  if (!IsJobId(s.id, impl_->options.id_prefix))
    return {MediaJobResult::kInvalid, std::nullopt, "invalid job ID"};
  s.model = std::move(model);
  s.metadata = std::move(metadata);
  s.created_at = impl_->options.now();
  s.phase = "queued";
  job->work = std::move(work);
  job->directory = impl_->options.root / s.id;
  MediaJobSnapshot admitted;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepLocked();
    if (impl_->queue.size() >= impl_->options.queue_capacity)
      return {MediaJobResult::kQueueFull, std::nullopt,
              "the job queue is full"};
    std::error_code error;
    if (impl_->jobs.contains(s.id) ||
        std::filesystem::exists(job->directory, error))
      return {MediaJobResult::kInvalid, std::nullopt, "duplicate job ID"};
    if (!impl_->Persist(s))
      return {MediaJobResult::kIoError, std::nullopt, "cannot persist the job"};
    impl_->jobs.emplace(s.id, job);
    impl_->queue.push_back(job);
    admitted = s;
  }
  Logger::Info(impl_->options.id_prefix,
               "job=" + admitted.id + " event=queued model=" + admitted.model);
  impl_->condition.notify_one();
  return {MediaJobResult::kOk, std::move(admitted), {}};
}

std::optional<MediaJobSnapshot> MediaJobStore::Get(std::string_view id) {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->SweepLocked();
  const auto found = impl_->jobs.find(id);
  if (found == impl_->jobs.end())
    return std::nullopt;
  return found->second->snapshot;
}

MediaJobResult MediaJobStore::File(std::string_view id, std::string_view name,
                                   MediaJobFile* file) {
  std::shared_ptr<Impl::Job> job;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->SweepLocked();
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end())
      return MediaJobResult::kNotFound;
    job = found->second;
    if (job->snapshot.status != MediaJobStatus::kCompleted)
      return MediaJobResult::kNotReady;
    const auto type = job->snapshot.files.find(name);
    if (type == job->snapshot.files.end())
      return MediaJobResult::kNotFound;
    file->media_type = type->second;
  }
  file->path = job->directory / std::string(name);
  std::error_code error;
  const auto status = std::filesystem::symlink_status(file->path, error);
  if (error || !std::filesystem::is_regular_file(status))
    return MediaJobResult::kIoError;
  file->bytes = std::filesystem::file_size(file->path, error);
  return error ? MediaJobResult::kIoError : MediaJobResult::kOk;
}

std::optional<MediaJobSnapshot> MediaJobStore::Delete(std::string_view id) {
  std::shared_ptr<Impl::Job> job;
  bool running = false;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end())
      return std::nullopt;
    job = found->second;
    job->deleted = true;
    job->cancelled.store(true, std::memory_order_relaxed);
    running = job->snapshot.status == MediaJobStatus::kInProgress;
    std::erase(impl_->queue, job);
    impl_->jobs.erase(found);
  }
  // A running job's directory goes when its worker returns.
  if (!running) {
    std::error_code ignored;
    std::filesystem::remove_all(job->directory, ignored);
  }
  return job->snapshot;
}

namespace {

bool ParseUnsigned(std::string_view text, std::uint64_t* value) {
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return !text.empty() && parsed.ec == std::errc() &&
         parsed.ptr == text.data() + text.size();
}

}  // namespace

HttpResponse ServeMediaFile(const HttpRequest& request,
                            const MediaJobFile& file) {
  std::uint64_t first = 0;
  std::uint64_t last = file.bytes == 0 ? 0 : file.bytes - 1;
  bool partial = false;
  const std::string range = request.header("range");
  if (!range.empty()) {
    bool valid = range.starts_with("bytes=") && file.bytes != 0 &&
                 range.find(',') == std::string::npos;
    const std::string_view spec =
        valid ? std::string_view(range).substr(6) : std::string_view();
    const std::size_t dash = spec.find('-');
    valid = valid && dash != std::string_view::npos;
    if (valid && dash == 0) {
      std::uint64_t suffix = 0;
      valid = ParseUnsigned(spec.substr(1), &suffix) && suffix != 0;
      first = file.bytes - std::min(suffix, file.bytes);
    } else if (valid) {
      valid = ParseUnsigned(spec.substr(0, dash), &first) && first < file.bytes;
      if (valid && dash + 1 < spec.size())
        valid = ParseUnsigned(spec.substr(dash + 1), &last) && last >= first;
      last = std::min(last, file.bytes - 1);
    }
    if (!valid) {
      HttpResponse response{.status = 416,
                            .reason = "Range Not Satisfiable",
                            .body = {},
                            .headers = {},
                            .streaming_body = {}};
      response.headers.emplace_back("Content-Range",
                                    "bytes */" + std::to_string(file.bytes));
      return response;
    }
    partial = true;
  }
  const std::uint64_t count = file.bytes == 0 ? 0 : last - first + 1;
  std::ifstream input(file.path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(first));
  std::string body(static_cast<std::size_t>(count), '\0');
  input.read(body.data(), static_cast<std::streamsize>(count));
  if (input.gcount() != static_cast<std::streamsize>(count))
    return {
        .status = 503,
        .reason = "Service Unavailable",
        .body =
            R"({"error":{"message":"content is unavailable","type":"server_error","code":"content_unavailable"}})",
        .headers = {},
        .streaming_body = {}};
  HttpResponse response{.status = partial ? 206 : 200,
                        .reason = partial ? "Partial Content" : "OK",
                        .body = std::move(body),
                        .headers = {{"Content-Type", file.media_type},
                                    {"Accept-Ranges", "bytes"}},
                        .streaming_body = {}};
  if (partial)
    response.headers.emplace_back("Content-Range",
                                  "bytes " + std::to_string(first) + "-" +
                                      std::to_string(last) + "/" +
                                      std::to_string(file.bytes));
  return response;
}

}  // namespace gufo::server
