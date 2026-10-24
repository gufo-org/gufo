#include "src/cli/serve/video_jobs.hpp"

#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace h3 = gufo::minimax_h3;
using gufo::server::VideoJobRequest;
using gufo::server::VideoJobResult;
using gufo::server::VideoJobService;
using gufo::server::VideoJobServiceOptions;
using gufo::server::VideoJobStatus;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL video_jobs_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

struct FakeRunner {
  std::mutex mutex;
  std::condition_variable condition;
  int calls{0};
  bool release{false};
  std::vector<h3::GenerationRequest> requests;

  bool Run(const h3::GenerationRequest& request,
           const h3::CancellationToken* cancellation,
           h3::GenerationProgress progress, void* progress_opaque,
           h3::GenerationTelemetry* telemetry, std::string* error) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      ++calls;
      requests.push_back(request);
    }
    condition.notify_all();
    progress("inventory", 1, 1, progress_opaque);
    progress("prompt", 1, 1, progress_opaque);
    progress("denoiser", 2, 4, progress_opaque);
    progress("video_vae", 1, 1, progress_opaque);
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (!release &&
             (cancellation == nullptr || !cancellation->IsCancelled())) {
        condition.wait_for(lock, std::chrono::milliseconds(5));
      }
    }
    if (cancellation != nullptr && cancellation->IsCancelled()) {
      if (error != nullptr) {
        *error = "cancelled";
      }
      return false;
    }
    std::error_code filesystem_error;
    if (request.parameters.mux) {
      std::filesystem::create_directories(request.output_path.parent_path(),
                                          filesystem_error);
      std::ofstream output(request.output_path, std::ios::binary);
      output << "fake-mp4-content";
    } else {
      std::filesystem::create_directories(request.frames_directory,
                                          filesystem_error);
      const int frame = request.parameters.selected_frames.front();
      std::ostringstream filename;
      filename << "frame-" << std::setw(4) << std::setfill('0') << frame
               << ".ppm";
      std::ofstream output(request.frames_directory / filename.str(),
                           std::ios::binary);
      output << "P6\n1 1\n255\nabc";
    }
    if (filesystem_error) {
      if (error != nullptr) {
        *error = filesystem_error.message();
      }
      return false;
    }
    if (telemetry != nullptr) {
      telemetry->prompt_tokens = 6;
      telemetry->denoiser.forward_ms = {1.0, 2.0};
    }
    return true;
  }

  void WaitForCalls(int expected) {
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(lock, std::chrono::seconds(5),
                            [&] { return calls >= expected; })) {
      Fail("timed out waiting for fake video runner");
    }
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release = true;
    }
    condition.notify_all();
  }
};

std::string NextId() {
  static int counter = 0;
  ++counter;
  std::ostringstream output;
  output << "video_" << std::setw(24) << std::setfill('0') << std::hex
         << counter;
  return output.str();
}

VideoJobRequest Mp4Request(std::string prompt = "A red fox") {
  auto parameters = h3::ResolveGenerationPreset("fast", nullptr);
  Check(parameters.has_value(), "fast preset resolves");
  return {
      .model = "minimax-h3",
      .prompt = std::move(prompt),
      .size = "512x512",
      .seconds = "1",
      .output_format = "mp4",
      .parameters = std::move(*parameters),
      .seed = 42,
  };
}

template<typename Predicate>
void WaitUntil(Predicate predicate, std::string_view message) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  Fail(std::string(message));
}

std::string ReadAll(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void WriteCompletedJob(const std::filesystem::path& directory,
                       std::string_view id, std::string_view format = "mp4") {
  std::filesystem::create_directories(directory);
  std::ofstream metadata(directory / "job.json");
  metadata << "{\"schema\":\"gufo.video-job.v1\",\"id\":\"" << id
           << "\",\"status\":\"completed\",\"progress\":100,"
              "\"created_at\":3000,\"completed_at\":3001,"
              "\"expires_at\":4000,\"model\":\"minimax-h3\","
              "\"size\":\""
           << (format == "mp4" ? "512x512" : "256x256")
           << "\",\"seconds\":\"1\",\"output_format\":\"" << format
           << "\",\"error\":null}\n";
}

void TestQueueLifecycleAndRecovery(const std::filesystem::path& root) {
  FakeRunner runner;
  std::int64_t now = 1000;
  VideoJobServiceOptions options{
      .model_root = "/private/models/MiniMax-H3",
      .source_manifest = "/private/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::seconds(60),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [&] { return now; },
      .runner =
          [&](const h3::GenerationRequest& request,
              const h3::CancellationToken* cancellation,
              h3::GenerationProgress progress, void* progress_opaque,
              h3::GenerationTelemetry* telemetry, std::string* error) {
            return runner.Run(request, cancellation, progress, progress_opaque,
                              telemetry, error);
          },
  };

  std::string completed_id;
  {
    VideoJobService service(options);
    Check(service.ready(), service.initialization_error());
    const auto first = service.Create(Mp4Request("private prompt one"));
    Check(first.result == VideoJobResult::kOk && first.job.has_value(),
          "first video job is admitted");
    const std::string first_id = first.job->id;
    runner.WaitForCalls(1);
    WaitUntil(
        [&] {
          const auto lookup = service.Get(first_id);
          return lookup.job.has_value() &&
                 lookup.job->status == VideoJobStatus::kInProgress &&
                 lookup.job->progress >= 80;
        },
        "first job did not publish expected progress");
    const auto running = service.Get(first_id);
    Check(running.job.has_value() &&
              running.job->status == VideoJobStatus::kInProgress,
          "first job enters in_progress");
    Check(running.job->progress >= 80 && running.job->progress < 100,
          "progress is monotonic and bounded before completion");

    auto longer_request = Mp4Request("private prompt two");
    longer_request.seconds = "14.375";
    longer_request.parameters.frames = 345;
    const auto second = service.Create(longer_request);
    Check(second.result == VideoJobResult::kOk && second.job.has_value(),
          "one queued video job is admitted");
    completed_id = second.job->id;
    const auto saturated = service.Create(Mp4Request("third prompt"));
    Check(saturated.result == VideoJobResult::kQueueFull,
          "bounded queue rejects saturation");

    const auto deleted = service.Delete(first_id);
    Check(deleted.result == VideoJobResult::kOk,
          "active job deletion requests cancellation");
    Check(service.Get(first_id).result == VideoJobResult::kNotFound,
          "deleted active job disappears immediately");
    runner.WaitForCalls(2);
    runner.Release();

    WaitUntil(
        [&] {
          const auto lookup = service.Get(completed_id);
          return lookup.job.has_value() &&
                 lookup.job->status == VideoJobStatus::kCompleted;
        },
        "queued job did not complete");
    const auto completed = service.Get(completed_id);
    Check(completed.job->progress == 100, "completed progress reaches 100");
    const auto content = service.Content(completed_id);
    Check(content.result == VideoJobResult::kOk &&
              content.content.has_value() &&
              content.content->media_type == "video/mp4" &&
              content.content->bytes ==
                  std::string_view("fake-mp4-content").size(),
          "completed MP4 content is retrievable");

    {
      const std::lock_guard<std::mutex> lock(runner.mutex);
      Check(runner.requests.size() == 2, "runner request count");
      const auto& request = runner.requests.back();
      Check(request.parameters.internal_width == 384 &&
                request.parameters.output_width == 512 &&
                request.parameters.evaluations == 19 &&
                request.parameters.active_blocks == 45 &&
                request.parameters.reuse_interval == 2 &&
                request.parameters.frames == 345 && request.seed == 42,
            "server request matches direct fast CLI parameters");
    }
    const std::filesystem::path directory = root / completed_id;
    const std::string metadata = ReadAll(directory / "job.json");
    const std::string parameters = ReadAll(directory / "parameters.json");
    const std::string telemetry = ReadAll(directory / "telemetry.json");
    Check(metadata.find("\"status\":\"completed\"") != std::string::npos &&
              telemetry.find("gufo.minimax-h3-generation-telemetry.v1") !=
                  std::string::npos,
          "completed state is published with telemetry");
    Check(metadata.find("private prompt") == std::string::npos &&
              metadata.find("/private/") == std::string::npos &&
              parameters.find("private prompt") == std::string::npos &&
              parameters.find("/private/") == std::string::npos,
          "persisted job data excludes prompts and private paths");
    std::ofstream(directory / "job.json.gufo-partial-injected") << "incomplete";
  }

  {
    VideoJobService recovered(options);
    Check(recovered.ready(), recovered.initialization_error());
    Check(!std::filesystem::exists(root / completed_id /
                                   "job.json.gufo-partial-injected"),
          "restart removes partial metadata files");
    const auto lookup = recovered.Get(completed_id);
    Check(lookup.job.has_value() &&
              lookup.job->status == VideoJobStatus::kCompleted,
          "completed job survives service restart");
    Check(recovered.Content(completed_id).result == VideoJobResult::kOk,
          "recovered content remains retrievable");
    Check(recovered.Delete(completed_id).result == VideoJobResult::kOk,
          "recovered job can be deleted");
    Check(!std::filesystem::exists(root / completed_id),
          "delete reclaims completed artifacts");
  }
}

void TestPpmAndExpiry(const std::filesystem::path& root) {
  FakeRunner runner;
  runner.release = true;
  std::int64_t now = 2000;
  VideoJobService service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::seconds(5),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [&] { return now; },
      .runner =
          [&](const h3::GenerationRequest& request,
              const h3::CancellationToken* cancellation,
              h3::GenerationProgress progress, void* progress_opaque,
              h3::GenerationTelemetry* telemetry, std::string* error) {
            return runner.Run(request, cancellation, progress, progress_opaque,
                              telemetry, error);
          },
  });
  auto parameters = h3::ResolveGenerationPreset("dev", nullptr);
  Check(parameters.has_value(), "development preset resolves");
  parameters->selected_frames = {11};
  const auto created = service.Create({
      .model = "minimax-h3",
      .prompt = "diagnostic",
      .size = "256x256",
      .seconds = "1",
      .output_format = "ppm",
      .parameters = *parameters,
      .seed = 7,
  });
  Check(created.job.has_value(), "single-frame PPM job admitted");
  const std::string id = created.job->id;
  WaitUntil(
      [&] {
        const auto lookup = service.Get(id);
        return lookup.job.has_value() &&
               lookup.job->status == VideoJobStatus::kCompleted;
      },
      "PPM job did not complete");
  const auto content = service.Content(id);
  Check(content.content.has_value() &&
            content.content->media_type == "image/x-portable-pixmap",
        "diagnostic PPM content type");
  now += 6;
  Check(service.Get(id).result == VideoJobResult::kNotFound,
        "expired job is no longer visible");
  Check(!std::filesystem::exists(root / id), "expired artifacts are reclaimed");
}

void TestValidation(const std::filesystem::path& root) {
  VideoJobService unavailable({
      .model_root = root / "missing-model",
      .source_manifest = root / "missing-manifest.json",
      .storage_root = root / "unavailable",
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::hours(1),
      .validate_model_inventory = true,
      .id_factory = {},
      .now = {},
      .runner = {},
  });
  Check(!unavailable.ready(),
        "invalid checkpoint fails before video request admission");

  VideoJobService service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = {},
      .runner = [](const h3::GenerationRequest&, const h3::CancellationToken*,
                   h3::GenerationProgress, void*, h3::GenerationTelemetry*,
                   std::string*) { return false; },
  });
  Check(service.Create(Mp4Request("")).result == VideoJobResult::kInvalid,
        "empty prompt is rejected");
  Check(service.Create(Mp4Request(std::string("A\0B", 3))).result ==
            VideoJobResult::kInvalid,
        "embedded NUL in prompt is rejected");
  auto invalid = Mp4Request();
  invalid.seconds = "10";
  Check(service.Create(invalid).result == VideoJobResult::kInvalid,
        "job admission rejects seconds/frame mismatch independently of HTTP");
  invalid.parameters.frames = 362;
  invalid.seconds = "15";
  Check(service.Create(invalid).result == VideoJobResult::kInvalid,
        "job admission rejects alignment beyond the released limit");
  invalid = Mp4Request();
  invalid.model = "../escape";
  Check(service.Create(invalid).result == VideoJobResult::kInvalid,
        "invalid model and traversal-like value rejected");
  invalid = Mp4Request();
  invalid.model = "minimax-h3-exact";
  Check(service.Create(invalid).result == VideoJobResult::kInvalid,
        "model alias cannot disagree with the frozen preset");

  auto development = h3::ResolveGenerationPreset("dev", nullptr);
  Check(development.has_value(), "development preset resolves");
  development->selected_frames = {11};
  const VideoJobRequest wrong_ppm_size{
      .model = "minimax-h3-dev",
      .prompt = "diagnostic",
      .size = "512x512",
      .seconds = "1",
      .output_format = "ppm",
      .parameters = *development,
      .seed = 42,
  };
  Check(service.Create(wrong_ppm_size).result == VideoJobResult::kInvalid,
        "direct PPM jobs enforce development output geometry");

  const std::string collision_id = "video_0000000000000000000000d1";
  const std::filesystem::path collision_root = root / "collision";
  VideoJobService collision_service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = collision_root,
      .queue_capacity = 1,
      .validate_model_inventory = false,
      .id_factory = [collision_id] { return collision_id; },
      .now = {},
      .runner = [](const h3::GenerationRequest&, const h3::CancellationToken*,
                   h3::GenerationProgress, void*, h3::GenerationTelemetry*,
                   std::string*) { return false; },
  });
  Check(collision_service.ready(), collision_service.initialization_error());
  std::filesystem::create_directories(collision_root / collision_id);
  std::ofstream(collision_root / collision_id / "sentinel") << "preserve";
  Check(
      collision_service.Create(Mp4Request()).result == VideoJobResult::kInvalid,
      "pre-existing job destinations are never reused");
  Check(std::filesystem::is_regular_file(collision_root / collision_id /
                                         "sentinel"),
        "destination collision preserves existing data");
}

void TestRecoveryHardening(const std::filesystem::path& root) {
  const std::string directory_id = "video_0000000000000000000000a1";
  const std::string claimed_id = "video_0000000000000000000000a2";
  const std::filesystem::path mismatched = root / directory_id;
  WriteCompletedJob(mismatched, claimed_id);
  std::ofstream(mismatched / "output.mp4") << "wrong-id";

  const std::string linked_content_id = "video_0000000000000000000000b1";
  const std::filesystem::path linked_content = root / linked_content_id;
  WriteCompletedJob(linked_content, linked_content_id);
  const std::filesystem::path secret = root / "outside-secret.mp4";
  std::ofstream(secret) << "must-not-be-served";
  std::error_code link_error;
  std::filesystem::create_symlink(secret, linked_content / "output.mp4",
                                  link_error);
  Check(!link_error, "content symlink fixture created");

  const std::string linked_directory_id = "video_0000000000000000000000c1";
  const std::filesystem::path external = root / "external-job";
  WriteCompletedJob(external, linked_directory_id);
  std::ofstream(external / "output.mp4") << "outside-directory";
  std::filesystem::create_directory_symlink(
      external, root / linked_directory_id, link_error);
  Check(!link_error, "directory symlink fixture created");

  VideoJobService service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::hours(1),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 3100; },
      .runner = [](const h3::GenerationRequest&, const h3::CancellationToken*,
                   h3::GenerationProgress, void*, h3::GenerationTelemetry*,
                   std::string*) { return false; },
  });
  Check(service.ready(), service.initialization_error());
  Check(service.Get(claimed_id).result == VideoJobResult::kNotFound,
        "recovery rejects metadata/directory ID mismatch");
  Check(service.Get(linked_content_id).result == VideoJobResult::kNotFound,
        "recovery rejects symlinked content");
  Check(service.Get(linked_directory_id).result == VideoJobResult::kNotFound,
        "recovery rejects symlinked job directory");
  Check(std::filesystem::is_regular_file(secret),
        "recovery preserves the symlink target");
  Check(std::filesystem::is_regular_file(external / "output.mp4"),
        "recovery does not traverse the directory symlink");
}

void TestRunnerExceptionRecovery(const std::filesystem::path& root) {
  int calls = 0;
  VideoJobService service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::hours(1),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 5000; },
      .runner =
          [&](const h3::GenerationRequest& request,
              const h3::CancellationToken*, h3::GenerationProgress, void*,
              h3::GenerationTelemetry*, std::string*) {
            ++calls;
            if (calls == 1) {
              throw std::runtime_error("injected backend exception");
            }
            std::filesystem::create_directories(
                request.output_path.parent_path());
            std::ofstream(request.output_path, std::ios::binary)
                << "recovered-worker-output";
            return true;
          },
  });
  Check(service.ready(), service.initialization_error());
  const auto first = service.Create(Mp4Request("throwing request"));
  Check(first.job.has_value(), "throwing backend request admitted");
  WaitUntil(
      [&] {
        const auto lookup = service.Get(first.job->id);
        return lookup.job.has_value() &&
               lookup.job->status == VideoJobStatus::kFailed;
      },
      "throwing backend did not become a failed job");

  const auto second = service.Create(Mp4Request("recovery request"));
  Check(second.job.has_value(), "post-exception request admitted");
  WaitUntil(
      [&] {
        const auto lookup = service.Get(second.job->id);
        return lookup.job.has_value() &&
               lookup.job->status == VideoJobStatus::kCompleted;
      },
      "worker did not recover after backend exception");
  Check(service.Content(second.job->id).result == VideoJobResult::kOk,
        "post-exception worker publishes content");
}

void TestRunnerOutputBoundary(const std::filesystem::path& root) {
  const std::filesystem::path external =
      root.parent_path() / "runner-output-secret.mp4";
  std::ofstream(external, std::ios::binary) << "must-not-be-published";
  VideoJobService service({
      .model_root = "/models/h3",
      .source_manifest = "/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::hours(1),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 6000; },
      .runner =
          [&](const h3::GenerationRequest& request,
              const h3::CancellationToken*, h3::GenerationProgress, void*,
              h3::GenerationTelemetry*, std::string*) {
            std::filesystem::create_directories(
                request.output_path.parent_path());
            std::error_code link_error;
            std::filesystem::create_symlink(external, request.output_path,
                                            link_error);
            return !link_error;
          },
  });
  Check(service.ready(), service.initialization_error());
  const auto created = service.Create(Mp4Request("boundary request"));
  Check(created.job.has_value(), "output-boundary request admitted");
  WaitUntil(
      [&] {
        const auto lookup = service.Get(created.job->id);
        return lookup.job.has_value() &&
               lookup.job->status == VideoJobStatus::kFailed;
      },
      "symlinked runner output did not fail");
  Check(service.Content(created.job->id).result == VideoJobResult::kNotReady,
        "symlinked runner output is not published");
  Check(ReadAll(external) == "must-not-be-published",
        "runner output rejection preserves the external target");
}

}  // namespace

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("gufo-video-jobs-test-" + std::to_string(getpid()));
  std::error_code ignored;
  std::filesystem::remove_all(base, ignored);
  std::filesystem::create_directories(base);
  TestQueueLifecycleAndRecovery(base / "lifecycle");
  TestPpmAndExpiry(base / "ppm");
  TestValidation(base / "validation");
  TestRecoveryHardening(base / "recovery-hardening");
  TestRunnerExceptionRecovery(base / "runner-exception");
  TestRunnerOutputBoundary(base / "runner-output-boundary");
  std::filesystem::remove_all(base, ignored);
  std::cout << "Video job lifecycle tests passed.\n";
  return 0;
}
