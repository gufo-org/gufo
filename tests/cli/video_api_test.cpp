#include "src/cli/serve/video_api.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/json.hpp"

namespace {

namespace h3 = strix::minimax_h3;
using strix::server::HandleVideoApiRequest;
using strix::server::HttpRequest;
using strix::server::HttpResponse;
using strix::server::VideoJobService;
using strix::server::VideoJobServiceOptions;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL video_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

std::string NextId() {
  static std::atomic_uint64_t counter{0};
  std::ostringstream output;
  output << "video_" << std::setw(24) << std::setfill('0') << std::hex
         << counter.fetch_add(1, std::memory_order_relaxed) + 1;
  return output.str();
}

class FakeRunner {
public:
  explicit FakeRunner(bool blocked = false) : blocked_(blocked) {}

  bool Run(const h3::GenerationRequest& request,
           const h3::CancellationToken* cancellation,
           h3::GenerationProgress progress, void* progress_opaque,
           h3::GenerationTelemetry*, std::string* error) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      ++calls_;
      latest_ = request;
    }
    condition_.notify_all();
    progress("denoiser", 1, 2, progress_opaque);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (blocked_ && !released_ &&
             (cancellation == nullptr || !cancellation->IsCancelled())) {
        condition_.wait_for(lock, std::chrono::milliseconds(5));
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
      output << "fake-video-content";
    } else {
      std::filesystem::create_directories(request.frames_directory,
                                          filesystem_error);
      std::ostringstream filename;
      filename << "frame-" << std::setw(4) << std::setfill('0')
               << request.parameters.selected_frames.front() << ".ppm";
      std::ofstream output(request.frames_directory / filename.str(),
                           std::ios::binary);
      output << "P6\n1 1\n255\nabc";
    }
    return !filesystem_error;
  }

  void WaitForCalls(int expected) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(5),
                             [&] { return calls_ >= expected; })) {
      Fail("timed out waiting for fake runner");
    }
  }

  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    condition_.notify_all();
  }

  h3::GenerationRequest latest() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

private:
  bool blocked_{false};
  bool released_{false};
  int calls_{0};
  h3::GenerationRequest latest_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

VideoJobServiceOptions Options(const std::filesystem::path& root,
                               FakeRunner* runner) {
  return {
      .model_root = "/private/models/MiniMax-H3",
      .source_manifest = "/private/manifest.json",
      .storage_root = root,
      .queue_capacity = 1,
      .artifact_ttl = std::chrono::seconds(60),
      .validate_model_inventory = false,
      .id_factory = NextId,
      .now = [] { return 1000; },
      .runner =
          [runner](const h3::GenerationRequest& request,
                   const h3::CancellationToken* cancellation,
                   h3::GenerationProgress progress, void* progress_opaque,
                   h3::GenerationTelemetry* telemetry, std::string* error) {
            return runner->Run(request, cancellation, progress, progress_opaque,
                               telemetry, error);
          },
  };
}

HttpResponse Send(VideoJobService& service, std::string method,
                  std::string path, std::string body = {},
                  std::string range = {}, std::string content_type = {}) {
  HttpRequest request{
      .method = std::move(method),
      .path = std::move(path),
      .query = {},
      .body = std::move(body),
      .headers = {},
      .is_cancelled = {},
  };
  if (!range.empty()) {
    request.headers.emplace_back("RaNgE", std::move(range));
  }
  if (!content_type.empty()) {
    request.headers.emplace_back("CoNtEnT-TyPe", std::move(content_type));
  }
  return HandleVideoApiRequest(request, service);
}

std::string Multipart(
    std::string_view boundary,
    const std::vector<std::pair<std::string, std::string>>& fields) {
  std::string body;
  for (const auto& [name, value] : fields) {
    body += "--";
    body += boundary;
    body += "\r\nContent-Disposition: form-data; name=\"";
    body += name;
    body += "\"\r\n\r\n";
    body += value;
    body += "\r\n";
  }
  body += "--";
  body += boundary;
  body += "--\r\n";
  return body;
}

std::string CreatedId(const HttpResponse& response) {
  const auto body = strix::server::json::parse(response.body);
  return body.member_str("id");
}

void WaitCompleted(VideoJobService& service, std::string_view id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    const HttpResponse response =
        Send(service, "GET", "/v1/videos/" + std::string(id));
    if (response.body.find("\"status\":\"completed\"") != std::string::npos) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  Fail("video API job did not complete");
}

void TestLifecycleAndRange(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  Check(service.ready(), service.initialization_error());

  const std::string request =
      R"({"model":"minimax-h3-dev","prompt":"private fox prompt","size":"256x256","seconds":"1","strix":{"seed":7,"frames":22,"output_format":"ppm","selected_frame":4}})";
  const HttpResponse created = Send(service, "POST", "/v1/videos", request, {},
                                    "application/json; charset=utf-8");
  Check(created.status == 202 &&
            created.body.find("private fox prompt") == std::string::npos &&
            created.body.find("\"status\":\"queued\"") != std::string::npos,
        "create returns a private, queued video object");
  const std::string id = CreatedId(created);
  Check(!id.empty(), "create response includes an ID");
  runner.WaitForCalls(1);
  WaitCompleted(service, id);

  const h3::GenerationRequest generation = runner.latest();
  Check(generation.seed == 7 &&
            generation.parameters.preset == "development-256" &&
            generation.parameters.frames == 22 &&
            generation.parameters.selected_frames == std::vector<int>({4}) &&
            !generation.parameters.mux,
        "HTTP adapter preserves direct generation parameters");

  const HttpResponse content =
      Send(service, "GET", "/v1/videos/" + id + "/content");
  Check(content.status == 200 && content.body == "P6\n1 1\n255\nabc",
        "completed diagnostic frame is returned");

  const HttpResponse partial =
      Send(service, "GET", "/v1/videos/" + id + "/content", {}, "bytes=0-4");
  Check(partial.status == 206 && partial.body == "P6\n1 ",
        "single HTTP byte range is returned");
  Check(std::ranges::any_of(partial.headers,
                            [](const auto& header) {
                              return header.first == "Content-Range" &&
                                     header.second == "bytes 0-4/14";
                            }),
        "partial content includes a stable Content-Range");

  const HttpResponse invalid_range =
      Send(service, "GET", "/v1/videos/" + id + "/content", {}, "bytes=99-100");
  Check(invalid_range.status == 416, "unsatisfiable range returns HTTP 416");

  const HttpResponse deleted = Send(service, "DELETE", "/v1/videos/" + id);
  Check(deleted.status == 200 &&
            deleted.body.find("\"deleted\":true") != std::string::npos,
        "DELETE returns a deletion object");
  Check(Send(service, "GET", "/v1/videos/" + id).status == 404,
        "deleted jobs are no longer visible");
}

void TestValidation(const std::filesystem::path& root) {
  FakeRunner runner;
  VideoJobService service(Options(root, &runner));
  const HttpResponse fullres = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"A goalkeeper saves a shot","size":"1344x768","seconds":5})");
  Check(fullres.status == 202, "released full-resolution API request accepted");
  runner.WaitForCalls(1);
  const h3::GenerationRequest fullres_generation = runner.latest();
  Check(fullres_generation.parameters.preset == "exact-1344x768" &&
            fullres_generation.parameters.internal_width == 1344 &&
            fullres_generation.parameters.internal_height == 768 &&
            fullres_generation.parameters.frames == 124 &&
            fullres_generation.parameters.evaluations == 49,
        "API maps 1344x768 five-second request to the exact preset");
  WaitCompleted(service, CreatedId(fullres));

  const HttpResponse reference = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","input_reference":"secret.png"})");
  Check(reference.status == 400 &&
            reference.body.find("unsupported_input_reference") !=
                std::string::npos,
        "future conditioning inputs fail closed");

  const HttpResponse bad_output = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","strix":{"preset":"dev","output_format":"mp4"}})");
  Check(bad_output.status == 400,
        "development preset cannot silently produce MP4");

  const HttpResponse mismatch = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3-fast","prompt":"x","size":"512x512","seconds":"1","strix":{"preset":"exact"}})");
  Check(mismatch.status == 400 &&
            mismatch.body.find("preset_model_mismatch") != std::string::npos,
        "model aliases cannot drift from the frozen preset");

  const HttpResponse too_short = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3-dev","prompt":"x","size":"256x256","seconds":"1","strix":{"frames":5,"output_format":"ppm","selected_frame":4}})");
  Check(too_short.status == 400 &&
            too_short.body.find("invalid_frames") != std::string::npos &&
            too_short.body.find("at least 22 aligned frames") !=
                std::string::npos,
        "sub-chunk requests fail before the released VisualVAE is invoked");

  const HttpResponse unknown = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","silent_typo":true})");
  Check(unknown.status == 400 &&
            unknown.body.find("unsupported_field") != std::string::npos,
        "unknown top-level fields fail closed");

  const HttpResponse duplicate_json = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","model":"minimax-h3-fast","prompt":"x","seconds":"1"})");
  Check(duplicate_json.status == 400 &&
            duplicate_json.body.find("parse_error") != std::string::npos,
        "duplicate JSON fields fail closed");

  const HttpResponse trailing_json =
      Send(service, "POST", "/v1/videos",
           R"({"model":"minimax-h3","prompt":"x","seconds":"1"} trailing)");
  Check(trailing_json.status == 400 &&
            trailing_json.body.find("parse_error") != std::string::npos,
        "trailing JSON input fails closed");

  const HttpResponse malformed_number = Send(
      service, "POST", "/v1/videos",
      R"({"model":"minimax-h3","prompt":"x","seconds":"1","strix":{"seed":1e}})");
  Check(malformed_number.status == 400 &&
            malformed_number.body.find("parse_error") != std::string::npos,
        "malformed JSON numbers fail closed");

  const HttpResponse wrong_size_type =
      Send(service, "POST", "/v1/videos",
           R"({"model":"minimax-h3","prompt":"x","size":512,"seconds":"1"})");
  Check(wrong_size_type.status == 400 &&
            wrong_size_type.body.find("invalid_parameter_type") !=
                std::string::npos,
        "wrongly typed optional fields do not fall back to defaults");

  const std::string oversized = "{\"model\":\"minimax-h3\",\"prompt\":\"" +
                                std::string(4097, 'x') +
                                "\",\"seconds\":\"1\"}";
  const HttpResponse too_large = Send(service, "POST", "/v1/videos", oversized);
  Check(too_large.status == 413 &&
            too_large.body.find("prompt_too_large") != std::string::npos,
        "oversized prompts use the documented HTTP 413 contract");

  Check(Send(service, "GET", "/v1/videos/../../private").status == 404,
        "path traversal-like IDs fail closed");

  constexpr std::string_view kBoundary = "strix-openai-video-boundary";
  const HttpResponse multipart =
      Send(service, "POST", "/v1/videos",
           Multipart(kBoundary,
                     {
                         {"model", "minimax-h3-fast"},
                         {"prompt", "A red fox walking through snow"},
                         {"size", "512x512"},
                         {"seconds", "1"},
                     }),
           {}, "multipart/form-data; boundary=\"strix-openai-video-boundary\"");
  Check(multipart.status == 202,
        "OpenAI-style multipart text-to-video request is admitted");
  runner.WaitForCalls(2);
  Check(runner.latest().parameters.preset == "fast-384",
        "multipart request preserves the selected H3 mode");

  const HttpResponse duplicate =
      Send(service, "POST", "/v1/videos",
           Multipart(kBoundary,
                     {
                         {"model", "minimax-h3"},
                         {"prompt", "first"},
                         {"prompt", "second"},
                         {"seconds", "1"},
                     }),
           {}, "multipart/form-data; boundary=strix-openai-video-boundary");
  Check(duplicate.status == 400 &&
            duplicate.body.find("invalid_multipart_form") != std::string::npos,
        "multipart duplicate fields fail closed");

  const std::string reference_body =
      "--strix-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
      "minimax-h3\r\n"
      "--strix-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
      "x\r\n"
      "--strix-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"seconds\"\r\n\r\n"
      "1\r\n"
      "--strix-openai-video-boundary\r\n"
      "Content-Disposition: form-data; name=\"input_reference\"; "
      "filename=\"frame.png\"\r\n"
      "Content-Type: image/png\r\n\r\n"
      "not-an-image\r\n"
      "--strix-openai-video-boundary--\r\n";
  const HttpResponse multipart_reference =
      Send(service, "POST", "/v1/videos", reference_body, {},
           "multipart/form-data; boundary=strix-openai-video-boundary");
  Check(multipart_reference.status == 400 &&
            multipart_reference.body.find("unsupported_input_reference") !=
                std::string::npos,
        "multipart first-frame conditioning remains explicit future work");

  const HttpResponse unsupported_media =
      Send(service, "POST", "/v1/videos", "model=minimax-h3", {},
           "application/x-www-form-urlencoded");
  Check(unsupported_media.status == 415 &&
            unsupported_media.body.find("unsupported_media_type") !=
                std::string::npos,
        "unsupported create content types fail explicitly");
}

void TestQueueSaturation(const std::filesystem::path& root) {
  FakeRunner runner(true);
  VideoJobService service(Options(root, &runner));
  const std::string body =
      R"({"model":"minimax-h3","prompt":"x","size":"512x512","seconds":"1","strix":{"preset":"fast"}})";
  const HttpResponse first = Send(service, "POST", "/v1/videos", body);
  runner.WaitForCalls(1);
  const HttpResponse second = Send(service, "POST", "/v1/videos", body);
  const HttpResponse third = Send(service, "POST", "/v1/videos", body);
  Check(first.status == 202 && second.status == 202 && third.status == 429,
        "one active and one queued request enforce bounded admission");
  Check(std::ranges::any_of(
            third.headers,
            [](const auto& header) { return header.first == "Retry-After"; }),
        "queue saturation supplies Retry-After");
  Check(Send(service, "DELETE", "/v1/videos/" + CreatedId(first)).status == 200,
        "active request can be cancelled through DELETE");
  runner.WaitForCalls(2);
  runner.Release();
}

}  // namespace

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("strix-video-api-test-" + std::to_string(getpid()));
  std::error_code ignored;
  std::filesystem::remove_all(base, ignored);
  std::filesystem::create_directories(base);
  TestLifecycleAndRange(base / "lifecycle");
  TestValidation(base / "validation");
  TestQueueSaturation(base / "queue");
  std::filesystem::remove_all(base, ignored);
  std::cout << "Video API contract tests passed.\n";
  return 0;
}
