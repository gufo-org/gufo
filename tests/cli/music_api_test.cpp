#include "src/cli/serve/music_api.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/cli/serve/media_jobs.hpp"
#include "src/core/json.hpp"
#include "src/core/platform/os_info.hpp"

// The `gufo serve music` routes and the media job store against a fake
// runner: request mapping, the job lifecycle, files, cancellation, restart
// recovery and expiry. No sockets and no model.

namespace {

using gufo::ext::MusicCapabilities;
using gufo::ext::MusicDurationControl;
using gufo::ext::MusicFinish;
using gufo::ext::MusicRequest;
using gufo::ext::MusicResult;
using gufo::ext::MusicSupport;
using gufo::server::HandleMusicApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::MediaJobContext;
using gufo::server::MediaJobStore;
using gufo::server::MediaJobStoreOptions;
using gufo::server::MusicService;
using gufo::server::MusicServiceOptions;
namespace json = gufo::json;

constexpr std::uint32_t kRate = 8000;
constexpr std::size_t kFrames = 800;
constexpr std::string_view kScore = "X:1\nT:fake\nK:C\nV:Vocal\nCDEF|\n";

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL music_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition)
    Fail(message);
}

// Records every request; optionally holds each generation until released or
// cancelled.
struct FakeState {
  std::mutex mutex;
  std::condition_variable condition;
  bool blocked{false};
  bool released{false};
  bool fail{false};
  bool truncate{false};
  int started{0};
  int cancelled{0};
  MusicRequest latest;

  void WaitStarted(int expected) {
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(lock, std::chrono::seconds(5),
                            [&] { return started >= expected; }))
      Fail("timed out waiting for the fake runner to start");
  }
  void Release() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      released = true;
    }
    condition.notify_all();
  }
};

class FakeRunner final : public gufo::ext::MusicRunner {
public:
  explicit FakeRunner(std::shared_ptr<FakeState> state)
      : state_(std::move(state)) {}

  bool Generate(const MusicRequest& request,
                const gufo::ext::CancellationCheck& is_cancelled,
                const gufo::ext::MusicProgress& progress, MusicResult* result,
                std::string* error) override {
    {
      const std::lock_guard<std::mutex> lock(state_->mutex);
      ++state_->started;
      state_->latest = request;
    }
    state_->condition.notify_all();
    if (!progress("plan", 1, 0)) {
      *error = "cancelled";
      return false;
    }
    {
      std::unique_lock<std::mutex> lock(state_->mutex);
      while (state_->blocked && !state_->released && !is_cancelled())
        state_->condition.wait_for(lock, std::chrono::milliseconds(5));
    }
    if (is_cancelled() || !progress("acoustic", 2, 4)) {
      const std::lock_guard<std::mutex> lock(state_->mutex);
      ++state_->cancelled;
      *error = "cancelled";
      return false;
    }
    if (state_->fail) {
      *error = "fake failure";
      return false;
    }
    result->score = request.score.empty() ? std::string(kScore) : request.score;
    result->score_format = "abc";
    result->finish =
        state_->truncate ? MusicFinish::kTruncated : MusicFinish::kComplete;
    result->telemetry_json = R"({"fake_ms":1})";
    if (!request.plan_only) {
      result->audio.sample_rate = kRate;
      result->audio.channels = 2;
      result->audio.samples.resize(kFrames * 2);
      for (std::size_t i = 0; i < result->audio.samples.size(); ++i)
        result->audio.samples[i] = (i % 2 == 0 ? 0.25F : -0.25F);
    }
    return true;
  }

  MusicCapabilities Capabilities() const override {
    MusicCapabilities capabilities;
    capabilities.lyrics = MusicSupport::kOptional;
    capabilities.language = MusicSupport::kUnsupported;
    capabilities.tempo_key_meter = MusicSupport::kUnsupported;
    capabilities.score = MusicSupport::kOptional;
    capabilities.score_formats = {"abc"};
    capabilities.score_output = true;
    capabilities.plan_only = true;
    capabilities.guidance = true;
    capabilities.steps = true;
    capabilities.duration = MusicDurationControl::kMaximum;
    capabilities.max_seconds = 360.0;
    return capabilities;
  }

  bool ValidateRequest(const MusicRequest& request,
                       std::string* error) const override {
    if (request.provider_json.find("\"bad\"") != std::string::npos) {
      *error = "unknown provider option 'bad'";
      return false;
    }
    return true;
  }

  std::uint32_t SampleRate() const override { return kRate; }
  std::string ModelId() const override { return "fake-music"; }

private:
  std::shared_ptr<FakeState> state_;
};

std::unique_ptr<MusicService> Service(const std::filesystem::path& root,
                                      std::shared_ptr<FakeState> state,
                                      std::size_t queue_capacity = 4) {
  MusicServiceOptions options;
  options.storage_root = root;
  options.queue_capacity = queue_capacity;
  options.model_aliases = {"music-alias"};
  auto service = std::make_unique<MusicService>(
      std::make_unique<FakeRunner>(std::move(state)), std::move(options));
  Check(service->ready(), "service ready: " + service->initialization_error());
  return service;
}

HttpResponse Send(MusicService& service, std::string method, std::string path,
                  std::string body = {}, std::string range = {}) {
  HttpRequest request;
  request.method = std::move(method);
  request.path = std::move(path);
  request.body = std::move(body);
  if (!range.empty())
    request.headers.emplace_back("RaNgE", std::move(range));
  return HandleMusicApiRequest(request, service);
}

json::Value Body(const HttpResponse& response) {
  try {
    return json::parse(response.body);
  } catch (const std::exception& exception) {
    Fail(std::string("response is not JSON: ") + exception.what() + ": " +
         response.body);
  }
}

std::string Header(const HttpResponse& response, std::string_view name) {
  for (const auto& [key, value] : response.headers)
    if (key == name)
      return value;
  return {};
}

void ExpectError(const HttpResponse& response, int status,
                 std::string_view code, std::string_view param,
                 const std::string& label) {
  Check(response.status == status,
        label + ": status " + std::to_string(response.status) +
            " != " + std::to_string(status) + " " + response.body);
  const json::Value body = Body(response);
  const json::Value* error = body.find("error");
  Check(error != nullptr && error->is_object(), label + ": no error object");
  Check(error->member_str("code") == code,
        label + ": code '" + error->member_str("code") + "' != '" +
            std::string(code) + "'");
  if (!param.empty())
    Check(error->member_str("param") == param,
          label + ": param '" + error->member_str("param") + "' != '" +
              std::string(param) + "'");
}

std::string Create(MusicService& service, const std::string& body) {
  const HttpResponse response = Send(service, "POST", "/v1/audio/music", body);
  Check(response.status == 202, "create: status " +
                                    std::to_string(response.status) + " " +
                                    response.body);
  const json::Value job = Body(response);
  Check(job.member_str("object") == "audio.music", "create: object");
  Check(job.member_str("status") == "queued", "create: status queued");
  const std::string id = job.member_str("id");
  Check(id.starts_with("music_") && id.size() == 6 + 24,
        "create: job id " + id);
  return id;
}

json::Value WaitFor(MusicService& service, const std::string& id,
                    std::string_view status) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  json::Value job;
  while (std::chrono::steady_clock::now() < deadline) {
    const HttpResponse response = Send(service, "GET", "/v1/audio/music/" + id);
    Check(response.status == 200, "poll: status " + response.body);
    job = Body(response);
    if (job.member_str("status") == status)
      return job;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  Fail("job " + id + " never reached " + std::string(status) +
       "; last: " + job.dump());
}

std::string ReadAll(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void TestCapabilitiesJson() {
  auto state = std::make_shared<FakeState>();
  FakeRunner runner(state);
  const json::Value value =
      gufo::server::MusicCapabilitiesJson(runner.Capabilities());
  Check(value.member_str("lyrics") == "optional", "capabilities: lyrics");
  Check(value.member_str("tempo_key_meter") == "unsupported",
        "capabilities: tempo_key_meter");
  Check(value.member_str("score") == "optional", "capabilities: score");
  Check(value.member_str("duration") == "maximum", "capabilities: duration");
  Check(value.member_double("max_seconds") == 360.0,
        "capabilities: max_seconds");
  const json::Value* formats = value.find("score_formats");
  Check(formats != nullptr && formats->is_array() && formats->size() == 1,
        "capabilities: score_formats");
  Check(gufo::server::IsMusicApiPath("/v1/audio/music") &&
            gufo::server::IsMusicApiPath("/v1/audio/music/x/content") &&
            !gufo::server::IsMusicApiPath("/v1/audio/musical") &&
            !gufo::server::IsMusicApiPath("/v1/audio/speech"),
        "IsMusicApiPath");
}

void TestValidation(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  auto service = Service(root, state);
  auto& s = *service;
  const auto post = [&](const std::string& body) {
    return Send(s, "POST", "/v1/audio/music", body);
  };
  ExpectError(post("{"), 400, "invalid_json", "", "invalid JSON");
  ExpectError(post("[]"), 400, "invalid_request", "", "array body");
  ExpectError(post(R"({"prompt":"pop","voice":"x"})"), 400,
              "unsupported_parameter", "voice", "unknown top-level field");
  ExpectError(post(R"({"prompt":"pop","gufo":{"cfg":1}})"), 400,
              "unsupported_parameter", "cfg", "unknown gufo field");
  ExpectError(post(R"({"model":"other","prompt":"pop"})"), 404,
              "model_not_found", "model", "unknown model");
  ExpectError(post(R"({"prompt":""})"), 400, "invalid_request", "prompt",
              "no prompt or lyrics");
  ExpectError(post(R"({"prompt":7})"), 400, "invalid_request", "prompt",
              "prompt type");
  ExpectError(post(R"({"prompt":"pop","lyrics":"la","instrumental":true})"),
              400, "invalid_request", "lyrics", "instrumental with lyrics");
  ExpectError(post(R"({"prompt":"pop","seconds":"10"})"), 400,
              "invalid_request", "seconds", "seconds type");
  ExpectError(post(R"({"prompt":"pop","seconds":0})"), 400,
              "unsupported_parameter", "seconds", "zero seconds");
  ExpectError(post(R"({"prompt":"pop","seconds":361})"), 400,
              "unsupported_parameter", "seconds", "seconds over the maximum");
  ExpectError(post(R"({"prompt":"pop","response_format":"mp3"})"), 400,
              "invalid_request", "response_format", "response format");
  ExpectError(post(R"({"prompt":"pop","gufo":{"bpm":120}})"), 400,
              "unsupported_parameter", "bpm", "bpm unsupported");
  ExpectError(post(R"({"prompt":"pop","gufo":{"key":"C major"}})"), 400,
              "unsupported_parameter", "key", "key unsupported");
  ExpectError(post(R"({"prompt":"pop","gufo":{"time_signature":"4/4"}})"), 400,
              "unsupported_parameter", "time_signature", "meter unsupported");
  for (const char* meter : {"4", "x/4", "4/x", "4/3", "/4", "4/", "4/4/4"})
    ExpectError(
        post(std::string(R"({"prompt":"pop","gufo":{"time_signature":")") +
             meter + "\"}}"),
        400, "invalid_request", "time_signature",
        std::string("meter syntax ") + meter);
  ExpectError(post(R"({"prompt":"pop","gufo":{"language":"en"}})"), 400,
              "unsupported_parameter", "language", "language unsupported");
  ExpectError(post(R"({"prompt":"pop","gufo":{"seed":-1}})"), 400,
              "invalid_request", "seed", "negative seed");
  ExpectError(post(R"({"prompt":"pop","gufo":{"seed":1.5}})"), 400,
              "invalid_request", "seed", "fractional seed");
  ExpectError(post(R"({"prompt":"pop","gufo":{"steps":0}})"), 400,
              "invalid_request", "steps", "zero steps");
  ExpectError(post(R"({"prompt":"pop","gufo":{"score":"X:1"}})"), 400,
              "invalid_request", "score", "score shape");
  ExpectError(
      post(
          R"({"prompt":"pop","gufo":{"score":{"format":"musicxml","text":"<x/>"}}})"),
      400, "unsupported_parameter", "score", "score format");
  ExpectError(
      post(
          R"({"prompt":"pop","gufo":{"plan_only":true,"score":{"text":"X:1"}}})"),
      400, "unsupported_parameter", "plan_only", "plan_only with a score");
  ExpectError(post(R"({"prompt":"pop","gufo":{"options":{"bad":1}}})"), 400,
              "invalid_request", "gufo", "provider option rejected");
  ExpectError(post(R"({"prompt":"pop","gufo":{"options":[]}})"), 400,
              "invalid_request", "options", "options type");

  ExpectError(Send(s, "GET", "/v1/audio/music"), 405, "method_not_allowed", "",
              "GET on the collection");
  ExpectError(Send(s, "PUT", "/v1/audio/music/music_0"), 405,
              "method_not_allowed", "", "PUT on a job");
  ExpectError(Send(s, "GET", "/v1/audio/music/music_missing"), 404,
              "job_not_found", "", "unknown job");
  ExpectError(Send(s, "DELETE", "/v1/audio/music/music_missing"), 404,
              "job_not_found", "", "delete unknown job");
  ExpectError(Send(s, "GET", "/v1/audio/music/music_missing/content"), 404,
              "not_found", "", "content of unknown job");
  ExpectError(Send(s, "POST", "/v1/audio/music/music_missing/content"), 405,
              "method_not_allowed", "", "POST on content");
  {
    const std::lock_guard<std::mutex> lock(state->mutex);
    Check(state->started == 0, "validation: no request reached the runner");
  }
}

void TestLifecycle(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  auto service = Service(root, state);
  auto& s = *service;
  const std::string prompt = "secret style prompt";
  const std::string id =
      Create(s, R"({"model":"music-alias","prompt":")" + prompt +
                    R"(","lyrics":"[verse]\nla la","seconds":30,)"
                    R"("gufo":{"seed":42,"guidance_scale":1.5,"steps":8,)"
                    R"("options":{"cot":"full"}}})");
  const json::Value job = WaitFor(s, id, "completed");
  Check(job.member_str("model") == "music-alias", "lifecycle: model alias");
  Check(job.member_double("progress") == 100, "lifecycle: progress");
  Check(std::abs(job.member_double("seconds") -
                 static_cast<double>(kFrames) / kRate) < 1e-9,
        "lifecycle: seconds");
  Check(job.member_double("completed_at") > 0 &&
            job.member_double("expires_at") > job.member_double("completed_at"),
        "lifecycle: timestamps");
  const json::Value* gufo = job.find("gufo");
  Check(gufo != nullptr && gufo->member_str("finish") == "complete",
        "lifecycle: finish");
  Check(gufo->member_double("seed") == 42, "lifecycle: seed echoed");
  const json::Value* has_audio = gufo->find("has_audio");
  const json::Value* has_score = gufo->find("has_score");
  Check(has_audio != nullptr && has_audio->as_bool() && has_score != nullptr &&
            has_score->as_bool(),
        "lifecycle: files flagged");
  {
    const std::lock_guard<std::mutex> lock(state->mutex);
    const MusicRequest& request = state->latest;
    Check(request.prompt == prompt, "lifecycle: prompt reached the runner");
    Check(request.lyrics == "[verse]\nla la", "lifecycle: lyrics");
    Check(request.duration_seconds && *request.duration_seconds == 30.0,
          "lifecycle: seconds reached the runner");
    Check(request.seed == 42, "lifecycle: seed reached the runner");
    Check(request.guidance && *request.guidance == 1.5F, "lifecycle: guidance");
    Check(request.steps && *request.steps == 8, "lifecycle: steps");
    Check(json::parse(request.provider_json).member_str("cot") == "full",
          "lifecycle: provider options");
  }

  const HttpResponse content =
      Send(s, "GET", "/v1/audio/music/" + id + "/content");
  Check(content.status == 200, "content: status " + content.body);
  Check(Header(content, "Content-Type") == "audio/wav", "content: type");
  Check(Header(content, "Accept-Ranges") == "bytes", "content: ranges");
  Check(content.body.size() == 44 + kFrames * 2 * 2,
        "content: PCM16 size " + std::to_string(content.body.size()));
  Check(content.body.starts_with("RIFF") && content.body.substr(8, 4) == "WAVE",
        "content: RIFF header");

  const HttpResponse head =
      Send(s, "GET", "/v1/audio/music/" + id + "/content", {}, "bytes=0-3");
  Check(head.status == 206 && head.body == "RIFF", "range: first bytes");
  Check(Header(head, "Content-Range") ==
            "bytes 0-3/" + std::to_string(content.body.size()),
        "range: Content-Range");
  const HttpResponse tail =
      Send(s, "GET", "/v1/audio/music/" + id + "/content", {}, "bytes=-2");
  Check(tail.status == 206 &&
            tail.body == content.body.substr(content.body.size() - 2),
        "range: suffix");
  const HttpResponse open =
      Send(s, "GET", "/v1/audio/music/" + id + "/content", {}, "bytes=40-");
  Check(open.status == 206 && open.body == content.body.substr(40),
        "range: open end");
  for (const char* range : {"bytes=999999-", "bytes=5-2", "bytes=0-1,4-5",
                            "items=0-1", "bytes=-0", "bytes=x-"}) {
    const HttpResponse bad =
        Send(s, "GET", "/v1/audio/music/" + id + "/content", {}, range);
    Check(bad.status == 416, std::string("range: 416 for ") + range);
    Check(Header(bad, "Content-Range") ==
              "bytes */" + std::to_string(content.body.size()),
          std::string("range: unsatisfied Content-Range for ") + range);
  }

  const HttpResponse score = Send(s, "GET", "/v1/audio/music/" + id + "/score");
  Check(score.status == 200 && score.body == kScore, "score: body");
  Check(Header(score, "Content-Type").starts_with("text/vnd.abc"),
        "score: type");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + id + "/stems"), 404,
              "not_found", "", "unknown job resource");

  // The persisted job keeps hashes of the prompt and lyrics, not the text.
  const std::string saved = ReadAll(root / id / "job.json");
  Check(!saved.empty(), "persisted job.json exists");
  Check(saved.find(prompt) == std::string::npos &&
            saved.find("la la") == std::string::npos,
        "persisted job.json holds no prompt or lyrics text");
  Check(saved.find("prompt_sha256") != std::string::npos,
        "persisted job.json holds the prompt hash");

  const HttpResponse deleted = Send(s, "DELETE", "/v1/audio/music/" + id);
  Check(deleted.status == 200, "delete completed: status");
  const json::Value deleted_body = Body(deleted);
  Check(deleted_body.member_str("object") == "audio.music.deleted",
        "delete completed: object");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + id), 404, "job_not_found", "",
              "deleted job");
  Check(!std::filesystem::exists(root / id), "delete removed the directory");
}

void TestFloatAndTruncation(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  state->truncate = true;
  auto service = Service(root, state);
  const std::string id =
      Create(*service, R"({"prompt":"pop","response_format":"wav_float"})");
  const json::Value job = WaitFor(*service, id, "completed");
  Check(job.find("gufo")->member_str("finish") == "truncated",
        "truncated: finish");
  const HttpResponse content =
      Send(*service, "GET", "/v1/audio/music/" + id + "/content");
  Check(content.status == 200 &&
            content.body.size() > kFrames * 2 * sizeof(float) &&
            content.body.size() < kFrames * 2 * sizeof(float) + 128,
        "wav_float: float32 payload " + std::to_string(content.body.size()));
  // WAVE_FORMAT_IEEE_FLOAT (3) or WAVE_FORMAT_EXTENSIBLE around it.
  const auto format = static_cast<unsigned char>(content.body[20]);
  Check(format == 3 || format == 0xFE, "wav_float: format tag");
}

void TestPlanOnlyAndScoreEdit(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  auto service = Service(root, state);
  auto& s = *service;
  const std::string plan =
      Create(s, R"({"prompt":"folk","gufo":{"plan_only":true}})");
  const json::Value job = WaitFor(s, plan, "completed");
  const json::Value* gufo = job.find("gufo");
  Check(
      !gufo->find("has_audio")->as_bool() && gufo->find("has_score")->as_bool(),
      "plan-only: files");
  Check(job.find("seconds")->is_null(), "plan-only: no seconds");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + plan + "/content"), 404,
              "not_found", "", "plan-only: no audio");
  const HttpResponse score =
      Send(s, "GET", "/v1/audio/music/" + plan + "/score");
  Check(score.status == 200 && score.body == kScore, "plan-only: score");

  // Edit the score and play it back.
  const std::string edited = "X:1\nT:edited\nK:G\nGABc|\n";
  json::Value body = json::Value::object();
  body["prompt"] = "folk";
  json::Value options = json::Value::object();
  json::Value given = json::Value::object();
  given["format"] = "abc";
  given["text"] = edited;
  options["score"] = std::move(given);
  body["gufo"] = std::move(options);
  const std::string play = Create(s, body.dump());
  WaitFor(s, play, "completed");
  {
    const std::lock_guard<std::mutex> lock(state->mutex);
    Check(state->latest.score == edited && state->latest.score_format == "abc",
          "score edit: score reached the runner");
    Check(!state->latest.plan_only, "score edit: plays audio");
  }
  const HttpResponse played =
      Send(s, "GET", "/v1/audio/music/" + play + "/score");
  Check(played.status == 200 && played.body == edited,
        "score edit: score echoed");
}

void TestFailure(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  state->fail = true;
  auto service = Service(root, state);
  const std::string id = Create(*service, R"({"prompt":"pop"})");
  const json::Value job = WaitFor(*service, id, "failed");
  const json::Value* error = job.find("error");
  Check(error != nullptr && error->member_str("code") == "generation_failed" &&
            error->member_str("message") == "fake failure",
        "failure: error " + job.dump());
  ExpectError(Send(*service, "GET", "/v1/audio/music/" + id + "/content"), 409,
              "job_not_ready", "", "failure: content");
  Check(!std::filesystem::exists(root / id / "audio.wav") &&
            !std::filesystem::exists(root / id / "score.abc"),
        "failure: no published files");
}

void TestQueueAndCancellation(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  state->blocked = true;
  auto service = Service(root, state, /*queue_capacity=*/1);
  auto& s = *service;
  const std::string running = Create(s, R"({"prompt":"one"})");
  state->WaitStarted(1);
  const json::Value progress = WaitFor(s, running, "in_progress");
  Check(progress.find("gufo")->member_str("phase") == "plan",
        "queue: phase reported");
  Check(progress.find("gufo")->member_double("total") == 0,
        "queue: unknown total reported as 0");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + running + "/content"), 409,
              "job_not_ready", "", "queue: content before completion");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + running + "/score"), 409,
              "job_not_ready", "", "queue: score before completion");

  const std::string queued = Create(s, R"({"prompt":"two"})");
  ExpectError(Send(s, "POST", "/v1/audio/music", R"({"prompt":"three"})"), 429,
              "queue_full", "", "queue: full");

  // Deleting a queued job frees its slot without running it.
  Check(Send(s, "DELETE", "/v1/audio/music/" + queued).status == 200,
        "queue: delete queued");
  Check(!std::filesystem::exists(root / queued), "queue: queued dir removed");
  const std::string replacement = Create(s, R"({"prompt":"four"})");

  // Deleting the running job cancels it; the worker moves on.
  Check(Send(s, "DELETE", "/v1/audio/music/" + running).status == 200,
        "cancel: delete running");
  ExpectError(Send(s, "GET", "/v1/audio/music/" + running), 404,
              "job_not_found", "", "cancel: running job gone");
  state->WaitStarted(2);
  {
    const std::lock_guard<std::mutex> lock(state->mutex);
    Check(state->cancelled == 1, "cancel: runner saw the cancellation");
    Check(state->latest.prompt == "four",
          "cancel: the deleted queued job never ran");
  }
  state->Release();
  WaitFor(s, replacement, "completed");
  Check(!std::filesystem::exists(root / running),
        "cancel: running dir removed after the worker returned");
}

void TestRestartRecovery(const std::filesystem::path& root) {
  auto state = std::make_shared<FakeState>();
  std::string completed;
  std::string interrupted;
  {
    auto service = Service(root, state);
    completed = Create(*service, R"({"prompt":"keep"})");
    WaitFor(*service, completed, "completed");
    state->blocked = true;
    interrupted = Create(*service, R"({"prompt":"lose"})");
    state->WaitStarted(2);
    // The service shuts down with the job still running.
  }
  Check(std::filesystem::exists(root / interrupted),
        "restart: interrupted job left on disk");
  // Debris a crash can leave: a stray partial file and a foreign directory.
  {
    std::ofstream(root / completed / "audio.wav.gufo-partial-1-1") << "x";
    std::filesystem::create_directories(root / "not-a-job");
  }
  state->blocked = false;
  auto service = Service(root, state);
  const HttpResponse job =
      Send(*service, "GET", "/v1/audio/music/" + completed);
  Check(job.status == 200 && Body(job).member_str("status") == "completed",
        "restart: completed job recovered");
  const HttpResponse content =
      Send(*service, "GET", "/v1/audio/music/" + completed + "/content");
  Check(content.status == 200 && content.body.starts_with("RIFF"),
        "restart: recovered content");
  ExpectError(Send(*service, "GET", "/v1/audio/music/" + interrupted), 404,
              "job_not_found", "", "restart: interrupted job dropped");
  Check(!std::filesystem::exists(root / interrupted),
        "restart: interrupted directory removed");
  Check(std::filesystem::exists(root / "not-a-job"),
        "restart: foreign directories untouched");
  Check(
      !std::filesystem::exists(root / completed / "audio.wav.gufo-partial-1-1"),
      "restart: partial writes removed");
}

// The store directly, with a fake clock for expiry.
void TestStoreExpiry(const std::filesystem::path& root) {
  auto now = std::make_shared<std::int64_t>(1000);
  auto counter = std::make_shared<int>(0);
  const auto options = [&] {
    MediaJobStoreOptions o;
    o.root = root;
    o.id_prefix = "media";
    o.schema = "gufo.media-test.v1";
    o.queue_capacity = 2;
    o.ttl = std::chrono::seconds(60);
    o.now = [now] { return *now; };
    o.id_factory = [counter] {
      std::ostringstream id;
      id << "media_" << std::setw(24) << std::setfill('0') << std::hex
         << ++*counter;
      return id.str();
    };
    return o;
  };
  const auto wait_done = [](MediaJobStore& store, const std::string& id) {
    for (int i = 0; i < 2500; ++i) {
      const auto job = store.Get(id);
      if (job && job->status != gufo::server::MediaJobStatus::kQueued &&
          job->status != gufo::server::MediaJobStatus::kInProgress)
        return *job;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Fail("store job never finished");
  };
  std::string kept;
  std::string expired;
  {
    MediaJobStore store(options());
    Check(store.ready(), "store ready");
    auto work = [](MediaJobContext& context, std::string*) {
      return context.WriteFile("out.bin", "application/octet-stream", "data") &&
             !context.WriteFile("../escape", "x", "x") &&
             !context.WriteFile("job.json", "x", "x") &&
             !context.WriteFile(".hidden", "x", "x");
    };
    const auto first = store.Create("m", json::Value::object(), work);
    Check(first.result == gufo::server::MediaJobResult::kOk && first.job,
          "store: create");
    expired = first.job->id;
    const auto done = wait_done(store, expired);
    Check(done.status == gufo::server::MediaJobStatus::kCompleted,
          "store: completed (published names are checked)");
    Check(done.expires_at == 1060, "store: expiry from the fake clock");
    *now = 1030;
    const auto second = store.Create("m", json::Value::object(), work);
    kept = second.job->id;
    wait_done(store, kept);
    gufo::server::MediaJobFile file;
    Check(store.File(expired, "out.bin", &file) ==
                  gufo::server::MediaJobResult::kOk &&
              file.bytes == 4,
          "store: file");
    Check(store.File(expired, "missing", &file) ==
              gufo::server::MediaJobResult::kNotFound,
          "store: missing file");
    *now = 1060;
    Check(!store.Get(expired), "store: expired job swept");
    Check(!std::filesystem::exists(root / expired), "store: expired dir gone");
    Check(store.Get(kept).has_value(), "store: unexpired job kept");
  }
  *now = 1089;
  {
    MediaJobStore store(options());
    Check(store.Get(kept).has_value(), "store: recovered before expiry");
  }
  *now = 1090;
  {
    MediaJobStore store(options());
    Check(!store.Get(kept), "store: recovery drops expired jobs");
    Check(!std::filesystem::exists(root / kept),
          "store: recovery removes expired dirs");
  }
  {
    MediaJobStoreOptions bad = options();
    bad.queue_capacity = 0;
    MediaJobStore store(std::move(bad));
    Check(!store.ready() && !store.initialization_error().empty(),
          "store: invalid configuration");
    Check(store.Create("m", json::Value::object(), {}).result ==
              gufo::server::MediaJobResult::kIoError,
          "store: create on an unready store");
  }
}

}  // namespace

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("gufo-music-api-test-" + std::to_string(gufo::platform::ProcessId()));
  std::error_code ignored;
  std::filesystem::remove_all(base, ignored);
  std::filesystem::create_directories(base);
  TestCapabilitiesJson();
  TestValidation(base / "validation");
  TestLifecycle(base / "lifecycle");
  TestFloatAndTruncation(base / "float");
  TestPlanOnlyAndScoreEdit(base / "plan");
  TestFailure(base / "failure");
  TestQueueAndCancellation(base / "queue");
  TestRestartRecovery(base / "restart");
  TestStoreExpiry(base / "store");
  std::filesystem::remove_all(base, ignored);
  std::cout << "Music API contract tests passed.\n";
  return 0;
}
