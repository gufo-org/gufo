#include "src/cli/serve/music_api.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>

#include "src/cli/serve/music_wav.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kMusicPath = "/v1/audio/music";
constexpr std::string_view kSchema = "gufo.music-job.v1";
constexpr std::string_view kAudioFile = "audio.wav";
constexpr std::string_view kScoreFile = "score.abc";

HttpResponse Json(int status, std::string reason, const json::Value& body) {
  return {.status = status,
          .reason = std::move(reason),
          .body = body.dump(),
          .headers = {},
          .streaming_body = {}};
}

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code, std::string param = {}) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status >= 500 ? "server_error" : "invalid_request_error";
  error["code"] = std::move(code);
  error["param"] = param.empty() ? json::Value() : json::Value(param);
  root["error"] = std::move(error);
  return Json(status, std::move(reason), root);
}

HttpResponse BadRequest(std::string message, std::string param,
                        std::string code = "invalid_request") {
  return Error(400, "Bad Request", std::move(message), std::move(code),
               std::move(param));
}

std::string Sha256(std::string_view text) {
  return crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}

std::string_view SupportName(ext::MusicSupport support) {
  switch (support) {
    case ext::MusicSupport::kUnsupported:
      return "unsupported";
    case ext::MusicSupport::kOptional:
      return "optional";
    case ext::MusicSupport::kRequired:
      break;
  }
  return "required";
}

// Coarse progress: the AR phases have no total until the model stops.
int ProgressPercent(const MediaJobSnapshot& job) {
  const auto scaled = [&](int begin, int end) {
    if (job.total == 0)
      return begin;
    return begin +
           static_cast<int>((end - begin) * std::min(job.completed, job.total) /
                            job.total);
  };
  if (job.status == MediaJobStatus::kCompleted)
    return 100;
  if (job.status == MediaJobStatus::kQueued)
    return 0;
  if (job.phase == "plan")
    return 2;
  if (job.phase == "semantic")
    return 10;
  if (job.phase == "acoustic")
    return scaled(50, 90);
  if (job.phase == "decode")
    return scaled(90, 99);
  return 1;
}

json::Value JobJson(const MediaJobSnapshot& job) {
  json::Value root = json::Value::object();
  root["id"] = job.id;
  root["object"] = "audio.music";
  root["model"] = job.model;
  root["status"] = std::string(ToString(job.status));
  root["progress"] = ProgressPercent(job);
  root["created_at"] = static_cast<long long>(job.created_at);
  root["completed_at"] =
      job.completed_at > 0
          ? json::Value(static_cast<long long>(job.completed_at))
          : json::Value();
  root["expires_at"] = job.expires_at > 0
                           ? json::Value(static_cast<long long>(job.expires_at))
                           : json::Value();
  const json::Value* seconds = job.metadata.find("audio_seconds");
  root["seconds"] = seconds != nullptr ? *seconds : json::Value();
  if (job.error_code.empty()) {
    root["error"] = json::Value();
  } else {
    json::Value error = json::Value::object();
    error["code"] = job.error_code;
    error["message"] = job.error_message;
    root["error"] = std::move(error);
  }
  json::Value gufo = json::Value::object();
  gufo["phase"] = job.phase;
  gufo["completed"] = static_cast<long long>(job.completed);
  gufo["total"] = static_cast<long long>(job.total);
  for (const char* key : {"finish", "score_format", "plan_only", "seed"}) {
    if (const json::Value* value = job.metadata.find(key))
      gufo[key] = *value;
  }
  gufo["has_audio"] = job.files.contains(kAudioFile);
  gufo["has_score"] = job.files.contains(kScoreFile);
  root["gufo"] = std::move(gufo);
  return root;
}

bool HasOnly(const json::Value& value,
             std::initializer_list<std::string_view> allowed,
             std::string* unknown) {
  for (const auto& [key, member] : value.members()) {
    if (std::ranges::find(allowed, key) == allowed.end()) {
      *unknown = key;
      return false;
    }
  }
  return true;
}

// Present and not null.
const json::Value* Given(const json::Value& object, const std::string& key) {
  const json::Value* value = object.find(key);
  return value == nullptr || value->is_null() ? nullptr : value;
}

struct ParsedRequest {
  ext::MusicRequest music;
  std::string model;
  bool float32{false};
};

std::optional<HttpResponse> ParseRequest(const HttpRequest& http,
                                         MusicService& service,
                                         ParsedRequest* parsed) {
  json::Value body;
  try {
    body = json::parse(http.body);
  } catch (const std::exception& exception) {
    return BadRequest(exception.what(), {}, "invalid_json");
  }
  if (!body.is_object())
    return BadRequest("the request body must be a JSON object", {});
  std::string unknown;
  if (!HasOnly(body,
               {"model", "prompt", "lyrics", "instrumental", "seconds",
                "response_format", "gufo"},
               &unknown))
    return BadRequest("unknown field '" + unknown + "'", unknown,
                      "unsupported_parameter");
  auto& request = parsed->music;
  parsed->model = service.model_id();
  if (const json::Value* model = Given(body, "model")) {
    if (!model->is_string() || !service.serves_model(model->str()))
      return Error(404, "Not Found", "unknown music model", "model_not_found",
                   "model");
    parsed->model = model->str();
  }
  const auto string_field =
      [&](const json::Value& object, const std::string& key,
          std::string* output) -> std::optional<HttpResponse> {
    if (const json::Value* value = Given(object, key)) {
      if (!value->is_string())
        return BadRequest("'" + key + "' must be a string", key);
      *output = value->str();
    }
    return std::nullopt;
  };
  if (auto error = string_field(body, "prompt", &request.prompt))
    return error;
  if (auto error = string_field(body, "lyrics", &request.lyrics))
    return error;
  if (const json::Value* instrumental = Given(body, "instrumental")) {
    if (!instrumental->is_bool())
      return BadRequest("'instrumental' must be a boolean", "instrumental");
    if (instrumental->as_bool() && !request.lyrics.empty())
      return BadRequest("an instrumental request has no lyrics", "lyrics");
  }
  if (request.prompt.empty() && request.lyrics.empty())
    return BadRequest("give a prompt or lyrics", "prompt");
  if (const json::Value* seconds = Given(body, "seconds")) {
    if (!seconds->is_number())
      return BadRequest("'seconds' must be a number", "seconds");
    request.duration_seconds = seconds->as_double(0.0);
  }
  if (const json::Value* format = Given(body, "response_format")) {
    if (!format->is_string() ||
        (format->str() != "wav" && format->str() != "wav_float"))
      return BadRequest("response_format must be wav or wav_float",
                        "response_format");
    parsed->float32 = format->str() == "wav_float";
  }
  if (const json::Value* gufo = Given(body, "gufo")) {
    if (!gufo->is_object())
      return BadRequest("'gufo' must be an object", "gufo");
    if (!HasOnly(*gufo,
                 {"seed", "score", "plan_only", "guidance_scale", "steps",
                  "bpm", "key", "time_signature", "language", "options"},
                 &unknown))
      return BadRequest("unknown field 'gufo." + unknown + "'", unknown,
                        "unsupported_parameter");
    if (const json::Value* seed = Given(*gufo, "seed")) {
      const double value = seed->is_number() ? seed->as_double(-1) : -1;
      if (value < 0 || std::floor(value) != value || value > 9007199254740991.0)
        return BadRequest("seed must be a non-negative integer", "seed");
      request.seed = static_cast<std::uint64_t>(value);
    }
    if (const json::Value* score = Given(*gufo, "score")) {
      if (!score->is_object() || !Given(*score, "text") ||
          !Given(*score, "text")->is_string())
        return BadRequest("score must be {\"format\", \"text\"}", "score");
      request.score = Given(*score, "text")->str();
      request.score_format = score->member_str("format", "abc");
    }
    if (const json::Value* plan_only = Given(*gufo, "plan_only")) {
      if (!plan_only->is_bool())
        return BadRequest("plan_only must be a boolean", "plan_only");
      request.plan_only = plan_only->as_bool();
    }
    if (const json::Value* guidance = Given(*gufo, "guidance_scale")) {
      if (!guidance->is_number())
        return BadRequest("guidance_scale must be a number", "guidance_scale");
      request.guidance = static_cast<float>(guidance->as_double(0.0));
    }
    if (const json::Value* steps = Given(*gufo, "steps")) {
      const double value = steps->is_number() ? steps->as_double(0) : 0;
      if (std::floor(value) != value || value < 1 || value > 1000)
        return BadRequest("steps must be an integer in [1, 1000]", "steps");
      request.steps = static_cast<int>(value);
    }
    if (const json::Value* bpm = Given(*gufo, "bpm")) {
      if (!bpm->is_number())
        return BadRequest("bpm must be a number", "bpm");
      request.bpm = bpm->as_double(0.0);
    }
    if (auto error = string_field(*gufo, "key", &unknown))
      return error;
    if (Given(*gufo, "key"))
      request.key = unknown;
    if (const json::Value* meter = Given(*gufo, "time_signature")) {
      const std::string text = meter->is_string() ? meter->str() : "";
      int beats = 0;
      const auto slash = text.find('/');
      const auto result =
          std::from_chars(text.data(), text.data() + slash, beats);
      if (slash == std::string::npos || result.ec != std::errc() ||
          result.ptr != text.data() + slash)
        return BadRequest("time_signature must look like \"4/4\"",
                          "time_signature");
      request.beats_per_bar = beats;
    }
    if (const json::Value* language = Given(*gufo, "language")) {
      if (!language->is_string())
        return BadRequest("language must be a string", "language");
      request.language = language->str();
    }
    if (const json::Value* options = Given(*gufo, "options")) {
      if (!options->is_object())
        return BadRequest("options must be an object", "options");
      request.provider_json = options->dump();
    }
  }
  std::string field, error;
  if (!ext::ValidateMusicRequest(service.capabilities(), request, &field,
                                 &error))
    return BadRequest(error, field, "unsupported_parameter");
  if (!service.runner().ValidateRequest(request, &error))
    return BadRequest(error, "gufo");
  return std::nullopt;
}

HttpResponse CreateJob(const HttpRequest& http, MusicService& service) {
  auto parsed = std::make_shared<ParsedRequest>();
  if (auto error = ParseRequest(http, service, parsed.get()))
    return *error;
  const auto& request = parsed->music;
  json::Value metadata = json::Value::object();
  metadata["prompt_sha256"] = Sha256(request.prompt);
  metadata["lyrics_sha256"] = Sha256(request.lyrics);
  metadata["seed"] = static_cast<long long>(request.seed);
  metadata["plan_only"] = request.plan_only;
  metadata["response_format"] = parsed->float32 ? "wav_float" : "wav";
  ext::MusicRunner* runner = &service.runner();
  MediaWork work = [parsed, runner](MediaJobContext& context,
                                    std::string* error) {
    ext::MusicResult result;
    const bool ok = runner->Generate(
        parsed->music, [&context] { return context.cancelled(); },
        [&context](std::string_view phase, std::uint64_t completed,
                   std::uint64_t total) {
          return context.Progress(phase, completed, total);
        },
        &result, error);
    if (!ok)
      return false;
    json::Value outcome = json::Value::object();
    outcome["finish"] = result.finish == ext::MusicFinish::kTruncated
                            ? "truncated"
                            : "complete";
    outcome["score_format"] = result.score_format;
    if (!result.telemetry_json.empty())
      outcome["telemetry"] = json::parse(result.telemetry_json);
    if (!result.score.empty() &&
        !context.WriteFile(kScoreFile, "text/vnd.abc; charset=utf-8",
                           result.score)) {
      *error = "cannot store the score";
      return false;
    }
    if (!parsed->music.plan_only) {
      const std::string wav = EncodeMusicWav(result.audio, parsed->float32);
      if (wav.empty() || !context.WriteFile(kAudioFile, "audio/wav", wav)) {
        *error = "cannot store the audio";
        return false;
      }
      outcome["audio_seconds"] =
          static_cast<double>(result.audio.samples.size()) /
          result.audio.channels / result.audio.sample_rate;
    }
    context.SetMetadata(outcome);
    return true;
  };
  auto created = service.store().Create(parsed->model, std::move(metadata),
                                        std::move(work));
  if (created.result == MediaJobResult::kQueueFull)
    return Error(429, "Too Many Requests", created.error, "queue_full");
  if (created.result != MediaJobResult::kOk || !created.job)
    return Error(503, "Service Unavailable", created.error, "storage_error");
  HttpResponse response = Json(202, "Accepted", JobJson(*created.job));
  response.log_details = "job=" + created.job->id;
  return response;
}

}  // namespace

MusicService::MusicService(std::unique_ptr<ext::MusicRunner> runner,
                           MusicServiceOptions options)
    : runner_(std::move(runner)),
      capabilities_(runner_->Capabilities()),
      model_id_(options.model_id.empty() ? runner_->ModelId()
                                         : options.model_id),
      aliases_(std::move(options.model_aliases)) {
  if (model_id_.empty())
    model_id_ = "music";
  store_ = std::make_unique<MediaJobStore>(MediaJobStoreOptions{
      .root = options.storage_root,
      .id_prefix = "music",
      .schema = std::string(kSchema),
      .queue_capacity = options.queue_capacity,
      .ttl = options.ttl,
      .id_factory = {},
      .now = {},
  });
}

bool MusicService::ready() const {
  return store_->ready();
}

std::string MusicService::initialization_error() const {
  return store_->initialization_error();
}

bool MusicService::serves_model(std::string_view name) const {
  return name == model_id_ ||
         std::ranges::find(aliases_, name) != aliases_.end();
}

json::Value MusicCapabilitiesJson(const ext::MusicCapabilities& c) {
  json::Value root = json::Value::object();
  root["lyrics"] = std::string(SupportName(c.lyrics));
  root["tempo_key_meter"] = std::string(SupportName(c.tempo_key_meter));
  root["language"] = std::string(SupportName(c.language));
  root["reference_audio"] = std::string(SupportName(c.reference_audio));
  root["score"] = std::string(SupportName(c.score));
  json::Value formats = json::Value::array();
  for (const auto& format : c.score_formats)
    formats.push_back(format);
  root["score_formats"] = std::move(formats);
  root["score_output"] = c.score_output;
  root["plan_only"] = c.plan_only;
  root["guidance"] = c.guidance;
  root["steps"] = c.steps;
  root["duration"] = c.duration == ext::MusicDurationControl::kExact ? "exact"
                     : c.duration == ext::MusicDurationControl::kMaximum
                         ? "maximum"
                         : "none";
  root["max_seconds"] = c.max_seconds;
  return root;
}

bool IsMusicApiPath(std::string_view path) noexcept {
  return path == kMusicPath ||
         (path.starts_with(kMusicPath) && path.size() > kMusicPath.size() &&
          path[kMusicPath.size()] == '/');
}

HttpResponse HandleMusicApiRequest(const HttpRequest& request,
                                   MusicService& service) {
  if (request.path == kMusicPath) {
    if (request.method != "POST")
      return Error(405, "Method Not Allowed", "use POST", "method_not_allowed");
    return CreateJob(request, service);
  }
  std::string_view rest =
      std::string_view(request.path).substr(kMusicPath.size() + 1);
  std::string_view file;
  if (const auto slash = rest.find('/'); slash != std::string_view::npos) {
    file = rest.substr(slash + 1);
    rest = rest.substr(0, slash);
  }
  const std::string id(rest);
  if (file.empty()) {
    if (request.method == "GET") {
      const auto job = service.store().Get(id);
      if (!job)
        return Error(404, "Not Found", "music job not found", "job_not_found");
      return Json(200, "OK", JobJson(*job));
    }
    if (request.method == "DELETE") {
      const auto job = service.store().Delete(id);
      if (!job)
        return Error(404, "Not Found", "music job not found", "job_not_found");
      json::Value root = json::Value::object();
      root["id"] = job->id;
      root["object"] = "audio.music.deleted";
      root["deleted"] = true;
      return Json(200, "OK", root);
    }
    return Error(405, "Method Not Allowed", "use GET or DELETE",
                 "method_not_allowed");
  }
  if (request.method != "GET")
    return Error(405, "Method Not Allowed", "use GET", "method_not_allowed");
  if (file != "content" && file != "score")
    return Error(404, "Not Found", "unknown music job resource", "not_found");
  MediaJobFile content;
  const auto result = service.store().File(
      id, file == "content" ? kAudioFile : kScoreFile, &content);
  if (result == MediaJobResult::kNotReady)
    return Error(409, "Conflict", "the job has not completed", "job_not_ready");
  if (result == MediaJobResult::kNotFound)
    return Error(
        404, "Not Found",
        file == "content" ? "no audio for this job" : "no score for this job",
        "not_found");
  if (result != MediaJobResult::kOk)
    return Error(503, "Service Unavailable", "content is unavailable",
                 "content_unavailable");
  return ServeMediaFile(request, content);
}

}  // namespace gufo::server
