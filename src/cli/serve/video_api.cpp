#include "src/cli/serve/video_api.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/serve/multipart.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/ext/video.hpp"

namespace gufo::server {
namespace {

std::optional<int> ResolveDurationFrames(std::string_view seconds,
                                         std::string* error = nullptr) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->resolve_duration_frames) {
    return provider->resolve_duration_frames(seconds, error);
  }
  if (error)
    *error = "video provider 'minimax-h3' not available";
  return std::nullopt;
}

std::optional<ext::VideoParameters> ResolveGenerationPreset(
    std::string_view preset, std::string* error = nullptr) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->resolve_preset) {
    return provider->resolve_preset(preset, error);
  }
  if (error)
    *error = "video provider 'minimax-h3' not available";
  return std::nullopt;
}

bool ValidateGenerationParameters(const ext::VideoParameters& parameters,
                                  std::string* error = nullptr) {
  const auto* provider = ext::FindVideoProvider("minimax-h3");
  if (provider && provider->validate_parameters) {
    return provider->validate_parameters(parameters, error);
  }
  if (error)
    *error = "video provider 'minimax-h3' not available";
  return false;
}

constexpr std::string_view kVideosPath = "/v1/videos";
constexpr double kMaximumExactJsonInteger = 9007199254740991.0;
constexpr std::size_t kMaximumMultipartFields = 16;

HttpResponse Error(int status, std::string reason, std::string message,
                   std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = std::move(code);
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::move(reason),
      .body = root.dump(),
      .headers = {},
      .streaming_body = {},
  };
}

json::Value SnapshotValue(const VideoJobSnapshot& snapshot) {
  json::Value root = json::Value::object();
  root["id"] = snapshot.id;
  root["object"] = "video";
  root["schema"] = std::string(kVideoApiSchema);
  root["created_at"] = static_cast<long long>(snapshot.created_at);
  root["status"] = std::string(ToString(snapshot.status));
  root["progress"] = snapshot.progress;
  root["model"] = snapshot.model;
  root["size"] = snapshot.size;
  root["seconds"] = snapshot.seconds;
  root["output_format"] = snapshot.output_format;
  if (snapshot.completed_at > 0) {
    root["completed_at"] = static_cast<long long>(snapshot.completed_at);
  } else {
    root["completed_at"] = json::Value();
  }
  if (snapshot.expires_at > 0) {
    root["expires_at"] = static_cast<long long>(snapshot.expires_at);
  } else {
    root["expires_at"] = json::Value();
  }
  if (snapshot.error_code.empty()) {
    root["error"] = json::Value();
  } else {
    json::Value error = json::Value::object();
    error["code"] = snapshot.error_code;
    error["message"] = snapshot.error_message;
    root["error"] = std::move(error);
  }
  return root;
}

HttpResponse SnapshotResponse(int status, std::string reason,
                              const VideoJobSnapshot& snapshot) {
  return {
      .status = status,
      .reason = std::move(reason),
      .body = SnapshotValue(snapshot).dump(),
      .headers = {},
      .streaming_body = {},
      .log_details = "job=" + snapshot.id +
                     " state=" + std::string(ToString(snapshot.status)) +
                     " model=" + snapshot.model + " size=" + snapshot.size,
  };
}

bool HasOnlyMembers(const json::Value& value,
                    std::initializer_list<std::string_view> allowed) {
  return std::ranges::all_of(value.members(), [&](const auto& member) {
    return std::ranges::find(allowed, member.first) != allowed.end();
  });
}

std::string_view TrimAscii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

bool EqualCaseInsensitive(std::string_view left, std::string_view right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (std::tolower(static_cast<unsigned char>(left[index])) !=
        std::tolower(static_cast<unsigned char>(right[index]))) {
      return false;
    }
  }
  return true;
}

// The form's fields as a JSON object; nullopt with *error when it is
// malformed, or with *form_data false when the body is not a form.
std::optional<json::Value> ParseMultipartBody(
    const HttpRequest& request,
    std::map<std::string, std::string>* binary_fields, std::string* error,
    bool* form_data) {
  *form_data = true;
  json::Value result = json::Value::object();
  MultipartResult form = ParseMultipart(
      request.header("content-type"), request.body,
      {.max_fields = kMaximumMultipartFields, .unique_names = true},
      [&](const MultipartField& field) {
        const std::string field_name(field.name);
        const std::string_view content = field.content;
        if (field_name == "gufo" && !field.has_filename) {
          try {
            json::Value extension = json::parse(std::string(content));
            if (!extension.is_object()) {
              *error = "multipart 'gufo' field must contain a JSON object";
              return false;
            }
            result[field_name] = std::move(extension);
          } catch (const std::exception&) {
            *error = "multipart 'gufo' field must contain valid JSON";
            return false;
          }
        } else if (field.has_filename && binary_fields != nullptr &&
                   (field_name == "first_frame" ||
                    field_name == "last_frame")) {
          // Keyframe anchors keep their raw bytes out of the JSON value; the
          // empty sentinel only marks presence for CreateVideo.
          (*binary_fields)[field_name] = std::string(content);
          result[field_name] = std::string();
        } else {
          result[field_name] =
              field.has_filename ? std::string() : std::string(content);
        }
        return true;
      });
  switch (form.status) {
    case MultipartStatus::kComplete:
      return result;
    case MultipartStatus::kStopped:
      break;
    case MultipartStatus::kNotFormData:
      *form_data = false;
      break;
    case MultipartStatus::kBadBoundary:
      *error = "multipart video request has an invalid boundary";
      break;
    case MultipartStatus::kTooManyFields:
      *error = "multipart video request contains too many fields";
      break;
    case MultipartStatus::kDuplicateField:
      *error = "multipart video request repeats a field";
      break;
    case MultipartStatus::kMalformed:
      *error = std::move(form.detail);
      break;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> JsonUnsigned(const json::Value* value) {
  if (value == nullptr || !value->is_number()) {
    return std::nullopt;
  }
  const double number = value->as_double(-1.0);
  if (!std::isfinite(number) || number < 0.0 ||
      number > kMaximumExactJsonInteger || std::floor(number) != number) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(number);
}

std::optional<int> JsonFrame(const json::Value* value) {
  const auto parsed = JsonUnsigned(value);
  if (!parsed.has_value() ||
      *parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  return static_cast<int>(*parsed);
}

bool IsUnsupportedConditioning(const json::Value& body) {
  static constexpr std::string_view kUnsupported[] = {"input_reference",
                                                      "references"};
  return std::ranges::any_of(kUnsupported, [&](std::string_view key) {
    return body.contains(std::string(key));
  });
}

HttpResponse CreateVideo(const HttpRequest& request, VideoJobService& service) {
  json::Value body;
  std::map<std::string, std::string> binary_fields;
  const std::string content_type = request.header("content-type");
  std::string multipart_error;
  bool form_data = false;
  auto parsed =
      ParseMultipartBody(request, &binary_fields, &multipart_error, &form_data);
  if (form_data) {
    if (!parsed.has_value()) {
      return Error(400, "Bad Request", std::move(multipart_error),
                   "invalid_multipart_form");
    }
    body = std::move(*parsed);
  } else {
    if (!content_type.empty() &&
        !EqualCaseInsensitive(
            TrimAscii(content_type.substr(0, content_type.find(';'))),
            "application/json")) {
      return Error(415, "Unsupported Media Type",
                   "video requests must use application/json or "
                   "multipart/form-data",
                   "unsupported_media_type");
    }
    try {
      body = json::parse(request.body);
    } catch (const std::exception&) {
      return Error(400, "Bad Request", "request body must be valid JSON",
                   "parse_error");
    }
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }
  // Unknown members are ignored, as in the other OpenAI-compatible servers;
  // known conditioning fields are refused rather than silently dropped.
  if (IsUnsupportedConditioning(body)) {
    return Error(400, "Bad Request",
                 "ordered-reference conditioning is not supported",
                 "unsupported_input_reference");
  }

  const json::Value* model_value = body.find("model");
  const json::Value* prompt_value = body.find("prompt");
  const json::Value* size_value = body.find("size");
  if (model_value == nullptr || !model_value->is_string() ||
      prompt_value == nullptr || !prompt_value->is_string() ||
      (size_value != nullptr && !size_value->is_string())) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and optional 'size' must be strings",
                 "invalid_parameter_type");
  }
  // Alias names behave as the root model; preset-selecting variant names
  // stay canonical (#957).
  const std::string model = service.normalize_model(model_value->str());
  const std::string prompt = prompt_value->str();
  const std::string size = body.member_str("size", "512x512");
  const json::Value* seconds_value = body.find("seconds");
  std::string seconds;
  if (seconds_value != nullptr && seconds_value->is_string()) {
    seconds = seconds_value->str();
  } else if (seconds_value != nullptr && seconds_value->is_number()) {
    seconds = seconds_value->dump();
  }
  if (model.empty() || prompt.empty() || seconds.empty()) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and 'seconds' are required",
                 "missing_required_parameter");
  }
  std::string duration_error;
  const auto duration_frames = ResolveDurationFrames(seconds, &duration_error);
  if (!duration_frames) {
    return Error(400, "Bad Request", duration_error, "invalid_seconds");
  }
  std::string preset = size == "1344x768" ? "exact-1344x768" : "exact";
  std::string model_preset;
  if (model == "minimax-h3-exact") {
    model_preset = "exact";
  } else if (model == "minimax-h3-fast") {
    model_preset = "fast";
  } else if (model == "minimax-h3-aggressive") {
    model_preset = "aggressive";
  } else if (model == "minimax-h3-dev") {
    model_preset = "dev";
  } else if (model == "minimax-h3-fullres") {
    model_preset = "exact-1344x768";
  } else if (model != "minimax-h3") {
    return Error(400, "Bad Request", "unsupported video model",
                 "invalid_model");
  }
  if (!model_preset.empty()) {
    preset = model_preset;
  }
  std::string output_format = "mp4";
  std::uint64_t seed = 42;
  std::optional<int> frames;
  std::optional<int> selected_frame;
  bool preset_explicit = false;
  if (const json::Value* extension = body.find("gufo")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension, {"preset", "seed", "frames",
                                     "output_format", "selected_frame"})) {
      return Error(400, "Bad Request",
                   "'gufo' must contain only supported video options",
                   "invalid_strix_options");
    }
    if (extension->contains("preset")) {
      if (!extension->find("preset")->is_string()) {
        return Error(400, "Bad Request", "'gufo.preset' must be a string",
                     "invalid_preset");
      }
      preset = extension->member_str("preset");
      preset_explicit = true;
    }
    if (extension->contains("output_format") &&
        !extension->find("output_format")->is_string()) {
      return Error(400, "Bad Request", "'gufo.output_format' must be a string",
                   "invalid_output_format");
    }
    output_format = extension->member_str("output_format", output_format);
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.seed' must be a non-negative exact integer",
                     "invalid_seed");
      }
      seed = *parsed;
    }
    if (extension->contains("frames")) {
      frames = JsonFrame(extension->find("frames"));
      if (!frames.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.frames' must be a positive legal frame count",
                     "invalid_frames");
      }
    }
    if (extension->contains("selected_frame")) {
      selected_frame = JsonFrame(extension->find("selected_frame"));
      if (!selected_frame.has_value()) {
        return Error(400, "Bad Request",
                     "'gufo.selected_frame' must be a non-negative integer",
                     "invalid_selected_frame");
      }
    }
  }
  if (preset_explicit && !model_preset.empty() && preset != model_preset) {
    return Error(400, "Bad Request",
                 "model alias and 'gufo.preset' must select the same preset",
                 "preset_model_mismatch");
  }

  std::string preset_error;
  auto parameters = ResolveGenerationPreset(preset, &preset_error);
  if (!parameters.has_value()) {
    return Error(400, "Bad Request", preset_error, "invalid_preset");
  }
  parameters->frames = frames.value_or(*duration_frames);
  if (output_format == "ppm") {
    if (parameters->preset != "development-256" || size != "256x256") {
      return Error(400, "Bad Request",
                   "PPM output requires preset 'dev' and size '256x256'",
                   "invalid_output_format");
    }
    parameters->selected_frames = {
        selected_frame.value_or(parameters->frames / 2)};
  } else if (output_format == "mp4") {
    if (!parameters->mux || selected_frame.has_value()) {
      return Error(400, "Bad Request",
                   "MP4 output requires a supported preset, duration, and "
                   "resolution combination",
                   "invalid_output_format");
    }
  } else {
    return Error(400, "Bad Request",
                 "'gufo.output_format' must be 'mp4' or 'ppm'",
                 "invalid_output_format");
  }
  std::string parameter_error;
  if (!ValidateGenerationParameters(*parameters, &parameter_error)) {
    return Error(400, "Bad Request", parameter_error, "invalid_frames");
  }
  if (parameters->frames != *duration_frames) {
    return Error(400, "Bad Request",
                 "'gufo.frames' must match the aligned 'seconds' duration",
                 "invalid_frames");
  }

  // Keyframe anchors (issue #177): multipart file parts or base64 data URLs,
  // validated to decodable PNG/JPEG/WebP (display orientation applied by
  // the shared decoder) before admission; raw bytes are staged into the
  // job directory by the service.
  std::vector<std::uint8_t> first_frame;
  std::vector<std::uint8_t> last_frame;
  const auto load_keyframe = [&](std::string_view field,
                                 std::vector<std::uint8_t>* out) {
    const json::Value* value = body.find(std::string(field));
    if (value == nullptr) {
      return std::optional<std::string>{};
    }
    const auto uploaded = binary_fields.find(std::string(field));
    if (uploaded != binary_fields.end()) {
      out->assign(uploaded->second.begin(), uploaded->second.end());
    } else if (value->is_string() && value->str().starts_with("data:")) {
      // Only inline data URLs: the video endpoint never fetches remote
      // imagery on the client's behalf.
      try {
        core::ImageReadBudget budget;
        const std::vector<std::uint8_t> bytes =
            core::ReadImageUrl(value->str(), budget);
        out->assign(bytes.begin(), bytes.end());
      } catch (const std::exception&) {
        out->clear();
        return std::optional<std::string>{
            "'" + std::string(field) +
            "' could not be read; provide an image data URL or file upload"};
      }
    } else {
      return std::optional<std::string>{
          "'" + std::string(field) +
          "' must be a base64 image data URL or file upload"};
    }
    if (out->empty() || out->size() > core::kMaxEncodedImageBytes) {
      return std::optional<std::string>{
          "'" + std::string(field) +
          "' is empty or exceeds the encoded image limit"};
    }
    try {
      const core::Image decoded =
          core::DecodeImage(std::span<const std::uint8_t>(*out));
      if (decoded.width == 0 || decoded.height == 0) {
        throw std::invalid_argument("empty decode");
      }
    } catch (const std::exception&) {
      return std::optional<std::string>{
          "'" + std::string(field) +
          "' is not a decodable PNG, JPEG, or WebP image"};
    }
    return std::optional<std::string>{};
  };
  const std::optional<std::string> first_error =
      load_keyframe("first_frame", &first_frame);
  const std::optional<std::string> last_error =
      load_keyframe("last_frame", &last_frame);
  if (first_error.has_value() || last_error.has_value()) {
    return Error(400, "Bad Request",
                 first_error.has_value() ? *first_error : *last_error,
                 "invalid_frame_image");
  }

  const VideoJobCreateResult created = service.Create({
      .model = model,
      .prompt = prompt,
      .size = size,
      .seconds = seconds,
      .output_format = output_format,
      .parameters = std::move(*parameters),
      .seed = seed,
      .first_frame_bytes = std::move(first_frame),
      .last_frame_bytes = std::move(last_frame),
  });
  if (created.result == VideoJobResult::kQueueFull) {
    HttpResponse response =
        Error(429, "Too Many Requests", created.error, "video_queue_full");
    response.headers.emplace_back("Retry-After", "1");
    return response;
  }
  if (created.result != VideoJobResult::kOk || !created.job.has_value()) {
    const int status = created.result == VideoJobResult::kIoError ? 503 : 400;
    return Error(
        status, status == 503 ? "Service Unavailable" : "Bad Request",
        created.error,
        status == 503 ? "video_service_unavailable" : "invalid_video_request");
  }
  HttpResponse response = SnapshotResponse(202, "Accepted", *created.job);
  response.headers.emplace_back("Location", "/v1/videos/" + created.job->id);
  return response;
}

HttpResponse LookupVideo(std::string_view id, VideoJobService& service) {
  const VideoJobLookupResult lookup = service.Get(id);
  if (lookup.result != VideoJobResult::kOk || !lookup.job.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  return SnapshotResponse(200, "OK", *lookup.job);
}

struct ByteRange {
  std::uint64_t first{0};
  std::uint64_t last{0};
};

bool ParseUnsigned(std::string_view text, std::uint64_t* result) {
  if (text.empty() || result == nullptr) {
    return false;
  }
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, *result);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

std::optional<ByteRange> ParseRange(std::string_view header,
                                    std::uint64_t bytes) {
  if (!header.starts_with("bytes=") || bytes == 0) {
    return std::nullopt;
  }
  const std::string_view value = header.substr(6);
  if (value.find(',') != std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t dash = value.find('-');
  if (dash == std::string_view::npos) {
    return std::nullopt;
  }
  const std::string_view first_text = value.substr(0, dash);
  const std::string_view last_text = value.substr(dash + 1);
  if (first_text.empty()) {
    std::uint64_t suffix = 0;
    if (!ParseUnsigned(last_text, &suffix) || suffix == 0) {
      return std::nullopt;
    }
    suffix = std::min(suffix, bytes);
    return ByteRange{.first = bytes - suffix, .last = bytes - 1};
  }
  std::uint64_t first = 0;
  if (!ParseUnsigned(first_text, &first) || first >= bytes) {
    return std::nullopt;
  }
  std::uint64_t last = bytes - 1;
  if (!last_text.empty() &&
      (!ParseUnsigned(last_text, &last) || last < first)) {
    return std::nullopt;
  }
  return ByteRange{.first = first, .last = std::min(last, bytes - 1)};
}

HttpResponse VideoContent(std::string_view id, const HttpRequest& request,
                          VideoJobService& service) {
  const VideoJobContentResult result = service.Content(id);
  if (result.result == VideoJobResult::kNotFound) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  if (result.result == VideoJobResult::kNotReady) {
    return Error(409, "Conflict", "video content is not ready",
                 "video_not_ready");
  }
  if (result.result != VideoJobResult::kOk || !result.content.has_value()) {
    return Error(503, "Service Unavailable", "video content is unavailable",
                 "video_content_unavailable");
  }

  const VideoJobContent& content = *result.content;
  const std::string range_header = request.header("range");
  ByteRange range{
      .first = 0,
      .last = content.bytes == 0 ? 0 : content.bytes - 1,
  };
  bool partial = false;
  if (!range_header.empty()) {
    const auto parsed = ParseRange(range_header, content.bytes);
    if (!parsed.has_value()) {
      HttpResponse response = Error(416, "Range Not Satisfiable",
                                    "invalid byte range", "invalid_range");
      response.headers.emplace_back("Content-Range",
                                    "bytes */" + std::to_string(content.bytes));
      response.headers.emplace_back("Accept-Ranges", "bytes");
      return response;
    }
    range = *parsed;
    partial = true;
  }

  const std::uint64_t count =
      content.bytes == 0 ? 0 : range.last - range.first + 1;
  if (count >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return Error(503, "Service Unavailable", "video content is too large",
                 "video_content_unavailable");
  }
  std::ifstream input(content.path, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(range.first));
  std::string body(static_cast<std::size_t>(count), '\0');
  input.read(body.data(), static_cast<std::streamsize>(count));
  if (!input && input.gcount() != static_cast<std::streamsize>(count)) {
    return Error(503, "Service Unavailable", "video content is unavailable",
                 "video_content_unavailable");
  }

  HttpResponse response{
      .status = partial ? 206 : 200,
      .reason = partial ? "Partial Content" : "OK",
      .body = std::move(body),
      .headers =
          {
              {"Content-Type", content.media_type},
              {"Accept-Ranges", "bytes"},
              {"Content-Disposition",
               "attachment; filename=\"" + std::string(id) +
                   (content.media_type == "video/mp4" ? ".mp4\"" : ".ppm\"")},
          },
      .streaming_body = {},
  };
  if (partial) {
    response.headers.emplace_back("Content-Range",
                                  "bytes " + std::to_string(range.first) + "-" +
                                      std::to_string(range.last) + "/" +
                                      std::to_string(content.bytes));
  }
  return response;
}

HttpResponse DeleteVideo(std::string_view id, VideoJobService& service) {
  const VideoJobLookupResult deleted = service.Delete(id);
  if (deleted.result != VideoJobResult::kOk || !deleted.job.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  json::Value root = json::Value::object();
  root["id"] = deleted.job->id;
  root["object"] = "video.deleted";
  root["deleted"] = true;
  return {
      .status = 200,
      .reason = "OK",
      .body = root.dump(),
      .headers = {},
      .streaming_body = {},
  };
}

std::optional<std::pair<std::string_view, bool>> ParseVideoPath(
    std::string_view path) {
  constexpr std::string_view prefix = "/v1/videos/";
  if (!path.starts_with(prefix)) {
    return std::nullopt;
  }
  std::string_view remainder = path.substr(prefix.size());
  bool content = false;
  constexpr std::string_view kContentSuffix = "/content";
  if (remainder.ends_with(kContentSuffix)) {
    remainder.remove_suffix(kContentSuffix.size());
    content = true;
  }
  if (remainder.empty() || remainder.find('/') != std::string_view::npos) {
    return std::nullopt;
  }
  return std::pair{remainder, content};
}

}  // namespace

bool IsVideoApiPath(std::string_view path) noexcept {
  return path == kVideosPath || path.starts_with("/v1/videos/");
}

HttpResponse HandleVideoApiRequest(const HttpRequest& request,
                                   VideoJobService& service) {
  if (request.path == kVideosPath) {
    if (request.method == "POST") {
      return CreateVideo(request, service);
    }
    return Error(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
  }

  const auto parsed = ParseVideoPath(request.path);
  if (!parsed.has_value()) {
    return Error(404, "Not Found", "video job not found", "video_not_found");
  }
  const auto [id, content] = *parsed;
  if (content) {
    if (request.method == "GET") {
      return VideoContent(id, request, service);
    }
    return Error(405, "Method Not Allowed", "method is not allowed",
                 "method_not_allowed");
  }
  if (request.method == "GET") {
    return LookupVideo(id, service);
  }
  if (request.method == "DELETE") {
    return DeleteVideo(id, service);
  }
  return Error(405, "Method Not Allowed", "method is not allowed",
               "method_not_allowed");
}

}  // namespace gufo::server
