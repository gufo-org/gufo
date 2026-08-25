#include "src/cli/serve/video_api.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "src/cli/serve/json.hpp"

namespace strix::server {
namespace {

constexpr std::string_view kVideosPath = "/v1/videos";
constexpr double kMaximumExactJsonInteger = 9007199254740991.0;
constexpr std::size_t kMaximumPromptBytes = 4096;
constexpr std::size_t kMaximumMultipartHeadersBytes = 16U << 10U;
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

std::vector<std::string_view> HeaderSegments(std::string_view value,
                                             bool* valid) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  bool quoted = false;
  bool escaped = false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const char character = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quoted && character == '\\') {
      escaped = true;
      continue;
    }
    if (character == '"') {
      quoted = !quoted;
    } else if (character == ';' && !quoted) {
      result.push_back(TrimAscii(value.substr(begin, index - begin)));
      begin = index + 1;
    }
  }
  if (quoted || escaped) {
    *valid = false;
    return {};
  }
  result.push_back(TrimAscii(value.substr(begin)));
  *valid = true;
  return result;
}

bool DecodeHeaderParameter(std::string_view value, std::string* output) {
  value = TrimAscii(value);
  if (value.empty()) {
    output->clear();
    return true;
  }
  if (value.front() != '"') {
    if (value.find_first_of("\"\r\n") != std::string_view::npos) {
      return false;
    }
    *output = value;
    return true;
  }
  if (value.size() < 2 || value.back() != '"') {
    return false;
  }
  output->clear();
  const std::string_view inner = value.substr(1, value.size() - 2);
  bool escaped = false;
  for (const char character : inner) {
    if (escaped) {
      output->push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"' || character == '\r' || character == '\n') {
      return false;
    } else {
      output->push_back(character);
    }
  }
  return !escaped;
}

bool HeaderParameter(std::string_view value, std::string_view parameter,
                     std::string* output, bool* present) {
  bool valid = false;
  const std::vector<std::string_view> segments = HeaderSegments(value, &valid);
  if (!valid || segments.empty()) {
    return false;
  }
  *present = false;
  for (std::size_t index = 1; index < segments.size(); ++index) {
    const std::size_t equals = segments[index].find('=');
    if (equals == std::string_view::npos) {
      return false;
    }
    const std::string_view name = TrimAscii(segments[index].substr(0, equals));
    if (!EqualCaseInsensitive(name, parameter)) {
      continue;
    }
    if (*present ||
        !DecodeHeaderParameter(segments[index].substr(equals + 1), output)) {
      return false;
    }
    *present = true;
  }
  return true;
}

bool IsMultipartFormData(std::string_view content_type) {
  const std::size_t semicolon = content_type.find(';');
  return EqualCaseInsensitive(TrimAscii(content_type.substr(0, semicolon)),
                              "multipart/form-data");
}

std::optional<std::string> MultipartBoundary(std::string_view content_type) {
  std::string boundary;
  bool present = false;
  if (!HeaderParameter(content_type, "boundary", &boundary, &present) ||
      !present || boundary.empty() || boundary.size() > 200 ||
      boundary.find_first_of("\r\n") != std::string::npos) {
    return std::nullopt;
  }
  return boundary;
}

std::optional<json::Value> ParseMultipartBody(const HttpRequest& request,
                                              std::string* error) {
  const auto boundary = MultipartBoundary(request.header("content-type"));
  if (!boundary.has_value()) {
    *error = "multipart video request has an invalid boundary";
    return std::nullopt;
  }
  const std::string delimiter = "--" + *boundary;
  const std::string_view body = request.body;
  if (!body.starts_with(delimiter)) {
    *error = "multipart video request does not start with its boundary";
    return std::nullopt;
  }

  json::Value result = json::Value::object();
  std::size_t cursor = delimiter.size();
  std::size_t fields = 0;
  while (true) {
    if (body.substr(cursor).starts_with("--")) {
      cursor += 2;
      if (body.substr(cursor).starts_with("\r\n")) {
        cursor += 2;
      }
      if (cursor != body.size()) {
        *error = "multipart video request has data after its final boundary";
        return std::nullopt;
      }
      return result;
    }
    if (!body.substr(cursor).starts_with("\r\n")) {
      *error = "multipart video request has an invalid boundary separator";
      return std::nullopt;
    }
    cursor += 2;
    const std::size_t headers_end = body.find("\r\n\r\n", cursor);
    if (headers_end == std::string_view::npos ||
        headers_end - cursor > kMaximumMultipartHeadersBytes) {
      *error = "multipart video request has invalid part headers";
      return std::nullopt;
    }

    std::string content_disposition;
    std::string_view headers = body.substr(cursor, headers_end - cursor);
    while (!headers.empty()) {
      const std::size_t newline = headers.find("\r\n");
      const std::string_view line = headers.substr(0, newline);
      const std::size_t colon = line.find(':');
      if (colon == std::string_view::npos) {
        *error = "multipart video request has a malformed part header";
        return std::nullopt;
      }
      const std::string_view name = TrimAscii(line.substr(0, colon));
      const std::string_view value = TrimAscii(line.substr(colon + 1));
      if (EqualCaseInsensitive(name, "content-disposition")) {
        if (!content_disposition.empty()) {
          *error = "multipart video request repeats Content-Disposition";
          return std::nullopt;
        }
        content_disposition = value;
      }
      if (newline == std::string_view::npos) {
        break;
      }
      headers.remove_prefix(newline + 2);
    }

    bool disposition_valid = false;
    const std::vector<std::string_view> disposition_segments =
        HeaderSegments(content_disposition, &disposition_valid);
    std::string field_name;
    std::string filename;
    bool name_present = false;
    bool filename_present = false;
    if (!disposition_valid || disposition_segments.empty() ||
        !EqualCaseInsensitive(disposition_segments.front(), "form-data") ||
        !HeaderParameter(content_disposition, "name", &field_name,
                         &name_present) ||
        !HeaderParameter(content_disposition, "filename", &filename,
                         &filename_present) ||
        !name_present || field_name.empty()) {
      *error = "multipart video request has invalid Content-Disposition";
      return std::nullopt;
    }

    const std::size_t content_begin = headers_end + 4;
    const std::string next_boundary = "\r\n" + delimiter;
    const std::size_t content_end = body.find(next_boundary, content_begin);
    if (content_end == std::string_view::npos) {
      *error = "multipart video request is missing a closing boundary";
      return std::nullopt;
    }
    if (++fields > kMaximumMultipartFields || result.contains(field_name)) {
      *error = fields > kMaximumMultipartFields
                   ? "multipart video request contains too many fields"
                   : "multipart video request repeats a field";
      return std::nullopt;
    }
    const std::string_view content =
        body.substr(content_begin, content_end - content_begin);
    if (field_name == "strix" && !filename_present) {
      try {
        json::Value extension = json::parse(std::string(content));
        if (!extension.is_object()) {
          *error = "multipart 'strix' field must contain a JSON object";
          return std::nullopt;
        }
        result[field_name] = std::move(extension);
      } catch (const std::exception&) {
        *error = "multipart 'strix' field must contain valid JSON";
        return std::nullopt;
      }
    } else {
      result[field_name] =
          filename_present ? std::string() : std::string(content);
    }
    cursor = content_end + next_boundary.size();
  }
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
  static constexpr std::string_view kUnsupported[] = {
      "input_reference", "first_frame", "last_frame", "references"};
  return std::ranges::any_of(kUnsupported, [&](std::string_view key) {
    return body.contains(std::string(key));
  });
}

HttpResponse CreateVideo(const HttpRequest& request, VideoJobService& service) {
  json::Value body;
  const std::string content_type = request.header("content-type");
  if (IsMultipartFormData(content_type)) {
    std::string multipart_error;
    auto parsed = ParseMultipartBody(request, &multipart_error);
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
  if (IsUnsupportedConditioning(body)) {
    return Error(400, "Bad Request",
                 "image and ordered-reference conditioning are not supported",
                 "unsupported_input_reference");
  }
  if (!HasOnlyMembers(body, {"model", "prompt", "size", "seconds", "strix"})) {
    return Error(400, "Bad Request",
                 "request contains an unsupported video field",
                 "unsupported_field");
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
  const std::string model = model_value->str();
  const std::string prompt = prompt_value->str();
  const std::string size = body.member_str("size", "512x512");
  const json::Value* seconds_value = body.find("seconds");
  std::string seconds;
  if (seconds_value != nullptr && seconds_value->is_string()) {
    seconds = seconds_value->str();
  } else if (seconds_value != nullptr && seconds_value->is_number() &&
             (seconds_value->as_double() == 1.0 ||
              seconds_value->as_double() == 5.0)) {
    seconds = seconds_value->as_double() == 5.0 ? "5" : "1";
  }
  if (model.empty() || prompt.empty() || seconds.empty()) {
    return Error(400, "Bad Request",
                 "'model', 'prompt', and 'seconds' are required",
                 "missing_required_parameter");
  }
  if (prompt.size() > kMaximumPromptBytes) {
    return Error(413, "Payload Too Large",
                 "video prompt exceeds 4096 UTF-8 bytes", "prompt_too_large");
  }

  std::string preset =
      size == "1344x768" && seconds == "5" ? "exact-1344x768" : "exact";
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
  if (const json::Value* extension = body.find("strix")) {
    if (!extension->is_object() ||
        !HasOnlyMembers(*extension, {"preset", "seed", "frames",
                                     "output_format", "selected_frame"})) {
      return Error(400, "Bad Request",
                   "'strix' must contain only supported video options",
                   "invalid_strix_options");
    }
    if (extension->contains("preset")) {
      if (!extension->find("preset")->is_string()) {
        return Error(400, "Bad Request", "'strix.preset' must be a string",
                     "invalid_preset");
      }
      preset = extension->member_str("preset");
      preset_explicit = true;
    }
    if (extension->contains("output_format") &&
        !extension->find("output_format")->is_string()) {
      return Error(400, "Bad Request", "'strix.output_format' must be a string",
                   "invalid_output_format");
    }
    output_format = extension->member_str("output_format", output_format);
    if (extension->contains("seed")) {
      const auto parsed = JsonUnsigned(extension->find("seed"));
      if (!parsed.has_value()) {
        return Error(400, "Bad Request",
                     "'strix.seed' must be a non-negative exact integer",
                     "invalid_seed");
      }
      seed = *parsed;
    }
    if (extension->contains("frames")) {
      frames = JsonFrame(extension->find("frames"));
      if (!frames.has_value()) {
        return Error(400, "Bad Request",
                     "'strix.frames' must be a positive legal frame count",
                     "invalid_frames");
      }
    }
    if (extension->contains("selected_frame")) {
      selected_frame = JsonFrame(extension->find("selected_frame"));
      if (!selected_frame.has_value()) {
        return Error(400, "Bad Request",
                     "'strix.selected_frame' must be a non-negative integer",
                     "invalid_selected_frame");
      }
    }
  }
  if (preset_explicit && !model_preset.empty() && preset != model_preset) {
    return Error(400, "Bad Request",
                 "model alias and 'strix.preset' must select the same preset",
                 "preset_model_mismatch");
  }

  std::string preset_error;
  auto parameters = minimax_h3::ResolveGenerationPreset(preset, &preset_error);
  if (!parameters.has_value()) {
    return Error(400, "Bad Request", preset_error, "invalid_preset");
  }
  if (frames.has_value()) {
    parameters->frames = *frames;
  }
  if (output_format == "ppm") {
    if (parameters->preset != "development-256" || size != "256x256") {
      return Error(400, "Bad Request",
                   "PPM output requires preset 'dev' and size '256x256'",
                   "invalid_output_format");
    }
    parameters->selected_frames = {
        selected_frame.value_or(parameters->frames / 2)};
  } else if (output_format == "mp4") {
    const bool one_second = size == "512x512" && seconds == "1" &&
                            parameters->frames == 22 &&
                            (parameters->preset == "exact-512" ||
                             parameters->preset == "fast-384" ||
                             parameters->preset == "aggressive-320");
    const bool released_full_resolution =
        size == "1344x768" && seconds == "5" &&
        parameters->preset == "exact-1344x768" && parameters->frames == 124;
    if (!parameters->mux || selected_frame.has_value() ||
        (!one_second && !released_full_resolution)) {
      return Error(400, "Bad Request",
                   "MP4 output requires a supported preset, duration, and "
                   "resolution combination",
                   "invalid_output_format");
    }
  } else {
    return Error(400, "Bad Request",
                 "'strix.output_format' must be 'mp4' or 'ppm'",
                 "invalid_output_format");
  }
  std::string parameter_error;
  if (!minimax_h3::ValidateGenerationParameters(*parameters,
                                                &parameter_error)) {
    return Error(400, "Bad Request", parameter_error, "invalid_frames");
  }

  const VideoJobCreateResult created = service.Create({
      .model = model,
      .prompt = prompt,
      .size = size,
      .seconds = seconds,
      .output_format = output_format,
      .parameters = std::move(*parameters),
      .seed = seed,
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

}  // namespace strix::server
