#include "src/cli/serve/audio_asr_api.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kTranscriptionsPath = "/v1/audio/transcriptions";
constexpr std::size_t kMaximumTextFieldBytes = 16U << 10U;
constexpr std::size_t kMaximumNewTokens = 512;

struct MultipartPart {
  std::string filename;
  std::string body;
};

using Multipart = std::unordered_map<std::string, MultipartPart>;

std::string_view Trim(std::string_view value) {
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

std::string Lower(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

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
      .log_details = {},
  };
}

std::optional<std::string> Parameter(std::string_view value,
                                     std::string_view name) {
  const std::string target = Lower(name);
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t semicolon = value.find(';', begin);
    const std::size_t end =
        semicolon == std::string_view::npos ? value.size() : semicolon;
    const std::string_view item = Trim(value.substr(begin, end - begin));
    const std::size_t equals = item.find('=');
    if (equals != std::string_view::npos &&
        Lower(Trim(item.substr(0, equals))) == target) {
      std::string_view parameter = Trim(item.substr(equals + 1U));
      if (parameter.size() >= 2U && parameter.front() == '"' &&
          parameter.back() == '"') {
        parameter.remove_prefix(1);
        parameter.remove_suffix(1);
      }
      if (parameter.find_first_of("\r\n") != std::string_view::npos) {
        return std::nullopt;
      }
      return std::string(parameter);
    }
    if (semicolon == std::string_view::npos) {
      break;
    }
    begin = semicolon + 1U;
  }
  return std::nullopt;
}

bool ExtractBoundary(std::string_view content_type, std::string* boundary) {
  const std::size_t semicolon = content_type.find(';');
  const std::string media_type = Lower(Trim(content_type.substr(0, semicolon)));
  if (media_type != "multipart/form-data") {
    return false;
  }
  const std::optional<std::string> parsed = Parameter(content_type, "boundary");
  if (!parsed.has_value() || parsed->empty() || parsed->size() > 70U ||
      parsed->find_first_of("\r\n") != std::string::npos) {
    return false;
  }
  *boundary = *parsed;
  return true;
}

bool ParseMultipart(std::string_view body, std::string_view boundary,
                    Multipart* output, std::string* error) {
  const std::string delimiter = "--" + std::string(boundary);
  if (!body.starts_with(delimiter)) {
    *error = "multipart body does not start with its boundary";
    return false;
  }
  std::size_t cursor = delimiter.size();
  while (true) {
    if (body.substr(cursor, 2U) == "--") {
      return true;
    }
    if (body.substr(cursor, 2U) != "\r\n") {
      *error = "multipart boundary is malformed";
      return false;
    }
    cursor += 2U;
    const std::size_t headers_end = body.find("\r\n\r\n", cursor);
    if (headers_end == std::string_view::npos ||
        headers_end - cursor > kMaximumTextFieldBytes) {
      *error = "multipart part headers are malformed";
      return false;
    }

    std::string disposition;
    for (std::size_t line_begin = cursor; line_begin < headers_end;) {
      const std::size_t line_end = body.find("\r\n", line_begin);
      const std::size_t bounded_end =
          line_end == std::string_view::npos || line_end > headers_end
              ? headers_end
              : line_end;
      const std::string_view line =
          body.substr(line_begin, bounded_end - line_begin);
      const std::size_t colon = line.find(':');
      if (colon != std::string_view::npos &&
          Lower(Trim(line.substr(0, colon))) == "content-disposition") {
        disposition = std::string(Trim(line.substr(colon + 1U)));
      }
      if (bounded_end == headers_end) {
        break;
      }
      line_begin = bounded_end + 2U;
    }
    const std::optional<std::string> name = Parameter(disposition, "name");
    if (!name.has_value() || name->empty()) {
      *error = "multipart part is missing a field name";
      return false;
    }

    const std::size_t payload_begin = headers_end + 4U;
    const std::string next_marker = "\r\n" + delimiter;
    const std::size_t next = body.find(next_marker, payload_begin);
    if (next == std::string_view::npos) {
      *error = "multipart body is missing a closing boundary";
      return false;
    }
    MultipartPart part{
        .filename = Parameter(disposition, "filename").value_or(""),
        .body = std::string(body.substr(payload_begin, next - payload_begin)),
    };
    if (!output->emplace(*name, std::move(part)).second) {
      *error = "multipart field is duplicated: " + *name;
      return false;
    }
    if (output->size() > 12U) {
      *error = "multipart request contains too many fields";
      return false;
    }
    cursor = next + 2U + delimiter.size();
  }
}

const MultipartPart* Find(const Multipart& parts, std::string_view name) {
  const auto iterator = parts.find(std::string(name));
  return iterator == parts.end() ? nullptr : &iterator->second;
}

bool ParsePositiveSize(std::string_view value, std::size_t maximum,
                       std::size_t* output) {
  std::size_t parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      parsed == 0U || parsed > maximum) {
    return false;
  }
  *output = parsed;
  return true;
}

bool IsKnownField(std::string_view name) {
  static constexpr std::string_view kKnown[] = {
      "file",       "model",           "language",
      "prompt",     "response_format", "temperature",
      "max_tokens", "stream",          "timestamp_granularities[]",
  };
  return std::ranges::find(kKnown, name) != std::end(kKnown);
}

std::string ServerTiming(
    const models::qwen3_asr::TranscriptionTimings& timings) {
  std::ostringstream value;
  value << "decode_audio;dur=" << timings.decode_audio_ms
        << ", features;dur=" << timings.feature_extraction_ms
        << ", audio_encoder;dur=" << timings.audio_encoder_ms
        << ", text_decoder;dur=" << timings.text_decoder_ms;
  return value.str();
}

HttpResponse Transcribe(const HttpRequest& request, AsrService& service) {
  std::string boundary;
  if (!ExtractBoundary(request.header("content-type"), &boundary)) {
    return Error(400, "Bad Request",
                 "Content-Type must be multipart/form-data with a boundary",
                 "invalid_content_type");
  }
  Multipart parts;
  std::string parse_error;
  if (!ParseMultipart(request.body, boundary, &parts, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error),
                 "invalid_multipart");
  }
  for (const auto& [name, unused] : parts) {
    (void)unused;
    if (!IsKnownField(name)) {
      return Error(400, "Bad Request",
                   "unsupported transcription field: " + name,
                   "unsupported_field");
    }
  }

  const MultipartPart* file = Find(parts, "file");
  if (file == nullptr || file->body.empty()) {
    return Error(400, "Bad Request", "a non-empty WAV 'file' is required",
                 "missing_file");
  }
  const MultipartPart* model = Find(parts, "model");
  if (model != nullptr && model->body != service.model_id() &&
      model->body != "qwen3-asr" && model->body != "qwen3-asr-1.7b") {
    return Error(400, "Bad Request", "unsupported transcription model",
                 "invalid_model");
  }
  const MultipartPart* language = Find(parts, "language");
  const MultipartPart* prompt = Find(parts, "prompt");
  if ((language != nullptr && language->body.size() > kMaximumTextFieldBytes) ||
      (prompt != nullptr && prompt->body.size() > kMaximumTextFieldBytes)) {
    return Error(400, "Bad Request",
                 "transcription text fields exceed 16384 bytes",
                 "field_too_large");
  }
  const MultipartPart* response_format = Find(parts, "response_format");
  const std::string format =
      response_format == nullptr ? "json" : response_format->body;
  if (format != "json" && format != "text" && format != "verbose_json") {
    return Error(400, "Bad Request",
                 "response_format must be json, text, or verbose_json",
                 "invalid_response_format");
  }
  if (const MultipartPart* temperature = Find(parts, "temperature");
      temperature != nullptr) {
    double value = 0.0;
    const auto parsed = std::from_chars(
        temperature->body.data(),
        temperature->body.data() + temperature->body.size(), value);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != temperature->body.data() + temperature->body.size() ||
        !std::isfinite(value) || value != 0.0) {
      return Error(400, "Bad Request",
                   "native Qwen3-ASR supports temperature 0 only",
                   "invalid_temperature");
    }
  }
  if (Find(parts, "timestamp_granularities[]") != nullptr) {
    return Error(400, "Bad Request",
                 "timestamp granularities are not supported",
                 "unsupported_option");
  }
  bool stream = false;
  if (const auto* value = Find(parts, "stream")) {
    if (value->body != "true" && value->body != "false") {
      return Error(400, "Bad Request", "'stream' must be true or false",
                   "invalid_parameter_type");
    }
    stream = value->body == "true";
  }
  if (stream && format != "json") {
    return Error(400, "Bad Request", "streaming transcription requires json",
                 "invalid_response_format");
  }
  std::size_t maximum_tokens = 256;
  if (const MultipartPart* value = Find(parts, "max_tokens");
      value != nullptr &&
      !ParsePositiveSize(value->body, kMaximumNewTokens, &maximum_tokens)) {
    return Error(400, "Bad Request", "max_tokens must be between 1 and 512",
                 "invalid_max_tokens");
  }

  const auto* wav_data = reinterpret_cast<const std::byte*>(file->body.data());
  if (stream) {
    HttpResponse response;
    response.headers = {{"Content-Type", "text/event-stream"},
                        {"Cache-Control", "no-cache"},
                        {"X-Accel-Buffering", "no"},
                        {"X-Gufo-Schema", std::string(kAudioAsrApiSchema)}};
    response.stream_log = std::make_shared<HttpResponse::StreamLog>();
    response.streaming_body =
        [&service, wav = file->body,
         context = prompt == nullptr ? std::string{} : prompt->body,
         language = language == nullptr || language->body.empty()
                        ? std::optional<std::string>{}
                        : std::optional<std::string>{language->body},
         maximum_tokens, cancelled = request.is_cancelled,
         log = response.stream_log](const HttpResponse::BodyWriter& write) {
          bool connected = true;
          auto send = [&](json::Value event) {
            if (!connected)
              return false;
            connected = write("event: " + event.member_str("type") +
                              "\ndata: " + event.dump() + "\n\n");
            return connected;
          };
          std::string published;
          const auto progress = [&](std::string_view text) {
            if (!text.starts_with(published)) {
              // Committed SSE deltas cannot be retracted.
              throw std::runtime_error(
                  "ASR transcript changed its committed prefix");
            }
            if (text.size() == published.size())
              return connected;
            auto event = json::Value::object();
            event["type"] = "transcript.text.delta";
            event["delta"] = std::string(text.substr(published.size()));
            published = text;
            return send(std::move(event));
          };
          models::qwen3_asr::TranscriptionRequest input{
              .wav = {reinterpret_cast<const std::byte*>(wav.data()),
                      wav.size()},
              .context = context,
              .language = language,
              .max_new_tokens = maximum_tokens,
              .on_text = progress,
          };
          models::qwen3_asr::TranscriptionResult result;
          std::string error;
          const auto stop = [&] {
            return !connected || (cancelled && cancelled());
          };
          if (!service.Transcribe(input, stop, &result, &error)) {
            log->error_code = stop() ? "cancelled" : "transcription_failed";
            if (!stop()) {
              auto event = json::Value::object();
              event["type"] = "error";
              event["code"] = log->error_code;
              event["message"] = error;
              log->error_event_sent = send(std::move(event));
            }
            return;
          }
          if (!progress(result.text))
            return;
          auto done = json::Value::object();
          done["type"] = "transcript.text.done";
          done["text"] = result.text;
          send(std::move(done));
          log->details = "model=" + service.model_id() +
                         " chunks=" + std::to_string(result.chunks) +
                         " generated_tokens=" +
                         std::to_string(result.generated_ids.size());
        };
    return response;
  }
  models::qwen3_asr::TranscriptionResult result;
  std::string error;
  if (!service.Transcribe(
          models::qwen3_asr::TranscriptionRequest{
              .wav = std::span<const std::byte>(wav_data, file->body.size()),
              .context = prompt == nullptr ? std::string{} : prompt->body,
              .language = language == nullptr || language->body.empty()
                              ? std::nullopt
                              : std::optional<std::string>(language->body),
              .max_new_tokens = maximum_tokens,
          },
          request.is_cancelled, &result, &error)) {
    if (request.is_cancelled && request.is_cancelled()) {
      return Error(499, "Client Closed Request", "transcription cancelled",
                   "cancelled");
    }
    return Error(500, "Internal Server Error", std::move(error),
                 "transcription_failed");
  }

  HttpResponse response;
  response.status = 200;
  response.reason = "OK";
  if (format == "text") {
    response.body = core::Utf8Decoder{}.Push(result.text, true);
    response.headers.emplace_back("Content-Type", "text/plain; charset=utf-8");
  } else {
    json::Value body = json::Value::object();
    body["text"] = core::Utf8Decoder{}.Push(result.text, true);
    if (format == "verbose_json") {
      body["task"] = "transcribe";
      body["language"] = Lower(result.language);
      body["duration"] =
          static_cast<double>(result.audio_samples) / result.sample_rate;
      body["segments"] = json::Value::array();
    }
    response.body = body.dump();
  }
  response.headers.emplace_back("X-Gufo-Schema",
                                std::string(kAudioAsrApiSchema));
  response.headers.emplace_back("X-Gufo-ASR-Backend", service.backend_name());
  response.headers.emplace_back("Server-Timing", ServerTiming(result.timings));
  response.log_details =
      "model=" + service.model_id() +
      " chunks=" + std::to_string(result.chunks) +
      " audio_tokens=" + std::to_string(result.audio_tokens) +
      " generated_tokens=" + std::to_string(result.generated_ids.size()) +
      " audio_ms=" +
      std::to_string(1000ULL * result.audio_samples /
                     std::max(1U, result.sample_rate)) +
      " feature_ms=" + std::to_string(result.timings.feature_extraction_ms) +
      " encoder_ms=" + std::to_string(result.timings.audio_encoder_ms) +
      " decoder_ms=" + std::to_string(result.timings.text_decoder_ms);
  return response;
}

}  // namespace

bool IsAudioAsrApiPath(std::string_view path) noexcept {
  return path == kTranscriptionsPath;
}

HttpResponse HandleAudioAsrApiRequest(const HttpRequest& request,
                                      AsrService& service) {
  if (request.path == kTranscriptionsPath && request.method == "POST") {
    return Transcribe(request, service);
  }
  return Error(405, "Method Not Allowed",
               "HTTP method is not supported for this transcription endpoint",
               "method_not_allowed");
}

}  // namespace gufo::server
