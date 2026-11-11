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

#include "src/cli/serve/multipart.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kTranscriptionsPath = "/v1/audio/transcriptions";
constexpr std::size_t kMaximumTextFieldBytes = 16U << 10U;
constexpr std::size_t kMaximumNewTokens = 512;
constexpr std::size_t kMaximumFields = 12;

struct MultipartPart {
  std::string body;
};

using Multipart = std::unordered_map<std::string, MultipartPart>;

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

std::string ServerTiming(const ext::TranscriptionTimings& timings) {
  std::ostringstream value;
  value << "decode_audio;dur=" << timings.decode_audio_ms
        << ", features;dur=" << timings.feature_extraction_ms
        << ", audio_encoder;dur=" << timings.audio_encoder_ms
        << ", text_decoder;dur=" << timings.text_decoder_ms;
  return value.str();
}

HttpResponse Transcribe(const HttpRequest& request, AsrService& service) {
  Multipart parts;
  MultipartResult form = ParseMultipart(
      request.header("content-type"), request.body,
      {.max_fields = kMaximumFields, .unique_names = true},
      [&parts](const MultipartField& field) {
        parts.emplace(std::string(field.name),
                      MultipartPart{.body = std::string(field.content)});
        return true;
      });
  if (form.status == MultipartStatus::kNotFormData ||
      form.status == MultipartStatus::kBadBoundary) {
    return Error(400, "Bad Request",
                 "Content-Type must be multipart/form-data with a boundary",
                 "invalid_content_type");
  }
  if (form.status == MultipartStatus::kTooManyFields) {
    form.detail = "multipart request contains too many fields";
  } else if (form.status == MultipartStatus::kDuplicateField) {
    form.detail = "multipart field is duplicated: " + form.detail;
  }
  if (form.status != MultipartStatus::kComplete) {
    return Error(400, "Bad Request", std::move(form.detail),
                 "invalid_multipart");
  }
  // As in vLLM and the other OpenAI-compatible servers, unknown fields are
  // ignored.

  const MultipartPart* file = Find(parts, "file");
  if (file == nullptr || file->body.empty()) {
    return Error(400, "Bad Request", "a non-empty WAV 'file' is required",
                 "missing_file");
  }
  const MultipartPart* model = Find(parts, "model");
  if (model != nullptr && !service.serves_model(model->body)) {
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
  // `include[]` only requests logprobs, which this server cannot return.
  if (Find(parts, "include[]") != nullptr) {
    return Error(400, "Bad Request", "transcription logprobs are not supported",
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
          ext::TranscriptionRequest input{
              .wav = {reinterpret_cast<const std::byte*>(wav.data()),
                      wav.size()},
              .context = context,
              .language = language,
              .max_new_tokens = maximum_tokens,
              .on_text = progress,
          };
          ext::TranscriptionResult result;
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
  ext::TranscriptionResult result;
  std::string error;
  if (!service.Transcribe(
          ext::TranscriptionRequest{
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
